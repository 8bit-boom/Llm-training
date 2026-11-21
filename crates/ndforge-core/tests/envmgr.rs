use std::sync::atomic::{AtomicBool, Ordering};

use ndforge_core::envmgr::gpu::{cuda_wheel_for, driver_version, parse_nvidia_smi, Precision};
use ndforge_core::envmgr::health::{can_install, evaluate, Facts, Light};
use ndforge_core::envmgr::jobs::{run_job, WorkerEvent};
use ndforge_core::envmgr::process::{run_lines, Stream};
use serde_json::json;

#[test]
fn parses_nvidia_smi_for_the_owner_s_and_the_servers_cards() {
    let g = parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607, 570.133.07, 12.0\nTesla T10, 16384, 535.104.05, 7.5\n");
    assert_eq!(g.len(), 2);
    assert_eq!((g[0].vram_mb, g[0].compute_cap, g[0].precision()), (32607, Some((12, 0)), Precision::Bf16));
    assert!(g[0].is_blackwell() && !g[1].is_blackwell());
    assert_eq!(g[1].precision(), Precision::Fp16); // the T10 (Turing) has no bf16
    assert!((g[0].vram_gb() - 31.8).abs() < 0.2);
}

#[test]
fn guesses_compute_capability_from_the_name_on_old_drivers() {
    let g = parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607, 570.86.16\nTesla V100-SXM2-16GB, 16384, 515.65.01\nMystery Card, 8000, 550.1\nbroken line\n");
    assert_eq!(g.len(), 3);
    assert_eq!(g[0].compute_cap, Some((12, 0)));
    assert_eq!((g[1].compute_cap, g[1].precision()), (Some((7, 0)), Precision::Fp16));
    assert_eq!((g[2].compute_cap, g[2].precision()), (None, Precision::Fp16));
}

#[test]
fn picks_the_cuda_build_from_the_driver() {
    let mk = |s: &str| parse_nvidia_smi(s).remove(0);
    assert_eq!(driver_version("570.133.07"), Some((570, 133)));
    assert_eq!(cuda_wheel_for(&mk("RTX 5090, 32607, 570.133.07, 12.0")).unwrap(), "cu128");
    assert!(cuda_wheel_for(&mk("RTX 5090, 32607, 560.35.03, 12.0")).unwrap_err().contains("570"));   // Blackwell needs 570+
    assert_eq!(cuda_wheel_for(&mk("RTX 4090, 24564, 560.35.03, 8.9")).unwrap(), "cu126");
    assert!(cuda_wheel_for(&mk("Tesla T10, 16384, 535.104.05, 7.5")).unwrap_err().contains("too old"));
    assert!(cuda_wheel_for(&mk("X, 8000, abc, 8.0")).unwrap_err().contains("could not be read"));
}

fn facts(smi: Result<&str, &str>, disk: Option<f64>, uv: bool, env: bool, test: bool) -> Facts {
    Facts { gpus: smi.map(parse_nvidia_smi).map_err(String::from), free_disk_gb: disk, uv_found: uv, env_installed: env, selftest_passed: test, env_cuda_unchecked: false, env_requirements: Some(ndforge_core::envmgr::health::ENV_REQUIREMENTS_VERSION), fast_kernels: None }
}

fn lights(f: &Facts) -> Vec<(&'static str, Light)> {
    evaluate(f).iter().map(|c| (c.id, c.light)).collect()
}

#[test]
fn health_lights() {
    let ready = facts(Ok("RTX 5090, 32607, 570.133.07, 12.0"), Some(200.0), true, true, true);
    assert!(lights(&ready).iter().all(|(_, l)| *l == Light::Ok), "{:?}", lights(&ready));
    assert!(can_install(&evaluate(&ready)));
    let fresh = facts(Ok("RTX 5090, 32607, 570.133.07, 12.0"), Some(200.0), true, false, false);
    let l = lights(&fresh);
    assert!(l.contains(&("env", Light::Warn)) && l.contains(&("selftest", Light::Warn)));
    assert!(can_install(&evaluate(&fresh)));
    // no driver, old driver, little memory, little disk, no uv: each is red and blocks the install
    for bad in [
        facts(Err("nvidia-smi was not found."), Some(200.0), true, false, false),
        facts(Ok("RTX 5090, 32607, 535.0, 12.0"), Some(200.0), true, false, false),
        facts(Ok("GTX 1050, 2048, 570.1, 6.1"), Some(200.0), true, false, false),
        facts(Ok("RTX 5090, 32607, 570.133.07, 12.0"), Some(5.0), true, false, false),
        facts(Ok("RTX 5090, 32607, 570.133.07, 12.0"), Some(200.0), false, false, false),
    ] {
        assert!(!can_install(&evaluate(&bad)), "{:?}", lights(&bad));
        assert!(evaluate(&bad).iter().filter(|c| c.light == Light::Fail).all(|c| !c.fix.is_empty()));
    }
    assert_eq!(lights(&facts(Ok("RTX 4060, 8188, 570.1, 8.9"), Some(30.0), true, true, true)).iter().filter(|(_, l)| *l == Light::Warn).count(), 2); // vram + disk
}

#[cfg(unix)]
#[test]
fn run_lines_streams_both_outputs_and_can_be_cancelled() {
    let mut got = Vec::new();
    let mut cmd = std::process::Command::new("sh");
    cmd.args(["-c", "echo out1; echo err1 1>&2; exit 3"]);
    let st = run_lines(cmd, &AtomicBool::new(false), &mut |s: Stream, l: &str| got.push((s, l.to_string()))).unwrap();
    assert_eq!(st.code(), Some(3));
    assert!(got.contains(&(Stream::Out, "out1".into())) && got.contains(&(Stream::Err, "err1".into())));
    // cancel: a long sleep is killed promptly
    let cancel = std::sync::Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let t = std::thread::spawn(move || { std::thread::sleep(std::time::Duration::from_millis(300)); c2.store(true, Ordering::SeqCst); });
    let started = std::time::Instant::now();
    let mut cmd = std::process::Command::new("sleep");
    cmd.arg("30");
    let r = run_lines(cmd, &cancel, &mut |_: Stream, _: &str| {});
    t.join().unwrap();
    assert!(r.unwrap_err().0.contains("Cancelled") && started.elapsed().as_secs() < 5);
}

fn python3() -> Option<std::path::PathBuf> {
    std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("python3")).find(|p| p.is_file()))
}

fn worker_dir() -> std::path::PathBuf {
    ndforge_core::envmgr::worker_dir()
}

/// Every job saves a log in the settings folder: these tests get a private one (kept for the whole test run), never the real one.
fn private_config_dir() {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let d = DIR.get_or_init(|| tempfile::tempdir().unwrap());
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
}

#[test]
fn the_real_worker_answers_hello_and_explains_a_missing_torch() {
    private_config_dir();
    let Some(py) = python3() else { return };
    let mut events = Vec::new();
    let r = run_job(&py, &worker_dir(), &json!({"job": "hello"}), &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert_eq!(r["worker"], "0.1.0");
    let unknown = run_job(&py, &worker_dir(), &json!({"job": "nope"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(unknown.0.contains("Unknown job"));
    // the self-test on a machine without PyTorch: a plain-language error (skipped where torch happens to be installed)
    let has_torch = std::process::Command::new(&py).args(["-c", "import torch"]).output().map(|o| o.status.success()).unwrap_or(false);
    if !has_torch {
        let e = run_job(&py, &worker_dir(), &json!({"job": "selftest"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err();
        assert!(e.0.contains("PyTorch is not installed"), "{e}");
    }
}

#[cfg(unix)]
#[test]
fn worker_events_and_crashes_are_reported() {
    use std::os::unix::fs::PermissionsExt;
    private_config_dir();
    // a fake "python" that prints protocol lines
    let d = tempfile::tempdir().unwrap();
    let fake = d.path().join("python");
    std::fs::write(&fake, "#!/bin/sh\necho '{\"type\":\"log\",\"text\":\"hi\"}'\necho '{\"type\":\"progress\",\"step\":\"s\",\"value\":0.5,\"text\":\"half\"}'\necho 'not json'\necho '{\"type\":\"result\",\"data\":{\"ok\":true}}'\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut ev = Vec::new();
    let r = run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |e| ev.push(e)).unwrap();
    assert_eq!(r, json!({"ok": true}));
    assert_eq!(ev, vec![WorkerEvent::Log("hi".into()), WorkerEvent::Progress { step: "s".into(), value: 0.5, text: "half".into() }, WorkerEvent::Log("not json".into())]);
    // a progress line with `detail` keeps it, tagged "progress"; one without stays a plain Progress
    std::fs::write(&fake, "#!/bin/sh\necho '{\"type\":\"progress\",\"step\":\"cache\",\"value\":0.5,\"text\":\"t\",\"detail\":{\"unit\":\"tokens\",\"done\":5,\"total\":10}}'\necho '{\"type\":\"result\",\"data\":{}}'\n").unwrap();
    let mut ev = Vec::new();
    run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |e| ev.push(e)).unwrap();
    match &ev[0] {
        WorkerEvent::Other(v) => {
            assert_eq!(v["kind"], "progress");
            assert_eq!(v["step"], "cache");
            assert_eq!(v["detail"]["total"], 10);
        }
        other => panic!("{other:?}"),
    }
    std::fs::write(&fake, "#!/bin/sh\nexit 7\n").unwrap();
    assert!(run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("exit code 7"));
    std::fs::write(&fake, "#!/bin/sh\necho '{\"type\":\"error\",\"code\":\"c\",\"message\":\"Plain words\"}'\nexit 1\n").unwrap();
    // the message is the worker's, followed by where the saved log is
    let msg = run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert!(msg.starts_with("Plain words (details: logs/jobs/") && msg.ends_with(".log)"), "{msg}");
    std::fs::write(&fake, "#!/bin/sh\nexit 0\n").unwrap();
    assert!(run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("without an answer"));
}

#[cfg(unix)]
#[test]
fn setup_runs_the_uv_steps_in_order_and_writes_the_marker() {
    use ndforge_core::envmgr::setup::{run_setup, Event, SetupPlan};
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    let log = d.path().join("uv.log");
    let uv = d.path().join("uv");
    std::fs::write(&uv, format!("#!/bin/sh\necho \"$@\" >> {}\necho \"running $1\"\nif [ \"$FAIL_AT\" = \"$2\" ]; then exit 2; fi\nexit 0\n", log.display())).unwrap();
    std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
    let plan = SetupPlan { env_dir: d.path().join("env"), uv: uv.clone(), worker_dir: d.path().join("worker"), wheel: "cu128".into(), python_version: "3.13".into(), unsloth: true };
    let mut events = Vec::new();
    run_setup(&plan, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    let lines: Vec<String> = std::fs::read_to_string(&log).unwrap().lines().map(String::from).collect();
    // venv, PyTorch (with its sister libraries), the freeze that pins them, Unsloth, the fast kernels (on this path too), the worker
    let kernels = ndforge_core::envmgr::setup::kernel_extras();
    assert_eq!(lines.len(), 5 + kernels.len(), "{lines:?}");
    assert!(lines[0].starts_with("venv --python 3.13 --seed") && lines[1].contains("https://download.pytorch.org/whl/cu128") && lines[1].contains("torch torchvision torchaudio")
            && lines[2].starts_with("pip freeze") && lines[3].contains("unsloth") && lines.last().unwrap().contains("worker"));
    for (i, k) in kernels.iter().enumerate() {
        assert!(lines[4 + i].contains(k) && lines[4 + i].contains("--only-binary=:all:"), "{}", lines[4 + i]);
    }
    assert_eq!(events.iter().filter(|e| matches!(e, Event::Step { total: 5, .. })).count(), 5);
    assert!(std::fs::read_to_string(plan.env_dir.join("ndforge_env.json")).unwrap().contains("\"unsloth\": true"));
    assert!(events.contains(&Event::Log("running pip".into())));
    assert!(plan.env_dir.join("ndforge_env.json").is_file());
    // a failing step stops the install with a readable message and no marker
    let plan2 = SetupPlan { env_dir: d.path().join("env2"), ..plan.clone() };
    std::fs::write(&uv, "#!/bin/sh\nexit 2\n").unwrap();
    let e = run_setup(&plan2, &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("failed") && e.0.contains("exit code 2"), "{e}");
    assert!(!plan2.env_dir.join("ndforge_env.json").exists());

    // Unsloth cannot be installed: the plain libraries are installed instead and the install still succeeds
    let log2 = d.path().join("uv2.log");
    std::fs::write(&uv, format!("#!/bin/sh\necho \"$@\" >> {}\ncase \"$*\" in *unsloth*) exit 3;; esac\nexit 0\n", log2.display())).unwrap();
    let plan3 = SetupPlan { env_dir: d.path().join("env3"), ..plan.clone() };
    let mut ev3 = Vec::new();
    run_setup(&plan3, &AtomicBool::new(false), &mut |e| ev3.push(e)).unwrap();
    let lines3: Vec<String> = std::fs::read_to_string(&log2).unwrap().lines().map(String::from).collect();
    // venv, torch, freeze, Unsloth (fails), the standard libraries, the fast kernels, the worker
    let n_extras = ndforge_core::envmgr::setup::kernel_extras().len();
    assert_eq!(lines3.len(), 6 + n_extras, "{lines3:?}");
    assert!(lines3[4].contains("transformers>=5") && lines3[4].contains("peft") && lines3[4].contains("bitsandbytes") && lines3[5].contains("flash-linear-attention") && lines3.last().unwrap().contains("worker"));
    assert!(ev3.iter().any(|e| matches!(e, Event::Log(l) if l.contains("Unsloth could not be installed"))));
    assert!(std::fs::read_to_string(plan3.env_dir.join("ndforge_env.json")).unwrap().contains("\"unsloth\": false"));
}

/// A fake uv that makes a fake Python in each environment: that Python says "CUDA None" (exit 3) until `fixed` exists; a `--reinstall` creates it
/// when `reinstall_fixes`. `pip freeze` lists the torch builds and one other package.
#[cfg(unix)]
fn fake_uv(d: &std::path::Path, log: &std::path::Path, python_says: &str, reinstall_fixes: bool) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let uv = d.join("uv");
    let fixed = d.join("fixed");
    let script = match python_says {
        "cuda" => "#!/bin/sh\necho 'PyTorch 2.8.0 CUDA 12.8'\nexit 0\n".to_string(),
        _ => format!("#!/bin/sh\nif [ -f {f} ]; then echo 'PyTorch 2.8.0 CUDA 12.8'; exit 0; fi\necho 'PyTorch 2.8.0 CUDA None'\nexit 3\n", f = fixed.display()),
    };
    let python_template = d.join("python_template");
    std::fs::write(&python_template, script).unwrap();
    let touch = if reinstall_fixes { format!("case \"$*\" in *--reinstall*) touch {};; esac\n", fixed.display()) } else { String::new() };
    std::fs::write(&uv, format!(
        "#!/bin/sh\necho \"$@\" >> {log}\ncase \"$1\" in\n  venv) for last; do :; done; mkdir -p \"$last/bin\"; cp {tpl} \"$last/bin/python\"; chmod +x \"$last/bin/python\";;\n  pip) if [ \"$2\" = freeze ]; then printf 'numpy==2.0.0\\ntorch==2.8.0+cu128\\ntorchvision==0.23.0+cu128\\ntorchaudio==2.8.0+cu128\\n'; fi;;\nesac\n{touch}exit 0\n",
        log = log.display(), tpl = python_template.display(), touch = touch)).unwrap();
    std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
    uv
}

#[cfg(unix)]
#[test]
fn setup_keeps_the_nvidia_pytorch_and_puts_it_back_when_a_package_replaced_it() {
    use ndforge_core::envmgr::setup::{run_setup, Event, SetupPlan};
    let plan_for = |d: &std::path::Path, uv: std::path::PathBuf, name: &str| SetupPlan { env_dir: d.join(name), uv, worker_dir: d.join("worker"), wheel: "cu128".into(), python_version: "3.13".into(), unsloth: true };
    let marker = |p: &SetupPlan| std::fs::read_to_string(p.env_dir.join("ndforge_env.json")).unwrap();

    // 1. the installed torch builds are pinned while everything else is installed, and extra packages may come from the NVIDIA index too
    let d = tempfile::tempdir().unwrap();
    let log = d.path().join("log");
    let plan = plan_for(d.path(), fake_uv(d.path(), &log, "cuda", false), "env");
    run_setup(&plan, &AtomicBool::new(false), &mut |_| {}).unwrap();
    let pins = std::fs::read_to_string(plan.env_dir.join("constraints.txt")).unwrap();
    assert_eq!(pins, "torch==2.8.0+cu128\ntorchvision==0.23.0+cu128\ntorchaudio==2.8.0+cu128\n");
    let lines: Vec<String> = std::fs::read_to_string(&log).unwrap().lines().map(String::from).collect();
    for l in lines.iter().filter(|l| l.contains("unsloth") || l.contains("worker")) {
        assert!(l.contains("--constraint") && l.contains("constraints.txt") && l.contains("--extra-index-url https://download.pytorch.org/whl/cu128") && l.contains("--index-strategy unsafe-best-match"), "{l}");
    }
    assert!(!lines.iter().any(|l| l.contains("--reinstall")), "PyTorch was fine: nothing to put back");
    assert!(marker(&plan).contains("\"cuda_torch\": true"));

    // 2. a package replaced PyTorch by the CPU-only build: it is found and put back
    let d = tempfile::tempdir().unwrap();
    let log = d.path().join("log");
    let plan = plan_for(d.path(), fake_uv(d.path(), &log, "cpu", true), "env");
    let mut ev = Vec::new();
    run_setup(&plan, &AtomicBool::new(false), &mut |e| ev.push(e)).unwrap();
    let text = std::fs::read_to_string(&log).unwrap();
    let re = text.lines().find(|l| l.contains("--reinstall")).expect("reinstalled");
    assert!(re.contains("torch torchvision torchaudio") && re.contains("--index-url https://download.pytorch.org/whl/cu128"), "{re}");
    assert!(ev.iter().any(|e| matches!(e, Event::Log(l) if l.contains("CPU-only build"))) && ev.iter().any(|e| matches!(e, Event::Log(l) if l == "PyTorch 2.8.0 CUDA 12.8")));
    assert!(marker(&plan).contains("\"cuda_torch\": true"));

    // 3. reinstalling does not help (for example the NVIDIA index cannot be reached): a plain message, no marker, so the page does not say "Installed"
    let d = tempfile::tempdir().unwrap();
    let log = d.path().join("log");
    let plan = plan_for(d.path(), fake_uv(d.path(), &log, "cpu", false), "env");
    let e = run_setup(&plan, &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("still the CPU-only build") && e.0.contains("download.pytorch.org"), "{e}");
    assert!(!plan.env_dir.join("ndforge_env.json").exists());
}

#[test]
fn an_install_made_before_pytorch_was_checked_is_flagged() {
    use ndforge_core::envmgr::health::{env_cuda_unchecked, evaluate, Light};
    use ndforge_core::envmgr::gpu::parse_nvidia_smi;
    let d = tempfile::tempdir().unwrap();
    let env = d.path().join("env");
    std::fs::create_dir_all(&env).unwrap();
    assert!(!env_cuda_unchecked(&env), "no environment, nothing to flag");
    std::fs::write(env.join("ndforge_env.json"), "{\"wheel\": \"cu128\", \"python\": \"3.13\", \"unsloth\": true}\n").unwrap();
    assert!(env_cuda_unchecked(&env));
    std::fs::write(env.join("ndforge_env.json"), "{\"wheel\": \"cu128\", \"unsloth\": true, \"cuda_torch\": true}\n").unwrap();
    assert!(!env_cuda_unchecked(&env));
    let facts = |unchecked| ndforge_core::envmgr::health::Facts { gpus: Ok(parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607, 576.80, 12.0")), free_disk_gb: Some(100.0), uv_found: true, env_installed: true, selftest_passed: false, env_cuda_unchecked: unchecked, env_requirements: Some(ndforge_core::envmgr::health::ENV_REQUIREMENTS_VERSION), fast_kernels: None };
    let env_check = |f: &ndforge_core::envmgr::health::Facts| evaluate(f).into_iter().find(|c| c.id == "env").unwrap();
    let flagged = env_check(&facts(true));
    assert_eq!(flagged.light, Light::Warn);
    assert!(flagged.detail.contains("not checked") && flagged.fix.contains("CPU-only"), "{flagged:?}");
    assert_eq!(env_check(&facts(false)).light, Light::Ok);
}

#[cfg(unix)]
#[test]
fn setup_without_unsloth_installs_only_the_standard_libraries_and_never_fails_on_the_optional_extras() {
    use ndforge_core::envmgr::health::env_engine;
    use ndforge_core::envmgr::setup::{run_setup, Event, SetupPlan};
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    let log = d.path().join("log");
    let uv = d.path().join("uv");
    // an optional extra that cannot be installed
    std::fs::write(&uv, format!("#!/bin/sh\necho \"$@\" >> {}\ncase \"$*\" in *flash-linear-attention*) exit 4;; esac\nexit 0\n", log.display())).unwrap();
    std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
    let plan = SetupPlan { env_dir: d.path().join("env"), uv: uv.clone(), worker_dir: d.path().join("worker"), wheel: "cu128".into(), python_version: "3.13".into(), unsloth: false };
    let mut events = Vec::new();
    run_setup(&plan, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(!text.contains("unsloth"), "Unsloth must not even be tried:\n{text}");
    let lib = text.lines().find(|l| l.contains("bitsandbytes")).expect("the standard libraries were installed");
    assert!(lib.contains("transformers>=5") && lib.contains("trl") && lib.contains("peft") && lib.contains("accelerate") && lib.contains("--extra-index-url https://download.pytorch.org/whl/cu128"));
    assert!(text.lines().any(|l| l.contains("flash-linear-attention")) && text.lines().last().unwrap().contains("worker"));
    assert!(events.iter().any(|e| matches!(e, Event::Log(l) if l.contains("Unsloth is skipped, as you chose"))));
    assert!(events.iter().any(|e| matches!(e, Event::Log(l) if l.contains("flash-linear-attention could not be installed") && l.contains("several times slower"))));
    let marker = std::fs::read_to_string(plan.env_dir.join("ndforge_env.json")).unwrap();
    assert!(marker.contains("\"unsloth\": false") && marker.contains("\"engine\": \"standard\"") && marker.contains("causal-conv1d") && !marker.contains("\"flash-linear-attention\""), "{marker}");
    assert_eq!(env_engine(&plan.env_dir).as_deref(), Some("standard"));
    // the libraries themselves cannot be installed: that is an error (an environment without them cannot train)
    std::fs::write(&uv, "#!/bin/sh\ncase \"$*\" in *bitsandbytes*) exit 5;; esac\nexit 0\n").unwrap();
    let plan2 = SetupPlan { env_dir: d.path().join("env2"), ..plan.clone() };
    let e = run_setup(&plan2, &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("without Unsloth") && e.0.contains("exit code 5"), "{e}");
    assert!(!plan2.env_dir.join("ndforge_env.json").exists());
    // with Unsloth installed the engine says so
    std::fs::write(&uv, "#!/bin/sh\nexit 0\n").unwrap();
    let plan3 = SetupPlan { env_dir: d.path().join("env3"), unsloth: true, ..plan.clone() };
    run_setup(&plan3, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(env_engine(&plan3.env_dir).as_deref(), Some("unsloth"));
    assert_eq!(env_engine(&d.path().join("nowhere")), None);
}

#[test]
fn the_graphics_card_is_asked_once_a_minute_not_once_per_page() {
    use ndforge_core::envmgr::gpu::detect_gpus_cached;
    let first = detect_gpus_cached();
    let t0 = std::time::Instant::now();
    for _ in 0..200 {
        assert_eq!(detect_gpus_cached(), first, "the remembered answer is the same one");
    }
    assert!(t0.elapsed() < std::time::Duration::from_secs(2), "no program is started for the remembered answer");
}

#[test]
fn an_environment_made_by_an_older_program_is_told_to_install_again_and_the_marker_keeps_the_version() {
    use ndforge_core::envmgr::health::{env_requirements, ENV_REQUIREMENTS_VERSION};
    let d = tempfile::tempdir().unwrap();
    assert_eq!(env_requirements(d.path()), None);
    std::fs::write(d.path().join("ndforge_env.json"), "{\"wheel\": \"cu128\", \"cuda_torch\": true}\n").unwrap();
    assert_eq!(env_requirements(d.path()), None, "a setup from before the version existed");
    std::fs::write(d.path().join("ndforge_env.json"), format!("{{\"wheel\": \"cu128\", \"cuda_torch\": true, \"requirements\": {ENV_REQUIREMENTS_VERSION}}}\n")).unwrap();
    assert_eq!(env_requirements(d.path()), Some(ENV_REQUIREMENTS_VERSION));
    let f = |req: Option<u32>| {
        let mut x = facts(Ok("NVIDIA GeForce RTX 5090, 32607, 576.80, 12.0"), Some(100.0), true, true, true);
        x.env_requirements = req;
        evaluate(&x).into_iter().find(|c| c.id == "env").unwrap()
    };
    let old = f(None);
    assert_eq!(old.light, Light::Warn);
    assert!(old.detail.contains("older version") && old.fix.contains("Install the training software"), "{old:?}");
    assert_eq!(f(Some(ENV_REQUIREMENTS_VERSION - 1)).light, Light::Warn);
    assert_eq!(f(Some(ENV_REQUIREMENTS_VERSION)).light, Light::Ok);
    assert_eq!(f(Some(ENV_REQUIREMENTS_VERSION + 1)).light, Light::Ok, "an environment newer than the program is not nagged");
}

#[test]
fn the_fast_kernels_light_follows_the_environment_and_never_blocks_the_install() {
    use ndforge_core::envmgr::health::{can_install, env_fast_kernels, evaluate, Light};
    let smi = Ok("NVIDIA GeForce RTX 5090, 32607, 576.80, 12.0");
    let find = |f: &Facts| evaluate(f).into_iter().find(|c| c.id == "kernels");
    let mut f = facts(smi, Some(500.0), true, true, true);
    assert!(find(&f).is_none(), "no answer, no light");
    f.fast_kernels = Some(true);
    assert_eq!(find(&f).unwrap().light, Light::Ok);
    f.fast_kernels = Some(false);
    let c = find(&f).unwrap();
    assert_eq!(c.light, Light::Warn);
    assert!(c.detail.contains("flash-linear-attention") && c.fix.contains("fast kernels"));
    assert!(can_install(&evaluate(&f)));
    // the real look: a fake environment folder with and without the package's dist-info (both layouts)
    let d = tempfile::tempdir().unwrap();
    assert_eq!(env_fast_kernels(d.path()), None, "no environment marker");
    std::fs::write(d.path().join("ndforge_env.json"), "{}").unwrap();
    assert_eq!(env_fast_kernels(d.path()), Some(false));
    let unix = d.path().join("lib").join("python3.12").join("site-packages");
    std::fs::create_dir_all(&unix).unwrap();
    assert_eq!(env_fast_kernels(d.path()), Some(false));
    std::fs::create_dir(unix.join("flash_linear_attention-0.4.1.dist-info")).unwrap();
    assert_eq!(env_fast_kernels(d.path()), Some(true));
    let e = tempfile::tempdir().unwrap();
    std::fs::write(e.path().join("ndforge_env.json"), "{}").unwrap();
    let win = e.path().join("Lib").join("site-packages");
    std::fs::create_dir_all(&win).unwrap();
    std::fs::create_dir(win.join("flash_linear_attention-0.4.1.dist-info")).unwrap();
    assert_eq!(env_fast_kernels(e.path()), Some(true));
}

#[test]
fn a_kernel_package_is_installed_from_a_ready_made_wheel_only_and_a_failure_is_plain_words() {
    use ndforge_core::envmgr::setup::{kernel_extra_args, kernel_extras};
    let args = kernel_extra_args("flash-linear-attention", "/env/bin/python");
    assert_eq!(args, vec!["pip", "install", "--python", "/env/bin/python", "--only-binary=:all:", "flash-linear-attention"]);
    let names = kernel_extras();
    assert!(names.contains(&"flash-linear-attention") && names.contains(&"causal-conv1d"));
    assert_eq!(names.contains(&"triton-windows"), cfg!(windows));
    assert_eq!(names.iter().position(|n| *n == "flash-linear-attention").unwrap() > names.iter().position(|n| *n == "triton-windows").unwrap_or(0) || !cfg!(windows), true);
}

#[cfg(unix)]
#[test]
fn a_kernel_that_cannot_be_installed_is_a_note_and_the_rest_goes_on_and_the_button_updates_the_marker() {
    use ndforge_core::envmgr::setup::{run_kernels_only, run_setup, Event, SetupPlan};
    use std::os::unix::fs::PermissionsExt;
    let d = tempfile::tempdir().unwrap();
    let log = d.path().join("uv.log");
    let uv = d.path().join("uv");
    std::fs::write(&uv, format!("#!/bin/sh\necho \"$@\" >> {}\ncase \"$*\" in *causal-conv1d*) exit 1;; esac\nexit 0\n", log.display())).unwrap();
    std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
    let plan = SetupPlan { env_dir: d.path().join("env"), uv, worker_dir: d.path().join("worker"), wheel: "cu128".into(), python_version: "3.13".into(), unsloth: true };
    let mut ev = Vec::new();
    // no environment yet: the button says so
    assert!(run_kernels_only(&plan, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("not installed yet"));
    // the full setup (its fake Python is missing: the CUDA check cannot be asked and the install still writes the marker)
    run_setup(&plan, &AtomicBool::new(false), &mut |e| ev.push(e)).unwrap();
    assert!(ev.iter().any(|e| matches!(e, Event::Log(l) if l.starts_with("causal-conv1d has no ready-made package"))), "{ev:?}");
    let marker = std::fs::read_to_string(plan.env_dir.join("ndforge_env.json")).unwrap();
    assert!(marker.contains("flash-linear-attention") && !marker.contains("\"causal-conv1d\""), "{marker}");
    // the button: only the kernels are tried again, the marker keeps its other keys
    std::fs::create_dir_all(plan.env_dir.join("bin")).unwrap();
    std::fs::write(plan.env_dir.join("bin").join("python"), "").unwrap();
    std::fs::write(&log, "").unwrap();
    let done = run_kernels_only(&plan, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(done, vec!["flash-linear-attention".to_string()]);
    let lines: Vec<String> = std::fs::read_to_string(&log).unwrap().lines().map(String::from).collect();
    assert_eq!(lines.len(), ndforge_core::envmgr::setup::kernel_extras().len());
    assert!(lines.iter().all(|l| l.contains("--only-binary=:all:")));
    let after = std::fs::read_to_string(plan.env_dir.join("ndforge_env.json")).unwrap();
    assert!(after.contains("\"unsloth\": true") && after.contains("flash-linear-attention"), "{after}");
}

// ---- pinned fast kernels (docs/HEAL_SPEED.md)

#[test]
fn triton_is_pinned_from_the_torch_version_and_an_unknown_torch_is_left_alone() {
    use ndforge_core::envmgr::setup::{kernel_requirement, triton_pin};
    assert_eq!(triton_pin("2.14.1+cu130"), Some(">=3.8,<3.9"));
    assert_eq!(triton_pin("2.14.0"), Some(">=3.8,<3.9"));
    assert_eq!(triton_pin("2.15.0+cu130"), Some(">=3.9,<3.10"));
    assert_eq!(triton_pin("2.13.0"), None);
    assert_eq!(triton_pin(""), None);
    assert_eq!(kernel_requirement("triton-windows", "2.14.1+cu130").unwrap(), "triton-windows>=3.8,<3.9");
    let why = kernel_requirement("triton-windows", "2.99.0").unwrap_err();
    assert!(why.contains("2.99.0") && why.contains("left as it is") && why.contains("unpinned"), "{why}");
    assert_eq!(kernel_requirement("flash-linear-attention", "x").unwrap(), "flash-linear-attention>=0.6");
    assert_eq!(kernel_requirement("causal-conv1d", "x").unwrap(), "causal-conv1d");
}

#[test]
fn the_kernel_check_says_plainly_what_works_and_rolls_nothing_back() {
    use ndforge_core::envmgr::setup::{parse_kernel_check, KERNEL_CHECK};
    assert!(KERNEL_CHECK.contains("fla.ops.gated_delta_rule") && KERNEL_CHECK.contains("backward") && KERNEL_CHECK.contains("cuda"));
    let ok = parse_kernel_check(Some(0), "...\nOK\n");
    assert!(ok.ok && ok.text.contains("work"));
    let imp = parse_kernel_check(Some(4), "IMPORT ModuleNotFoundError No module named 'triton'");
    assert!(!imp.ok && imp.text.contains("does not import") && imp.text.contains("slow PyTorch kernels") && imp.text.contains("Nothing was removed"), "{}", imp.text);
    let run = parse_kernel_check(Some(5), "RUN RuntimeError no kernel image is available for execution on the device");
    assert!(!run.ok && run.text.contains("failed on this graphics card") && run.text.contains("no kernel image"));
    let none = parse_kernel_check(None, "could not start");
    assert!(!none.ok && none.text.contains("could not be run"));
}

fn reqs(v: &str) -> Vec<String> {
    vec![format!("triton=={v}; platform_system == \"Linux\" and platform_machine == \"x86_64\""), "pytorch-triton-rocm==9.9; extra == 'rocm'".to_string(), "triton-windows==1.0".to_string()]
}

#[test]
fn the_triton_pin_is_read_from_the_torch_wheel_then_the_table_then_unknown() {
    use ndforge_core::envmgr::setup::{kernel_requirement_with, triton_pin_for, PinSource};
    assert_eq!(triton_pin_for("2.11.0+cu128", &reqs("3.6.0")), Some((">=3.6,<3.7".into(), PinSource::Wheel)));
    assert_eq!(triton_pin_for("2.14.1+cu130", &reqs("3.8.0")), Some((">=3.8,<3.9".into(), PinSource::Wheel)));
    assert_eq!(triton_pin_for("2.15.0", &reqs("3.9.1")), Some((">=3.9,<3.10".into(), PinSource::Wheel)));
    assert_eq!(triton_pin_for("2.14.1", &[]), Some((">=3.8,<3.9".into(), PinSource::Table)), "no requirement: the table");
    assert_eq!(triton_pin_for("2.11.0", &[]), None, "no requirement and not in the table: unknown");
    assert_eq!(triton_pin_for("2.11.0", &["triton (==3.6.0)".to_string()]), Some((">=3.6,<3.7".into(), PinSource::Wheel)), "the old requirement form");
    assert_eq!(kernel_requirement_with("triton-windows", "2.11.0+cu128", &reqs("3.6.0")).unwrap(), "triton-windows>=3.6,<3.7");
    let why = kernel_requirement_with("triton-windows", "2.11.0", &[]).unwrap_err();
    assert!(why.contains("unknown") && why.contains("left as it is"), "{why}");
}

#[test]
fn an_installed_triton_that_does_not_belong_to_the_torch_is_said_plainly_and_the_light_points_at_the_button() {
    use ndforge_core::envmgr::health::{triton_check, Light};
    use ndforge_core::envmgr::setup::triton_status;
    let bad = triton_status("2.11.0+cu128", Some("3.9.0"), &reqs("3.6.0"));
    assert_eq!(bad.matches, Some(false));
    assert_eq!(bad.line(), "Triton 3.9.0 does not match torch 2.11.0 (wants 3.6.x): press Install fast kernels to fix.");
    let c = triton_check(&bad).unwrap();
    assert_eq!(c.light, Light::Warn);
    assert!(c.detail.contains("press Install fast kernels") && c.fix.contains("Install fast kernels"), "{c:?}");
    let good = triton_status("2.11.0+cu128", Some("3.6.2"), &reqs("3.6.0"));
    assert!(good.line().contains("matches") && triton_check(&good).unwrap().light == Light::Ok);
    assert!(triton_check(&triton_status("2.11.0", None, &reqs("3.6.0"))).is_none(), "no Triton installed: nothing to compare");
    assert!(triton_check(&triton_status("2.11.0", Some("3.9.0"), &[])).is_none(), "unknown pin: no light");
}

#[test]
fn the_metadata_script_reads_a_fake_torch_wheel() {
    use ndforge_core::envmgr::setup::{parse_triton_facts, TRITON_FACTS};
    let py = if cfg!(windows) { "python" } else { "python3" };
    if std::process::Command::new(py).arg("--version").output().is_err() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    for (name, version, extra) in [("torch", "2.11.0+cu128", "Requires-Dist: triton==3.6.0; platform_system == \"Linux\"\n"), ("triton_windows", "3.9.0", "")] {
        let di = d.path().join(format!("{name}-{version}.dist-info"));
        std::fs::create_dir_all(&di).unwrap();
        std::fs::write(di.join("METADATA"), format!("Metadata-Version: 2.1\nName: {}\nVersion: {version}\n{extra}", name.replace('_', "-"))).unwrap();
    }
    let o = std::process::Command::new(py).args(["-I", "-c", &format!("import sys; sys.path.insert(0, {:?})\n{}", d.path().to_string_lossy(), TRITON_FACTS)]).output().unwrap();
    let t = parse_triton_facts(&String::from_utf8_lossy(&o.stdout)).unwrap_or_else(|| panic!("{}", String::from_utf8_lossy(&o.stderr)));
    assert_eq!((t.torch.as_str(), t.triton.as_deref(), t.matches), ("2.11.0+cu128", Some("3.9.0"), Some(false)), "{t:?}");
    assert_eq!(t.wants().as_deref(), Some("3.6"));
}
