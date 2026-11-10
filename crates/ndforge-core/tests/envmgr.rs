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
    Facts { gpus: smi.map(parse_nvidia_smi).map_err(String::from), free_disk_gb: disk, uv_found: uv, env_installed: env, selftest_passed: test, env_cuda_unchecked: false }
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

#[test]
fn the_real_worker_answers_hello_and_explains_a_missing_torch() {
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
    // a fake "python" that prints protocol lines
    let d = tempfile::tempdir().unwrap();
    let fake = d.path().join("python");
    std::fs::write(&fake, "#!/bin/sh\necho '{\"type\":\"log\",\"text\":\"hi\"}'\necho '{\"type\":\"progress\",\"step\":\"s\",\"value\":0.5,\"text\":\"half\"}'\necho 'not json'\necho '{\"type\":\"result\",\"data\":{\"ok\":true}}'\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut ev = Vec::new();
    let r = run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |e| ev.push(e)).unwrap();
    assert_eq!(r, json!({"ok": true}));
    assert_eq!(ev, vec![WorkerEvent::Log("hi".into()), WorkerEvent::Progress { step: "s".into(), value: 0.5, text: "half".into() }, WorkerEvent::Log("not json".into())]);
    std::fs::write(&fake, "#!/bin/sh\nexit 7\n").unwrap();
    assert!(run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("exit code 7"));
    std::fs::write(&fake, "#!/bin/sh\necho '{\"type\":\"error\",\"code\":\"c\",\"message\":\"Plain words\"}'\nexit 1\n").unwrap();
    assert_eq!(run_job(&fake, d.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0, "Plain words");
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
    let plan = SetupPlan { env_dir: d.path().join("env"), uv: uv.clone(), worker_dir: d.path().join("worker"), wheel: "cu128".into(), python_version: "3.13".into() };
    let mut events = Vec::new();
    run_setup(&plan, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    let lines: Vec<String> = std::fs::read_to_string(&log).unwrap().lines().map(String::from).collect();
    // venv, PyTorch (with its sister libraries), the freeze that pins them, Unsloth, the worker
    assert_eq!(lines.len(), 5, "{lines:?}");
    assert!(lines[0].starts_with("venv --python 3.13 --seed") && lines[1].contains("https://download.pytorch.org/whl/cu128") && lines[1].contains("torch torchvision torchaudio")
            && lines[2].starts_with("pip freeze") && lines[3].contains("unsloth") && lines[4].contains("worker"));
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
    assert_eq!(lines3.len(), 6);
    assert!(lines3[4].contains("transformers") && lines3[4].contains("peft") && lines3[4].contains("bitsandbytes") && lines3[5].contains("worker"));
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
    let plan_for = |d: &std::path::Path, uv: std::path::PathBuf, name: &str| SetupPlan { env_dir: d.join(name), uv, worker_dir: d.join("worker"), wheel: "cu128".into(), python_version: "3.13".into() };
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
    let facts = |unchecked| ndforge_core::envmgr::health::Facts { gpus: Ok(parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607, 576.80, 12.0")), free_disk_gb: Some(100.0), uv_found: true, env_installed: true, selftest_passed: false, env_cuda_unchecked: unchecked };
    let env_check = |f: &ndforge_core::envmgr::health::Facts| evaluate(f).into_iter().find(|c| c.id == "env").unwrap();
    let flagged = env_check(&facts(true));
    assert_eq!(flagged.light, Light::Warn);
    assert!(flagged.detail.contains("not checked") && flagged.fix.contains("CPU-only"), "{flagged:?}");
    assert_eq!(env_check(&facts(false)).light, Light::Ok);
}
