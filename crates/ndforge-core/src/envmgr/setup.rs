//! Installing the Python environment with `uv`: create it, install the CUDA build of PyTorch, keep it (a later install must not swap it for the
//! CPU-only build from the normal package index, which is what Windows gets there), install the worker package, and check that PyTorch really has CUDA.
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use super::process::{run_lines, Stream};
use super::python_path;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct SetupPlan {
    pub env_dir: PathBuf,
    pub uv: PathBuf,
    pub worker_dir: PathBuf,
    /// "cu128" / "cu126": the PyTorch CUDA build (see gpu::cuda_wheel_for)
    pub wheel: String,
    pub python_version: String,
    /// try to install Unsloth (faster, less memory); false = the standard libraries only (Transformers + PEFT + TRL), which install everywhere
    pub unsloth: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Step { index: usize, total: usize, text: String },
    Log(String),
}

const TOTAL: usize = 5;

/// The libraries the plain Transformers + PEFT + TRL fallback needs (what Unsloth would otherwise bring along).
const FALLBACK_LIBS: [&str; 6] = ["transformers>=5", "trl", "peft", "datasets", "accelerate", "bitsandbytes"];

/// The fast kernels of Qwen3.5's linear-attention layers, each tried on its own on both paths (with and without Unsloth): a failure never fails the install.
/// triton-windows: the Triton they need, which PyTorch does not ship for Windows; flash-linear-attention: the delta-rule kernels (the big speed-up);
/// causal-conv1d: the short convolution (a small part of the time; it only comes as source, so a ready-made package is rarely found).
pub fn kernel_extras() -> Vec<&'static str> {
    let mut v = vec!["flash-linear-attention", "causal-conv1d"];
    if cfg!(windows) {
        v.insert(0, "triton-windows");
    }
    v
}

/// The `uv` arguments that install one kernel package: ready-made packages only (never a build from source), into this Python. The pins are added by the caller.
pub fn kernel_extra_args(pkg: &str, py: &str) -> Vec<String> {
    ["pip", "install", "--python", py, "--only-binary=:all:", pkg].iter().map(|s| s.to_string()).collect()
}

/// What the log says when a kernel package could not be installed.
fn kernel_failure_note(pkg: &str) -> String {
    match pkg {
        "causal-conv1d" => "causal-conv1d has no ready-made package for this system; the short convolution uses the PyTorch version (a small part of the time).".into(),
        "flash-linear-attention" => "flash-linear-attention could not be installed: Qwen3.5's linear-attention layers will be several times slower when measuring, healing and checking.".into(),
        other => format!("{other} could not be installed; flash-linear-attention may not work without it."),
    }
}

fn step(on_event: &mut dyn FnMut(Event), index: usize, text: &str) {
    on_event(Event::Step { index, total: TOTAL, text: text.to_string() });
}

fn run_uv(plan: &SetupPlan, args: &[String], cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<std::process::ExitStatus> {
    let mut cmd = Command::new(&plan.uv);
    cmd.args(args);
    run_lines(cmd, cancel, &mut |_: Stream, line: &str| on_event(Event::Log(line.to_string())))
}

fn fail(text: &str, code: Option<i32>) -> Error {
    Error::new(format!("The step \"{text}\" failed (exit code {}). The log above shows why; fix that and press the button again.", code.unwrap_or(-1)))
}

pub fn run_setup(plan: &SetupPlan, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<()> {
    let py = python_path(&plan.env_dir).to_string_lossy().to_string();
    let a = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).collect() };
    std::fs::create_dir_all(plan.env_dir.parent().unwrap_or(&plan.env_dir))?;

    let t1 = "Creating the Python environment";
    step(on_event, 1, t1);
    let st = run_uv(plan, &a(&["venv", "--python", &plan.python_version, "--seed", &plan.env_dir.to_string_lossy()]), cancel, on_event)?;
    if !st.success() {
        return Err(fail(t1, st.code()));
    }

    let index = format!("https://download.pytorch.org/whl/{}", plan.wheel);
    let t2 = format!("Installing PyTorch for NVIDIA ({}) - several GB, please wait", plan.wheel);
    step(on_event, 2, &t2);
    let torch_libs = ["torch", "torchvision", "torchaudio"];
    let mut args = a(&["pip", "install", "--python", &py]);
    args.extend(torch_libs.iter().map(|s| s.to_string()));
    args.extend(a(&["--index-url", &index]));
    let st = run_uv(plan, &args, cancel, on_event)?;
    if !st.success() {
        return Err(fail(&t2, st.code()));
    }

    // Everything installed after this may use the normal package index too, but must keep exactly these builds: Unsloth and its friends ask for
    // torchvision and others whose own requirements would otherwise pull in a CPU-only torch from PyPI.
    let constraints = plan.env_dir.join("constraints.txt");
    let pinned = pin_torch(plan, &py, &torch_libs, &constraints);
    // the upper bounds the CI's worker tests proved (worker/constraints.txt); Unsloth is not pinned
    let tested = plan.worker_dir.join("constraints.txt");
    let with_pins = |mut v: Vec<String>| -> Vec<String> {
        if pinned {
            v.extend(a(&["--constraint", &constraints.to_string_lossy()]));
        }
        if tested.is_file() {
            v.extend(a(&["--constraint", &tested.to_string_lossy()]));
        }
        v.extend(a(&["--extra-index-url", &index, "--index-strategy", "unsafe-best-match"]));
        v
    };

    // (the kernel packages are installed after step 3, see `install_kernels`)
    // Unsloth (fast, small memory) is preferred when asked for; if it cannot be installed (it is picky about versions and about Windows) or was not wanted,
    // the standard libraries are installed instead and training still works, only slower and with more memory.
    let t3 = if plan.unsloth { "Installing Unsloth and the training libraries" } else { "Installing the training libraries (without Unsloth)" };
    step(on_event, 3, t3);
    let unsloth = if plan.unsloth {
        let st = run_uv(plan, &with_pins(a(&["pip", "install", "--python", &py, "unsloth"])), cancel, on_event)?;
        if !st.success() {
            on_event(Event::Log("Unsloth could not be installed. Installing the standard Transformers + PEFT + TRL libraries instead (slower, needs more graphics memory).".into()));
        }
        st.success()
    } else {
        on_event(Event::Log("Unsloth is skipped, as you chose. Installing the standard Transformers + PEFT + TRL libraries (slower than Unsloth, needs more graphics memory).".into()));
        false
    };
    let mut extras: Vec<String> = Vec::new();
    if !unsloth {
        let mut args = a(&["pip", "install", "--python", &py]);
        args.extend(FALLBACK_LIBS.iter().map(|s| s.to_string()));
        let st = run_uv(plan, &with_pins(args), cancel, on_event)?;
        if !st.success() {
            return Err(fail(t3, st.code()));
        }
    }
    extras.extend(install_kernels(plan, &py, &with_pins, cancel, on_event)?);

    let t4 = "Installing the ND Lore Forge worker";
    step(on_event, 4, t4);
    let st = run_uv(plan, &with_pins(a(&["pip", "install", "--python", &py, &plan.worker_dir.to_string_lossy()])), cancel, on_event)?;
    if !st.success() {
        return Err(fail(t4, st.code()));
    }

    let t5 = "Checking that PyTorch can use the graphics card";
    step(on_event, 5, t5);
    let mut verified = torch_has_cuda(&py, on_event);
    if verified == Some(false) {
        on_event(Event::Log("PyTorch in the environment is the CPU-only build (a package replaced it). Putting the NVIDIA build back.".into()));
        let mut args = a(&["pip", "install", "--python", &py, "--reinstall"]);
        args.extend(torch_libs.iter().map(|s| s.to_string()));
        args.extend(a(&["--index-url", &index]));
        let st = run_uv(plan, &args, cancel, on_event)?;
        if !st.success() {
            return Err(fail(t5, st.code()));
        }
        verified = torch_has_cuda(&py, on_event);
        if verified == Some(false) {
            return Err(Error::new(format!(
                "PyTorch in the training environment is still the CPU-only build after reinstalling it from {index}. Check that this computer can reach download.pytorch.org (a proxy or a firewall may block it), then press the install button again.")));
        }
    }

    std::fs::create_dir_all(&plan.env_dir)?;
    let extras_json = extras.iter().map(|e| format!("\"{e}\"")).collect::<Vec<_>>().join(", ");
    let marker = format!("{{\"wheel\": \"{}\", \"python\": \"{}\", \"unsloth\": {}, \"engine\": \"{}\", \"extras\": [{}], \"cuda_torch\": {}, \"requirements\": {}}}\n",
                         plan.wheel, plan.python_version, unsloth, if unsloth { "unsloth" } else { "standard" }, extras_json, verified.unwrap_or(false), super::health::ENV_REQUIREMENTS_VERSION);
    std::fs::write(plan.env_dir.join("ndforge_env.json"), marker)?;
    let _ = std::fs::remove_file(plan.env_dir.join("selftest.json")); // a new environment has to pass the test again
    Ok(())
}

/// Tries each kernel package (see `kernel_extras`), with the same pins as everything else so that PyTorch is never swapped. Returns the ones that were installed;
/// a failure is only logged in plain words. An error is returned only when the install was cancelled.
fn install_kernels(plan: &SetupPlan, py: &str, with_pins: &dyn Fn(Vec<String>) -> Vec<String>, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<Vec<String>> {
    let mut done = Vec::new();
    for pkg in kernel_extras() {
        let st = run_uv(plan, &with_pins(kernel_extra_args(pkg, py)), cancel, on_event)?;
        if st.success() {
            done.push(pkg.to_string());
        } else {
            on_event(Event::Log(kernel_failure_note(pkg)));
        }
    }
    Ok(done)
}

/// The "Install the fast kernels" button: only the kernel packages, into the environment that exists. Updates `extras` in the marker. Returns what was installed.
pub fn run_kernels_only(plan: &SetupPlan, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<Vec<String>> {
    let py = python_path(&plan.env_dir).to_string_lossy().to_string();
    if !std::path::Path::new(&py).is_file() || !plan.env_dir.join("ndforge_env.json").is_file() {
        return Err(Error::new("The training software is not installed yet. Press \"Install the training software\" first."));
    }
    let index = format!("https://download.pytorch.org/whl/{}", plan.wheel);
    let constraints = plan.env_dir.join("constraints.txt");
    let tested = plan.worker_dir.join("constraints.txt");
    let with_pins = |mut v: Vec<String>| -> Vec<String> {
        for c in [&constraints, &tested] {
            if c.is_file() {
                v.extend(["--constraint".to_string(), c.to_string_lossy().to_string()]);
            }
        }
        v.extend(["--extra-index-url".to_string(), index.clone(), "--index-strategy".to_string(), "unsafe-best-match".to_string()]);
        v
    };
    let done = install_kernels(plan, &py, &with_pins, cancel, on_event)?;
    // the marker keeps what was installed (the other keys stay as they are)
    let marker = plan.env_dir.join("ndforge_env.json");
    if let Ok(mut v) = std::fs::read(&marker).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).map_err(|e| e.to_string())) {
        let mut all: Vec<String> = v["extras"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
        for d in &done {
            if !all.contains(d) {
                all.push(d.clone());
            }
        }
        v["extras"] = serde_json::json!(all);
        if let Ok(t) = serde_json::to_string_pretty(&v) {      // pretty: the checks that read the marker look for `"key": value` with the space
            let _ = std::fs::write(&marker, t + "\n");
        }
    }
    on_event(Event::Log(if done.is_empty() { "No fast kernel could be installed.".to_string() } else { format!("Installed: {}.", done.join(", ")) }));
    Ok(done)
}

/// Writes `constraints.txt` with the exact versions of the torch libraries that are installed now; false when they could not be read.
fn pin_torch(plan: &SetupPlan, py: &str, libs: &[&str], file: &std::path::Path) -> bool {
    let Ok(out) = Command::new(&plan.uv).args(["pip", "freeze", "--python", py]).output() else { return false };
    if !out.status.success() {
        return false;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let pins: Vec<&str> = text.lines().map(str::trim).filter(|l| libs.iter().any(|n| l.starts_with(&format!("{n}==")))).collect();
    if pins.is_empty() || std::fs::write(file, pins.join("\n") + "\n").is_err() {
        return false;
    }
    true
}

/// Does the environment's PyTorch have CUDA support (a build, not whether a card is present)? None when it could not be asked.
fn torch_has_cuda(py: &str, on_event: &mut dyn FnMut(Event)) -> Option<bool> {
    let script = "import sys, torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda); sys.exit(0 if torch.version.cuda else 3)";
    let mut cmd = Command::new(py);
    cmd.args(["-c", script]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // no console window
    }
    let out = cmd.output().ok()?;
    for l in String::from_utf8_lossy(&out.stdout).lines().chain(String::from_utf8_lossy(&out.stderr).lines().take(3)) {
        on_event(Event::Log(l.to_string()));
    }
    match out.status.code() {
        Some(0) => Some(true),
        Some(3) => Some(false),
        _ => None,
    }
}
