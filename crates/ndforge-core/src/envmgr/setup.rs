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

/// Optional extras of the standard path, each tried on its own: a failure never fails the install. (flash-linear-attention: Qwen3.5's fast kernels;
/// triton-windows: the Triton they need, which PyTorch does not ship for Windows.)
fn standard_extras() -> Vec<&'static str> {
    let mut v = vec!["flash-linear-attention"];
    if cfg!(windows) {
        v.insert(0, "triton-windows");
    }
    v
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
    let with_pins = |mut v: Vec<String>| -> Vec<String> {
        if pinned {
            v.extend(a(&["--constraint", &constraints.to_string_lossy()]));
        }
        v.extend(a(&["--extra-index-url", &index, "--index-strategy", "unsafe-best-match"]));
        v
    };

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
        for extra in standard_extras() {
            let st = run_uv(plan, &with_pins(a(&["pip", "install", "--python", &py, extra])), cancel, on_event)?;
            if st.success() {
                extras.push(extra.to_string());
            } else {
                on_event(Event::Log(format!("Optional: {extra} could not be installed. Training works without it; {}.", if extra == "flash-linear-attention" { "Qwen3.5 will be much slower" } else { "some speed-ups are unavailable" })));
            }
        }
    }

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
    let marker = format!("{{\"wheel\": \"{}\", \"python\": \"{}\", \"unsloth\": {}, \"engine\": \"{}\", \"extras\": [{}], \"cuda_torch\": {}}}\n",
                         plan.wheel, plan.python_version, unsloth, if unsloth { "unsloth" } else { "standard" }, extras_json, verified.unwrap_or(false));
    std::fs::write(plan.env_dir.join("ndforge_env.json"), marker)?;
    let _ = std::fs::remove_file(plan.env_dir.join("selftest.json")); // a new environment has to pass the test again
    Ok(())
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
