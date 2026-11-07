//! Installing the Python environment with `uv`: create it, install the CUDA build of PyTorch, install the worker package.
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Step { index: usize, total: usize, text: String },
    Log(String),
}

const TOTAL: usize = 4;

/// The libraries the plain Transformers + PEFT + TRL fallback needs (what Unsloth would otherwise bring along).
const FALLBACK_LIBS: [&str; 6] = ["transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"];

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

    let t2 = format!("Installing PyTorch for NVIDIA ({}) - several GB, please wait", plan.wheel);
    step(on_event, 2, &t2);
    let st = run_uv(plan, &a(&["pip", "install", "--python", &py, "torch", "--index-url", &format!("https://download.pytorch.org/whl/{}", plan.wheel)]), cancel, on_event)?;
    if !st.success() {
        return Err(fail(&t2, st.code()));
    }

    // Unsloth (fast, small memory) is preferred; if it cannot be installed (it is picky about versions and about Windows) the plain libraries are
    // installed instead and training still works, only slower and with more memory.
    let t3 = "Installing Unsloth and the training libraries";
    step(on_event, 3, t3);
    let st = run_uv(plan, &a(&["pip", "install", "--python", &py, "unsloth"]), cancel, on_event)?;
    let unsloth = st.success();
    if !unsloth {
        on_event(Event::Log("Unsloth could not be installed. Installing the plain Transformers + PEFT + TRL libraries instead (slower, needs more graphics memory).".into()));
        let mut args = a(&["pip", "install", "--python", &py]);
        args.extend(FALLBACK_LIBS.iter().map(|s| s.to_string()));
        let st = run_uv(plan, &args, cancel, on_event)?;
        if !st.success() {
            return Err(fail(t3, st.code()));
        }
    }

    let t4 = "Installing the ND Lore Forge worker";
    step(on_event, 4, t4);
    let st = run_uv(plan, &a(&["pip", "install", "--python", &py, &plan.worker_dir.to_string_lossy()]), cancel, on_event)?;
    if !st.success() {
        return Err(fail(t4, st.code()));
    }

    std::fs::create_dir_all(&plan.env_dir)?;
    let marker = format!("{{\"wheel\": \"{}\", \"python\": \"{}\", \"unsloth\": {}}}\n", plan.wheel, plan.python_version, unsloth);
    std::fs::write(plan.env_dir.join("ndforge_env.json"), marker)?;
    let _ = std::fs::remove_file(plan.env_dir.join("selftest.json")); // a new environment has to pass the test again
    Ok(())
}
