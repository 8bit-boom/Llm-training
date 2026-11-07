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

const TOTAL: usize = 3;

pub fn run_setup(plan: &SetupPlan, cancel: &AtomicBool, on_event: &mut dyn FnMut(Event)) -> Result<()> {
    let py = python_path(&plan.env_dir);
    let env_s = plan.env_dir.to_string_lossy().to_string();
    let py_s = py.to_string_lossy().to_string();
    let wheel_url = format!("https://download.pytorch.org/whl/{}", plan.wheel);
    let steps: Vec<(String, Vec<String>)> = vec![
        ("Creating the Python environment".into(), vec!["venv".into(), "--python".into(), plan.python_version.clone(), "--seed".into(), env_s.clone()]),
        (format!("Installing PyTorch for NVIDIA ({}) - several GB, please wait", plan.wheel),
         vec!["pip".into(), "install".into(), "--python".into(), py_s.clone(), "torch".into(), "--index-url".into(), wheel_url]),
        ("Installing the ND Lore Forge worker".into(), vec!["pip".into(), "install".into(), "--python".into(), py_s.clone(), plan.worker_dir.to_string_lossy().to_string()]),
    ];
    std::fs::create_dir_all(plan.env_dir.parent().unwrap_or(&plan.env_dir))?;
    for (i, (text, args)) in steps.into_iter().enumerate() {
        on_event(Event::Step { index: i + 1, total: TOTAL, text: text.clone() });
        let mut cmd = Command::new(&plan.uv);
        cmd.args(&args);
        let status = run_lines(cmd, cancel, &mut |_: Stream, line: &str| on_event(Event::Log(line.to_string())))?;
        if !status.success() {
            return Err(Error::new(format!("The step \"{text}\" failed (exit code {}). The log above shows why; fix that and press the button again.", status.code().unwrap_or(-1))));
        }
    }
    std::fs::create_dir_all(&plan.env_dir)?;
    let marker = format!("{{\"wheel\": \"{}\", \"python\": \"{}\"}}\n", plan.wheel, plan.python_version);
    std::fs::write(plan.env_dir.join("ndforge_env.json"), marker)?;
    let _ = std::fs::remove_file(plan.env_dir.join("selftest.json")); // a new environment has to pass the test again
    Ok(())
}
