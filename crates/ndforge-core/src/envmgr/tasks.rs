//! The long jobs of the setup page (install, self-test) as one call each; progress goes to a callback as small JSON objects the window shows:
//! `{"kind": "step"|"log"|"progress"|"done"|"error", ...}`.
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::gpu::cuda_wheel_for;
use super::health::{can_install, evaluate, gather, Check};
use super::jobs::{run_job, WorkerEvent};
use super::setup::{run_setup, Event, SetupPlan};
use super::{env_dir, find_uv, python_path, worker_dir};
use crate::error::{Error, Result};

/// The checklist and the GPU for the setup page.
pub fn status() -> Value {
    let facts = gather();
    let checks: Vec<Check> = evaluate(&facts);
    let gpu = facts.gpus.as_ref().ok().and_then(|g| g.first().cloned());
    json!({
        "checks": checks,
        "can_install": can_install(&checks),
        "gpu": gpu.as_ref().map(|g| json!({"name": g.name, "vram_gb": (g.vram_gb() * 10.0).round() / 10.0, "driver": g.driver, "precision": g.precision(), "blackwell": g.is_blackwell()})),
        "env_installed": facts.env_installed,
        "selftest_passed": facts.selftest_passed,
        "env_dir": env_dir(),
    })
}

pub fn install(cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<()> {
    let facts = gather();
    let checks = evaluate(&facts);
    if let Some(bad) = checks.iter().find(|c| matches!(c.id, "gpu" | "driver" | "vram" | "disk" | "uv") && c.light == super::health::Light::Fail) {
        return Err(Error::new(format!("{}: {} {}", bad.label, bad.detail, bad.fix)));
    }
    let gpu = facts.gpus.as_ref().ok().and_then(|g| g.first().cloned()).ok_or_else(|| Error::new("No graphics card found."))?;
    let wheel = cuda_wheel_for(&gpu).map_err(Error::new)?;
    let plan = SetupPlan {
        env_dir: env_dir(), uv: find_uv().ok_or_else(|| Error::new("uv was not found."))?, worker_dir: worker_dir(),
        wheel: wheel.to_string(), python_version: "3.12".to_string(),
    };
    run_setup(&plan, cancel, &mut |e| on_event(match e {
        Event::Step { index, total, text } => json!({"kind": "step", "index": index, "total": total, "text": text}),
        Event::Log(l) => json!({"kind": "log", "text": l}),
    }))
}

/// Runs the GPU self-test in the worker and remembers a pass (`selftest.json`), which turns the last light green.
pub fn selftest(cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let env = env_dir();
    let py = python_path(&env);
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Press \"Install the training software\" first."));
    }
    let _ = std::fs::remove_file(env.join("selftest.json"));
    let data = run_job(&py, &worker_dir(), &json!({"job": "selftest"}), cancel, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
    }))?;
    std::fs::write(env.join("selftest.json"), serde_json::to_vec_pretty(&data)?)?;
    Ok(data)
}
