//! "Measure heal speed" (docs/HEAL_SPEED.md): where does a heal's time go on THIS computer? One worker job, the `distill` bench with the additive keys `profile`, `microbench` and
//! (optionally, it loads the original) `teacher_bench`, and `out_dir`: the worker writes `heal_speed_report.md/.json` there on every end. Nothing runs unless the person presses the button.
//! The research report's speed claims are hypotheses; this is the measurement.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::runs::{bench_base, HealSpec};
use crate::envmgr::jobs::{run_with_grace, WorkerEvent};
use crate::error::Result;
use crate::project::Project;

pub const STEPS: u32 = 6;
pub const WARMUP: u32 = 3;
/// The token budgets of the student's micro-batches the report measures on real rows (0 = one row per forward pass, today's way).
pub const BATCH_TOKENS_SWEEP: [u32; 5] = [0, 2048, 4096, 8192, 16384];
/// The token budgets of the original's answers the tuner measures (0 = the fixed number of rows per forward pass; the Settings default is 16,384).
pub const TEACHER_BUDGETS: [u32; 6] = [0, 4096, 8192, 16384, 32768, 65536];

/// Where the report of a made model is kept: `benchmarks/heal_speed-<model>/`.
pub fn report_dir(p: &Project, name: &str) -> PathBuf {
    p.folder.join("benchmarks").join(format!("heal_speed-{name}"))
}

/// The worker job of the measurement (pure: tested without a worker). `teacher`: also time the original's answers per rows-per-forward (loads the original: slow).
pub fn job_of(base_job: &Value, teacher: Option<(&str, &str)>, out_dir: &std::path::Path, windows_hints: &[String]) -> Value {
    let mut job = base_job.clone();
    let mut bench = json!({"steps": STEPS, "warmup": WARMUP, "targets": "rows", "profile": true, "microbench": true, "out_dir": out_dir,
                              "batch_tokens_sweep": BATCH_TOKENS_SWEEP});
    if let Some((repo, load)) = teacher {
        bench["teacher_bench"] = json!(true);
        job["distill"]["teacher"] = json!({"repo": repo, "load": load});
    }
    if !windows_hints.is_empty() {
        bench["windows_hints"] = json!(windows_hints);
    }
    job["bench"] = bench;
    job
}

/// The token-budget sweep for the tuner (docs/TUNE.md): the same bench job as the speed report, without the profile and the micro-benchmark, with the repeats that measure the noise and the worst-case update
/// that guards the memory. The report is written to `dir`; its JSON is returned.
pub fn tune_sweep(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>, dir: &std::path::Path, stop: Arc<AtomicBool>, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let b = bench_base(p, name, spec, vram_gb)?;
    let _ = std::fs::create_dir_all(dir);
    let mut job = b.job_base(STEPS, WARMUP).job;
    job["bench"] = json!({"steps": 3, "warmup": 1, "targets": "rows", "rows_tokens": 600_000, "out_dir": dir, "batch_tokens_sweep": BATCH_TOKENS_SWEEP, "sweep_repeat": 2, "worst_case": true, "sweep_updates": 2, "sweep_warmup": 1});
    let outcome = run_with_grace(&b.py, &job, &dir.join("STOP"), &stop, Duration::from_secs(60), &mut |e: WorkerEvent| emit(super::runs::event_json(e)));
    let report: Value = std::fs::read(dir.join("heal_speed_report.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    match outcome {
        Ok(_) if report["batch_sweep"]["results"].is_array() => Ok(report),
        Ok(_) => Err(crate::error::Error::new("The speed test ended without the token-budget measurement.")),
        Err(e) if stop.load(std::sync::atomic::Ordering::SeqCst) => {
            let mut r = report;
            r["stopped"] = json!(true);
            let _ = e;
            Ok(r)
        }
        Err(e) => Err(e),
    }
}

/// The original's tokens per forward pass for the tuner (docs/TUNE.md): the worker loads the original as the heal does and answers the same real rows at several token budgets (0 = the fixed rows per pass),
/// with repeats for the noise. Returns the worker's result (`batch_sweep.results` shaped like the student's sweep); `stopped: true` when Stop was pressed.
pub fn tune_teacher_sweep(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>, dir: &std::path::Path, stop: Arc<AtomicBool>, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let b = bench_base(p, name, spec, vram_gb)?;
    let _ = std::fs::create_dir_all(dir);
    let o = super::heal_source::read_origin(&p.folder.join("models").join(name))?;
    let load = b.ch.teacher_load.clone().unwrap_or_else(|| "4bit".into());
    let mut job = b.job_base(STEPS, WARMUP).job;
    job["distill"]["teacher"] = json!({"repo": o.teacher, "load": load});
    job["bench"] = json!({"steps": 1, "warmup": 0, "targets": "rows", "rows_tokens": 600_000, "teacher_sweep": TEACHER_BUDGETS, "sweep_repeat": 2});
    let outcome = run_with_grace(&b.py, &job, &dir.join("STOP"), &stop, Duration::from_secs(60), &mut |e: WorkerEvent| emit(super::runs::event_json(e)));
    match outcome {
        Ok(r) if r["batch_sweep"]["results"].is_array() => Ok(r),
        Ok(_) => Err(crate::error::Error::new("The test of the original's batch ended without a measurement.")),
        Err(e) if stop.load(std::sync::atomic::Ordering::SeqCst) => {
            let _ = e;
            Ok(json!({"stopped": true, "batch_sweep": {"results": []}}))
        }
        Err(e) => Err(e),
    }
}

/// Runs it. Returns `{"report", "report_md", "dir", "result"}`; a failed worker still leaves the report the worker wrote (and its path is returned with `error`).
pub fn run(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>, with_teacher: bool, stop: Arc<AtomicBool>, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let b = bench_base(p, name, spec, vram_gb)?;
    let dir = report_dir(p, name);
    let _ = std::fs::create_dir_all(&dir);
    let o = super::heal_source::read_origin(&p.folder.join("models").join(name))?;
    let load = b.ch.teacher_load.clone().unwrap_or_else(|| "4bit".into());
    let hints = crate::winhints::hints();
    let job = job_of(&b.job_base(STEPS, WARMUP).job, with_teacher.then_some((o.teacher.as_str(), load.as_str())), &dir, &hints);
    emit(json!({"kind": "heal_speed", "text": "Measuring the heal's speed (the student is loaded, a few real steps are profiled)"}));
    let outcome = run_with_grace(&b.py, &job, &dir.join("STOP"), &stop, Duration::from_secs(60), &mut |e: WorkerEvent| {
        emit(super::runs::event_json(e));
    });
    let mut out = json!({"dir": dir, "report": dir.join("heal_speed_report.json"), "report_md": dir.join("heal_speed_report.md"), "windows_hints": hints});
    match outcome {
        Ok(r) => out["result"] = r,
        Err(e) => out["error"] = json!(e.0),
    }
    out["exists"] = json!(dir.join("heal_speed_report.md").is_file());
    Ok(out)
}

/// The saved report of a made model: `{"dir", "markdown", "exists"}`.
pub fn files(p: &Project, name: &str) -> Value {
    let dir = report_dir(p, name);
    let md = std::fs::read_to_string(dir.join("heal_speed_report.md")).ok();
    // the token budget the measurement recommends (never applied by itself): `{"value", "speedup", "why"}` when the report has the measurement, else null
    let recommended = std::fs::read(dir.join("heal_speed_report.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).map(|r| r["batch_sweep"]["recommended"].clone()).filter(|r| r.is_object()).unwrap_or(Value::Null);
    json!({"dir": dir.display().to_string(), "markdown": md, "exists": md.is_some(), "recommended": recommended})
}
