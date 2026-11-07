//! Running a job in the Python worker: write the job file, start `python -m ndforge_worker`, read its JSON lines, stop it on request.
//! Protocol: worker/ndforge_worker/protocol.py.
use std::path::Path;
use std::process::Command;
use std::sync::atomic::AtomicBool;

use serde_json::Value;

use super::process::{run_lines, Stream};
use crate::error::{Error, Result};

/// What the worker reports while it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerEvent {
    Log(String),
    Progress { step: String, value: f64, text: String },
}

/// Runs `job` and returns the worker's result data. A worker error becomes `Err` with the worker's plain-language message.
pub fn run_job(python: &Path, worker_dir: &Path, job: &Value, cancel: &AtomicBool, on_event: &mut dyn FnMut(WorkerEvent)) -> Result<Value> {
    let dir = std::env::temp_dir().join(format!("ndforge-job-{}-{}", std::process::id(), chrono::Utc::now().timestamp_micros()));
    std::fs::create_dir_all(&dir)?;
    let file = dir.join("job.json");
    std::fs::write(&file, serde_json::to_vec(job)?)?;
    let mut cmd = Command::new(python);
    cmd.args(["-X", "utf8", "-m", "ndforge_worker"]).arg(&file).current_dir(worker_dir)
        .env("PYTHONPATH", worker_dir).env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8");
    let mut result: Option<Value> = None;
    let mut failure: Option<String> = None;
    let status = run_lines(cmd, cancel, &mut |stream: Stream, line: &str| {
        if stream == Stream::Err {
            on_event(WorkerEvent::Log(line.to_string()));
            return;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(v) => match v["type"].as_str() {
                Some("log") => on_event(WorkerEvent::Log(v["text"].as_str().unwrap_or("").to_string())),
                Some("progress") => on_event(WorkerEvent::Progress {
                    step: v["step"].as_str().unwrap_or("").to_string(), value: v["value"].as_f64().unwrap_or(0.0), text: v["text"].as_str().unwrap_or("").to_string(),
                }),
                Some("result") => result = Some(v["data"].clone()),
                Some("error") => failure = Some(v["message"].as_str().unwrap_or("The worker reported an error.").to_string()),
                _ => on_event(WorkerEvent::Log(line.to_string())),
            },
            Err(_) => on_event(WorkerEvent::Log(line.to_string())),
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    let status = status?;
    if let Some(msg) = failure {
        return Err(Error::new(msg));
    }
    match (status.success(), result) {
        (true, Some(r)) => Ok(r),
        (true, None) => Err(Error::new("The worker finished without an answer.")),
        (false, _) => Err(Error::new(format!("The worker stopped unexpectedly (exit code {}).", status.code().unwrap_or(-1)))),
    }
}
