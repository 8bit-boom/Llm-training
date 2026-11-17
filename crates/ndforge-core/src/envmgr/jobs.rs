//! Running a job in the Python worker: write the job file, start `python -m ndforge_worker`, read its JSON lines, stop it on request.
//! Protocol: worker/ndforge_worker/protocol.py.
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::Value;

use super::process::{run_watched, Item, Stream};
use crate::error::{Error, Result};

/// The worker's error code when llama.cpp (or a part of it) is not there, and the mark the error text then ends with.
pub const LLAMA_CPP_MISSING: &str = "llama_cpp_missing";
pub const LLAMA_CPP_MISSING_MARK: &str = "[llama_cpp_missing]";

/// Worker error codes the window can offer a button for. The error text then ends with ` [fix:<code>]` (after the `(details: ...)` part, before nothing else): the window
/// strips the mark for display and shows the buttons (`fixActions` in ui/app.js). `[llama_cpp_missing]` is the older mark of the same kind and stays as it was.
pub const FIX_CODES: [&str; 5] = ["disk_full", "distill_oom", "distill_unstable", "reshape_refused", "bad_job"];
/// The mark for a text whose worker said a Python library is missing (any job: the worker explains it the same way everywhere).
pub const FIX_MISSING_MODULE: &str = "[fix:missing_module]";

/// The ` [fix:...]` mark for a worker error, or None.
pub fn fix_mark(code: &str, text: &str) -> Option<String> {
    let low = text.to_lowercase();
    if FIX_CODES.contains(&code) {
        return Some(format!("[fix:{code}]"));
    }
    ["no module named", "training library is missing", "training software is incomplete"].iter().any(|m| low.contains(m)).then(|| FIX_MISSING_MODULE.to_string())
}

/// What the worker reports while it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerEvent {
    Log(String),
    Progress { step: String, value: f64, text: String },
    /// any other event the worker sends (start, metric, sample ...), as it came
    Other(Value),
}

/// The environment variables a job gets besides the usual ones: HF_TOKEN (the person's Hugging Face token, kept with the other secrets) for the jobs that read
/// general text from the Hub. The token is never part of the job file.
pub fn job_env(job: &Value) -> Vec<(String, String)> {
    let reads_general = match job["job"].as_str() {
        Some("general_check") => true,
        Some("distill_cache") => job["data"]["general"].is_object(),
        Some("distill") => job["dataset"]["general"].is_object(),
        Some("export") => job["imatrix"]["enabled"] == true,
        Some("imatrix_make") => true,           // the general text, and a model that is a Hugging Face id (a gated one needs the token)
        _ => false,
    };
    if !reads_general {
        return Vec::new();
    }
    crate::secrets::get(crate::general::TOKEN_SECRET).map(|t| vec![("HF_TOKEN".to_string(), t)]).unwrap_or_default()
}

/// How many minutes without any output make the program say "no sign of progress": the job key `stall_minutes`, else 60 for the jobs that are quiet for long
/// stretches (the teacher's answers, an export, a matrix) and 20 for the others. The job is only warned about, never stopped.
pub fn stall_minutes(job: &Value) -> f64 {
    if let Some(m) = job["stall_minutes"].as_f64().filter(|m| *m > 0.0) {
        return m;
    }
    match job["job"].as_str() {
        Some("distill_cache") | Some("export") | Some("imatrix_make") => 60.0,
        _ => 20.0,
    }
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
    for (k, v) in job_env(job) {
        cmd.env(k, v);
    }
    // the llama.cpp this program installed (docs/LLAMA_CPP_INSTALL.md): the worker's finder looks there after an explicit LLAMA_CPP_DIR
    for (k, v) in super::llamacpp::worker_env(&super::llamacpp::Config::from_env()) {
        cmd.env(k, v);
    }
    let scrub = crate::unsloth::client::scrub;
    let mut log = super::joblog::JobLog::open(job);
    let mut result: Option<Value> = None;
    let mut failure: Option<String> = None;
    let mut failure_code = String::new();
    let stall = stall_minutes(job);
    let status = run_watched(cmd, cancel, Some(Duration::from_secs_f64(stall * 60.0)), &mut |item: Item| {
        let (stream, line) = match item {
            Item::Line(stream, line) => (stream, line),
            Item::Stall { periods } => {
                let minutes = stall * periods as f64;
                log.line("stall", &format!("no output for {minutes} minutes"));
                on_event(WorkerEvent::Other(serde_json::json!({"type": "stall", "kind": "stall", "minutes": if minutes.fract() == 0.0 { serde_json::json!(minutes as u64) } else { serde_json::json!(minutes) }})));
                return;
            }
        };
        log.line(if stream == Stream::Err { "err" } else { "out" }, line);
        if stream == Stream::Err {
            on_event(WorkerEvent::Log(scrub(line)));
            return;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(v) => match v["type"].as_str() {
                Some("log") => on_event(WorkerEvent::Log(scrub(v["text"].as_str().unwrap_or("")))),
                Some("progress") => on_event(WorkerEvent::Progress {
                    step: v["step"].as_str().unwrap_or("").to_string(), value: v["value"].as_f64().unwrap_or(0.0), text: v["text"].as_str().unwrap_or("").to_string(),
                }),
                Some("result") => result = Some(v["data"].clone()),
                Some("error") => {
                    failure = Some(scrub(v["message"].as_str().unwrap_or("The worker reported an error.")));
                    failure_code = v["code"].as_str().unwrap_or("").to_string();
                }
                Some(_) => on_event(WorkerEvent::Other(v.clone())),
                None => on_event(WorkerEvent::Log(scrub(line))),
            },
            Err(_) => on_event(WorkerEvent::Log(scrub(line))),
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    let status = match status {
        Ok(s) => s,
        Err(e) => {
            log.finish(if e.is_cancelled() { "stopped on request" } else { "could not run" }, "", if e.is_cancelled() { None } else { Some(&e.0) });
            return Err(e);
        }
    };
    let outcome: Result<Value> = if let Some(msg) = failure.clone() {
        Err(Error::new(msg))
    } else {
        match (status.success(), result) {
            (true, Some(r)) => Ok(r),
            (true, None) => Err(Error::new("The worker finished without an answer.")),
            (false, _) => Err(Error::new(format!("The worker stopped unexpectedly (exit code {}).", status.code().unwrap_or(-1)))),
        }
    };
    log.finish(&format!("exit code {}", status.code().map(|c| c.to_string()).unwrap_or_else(|| "none".into())), &failure_code, outcome.as_ref().err().map(|e| e.0.as_str()));
    match outcome {
        Ok(r) => Ok(r),
        // the text points at the saved log; the page offers "Install llama.cpp now" when it finds the mark, which stays at the very end
        Err(e) => {
            let mut text = e.0;
            // a stop answered by the worker ("Cancelled.") stays exactly that word: callers compare it
            if let Some(name) = log.short_name().filter(|_| text != crate::error::CANCELLED) {
                text = format!("{text} (details: {name})");
            }
            if failure_code == LLAMA_CPP_MISSING {
                text = format!("{text} {LLAMA_CPP_MISSING_MARK}");
            } else if let Some(mark) = fix_mark(&failure_code, &text).filter(|_| failure.is_some()) {
                text = format!("{text} {mark}");
            }
            Err(Error::new(text))
        }
    }
}

/// Runs `job` like `run_job`, but a stop request (`stop` becomes true) is first passed on politely: the file `stop_file` is written, which a worker that looks for it answers
/// by saving what it has and ending by itself. Only when it is still running `grace` later is it killed (its error is then "Cancelled."). The file is removed before the start
/// (a left-over one would stop the next job at once) and afterwards.
pub fn run_with_grace(py: &Path, job: &Value, stop_file: &Path, stop: &AtomicBool, grace: Duration, on_event: &mut dyn FnMut(WorkerEvent)) -> Result<Value> {
    if let Some(dir) = stop_file.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let _ = std::fs::remove_file(stop_file);
    let (kill, done) = (AtomicBool::new(false), AtomicBool::new(false));
    let result = std::thread::scope(|s| {
        s.spawn(|| {
            let mut asked: Option<std::time::Instant> = None;
            while !done.load(Ordering::SeqCst) {
                if stop.load(Ordering::SeqCst) && asked.is_none() {
                    let _ = std::fs::write(stop_file, b"");
                    asked = Some(std::time::Instant::now());
                }
                if asked.is_some_and(|t| t.elapsed() > grace) {
                    kill.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let r = run_job(py, &super::worker_dir(), job, &kill, on_event);
        done.store(true, Ordering::SeqCst);
        r
    });
    let _ = std::fs::remove_file(stop_file);
    result
}
