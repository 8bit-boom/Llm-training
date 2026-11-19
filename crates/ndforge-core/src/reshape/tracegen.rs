//! "Make thinking answers with the original model" (worker job `trace_gen`, worker/ndforge_worker/tracegen.py): the original (or any model) writes thinking answers for prompts, kept in
//! a jsonl file in the format of a heal's "My own file" general source, so a heal can learn from them.
//!
//! The way is chosen by the worker: the experimental vLLM engine when it is switched on and set up (estimated 5-10x faster than the standard engine: numbers from a research report, not
//! measured on this PC), else llama-server for a GGUF file, else the standard engine. The answers are filtered (the thinking is closed, the answer is not empty, the limit did not cut it, no
//! loop, no garbled text); a second run goes on where the first stopped (by the hash of the prompt). Stop keeps what was written.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::heal_source;
use super::runs::{event_json, models_dir, python, valid_name};
use crate::error::{Error, Result};
use crate::project::Project;

pub const DEFAULT_LIMIT: u32 = 200;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TraceSpec {
    /// the folder name under `models/`: the model that writes the answers is ITS original (`model` overrides it)
    pub name: String,
    /// a model folder, a GGUF file or a Hugging Face id to use instead of the original of `name`
    pub model: Option<String>,
    /// the prompts: a local file (jsonl, json, csv, txt) or a Hugging Face dataset id
    pub source: String,
    /// the column that holds the prompt or the chat (None: found out)
    pub column: Option<String>,
    /// how many prompts to try in this run
    pub limit: Option<u32>,
    /// the longest answer in tokens (the worker's own default when None)
    pub max_new: Option<u32>,
    /// "auto" (default) | "vllm" | "llama" | "hf"
    pub backend: Option<String>,
    /// the vLLM settings, filled in by the window's command from the settings (the page does not send it)
    pub engine: Option<Value>,
}

pub fn trace_dir(p: &Project, name: &str) -> PathBuf {
    models_dir(p).join(name).join("trace_gen")
}

pub fn out_file(p: &Project, name: &str) -> PathBuf {
    trace_dir(p, name).join("answers.jsonl")
}

fn is_file_source(s: &str) -> bool {
    std::path::Path::new(s).is_file() || s.contains('\\') || s.starts_with('/') || [".jsonl", ".json", ".csv", ".txt", ".ndjson"].iter().any(|e| s.to_ascii_lowercase().ends_with(e))
}

pub fn build_job(p: &Project, spec: &TraceSpec) -> Result<Value> {
    valid_name(&spec.name)?;
    let dir = models_dir(p).join(&spec.name);
    if !dir.is_dir() {
        return Err(Error::new("That model is not in this project."));
    }
    let model = match spec.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        Some(m) => m.to_string(),
        None => heal_source::read_origin(&dir)?.teacher,
    };
    let src = spec.source.trim();
    if src.is_empty() {
        return Err(Error::new("Choose the prompts: a file (jsonl, json, csv, txt) or a Hugging Face dataset."));
    }
    let source = if is_file_source(src) {
        if !std::path::Path::new(src).is_file() {
            return Err(Error::new(format!("The file {src} was not found.")));
        }
        json!({"file": src})
    } else {
        json!({"id": src})
    };
    let td = trace_dir(p, &spec.name);
    let backend = spec.backend.as_deref().filter(|b| matches!(*b, "auto" | "vllm" | "llama" | "hf")).unwrap_or("auto");
    let mut job = json!({"job": "trace_gen", "model": model, "backend": backend, "source": source, "out_file": out_file(p, &spec.name), "stop_file": td.join("STOP"),
                         "limit": spec.limit.filter(|l| *l >= 1).unwrap_or(DEFAULT_LIMIT)});
    if let Some(c) = spec.column.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
        job["column"] = json!(c);
    }
    if let Some(n) = spec.max_new.filter(|n| (16..=65536).contains(n)) {
        job["max_new"] = json!(n);
    }
    if let Some(e) = spec.engine.as_ref().filter(|_| backend != "hf" && backend != "llama") {
        let mut e = e.clone();
        e["original"] = json!(spec.model.as_deref().is_none_or(|m| m.trim().is_empty()));       // the original: the pre-quantized checkpoint may be served (vllmquant.pick)
        job["engine"] = e;
    }
    Ok(job)
}

static RUNNING: Mutex<bool> = Mutex::new(false);

/// Runs one job to the end. Events: `progress` (step `trace`), `log`. Returns the worker's result with `name`.
pub fn run(p: &Project, spec: &TraceSpec, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let job = build_job(p, spec)?;
    let py = python()?;
    {
        let mut r = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
        if *r {
            return Err(Error::new("Thinking answers are already being made. Wait until that is done, or stop it."));
        }
        *r = true;
    }
    let dir = trace_dir(p, &spec.name);
    let _ = std::fs::remove_file(dir.join("STOP"));
    let out = crate::train::runs::drive(&py, &dir, &job, stop, Duration::from_secs(60), &mut |e| on_event(event_json(e)));
    *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = false;
    let mut r = out?;
    r["name"] = json!(spec.name);
    Ok(r)
}

/// What is on disk for a made model: `{"file", "count", "rejected", "bytes"}` (counts of lines; both 0 when nothing was made yet).
pub fn state(p: &Project, name: &str) -> Result<Value> {
    valid_name(name)?;
    let f = out_file(p, name);
    let lines = |path: &std::path::Path| std::fs::read_to_string(path).map(|t| t.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0);
    let rej = f.with_file_name("answers.jsonl.rejected.jsonl");
    Ok(json!({"file": f, "count": lines(&f), "rejected": lines(&rej), "bytes": std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0)}))
}
