//! Training runs: `runs/<name>/` holds config.json (the job), state.json (status), metrics.jsonl (what the charts draw), the checkpoints and the
//! adapter. Starting, stopping (a STOP file the worker honours after the current step, then a kill after a grace period) and resuming.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::catalog::ModelEntry;
use super::plan::{Hyper, Method};
use crate::envmgr::jobs::{run_job, WorkerEvent};
use crate::envmgr::{env_dir, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::paths::config_dir;
use crate::project::Project;

pub fn runs_dir(p: &Project) -> PathBuf {
    p.folder.join("runs")
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(v)?)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

fn read_json(path: &Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

/// The newest saved dataset version: (number, manifest).
pub fn latest_version(p: &Project) -> Option<(u32, Value)> {
    crate::dataset::store::versions(p).into_iter().filter_map(|m| Some((m["version"].as_u64()? as u32, m))).max_by_key(|(n, _)| *n)
}

pub fn data_stats(manifest: &Value) -> super::plan::DataStats {
    let c = &manifest["counts"];
    let (text, chat) = (c["world_book_texts"].as_u64().unwrap_or(0), c["train_chat"].as_u64().unwrap_or(0));
    super::plan::DataStats { examples: text + chat, tokens: manifest["tokens"]["train"].as_u64().unwrap_or(0), text_examples: text, chat_examples: chat }
}

/// A few questions the training shows answers to now and then: identity questions about distinct entries, and one about secrets, so the owner can
/// watch the model learn. Taken from the training file (not the test file).
pub fn probes_from_dataset(train_jsonl: &Path, n: usize) -> Vec<Value> {
    let Ok(text) = std::fs::read_to_string(train_jsonl) else { return Vec::new() };
    let mut out: Vec<Value> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for want in ["identity", "relations", "secrets", "fields"] {
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
            if v["type"] != "chat" || v["builder"] != want || !seen.insert(format!("{want}|{}", v["source"].as_str().unwrap_or(""))) {
                continue;
            }
            let msgs = v["messages"].as_array().cloned().unwrap_or_default();
            let (sys, user) = (msgs.iter().find(|m| m["role"] == "system"), msgs.iter().find(|m| m["role"] == "user"));
            if let (Some(s), Some(u)) = (sys, user) {
                out.push(json!({"system": s["content"], "question": u["content"]}));
                break;
            }
        }
        if out.len() >= n {
            break;
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------------- measured memory

fn measured_file() -> PathBuf {
    config_dir().join("measured.json")
}

/// (peak GB, sequence length, batch) of the last finished run of this model and method on this PC.
pub fn load_measured(model_id: &str, method: Method) -> Option<(f64, u32, u32)> {
    let v = read_json(&measured_file())?;
    let e = &v[model_id][method.key()];
    Some((e["gb"].as_f64()?, e["seq"].as_u64()? as u32, e["batch"].as_u64()? as u32))
}

fn record_measured(model_id: &str, method: Method, gb: f64, seq: u32, batch: u32) {
    let mut v = read_json(&measured_file()).unwrap_or_else(|| json!({}));
    v[model_id][method.key()] = json!({"gb": (gb * 10.0).round() / 10.0, "seq": seq, "batch": batch, "when": chrono::Utc::now().format("%Y-%m-%d").to_string()});
    let _ = write_json(&measured_file(), &v);
}

// ------------------------------------------------------------------------------------------------------------------ runs

pub fn run_name(model_id: &str) -> String {
    format!("{}-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"), model_id)
}

/// The job the worker gets.
pub fn build_job(p: &Project, m: &ModelEntry, method: Method, hyper: &Hyper, version: u32, name: &str, resume: bool) -> Result<Value> {
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    if !train.is_file() {
        return Err(Error::new("The training data is missing. Save a version in Step 2 first."));
    }
    let mut job = json!({
        "job": "train", "model": {"repo": m.repo, "family": m.family}, "method": method, "backend": "auto",
        "dataset": {"train": train, "version": version}, "hyper": hyper, "out": runs_dir(p).join(name), "resume": resume,
        "probes": probes_from_dataset(&train, 3), "probe_every": (hyper.save_steps).max(10), "probe_tokens": 120,
    });
    // two stages: read the world book first, then practise answering (a fixed number of steps keeps one stage)
    if hyper.stages == "book_first" && hyper.max_steps.is_none() {
        job["stages"] = json!([
            {"name": "world_book", "filter": "text", "epochs": hyper.book_epochs},
            {"name": "practice", "filter": "chat", "epochs": hyper.epochs},
        ]);
    }
    Ok(job)
}

fn set_state(dir: &Path, patch: Value) {
    let mut v = read_json(&dir.join("state.json")).unwrap_or_else(|| json!({}));
    if let (Some(o), Some(pa)) = (v.as_object_mut(), patch.as_object()) {
        for (k, x) in pa {
            o.insert(k.clone(), x.clone());
        }
        o.insert("updated".into(), json!(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()));
    }
    let _ = write_json(&dir.join("state.json"), &v);
}

/// Turns what the worker says into the events the window shows, keeps the charts' events in `metrics.jsonl` and the progress in `state.json`.
pub struct Recorder {
    dir: PathBuf,
    metrics: std::fs::File,
    last_step: u64,
    total: u64,
}

impl Recorder {
    pub fn new(dir: &Path) -> Result<Recorder> {
        std::fs::create_dir_all(dir)?;
        Ok(Recorder { dir: dir.to_path_buf(), metrics: std::fs::OpenOptions::new().create(true).append(true).open(dir.join("metrics.jsonl"))?, last_step: 0, total: 0 })
    }

    pub fn from_worker(&mut self, e: WorkerEvent) -> Value {
        match e {
            WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
            WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
            WorkerEvent::Other(mut v) => {
                let kind = v["type"].as_str().unwrap_or("event").to_string();
                v["kind"] = json!(kind);
                self.record(&v);
                v
            }
        }
    }

    /// An event that already has a "kind" (from a remote worker): keep what the charts need.
    pub fn record(&mut self, v: &Value) {
        let kind = v["kind"].as_str().unwrap_or("");
        if matches!(kind, "metric" | "sample" | "start") {
            use std::io::Write;
            let mut line = v.clone();
            line["type"] = json!(kind);
            let _ = writeln!(self.metrics, "{line}");
        }
        if kind == "metric" {
            self.last_step = v["step"].as_u64().unwrap_or(self.last_step);
            self.total = v["total_steps"].as_u64().unwrap_or(self.total);
            set_state(&self.dir, json!({"step": self.last_step, "total_steps": self.total, "loss": v["loss"], "peak_vram_mb": v["vram_mb"]}));
        }
    }
}

/// Runs a training job with the worker until it ends. `stop` (the Stop button) asks the worker to finish the current step, save a checkpoint and
/// exit (a STOP file in `out_dir`); after `grace` it is killed. Returns the worker's result, or an error (a kill is "Cancelled.").
pub fn drive(py: &Path, out_dir: &Path, job: &Value, stop: Arc<AtomicBool>, grace: Duration, on_worker: &mut dyn FnMut(WorkerEvent)) -> Result<Value> {
    std::fs::create_dir_all(out_dir)?;
    let kill = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let watcher = {
        let (kill, done, stop, stop_file) = (kill.clone(), done.clone(), stop, out_dir.join("STOP"));
        std::thread::spawn(move || {
            let mut asked: Option<std::time::Instant> = None;
            while !done.load(Ordering::SeqCst) {
                if stop.load(Ordering::SeqCst) && asked.is_none() {
                    let _ = std::fs::write(&stop_file, b"");
                    asked = Some(std::time::Instant::now());
                }
                if asked.is_some_and(|t| t.elapsed() > grace) {
                    kill.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        })
    };
    let outcome = run_job(py, &worker_dir(), job, &kill, on_worker);
    done.store(true, Ordering::SeqCst);
    let _ = watcher.join();
    let _ = std::fs::remove_file(out_dir.join("STOP"));
    outcome
}

/// Runs the job on this PC to the end, stopping or failing, and keeps the run's state. Every event goes to `on_event` as JSON ({"kind": ...}).
pub fn execute(p: &Project, m: &ModelEntry, method: Method, hyper: &Hyper, name: &str, job: Value, stop: Arc<AtomicBool>, grace: Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let dir = runs_dir(p).join(name);
    std::fs::create_dir_all(&dir)?;
    write_json(&dir.join("config.json"), &job)?;
    set_state(&dir, json!({"name": name, "model_id": m.id, "model": m.name, "method": method, "status": "running", "created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                           "seq": hyper.max_seq, "batch": hyper.batch, "error": null}));
    let mut rec = Recorder::new(&dir)?;
    let outcome = drive(&py, &dir, &job, stop, grace, &mut |e| {
        let ev = rec.from_worker(e);
        on_event(ev);
    });
    finish(&dir, m, method, hyper, outcome, true)
}

/// Writes the final state of a run from the worker's outcome (shared by local and remote runs). `record_memory`: the peak memory is of THIS PC's card.
pub fn finish(dir: &Path, m: &ModelEntry, method: Method, hyper: &Hyper, outcome: Result<Value>, record_memory: bool) -> Result<Value> {
    match outcome {
        Ok(result) => {
            let stopped = result["stopped"].as_bool().unwrap_or(false);
            if stopped {
                set_state(dir, json!({"status": "stopped", "step": result["steps"], "peak_vram_mb": result["peak_vram_mb"]}));
            } else {
                write_json(&dir.join("result.json"), &result)?;
                set_state(dir, json!({"status": "done", "step": result["steps"], "total_steps": result["total_steps"], "loss": result["final_loss"], "peak_vram_mb": result["peak_vram_mb"]}));
                if let Some(mb) = result["peak_vram_mb"].as_f64().filter(|mb| *mb > 0.0 && record_memory) {
                    record_measured(&m.id, method, mb / 1024.0, hyper.max_seq, hyper.batch);
                }
            }
            Ok(result)
        }
        Err(e) if e.0 == "Cancelled." => {
            // killed after the grace period: the last checkpoint is kept and the run can be continued
            set_state(dir, json!({"status": "stopped"}));
            Err(Error::new("Stopped. The run can be continued from its last saved point."))
        }
        Err(e) => {
            set_state(dir, json!({"status": "failed", "error": e.0}));
            Err(e)
        }
    }
}

pub fn set_run_state(dir: &Path, patch: Value) {
    set_state(dir, patch);
}

/// Every run, newest first, with its state.
pub fn list_runs(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(runs_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let mut st = read_json(&e.path().join("state.json"))?;
        st["has_adapter"] = json!(e.path().join("adapter").join("adapter_config.json").is_file());
        st["has_model"] = json!(e.path().join("model").join("config.json").is_file());
        Some(st)
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

/// A run's charts: every metric, sample and start event in order.
pub fn read_metrics(p: &Project, name: &str) -> Vec<Value> {
    if name.contains(['/', '\\']) || name.contains("..") {
        return Vec::new();
    }
    std::fs::read_to_string(runs_dir(p).join(name).join("metrics.jsonl")).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
}

/// A run that was stopped or failed after saving a checkpoint can be continued (a remote run keeps its checkpoints on the server).
pub fn resumable(p: &Project, name: &str) -> bool {
    if read_json(&runs_dir(p).join(name).join("state.json")).is_some_and(|s| s["server"].is_string() && matches!(s["status"].as_str(), Some("stopped" | "failed" | "interrupted"))) {
        return true;
    }
    // one stage keeps its checkpoints directly in checkpoints/, a two-stage run in checkpoints/<stage>/
    let d = runs_dir(p).join(name).join("checkpoints");
    let has = |dir: &Path| std::fs::read_dir(dir).map(|r| r.flatten().any(|e| e.file_name().to_string_lossy().starts_with("checkpoint-"))).unwrap_or(false);
    has(&d) || std::fs::read_dir(&d).map(|r| r.flatten().any(|e| e.path().is_dir() && has(&e.path()))).unwrap_or(false)
}
