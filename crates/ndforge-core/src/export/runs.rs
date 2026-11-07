//! Export runs: `exports/<run>-<quant>/` holds model-<quant>.gguf, Modelfile, MODEL_CARD.md, meta.json (what it was made from and where it was
//! installed) and export.json (the worker's summary).
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::modelfile::{model_card, modelfile_text, ollama_name, spec_for, CardInfo};
use crate::envmgr::jobs::{run_job, WorkerEvent};
use crate::envmgr::{env_dir, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::runs::runs_dir;

pub const QUANTS: [(&str, &str); 4] = [
    ("q4_k_m", "Q4_K_M: smallest, good quality (recommended)"),
    ("q5_k_m", "Q5_K_M: a little larger, a little better"),
    ("q8_0", "Q8_0: large, very close to the original"),
    ("f16", "F16: the original precision (very large)"),
];

pub fn exports_dir(p: &Project) -> PathBuf {
    p.folder.join("exports")
}

fn read_json(path: &std::path::Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn write_json(path: &std::path::Path, v: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}

/// Runs that can be exported: they have a saved adapter. Newest first.
pub fn exportable_runs(p: &Project) -> Vec<Value> {
    crate::train::runs::list_runs(p).into_iter().filter(|r| r["has_adapter"] == true).collect()
}

pub fn list_exports(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(exports_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let mut meta = read_json(&e.path().join("meta.json"))?;
        let gguf = e.path().join(meta["gguf_file"].as_str()?);
        meta["gguf_exists"] = json!(gguf.is_file());
        meta["dir"] = json!(e.path());
        meta["size_mb"] = json!(std::fs::metadata(&gguf).map(|m| (m.len() as f64 / 1048576.0 * 10.0).round() / 10.0).ok());
        Some(meta)
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

/// Makes the GGUF of a run in the worker and writes the Modelfile and the model card next to it.
pub fn execute(p: &Project, run: &str, quant: &str, num_ctx: u32, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    if !QUANTS.iter().any(|(q, _)| *q == quant) {
        return Err(Error::new(format!("Unknown file size: {quant}")));
    }
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let rdir = runs_dir(p).join(run);
    let adapter = rdir.join("adapter");
    if !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new("This run has no saved model to export."));
    }
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of this run are missing."))?;
    let summary = read_json(&rdir.join("summary.json")).unwrap_or(Value::Null);
    let (repo, family) = (cfg["model"]["repo"].as_str().unwrap_or("").to_string(), cfg["model"]["family"].as_str().unwrap_or("generic").to_string());
    let name = format!("{run}-{quant}");
    let out = exports_dir(p).join(&name);
    std::fs::create_dir_all(&out)?;
    let job = json!({"job": "export", "adapter": adapter, "base": {"repo": repo, "family": family}, "out": out, "quant": quant, "backend": "auto", "gguf": true,
                     "max_seq": cfg["hyper"]["max_seq"]});
    let result = run_job(&py, &worker_dir(), &job, kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    let gguf = result["gguf"].as_str().map(PathBuf::from).ok_or_else(|| Error::new("No GGUF file was made."))?;
    let gguf_file = gguf.file_name().and_then(|n| n.to_str()).unwrap_or("model.gguf").to_string();
    let oname = ollama_name(&p.world_name, "");
    std::fs::write(out.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&family, num_ctx)))?;
    let version = cfg["dataset"]["version"].as_u64().unwrap_or(0);
    std::fs::write(out.join("MODEL_CARD.md"), model_card(&CardInfo {
        world: &p.world_name, model_name: &oname, base_repo: &repo, method: cfg["method"].as_str().unwrap_or("lora"), quant,
        dataset_version: version, steps: summary["steps"].as_u64(), final_loss: summary["final_loss"].as_f64(), ollama_name: &oname,
    }))?;
    let meta = json!({"name": name, "run": run, "quant": quant, "family": family, "base_repo": repo, "gguf_file": gguf_file, "ollama_name": oname, "num_ctx": num_ctx,
                      "created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), "world": p.world_name, "installed": []});
    write_json(&out.join("meta.json"), &meta)?;
    Ok(meta)
}

/// Remember that an export was installed into a server under a name.
pub fn record_install(p: &Project, export: &str, url: &str, model: &str) -> Result<()> {
    let f = exports_dir(p).join(export).join("meta.json");
    let mut meta = read_json(&f).ok_or_else(|| Error::new("That export no longer exists."))?;
    let entry = json!({"url": url, "model": model, "when": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()});
    let mut list: Vec<Value> = meta["installed"].as_array().cloned().unwrap_or_default().into_iter().filter(|e| !(e["url"] == entry["url"] && e["model"] == entry["model"])).collect();
    list.push(entry);
    meta["installed"] = json!(list);
    write_json(&f, &meta)
}

pub fn find_export(p: &Project, export: &str) -> Result<(PathBuf, Value)> {
    if export.contains(['/', '\\']) || export.contains("..") {
        return Err(Error::new("Unknown export."));
    }
    let dir = exports_dir(p).join(export);
    let meta = read_json(&dir.join("meta.json")).ok_or_else(|| Error::new("That export no longer exists."))?;
    let gguf = dir.join(meta["gguf_file"].as_str().unwrap_or("model.gguf"));
    if !gguf.is_file() {
        return Err(Error::new("The GGUF file of that export is missing."));
    }
    Ok((gguf, meta))
}
