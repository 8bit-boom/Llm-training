//! Export runs: `exports/<run>-<quant>/` holds model-<quant>.gguf, Modelfile, MODEL_CARD.md, meta.json (what it was made from and where it was
//! installed) and export.json (the worker's summary).
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::modelfile::{model_card, modelfile_text, ollama_name, spec_for, CardInfo};
use crate::envmgr::jobs::{run_job, WorkerEvent};
use crate::envmgr::{env_dir, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::runs::runs_dir;

pub const QUANTS: [(&str, &str); 5] = [
    ("q4_k_m", "Q4_K_M: smallest, good quality (recommended)"),
    ("q5_k_m", "Q5_K_M: a little larger, a little better"),
    ("q6_k", "Q6_K: larger again, very close to Q8"),
    ("q8_0", "Q8_0: large, very close to the original"),
    ("f16", "F16: the original precision (very large)"),
];

/// The optional importance matrix of a GGUF export (docs/IMATRIX.md): off by default. llama.cpp measures on calibration text which weights matter most and
/// rounds the sizes below Q8_0 with more care. `chunks` is the amount of calibration text (chunks of IMATRIX_CTX tokens); `reuse` takes a saved matrix of the
/// same model and data instead of measuring again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImatrixOptions {
    pub enabled: bool,
    pub chunks: u32,
    pub reuse: bool,
    /// where the general text comes from; None = the program's default
    #[serde(default)]
    pub general: Option<crate::general::GeneralSource>,
}

impl Default for ImatrixOptions {
    fn default() -> Self {
        ImatrixOptions { enabled: false, chunks: IMATRIX_DEFAULT_CHUNKS, reuse: true, general: None }
    }
}

pub const IMATRIX_CHUNKS: [u32; 3] = [64, 128, 256];
pub const IMATRIX_DEFAULT_CHUNKS: u32 = 128;
/// The context of one calibration chunk (tokens).
pub const IMATRIX_CTX: u32 = 512;
/// About this share of the calibration text is general text, the rest is the project's own data.
pub const IMATRIX_GENERAL_SHARE: f64 = 0.2;
/// After Stop the worker has this long to end the matrix step by itself (it kills llama-imatrix at once); then the worker is killed.
const IMATRIX_STOP_GRACE: Duration = Duration::from_secs(20);

/// Does the importance matrix do anything for this size? Not for Q8_0 and the 16-bit files.
pub fn imatrix_helps(quant: &str) -> bool {
    !matches!(quant, "q8_0" | "f16" | "bf16")
}

/// A rough guess of how long the matrix takes, for the export page: the text is `chunks` x `ctx` tokens read once through the 16-bit model.
/// **A guess, not a measurement** (docs/IMATRIX.md "Not verified yet"): about 0.15 s per chunk per billion parameters on a graphics card, 30 times that on the processor.
pub fn imatrix_estimate(chunks: u32, params_b: Option<f64>) -> Value {
    let b = params_b.filter(|b| *b > 0.0).unwrap_or(9.0);
    let gpu = chunks as f64 * 0.15 * b / 60.0 + 1.0;
    json!({"chunks": chunks, "ctx": IMATRIX_CTX, "tokens": chunks * IMATRIX_CTX, "gpu_minutes": gpu.ceil(), "cpu_minutes": (gpu * 30.0).ceil()})
}

/// Folders of earlier exports of `run` that hold a saved matrix (imatrix.gguf + imatrix.json made for this run), newest first.
pub fn saved_imatrices(p: &Project, run: &str) -> Vec<(PathBuf, Value)> {
    let mut found: Vec<(PathBuf, Value)> = std::fs::read_dir(exports_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let info = read_json(&e.path().join("imatrix.json"))?;
        (info["model"] == run && e.path().join("imatrix.gguf").is_file()).then(|| (e.path(), info))
    }).collect()).unwrap_or_default();
    found.sort_by(|a, b| b.1["created"].as_str().cmp(&a.1["created"].as_str()));
    found
}

/// What the export page needs for the importance matrix: the choices, and per run the saved matrix (if any) and the time guesses for each amount of text.
pub fn imatrix_state(p: &Project) -> Value {
    let mut saved = serde_json::Map::new();
    let mut estimates = serde_json::Map::new();
    for r in exportable_runs(p) {
        let Some(run) = r["name"].as_str() else { continue };
        if let Some((dir, info)) = saved_imatrices(p, run).into_iter().next() {
            saved.insert(run.to_string(), json!({"dir": dir, "chunks": info["chunks"], "ctx": info["ctx"], "created": info["created"], "general_used": info["general_used"], "data_version": info["data_version"]}));
        }
        let params = r["model_id"].as_str().and_then(crate::train::catalog::find).map(|m| m.params_b);
        let mut per = serde_json::Map::new();
        for c in IMATRIX_CHUNKS {
            per.insert(c.to_string(), imatrix_estimate(c, params));
        }
        estimates.insert(run.to_string(), Value::Object(per));
    }
    json!({"chunks": IMATRIX_CHUNKS, "default_chunks": IMATRIX_DEFAULT_CHUNKS, "ctx": IMATRIX_CTX, "general_share": IMATRIX_GENERAL_SHARE,
           "no_matrix_for": ["q8_0", "f16"], "saved": saved, "estimates": estimates, "general": crate::general::state(Some(p))})
}

/// The job's "imatrix" block (None when it is off or no GGUF is made). `out` is the folder the worker writes into.
pub fn imatrix_block(p: &Project, run: &str, cfg: &Value, opts: &ImatrixOptions, out: &std::path::Path) -> Result<Option<Value>> {
    if !opts.enabled {
        return Ok(None);
    }
    if !IMATRIX_CHUNKS.contains(&opts.chunks) {
        return Err(Error::new(format!("Unknown amount of calibration text: {} chunks. Choose 64, 128 or 256.", opts.chunks)));
    }
    let version = cfg["dataset"]["version"].as_u64().unwrap_or(0);
    let train = cfg["dataset"]["train"].as_str().map(PathBuf::from).unwrap_or_else(|| p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl"));
    let general = crate::general::job_block(&crate::general::resolve(opts.general.as_ref(), true)?, IMATRIX_GENERAL_SHARE);
    Ok(Some(json!({"enabled": true, "general": general, "seed": crate::general::SEED, "model": run, "data_version": version, "train": train, "chunks": opts.chunks, "ctx": IMATRIX_CTX, "general_share": IMATRIX_GENERAL_SHARE,
                   "reuse": opts.reuse, "reuse_dirs": saved_imatrices(p, run).into_iter().map(|(d, _)| d).collect::<Vec<_>>(), "stop_file": out.join("STOP")})))
}

/// Runs the export job. With the importance matrix the Stop button first writes a STOP file (the worker kills llama-imatrix and ends cleanly) and kills the
/// worker only after IMATRIX_STOP_GRACE; without it the worker is killed at once, as before.
fn run_export_job(py: &std::path::Path, job: &Value, out: &std::path::Path, graceful: Option<Duration>, kill: &AtomicBool, on_event: &mut dyn FnMut(WorkerEvent)) -> Result<Value> {
    let Some(grace) = graceful else { return run_job(py, &worker_dir(), job, kill, on_event) };
    let (inner, done, stop_file) = (AtomicBool::new(false), AtomicBool::new(false), out.join("STOP"));
    let _ = std::fs::remove_file(&stop_file);
    let result = std::thread::scope(|s| {
        s.spawn(|| {
            let mut asked: Option<Instant> = None;
            while !done.load(Ordering::SeqCst) {
                if kill.load(Ordering::SeqCst) && asked.is_none() {
                    let _ = std::fs::write(&stop_file, b"");
                    asked = Some(Instant::now());
                }
                if asked.is_some_and(|t| t.elapsed() > grace) {
                    inner.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        let r = run_job(py, &worker_dir(), job, &inner, on_event);
        done.store(true, Ordering::SeqCst);
        r
    });
    let _ = std::fs::remove_file(&stop_file);
    result
}

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
    crate::train::runs::list_runs(p).into_iter().filter(|r| r["has_adapter"] == true || r["has_model"] == true).collect()
}

pub fn list_exports(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(exports_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let mut meta = read_json(&e.path().join("meta.json"))?;
        // a "merged model only" export has no GGUF
        let gguf = meta["gguf_file"].as_str().map(|f| e.path().join(f));
        meta["gguf_exists"] = json!(gguf.as_ref().is_some_and(|g| g.is_file()));
        meta["dir"] = json!(e.path());
        meta["size_mb"] = json!(gguf.as_ref().and_then(|g| std::fs::metadata(g).ok()).map(|m| (m.len() as f64 / 1048576.0 * 10.0).round() / 10.0));
        meta["has_vision"] = json!(meta["mmproj_file"].as_str().is_some_and(|f| e.path().join(f).is_file()));
        meta["merged_exists"] = json!(meta["merged_dir"].as_str().is_some_and(|d| std::path::Path::new(d).join("config.json").is_file()));
        Some(meta)
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

/// Makes the GGUF of a run in the worker and writes the Modelfile and the model card next to it.
pub fn execute(p: &Project, run: &str, quant: &str, num_ctx: u32, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    execute_what(p, run, quant, num_ctx, "gguf", kill, on_event)
}

/// What an export makes: "gguf" (GGUF + Modelfile), "gguf_merged" (also keeps the merged 16-bit model as safetensors, 10 to 20 GB) or "merged" (only that model).
pub const WHAT: [&str; 3] = ["gguf", "gguf_merged", "merged"];

pub fn execute_what(p: &Project, run: &str, quant: &str, num_ctx: u32, what: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let mut v = execute_many(p, run, &[quant.to_string()], num_ctx, what, kill, on_event)?;
    Ok(v.remove(0))
}

/// Several GGUF sizes of one run in one go: the model is merged and converted once, then shrunk to each size. Every size becomes its own export
/// (`exports/<run>-<size>/`, with its own Modelfile and meta). With "gguf_merged" the merged model is kept in the first one. Returns the metas.
pub fn execute_many(p: &Project, run: &str, quants: &[String], num_ctx: u32, what: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    execute_many_with(p, run, quants, num_ctx, what, &ImatrixOptions::default(), kill, on_event)
}

/// `execute_many` with the options of the importance matrix (only used for a GGUF).
#[allow(clippy::too_many_arguments)]
pub fn execute_many_with(p: &Project, run: &str, quants: &[String], num_ctx: u32, what: &str, imatrix: &ImatrixOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    if !WHAT.contains(&what) {
        return Err(Error::new(format!("Unknown export kind: {what}")));
    }
    let (want_gguf, keep_merged) = (what != "merged", what != "gguf");
    let mut list: Vec<String> = Vec::new();
    for q in quants {
        if !QUANTS.iter().any(|(id, _)| id == q) {
            return Err(Error::new(format!("Unknown file size: {q}")));
        }
        if !list.contains(q) {
            list.push(q.clone());
        }
    }
    if list.is_empty() {
        return Err(Error::new("Choose at least one file size."));
    }
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let rdir = runs_dir(p).join(run);
    let adapter = rdir.join("adapter");
    let full = rdir.join("model");
    let is_full = full.join("config.json").is_file();
    if !is_full && !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new("This run has no saved model to export."));
    }
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of this run are missing."))?;
    let summary = read_json(&rdir.join("summary.json")).unwrap_or(Value::Null);
    let (repo, family) = (cfg["model"]["repo"].as_str().unwrap_or("").to_string(), cfg["model"]["family"].as_str().unwrap_or("generic").to_string());
    // one size: the worker writes straight into its export; several: into a work folder, from which each size moves into its own export
    let list = if want_gguf { list } else { list[..1].to_vec() };
    let many = want_gguf && list.len() > 1;
    let name_of = |q: &str| if want_gguf { format!("{run}-{q}") } else { format!("{run}-merged") };
    let out = if many { exports_dir(p).join(format!("{run}-gguf-work")) } else { exports_dir(p).join(name_of(&list[0])) };
    let _ = std::fs::remove_dir_all(if many { out.clone() } else { out.join("__none__") });
    std::fs::create_dir_all(&out)?;
    // a base made by "Slim or grow a model" keeps the original's vision part: its vision file comes from the original model
    let origin = read_json(&std::path::Path::new(&repo).join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from));
    let mut job = json!({"job": "export", "base": {"repo": repo, "family": family}, "out": out, "quant": list[0], "quants": list, "backend": "auto", "gguf": want_gguf, "keep_merged": keep_merged,
                     "max_seq": cfg["hyper"]["max_seq"]});
    if let Some(o) = &origin {
        job["base"]["vision_repo"] = json!(o);
    }
    let matrix = if want_gguf { imatrix_block(p, run, &cfg, imatrix, &out)? } else { None };
    if let Some(m) = &matrix {
        job["imatrix"] = m.clone();
    }
    // a full fine-tune saved the whole model (nothing to merge); a LoRA run saved an adapter
    if is_full {
        job["full_model"] = json!(full);
    } else {
        job["adapter"] = json!(adapter);
    }
    let result = run_export_job(&py, &job, &out, matrix.is_some().then_some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    let oname = ollama_name(&p.model_label(), "");
    let created = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut merged_dir = result["merged_dir"].as_str().filter(|_| keep_merged).map(String::from);
    if !want_gguf {
        if merged_dir.is_none() {
            return Err(Error::new("No merged model was made."));
        }
        let meta = json!({"name": name_of(""), "run": run, "quant": null, "family": family, "base_repo": repo, "gguf_file": null, "merged_dir": merged_dir, "ollama_name": oname, "num_ctx": num_ctx,
                          "created": created, "world": p.display_name(), "installed": []});
        write_json(&out.join("meta.json"), &meta)?;
        return Ok(vec![meta]);
    }
    let license = crate::train::catalog::models().into_iter().find(|m| m.repo == repo).map(|m| m.license).unwrap_or_else(|| "see the model page for its licence".to_string());
    let version = cfg["dataset"]["version"].as_u64().unwrap_or(0);
    let mut metas = Vec::new();
    for (i, quant) in list.iter().enumerate() {
        let gguf = result["ggufs"][quant.as_str()].as_str().or_else(|| if list.len() == 1 { result["gguf"].as_str() } else { None }).map(PathBuf::from)
            .ok_or_else(|| Error::new(format!("No GGUF file was made for {quant}.")))?;
        let dir = exports_dir(p).join(name_of(quant));
        if dir != out {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir)?;
        }
        let gguf_file = gguf.file_name().and_then(|n| n.to_str()).unwrap_or("model.gguf").to_string();
        if dir != out {
            std::fs::rename(&gguf, dir.join(&gguf_file))?;
        }
        // the merged model stays once: in the first export
        if let (true, Some(m)) = (i == 0 && dir != out, merged_dir.as_ref()) {
            let to = dir.join("merged");
            std::fs::rename(m, &to)?;
            merged_dir = Some(to.display().to_string());
        }
        let keep_here = if i == 0 { merged_dir.clone() } else { None };
        // the base model's vision file (images; the language layers are all that training changed): once per export, next to its GGUF
        let mmproj_file = match result["mmproj"].as_str().map(PathBuf::from).filter(|m| m.is_file()) {
            Some(m) => {
                let name = m.file_name().and_then(|n| n.to_str()).unwrap_or("mmproj-F16.gguf").to_string();
                if dir != out {
                    if i == list.len() - 1 {
                        std::fs::rename(&m, dir.join(&name))?;
                    } else {
                        std::fs::copy(&m, dir.join(&name))?;
                    }
                }
                std::fs::write(dir.join("VISION.md"), vision_readme(&gguf_file, &name))?;
                Some(name)
            }
            None => None,
        };
        // the importance matrix (and what it was made from) sits next to every GGUF of the batch, so a later export finds it whichever size is kept
        let matrix_file = match result["imatrix"]["file"].as_str().map(PathBuf::from).filter(|m| m.is_file()) {
            Some(m) => {
                let info = m.with_file_name("imatrix.json");
                if dir != out {
                    for f in [&m, &info].into_iter().filter(|f| f.is_file()) {
                        let to = dir.join(f.file_name().unwrap_or_default());
                        if i == list.len() - 1 { std::fs::rename(f, to)?; } else { std::fs::copy(f, to)?; }
                    }
                }
                Some("imatrix.gguf")
            }
            None => None,
        };
        std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&family, num_ctx)))?;
        std::fs::write(dir.join("MODEL_CARD.md"), model_card(&CardInfo {
            world: &p.display_name(), model_name: &oname, base_repo: &repo, license: &license, method: cfg["method"].as_str().unwrap_or("lora"), quant,
            dataset_version: version, steps: summary["steps"].as_u64(), final_loss: summary["final_loss"].as_f64(), ollama_name: &oname, derived_from: origin.as_deref(),
        }))?;
        let meta = json!({"name": name_of(quant), "run": run, "quant": quant, "family": family, "base_repo": repo, "gguf_file": gguf_file, "mmproj_file": mmproj_file, "modalities": result["modalities"], "vision_note": result["vision_note"], "merged_dir": keep_here, "ollama_name": oname, "num_ctx": num_ctx,
                          "created": created, "world": p.display_name(), "installed": []});
        let mut meta = meta;
        meta["imatrix"] = imatrix_meta(matrix.as_ref(), &result["imatrix"], quant, matrix_file);
        write_json(&dir.join("meta.json"), &meta)?;
        metas.push(meta);
    }
    if many {
        let _ = std::fs::remove_dir_all(&out);
    }
    Ok(metas)
}

/// What meta.json says about the importance matrix of one size: null when it was not asked for. `applied` is whether THIS size was made with it.
fn imatrix_meta(asked: Option<&Value>, report: &Value, quant: &str, file: Option<&str>) -> Value {
    let Some(asked) = asked else { return Value::Null };
    let applied = report["applied"].as_array().is_some_and(|a| a.iter().any(|q| q == quant));
    let reason = report["not_applied"][quant].as_str();
    json!({"requested": true, "applied": applied, "made": report["made"] == true, "reused": report["reused"] == true, "file": file, "chunks": asked["chunks"], "ctx": asked["ctx"],
           "general_used": report["info"]["general_used"], "llama_cpp": report["info"]["llama_cpp"], "reason": reason, "note": report["note"]})
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
    let Some(file) = meta["gguf_file"].as_str() else { return Err(Error::new("That export has only the merged model, no GGUF file: Ollama needs a GGUF. Make the files with a GGUF.")) };
    let gguf = dir.join(file);
    if !gguf.is_file() {
        return Err(Error::new("The GGUF file of that export is missing."));
    }
    Ok((gguf, meta))
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), target)?;
        }
    }
    Ok(())
}

/// Copies the LoRA adapter of a run (adapter files + tokenizer) to `<dest>/<run>-lora/` with a short README that says which base model it belongs to and how to
/// load it in Unsloth and in plain PEFT. Returns the new folder. A full fine-tune has no adapter.
pub fn copy_adapter(p: &Project, run: &str, dest: &std::path::Path) -> Result<PathBuf> {
    if run.contains(['/', '\\']) || run.contains("..") {
        return Err(Error::new("Unknown run."));
    }
    let rdir = runs_dir(p).join(run);
    let adapter = rdir.join("adapter");
    if !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new(if rdir.join("model").join("config.json").is_file() { "A full fine-tune has no separate LoRA add-on: the whole model is saved in the run's folder." } else { "This run has no saved LoRA add-on." }));
    }
    let to = dest.join(format!("{run}-lora"));
    if to.exists() {
        return Err(Error::new(format!("{} already exists. Choose another folder or remove it.", to.display())));
    }
    copy_dir(&adapter, &to)?;
    let cfg = read_json(&rdir.join("config.json")).unwrap_or(Value::Null);
    let (repo, family) = (cfg["model"]["repo"].as_str().unwrap_or("(see adapter_config.json)"), cfg["model"]["family"].as_str().unwrap_or("generic"));
    let loader = if family == "qwen3_5" || family == "gemma4" { "FastModel" } else { "FastLanguageModel" };
    let readme = format!("# LoRA add-on of {run}\n\nTrained for the world \"{world}\" on **{repo}** (use exactly this base model, not a different one).\n\n\
## Unsloth\n```python\nfrom unsloth import {loader}\nmodel, tokenizer = {loader}.from_pretrained(model_name=\"{path}\", max_seq_length=4096, load_in_4bit=False)\n{loader}.for_inference(model)\n```\n\n\
## Plain PEFT\n```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import PeftModel\nbase = AutoModelForCausalLM.from_pretrained(\"{repo}\", dtype=\"auto\")\nmodel = PeftModel.from_pretrained(base, \"{path}\")\ntokenizer = AutoTokenizer.from_pretrained(\"{path}\")\n```\n\n\
## Bake it into the model (merge)\n```python\nmerged = model.merge_and_unload()   # PEFT\nmerged.save_pretrained(\"merged\"); tokenizer.save_pretrained(\"merged\")\n# Unsloth: model.save_pretrained_merged(\"merged\", tokenizer, save_method=\"merged_16bit\")\n```\n\n\
Ask questions the way the training did: the same kind of system message, and for Qwen3.5 / Gemma 4 with thinking off. This add-on knows everything in the world, including GM secrets.\n",
        world = p.display_name(), path = to.display());
    std::fs::write(to.join("README.md"), readme)?;
    Ok(to)
}

/// What to do with the vision file that sits next to the GGUF.
fn vision_readme(gguf: &str, mmproj: &str) -> String {
    format!("# Images with this model\n\nThis folder holds the fine-tuned text model (`{gguf}`) and the vision file of its base model (`{mmproj}`). Training only changed the language layers, so the base model's vision part still fits.\n\n\
## Unsloth Studio / llama.cpp (images work)\n- **Unsloth Studio**: load this folder (or the GGUF; Studio finds the `mmproj` file next to it) and attach pictures in the chat.\n- **llama.cpp**: `llama-server -m {gguf} --mmproj {mmproj}`, or `llama-mtmd-cli -m {gguf} --mmproj {mmproj} --image picture.png -p \"What is in this picture?\"`.\n\n\
## Ollama (text only)\nOllama's GGUF import takes one file, so the program installs only `{gguf}` and the model answers text. Keep using Unsloth Studio or llama.cpp for pictures, or run a normal vision model next to it in Ollama for the photo tools.\n\n\
Pictures are not part of the training: the model will not know your world's maps or art, only what it could read from a picture before.\n")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn fake_python(dir: &std::path::Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let py = dir.join("python");
        std::fs::write(&py, format!("#!/bin/sh\nout=$(sed -n 's/.*\"stop_file\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\n{body}\n")).unwrap();
        std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
        py
    }

    /// Stop with the importance matrix: a STOP file first (a worker that obeys ends by itself), the kill only after the grace time (a worker that does not).
    #[test]
    fn stop_writes_the_stop_file_first_and_kills_only_after_the_grace_time() {
        let d = tempfile::tempdir().unwrap();
        let stop = d.path().join("STOP");
        let job = json!({"job": "export", "stop_file": stop});
        // obeys: waits for the file, answers "Cancelled."
        let py = fake_python(d.path(), "while [ ! -f \"$out\" ]; do sleep 0.05; done\necho '{\"type\":\"error\",\"code\":\"export_stopped\",\"message\":\"Cancelled.\"}'\nexit 1");
        let kill = AtomicBool::new(false);
        let t0 = Instant::now();
        let r = std::thread::scope(|s| {
            s.spawn(|| { std::thread::sleep(Duration::from_millis(300)); kill.store(true, Ordering::SeqCst); });
            run_export_job(&py, &job, d.path(), Some(Duration::from_secs(60)), &kill, &mut |_| {})
        });
        assert_eq!(r.unwrap_err().0, "Cancelled.");
        assert!(t0.elapsed() < Duration::from_secs(20), "the worker ended by itself, long before the grace time");
        assert!(!stop.exists(), "the STOP file is removed afterwards");
        // ignores the file: killed when the grace time is over
        let py = fake_python(d.path(), "sleep 30");
        let kill = AtomicBool::new(true);
        let t0 = Instant::now();
        let r = run_export_job(&py, &job, d.path(), Some(Duration::from_millis(400)), &kill, &mut |_| {});
        assert_eq!(r.unwrap_err().0, "Cancelled.");
        assert!(t0.elapsed() < Duration::from_secs(20) && !stop.exists());
    }
}
