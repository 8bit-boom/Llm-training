//! Compare quants (docs/QUANT_COMPARE.md): how far is each quantised GGUF of a model from its 16-bit file, as KL divergence measured by llama.cpp's
//! `llama-perplexity` in the worker (`worker/ndforge_worker/quantcheck.py`). Each comparison lives in `<project>/quantcheck/<id>/` (`results.json`, `test.txt`, and the
//! big `base.kld` until the comparison is over, unless it is kept); `results.json` is written by the worker after every pass, so a Stop keeps what was finished.
//!
//! This file is the program's side: what the page offers (the files of the exports, the disk), the job the worker gets (every mistake a person could make is a plain
//! message here), listing, deleting, resuming.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::imatrix_lib::{build_source, free_bytes, short_name, SourceSpec};
use super::runs::{exportable_runs, list_exports, read_json, run_export_job, write_json, IMATRIX_STOP_GRACE};
use crate::envmgr::jobs::WorkerEvent;
use crate::envmgr::{env_dir, python_path};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::runs::runs_dir;

/// The amounts of test text offered (chunks of `CTX` tokens); the worker shows the disk need of each before anything runs.
pub const CHUNKS: [u32; 6] = [4, 8, 16, 32, 64, 128];
pub const DEFAULT_CHUNKS: u32 = 16;
pub const CTX: u32 = 512;

/// The folder of all comparisons of a project. Old projects have none: it is made with the first one.
pub fn dir(p: &Project) -> PathBuf {
    p.folder.join("quantcheck")
}

// ------------------------------------------------------------------------------------------------------------------ the choice

/// One quantised file to compare, with the words the table uses for it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantPick {
    pub path: String,
    pub label: String,
    /// the size name ("Q4_K_M"): files of one size name with and without a matrix are compared in words
    pub group: Option<String>,
    /// made with an importance matrix (Some(true)), without (Some(false)), not applicable (None: Q8_0, 16-bit files, a file typed by hand)
    pub matrix: Option<bool>,
    /// the id of the mixed-precision preset it was made with
    pub mixed: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ReferenceSpec {
    /// "file" (a 16-bit GGUF) | "make" (the worker makes one with llama.cpp's converter from `source`)
    pub kind: String,
    pub file: Option<String>,
    pub source: Option<SourceSpec>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextSpec {
    /// "data" (the held-out part of the saved data version, else a sample of it) | "file" (a text file of the person's own) | "general" (general text, seed 1234: for a model
    /// folder that has no data of its own of this project, as in Auto-cut)
    pub kind: String,
    pub file: Option<String>,
    /// data: the run whose model the files were made from (its chat template writes the conversations)
    pub run: Option<String>,
    /// data: a model folder or Hugging Face id instead of a run
    pub tokenizer: Option<String>,
    /// general: where the text comes from; None = the program's default general text
    pub general: Option<crate::general::GeneralSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompareSpec {
    pub reference: ReferenceSpec,
    pub quants: Vec<QuantPick>,
    pub text: TextSpec,
    pub chunks: u32,
    /// keep the big file of the 16-bit model's answers when the comparison is over
    pub keep_logits: bool,
    /// the id of an earlier comparison to go on with (finished passes are skipped, new files are added)
    pub resume: Option<String>,
}

impl Default for CompareSpec {
    fn default() -> Self {
        CompareSpec { reference: ReferenceSpec { kind: "file".into(), ..Default::default() }, quants: vec![], text: TextSpec { kind: "data".into(), ..Default::default() }, chunks: DEFAULT_CHUNKS, keep_logits: false, resume: None }
    }
}

// ------------------------------------------------------------------------------------------------------------------ which comparison is running

static RUNNING: Mutex<Vec<String>> = Mutex::new(Vec::new());

struct Running(String);

impl Drop for Running {
    fn drop(&mut self) {
        let mut v = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = v.iter().position(|x| *x == self.0) {
            v.remove(i);
        }
    }
}

fn mark_running(id: &str) -> Running {
    RUNNING.lock().unwrap_or_else(|e| e.into_inner()).push(id.to_string());
    Running(id.to_string())
}

pub fn running(id: &str) -> bool {
    RUNNING.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|x| x == id)
}

// ------------------------------------------------------------------------------------------------------------------ the saved comparisons

pub fn valid_id(id: &str) -> Result<()> {
    if id.is_empty() || id.starts_with(['_', '.']) || id.contains(['/', '\\']) || id.contains("..") {
        return Err(Error::new("That is not the name of a saved comparison."));
    }
    Ok(())
}

fn mb(bytes: u64) -> f64 {
    (bytes as f64 / 1048576.0 * 100.0).round() / 100.0
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// One comparison as the list shows it: the short facts of `results.json`, plus `id`, `dir`, `running` and the size of the folder. A folder without results is not an entry.
pub fn entry(d: &Path) -> Option<Value> {
    let r = read_json(&d.join("results.json")).filter(|v| v.is_object())?;
    let id = d.file_name()?.to_str()?.to_string();
    let rows = r["rows"].as_array().cloned().unwrap_or_default();
    let done = rows.iter().filter(|x| x["status"] == "done").count();
    let base = d.join("base.kld");
    Some(json!({"id": id, "dir": d, "created": r["created"], "updated": r["updated"], "state": r["state"], "reference": r["reference"]["name"], "text": r["text"]["source"],
                "chunks": r["settings"]["chunks"], "ctx": r["settings"]["ctx"], "rows": rows.len(), "rows_done": done, "base_kept": base.is_file(), "base_mb": mb(file_len(&base)),
                "running": running(&id), "has_reference": d.join("reference-f16.gguf").is_file()}))
}

/// Every saved comparison, newest first.
pub fn list(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(dir(p)).map(|d| d.flatten().filter(|e| e.path().is_dir()).filter_map(|e| entry(&e.path())).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
    out
}

/// The whole `results.json` of a comparison (the table, the line, the words) and its folder.
pub fn get(p: &Project, id: &str) -> Result<(PathBuf, Value)> {
    valid_id(id)?;
    let d = dir(p).join(id);
    let mut r = read_json(&d.join("results.json")).filter(|v| v.is_object()).ok_or_else(|| Error::new("That comparison no longer exists."))?;
    r["dir"] = json!(d);
    r["running"] = json!(running(id));
    r["base_kept"] = json!(d.join("base.kld").is_file());
    Ok((d, r))
}

/// Deletes a comparison with everything in it (also a 16-bit reference the worker made for it). Refused while it runs.
pub fn delete(p: &Project, id: &str) -> Result<()> {
    let (d, _) = get(p, id)?;
    if running(id) {
        return Err(Error::new("This comparison is running right now. Press Stop first, then delete it."));
    }
    std::fs::remove_dir_all(d)?;
    Ok(())
}

/// Deletes only the big file of the 16-bit model's answers (`base.kld`); the table stays, a later "continue" has to read the text again. Returns the freed bytes.
pub fn delete_logits(p: &Project, id: &str) -> Result<u64> {
    let (d, mut r) = get(p, id)?;
    if running(id) {
        return Err(Error::new("This comparison is running right now. Press Stop first."));
    }
    let f = d.join("base.kld");
    let n = file_len(&f);
    if f.is_file() {
        std::fs::remove_file(&f)?;
    }
    for k in ["dir", "running", "base_kept"] {
        r.as_object_mut().map(|o| o.remove(k));
    }
    r["base"]["kept"] = json!(false);
    r["base"]["deleted"] = json!(true);
    r["base"]["done"] = json!(false);
    write_json(&d.join("results.json"), &r)?;
    Ok(n)
}

// ------------------------------------------------------------------------------------------------------------------ what the page offers

/// The words the table uses for an export: ("Q4_K_M, with matrix", Some("Q4_K_M"), Some(true), mixed preset id). A mixed-precision export carries `mixed` in its meta.json.
pub fn pick_for_export(meta: &Value, with_run: bool) -> (String, Option<String>, Option<bool>, Option<String>) {
    let quant = meta["quant"].as_str().unwrap_or("").to_string();
    let mixed = meta["mixed"]["id"].as_str().map(String::from);
    let group = if let Some(l) = meta["mixed"]["label"].as_str() { l.to_string() } else { quant.to_uppercase() };
    let high = matches!(quant.as_str(), "q8_0" | "f16" | "bf16");
    let matrix = if high { None } else { Some(meta["imatrix"]["applied"] == true) };
    let mut label = group.clone();
    match matrix {
        Some(true) => label.push_str(", with matrix"),
        Some(false) => label.push_str(", no matrix"),
        None => {}
    }
    if with_run {
        if let Some(r) = meta["run"].as_str() {
            label = format!("{r}: {label}");
        }
    }
    (label, if group.is_empty() { None } else { Some(group) }, matrix, mixed)
}

/// Files the page can offer. `references`: 16-bit GGUFs (an F16 export, a `model-f16.gguf` left in an export folder, a reference an earlier comparison kept); `quants`: every
/// GGUF of the exports, with the words for the table. Only files that exist.
pub fn candidates(p: &Project) -> Value {
    let exports = list_exports(p);
    let runs: std::collections::BTreeSet<&str> = exports.iter().filter_map(|m| m["run"].as_str()).collect();
    let many = runs.len() > 1;
    let (mut refs, mut quants) = (Vec::<Value>::new(), Vec::<Value>::new());
    for m in &exports {
        let Some(dir) = m["dir"].as_str().map(PathBuf::from) else { continue };
        let name = m["name"].as_str().unwrap_or("");
        if let Some(g) = m["gguf_file"].as_str().map(|g| dir.join(g)).filter(|g| g.is_file()) {
            let (label, group, matrix, mixed) = pick_for_export(m, many);
            let item = json!({"path": g, "export": name, "run": m["run"], "quant": m["quant"], "label": label, "group": group, "matrix": matrix, "mixed": mixed, "size_mb": mb(file_len(&g))});
            if matches!(m["quant"].as_str(), Some("f16" | "bf16")) {
                refs.push(json!({"path": g, "label": format!("{name} (16-bit, from the export)"), "run": m["run"], "size_mb": mb(file_len(&g)), "kind": "export"}));
            } else {
                quants.push(item);
            }
        }
        let side = dir.join("model-f16.gguf");
        if side.is_file() && m["gguf_file"].as_str() != Some("model-f16.gguf") {
            refs.push(json!({"path": side, "label": format!("{name}: model-f16.gguf"), "run": m["run"], "size_mb": mb(file_len(&side)), "kind": "export"}));
        }
    }
    for e in list(p) {
        let f = dir(p).join(e["id"].as_str().unwrap_or("")).join("reference-f16.gguf");
        if f.is_file() {
            refs.push(json!({"path": f, "label": format!("the reference kept by the comparison {}", e["id"].as_str().unwrap_or("")), "run": null, "size_mb": mb(file_len(&f)), "kind": "comparison"}));
        }
    }
    json!({"references": refs, "quants": quants})
}

/// Everything the "Compare quants" card needs.
pub fn state(p: &Project) -> Value {
    let runs: Vec<Value> = exportable_runs(p).iter().map(|r| json!({"name": r["name"], "model": r["model"], "status": r["status"]})).collect();
    let version = crate::train::runs::latest_version(p).map(|(n, _)| n);
    let test_rows = version.map(|n| std::fs::metadata(p.folder.join("datasets").join(format!("v{n}")).join("test.jsonl")).map(|m| m.len() > 0).unwrap_or(false));
    let models: Vec<Value> = crate::train::catalog::models().into_iter().map(|m| json!({"id": m.id, "name": m.name, "repo": m.repo, "family": m.family, "params_b": m.params_b})).collect();
    json!({"entries": list(p), "candidates": candidates(p), "runs": runs, "models": models, "data_version": version, "has_test_rows": test_rows, "chunks": CHUNKS, "default_chunks": DEFAULT_CHUNKS,
           "ctx": CTX, "free_bytes": free_bytes(&dir(p)), "env_ready": crate::envmgr::health::env_installed(&env_dir())})
}

// ------------------------------------------------------------------------------------------------------------------ the job

/// The tokenizer sources (the model's chat template writes the practice conversations) and the family of the run or typed model.
fn tokenizer_for(p: &Project, t: &TextSpec) -> Result<(Vec<String>, String)> {
    if let Some(run) = t.run.as_deref().filter(|r| !r.is_empty()) {
        if run.contains(['/', '\\']) || run.contains("..") {
            return Err(Error::new("Unknown run."));
        }
        let rdir = runs_dir(p).join(run);
        let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of that run are missing."))?;
        let mut src: Vec<String> = vec![];
        for d in [rdir.join("model"), rdir.join("adapter")] {
            if d.join("tokenizer_config.json").is_file() {
                src.push(d.display().to_string());
            }
        }
        if let Some(r) = cfg["model"]["repo"].as_str().filter(|r| !r.is_empty()) {
            src.push(r.to_string());
        }
        return Ok((src, cfg["model"]["family"].as_str().unwrap_or("generic").to_string()));
    }
    if let Some(m) = t.tokenizer.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        let family = crate::train::catalog::models().into_iter().find(|x| x.repo == m).map(|x| x.family).unwrap_or_else(|| "generic".into());
        return Ok((vec![m.to_string()], family));
    }
    Err(Error::new("Say which model the files were made from (a trained run, or a model folder or Hugging Face id): its chat template writes the test conversations. Or choose a text file of your own."))
}

/// The job for the worker. `id` is the folder name (a new one, or the one being resumed).
pub fn build_job(p: &Project, spec: &CompareSpec, id: &str, stop_file: &Path) -> Result<Value> {
    if !(2..=1024).contains(&spec.chunks) {
        return Err(Error::new("Choose between 2 and 1024 chunks of test text."));
    }
    if spec.quants.is_empty() {
        return Err(Error::new("Choose at least one quantised GGUF file to compare."));
    }
    let mut seen = std::collections::HashSet::new();
    let mut quants = vec![];
    for q in &spec.quants {
        let path = q.path.trim();
        if path.is_empty() {
            return Err(Error::new("A file to compare has no path."));
        }
        if !path.to_ascii_lowercase().ends_with(".gguf") {
            return Err(Error::new(format!("{path} is not a GGUF file (its name should end in .gguf).")));
        }
        if !Path::new(path).is_file() {
            return Err(Error::new(format!("The file {path} was not found.")));
        }
        if !seen.insert(path.to_string()) {
            return Err(Error::new(format!("The file {path} is in the list twice.")));
        }
        let label = if q.label.trim().is_empty() { Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path).to_string() } else { q.label.trim().chars().take(80).collect() };
        quants.push(json!({"path": path, "label": label, "group": q.group, "matrix": q.matrix, "mixed": q.mixed}));
    }
    let reference = match spec.reference.kind.as_str() {
        "file" => {
            let f = spec.reference.file.as_deref().map(str::trim).filter(|f| !f.is_empty()).ok_or_else(|| Error::new("Choose the 16-bit reference file (an F16 or BF16 GGUF), or let the program make one."))?;
            if !Path::new(f).is_file() {
                return Err(Error::new(format!("The reference file {f} was not found.")));
            }
            if seen.contains(f) {
                return Err(Error::new("The reference file is also in the list of files to compare: it would be compared with itself. Take it out of the list."));
            }
            json!({"kind": "file", "file": f})
        }
        "make" => {
            let src = spec.reference.source.as_ref().ok_or_else(|| Error::new("Choose what the 16-bit reference is made from: a trained run, a model, or a GGUF file."))?;
            json!({"kind": "make", "source": build_source(p, src)?.0})
        }
        _ => return Err(Error::new("Choose the 16-bit reference file, or let the program make one.")),
    };
    // a comparison that is continued has its test text saved in its folder: the worker measures that same text again
    let resumed_text = spec.resume.as_deref().filter(|r| !r.is_empty() && valid_id(r).is_ok()).is_some_and(|r| dir(p).join(r).join("test.txt").is_file());
    let text = match spec.text.kind.as_str() {
        _ if resumed_text => json!({"kind": "saved"}),
        "file" => {
            let f = spec.text.file.as_deref().map(str::trim).filter(|f| !f.is_empty()).ok_or_else(|| Error::new("Choose the text file."))?;
            if !Path::new(f).is_file() {
                return Err(Error::new(format!("The text file {f} was not found.")));
            }
            json!({"kind": "file", "file": f})
        }
        "data" => {
            let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a version of the training data first (Step 2), or choose a text file of your own as the test text."))?;
            let vdir = p.folder.join("datasets").join(format!("v{version}"));
            let (tokenizer, family) = tokenizer_for(p, &spec.text)?;
            let has_test = std::fs::metadata(vdir.join("test.jsonl")).map(|m| m.len() > 0).unwrap_or(false);
            json!({"kind": "data", "train": vdir.join("train.jsonl"), "test": if has_test { json!(vdir.join("test.jsonl")) } else { Value::Null }, "family": family, "tokenizer": tokenizer,
                   "seed": 1234, "data_version": version})
        }
        "general" => {
            let (tokenizer, family) = tokenizer_for(p, &spec.text)?;
            let source = crate::general::job_block(&crate::general::resolve(spec.text.general.as_ref(), true)?, 1.0);
            json!({"kind": "general", "source": source, "family": family, "tokenizer": tokenizer, "seed": 1234})
        }
        _ => return Err(Error::new("Choose the test text: the held-out part of your saved data, a general text, or a text file of your own.")),
    };
    Ok(json!({"job": "quantcheck", "root": dir(p), "id": id, "resume": spec.resume.is_some(), "stop_file": stop_file, "chunks": spec.chunks, "ctx": CTX, "ngl": 99,
              "keep_logits": spec.keep_logits, "reference": reference, "quants": quants, "text": text}))
}

fn python() -> Result<PathBuf> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first (it also runs the comparison)."));
    }
    Ok(py)
}

fn event(e: WorkerEvent) -> Value {
    crate::reshape::runs::event_json(e)
}

/// `yyyymmdd-hhmm-<short name of the reference>`, a number added when the name is taken.
pub fn new_id(p: &Project, name: &str) -> String {
    id_at(p, &chrono::Local::now().format("%Y%m%d-%H%M").to_string(), name)
}

pub fn id_at(p: &Project, stamp: &str, name: &str) -> String {
    let base = format!("{stamp}-{}", short_name(name));
    let (mut id, mut n) = (base.clone(), 2);
    while dir(p).join(&id).exists() {
        id = format!("{base}-{n}");
        n += 1;
    }
    id
}

/// The need of the base file before anything runs (the worker reads the vocabulary from the reference's header): {"options": [{chunks, bytes, gb, fits}], "default_chunks",
/// "refusal", "free", ...}. `reference` None: the vocabulary is assumed.
pub fn plan(p: &Project, reference: Option<&str>) -> Result<Value> {
    let py = python()?;
    std::fs::create_dir_all(dir(p))?;
    let job = json!({"job": "quantcheck_plan", "reference": reference.map(str::trim).filter(|r| !r.is_empty()), "ctx": CTX, "root": dir(p)});
    run_export_job(&py, &job, &dir(p), None, &AtomicBool::new(false), &mut |_| {})
}

/// Runs a comparison in the worker. Stop writes the STOP file first (the worker kills llama-perplexity and ends cleanly, keeping the finished passes), the worker is killed
/// after the grace time. Returns the results (`results.json`) with `entry`.
pub fn compare(p: &Project, spec: &CompareSpec, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python()?;
    std::fs::create_dir_all(dir(p))?;
    let id = match spec.resume.as_deref().filter(|r| !r.is_empty()) {
        Some(r) => {
            valid_id(r)?;
            if !dir(p).join(r).join("results.json").is_file() {
                return Err(Error::new("The comparison to continue no longer exists."));
            }
            r.to_string()
        }
        None => {
            let name = match spec.reference.kind.as_str() {
                "make" => spec.reference.source.as_ref().and_then(|s| s.run.clone().or(s.model.clone()).or(s.path.clone())).unwrap_or_else(|| "reference".into()),
                _ => Path::new(spec.reference.file.as_deref().unwrap_or("")).file_stem().and_then(|s| s.to_str()).unwrap_or("reference").to_string(),
            };
            new_id(p, &name)
        }
    };
    let stop = dir(p).join("STOP");
    let job = build_job(p, spec, &id, &stop)?;
    let _g = mark_running(&id);
    let mut result = run_export_job(&py, &job, &dir(p), Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(event(e)))?;
    result["entry"] = entry(&dir(p).join(&id)).unwrap_or(Value::Null);
    Ok(result)
}

/// Exports that exist (for the "Compare these sizes" button of the export results): the picks of every GGUF of one run, newest first.
pub fn picks_of_run(p: &Project, run: &str) -> Vec<Value> {
    candidates(p)["quants"].as_array().cloned().unwrap_or_default().into_iter().filter(|q| q["run"] == run).collect()
}
