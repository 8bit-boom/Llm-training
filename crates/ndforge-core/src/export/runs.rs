//! Export runs: `exports/<run>-<quant>/` holds model-<quant>.gguf, Modelfile, MODEL_CARD.md, meta.json (what it was made from and where it was
//! installed) and export.json (the worker's summary).
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

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

/// The mixed-precision sizes ("Mixed precision (our own recipe)", docs/MIXED_QUANTS.md): the recipes are in ONE table in the worker (`worker/ndforge_worker/mixedquant.py`); the ids and
/// the labels here must say the same (a worker test reads this file and checks). Made through llama.cpp only (per-tensor types); never called anything else.
pub const MIXED_QUANTS: [(&str, &str); 4] = [
    ("mixed_4", "Mixed 4-bit (balanced) · Mixed precision (our own recipe): about 5 bits per weight"),
    ("mixed_3", "Mixed 3-bit (small) · Mixed precision (our own recipe): about 4 bits per weight"),
    ("mixed_2", "Mixed 2-bit (smallest: short answers only) · Mixed precision (our own recipe): about 3 bits per weight"),
    ("mixed_fit", "Mixed: fit to a size (experimental) · Mixed precision (our own recipe): made to fit the size you give, from a saved matrix"),
];
/// What the recipes are called wherever they are shown.
pub const MIXED_RECIPE: &str = "Mixed precision (our own recipe)";

pub fn is_mixed(quant: &str) -> bool {
    MIXED_QUANTS.iter().any(|(id, _)| *id == quant)
}

/// The short name of a size for a checkbox or a table: "Q4_K_M" for "Q4_K_M: smallest, good quality (recommended)", "Mixed 4-bit (balanced)" for a mixed one.
pub fn quant_label(id: &str, name: &str) -> String {
    if is_mixed(id) { name.split(" · ").next().unwrap_or(name).to_string() } else { name.split(':').next().unwrap_or(name).to_string() }
}

/// "Fit to a size" (experimental): the size asked for and, optionally, the saved matrix (a library id) whose per-tensor statistics decide; without one the export's own matrix is used.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MixedFit {
    pub target_gb: f64,
    pub matrix: Option<String>,
}

/// Options of the mixed-precision sizes. Old specs have none.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MixedOptions {
    pub fit: Option<MixedFit>,
}

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
    /// also measure the speculative-decoding (MTP) head. On by default: the worker uses it only when the model has a head and llama-imatrix has `--nextn`
    /// (else it says so in a note); the page shows the tick only for a model with a head.
    #[serde(default = "default_nextn")]
    pub nextn: bool,
    /// the id of a matrix of the library (`export::imatrix_lib`) to use as it is: nothing is measured. Old specs have none.
    #[serde(default)]
    pub use_saved: Option<String>,
    /// conversations with thinking (written the way a thinking model writes them) in the calibration text: that share of its tokens. None = the project's text and the
    /// general text only. The matrix of a model that thinks is then measured on text that looks like what it will be asked (docs/IMATRIX.md, "Thinking models").
    #[serde(default)]
    pub reasoning: Option<crate::reshape::plan::ReasoningCalib>,
}

fn default_nextn() -> bool {
    true
}

impl Default for ImatrixOptions {
    fn default() -> Self {
        ImatrixOptions { enabled: false, chunks: IMATRIX_DEFAULT_CHUNKS, reuse: true, general: None, nextn: true, use_saved: None, reasoning: None }
    }
}

pub const IMATRIX_CHUNKS: [u32; 3] = [64, 128, 256];
pub const IMATRIX_DEFAULT_CHUNKS: u32 = 128;
/// The context of one calibration chunk (tokens).
pub const IMATRIX_CTX: u32 = 512;
/// About this share of the calibration text is general text, the rest is the project's own data.
pub const IMATRIX_GENERAL_SHARE: f64 = 0.2;
/// After Stop the worker has this long to end the matrix step by itself (it kills llama-imatrix at once); then the worker is killed.
pub(crate) const IMATRIX_STOP_GRACE: Duration = Duration::from_secs(20);

/// Bits per weight of the plain GGUF sizes (llama.cpp's own table, rounded up a little); a mixed size is taken as 5.5.
fn bits_per_weight(quant: &str) -> f64 {
    match quant {
        "q4_k_m" => 4.85,
        "q5_k_m" => 5.7,
        "q6_k" => 6.6,
        "q8_0" => 8.5,
        "f16" => 16.0,
        _ => 5.5,
    }
}

/// The most disk space an export needs at one moment, in GB: the merged 16-bit model (2 bytes per weight, only when there is an adapter to merge), the 16-bit GGUF
/// (2 bytes per weight) and the sizes made from it. Converting needs merged + 16-bit; shrinking needs the 16-bit file + every size (+ the merged model while it is kept).
/// No sizes (a "merged model only" export) needs just the merged model. `params_b` in billions.
pub fn export_need_gb(params_b: f64, has_adapter: bool, quants: &[String], keep_merged: bool) -> f64 {
    let merged = if has_adapter { 2.0 * params_b } else { 0.0 };
    if quants.is_empty() {
        return merged;
    }
    let f16 = 2.0 * params_b;
    let sizes: f64 = quants.iter().filter(|q| q.as_str() != "f16").map(|q| params_b * bits_per_weight(q) / 8.0).sum();
    (merged + f16).max(if keep_merged { merged } else { 0.0 } + f16 + sizes)
}

/// Refuses an export that would run the disk full, in plain words with both numbers. `free_gb` None (not known) never refuses.
pub fn check_export_disk(need_gb: f64, free_gb: Option<f64>, dir: &std::path::Path) -> Result<()> {
    match free_gb {
        Some(free) if need_gb > 0.0 && free < need_gb * 1.05 => Err(Error::new(format!(
            "There is not enough free space for this export: about {need_gb:.1} GB are needed in {} and {free:.1} GB are free. Free some space, or choose fewer or smaller file sizes.", dir.display()))),
        _ => Ok(()),
    }
}

/// Billions of weights of a model folder, from the size of its 16-bit safetensors files (a rough figure for the disk check).
fn folder_params_b(dir: &std::path::Path) -> f64 {
    let bytes: u64 = std::fs::read_dir(dir).map(|d| d.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0);
    bytes as f64 / 2e9
}

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

/// Does this model folder carry a speculative-decoding (MTP / NextN) head? `Some(true)` / `Some(false)`, `None` when the folder tells nothing (no config, no tensors read).
/// Reads config.json (`mtp_num_hidden_layers` / `num_nextn_predict_layers`, also inside `text_config`), else the tensor names of model.safetensors.index.json or of the
/// header of each .safetensors file. The worker does the same (imatrix.folder_has_mtp).
pub fn mtp_head_in(dir: &std::path::Path) -> Option<bool> {
    if !dir.is_dir() {
        return None;
    }
    let cfg = read_json(&dir.join("config.json"));
    let key = |v: &Value| ["mtp_num_hidden_layers", "num_nextn_predict_layers"].iter().any(|k| v[*k].as_u64().is_some_and(|n| n > 0));
    if let Some(c) = &cfg {
        if key(c) || key(&c["text_config"]) {
            return Some(true);
        }
    }
    let is_mtp = |n: &str| n.starts_with("mtp.") || n.starts_with("model.mtp.");
    if let Some(idx) = read_json(&dir.join("model.safetensors.index.json")) {
        return idx["weight_map"].as_object().map(|m| m.keys().any(|k| is_mtp(k))).or(cfg.as_ref().map(|_| false));
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|x| x == "safetensors")).collect();
    files.sort();
    if files.is_empty() {
        return cfg.as_ref().map(|_| false);
    }
    for f in files {
        use std::io::Read;
        let mut fh = std::fs::File::open(&f).ok()?;
        let mut len = [0u8; 8];
        fh.read_exact(&mut len).ok()?;
        let n = u64::from_le_bytes(len);
        if n > 1 << 28 {
            return None;
        }
        let mut raw = vec![0u8; n as usize];
        fh.read_exact(&mut raw).ok()?;
        let header: Value = serde_json::from_slice(&raw).ok()?;
        if header.as_object()?.keys().any(|k| is_mtp(k)) {
            return Some(true);
        }
    }
    Some(false)
}

/// Snapshot folders of a Hugging Face model id in the local cache (HF_HUB_CACHE, HF_HOME/hub, ~/.cache/huggingface/hub), newest first.
pub(crate) fn hf_cache_snapshots(repo: &str) -> Vec<PathBuf> {
    if !repo.contains('/') || std::path::Path::new(repo).exists() {
        return vec![];
    }
    let mut roots: Vec<PathBuf> = vec![];
    if let Ok(v) = std::env::var("HF_HUB_CACHE") { roots.push(PathBuf::from(v)); }
    if let Ok(v) = std::env::var("HF_HOME") { roots.push(PathBuf::from(v).join("hub")); }
    if let Some(h) = dirs::home_dir() { roots.push(h.join(".cache").join("huggingface").join("hub")); }
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = vec![];
    for r in roots {
        let snaps = r.join(format!("models--{}", repo.replace('/', "--"))).join("snapshots");
        for e in std::fs::read_dir(snaps).into_iter().flatten().flatten() {
            if e.path().join("config.json").is_file() {
                found.push((e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), e.path()));
            }
        }
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    found.into_iter().map(|f| f.1).collect()
}

/// Does the model of this run have an MTP head? The run's own full model, else its base model (a folder, or the Hugging Face cache); `None` when it cannot be told.
pub fn run_has_mtp_head(p: &Project, run: &str) -> Option<bool> {
    let rdir = runs_dir(p).join(run);
    if let Some(v) = mtp_head_in(&rdir.join("model")) {
        return Some(v);
    }
    let repo = read_json(&rdir.join("config.json"))?["model"]["repo"].as_str()?.to_string();
    std::iter::once(PathBuf::from(&repo)).chain(hf_cache_snapshots(&repo)).find_map(|d| mtp_head_in(&d))
}

/// What the export page needs for the importance matrix: the choices, and per run the saved matrix (if any) and the time guesses for each amount of text.
pub fn imatrix_state(p: &Project) -> Value {
    let mut saved = serde_json::Map::new();
    let mut estimates = serde_json::Map::new();
    let mut mtp = serde_json::Map::new();
    for r in exportable_runs(p) {
        let Some(run) = r["name"].as_str() else { continue };
        mtp.insert(run.to_string(), json!(run_has_mtp_head(p, run)));
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
           "no_matrix_for": ["q8_0", "f16"], "saved": saved, "estimates": estimates,
           "mtp": mtp, "nextn_supported": Value::Null, "general": crate::general::state(Some(p)), "library": super::imatrix_lib::brief(p)})
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
    let reasoning = match &opts.reasoning {
        Some(r) => {
            r.validate()?;
            r.json()
        }
        None => Value::Null,
    };
    // a saved matrix of the library: nothing is measured, the worker copies the file into the export folder and gives it to llama-quantize / Unsloth as a fresh one
    let (use_file, use_id) = match opts.use_saved.as_deref().filter(|s| !s.is_empty()) {
        Some(id) => { let (id, f) = super::imatrix_lib::for_export(p, id)?; (json!(f), json!(id)) }
        None => (Value::Null, Value::Null),
    };
    Ok(Some(json!({"use_file": use_file, "use_id": use_id, "enabled": true, "general": general, "reasoning": reasoning, "seed": crate::general::SEED, "model": run, "data_version": version, "train": train, "chunks": opts.chunks, "ctx": IMATRIX_CTX, "general_share": IMATRIX_GENERAL_SHARE,
                   "reuse": opts.reuse, "nextn": opts.nextn, "reuse_dirs": saved_imatrices(p, run).into_iter().map(|(d, _)| d).collect::<Vec<_>>(), "stop_file": out.join("STOP")})))
}

/// Runs the export job. The Stop button first writes a STOP file (the worker kills llama-imatrix / the converter / llama-quantize and ends cleanly) and kills the
/// worker only after IMATRIX_STOP_GRACE (`graceful: None` kills at once; the program always gives a grace time).
pub(crate) fn run_export_job(py: &std::path::Path, job: &Value, out: &std::path::Path, graceful: Option<Duration>, kill: &AtomicBool, on_event: &mut dyn FnMut(WorkerEvent)) -> Result<Value> {
    match graceful {
        None => run_job(py, &worker_dir(), job, kill, on_event),
        Some(grace) => crate::envmgr::jobs::run_with_grace(py, job, &out.join("STOP"), kill, grace, on_event),
    }
}

pub fn exports_dir(p: &Project) -> PathBuf {
    p.folder.join("exports")
}

pub(crate) fn read_json(path: &std::path::Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

pub(crate) fn write_json(path: &std::path::Path, v: &Value) -> Result<()> {
    crate::fsx::write_json_atomic(path, v)
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

/// What the exported chat template does about thinking (the worker's job key `thinking_default`): "keep" when the base model was healed with the thinking recipe
/// (`healed.recipe == "thinking"` in its `ndforge_reshape.json`: it was taught to think, so the template stays as the model ships it), else "off" (answers without
/// thinking unless asked, what a model trained thinking-off expects).
pub fn thinking_default(base: &std::path::Path) -> &'static str {
    match read_json(&base.join("ndforge_reshape.json")) {
        Some(m) if m["healed"]["recipe"].as_str() == Some("thinking") => "keep",
        _ => "off",
    }
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
    execute_many_full(p, run, quants, num_ctx, what, imatrix, &MixedOptions::default(), kill, on_event)
}

/// `execute_many_with` with the options of the mixed-precision sizes.
#[allow(clippy::too_many_arguments)]
pub fn execute_many_full(p: &Project, run: &str, quants: &[String], num_ctx: u32, what: &str, imatrix: &ImatrixOptions, mixed: &MixedOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    if !WHAT.contains(&what) {
        return Err(Error::new(format!("Unknown export kind: {what}")));
    }
    let (want_gguf, keep_merged) = (what != "merged", what != "gguf");
    // a saved matrix of the library cannot be deleted while this export uses it
    let _in_use = imatrix.use_saved.as_deref().filter(|s| !s.is_empty() && imatrix.enabled && want_gguf).map(super::imatrix_lib::mark_in_use);
    let mut list: Vec<String> = Vec::new();
    for q in quants {
        if !QUANTS.iter().chain(MIXED_QUANTS.iter()).any(|(id, _)| id == q) {
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
                     "max_seq": cfg["hyper"]["max_seq"], "thinking_default": thinking_default(std::path::Path::new(&repo))});
    // the disk is checked before anything is made (the worker looks again before it starts: `need_gb`)
    if let Some(m) = crate::train::catalog::find(cfg["model"]["id"].as_str().unwrap_or("")) {
        let need = export_need_gb(m.params_b, !is_full, if want_gguf { &list } else { &[] }, keep_merged);
        check_export_disk(need, crate::envmgr::health::free_disk_gb(&exports_dir(p)), &exports_dir(p))?;
        job["need_gb"] = json!((need * 10.0).round() / 10.0);
    }
    if let Some(o) = &origin {
        job["base"]["vision_repo"] = json!(o);
    }
    let matrix = if want_gguf { imatrix_block(p, run, &cfg, imatrix, &out)? } else { None };
    if want_gguf && list.iter().any(|q| q == "mixed_fit") {
        let fit = mixed.fit.as_ref().filter(|f| f.target_gb.is_finite() && f.target_gb > 0.0).ok_or_else(|| Error::new("Give the size in GB that \"Mixed: fit to a size\" should aim for (a number above 0)."))?;
        let file = match fit.matrix.as_deref().filter(|m| !m.is_empty()) {
            Some(id) => Some(super::imatrix_lib::for_export(p, id)?.1),
            None => None,
        };
        job["mixed_fit"] = json!({"target_gb": fit.target_gb, "matrix": file});
    }
    if let Some(m) = &matrix {
        job["imatrix"] = m.clone();
    }
    // a full fine-tune saved the whole model (nothing to merge); a LoRA run saved an adapter
    if is_full {
        job["full_model"] = json!(full);
    } else {
        job["adapter"] = json!(adapter);
    }
    job["stop_file"] = json!(out.join("STOP"));
    let result = run_export_job(&py, &job, &out, Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(match e {
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
                m.file_name().and_then(|n| n.to_str()).map(String::from)
            }
            None => None,
        };
        std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&family, num_ctx)))?;
        let quant_shown = if is_mixed(quant) { format!("{} ({MIXED_RECIPE})", MIXED_QUANTS.iter().find(|(id, _)| id == quant).map(|(id, n)| quant_label(id, n)).unwrap_or_default()) } else { quant.clone() };
        std::fs::write(dir.join("MODEL_CARD.md"), model_card(&CardInfo {
            world: &p.display_name(), model_name: &oname, base_repo: &repo, license: &license, method: cfg["method"].as_str().unwrap_or("lora"), quant: &quant_shown,
            dataset_version: version, steps: summary["steps"].as_u64(), final_loss: summary["final_loss"].as_f64(), ollama_name: &oname, derived_from: origin.as_deref(),
        }))?;
        let meta = json!({"name": name_of(quant), "run": run, "quant": quant, "family": family, "base_repo": repo, "gguf_file": gguf_file, "mmproj_file": mmproj_file, "modalities": result["modalities"], "vision_note": result["vision_note"], "merged_dir": keep_here, "ollama_name": oname, "num_ctx": num_ctx,
                          "created": created, "world": p.display_name(), "installed": [], "via": result["via"]});
        let mut meta = meta;
        meta["imatrix"] = imatrix_meta(matrix.as_ref(), &result["imatrix"], quant, matrix_file.as_deref());
        if is_mixed(quant) {
            // what the recipe was (the rules llama-quantize got, the size estimate, whether the matrix was used), always labelled as our own
            let mut m = result["mixed"][quant.as_str()].clone();
            if !m.is_object() {
                m = json!({"id": quant, "label": MIXED_QUANTS.iter().find(|(id, _)| id == quant).map(|(id, n)| quant_label(id, n)), "recipe": MIXED_RECIPE});
            }
            m["note"] = json!(result["mixed_notes"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" ")).unwrap_or_default());
            meta["mixed"] = m;
        }
        write_json(&dir.join("meta.json"), &meta)?;
        metas.push(meta);
    }
    if many {
        let _ = std::fs::remove_dir_all(&out);
    }
    Ok(metas)
}

/// The GGUF files of a model folder that is not a training run (the healed model of Auto-cut): the whole model is converted once and shrunk to every size, then each size
/// becomes an export (`exports/<name>-<size>/` with its Modelfile and meta.json, which names the folder as its base). `thinking`: the exported chat template keeps the
/// model's own thinking (a model healed with the thinking recipe), else it answers without thinking unless asked (`thinking_default`). Returns the metas.
#[allow(clippy::too_many_arguments)]
pub fn execute_model_folder(p: &Project, name: &str, model_dir: &std::path::Path, family: &str, quants: &[String], num_ctx: u32, thinking: bool, imatrix: &ImatrixOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
        return Err(Error::new("That is not a name for an export."));
    }
    if !model_dir.join("config.json").is_file() {
        return Err(Error::new("The model to export is missing."));
    }
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
    let work = exports_dir(p).join(format!("{name}-gguf-work"));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)?;
    let repo = model_dir.to_string_lossy().to_string();
    let origin = read_json(&model_dir.join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from));
    let mut job = json!({"job": "export", "base": {"repo": repo, "family": family}, "out": work, "quant": list[0], "quants": list, "backend": "auto", "gguf": true, "keep_merged": false,
                         "max_seq": 4096, "thinking_default": if thinking { "keep" } else { "off" }, "full_model": model_dir});
    let need = export_need_gb(folder_params_b(model_dir), false, &list, false);
    check_export_disk(need, crate::envmgr::health::free_disk_gb(&exports_dir(p)), &exports_dir(p))?;
    job["need_gb"] = json!((need * 10.0).round() / 10.0);
    if let Some(o) = &origin {
        job["base"]["vision_repo"] = json!(o);
    }
    let matrix = if imatrix.enabled {
        let (version, train) = match crate::train::runs::latest_version(p) {
            Some((v, _)) => (v, p.folder.join("datasets").join(format!("v{v}")).join("train.jsonl")),
            None => return Err(Error::new("Save a training-data version in Step 2 first: the importance matrix is measured on your world's text too.")),
        };
        imatrix_block(p, name, &json!({"dataset": {"version": version, "train": train}}), imatrix, &work)?
    } else {
        None
    };
    if let Some(m) = &matrix {
        job["imatrix"] = m.clone();
    }
    job["stop_file"] = json!(work.join("STOP"));
    let result = run_export_job(&py, &job, &work, Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    if result["stopped"].as_bool().unwrap_or(false) {
        let _ = std::fs::remove_dir_all(&work);
        return Ok(Vec::new());
    }
    let oname = ollama_name(&p.model_label(), "");
    let created = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut metas = Vec::new();
    for (i, quant) in list.iter().enumerate() {
        let gguf = result["ggufs"][quant.as_str()].as_str().or_else(|| if list.len() == 1 { result["gguf"].as_str() } else { None }).map(PathBuf::from)
            .ok_or_else(|| Error::new(format!("No GGUF file was made for {quant}.")))?;
        let dir = exports_dir(p).join(format!("{name}-{quant}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        let gguf_file = gguf.file_name().and_then(|n| n.to_str()).unwrap_or("model.gguf").to_string();
        std::fs::rename(&gguf, dir.join(&gguf_file))?;
        let mmproj_file = match result["mmproj"].as_str().map(PathBuf::from).filter(|m| m.is_file()) {
            Some(m) => {
                let file = m.file_name().and_then(|n| n.to_str()).unwrap_or("mmproj-F16.gguf").to_string();
                if i == list.len() - 1 { std::fs::rename(&m, dir.join(&file))?; } else { std::fs::copy(&m, dir.join(&file))?; }
                Some(file)
            }
            None => None,
        };
        std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(family, num_ctx)))?;
        let mut meta = json!({"name": format!("{name}-{quant}"), "run": name, "quant": quant, "family": family, "base_repo": repo, "gguf_file": gguf_file, "mmproj_file": mmproj_file,
                              "modalities": result["modalities"], "vision_note": result["vision_note"], "merged_dir": null, "ollama_name": oname, "num_ctx": num_ctx,
                              "created": created, "world": p.display_name(), "installed": [], "via": result["via"], "auto_cut": true});
        meta["imatrix"] = imatrix_meta(matrix.as_ref(), &result["imatrix"], quant, None);
        write_json(&dir.join("meta.json"), &meta)?;
        metas.push(meta);
    }
    let _ = std::fs::remove_dir_all(&work);
    Ok(metas)
}

/// What meta.json says about the importance matrix of one size: null when it was not asked for. `applied` is whether THIS size was made with it.
fn imatrix_meta(asked: Option<&Value>, report: &Value, quant: &str, file: Option<&str>) -> Value {
    let Some(asked) = asked else { return Value::Null };
    let applied = report["applied"].as_array().is_some_and(|a| a.iter().any(|q| q == quant));
    let reason = report["not_applied"][quant].as_str();
    json!({"requested": true, "applied": applied, "made": report["made"] == true, "reused": report["reused"] == true, "file": file, "chunks": asked["chunks"], "ctx": asked["ctx"],
           "general_used": report["info"]["general_used"], "llama_cpp": report["info"]["llama_cpp"], "reason": reason, "note": report["note"],
           "nextn": report["nextn"] == true, "nextn_asked": asked["nextn"] == true, "via": report["via"], "library": report["from_library"]})
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
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    /// Jobs save a log in the settings folder: these tests use a private one for the whole run.
    fn private_config_dir() {
        static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
        let d = DIR.get_or_init(|| tempfile::tempdir().unwrap());
        std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
    }

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
        private_config_dir();
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

#[cfg(test)]
mod disk_tests {
    use super::*;

    #[test]
    fn the_disk_an_export_needs_follows_the_sizes_asked_for() {
        let q = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        // 10 billion weights, a LoRA run: merged model 20 GB + 16-bit file 20 GB
        assert!(close(export_need_gb(10.0, true, &q(&["q4_k_m"]), false), 40.0), "converting needs the merged model and the 16-bit file: {}", export_need_gb(10.0, true, &q(&["q4_k_m"]), false));
        // keeping the merged model: while shrinking it is still there next to the 16-bit file and the sizes
        assert!(close(export_need_gb(10.0, true, &q(&["q4_k_m", "q8_0"]), true), 20.0 + 20.0 + 10.0 * 4.85 / 8.0 + 10.0 * 8.5 / 8.0));
        // a full fine-tune has nothing to merge; the 16-bit file as an output is not counted twice
        assert!(close(export_need_gb(10.0, false, &q(&["f16"]), false), 20.0));
        assert!(close(export_need_gb(10.0, false, &q(&["q6_k", "q5_k_m"]), false), 20.0 + 10.0 * 6.6 / 8.0 + 10.0 * 5.7 / 8.0));
        // a mixed size counts 5.5 bits per weight; "merged model only" is just the merged model, or nothing for a full model
        assert!(close(export_need_gb(10.0, false, &q(&["mixed_4"]), false), 20.0 + 10.0 * 5.5 / 8.0));
        assert!(close(export_need_gb(10.0, true, &[], true), 20.0) && close(export_need_gb(10.0, false, &[], true), 0.0));
    }

    #[test]
    fn an_export_that_does_not_fit_is_refused_with_both_numbers() {
        let dir = std::path::Path::new("exports");
        let e = check_export_disk(40.0, Some(41.0), dir).unwrap_err().0;
        assert!(e.contains("about 40.0 GB are needed") && e.contains("41.0 GB are free") && e.contains("not enough free space"), "{e}");
        assert!(check_export_disk(40.0, Some(42.0), dir).is_ok(), "5 % to spare is enough");
        assert!(check_export_disk(40.0, None, dir).is_ok() && check_export_disk(0.0, Some(0.0), dir).is_ok(), "an unknown disk never refuses");
    }
}
