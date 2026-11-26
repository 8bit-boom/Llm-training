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
pub const MIXED_QUANTS: [(&str, &str); 5] = [
    ("mixed_4", "Mixed 4-bit (balanced) · Mixed precision (our own recipe): about 5 bits per weight"),
    ("mixed_3", "Mixed 3-bit (small) · Mixed precision (our own recipe): about 4 bits per weight"),
    ("mixed_2", "Mixed 2-bit (smallest: short answers only) · Mixed precision (our own recipe): about 3 bits per weight"),
    ("mixed_fit", "Mixed: fit to a size (experimental) · Mixed precision (our own recipe): made to fit the size you give, from a saved matrix"),
    ("mixed_measured", "Mixed: measured fit to a size · Mixed precision (our own recipe): each tensor's error measured on your weights, weighted by the importance matrix"),
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

/// "Mixed: measured fit to a size" (docs/MIXED_QUANTS.md, "Fit to a size (measured)"): the size asked for, the saved matrix (a library id) whose statistics weight the error (without one the
/// export's own matrix), the lowest and highest type the plan may use (`Q2_K`..`Q8_0`; default Q3_K and Q8_0), how many rows of every tensor are looked at (0 = 128) and the engine
/// (`auto`, `llama` or `reference`; default auto).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MixedMeasured {
    pub target_gb: f64,
    pub matrix: Option<String>,
    pub floor: Option<String>,
    pub ceiling: Option<String>,
    pub sample_rows: u32,
    pub engine: Option<String>,
}

/// The types of the ladder of the measured fit, lowest first.
pub const MEASURED_LADDER: [&str; 6] = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"];
pub const MEASURED_ENGINES: [&str; 3] = ["auto", "llama", "reference"];

impl MixedMeasured {
    /// Plain-language refusal of an unusable setting, else Ok. The size must be a number above 0.
    pub fn check(&self) -> Result<()> {
        if !(self.target_gb.is_finite() && self.target_gb > 0.0) {
            return Err(Error::new("Give the size in GB that \"Mixed: measured fit to a size\" should aim for (a number above 0)."));
        }
        let pos = |t: &Option<String>, default: usize| -> Result<usize> {
            match t.as_deref().filter(|s| !s.is_empty()) {
                None => Ok(default),
                Some(x) => MEASURED_LADDER.iter().position(|l| *l == x).ok_or_else(|| Error::new(format!("{x} is not one of the types the measured fit can use ({}).", MEASURED_LADDER.join(", ")))),
            }
        };
        if pos(&self.floor, 1)? > pos(&self.ceiling, 5)? {
            return Err(Error::new("The lowest type is above the highest one. Choose a lowest type that is not above the highest."));
        }
        if let Some(e) = self.engine.as_deref().filter(|e| !e.is_empty()) {
            if !MEASURED_ENGINES.contains(&e) {
                return Err(Error::new(format!("Unknown way of measuring: {e} (use auto, llama or reference).")));
            }
        }
        if self.sample_rows != 0 && !(8..=4096).contains(&self.sample_rows) {
            return Err(Error::new("The number of rows looked at in every tensor must be between 8 and 4096 (the usual choice is 128)."));
        }
        Ok(())
    }

    /// The `mixed_measured` block of an export job. `matrix_file` is the saved matrix's file when one was chosen; `cache_dir` is where the measures are kept between runs.
    pub fn job_block(&self, matrix_file: Option<String>, cache_dir: &std::path::Path) -> Value {
        json!({"target_gb": self.target_gb, "matrix": matrix_file, "floor": self.floor.as_deref().filter(|s| !s.is_empty()).unwrap_or("Q3_K"),
               "ceiling": self.ceiling.as_deref().filter(|s| !s.is_empty()).unwrap_or("Q8_0"), "sample_rows": if self.sample_rows == 0 { 128 } else { self.sample_rows },
               "engine": self.engine.as_deref().filter(|s| !s.is_empty()).unwrap_or("auto"), "cache_dir": cache_dir})
    }
}

/// Options of the mixed-precision sizes. Old specs have none.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MixedOptions {
    pub fit: Option<MixedFit>,
    pub measured: Option<MixedMeasured>,
}

/// Where the measures of "Mixed: measured fit to a size" are kept between runs (`<project>/quantfit/<key>/`).
pub fn quantfit_cache_dir(p: &Project) -> PathBuf {
    p.folder.join("quantfit")
}

/// The `mixed_fit` and `mixed_measured` blocks of an export job, when those sizes are in `list`; a plain refusal when their target is missing.
fn add_mixed_blocks(p: &Project, list: &[String], mixed: &MixedOptions, job: &mut Value) -> Result<()> {
    let file_of = |m: Option<&str>| -> Result<Option<String>> {
        match m.filter(|m| !m.is_empty()) {
            Some(id) => Ok(Some(super::imatrix_lib::for_export(p, id)?.1.to_string_lossy().to_string())),
            None => Ok(None),
        }
    };
    if list.iter().any(|q| q == "mixed_fit") {
        let fit = mixed.fit.as_ref().filter(|f| f.target_gb.is_finite() && f.target_gb > 0.0).ok_or_else(|| Error::new("Give the size in GB that \"Mixed: fit to a size\" should aim for (a number above 0)."))?;
        job["mixed_fit"] = json!({"target_gb": fit.target_gb, "matrix": file_of(fit.matrix.as_deref())?});
    }
    if list.iter().any(|q| q == "mixed_measured") {
        let m = mixed.measured.as_ref().ok_or_else(|| Error::new("Give the size in GB that \"Mixed: measured fit to a size\" should aim for (a number above 0)."))?;
        m.check()?;
        job["mixed_measured"] = m.job_block(file_of(m.matrix.as_deref())?, &quantfit_cache_dir(p));
    }
    Ok(())
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
    // the measured fit keeps a small sample file of the weights next to the 16-bit file while it measures
    let sample = if quants.iter().any(|q| q == "mixed_measured") { 2.0 } else { 0.0 };
    (merged + f16).max(if keep_merged { merged } else { 0.0 } + f16 + sizes) + sample
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
        // a GGUF export whose model has a vision tower but no projector can get one back via "Restore vision"
        let base_vision = meta["base_repo"].as_str().is_some_and(|r| config_has_vision(&read_json(&std::path::Path::new(r).join("config.json")).unwrap_or(Value::Null)));
        meta["can_restore_vision"] = json!(base_vision && meta["has_vision"] == false && meta["gguf_exists"] == true);
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

/// Whether a model config describes a vision-capable model (a vision tower, or a multimodal
/// architecture). The same rule the worker's `vision.base_modalities` uses for "image".
fn config_has_vision(cfg: &Value) -> bool {
    let archs = cfg["architectures"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" ")).unwrap_or_default();
    cfg.get("vision_config").is_some() || archs.contains("ConditionalGeneration") || archs.contains("VL")
}

/// Regenerate the vision projector of an export whose model has an intact vision tower but no mmproj
/// file — for example when the recorded origin was an intermediate cut that was deleted after a later
/// cut or heal, so the export skipped the vision file. Converts the projector from the export's OWN
/// base model folder (the vision part is copied unchanged through cuts) and re-wires meta.json, the
/// Modelfile and VISION.md exactly as a fresh export would. Returns the updated meta.
pub fn restore_vision(p: &Project, export: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let dir = exports_dir(p).join(export);
    let mut meta = read_json(&dir.join("meta.json")).ok_or_else(|| Error::new("That export is not there any more."))?;
    if meta["mmproj_file"].as_str().is_some_and(|f| dir.join(f).is_file()) {
        return Err(Error::new("This export already has a vision file; there is nothing to restore."));
    }
    let base = meta["base_repo"].as_str().map(PathBuf::from).filter(|d| d.join("config.json").is_file())
        .ok_or_else(|| Error::new("The export's base model folder is missing, so its vision part cannot be read."))?;
    if !config_has_vision(&read_json(&base.join("config.json")).unwrap_or(Value::Null)) {
        return Err(Error::new("The base model has no vision part, so there is nothing to restore."));
    }
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let vision_repo = read_json(&base.join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from))
        .unwrap_or_else(|| base.to_string_lossy().to_string());
    let job = json!({"job": "restore_vision", "base": {"repo": base.to_string_lossy(), "family": meta["family"].as_str().unwrap_or("generic"), "vision_repo": vision_repo}, "out": dir});
    let result = run_export_job(&py, &job, &dir, None, kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    let mmproj = result["mmproj"].as_str().map(PathBuf::from).filter(|m| m.is_file())
        .ok_or_else(|| Error::new("The vision file was not made. See the log."))?;
    let file = mmproj.file_name().and_then(|n| n.to_str()).unwrap_or("mmproj-F16.gguf").to_string();
    if mmproj != dir.join(&file) {
        std::fs::copy(&mmproj, dir.join(&file))?;
    }
    let gguf_file = meta["gguf_file"].as_str().unwrap_or("model.gguf").to_string();
    let family = meta["family"].as_str().unwrap_or("generic").to_string();
    let num_ctx = meta["num_ctx"].as_u64().unwrap_or(8192) as u32;
    meta["mmproj_file"] = json!(file);
    meta["modalities"] = result["modalities"].clone();
    meta["vision_note"] = result["vision_note"].clone();
    std::fs::write(dir.join("VISION.md"), vision_readme(&gguf_file, &file))?;
    std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&family, num_ctx), Some(&file)))?;
    write_json(&dir.join("meta.json"), &meta)?;
    Ok(meta)
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
    if want_gguf {
        add_mixed_blocks(p, &list, mixed, &mut job)?;
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
        std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&family, num_ctx), mmproj_file.as_deref()))?;
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

/// Where a model to export without a training run comes from. The Slim or grow page's "Save a model as GGUF" card offers all three: a model it made (the cut or grown one,
/// or its healed copy), any folder on this computer that holds a model, and every model of the list (built-in or added) — that one the worker downloads first when it is
/// not on this computer yet. None of them needs a training run: there is no adapter, so there is nothing to merge.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelSource {
    /// `models/<name>/` of this project, and which copy: "model" (as cut or grown) or "healed" (after a heal).
    Made { name: String, sub: String },
    /// A folder on this computer with config.json and the 16-bit .safetensors files in it.
    Folder { dir: String, name: Option<String> },
    /// A model of the list: its id (for example `qwen3.5-4b`) or its `repo` (a Hugging Face id, or the folder a custom model points at).
    Hub { repo: String },
}

/// What one no-training export is made of, once its source is found on disk (or known to be fetched by the worker): the export name, what the job names as its base model,
/// the family for the Modelfile, the size for the disk check, and what meta.json should remember.
struct FolderJob {
    /// The export name: every size becomes `exports/<name>-<size>/`.
    name: String,
    /// The model's folder on this computer; None for a model of the list that is not here (the worker resolves and downloads `base_repo` itself).
    model_dir: Option<std::path::PathBuf>,
    /// What meta.json and the worker's job name as the base: the folder path, or the Hugging Face id.
    base_repo: String,
    family: String,
    /// Billions of weights, for the disk check.
    params_b: f64,
    /// The model whose vision file the export keeps (a made model keeps its origin's).
    vision_repo: Option<String>,
    /// Extra disk for fetching the model itself (a Hugging Face download), on top of the sizes.
    download_gb: f64,
    /// Goes into meta.json as "source" (what the files were made from, when not from a training run).
    source: Value,
    /// The one-click Auto-cut writes `auto_cut: true` as it always did; the card's exports say "source" instead.
    auto_cut: bool,
}

/// Family and size of a made model: its entry in the model list of Step 3 knows both; the reshape meta or the folder itself are the fallback.
fn made_family_params(sub: &str, dir: &std::path::Path) -> (String, f64) {
    let state = read_json(&dir.parent().unwrap_or(dir).join("state.json")).unwrap_or(Value::Null);
    let id = state[if sub == "healed" { "healed_model_id" } else { "model_id" }].as_str().map(String::from);
    let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()).to_string_lossy().to_string();
    let entry = id.and_then(|i| crate::train::catalog::find(&i))
        .or_else(|| crate::train::catalog::models().into_iter().find(|m| m.repo == canonical));
    let meta_params = read_json(&dir.join("ndforge_reshape.json")).and_then(|m| m["params"].as_u64()).filter(|t| *t > 0).map(|t| t as f64 / 1e9);
    let params_b = entry.as_ref().map(|m| m.params_b).filter(|b| *b > 0.0).or(meta_params).unwrap_or_else(|| folder_params_b(dir));
    (entry.map(|m| m.family).unwrap_or_else(|| "generic".into()), params_b)
}

/// The models the Slim or grow page can save as GGUF without a training run: every made model that still has its cut or grown copy (`model`) or its healed copy, newest
/// first. `source` is exactly what `execute_no_train` takes back, and `thinking` is what the exported chat template should do about the model's own thinking
/// (`thinking_default`): the checkbox of the card starts here.
pub fn exportable_models(p: &Project) -> Vec<Value> {
    crate::reshape::runs::list(p).into_iter().filter_map(|m| {
        let name = m["name"].as_str()?.to_string();
        let mut out: Vec<Value> = Vec::new();
        for (sub, size_gb) in [("model", m["size_gb"].as_f64().unwrap_or(0.0)), ("healed", m["healed_size_gb"].as_f64().unwrap_or(0.0))] {
            let dir = crate::reshape::runs::models_dir(p).join(&name).join(sub);
            if !dir.join("config.json").is_file() {
                continue;
            }
            let (family, params_b) = made_family_params(sub, &dir);
            let export_name = if sub == "healed" { format!("{name}-healed") } else { name.clone() };
            out.push(json!({"kind": "made", "name": name, "sub": sub, "label": format!("{name} · {}", if sub == "healed" { "healed copy" } else { "as made" }),
                            "export_name": export_name, "family": family, "params_b": params_b, "size_gb": size_gb, "thinking": thinking_default(&dir) == "keep",
                            "source": {"kind": "made", "name": name, "sub": sub}}));
        }
        (!out.is_empty()).then_some(out)
    }).flatten().collect()
}

/// Is a model of the list already on this computer (a local folder, or a snapshot in the Hugging Face cache)? The card says "downloads first" for the others.
pub fn hub_model_local(repo: &str) -> bool {
    std::path::Path::new(repo).is_dir() || !hf_cache_snapshots(repo).is_empty()
}

/// Works out what one source of the card means on disk, and what the export needs from it.
fn resolve_source(p: &Project, src: &ModelSource) -> Result<FolderJob> {
    Ok(match src {
        ModelSource::Made { name, sub } => {
            if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
                return Err(Error::new("That is not the name of a model made here."));
            }
            if sub != "model" && sub != "healed" {
                return Err(Error::new("Choose the made copy or the healed copy."));
            }
            let dir = crate::reshape::runs::models_dir(p).join(name).join(sub);
            if !dir.join("config.json").is_file() {
                return Err(Error::new(if sub == "healed" { format!("The healed copy of {name} is not there (it was never healed, or it was deleted).") } else { format!("The model {name} is not there any more.") }));
            }
            let (family, params_b) = made_family_params(sub, &dir);
            let vision = read_json(&dir.join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from));
            FolderJob {
                name: if sub == "healed" { format!("{name}-healed") } else { name.clone() },
                model_dir: Some(dir.clone()), base_repo: dir.to_string_lossy().to_string(), family, params_b, vision_repo: vision, download_gb: 0.0,
                source: json!({"kind": "made", "name": name, "sub": sub}), auto_cut: false,
            }
        }
        ModelSource::Folder { dir, name } => {
            let path = std::path::PathBuf::from(dir.trim().trim_matches(|c| c == '"' || c == '\''));
            if !path.join("config.json").is_file() {
                return Err(Error::new(format!("There is no config.json in {}. Choose the model's own folder (the one that holds config.json and the .safetensors files).", path.display())));
            }
            let export_name = name.clone().filter(|n| !n.trim().is_empty()).map(|n| n.trim().to_string())
                .or_else(|| path.file_name().map(|n| n.to_string_lossy().to_string())).unwrap_or_else(|| "model".into());
            let canonical = path.canonicalize().unwrap_or_else(|_| path.clone()).to_string_lossy().to_string();
            let entry = crate::train::catalog::models().into_iter().find(|m| m.repo == canonical)
                .or_else(|| crate::train::hub::lookup(dir.trim()).ok().map(|f| f.entry));
            let params_b = entry.as_ref().map(|m| m.params_b).filter(|b| *b > 0.0).unwrap_or_else(|| folder_params_b(&path));
            let family = entry.map(|m| m.family).unwrap_or_else(|| "generic".into());
            let vision = read_json(&path.join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from));
            FolderJob {
                name: export_name, model_dir: Some(path), base_repo: canonical, family, params_b, vision_repo: vision, download_gb: 0.0,
                source: json!({"kind": "folder", "dir": dir.trim()}), auto_cut: false,
            }
        }
        ModelSource::Hub { repo } => {
            let repo_t = repo.trim();
            let entry = crate::train::catalog::find(repo_t).or_else(|| crate::train::catalog::models().into_iter().find(|m| m.repo == repo_t));
            let Some(entry) = entry else { return Err(Error::new(format!("{repo_t} is not in the model list. Add it in Step 3 first, or choose one of the models there."))); };
            let name = entry.repo.rsplit('/').next().unwrap_or(&entry.repo).to_string();
            if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
                return Err(Error::new("That is not a name for an export."));
            }
            let local = hub_model_local(&entry.repo);
            FolderJob {
                name, model_dir: None, base_repo: entry.repo.clone(), family: entry.family.clone(), params_b: entry.params_b, vision_repo: None,
                download_gb: if local { 0.0 } else { (entry.params_b * 2.0 * 10.0).round() / 10.0 },
                source: json!({"kind": "hub", "repo": entry.repo}), auto_cut: false,
            }
        }
    })
}

/// The GGUF files of a model folder that is not a training run (the healed model of Auto-cut): the whole model is converted once and shrunk to every size, then each size
/// becomes an export (`exports/<name>-<size>/` with its Modelfile and meta.json, which names the folder as its base). `thinking`: the exported chat template keeps the
/// model's own thinking (a model healed with the thinking recipe), else it answers without thinking unless asked (`thinking_default`). Returns the metas.
#[allow(clippy::too_many_arguments)]
pub fn execute_model_folder(p: &Project, name: &str, model_dir: &std::path::Path, family: &str, quants: &[String], num_ctx: u32, thinking: bool, imatrix: &ImatrixOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    execute_model_folder_full(p, name, model_dir, family, quants, num_ctx, thinking, imatrix, &MixedOptions::default(), kill, on_event)
}

/// `execute_model_folder` that also makes the mixed-precision sizes (the plain ones and `mixed_4` .. `mixed_measured`), with their options.
#[allow(clippy::too_many_arguments)]
pub fn execute_model_folder_full(p: &Project, name: &str, model_dir: &std::path::Path, family: &str, quants: &[String], num_ctx: u32, thinking: bool, imatrix: &ImatrixOptions, mixed: &MixedOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    let vision = read_json(&model_dir.join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from));
    let job = FolderJob {
        name: name.to_string(), model_dir: Some(model_dir.to_path_buf()), base_repo: model_dir.to_string_lossy().to_string(), family: family.to_string(),
        params_b: folder_params_b(model_dir), vision_repo: vision, download_gb: 0.0, source: json!({"made_by": "auto_cut"}), auto_cut: true,
    };
    execute_folder(p, &job, quants, num_ctx, thinking, imatrix, mixed, kill, on_event)
}

/// The GGUF files of a model without a training run (the "Save a model as GGUF" card of the Slim or grow page): a model made there (cut, grown or healed), a folder on this
/// computer, or a model of the list — that one the worker fetches first when it is not on this computer yet. The same files and options as the export of a training run:
/// every size becomes its own export (`exports/<name>-<size>/` with its Modelfile and meta.json). Returns the metas.
#[allow(clippy::too_many_arguments)]
pub fn execute_no_train(p: &Project, src: &ModelSource, quants: &[String], num_ctx: u32, thinking: bool, imatrix: &ImatrixOptions, mixed: &MixedOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    let job = resolve_source(p, src)?;
    execute_folder(p, &job, quants, num_ctx, thinking, imatrix, mixed, kill, on_event)
}

/// The shared body of the folder exports (`execute_model_folder_full`, `execute_no_train`): the model is converted once and shrunk to every size, then each size becomes
/// an export with its Modelfile and meta.json. A job without `model_dir` (a model of the list that is not on this computer) is resolved and fetched by the worker itself.
#[allow(clippy::too_many_arguments)]
fn execute_folder(p: &Project, j: &FolderJob, quants: &[String], num_ctx: u32, thinking: bool, imatrix: &ImatrixOptions, mixed: &MixedOptions, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    if j.name.is_empty() || j.name.contains(['/', '\\']) || j.name.contains("..") {
        return Err(Error::new("That is not a name for an export."));
    }
    if j.model_dir.as_ref().is_some_and(|dir| !dir.join("config.json").is_file()) {
        return Err(Error::new("The model to export is missing."));
    }
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
    let work = exports_dir(p).join(format!("{}-gguf-work", j.name));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work)?;
    let mut job = json!({"job": "export", "base": {"repo": j.base_repo, "family": j.family}, "out": work, "quant": list[0], "quants": list, "backend": "auto", "gguf": true, "keep_merged": false,
                         "max_seq": 4096, "thinking_default": if thinking { "keep" } else { "off" }});
    if let Some(dir) = &j.model_dir {
        job["full_model"] = json!(dir);
    }
    let need = export_need_gb(j.params_b, false, &list, false) + j.download_gb;
    check_export_disk(need, crate::envmgr::health::free_disk_gb(&exports_dir(p)), &exports_dir(p))?;
    job["need_gb"] = json!((need * 10.0).round() / 10.0);
    if let Some(o) = &j.vision_repo {
        job["base"]["vision_repo"] = json!(o);
    }
    add_mixed_blocks(p, &list, mixed, &mut job)?;
    let matrix = if imatrix.enabled {
        let (version, train) = match crate::train::runs::latest_version(p) {
            Some((v, _)) => (v, p.folder.join("datasets").join(format!("v{v}")).join("train.jsonl")),
            None => return Err(Error::new("Save a training-data version in Step 2 first: the importance matrix is measured on your world's text too.")),
        };
        imatrix_block(p, &j.name, &json!({"dataset": {"version": version, "train": train}}), imatrix, &work)?
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
        let dir = exports_dir(p).join(format!("{}-{}", j.name, quant));
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
        if let Some(m) = &mmproj_file {
            std::fs::write(dir.join("VISION.md"), vision_readme(&gguf_file, m))?;
        }
        std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&j.family, num_ctx), mmproj_file.as_deref()))?;
        let mut meta = json!({"name": format!("{}-{}", j.name, quant), "run": j.name, "quant": quant, "family": j.family, "base_repo": j.base_repo, "gguf_file": gguf_file, "mmproj_file": mmproj_file,
                              "modalities": result["modalities"], "vision_note": result["vision_note"], "merged_dir": null, "ollama_name": oname, "num_ctx": num_ctx,
                              "created": created, "world": p.display_name(), "installed": [], "via": result["via"]});
        if j.auto_cut {
            meta["auto_cut"] = json!(true);
        }
        meta["source"] = j.source.clone();
        meta["imatrix"] = imatrix_meta(matrix.as_ref(), &result["imatrix"], quant, None);
        if is_mixed(quant) {
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
    let loader = if family == "qwen3_5" || family == "gemma4" || family == "inkling2" { "FastModel" } else { "FastLanguageModel" };
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
## Ollama (pictures when the build takes the projector)\nThe Modelfile lists the vision file as a second `FROM`, so `ollama create <name> -f Modelfile` keeps pictures when your Ollama build can load this family's projector. If it answers but ignores pictures, or refuses to load, your build does not: keep Ollama for text and use llama.cpp or Unsloth Studio for pictures. (Installing through the app sends the vision file along automatically, the same way.)\n\n\
Pictures are not part of the training: the model will not know your world's maps or art, only what it could read from a picture before. A base model that was slimmed and healed on text only can be weaker with pictures than its original; the vision part itself is unchanged.\n")
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
