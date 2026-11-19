//! The thinking check of a made model (docs/THINKING_CHECK.md): does the model still think before it answers, after it was cut and healed?
//!
//! The worker (`worker/ndforge_worker/thinkcheck.py`) asks a suite of 32 questions with thinking on and reports numbers; this file is the program's side: which model a
//! check of the "original", the "cut", the "healed" model or a "gguf" file is about, the job, where the results are kept (`models/<name>/think_check/<target>.<full|quick>.json`,
//! written by the worker, which also answers a repeated check of the same weights from that file) and the gates that compare a model with the original. The gates are the
//! same rule as `thinkcheck.gates` in the worker; both are tested against `worker/tests/data/think_gate_cases.json`.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::heal_source;
use super::runs::{event_json, models_dir, python, read_json, valid_name};
use crate::envmgr::jobs::run_job;
use crate::envmgr::worker_dir;
use crate::error::{Error, Result};
use crate::project::Project;

/// The version of the question suite the worker has; results of another version are not compared.
pub const SUITE_VERSION: u64 = 1;
/// The gates, in the order the table shows them.
pub const GATES: [&str; 5] = ["closed_rate", "accuracy", "loop_rate", "garbage_rate", "median_think_tokens"];
/// "Warn" when a passed gate is within this share of its limit.
pub const WARN_MARGIN: f64 = 0.05;
/// The things a check can be about, in the order of the table.
pub const TARGETS: [&str; 4] = ["original", "cut", "healed", "gguf"];

fn num(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or(0.0)
}

/// The limit of every gate, from the ORIGINAL model's numbers: (name, "min" | "max", limit).
pub fn gate_limits(original: &Value) -> Vec<(&'static str, &'static str, f64)> {
    vec![
        ("closed_rate", "min", (0.9 * num(original, "closed_rate")).max(0.75)),
        ("accuracy", "min", 0.8 * num(original, "accuracy")),
        ("loop_rate", "max", num(original, "loop_rate") + 0.10),
        ("garbage_rate", "max", 0.05),
        ("median_think_tokens", "max", 2.5 * num(original, "median_think_tokens").max(1.0)),
    ]
}

/// `{"verdict": "pass" | "warn" | "fail", "checks": [...], "failed": [names], "warned": [names]}`. Fail: any gate missed. Warn: all met, one within 5 % of its limit.
pub fn gates(original: &Value, candidate: &Value) -> Value {
    let eps = 1e-9;
    let (mut checks, mut failed, mut warned) = (vec![], vec![], vec![]);
    for (name, dir, limit) in gate_limits(original) {
        let v = num(candidate, name);
        let (ok, warn) = if dir == "min" {
            let ok = v >= limit - eps;
            (ok, ok && limit > 0.0 && v < limit * (1.0 + WARN_MARGIN) - eps)
        } else {
            let ok = v <= limit + eps;
            (ok, ok && limit > 0.0 && v > limit * (1.0 - WARN_MARGIN) + eps)
        };
        if !ok {
            failed.push(name);
        } else if warn {
            warned.push(name);
        }
        let r4 = |x: f64| (x * 10000.0).round() / 10000.0;
        checks.push(json!({"name": name, "direction": dir, "limit": r4(limit), "value": r4(v), "ok": ok, "warn": warn}));
    }
    let verdict = if !failed.is_empty() { "fail" } else if !warned.is_empty() { "warn" } else { "pass" };
    json!({"verdict": verdict, "checks": checks, "failed": failed, "warned": warned})
}

// ------------------------------------------------------------------------------------------------------------------ choosing by thinking

/// Two thinking scores this close are a tie: the check asks few questions (16 in the quick suite) and is noisy, so the lower loss decides.
pub const THINK_TIE: f64 = 0.125;

/// One number for "how well does it still think": the share of answers that finish their reasoning, minus the share that loop, minus the share of garbage. None when the
/// row has no numbers.
pub fn think_score(main: &Value) -> Option<f64> {
    let get = |k: &str| main[k].as_f64();
    let closed = get("closed_rate")?;
    Some(closed - get("loop_rate").unwrap_or(0.0) - get("garbage_rate").unwrap_or(0.0))
}

/// The best candidate of `(loss, thinking score)` rows, or None for no rows. The highest score wins; scores within `THINK_TIE` of it are decided by the lower loss; a row
/// without a score only competes when no row has one (then the lowest loss wins). A row without a loss loses every tie.
pub fn rank_thinking(rows: &[(Option<f64>, Option<f64>)]) -> Option<usize> {
    let by_loss = |idx: Vec<usize>| -> Option<usize> {
        idx.into_iter().min_by(|a, b| {
            let (la, lb) = (rows[*a].0.unwrap_or(f64::INFINITY), rows[*b].0.unwrap_or(f64::INFINITY));
            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
        })
    };
    let top = rows.iter().filter_map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    if top.is_finite() {
        by_loss((0..rows.len()).filter(|i| rows[*i].1.is_some_and(|s| s >= top - THINK_TIE - 1e-12)).collect())
    } else if rows.iter().any(|r| r.0.is_some()) {
        by_loss((0..rows.len()).collect())
    } else {
        None
    }
}

/// Did the candidate lose much of its ability to finish its reasoning: its `closed_rate` is below 0.6 times the original's.
pub fn lost_much(original_main: &Value, cand_main: &Value) -> bool {
    match (original_main["closed_rate"].as_f64(), cand_main["closed_rate"].as_f64()) {
        (Some(o), Some(c)) => o > 0.0 && c < 0.6 * o,
        _ => false,
    }
}

// ------------------------------------------------------------------------------------------------------------------ what is checked

fn default_target() -> String {
    "cut".into()
}

/// A request to check one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThinkSpec {
    /// the folder name under `models/`
    pub name: String,
    /// "original" | "cut" | "healed" | "gguf"
    pub target: String,
    /// 16 questions instead of 32 (and a shorter limit per answer)
    pub quick: bool,
    /// target "gguf": the file (llama.cpp's llama-server runs it)
    pub gguf: Option<String>,
    /// "auto" | "16bit" | "4bit" | "offload" (the transformers way)
    pub load: Option<String>,
    /// how many questions are asked at once (the transformers way); 1 = one at a time
    pub batch: Option<u32>,
    /// the most new tokens per answer (the worker's own limit when None); only the end-to-end test on a tiny model sets it
    pub max_new: Option<u32>,
    /// target "round": the round of a heal in rounds whose adapter is checked on top of the student, NOT merged (docs/HEAL_ROUNDS.md)
    pub round: Option<u32>,
    /// the way the questions are answered: None / "hf" (the standard engine), or "vllm" (EXPERIMENTAL, docs/VLLM.md; a GGUF file always uses llama.cpp)
    pub backend: Option<String>,
    /// the vLLM engine settings the worker gets (`vllm::engine_job`); the window's command fills it from the settings, the page does not send it
    pub engine: Option<Value>,
}

impl Default for ThinkSpec {
    fn default() -> Self {
        ThinkSpec { name: String::new(), target: default_target(), quick: false, gguf: None, load: None, batch: None, max_new: None, round: None, backend: None, engine: None }
    }
}

/// Does this check go through the experimental vLLM engine? (Its numbers are kept in their own files: the gates compare a model with the original of the SAME way.)
pub fn uses_vllm(spec: &ThinkSpec) -> bool {
    spec.backend.as_deref() == Some("vllm") && spec.target != "gguf"
}

pub fn think_dir(p: &Project, name: &str) -> PathBuf {
    models_dir(p).join(name).join("think_check")
}

/// Has the cut of the made model `name` had a thinking check of either size (saved by the worker)?
pub fn cut_checked(p: &Project, name: &str) -> bool {
    valid_name(name).is_ok() && [true, false].iter().any(|q| think_dir(p, name).join(file_name("cut", *q)).is_file())
}

fn file_name(target: &str, quick: bool) -> String {
    format!("{target}.{}.json", if quick { "quick" } else { "full" })
}

fn file_name_for(target: &str, quick: bool, vllm: bool) -> String {
    if vllm { format!("{target}.{}.vllm.json", if quick { "quick" } else { "full" }) } else { file_name(target, quick) }
}

/// Where the result of a check is kept: `<target>.<mode>.json`, for a round of a heal in rounds `round-r<n>.<mode>.json`.
pub fn cache_name(spec: &ThinkSpec) -> String {
    match (spec.target.as_str(), spec.round) {
        ("round", Some(n)) => format!("round-r{n}.{}{}.json", if spec.quick { "quick" } else { "full" }, if uses_vllm(spec) { ".vllm" } else { "" }),
        _ => file_name_for(&spec.target, spec.quick, uses_vllm(spec)),
    }
}

/// The LoRA folder of round `n` of the heal of `name` (written by the worker's distill job).
pub fn round_adapter(p: &Project, name: &str, n: u32) -> PathBuf {
    models_dir(p).join(name).join("distill").join("run").join("rounds").join(format!("r{n}")).join("adapter")
}

/// How the student of the heal was loaded for training: `heal_config.json` is the distill job the program wrote.
fn heal_config(dir: &Path) -> Option<Value> {
    read_json(&dir.join("heal_config.json"))
}

/// (the model the worker loads: a folder, a Hugging Face id or a GGUF file, the way it runs: "hf" | "llama")
pub fn model_of(p: &Project, spec: &ThinkSpec) -> Result<(String, &'static str)> {
    valid_name(&spec.name)?;
    let dir = models_dir(p).join(&spec.name);
    if !dir.is_dir() {
        return Err(Error::new("That model is not in this project."));
    }
    match spec.target.as_str() {
        "original" => Ok((heal_source::read_origin(&dir)?.teacher, "hf")),
        "cut" => Ok((heal_source::resolve_student(&dir)?.to_string_lossy().to_string(), "hf")),
        "healed" => {
            let h = dir.join("healed");
            if !h.join("config.json").is_file() {
                return Err(Error::new("This model has not been healed yet."));
            }
            Ok((h.to_string_lossy().to_string(), "hf"))
        }
        "round" => {
            // the exact student the distill job used (a retry student folder too), else the made model's folder
            if let Some(repo) = heal_config(&dir).and_then(|c| c["model"]["repo"].as_str().map(String::from)).filter(|r| !r.is_empty()) {
                return Ok((repo, "hf"));
            }
            Ok((heal_source::resolve_student(&dir)?.to_string_lossy().to_string(), "hf"))
        }
        "gguf" => {
            let f = spec.gguf.as_deref().map(str::trim).filter(|f| !f.is_empty()).ok_or_else(|| Error::new("Choose the GGUF file to check."))?;
            if !f.to_ascii_lowercase().ends_with(".gguf") {
                return Err(Error::new(format!("{f} is not a GGUF file (its name should end in .gguf).")));
            }
            if !Path::new(f).is_file() {
                return Err(Error::new(format!("The file {f} was not found.")));
            }
            Ok((f.to_string(), "llama"))
        }
        other => Err(Error::new(format!("Unknown thing to check: {other}"))),
    }
}

pub fn build_job(p: &Project, spec: &ThinkSpec) -> Result<Value> {
    let (model, backend) = model_of(p, spec)?;
    let load = spec.load.as_deref().filter(|l| matches!(*l, "auto" | "16bit" | "4bit" | "offload")).unwrap_or("auto");
    let dir = think_dir(p, &spec.name);
    let mut job = json!({"job": "think_check", "model": model, "backend": backend, "quick": spec.quick, "control": true, "load": load,
                         "stop_file": dir.join("STOP"), "cache_file": dir.join(cache_name(spec))});
    if spec.target == "round" {
        let n = spec.round.ok_or_else(|| Error::new("Which round should be checked?"))?;
        let adapter = round_adapter(p, &spec.name, n);
        if !adapter.join("adapter_config.json").is_file() {
            return Err(Error::new(format!("The adapter of round {n} was not found. A round is checked right after it was trained; its adapter is deleted when the heal is finished.")));
        }
        // the model as it is trained: the base the way the student was loaded (4-bit NF4 with double quantization for QLoRA, else 16-bit) and the LoRA on top, not merged
        let qlora = heal_config(&models_dir(p).join(&spec.name)).is_none_or(|c| c["method"] != json!("lora"));
        job["adapter"] = json!(adapter);
        job["double_quant"] = json!(qlora);
        job["load"] = json!(if qlora { "4bit" } else { "16bit" });
    }
    if uses_vllm(spec) {
        job["backend"] = json!("vllm");
        job["engine"] = spec.engine.clone().unwrap_or_else(|| json!({"kind": "auto"}));
        if spec.target == "original" {
            job["engine"]["original"] = json!(true);          // the worker serves the configured pre-quantized checkpoint when the original does not fit (vllmquant.pick)
        }
    }
    if let Some(b) = spec.batch.filter(|b| (1..=64).contains(b)) {
        job["batch"] = json!(b);
    }
    if let Some(n) = spec.max_new.filter(|n| (1..=65536).contains(n)) {
        job["max_new"] = json!(n);
    }
    Ok(job)
}

// ------------------------------------------------------------------------------------------------------------------ running

static RUNNING: Mutex<bool> = Mutex::new(false);

struct Running;

impl Running {
    fn begin() -> Result<Running> {
        let mut r = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
        if *r {
            return Err(Error::new("A thinking check is already running. Wait until it is done, or stop it."));
        }
        *r = true;
        Ok(Running)
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = false;
    }
}

/// What a check would do (the questions, the settings, a rough size) without loading anything.
pub fn plan(spec: &ThinkSpec, kill: &AtomicBool) -> Result<Value> {
    let py = python()?;
    let backend = if spec.target == "gguf" { "llama" } else if uses_vllm(spec) { "vllm" } else { "hf" };
    run_job(&py, &worker_dir(), &json!({"job": "think_check_plan", "backend": backend, "quick": spec.quick}), kill, &mut |_| {})
}

/// Runs one check. Events: `progress` (steps `load`, `think`), `think_item` for every answered question, `log`. Stop writes the STOP file (the worker ends between questions and
/// keeps nothing half-written); a check that was stopped is not saved. Returns the worker's result with `name` and `target`.
pub fn run(p: &Project, spec: &ThinkSpec, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let job = build_job(p, spec)?;
    let py = python()?;
    let _running = Running::begin()?;
    let dir = think_dir(p, &spec.name);
    let mut r = crate::train::runs::drive(&py, &dir, &job, stop, Duration::from_secs(60), &mut |e| on_event(event_json(e)))?;
    r["name"] = json!(spec.name);
    r["target"] = json!(spec.target);
    Ok(r)
}

// ------------------------------------------------------------------------------------------------------------------ the saved results

fn saved(dir: &Path, target: &str, quick: bool, vllm: bool) -> Option<Value> {
    let f = read_json(&dir.join(file_name_for(target, quick, vllm)))?;
    let r = f["result"].clone();
    (r.is_object() && r["suite_version"] == json!(SUITE_VERSION) && r["main"].is_object()).then_some(r)
}

/// The ways a made cut goes beyond what Auto-cut does by itself (see `auto_plan::guardrail_messages`), read from the cut's own record (`ndforge_reshape.json`): the MLP kept
/// below `WIDTH_FIRST_KEEP` of the original's, more than `MAX_BLOCKS_FREE` blocks removed. Empty for a cut inside the rules or a record that cannot be read.
pub fn cut_guardrails(dir: &Path) -> Vec<String> {
    let Some(meta) = read_json(&dir.join("model").join("ndforge_reshape.json")) else { return Vec::new() };
    if meta["mode"] != json!("prune") {
        return Vec::new();
    }
    let blocks = (meta["dropped"].as_array().map(|d| d.len()).unwrap_or(0) as u32) / meta["period"].as_u64().unwrap_or(4).max(1) as u32;
    let w = &meta["width"]["intermediate_size"];
    let (full, after) = (w[0].as_u64().map(|x| x as u32), w[1].as_u64().map(|x| x as u32));
    let floor = full.map(|f| (f as f64 * super::auto_plan::WIDTH_FIRST_KEEP).ceil() as u32);
    super::auto_plan::guardrail_messages(blocks, after, full, floor)
}

/// What the result page shows: `{"suite_version", "modes": {"full": {"rows": {target: result}, "gates": {target: gates}}, "quick": {...}}, "targets": {"cut": bool, ...}, "ggufs": [...]}`.
/// The gates are there for every row once the original of the same mode has been checked. A result of another suite version is not shown.
pub fn results(p: &Project, name: &str) -> Result<Value> {
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    let tdir = think_dir(p, name);
    let modes_of = |vllm: bool| {
        let mut modes = serde_json::Map::new();
        for (mode, quick) in [("full", false), ("quick", true)] {
            let mut rows = serde_json::Map::new();
            for t in TARGETS {
                if let Some(r) = saved(&tdir, t, quick, vllm) {
                    rows.insert(t.to_string(), r);
                }
            }
            let mut gs = serde_json::Map::new();
            if let Some(orig) = rows.get("original").map(|o| o["main"].clone()) {
                for (t, r) in &rows {
                    if t != "original" {
                        gs.insert(t.clone(), gates(&orig, &r["main"]));
                    }
                }
            }
            modes.insert(mode.into(), json!({"rows": rows, "gates": gs}));
        }
        modes
    };
    let modes = modes_of(false);
    let vllm_modes = modes_of(true);
    let ggufs: Vec<Value> = crate::export::runs::list_exports(p).into_iter().filter(|m| m["gguf_exists"] == json!(true))
        .filter_map(|m| {
            let path = Path::new(m["dir"].as_str()?).join(m["gguf_file"].as_str()?);
            Some(json!({"name": m["name"], "path": path}))
        }).collect();
    Ok(json!({"suite_version": SUITE_VERSION, "modes": modes, "vllm_modes": vllm_modes, "ggufs": ggufs, "guardrails": cut_guardrails(&dir),
              "targets": {"original": heal_source::read_origin(&dir).is_ok(), "cut": dir.join("model").join("config.json").is_file() || heal_source::student_folder(&dir).ok().flatten().is_some(),
                          "healed": dir.join("healed").join("config.json").is_file()}}))
}
