//! Slimmed and grown models: `models/<name>/` in the project holds state.json (status), result.json (what the worker found), scores.json (the layer
//! measurements for the chart), test.json (original vs this model) and `model/` (the new model folder, which is also added to the model list).
//! While the worker writes, the files are in `model.partial/`; it is renamed to `model/` at the end and removed after a failure or a stop.
//! Thinner layers (width) also keep the measurement of a model once per model, data and sizes in `models/_scores/<key>/`, and healing (`heal`) keeps
//! `heal.json` (state), `distill/cache/` (the original's saved answers, kept until the model is deleted), `distill/run/` (the LoRA; its checkpoints are
//! deleted after the merge) and `healed/` (the healed model, which is also added to the model list).
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use serde::{Deserialize, Serialize};

use super::heal_rounds;
use super::heal_source;
use super::plan::{self, Inspect, Mode, ReasoningCalib, ReshapeSpec, Width};
use crate::envmgr::jobs::{run_job, WorkerEvent};
use crate::envmgr::{env_dir, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::paths::config_dir;
use crate::project::Project;
use crate::train::catalog::{self, ModelEntry};
use crate::train::hub::estimate_vram;

pub fn models_dir(p: &Project) -> PathBuf {
    p.folder.join("models")
}

pub(super) fn write_json(path: &Path, v: &Value) -> Result<()> {
    crate::fsx::write_json_atomic(path, v)
}

pub(super) fn read_json(path: &Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

pub(super) fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Folder names come from the window: no path tricks.
pub(super) fn valid_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with('_') || name.contains(['/', '\\']) || name.contains("..") {
        return Err(Error::new("That is not the name of a made model."));
    }
    Ok(())
}

fn set_state(dir: &Path, patch: Value) {
    let mut v = read_json(&dir.join("state.json")).unwrap_or_else(|| json!({}));
    if let (Some(o), Some(pa)) = (v.as_object_mut(), patch.as_object()) {
        for (k, x) in pa {
            o.insert(k.clone(), x.clone());
        }
        o.insert("updated".into(), json!(now()));
    }
    let _ = write_json(&dir.join("state.json"), &v);
}

/// "qwen3.5-27b" from "Qwen3.5 27B" (letters, digits and dots; the rest becomes a dash)
pub(super) fn slug(text: &str) -> String {
    let s: String = text.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '-' }).collect();
    let s = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    s.chars().take(40).collect::<String>().trim_matches('-').to_string()
}

/// `20261113-141500-qwen3.5-27b-slim-48L`
pub fn name_for(src: &ModelEntry, mode: Mode, layers_after: u32) -> String {
    format!("{}-{}-{}-{}L", chrono::Utc::now().format("%Y%m%d-%H%M%S"), slug(&src.name), mode.word(), layers_after)
}

/// The name of a made model by what was done: depth `…-slim-48L`, width `…-slim-w10752`, both `…-slim-60L-w12288` (Expert axes add `-v32` / `-h16`).
pub fn name_for_shape(src: &ModelEntry, mode: Mode, layers_after: u32, width: Option<&Width>, depth_removed: bool) -> String {
    let Some(w) = width else { return name_for(src, mode, layers_after) };
    let mut tail = format!("w{}", w.intermediate);
    if let Some(v) = w.lin_v_heads {
        tail.push_str(&format!("-v{v}"));
    }
    if let Some(h) = w.heads {
        tail.push_str(&format!("-h{h}"));
    }
    let layers = if depth_removed { format!("{layers_after}L-") } else { String::new() };
    format!("{}-{}-{}-{layers}{tail}", chrono::Utc::now().format("%Y%m%d-%H%M%S"), slug(&src.name), mode.word())
}

/// The key of a measurement: model, data version, samples, length and the way it was loaded. The same key is the same measurement.
pub fn scores_key(source_repo: &str, version: Option<u32>, samples: u32, len: u32, load: &str) -> String {
    scores_key_with(source_repo, version, samples, len, load, None)
}

/// `scores_key` for a measuring text with a reasoning part: the same key plus `ReasoningCalib::tag` (another reasoning text is another measurement).
pub fn scores_key_with(source_repo: &str, version: Option<u32>, samples: u32, len: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> String {
    format!("{}-v{}-{samples}x{len}-{load}{}", slug(source_repo), version.unwrap_or(0), reasoning.map(|r| r.tag()).unwrap_or_default())
}

pub fn scores_dir(p: &Project, key: &str) -> PathBuf {
    models_dir(p).join("_scores").join(key)
}

/// Deletes a kept measurement (Expert: "measure again").
pub fn remove_scores(p: &Project, key: &str) -> Result<()> {
    if key.is_empty() || key.contains(['/', '\\']) || key.contains("..") {
        return Err(Error::new("That is not the name of a measurement."));
    }
    let dir = scores_dir(p, key);
    if dir.is_dir() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}

/// A worker event as the window gets it ({"kind": ...}).
pub fn event_json(e: WorkerEvent) -> Value {
    match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(mut v) => {
            let kind = v["type"].as_str().unwrap_or("event").to_string();
            v["kind"] = json!(kind);
            v
        }
    }
}

pub(super) fn python() -> Result<PathBuf> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    Ok(py)
}

// ---------------------------------------------------------------------------------------------------------------- inspect

fn cache_file() -> PathBuf {
    config_dir().join("reshape_inspect.json")
}

/// What identifies the model's files: a folder's config.json modification time (a Hub model never changes under its name).
fn cache_key(m: &ModelEntry) -> String {
    let dir = Path::new(&m.repo);
    if dir.is_dir() {
        let mtime = std::fs::metadata(dir.join("config.json")).and_then(|x| x.modified()).ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
        format!("{}@{mtime}", m.repo)
    } else {
        m.repo.clone()
    }
}

/// What the model is made of (layers, blocks, parameters per layer ...), from the worker; remembered per model so the page opens fast the next time.
pub fn inspect(m: &ModelEntry, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Inspect> {
    let key = cache_key(m);
    let mut cache = read_json(&cache_file()).unwrap_or_else(|| json!({}));
    // an answer from before thinner layers existed has no "width_supported": it is asked again
    if let Some(ins) = cache.get(&key).filter(|v| v.get("width_supported").is_some()).and_then(|v| serde_json::from_value::<Inspect>(v.clone()).ok()) {
        return Ok(ins);
    }
    let py = python()?;
    let job = json!({"job": "reshape_inspect", "source": {"repo": m.repo, "family": m.family}});
    let result = run_job(&py, &worker_dir(), &job, kill, &mut |e| on_event(event_json(e)))?;
    let ins: Inspect = serde_json::from_value(result.clone()).map_err(|e| Error::new(format!("The worker's description of the model was not readable: {e}")))?;
    cache[&key] = result;
    let _ = write_json(&cache_file(), &cache);
    Ok(ins)
}

// -------------------------------------------------------------------------------------------------------------------- job

/// The job the worker gets. Slimming by measuring needs a saved training-data version (the layers are measured on the project's own text).
pub fn build_job(p: &Project, m: &ModelEntry, spec: &ReshapeSpec, ins: &Inspect, name: &str) -> Result<Value> {
    plan::validate(spec, ins)?;
    let measures = spec.mode == Mode::Prune && (spec.drop_layers.is_none() || spec.width.is_some());
    let version = crate::train::runs::latest_version(p).map(|(n, _)| n);
    let train = version.map(|n| p.folder.join("datasets").join(format!("v{n}")).join("train.jsonl")).filter(|f| f.is_file());
    if measures && train.is_none() {
        return Err(Error::new("Save a training-data version in Step 2 first: the layers are measured on your own text."));
    }
    let dir = models_dir(p).join(name);
    let samples = if spec.width.is_some() { spec.width_samples } else { spec.calib_samples };
    let mut job = json!({
        "job": "reshape", "source": {"repo": m.repo, "family": m.family}, "mode": spec.mode, "blocks": spec.blocks, "strategy": spec.strategy,
        "protect": {"first": spec.protect_first, "last": spec.protect_last}, "drop_layers": spec.drop_layers,
        "grow": {"layout": "solar", "init": spec.init},
        "calibration": calibration_json(train.as_deref(), samples, spec.calib_len, version, spec.reasoning.as_ref()),
        "load": spec.load, "check": !spec.skip_check, "shard_gb": 5,
        "out": dir.join("model.partial"), "final": dir.join("model"),
    });
    if let Some(w) = &spec.width {
        job["width"] = json!({"intermediate_size": w.intermediate, "linear_num_value_heads": w.lin_v_heads, "num_attention_heads": w.heads});
        job["scores_dir"] = json!(scores_dir(p, &scores_key_with(&m.repo, version, samples, spec.calib_len, &spec.load, spec.reasoning.as_ref())));
        if spec.width_scoring.as_deref() == Some("output") {
            job["width_scoring"] = json!("output");
        }
        if let Some(r) = &spec.reconstruct {
            job["reconstruct"] = json!({"variant": r.variant, "alpha": r.alpha});
        }
    }
    Ok(job)
}

/// The `calibration` of a worker job: the project's training file, how many samples of which length and, when asked, the reasoning part.
fn calibration_json(train: Option<&Path>, samples: u32, len: u32, version: Option<u32>, reasoning: Option<&ReasoningCalib>) -> Value {
    let mut c = json!({"train": train, "samples": samples, "max_len": len, "seed": 3407, "version": version});
    if let Some(r) = reasoning {
        c["reasoning"] = r.json();
    }
    c
}

fn active() -> &'static std::sync::Mutex<std::collections::HashSet<PathBuf>> {
    static A: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<PathBuf>>> = std::sync::OnceLock::new();
    A.get_or_init(Default::default)
}

/// Marks a made-model folder as being written by THIS program while it is alive (opening a project must not take it for an interrupted one).
struct Active(PathBuf);

impl Active {
    fn begin(dir: &Path) -> Result<Active> {
        let mut a = active().lock().map_err(|_| Error::new("Internal error (lock)."))?;
        if !a.is_empty() {
            return Err(Error::new("Another model is being made right now. Wait until it is done, or stop it."));
        }
        a.insert(dir.to_path_buf());
        Ok(Active(dir.to_path_buf()))
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        if let Ok(mut a) = active().lock() {
            a.remove(&self.0);
        }
    }
}

fn layers_after(ins: &Inspect, spec: &ReshapeSpec) -> u32 {
    match (&spec.drop_layers, spec.mode) {
        (Some(d), Mode::Prune) => ins.layers.saturating_sub(d.len() as u32),
        (_, Mode::Prune) => ins.layers.saturating_sub(spec.blocks * ins.period),
        (_, Mode::Grow) => ins.layers + spec.blocks * ins.period,
    }
}

/// Makes the model: inspects the source, runs the job, and on success keeps the result and adds the new model to the model list.
/// `stop` (the Stop button) asks the worker to stop between tensors (a STOP file); after 60 seconds it is killed. Returns the worker's result
/// plus "name" and "model_id" (or `{"stopped": true}`). Every event goes to `on_event` as JSON ({"kind": ...}).
pub fn execute(p: &Project, spec: &ReshapeSpec, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    execute_with_grace(p, spec, stop, Duration::from_secs(60), on_event)
}

pub fn execute_with_grace(p: &Project, spec: &ReshapeSpec, stop: Arc<AtomicBool>, grace: Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let m = catalog::find(&spec.source_model_id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
    let py = python()?;
    let ins = inspect(&m, &stop, on_event)?;
    plan::validate(spec, &ins)?;
    let depth_removed = spec.blocks > 0 || spec.drop_layers.is_some();
    let make_name = || name_for_shape(&m, spec.mode, layers_after(&ins, spec), spec.width.as_ref(), depth_removed);
    let mut name = make_name();
    // two models made in the same second (or a folder left over) never share a folder
    let mut n = 1;
    while models_dir(p).join(&name).exists() {
        n += 1;
        name = format!("{}-{n}", make_name());
    }
    let job = build_job(p, &m, spec, &ins, &name)?;
    let dir = models_dir(p).join(&name);
    let _active = Active::begin(&dir)?;
    std::fs::create_dir_all(&dir)?;
    write_json(&dir.join("config.json"), &job)?;
    set_state(&dir, json!({"name": name, "status": "running", "created": now(), "spec": spec, "source_model_id": m.id, "source_name": m.name, "model_id": null, "error": null}));
    let outcome = crate::train::runs::drive(&py, &dir, &job, stop, grace, &mut |e| {
        let ev = event_json(e);
        if ev["kind"] == "reshape_scores" {
            let _ = write_json(&dir.join("scores.json"), &ev);
        }
        if ev["kind"] == "width_scores" {
            let _ = write_json(&dir.join("width_scores.json"), &ev);
        }
        on_event(ev);
    });
    let partial = dir.join("model.partial");
    match outcome {
        Ok(result) if result["stopped"].as_bool().unwrap_or(false) => {
            let _ = std::fs::remove_dir_all(&partial);
            set_state(&dir, json!({"status": "stopped"}));
            Ok(json!({"stopped": true, "name": name}))
        }
        Ok(mut result) => {
            let _ = std::fs::remove_dir_all(&partial);
            if !dir.join("model").join("config.json").is_file() {
                let e = Error::new("The worker finished but the new model is missing.");
                set_state(&dir, json!({"status": "failed", "error": e.0}));
                return Err(e);
            }
            write_json(&dir.join("result.json"), &result)?;
            set_state(&dir, json!({"status": "done"}));
            let id = register(p, &name)?;
            result["name"] = json!(name);
            result["model_id"] = json!(id);
            Ok(result)
        }
        Err(e) if e.is_cancelled() => {
            let _ = std::fs::remove_dir_all(&partial);
            set_state(&dir, json!({"status": "stopped"}));
            Err(Error::new("Stopped."))
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&partial);
            set_state(&dir, json!({"status": "failed", "error": e.0}));
            Err(e)
        }
    }
}

/// Auto-cut's clean-up of a made model whose heal has passed its check: the unhealed model (`model/`, and its entry in the model list) and the teacher's saved answers go;
/// the folder, its records, the thinking checks and a healed model stay. Refused while the model is being made.
pub fn remove_cut(p: &Project, name: &str) -> Result<()> {
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    if !dir.is_dir() {
        return Ok(());
    }
    if active().lock().map(|a| a.contains(&dir)).unwrap_or(false) {
        return Err(Error::new("This model is being made right now. Stop it first."));
    }
    let model = dir.join("model");
    let repo = model.canonicalize().unwrap_or(model.clone()).to_string_lossy().to_string();
    if let Some(id) = catalog::models().into_iter().find(|m| m.repo == repo).map(|m| m.id) {
        catalog::remove_custom(&id)?;
    }
    for sub in [model, dir.join("distill").join("cache")] {
        if sub.is_dir() {
            std::fs::remove_dir_all(sub)?;
        }
    }
    set_state(&dir, json!({"model_removed": true}));
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------- the list

/// Adds `models/<name>/model` to the model list so it can be chosen as a base model in Step 3. Returns the model's id (the same one when called again).
pub fn register(p: &Project, name: &str) -> Result<String> {
    register_folder(p, name, "model")
}

/// The same for `models/<name>/healed` (id `custom-local-<name>-healed`).
pub fn register_healed(p: &Project, name: &str) -> Result<String> {
    register_folder(p, name, "healed")
}

pub(super) fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn register_folder(p: &Project, name: &str, sub: &str) -> Result<String> {
    valid_name(name)?;
    let healed = sub == "healed";
    let dir = models_dir(p).join(name);
    let model = dir.join(sub);
    let repo = model.canonicalize().unwrap_or(model.clone()).to_string_lossy().to_string();
    let id_key = if healed { "healed_model_id" } else { "model_id" };
    if let Some(m) = catalog::models().into_iter().find(|m| m.repo == repo) {
        set_state(&dir, json!({id_key: m.id}));
        return Ok(m.id);
    }
    let mut entry = crate::train::hub::lookup(&repo)?.entry;
    if healed && heal_source::is_external(&dir) {
        return register_external_healed(&dir, entry, &model, name);
    }
    let meta = read_json(&model.join("ndforge_reshape.json")).unwrap_or(Value::Null);
    let state = read_json(&dir.join("state.json")).unwrap_or(Value::Null);
    let source = state["source_model_id"].as_str().and_then(catalog::find);
    let origin = meta["origin"]["repo"].as_str().map(String::from).or_else(|| source.as_ref().map(|s| s.repo.clone()));
    let (before, after) = (meta["origin"]["layers"].as_u64().unwrap_or(0), meta["layers"].as_u64().unwrap_or(0));
    let slimmed = meta["mode"] == "prune";
    let wide = meta["width"].is_object();
    let base_name = source.as_ref().map(|s| s.name.clone()).unwrap_or_else(|| entry.name.clone());
    // "slim 48 layers", "slim MLP 10,752", "slim 60 layers, MLP 12,288"
    let wtxt = wide.then(|| format!("MLP {}", thousands(meta["width"]["intermediate_size"][1].as_u64().unwrap_or(0))));
    let what = match (before != after, &wtxt) {
        (true, Some(w)) => format!("{after} layers, {w}"),
        (false, Some(w)) => w.clone(),
        _ => format!("{after} layers"),
    };
    entry.id = format!("custom-local-{name}{}", if healed { "-healed" } else { "" });
    entry.name = format!("{base_name} · {} {what}{}", if slimmed { "slim" } else { "grow" }, if healed { " · healed" } else { "" });
    if let Some(s) = &source {
        entry.family = s.family.clone();
    }
    if let Some(total) = meta["params"].as_u64().filter(|t| *t > 0) {
        entry.params_b = total as f64 / 1e9;
        let (lora, qlora) = estimate_vram(entry.params_b);
        entry.vram_lora_gb = lora;
        entry.vram_qlora_gb = qlora;
    }
    entry.derived_from = origin.clone();
    let how = if slimmed { "slimmed" } else { "grown" };
    entry.reshape = Some(match (healed, &wtxt) {
        (true, Some(w)) => format!("healed: {before} → {after} layers, {w}"),
        (true, None) => format!("healed: {before} → {after} layers"),
        (false, Some(w)) => format!("{how}: {before} → {after} layers, {w} (from {})", thousands(meta["width"]["intermediate_size"][0].as_u64().unwrap_or(0))),
        (false, None) => format!("{how}: {before} → {after} layers"),
    });
    entry.good_for = format!("Made on the Slim or grow page from {}.", origin.as_deref().unwrap_or("another model"));
    entry.note = if healed {
        "Healed: the original model taught this thinner model to answer like it did. It is the better base for Step 3. Memory figures are estimates.".to_string()
    } else if wide {
        "A thinner model answers worse than the original until it has been healed (the Heal step) and trained again on your world (Step 3), then test it. Memory figures are estimates.".to_string()
    } else if slimmed {
        "A slimmed model answers worse than the original until it is trained again on your world (Step 3), then test it. Memory figures are estimates.".to_string()
    } else {
        "A grown model starts out answering exactly like the original; a world's data alone gives little gain, and it needs more memory than the original. Memory figures are estimates.".to_string()
    };
    let id = entry.id.clone();
    catalog::add_custom(entry)?;
    set_state(&dir, json!({id_key: id}));
    Ok(id)
}

/// The healed copy of a model the Slim page did not make: "<student> · healed", derived from the student (its folder or Hugging Face id).
fn register_external_healed(dir: &Path, mut entry: ModelEntry, model: &Path, name: &str) -> Result<String> {
    let h = heal_source::read(dir).ok_or_else(|| Error::new("This heal has no heal_source.json."))?;
    let meta = read_json(&model.join("ndforge_reshape.json")).unwrap_or(Value::Null);
    let student = match &h.student {
        heal_source::Source::Folder { path } => path.trim().to_string(),
        heal_source::Source::Hub { id } => id.trim().to_string(),
        heal_source::Source::Made { name } => name.clone(),
    };
    let teacher = meta["origin"]["repo"].as_str().map(String::from).or_else(|| h.teacher.as_ref().map(|t| t.label())).unwrap_or_else(|| "another model".into());
    entry.id = format!("custom-local-{name}-healed");
    entry.name = format!("{} · healed", h.student_name);
    if let Some(total) = meta["params"].as_u64().filter(|t| *t > 0) {
        entry.params_b = total as f64 / 1e9;
        let (lora, qlora) = estimate_vram(entry.params_b);
        entry.vram_lora_gb = lora;
        entry.vram_qlora_gb = qlora;
    }
    entry.derived_from = Some(student.clone());
    entry.reshape = Some(format!("healed: {} taught by {}", h.student_name, h.teacher_name));
    entry.good_for = format!("Healed on the Heal page: {} taught by {teacher}.", h.student_name);
    entry.note = "Healed: another model taught this one to answer like it did (distillation). The copy is a derivative of both models: check their licences. It is the better base for Step 3. Memory figures are estimates.".to_string();
    let id = entry.id.clone();
    catalog::add_custom(entry)?;
    set_state(dir, json!({"healed_model_id": id}));
    Ok(id)
}

fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|d| d.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0)
}

// ------------------------------------------------------------------------------------------------------ the shared teacher cache
// The teacher's answers (the heavy part files) live in `models/_teacher_cache/<id>/`, shared by every heal that has the same teacher, tokenizer and rows (the worker
// chooses the folder, see distilljob.py "the shared cache"). A model's own `distill/cache/` then only holds index.json (whose `store` says where the parts are) and rows.npz.

/// The folder of the shared teacher answers of this project.
pub fn teacher_cache_dir(p: &Project) -> PathBuf {
    models_dir(p).join("_teacher_cache")
}

fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The shared folder a model's cache points at (`store` in its index.json, relative to the cache folder), or None for a cache of its own.
fn store_of(cache: &Path) -> Option<PathBuf> {
    let store = read_json(&cache.join("index.json"))?["store"].as_str()?.to_string();
    Some(lexical(&cache.join(store)))
}

/// Disk the model's teacher answers take: its own folder and the parts it uses in the shared one.
fn cache_bytes(cache: &Path) -> u64 {
    let own = dir_size(cache);
    let shared: u64 = match (store_of(cache), read_json(&cache.join("index.json"))) {
        (Some(store), Some(idx)) => idx["parts"].as_array().map(|ps| ps.iter().filter_map(|p| p["file"].as_str()).filter_map(|f| std::fs::metadata(store.join(f)).ok()).map(|m| m.len()).sum()).unwrap_or(0),
        _ => 0,
    };
    own + shared
}

/// Every shared teacher cache folder with the models whose cache points at it.
fn teacher_cache_uses(p: &Project) -> Vec<(PathBuf, Vec<String>)> {
    let base = teacher_cache_dir(p);
    let mut out: Vec<(PathBuf, Vec<String>)> = std::fs::read_dir(&base).map(|d| d.flatten().filter(|e| e.path().is_dir()).map(|e| (lexical(&e.path()), Vec::new())).collect()).unwrap_or_default();
    if let Ok(rd) = std::fs::read_dir(models_dir(p)) {
        for e in rd.flatten() {
            if let Some(store) = store_of(&e.path().join("distill").join("cache")) {
                if let Some(slot) = out.iter_mut().find(|(d, _)| *d == store) {
                    slot.1.push(e.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    out
}

fn any_heal_active() -> bool {
    active().lock().map(|a| !a.is_empty()).unwrap_or(true)
}

/// Deletes the shared teacher answers that no model uses any more (after a model was deleted). Does nothing while a model is being made: its cache folder may not
/// point at its shared folder yet. Returns how many folders went.
pub fn sweep_teacher_caches(p: &Project) -> usize {
    if any_heal_active() {
        return 0;
    }
    teacher_cache_uses(p).into_iter().filter(|(_, users)| users.is_empty()).filter(|(d, _)| std::fs::remove_dir_all(d).is_ok()).count()
}

/// The end of an Auto-cut run: the shared teacher answers that only models of this run (`names`) use are not needed any more (the last stage has been healed), so they go,
/// and so do those models' links to them. A shared folder that any other model uses, such as a heal the person started on the Slim page, stays. Returns how many went.
pub fn drop_teacher_caches_of(p: &Project, names: &[String]) -> usize {
    if any_heal_active() {
        return 0;
    }
    let mut gone = 0;
    for (dir, users) in teacher_cache_uses(p) {
        if users.iter().all(|u| names.contains(u)) && std::fs::remove_dir_all(&dir).is_ok() {
            for u in users {
                let _ = std::fs::remove_dir_all(models_dir(p).join(u).join("distill").join("cache"));
            }
            gone += 1;
        }
    }
    gone
}

/// Every made model, newest first, with its state, a short summary of the result and whether it has been tested.
pub fn list(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(models_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let dir = e.path();
        let mut st = read_json(&dir.join("state.json"))?;
        if let Some(r) = read_json(&dir.join("result.json")) {
            st["summary"] = json!({
                "mode": r["mode"], "layers_before": r["layers_before"], "layers_after": r["layers_after"], "dropped": r["dropped"], "duplicated": r["duplicated"],
                "params_before": r["params_before"], "params_after": r["params_after"], "loss_before": r["loss_before"], "loss_after": r["loss_after"],
                "ppl_before": r["ppl_before"], "ppl_after": r["ppl_after"], "stable": r["stable"], "chosen_blocks": r["chosen_blocks"],
                "boundary_distance": r["boundary_distance"], "notes": r["notes"], "load": r["load"], "shape": r["shape"], "width": r["width"],
            });
        }
        st["scores"] = read_json(&dir.join("scores.json")).unwrap_or(Value::Null);
        st["width_scores"] = read_json(&dir.join("width_scores.json")).unwrap_or(Value::Null);
        st["heal"] = read_json(&dir.join("heal.json")).unwrap_or(Value::Null);
        st["healed"] = json!(dir.join("healed").join("config.json").is_file());
        st["test_healed"] = read_json(&dir.join("test_healed.json")).map(|t| json!({"scores": t["scores"], "verdict": t["verdict"], "n": t["n"], "method": t["method"], "created": t["created"], "against": t["against"], "loaded": t["loaded"], "original": t["original"]})).unwrap_or(Value::Null);
        st["cache_gb"] = json!((cache_bytes(&dir.join("distill").join("cache")) as f64 / 1e9 * 10.0).round() / 10.0);
        let test = read_json(&dir.join("test.json"));
        st["tested"] = json!(test.is_some());
        // the headline and the scores (not the hundreds of answers) for the page
        st["test"] = test.map(|t| json!({"scores": t["scores"], "verdict": t["verdict"], "n": t["n"], "method": t["method"], "created": t["created"], "against": t["against"], "loaded": t["loaded"], "original": t["original"]})).unwrap_or(Value::Null);
        st["has_model"] = json!(dir.join("model").join("config.json").is_file());
        st["size_gb"] = json!(if st["has_model"] == true { (dir_size(&dir.join("model")) as f64 / 1e9 * 10.0).round() / 10.0 } else { 0.0 });
        st["healed_size_gb"] = json!(if st["healed"] == true { (dir_size(&dir.join("healed")) as f64 / 1e9 * 10.0).round() / 10.0 } else { 0.0 });
        Some(st)
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

/// Deletes a made model (its folder and its entry in the model list). Refused while it is being made or while a training run used it as its base.
pub fn remove(p: &Project, name: &str) -> Result<()> {
    remove_model(p, name)?;
    sweep_teacher_caches(p);
    Ok(())
}

/// `remove` for Auto-cut's clean-up: the teacher's shared answers stay (the next stage still needs them; they go at the end of the run, see `drop_teacher_caches_of`).
pub fn remove_keeping_teacher_cache(p: &Project, name: &str) -> Result<()> {
    remove_model(p, name)
}

fn remove_model(p: &Project, name: &str) -> Result<()> {
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    if !dir.is_dir() {
        return Err(Error::new("That model was not found."));
    }
    let live = active().lock().map(|a| a.contains(&dir)).unwrap_or(false);
    if live {
        return Err(Error::new("This model is being made right now. Stop it first."));
    }
    let folders: Vec<(PathBuf, String)> = ["model", "healed"].iter().map(|sub| {
        let f = dir.join(sub);
        let repo = f.canonicalize().unwrap_or(f.clone()).to_string_lossy().to_string();
        (f, repo)
    }).collect();
    if let Ok(rd) = std::fs::read_dir(crate::train::runs::runs_dir(p)) {
        for e in rd.flatten() {
            let used = read_json(&e.path().join("config.json")).is_some_and(|c| c["model"]["repo"].as_str().is_some_and(|r| folders.iter().any(|(f, repo)| r == repo || Path::new(r) == f)));
            if used {
                return Err(Error::new(format!("Run {} was trained on this model. Delete that run first, or keep the model.", e.file_name().to_string_lossy())));
            }
        }
    }
    for (_, repo) in &folders {
        if let Some(id) = catalog::models().into_iter().find(|m| &m.repo == repo).map(|m| m.id) {
            catalog::remove_custom(&id)?;
        }
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A model still marked "running" when a project is opened was cut off (the program was closed, or it crashed): it becomes "failed" and its half-written
/// folder is removed. A model being made by this very program is left alone. Returns how many were marked.
pub fn mark_interrupted(p: &Project) -> usize {
    let mut n = 0;
    let Ok(rd) = std::fs::read_dir(models_dir(p)) else { return 0 };
    for e in rd.flatten() {
        let dir = e.path();
        // a heal of a model the Slim page did not make has no state.json: only its heal.json can be cut off
        let st = read_json(&dir.join("state.json")).unwrap_or(Value::Null);
        if st.is_null() && !heal_source::is_external(&dir) {
            continue;
        }
        let live = active().lock().map(|a| a.contains(&dir)).unwrap_or(false);
        // healing that was cut off can be continued: its checkpoints and the teacher's answers are kept
        if !live && read_json(&dir.join("heal.json")).is_some_and(|h| h["status"] == "running") {
            let _ = std::fs::remove_dir_all(dir.join("healed.partial"));
            set_heal(&dir, json!({"status": "stopped", "interrupted": true, "error": "The program was closed while this model was being healed. Continue to go on where it stopped."}));
            n += 1;
        }
        if st["status"] == "running" && !live {
            let _ = std::fs::remove_dir_all(dir.join("model.partial"));
            set_state(&dir, json!({"status": "failed", "error": "The program was closed while this model was being made. Start it again."}));
            n += 1;
        }
    }
    n
}

// -------------------------------------------------------------------------------------------------------------------- test

/// Tests a made model against the model it was made from: the same questions as Step 4 (about the project's own world), both answered by the worker
/// (a model too big for the card loads in 4-bit), scored the same way. Saved as `models/<name>/test.json`.
pub fn test(p: &Project, name: &str, judge: Option<&crate::teacher::Teacher>, vram_gb: Option<f64>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    test_target(p, name, "slim", judge, vram_gb, kill, on_event)
}

/// `target`: "slim" (the made model, `test.json`) or "healed" (the healed model, `test_healed.json`); both against the model they were made from.
pub fn test_target(p: &Project, name: &str, target: &str, judge: Option<&crate::teacher::Teacher>, vram_gb: Option<f64>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    use crate::evaluate::runs::{execute_pair, Pair, Side};
    valid_name(name)?;
    let healed = match target {
        "slim" => false,
        "healed" => true,
        other => return Err(Error::new(format!("Unknown test target: {other}"))),
    };
    let dir = models_dir(p).join(name);
    let model = dir.join(if healed { "healed" } else { "model" });
    if !model.join("config.json").is_file() {
        return Err(Error::new(if healed { "This model has not been healed yet." } else { "That model is not ready (it was stopped or failed)." }));
    }
    let meta = read_json(&model.join("ndforge_reshape.json")).ok_or_else(|| Error::new("This model has no ndforge_reshape.json, so its original is unknown."))?;
    let origin = meta["origin"]["repo"].as_str().ok_or_else(|| Error::new("This model's original is unknown."))?.to_string();
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: the test asks questions about your world."))?;
    let big = |params: Option<f64>| vram_gb.zip(params).is_some_and(|(v, b)| b * 2.0 + 3.0 > v * 0.9);
    let pair = Pair {
        original: origin, family: meta["origin"]["family"].as_str().unwrap_or("generic").to_string(), trained: Side::Full(model.clone()),
        four_bit: (big(meta["origin"]["params"].as_f64().map(|x| x / 1e9)), big(meta["params"].as_f64().map(|x| x / 1e9))), against: "origin",
    };
    let (label, file) = if healed { (format!("model:{name}-healed"), "test_healed.json") } else { (format!("model:{name}"), "test.json") };
    execute_pair(p, &pair, version, &label, &dir.join(file), 30, judge, kill, on_event)
}

pub fn read_test(p: &Project, name: &str) -> Option<Value> {
    valid_name(name).ok()?;
    read_json(&models_dir(p).join(name).join("test.json"))
}

pub fn read_test_healed(p: &Project, name: &str) -> Option<Value> {
    valid_name(name).ok()?;
    read_json(&models_dir(p).join(name).join("test_healed.json"))
}

// ------------------------------------------------------------------------------------------------------------------ heal

fn half() -> f64 {
    0.5
}
fn one_f() -> f64 {
    1.0
}
fn lr_default() -> f64 {
    1e-4
}

/// What the person chose for healing (Expert fields have defaults). `tokens_m`: millions of tokens, one of 10 / 20 / 40 / 80.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealSpec {
    pub tokens_m: u32,
    /// The exact number of tokens, instead of `tokens_m` million: only for tests and the end-to-end run on a tiny model (the window never sends it; 1,000 to 100,000,000).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_exact: Option<u64>,
    /// the share of the heal tokens that is general text (0 = the project's own text only, no download)
    #[serde(default = "half")]
    pub general_share: f64,
    /// where the general text comes from (a preset, a dataset the person added, a file); None = the program's default (`general::DEFAULT_ID`)
    #[serde(default)]
    pub general: Option<crate::general::GeneralSource>,
    /// older windows sent only a dataset id and a column
    #[serde(default)]
    pub general_dataset: Option<String>,
    #[serde(default)]
    pub general_column: Option<String>,
    /// weight of the KL term; below 1 the training text's own next words count too (Expert)
    #[serde(default = "one_f")]
    pub alpha: f64,
    /// LoRA size; None = by the student's size (32 from 15 B, else 64)
    #[serde(default)]
    pub rank: Option<u32>,
    #[serde(default = "lr_default")]
    pub lr: f64,
    /// "cache" | "online"; None = what fits the card
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    // Healing settings (Expert, docs/HEAL_SETTINGS.md): None = the automatic value for this student and card (`plan::heal_defaults`). Older windows send none of them.
    /// sequence length: 512 | 1024 | 2048 | 4096 | 8192
    #[serde(default)]
    pub seq: Option<u32>,
    /// sequences per step on the card: 1 | 2 | 4
    #[serde(default)]
    pub batch: Option<u32>,
    /// sequences per update (batch x accumulation, default 32); the accumulation is computed from it
    #[serde(default)]
    pub per_update: Option<u32>,
    /// "auto" | "on" | "off"
    #[serde(default)]
    pub checkpointing: Option<String>,
    /// "qlora" | "lora"
    #[serde(default)]
    pub method: Option<String>,
    /// "adamw_8bit" | "adamw"
    #[serde(default)]
    pub optimizer: Option<String>,
    /// tokens per chunk of the KL loss, 64 to 512
    #[serde(default)]
    pub kl_chunk: Option<u32>,
    /// how many of the original's answers per token are kept (top-k), 8 to 128; changing it makes the original answer again
    #[serde(default)]
    pub k: Option<u32>,
    /// how the original is loaded: "auto" | "4bit" | "16bit"
    #[serde(default)]
    pub teacher_load: Option<String>,
    #[serde(default)]
    pub warmup: Option<u32>,
    /// QLoRA only: keep PEFT's 32-bit copy of the embedding and the output layer (the old behaviour, about 4 to 5 GB more graphics memory). None = off. It changes only
    /// the numerics of the frozen tables: the training key and the teacher's cache key do not include it (docs/HEAL_SETTINGS.md).
    #[serde(default)]
    pub kbit_upcast: Option<bool>,
    /// "on" (the default) | "off": keep the output layer's logits for the backward pass instead of making them again (faster, the same loss, about tokens x vocabulary x 10 bytes
    /// more graphics memory). Student-side only: the training key and the teacher's cache key do not include it. None = on; the job then carries no key for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_checkpoint: Option<String>,
    /// "thinking": heal so that the model keeps thinking (`plan::THINKING_RECIPE`). The window sends the recipe's numbers (`heal_options` says them: tokens, general share);
    /// what it leaves empty is filled here: the general text (the reasoning traces of Qwen3.8-27B), the sequence length, the batch and the LoRA size. Written to `heal.json`
    /// and, once merged, to the healed model's `ndforge_reshape.json` (`healed.recipe`), where the export finds it. Only for Qwen3.5 / Qwen3.8 students.
    #[serde(default)]
    pub recipe: Option<String>,
    /// Heal in rounds with a thinking check after each (experimental, docs/HEAL_ROUNDS.md). None = exactly the heal as it always was (the key is then not even written).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounds: Option<super::heal_rounds::RoundSpec>,
    /// Make the original's answers with vLLM (experimental, docs/VLLM.md; needs the settings' switches too). None = the heal as it always was (the key is then not even written).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vllm: Option<HealVllm>,
}

/// The Heal card's vLLM option.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealVllm {
    /// the original's answers (the teacher cache) are made by vLLM
    #[serde(default)]
    pub teacher_cache: bool,
    /// only for "Check the vLLM teacher": measure the 8-bit KV cache ("fp8") against the exact reference
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_dtype: Option<String>,
}

impl HealSpec {
    /// vLLM is asked for the original's answers.
    pub fn vllm_teacher(&self) -> bool {
        self.vllm.as_ref().is_some_and(|v| v.teacher_cache)
    }

    pub fn thinking(&self) -> bool {
        self.recipe.as_deref() == Some(plan::THINKING_RECIPE)
    }

    /// How many tokens the heal reads.
    pub fn tokens(&self) -> u64 {
        self.tokens_exact.unwrap_or(self.tokens_m as u64 * 1_000_000)
    }

    pub fn validate(&self) -> Result<()> {
        match self.tokens_exact {
            Some(n) if !(1_000..=100_000_000).contains(&n) => return Err(Error::new("The exact number of tokens must be between 1,000 and 100,000,000.")),
            Some(_) => {}
            None if !plan::HEAL_TOKENS_M.contains(&self.tokens_m) => return Err(Error::new("Heal on 10, 20, 40 or 80 million tokens.")),
            None => {}
        }
        if !(0.0..=0.9).contains(&self.general_share) || !(0.0..=1.0).contains(&self.alpha) || !self.rank.is_none_or(|r| (4..=256).contains(&r)) || !(1e-6..=1e-2).contains(&self.lr) {
            return Err(Error::new("Healing settings out of range: general share 0 to 0.9, KL weight 0 to 1, LoRA size 4 to 256, learning rate 0.000001 to 0.01."));
        }
        if self.mode.as_deref().is_some_and(|m| !["cache", "online"].contains(&m)) {
            return Err(Error::new("Unknown way of healing."));
        }
        if self.recipe.as_deref().is_some_and(|r| r != plan::THINKING_RECIPE) {
            return Err(Error::new("Unknown healing recipe. The only one is \"thinking\"."));
        }
        let bad = |what: &str| Err(Error::new(format!("Healing settings out of range: {what}.")));
        if self.seq.is_some_and(|s| ![512, 1024, 2048, 4096, 8192].contains(&s)) {
            return bad("sequence length 512, 1024, 2048, 4096 or 8192");
        }
        if self.batch.is_some_and(|b| ![1, 2, 4].contains(&b)) {
            return bad("batch per step 1, 2 or 4");
        }
        if self.per_update.is_some_and(|u| !(1..=256).contains(&u)) {
            return bad("sequences per update 1 to 256");
        }
        if self.checkpointing.as_deref().is_some_and(|c| !["auto", "on", "off"].contains(&c)) || self.method.as_deref().is_some_and(|m| !["qlora", "lora"].contains(&m)) {
            return bad("gradient checkpointing auto, on or off; method qlora or lora");
        }
        if self.head_checkpoint.as_deref().is_some_and(|c| !["on", "off"].contains(&c)) {
            return bad("output-layer checkpointing on or off");
        }
        if self.optimizer.as_deref().is_some_and(|o| !["adamw_8bit", "adamw"].contains(&o)) || self.teacher_load.as_deref().is_some_and(|l| !["auto", "4bit", "16bit"].contains(&l)) {
            return bad("optimizer adamw_8bit or adamw; original loaded auto, 4bit or 16bit");
        }
        if self.kl_chunk.is_some_and(|c| !(32..=1024).contains(&c)) || self.k.is_some_and(|k| !(8..=128).contains(&k)) || self.warmup.is_some_and(|w| w > 1000) {
            return bad("KL chunk 32 to 1024, top-k 8 to 128, warm-up 0 to 1000");
        }
        if let Some(r) = &self.rounds {
            r.validate(self)?;
            if self.mode.as_deref() == Some("online") {
                return bad("healing in rounds uses the saved answers of the original (not the online way)");
            }
        }
        Ok(())
    }
}

/// The program's default general text for healing: open (Apache-2.0) chat data; the page says it is downloaded.
pub const DEFAULT_GENERAL_DATASET: &str = crate::general::DEFAULT_ID;

impl HealSpec {
    /// The source of the general text with the gaps filled in, or None when healing uses the project's text only. Fails when a preset needs an acknowledgement
    /// that was not given.
    pub fn general_source(&self) -> Result<Option<crate::general::GeneralSource>> {
        if self.general_share <= 0.0 {
            return Ok(None);
        }
        let legacy = self.general_dataset.as_ref().filter(|d| !d.trim().is_empty()).map(|d| crate::general::GeneralSource { id: Some(d.clone()), column: self.general_column.clone(), ..Default::default() });
        let recipe = self.thinking().then(|| crate::general::GeneralSource::hub(plan::THINKING_GENERAL_ID));
        crate::general::resolve(self.general.as_ref().or(legacy.as_ref()).or(recipe.as_ref()), true).map(Some)
    }
}

/// What a healing run will use: the automatic settings for this student and card (`plan::heal_defaults`) with the person's choices on top.
#[derive(Debug, Clone, PartialEq)]
pub struct HealChoice {
    pub knobs: plan::HealKnobs,
    /// "adamw_8bit" | "adamw"
    pub optimizer: String,
    pub k: u32,
    pub teacher_load: Option<String>,
    pub warmup: u32,
}

pub fn heal_choice(spec: &HealSpec, student_b: f64, vram_gb: Option<f64>, d: &plan::HealDims) -> HealChoice {
    let def = plan::heal_defaults(student_b, vram_gb, d);
    // the thinking recipe changes what the automatic values are; what the person chose still wins
    let (rseq, rbatch, rrank) = if spec.thinking() { plan::thinking_knobs(student_b, vram_gb, d) } else { (def.seq, def.batch, def.rank) };
    let batch = spec.batch.unwrap_or(rbatch);
    let per_update = spec.per_update.unwrap_or(def.batch * def.grad_accum).max(batch);
    let mut knobs = plan::HealKnobs {
        seq: spec.seq.unwrap_or(rseq), batch, grad_accum: per_update.div_ceil(batch).max(1), checkpointing: def.checkpointing, rank: spec.rank.unwrap_or(rrank),
        kl_chunk: spec.kl_chunk.unwrap_or(def.kl_chunk), qlora: match spec.method.as_deref() { Some("qlora") => true, Some("lora") => false, _ => def.qlora }, upcast: false,
        head_ckpt: spec.head_checkpoint.as_deref() != Some("off"),
    };
    knobs.upcast = knobs.qlora && spec.kbit_upcast.unwrap_or(false);      // 16-bit LoRA has no such copy
    knobs.checkpointing = match spec.checkpointing.as_deref() {
        Some("on") => true,
        Some("off") => false,
        _ => plan::heal_checkpointing_auto(student_b, d, &knobs, vram_gb),
    };
    HealChoice { knobs, optimizer: spec.optimizer.clone().unwrap_or_else(|| "adamw_8bit".into()), k: spec.k.unwrap_or(32), teacher_load: spec.teacher_load.clone().filter(|l| l != "auto"), warmup: spec.warmup.unwrap_or(30) }
}

/// The settings that change the maths of the training step: when one of them differs from the stopped run, its checkpoints cannot be continued.
/// (Batch, accumulation, gradient checkpointing, the optimizer's 8 bits, the KL chunk, how the original is loaded and the 32-bit embedding copy (`upcast`) are not
/// in it: they only change the memory, or the numerics of frozen weights, and the checkpoints of the LoRA resume.)
///
/// A heal in rounds has no `tokens` (nor the cap) in its key, so that a higher cap (one more stretch after a failed check) continues the same run; the round size and the
/// version of the round schedule are in it instead (another round size changes where the rounds begin and end).
pub fn heal_train_key(spec: &HealSpec, c: &HealChoice, tokens: u64) -> Value {
    match &spec.rounds {
        Some(r) => json!({"seq": c.knobs.seq, "rank": c.knobs.rank, "qlora": c.knobs.qlora, "k": c.k, "alpha": spec.alpha, "lr": spec.lr,
                          "per_update": c.knobs.batch * c.knobs.grad_accum, "warmup": c.warmup, "rounds": {"round": r.round_tokens(), "schedule_v": 1}}),
        None => json!({"seq": c.knobs.seq, "rank": c.knobs.rank, "qlora": c.knobs.qlora, "k": c.k, "alpha": spec.alpha, "lr": spec.lr, "tokens": tokens,
                       "per_update": c.knobs.batch * c.knobs.grad_accum, "warmup": c.warmup}),
    }
}

/// The settings the worker reports ("heal_settings" / "heal_retry" events) as knobs again.
fn knobs_from_worker(s: &Value) -> Option<plan::HealKnobs> {
    let n = |k: &str| s[k].as_u64().map(|v| v as u32);
    Some(plan::HealKnobs { seq: n("max_seq")?, batch: n("batch")?, grad_accum: n("grad_accum")?, checkpointing: s["checkpointing"].as_bool()?, rank: n("rank")?, kl_chunk: n("chunk")?, qlora: s["method"] == "qlora", upcast: s["upcast"].as_bool().unwrap_or(false),
                            head_ckpt: s["head_checkpoint"].as_bool().unwrap_or(true) })
}

/// After an out-of-memory failure: the next safer setting from the settings of the last try, as what the page sets in the Heal card. `Null` when there is nothing
/// to offer (the ladder is used up: the person must shorten the sequence length, use QLoRA, lower the rank or choose a smaller model).
fn heal_safer(h: &Value) -> Value {
    let last = knobs_from_worker(&h["knobs_now"]).or_else(|| serde_json::from_value::<plan::HealKnobs>(h["knobs"].clone()).ok());
    match last.and_then(|k| plan::heal_next_safer(&k)) {
        Some((k, label)) => json!({"label": label, "kbit_upcast": k.upcast, "head_checkpoint": if k.head_ckpt { "on" } else { "off" }, "checkpointing": if k.checkpointing { "on" } else { "off" }, "batch": k.batch, "per_update": k.batch * k.grad_accum, "kl_chunk": k.kl_chunk}),
        None => Value::Null,
    }
}

fn set_heal(dir: &Path, patch: Value) {
    let mut v = read_json(&dir.join("heal.json")).unwrap_or_else(|| json!({}));
    if let (Some(o), Some(pa)) = (v.as_object_mut(), patch.as_object()) {
        for (k, x) in pa {
            o.insert(k.clone(), x.clone());
        }
        o.insert("updated".into(), json!(now()));
    }
    let _ = write_json(&dir.join("heal.json"), &v);
}

/// Keeps what the rows of the heal cover (`coverage` of the worker's `distill_cache` result: row counts, complete or cut conversations, reasoning rows, token shares, lengths)
/// in heal.json for the Heal card. A worker that does not report it (an older one) leaves heal.json as it was.
fn set_coverage(dir: &Path, result: &Value) {
    if result["coverage"].is_object() {
        set_heal(dir, json!({"coverage": result["coverage"].clone()}));
    }
}

/// One healing job: runs it, forwards its events tagged {"phase": "heal", "heal_step": step}, keeps the chart numbers (`heal_metrics.jsonl`).
fn heal_job(py: &Path, dir: &Path, job: &Value, step: &str, stop: &Arc<AtomicBool>, grace: Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    set_heal(dir, json!({"step": step}));
    crate::train::runs::drive(py, dir, job, stop.clone(), grace, &mut |e| {
        let mut ev = event_json(e);
        ev["phase"] = json!("heal");
        ev["heal_step"] = json!(step);
        match ev["kind"].as_str() {
            // the settings the worker really uses, and every automatic retry after an out-of-memory error (the card shows them; "Try again with safer settings" starts from them)
            Some("heal_settings") => set_heal(dir, json!({"knobs_now": ev["settings"]})),
            Some("heal_retry") => {
                let mut list = read_json(&dir.join("heal.json")).and_then(|h| h["retries"].as_array().cloned()).unwrap_or_default();
                list.push(json!({"attempt": ev["attempt"], "change": ev["change"]}));
                set_heal(dir, json!({"retries": list, "knobs_now": ev["settings"]}));
            }
            _ => {}
        }
        if matches!(ev["kind"].as_str(), Some("distill_metric" | "distill_eval")) {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("heal_metrics.jsonl")) {
                let _ = writeln!(f, "{ev}");
            }
        }
        on_event(ev);
    })
}

/// The vLLM teacher of a heal (docs/VLLM.md, "Teacher cache"): None for the heal as it always was; else `(the worker's engine object, the identity {engine, teacher_quant, route})`.
/// Refuses, in plain words, when vLLM is asked for but not switched on, when the heal does not save the original's answers, when the route would not teach the person's original
/// (a pre-quantized copy of another model) or when the saved check of the vLLM teacher says it is a different teacher. An unchecked teacher only warns (the worker logs it).
pub fn vllm_teacher_gate(dir: &Path, spec: &HealSpec, origin: &str, mode: &str) -> Result<Option<(Value, Value)>> {
    if !spec.vllm_teacher() {
        return Ok(None);
    }
    let s = crate::settings::Settings::load().vllm;
    if !s.enabled || !s.teacher.enabled {
        return Err(Error::new("vLLM for the original's answers is not switched on: open Settings, vLLM engine, and tick \"use vLLM for the original's answers\" (experimental)."));
    }
    if mode != "cache" {
        return Err(Error::new("vLLM makes the original's SAVED answers: this card heals with the original next to the student (no saved answers), so vLLM cannot be used here."));
    }
    let route = crate::vllm::teacher_route(&s.teacher, origin);
    if route["route"] == "refused" {
        return Err(Error::new(route["reason"].as_str().unwrap_or("The vLLM teacher route is refused.")));
    }
    let check = crate::vllm::read_teacher_check(dir, &s, origin);
    if check["verdict"] == "different" && check["current"] == true {
        return Err(Error::new("The check of the vLLM teacher says it gives different answers than the standard teacher, so it is not used for healing. See \"Check the vLLM teacher\" on the Heal card."));
    }
    Ok(Some((crate::vllm::teacher_engine_job(&s, Some(&check).filter(|c| c["current"] == true)), crate::vllm::teacher_identity(&s, origin))))
}

/// "Check the vLLM teacher" (docs/VLLM.md): the worker job `vllm_teacher_check` compares the vLLM teacher's top answers with the standard 4-bit teacher's on 100 real rows of the heal's data at
/// the heal's sequence length (an existing complete standard cache of this original is the reference when there is one; else the standard teacher runs first, never both on the card at once)
/// and writes `models/<name>/vllm_teacher_check.json`. Returns the worker's result (verdict, metrics, start-up phases, speed).
pub fn vllm_teacher_check(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>, stop: Arc<AtomicBool>, grace: Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    valid_name(name)?;
    spec.validate()?;
    let dir = models_dir(p).join(name);
    let s = crate::settings::Settings::load().vllm;
    if !s.enabled || !s.teacher.enabled {
        return Err(Error::new("vLLM for the original's answers is not switched on: open Settings, vLLM engine, and tick \"use vLLM for the original's answers\" (experimental)."));
    }
    let o = heal_source::read_origin(&dir)?;
    let model = heal_source::student_folder(&dir)?.ok_or_else(|| Error::new("The model's folder is missing."))?;
    let route = crate::vllm::teacher_route(&s.teacher, &o.teacher);
    if route["route"] == "refused" {
        return Err(Error::new(route["reason"].as_str().unwrap_or("The vLLM teacher route is refused.")));
    }
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: the check uses your world's text."))?;
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    let dims = plan::HealDims::from_config(&o.config, o.student_b);
    let ch = heal_choice(spec, o.student_b, vram_gb, &dims);
    let source = spec.general_source()?;
    let general = source.as_ref().map(|g| crate::general::job_block(g, spec.general_share)).unwrap_or(Value::Null);
    let common = json!({"teacher": {"repo": o.teacher, "load": ch.teacher_load.clone().unwrap_or_else(|| "4bit".into())}, "student": model, "model": {"family": o.family},
                        "data": {"train": train, "tokens": spec.tokens(), "general": general}, "k": ch.k, "max_seq": ch.knobs.seq, "seed": crate::general::SEED});
    let cache = dir.join("distill").join("cache");
    let reference = read_json(&cache.join("index.json")).filter(|i| i["complete"] == true && i["engine"].is_null()).map(|_| cache);
    let job = crate::vllm::teacher_check_job(&s, &common, &dir, reference.as_deref(), spec.vllm.as_ref().and_then(|v| v.kv_cache_dtype.as_deref()));
    let py = python()?;
    let _active = Active::begin(&dir)?;
    let _ = std::fs::remove_file(dir.join("STOP"));
    let r = crate::train::runs::drive(&py, &dir, &job, stop, grace, &mut |e| {
        let mut ev = event_json(e);
        ev["phase"] = json!("vllm_teacher_check");
        on_event(ev);
    });
    // never leave the engine running: the next job (training, a check) needs the card
    let _ = crate::vllm::stop_leftovers(&s, &crate::vllm::real_runner);
    r
}

/// Keeps what the worker says about the vLLM teacher (`teacher_engine` of the cache result: engine, quantization, start-up, speed, fallback) in heal.json for the card and the report.
fn record_teacher_engine(dir: &Path, r: &Value) {
    let t = r["teacher_engine"].clone();
    if t.is_object() {
        let mut rec = t;
        rec["requested"] = json!(true);
        set_heal(dir, json!({"teacher_engine": rec}));
    }
}

/// How many general rows are kept back from the heal and only measured (the held-out loss of the Heal card).
pub const HOLDOUT_ROWS: u32 = 64;

/// What the speed test of the loss settings (`heal_bench`) needs of a made model: the worker's python, the student's folder and family, the plan and settings the Heal card would use,
/// the training file and the cache folder (used for the teacher's answers only when it is complete for this sequence length).
pub(super) struct BenchBase {
    pub py: PathBuf,
    pub model: PathBuf,
    pub family: String,
    pub method: String,
    pub ch: HealChoice,
    pub train: PathBuf,
    pub cache: Option<PathBuf>,
    pub student_b: f64,
    pub dims: plan::HealDims,
    pub lr: f64,
}

/// A bench base with its worker job (see `BenchBase::job_base`).
pub(super) struct BenchBaseJob {
    pub py: PathBuf,
    pub job: Value,
}

impl BenchBase {
    /// The `distill` job every setting of the speed test starts from: the student, the training file, the settings the Heal card would use, and (when its answers are complete)
    /// the teacher's cache. Each setting changes the KL chunk and the output-layer checkpointing and adds the `bench` key.
    pub(super) fn job_base(&self, steps: u32, warmup: u32) -> BenchBaseJob {
        let k = &self.ch.knobs;
        let mut job = json!({
            "job": "distill", "model": {"repo": self.model, "family": self.family}, "method": self.method, "backend": "hf",
            "dataset": {"train": self.train, "general": Value::Null},
            "hyper": {"rank": k.rank, "alpha": k.rank, "lr": self.lr, "max_seq": k.seq, "batch": k.batch, "grad_accum": k.grad_accum, "warmup": self.ch.warmup, "weight_decay": 0.0,
                      "scheduler": "cosine", "seed": 3407, "save_steps": 1_000_000, "logging_steps": 1, "tokens": 1_000_000, "checkpointing": if k.checkpointing { "on" } else { "off" },
                      "optimizer": self.ch.optimizer, "student_b": self.student_b},
            "distill": {"mode": "cache", "k": self.ch.k, "chunk": k.kl_chunk, "alpha": 1.0, "kbit_upcast": k.upcast},
            "bench": {"steps": steps, "warmup": warmup, "targets": "cache"},
        });
        if let Some(c) = &self.cache {
            job["distill"]["cache"] = json!(c);
        }
        BenchBaseJob { py: self.py.clone(), job }
    }
}

pub(super) fn bench_base(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>) -> Result<BenchBase> {
    valid_name(name)?;
    spec.validate()?;
    let dir = models_dir(p).join(name);
    let model = heal_source::student_folder(&dir)?.ok_or_else(|| Error::new("The speed test needs the model on this computer: heal it once, or choose a model that is a folder."))?;
    let o = heal_source::read_origin(&dir)?;
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: the speed is measured on your world's text."))?;
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    if !train.is_file() {
        return Err(Error::new("The saved training data of this project is missing. Save a version in Step 2 again."));
    }
    let dims = plan::HealDims::from_config(&o.config, o.student_b);
    let ch = heal_choice(spec, o.student_b, vram_gb, &dims);
    let hp = plan::heal_plan_with(o.student_b, o.teacher_b, vram_gb, &dims, &ch.knobs);
    let cache = dir.join("distill").join("cache");
    let complete = read_json(&cache.join("index.json")).is_some_and(|i| i["complete"] == true && i["k"] == ch.k && i["max_len"] == ch.knobs.seq);
    Ok(BenchBase { py: python()?, model, family: o.family.clone(), method: hp.student_method.to_string(), ch, train, cache: complete.then_some(cache), student_b: o.student_b, dims, lr: spec.lr })
}

/// The teacher's answers are all there AND were made from this general text (an older cache has no key and counts as made from other data), with this many
/// answers per token (`k`) and for this sequence length (`seq`). Student-side settings (batch, accumulation, checkpointing, rank, optimizer, KL chunk, learning
/// rate) are not part of it: the answers stay when only those change.
///
/// `rows_key` (see `rows_key()`) says which rows the answers are for: the training file, the amount of text, the teacher and how it was loaded. A cache made before the
/// key was kept has none and is judged as before; one that has it must match.
fn cache_complete(cache: &Path, want: &Value, k: u32, seq: u32, rows_key: &str) -> bool {
    read_json(&cache.join("index.json")).is_some_and(|i| {
        i["complete"] == true && i["general_key"] == *want && i["k"] == k && i["max_len"] == seq && i.get("rows_key").and_then(|v| v.as_str()).is_none_or(|have| have == rows_key)
    })
}

/// A fingerprint of what decides the teacher's rows besides the general text: the training file (path, size, modification time), the amount of text in tokens, the
/// teacher and how it is loaded. The worker stores it as `rows_key` in the cache's index.json; the program compares it before skipping the teacher pass.
pub fn rows_key(train: &Path, tokens: u64, teacher_repo: &str, teacher_load: &str) -> String {
    rows_key_with(train, tokens, teacher_repo, teacher_load, None)
}

/// `rows_key` for answers made by another engine (`engine`: the vLLM identity `{"engine", "teacher_quant"}`): a cache of the standard teacher and one of the vLLM teacher are never
/// taken for each other. With None the key is exactly what it always was.
pub fn rows_key_with(train: &Path, tokens: u64, teacher_repo: &str, teacher_load: &str, engine: Option<&Value>) -> String {
    use sha2::{Digest, Sha256};
    let meta = std::fs::metadata(train).ok();
    let mtime = meta.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
    let mut text = json!({"train": train.to_string_lossy(), "size": meta.map(|m| m.len()).unwrap_or(0), "mtime": mtime, "tokens": tokens, "teacher": teacher_repo, "load": teacher_load});
    if let Some(e) = engine {
        text["engine"] = json!({"engine": e["engine"], "teacher_quant": e["teacher_quant"]});
        if !e["kv_cache"].is_null() {
            text["engine"]["kv_cache"] = e["kv_cache"].clone();          // an 8-bit KV cache is another cache (only written when on: the key is otherwise what it was)
        }
    }
    // a teacher that is a local folder also counts by its weight files (a replaced checkpoint must not reuse the old answers); a Hub id keeps the key it always had
    if let Some(files) = local_teacher_files(Path::new(teacher_repo)) {
        text["files"] = Value::String(files);
    }
    Sha256::digest(text.to_string().as_bytes()).iter().take(16).map(|b| format!("{b:02x}")).collect()
}

/// A fingerprint of the weight files and config of a local teacher folder (sorted name:size:mtime of `*.safetensors` and `config.json`); None for anything that is not a folder
/// with such files (a Hub id).
fn local_teacher_files(dir: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut names: Vec<String> = std::fs::read_dir(dir).ok()?.flatten().filter_map(|e| e.file_name().into_string().ok()).filter(|n| n == "config.json" || n.ends_with(".safetensors")).collect();
    if names.is_empty() {
        return None;
    }
    names.sort();
    let mut text = String::new();
    for n in names {
        let m = std::fs::metadata(dir.join(&n)).ok();
        let mtime = m.as_ref().and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
        text.push_str(&format!("{n}:{}:{mtime};", m.map(|m| m.len()).unwrap_or(0)));
    }
    Some(Sha256::digest(text.as_bytes()).iter().take(16).map(|b| format!("{b:02x}")).collect())
}

fn training_done(run: &Path) -> bool {
    run.join("adapter").join("adapter_config.json").is_file() && read_json(&run.join("summary.json")).is_some_and(|s| s["stopped"] == false)
}

/// Heals a made model: the original (teacher) teaches the thinner one (student) to answer as it did. Three jobs one after the other, each resumable:
/// 1. `distill_cache` (when the teacher and the student do not fit the card together): the teacher's top answers for every row, saved under
///    `distill/cache/`; 2. `distill`: the student learns (a LoRA, checkpoints in `distill/run/`); 3. `distill_merge`: the LoRA is added into a new
///    base model, `healed/`, which is added to the model list. The checkpoints are deleted afterwards; the teacher's answers stay until the model is
///    deleted. Calling it again after a stop or a crash goes on where it was. Returns the result ("name", "model_id", the training numbers) or
///    `{"stopped": true, "name"}`. Every event goes to `on_event` tagged with "phase": "heal".
pub fn heal(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>, stop: Arc<AtomicBool>, grace: Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    valid_name(name)?;
    spec.validate()?;
    let dir = models_dir(p).join(name);
    let mut model = heal_source::student_folder(&dir)?;     // None: a Hugging Face model that is downloaded first (below)
    if dir.join("healed").join("config.json").is_file() && !rounds_can_extend(&dir, spec) {
        return Err(Error::new("This model has already been healed."));
    }
    let o = heal_source::read_origin(&dir)?;
    let (origin, family) = (o.teacher.clone(), o.family.clone());
    if spec.thinking() && family != "qwen3_5" {
        return Err(Error::new("The thinking recipe is for Qwen3.5 and Qwen3.8 models (their chat template writes the reasoning). Untick \"Keep thinking\" for this model."));
    }
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: healing uses your world's text."))?;
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    if !train.is_file() {
        return Err(Error::new("The saved training data of this project is missing. Save a version in Step 2 again."));
    }
    let (student_b, teacher_b) = (o.student_b, o.teacher_b);
    let dims = plan::HealDims::from_config(&o.config, student_b);
    let ch = heal_choice(spec, student_b, vram_gb, &dims);
    let hp = plan::heal_plan_with(student_b, teacher_b, vram_gb, &dims, &ch.knobs);
    if !hp.fits {
        return Err(Error::new(hp.why.clone().unwrap_or_else(|| "Healing does not fit this card.".into())));
    }
    let rspec = spec.rounds.as_ref();
    if let Some(r) = rspec.filter(|r| r.round_exact.is_none()) {
        heal_rounds::check_round_size(ch.knobs.seq, ch.knobs.batch * ch.knobs.grad_accum, r.round_tokens())?;
    }
    let tokens = rspec.map(|r| r.cap_tokens(spec)).unwrap_or_else(|| spec.tokens());       // a heal in rounds reads at most the cap
    let train_key = heal_train_key(spec, &ch, tokens);
    // healing in rounds always works from the teacher's saved answers: the answers of a round are asked for one chunk at a time
    let mode = if rspec.is_some() { "cache".to_string() } else { spec.mode.clone().unwrap_or_else(|| hp.mode.to_string()) };
    let source = spec.general_source()?;
    let general = source.as_ref().map(|s| crate::general::job_block(s, spec.general_share)).unwrap_or(Value::Null);
    let general_key = crate::general::key(&general, crate::general::SEED);
    let (cache_dir, run_dir) = (dir.join("distill").join("cache"), dir.join("distill").join("run"));
    let teacher_load = ch.teacher_load.clone().unwrap_or_else(|| hp.teacher_load.to_string());
    let vllm_engine = vllm_teacher_gate(&dir, spec, &origin, &mode)?;
    let rows_key = rows_key_with(&train, tokens, &origin, &teacher_load, vllm_engine.as_ref().map(|e| e.1.clone()).as_ref());
    if mode == "cache" && !cache_complete(&cache_dir, &general_key, ch.k, ch.knobs.seq, &rows_key) {
        let need = tokens as f64 * 196.0 / 1e9 + if rspec.is_some() { 4.0 } else { 0.0 };       // rounds keep two adapters and two checkpoints besides
        if let Some(free) = crate::envmgr::health::free_disk_gb(&dir).filter(|f| *f < need * 1.05) {
            return Err(Error::new(format!("The teacher's answers need {need:.1} GB in {}; {free:.1} GB free.", cache_dir.display())));
        }
    }
    let py = python()?;
    let _active = Active::begin(&dir)?;
    // a stopped heal that goes on with another general text cannot continue from the old one's checkpoints
    // (and likewise one that goes on with other settings that change the maths: sequence length, rank, top-k, learning rate, sequences per update ...; a run
    // from before the key was kept cannot be checked, so it starts again too). The teacher's answers are judged by `cache_complete` and are not deleted here.
    let old = read_json(&dir.join("heal.json"));
    if old.as_ref().map(|h| h["general_key"].clone()).as_ref() != Some(&general_key) || old.as_ref().map(|h| h["train_key"].clone()).as_ref() != Some(&train_key) {
        if run_dir.exists() {
            let _ = std::fs::remove_dir_all(&run_dir);
        }
        if old.as_ref().is_some_and(|h| !h["rounds"].is_null()) {
            set_heal(&dir, json!({"rounds": Value::Null}));          // the rounds of another run are not this run's
        }
    }
    let cal = &o.calibration;
    let mut calibration = json!({"train": train, "samples": cal["samples"].as_u64().unwrap_or(128), "max_len": cal["max_len"].as_u64().unwrap_or(512), "batch": 4, "seed": 3407, "version": version});
    if cal["reasoning"].is_object() {
        calibration["reasoning"] = cal["reasoning"].clone();         // the loss before and after is measured on the text the cut was measured on
    }
    let stop_file = dir.join("STOP");
    let started_at = read_json(&dir.join("heal.json")).and_then(|h| h["created"].as_str().map(String::from)).unwrap_or_else(now);
    set_heal(&dir, json!({"status": "running", "interrupted": false, "step": "start", "spec": spec, "general": general, "general_key": general_key, "created": started_at, "error": null, "oom": false, "mode": mode, "plan": hp, "knobs": ch.knobs, "knobs_now": Value::Null, "train_key": train_key, "retries": [], "recipe": spec.recipe}));
    let stopped = |step: &str| {
        set_heal(&dir, json!({"status": "stopped", "step": step}));
        json!({"stopped": true, "name": name})
    };
    let failed = |e: Error| {
        if e.is_cancelled() {
            set_heal(&dir, json!({"status": "stopped"}));
            return Error::new("Stopped. Healing can be continued where it stopped.");
        }
        set_heal(&dir, json!({"status": "failed", "error": e.0, "oom": e.0.to_lowercase().contains("out of memory")}));
        e
    };
    // a Hugging Face student is downloaded into the Hugging Face cache first (the weights are read there, never copied)
    let model = match model.take() {
        Some(m) => m,
        None => {
            set_heal(&dir, json!({"step": "download"}));
            heal_source::fetch_student(&dir, &stop, on_event).map_err(failed)?
        }
    };
    if let Some(rs) = rspec {
        let ctx = RoundsCtx {
            p, name, dir: dir.clone(), py: py.clone(), model: model.clone(), origin: origin.clone(), family: family.clone(), spec, ch: ch.clone(), method: hp.student_method.to_string(),
            teacher_load: ch.teacher_load.clone().unwrap_or_else(|| hp.teacher_load.to_string()), train: train.clone(), tokens, general: general.clone(), rows_key: rows_key.clone(),
            vllm: vllm_engine.as_ref().map(|e| e.0.clone()),
            cache_dir: cache_dir.clone(), run_dir: run_dir.clone(), stop: stop.clone(), grace, stop_file: stop_file.clone(), calibration: calibration.clone(),
            student_b, o: &o,
        };
        let share = if source.is_some() { spec.general_share } else { 0.0 };
        let plan = heal_rounds::Plan { spec: rs.clone(), cap: tokens, round: rs.round_tokens(), first_check: rs.first_check_tokens(spec), share, thinking: spec.thinking(), tail_secs: heal_rounds::TAIL_SECS_DEFAULT };
        return match heal_in_rounds(&ctx, &plan, on_event) {
            Ok(Some(r)) => Ok(r),
            Ok(None) => Ok(json!({"stopped": true, "name": name})),
            Err(e) => Err(failed(e)),
        };
    }
    if mode == "cache" && !cache_complete(&cache_dir, &general_key, ch.k, ch.knobs.seq, &rows_key) {
        let mut job = json!({
            "job": "distill_cache", "teacher": {"repo": origin, "load": ch.teacher_load.clone().unwrap_or_else(|| hp.teacher_load.to_string())}, "student": model, "model": {"family": family},
            "data": {"train": train, "tokens": tokens, "general": general}, "k": ch.k, "max_seq": ch.knobs.seq, "batch": 4, "chunk": 256, "seed": crate::general::SEED,
            "out": cache_dir, "stop_file": stop_file, "rows_key": rows_key, "shared_cache": true, "shared_dir": teacher_cache_dir(p),
        });
        if !general.is_null() {
            job["holdout"] = json!({"rows": HOLDOUT_ROWS});          // general rows read past the ones the heal learns from: the loss on them is not optimistic
        }
        if let Some((e, ident)) = &vllm_engine {
            job["engine"] = e.clone();
            set_heal(&dir, json!({"teacher_engine": {"requested": true, "identity": ident}}));
        }
        let r = heal_job(&py, &dir, &job, "cache", &stop, grace, on_event).map_err(failed)?;
        if r["stopped"].as_bool().unwrap_or(false) {
            return Ok(stopped("cache"));
        }
        if vllm_engine.is_some() {
            record_teacher_engine(&dir, &r);
        }
        let note = r["notes"].as_array().and_then(|n| n.first()).and_then(|n| n.as_str());
        // the page shows which general text was used or skipped, and why
        set_heal(&dir, json!({"note": note, "note_code": r["stats"]["general_error_code"], "general_used": {"source": r["stats"]["general_source"], "rows": r["stats"]["general_rows"],
                              "tokens": r["stats"]["general_tokens"], "detected": r["stats"]["general_detected"], "skipped": note.is_some()}}));
        set_coverage(&dir, &r);
    }
    let mut summary = read_json(&run_dir.join("summary.json")).filter(|_| training_done(&run_dir)).unwrap_or(Value::Null);
    if summary.is_null() {
        let resume = std::fs::read_dir(run_dir.join("checkpoints")).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().starts_with("checkpoint-"))).unwrap_or(false);
        let mut job = json!({
            "job": "distill", "model": {"repo": model, "family": family}, "method": hp.student_method, "backend": spec.backend.clone().unwrap_or_else(|| "hf".into()),
            "dataset": {"train": train, "general": general},
            "hyper": {"rank": ch.knobs.rank, "alpha": ch.knobs.rank, "lr": spec.lr, "max_seq": ch.knobs.seq, "batch": ch.knobs.batch, "grad_accum": ch.knobs.grad_accum, "warmup": ch.warmup,
                      "weight_decay": 0.0, "scheduler": "cosine", "seed": 3407, "save_steps": 50, "logging_steps": 1, "tokens": tokens,
                      "checkpointing": if ch.knobs.checkpointing { "on" } else { "off" }, "optimizer": ch.optimizer, "student_b": student_b},
            "distill": {"mode": mode, "cache": cache_dir, "teacher": {"repo": origin, "load": ch.teacher_load.clone().unwrap_or_else(|| hp.teacher_load.to_string())}, "alpha": spec.alpha, "k": ch.k,
                        "chunk": ch.knobs.kl_chunk, "eval_every": 50, "kbit_upcast": ch.knobs.upcast},
            "eval": {"calibration": calibration, "teacher_loss": o.loss_before, "slim_loss": o.loss_after, "holdout": {"rows": HOLDOUT_ROWS}},
            "out": run_dir, "resume": resume, "stop_file": stop_file,
        });
        if !ch.knobs.head_ckpt {
            job["distill"]["head_checkpoint"] = json!("off");          // only an explicit "off" is sent: a default job is the job it always was
        }
        std::fs::create_dir_all(&run_dir)?;
        write_json(&dir.join("heal_config.json"), &job)?;
        let r = heal_job(&py, &dir, &job, "train", &stop, grace, on_event).map_err(failed)?;
        if r["stopped"].as_bool().unwrap_or(false) {
            return Ok(stopped("train"));
        }
        if let Some(e) = r["general"]["general_error"].as_str() {
            set_heal(&dir, json!({"note": e, "note_code": r["general"]["general_error_code"]}));
        }
        summary = r;
    }
    let mut job = json!({"job": "distill_merge", "base": model, "adapter": run_dir.join("adapter"), "out": dir.join("healed.partial"), "final": dir.join("healed"), "summary": summary, "stop_file": stop_file});
    if let Some(r) = &spec.recipe {
        job["recipe"] = json!(r);          // the merged model's ndforge_reshape.json says how it was healed (healed.recipe): the export keeps its thinking template for "thinking"
    }
    if o.external || o.teacher_chosen {
        // a model the Slim page did not make has no recorded original: the merged folder records the teacher (and the student's name) for Step 3 and Step 5
        job["origin"] = json!({"repo": origin, "family": family, "params": (teacher_b * 1e9).round()});
        job["student_label"] = json!(o.student_label);
    }
    let outcome = heal_job(&py, &dir, &job, "merge", &stop, grace, on_event);
    let _ = std::fs::remove_dir_all(dir.join("healed.partial"));
    let r = outcome.map_err(failed)?;
    if r["stopped"].as_bool().unwrap_or(false) {
        return Ok(stopped("merge"));
    }
    if !dir.join("healed").join("config.json").is_file() {
        return Err(failed(Error::new("The worker finished but the healed model is missing.")));
    }
    let _ = std::fs::remove_dir_all(run_dir.join("checkpoints"));
    let id = register_healed(p, name)?;
    let mut result = summary.clone();
    result["name"] = json!(name);
    result["model_id"] = json!(id);
    result["healed_dir"] = json!(dir.join("healed"));
    write_json(&dir.join("heal_result.json"), &result)?;
    set_heal(&dir, json!({"status": "done", "step": "done", "result": result.clone(), "error": null}));
    Ok(result)
}

// ----------------------------------------------------------------------------------------------------------------- healing in rounds (docs/HEAL_ROUNDS.md)

/// A heal that ended without a passed check of the merged model may be taken further: `rounds.extend` with a higher cap opens it again.
fn rounds_can_extend(dir: &Path, spec: &HealSpec) -> bool {
    let Some(r) = spec.rounds.as_ref().filter(|r| r.extend) else { return false };
    let Some(st) = read_json(&dir.join("heal.json")).map(|h| h["rounds"].clone()) else { return false };
    let tokens = st["rows"].as_array().and_then(|a| a.last()).and_then(|x| x["tokens"].as_u64()).unwrap_or(0);
    st["state"] == "ended" && !matches!(st["final"]["verdict"].as_str(), Some("pass" | "warn")) && r.cap_tokens(spec) > tokens
}

/// Everything the jobs of a heal in rounds need that `heal()` has worked out.
struct RoundsCtx<'a> {
    p: &'a Project,
    name: &'a str,
    dir: PathBuf,
    py: PathBuf,
    model: PathBuf,
    origin: String,
    family: String,
    spec: &'a HealSpec,
    ch: HealChoice,
    method: String,
    teacher_load: String,
    train: PathBuf,
    /// the cap in tokens: the rows of the whole run
    tokens: u64,
    general: Value,
    rows_key: String,
    /// the vLLM teacher's engine object (`vllm_teacher_gate`); None = the standard teacher
    vllm: Option<Value>,
    cache_dir: PathBuf,
    run_dir: PathBuf,
    stop: Arc<AtomicBool>,
    grace: Duration,
    stop_file: PathBuf,
    calibration: Value,
    student_b: f64,
    o: &'a heal_source::Origin,
}

struct RealRoundJobs<'a, 'b> {
    c: &'a RoundsCtx<'a>,
    events: &'a std::cell::RefCell<&'b mut dyn FnMut(Value)>,
    round: u32,
}

impl RealRoundJobs<'_, '_> {
    /// The settings the worker really used last (an out-of-memory retry may have made them safer): the next round starts there, so the ladder is not climbed again.
    fn knobs(&self) -> plan::HealKnobs {
        let per_update = self.c.ch.knobs.batch * self.c.ch.knobs.grad_accum;
        read_json(&self.c.dir.join("heal.json")).and_then(|h| knobs_from_worker(&h["knobs_now"])).filter(|k| k.batch * k.grad_accum == per_update && k.seq == self.c.ch.knobs.seq).unwrap_or_else(|| self.c.ch.knobs.clone())
    }

    fn job(&mut self, job: &Value, step: &str) -> Result<Value> {
        let events = self.events;
        let round = self.round;
        let mut f = |mut ev: Value| {
            ev["round"] = json!(round);
            (events.borrow_mut())(ev);
        };
        heal_job(&self.c.py, &self.c.dir, job, step, &self.c.stop, self.c.grace, &mut f)
    }

    fn think(&mut self, target: &str, quick: bool, round: Option<u32>) -> Result<Value> {
        let spec = super::thinkcheck::ThinkSpec { name: self.c.name.to_string(), target: target.into(), quick, round, ..Default::default() };
        let events = self.events;
        let n = self.round;
        let mut f = |mut ev: Value| {
            ev["phase"] = json!("heal");
            ev["heal_step"] = json!("check");
            ev["round"] = json!(n);
            (events.borrow_mut())(ev);
        };
        set_heal(&self.c.dir, json!({"step": "check"}));
        super::thinkcheck::run(self.c.p, &spec, self.c.stop.clone(), &mut f)
    }

    fn last_round(&self) -> u32 {
        read_json(&self.c.dir.join("heal.json")).and_then(|h| h["rounds"]["rows"].as_array().and_then(|r| r.last().and_then(|x| x["n"].as_u64()))).unwrap_or(1) as u32
    }
}

impl heal_rounds::RoundJobs for RealRoundJobs<'_, '_> {
    fn load(&mut self) -> Option<Value> {
        read_json(&self.c.dir.join("heal.json")).map(|h| h["rounds"].clone()).filter(|r| r.is_object())
    }

    fn save(&mut self, state: &Value) {
        set_heal(&self.c.dir, json!({"rounds": state}));
    }

    fn cache(&mut self, n: u32, upto_general_tokens: u64) -> Result<Value> {
        self.round = n;
        let c = self.c;
        let mut job = json!({
            "job": "distill_cache", "teacher": {"repo": c.origin, "load": c.teacher_load}, "student": c.model, "model": {"family": c.family},
            "data": {"train": c.train, "tokens": c.tokens, "general": c.general}, "k": c.ch.k, "max_seq": c.ch.knobs.seq, "batch": 4, "chunk": 256, "seed": crate::general::SEED,
            "out": c.cache_dir, "stop_file": c.stop_file, "rows_key": c.rows_key, "shared_cache": true, "shared_dir": teacher_cache_dir(c.p),
            "upto_general_tokens": upto_general_tokens, "reuse_rows": true,
        });
        if !c.general.is_null() {
            job["holdout"] = json!({"rows": HOLDOUT_ROWS});
        }
        if let Some(e) = &c.vllm {
            job["engine"] = e.clone();          // every round's chunk starts the engine again (its start-up is paid each time: the card warns)
        }
        let r = self.job(&job, "cache")?;
        if c.vllm.is_some() {
            record_teacher_engine(&c.dir, &r);
        }
        if !r["stopped"].as_bool().unwrap_or(false) && n == 1 {
            let note = r["notes"].as_array().and_then(|a| a.first()).and_then(|x| x.as_str());
            set_heal(&c.dir, json!({"note": note, "note_code": r["stats"]["general_error_code"], "general_used": {"source": r["stats"]["general_source"], "rows": r["stats"]["general_rows"],
                                    "tokens": r["stats"]["general_tokens"], "detected": r["stats"]["general_detected"], "skipped": note.is_some()}}));
            set_coverage(&c.dir, &r);
        }
        Ok(r)
    }

    fn train(&mut self, t: &heal_rounds::TrainRound) -> Result<Value> {
        self.round = t.n;
        let c = self.c;
        let k = self.knobs();
        let resume = std::fs::read_dir(c.run_dir.join("checkpoints")).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().starts_with("checkpoint-"))).unwrap_or(false);
        let mut job = json!({
            "job": "distill", "model": {"repo": c.model, "family": c.family}, "method": c.method, "backend": c.spec.backend.clone().unwrap_or_else(|| "hf".into()),
            "dataset": {"train": c.train, "general": c.general},
            "hyper": {"rank": k.rank, "alpha": k.rank, "lr": c.spec.lr, "max_seq": k.seq, "batch": k.batch, "grad_accum": k.grad_accum, "warmup": c.ch.warmup,
                      "weight_decay": 0.0, "scheduler": "cosine", "seed": 3407, "save_steps": 50, "logging_steps": 1, "tokens": c.tokens,
                      "checkpointing": if k.checkpointing { "on" } else { "off" }, "optimizer": c.ch.optimizer, "student_b": c.student_b},
            "distill": {"mode": "cache", "cache": c.cache_dir, "teacher": {"repo": c.origin, "load": c.teacher_load}, "alpha": c.spec.alpha, "k": c.ch.k,
                        "chunk": k.kl_chunk, "eval_every": 50, "kbit_upcast": k.upcast},
            "eval": {"calibration": c.calibration, "teacher_loss": c.o.loss_before, "slim_loss": c.o.loss_after, "holdout": {"rows": HOLDOUT_ROWS}},
            "rounds": {"index": t.n, "round_tokens": t.round_tokens, "general_from": t.general_from, "general_to": t.general_to, "s_start": t.s_start},
            "out": c.run_dir, "resume": resume, "stop_file": c.stop_file,
        });
        if !k.head_ckpt {
            job["distill"]["head_checkpoint"] = json!("off");
        }
        std::fs::create_dir_all(&c.run_dir)?;
        write_json(&c.dir.join("heal_config.json"), &job)?;
        let r = self.job(&job, "train")?;
        if !r["stopped"].as_bool().unwrap_or(false) {
            if let Some(e) = r["general"]["general_error"].as_str() {
                set_heal(&c.dir, json!({"note": e, "note_code": r["general"]["general_error_code"]}));
            }
            if t.n > 2 {
                let _ = std::fs::remove_dir_all(c.run_dir.join("rounds").join(format!("r{}", t.n - 2)));        // two round adapters are kept
            }
        }
        Ok(r)
    }

    fn check_round(&mut self, n: u32, quick: bool) -> Result<Value> {
        self.round = n;
        self.think("round", quick, Some(n))
    }

    fn check_original(&mut self, quick: bool) -> Result<Value> {
        self.think("original", quick, None)
    }

    fn merge(&mut self) -> Result<Value> {
        let c = self.c;
        if c.dir.join("healed").join("config.json").is_file() {
            return Ok(json!({"stopped": false, "already": true}));
        }
        let n = self.last_round();
        self.round = n;
        let adapter = [c.run_dir.join("rounds").join(format!("r{n}")).join("adapter"), c.run_dir.join("adapter")].into_iter().find(|a| a.join("adapter_config.json").is_file()).unwrap_or_else(|| c.run_dir.join("adapter"));
        let summary = read_json(&c.run_dir.join("rounds").join(format!("r{n}")).join("summary.json")).or_else(|| read_json(&c.run_dir.join("summary.json"))).unwrap_or(Value::Null);
        let mut job = json!({"job": "distill_merge", "base": c.model, "adapter": adapter, "out": c.dir.join("healed.partial"), "final": c.dir.join("healed"), "summary": summary, "stop_file": c.stop_file});
        if let Some(r) = &c.spec.recipe {
            job["recipe"] = json!(r);
        }
        if c.o.external || c.o.teacher_chosen {
            job["origin"] = json!({"repo": c.origin, "family": c.family, "params": (c.o.teacher_b * 1e9).round()});
            job["student_label"] = json!(c.o.student_label);
        }
        let outcome = self.job(&job, "merge");
        let _ = std::fs::remove_dir_all(c.dir.join("healed.partial"));
        let r = outcome?;
        if !r["stopped"].as_bool().unwrap_or(false) && !c.dir.join("healed").join("config.json").is_file() {
            return Err(Error::new("The worker finished but the healed model is missing."));
        }
        Ok(r)
    }

    fn check_merged(&mut self, quick: bool) -> Result<Value> {
        self.round = self.last_round();
        self.think("healed", quick, None)
    }

    fn unmerge(&mut self) -> Result<()> {
        let c = self.c;
        let healed = c.dir.join("healed");
        let repo = healed.canonicalize().unwrap_or(healed.clone()).to_string_lossy().to_string();
        if let Some(id) = catalog::models().into_iter().find(|m| m.repo == repo).map(|m| m.id) {
            catalog::remove_custom(&id)?;
        }
        let _ = std::fs::remove_dir_all(&healed);
        // the saved checks of the merged model are of the old weights; the worker's cache key (file names and sizes) would take them for the new ones
        for mode in ["quick", "full"] {
            let _ = std::fs::remove_file(super::thinkcheck::think_dir(c.p, c.name).join(format!("healed.{mode}.json")));
        }
        set_state(&c.dir, json!({"healed_model_id": Value::Null}));
        Ok(())
    }

    fn register(&mut self) -> Result<Value> {
        let id = register_healed(self.c.p, self.c.name)?;
        Ok(json!({"model_id": id, "healed_dir": self.c.dir.join("healed")}))
    }

    fn now_secs(&self) -> u64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    }
}

/// The heal in rounds; `Ok(None)` when Stop was pressed. The result is the one of a normal heal (the numbers of the last round's training, the model's name and id) with `rounds`.
fn heal_in_rounds(c: &RoundsCtx, plan: &heal_rounds::Plan, on_event: &mut dyn FnMut(Value)) -> Result<Option<Value>> {
    let cell = std::cell::RefCell::new(on_event);
    let mut jobs = RealRoundJobs { c, events: &cell, round: 1 };
    let mut tell = |e: Value| (cell.borrow_mut())(e);
    match heal_rounds::run(plan, &mut jobs, &mut tell)? {
        heal_rounds::Outcome::Stopped(step) => {
            set_heal(&c.dir, json!({"status": "stopped", "step": step}));
            Ok(None)
        }
        heal_rounds::Outcome::Done(rounds) => {
            let passed = c.spec.thinking().then(|| matches!(rounds["final"]["verdict"].as_str(), Some("pass" | "warn"))).unwrap_or(true);
            let last = rounds["rows"].as_array().and_then(|r| r.last()).and_then(|r| r["n"].as_u64()).unwrap_or(1);
            let mut result = read_json(&c.run_dir.join("rounds").join(format!("r{last}")).join("summary.json")).or_else(|| read_json(&c.run_dir.join("summary.json"))).unwrap_or_else(|| json!({}));
            result["name"] = json!(c.name);
            result["model_id"] = jobs_registered(c);
            result["healed_dir"] = json!(c.dir.join("healed"));
            result["rounds"] = rounds;
            if passed {
                let _ = std::fs::remove_dir_all(c.run_dir.join("checkpoints"));       // kept otherwise: "Heal more" goes on from the last round
            }
            write_json(&c.dir.join("heal_result.json"), &result)?;
            set_heal(&c.dir, json!({"status": "done", "step": "done", "result": result.clone(), "error": null}));
            Ok(Some(result))
        }
    }
}

fn jobs_registered(c: &RoundsCtx) -> Value {
    read_json(&c.dir.join("state.json")).map(|s| s["healed_model_id"].clone()).unwrap_or(Value::Null)
}

/// The chart numbers of a healing run (KL, calibration loss), oldest first, for the page after a restart.
pub fn heal_metrics(p: &Project, name: &str) -> Vec<Value> {
    if valid_name(name).is_err() {
        return Vec::new();
    }
    std::fs::read_to_string(models_dir(p).join(name).join("heal_metrics.jsonl")).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
}

// ----------------------------------------------------------------------------------------------------------------- measure

/// What is kept of an earlier measurement of this model on this data with these settings (None when there is none yet).
pub fn measurement_info(p: &Project, m: &ModelEntry, samples: u32, len: u32, load: &str) -> Value {
    measurement_info_with(p, m, samples, len, load, None)
}

/// `measurement_info` for a measuring text with a reasoning part.
pub fn measurement_info_with(p: &Project, m: &ModelEntry, samples: u32, len: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Value {
    let version = crate::train::runs::latest_version(p).map(|(n, _)| n);
    let key = scores_key_with(&m.repo, version, samples, len, load, reasoning);
    let meta = read_json(&scores_dir(p, &key).join("scores.json"));
    json!({"key": key, "exists": meta.is_some(), "tokens": meta.as_ref().map(|x| x["tokens"].clone()), "load": meta.as_ref().map(|x| x["load"].clone()),
           "created": meta.as_ref().map(|x| x["created"].clone()), "data_version": version})
}

/// "Measure now" (Expert: also "measure again"): measures the model once and keeps the result for every later size choice on the same data. Making a
/// thinner model measures by itself when nothing is kept, so this is optional. Returns the worker's result.
pub fn measure(p: &Project, model_id: &str, samples: u32, len: u32, load: &str, force: bool, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    measure_with(p, model_id, samples, len, load, None, force, stop, on_event)
}

/// `measure` with a reasoning part in the measuring text (half of the samples are conversations with thinking, written the way the model thinks).
#[allow(clippy::too_many_arguments)]
pub fn measure_with(p: &Project, model_id: &str, samples: u32, len: u32, load: &str, reasoning: Option<&ReasoningCalib>, force: bool, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    if let Some(r) = reasoning {
        r.validate()?;
    }
    if !["auto", "16bit", "4bit", "offload"].contains(&load) || !(8..=4096).contains(&samples) || !(32..=8192).contains(&len) {
        return Err(Error::new("Measure on 8 to 4096 samples of 32 to 8192 tokens."));
    }
    let m = catalog::find(model_id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: the layers are measured on your own text."))?;
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    if !train.is_file() {
        return Err(Error::new("The saved training data of this project is missing. Save a version in Step 2 again."));
    }
    let py = python()?;
    let key = scores_key_with(&m.repo, Some(version), samples, len, load, reasoning);
    let folder = scores_dir(p, &key);
    let _active = Active::begin(&folder)?;
    let job = json!({
        "job": "reshape_measure", "source": {"repo": m.repo, "family": m.family}, "load": load, "out_dir": folder, "force": force,
        "calibration": calibration_json(Some(&train), samples, len, Some(version), reasoning),
    });
    let stop_dir = models_dir(p).join("_scores");
    let mut result = crate::train::runs::drive(&py, &stop_dir, &job, stop, Duration::from_secs(60), &mut |e| on_event(event_json(e)))?;
    result["key"] = json!(key);
    Ok(result)
}

// ------------------------------------------------------------------------------------------------------------ try before cutting

fn try_one() -> u32 {
    1
}
fn try_samples() -> u32 {
    plan::TRY_QUICK_SAMPLES
}
fn try_len() -> u32 {
    512
}
fn try_measure_samples() -> u32 {
    256
}
fn try_auto() -> String {
    "auto".into()
}
fn try_contiguous() -> String {
    "contiguous".into()
}

/// One cut the person wants an exact answer for: blocks (clicked or typed) and/or an MLP width.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TryChoice {
    #[serde(default)]
    pub blocks: Vec<u32>,
    #[serde(default)]
    pub width: Option<u32>,
    /// "activation" (None: the default) or "output": how the neurons of the width are ranked (needs a width). The worker then also tries the plain cut, to compare the two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring: Option<String>,
    /// Refit the kept neurons after the width cut (experimental): "pruned" (from the cut model's own activations) or "original" (from the original's); needs a width. The worker
    /// then also tries the plain cut, to compare. None = off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reconstruct: Option<String>,
}

/// What "Try first" is asked to find out: which cuts, on how much text. Nothing is written but `reshape_try.json` in the measurement's folder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrySpec {
    pub model_id: String,
    /// Quick (64) or Full (128) samples of the project's text; the answers are kept per number of samples
    #[serde(default = "try_samples")]
    pub samples: u32,
    #[serde(default = "try_len")]
    pub len: u32,
    #[serde(default = "try_auto")]
    pub load: String,
    /// the samples of the measurement whose neuron scores (and influence) are used
    #[serde(default = "try_measure_samples")]
    pub measure_samples: u32,
    #[serde(default = "try_contiguous")]
    pub strategy: String,
    #[serde(default = "try_one")]
    pub protect_first: u32,
    #[serde(default = "try_one")]
    pub protect_last: u32,
    /// every block on its own
    #[serde(default)]
    pub blocks: bool,
    /// run lengths: every run of that many neighbouring blocks
    #[serde(default)]
    pub runs: Vec<u32>,
    /// MLP widths (the stops of the size slider)
    #[serde(default)]
    pub widths: Vec<u32>,
    /// exact choices (blocks and/or width)
    #[serde(default)]
    pub choices: Vec<TryChoice>,
    /// throw the saved answers away first
    #[serde(default)]
    pub force: bool,
    /// the model's size in billions of parameters (for the first time estimate)
    #[serde(default)]
    pub params_b: Option<f64>,
    /// a measuring text with a reasoning part (the answers and the measurement of the neurons are kept per reasoning text)
    #[serde(default)]
    pub reasoning: Option<ReasoningCalib>,
    /// the automatic search of the best cut (reshape::search): `{groups, budget_min, thorough, ...}`; the worker plans the rounds itself and measures in one job
    #[serde(default)]
    pub search: Option<Value>,
}

/// The most cuts one request may ask for (a safety net; the page asks for far fewer).
pub const TRY_MAX_PASSES: usize = 400;
/// The most target sizes one search may look at (the trade-off mode uses five).
pub const TRY_SEARCH_MAX_GROUPS: usize = 8;

impl TrySpec {
    pub fn validate(&self) -> Result<()> {
        if !["auto", "16bit", "4bit", "offload"].contains(&self.load.as_str()) {
            return Err(Error::new(format!("Unknown way of loading the model: {}.", self.load)));
        }
        if !["contiguous", "separate"].contains(&self.strategy.as_str()) {
            return Err(Error::new(format!("Unknown way of choosing layers: {}.", self.strategy)));
        }
        if !(8..=4096).contains(&self.samples) || !(8..=4096).contains(&self.measure_samples) || !(32..=8192).contains(&self.len) {
            return Err(Error::new("Try on 8 to 4096 samples of 32 to 8192 tokens."));
        }
        if let Some(r) = &self.reasoning {
            r.validate()?;
        }
        if self.protect_first > 8 || self.protect_last > 8 {
            return Err(Error::new("Protect at most 8 blocks at each end."));
        }
        if self.runs.iter().any(|k| !(1..=64).contains(k)) || self.widths.iter().any(|w| *w == 0) {
            return Err(Error::new("The run lengths and widths to try are not valid."));
        }
        if self.choices.iter().any(|c| c.blocks.is_empty() && c.width.is_none()) {
            return Err(Error::new("A choice to try needs at least one block or a width."));
        }
        if self.choices.iter().any(|c| c.scoring.as_deref().is_some_and(|s| !["activation", "output"].contains(&s)) || (c.scoring.as_deref() == Some("output") && c.width.is_none())) {
            return Err(Error::new("Ranking the neurons by their output (or by their activation) needs a width to try."));
        }
        if self.choices.iter().any(|c| c.reconstruct.as_deref().is_some_and(|r| !["pruned", "original"].contains(&r) || c.width.is_none())) {
            return Err(Error::new("Refitting the kept neurons (pruned or original) needs a width to try."));
        }
        if self.choices.len() + self.widths.len() + self.runs.len() * 64 > TRY_MAX_PASSES {
            return Err(Error::new("That asks for too many cuts at once."));
        }
        if let Some(s) = &self.search {
            let groups = s["groups"].as_array().map(|g| g.len()).unwrap_or(0);
            if groups == 0 || groups > TRY_SEARCH_MAX_GROUPS {
                return Err(Error::new(format!("The search needs 1 to {TRY_SEARCH_MAX_GROUPS} target sizes.")));
            }
            let minutes = s["budget_min"].as_f64().unwrap_or(20.0);
            if !(5.0..=240.0).contains(&minutes) {
                return Err(Error::new("The search may take 5 to 240 minutes."));
            }
            if s["max_passes"].as_u64().is_some_and(|n| n as usize > TRY_MAX_PASSES) {
                return Err(Error::new("The search may measure at most 400 cuts."));
            }
        }
        Ok(())
    }

    /// The `want` of the worker job.
    pub fn want(&self) -> Value {
        let mut w = json!({"blocks": self.blocks, "runs": self.runs, "widths": self.widths, "choices": self.choices});
        if let Some(s) = &self.search {
            w["search"] = s.clone();
        }
        w
    }
}

/// The folder where the answers for this model, data and settings are kept (the measurement's own folder when the samples are the same).
pub fn try_dir(p: &Project, m: &ModelEntry, version: u32, spec: &TrySpec) -> PathBuf {
    scores_dir(p, &scores_key_with(&m.repo, Some(version), spec.samples, spec.len, &spec.load, spec.reasoning.as_ref()))
}

/// The worker's job for a spec. `plan_only`: count the passes and estimate the time, run nothing.
pub fn try_job(p: &Project, m: &ModelEntry, spec: &TrySpec, plan_only: bool) -> Result<Value> {
    spec.validate()?;
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: the cuts are tried on your own text."))?;
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    if !train.is_file() {
        return Err(Error::new("The saved training data of this project is missing. Save a version in Step 2 again."));
    }
    Ok(json!({
        "job": "reshape_try", "source": {"repo": m.repo, "family": m.family}, "load": spec.load, "out_dir": try_dir(p, m, version, spec), "plan_only": plan_only, "force": spec.force,
        "calibration": calibration_json(Some(&train), spec.samples, spec.len, Some(version), spec.reasoning.as_ref()),
        "measure_dir": scores_dir(p, &scores_key_with(&m.repo, Some(version), spec.measure_samples, spec.len, &spec.load, spec.reasoning.as_ref())), "measure_samples": spec.measure_samples,
        "protect": {"first": spec.protect_first, "last": spec.protect_last}, "strategy": spec.strategy, "want": spec.want(), "params_b": spec.params_b,
    }))
}

fn try_model(spec: &TrySpec) -> Result<ModelEntry> {
    catalog::find(&spec.model_id).ok_or_else(|| Error::new("That model is not in the list any more."))
}

/// How many cuts the request needs, how many are saved already and the time the rest will take (a guess the first time, then measured). Loads nothing.
pub fn try_plan(p: &Project, spec: &TrySpec, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let m = try_model(spec)?;
    let job = try_job(p, &m, spec, true)?;
    let py = python()?;
    let mut r = run_job(&py, &worker_dir(), &job, kill, &mut |e| on_event(event_json(e)))?;
    r["dir"] = job["out_dir"].clone();
    Ok(r)
}

/// The answers kept for this model, data and settings (`{"exists": false}` when there are none). Whether they still belong to the weights is checked by the worker.
pub fn try_results(p: &Project, model_id: &str, samples: u32, len: u32, load: &str) -> Result<Value> {
    try_results_with(p, model_id, samples, len, load, None)
}

/// `try_results` for a measuring text with a reasoning part.
pub fn try_results_with(p: &Project, model_id: &str, samples: u32, len: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Result<Value> {
    let m = catalog::find(model_id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
    let Some((version, _)) = crate::train::runs::latest_version(p) else { return Ok(json!({"exists": false})) };
    let file = scores_dir(p, &scores_key_with(&m.repo, Some(version), samples, len, load, reasoning)).join("reshape_try.json");
    let Some(v) = read_json(&file) else { return Ok(json!({"exists": false})) };
    if v["key"]["source"] != json!(m.repo) || v["key"]["data_version"] != json!(version) {
        return Ok(json!({"exists": false}));
    }
    Ok(json!({"exists": true, "analysis": v["analysis"], "seconds_per_pass": v["seconds_per_pass"], "tokens": v["tokens"], "load": v["load_used"], "passes": v["passes"].as_object().map(|o| o.len()).unwrap_or(0)}))
}

/// Runs the cuts that are not saved yet (a stop keeps the finished ones). Events: `try_pass` after every cut, `progress`, `log`. Returns the worker's result.
pub fn try_run(p: &Project, spec: &TrySpec, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let m = try_model(spec)?;
    let job = try_job(p, &m, spec, false)?;
    let py = python()?;
    let folder = PathBuf::from(job["out_dir"].as_str().unwrap_or_default());
    let _active = Active::begin(&folder)?;
    let stop_dir = models_dir(p).join("_scores");
    let mut result = crate::train::runs::drive(&py, &stop_dir, &job, stop, Duration::from_secs(60), &mut |e| on_event(event_json(e)))?;
    result["samples"] = json!(spec.samples);
    Ok(result)
}

/// What the Heal card shows for a made model: how it would run on this card, the budgets with the disk and time they need, and where it stands.
pub fn heal_options(p: &Project, name: &str, vram_gb: Option<f64>) -> Result<Value> {
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    let o = heal_source::read_origin(&dir)?;
    Ok(heal_options_with(p, Some((name, &dir)), &o, vram_gb))
}

/// The same for a pair that may have no folder yet (`dir` None: nothing was written, so there is no state, no chart and nothing healed).
pub fn heal_options_with(p: &Project, dir: Option<(&str, &Path)>, o: &heal_source::Origin, vram_gb: Option<f64>) -> Value {
    let (student_b, teacher_b) = (o.student_b, o.teacher_b);
    let dims = plan::HealDims::from_config(&o.config, student_b);
    let defaults = plan::heal_defaults(student_b, vram_gb, &dims);
    let hp = plan::heal_plan_with(student_b, teacher_b, vram_gb, &dims, &defaults);
    let tokens: Vec<Value> = plan::HEAL_TOKENS_M.iter().map(|&t| {
        let m = t as f64 / 10.0;
        json!({"tokens_m": t, "cache_gb": (hp.cache_gb_per_10m * m * 10.0).round() / 10.0, "hours": [((hp.hours_per_10m.0 * m) * 10.0).round() / 10.0, ((hp.hours_per_10m.1 * m) * 10.0).round() / 10.0]})
    }).collect();
    let heal = dir.and_then(|(_, d)| read_json(&d.join("heal.json")));
    let free_in = dir.map(|(_, d)| d.to_path_buf()).unwrap_or_else(|| models_dir(p));
    json!({
        "name": dir.map(|(n, _)| n), "plan": hp, "defaults": defaults, "dims": dims, "upcast_gb": plan::heal_upcast_extra_gb(student_b, &dims, &defaults), "head_off_gb": plan::heal_head_off_extra_gb(student_b, &dims, &defaults),
        "head_off_fits": vram_gb.map(|v| plan::heal_memory_gb(student_b, &dims, &plan::HealKnobs { head_ckpt: false, ..defaults.clone() }) <= v), "safer": heal.as_ref().filter(|h| h["status"] == "failed" && h["oom"] == true).map(heal_safer).unwrap_or(Value::Null), "tokens": tokens, "default_tokens_m": plan::default_heal_tokens_m(student_b), "student_b": (student_b * 100.0).round() / 100.0,
        "teacher": o.teacher, "external": o.external, "teacher_chosen": o.teacher_chosen, "student_label": o.student_label, "general_dataset": DEFAULT_GENERAL_DATASET, "general": crate::general::state(Some(p)), "has_data": crate::train::runs::latest_version(p).is_some(),
        "healed": dir.is_some_and(|(_, d)| d.join("healed").join("config.json").is_file()), "heal": heal, "metrics": dir.map(|(n, _)| heal_metrics(p, n)).unwrap_or_default(), "vram_gb": vram_gb,
        "free_disk_gb": crate::envmgr::health::free_disk_gb(&free_in).map(|g| (g * 10.0).round() / 10.0),
        "calibration": o.calibration, "loss": {"original": o.loss_before, "thinned": o.loss_after}, "recipe": thinking_recipe_json(o, &dims, vram_gb),
    })
}

/// What the Heal card needs for "Keep thinking": the recipe's numbers for this student and card, and whether it is on by default (Qwen3.5 / Qwen3.8 students). Null for other models.
fn thinking_recipe_json(o: &heal_source::Origin, dims: &plan::HealDims, vram_gb: Option<f64>) -> Value {
    if o.family != "qwen3_5" {
        return Value::Null;
    }
    let (seq, batch, rank) = plan::thinking_knobs(o.student_b, vram_gb, dims);
    let cut = if o.teacher_b > 0.0 { (1.0 - o.student_b / o.teacher_b).max(0.0) } else { 0.0 };
    json!({"id": plan::THINKING_RECIPE, "tokens_m": plan::thinking_tokens_m(o.student_b, o.teacher_b), "general_share": plan::THINKING_SHARE, "general": plan::THINKING_GENERAL_ID, "seq": seq, "batch": batch,
           "rank": rank, "cut_share": (cut * 1000.0).round() / 1000.0, "default_on": true})
}

/// The Heal card's live estimate: the plan (memory in plain words, the breakdown, the way that fits) for the settings the person has chosen so far, and the
/// settings that would really be used (`knobs`, the automatic value where nothing was chosen).
pub fn heal_estimate(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>) -> Result<Value> {
    valid_name(name)?;
    spec.validate()?;
    let dir = models_dir(p).join(name);
    let o = heal_source::read_origin(&dir)?;
    let mut out = heal_estimate_with(&o, spec, vram_gb);
    // the project's own files: the saved check of the vLLM teacher and the speed its scoring measured
    let st = crate::settings::Settings::load().vllm;
    let check = crate::vllm::read_teacher_check(&dir, &st, &o.teacher);
    out["vllm_teacher"] = vllm_teacher_estimate(&o, spec, &crate::vllm::history_with_measured(crate::timing::History::default(), Some(p)), &st, Some(&check));
    out["vllm_teacher"]["files"] = crate::vllm::teacher_check_files(&dir);
    Ok(out)
}

/// The estimate for a pair (`heal_source::origin_from_check`) that may have no folder yet.
pub fn heal_estimate_with(o: &heal_source::Origin, spec: &HealSpec, vram_gb: Option<f64>) -> Value {
    let (student_b, teacher_b) = (o.student_b, o.teacher_b);
    let dims = plan::HealDims::from_config(&o.config, student_b);
    let ch = heal_choice(spec, student_b, vram_gb, &dims);
    let hp = plan::heal_plan_with(student_b, teacher_b, vram_gb, &dims, &ch.knobs);
    let mut out = json!({"plan": hp, "knobs": ch.knobs, "upcast_gb": plan::heal_upcast_extra_gb(student_b, &dims, &ch.knobs),
                          "head_off_gb": plan::heal_head_off_extra_gb(student_b, &dims, &ch.knobs),
                          "head_off_fits": vram_gb.map(|v| plan::heal_memory_gb(student_b, &dims, &plan::HealKnobs { head_ckpt: false, ..ch.knobs.clone() }) <= v), "optimizer": ch.optimizer, "k": ch.k, "per_update": ch.knobs.batch * ch.knobs.grad_accum});
    if let Some(r) = &spec.rounds {
        out["rounds"] = rounds_estimate_json(o, spec, r, &ch);
    }
    out["vllm_teacher"] = vllm_teacher_estimate(o, spec, &crate::vllm::history_with_measured(crate::timing::History::default(), None), &crate::settings::Settings::load().vllm, None);
    out
}

/// What the Heal card shows about making the original's answers with vLLM (experimental): the route (which weights, and why), the time with the standard engine and with vLLM from the
/// numbers measured on THIS PC (the self-test's start-up, the teacher check's scoring speed) or, where there are none, the report's estimates (said so), the warning that a heal in rounds
/// starts the engine every round, and the saved check of the vLLM teacher. `check`: the saved `vllm_teacher_check.json` of the model (None for a pair that has no folder yet).
pub fn vllm_teacher_estimate(o: &heal_source::Origin, spec: &HealSpec, h: &crate::timing::History, s: &crate::settings::VllmSettings, check: Option<&Value>) -> Value {
    use crate::vllm_plan as vp;
    let n = vp::Numbers::from_history(h);
    let tokens_m = match &spec.rounds {
        Some(r) => r.cap_tokens(spec) as f64 / 1e6,
        None => spec.tokens() as f64 / 1e6,
    };
    let std_min = super::auto_time::cache_minutes(&crate::timing::History::default(), o.teacher_b, tokens_m);
    let start = vp::startup(&n, o.teacher_b);
    let score = vp::scoring(&n, tokens_m);
    let starts = spec.rounds.as_ref().map(|r| (r.cap_tokens(spec) as f64 / r.round_tokens().max(1) as f64).ceil().max(1.0)).unwrap_or(1.0);
    let vllm = (start.lo * starts + score.lo, start.hi * starts + score.hi);
    let route = crate::vllm::teacher_route(&s.teacher, &o.teacher);
    let better = (vllm.0 + vllm.1) < (std_min.0 + std_min.1) * vp::AUTO_MARGIN;
    let text = if route["route"] == "refused" {
        route["reason"].as_str().unwrap_or("").to_string()
    } else {
        format!("The original's answers: about {:.1} to {:.1} hours with the standard engine, about {:.1} to {:.1} hours with vLLM ({}start-up {:.1} to {:.1} minutes{}, scoring {}).{}",
                std_min.0 / 60.0, std_min.1 / 60.0, vllm.0 / 60.0, vllm.1 / 60.0, if starts > 1.0 { format!("{starts:.0} engine starts, ") } else { String::new() }, start.lo, start.hi,
                if start.measured { " measured by your self-test" } else { ", not measured on this PC" }, if score.measured { "measured by your teacher check" } else { "the research report's estimate, not measured here" },
                if better { "" } else { " vLLM is not faster than the standard engine by this estimate." })
    };
    let mut warnings = Vec::new();
    if starts > 1.0 {
        warnings.push(format!("A heal in rounds asks the original for one chunk of answers per round, and every chunk starts the engine again: about {:.0} to {:.0} minutes of start-up per round, {starts:.0} rounds at most. Keep the standard engine for rounds unless a round is long.", start.lo, start.hi));
    }
    if matches!(route["route"].as_str(), Some("bnb")) {
        warnings.push("The in-flight 4-bit route has never been run: how vLLM does it and how fast it is are unknown (the research report says only that it is much slower than the pre-quantized weights).".into());
    }
    if check.is_none_or(|c| c.is_null()) {
        warnings.push("The vLLM teacher has not been checked against the standard teacher yet: use \"Check the vLLM teacher\" first (about 100 rows).".into());
    } else if check.is_some_and(|c| c["current"] == false) {
        warnings.push("The saved check of the vLLM teacher was made for other settings (another engine version or teacher route): check it again.".into());
    }
    json!({"asked": spec.vllm_teacher(), "settings_on": s.enabled && s.teacher.enabled, "route": route, "identity": crate::vllm::teacher_identity(s, &o.teacher), "tokens_m": tokens_m,
           "standard_min": [std_min.0.round(), std_min.1.round()], "vllm_min": [vllm.0.round(), vllm.1.round()], "startup_min": [(start.lo * 10.0).round() / 10.0, (start.hi * 10.0).round() / 10.0],
           "startup_measured": start.measured, "scoring_measured": score.measured, "engine_starts": starts, "faster": better, "text": text, "warnings": warnings, "basis": vp::basis_notes(&n, o.teacher_b),
           "check": check.cloned().unwrap_or(Value::Null), "required": s.teacher.required})
}

/// What the Heal card shows for a heal in rounds, additive to the estimate: the steps of a round, the overhead, and the time if the first check passes and if it runs to the cap
/// (formulas on the reference run: this computer's own timings are not read here).
fn rounds_estimate_json(o: &heal_source::Origin, spec: &HealSpec, r: &heal_rounds::RoundSpec, ch: &HealChoice) -> Value {
    use super::auto_time::{rounds_estimate, RoundsIn};
    let (seq, per_update) = (ch.knobs.seq, ch.knobs.batch * ch.knobs.grad_accum);
    let cap = r.cap_tokens(spec);
    let steps = heal_rounds::est_steps(seq, per_update, r.round_tokens());
    let cap_m = (cap as f64 / 1e6).ceil() as u32;
    let round_m = ((r.round_tokens() as f64 / 1e6).ceil() as u32).max(1);
    let rin = RoundsIn {
        teacher_b: o.teacher_b, student_b: o.student_b, teacher_gb: o.teacher_b * 2.0, student_gb: o.student_b * 2.0, teacher_load: ch.teacher_load.clone().unwrap_or_else(|| "4bit".into()), reuse_m: 0, cap_m,
        round_m, first_check_m: (r.first_check_tokens(spec) as f64 / 1e6).ceil() as u32, check_quick: r.quick(), final_quick: r.final_is_quick(), thinking: spec.thinking(),
    };
    let (parts, t) = rounds_estimate(&crate::timing::History::default(), &rin);
    let normal = {
        let n = RoundsIn { round_m: cap_m, first_check_m: cap_m, thinking: false, ..rin.clone() };
        rounds_estimate(&crate::timing::History::default(), &n).1.worst
    };
    json!({"steps_per_round": (steps * 10.0).round() / 10.0, "min_steps": heal_rounds::MIN_ROUND_STEPS, "too_few_steps": r.round_exact.is_none() && steps < heal_rounds::MIN_ROUND_STEPS as f64,
           "time": t, "parts": parts, "normal_heal": normal, "warn_overhead": t.overhead_share.1 > 0.2, "basis": "formula"})
}
