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

use super::plan::{self, Inspect, Mode, ReshapeSpec, Width};
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

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Folder names come from the window: no path tricks.
fn valid_name(name: &str) -> Result<()> {
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
fn slug(text: &str) -> String {
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
    format!("{}-v{}-{samples}x{len}-{load}", slug(source_repo), version.unwrap_or(0))
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

fn python() -> Result<PathBuf> {
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
        "calibration": {"train": train, "samples": samples, "max_len": spec.calib_len, "seed": 3407, "version": version},
        "load": spec.load, "check": true, "shard_gb": 5,
        "out": dir.join("model.partial"), "final": dir.join("model"),
    });
    if let Some(w) = &spec.width {
        job["width"] = json!({"intermediate_size": w.intermediate, "linear_num_value_heads": w.lin_v_heads, "num_attention_heads": w.heads});
        job["scores_dir"] = json!(scores_dir(p, &scores_key(&m.repo, version, samples, spec.calib_len, &spec.load)));
    }
    Ok(job)
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
        Err(e) if e.0 == "Cancelled." => {
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

// ------------------------------------------------------------------------------------------------------------- the list

/// Adds `models/<name>/model` to the model list so it can be chosen as a base model in Step 3. Returns the model's id (the same one when called again).
pub fn register(p: &Project, name: &str) -> Result<String> {
    register_folder(p, name, "model")
}

/// The same for `models/<name>/healed` (id `custom-local-<name>-healed`).
pub fn register_healed(p: &Project, name: &str) -> Result<String> {
    register_folder(p, name, "healed")
}

fn thousands(n: u64) -> String {
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

fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|d| d.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()).unwrap_or(0)
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
        st["cache_gb"] = json!((dir_size(&dir.join("distill").join("cache")) as f64 / 1e9 * 10.0).round() / 10.0);
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
        let Some(st) = read_json(&dir.join("state.json")) else { continue };
        let live = active().lock().map(|a| a.contains(&dir)).unwrap_or(false);
        // healing that was cut off can be continued: its checkpoints and the teacher's answers are kept
        if !live && read_json(&dir.join("heal.json")).is_some_and(|h| h["status"] == "running") {
            let _ = std::fs::remove_dir_all(dir.join("healed.partial"));
            set_heal(&dir, json!({"status": "stopped", "error": "The program was closed while this model was being healed. Continue to go on where it stopped."}));
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
fn r64() -> u32 {
    64
}
fn lr_default() -> f64 {
    1e-4
}

/// What the person chose for healing (Expert fields have defaults). `tokens_m`: millions of tokens, one of 10 / 20 / 40 / 80.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealSpec {
    pub tokens_m: u32,
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
    #[serde(default = "r64")]
    pub rank: u32,
    #[serde(default = "lr_default")]
    pub lr: f64,
    /// "cache" | "online"; None = what fits the card
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
}

impl HealSpec {
    pub fn validate(&self) -> Result<()> {
        if !plan::HEAL_TOKENS_M.contains(&self.tokens_m) {
            return Err(Error::new("Heal on 10, 20, 40 or 80 million tokens."));
        }
        if !(0.0..=0.9).contains(&self.general_share) || !(0.0..=1.0).contains(&self.alpha) || !(4..=256).contains(&self.rank) || !(1e-6..=1e-2).contains(&self.lr) {
            return Err(Error::new("Healing settings out of range: general share 0 to 0.9, KL weight 0 to 1, LoRA size 4 to 256, learning rate 0.000001 to 0.01."));
        }
        if self.mode.as_deref().is_some_and(|m| !["cache", "online"].contains(&m)) {
            return Err(Error::new("Unknown way of healing."));
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
        crate::general::resolve(self.general.as_ref().or(legacy.as_ref()), true).map(Some)
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

/// One healing job: runs it, forwards its events tagged {"phase": "heal", "heal_step": step}, keeps the chart numbers (`heal_metrics.jsonl`).
fn heal_job(py: &Path, dir: &Path, job: &Value, step: &str, stop: &Arc<AtomicBool>, grace: Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    set_heal(dir, json!({"step": step}));
    crate::train::runs::drive(py, dir, job, stop.clone(), grace, &mut |e| {
        let mut ev = event_json(e);
        ev["phase"] = json!("heal");
        ev["heal_step"] = json!(step);
        if matches!(ev["kind"].as_str(), Some("distill_metric" | "distill_eval")) {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("heal_metrics.jsonl")) {
                let _ = writeln!(f, "{ev}");
            }
        }
        on_event(ev);
    })
}

/// The teacher's answers are all there AND were made from this general text (an older cache has no key and counts as made from other data).
fn cache_complete(cache: &Path, want: &Value) -> bool {
    read_json(&cache.join("index.json")).is_some_and(|i| i["complete"] == true && i["general_key"] == *want)
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
    let model = dir.join("model");
    if !model.join("config.json").is_file() {
        return Err(Error::new("That model is not ready (it was stopped or failed)."));
    }
    if dir.join("healed").join("config.json").is_file() {
        return Err(Error::new("This model has already been healed."));
    }
    let meta = read_json(&model.join("ndforge_reshape.json")).ok_or_else(|| Error::new("This model has no ndforge_reshape.json, so its original is unknown."))?;
    let origin = meta["origin"]["repo"].as_str().ok_or_else(|| Error::new("This model's original is unknown."))?.to_string();
    let family = meta["origin"]["family"].as_str().unwrap_or("generic").to_string();
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: healing uses your world's text."))?;
    let train = p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl");
    if !train.is_file() {
        return Err(Error::new("The saved training data of this project is missing. Save a version in Step 2 again."));
    }
    let student_b = meta["params"].as_f64().unwrap_or(0.0) / 1e9;
    let teacher_b = meta["origin"]["params"].as_f64().map(|x| x / 1e9).unwrap_or(student_b * 1.3);
    let hp = plan::heal_plan(student_b, teacher_b, vram_gb);
    if !hp.fits {
        return Err(Error::new(hp.why.clone().unwrap_or_else(|| "Healing does not fit this card.".into())));
    }
    let mode = spec.mode.clone().unwrap_or_else(|| hp.mode.to_string());
    let tokens = spec.tokens_m as u64 * 1_000_000;
    let source = spec.general_source()?;
    let general = source.as_ref().map(|s| crate::general::job_block(s, spec.general_share)).unwrap_or(Value::Null);
    let general_key = crate::general::key(&general, crate::general::SEED);
    let (cache_dir, run_dir) = (dir.join("distill").join("cache"), dir.join("distill").join("run"));
    if mode == "cache" && !cache_complete(&cache_dir, &general_key) {
        let need = tokens as f64 * 196.0 / 1e9;
        if let Some(free) = crate::envmgr::health::free_disk_gb(&dir).filter(|f| *f < need * 1.05) {
            return Err(Error::new(format!("The teacher's answers need {need:.1} GB in {}; {free:.1} GB free.", cache_dir.display())));
        }
    }
    let py = python()?;
    let _active = Active::begin(&dir)?;
    // a stopped heal that goes on with another general text cannot continue from the old one's checkpoints
    if read_json(&dir.join("heal.json")).map(|h| h["general_key"].clone()).as_ref() != Some(&general_key) && run_dir.exists() {
        let _ = std::fs::remove_dir_all(&run_dir);
    }
    let cal = &meta["calibration"];
    let calibration = json!({"train": train, "samples": cal["samples"].as_u64().unwrap_or(128), "max_len": cal["max_len"].as_u64().unwrap_or(512), "batch": 4, "seed": 3407, "version": version});
    let stop_file = dir.join("STOP");
    let started_at = read_json(&dir.join("heal.json")).and_then(|h| h["created"].as_str().map(String::from)).unwrap_or_else(now);
    set_heal(&dir, json!({"status": "running", "step": "start", "spec": spec, "general": general, "general_key": general_key, "created": started_at, "error": null, "mode": mode, "plan": hp}));
    let stopped = |step: &str| {
        set_heal(&dir, json!({"status": "stopped", "step": step}));
        json!({"stopped": true, "name": name})
    };
    let failed = |e: Error| {
        if e.0 == "Cancelled." {
            set_heal(&dir, json!({"status": "stopped"}));
            return Error::new("Stopped. Healing can be continued where it stopped.");
        }
        set_heal(&dir, json!({"status": "failed", "error": e.0}));
        e
    };
    if mode == "cache" && !cache_complete(&cache_dir, &general_key) {
        let job = json!({
            "job": "distill_cache", "teacher": {"repo": origin, "load": hp.teacher_load}, "student": model, "model": {"family": family},
            "data": {"train": train, "tokens": tokens, "general": general}, "k": 32, "max_seq": 2048, "batch": 4, "chunk": 256, "seed": crate::general::SEED,
            "out": cache_dir, "stop_file": stop_file,
        });
        let r = heal_job(&py, &dir, &job, "cache", &stop, grace, on_event).map_err(failed)?;
        if r["stopped"].as_bool().unwrap_or(false) {
            return Ok(stopped("cache"));
        }
        let note = r["notes"].as_array().and_then(|n| n.first()).and_then(|n| n.as_str());
        // the page shows which general text was used or skipped, and why
        set_heal(&dir, json!({"note": note, "note_code": r["stats"]["general_error_code"], "general_used": {"source": r["stats"]["general_source"], "rows": r["stats"]["general_rows"],
                              "tokens": r["stats"]["general_tokens"], "detected": r["stats"]["general_detected"], "skipped": note.is_some()}}));
    }
    let mut summary = read_json(&run_dir.join("summary.json")).filter(|_| training_done(&run_dir)).unwrap_or(Value::Null);
    if summary.is_null() {
        let resume = std::fs::read_dir(run_dir.join("checkpoints")).map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().starts_with("checkpoint-"))).unwrap_or(false);
        let job = json!({
            "job": "distill", "model": {"repo": model, "family": family}, "method": hp.student_method, "backend": spec.backend.clone().unwrap_or_else(|| "hf".into()),
            "dataset": {"train": train, "general": general},
            "hyper": {"rank": spec.rank, "alpha": spec.rank, "lr": spec.lr, "max_seq": 2048, "batch": 4, "grad_accum": 8, "warmup": 30, "weight_decay": 0.0, "scheduler": "cosine",
                      "seed": 3407, "save_steps": 50, "logging_steps": 1, "tokens": tokens},
            "distill": {"mode": mode, "cache": cache_dir, "teacher": {"repo": origin, "load": hp.teacher_load}, "alpha": spec.alpha, "k": 32, "chunk": 256, "eval_every": 50},
            "eval": {"calibration": calibration, "teacher_loss": meta["loss_before"], "slim_loss": meta["loss_after"]},
            "out": run_dir, "resume": resume, "stop_file": stop_file,
        });
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
    let job = json!({"job": "distill_merge", "base": model, "adapter": run_dir.join("adapter"), "out": dir.join("healed.partial"), "final": dir.join("healed"), "summary": summary, "stop_file": stop_file});
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
    let version = crate::train::runs::latest_version(p).map(|(n, _)| n);
    let key = scores_key(&m.repo, version, samples, len, load);
    let meta = read_json(&scores_dir(p, &key).join("scores.json"));
    json!({"key": key, "exists": meta.is_some(), "tokens": meta.as_ref().map(|x| x["tokens"].clone()), "load": meta.as_ref().map(|x| x["load"].clone()),
           "created": meta.as_ref().map(|x| x["created"].clone()), "data_version": version})
}

/// "Measure now" (Expert: also "measure again"): measures the model once and keeps the result for every later size choice on the same data. Making a
/// thinner model measures by itself when nothing is kept, so this is optional. Returns the worker's result.
pub fn measure(p: &Project, model_id: &str, samples: u32, len: u32, load: &str, force: bool, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
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
    let key = scores_key(&m.repo, Some(version), samples, len, load);
    let folder = scores_dir(p, &key);
    let _active = Active::begin(&folder)?;
    let job = json!({
        "job": "reshape_measure", "source": {"repo": m.repo, "family": m.family}, "load": load, "out_dir": folder, "force": force,
        "calibration": {"train": train, "samples": samples, "max_len": len, "seed": 3407, "version": version},
    });
    let stop_dir = models_dir(p).join("_scores");
    let mut result = crate::train::runs::drive(&py, &stop_dir, &job, stop, Duration::from_secs(60), &mut |e| on_event(event_json(e)))?;
    result["key"] = json!(key);
    Ok(result)
}

/// What the Heal card shows for a made model: how it would run on this card, the budgets with the disk and time they need, and where it stands.
pub fn heal_options(p: &Project, name: &str, vram_gb: Option<f64>) -> Result<Value> {
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    let meta = read_json(&dir.join("model").join("ndforge_reshape.json")).ok_or_else(|| Error::new("This model has no ndforge_reshape.json, so its original is unknown."))?;
    let student_b = meta["params"].as_f64().unwrap_or(0.0) / 1e9;
    let teacher_b = meta["origin"]["params"].as_f64().map(|x| x / 1e9).unwrap_or(student_b * 1.3);
    let hp = plan::heal_plan(student_b, teacher_b, vram_gb);
    let tokens: Vec<Value> = plan::HEAL_TOKENS_M.iter().map(|&t| {
        let m = t as f64 / 10.0;
        json!({"tokens_m": t, "cache_gb": (hp.cache_gb_per_10m * m * 10.0).round() / 10.0, "hours": [((hp.hours_per_10m.0 * m) * 10.0).round() / 10.0, ((hp.hours_per_10m.1 * m) * 10.0).round() / 10.0]})
    }).collect();
    let heal = read_json(&dir.join("heal.json"));
    Ok(json!({
        "name": name, "plan": hp, "tokens": tokens, "default_tokens_m": plan::default_heal_tokens_m(student_b), "student_b": (student_b * 100.0).round() / 100.0,
        "teacher": meta["origin"]["repo"], "general_dataset": DEFAULT_GENERAL_DATASET, "general": crate::general::state(Some(p)), "has_data": crate::train::runs::latest_version(p).is_some(),
        "healed": dir.join("healed").join("config.json").is_file(), "heal": heal, "metrics": heal_metrics(p, name), "vram_gb": vram_gb,
        "free_disk_gb": crate::envmgr::health::free_disk_gb(&dir).map(|g| (g * 10.0).round() / 10.0),
        "calibration": meta["calibration"], "loss": {"original": meta["loss_before"], "thinned": meta["loss_after"]},
    }))
}
