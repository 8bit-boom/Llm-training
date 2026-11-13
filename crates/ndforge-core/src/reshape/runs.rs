//! Slimmed and grown models: `models/<name>/` in the project holds state.json (status), result.json (what the worker found), scores.json (the layer
//! measurements for the chart), test.json (original vs this model) and `model/` (the new model folder, which is also added to the model list).
//! While the worker writes, the files are in `model.partial/`; it is renamed to `model/` at the end and removed after a failure or a stop.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::plan::{self, Inspect, Mode, ReshapeSpec};
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
    if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
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
    if let Some(ins) = cache.get(&key).and_then(|v| serde_json::from_value::<Inspect>(v.clone()).ok()) {
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
    let measures = spec.mode == Mode::Prune && spec.drop_layers.is_none();
    let version = crate::train::runs::latest_version(p).map(|(n, _)| n);
    let train = version.map(|n| p.folder.join("datasets").join(format!("v{n}")).join("train.jsonl")).filter(|f| f.is_file());
    if measures && train.is_none() {
        return Err(Error::new("Save a training-data version in Step 2 first: the layers are measured on your own text."));
    }
    let dir = models_dir(p).join(name);
    Ok(json!({
        "job": "reshape", "source": {"repo": m.repo, "family": m.family}, "mode": spec.mode, "blocks": spec.blocks, "strategy": spec.strategy,
        "protect": {"first": spec.protect_first, "last": spec.protect_last}, "drop_layers": spec.drop_layers,
        "grow": {"layout": "solar", "init": spec.init},
        "calibration": {"train": train, "samples": spec.calib_samples, "max_len": spec.calib_len, "seed": 3407, "version": version},
        "load": spec.load, "check": true, "shard_gb": 5,
        "out": dir.join("model.partial"), "final": dir.join("model"),
    }))
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
    let mut name = name_for(&m, spec.mode, layers_after(&ins, spec));
    // two models made in the same second (or a folder left over) never share a folder
    let mut n = 1;
    while models_dir(p).join(&name).exists() {
        n += 1;
        name = format!("{}-{n}", name_for(&m, spec.mode, layers_after(&ins, spec)));
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
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    let model = dir.join("model");
    let repo = model.canonicalize().unwrap_or(model.clone()).to_string_lossy().to_string();
    if let Some(m) = catalog::models().into_iter().find(|m| m.repo == repo) {
        set_state(&dir, json!({"model_id": m.id}));
        return Ok(m.id);
    }
    let mut entry = crate::train::hub::lookup(&repo)?.entry;
    let meta = read_json(&model.join("ndforge_reshape.json")).unwrap_or(Value::Null);
    let state = read_json(&dir.join("state.json")).unwrap_or(Value::Null);
    let source = state["source_model_id"].as_str().and_then(catalog::find);
    let origin = meta["origin"]["repo"].as_str().map(String::from).or_else(|| source.as_ref().map(|s| s.repo.clone()));
    let (before, after) = (meta["origin"]["layers"].as_u64().unwrap_or(0), meta["layers"].as_u64().unwrap_or(0));
    let slimmed = meta["mode"] == "prune";
    let base_name = source.as_ref().map(|s| s.name.clone()).unwrap_or_else(|| entry.name.clone());
    entry.id = format!("custom-local-{name}");
    entry.name = format!("{base_name} · {} {after} layers", if slimmed { "slim" } else { "grow" });
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
    entry.reshape = Some(format!("{}: {before} → {after} layers", if slimmed { "slimmed" } else { "grown" }));
    entry.good_for = format!("Made on the Slim or grow page from {}.", origin.as_deref().unwrap_or("another model"));
    entry.note = if slimmed {
        "A slimmed model answers worse than the original until it is trained again on your world (Step 3), then test it. Memory figures are estimates.".to_string()
    } else {
        "A grown model starts out answering exactly like the original; a world's data alone gives little gain, and it needs more memory than the original. Memory figures are estimates.".to_string()
    };
    let id = entry.id.clone();
    catalog::add_custom(entry)?;
    set_state(&dir, json!({"model_id": id}));
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
                "boundary_distance": r["boundary_distance"], "notes": r["notes"], "load": r["load"],
            });
        }
        st["scores"] = read_json(&dir.join("scores.json")).unwrap_or(Value::Null);
        let test = read_json(&dir.join("test.json"));
        st["tested"] = json!(test.is_some());
        // the headline and the scores (not the hundreds of answers) for the page
        st["test"] = test.map(|t| json!({"scores": t["scores"], "verdict": t["verdict"], "n": t["n"], "method": t["method"], "created": t["created"], "against": t["against"], "loaded": t["loaded"], "original": t["original"]})).unwrap_or(Value::Null);
        st["has_model"] = json!(dir.join("model").join("config.json").is_file());
        st["size_gb"] = json!(if st["has_model"] == true { (dir_size(&dir.join("model")) as f64 / 1e9 * 10.0).round() / 10.0 } else { 0.0 });
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
    let model = dir.join("model");
    let repo = model.canonicalize().unwrap_or(model.clone()).to_string_lossy().to_string();
    if let Ok(rd) = std::fs::read_dir(crate::train::runs::runs_dir(p)) {
        for e in rd.flatten() {
            let used = read_json(&e.path().join("config.json")).is_some_and(|c| c["model"]["repo"].as_str().is_some_and(|r| r == repo || Path::new(r) == model));
            if used {
                return Err(Error::new(format!("Run {} was trained on this model. Delete that run first, or keep the model.", e.file_name().to_string_lossy())));
            }
        }
    }
    if let Some(id) = catalog::models().into_iter().find(|m| m.repo == repo).map(|m| m.id) {
        catalog::remove_custom(&id)?;
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
    use crate::evaluate::runs::{execute_pair, Pair, Side};
    valid_name(name)?;
    let dir = models_dir(p).join(name);
    let model = dir.join("model");
    if !model.join("config.json").is_file() {
        return Err(Error::new("That model is not ready (it was stopped or failed)."));
    }
    let meta = read_json(&model.join("ndforge_reshape.json")).ok_or_else(|| Error::new("This model has no ndforge_reshape.json, so its original is unknown."))?;
    let origin = meta["origin"]["repo"].as_str().ok_or_else(|| Error::new("This model's original is unknown."))?.to_string();
    let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a training-data version in Step 2 first: the test asks questions about your world."))?;
    let big = |params: Option<f64>| vram_gb.zip(params).is_some_and(|(v, b)| b * 2.0 + 3.0 > v * 0.9);
    let pair = Pair {
        original: origin, family: meta["origin"]["family"].as_str().unwrap_or("generic").to_string(), trained: Side::Full(model.clone()),
        four_bit: (big(meta["origin"]["params"].as_f64().map(|x| x / 1e9)), big(meta["params"].as_f64().map(|x| x / 1e9))), against: "origin",
    };
    execute_pair(p, &pair, version, &format!("model:{name}"), &dir.join("test.json"), 30, judge, kill, on_event)
}

pub fn read_test(p: &Project, name: &str) -> Option<Value> {
    valid_name(name).ok()?;
    read_json(&models_dir(p).join(name).join("test.json"))
}
