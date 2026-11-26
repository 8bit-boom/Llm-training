//! What the window asks about presets, without the window: the identity of a made model, the machine now, the list for a model, saving the current card, and the preview of using a preset (the changes with
//! their measured gain). Everything here is read-only except `save_card`, which only writes a preset file.
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::identity::{detect, model_identity, MachineId, ModelIdentity};
use super::preset::{diff, Preset};
use super::store::{find, kind_of, MatchKind, Scope, Store};
use crate::error::{Error, Result};
use crate::project::Project;

/// The time as a preset records it (UTC, to the second).
pub fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// The identity of a made model (`models/<name>`): its student folder and where it came from.
pub fn model_for(p: &Project, name: &str) -> Result<ModelIdentity> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
        return Err(Error::new("Choose a model."));
    }
    let dir = p.folder.join("models").join(name);
    let folder = crate::reshape::heal_source::student_folder(&dir)?.ok_or_else(|| Error::new("The model is not on this computer yet: heal it once, or choose a model that is a folder."))?;
    let mut lineage = Vec::new();
    if let Ok(o) = crate::reshape::heal_source::read_origin(&dir) {
        lineage.push(o.teacher);
    }
    model_identity(&folder, name, lineage)
}

static MACHINE: Mutex<Option<(Instant, MachineId)>> = Mutex::new(None);

/// The software versions of the training environment, as the worker's Python sees them (empty when it is not installed or does not answer).
fn software() -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let py = crate::envmgr::python_path(&crate::envmgr::env_dir());
    if py.is_file() {
        let code = "import json, importlib.metadata as m\nd = {}\nfor k, p in (('torch','torch'),('transformers','transformers'),('peft','peft'),('bitsandbytes','bitsandbytes'),('triton','triton'),('triton','triton-windows'),('fla','flash-linear-attention'),('trl','trl')):\n    if k in d: continue\n    try: d[k] = m.version(p)\n    except Exception: pass\nprint(json.dumps(d))";
        let mut cmd = std::process::Command::new(&py);
        cmd.args(["-c", code]).stdin(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        if let Some(v) = cmd.output().ok().filter(|o| o.status.success()).and_then(|o| serde_json::from_slice::<Map<String, Value>>(&o.stdout).ok()) {
            for (k, v) in v {
                if let Some(s) = v.as_str() {
                    out.insert(k, s.to_string());
                }
            }
        }
    }
    out.insert("vllm_image".into(), crate::settings::Settings::load().vllm.image);
    out
}

/// This computer: the first graphics card and the software versions (cached for ten minutes: starting Python for the versions takes a moment).
pub fn machine_now() -> MachineId {
    let mut g = MACHINE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, m)) = g.as_ref() {
        if at.elapsed() < Duration::from_secs(600) {
            return m.clone();
        }
    }
    let m = detect(software());
    *g = Some((Instant::now(), m.clone()));
    m
}

pub fn forget_machine() {
    *MACHINE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

fn preset_json(m: &super::store::Match) -> Value {
    let p = &m.preset;
    json!({"id": p.id, "scope": m.scope, "name": p.name, "created": p.created, "kind": m.kind, "note": m.note, "whole": m.kind.whole(), "model": p.model.name, "params": p.model.params,
           "machine": {"gpu": p.machine.gpu, "vram_mb": p.machine.vram_mb}, "gains": p.gains().iter().map(|(k, g)| json!({"function": k, "gain": g})).collect::<Vec<_>>(),
           "stale": m.stale.iter().map(|s| json!({"function": s.function, "reasons": s.reasons})).collect::<Vec<_>>(), "report": p.report, "notes": p.notes,
           "classes": p.functions.iter().map(|(k, f)| json!({"function": k, "class": f.class, "tier": f.tier})).collect::<Vec<_>>()})
}

/// The window's list for a model: the model and machine facts, every preset that fits (with kind, gains, staleness) and the files that could not be read.
pub fn presets_view(p: &Project, model: &str) -> Result<Value> {
    let id = model_for(p, model)?;
    let machine = machine_now();
    let store = Store::new(Some(&p.folder));
    let matches = find(&store, &id, &machine);
    let mut unreadable = Vec::new();
    for scope in [Scope::Project, Scope::Library] {
        for u in store.list(scope).1 {
            unreadable.push(json!({"scope": scope, "file": u.file, "kept_as": u.kept_as, "why": u.why}));
        }
    }
    Ok(json!({"model": {"name": id.name, "shape_id": id.shape_id, "weights_id": id.weights_id, "layers": id.layers, "hidden": id.hidden, "params": id.params, "arch": id.arch, "lineage": id.lineage},
              "machine": {"key": machine.key, "gpu": machine.gpu, "vram_mb": machine.vram_mb, "driver": machine.driver, "software": machine.software},
              "presets": matches.iter().map(preset_json).collect::<Vec<_>>(), "unreadable": unreadable}))
}

/// Saves the current Heal card (every knob with its value) as a preset of this model on this machine, with no measured numbers (the person's own settings).
pub fn save_card(p: &Project, model: &str, name: &str, scope: Scope, card: Map<String, Value>, now: &str) -> Result<Preset> {
    let id = model_for(p, model)?;
    let machine = machine_now();
    let mut preset = Preset::new(name, now, "", id, machine);
    preset.card = card;
    preset.notes.push("Saved from the settings of the Heal card: no speed test is behind it.".into());
    Store::new(Some(&p.folder)).save(scope, &preset)?;
    Ok(preset)
}

/// The knobs each target of "Use" takes from a preset (None: the page takes whatever the preset sets).
fn target_keys(target: &str) -> Option<&'static [&'static str]> {
    match target {
        "settings_teacher" => Some(&["teacher_batch_tokens"]),
        "settings_vllm" => Some(&["max_num_seqs", "vllm_batch_tokens", "gpu_util_pct"]),
        "settings_llama" => Some(&["llama_ngl"]),
        "settings_forward" => Some(&["try_calib_batch", "think_batch"]),
        _ => None,
    }
}

/// The values of the saved settings a settings target compares with.
pub fn settings_current(s: &crate::settings::Settings) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("teacher_batch_tokens".into(), json!(s.teacher_batch_tokens));
    m.insert("max_num_seqs".into(), json!(s.vllm.teacher.max_num_seqs));
    m.insert("vllm_batch_tokens".into(), json!(s.vllm.teacher.batch_tokens));
    m.insert("gpu_util_pct".into(), json!(s.vllm.teacher.gpu_util_pct));
    m.insert("llama_ngl".into(), json!(s.llama_ngl.unwrap_or(super::llama::ALL)));
    m.insert("try_calib_batch".into(), json!(s.try_calib_batch));
    m.insert("think_batch".into(), json!(s.think_batch));
    m
}

/// Adds the saved settings the tests start from (the original's tokens per forward pass, the vLLM teacher's engine settings) to a run's card where the window did not send them.
pub fn with_settings(card: &mut Map<String, Value>) {
    for (k, v) in settings_current(&crate::settings::Settings::load()) {
        card.entry(k).or_insert(v);
    }
}

/// Applies a previewed change to the saved settings; the token must be the one of a fresh preview of the same preset against the settings as they are now (a stale or missing token changes nothing).
pub fn apply_settings(p: &Project, scope: Scope, id: &str, model: &str, target: &str, token: &str) -> Result<Value> {
    if !matches!(target, "settings_teacher" | "settings_vllm" | "settings_llama" | "settings_forward") {
        return Err(Error::new("Only the settings targets are applied here; the pages fill their own settings."));
    }
    let pv = apply_preview(p, scope, id, model, target, &Map::new())?;
    if token.is_empty() || pv["token"].as_str() != Some(token) {
        return Err(Error::new("The settings or the preset changed since the preview. Look at the new preview and confirm again. Nothing was changed."));
    }
    let mut s = crate::settings::Settings::load();
    let mut done = Vec::new();
    for c in pv["changes"].as_array().cloned().unwrap_or_default() {
        let (k, to) = (c["setting"].as_str().unwrap_or(""), c["to"].as_u64().unwrap_or(0) as u32);
        match k {
            "teacher_batch_tokens" => s.teacher_batch_tokens = to,
            "max_num_seqs" => s.vllm.teacher.max_num_seqs = to,
            "vllm_batch_tokens" => s.vllm.teacher.batch_tokens = to,
            "gpu_util_pct" => s.vllm.teacher.gpu_util_pct = to,
            "try_calib_batch" => s.try_calib_batch = to,
            "think_batch" => s.think_batch = to,
            "llama_ngl" => s.llama_ngl = Some(to).filter(|n| *n < super::llama::ALL),          // "all of them" is the empty setting
            _ => continue,
        }
        done.push(c);
    }
    s.save();
    Ok(json!({"applied": done}))
}

/// A readable label and a short reason for the knobs a preset sets.
fn label(knob: &str) -> &str {
    match knob {
        "seq" => "Sequence length",
        "batch" => "Batch per step",
        "per_update" => "Sequences per update",
        "checkpointing" => "Gradient checkpointing",
        "rank" => "LoRA size",
        "method" => "Method",
        "optimizer" => "Optimizer",
        "kl_chunk" => "Loss chunk",
        "k" => "Answers kept per token (top-k)",
        "teacher_load" => "The original is loaded as",
        "warmup" => "Warm-up steps",
        "lr" => "Learning rate",
        "head_checkpoint" => "Output-layer logits",
        "student_batch_tokens" => "Tokens per forward pass",
        "teacher_batch_tokens" => "Teacher tokens per forward pass",
        "max_num_seqs" => "vLLM rows scored at once",
        "llama_ngl" => "llama.cpp: layers on the card",
        "try_calib_batch" => "Try first: rows measured together",
        "think_batch" => "Thinking check: questions asked together",
        "vllm_batch_tokens" => "vLLM tokens per batch",
        "gpu_util_pct" => "vLLM share of the card (%)",
        other => other,
    }
}

/// What using a preset would change, against `current` (the card's values as the window has them): `{changes: [{setting, label, from, to, gain, function, class, why}], whole, kind, note, stale, token, gain}`.
/// An exact or same-shape match offers the whole preset; a parent or nearest match only the knobs that carry over; another PC is refused. The token is a hash of the preset and the changes
/// (`tune_apply` of a settings target must carry it back).
pub fn apply_preview(p: &Project, scope: Scope, id: &str, model: &str, target: &str, current: &Map<String, Value>) -> Result<Value> {
    let settings_now;
    let current = match target {
        "settings_teacher" | "settings_vllm" | "settings_llama" | "settings_forward" => {
            settings_now = settings_current(&crate::settings::Settings::load());
            &settings_now
        }
        "heal_card" | "auto" | "try" | "think" | "imatrix" => current,
        _ => return Err(Error::new("Apply to what? heal_card, auto, settings_teacher, settings_vllm, settings_llama, settings_forward, try, think or imatrix.")),
    };
    let store = Store::new(Some(&p.folder));
    let preset = store.get(scope, id)?;
    let mid = model_for(p, model)?;
    let machine = machine_now();
    let (kind, note) = kind_of(&preset, &mid, &machine).ok_or_else(|| Error::new("That preset does not belong to this model."))?;
    if kind == MatchKind::OtherPc {
        return Err(Error::new(format!("That preset was measured on another PC ({}): it is listed but never applied here.", preset.machine.gpu)));
    }
    let mut values = if kind.whole() { preset.values() } else { preset.transferable_values() };
    match target_keys(target) {
        Some(keys) => values.retain(|k, _| keys.contains(&k.as_str())),          // a settings target only takes its own knobs
        None => values.retain(|k, _| !super::real::SETTINGS_KEYS.contains(&k.as_str())),          // the pages' own fields never take a saved setting
    }
    let mut changes = Vec::new();
    for (k, from, to) in diff(current, &values) {
        let f = preset.functions.iter().find(|(_, f)| f.chosen.contains_key(&k));
        changes.push(json!({"setting": k, "label": label(&k), "from": from, "to": to, "function": f.map(|(n, _)| n.clone()), "gain": f.map(|(_, f)| f.gain),
                            "class": f.map(|(_, f)| f.class.clone()), "why": f.map(|(_, f)| f.note.clone()).unwrap_or_default()}));
    }
    let stale: Vec<Value> = super::store::stale_of(&preset, &machine).iter().map(|s| json!({"function": s.function, "reasons": s.reasons})).collect();
    let mut h = Sha256::new();
    h.update(format!("{id}|{}|{}", scope as u8, serde_json::to_string(&changes).unwrap_or_default()));
    let token = format!("a{}", &h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()[..16]);
    let gain = preset.functions.values().filter(|f| f.gain > 0.0 && f.tier == 1).map(|f| f.gain).fold(0.0, f64::max);
    Ok(json!({"changes": changes, "whole": kind.whole(), "kind": kind, "note": note, "stale": stale, "token": token, "gain": gain, "name": preset.name, "context": preset.functions.values().map(|f| f.context.clone()).collect::<Vec<_>>()}))
}

/// Every preset of this PC (project and library), for the Auto page's "Use tested settings": the ones measured on another card are left out. `[{id, scope, name, model, created, gains}]`.
pub fn presets_here(p: Option<&Project>) -> Vec<Value> {
    let machine = machine_now();
    let store = Store::new(p.map(|p| p.folder.as_path()));
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    let mut out = Vec::new();
    for scope in [Scope::Project, Scope::Library] {
        for pr in store.list(scope).0 {
            if pr.machine.key != machine.key || seen.insert(pr.id.clone(), ()).is_some() {
                continue;
            }
            out.push(json!({"id": pr.id, "scope": scope, "name": pr.name, "model": pr.model.name, "created": pr.created, "gains": pr.gains().iter().map(|(k, g)| json!({"function": k, "gain": g})).collect::<Vec<_>>()}));
        }
    }
    out
}

/// A preset as an Auto-cut snapshot (`AutoSpec.tuned`): its values and the measured speed of the student heal.
pub fn tuned_snapshot(p: &Project, scope: Scope, id: &str) -> Result<crate::reshape::auto_plan::TunedUse> {
    let preset = Store::new(Some(&p.folder)).get(scope, id)?;
    let machine = machine_now();
    if preset.machine.key != machine.key {
        return Err(Error::new("That preset was measured on another PC: it is not used here."));
    }
    let sb = preset.functions.get("heal.student_batch");
    let m = sb.map(|f| &f.measured);
    Ok(crate::reshape::auto_plan::TunedUse {
        preset_id: preset.id.clone(), scope: format!("{}", match scope { Scope::Project => "project", Scope::Library => "library" }), name: preset.name.clone(), match_kind: "chosen".into(), values: preset.values(),
        tokens_per_s: m.and_then(|m| m["tokens_per_s"].as_f64()), measured_student_b: m.and_then(|m| m["student_b"].as_f64()).or(Some(preset.model.params as f64 / 1e9).filter(|b| *b > 0.0)),
        default_tokens_per_s: m.and_then(|m| m["default_tokens_per_s"].as_f64()),
        chosen_keys: preset.functions.values().flat_map(|f| f.chosen.keys().cloned()).collect(),
    })
}
