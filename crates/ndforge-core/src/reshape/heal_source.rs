//! Healing any model, not only the ones the Slim page made (docs/HEAL_ANY_MODEL.md).
//!
//! A heal needs two models: the student (the one that is healed) and the teacher (the original that it learns from). For a model made on the Slim page the
//! student is `models/<name>/model` and the teacher is the original recorded in its `ndforge_reshape.json`. For any other student the program makes
//! `models/<name>/` with a `heal_source.json` (student, teacher, the file-level facts the Check found) and reads the student where it is (never a copy).
//! `read_origin` and `student_folder` are the two places the heal code asks "which student, which original": for a made model they answer from
//! `model/` and `ndforge_reshape.json` exactly as before.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::runs::{event_json, models_dir, now, python, read_json, slug, valid_name, write_json};
use crate::envmgr::jobs::run_job;
use crate::envmgr::worker_dir;
use crate::error::{Error, Result};
use crate::paths::config_dir;
use crate::project::Project;
use crate::train::catalog;

/// Where a model comes from. `Made` is a model of the Slim page (its folder is `models/<name>/model`); `Folder` a folder on this computer with config.json and
/// the .safetensors files; `Hub` a Hugging Face id (`owner/model`), read through the Hugging Face cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Source {
    Made { name: String },
    Folder { path: String },
    Hub { id: String },
}

/// `owner/name`: letters, digits, dot, dash, underscore
pub fn is_hub_id(s: &str) -> bool {
    let ok = |p: &str| !p.is_empty() && !p.starts_with(['.', '-']) && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    let mut it = s.split('/');
    matches!((it.next(), it.next(), it.next()), (Some(a), Some(b), None) if ok(a) && ok(b))
}

impl Source {
    pub fn validate(&self) -> Result<()> {
        match self {
            Source::Made { name } => valid_name(name),
            Source::Folder { path } if path.trim().is_empty() => Err(Error::new("Choose a folder.")),
            Source::Folder { .. } => Ok(()),
            Source::Hub { id } if is_hub_id(id.trim()) => Ok(()),
            Source::Hub { id } => Err(Error::new(format!("\"{id}\" is not a Hugging Face model name. It looks like owner/model."))),
        }
    }

    /// What the worker gets: the folder or the Hub id. A made model is its `model/` folder.
    pub fn repr(&self, p: &Project) -> String {
        match self {
            Source::Made { name } => models_dir(p).join(name).join("model").to_string_lossy().to_string(),
            Source::Folder { path } => path.trim().trim_matches(|c| c == '"' || c == '\'').to_string(),
            Source::Hub { id } => id.trim().to_string(),
        }
    }

    /// A short name for lists and file names
    pub fn label(&self) -> String {
        match self {
            Source::Made { name } => name.clone(),
            Source::Folder { path } => Path::new(path.trim().trim_end_matches(['/', '\\'])).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path.clone()),
            Source::Hub { id } => id.trim().to_string(),
        }
    }

    pub fn is_made(&self) -> bool {
        matches!(self, Source::Made { .. })
    }
}

/// `models/<name>/heal_source.json`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealSource {
    pub student: Source,
    /// None for a made model: its recorded original
    #[serde(default)]
    pub teacher: Option<Source>,
    pub created: String,
    pub student_name: String,
    #[serde(default)]
    pub teacher_name: String,
    /// where a Hugging Face student was put by the Hugging Face cache (set when its download is done)
    #[serde(default)]
    pub student_dir: Option<String>,
    /// the file-level facts of both models (the Check's answer): sizes, layers, hidden size, family
    #[serde(default)]
    pub check: Value,
}

pub fn file(dir: &Path) -> PathBuf {
    dir.join("heal_source.json")
}

pub fn read(dir: &Path) -> Option<HealSource> {
    read_json(&file(dir)).and_then(|v| serde_json::from_value(v).ok())
}

pub fn write(dir: &Path, h: &HealSource) -> Result<()> {
    write_json(&file(dir), &serde_json::to_value(h)?)
}

/// A heal of a model the Slim page did not make
pub fn is_external(dir: &Path) -> bool {
    read(dir).is_some_and(|h| !h.student.is_made())
}

/// What the heal code needs to know about the pair.
#[derive(Debug, Clone)]
pub struct Origin {
    /// the teacher: a Hugging Face id or a folder
    pub teacher: String,
    /// the chat family the worker formats the text for ("qwen3_5" or "generic")
    pub family: String,
    pub student_b: f64,
    pub teacher_b: f64,
    /// the student's sizes (config.json of the student, or the facts of the Check): layers, hidden size, vocabulary
    pub config: Value,
    pub calibration: Value,
    pub loss_before: Value,
    pub loss_after: Value,
    pub external: bool,
    /// the teacher was chosen by the person, not recorded by the Slim page
    pub teacher_chosen: bool,
    pub student_label: String,
}

fn billions(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| *x > 0.0).map(|x| x / 1e9)
}

fn family_of(facts_family: &Value) -> String {
    if facts_family == "qwen3_5" { "qwen3_5".into() } else { "generic".into() }
}

/// The student's sizes as the memory estimate reads them (`plan::HealDims::from_config`), from the Check's facts.
fn config_from_facts(f: &Value) -> Value {
    json!({"num_hidden_layers": f["layers"], "hidden_size": f["hidden"], "vocab_size": f["vocab_size"]})
}

/// The student and its original for the heal in `dir`. Made model: from `model/ndforge_reshape.json` (unchanged); a `heal_source.json` there may name another
/// teacher. Any other student: from `heal_source.json`.
pub fn read_origin(dir: &Path) -> Result<Origin> {
    let hs = read(dir);
    match hs.as_ref().filter(|h| !h.student.is_made()) {
        None => {
            let meta = read_json(&dir.join("model").join("ndforge_reshape.json")).ok_or_else(|| Error::new("This model has no ndforge_reshape.json, so its original is unknown."))?;
            let recorded = meta["origin"]["repo"].as_str().ok_or_else(|| Error::new("This model's original is unknown."))?.to_string();
            let student_b = meta["params"].as_f64().unwrap_or(0.0) / 1e9;
            let (mut teacher, mut teacher_b, mut chosen) = (recorded, meta["origin"]["params"].as_f64().map(|x| x / 1e9).unwrap_or(student_b * 1.3), false);
            let mut family = meta["origin"]["family"].as_str().unwrap_or("generic").to_string();
            if let Some(t) = hs.as_ref().and_then(|h| h.teacher.as_ref()) {
                // another teacher for a made model: its repr is a folder or an id; its size comes from the Check
                let repr = match t {
                    Source::Folder { path } => path.trim().to_string(),
                    Source::Hub { id } => id.trim().to_string(),
                    Source::Made { name } => dir.parent().map(|m| m.join(name).join("model").to_string_lossy().to_string()).unwrap_or_default(),
                };
                if repr != teacher {
                    let tf = &hs.as_ref().unwrap().check["teacher"];
                    teacher_b = billions(&tf["params"]).unwrap_or(student_b * 1.3);
                    family = family_of(&tf["family"]);
                    teacher = repr;
                    chosen = true;
                }
            }
            Ok(Origin {
                teacher, family, student_b, teacher_b, config: read_json(&dir.join("model").join("config.json")).unwrap_or(Value::Null), calibration: meta["calibration"].clone(),
                loss_before: meta["loss_before"].clone(), loss_after: meta["loss_after"].clone(), external: false, teacher_chosen: chosen,
                student_label: dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            })
        }
        Some(h) => {
            let t = h.teacher.as_ref().ok_or_else(|| Error::new("The original model (the teacher) of this heal is not chosen."))?;
            let teacher = match t {
                Source::Folder { path } => path.trim().to_string(),
                Source::Hub { id } => id.trim().to_string(),
                Source::Made { .. } => return Err(Error::new("The teacher must be a model folder or a Hugging Face model.")),
            };
            let sf = &h.check["student"];
            let student_b = billions(&sf["params"]).unwrap_or(0.0);
            let teacher_b = billions(&h.check["teacher"]["params"]).unwrap_or(student_b * 1.3);
            Ok(Origin {
                teacher, family: family_of(&sf["family"]), student_b, teacher_b, config: config_from_facts(sf), calibration: Value::Null, loss_before: Value::Null, loss_after: Value::Null,
                external: true, teacher_chosen: true, student_label: h.student_name.clone(),
            })
        }
    }
}

/// The folder the student is read from. Made: `model/` (it must be ready). Folder: the folder where it is. Hugging Face: the Hugging Face cache folder, which
/// is known once the download is done (`Ok(None)` before: the heal downloads it first).
pub fn student_folder(dir: &Path) -> Result<Option<PathBuf>> {
    let ready = |f: PathBuf| if f.join("config.json").is_file() { Ok(Some(f)) } else { Err(Error::new("That model is not ready (it was stopped or failed).")) };
    match read(dir).filter(|h| !h.student.is_made()) {
        None => ready(dir.join("model")),
        Some(h) => match &h.student {
            Source::Folder { path } => {
                let f = PathBuf::from(path.trim());
                if f.join("config.json").is_file() { Ok(Some(f)) } else { Err(Error::new(format!("The folder {} is not there any more (or has no config.json). Choose the model again.", f.display()))) }
            }
            Source::Hub { .. } => Ok(h.student_dir.as_ref().map(PathBuf::from).filter(|f| f.join("config.json").is_file())),
            Source::Made { .. } => ready(dir.join("model")),
        },
    }
}

/// The folder the student is read from; an error when a Hugging Face model has not been downloaded yet.
pub fn resolve_student(dir: &Path) -> Result<PathBuf> {
    student_folder(dir)?.ok_or_else(|| Error::new("This model has not been downloaded yet. Start the heal: it is downloaded first."))
}

// ------------------------------------------------------------------------------------------------------------------ the Check

fn facts_cache_file() -> PathBuf {
    config_dir().join("heal_facts.json")
}

fn epoch() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// What identifies the files of a model: a folder's config.json modification time; a Hugging Face model is asked again after ten minutes.
fn facts_key(repr: &str) -> (String, Option<u64>) {
    let dir = Path::new(repr);
    if dir.is_dir() {
        let mtime = std::fs::metadata(dir.join("config.json")).and_then(|x| x.modified()).ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
        (format!("v2|{repr}@{mtime}"), None)
    } else {
        (format!("v2|{repr}"), Some(600))
    }
}

/// The file-level facts of one or two models from the worker (`heal_check` job: configs, tokenizer files and .safetensors headers; nothing is loaded), kept per
/// model so that the window can ask again for free. `refresh` asks the worker again.
pub fn facts(student: &str, teacher: Option<&str>, refresh: bool, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<(Value, Option<Value>)> {
    let mut cache = read_json(&facts_cache_file()).unwrap_or_else(|| json!({}));
    let fresh = |repr: &str| -> Option<Value> {
        if refresh {
            return None;
        }
        let (key, ttl) = facts_key(repr);
        let e = cache.get(&key)?;
        if ttl.is_some_and(|t| epoch().saturating_sub(e["at"].as_u64().unwrap_or(0)) > t) {
            return None;
        }
        Some(e["facts"].clone())
    };
    let (mut s, mut t) = (fresh(student), teacher.and_then(fresh));
    let mut ask = serde_json::Map::new();
    if s.is_none() {
        ask.insert("student".into(), json!(student));
    }
    if let (None, Some(tr)) = (&t, teacher) {
        ask.insert("teacher".into(), json!(tr));
    }
    if !ask.is_empty() {
        let py = python()?;
        let r = run_job(&py, &worker_dir(), &json!({"job": "heal_check", "sources": ask}), kill, &mut |e| on_event(event_json(e)))?;
        for (side, repr, slot) in [("student", Some(student), &mut s), ("teacher", teacher, &mut t)] {
            if let (Some(repr), true) = (repr, r.get(side).is_some()) {
                cache[facts_key(repr).0] = json!({"at": epoch(), "facts": r[side]});
                *slot = Some(r[side].clone());
            }
        }
        let _ = write_json(&facts_cache_file(), &cache);
    }
    let s = s.ok_or_else(|| Error::new("The worker did not describe the model."))?;
    if let Some(t) = t.as_mut() {
        attach_pair_probe(&s, t, &mut cache, kill, on_event);
    }
    Ok((s, t))
}

/// When the two tokenizers' ids hashes differ (and the vocabularies are not known to differ in size), the worker compares what they do with a fixed set of texts
/// (`tokpair.decide_pair`); its answer is put on the teacher's facts as `pair_probe` and kept per pair. Without an answer the verdict refuses, as before.
fn attach_pair_probe(s: &Value, t: &mut Value, cache: &mut Value, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) {
    if !s["error"].is_null() || !t["error"].is_null() {
        return;
    }
    let (Some(sh), Some(th)) = (s["ids_hash"].as_str(), t["ids_hash"].as_str()) else { return };
    let (Some(sd), Some(td)) = (s["tokenizer_dir"].as_str(), t["tokenizer_dir"].as_str()) else { return };
    if sh == th || matches!((s["vocab_tokens"].as_u64(), t["vocab_tokens"].as_u64()), (Some(a), Some(b)) if a != b) {
        return;
    }
    let key = format!("pair|{sh}|{th}");
    let kept = cache.get(&key).filter(|e| e["probe"]["level"].is_string()).map(|e| e["probe"].clone());
    let probe = kept.or_else(|| {
        let py = python().ok()?;
        let r = run_job(&py, &worker_dir(), &json!({"job": "heal_check", "pair": {"student": sd, "teacher": td}}), kill, &mut |e| on_event(event_json(e))).ok()?;
        let probe = r.get("pair").filter(|p| p["level"].is_string())?.clone();
        // an answer that only says "could not run" is not kept: the next Check tries again
        if matches!(probe["probe"]["status"].as_str(), Some("passed") | Some("different")) {
            cache[&key] = json!({"at": epoch(), "probe": probe});
            let _ = write_json(&facts_cache_file(), cache);
        }
        Some(probe)
    });
    if let Some(p) = probe {
        t["pair_probe"] = p;
    }
}

fn gb(bytes: f64) -> f64 {
    (bytes / 1e8).round() / 10.0
}

fn b(params: &Value) -> String {
    match params.as_f64() {
        Some(x) if x >= 1e8 => format!("{:.2} billion", x / 1e9),
        Some(x) => format!("{:.0} million", x / 1e6),
        None => "an unknown number of".into(),
    }
}

fn n(v: &Value) -> String {
    v.as_u64().map(|x| crate::reshape::runs::thousands(x)).unwrap_or_else(|| "?".into())
}

/// What the person's computer offers the comparison.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ctx {
    pub vram_gb: Option<f64>,
    pub free_disk_gb: Option<f64>,
}

fn item(level: &str, text: impl Into<String>) -> Value {
    json!({"level": level, "text": text.into()})
}

/// The checks of one model that does not depend on the other. `who`: "Model to heal" / "Original (teacher)".
fn side_items(who: &str, f: &Value, out: &mut Vec<Value>) -> bool {
    if let Some(e) = f["error"].as_str() {
        out.push(item("refuse", format!("{who}: {e}")));
        return false;
    }
    let mut ok = true;
    if f["gguf"] == true {
        out.push(item("refuse", format!("{who}: this is a GGUF file. Healing needs the full-precision .safetensors model (as the Slim page does).")));
        return false;
    }
    if f["quantized"] == true {
        out.push(item("refuse", format!("{who}: this model is stored in a compressed (quantized) form. Healing needs the full-precision .safetensors model (as the Slim page does).")));
        ok = false;
    } else if f["supported"] == false {
        out.push(item("refuse", format!("{who}: {}", f["why"].as_str().unwrap_or("this kind of model is not supported yet."))));
        ok = false;
    }
    if ok {
        let mlp = f["intermediate"].as_u64().map(|i| format!(", MLP width {}", crate::reshape::runs::thousands(i))).unwrap_or_default();
        out.push(item("ok", format!("{who}: {} parameters, {} layers, hidden size {}{mlp}, family {}.", b(&f["params"]), n(&f["layers"]), n(&f["hidden"]), f["model_type"].as_str().unwrap_or("unknown"))));
    }
    ok
}

/// The comparison of the two models in plain words: every line is {"level": "ok" | "warn" | "refuse" | "note", "text"}. `ok` is true when no line refuses.
/// `student_was_made`: made on the Slim page (nothing to say about its licence or an unmodified model). `teacher` None = not chosen yet.
pub fn verdict(student: &Value, teacher: Option<&Value>, ctx: &Ctx) -> Value {
    let mut items = Vec::new();
    let s_ok = side_items("Model to heal", student, &mut items);
    let mut t_ok = false;
    match teacher {
        None => items.push(item("refuse", "Choose the original model (the teacher): the model that the one you heal was made from.")),
        Some(t) => {
            t_ok = side_items("Original (teacher)", t, &mut items);
            if student["error"].is_null() && t["error"].is_null() {
                tokenizer_items(student, t, &mut items);
                if let (Some(sp), Some(tp)) = (student["params"].as_f64(), t["params"].as_f64()) {
                    if tp < sp {
                        items.push(item("warn", format!("The teacher ({}) is smaller than the model you heal ({}). It can only teach what it knows; healing usually uses a bigger original.", b(&t["params"]), b(&student["params"]))));
                    }
                }
                if let (Some(sh), Some(th)) = (student["hidden"].as_u64(), t["hidden"].as_u64()) {
                    if sh != th {
                        items.push(item("note", format!("The hidden sizes differ ({sh} and {th}). That is fine: healing compares the two models' output probabilities only.")));
                    }
                }
            }
        }
    }
    let teacher_gb = teacher.and_then(|t| t["params"].as_f64()).map(|p| (p / 1e9 * 0.6 * 10.0).round() / 10.0);
    if let (Some(need), Some(card)) = (teacher_gb, ctx.vram_gb) {
        if need + 2.0 > card {
            items.push(item("warn", format!("The teacher needs about {need:.1} GB of graphics memory in 4-bit; this card has {card:.0} GB. Healing may not fit; the estimate below says so.")));
        } else {
            items.push(item("ok", format!("The teacher needs about {need:.1} GB in 4-bit; this card has {card:.0} GB.")));
        }
    }
    // what has to be downloaded, and the room for it and for the healed copy
    let dl: f64 = [Some(student), teacher].iter().flatten().map(|f| f["download_bytes"].as_f64().unwrap_or(0.0)).sum();
    let copy = student["bytes"].as_f64().unwrap_or(0.0);
    for (who, f) in [("Model to heal", Some(student)), ("Original (teacher)", teacher)] {
        let Some(f) = f.filter(|f| f["error"].is_null() && f["kind"] == "hub") else { continue };
        if f["on_disk"] == true {
            items.push(item("ok", format!("{who}: already on this computer (Hugging Face cache).")));
        } else {
            items.push(item("warn", format!("{who}: not on this computer. Starting downloads {:.1} GB from Hugging Face.", gb(f["download_bytes"].as_f64().unwrap_or(0.0)))));
        }
    }
    let need = dl + copy;
    if need > 0.0 {
        match ctx.free_disk_gb {
            Some(free) if free * 1e9 < dl * 1.05 => items.push(item("refuse", format!("There is not enough free disk space: {:.1} GB to download, {free:.1} GB free.", gb(dl)))),
            Some(free) if free * 1e9 < need * 1.1 => items.push(item("warn", format!("Disk space is tight: the download ({:.1} GB) and the healed copy ({:.1} GB) need about {:.1} GB; {free:.1} GB free.", gb(dl), gb(copy), gb(need)))),
            Some(free) => items.push(item("ok", format!("Disk space: about {:.1} GB needed for the healed copy and downloads; {free:.1} GB free.", gb(need)))),
            None => {}
        }
    }
    let lic = |f: &Value| f["license"].as_str().map(String::from);
    if let (Some(s), true) = (lic(student).or_else(|| Some("not stated".into())), s_ok) {
        let t = teacher.and_then(lic).unwrap_or_else(|| "not stated".into());
        items.push(item("note", format!("Licences: the model to heal: {s}; the original: {t}. The healed copy is a derivative of both; check that their licences allow what you plan to do.")));
    }
    let ok = s_ok && t_ok && !items.iter().any(|i| i["level"] == "refuse");
    json!({"ok": ok, "items": items, "teacher_4bit_gb": teacher_gb, "download_gb": gb(dl)})
}

fn tokenizer_items(student: &Value, t: &Value, out: &mut Vec<Value>) {
    // `ids_hash` (what decides token ids) is what the pair is compared by; facts of an older worker only have the strict `tokenizer_hash`
    let hash = |f: &Value| f["ids_hash"].as_str().or_else(|| f["tokenizer_hash"].as_str()).map(String::from);
    let (sh, th) = (hash(student), hash(t));
    let (sv, tv) = (student["vocab_tokens"].as_u64(), t["vocab_tokens"].as_u64());
    match (sh, th) {
        (Some(a), Some(b)) if a == b && sv == tv => out.push(item("ok", format!("The tokenizer is identical{}.", sv.map(|v| format!(" ({} tokens)", crate::reshape::runs::thousands(v))).unwrap_or_default()))),
        (Some(_), Some(_)) => {
            if let Some((a, b)) = sv.zip(tv).filter(|(a, b)| a != b) {
                out.push(item("refuse", format!("The two models do not use the same tokenizer, so the teacher's answers cannot teach this model. The vocabularies have {} and {} tokens. Choose the model this one was made from.", crate::reshape::runs::thousands(a), crate::reshape::runs::thousands(b))));
            } else if let Some(p) = t["pair_probe"].as_object().filter(|p| p.get("level").is_some_and(|l| l == "warn" || l == "refuse")) {
                let level = p["level"].as_str().unwrap_or("refuse");
                out.push(item(level, p.get("text").and_then(|x| x.as_str()).unwrap_or("The tokenizers could not be compared.")));
            } else {
                out.push(item("refuse", "The two models do not use the same tokenizer, so the teacher's answers cannot teach this model. The vocabularies have the same size, but the tokenizer files differ. Choose the model this one was made from."));
            }
        }
        _ => out.push(item("refuse", "A tokenizer's files are missing in one of the models, so the program cannot tell whether they use the same tokenizer. Healing needs them (tokenizer.json or vocab.json).")),
    }
    // a chat template that differs never refuses: the healing texts are written with the student's own template and the teacher reads the same ids
    if student["template_hash"] != t["template_hash"] && (student["template_hash"].is_string() || t["template_hash"].is_string()) {
        out.push(item("warn", "The two models use different chat templates. That is fine for healing: the texts are written with the template of the model you heal, and the original then reads exactly the same token ids, so the original's own template is not used. The healed copy keeps the template of the model you heal."));
    }
    if let (Some(a), Some(b)) = (student["vocab_size"].as_u64(), t["vocab_size"].as_u64()) {
        if a != b {
            out.push(item("warn", format!("The models' output layers have different sizes ({} and {}). The tokenizer is the same, so the extra entries are padding; the program only uses the real tokens.", crate::reshape::runs::thousands(a), crate::reshape::runs::thousands(b))));
        }
    }
}

/// True for a model that ships with the program (an unmodified catalog model): healing it only makes sense when it was changed.
fn is_builtin(repr: &str) -> bool {
    catalog::builtin_models().iter().any(|m| m.repo.eq_ignore_ascii_case(repr) || m.official.eq_ignore_ascii_case(repr))
}

/// The Check of the Heal section: the file-level facts of both models and what they mean together. `teacher` None = the recorded original for a made student
/// (an error when there is none), nothing yet for another student (the answer says to choose one and offers a guess).
pub fn heal_check(p: &Project, student: &Source, teacher: Option<&Source>, ctx: Ctx, refresh: bool, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    student.validate()?;
    if let Some(t) = teacher {
        t.validate()?;
        if t.is_made() {
            return Err(Error::new("The teacher must be a model folder or a Hugging Face model."));
        }
    }
    let srepr = student.repr(p);
    let mut recorded = None;
    let trepr: Option<String> = match (teacher, student) {
        (Some(t), _) => Some(t.repr(p)),
        (None, Source::Made { name }) => {
            let o = read_origin(&models_dir(p).join(name))?;
            recorded = Some(o.teacher.clone());
            Some(o.teacher)
        }
        (None, _) => None,
    };
    let (sf, tf) = facts(&srepr, trepr.as_deref(), refresh, kill, on_event)?;
    let mut v = verdict(&sf, tf.as_ref(), &ctx);
    let mut extra = Vec::new();
    if !student.is_made() && sf["error"].is_null() && is_builtin(&srepr) {
        extra.push(item("warn", "This is an unmodified model that ships with the program. Healing only makes sense for a model that was changed (thinned, pruned or otherwise cut down); healing a whole one changes little."));
    }
    if trepr.as_deref() == Some(srepr.as_str()) {
        extra.push(item("refuse", "The model to heal and the original are the same model. A model cannot learn from itself."));
    }
    if !extra.is_empty() {
        let items = v["items"].as_array_mut().unwrap();
        for e in extra.into_iter().rev() {
            items.insert(0, e);
        }
        v["ok"] = json!(v["ok"] == true && !v["items"].as_array().unwrap().iter().any(|i| i["level"] == "refuse"));
    }
    // a guess at the original (never chosen for the person): only when the model to heal names a base model
    let guess = (teacher.is_none() && recorded.is_none()).then(|| sf["base_guess"].clone()).filter(|g| !g.is_null());
    v["student"] = sf;
    v["teacher"] = tf.unwrap_or(Value::Null);
    v["teacher_guess"] = guess.unwrap_or(Value::Null);
    v["teacher_recorded"] = json!(recorded);
    v["free_disk_gb"] = json!(ctx.free_disk_gb.map(|g| (g * 10.0).round() / 10.0));
    v["vram_gb"] = json!(ctx.vram_gb);
    Ok(v)
}

// -------------------------------------------------------------------------------------------------------------------- the lists

/// What the pickers of the Heal section offer: the models of the Slim page ("made"), the entries of the model list that are folders or Hugging Face ids
/// ("models", custom ones and the ones that ship with the program) and the heals of other models that were started before ("runs").
pub fn heal_candidates(p: &Project) -> Value {
    let made_models: Vec<Value> = super::runs::list(p).into_iter().filter(|m| m["has_model"] == true && (m["summary"]["mode"] == "prune" || (m["summary"].is_null() && m["spec"]["mode"] == "prune")) && m["status"] == "done").map(|m| {
        json!({"name": m["name"], "healed": m["healed"], "heal_status": m["heal"]["status"], "size_gb": m["size_gb"], "model_id": m["model_id"], "healed_model_id": m["healed_model_id"]})
    }).collect();
    let mine = models_dir(p);
    let own_folders: Vec<String> = made_models.iter().filter_map(|m| m["name"].as_str()).map(|n| mine.join(n).join("model").canonicalize().unwrap_or_else(|_| mine.join(n).join("model")).to_string_lossy().to_string()).collect();
    let (mut custom, mut builtin) = (Vec::new(), Vec::new());
    for m in catalog::models() {
        if own_folders.contains(&m.repo) {
            continue;
        }
        let source = if Path::new(&m.repo).is_dir() { json!({"kind": "folder", "path": m.repo}) } else if is_hub_id(&m.repo) { json!({"kind": "hub", "id": m.repo}) } else { continue };
        let e = json!({"id": m.id, "name": m.name, "repo": m.repo, "source": source, "params_b": m.params_b, "custom": m.custom, "derived_from": m.derived_from, "reshape": m.reshape});
        if m.custom { custom.push(e) } else { builtin.push(e) }
    }
    let mut runs = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&mine) {
        for e in rd.flatten() {
            let dir = e.path();
            let Some(h) = read(&dir).filter(|h| !h.student.is_made()) else { continue };
            let heal = read_json(&dir.join("heal.json"));
            runs.push(json!({"name": e.file_name().to_string_lossy(), "student": h.student, "teacher": h.teacher, "student_name": h.student_name, "teacher_name": h.teacher_name,
                             "created": h.created, "healed": dir.join("healed").join("config.json").is_file(), "heal": heal, "model_id": heal.as_ref().and_then(|x| x["result"]["model_id"].as_str().map(String::from))}));
        }
    }
    runs.sort_by(|a, b| b["created"].as_str().cmp(&a["created"].as_str()));
    json!({"made": made_models, "models": {"mine": custom, "builtin": builtin}, "runs": runs})
}

/// The student and teacher of a pair that has not been started (no folder yet), from the Check's facts. A made student keeps what its folder records, with the
/// teacher chosen on top.
pub fn origin_from_check(p: &Project, student: &Source, teacher: Option<&Source>, check: &Value) -> Result<Origin> {
    if let Source::Made { name } = student {
        let mut o = read_origin(&models_dir(p).join(name))?;
        if let Some(t) = teacher.filter(|t| t.repr(p) != o.teacher) {
            o.teacher = t.repr(p);
            o.teacher_b = billions(&check["teacher"]["params"]).unwrap_or(o.student_b * 1.3);
            o.family = family_of(&check["teacher"]["family"]);
            o.teacher_chosen = true;
        }
        return Ok(o);
    }
    let t = teacher.ok_or_else(|| Error::new("Choose the original model (the teacher)."))?;
    let sf = &check["student"];
    let student_b = billions(&sf["params"]).unwrap_or(0.0);
    Ok(Origin {
        teacher: t.repr(p), family: family_of(&sf["family"]), student_b, teacher_b: billions(&check["teacher"]["params"]).unwrap_or(student_b * 1.3), config: config_from_facts(sf),
        calibration: Value::Null, loss_before: Value::Null, loss_after: Value::Null, external: true, teacher_chosen: true, student_label: student.label(),
    })
}

/// The folder of an earlier heal of this pair: the one that is not finished first (so Continue goes on where it was), else a finished one (its healed model is shown).
pub fn find_run(p: &Project, student: &Source, teacher: Option<&Source>) -> Option<String> {
    if let Source::Made { name } = student {
        return Some(name.clone());
    }
    let mut finished = None;
    for e in std::fs::read_dir(models_dir(p)).ok()?.flatten() {
        let dir = e.path();
        let Some(h) = read(&dir) else { continue };
        if &h.student == student && h.teacher.as_ref() == teacher {
            if dir.join("healed").join("config.json").is_file() {
                finished = Some(e.file_name().to_string_lossy().to_string());
            } else {
                return Some(e.file_name().to_string_lossy().to_string());
            }
        }
    }
    finished
}

/// What the Heal section shows after "Check": the comparison of the pair and, when it passes, the Heal card's options for it (`options` is null when it does not).
pub fn heal_pair(p: &Project, student: &Source, teacher: Option<&Source>, vram_gb: Option<f64>, refresh: bool, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let ctx = Ctx { vram_gb, free_disk_gb: crate::envmgr::health::free_disk_gb(&models_dir(p)) };
    let check = heal_check(p, student, teacher, ctx, refresh, kill, on_event)?;
    if check["ok"] != true {
        return Ok(json!({"check": check, "options": null, "name": null}));
    }
    let run = find_run(p, student, teacher);
    let mut o = origin_from_check(p, student, teacher, &check)?;
    // a finished or stopped heal of this pair keeps its own record of the models (the sizes it was started with)
    let dir = run.as_ref().map(|n| models_dir(p).join(n));
    if let Some(d) = dir.as_ref().filter(|d| is_external(d)) {
        if let Ok(rec) = read_origin(d) {
            o = rec;
        }
    }
    let opts = super::runs::heal_options_with(p, run.as_deref().zip(dir.as_deref()), &o, vram_gb);
    Ok(json!({"check": check, "options": opts, "name": run}))
}

/// The live estimate of the section for the settings chosen so far (the Check's answer is kept, so this does not ask the worker again).
pub fn heal_pair_estimate(p: &Project, student: &Source, teacher: Option<&Source>, spec: &super::runs::HealSpec, vram_gb: Option<f64>) -> Result<Value> {
    spec.validate()?;
    let ctx = Ctx { vram_gb, free_disk_gb: crate::envmgr::health::free_disk_gb(&models_dir(p)) };
    let check = heal_check(p, student, teacher, ctx, false, &AtomicBool::new(false), &mut |_| {})?;
    if check["ok"] != true {
        return Err(Error::new("The two models did not pass the check."));
    }
    let o = origin_from_check(p, student, teacher, &check)?;
    Ok(super::runs::heal_estimate_with(&o, spec, vram_gb))
}

/// Makes (or finds again) the folder of a heal and returns its name. A made model keeps its folder (a `heal_source.json` only when another teacher was chosen);
/// any other student gets `models/heal-<time>-<name>/`, or the folder of an earlier heal of the same pair that is not finished (so Continue goes on where it was).
/// Nothing but this small file is written. The pair must have passed the Check (`check` is its answer).
pub fn heal_prepare(p: &Project, student: &Source, teacher: Option<&Source>, check: &Value) -> Result<String> {
    student.validate()?;
    if check["ok"] != true {
        return Err(Error::new("The two models did not pass the check, so healing does not start."));
    }
    let mine = models_dir(p);
    let keep = json!({"student": check["student"], "teacher": check["teacher"]});
    if let Source::Made { name } = student {
        let dir = mine.join(name);
        let o = read_origin(&dir)?;
        // the recorded original (or no choice): nothing to write, and an earlier choice of another teacher is dropped unless that heal already happened
        let recorded = teacher.is_none_or(|t| t.repr(p) == o.teacher && !o.teacher_chosen);
        if recorded {
            if o.teacher_chosen && !dir.join("healed").join("config.json").is_file() {
                let _ = std::fs::remove_file(file(&dir));
            }
            return Ok(name.clone());
        }
        let t = teacher.unwrap();
        write(&dir, &HealSource { student: student.clone(), teacher: Some(t.clone()), created: now(), student_name: name.clone(), teacher_name: t.label(), student_dir: None, check: keep })?;
        return Ok(name.clone());
    }
    let teacher = teacher.ok_or_else(|| Error::new("Choose the original model (the teacher)."))?;
    if let Ok(rd) = std::fs::read_dir(&mine) {
        for e in rd.flatten() {
            let dir = e.path();
            if let Some(h) = read(&dir) {
                if &h.student == student && h.teacher.as_ref() == Some(teacher) && !dir.join("healed").join("config.json").is_file() {
                    write(&dir, &HealSource { check: keep, ..h })?;
                    return Ok(e.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    let name = format!("heal-{}-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"), slug(&student.label()));
    let base = name.trim_end_matches('-').to_string();
    let (mut name, mut n) = (base.clone(), 1);
    while mine.join(&name).exists() {
        n += 1;
        name = format!("{base}-{n}");
    }
    write(&mine.join(&name), &HealSource { student: student.clone(), teacher: Some(teacher.clone()), created: now(), student_name: student.label(), teacher_name: teacher.label(), student_dir: None, check: keep })?;
    Ok(name)
}

/// Downloads a Hugging Face student into the Hugging Face cache (the worker's `_resolve_source`) and remembers where it is. Does nothing for a folder.
pub fn fetch_student(dir: &Path, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<PathBuf> {
    if let Some(f) = student_folder(dir)? {
        return Ok(f);
    }
    let mut h = read(dir).ok_or_else(|| Error::new("This heal has no model to download."))?;
    let Source::Hub { id } = &h.student else { return Err(Error::new("This model is not there.")) };
    let r = run_job(&python()?, &worker_dir(), &json!({"job": "heal_check", "sources": {"student": id}, "fetch": true}), kill, &mut |e| {
        let mut ev = event_json(e);
        ev["phase"] = json!("heal");
        ev["heal_step"] = json!("download");
        on_event(ev)
    })?;
    let f = r["student"]["folder"].as_str().map(PathBuf::from).filter(|f| f.join("config.json").is_file()).ok_or_else(|| Error::new(r["student"]["error"].as_str().unwrap_or("The model could not be downloaded.").to_string()))?;
    h.student_dir = Some(f.to_string_lossy().to_string());
    write(dir, &h)?;
    Ok(f)
}

/// Checks the pair, makes the heal's folder and heals: the one call behind the section's Start. Returns the heal's result with the folder's `name` in it.
#[allow(clippy::too_many_arguments)]
pub fn heal_from_source(p: &Project, student: &Source, teacher: Option<&Source>, spec: &super::runs::HealSpec, vram_gb: Option<f64>, stop: std::sync::Arc<AtomicBool>, grace: std::time::Duration, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let ctx = Ctx { vram_gb, free_disk_gb: crate::envmgr::health::free_disk_gb(&models_dir(p)) };
    let check = heal_check(p, student, teacher, ctx, false, &stop, on_event)?;
    let name = heal_prepare(p, student, teacher, &check)?;
    super::runs::heal(p, &name, spec, vram_gb, stop, grace, on_event)
}
