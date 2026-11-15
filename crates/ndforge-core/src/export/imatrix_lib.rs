//! The matrix library (docs/IMATRIX.md, "The matrix library"): importance matrices that are made on their own, without exporting a GGUF, kept in the project
//! (`imatrix/<id>/`: `imatrix.gguf`, an optional legacy `imatrix.dat`, `imatrix.json`), reused by later exports and copied out to be used elsewhere.
//!
//! The matrices that past exports saved next to their GGUFs (`exports/<run>-<size>/imatrix.gguf`) are not moved or changed: "Add to library" copies one in.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::gguf_meta;
use super::runs::{exportable_runs, exports_dir, hf_cache_snapshots, imatrix_estimate, read_json, run_export_job, write_json, IMATRIX_CHUNKS, IMATRIX_CTX, IMATRIX_DEFAULT_CHUNKS, IMATRIX_GENERAL_SHARE,
                  IMATRIX_STOP_GRACE};
use crate::envmgr::jobs::WorkerEvent;
use crate::envmgr::{env_dir, python_path};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::runs::runs_dir;

/// The library folder of a project. Old projects have none: it is made with the first matrix.
pub fn lib_dir(p: &Project) -> PathBuf {
    p.folder.join("imatrix")
}

/// What a person calls a matrix is at most this long.
const LABEL_MAX: usize = 80;

// ------------------------------------------------------------------------------------------------------------------ who is using a matrix

static IN_USE: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// A matrix of the library that a job uses right now (an export that was given it, the making of its .dat): it cannot be deleted while this guard lives.
pub struct InUse(String);

impl Drop for InUse {
    fn drop(&mut self) {
        let mut v = IN_USE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = v.iter().position(|x| *x == self.0) {
            v.remove(i);
        }
    }
}

pub fn mark_in_use(id: &str) -> InUse {
    IN_USE.lock().unwrap_or_else(|e| e.into_inner()).push(id.to_string());
    InUse(id.to_string())
}

pub fn in_use(id: &str) -> bool {
    IN_USE.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|x| x == id)
}

// ------------------------------------------------------------------------------------------------------------------ the entries

/// Ids come from the window: no path tricks.
pub fn valid_id(id: &str) -> Result<()> {
    if id.is_empty() || id.starts_with(['_', '.']) || id.contains(['/', '\\']) || id.contains("..") {
        return Err(Error::new("That is not the name of a saved matrix."));
    }
    Ok(())
}

/// The file a library entry gives to llama-quantize: the GGUF one, else the legacy .dat (an imported matrix may be only that).
pub fn matrix_file(dir: &Path) -> Option<PathBuf> {
    ["imatrix.gguf", "imatrix.dat"].iter().map(|n| dir.join(n)).find(|f| f.is_file())
}

fn mb(bytes: u64) -> f64 {
    (bytes as f64 / 1048576.0 * 100.0).round() / 100.0
}

fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// One entry as the window shows it: what imatrix.json says plus `id`, `dir`, `file`, `has_gguf`, `has_dat`, `size_mb` and `in_use`. A folder without any matrix file is not an entry.
pub fn entry(dir: &Path) -> Option<Value> {
    let file = matrix_file(dir)?;
    let id = dir.file_name()?.to_str()?.to_string();
    let mut v = match read_json(&dir.join("imatrix.json")) {
        Some(v) if v.is_object() => v,
        _ => json!({}),
    };
    let (has_gguf, has_dat) = (dir.join("imatrix.gguf").is_file(), dir.join("imatrix.dat").is_file());
    v["size_mb"] = json!(mb(file_len(&file)));
    v["in_use"] = json!(in_use(&id));
    v["has_gguf"] = json!(has_gguf);
    v["has_dat"] = json!(has_dat);
    v["file"] = json!(file);
    v["dir"] = json!(dir);
    if v["label"].is_null() {
        v["label"] = json!("");
    }
    v["created"] = if v["created"].is_string() { v["created"].clone() } else { json!("") };
    v["id"] = json!(id);
    Some(v)
}

/// Every matrix of the library, newest first.
pub fn list(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(lib_dir(p)).map(|d| d.flatten().filter(|e| e.path().is_dir()).filter_map(|e| entry(&e.path())).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["id"].as_str().cmp(&a["id"].as_str()));
    out
}

pub fn get(p: &Project, id: &str) -> Result<(PathBuf, Value)> {
    valid_id(id)?;
    let dir = lib_dir(p).join(id);
    let e = entry(&dir).ok_or_else(|| Error::new("That saved matrix no longer exists."))?;
    Ok((dir, e))
}

/// Short, plain characters for a folder name.
pub fn short_name(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars().flat_map(|c| c.to_lowercase()) {
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let out: String = out.trim_matches('-').chars().take(24).collect();
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "matrix".to_string() } else { out }
}

/// `yyyymmdd-hhmm-<short model name>`, a number added when the name is taken (two matrices in one minute).
pub fn new_id(p: &Project, name: &str) -> String {
    id_at(p, &chrono::Local::now().format("%Y%m%d-%H%M").to_string(), name)
}

pub fn id_at(p: &Project, stamp: &str, name: &str) -> String {
    let base = format!("{stamp}-{}", short_name(name));
    let mut id = base.clone();
    let mut n = 2;
    while lib_dir(p).join(&id).exists() {
        id = format!("{base}-{n}");
        n += 1;
    }
    id
}

fn tidy_label(label: &str) -> String {
    label.trim().chars().filter(|c| !c.is_control()).take(LABEL_MAX).collect()
}

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

// ------------------------------------------------------------------------------------------------------------------ disk space

/// Free bytes on the disk that holds `dir` (or its nearest existing parent); None when it cannot be asked.
pub fn free_bytes(dir: &Path) -> Option<u64> {
    let mut d = dir;
    while !d.exists() {
        d = d.parent()?;
    }
    fs2::available_space(d).ok()
}

fn check_space(dest: &Path, need: u64, free: &dyn Fn(&Path) -> Option<u64>) -> Result<()> {
    if let Some(f) = free(dest) {
        if f < need + need / 20 + 1 {
            return Err(Error::new(format!("There is not enough free space in {}: {:.1} MB are needed and {:.1} MB are free. Free some space or choose another folder.", dest.display(), need as f64 / 1048576.0, f as f64 / 1048576.0)));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------ import, copy, delete

fn file_kind(path: &Path) -> Result<&'static str> {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("gguf") => Ok("gguf"),
        Some("dat") => Ok("dat"),
        _ => Err(Error::new("Choose an importance matrix file: imatrix.gguf (the current format) or a legacy .dat file.")),
    }
}

fn start_entry(p: &Project, name: &str) -> Result<(String, PathBuf)> {
    let id = new_id(p, name);
    let dir = lib_dir(p).join(&id);
    std::fs::create_dir_all(&dir)?;
    Ok((id, dir))
}

/// "Add a matrix file from my computer": copies a .gguf or .dat matrix (made elsewhere, or a copy from the earlier Save a copy) into the library with a label.
/// Returns the entry and, for a file that does not look like a matrix, a warning (it is still added: llama-quantize decides).
pub fn import_file(p: &Project, file: &Path, label: &str, free: &dyn Fn(&Path) -> Option<u64>) -> Result<Value> {
    let kind = file_kind(file)?;
    if !file.is_file() {
        return Err(Error::new(format!("The file {} was not found.", file.display())));
    }
    let size = file_len(file);
    if size == 0 {
        return Err(Error::new(format!("The file {} is empty.", file.display())));
    }
    std::fs::create_dir_all(lib_dir(p))?;
    check_space(&lib_dir(p), size, free)?;
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("matrix");
    let (id, dir) = start_entry(p, stem)?;
    let name = if kind == "dat" { "imatrix.dat" } else { "imatrix.gguf" };
    let done = (|| -> Result<Value> {
        std::fs::copy(file, dir.join(name))?;
        let summary = if kind == "gguf" { gguf_meta::matrix_summary(&dir.join(name)) } else { None };
        let warning = match (&summary, kind) {
            (None, "gguf") => Some("This file could not be read as a GGUF matrix here; it was added anyway. llama-quantize will tell if it does not fit."),
            (None, _) => Some("A legacy .dat matrix cannot be looked into here; it was added as it is."),
            _ => None,
        };
        let info = json!({"format": 1, "id": id, "label": tidy_label(label), "model": "", "source": format!("a matrix file from this computer ({})", file.file_name().and_then(|n| n.to_str()).unwrap_or("")),
                          "source_kind": "imported", "via": "imported", "created": now(), "size": size, "has_dat": kind == "dat", "chunks": summary.as_ref().and_then(|s| s.chunks),
                          "arch": summary.as_ref().and_then(|s| s.arch.clone()), "layers": summary.as_ref().and_then(|s| s.layers), "entries": summary.as_ref().map(|s| s.entries), "nextn": null,
                          "general_used": null, "notes": []});
        write_json(&dir.join("imatrix.json"), &info)?;
        let mut e = entry(&dir).ok_or_else(|| Error::new("The matrix could not be added."))?;
        e["warning"] = json!(warning);
        Ok(e)
    })();
    if done.is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    done
}

/// "Add to library" for a matrix that an earlier export saved next to its GGUF (`exports/<export>/imatrix.gguf` + imatrix.json): copied in, the export is not touched.
pub fn import_export(p: &Project, export: &str, label: &str, free: &dyn Fn(&Path) -> Option<u64>) -> Result<Value> {
    if export.contains(['/', '\\']) || export.contains("..") || export.is_empty() {
        return Err(Error::new("Unknown export."));
    }
    let from = exports_dir(p).join(export);
    let Some(src) = matrix_file(&from) else { return Err(Error::new("That export has no saved importance matrix.")) };
    std::fs::create_dir_all(lib_dir(p))?;
    check_space(&lib_dir(p), file_len(&src), free)?;
    let (id, dir) = start_entry(p, export)?;
    let done = (|| -> Result<Value> {
        let name = src.file_name().and_then(|n| n.to_str()).unwrap_or("imatrix.gguf").to_string();
        std::fs::copy(&src, dir.join(&name))?;
        let mut info = read_json(&from.join("imatrix.json")).filter(|v| v.is_object()).unwrap_or_else(|| json!({}));
        let meta = read_json(&from.join("meta.json")).unwrap_or(Value::Null);
        let summary = gguf_meta::matrix_summary(&dir.join(&name));
        info["id"] = json!(id);
        info["label"] = json!(tidy_label(label));
        info["source"] = json!(format!("the matrix saved next to the export {export}"));
        info["source_kind"] = json!("export");
        info["via"] = if meta["imatrix"]["via"].is_string() { meta["imatrix"]["via"].clone() } else { json!("llama.cpp") };
        info["size"] = json!(file_len(&src));
        info["has_dat"] = json!(name.ends_with(".dat"));
        info["copied_from"] = json!(export);
        for (k, v) in [("arch", summary.as_ref().and_then(|s| s.arch.clone()).map(|a| json!(a))), ("layers", summary.as_ref().and_then(|s| s.layers).map(|l| json!(l))), ("entries", summary.as_ref().map(|s| json!(s.entries)))] {
            if let Some(v) = v {
                info[k] = v;
            }
        }
        if !info["created"].is_string() {
            info["created"] = json!(now());
        }
        write_json(&dir.join("imatrix.json"), &info)?;
        entry(&dir).ok_or_else(|| Error::new("The matrix could not be added."))
    })();
    if done.is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    done
}

/// Exports that have a matrix next to their GGUF (that can be added to the library), newest first.
pub fn exports_with_matrix(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(exports_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let f = matrix_file(&e.path())?;
        let info = read_json(&e.path().join("imatrix.json")).unwrap_or(Value::Null);
        let name = e.file_name().to_string_lossy().to_string();
        let added = list(p).iter().any(|x| x["copied_from"] == name.as_str());
        Some(json!({"name": name, "model": info["model"], "chunks": info["chunks"], "created": info["created"], "size_mb": mb(file_len(&f)), "added": added}))
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

fn unique_dir(dest: &Path, name: &str) -> PathBuf {
    let mut to = dest.join(name);
    let mut n = 2;
    while to.exists() {
        to = dest.join(format!("{name}-{n}"));
        n += 1;
    }
    to
}

/// "Save a copy...": `imatrix.gguf` (and `imatrix.dat` when there is one) plus `imatrix.json` as `imatrix-info.json`, in a new folder `<dest>/<id>` (numbered when that name
/// is taken: nothing is ever overwritten). The free space is checked first (`free` is the disk question, so that tests can answer it). Returns the new folder.
pub fn save_copy(p: &Project, id: &str, dest: &Path, free: &dyn Fn(&Path) -> Option<u64>) -> Result<PathBuf> {
    let (dir, _) = get(p, id)?;
    let files: Vec<(PathBuf, &str)> = [("imatrix.gguf", "imatrix.gguf"), ("imatrix.dat", "imatrix.dat"), ("imatrix.json", "imatrix-info.json")].iter()
        .map(|(a, b)| (dir.join(a), *b)).filter(|(f, _)| f.is_file()).collect();
    if dest.as_os_str().is_empty() {
        return Err(Error::new("Choose the folder to save the copy in."));
    }
    if dest.exists() && !dest.is_dir() {
        return Err(Error::new(format!("{} is a file, not a folder. Choose a folder.", dest.display())));
    }
    let need: u64 = files.iter().map(|(f, _)| file_len(f)).sum();
    check_space(dest, need, free)?;
    std::fs::create_dir_all(dest).map_err(|e| Error::new(format!("The folder {} could not be made ({e}). Choose another folder.", dest.display())))?;
    let to = unique_dir(dest, id);
    std::fs::create_dir(&to)?;
    for (f, name) in &files {
        if let Err(e) = std::fs::copy(f, to.join(name)) {
            let _ = std::fs::remove_dir_all(&to);
            return Err(Error::new(format!("The copy failed ({e}); nothing was left behind in {}.", dest.display())));
        }
    }
    Ok(to)
}

/// Deletes an entry. Refused while a job uses it.
pub fn delete(p: &Project, id: &str) -> Result<()> {
    let (dir, _) = get(p, id)?;
    if in_use(id) {
        return Err(Error::new("This matrix is being used by a job right now (an export or the making of its .dat file). Wait for it to end, then delete it."));
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

/// "Open folder": the library folder of an entry, or a folder a copy was saved in. There is no opener plug-in in this program, so the system's own tool is started.
pub fn open_folder(p: &Project, id: Option<&str>, path: Option<&str>) -> Result<()> {
    let dir = match (id, path) {
        (Some(id), _) => get(p, id)?.0,
        (None, Some(path)) if Path::new(path).is_dir() => PathBuf::from(path),
        _ => return Err(Error::new("That folder does not exist.")),
    };
    let tool = if cfg!(windows) { "explorer" } else if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    std::process::Command::new(tool).arg(&dir).spawn().map(|_| ())
        .map_err(|_| Error::new(format!("The folder could not be opened from here. It is: {}", dir.display())))
}

// ------------------------------------------------------------------------------------------------------------------ does a matrix fit the model?

/// What the model of a run is, from its files: the full model's folder, else its base model (a folder, or in the Hugging Face cache).
pub fn run_facts(p: &Project, run: &str) -> Option<gguf_meta::ModelFacts> {
    if run.contains(['/', '\\']) || run.contains("..") {
        return None;
    }
    let rdir = runs_dir(p).join(run);
    let mut folders = vec![rdir.join("model")];
    if let Some(repo) = read_json(&rdir.join("config.json")).and_then(|c| c["model"]["repo"].as_str().map(String::from)) {
        folders.push(PathBuf::from(&repo));
        folders.extend(hf_cache_snapshots(&repo));
    }
    folders.into_iter().map(|f| gguf_meta::folder_facts(&f)).find(|f| f.layers.is_some())
}

/// The light check before an export: the matrix's header against the model's. Never a refusal; the page shows the warnings in yellow.
/// {"readable", "summary": {arch, layers, entries, chunks} | null, "model": {arch, layers} | null, "warnings": [...]}
pub fn check(p: &Project, id: &str, run: Option<&str>) -> Result<Value> {
    let (dir, _) = get(p, id)?;
    let Some(file) = matrix_file(&dir) else { return Err(Error::new("That saved matrix has no file.")) };
    let summary = gguf_meta::matrix_summary(&file);
    let facts = run.and_then(|r| run_facts(p, r));
    let mut warnings = gguf_meta::compat_warnings(&file, &facts.clone().unwrap_or_default());
    if run.is_some() && facts.is_none() && summary.is_some() {
        warnings.push("The layers of this model could not be read, so the matrix could not be compared with it. llama-quantize will tell if it does not fit.".to_string());
    }
    Ok(json!({"readable": summary.is_some(), "file": file.file_name().and_then(|n| n.to_str()),
              "summary": summary.map(|s| json!({"arch": s.arch, "layers": s.layers, "entries": s.entries, "chunks": s.chunks})),
              "model": facts.map(|f| json!({"arch": f.arch, "layers": f.layers})), "warnings": warnings}))
}

// ------------------------------------------------------------------------------------------------------------------ making a matrix

/// Where the model comes from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SourceSpec {
    /// "run" | "model" | "gguf"
    pub kind: String,
    /// run: the name of the trained run
    pub run: Option<String>,
    /// model: an id of the model list (also a made, slimmed or healed model)
    pub model: Option<String>,
    /// model: a local folder or a Hugging Face id typed by the person (instead of `model`)
    pub path: Option<String>,
    /// gguf: the file
    pub file: Option<String>,
    /// gguf: the model folder or Hugging Face id the file was made from (its chat template writes the practice conversations)
    pub tokenizer: Option<String>,
}

/// What the person chose on the "Create a matrix" form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MakeSpec {
    pub source: SourceSpec,
    /// "data" (the saved data version as the model sees it, plus general text) | "text" (a text file of the person's own, used as it is)
    pub calibration: String,
    pub text_file: Option<String>,
    pub general: Option<crate::general::GeneralSource>,
    /// the share of general text; None = 20 %
    pub general_share: Option<f64>,
    pub chunks: u32,
    pub nextn: bool,
    /// also make the legacy .dat
    pub dat: bool,
    pub label: String,
}

impl Default for MakeSpec {
    fn default() -> Self {
        MakeSpec { source: SourceSpec::default(), calibration: "data".into(), text_file: None, general: None, general_share: None, chunks: IMATRIX_DEFAULT_CHUNKS, nextn: true, dat: false, label: String::new() }
    }
}

/// The job for the worker, and the short name the id is made from. Everything the person could get wrong is a plain message here, before anything runs.
pub fn build_job(p: &Project, spec: &MakeSpec, id: &str, stop_file: &Path) -> Result<Value> {
    if !IMATRIX_CHUNKS.contains(&spec.chunks) {
        return Err(Error::new(format!("Unknown amount of calibration text: {} chunks. Choose 64, 128 or 256.", spec.chunks)));
    }
    let s = &spec.source;
    let (source, name) = match s.kind.as_str() {
        "run" => {
            let run = s.run.as_deref().filter(|r| !r.is_empty()).ok_or_else(|| Error::new("Choose a trained run."))?;
            if run.contains(['/', '\\']) || run.contains("..") {
                return Err(Error::new("Unknown run."));
            }
            let rdir = runs_dir(p).join(run);
            let (adapter, full) = (rdir.join("adapter"), rdir.join("model"));
            let is_full = full.join("config.json").is_file();
            if !is_full && !adapter.join("adapter_config.json").is_file() {
                return Err(Error::new("This run has no saved model to measure."));
            }
            let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of this run are missing."))?;
            let (repo, family) = (cfg["model"]["repo"].as_str().unwrap_or("").to_string(), cfg["model"]["family"].as_str().unwrap_or("generic").to_string());
            // a 16-bit GGUF that an export of this run left behind is used instead of merging again (the worker checks that it is newer than the weights)
            let f16s: Vec<PathBuf> = std::fs::read_dir(exports_dir(p)).map(|d| d.flatten().filter(|e| read_json(&e.path().join("meta.json")).is_some_and(|m| m["run"] == run))
                .map(|e| e.path().join("model-f16.gguf")).filter(|f| f.is_file()).collect()).unwrap_or_default();
            (json!({"kind": "run", "run": run, "adapter": if is_full { Value::Null } else { json!(adapter) }, "full_model": if is_full { json!(full) } else { Value::Null },
                    "base": {"repo": repo, "family": family}, "f16_candidates": f16s}), run.to_string())
        }
        "model" => {
            if let Some(id) = s.model.as_deref().filter(|m| !m.is_empty()) {
                let m = crate::train::catalog::find(id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
                (json!({"kind": "model", "model": m.repo, "family": m.family, "name": m.name}), m.name)
            } else {
                let path = s.path.as_deref().map(str::trim).filter(|m| !m.is_empty()).ok_or_else(|| Error::new("Choose a model: one of the list, a folder, or a Hugging Face id."))?;
                let family = crate::train::catalog::models().into_iter().find(|m| m.repo == path).map(|m| m.family).unwrap_or_else(|| "generic".into());
                let short = path.rsplit(['/', '\\']).find(|x| !x.is_empty()).unwrap_or(path).to_string();
                (json!({"kind": "model", "model": path, "family": family, "name": short}), short)
            }
        }
        "gguf" => {
            let file = s.file.as_deref().map(str::trim).filter(|f| !f.is_empty()).ok_or_else(|| Error::new("Choose the GGUF file."))?;
            if !Path::new(file).is_file() {
                return Err(Error::new(format!("The file {file} was not found.")));
            }
            let tok = s.tokenizer.as_deref().map(str::trim).filter(|t| !t.is_empty());
            let family = tok.and_then(|t| crate::train::catalog::models().into_iter().find(|m| m.repo == t)).map(|m| m.family).unwrap_or_else(|| "generic".into());
            let stem = Path::new(file).file_stem().and_then(|x| x.to_str()).unwrap_or("gguf").to_string();
            (json!({"kind": "gguf", "file": file, "tokenizer": tok, "family": family}), stem)
        }
        _ => return Err(Error::new("Choose where the model comes from: a trained run, a model, or a GGUF file.")),
    };
    let calibration = match spec.calibration.as_str() {
        "text" => {
            let f = spec.text_file.as_deref().map(str::trim).filter(|f| !f.is_empty()).ok_or_else(|| Error::new("Choose the text file."))?;
            if !Path::new(f).is_file() {
                return Err(Error::new(format!("The text file {f} was not found.")));
            }
            json!({"kind": "text", "file": f})
        }
        "data" => {
            let (version, _) = crate::train::runs::latest_version(p).ok_or_else(|| Error::new("Save a version of the training data first (Step 2), or choose a text file of your own as the calibration text."))?;
            let share = spec.general_share.unwrap_or(IMATRIX_GENERAL_SHARE);
            if !(0.0..=0.5).contains(&share) {
                return Err(Error::new("The share of general text is between 0 and 50 %."));
            }
            let general = if share > 0.0 { crate::general::job_block(&crate::general::resolve(spec.general.as_ref(), true)?, share) } else { Value::Null };
            json!({"kind": "data", "train": p.folder.join("datasets").join(format!("v{version}")).join("train.jsonl"), "data_version": version, "general_share": share,
                   "general": general, "seed": crate::general::SEED})
        }
        _ => return Err(Error::new("Choose the calibration text: your saved data, or a text file of your own.")),
    };
    Ok(json!({"job": "imatrix_make", "root": lib_dir(p), "id": id, "label": tidy_label(&spec.label), "model_name": name, "stop_file": stop_file, "chunks": spec.chunks, "ctx": IMATRIX_CTX,
              "nextn": spec.nextn, "ngl": 99, "dat": spec.dat, "statistics": true, "source": source, "calibration": calibration}))
}

/// The name an id is made from (the same one `build_job` puts in the job).
fn name_for(job: &Value) -> String {
    job["model_name"].as_str().unwrap_or("matrix").to_string()
}

fn event(e: WorkerEvent) -> Value {
    crate::reshape::runs::event_json(e)
}

fn python() -> Result<PathBuf> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    Ok(py)
}

/// Makes a matrix in the worker. Stop writes the STOP file first (the worker kills llama-imatrix and ends cleanly, "Cancelled."), the worker is killed after the grace time.
/// Returns the worker's answer with `entry` (the new library entry).
pub fn create(p: &Project, spec: &MakeSpec, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python()?;
    std::fs::create_dir_all(lib_dir(p))?;
    let probe = build_job(p, spec, "x", &lib_dir(p).join("STOP"))?;
    let id = new_id(p, &name_for(&probe));
    let job = build_job(p, spec, &id, &lib_dir(p).join("STOP"))?;
    let mut result = run_export_job(&py, &job, &lib_dir(p), Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(event(e)))?;
    result["entry"] = entry(&lib_dir(p).join(&id)).unwrap_or(Value::Null);
    Ok(result)
}

/// Makes the legacy .dat of an entry that has none.
pub fn make_dat(p: &Project, id: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python()?;
    let (dir, e) = get(p, id)?;
    if e["has_gguf"] != true {
        return Err(Error::new("This matrix has no GGUF file to convert."));
    }
    let _g = mark_in_use(id);
    let job = json!({"job": "imatrix_dat", "dir": dir, "stop_file": lib_dir(p).join("STOP")});
    let mut result = run_export_job(&py, &job, &lib_dir(p), Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(event(e)))?;
    result["entry"] = entry(&dir).unwrap_or(Value::Null);
    Ok(result)
}

// ------------------------------------------------------------------------------------------------------------------ what the page shows

/// Everything the "Importance matrices" card needs: the library, what a matrix can be made from, the amounts of text with time guesses, the general-text chooser.
pub fn state(p: &Project) -> Value {
    let runs = exportable_runs(p);
    let mut estimates = serde_json::Map::new();
    let est = |params: Option<f64>| { let mut per = serde_json::Map::new(); for c in IMATRIX_CHUNKS { per.insert(c.to_string(), imatrix_estimate(c, params)); } Value::Object(per) };
    for r in &runs {
        if let Some(n) = r["name"].as_str() {
            estimates.insert(format!("run:{n}"), est(r["model_id"].as_str().and_then(crate::train::catalog::find).map(|m| m.params_b)));
        }
    }
    let models: Vec<Value> = crate::train::catalog::models().into_iter().map(|m| {
        estimates.insert(format!("model:{}", m.id), est(Some(m.params_b)));
        json!({"id": m.id, "name": m.name, "repo": m.repo, "family": m.family, "params_b": m.params_b, "reshape": m.reshape})
    }).collect();
    estimates.insert("default".into(), est(None));
    json!({"entries": list(p), "exports": exports_with_matrix(p), "runs": runs.iter().map(|r| json!({"name": r["name"], "model": r["model"], "status": r["status"], "loss": r["loss"]})).collect::<Vec<_>>(),
           "models": models, "data_version": crate::train::runs::latest_version(p).map(|(n, _)| n), "chunks": IMATRIX_CHUNKS, "default_chunks": IMATRIX_DEFAULT_CHUNKS, "ctx": IMATRIX_CTX,
           "general_share": IMATRIX_GENERAL_SHARE, "estimates": estimates, "general": crate::general::state(Some(p)),
           "env_ready": crate::envmgr::health::env_installed(&env_dir())})
}

/// The compact list the export page offers under "Use a saved matrix".
pub fn brief(p: &Project) -> Vec<Value> {
    list(p).into_iter().map(|e| json!({"id": e["id"], "label": e["label"], "model": e["model"], "chunks": e["chunks"], "size_mb": e["size_mb"], "created": e["created"], "nextn": e["nextn"],
                                      "has_dat": e["has_dat"], "source": e["source"]})).collect()
}

/// The library file for an export that is given a saved matrix: (id, file). Errors in plain words when the entry or its file is gone.
pub fn for_export(p: &Project, id: &str) -> Result<(String, PathBuf)> {
    let (dir, _) = get(p, id)?;
    let f = matrix_file(&dir).ok_or_else(|| Error::new("That saved matrix has no file any more."))?;
    Ok((id.to_string(), f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_names_are_plain_and_short() {
        assert_eq!(short_name("Qwen3.5-9B Instruct"), "qwen3-5-9b-instruct");
        assert_eq!(short_name("../etc/passwd"), "etc-passwd");
        assert_eq!(short_name("ÄÖ äö"), "matrix");
        assert!(short_name("a-very-long-model-name-that-goes-on-and-on").len() <= 24);
        assert_eq!(short_name(""), "matrix");
    }

    #[test]
    fn ids_are_checked() {
        for bad in ["", "..", "a/b", "a\\b", "_x", ".hidden", "x..y"] {
            assert!(valid_id(bad).is_err(), "{bad}");
        }
        assert!(valid_id("20261115-1030-run1").is_ok());
    }

    #[test]
    fn the_in_use_guard_lives_exactly_as_long_as_the_job() {
        let id = "20261115-guard-test";
        assert!(!in_use(id));
        {
            let _a = mark_in_use(id);
            let _b = mark_in_use(id);
            assert!(in_use(id));
        }
        assert!(!in_use(id));
    }
}
