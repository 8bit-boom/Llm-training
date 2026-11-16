//! Adapting an importance matrix to a model that was cut on the Slim page (docs/IMATRIX.md, "Adapting a matrix to a cut model"). The worker does the work
//! (`worker/ndforge_worker/imatrixadapt.py`: job `imatrix_adapt`, a new library entry with `source_kind: "adapted"`); this module is the part the page needs:
//! the choice (`AdaptSpec`), what the made models say about how they were cut (`cut_facts`, the same refusals and texts as the worker), the numbers shown before anything is
//! made (`preview`: layers kept, `ffn_down` before and after, the size of the file, exactly when the matrix file is at hand), the job, and the run.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::gguf_meta;
use super::imatrix_lib::{self as lib, MergeInput};
use super::runs::{read_json, run_export_job, IMATRIX_STOP_GRACE};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::reshape::runs::models_dir;

/// What the person chose in "Adapt to my cut model".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptSpec {
    /// the matrix to adapt: a matrix of the library (`id`), a matrix file (`file`) or Unsloth's published matrix for a base model (`base`)
    pub source: MergeInput,
    /// the made model (its folder name under `models/`) to adapt it to
    pub model: String,
    /// multiplies sums and counts (None = 1): what the matrix weighs when it is merged
    pub weight: Option<f64>,
    pub label: String,
    /// only for a model made by an older version whose record has only counts: the layers of the original it kept (numbers from 0)
    pub kept_layers: Option<Vec<u32>>,
    /// the size of the published matrix as the Hub states it, bytes (an estimate before it is downloaded)
    pub source_size: Option<u64>,
}

/// The largest weight (the worker's `WEIGHT_MAX`).
pub const WEIGHT_MAX: f64 = 1e6;

pub fn thousands(n: u64) -> String {
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

pub fn check_weight(w: Option<f64>) -> Result<f64> {
    match w {
        None => Ok(1.0),
        Some(w) if w.is_finite() && w > 0.0 && w <= WEIGHT_MAX => Ok(w),
        Some(w) => Err(Error::new(format!("The weight must be a number above 0 (and at most {WEIGHT_MAX:e}), not {w}."))),
    }
}

// ------------------------------------------------------------------------------------------------------------------ what a made model says about how it was cut

/// The folder `models/<name>` of a model made on the Slim page. Names come from the window: no path tricks.
pub fn model_dir(p: &Project, name: &str) -> Result<PathBuf> {
    if name.is_empty() || name.starts_with(['_', '.']) || name.contains(['/', '\\']) || name.contains("..") {
        return Err(Error::new("Choose the cut model to adapt the matrix to."));
    }
    let dir = models_dir(p).join(name);
    if !dir.join("model").join("ndforge_reshape.json").is_file() && !dir.join("ndforge_reshape.json").is_file() {
        return Err(Error::new(format!("The model {name} has no record of how it was cut (no ndforge_reshape.json), so the layers and neurons it kept are unknown. Only a model made on the Slim or grow \
                                       page can have a matrix adapted to it. For any other model, measure a matrix on the model itself (Create a matrix).")));
    }
    Ok(dir)
}

fn record(dir: &Path) -> Option<Value> {
    read_json(&dir.join("model").join("ndforge_reshape.json")).or_else(|| read_json(&dir.join("ndforge_reshape.json"))).filter(|v| v.is_object())
}

/// How a model was cut, from its `ndforge_reshape.json` (the same reading and the same refusal texts as the worker's `read_cut`).
#[derive(Debug, Clone, PartialEq)]
pub struct CutFacts {
    pub name: String,
    pub origin_repo: Option<String>,
    pub origin_layers: u64,
    /// the original layer that became layer j
    pub kept: Option<Vec<u64>>,
    pub layers: u64,
    /// the MLP width before and after, when the record has a width block
    pub inter: Option<(u64, u64)>,
    /// (code, plain text) when this model cannot have a matrix adapted to it
    pub refusal: Option<(String, String)>,
}

impl CutFacts {
    pub fn thinner(&self) -> bool {
        self.inter.is_some_and(|(a, b)| a != b)
    }
    pub fn dropped(&self) -> Vec<u64> {
        let k = self.kept.clone().unwrap_or_default();
        (0..self.origin_layers).filter(|i| !k.contains(i)).collect()
    }
}

fn u64s(v: &Value) -> Option<Vec<u64>> {
    v.as_array()?.iter().map(|x| x.as_u64()).collect()
}

/// `kept_override`: the person's answer for a record that has only counts. Only the record is read here (the worker reads `width_kept.npz` and checks it).
pub fn cut_facts(dir: &Path, name: &str, kept_override: Option<&[u32]>) -> Option<CutFacts> {
    let meta = record(dir)?;
    let n_orig = meta["origin"]["layers"].as_u64().filter(|n| *n >= 1);
    let mut f = CutFacts { name: name.to_string(), origin_repo: meta["origin"]["repo"].as_str().map(String::from), origin_layers: n_orig.unwrap_or(0), kept: None, layers: meta["layers"].as_u64().unwrap_or(0), inter: None, refusal: None };
    let refuse = |f: &mut CutFacts, code: &str, text: String| f.refusal = Some((code.to_string(), text));
    if n_orig.is_none() {
        refuse(&mut f, "no_metadata", "This model's record does not say how many layers its original had (it was not made on the Slim or grow page, or by a much older version), so a matrix cannot be adapted to it. Measure a matrix on the model itself (Create a matrix).".into());
        return Some(f);
    }
    let n_orig = n_orig.unwrap();
    if meta["duplicated"].as_array().is_some_and(|a| !a.is_empty()) || meta["zero_init"].as_array().is_some_and(|a| !a.is_empty()) || meta["mode"] == "grow" {
        refuse(&mut f, "grown", format!("The model {name} has layers that were added (a grown model). A matrix measures the layers of the original; there is nothing measured for a copy, so it is not adapted. Measure a matrix on the model itself (Create a matrix)."));
        return Some(f);
    }
    if meta["width"].is_object() {
        for (key, what) in [("num_attention_heads", "attention heads"), ("linear_num_value_heads", "linear-attention value heads")] {
            if let Some(pair) = u64s(&meta["width"][key]).filter(|p| p.len() == 2 && p[0] != p[1]) {
                refuse(&mut f, "heads", format!("The model {name} was made thinner in more than its MLP: its {what} were cut from {} to {}. That changes the input of the attention output of every layer, which a matrix of the original cannot be cut to without measuring (the head axes are experimental). Only a cut of whole layers and of the MLP can be adapted; measure a matrix on the model itself (Create a matrix).", pair[0], pair[1]));
                return Some(f);
            }
        }
        if let Some(pair) = u64s(&meta["width"]["intermediate_size"]).filter(|p| p.len() == 2) {
            f.inter = Some((pair[0], pair[1]));
        }
    }
    f.kept = u64s(&meta["src_layers"]).filter(|k| !k.is_empty()).or_else(|| u64s(&meta["dropped"]).map(|d| (0..n_orig).filter(|i| !d.contains(i)).collect::<Vec<_>>()).filter(|k| meta["layers"].as_u64().is_none_or(|l| l == k.len() as u64)));
    if f.kept.is_none() {
        if let Some(o) = kept_override {
            f.kept = Some(o.iter().map(|x| *x as u64).collect());
        }
    }
    match f.kept.clone() {
        None => refuse(&mut f, "needs_layer_map", format!("The record of the model {name} says that it has {} layers of the original {n_orig}, but not which ones (it was made by an older version that kept only the counts). Tell which layers of the original it kept (their numbers, counted from 0, for example 0, 1, 2, 3, 8, 9, 10, 11) and try again.", meta["layers"].as_u64().map(|l| l.to_string()).unwrap_or_else(|| "?".into()))),
        Some(k) => {
            if k.windows(2).any(|w| w[0] >= w[1]) || k.iter().any(|i| *i >= n_orig) {
                refuse(&mut f, "bad_kept", format!("The kept layers of {name} are not a list of different layers of the original in order, so they cannot be used."));
            } else if meta["layers"].as_u64().is_some_and(|l| l != k.len() as u64) {
                refuse(&mut f, "bad_kept", format!("The record of {name} says {} layers but lists {} kept layers. Its record is inconsistent, so it is not used.", meta["layers"], k.len()));
            } else {
                f.layers = k.len() as u64;
            }
        }
    }
    Some(f)
}

/// The made models of the project with what the page needs to offer them: name, how they were cut, and the reason one cannot be used.
pub fn cut_models(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(models_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let dir = e.path();
        let name = e.file_name().to_str()?.to_string();
        if name.starts_with(['_', '.']) {
            return None;
        }
        let f = cut_facts(&dir, &name, None)?;
        let healed = dir.join("healed").join("config.json").is_file();
        Some(json!({"name": name, "origin_repo": f.origin_repo, "layers_before": f.origin_layers, "layers_after": f.layers, "intermediate_before": f.inter.map(|x| x.0), "intermediate_after": f.inter.map(|x| x.1),
                    "thinner": f.thinner(), "healed": healed, "refusal": f.refusal.as_ref().map(|r| r.1.clone()), "refusal_code": f.refusal.as_ref().map(|r| r.0.clone()), "needs_layer_map": f.refusal.as_ref().is_some_and(|r| r.0 == "needs_layer_map"),
                    "kept": f.kept}))
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

// ------------------------------------------------------------------------------------------------------------------ the matrix to adapt

/// Unsloth's repo for a base model (the worker's `published_repo`, same rules); None for a folder.
pub fn published_repo(base: &str) -> Option<String> {
    let mut name = base.trim().to_string();
    if name.is_empty() || Path::new(&name).is_dir() {
        return None;
    }
    for suffix in ["-unsloth-bnb-4bit", "-bnb-4bit"] {
        if let Some(s) = name.strip_suffix(suffix) {
            name = s.to_string();
        }
    }
    Some(if name.ends_with("-GGUF") { name } else { format!("unsloth/{}-GGUF", name.rsplit('/').next().unwrap_or(&name)) })
}

/// The file of the matrix to adapt when it is at hand: a library entry's `imatrix.gguf`, a file, a published matrix that is already downloaded. Err for a choice that can never be adapted.
pub fn source_file(p: &Project, s: &MergeInput) -> Result<Option<PathBuf>> {
    match s.kind.as_str() {
        "library" => {
            let id = s.id.as_deref().filter(|x| !x.is_empty()).ok_or_else(|| Error::new("Choose a saved matrix."))?;
            let (dir, _) = lib::get(p, id)?;
            let f = dir.join("imatrix.gguf");
            if !f.is_file() {
                return Err(Error::new("This saved matrix has only a legacy .dat file, which cannot be adapted; use a .gguf matrix."));
            }
            Ok(Some(f))
        }
        "file" => {
            let f = s.file.as_deref().map(str::trim).filter(|f| !f.is_empty()).ok_or_else(|| Error::new("Choose the matrix file."))?;
            if !f.to_ascii_lowercase().ends_with(".gguf") {
                return Err(Error::new(format!("{} is not a GGUF matrix: only an imatrix.gguf can be adapted (a legacy .dat file cannot be looked into).", Path::new(f).file_name().and_then(|n| n.to_str()).unwrap_or(f))));
            }
            if !Path::new(f).is_file() {
                return Err(Error::new(format!("The file {f} was not found.")));
            }
            Ok(Some(PathBuf::from(f)))
        }
        "published" => {
            let b = s.base.as_deref().map(str::trim).filter(|b| !b.is_empty()).ok_or_else(|| Error::new("Choose the base model whose published matrix to use."))?;
            let repo = published_repo(b).ok_or_else(|| Error::new(format!("{b} is a folder on this computer, so it has no published matrix on the Hub. Choose the original model it was made from.")))?;
            let f = lib::downloads_dir(p).join(repo.replace('/', "--")).join("imatrix_unsloth.gguf");
            Ok(f.is_file().then_some(f))
        }
        _ => Err(Error::new("Choose the matrix to adapt: a saved matrix, a matrix file, or Unsloth's published matrix.")),
    }
}

fn round_up(n: u64, a: u64) -> u64 {
    n.div_ceil(a) * a
}

/// The numbers the page shows before anything is made, or a refusal: {"ok": true, "model", "origin_repo", "layers_before", "layers_after", "kept", "dropped", "ffn_down_before",
/// "ffn_down_after", "thinner", "size_before_mb", "size_after_mb", "exact", "tensors_before", "tensors_after", "weight", "text", "notes"} or {"ok": false, "code", "text"}.
/// The size is exact when the matrix file is at hand (the layout of the real file: 32-byte alignment, one dimension per tensor), else an estimate from `source_size`, else null.
pub fn preview(p: &Project, spec: &AdaptSpec) -> Result<Value> {
    let weight = check_weight(spec.weight)?;
    let dir = model_dir(p, &spec.model)?;
    let facts = cut_facts(&dir, &spec.model, spec.kept_layers.as_deref()).ok_or_else(|| Error::new("This model has no readable record of how it was cut."))?;
    let no = |code: &str, text: String| Ok(json!({"ok": false, "code": code, "text": text, "model": spec.model, "layers_before": facts.origin_layers, "layers_after": facts.layers}));
    if let Some((code, text)) = &facts.refusal {
        return no(code, text.clone());
    }
    let kept = facts.kept.clone().unwrap_or_default();
    let file = source_file(p, &spec.source)?;
    let thin = facts.thinner();
    let mut v = json!({"ok": true, "model": spec.model, "origin_repo": facts.origin_repo, "layers_before": facts.origin_layers, "layers_after": kept.len(), "kept": kept, "dropped": facts.dropped(),
                       "thinner": thin, "weight": weight, "exact": false, "size_before_mb": null, "size_after_mb": null, "ffn_down_before": facts.inter.map(|x| x.0), "ffn_down_after": facts.inter.map(|x| x.1),
                       "tensors_before": null, "tensors_after": null, "notes": []});
    let mut notes: Vec<String> = vec![];
    if let Some(f) = file {
        let Some(h) = gguf_meta::read_full(&f) else {
            return no("bad_matrix", format!("The matrix {} could not be read as an importance matrix ({}). A legacy .dat file cannot be adapted; use a .gguf matrix.", f.file_name().and_then(|n| n.to_str()).unwrap_or(""), "not a readable GGUF"));
        };
        if h.meta.get("imatrix.datasets").and_then(|d| d.as_array()).is_some_and(|a| a.iter().any(|x| x.as_str().is_some_and(|s| s.starts_with("(adapted to ")))) {
            return no("already_adapted", "This matrix was already adapted to a cut model (its record says so). Adapt the original matrix instead; adapting twice would renumber layers that are already renumbered.".into());
        }
        let layer_of = |n: &str| n.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|x| x.parse::<u64>().ok());
        let max_layer = h.tensors.iter().filter_map(|t| layer_of(&t.name)).max();
        let present: std::collections::BTreeSet<u64> = h.tensors.iter().filter_map(|t| layer_of(&t.name)).filter(|l| *l < facts.origin_layers).collect();
        if max_layer.is_none() || present != (0..facts.origin_layers).collect() {
            return no("wrong_original", format!("This matrix has data for {} layers, but the original of the model {} ({}) has {}. It was not measured on that model (it may belong to another model, or to a model that was cut already). Choose the matrix of the original model.",
                                               max_layer.map(|m| m + 1).unwrap_or(0), spec.model, facts.origin_repo.as_deref().unwrap_or("its original"), facts.origin_layers));
        }
        let down = |t: &gguf_meta::TensorInfo| t.name.ends_with(".ffn_down.weight.in_sum2").then(|| t.dims.first().copied().unwrap_or(0));
        let widths: std::collections::BTreeSet<u64> = h.tensors.iter().filter(|t| layer_of(&t.name).is_some_and(|l| l < facts.origin_layers)).filter_map(down).collect();
        let before = widths.iter().next().copied().unwrap_or(0);
        if widths.len() != 1 || facts.inter.is_some_and(|(b, _)| b != before) {
            return no("wrong_original", format!("In this matrix the MLP (ffn_down) input is {} wide, but the original of the model {} has {}. It was not measured on that model. Choose the matrix of the original model.",
                                               thousands(before), spec.model, facts.inter.map(|x| thousands(x.0)).unwrap_or_else(|| "a different width".into())));
        }
        let after = facts.inter.map(|x| x.1).unwrap_or(before);
        let mut entries: Vec<(String, u64)> = vec![];
        let mut dropped_mtp = false;
        for t in &h.tensors {
            let n = t.dims.first().copied().unwrap_or(0);
            match layer_of(&t.name) {
                None => entries.push((t.name.clone(), n)),
                Some(l) if l >= facts.origin_layers => dropped_mtp = true,
                Some(l) => {
                    if let Some(j) = kept.iter().position(|k| *k == l) {
                        let rest = t.name.splitn(3, '.').nth(2).unwrap_or("");
                        entries.push((format!("blk.{j}.{rest}"), if rest == "ffn_down.weight.in_sum2" { after } else { n }));
                    }
                }
            }
        }
        let note = format!("(adapted to {})", spec.model);
        let has_ds = h.meta.contains_key("imatrix.datasets");
        let kv_after = h.kv_end + if has_ds { 8 + note.len() as u64 } else { (8 + "imatrix.datasets".len() + 4 + 4 + 8 + 8 + note.len()) as u64 };
        let info_after: u64 = entries.iter().map(|(n, _)| 32 + n.len() as u64).sum();
        let header_after = round_up(kv_after + info_after, 32);
        let size_after = header_after + entries.iter().map(|(_, n)| round_up(4 * n, 32)).sum::<u64>();
        let size_before = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        v["size_before_mb"] = json!(mb(size_before));
        v["size_after_mb"] = json!(mb(size_after));
        v["exact"] = json!(true);
        v["ffn_down_before"] = json!(before);
        v["ffn_down_after"] = json!(after);
        v["tensors_before"] = json!(h.tensors.len());
        v["tensors_after"] = json!(entries.len());
        v["size_after_bytes"] = json!(size_after);
        if dropped_mtp {
            notes.push("The statistics of the speculative-decoding head were left out: Slim cuts that head without measuring it.".into());
        }
    } else if let Some(sz) = spec.source_size.filter(|s| *s > 0) {
        let (b, a) = facts.inter.unwrap_or((1, 1));
        let share = 0.3;           // about the share of ffn_down in a layer's statistics (17,408 of about 58,000 values in the 27B)
        let est = sz as f64 * (kept.len() as f64 / facts.origin_layers.max(1) as f64) * (1.0 - share * (1.0 - a as f64 / b.max(1) as f64));
        v["size_before_mb"] = json!(mb(sz));
        v["size_after_mb"] = json!(mb(est as u64));
    }
    if weight != 1.0 {
        notes.push(format!("The weight {weight} multiplies the sums and the counts: a merge counts this matrix {weight} times as strongly; its mean square per channel is unchanged."));
    }
    v["notes"] = json!(notes);
    let ffn = match (v["ffn_down_before"].as_u64(), v["ffn_down_after"].as_u64()) {
        (Some(b), Some(a)) if b != a => format!("ffn_down {} -> {} per layer", thousands(b), thousands(a)),
        (Some(b), _) => format!("ffn_down stays {} per layer", thousands(b)),
        _ => "ffn_down is cut as your model was cut".to_string(),
    };
    let size = v["size_after_mb"].as_f64().map(|m| format!("; {} {:.1} MB", if v["exact"] == true { "exactly" } else { "about" }, m)).unwrap_or_default();
    v["text"] = json!(format!("keeps {} of {} layers; {ffn}{size}", kept.len(), facts.origin_layers));
    Ok(v)
}

fn mb(bytes: u64) -> f64 {
    (bytes as f64 / 1048576.0 * 100.0).round() / 100.0
}

// ------------------------------------------------------------------------------------------------------------------ the job

/// The job for the worker. Everything the person could get wrong is a plain message here, before anything runs.
pub fn build_job(p: &Project, spec: &AdaptSpec, id: &str, stop_file: &Path) -> Result<Value> {
    let weight = check_weight(spec.weight)?;
    let dir = model_dir(p, &spec.model)?;
    let source = match spec.source.kind.as_str() {
        "library" => {
            let lid = spec.source.id.as_deref().filter(|x| !x.is_empty()).ok_or_else(|| Error::new("Choose a saved matrix."))?;
            let (d, e) = lib::get(p, lid)?;
            let f = d.join("imatrix.gguf");
            if !f.is_file() {
                return Err(Error::new("This saved matrix has only a legacy .dat file, which cannot be adapted; use a .gguf matrix."));
            }
            if e["source_kind"] == "adapted" {
                return Err(Error::new("This saved matrix is already an adapted one. Adapt the original matrix instead; adapting twice would renumber layers that are already renumbered."));
            }
            json!({"kind": "library", "id": lid, "file": f, "label": if e["label"].as_str().is_some_and(|l| !l.is_empty()) { e["label"].clone() } else { json!(lid) }, "chunks": e["chunks"], "source_kind": e["source_kind"]})
        }
        _ => {
            let f = source_file(p, &spec.source)?;
            match spec.source.kind.as_str() {
                "file" => json!({"kind": "file", "file": f}),
                _ => json!({"kind": "published", "base": spec.source.base.as_deref().map(str::trim)}),
            }
        }
    };
    Ok(json!({"job": "imatrix_adapt", "root": lib::lib_dir(p), "id": id, "label": lib::tidy_label(&spec.label), "stop_file": stop_file, "cache": lib::downloads_dir(p), "model_dir": dir,
              "model_name": spec.model, "weight": weight, "kept_layers": spec.kept_layers, "source": source, "notes": []}))
}

/// Adapts a matrix to a cut model in the worker (a new library entry; a library source cannot be deleted meanwhile). Stop as for the other matrix jobs. Returns the worker's answer with `entry`.
pub fn adapt(p: &Project, spec: &AdaptSpec, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = lib::python()?;
    std::fs::create_dir_all(lib::lib_dir(p))?;
    let id = lib::new_id(p, &format!("{}-adapted", spec.model));
    let job = build_job(p, spec, &id, &lib::lib_dir(p).join("STOP"))?;
    let _g = (spec.source.kind == "library").then(|| spec.source.id.as_deref().map(lib::mark_in_use)).flatten();
    let mut result = run_export_job(&py, &job, &lib::lib_dir(p), Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(lib::event(e)))?;
    result["entry"] = lib::entry(&lib::lib_dir(p).join(&id)).unwrap_or(Value::Null);
    Ok(result)
}

/// The folder and name of the cut model a library entry was measured on (`cut_model` in its record), when that folder still holds a record.
pub fn entry_cut_model(entry: &Value) -> Option<(String, String)> {
    let dir = entry["cut_model"]["dir"].as_str()?;
    let name = entry["cut_model"]["name"].as_str().filter(|n| !n.is_empty()).map(String::from).or_else(|| Path::new(dir).file_name().and_then(|n| n.to_str()).map(String::from))?;
    (Path::new(dir).join("model").join("ndforge_reshape.json").is_file() || Path::new(dir).join("ndforge_reshape.json").is_file()).then(|| (dir.to_string(), name))
}

/// `origin.repo` of the record of a cut model's folder (the model it was made from), when it is there.
pub fn origin_repo(dir: &Path) -> Option<String> {
    record(dir)?["origin"]["repo"].as_str().map(String::from)
}
