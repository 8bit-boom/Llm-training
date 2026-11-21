//! The Storage page: where the disk went, and a guarded way to give it back (docs/STORAGE.md).
//!
//! Deleting is permanent and destructive, so the rules are strict:
//! * The window only ever sends opaque ids from the latest scan. Every id is looked up again in a FRESH scan (the id holds the kind, the path and the folder's modification time, so a folder
//!   that changed since the scan is refused), and the path of the item is checked against the folders this program owns (`Roots`) before anything is touched.
//! * Links (symbolic links, Windows junctions) are never followed: a link is removed itself, what it points to stays.
//! * Nothing is deleted while a job runs, while an Auto-cut run is running, or when a stopped / failed Auto-cut run still needs the item to go on (Continue).
//! * `dry_run` returns exactly what a real call would remove.
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::UNIX_EPOCH;

use serde::Serialize;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::project::Project;

/// Above this many bytes a deletion needs the word DELETE typed (a model folder always does).
pub const TYPE_ABOVE_BYTES: u64 = 5_000_000_000;
const OLD_LOG_DAYS: i64 = 14;

// ------------------------------------------------------------------------------------------------------------------ what an item is

/// How a deletion is carried out (the plain ones are a guarded removal of the path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exec {
    /// the path, all of it
    Tree,
    /// `models/<name>`: the model list and the training runs are asked first (`reshape::runs::removal_checks`)
    ModelEntry(String),
    /// `models/<name>/model` of a healed model (the unhealed cut): `reshape::runs::remove_cut`
    CutOnly(String),
    /// the big file of a quant comparison: `export::quantcheck::delete_logits`
    QuantLogits(String),
    /// everything inside a folder of a kind of data (the folder itself stays): each part is guarded on its own
    Contents,
    /// job logs older than `OLD_LOG_DAYS` days in the folder
    OldLogs,
    /// the vLLM engine's model volume (a container tool call, not a path)
    Volume(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// something running or a run that will be continued needs it: refused
    InUse,
    /// a report, an export or another model points to it: allowed with a warning
    Referenced,
    Safe,
}

#[derive(Debug, Clone)]
pub struct Item {
    pub id: String,
    /// models | exports | heal | downloads | leftovers
    pub group: &'static str,
    pub kind: &'static str,
    pub label: String,
    pub path: PathBuf,
    pub size: Option<u64>,
    pub size_cached: bool,
    pub modified: Option<String>,
    pub from: Option<String>,
    pub status: Status,
    pub refs: Vec<String>,
    pub note: Option<String>,
    /// why deleting it is a good idea (a suggestion at the top; never ticked by itself)
    pub suggest: Option<String>,
    /// a model folder: the word DELETE has to be typed whatever the size
    pub model: bool,
    /// the path is a link: only the link is removed
    pub link: bool,
    pub project: bool,
    pub exec: Exec,
    /// the repo or model path the Slim / Auto page may have chosen (see `Options::pinned`)
    pub model_repo: Option<String>,
}

impl Item {
    fn new(group: &'static str, kind: &'static str, label: impl Into<String>, path: PathBuf, project: bool, exec: Exec) -> Item {
        let (modified, link) = stat_of(&path);
        let mut it = Item { id: String::new(), group, kind, label: label.into(), path, size: None, size_cached: false, modified, from: None, status: Status::Safe, refs: Vec::new(), note: None, suggest: None,
                            model: false, link, project, exec, model_repo: None };
        it.id = make_id(&it);
        it
    }

    pub fn needs_type(&self, total: u64) -> bool {
        self.model || total > TYPE_ABOVE_BYTES
    }

    /// The window's view of the item: never the full path (the window cannot send one), only a shortened one to read.
    pub fn json(&self) -> Value {
        json!({"id": self.id, "group": self.group, "kind": self.kind, "label": self.label, "path_short": shorten(&self.path), "size": self.size, "size_cached": self.size_cached, "modified": self.modified,
               "from": self.from, "status": self.status, "refs": self.refs, "note": self.note, "suggest": self.suggest, "model": self.model, "link": self.link})
    }
}

fn stat_of(path: &Path) -> (Option<String>, bool) {
    match std::fs::symlink_metadata(path) {
        Ok(m) => (m.modified().ok().map(stamp), is_link(&m)),
        Err(_) => (None, false),
    }
}

fn stamp(t: std::time::SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    chrono::DateTime::from_timestamp(secs, 0).map(|d| d.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_default()
}

/// The id of an item: kind, path and the modification time of the path (a folder that changed after the scan gets another id and is refused).
fn make_id(it: &Item) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (it.kind, it.path.to_string_lossy().as_ref(), it.modified.as_deref().unwrap_or(""), format!("{:?}", it.exec)).hash(&mut h);
    format!("s{:016x}", h.finish())
}

/// The path as a person reads it: the home folder as `~`, a long middle cut out.
pub fn shorten(p: &Path) -> String {
    let mut s = p.to_string_lossy().to_string();
    if let Some(h) = dirs::home_dir() {
        let h = h.to_string_lossy().to_string();
        if h.len() > 3 && s.starts_with(&h) {
            s = format!("~{}", &s[h.len()..]);
        }
    }
    let n = s.chars().count();
    if n > 90 {
        let head: String = s.chars().take(30).collect();
        let tail: String = s.chars().skip(n - 55).collect();
        s = format!("{head}…{tail}");
    }
    s
}

pub fn is_link(m: &std::fs::Metadata) -> bool {
    if m.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        if m.file_attributes() & 0x400 != 0 {
            return true;          // a reparse point: a junction or another kind of link
        }
    }
    false
}

// ------------------------------------------------------------------------------------------------------------------ the folders this program owns

/// The folders a deletion may happen under. Everything else is refused, whatever the id says.
#[derive(Debug, Clone, Default)]
pub struct Roots {
    list: Vec<PathBuf>,
    /// folders that must never be removed themselves (the project folder, a root)
    keep: Vec<PathBuf>,
}

impl Roots {
    pub fn new(p: Option<&Project>) -> Roots {
        let mut r = Roots::default();
        if let Some(p) = p {
            for sub in ["models", "exports", "quantcheck", "quantfit", "imatrix_downloads", "benchmarks", "runs"] {
                r.add(p.folder.join(sub));
            }
            r.keep.push(canon(&p.folder));
        }
        let cfg = crate::paths::config_dir();
        for sub in ["general_cache", "logs", "vllm_selftest"] {
            r.add(cfg.join(sub));
        }
        for hub in hf_cache_roots() {
            r.add(hub);
        }
        r
    }

    /// Tests and unusual set-ups add a root of their own.
    pub fn add(&mut self, p: PathBuf) {
        let c = canon(&p);
        self.keep.push(c.clone());
        self.list.push(c);
    }

    /// A folder of a kind of data itself, for removing what is inside it (never the folder): it must be one of the roots, with no link in the way.
    pub fn guard_contents(&self, path: &Path) -> Result<PathBuf> {
        let meta = std::fs::symlink_metadata(path).map_err(|_| Error::new("That folder is not there any more."))?;
        if is_link(&meta) {
            return Err(Error::new("A link is not emptied."));
        }
        let c = path.canonicalize().map_err(|_| Error::new("That folder is not there any more."))?;
        if !self.list.iter().any(|r| *r == c) {
            return Err(Error::new("That is not one of the folders this program owns, so it is not touched."));
        }
        Ok(c)
    }

    /// The path, if it lies strictly below one of the roots: no `..`, no root itself, no link in the way (the last part may be a link: it is removed itself).
    pub fn guard(&self, path: &Path) -> Result<PathBuf> {
        if path.components().any(|c| matches!(c, Component::ParentDir)) {
            return Err(Error::new("A path with `..` is refused."));
        }
        let meta = std::fs::symlink_metadata(path).map_err(|_| Error::new("That folder is not there any more."))?;
        let name = path.file_name().ok_or_else(|| Error::new("That path has no name."))?;
        let parent = path.parent().ok_or_else(|| Error::new("That path has no folder."))?;
        let parent_c = parent.canonicalize().map_err(|_| Error::new("The folder around it is not there any more."))?;
        let whole = parent_c.join(name);
        if self.keep.iter().any(|k| *k == whole) && !self.list.iter().any(|r| whole.starts_with(r) && whole != *r) {
            return Err(Error::new("That folder is one this program never deletes."));
        }
        if !self.list.iter().any(|r| parent_c.starts_with(r)) {
            return Err(Error::new("That is not inside a folder this program owns, so it is not touched."));
        }
        if self.list.iter().any(|r| *r == whole) {
            return Err(Error::new("The folder of a kind of data is never deleted as a whole."));
        }
        let _ = meta;
        Ok(whole)
    }
}

/// One spelling for a path, used for every root and every path that is compared with one: the deepest part that exists is canonicalized (real letters of the drive and the names,
/// no short `RUNNER~1` names, no links, the same verbatim prefix on both sides), the part that does not exist yet is added lexically (`.` and `..` resolved, a trailing
/// slash and mixed `/` and `\` do not matter). A root that is created later therefore has the same spelling as one that is there already.
pub fn canon(p: &Path) -> PathBuf {
    let mut rest: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = p.to_path_buf();
    loop {
        if let Ok(c) = cur.canonicalize() {
            let mut out = c;
            for part in rest.iter().rev() {
                if part == ".." {
                    out.pop();
                } else if part != "." {
                    out.push(part);
                }
            }
            return out;
        }
        match (cur.file_name().map(|n| n.to_os_string()), cur.parent().map(Path::to_path_buf)) {
            (Some(n), Some(parent)) => {
                rest.push(n);
                cur = parent;
            }
            (None, Some(parent)) if cur.ends_with("..") => {
                rest.push("..".into());
                cur = parent;
            }
            _ => return p.to_path_buf(),
        }
    }
}

/// The Hugging Face hub folders the program and its worker use (`HF_HUB_CACHE`, `HF_HOME/hub`, `~/.cache/huggingface/hub`).
pub fn hf_cache_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = vec![];
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(v).join("hub"));
    }
    if let Some(h) = dirs::home_dir() {
        roots.push(h.join(".cache").join("huggingface").join("hub"));
    }
    let mut seen = BTreeSet::new();
    roots.into_iter().filter(|r| r.is_dir() && seen.insert(canon(r))).collect()
}

// ------------------------------------------------------------------------------------------------------------------ sizes

/// Bytes of everything below `path` (links are not followed and count nothing). Stops early (returns what it has) when `cancel` is set.
pub fn dir_size(path: &Path, cancel: &AtomicBool) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(path) else { return 0 };
    if is_link(&m) {
        return 0;
    }
    if m.is_file() {
        return m.len();
    }
    let mut total = 0;
    let mut stack = vec![path.to_path_buf()];
    while let Some(d) = stack.pop() {
        if cancel.load(Ordering::SeqCst) {
            return total;
        }
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let Ok(m) = e.metadata() else { continue };
            let Ok(sm) = std::fs::symlink_metadata(e.path()) else { continue };
            if is_link(&sm) {
                continue;
            }
            if m.is_dir() {
                stack.push(e.path());
            } else {
                total += m.len();
            }
        }
    }
    total
}

fn size_cache_file() -> PathBuf {
    crate::paths::config_dir().join("storage_sizes.json")
}

fn read_size_cache() -> serde_json::Map<String, Value> {
    std::fs::read(size_cache_file()).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v.as_object().cloned()).unwrap_or_default()
}

// ------------------------------------------------------------------------------------------------------------------ the scan (structure only: no sizes)

/// What the scan needs to know from the program.
#[derive(Default, Clone)]
pub struct Ctx<'a> {
    pub project: Option<&'a Project>,
    /// a job runs right now (the one-job flag of the window): nothing may be deleted
    pub busy: bool,
    pub vllm: Option<&'a crate::settings::VllmSettings>,
}

fn read_json(p: &Path) -> Option<Value> {
    std::fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn read_dir_sorted(d: &Path) -> Vec<std::fs::DirEntry> {
    let mut v: Vec<_> = std::fs::read_dir(d).map(|r| r.flatten().collect()).unwrap_or_default();
    v.sort_by_key(|e| e.file_name());
    v
}

/// The unfinished Auto-cut runs of a project and the model names they use.
struct RunFacts {
    id: String,
    status: String,
    names: BTreeSet<String>,
    unfinished_steps: bool,
    heal_pending: bool,
}

fn run_facts(p: &Project) -> Vec<RunFacts> {
    crate::reshape::auto::list(p).into_iter().map(|st| {
        let mut names = BTreeSet::new();
        for s in &st.stages {
            for n in [s.cut.as_ref().map(|c| c.name.clone()), s.heal.as_ref().map(|h| h.name.clone()), s.retry_from.as_ref().map(|h| h.name.clone())].into_iter().flatten() {
                names.insert(n);
            }
        }
        for m in &st.made {
            names.insert(m.name.clone());
        }
        let unfinished_steps = st.steps.iter().any(|s| !matches!(s.status.as_str(), "done" | "skipped"));
        let heal_pending = st.steps.iter().any(|s| s.id.ends_with(".heal") && !matches!(s.status.as_str(), "done" | "skipped"));
        RunFacts { id: st.id.clone(), status: st.status.clone(), names, unfinished_steps, heal_pending }
    }).collect()
}

fn run_continues(r: &RunFacts) -> bool {
    matches!(r.status.as_str(), "running" | "stopped" | "failed") && r.unfinished_steps
}

/// Marks an item `in use` by the Auto-cut runs that will go on (or are going) and `referenced` by the finished ones.
fn mark_runs(it: &mut Item, name: &str, runs: &[RunFacts]) {
    for r in runs.iter().filter(|r| r.names.contains(name)) {
        if run_continues(r) {
            it.status = Status::InUse;
            it.refs.push(format!("Auto-cut run {} ({}) still needs it to go on", r.id, r.status));
        } else {
            if it.status == Status::Safe {
                it.status = Status::Referenced;
            }
            it.refs.push(format!("the report of Auto-cut run {} points to it", r.id));
        }
    }
}

fn heal_unfinished(models: &Path) -> bool {
    read_dir_sorted(models).iter().any(|e| read_json(&e.path().join("heal.json")).is_some_and(|h| matches!(h["status"].as_str(), Some("running" | "stopped" | "failed"))))
}

/// Every item the page can show, with ids, status and suggestions, and no sizes yet. Fast: it reads directory listings and a few small files.
pub fn scan_items(ctx: &Ctx) -> Vec<Item> {
    let mut out: Vec<Item> = Vec::new();
    if let Some(p) = ctx.project {
        project_items(p, &mut out);
    }
    app_items(ctx, &mut out);
    if ctx.busy {
        for it in &mut out {
            it.status = Status::InUse;
            it.refs.push("a job is running: nothing can be deleted until it has ended".into());
        }
    }
    out
}

fn project_items(p: &Project, out: &mut Vec<Item>) {
    let runs = run_facts(p);
    let models = p.folder.join("models");
    let exports = p.folder.join("exports");
    let finished_all = !runs.iter().any(run_continues) && !heal_unfinished(&models);
    let export_metas: Vec<(String, Value)> = read_dir_sorted(&exports).iter().filter_map(|e| Some((e.file_name().to_string_lossy().to_string(), read_json(&e.path().join("meta.json"))?))).collect();

    // a) models
    for e in read_dir_sorted(&models) {
        let name = e.file_name().to_string_lossy().to_string();
        let dir = e.path();
        if !dir.is_dir() && !stat_of(&dir).1 {
            continue;
        }
        if name == "_teacher_cache" {
            for c in read_dir_sorted(&dir) {
                let mut it = Item::new("heal", "teacher_cache", format!("The original's saved answers ({})", c.file_name().to_string_lossy()), c.path(), true, Exec::Tree);
                it.note = Some("Deleting it makes the original (the teacher) answer your texts AGAIN the next time a model is healed: that takes hours.".into());
                if !finished_all || runs.iter().any(|r| r.heal_pending && run_continues(r)) {
                    it.status = Status::InUse;
                    it.refs.push("a heal or an Auto-cut run that can be continued still needs it".into());
                } else {
                    it.suggest = Some("The healing that used it is finished.".into());
                }
                out.push(it);
            }
            continue;
        }
        if name == "_scores" || name == "_search" {
            for c in read_dir_sorted(&dir) {
                let kind = if name == "_scores" { "Measurement of the layers" } else { "Search for the best cut" };
                let mut it = Item::new("heal", "measurement", format!("{kind} ({})", c.file_name().to_string_lossy()), c.path(), true, Exec::Tree);
                it.note = Some("It is measured again when it is needed.".into());
                out.push(it);
            }
            continue;
        }
        if name.starts_with('_') {
            continue;          // `_auto`: the Auto-cut runs' own records (small, never deleted here)
        }
        let state = read_json(&dir.join("state.json")).unwrap_or(Value::Null);
        let reshape_meta = read_json(&dir.join("model").join("ndforge_reshape.json")).unwrap_or(Value::Null);
        let origin = reshape_meta["origin"]["repo"].as_str().or_else(|| state["source_name"].as_str()).map(String::from);
        let has_healed = dir.join("healed").is_dir();
        let used_by_exports: Vec<String> = export_metas.iter().filter(|(_, m)| {
            let b = m["base_repo"].as_str().unwrap_or("").replace('\\', "/");
            b.contains(&format!("models/{name}/"))
        }).map(|(n, _)| format!("the export {n} was made from it")).collect();
        let kind_word = match (dir.join("model").is_dir(), has_healed) {
            (true, true) => "cut and healed",
            (true, false) => "cut",
            (false, true) => "healed",
            _ => "model",
        };
        let mut it = Item::new("models", "model_entry", format!("{name} ({kind_word})"), dir.clone(), true, Exec::ModelEntry(name.clone()));
        it.model = true;
        it.from = origin.clone().map(|o| format!("made from {o}"));
        it.model_repo = Some(dir.join(if has_healed { "healed" } else { "model" }).to_string_lossy().to_string());
        mark_runs(&mut it, &name, &runs);
        if !used_by_exports.is_empty() {
            if it.status == Status::Safe {
                it.status = Status::Referenced;
            }
            it.refs.extend(used_by_exports.clone());
        }
        out.push(it);
        // the unhealed cut of a healed model
        if has_healed && dir.join("model").is_dir() && state["model_removed"] != true {
            let mut c = Item::new("models", "cut_only", format!("{name}: the cut before healing"), dir.join("model"), true, Exec::CutOnly(name.clone()));
            c.model = true;
            c.from = origin.map(|o| format!("made from {o}"));
            c.note = Some("The healed model stays. This is the same model before it was healed.".into());
            c.model_repo = Some(dir.join("model").to_string_lossy().to_string());
            mark_runs(&mut c, &name, &runs);
            if let Some(run) = runs.iter().find(|r| r.names.contains(&name)) {
                if !run_continues(run) {
                    c.suggest = Some(format!("The healed model of Auto-cut run {} exists; the cut before healing is only an intermediate.", run.id));
                }
            }
            out.push(c);
        }
        // checkpoints of a heal that is done
        let ckpt = dir.join("distill").join("run");
        if ckpt.is_dir() {
            let mut c = Item::new("heal", "heal_checkpoints", format!("{name}: checkpoints of the heal"), ckpt, true, Exec::Tree);
            let unfinished = read_json(&dir.join("heal.json")).is_some_and(|h| matches!(h["status"].as_str(), Some("running" | "stopped" | "failed")));
            if unfinished {
                c.status = Status::InUse;
                c.refs.push("the heal can be continued from them".into());
            } else if has_healed {
                c.suggest = Some("The healed model is made; the checkpoints are not needed any more.".into());
            }
            out.push(c);
        }
        let cache = dir.join("distill").join("cache");
        if cache.is_dir() {
            let mut c = Item::new("heal", "teacher_cache", format!("{name}: the original's answers kept with this model"), cache, true, Exec::Tree);
            c.note = Some("Deleting it makes the teacher answer again if this model is healed again.".into());
            if !finished_all {
                c.status = Status::InUse;
                c.refs.push("a heal or an Auto-cut run that can be continued still needs it".into());
            }
            out.push(c);
        }
        for leftover in ["model.partial", "healed.partial"] {
            let lp = dir.join(leftover);
            if lp.exists() {
                let mut c = Item::new("leftovers", "leftover", format!("{name}/{leftover}: half-written files of a model that was cut off"), lp, true, Exec::Tree);
                c.suggest = Some("Left behind by a job that did not finish.".into());
                out.push(c);
            }
        }
    }

    // b) exports
    for e in read_dir_sorted(&exports) {
        let name = e.file_name().to_string_lossy().to_string();
        let dir = e.path();
        if dir.is_file() && name.ends_with(".partial") {
            let mut c = Item::new("leftovers", "leftover", format!("exports/{name}: a half-written file"), dir, true, Exec::Tree);
            c.suggest = Some("Left behind by an export that did not finish.".into());
            out.push(c);
            continue;
        }
        if !dir.is_dir() {
            continue;
        }
        if name.ends_with("-gguf-work") {
            let mut c = Item::new("leftovers", "leftover", format!("exports/{name}: working files of an export"), dir, true, Exec::Tree);
            c.suggest = Some("The working folder of a GGUF export that is not running.".into());
            out.push(c);
            continue;
        }
        let meta = read_json(&dir.join("meta.json")).unwrap_or(Value::Null);
        let ggufs: Vec<PathBuf> = read_dir_sorted(&dir).into_iter().map(|f| f.path()).filter(|f| f.extension().is_some_and(|x| x == "gguf") && f.file_name().is_some_and(|n| n != "model-f16.gguf" && n != "imatrix.gguf")).collect();
        let mut it = Item::new("exports", "export", format!("{name}{}", meta["quant"].as_str().map(|q| format!(" ({q})")).unwrap_or_default()), dir.clone(), true, Exec::Tree);
        it.from = meta["run"].as_str().map(|r| format!("from the run {r}")).or_else(|| meta["base_repo"].as_str().map(|b| format!("from {}", shorten(Path::new(b)))));
        if let Some(sv) = runs.iter().find(|r| run_continues(r)) {
            let _ = sv;
        }
        out.push(it);
        let f16 = dir.join("model-f16.gguf");
        if f16.is_file() {
            let mut c = Item::new("exports", "export_f16", format!("{name}: the 16-bit GGUF (model-f16.gguf)"), f16, true, Exec::Tree);
            c.note = Some("The biggest file of an export. A later export of the same model makes it again (and the importance matrix reuses it).".into());
            if !ggufs.is_empty() {
                c.suggest = Some("The quantized GGUF files exist; the 16-bit one was only the step before them.".into());
            }
            out.push(c);
        }
        let merged = dir.join("merged");
        if merged.is_dir() {
            let mut c = Item::new("exports", "export_merged", format!("{name}: the merged 16-bit model (merged/)"), merged, true, Exec::Tree);
            c.model = true;
            if !ggufs.is_empty() {
                c.suggest = Some("The GGUF files exist; the merged model is only needed to export again.".into());
            }
            out.push(c);
        }
        for f in read_dir_sorted(&dir) {
            if f.file_name().to_string_lossy().ends_with(".partial") {
                let mut c = Item::new("leftovers", "leftover", format!("exports/{name}/{}: a half-written file", f.file_name().to_string_lossy()), f.path(), true, Exec::Tree);
                c.suggest = Some("Left behind by an export that did not finish.".into());
                out.push(c);
            }
        }
    }

    // c) comparisons and caches of the project
    for e in read_dir_sorted(&p.folder.join("quantcheck")) {
        let id = e.file_name().to_string_lossy().to_string();
        let kld = e.path().join("base.kld");
        if kld.is_file() {
            let mut c = Item::new("heal", "quant_logits", format!("Compare quants {id}: the 16-bit model's answers (base.kld)"), kld, true, Exec::QuantLogits(id));
            c.note = Some("The comparison's results stay; a new comparison of the same model measures the 16-bit model again.".into());
            c.suggest = Some("Only needed to compare more sizes against the same 16-bit answers.".into());
            out.push(c);
        }
    }
    for (sub, label) in [("quantfit", "Fitting of mixed quants"), ("imatrix_downloads", "Downloaded importance matrices"), ("benchmarks", "Speed measurements")] {
        let d = p.folder.join(sub);
        if d.is_dir() && read_dir_sorted(&d).iter().next().is_some() {
            out.push(Item::new("heal", "cache_dir", label.to_string(), d, true, Exec::Contents));
        }
    }
    for e in read_dir_sorted(&p.folder.join("runs")) {
        let ck = e.path().join("checkpoints");
        if ck.is_dir() {
            let mut c = Item::new("heal", "train_checkpoints", format!("Training run {}: checkpoints", e.file_name().to_string_lossy()), ck, true, Exec::Tree);
            c.note = Some("The finished adapter stays. Training cannot be continued from a checkpoint that is deleted.".into());
            out.push(c);
        }
    }
}

fn app_items(ctx: &Ctx, out: &mut Vec<Item>) {
    let cfg = crate::paths::config_dir();
    let known: BTreeSet<String> = crate::train::catalog::models().into_iter().map(|m| m.repo.to_lowercase()).collect();
    let general = cfg.join("general_cache");
    if general.is_dir() && read_dir_sorted(&general).iter().next().is_some() {
        let mut c = Item::new("heal", "cache_dir", "General text kept for healing (all projects)".to_string(), general, false, Exec::Contents);
        c.note = Some("It is downloaded again when a heal needs it.".into());
        out.push(c);
    }
    let st = cfg.join("vllm_selftest");
    if st.is_dir() && read_dir_sorted(&st).iter().next().is_some() {
        out.push(Item::new("heal", "cache_dir", "vLLM self-test reports and files".to_string(), st, false, Exec::Contents));
    }
    let logs = cfg.join("logs").join("jobs");
    if logs.is_dir() {
        let old = old_logs(&logs).len();
        if old > 0 {
            let mut c = Item::new("leftovers", "old_logs", format!("{old} job logs older than {OLD_LOG_DAYS} days"), logs, false, Exec::OldLogs);
            c.suggest = Some("Old logs of finished jobs.".into());
            out.push(c);
        }
    }
    for hub in hf_cache_roots() {
        for e in read_dir_sorted(&hub) {
            let n = e.file_name().to_string_lossy().to_string();
            let Some(rest) = n.strip_prefix("models--") else { continue };
            let repo = rest.replacen("--", "/", 1);
            let mut c = Item::new("downloads", "hf_repo", repo.clone(), e.path(), false, Exec::Tree);
            c.model = true;
            c.model_repo = Some(repo.clone());
            if known.contains(&repo.to_lowercase()) {
                c.status = Status::Referenced;
                c.refs.push("it is in this program's model list (it is downloaded again when it is used)".into());
            } else {
                c.note = Some("This program does not know it: another program may have downloaded it and may still use it.".into());
            }
            c.note = Some(format!("{} Other programs can share this cache (the Hugging Face cache).", c.note.clone().unwrap_or_default()).trim().to_string());
            out.push(c);
        }
    }
    if let Some(v) = ctx.vllm {
        if matches!(v.kind.as_str(), "docker" | "docker_wsl" | "wslc") && !v.model_volume.trim().is_empty() {
            let mut c = Item::new("downloads", "vllm_volume", format!("vLLM engine volume {}", v.model_volume.trim()), PathBuf::from(format!("volume:{}", v.model_volume.trim())), false, Exec::Volume(v.model_volume.trim().to_string()));
            c.model = true;
            c.note = Some("The models vLLM downloaded into its container volume. Size not available. Refused while an engine container runs.".into());
            out.push(c);
        }
    }
}

fn old_logs(dir: &Path) -> Vec<PathBuf> {
    let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(OLD_LOG_DAYS as u64 * 86400);
    read_dir_sorted(dir).into_iter().filter(|e| e.file_type().is_ok_and(|t| t.is_file()) && e.metadata().and_then(|m| m.modified()).is_ok_and(|t| t < cutoff)).map(|e| e.path()).collect()
}

// ------------------------------------------------------------------------------------------------------------------ the scan with sizes

/// The suggestions at the top: the items with a reason, their total, and the preset "keep only the final healed model and the GGUF files".
pub fn suggestions(items: &[Item]) -> Value {
    let list: Vec<&Item> = items.iter().filter(|i| i.suggest.is_some() && i.status != Status::InUse).collect();
    let total: u64 = list.iter().filter_map(|i| i.size).sum();
    let preset: Vec<&Item> = list.iter().copied().filter(|i| matches!(i.kind, "cut_only" | "teacher_cache" | "export_f16" | "export_merged" | "heal_checkpoints" | "leftover")).collect();
    json!({"ids": list.iter().map(|i| i.id.clone()).collect::<Vec<_>>(), "total_bytes": total,
           "preset": {"id": "keep_final", "label": "After a finished Auto-cut keep only the final healed model and the GGUF files", "ids": preset.iter().map(|i| i.id.clone()).collect::<Vec<_>>(), "total_bytes": preset.iter().filter_map(|i| i.size).sum::<u64>()}})
}

/// The whole scan: items first (as they are found, without sizes or with the cached size), then every size as it is measured, then `done` with everything.
pub fn scan(ctx: &Ctx, cancel: &AtomicBool, emit: &mut dyn FnMut(Value)) -> Value {
    let mut items = scan_items(ctx);
    let mut cache = read_size_cache();
    for it in &mut items {
        if let Some(c) = cache.get(&format!("{}|{}", it.path.to_string_lossy(), it.modified.clone().unwrap_or_default())).and_then(|v| v.as_u64()) {
            it.size = Some(c);
            it.size_cached = true;
        }
        emit(json!({"kind": "item", "item": it.json()}));
    }
    for it in &mut items {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        let size = match &it.exec {
            Exec::Volume(_) => None,
            Exec::OldLogs => Some(old_logs(&it.path).iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum()),
            _ => Some(dir_size(&it.path, cancel)),
        };
        it.size = size;
        it.size_cached = false;
        if let Some(s) = size {
            cache.insert(format!("{}|{}", it.path.to_string_lossy(), it.modified.clone().unwrap_or_default()), json!(s));
        }
        emit(json!({"kind": "size", "id": it.id, "size": size}));
    }
    let _ = crate::fsx::write_json_atomic(&size_cache_file(), &Value::Object(cache));
    let free = ctx.project.map(|p| p.folder.clone()).or_else(|| Some(crate::paths::config_dir())).and_then(|d| crate::envmgr::health::free_disk_gb(&d));
    json!({"items": items.iter().map(Item::json).collect::<Vec<_>>(), "suggestions": suggestions(&items), "free_gb": free, "busy": ctx.busy,
           "total_bytes": items.iter().filter_map(|i| i.size).sum::<u64>(), "cancelled": cancel.load(Ordering::SeqCst)})
}

// ------------------------------------------------------------------------------------------------------------------ the deletion

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub dry_run: bool,
    /// the word DELETE typed in the window
    pub confirm: String,
    /// repos / model folders the open Slim or Auto page has chosen; deleting one needs `allow_pinned`
    pub pinned: Vec<String>,
    pub allow_pinned: bool,
    /// the token of the preview (dry run) this deletion was confirmed from: a real deletion without it, or with one that no longer matches the plan, is refused
    pub token: Option<String>,
}

/// A fingerprint of a plan: the ids (which hold the kind, path, modification time and way of deleting) and the stamps, in a fixed order. A deletion needs the token of the preview of the SAME plan,
/// so a call that skipped the preview (a missing or mis-spelled argument, an old window) can never remove anything.
pub fn plan_token(plan: &[Item]) -> String {
    let mut parts: Vec<String> = plan.iter().map(|i| format!("{}|{}|{}", i.id, i.modified.clone().unwrap_or_default(), i.path.to_string_lossy())).collect();
    parts.sort();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut h);
    format!("p{:016x}", h.finish())
}

/// What a request would remove, after every check. Returns the items in the order they go.
pub fn plan_delete(ctx: &Ctx, ids: &[String], opts: &Options) -> Result<Vec<Item>> {
    if ids.is_empty() {
        return Err(Error::new("Nothing is ticked."));
    }
    if ctx.busy {
        return Err(Error::new("A job is running. Wait until it has ended (or stop it): nothing is deleted while one runs."));
    }
    let fresh = scan_items(ctx);
    let roots = Roots::new(ctx.project);
    let mut chosen: Vec<Item> = Vec::new();
    for id in ids {
        let it = fresh.iter().find(|i| &i.id == id).ok_or_else(|| Error::new("The list has changed since it was made (that item is not there any more, or it was changed). Scan again, then choose again."))?;
        if chosen.iter().any(|c| c.id == it.id) {
            continue;
        }
        if it.status == Status::InUse {
            return Err(Error::new(format!("{} is in use and is not deleted: {}.", it.label, it.refs.join("; "))));
        }
        if matches!(it.exec, Exec::Contents) {
            roots.guard_contents(&it.path).map_err(|e| Error::new(format!("{}: {}", it.label, e.0)))?;
        } else if !matches!(it.exec, Exec::Volume(_)) {
            roots.guard(&it.path).map_err(|e| Error::new(format!("{}: {}", it.label, e.0)))?;
        }
        if !opts.allow_pinned && it.model_repo.as_ref().is_some_and(|r| opts.pinned.iter().any(|p| !p.is_empty() && (p == r || canon(Path::new(r)).starts_with(canon(Path::new(p))) || canon(Path::new(p)).starts_with(canon(Path::new(r)))))) {
            return Err(Error::new(format!("{} is the model chosen on the Slim or Auto page right now. Confirm that you want it deleted.", it.label)));
        }
        chosen.push(it.clone());
    }
    // an item inside another chosen item goes with it
    let paths: Vec<PathBuf> = chosen.iter().map(|c| c.path.clone()).collect();
    chosen.retain(|c| !paths.iter().any(|o| *o != c.path && c.path.starts_with(o)));
    Ok(chosen)
}

#[derive(Debug, Clone, Serialize)]
pub struct Failure {
    pub path: String,
    pub error: String,
}

/// Removes `path` below the guard: links are removed themselves, never followed; read-only files are made writable; a file that Windows says is in use is tried a few times.
/// Returns the bytes freed. Stops early with `Err` when `cancel` is set.
pub fn remove_tree(path: &Path, cancel: &AtomicBool, failures: &mut Vec<Failure>) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(path) else { return 0 };
    if cancel.load(Ordering::SeqCst) {
        return 0;
    }
    if is_link(&m) {
        let r = std::fs::remove_file(path).or_else(|_| std::fs::remove_dir(path));
        if let Err(e) = r {
            failures.push(fail(path, &e));
        }
        return 0;
    }
    if m.is_dir() {
        let mut freed = 0;
        match std::fs::read_dir(long(path)) {
            Ok(rd) => {
                for e in rd.flatten() {
                    freed += remove_tree(&e.path(), cancel, failures);
                }
            }
            Err(e) => failures.push(fail(path, &e)),
        }
        if !cancel.load(Ordering::SeqCst) {
            if let Err(e) = retry(|| std::fs::remove_dir(long(path))) {
                failures.push(fail(path, &e));
            }
        }
        return freed;
    }
    let len = m.len();
    let mut perm = m.permissions();
    if perm.readonly() {
        #[allow(clippy::permissions_set_readonly_false)]
        perm.set_readonly(false);
        let _ = std::fs::set_permissions(long(path), perm);
    }
    match retry(|| std::fs::remove_file(long(path))) {
        Ok(()) => len,
        Err(e) => {
            failures.push(fail(path, &e));
            0
        }
    }
}

/// On Windows a long absolute path needs the `\\?\` prefix.
fn long(p: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let s = p.to_string_lossy();
        if p.is_absolute() && !s.starts_with(r"\\?\") && !s.starts_with(r"\\") {
            return PathBuf::from(format!(r"\\?\{s}"));
        }
    }
    p.to_path_buf()
}

fn retry(mut f: impl FnMut() -> std::io::Result<()>) -> std::io::Result<()> {
    let tries = if cfg!(windows) { 6 } else { 1 };
    let mut last = None;
    for n in 0..tries {
        match f() {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                last = Some(e);
                if n + 1 < tries {
                    std::thread::sleep(std::time::Duration::from_millis(300));
                }
            }
        }
    }
    Err(last.unwrap())
}

fn fail(path: &Path, e: &std::io::Error) -> Failure {
    let in_use = cfg!(windows) && matches!(e.raw_os_error(), Some(5) | Some(32) | Some(33));
    let hint = if in_use { " Another program is using it (an antivirus scan, the file explorer, or a program that opened the model). Close it and delete again." } else { "" };
    Failure { path: shorten(path), error: format!("{e}.{hint}") }
}

/// The full deletion: `plan_delete`, then every item one after the other; a failure of one item does not stop the others. `dry_run` only returns the plan.
pub fn delete(ctx: &Ctx, ids: &[String], opts: &Options, cancel: &AtomicBool, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let plan = plan_delete(ctx, ids, opts)?;
    let none = AtomicBool::new(false);
    let sizes: Vec<u64> = plan.iter().map(|i| i.size.unwrap_or_else(|| if matches!(i.exec, Exec::Volume(_)) { 0 } else { dir_size(&i.path, &none) })).collect();
    let total: u64 = sizes.iter().sum();
    let needs_word = plan.iter().any(|c| c.model) || total > TYPE_ABOVE_BYTES;
    let token = plan_token(&plan);
    if !opts.dry_run && opts.token.as_deref() != Some(token.as_str()) {
        return Err(Error::new("Nothing was deleted: this deletion does not come from a preview of the same list (the preview is missing or the folders changed since). Choose again and confirm again."));
    }
    if !opts.dry_run && needs_word && opts.confirm != "DELETE" {
        return Err(Error::new("The word DELETE has to be typed to delete a model or more than 5 GB."));
    }
    if opts.dry_run {
        return Ok(json!({"dry_run": true, "items": plan.iter().zip(&sizes).map(|(i, s)| { let mut j = i.json(); j["size"] = json!(s); j }).collect::<Vec<_>>(), "total_bytes": total, "needs_delete_word": needs_word, "token": token,
                         "warnings": plan.iter().filter(|i| !i.refs.is_empty()).map(|i| format!("{}: {}", i.label, i.refs.join("; "))).collect::<Vec<_>>()}));
    }
    let mut freed = 0u64;
    let mut done = Vec::new();
    let mut failed = Vec::new();
    for (n, it) in plan.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        emit(json!({"kind": "progress", "text": format!("Deleting {} ({} of {})", it.label, n + 1, plan.len()), "value": n as f64 / plan.len() as f64}));
        let mut failures: Vec<Failure> = Vec::new();
        let got = delete_one(ctx, it, cancel, &mut failures);
        match got {
            Ok(bytes) if failures.is_empty() => {
                freed += bytes;
                done.push(json!({"id": it.id, "label": it.label, "freed": bytes}));
                record(ctx, it, bytes);
            }
            Ok(bytes) => {
                freed += bytes;
                failed.push(json!({"id": it.id, "label": it.label, "freed": bytes, "errors": failures}));
            }
            Err(e) => failed.push(json!({"id": it.id, "label": it.label, "errors": [Failure { path: shorten(&it.path), error: e.0 }]})),
        }
    }
    let free = ctx.project.map(|p| p.folder.clone()).or_else(|| Some(crate::paths::config_dir())).and_then(|d| crate::envmgr::health::free_disk_gb(&d));
    Ok(json!({"dry_run": false, "freed_bytes": freed, "deleted": done, "failed": failed, "cancelled": cancel.load(Ordering::SeqCst), "free_gb": free}))
}

fn delete_one(ctx: &Ctx, it: &Item, cancel: &AtomicBool, failures: &mut Vec<Failure>) -> Result<u64> {
    let before = if matches!(it.exec, Exec::Volume(_)) { 0 } else { dir_size(&it.path, &AtomicBool::new(false)) };
    match &it.exec {
        Exec::Tree => {
            Roots::new(ctx.project).guard(&it.path)?;
            remove_tree(&it.path, cancel, failures);
        }
        Exec::Contents => {
            let roots = Roots::new(ctx.project);
            roots.guard_contents(&it.path)?;
            for e in read_dir_sorted(&it.path) {
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                match roots.guard(&e.path()) {
                    Ok(_) => {
                        remove_tree(&e.path(), cancel, failures);
                    }
                    Err(err) => failures.push(fail(&e.path(), &std::io::Error::other(err.0))),
                }
            }
        }
        Exec::ModelEntry(name) => {
            let p = ctx.project.ok_or_else(|| Error::new("No project is open."))?;
            let dir = crate::reshape::runs::removal_checks(p, name)?;
            let roots = Roots::new(Some(p));
            roots.guard(&dir)?;
            crate::reshape::runs::unregister_made_model(p, name)?;
            remove_tree(&dir, cancel, failures);
            crate::reshape::runs::sweep_teacher_caches(p);
        }
        Exec::CutOnly(name) => {
            let p = ctx.project.ok_or_else(|| Error::new("No project is open."))?;
            Roots::new(Some(p)).guard(&it.path)?;
            crate::reshape::runs::remove_cut(p, name)?;
        }
        Exec::QuantLogits(id) => {
            let p = ctx.project.ok_or_else(|| Error::new("No project is open."))?;
            Roots::new(Some(p)).guard(&it.path)?;
            crate::export::quantcheck::delete_logits(p, id)?;
        }
        Exec::OldLogs => {
            Roots::new(ctx.project).guard(&it.path)?;
            for f in old_logs(&it.path) {
                if cancel.load(Ordering::SeqCst) {
                    break;
                }
                remove_tree(&f, cancel, failures);
            }
        }
        Exec::Volume(name) => {
            volume_remove(ctx, name)?;
        }
    }
    let after = if matches!(it.exec, Exec::Volume(_)) { 0 } else { dir_size(&it.path, &AtomicBool::new(false)) };
    Ok(before.saturating_sub(after))
}

// ------------------------------------------------------------------------------------------------------------------ the vLLM volume

/// The command that removes the engine's volume, for the engine kind in the settings (the `wslc` form is a template the person can edit; its syntax is not verified).
pub fn volume_command(s: &crate::settings::VllmSettings, name: &str) -> Option<Vec<String>> {
    let v = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    match s.kind.as_str() {
        "docker" => Some(v(&["docker", "volume", "rm", name])),
        "docker_wsl" => {
            let mut c = v(&["wsl.exe"]);
            if !s.distro.trim().is_empty() {
                c.extend(v(&["-d", s.distro.trim()]));
            }
            c.extend(v(&["--", "docker", "volume", "rm", name]));
            Some(c)
        }
        "wslc" => {
            let t = s.wslc.get("volume_remove").and_then(|x| x.as_str()).unwrap_or("wslc volume remove {name}").replace("{name}", name);
            Some(t.split_whitespace().map(String::from).collect())
        }
        _ => None,
    }
}

/// The command that lists the engine containers (running or not), as JSON lines or a table.
fn containers_command(s: &crate::settings::VllmSettings) -> Option<Vec<String>> {
    let v = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    match s.kind.as_str() {
        "docker" => Some(v(&["docker", "ps", "--filter", "name=ndforge-vllm-", "--format", "{{.Names}}"])),
        "docker_wsl" => {
            let mut c = v(&["wsl.exe"]);
            if !s.distro.trim().is_empty() {
                c.extend(v(&["-d", s.distro.trim()]));
            }
            c.extend(v(&["--", "docker", "ps", "--filter", "name=ndforge-vllm-", "--format", "{{.Names}}"]));
            Some(c)
        }
        "wslc" => Some(v(&["wslc", "list", "--format", "json"])),
        _ => None,
    }
}

fn volume_remove(ctx: &Ctx, name: &str) -> Result<()> {
    let s = ctx.vllm.ok_or_else(|| Error::new("The vLLM engine is not set up."))?;
    if !name.starts_with("ndforge-vllm-") {
        return Err(Error::new("Only volumes named ndforge-vllm-... are touched."));
    }
    let list = containers_command(s).ok_or_else(|| Error::new("This engine kind has no volume to remove."))?;
    let args: Vec<&str> = list.iter().map(String::as_str).collect();
    let (ok, out) = crate::vllm::real_runner(&args).ok_or_else(|| Error::new("The container tool did not answer, so it is not known whether an engine runs. The volume is kept."))?;
    if !ok {
        return Err(Error::new("The container tool refused to list the containers, so it is not known whether an engine runs. The volume is kept."));
    }
    if out.contains("ndforge-vllm-") && (s.kind != "wslc" || out.to_lowercase().contains("running")) {
        return Err(Error::new("A vLLM engine container exists or runs right now. Stop it (or wait until the job has ended), then delete the volume."));
    }
    let cmd = volume_command(s, name).ok_or_else(|| Error::new("This engine kind has no volume to remove."))?;
    let args: Vec<&str> = cmd.iter().map(String::as_str).collect();
    match crate::vllm::real_runner(&args) {
        Some((true, _)) => Ok(()),
        Some((false, out)) => Err(Error::new(format!("The volume could not be removed: {}", out.trim().chars().take(200).collect::<String>()))),
        None => Err(Error::new("The container tool did not answer.")),
    }
}

// ------------------------------------------------------------------------------------------------------------------ what the other screens see

fn ledger_file(p: &Project) -> PathBuf {
    p.folder.join("storage_deleted.json")
}

/// The things deleted on the Storage page, newest last: `[{kind, label, at, bytes}]` (the other screens show "deleted" instead of failing on a missing folder).
pub fn deleted_ledger(p: &Project) -> Vec<Value> {
    read_json(&ledger_file(p)).and_then(|v| v.as_array().cloned()).unwrap_or_default()
}

/// Remembers a deletion: in the project's ledger, and in the Auto-cut runs that point to the model (their `made` entry is `removed`, their report gets a note).
fn record(ctx: &Ctx, it: &Item, bytes: u64) {
    let Some(p) = ctx.project.filter(|_| it.project) else { return };
    let at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut led = deleted_ledger(p);
    led.push(json!({"kind": it.kind, "label": it.label, "at": at, "bytes": bytes}));
    let _ = crate::fsx::write_json_atomic(&ledger_file(p), &Value::Array(led));
    let name = match &it.exec {
        Exec::ModelEntry(n) | Exec::CutOnly(n) => Some(n.clone()),
        _ => None,
    };
    let auto_dir = crate::reshape::auto::auto_dir(p);
    for e in read_dir_sorted(&auto_dir) {
        let dir = e.path();
        let Some(mut st) = crate::reshape::auto::load(&dir) else { continue };
        let mut changed = false;
        if let Some(n) = &name {
            for m in st.made.iter_mut().filter(|m| &m.name == n && !m.removed) {
                if matches!(it.exec, Exec::ModelEntry(_)) || m.kind == "cut" {
                    m.removed = true;
                    changed = true;
                }
            }
        }
        let referenced = name.as_ref().is_some_and(|n| st.made.iter().any(|m| &m.name == n)) || (it.kind.starts_with("export") && st.export.is_some());
        if referenced {
            st.notes.push(format!("Deleted on {at} on the Storage page: {} (the files are gone; this report keeps its numbers).", it.label));
            changed = true;
        }
        if changed {
            let _ = crate::reshape::auto::save(&dir, &mut st);
        }
    }
}
