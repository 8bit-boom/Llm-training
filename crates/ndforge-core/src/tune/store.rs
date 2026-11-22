//! Where presets live and how they are found again: the project (`<project>/tune/presets/`), the app-wide library (`<config folder>/tune_library/`), export and import, and the match of a preset to a model
//! on this machine (exact, same shape, from a parent, nearest size, another PC), with the staleness of every result.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use super::identity::{major_of, minor_of, MachineId, ModelIdentity};
use super::preset::{Preset, FORMAT};
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Project,
    Library,
}

impl Scope {
    pub fn parse(s: &str) -> Result<Scope> {
        match s {
            "project" => Ok(Scope::Project),
            "library" => Ok(Scope::Library),
            _ => Err(Error::new("The scope is project or library.")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Store {
    pub project: Option<PathBuf>,
    pub library: PathBuf,
}

/// A preset file that could not be read: kept under another name, listed so that nothing disappears silently.
#[derive(Debug, Clone, Serialize)]
pub struct Unreadable {
    pub file: String,
    pub kept_as: Option<String>,
    pub why: String,
}

fn valid_id(id: &str) -> bool {
    id.len() >= 3 && id.len() <= 40 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

impl Store {
    /// `project`: the project folder (None: only the library).
    pub fn new(project: Option<&Path>) -> Store {
        Store { project: project.map(Path::to_path_buf), library: crate::paths::config_dir().join("tune_library") }
    }

    pub fn dir(&self, scope: Scope) -> Result<PathBuf> {
        match scope {
            Scope::Project => Ok(self.project.as_ref().ok_or_else(|| Error::new("No project is open."))?.join("tune").join("presets")),
            Scope::Library => Ok(self.library.clone()),
        }
    }

    fn file(&self, scope: Scope, id: &str) -> Result<PathBuf> {
        if !valid_id(id) {
            return Err(Error::new("That is not a preset id."));
        }
        Ok(self.dir(scope)?.join(format!("{id}.json")))
    }

    /// Every readable preset of a scope, newest first, and the files that could not be read (moved aside as `.broken-<time>`).
    pub fn list(&self, scope: Scope) -> (Vec<Preset>, Vec<Unreadable>) {
        let Ok(dir) = self.dir(scope) else { return (Vec::new(), Vec::new()) };
        let mut ok = Vec::new();
        let mut bad = Vec::new();
        let Ok(rd) = std::fs::read_dir(&dir) else { return (ok, bad) };
        for e in rd.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if !name.ends_with(".json") {
                continue;
            }
            match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<Preset>(&b).map_err(|e| e.to_string())) {
                Ok(p) if p.format <= FORMAT && !p.id.is_empty() => ok.push(p),
                Ok(p) => bad.push(Unreadable { file: name, kept_as: None, why: format!("it is a newer format ({}) than this program reads ({FORMAT})", p.format) }),
                Err(why) => {
                    let kept = crate::fsx::set_aside_broken(&path).map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string());
                    bad.push(Unreadable { file: name, kept_as: kept, why });
                }
            }
        }
        ok.sort_by(|a, b| b.created.cmp(&a.created).then(a.id.cmp(&b.id)));
        (ok, bad)
    }

    pub fn get(&self, scope: Scope, id: &str) -> Result<Preset> {
        let f = self.file(scope, id)?;
        let b = std::fs::read(&f).map_err(|_| Error::new("That preset is not there any more."))?;
        serde_json::from_slice(&b).map_err(|e| Error::new(format!("That preset could not be read: {e}")))
    }

    pub fn save(&self, scope: Scope, p: &Preset) -> Result<PathBuf> {
        if p.name.trim().is_empty() || p.name.chars().count() > 120 {
            return Err(Error::new("Give the preset a name (up to 120 characters)."));
        }
        let f = self.file(scope, &p.id)?;
        crate::fsx::write_json_atomic(&f, p)?;
        Ok(f)
    }

    pub fn rename(&self, scope: Scope, id: &str, name: &str) -> Result<Preset> {
        let mut p = self.get(scope, id)?;
        p.name = name.trim().to_string();
        self.save(scope, &p)?;
        Ok(p)
    }

    /// A copy with another name and its own id (the creation time of the copy is `now`); the numbers are the same.
    pub fn duplicate(&self, scope: Scope, id: &str, name: &str, now: &str) -> Result<Preset> {
        let mut p = self.get(scope, id)?;
        p.name = name.trim().to_string();
        p.created = now.to_string();
        p.id = super::preset::preset_id(&format!("{}|copy|{}", p.model.weights_id, id), &p.machine.key, now);
        self.save(scope, &p)?;
        Ok(p)
    }

    /// Deletes a preset; `confirm` must be its exact name.
    pub fn delete(&self, scope: Scope, id: &str, confirm: &str) -> Result<()> {
        let p = self.get(scope, id)?;
        if confirm != p.name {
            return Err(Error::new("To delete a preset type its exact name. Nothing was deleted."));
        }
        std::fs::remove_file(self.file(scope, id)?)?;
        Ok(())
    }

    /// Writes `{"ndtune": 1, "preset": {...}}` to `path` (the home folder in the text is replaced by `~`).
    pub fn export(&self, scope: Scope, id: &str, path: &Path) -> Result<()> {
        let p = self.get(scope, id)?;
        let text = serde_json::to_string_pretty(&serde_json::json!({"ndtune": FORMAT, "preset": p})).map_err(|e| Error::new(e.to_string()))?;
        let home = dirs::home_dir().map(|h| h.to_string_lossy().to_string()).filter(|h| h.len() > 3);
        let text = match home {
            Some(h) => text.replace(&h, "~").replace(&h.replace('\\', "\\\\"), "~"),
            None => text,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        crate::fsx::write_atomic(path, text.as_bytes())
    }

    /// Reads an exported preset into the library. A newer format is refused; an id that exists needs `overwrite`.
    pub fn import(&self, path: &Path, overwrite: bool) -> Result<Preset> {
        let v: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| Error::new(format!("The file could not be read: {e}")))?).map_err(|_| Error::new("That is not a preset file."))?;
        let major = v["ndtune"].as_u64().ok_or_else(|| Error::new("That is not a preset file."))?;
        if major > FORMAT as u64 {
            return Err(Error::new(format!("That preset was made by a newer version of this program (format {major}); this one reads format {FORMAT}.")));
        }
        let p: Preset = serde_json::from_value(v["preset"].clone()).map_err(|e| Error::new(format!("That preset file is damaged: {e}")))?;
        if !valid_id(&p.id) {
            return Err(Error::new("That preset file has no valid id."));
        }
        if self.file(Scope::Library, &p.id)?.exists() && !overwrite {
            return Err(Error::new(format!("A preset with that id ({}) is already in the library. Import with overwrite to replace it.", p.name)));
        }
        self.save(Scope::Library, &p)?;
        Ok(p)
    }
}

// ------------------------------------------------------------------------------------------------------------------ matching

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    Exact,
    SameShape,
    Lineage,
    Nearest,
    OtherPc,
}

impl MatchKind {
    /// Whether the WHOLE preset may be used (an exact or same-shape match) or only the knobs a function marks as transferable.
    pub fn whole(self) -> bool {
        matches!(self, MatchKind::Exact | MatchKind::SameShape)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Stale {
    pub function: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Match {
    pub scope: Scope,
    pub kind: MatchKind,
    pub note: String,
    pub stale: Vec<Stale>,
    pub preset: Preset,
}

/// The value of one `depends_on` component on a machine: `driver_major`, `os_major`, `<package>_minor` or a plain software entry; None when unknown.
fn component(m: &MachineId, c: &str) -> Option<String> {
    match c {
        "gpu" => Some(m.key.clone()),
        "driver_major" => Some(major_of(&m.driver)).filter(|s| !s.is_empty()),
        "os_major" => m.os.split_whitespace().next().map(String::from),
        _ => match c.strip_suffix("_minor") {
            Some(pkg) => m.software.get(pkg).map(|v| minor_of(v)),
            None => m.software.get(c).cloned(),
        },
    }
}

/// For every function of a preset: what changed on this machine since the test, in words (an unknown component is not a change).
pub fn stale_of(p: &Preset, now: &MachineId) -> Vec<Stale> {
    let mut out = Vec::new();
    for (name, f) in &p.functions {
        let mut reasons = Vec::new();
        for c in &f.depends_on {
            if c == "gpu" {
                continue;
            }
            if let (Some(a), Some(b)) = (component(&p.machine, c), component(now, c)) {
                if a != b {
                    reasons.push(format!("{} {} -> {} since the test", c.trim_end_matches("_minor").replace('_', " "), a, b));
                }
            }
        }
        if !reasons.is_empty() {
            out.push(Stale { function: name.clone(), reasons });
        }
    }
    out
}

/// How one preset relates to the model in front of us on this machine, or None when it is of no use.
pub fn kind_of(p: &Preset, model: &ModelIdentity, machine: &MachineId) -> Option<(MatchKind, String)> {
    if p.machine.key != machine.key {
        return Some((MatchKind::OtherPc, format!("measured on {} ({} MB): another PC, not offered", p.machine.gpu, p.machine.vram_mb)));
    }
    if !p.model.weights_id.is_empty() && p.model.weights_id == model.weights_id {
        return Some((MatchKind::Exact, "tested for this model".into()));
    }
    if !p.model.shape_id.is_empty() && p.model.shape_id == model.shape_id {
        return Some((MatchKind::SameShape, format!("measured on {} with the same shape: the speed should be the same", p.model.name)));
    }
    if model.lineage.iter().any(|l| *l == p.model.name || *l == p.model.weights_id) {
        return Some((MatchKind::Lineage, format!("measured on {} ({:.1} B); this one is {:.1} B: only the settings that carry over are offered", p.model.name, p.model.params as f64 / 1e9, model.params as f64 / 1e9)));
    }
    if p.model.arch == model.arch && p.model.params > 0 && model.params > 0 {
        let r = model.params as f64 / p.model.params as f64;
        if (0.67..=1.5).contains(&r) {
            return Some((MatchKind::Nearest, format!("measured on {} ({:.1} B), the same architecture and about this size: re-test recommended", p.model.name, p.model.params as f64 / 1e9)));
        }
    }
    None
}

/// The presets that fit `model` on `machine`, best match first (exact, same shape, parent, nearest, another PC); inside a kind the project before the library, then the newest. An id in both scopes
/// is listed once (the project's).
pub fn find(store: &Store, model: &ModelIdentity, machine: &MachineId) -> Vec<Match> {
    let mut out: Vec<Match> = Vec::new();
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for scope in [Scope::Project, Scope::Library] {
        for p in store.list(scope).0 {
            if seen.contains_key(&p.id) {
                continue;
            }
            if let Some((kind, note)) = kind_of(&p, model, machine) {
                seen.insert(p.id.clone(), ());
                let stale = stale_of(&p, machine);
                out.push(Match { scope, kind, note, stale, preset: p });
            }
        }
    }
    let rank = |k: MatchKind| match k { MatchKind::Exact => 0, MatchKind::SameShape => 1, MatchKind::Lineage => 2, MatchKind::Nearest => 3, MatchKind::OtherPc => 4 };
    out.sort_by(|a, b| rank(a.kind).cmp(&rank(b.kind)).then((a.scope as u8).cmp(&(b.scope as u8))).then(b.preset.created.cmp(&a.preset.created)));
    out
}
