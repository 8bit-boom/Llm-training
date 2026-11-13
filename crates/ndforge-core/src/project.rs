//! A project = one world. Lives in a folder:
//!
//! ```text
//! project.toml             name, world, choices made so far
//! sources/<sha12>.ndtrain  every imported bundle (kept, so a diff to the previous one is possible)
//!                          a project may hold SEVERAL worlds: the first is the main one, further ones are listed under [[worlds]] in project.toml
//! datasets/vN/             saved training-data versions   (step 2)
//! runs/<name>/             training runs                  (step 3)
//! models/<name>/           slimmed or grown models        ("Slim or grow a model": state.json, result.json, test.json, model/)
//! exports/                 GGUF / Modelfile               (step 5)
//! ```
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bundle::{open_bundle, Bundle};
use crate::diff::{diff_bundles, BundleDiff};
use crate::error::{Error, Result};

pub const STEPS: [&str; 5] = ["data", "dataset", "train", "test", "use"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct WorldRef {
    #[serde(default)]
    name: String,
    #[serde(default)]
    slug: String,
    /// added to the `source` of this world's examples (empty for the first world, "@slug" for the others) so that ids of different worlds never collide
    #[serde(default)]
    suffix: String,
}

/// A further world of a project (the first one lives in the project's own fields).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WorldEntry {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub slug: String,
    #[serde(default)]
    pub suffix: String,
    #[serde(default)]
    pub current: String,
    #[serde(default)]
    pub all: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SourceRef {
    #[serde(default)]
    current: String,
    #[serde(default)]
    all: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ProjectFile {
    #[serde(default)]
    name: String,
    #[serde(default)]
    created: String,
    #[serde(default)]
    world: WorldRef,
    #[serde(default)]
    source: SourceRef,
    #[serde(default)]
    choices: toml::Table,
    #[serde(default)]
    worlds: Vec<WorldEntry>,
}

#[derive(Debug, Clone)]
pub struct Project {
    pub folder: PathBuf,
    pub name: String,
    pub world_name: String,
    pub world_slug: String,
    pub current_source: String,
    pub sources: Vec<String>,
    /// suffix of the first world's example sources (empty unless another world took over as the first one)
    pub world_suffix: String,
    /// the other worlds of the project
    pub extra_worlds: Vec<WorldEntry>,
    pub created: String,
    /// model, training type, stages, preset ... (filled by later steps)
    pub choices: toml::Table,
}

/// What `check_new_bundle` found out about a file a user wants to import.
#[derive(Debug, Clone)]
pub enum BundleCheck {
    /// the project has no data yet
    First,
    /// a newer file of a world the project already has, with what changed
    Update(BundleDiff),
    /// a file of a world the project does not have yet: it would be trained together with the others (the text names the worlds the project has)
    AddWorld(String),
    /// not usable here; the text says why
    Refused(String),
}

impl Project {
    pub fn file(&self) -> PathBuf {
        self.folder.join("project.toml")
    }

    pub fn create(parent: &Path, name: &str) -> Result<Project> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::new("Please give the project a name."));
        }
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect();
        let safe = safe.trim_matches(|c| c == ' ' || c == '.').to_string();
        let folder = parent.join(if safe.is_empty() { "project" } else { &safe });
        if folder.exists() && std::fs::read_dir(&folder)?.next().is_some() {
            return Err(Error::new(format!("The folder {} already exists and is not empty. Choose another name.", folder.display())));
        }
        for sub in ["sources", "datasets", "runs", "models", "exports"] {
            std::fs::create_dir_all(folder.join(sub))?;
        }
        let p = Project {
            folder, name: name.to_string(), world_name: String::new(), world_slug: String::new(), current_source: String::new(),
            sources: Vec::new(), world_suffix: String::new(), extra_worlds: Vec::new(), created: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), choices: toml::Table::new(),
        };
        p.save()?;
        Ok(p)
    }

    pub fn open(folder: impl AsRef<Path>) -> Result<Project> {
        let folder = folder.as_ref();
        let text = match std::fs::read_to_string(folder.join("project.toml")) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::new(format!("{} is not an ND Lore Forge project (no project.toml).", folder.display())))
            }
            Err(e) => return Err(e.into()),
        };
        let f: ProjectFile = toml::from_str(&text).map_err(|e| Error::new(format!("The project file is damaged: {e}")))?;
        Ok(Project {
            folder: folder.to_path_buf(),
            name: if f.name.is_empty() { folder.file_name().and_then(|n| n.to_str()).unwrap_or("project").to_string() } else { f.name },
            world_name: f.world.name, world_slug: f.world.slug, current_source: f.source.current, sources: f.source.all,
            world_suffix: f.world.suffix, extra_worlds: f.worlds, created: f.created, choices: f.choices,
        })
    }

    pub fn save(&self) -> Result<()> {
        let f = ProjectFile {
            name: self.name.clone(), created: self.created.clone(),
            world: WorldRef { name: self.world_name.clone(), slug: self.world_slug.clone(), suffix: self.world_suffix.clone() },
            source: SourceRef { current: self.current_source.clone(), all: self.sources.clone() },
            choices: self.choices.clone(), worlds: self.extra_worlds.clone(),
        };
        let text = toml::to_string_pretty(&f).map_err(|e| Error::new(format!("Could not save the project: {e}")))?;
        let tmp = self.folder.join("project.toml.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(tmp, self.file())?;
        Ok(())
    }

    // ----------------------------------------------------------------- sources
    pub fn current_bundle(&self) -> Option<Bundle> {
        if self.current_source.is_empty() {
            return None;
        }
        open_bundle(self.folder.join("sources").join(&self.current_source)).ok()
    }

    /// Every world of the project, the first one first (only worlds that have data).
    pub fn worlds(&self) -> Vec<WorldEntry> {
        let mut out = Vec::new();
        if !self.current_source.is_empty() {
            out.push(WorldEntry { name: self.world_name.clone(), slug: self.world_slug.clone(), suffix: self.world_suffix.clone(), current: self.current_source.clone(), all: self.sources.clone() });
        }
        out.extend(self.extra_worlds.iter().filter(|w| !w.current.is_empty()).cloned());
        out
    }

    /// The current bundle of every world, with its entry.
    pub fn bundles(&self) -> Vec<(WorldEntry, Bundle)> {
        self.worlds().into_iter().filter_map(|w| open_bundle(self.folder.join("sources").join(&w.current)).ok().map(|b| (w, b))).collect()
    }

    /// Names of the worlds ("Aria", "Aria and Bryn", "Aria, Bryn and Cael").
    pub fn display_name(&self) -> String {
        let names: Vec<String> = self.worlds().into_iter().map(|w| w.name).filter(|n| !n.is_empty()).collect();
        match names.len() {
            0 => self.world_name.clone(),
            1 => names[0].clone(),
            n => format!("{} and {}", names[..n - 1].join(", "), names[n - 1]),
        }
    }

    /// What names the trained model: the world, or the project when it holds several worlds.
    pub fn model_label(&self) -> String {
        if self.worlds().len() > 1 { self.name.clone() } else { self.world_name.clone() }
    }

    /// Identifies the world data a dataset was made from: the file's hash for one world, a hash of all the hashes for several.
    pub fn signature(&self) -> Option<String> {
        use sha2::{Digest, Sha256};
        let b = self.bundles();
        match b.len() {
            0 => None,
            1 => Some(b[0].1.sha256.clone()),
            _ => {
                let mut shas: Vec<String> = b.iter().map(|(w, x)| format!("{}:{}", w.slug, x.sha256)).collect();
                shas.sort();
                Some(format!("{:x}", Sha256::digest(shas.join("|").as_bytes())))
            }
        }
    }

    fn world_of(&self, bundle: &Bundle) -> Option<WorldEntry> {
        self.worlds().into_iter().find(|w| w.slug == bundle.world_slug())
    }

    pub fn check_new_bundle(&self, bundle: &Bundle) -> Result<BundleCheck> {
        if self.current_source.is_empty() {
            return Ok(BundleCheck::First);
        }
        let Some(w) = self.world_of(bundle) else {
            let names: Vec<String> = self.worlds().into_iter().map(|w| format!("'{}'", w.name)).collect();
            return Ok(BundleCheck::AddWorld(format!(
                "'{}' is a different world from {}. It will be added: the model is then trained on all of these worlds at once.", bundle.world_name(), names.join(", ")
            )));
        };
        let cur = open_bundle(self.folder.join("sources").join(&w.current))?;
        if bundle.sha256 == cur.sha256 {
            return Ok(BundleCheck::Refused("This is the same file that is already imported.".into()));
        }
        if bundle.created() < cur.created() {
            return Ok(BundleCheck::Refused("This file is older than the one already imported.".into()));
        }
        Ok(BundleCheck::Update(diff_bundles(&cur, bundle)?))
    }

    /// Copy a verified bundle into the project and make it the current snapshot of its world (a world the project does not have yet is added).
    pub fn import_bundle(&mut self, bundle: &Bundle) -> Result<Bundle> {
        let check = self.check_new_bundle(bundle)?;
        if let BundleCheck::Refused(why) = &check {
            return Err(Error::new(why.clone()));
        }
        let name = format!("{}.ndtrain", &bundle.sha256[..12]);
        let dest = self.folder.join("sources").join(&name);
        std::fs::create_dir_all(dest.parent().expect("has a parent"))?;
        if !dest.exists() {
            std::fs::copy(&bundle.path, &dest)?;
        }
        match check {
            BundleCheck::AddWorld(_) => {
                let slug = bundle.world_slug();
                let mut suffix = format!("@{}", if slug.is_empty() { name[..8].to_string() } else { slug.clone() });
                while self.worlds().iter().any(|w| w.suffix == suffix) {
                    suffix.push('_');
                }
                self.extra_worlds.push(WorldEntry { name: bundle.world_name(), slug, suffix, current: name.clone(), all: vec![name] });
            }
            _ => {
                let slug = bundle.world_slug();
                if self.current_source.is_empty() || self.world_slug == slug {
                    self.current_source = name.clone();
                    if !self.sources.contains(&name) {
                        self.sources.push(name);
                    }
                    self.world_name = bundle.world_name();
                    self.world_slug = slug;
                } else if let Some(w) = self.extra_worlds.iter_mut().find(|w| w.slug == slug) {
                    w.current = name.clone();
                    w.name = bundle.world_name();
                    if !w.all.contains(&name) {
                        w.all.push(name);
                    }
                }
            }
        }
        self.save()?;
        open_bundle(dest)
    }

    /// Takes a world out of the project (its saved files stay on disk). The last world cannot be removed. Returns the removed world's name.
    pub fn remove_world(&mut self, slug: &str) -> Result<String> {
        if self.worlds().len() < 2 {
            return Err(Error::new("A project needs at least one world. Create a new project instead."));
        }
        if self.world_slug == slug {
            let next = self.extra_worlds.remove(0);
            let gone = std::mem::replace(&mut self.world_name, next.name);
            self.world_slug = next.slug;
            self.world_suffix = next.suffix;
            self.current_source = next.current;
            self.sources = next.all;
            self.save()?;
            return Ok(gone);
        }
        let Some(i) = self.extra_worlds.iter().position(|w| w.slug == slug) else { return Err(Error::new("That world is not in this project.")) };
        let gone = self.extra_worlds.remove(i).name;
        self.save()?;
        Ok(gone)
    }

    // ---------------------------------------------------------------- progress
    fn has_any(&self, sub: &str, marker: &str) -> bool {
        std::fs::read_dir(self.folder.join(sub)).map(|d| d.flatten().any(|e| marker.is_empty() || e.path().join(marker).exists())).unwrap_or(false)
    }

    /// Which steps are finished; drives the ticks and unlocking in the sidebar.
    pub fn progress(&self) -> [(&'static str, bool); 5] {
        // an export is a folder exports/<run>-<quant>/ holding model-<quant>.gguf
        let exports = std::fs::read_dir(self.folder.join("exports")).map(|d| d.flatten().any(|e| {
            std::fs::read_dir(e.path()).map(|f| f.flatten().any(|x| x.path().extension().is_some_and(|ext| ext == "gguf"))).unwrap_or(false)
        })).unwrap_or(false);
        [
            ("data", !self.current_source.is_empty()),
            ("dataset", self.has_any("datasets", "manifest.json")),
            ("train", self.has_any("runs", "result.json")),
            ("test", self.has_any("runs", "test.json")),
            ("use", exports),
        ]
    }

    /// A step opens when the steps it needs are done. Testing is optional: "use" opens once something is trained.
    pub fn unlocked(&self) -> [(&'static str, bool); 5] {
        let done = |k: &str| self.progress().iter().any(|(n, d)| *n == k && *d);
        [
            ("data", true),
            ("dataset", done("data")),
            ("train", done("data") && done("dataset")),
            ("test", done("train")),
            ("use", done("train")),
        ]
    }
}
