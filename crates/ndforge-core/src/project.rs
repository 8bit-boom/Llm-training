//! A project = one world. Lives in a folder:
//!
//! ```text
//! project.toml             name, world, choices made so far
//! sources/<sha12>.ndtrain  every imported bundle (kept, so a diff to the previous one is possible)
//! datasets/vN/             saved training-data versions   (step 2)
//! runs/<name>/             training runs                  (step 3)
//! exports/                 GGUF / Modelfile               (step 5)
//! ```
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bundle::{open_bundle, Bundle};
use crate::diff::{diff_bundles, same_world, BundleDiff};
use crate::error::{Error, Result};

pub const STEPS: [&str; 5] = ["data", "dataset", "train", "test", "use"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct WorldRef {
    #[serde(default)]
    name: String,
    #[serde(default)]
    slug: String,
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
}

#[derive(Debug, Clone)]
pub struct Project {
    pub folder: PathBuf,
    pub name: String,
    pub world_name: String,
    pub world_slug: String,
    pub current_source: String,
    pub sources: Vec<String>,
    pub created: String,
    /// model, training type, stages, preset ... (filled by later steps)
    pub choices: toml::Table,
}

/// What `check_new_bundle` found out about a file a user wants to import.
#[derive(Debug, Clone)]
pub enum BundleCheck {
    /// the project has no data yet
    First,
    /// a newer file of the same world, with what changed
    Update(BundleDiff),
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
        for sub in ["sources", "datasets", "runs", "exports"] {
            std::fs::create_dir_all(folder.join(sub))?;
        }
        let p = Project {
            folder, name: name.to_string(), world_name: String::new(), world_slug: String::new(), current_source: String::new(),
            sources: Vec::new(), created: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), choices: toml::Table::new(),
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
            created: f.created, choices: f.choices,
        })
    }

    pub fn save(&self) -> Result<()> {
        let f = ProjectFile {
            name: self.name.clone(), created: self.created.clone(),
            world: WorldRef { name: self.world_name.clone(), slug: self.world_slug.clone() },
            source: SourceRef { current: self.current_source.clone(), all: self.sources.clone() },
            choices: self.choices.clone(),
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

    pub fn check_new_bundle(&self, bundle: &Bundle) -> Result<BundleCheck> {
        let Some(cur) = self.current_bundle() else { return Ok(BundleCheck::First) };
        if !same_world(&cur, bundle) {
            return Ok(BundleCheck::Refused(format!(
                "This file is for the world '{}', but this project is for '{}'. Create a new project for a different world.",
                bundle.world_name(), cur.world_name()
            )));
        }
        if bundle.sha256 == cur.sha256 {
            return Ok(BundleCheck::Refused("This is the same file that is already imported.".into()));
        }
        if bundle.created() < cur.created() {
            return Ok(BundleCheck::Refused("This file is older than the one already imported.".into()));
        }
        Ok(BundleCheck::Update(diff_bundles(&cur, bundle)?))
    }

    /// Copy a verified bundle into the project and make it the current snapshot.
    pub fn import_bundle(&mut self, bundle: &Bundle) -> Result<Bundle> {
        if let BundleCheck::Refused(why) = self.check_new_bundle(bundle)? {
            return Err(Error::new(why));
        }
        let name = format!("{}.ndtrain", &bundle.sha256[..12]);
        let dest = self.folder.join("sources").join(&name);
        std::fs::create_dir_all(dest.parent().expect("has a parent"))?;
        if !dest.exists() {
            std::fs::copy(&bundle.path, &dest)?;
        }
        self.current_source = name.clone();
        if !self.sources.contains(&name) {
            self.sources.push(name);
        }
        self.world_name = bundle.world_name();
        self.world_slug = bundle.world_slug();
        self.save()?;
        open_bundle(dest)
    }

    // ---------------------------------------------------------------- progress
    fn has_any(&self, sub: &str, marker: &str) -> bool {
        std::fs::read_dir(self.folder.join(sub)).map(|d| d.flatten().any(|e| marker.is_empty() || e.path().join(marker).exists())).unwrap_or(false)
    }

    /// Which steps are finished; drives the ticks and unlocking in the sidebar.
    pub fn progress(&self) -> [(&'static str, bool); 5] {
        let exports = std::fs::read_dir(self.folder.join("exports")).map(|d| d.flatten().any(|e| e.path().extension().is_some_and(|x| x == "gguf"))).unwrap_or(false);
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
