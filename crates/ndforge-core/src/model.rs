//! The application model behind the window: current project, settings and the actions the pages call. The Tauri shell
//! (`src-tauri`) only forwards calls to it and shows the JSON it returns, so everything here is testable without a window.
use std::path::Path;

use serde_json::{json, Value};

use crate::bundle::open_bundle;
use crate::dataset::builders::{Recipe, Status, BUILDERS};
use crate::dataset::store::{self, Filter};
use crate::error::{Error, Result};
use crate::paths::default_projects_dir;
use crate::project::{BundleCheck, Project};
use crate::settings::Settings;

/// (key, sidebar label)
pub const STEP_LABELS: [(&str, &str); 5] = [
    ("data", "1 · Get world data"),
    ("dataset", "2 · Make training data"),
    ("train", "3 · Train"),
    ("test", "4 · Test"),
    ("use", "5 · Use in nd-world"),
];

pub struct AppModel {
    pub settings: Settings,
    pub project: Option<Project>,
}

impl AppModel {
    pub fn load() -> Self {
        AppModel { settings: Settings::load(), project: None }
    }

    /// Everything the window needs to draw itself.
    pub fn state(&self) -> Value {
        let (progress, unlocked) = match &self.project {
            Some(p) => (p.progress(), p.unlocked()),
            None => ([("data", false), ("dataset", false), ("train", false), ("test", false), ("use", false)],
                     [("data", false), ("dataset", false), ("train", false), ("test", false), ("use", false)]),
        };
        let steps: Vec<Value> = STEP_LABELS.iter().enumerate().map(|(i, (key, label))| {
            json!({"key": key, "label": label, "done": progress[i].1, "unlocked": unlocked[i].1})
        }).collect();
        let bundle = self.project.as_ref().and_then(|p| p.current_bundle()).map(|b| {
            json!({"world": b.world_name(), "lines": b.summary_lines(), "created": b.created()})
        });
        let recent: Vec<Value> = self.settings.existing_recent().iter().map(|f| {
            json!({"folder": f, "name": Path::new(f).file_name().and_then(|n| n.to_str()).unwrap_or(f)})
        }).collect();
        json!({
            "project": self.project.as_ref().map(|p| json!({"name": p.name, "folder": p.folder, "world": p.world_name})),
            "steps": steps,
            "bundle": bundle,
            "recent": recent,
            "settings": {"expert_mode": self.settings.expert_mode, "show_help_panel": self.settings.show_help_panel},
            "projects_dir": default_projects_dir(),
        })
    }

    fn adopt(&mut self, project: Project) {
        self.settings.remember_project(&project.folder);
        self.settings.save();
        self.project = Some(project);
    }

    pub fn create_project(&mut self, name: &str) -> Result<()> {
        let p = Project::create(&default_projects_dir(), name)?;
        self.adopt(p);
        Ok(())
    }

    pub fn open_project(&mut self, folder: &str) -> Result<()> {
        let p = Project::open(folder)?;
        self.adopt(p);
        Ok(())
    }

    /// Verify a file and say what importing it would do: `check.kind` is "first", "update" (with `diff`) or "refused".
    pub fn check_bundle(&self, path: &str) -> Result<Value> {
        let project = self.project.as_ref().ok_or_else(|| Error::new("Create or open a project first (Welcome page)."))?;
        let bundle = open_bundle(path)?;
        let check = match project.check_new_bundle(&bundle)? {
            BundleCheck::First => json!({"kind": "first", "text": "This will be the world data of the project."}),
            BundleCheck::Update(d) => json!({"kind": "update", "text": d.summary()}),
            BundleCheck::Refused(why) => json!({"kind": "refused", "text": why}),
        };
        Ok(json!({"world": bundle.world_name(), "lines": bundle.summary_lines(), "check": check}))
    }

    /// Import the file (what the user confirmed). Returns the change summary, "" for a first import.
    pub fn apply_bundle(&mut self, path: &str) -> Result<String> {
        let project = self.project.as_mut().ok_or_else(|| Error::new("Create or open a project first (Welcome page)."))?;
        let bundle = open_bundle(path)?;
        let summary = match project.check_new_bundle(&bundle)? {
            BundleCheck::Update(d) => d.summary(),
            _ => String::new(),
        };
        project.import_bundle(&bundle)?;
        Ok(summary)
    }

    pub fn set_setting(&mut self, key: &str, value: bool) -> Result<()> {
        match key {
            "expert_mode" => self.settings.expert_mode = value,
            "show_help_panel" => self.settings.show_help_panel = value,
            "welcome_seen" => self.settings.welcome_seen = value,
            other => return Err(Error::new(format!("Unknown setting: {other}"))),
        }
        self.settings.save();
        Ok(())
    }

    // ------------------------------------------------------------------ step 2: training data
    fn project_ref(&self) -> Result<&Project> {
        self.project.as_ref().ok_or_else(|| Error::new("Create or open a project first (Welcome page)."))
    }

    /// The recipe form, the draft's counts and the saved versions.
    pub fn dataset_state(&self) -> Result<Value> {
        let p = self.project_ref()?;
        let recipe = store::load_recipe(p).unwrap_or_default();
        let builders: Vec<Value> = BUILDERS.iter().map(|(id, name)| json!({"id": id, "name": name})).collect();
        Ok(json!({"recipe": recipe, "builders": builders, "summary": store::summary(p)?, "versions": store::versions(p)}))
    }

    pub fn dataset_generate(&self, recipe: Value) -> Result<Value> {
        let p = self.project_ref()?;
        let recipe: Recipe = serde_json::from_value(recipe).map_err(|e| Error::new(format!("The recipe is not valid: {e}")))?;
        store::generate_draft(p, &recipe)
    }

    fn filter_from(v: &Value) -> Filter {
        let text = |k: &str| v[k].as_str().map(str::to_string).filter(|s| !s.is_empty());
        Filter {
            builder: text("builder"), search: text("search"),
            status: match v["status"].as_str() { Some("keep") => Some(Status::Keep), Some("drop") => Some(Status::Drop), _ => None },
        }
    }

    pub fn dataset_examples(&self, filter: &Value, offset: usize, limit: usize) -> Result<Value> {
        let (total, items) = store::list(self.project_ref()?, &Self::filter_from(filter), offset, limit)?;
        Ok(json!({"total": total, "items": items}))
    }

    pub fn dataset_set_status(&self, ids: Vec<String>, status: &str) -> Result<Value> {
        let p = self.project_ref()?;
        store::set_status(p, &ids, if status == "drop" { Status::Drop } else { Status::Keep })?;
        Ok(store::summary(p)?.unwrap_or(Value::Null))
    }

    /// Every example that matches the filter (for "drop all of this kind").
    pub fn dataset_set_status_matching(&self, filter: &Value, status: &str) -> Result<Value> {
        let p = self.project_ref()?;
        store::set_status_matching(p, &Self::filter_from(filter), if status == "drop" { Status::Drop } else { Status::Keep })?;
        Ok(store::summary(p)?.unwrap_or(Value::Null))
    }

    pub fn dataset_edit(&self, id: &str, text: &str) -> Result<()> {
        store::edit(self.project_ref()?, id, text)
    }

    pub fn dataset_save(&self) -> Result<Value> {
        store::save_version(self.project_ref()?)
    }
}
