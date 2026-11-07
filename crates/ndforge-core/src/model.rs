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
use crate::train::catalog::{find, models, ModelEntry};
use crate::train::plan::{plan, recommended_method, Hyper, Method, Preset};
use crate::train::runs;
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
            "settings": {"expert_mode": self.settings.expert_mode, "show_help_panel": self.settings.show_help_panel, "theme": self.settings.theme},
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

    /// "auto", "light" or "dark".
    pub fn set_theme(&mut self, theme: &str) -> Result<()> {
        if !["auto", "light", "dark"].contains(&theme) {
            return Err(Error::new(format!("Unknown theme: {theme}")));
        }
        self.settings.theme = theme.to_string();
        self.settings.save();
        Ok(())
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

    // ------------------------------------------------------------------ step 3: training
    /// Everything the Train page shows: the five models with fit marks for this card, the newest saved data version, the runs.
    pub fn train_state(&self) -> Result<Value> {
        let p = self.project_ref()?;
        let gpus = crate::envmgr::gpu::detect_gpus();
        let gpu = gpus.as_ref().ok().and_then(|g| g.first().cloned());
        let vram = gpu.as_ref().map(|g| g.vram_gb());
        let version = runs::latest_version(p);
        let stats = version.as_ref().map(|(_, m)| runs::data_stats(m)).unwrap_or_default();
        let cards: Vec<Value> = models().iter().map(|m| {
            let one = |method: Method| {
                let pl = plan(m, method, Preset::Normal, vram, stats, runs::load_measured(&m.id, method));
                json!({"est_vram_gb": pl.est_vram_gb, "fit": pl.fit, "measured": runs::load_measured(&m.id, method).map(|(gb, ..)| gb)})
            };
            json!({"model": m, "lora": one(Method::Lora), "qlora": one(Method::Qlora), "recommended": recommended_method(m, vram)})
        }).collect();
        Ok(json!({
            "models": cards,
            "gpu": gpu.as_ref().map(|g| json!({"name": g.name, "vram_gb": (g.vram_gb() * 10.0).round() / 10.0, "precision": g.precision()})),
            "gpu_problem": gpus.err(),
            "env_ready": crate::envmgr::health::env_installed(&crate::envmgr::env_dir()),
            "version": version.map(|(n, m)| json!({"number": n, "counts": m["counts"], "tokens": m["tokens"], "created": m["created"]})),
            "runs": runs::list_runs(p),
        }))
    }

    fn hyper_with(base: Hyper, overrides: &Value) -> Result<Hyper> {
        let mut v = serde_json::to_value(&base)?;
        if let Some(o) = overrides.as_object() {
            for (k, x) in o.iter().filter(|(_, x)| !x.is_null()) {
                if v.get(k).is_none() {
                    return Err(Error::new(format!("Unknown setting: {k}")));
                }
                v[k] = x.clone();
            }
        }
        let h: Hyper = serde_json::from_value(v).map_err(|e| Error::new(format!("A setting is not valid: {e}")))?;
        let bad = |what: &str| Err(Error::new(format!("{what} is out of range.")));
        if !(1..=256).contains(&h.rank) { return bad("The LoRA rank"); }
        if !(h.lr > 0.0 && h.lr <= 0.01) { return bad("The learning rate"); }
        if !(h.epochs > 0.0 && h.epochs <= 20.0) { return bad("The number of epochs"); }
        if !(1..=64).contains(&h.batch) || !(1..=512).contains(&h.grad_accum) { return bad("The batch size"); }
        if !(256..=32768).contains(&h.max_seq) { return bad("The sequence length"); }
        Ok(h)
    }

    pub fn train_plan(&self, model_id: &str, method: Method, preset: Preset) -> Result<Value> {
        let p = self.project_ref()?;
        let m = find(model_id).ok_or_else(|| Error::new("Unknown model."))?;
        let vram = crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.vram_gb()));
        let stats = runs::latest_version(p).map(|(_, mf)| runs::data_stats(&mf)).unwrap_or_default();
        Ok(serde_json::to_value(plan(&m, method, preset, vram, stats, runs::load_measured(&m.id, method)))?)
    }

    /// Everything needed to start (or continue) a run: the project, the model, the settings and the job for the worker.
    pub fn train_job(&self, model_id: &str, method: Method, preset: Preset, overrides: &Value, resume: Option<&str>) -> Result<TrainJob> {
        let p = self.project_ref()?.clone();
        if let Some(name) = resume {
            let cfg: Value = std::fs::read(runs::runs_dir(&p).join(name).join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).ok_or_else(|| Error::new("That run cannot be continued (its settings are missing)."))?;
            let m = find(cfg["model"]["repo"].as_str().and_then(|r| models().into_iter().find(|m| m.repo == r)).map(|m| m.id).as_deref().unwrap_or("")).ok_or_else(|| Error::new("The model of that run is no longer in the catalog."))?;
            let hyper: Hyper = serde_json::from_value(cfg["hyper"].clone()).map_err(|_| Error::new("The settings of that run are damaged."))?;
            let method: Method = serde_json::from_value(cfg["method"].clone()).unwrap_or(Method::Lora);
            let mut job = cfg;
            job["resume"] = json!(true);
            return Ok(TrainJob { project: p, model: m, method, hyper, name: name.to_string(), job });
        }
        let m = find(model_id).ok_or_else(|| Error::new("Choose a model first."))?;
        let (version, manifest) = runs::latest_version(&p).ok_or_else(|| Error::new("Save a version of the training data first (Step 2)."))?;
        let vram = crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.vram_gb()));
        let pl = plan(&m, method, preset, vram, runs::data_stats(&manifest), runs::load_measured(&m.id, method));
        let hyper = Self::hyper_with(pl.hyper, overrides)?;
        let name = runs::run_name(&m.id);
        let job = runs::build_job(&p, &m, method, &hyper, version, &name, false)?;
        Ok(TrainJob { project: p, model: m, method, hyper, name, job })
    }

    pub fn train_metrics(&self, run: &str) -> Result<Value> {
        Ok(Value::Array(runs::read_metrics(self.project_ref()?, run)))
    }
}

/// A run ready to start.
pub struct TrainJob {
    pub project: Project,
    pub model: ModelEntry,
    pub method: Method,
    pub hyper: Hyper,
    pub name: String,
    pub job: Value,
}
