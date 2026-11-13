//! The application model behind the window: current project, settings and the actions the pages call. The Tauri shell
//! (`src-tauri`) only forwards calls to it and shows the JSON it returns, so everything here is testable without a window.
use std::path::Path;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use crate::bundle::open_bundle;
use crate::dataset::builders::{Recipe, Status, BUILDERS};
use crate::dataset::store::{self, Filter};
use crate::error::{Error, Result};
use crate::paths::default_projects_dir;
use crate::project::{BundleCheck, Project};
use crate::train::catalog::{find, models, ModelEntry};
use crate::train::plan::{full_blockers, full_disk_gb, plan, recommended_method, Hyper, Method, Preset};
use crate::export::modelfile::ollama_name;
use crate::export::runs as exports;
use crate::settings::{OllamaServer, TrainServer};
use crate::remote::run::Target;
use crate::train::{retrain, runs};
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
    /// what each training server reported last time it was checked (its card), by address
    pub remote_status: std::collections::HashMap<String, Value>,
}

impl AppModel {
    pub fn load() -> Self {
        AppModel { settings: Settings::load(), project: None, remote_status: Default::default() }
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
        let worlds: Vec<Value> = self.project.as_ref().map(|p| p.bundles().into_iter().map(|(w, b)| {
            json!({"world": w.name, "slug": w.slug, "lines": b.summary_lines(), "created": b.created()})
        }).collect()).unwrap_or_default();
        json!({
            "project": self.project.as_ref().map(|p| json!({"name": p.name, "folder": p.folder, "world": p.display_name()})),
            "worlds": worlds,
            "steps": steps,
            "bundle": bundle,
            "recent": recent,
            "settings": {"keep_awake": self.settings.keep_awake, "expert_mode": self.settings.expert_mode, "show_help_panel": self.settings.show_help_panel, "theme": self.settings.theme},
            "projects_dir": default_projects_dir(),
        })
    }

    fn adopt(&mut self, project: Project) {
        self.settings.remember_project(&project.folder);
        self.settings.save();
        crate::reshape::runs::mark_interrupted(&project);   // a model half-made when the program closed is marked failed
        crate::train::runs::mark_interrupted(&project);   // a run left "running" by a closed program or a crash can be continued
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
            BundleCheck::AddWorld(text) => json!({"kind": "add", "text": text}),
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

    /// Takes a world out of the project. Returns its name. The data made from it stays in the saved dataset versions; make the examples again in Step 2.
    pub fn remove_world(&mut self, slug: &str) -> Result<String> {
        let project = self.project.as_mut().ok_or_else(|| Error::new("Create or open a project first (Welcome page)."))?;
        project.remove_world(slug)
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
            "keep_awake" => self.settings.keep_awake = value,
            other => return Err(Error::new(format!("Unknown setting: {other}"))),
        }
        self.settings.save();
        Ok(())
    }

    // ------------------------------------------------------------------ step 2: training data
    /// Fails with "open a project first" when there is none (for the window's off-thread calls).
    pub fn reshape_options_ready(&self) -> Result<()> {
        self.project_ref().map(|_| ())
    }

    fn project_ref(&self) -> Result<&Project> {
        self.project.as_ref().ok_or_else(|| Error::new("Create or open a project first (Welcome page)."))
    }

    /// The recipe form, the draft's counts and the saved versions.
    pub fn dataset_state(&self) -> Result<Value> {
        let p = self.project_ref()?;
        let recipe = store::load_recipe(p).unwrap_or_default();
        let builders: Vec<Value> = BUILDERS.iter().map(|(id, name)| json!({"id": id, "name": name})).collect();
        let kinds: Vec<Value> = crate::teacher::KINDS.iter().map(|(id, name, help)| json!({"id": id, "name": name, "help": help})).collect();
        Ok(json!({"recipe": recipe, "builders": builders, "summary": store::summary(p)?, "versions": store::versions(p),
                  "teacher": {"settings": self.settings.teacher, "has_key": crate::secrets::get("teacher").is_some(), "kinds": kinds, "options": crate::teacher::Options::default()}}))
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

    /// Saves the teacher's address, model and key (an empty key keeps the stored one; "-" removes it).
    pub fn teacher_set(&mut self, url: &str, model: &str, key: &str, unload_after: bool) -> Result<()> {
        self.settings.teacher = crate::settings::TeacherSettings { url: url.trim().to_string(), model: model.trim().to_string(), unload_after };
        self.settings.save();
        match key.trim() {
            "" => Ok(()),
            "-" => crate::secrets::remove("teacher"),
            k => crate::secrets::set("teacher", k),
        }
    }

    pub fn teacher_client(&self) -> crate::teacher::Teacher {
        let s = &self.settings.teacher;
        let mut cfg = crate::teacher::TeacherConfig::new(&s.url, &s.model, crate::secrets::get("teacher"));
        cfg.unload_after = s.unload_after;
        crate::teacher::Teacher::new(cfg)
    }

    pub fn teacher_options(v: Value) -> Result<crate::teacher::Options> {
        serde_json::from_value(v).map_err(|e| Error::new(format!("The options are not valid: {e}")))
    }

    pub fn teacher_clear(&self, kind: Option<&str>, forget_cache: bool) -> Result<Value> {
        let p = self.project_ref()?;
        store::clear_teacher(p, kind, forget_cache)?;
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
    pub fn train_state(&self, server: Option<&str>) -> Result<Value> {
        let p = self.project_ref()?;
        let gpus = crate::envmgr::gpu::detect_gpus();
        let gpu = gpus.as_ref().ok().and_then(|g| g.first().cloned());
        // the card the plan is for: this PC's, or the chosen server's as last reported
        let server_gpu = server.and_then(|u| self.remote_status.get(u)).and_then(|st| st["gpu"].as_object().cloned());
        let vram = match server {
            Some(_) => server_gpu.as_ref().and_then(|g| g.get("vram_gb")).and_then(|v| v.as_f64()),
            None => gpu.as_ref().map(|g| g.vram_gb()),
        };
        let version = runs::latest_version(p);
        let stats = version.as_ref().map(|(_, m)| runs::data_stats(m)).unwrap_or_default();
        // does the card have bf16 (needed for full fine-tuning)?
        let bf16 = match server {
            Some(_) => server_gpu.as_ref().and_then(|g| g.get("precision")).and_then(|v| v.as_str()).map(|p| p == "bf16"),
            None => gpu.as_ref().map(|g| g.precision() == crate::envmgr::gpu::Precision::Bf16),
        };
        let cards: Vec<Value> = models().iter().map(|m| {
            let one = |method: Method| {
                let measured = if server.is_some() { None } else { runs::load_measured(&m.id, method) };
                let pl = plan(m, method, Preset::Normal, vram, stats, measured);
                let mut v = json!({"est_vram_gb": pl.est_vram_gb, "fit": pl.fit, "measured": measured.map(|(gb, ..)| gb)});
                if method == Method::Full {
                    let why = full_blockers(m, vram, bf16, server.is_some());
                    v["available"] = json!(why.is_empty());
                    v["reasons"] = json!(why);
                    v["disk_gb"] = json!(full_disk_gb(m));
                }
                v
            };
            json!({"model": m, "lora": one(Method::Lora), "qlora": one(Method::Qlora), "full": one(Method::Full), "recommended": recommended_method(m, vram)})
        }).collect();
        Ok(json!({
            "models": cards,
            "gpu": gpu.as_ref().map(|g| json!({"name": g.name, "vram_gb": (g.vram_gb() * 10.0).round() / 10.0, "precision": g.precision()})),
            "gpu_problem": gpus.err(),
            "env_ready": crate::envmgr::health::env_installed(&crate::envmgr::env_dir()),
            "servers": self.settings.train_servers.iter().map(|s| json!({"name": s.name, "url": s.url, "status": self.remote_status.get(&s.url)})).collect::<Vec<_>>(),
            "version": version.map(|(n, m)| json!({"number": n, "counts": m["counts"], "tokens": m["tokens"], "created": m["created"]})),
            "runs": runs::list_runs(p),
        }))
    }

    /// `try_anyway` in the overrides is not a setting; it switches the "too big" check for full fine-tuning off (Expert mode).
    fn hyper_with(base: Hyper, overrides: &Value) -> Result<Hyper> {
        let mut overrides = overrides.clone();
        if let Some(o) = overrides.as_object_mut() {
            o.remove("try_anyway");
        }
        let overrides = &overrides;
        let mut v = serde_json::to_value(&base)?;
        if let Some(o) = overrides.as_object() {
            for (k, x) in o.iter().filter(|(_, x)| !x.is_null()) {
                if v.get(k).is_none() {
                    return Err(Error::new(format!("Unknown setting: {k}")));
                }
                v[k] = x.clone();
            }
        }
        let mut h: Hyper = serde_json::from_value(v).map_err(|e| Error::new(format!("A setting is not valid: {e}")))?;
        if !["mixed", "book_first"].contains(&h.stages.as_str()) {
            return Err(Error::new("The stages setting must be \"mixed\" or \"book_first\"."));
        }
        if !["auto", "on", "off"].contains(&h.bucket.as_str()) {
            return Err(Error::new("The batching setting must be \"auto\", \"on\" or \"off\"."));
        }
        if !["auto", "unsloth", "standard"].contains(&h.engine.as_str()) {
            return Err(Error::new("The training engine must be \"auto\", \"unsloth\" or \"standard\"."));
        }
        for (name, v) in [("torch.compile", &h.compile), ("Liger", &h.liger), ("padding-free batching", &h.padding_free)] {
            if !["off", "on"].contains(&v.as_str()) {
                return Err(Error::new(format!("The {name} setting must be \"off\" or \"on\".")));
            }
        }
        if h.stages == "book_first" && h.max_steps.is_some() {
            h.stages = "mixed".into();   // a fixed number of steps is a single stage (the quick test)
        }
        if !(h.book_epochs > 0.0 && h.book_epochs <= 20.0) {
            return Err(Error::new("The number of passes over the world book is out of range."));
        }
        let bad = |what: &str| Err(Error::new(format!("{what} is out of range.")));
        if method_needs_rank(&h) && !(1..=256).contains(&h.rank) { return bad("The LoRA rank"); }
        if !(h.lr > 0.0 && h.lr <= 0.01) { return bad("The learning rate"); }
        if !(h.epochs > 0.0 && h.epochs <= 20.0) { return bad("The number of epochs"); }
        if !(1..=64).contains(&h.batch) || !(1..=512).contains(&h.grad_accum) { return bad("The batch size"); }
        if !(256..=32768).contains(&h.max_seq) { return bad("The sequence length"); }
        Ok(h)
    }

    pub fn train_plan(&self, model_id: &str, method: Method, preset: Preset, server: Option<&str>, stages: Option<&str>) -> Result<Value> {
        let p = self.project_ref()?;
        let m = find(model_id).ok_or_else(|| Error::new("Unknown model."))?;
        let vram = match server {
            Some(u) => self.remote_status.get(u).and_then(|st| st["gpu"]["vram_gb"].as_f64()),
            None => crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.vram_gb())),
        };
        let stats = runs::latest_version(p).map(|(_, mf)| runs::data_stats(&mf)).unwrap_or_default();
        let measured = if server.is_some() { None } else { runs::load_measured(&m.id, method) };
        let mut pl = plan(&m, method, preset, vram, stats, measured);
        if server.is_none() && measured.is_none() && runs::backend_for("auto") == "hf" {
            pl.notes.push("This computer trains without Unsloth (the standard libraries). The memory figure above comes from Unsloth's guides; without it the program can use more, and a run is slower. A Quick test measures the real figure.".to_string());
        }
        // the stage choice changes only the number of steps (a fixed-step quick test is always one stage)
        if stages == Some("book_first") && pl.hyper.max_steps.is_none() {
            pl.hyper.stages = "book_first".into();
            pl.steps = crate::train::plan::steps_for(&pl.hyper, stats);
        }
        Ok(serde_json::to_value(pl)?)
    }

    /// Everything needed to start (or continue) a run: the project, the model, the settings and the job for the worker.
    pub fn train_job(&self, model_id: &str, method: Method, preset: Preset, overrides: &Value, resume: Option<&str>, server: Option<&str>) -> Result<TrainJob> {
        let p = self.project_ref()?.clone();
        if let Some(name) = resume {
            let cfg: Value = std::fs::read(runs::runs_dir(&p).join(name).join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).ok_or_else(|| Error::new("That run cannot be continued (its settings are missing)."))?;
            let m = find(cfg["model"]["repo"].as_str().and_then(|r| models().into_iter().find(|m| m.repo == r)).map(|m| m.id).as_deref().unwrap_or("")).ok_or_else(|| Error::new("The model of that run is no longer in the catalog."))?;
            let hyper: Hyper = serde_json::from_value(cfg["hyper"].clone()).map_err(|_| Error::new("The settings of that run are damaged."))?;
            let method: Method = serde_json::from_value(cfg["method"].clone()).unwrap_or(Method::Lora);
            let remote = match cfg["remote"]["url"].as_str() {
                Some(url) => Some(self.target_for(url)?),
                None => None,
            };
            let mut job = cfg;
            job["resume"] = json!(true);
            return Ok(TrainJob { project: p, model: m, method, hyper, name: name.to_string(), job, remote });
        }
        let m = find(model_id).ok_or_else(|| Error::new("Choose a model first."))?;
        if server.is_some() && std::path::Path::new(&m.repo).is_absolute() {
            return Err(Error::new(format!("{} is a folder on this computer, so it cannot be trained on a server. Train it on this computer, or add the model from Hugging Face instead.", m.name)));
        }
        let (version, manifest) = runs::latest_version(&p).ok_or_else(|| Error::new("Save a version of the training data first (Step 2)."))?;
        let remote = match server { Some(u) => Some(self.target_for(u)?), None => None };
        if method == Method::Full {
            let try_anyway = overrides["try_anyway"] == true;
            let bf16 = match server {
                Some(u) => self.remote_status.get(u).and_then(|st| st["gpu"]["precision"].as_str()).map(|p| p == "bf16"),
                None => crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.precision() == crate::envmgr::gpu::Precision::Bf16)),
            };
            let vram_for_check = match server {
                Some(u) => self.remote_status.get(u).and_then(|st| st["gpu"]["vram_gb"].as_f64()),
                None => crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.vram_gb())),
            };
            let mut why = full_blockers(&m, vram_for_check, bf16, server.is_some());
            if try_anyway {
                why.retain(|w| w.contains("needs about"));   // only the memory warning can be overridden; remote and bf16 cannot
            }
            if !why.is_empty() {
                return Err(Error::new(format!("Full fine-tuning is not possible here: {}", why.join(" "))));
            }
            if let Some(free) = crate::envmgr::health::free_disk_gb(&p.folder) {
                if free < full_disk_gb(&m) {
                    return Err(Error::new(format!("Full fine-tuning of {} needs about {:.0} GB of free disk space for its checkpoints; only {:.0} GB are free.", m.name, full_disk_gb(&m), free)));
                }
            }
        }
        let vram = match server {
            Some(u) => self.remote_status.get(u).and_then(|st| st["gpu"]["vram_gb"].as_f64()),
            None => crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.vram_gb())),
        };
        let measured = if server.is_some() { None } else { runs::load_measured(&m.id, method) };
        let pl = plan(&m, method, preset, vram, runs::data_stats(&manifest), measured);
        let hyper = Self::hyper_with(pl.hyper, overrides)?;
        let name = runs::run_name(&m.id);
        let job = runs::build_job(&p, &m, method, &hyper, version, &name, false)?;
        Ok(TrainJob { project: p, model: m, method, hyper, name, job, remote })
    }

    /// The settings the benchmark starts from (this computer only: a server's card is benchmarked by training there).
    pub fn bench_job(&self, model_id: &str, method: Method, preset: Preset, overrides: &Value) -> Result<TrainJob> {
        self.train_job(model_id, method, preset, overrides, None, None)
    }

    pub fn bench_last(&self, model_id: &str, method: Method) -> Result<Value> {
        let m = find(model_id).ok_or_else(|| Error::new("Choose a model first."))?;
        Ok(crate::train::bench::last(self.project_ref()?, &m, method).unwrap_or(Value::Null))
    }

    // ------------------------------------------------------------------ slim or grow a model
    /// Everything the Slim or grow page shows before a model is chosen: the model list, what was made, and what this PC has.
    pub fn reshape_state(&self) -> Result<Value> {
        let p = self.project_ref()?;
        let gpu = crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().cloned());
        Ok(json!({
            "models": models().iter().map(|m| json!({"id": m.id, "name": m.name, "family": m.family, "params_b": m.params_b, "custom": m.custom, "reshape": m.reshape, "derived_from": m.derived_from})).collect::<Vec<_>>(),
            "made": crate::reshape::runs::list(p),
            "dataset_version": runs::latest_version(p).map(|(n, _)| n),
            "env_ready": crate::envmgr::health::env_installed(&crate::envmgr::env_dir()),
            "vram_gb": gpu.as_ref().map(|g| (g.vram_gb() * 10.0).round() / 10.0),
            "gpu": gpu.as_ref().map(|g| g.name.clone()),
            "free_disk_gb": crate::envmgr::health::free_disk_gb(&p.folder).map(|g| (g * 10.0).round() / 10.0),
            "unsloth": crate::train::bench::env_has_unsloth(),
        }))
    }

    /// What can be done with one model: its layout, every feasible size for `mode` ("prune" or "grow") and what to know first. May ask the worker (and the
    /// internet) the first time, so the window calls it off the main thread.
    pub fn reshape_options(&self, model_id: &str, mode: &str) -> Result<Value> {
        self.project_ref()?;
        Self::reshape_options_for(model_id, mode)
    }

    /// `reshape_options` without the project, so the window can run it on a thread of its own.
    pub fn reshape_options_for(model_id: &str, mode: &str) -> Result<Value> {
        use crate::reshape::{plan, runs as rruns};
        let m = find(model_id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
        let mode = match mode {
            "prune" => plan::Mode::Prune,
            "grow" => plan::Mode::Grow,
            other => return Err(Error::new(format!("Unknown mode: {other}"))),
        };
        let gpu = crate::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().cloned());
        let vram = gpu.as_ref().map(|g| g.vram_gb());
        let ins = rruns::inspect(&m, &AtomicBool::new(false), &mut |_| {})?;
        let options = plan::options(&ins, mode, vram);
        // the first choice the page offers: a quarter of the blocks (at least one), within what is possible
        let blocks = options.last().map(|o| (o.blocks).min((ins.blocks / 4).max(1))).unwrap_or(0);
        let spec = plan::ReshapeSpec { source_model_id: m.id.clone(), mode, blocks, strategy: "contiguous".into(), protect_first: 1, protect_last: 1, init: "identity".into(), calib_samples: 128, calib_len: 512, load: "auto".into(), drop_layers: None };
        Ok(json!({
            "model": {"id": m.id, "name": m.name, "params_b": m.params_b}, "inspect": ins, "mode": mode.key(), "options": options,
            "load": plan::scoring_load(ins.total_params, ins.bytes_per_param, vram),
            "disk_gb": options.iter().map(|o| (o.blocks, plan::disk_needed_gb(&ins, mode, o.blocks))).collect::<Vec<_>>(),
            "default_blocks": blocks,
            "warnings": if ins.supported && blocks > 0 { plan::warnings(&spec, &ins, vram) } else { Vec::new() },
        }))
    }

    pub fn train_metrics(&self, run: &str) -> Result<Value> {
        Ok(Value::Array(runs::read_metrics(self.project_ref()?, run)))
    }
}

impl AppModel {
    // ------------------------------------------------------------------ step 5: use in nd-world
    pub fn export_state(&self) -> Result<Value> {
        let p = self.project_ref()?;
        Ok(json!({
            "runs": exports::exportable_runs(p),
            "exports": exports::list_exports(p),
            "quants": exports::QUANTS.iter().map(|(id, name)| json!({"id": id, "name": name})).collect::<Vec<_>>(),
            "servers": self.settings.ollama_servers,
            "suggested_name": ollama_name(&p.model_label(), ""),
            "world": p.display_name(),
            "env_ready": crate::envmgr::health::env_installed(&crate::envmgr::env_dir()),
        }))
    }

    // ------------------------------------------------------------------ step 4: test
    /// Runs that can be tested (they saved a model), the newest test of each, and the comparison table.
    pub fn test_state(&self) -> Result<Value> {
        let p = self.project_ref()?;
        let runs: Vec<Value> = crate::evaluate::runs::testable(p);
        let t = &self.settings.teacher;
        Ok(json!({
            "runs": runs, "compare": crate::evaluate::runs::compare(p), "env_ready": crate::envmgr::health::env_installed(&crate::envmgr::env_dir()),
            "teacher_ready": !t.url.trim().is_empty() && !t.model.trim().is_empty(), "teacher_model": t.model, "world": p.display_name(),
            "default_system": crate::dataset::builders::system_prompt(&p.display_name()),
        }))
    }

    pub fn test_result(&self, run: &str) -> Result<Value> {
        Ok(crate::evaluate::runs::read_test(self.project_ref()?, run).unwrap_or(Value::Null))
    }

    pub fn test_save_transcript(&self, run: &str, items: &[Value]) -> Result<String> {
        Ok(crate::evaluate::runs::save_transcript(self.project_ref()?, run, items)?.to_string_lossy().to_string())
    }

    /// Plain text for a bug report (no keys, no tokens).
    pub fn problem_report(&self) -> String {
        crate::diagnose::report(self.project.as_ref(), &self.settings)
    }

    // ------------------------------------------------------------------ models from Hugging Face
    pub fn custom_model_lookup(input: &str) -> Result<Value> {
        let f = crate::train::hub::lookup(input)?;
        Ok(json!({"model": f.entry, "warnings": f.warnings, "input": input.trim()}))
    }

    /// Adds the model (looked up again, so only a model that passes the checks is ever saved). Returns its id.
    pub fn custom_model_add(input: &str) -> Result<String> {
        let f = crate::train::hub::lookup(input)?;
        crate::train::catalog::add_custom(f.entry.clone())?;
        Ok(f.entry.id)
    }

    pub fn custom_model_remove(id: &str) -> Result<()> {
        crate::train::catalog::remove_custom(id)
    }

    pub fn project_clone(&self) -> Result<Project> {
        Ok(self.project_ref()?.clone())
    }

    pub fn add_ollama_server(&mut self, name: &str, url: &str) -> Result<()> {
        let (name, url) = (name.trim(), crate::export::ollama::normalize_url(url));
        if name.is_empty() || url == "http://" {
            return Err(Error::new("Give the server a name and an address (for example http://truenas:11434)."));
        }
        self.settings.ollama_servers.retain(|s| s.url != url);
        self.settings.ollama_servers.push(OllamaServer { name: name.to_string(), url });
        self.settings.save();
        Ok(())
    }

    pub fn remove_ollama_server(&mut self, url: &str) -> Result<()> {
        self.settings.ollama_servers.retain(|s| s.url != url);
        self.settings.save();
        Ok(())
    }

    /// (path of the GGUF, family, context length) of an export, for installing it.
    pub fn export_for_install(&self, export: &str) -> Result<(std::path::PathBuf, String, u32)> {
        let (gguf, meta) = exports::find_export(self.project_ref()?, export)?;
        Ok((gguf, meta["family"].as_str().unwrap_or("generic").to_string(), meta["num_ctx"].as_u64().unwrap_or(8192) as u32))
    }

    pub fn record_install(&self, export: &str, url: &str, model: &str) -> Result<()> {
        exports::record_install(self.project_ref()?, export, url, model)
    }

    /// A question to ask the installed model: one of the training's own sample questions (with its system message), else a generic one.
    pub fn probe_question(&self) -> Result<(String, String)> {
        let p = self.project_ref()?;
        if let Some((n, _)) = runs::latest_version(p) {
            let train = p.folder.join("datasets").join(format!("v{n}")).join("train.jsonl");
            if let Some(first) = runs::probes_from_dataset(&train, 1).first() {
                return Ok((first["system"].as_str().unwrap_or("").to_string(), first["question"].as_str().unwrap_or("").to_string()));
            }
        }
        Ok((format!("You are the game master's assistant for the tabletop world \"{}\".", p.display_name()), "Tell me about this world.".to_string()))
    }

    // ------------------------------------------------------------------ training servers
    pub fn add_train_server(&mut self, name: &str, url: &str, token: &str) -> Result<()> {
        let (name, url) = (name.trim(), crate::export::ollama::normalize_url(url));
        if name.is_empty() || url == "http://" || token.trim().is_empty() {
            return Err(Error::new("Give the server a name, an address (for example http://truenas:8765) and the access token (NDFORGE_TOKEN of the worker)."));
        }
        self.settings.train_servers.retain(|s| s.url != url);
        self.settings.train_servers.push(TrainServer { name: name.to_string(), url: url.clone() });
        self.settings.save();
        crate::secrets::set(&url, token.trim())
    }

    pub fn remove_train_server(&mut self, url: &str) -> Result<()> {
        self.settings.train_servers.retain(|s| s.url != url);
        self.remote_status.remove(url);
        self.settings.save();
        crate::secrets::remove(url)
    }

    /// name, address and token of a server the user added.
    pub fn target_for(&self, url: &str) -> Result<Target> {
        let s = self.settings.train_servers.iter().find(|s| s.url == url).ok_or_else(|| Error::new("That training server is no longer in the list. Add it again."))?;
        let token = crate::secrets::get(url).ok_or_else(|| Error::new("The access token of that server is missing. Add the server again."))?;
        Ok(Target { name: s.name.clone(), url: s.url.clone(), token })
    }

    pub fn set_remote_status(&mut self, url: &str, status: Value) {
        self.remote_status.insert(url.to_string(), status);
    }

    // ------------------------------------------------------------------ retrain with the new export
    pub fn retrain_info(&self) -> Result<Value> {
        retrain::info(self.project_ref()?)
    }

    /// Makes the new data version and returns the training run to start. `from_previous`: continue from the last model with a short run.
    pub fn retrain_job(&self, from_previous: bool) -> Result<TrainJob> {
        let p = self.project_ref()?.clone();
        let (version, cfg, hyper, last_name) = retrain::prepare(&p, from_previous)?;
        let m = models().into_iter().find(|m| Some(m.repo.as_str()) == cfg["model"]["repo"].as_str()).ok_or_else(|| Error::new("The model of the last run is no longer in the catalog."))?;
        let method: Method = serde_json::from_value(cfg["method"].clone()).unwrap_or(Method::Lora);
        let name = runs::run_name(&m.id);
        let mut job = runs::build_job(&p, &m, method, &hyper, version, &name, false)?;
        if from_previous {
            if method == Method::Full {
                job["init_model"] = json!(runs::runs_dir(&p).join(&last_name).join("model"));
            } else {
                job["init_adapter"] = json!(runs::runs_dir(&p).join(&last_name).join("adapter"));
            }
        }
        // a run that was made on a server is retrained there
        let remote = match cfg["remote"]["url"].as_str() { Some(url) => Some(self.target_for(url)?), None => None };
        Ok(TrainJob { project: p, model: m, method, hyper, name, job, remote })
    }
}

fn method_needs_rank(h: &Hyper) -> bool {
    h.rank > 0 || h.alpha > 0   // a full fine-tune has rank 0 (unused)
}

/// A run ready to start.
#[derive(Debug)]
pub struct TrainJob {
    pub remote: Option<Target>,
    pub project: Project,
    pub model: ModelEntry,
    pub method: Method,
    pub hyper: Hyper,
    pub name: String,
    pub job: Value,
}
