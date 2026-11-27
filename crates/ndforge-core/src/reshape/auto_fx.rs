//! Auto-cut's real effects (`auto::Effects`): every step calls the job the Slim page would call, with the Stop button wired to the same flag.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::auto::{CutRef, Effects, HealRef, Made, ModelRef, Out, Student};
use super::heal_source::{self, Source};
use super::plan::{Inspect, ReasoningCalib, ReshapeSpec};
use super::runs::{self, HealSpec, TrySpec};
use super::thinkcheck::{self, ThinkSpec};
use crate::error::{Error, Result};
use crate::export::runs::{ImatrixOptions, MixedMeasured, MixedOptions};
use crate::project::Project;
use crate::train::catalog;

/// How long a stopped job gets to end by itself before it is killed.
const GRACE: Duration = Duration::from_secs(60);
const NUM_CTX: u32 = 8192;

pub struct RealEffects<'a> {
    pub p: &'a Project,
    pub stop: Arc<AtomicBool>,
    pub vram_gb: Option<f64>,
    on_event: &'a mut dyn FnMut(Value),
    step: String,
    /// measure an importance matrix for the small sizes of the export (`AutoSpec::imatrix`)
    imatrix: bool,
    /// the progress of the whole run (`auto_progress` events, at most one a second)
    tracker: super::auto_progress::ProgressTracker,
}

impl<'a> RealEffects<'a> {
    pub fn new(p: &'a Project, stop: Arc<AtomicBool>, vram_gb: Option<f64>, on_event: &'a mut dyn FnMut(Value)) -> Self {
        RealEffects { p, stop, vram_gb, on_event, step: "plan".into(), imatrix: true, tracker: super::auto_progress::ProgressTracker::new() }
    }

    /// Is the made model `name` (a folder under `models/`) a cut of Inkling-2-Mini? Its thinking is checked through a GGUF file (llama.cpp), not through Transformers.
    fn is_inkling(&self, name: &str) -> bool {
        runs::valid_name(name).is_ok() && runs::read_json(&runs::models_dir(self.p).join(name).join("model").join("ndforge_reshape.json")).is_some_and(|m| m["origin"]["family"] == "inkling2")
    }

    /// The thinking check of a mixture of experts: the model (the original, the cut or the healed one) is made into a Q4_K_M GGUF, llama.cpp answers the questions with Inkling's own
    /// template arguments and sampling, and the file is deleted again. Each check pays for a conversion (minutes) and a load; the original is checked once per run.
    fn think_check_inkling(&mut self, name: &str, target: &str, quick: bool) -> Result<Out<Value>> {
        use crate::export::runs::{execute_no_train, exports_dir, ModelSource};
        let meta = runs::read_json(&runs::models_dir(self.p).join(name).join("model").join("ndforge_reshape.json")).unwrap_or(Value::Null);
        let src = match target {
            "original" => ModelSource::Hub { repo: meta["origin"]["repo"].as_str().ok_or_else(|| Error::new("The original of this model is not recorded."))?.to_string() },
            "cut" => ModelSource::Made { name: name.into(), sub: "model".into() },
            "healed" => ModelSource::Made { name: name.into(), sub: "healed".into() },
            other => return Err(Error::new(format!("Unknown thing to check: {other}"))),
        };
        let stop = self.stop.clone();
        let imatrix = ImatrixOptions { enabled: false, ..ImatrixOptions::default() };
        let metas = execute_no_train(self.p, &src, &["q4_k_m".to_string()], NUM_CTX, false, &imatrix, &MixedOptions::default(), &stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        if self.stop.load(Ordering::SeqCst) && metas.as_ref().map(|m| m.is_empty()).unwrap_or(true) {
            return Ok(Out::Stopped);
        }
        let metas = metas?;
        let exports = exports_dir(self.p);
        let (Some(dir_name), Some(file)) = (metas.first().and_then(|m| m["name"].as_str()), metas.first().and_then(|m| m["gguf_file"].as_str())) else {
            return Err(Error::new("The GGUF file for the thinking check was not made."));
        };
        let gguf = exports.join(dir_name).join(file);
        let spec = ThinkSpec { name: name.into(), target: "gguf".into(), quick, gguf: Some(gguf.to_string_lossy().to_string()), family: Some("inkling2".into()), ..ThinkSpec::default() };
        let r = thinkcheck::run(self.p, &spec, self.stop.clone(), &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        let _ = std::fs::remove_dir_all(exports.join(dir_name));           // the file was only made for the check
        done(r, Ok)
    }

    /// The export of the healed model with the options of the mixed sizes.
    #[allow(clippy::too_many_arguments)]
    fn export_with(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool, mixed: &MixedOptions) -> Result<Out<Value>> {
        let stop = self.stop.clone();
        let dir = std::path::PathBuf::from(&healed.dir);
        // the matrix for the small sizes is measured on the finished model, on text that looks like what it will be asked: with thinking when it keeps thinking (docs/IMATRIX.md)
        let imatrix = ImatrixOptions { enabled: self.imatrix, reasoning: keep_thinking.then(ReasoningCalib::default), ..ImatrixOptions::default() };
        let r = crate::export::runs::execute_model_folder_full(self.p, run, &dir, family, quants, NUM_CTX, keep_thinking, &imatrix, mixed, &stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        if self.stop.load(Ordering::SeqCst) && r.as_ref().map(|m| m.is_empty()).unwrap_or(true) {
            return Ok(Out::Stopped);
        }
        let metas = r?;
        let exports = crate::export::runs::exports_dir(self.p);
        let ggufs: Vec<Value> = metas.iter().filter_map(|m| {
            let (q, f, n) = (m["quant"].as_str()?, m["gguf_file"].as_str()?, m["name"].as_str()?);
            Some(json!({"quant": q, "path": exports.join(n).join(f), "export": n, "size_gb": std::fs::metadata(exports.join(n).join(f)).map(|x| (x.len() as f64 / 1e8).round() / 10.0).ok()}))
        }).collect();
        Ok(Out::Done(json!({"dir": exports, "hf_dir": healed.dir, "ggufs": ggufs, "exports": metas.iter().map(|m| m["name"].clone()).collect::<Vec<_>>()})))
    }

}

/// A worker event as the window gets it: the same events as on the Slim page, tagged with the step of the run.
pub fn tag(mut e: Value, step: &str) -> Value {
    e["phase"] = json!("auto");
    e["auto_step"] = json!(step);
    e
}

/// Forwards one event at once (while the job still runs), so the card shows a heal, a cut or a check as it happens.
fn send(on_event: &mut dyn FnMut(Value), tracker: &mut super::auto_progress::ProgressTracker, step: &str, e: Value) {
    let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
    let snap = tracker.on_event(&e, now);
    on_event(tag(e, step));
    if let Some(s) = snap {
        on_event(s);
    }
}

/// A job that ended because the person pressed Stop: `Err("Cancelled." | "Stopped...")`, or a result with `stopped: true`.
fn was_stopped(r: &Result<Value>) -> bool {
    match r {
        Ok(v) => v["stopped"].as_bool().unwrap_or(false),
        Err(e) => e.is_cancelled() || e.0.starts_with("Stopped"),
    }
}

fn done<T>(r: Result<Value>, f: impl FnOnce(Value) -> Result<T>) -> Result<Out<T>> {
    if was_stopped(&r) {
        return Ok(Out::Stopped);
    }
    f(r?).map(Out::Done)
}

fn source_of(repo: &str) -> Source {
    if heal_source::is_hub_id(repo) { Source::Hub { id: repo.to_string() } } else { Source::Folder { path: repo.to_string() } }
}

impl Effects for RealEffects<'_> {
    fn tracker(&mut self) -> Option<&mut super::auto_progress::ProgressTracker> {
        Some(&mut self.tracker)
    }

    fn emit(&mut self, event: Value) {
        if event["kind"] == "auto" && event["status"] == "running" {
            self.step = event["auto_step"].as_str().unwrap_or("").to_string();
        }
        (self.on_event)(event);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    fn model(&self, id: &str) -> Result<ModelRef> {
        let m = catalog::find(id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
        Ok(ModelRef { id: m.id, repo: m.repo, name: m.name, family: m.family })
    }

    fn has_training_data(&self) -> bool {
        crate::train::runs::latest_version(self.p).is_some()
    }

    fn vram_gb(&self) -> Option<f64> {
        self.vram_gb
    }

    fn history(&self) -> crate::timing::History {
        crate::vllm::history_with_measured(crate::timing::History::recent(60), Some(self.p))          // plus the numbers of your own vLLM self-test and teacher check
    }
    fn free_disk_gb(&self) -> Option<f64> {
        crate::envmgr::health::free_disk_gb(&runs::models_dir(self.p))
    }

    fn inspect(&mut self, model: &ModelRef) -> Result<Out<Inspect>> {
        let m = catalog::find(&model.id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
        let stop = self.stop.clone();
        let r = runs::inspect(&m, &stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        match r {
            Err(e) if e.is_cancelled() => Ok(Out::Stopped),
            r => r.map(Out::Done),
        }
    }

    fn measure(&mut self, model_id: &str, samples: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Result<Out<Value>> {
        let stop = self.stop.clone();
        let r = runs::measure_with(self.p, model_id, samples, 512, load, reasoning, false, stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        done(r, Ok)
    }

    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>> {
        let stop = self.stop.clone();
        let r = runs::try_run(self.p, spec, stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        let p = self.p;
        done(r, |_| Ok(runs::try_results_with(p, &spec.model_id, spec.samples, spec.len, &spec.load, spec.reasoning.as_ref())?["analysis"].clone()))
    }

    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>> {
        let stop = self.stop.clone();
        let r = runs::execute_with_grace(self.p, spec, stop, GRACE, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        done(r, |v| match (v["name"].as_str(), v["model_id"].as_str()) {
            (Some(name), Some(id)) => Ok(CutRef { name: name.into(), model_id: id.into() }),
            _ => Err(Error::new("The cut did not give a model.")),
        })
    }

    fn think_check(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>) -> Result<Out<Value>> {
        if gguf.is_none() && self.is_inkling(name) {
            return self.think_check_inkling(name, target, quick);
        }
        let spec = ThinkSpec { name: name.into(), target: target.into(), quick, gguf: gguf.map(String::from), load: None, batch: None, max_new: None, round: None, backend: None, engine: None, family: None };
        let stop = self.stop.clone();
        let r = thinkcheck::run(self.p, &spec, stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        done(r, Ok)
    }

    fn think_check_with(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>, engine: &str) -> Result<Out<Value>> {
        // "vllm@bnb-nf4": the route the original of this run was served with, so that the worker serves this model the same way when it fits (a like-for-like comparison)
        let (engine, route_hint) = match engine.split_once('@') { Some((e, h)) => (e, Some(h.to_string())), None => (engine, None) };
        if !engine.starts_with("vllm") || gguf.is_some() || target == "gguf" {
            return self.think_check(name, target, quick, gguf);
        }
        let s = crate::settings::Settings::load().vllm;
        let fall = |me: &mut Self, why: String| -> Result<Out<Value>> {
            let note = format!("{why} The standard engine does this check.");
            (me.on_event)(tag(json!({"kind": "log", "text": note.clone()}), &me.step));
            let r = me.think_check(name, target, quick, None)?;
            Ok(match r {
                Out::Done(mut v) => {
                    v["engine_used"] = json!({"engine": "hf", "fallback": true, "requested": true, "note": note});
                    Out::Done(v)
                }
                Out::Stopped => Out::Stopped,
            })
        };
        if !s.enabled {
            return fall(self, "vLLM is switched off in Settings.".into());
        }
        let facts = crate::vllm::preflight_now(&s);
        if let Some(bad) = facts.iter().find(|c| c.light == crate::envmgr::health::Light::Fail) {
            return fall(self, format!("vLLM cannot start now: {}", bad.detail));
        }
        let spec = ThinkSpec { name: name.into(), target: target.into(), quick, gguf: None, load: None, batch: None, max_new: None, round: None, backend: Some("vllm".into()), engine: Some(crate::vllm::engine_job_with_card(&s)), family: None };
        let mut spec = spec;
        if engine == "vllm_fp8kv" {
            if let Some(e) = spec.engine.as_mut() {
                e["kv_cache_dtype"] = json!("fp8");
            }
        }
        if let (Some(h), Some(e)) = (route_hint.as_ref().filter(|_| target != "original"), spec.engine.as_mut()) {
            e["route_hint"] = json!(h);
        }
        let stop = self.stop.clone();
        let r = thinkcheck::run(self.p, &spec, stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        let _ = crate::vllm::stop_leftovers(&s, &crate::vllm::real_runner);
        if was_stopped(&r) {
            return Ok(Out::Stopped);
        }
        match r {
            Ok(mut v) => {
                v["engine_used"] = json!({"engine": "vllm", "fallback": false, "engine_version": v["engine"]["vllm"], "startup_seconds": v["engine"]["startup_seconds"], "tokens_per_s": v["tokens_per_s"],
                                         "route": v["engine"]["pick"]["quant"], "like_for_like": v["engine"]["pick"]["like_for_like"]});
                Ok(Out::Done(v))
            }
            Err(e) => fall(self, format!("vLLM did not work ({}).", e.0.chars().take(200).collect::<String>())),
        }
    }

    fn engine_cleanup(&mut self) {
        let s = crate::settings::Settings::load().vllm;
        if s.enabled {
            let _ = crate::vllm::stop_leftovers(&s, &crate::vllm::real_runner);
        }
    }

    fn heal(&mut self, student: &Student, teacher_repo: &str, spec: &HealSpec) -> Result<Out<HealRef>> {
        let mut spec = spec.clone();
        let mut gate_note: Option<String> = None;
        if spec.vllm_teacher() {
            // the heal's own gate (settings, route, the saved check): a refusal is a fallback to the standard teacher with a note, unless the settings say vLLM is required
            let dir = match student {
                Student::Made(n) => runs::models_dir(self.p).join(n),
                Student::Folder(f) => std::path::PathBuf::from(f),
            };
            if let Err(e) = runs::vllm_teacher_gate(&dir, &spec, teacher_repo, "cache") {
                if crate::settings::Settings::load().vllm.teacher.required {
                    return Err(e);
                }
                spec.vllm = None;
                gate_note = Some(e.0);
            }
        }
        let spec = &spec;
        let student = match student {
            Student::Made(name) => Source::Made { name: name.clone() },
            Student::Folder(path) => Source::Folder { path: path.clone() },
        };
        let teacher = source_of(teacher_repo);
        let stop = self.stop.clone();
        let r = heal_source::heal_from_source(self.p, &student, Some(&teacher), spec, self.vram_gb, stop, GRACE, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        done(r, |v| match (v["name"].as_str(), v["model_id"].as_str(), v["healed_dir"].as_str()) {
            (Some(name), Some(id), Some(dir)) => {
                let keys = ["kd_first", "kd_last", "calib_loss_before", "calib_loss_after", "teacher_calib_loss", "tokens_seen", "steps"];
                let mut summary: serde_json::Map<String, Value> = keys.iter().filter(|k| !v[**k].is_null()).map(|k| (k.to_string(), v[*k].clone())).collect();
                // a heal in rounds says how many tokens it really trained (it may have stopped early) and keeps its rounds for the report
                let tokens_m = match v["rounds"]["tokens_m"].as_f64().filter(|_| v["rounds"].is_object()) {
                    Some(t) => {
                        summary.insert("rounds".into(), v["rounds"].clone());
                        t.ceil() as u32
                    }
                    None => spec.tokens_m,
                };
                // which engine made the original's answers (vLLM asked for): what the worker said, or why the heal went on with the standard teacher
                if let Some(note) = &gate_note {
                    summary.insert("teacher_engine".into(), json!({"used": false, "fallback": true, "problem": note}));
                } else if spec.vllm_teacher() {
                    let t = std::fs::read(runs::models_dir(self.p).join(name).join("heal.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).map(|h| h["teacher_engine"].clone());
                    if let Some(t) = t.filter(|t| t.is_object()) {
                        summary.insert("teacher_engine".into(), t);
                    }
                }
                Ok(HealRef { name: name.into(), model_id: id.into(), dir: dir.into(), tokens_m, summary: Some(Value::Object(summary)) })
            }
            _ => Err(Error::new("The heal did not give a model.")),
        })
    }

    fn remove(&mut self, what: &Made) -> Result<()> {
        match what.kind.as_str() {
            "cut" => runs::remove_cut(self.p, &what.name),
            "healed" => runs::remove_keeping_teacher_cache(self.p, &what.name),
            // the 16-bit GGUF kept as the reference of a comparison: its export folder (only ever one this run made, only under exports/)
            "export" => {
                let n = what.name.as_str();
                if n.is_empty() || n.contains(['/', '\\']) || n.contains("..") {
                    return Err(Error::new("That is not the name of an export."));
                }
                let dir = crate::export::runs::exports_dir(self.p).join(n);
                if dir.is_dir() {
                    std::fs::remove_dir_all(&dir)?;
                }
                Ok(())
            }
            other => Err(Error::new(format!("Auto-cut did not make a {other}."))),
        }
    }

    fn drop_teacher_cache(&mut self, models: &[String]) -> Result<()> {
        runs::drop_teacher_caches_of(self.p, models);
        Ok(())
    }

    fn export_options(&mut self, imatrix: bool) {
        self.imatrix = imatrix;
    }

    fn export(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool) -> Result<Out<Value>> {
        self.export_with(run, healed, family, quants, keep_thinking, &MixedOptions::default())
    }

    fn export_measured(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool, target_gb: f64) -> Result<Out<Value>> {
        // the matrix of the export (measured on the healed model, with its thinking text) decides; no saved one is chosen
        let mixed = MixedOptions { measured: Some(MixedMeasured { target_gb, ..Default::default() }), ..Default::default() };
        self.export_with(run, healed, family, quants, keep_thinking, &mixed)
    }

    fn compare_quants(&mut self, reference: &str, picks: &[Value]) -> Result<Out<Value>> {
        use crate::export::quantcheck::{self, CompareSpec, QuantPick, ReferenceSpec, TextSpec};
        // the model the files were made from names the chat template (its folder is the `base_repo` in the meta of the 16-bit export next to the reference file)
        let meta = std::path::Path::new(reference).parent().map(|d| d.join("meta.json")).and_then(|f| crate::export::runs::read_json(&f)).unwrap_or(Value::Null);
        let tokenizer = meta["base_repo"].as_str().map(String::from);
        // the text: the saved data of the project when there is any, else general text with the fixed seed (docs/QUANT_COMPARE.md)
        let kind = if self.has_training_data() { "data" } else { "general" };
        let text = TextSpec { kind: kind.into(), tokenizer, ..Default::default() };
        let quants: Vec<QuantPick> = picks.iter().filter_map(|p| serde_json::from_value(p.clone()).ok()).collect();
        let spec = CompareSpec { reference: ReferenceSpec { kind: "file".into(), file: Some(reference.to_string()), source: None }, quants, text, ..Default::default() };
        let stop = self.stop.clone();
        let r = quantcheck::compare(self.p, &spec, &stop, &mut |e| send(&mut *self.on_event, &mut self.tracker, &self.step, e));
        done(r, Ok)
    }

    fn publish(&mut self, export: &Value, model_name: &str) -> Result<Out<Value>> {
        use crate::unsloth::{client::UnslothClient, publish};
        let settings = crate::settings::Settings::load();
        let Some(share) = settings.studio_share.clone().filter(|s| !s.host_path.trim().is_empty()) else {
            return Ok(Out::Done(json!({"published": false, "why": "Unsloth Studio's share is not set up (Settings: \"Studio share\")."})));
        };
        // the API key is optional here: with it the container path is found out and Studio is told about the folder
        let client = settings.unsloth.as_ref().and_then(|u| crate::secrets::get(&format!("unsloth:{}", u.url)).filter(|k| !k.trim().is_empty()).map(|k| UnslothClient::new(&u.url, &k)));
        let files: Vec<std::path::PathBuf> = export["ggufs"].as_array().map(|a| a.iter().filter_map(|g| g["path"].as_str()).map(std::path::PathBuf::from).collect()).unwrap_or_default();
        let readme = format!("# {model_name}, cut and healed by ND Lore Forge (Auto-cut)\n\nGGUF files: {}.\n\nThis model was cut (blocks removed, MLP thinned) and healed against the original, which is its teacher. It can think: in the chat, switch\nthe Think option on. Its chat template keeps the thinking default of the original.\n",
                             files.iter().filter_map(|f| f.file_name()).map(|n| n.to_string_lossy().to_string()).collect::<Vec<_>>().join(", "));
        let spec = publish::PublishSpec { name: model_name.to_string(), files, hf_folder: export["hf_dir"].as_str().map(std::path::PathBuf::from), readme: Some(readme) };
        let free = crate::envmgr::health::free_disk_gb(std::path::Path::new(share.host_path.trim()));
        let ctx = publish::Ctx { client: client.as_ref(), free_gb: free, stop: &self.stop };
        let mut last = 0.0f64;
        let r = publish::publish(&share, &spec, &ctx, &mut |f| {
            if f - last >= 0.02 || f >= 1.0 {
                last = f;
                send(&mut *self.on_event, &mut self.tracker, &self.step, json!({"kind": "progress", "step": "publish", "value": f, "text": "Copying to the Unsloth Studio share"}));
            }
        });
        match r {
            Err(e) if e.0 == "Stopped." => Ok(Out::Stopped),
            r => r.map(Out::Done),
        }
    }
}
