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
use crate::export::runs::ImatrixOptions;
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
}

impl<'a> RealEffects<'a> {
    pub fn new(p: &'a Project, stop: Arc<AtomicBool>, vram_gb: Option<f64>, on_event: &'a mut dyn FnMut(Value)) -> Self {
        RealEffects { p, stop, vram_gb, on_event, step: "plan".into() }
    }

    /// A worker event as the window gets it: the same events as on the Slim page, tagged with the step of the run.
    fn forward(&mut self, mut e: Value) {
        e["phase"] = json!("auto");
        e["auto_step"] = json!(self.step);
        (self.on_event)(e);
    }
}

/// A job that ended because the person pressed Stop: `Err("Cancelled." | "Stopped...")`, or a result with `stopped: true`.
fn was_stopped(r: &Result<Value>) -> bool {
    match r {
        Ok(v) => v["stopped"].as_bool().unwrap_or(false),
        Err(e) => e.0 == "Cancelled." || e.0.starts_with("Stopped"),
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

    fn free_disk_gb(&self) -> Option<f64> {
        crate::envmgr::health::free_disk_gb(&runs::models_dir(self.p))
    }

    fn inspect(&mut self, model: &ModelRef) -> Result<Out<Inspect>> {
        let m = catalog::find(&model.id).ok_or_else(|| Error::new("That model is not in the list any more."))?;
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let r = runs::inspect(&m, &stop, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
        match r {
            Err(e) if e.0 == "Cancelled." => Ok(Out::Stopped),
            r => r.map(Out::Done),
        }
    }

    fn measure(&mut self, model_id: &str, samples: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Result<Out<Value>> {
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let r = runs::measure_with(self.p, model_id, samples, 512, load, reasoning, false, stop, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
        done(r, Ok)
    }

    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>> {
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let r = runs::try_run(self.p, spec, stop, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
        let p = self.p;
        done(r, |_| Ok(runs::try_results_with(p, &spec.model_id, spec.samples, spec.len, &spec.load, spec.reasoning.as_ref())?["analysis"].clone()))
    }

    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>> {
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let r = runs::execute_with_grace(self.p, spec, stop, GRACE, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
        done(r, |v| match (v["name"].as_str(), v["model_id"].as_str()) {
            (Some(name), Some(id)) => Ok(CutRef { name: name.into(), model_id: id.into() }),
            _ => Err(Error::new("The cut did not give a model.")),
        })
    }

    fn think_check(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>) -> Result<Out<Value>> {
        let spec = ThinkSpec { name: name.into(), target: target.into(), quick, gguf: gguf.map(String::from), load: None, batch: None };
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let r = thinkcheck::run(self.p, &spec, stop, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
        done(r, Ok)
    }

    fn heal(&mut self, student: &Student, teacher_repo: &str, spec: &HealSpec) -> Result<Out<HealRef>> {
        let student = match student {
            Student::Made(name) => Source::Made { name: name.clone() },
            Student::Folder(path) => Source::Folder { path: path.clone() },
        };
        let teacher = source_of(teacher_repo);
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let r = heal_source::heal_from_source(self.p, &student, Some(&teacher), spec, self.vram_gb, stop, GRACE, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
        done(r, |v| match (v["name"].as_str(), v["model_id"].as_str(), v["healed_dir"].as_str()) {
            (Some(name), Some(id), Some(dir)) => {
                let keys = ["kd_first", "kd_last", "calib_loss_before", "calib_loss_after", "teacher_calib_loss", "tokens_seen", "steps"];
                let summary: serde_json::Map<String, Value> = keys.iter().filter(|k| !v[**k].is_null()).map(|k| (k.to_string(), v[*k].clone())).collect();
                Ok(HealRef { name: name.into(), model_id: id.into(), dir: dir.into(), tokens_m: spec.tokens_m, summary: Some(Value::Object(summary)) })
            }
            _ => Err(Error::new("The heal did not give a model.")),
        })
    }

    fn remove(&mut self, what: &Made) -> Result<()> {
        match what.kind.as_str() {
            "cut" => runs::remove_cut(self.p, &what.name),
            "healed" => runs::remove(self.p, &what.name),
            other => Err(Error::new(format!("Auto-cut did not make a {other}."))),
        }
    }

    fn export(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool) -> Result<Out<Value>> {
        let stop = self.stop.clone();
        let mut events = Vec::new();
        let dir = std::path::PathBuf::from(&healed.dir);
        // the matrix for the small sizes is measured on the finished model, on text that looks like what it will be asked: with thinking when it keeps thinking (docs/IMATRIX.md)
        let imatrix = ImatrixOptions { enabled: true, reasoning: keep_thinking.then(ReasoningCalib::default), ..ImatrixOptions::default() };
        let r = crate::export::runs::execute_model_folder(self.p, run, &dir, family, quants, NUM_CTX, keep_thinking, &imatrix, &stop, &mut |e| events.push(e));
        for e in events {
            self.forward(e);
        }
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
        let step = self.step.clone();
        let mut last = 0.0f64;
        let mut events = Vec::new();
        let r = publish::publish(&share, &spec, &ctx, &mut |f| {
            if f - last >= 0.02 || f >= 1.0 {
                last = f;
                events.push(json!({"kind": "progress", "phase": "auto", "auto_step": step, "step": "publish", "value": f, "text": "Copying to the Unsloth Studio share"}));
            }
        });
        for e in events {
            (self.on_event)(e);
        }
        match r {
            Err(e) if e.0 == "Stopped." => Ok(Out::Stopped),
            r => r.map(Out::Done),
        }
    }
}
