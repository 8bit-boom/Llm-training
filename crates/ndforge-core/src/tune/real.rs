//! The real effects of the runner: the existing benches (the token-budget sweep of Measure heal speed, Find the fastest loss settings) run by the same code the buttons use.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::runner::{TestCtx, TestDef, TestOut, TuneFx};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::reshape::runs::HealSpec;

pub struct Real<'a> {
    pub project: &'a Project,
    pub vram_gb: Option<f64>,
    pub stop: Arc<AtomicBool>,
    pub emit_fn: &'a mut dyn FnMut(Value),
}

/// The saved settings (not Heal-card fields) a run reads and a preset may carry: the original's tokens per forward pass and the vLLM teacher's engine settings.
pub const SETTINGS_KEYS: [&str; 7] = ["teacher_batch_tokens", "max_num_seqs", "vllm_batch_tokens", "gpu_util_pct", "llama_ngl", "try_calib_batch", "think_batch"];

/// The Heal card's values as a heal spec (a null or a zero token budget means "automatic" and is left out).
pub fn heal_spec_of(card: &Map<String, Value>) -> Result<HealSpec> {
    let mut m = Map::new();
    for (k, v) in card {
        if v.is_null() || SETTINGS_KEYS.contains(&k.as_str()) || (k == "student_batch_tokens" && v.as_u64().unwrap_or(0) == 0) || (k == "head_checkpoint" && v == "on") || (k == "checkpointing" && v == "auto") {
            continue;
        }
        m.insert(k.clone(), v.clone());
    }
    m.entry("tokens_m".to_string()).or_insert(json!(1));
    serde_json::from_value(Value::Object(m)).map_err(|e| Error::new(format!("The card's settings are not valid for a speed test: {e}")))
}

impl TuneFx for Real<'_> {
    fn emit(&mut self, ev: Value) {
        (self.emit_fn)(ev);
    }
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
    fn now(&self) -> String {
        super::api::now()
    }
    fn gpu(&self) -> Option<(u64, u64, Vec<String>)> {
        use crate::memory::Env;
        let env = crate::memory::Real { vllm: crate::settings::Settings::load().vllm };
        let (total, used) = env.gpu_memory()?;
        let me = std::process::id();
        let procs = env.procs();
        let others = env.gpu_apps().into_iter().filter(|a| a.pid != me && !crate::memory::descends_from(&procs, a.pid, me)).map(|a| format!("{} {}", a.name, a.used_mb.map(|m| format!("{m} MB")).unwrap_or_default()).trim().to_string()).collect();
        Some((total.saturating_sub(used), total, others))
    }
    fn run_test(&mut self, d: &TestDef, ctx: &TestCtx) -> Result<TestOut> {
        let spec = heal_spec_of(ctx.card)?;
        match d.id {
            "heal.student_batch" => {
                let mut forward = |e: Value| (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": e}));
                let report = crate::reshape::heal_speed::tune_sweep(self.project, ctx.model, &spec, self.vram_gb, &ctx.dir, self.stop.clone(), &mut forward)?;
                let sweep = &report["batch_sweep"];
                for r in sweep["results"].as_array().cloned().unwrap_or_default() {
                    (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": r}));
                }
                Ok(TestOut { rows: sweep["results"].as_array().cloned().unwrap_or_default(), raw: json!({"spread": sweep["spread"], "rows": report["rows"], "card_mb": sweep["card_mb"]}), stopped: report["stopped"] == true || self.stopped(), note: String::new() })
            }
            "teacher.standard" => {
                let mut forward = |e: Value| (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": e}));
                let res = crate::reshape::heal_speed::tune_teacher_sweep(self.project, ctx.model, &spec, self.vram_gb, &ctx.dir, self.stop.clone(), &mut forward)?;
                let sweep = &res["batch_sweep"];
                for r in sweep["results"].as_array().cloned().unwrap_or_default() {
                    (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": r}));
                }
                Ok(TestOut { rows: sweep["results"].as_array().cloned().unwrap_or_default(), raw: json!({"spread": sweep["spread"], "rows": sweep["rows"], "tokens": sweep["tokens"], "card_mb": sweep["card_mb"]}), stopped: res["stopped"] == true || self.stopped(), note: String::new() })
            }
            "teacher.vllm" => {
                let variants = super::vllm::variants(ctx.card);
                let list: Vec<Value> = variants.iter().map(|v| json!({"id": v.id, "knob": v.knob, "value": v.value, "label": v.label, "engine": v.engine})).collect();
                let mut forward = |e: Value| (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": e}));
                let res = crate::reshape::runs::vllm_teacher_tune(self.project, ctx.model, &spec, self.vram_gb, &list, &ctx.dir, self.stop.clone(), std::time::Duration::from_secs(180), &mut forward);
                let res = match res {
                    Ok(r) => r,
                    Err(_) if self.stopped() => json!({"stopped": true, "rows": []}),
                    Err(e) => return Err(e),
                };
                for r in res["rows"].as_array().cloned().unwrap_or_default() {
                    (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": r}));
                }
                Ok(TestOut { rows: res["rows"].as_array().cloned().unwrap_or_default(), raw: json!({"reference": res["reference"], "rows_scored": res["rows_scored"], "tokens": res["tokens"]}), stopped: res["stopped"] == true || self.stopped(), note: String::new() })
            }
            "try.calib_batch" | "think.batch" => {
                let (what, sizes): (&str, &[u32]) = if d.id == "try.calib_batch" { ("calib", &[0, 1, 2, 8, 16]) } else { ("think", &[0, 4, 16, 32]) };
                let mut forward = |e: Value| (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": e}));
                let res = crate::reshape::runs::tune_batch_bench(self.project, ctx.model, &spec, self.vram_gb, what, sizes, &ctx.dir, self.stop.clone(), std::time::Duration::from_secs(120), &mut forward);
                let res = match res {
                    Ok(r) => r,
                    Err(_) if self.stopped() => json!({"stopped": true, "batch_sweep": {"results": []}}),
                    Err(e) => return Err(e),
                };
                let sweep = &res["batch_sweep"];
                for r in sweep["results"].as_array().cloned().unwrap_or_default() {
                    (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": r}));
                }
                Ok(TestOut { rows: sweep["results"].as_array().cloned().unwrap_or_default(), raw: json!({"spread": sweep["spread"], "card_mb": sweep["card_mb"], "load": res["load"]}), stopped: res["stopped"] == true || self.stopped(), note: String::new() })
            }
            "disk" => {
                let dir = crate::reshape::runs::models_dir(self.project).join(".tune-disk");
                let r = super::disk::probe(&dir, super::disk::DEFAULT_MB, &self.stop);
                let _ = std::fs::remove_dir(&dir);
                match r {
                    Ok(v) => {
                        (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": v}));
                        Ok(TestOut { rows: vec![v.clone()], raw: v, stopped: false, note: String::new() })
                    }
                    Err(_) if self.stopped() => Ok(TestOut { stopped: true, ..Default::default() }),
                    Err(e) => Err(e),
                }
            }
            "llama.offload" => {
                let layers = ctx.layers.max(1);
                let cands = super::llama::candidates(layers);
                let mut forward = |e: Value| (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": e}));
                let res = crate::export::quantcheck::offload_tune(self.project, &cands, &ctx.dir, &self.stop, &mut forward);
                let res = match res {
                    Ok(r) => r,
                    Err(_) if self.stopped() => json!({"stopped": true, "rows": []}),
                    Err(e) => return Err(e),
                };
                for r in res["rows"].as_array().cloned().unwrap_or_default() {
                    (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": r}));
                }
                Ok(TestOut { rows: res["rows"].as_array().cloned().unwrap_or_default(), raw: json!({"gguf": res["gguf"], "gguf_mb": res["gguf_mb"], "tool": res["tool"]}), stopped: res["stopped"] == true || self.stopped(), note: String::new() })
            }
            "heal.loss" => {
                let mut forward = |e: Value| (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "event": e}));
                let res = crate::reshape::heal_bench::run(self.project, ctx.model, &spec, self.vram_gb, crate::reshape::heal_bench::STEPS, self.stop.clone(), &mut forward)?;
                Ok(TestOut { rows: res["rows"].as_array().cloned().unwrap_or_default(), raw: res.clone(), stopped: res["stopped"] == true || self.stopped(), note: String::new() })
            }
            "heal.quality" => {
                let tokens = (if ctx.trial_tokens_k == 0 { 400 } else { ctx.trial_tokens_k } as u64) * 1000;
                let mut rows: Vec<Value> = Vec::new();
                let mut stopped = false;
                for (i, v) in super::quality::variants(ctx.card).into_iter().enumerate() {
                    if self.stopped() {
                        stopped = true;
                        break;
                    }
                    let seed = if v.id == "baseline2" { 2 } else { 1 };
                    let ov = super::quality::overrides(&v, seed);
                    (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": {"trial": v.id, "label": v.label, "n": i}}));
                    let dir = ctx.dir.join(format!("trial-{i}"));
                    let r = match crate::reshape::heal_bench::trial(self.project, ctx.model, &spec, self.vram_gb, &ov, tokens, &dir, &self.stop) {
                        Ok(r) => r,
                        Err(_) if self.stopped() => {
                            stopped = true;
                            break;
                        }
                        Err(e) => return Err(e),
                    };
                    let mut row = json!({"id": v.id, "knob": v.knob, "value": v.value, "label": v.label, "seed": seed});
                    if let (Some(a), Some(b)) = (row.as_object_mut(), r.as_object()) {
                        for (k, x) in b {
                            a.insert(k.clone(), x.clone());
                        }
                    }
                    (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": row}));
                    rows.push(row);
                }
                // the winners were tried one at a time: when two or more won, they are tried TOGETHER once (same seed) so that settings that do not add up are not applied together
                if !stopped {
                    let (decisions, _, _) = super::quality::decide_quality(&rows, ctx.card_mb);
                    let winners: Map<String, Value> = decisions.iter().filter(|k| k.changed).map(|k| (k.knob.clone(), k.chosen.clone())).collect();
                    if winners.len() >= 2 && !self.stopped() {
                        let v = super::quality::Variant { id: "combined".into(), knob: "combined".into(), value: Value::Object(winners), label: "The winning settings together".into() };
                        let ov = super::quality::overrides(&v, 1);
                        (self.emit_fn)(json!({"kind": "tune_progress", "id": d.id, "event": {"trial": v.id, "label": v.label, "n": rows.len()}}));
                        match crate::reshape::heal_bench::trial(self.project, ctx.model, &spec, self.vram_gb, &ov, tokens, &ctx.dir.join("trial-combined"), &self.stop) {
                            Ok(r) => {
                                let mut row = json!({"id": v.id, "knob": v.knob, "value": v.value, "label": v.label, "seed": 1});
                                if let (Some(a), Some(b)) = (row.as_object_mut(), r.as_object()) {
                                    a.extend(b.clone());
                                }
                                (self.emit_fn)(json!({"kind": "tune_row", "id": d.id, "row": row}));
                                rows.push(row);
                            }
                            Err(_) if self.stopped() => stopped = true,
                            Err(e) => rows.push(json!({"id": v.id, "knob": v.knob, "label": v.label, "ok": false, "error": e.0})),
                        }
                    }
                }
                Ok(TestOut { rows, raw: json!({"trial_tokens": tokens}), stopped, note: String::new() })
            }
            other => Err(Error::new(format!("The test {other} is not built in this version."))),
        }
    }
}
