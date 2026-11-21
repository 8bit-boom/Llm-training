//! The Auto-cut state machine (`reshape::auto`) with a stand-in for everything it does (no worker, so it runs on every system): the steps in order, stop at every
//! step and continue, the clean-up of the intermediate files, the preflight and the failures.
use std::collections::{BTreeMap, BTreeSet};

use ndforge_core::error::{Error, Result};
use ndforge_core::project::Project;
use ndforge_core::reshape::auto::*;
use ndforge_core::reshape::auto_plan::{AutoSpec, HealRoundsOpt};
use ndforge_core::reshape::plan::*;
use ndforge_core::reshape::runs::{HealSpec, TrySpec};
use serde_json::{json, Value};
use tempfile::tempdir;

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("worker").join("tests").join("data")
}

fn qwen27() -> Inspect {
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(data_dir().join("qwen3_5_27b_config.json")).unwrap()).unwrap();
    let tc = &cfg["text_config"];
    let u = |k: &str| tc[k].as_u64().unwrap_or(0) as u32;
    let d = WidthDims {
        hidden: u("hidden_size"), intermediate: u("intermediate_size"), heads: u("num_attention_heads"), kv_heads: u("num_key_value_heads"), head_dim: u("head_dim"),
        lin_k_heads: u("linear_num_key_heads"), lin_v_heads: u("linear_num_value_heads"), lin_k_dim: u("linear_key_head_dim"), lin_v_dim: u("linear_value_head_dim"),
        conv_kernel: u("linear_conv_kernel_dim"), mtp_layers: u("mtp_num_hidden_layers"), vocab: u("vocab_size"), tied: tc["tie_word_embeddings"].as_bool().unwrap_or(false),
    };
    let types: Vec<String> = tc["layer_types"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    Inspect {
        model_type: "qwen3_5".into(), supported: true, layers: 64, layer_types: types, period: 4, blocks: 16, vision_params: 460_730_096, other_params: d.hidden as u64,
        total_params: 27_781_427_952, bytes_per_param: 2.0, has_vision: true, local: true, width_supported: true, layer_params: vec![390_000_000; 64],
        width_choices: Some(WidthChoices { intermediate: (34..=68).map(|k| k * 256).collect(), lin_v_heads: vec![48, 32], heads: vec![24, 20, 16] }),
        width_dims: Some(d), ..Default::default()
    }
}

/// The metrics of a good thinking check (the original's)
fn good() -> Value {
    json!({"closed_rate": 1.0, "accuracy": 0.75, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800})
}

/// A stand-in for everything the steps do. Every call that can be stopped counts; `stop_at` makes the n-th one answer "stopped" (once).
struct Mock {
    ins: Inspect,
    n: usize,
    stop_at: Option<usize>,
    stop_flag: bool,
    /// the calls that finished, in order, with what mattered about them
    done: Vec<String>,
    /// every call, stopped ones too
    all: Vec<String>,
    events: Vec<Value>,
    cuts: u32,
    heals: u32,
    made: BTreeSet<String>,
    removed: Vec<Made>,
    /// the models named when the shared teacher answers were let go (one entry per call)
    dropped: Vec<Vec<String>>,
    specs: Vec<Value>,
    free_gb: Option<f64>,
    vram: Option<f64>,
    data: bool,
    /// the metrics a check of (name, target) answers; default: the original's
    checks: BTreeMap<String, Value>,
    /// how many times each (target) was asked
    asked: BTreeMap<String, u32>,
    /// loss of a cut in Try first: (blocks, width) -> (loss, h1, h2)
    losses: BTreeMap<(Vec<u32>, Option<u32>), (f64, f64, f64)>,
    try_acc: BTreeMap<u32, (Vec<(u32, f64)>, BTreeMap<u32, Vec<Value>>, Vec<Value>)>,
    fail_heal_once: bool,
    fail_text: Option<String>,
    publish_ok: bool,
    /// the measured file beats the plain file of its size in the comparison
    promote: bool,
    /// the comparison fails with this text
    compare_error: Option<String>,
    /// the export folders the run made (a 16-bit export may be removed)
    exports_made: BTreeSet<String>,
    /// the heal specs as the heal got them, and the `imatrix` of every export_options call
    heal_specs: Vec<Value>,
    imatrix: Vec<bool>,
    /// the clock of the time limit (seconds); None = the real one
    clock: Option<i64>,
    /// the progress of the run (`auto_progress` events); None = the stand-in has none, like most tests
    tracker: Option<ndforge_core::reshape::auto_progress::ProgressTracker>,
    /// the search requests the stand-in got, and whether its answer says the time was used up
    searches: Vec<Value>,
    search_budget_stop: bool,
    /// why each heal in rounds ends (pass | plateau | cap | time), one per heal; "cap" when the list is used up
    rounds_stops: std::collections::VecDeque<String>,
    /// vLLM (experimental): the numbers this PC measured, what the checks / heals did with the engine, and the simulated engine (running or not)
    hist: ndforge_core::timing::History,
    vllm_log: Vec<String>,
    engine_running: bool,
    cleanups: u32,
    /// a vLLM check leaves the engine running (a crash), until `engine_cleanup`
    leak_engine: bool,
    /// the vLLM checks fail (the standard engine does them: the result says so)
    vllm_checks_fail: bool,
    /// the route the mock's vLLM serves the original with (`engine.pick.quant` of its result); later checks get the hint of the engine string unless `pick_mismatch`
    pick_original: Option<String>,
    pick_mismatch: bool,
    /// what a heal that was asked to use vLLM reports: `teacher_engine` of its summary
    teacher_engine: Option<Value>,
}

impl Mock {
    fn new() -> Self {
        Mock {
            ins: qwen27(), n: 0, stop_at: None, stop_flag: false, done: Vec::new(), all: Vec::new(), events: Vec::new(), cuts: 0, heals: 0, made: BTreeSet::new(), removed: Vec::new(), dropped: Vec::new(), specs: Vec::new(),
            free_gb: Some(900.0), vram: Some(32.0), data: true, checks: BTreeMap::new(), asked: BTreeMap::new(), losses: BTreeMap::new(), try_acc: BTreeMap::new(), fail_heal_once: false, fail_text: None, publish_ok: true, promote: false, compare_error: None, exports_made: BTreeSet::new(), heal_specs: Vec::new(), imatrix: Vec::new(), clock: None, tracker: None, searches: Vec::new(), search_budget_stop: false, rounds_stops: std::collections::VecDeque::new(), pick_original: None, pick_mismatch: false,
            hist: ndforge_core::timing::History::default(), vllm_log: Vec::new(), engine_running: false, cleanups: 0, leak_engine: false, vllm_checks_fail: false, teacher_engine: None,
        }
    }

    fn stop_here(&mut self, what: String) -> bool {
        self.n += 1;
        self.all.push(what);
        if self.stop_at == Some(self.n) {
            self.stop_at = None;
            return true;
        }
        false
    }

    fn loss_of(&self, blocks: &[u32], width: Option<u32>) -> (f64, f64, f64) {
        self.losses.get(&(blocks.to_vec(), width)).copied().unwrap_or_else(|| {
            // binary fractions: they are the same after a round trip through the JSON file
            let l = (2048 + 512 * blocks.len() as u32 + width.map(|w| (17408 - w) / 16).unwrap_or(0)) as f64 / 1024.0;
            (l, l, l)
        })
    }
}

impl Effects for Mock {
    fn emit(&mut self, event: Value) {
        self.events.push(event);
    }
    fn tracker(&mut self) -> Option<&mut ndforge_core::reshape::auto_progress::ProgressTracker> {
        self.tracker.as_mut()
    }
    fn stopped(&self) -> bool {
        self.stop_flag
    }
    fn now_secs(&self) -> i64 {
        self.clock.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0))
    }
    fn model(&self, id: &str) -> Result<ModelRef> {
        Ok(ModelRef { id: id.into(), repo: "org/Qwen3.8-27B".into(), name: "Qwen3.8 27B".into(), family: "qwen3_5".into() })
    }
    fn has_training_data(&self) -> bool {
        self.data
    }
    fn vram_gb(&self) -> Option<f64> {
        self.vram
    }
    fn free_disk_gb(&self) -> Option<f64> {
        self.free_gb
    }
    fn inspect(&mut self, m: &ModelRef) -> Result<Out<Inspect>> {
        if self.stop_here(format!("inspect {}", m.id)) {
            return Ok(Out::Stopped);
        }
        self.done.push("inspect".into());
        Ok(Out::Done(self.ins.clone()))
    }
    fn measure(&mut self, id: &str, samples: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Result<Out<Value>> {
        if self.stop_here(format!("measure {id}")) {
            return Ok(Out::Stopped);
        }
        self.done.push(format!("measure {id} {samples} {load} reasoning={}", reasoning.is_some()));
        Ok(Out::Done(json!({"key": "k", "tokens": 1234, "load": load, "loss": 2.0})))
    }
    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>> {
        if self.stop_here(format!("try {} {:?} {:?}", spec.samples, spec.runs, spec.widths)) {
            return Ok(Out::Stopped);
        }
        if let Some(sv) = &spec.search {
            // the search: for every cut of the first group a neighbouring run and a set of every other block; the hidden best one is the separate set
            self.done.push(format!("search samples={} minutes={}", spec.samples, sv["budget_min"]));
            self.searches.push(sv.clone());
            let mut rows = serde_json::Map::new();
            let mut ranked: Vec<(f64, String)> = Vec::new();
            for c in sv["groups"][0]["cuts"].as_array().cloned().unwrap_or_default() {
                let (k, w) = (c["k"].as_u64().unwrap_or(0) as u32, c["width"].as_u64().map(|x| x as u32));
                if k == 0 {
                    continue;
                }
                let sets: Vec<Vec<u32>> = vec![(1..=k).collect(), (0..k).map(|i| 1 + 2 * i).collect(), (0..k).map(|i| 2 + 2 * i).collect()];
                for blocks in sets {
                    let (mut l, h1, h2) = self.loss_of(&blocks, w);
                    if blocks.windows(2).any(|p| p[1] - p[0] == 2) {
                        l -= 0.01;              // the separate set is the better one
                    }
                    let key = format!("d={}|w={}", blocks.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","), w.map(|x| x.to_string()).unwrap_or_default());
                    rows.insert(key.clone(), json!({"loss": l, "loss_h1": h1 - 0.01 * (l != h1) as u8 as f64, "loss_h2": h2 - 0.01 * (l != h2) as u8 as f64}));
                    ranked.push((l, key));
                }
            }
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            let keys: Vec<String> = ranked.iter().map(|r| r.1.clone()).collect();
            let fin: Vec<String> = keys.iter().step_by(2).take(3).cloned().collect();
            let stop_reason = if self.search_budget_stop { "budget" } else { "done" };
            return Ok(Out::Done(json!({"base": {"loss": 2.0}, "passes": keys.len() + 15, "search": {"ranking": {"0": keys}, "rows": rows, "finalists": {"0": fin}, "stop_reason": stop_reason}})));
        }
        self.done.push(format!("try samples={} runs={:?} widths={:?} choices={}", spec.samples, spec.runs, spec.widths, spec.choices.len()));
        let mut acc = self.try_acc.remove(&spec.samples).unwrap_or_default();
        for w in &spec.widths {
            let (l, ..) = self.loss_of(&[], Some(*w));
            acc.0.push((*w, l));
        }
        for k in &spec.runs {
            let rows: Vec<Value> = (1..=14u32).map(|a| {
                let blocks: Vec<u32> = (a..a + k).collect();
                let (l, h1, h2) = self.loss_of(&blocks, None);
                json!({"blocks": blocks, "loss": l, "loss_h1": h1, "loss_h2": h2})
            }).collect();
            acc.1.insert(*k, rows);
        }
        for c in &spec.choices {
            let blocks: Vec<u32> = c.blocks.clone();
            let (l, h1, h2) = self.loss_of(&blocks, c.width);
            acc.2.push(json!({"blocks": blocks, "intermediate": c.width, "loss": l, "loss_h1": h1, "loss_h2": h2}));
        }
        let widths: Vec<Value> = acc.0.iter().map(|(w, l)| json!({"intermediate": w, "loss": l, "loss_h1": l, "loss_h2": l})).collect();
        let mut runs = serde_json::Map::new();
        for (k, rows) in &acc.1 {
            let best = rows.iter().min_by(|a, b| a["loss"].as_f64().partial_cmp(&b["loss"].as_f64()).unwrap()).cloned().unwrap();
            runs.insert(k.to_string(), json!({"k": k, "windows": rows, "best": best}));
        }
        let analysis = json!({"base": {"loss": 2.0}, "widths": widths, "runs": runs, "choices": acc.2});
        self.try_acc.insert(spec.samples, acc);
        Ok(Out::Done(analysis))
    }
    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>> {
        if self.stop_here(format!("cut {}", spec.source_model_id)) {
            return Ok(Out::Stopped);
        }
        self.cuts += 1;
        let name = format!("cut-{}", self.cuts);
        self.made.insert(name.clone());
        self.specs.push(serde_json::to_value(spec).unwrap());
        self.done.push(format!("cut {} from {} blocks={} drop={:?} width={:?} reasoning={}", name, spec.source_model_id, spec.blocks, spec.drop_layers, spec.width.as_ref().map(|w| w.intermediate), spec.reasoning.is_some()));
        Ok(Out::Done(CutRef { model_id: format!("id-{name}"), name }))
    }
    fn history(&self) -> ndforge_core::timing::History {
        self.hist.clone()
    }
    fn think_check_with(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>, engine: &str) -> Result<Out<Value>> {
        let hint = engine.split_once('@').map(|(_, h)| h.to_string());
        if !engine.starts_with("vllm") {
            return self.think_check(name, target, quick, gguf);
        }
        self.vllm_log.push(format!("engine={engine}"));
        assert!(!self.engine_running, "a vLLM job must never start while another engine runs");
        self.vllm_log.push(format!("check {name} {target}"));
        self.engine_running = true;
        let r = self.think_check(name, target, quick, gguf);
        self.engine_running = self.leak_engine;
        let r = r?;
        Ok(match r {
            Out::Done(mut v) => {
                if let Some(orig) = self.pick_original.clone() {
                    let (quant, like) = if target == "original" { (orig, Value::Null) } else if self.pick_mismatch { ("fp8".to_string(), json!(false)) } else { (hint.clone().unwrap_or_else(|| "fp8".into()), json!(hint.is_some())) };
                    v["engine"] = json!({"pick": {"quant": quant, "like_for_like": like}});
                }
                v["engine_used"] = if self.vllm_checks_fail { json!({"engine": "hf", "fallback": true, "requested": true, "note": "vLLM did not start."}) } else { json!({"engine": "vllm", "fallback": false, "startup_seconds": 263.5}) };
                if v["engine"]["pick"].is_object() && v["engine_used"]["engine"] == "vllm" {
                    v["engine_used"]["route"] = v["engine"]["pick"]["quant"].clone();
                    v["engine_used"]["like_for_like"] = v["engine"]["pick"]["like_for_like"].clone();
                }
                Out::Done(v)
            }
            o => o,
        })
    }
    fn engine_cleanup(&mut self) {
        self.cleanups += 1;
        self.engine_running = false;
    }
    fn think_check(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>) -> Result<Out<Value>> {
        if self.stop_here(format!("check {name} {target}")) {
            return Ok(Out::Stopped);
        }
        *self.asked.entry(format!("{name}/{target}")).or_default() += 1;
        self.done.push(format!("check {name} {target} quick={quick} gguf={}", gguf.is_some()));
        let main = self.checks.get(&format!("{name}/{target}")).cloned().or_else(|| self.checks.get(target).cloned()).unwrap_or_else(good);
        Ok(Out::Done(json!({"main": main, "control": {"closed_rate": 1.0}, "suite_version": 1})))
    }
    fn heal(&mut self, student: &Student, teacher_repo: &str, spec: &HealSpec) -> Result<Out<HealRef>> {
        if self.stop_here(format!("heal {student:?}")) {
            return Ok(Out::Stopped);
        }
        if self.fail_heal_once {
            self.fail_heal_once = false;
            return Err(Error::new(self.fail_text.take().unwrap_or_else(|| "The graphics card ran out of memory.".into())));
        }
        assert!(!self.engine_running, "a training must never start while the vLLM engine runs");
        self.heals += 1;
        self.heal_specs.push(serde_json::to_value(spec).unwrap());
        if spec.vllm_teacher() {
            self.vllm_log.push(format!("heal teacher cache tokens={}", spec.tokens_m));
        }
        let name = match student {
            Student::Made(n) => n.clone(),
            Student::Folder(_) => format!("heal-{}", self.heals),
        };
        self.made.insert(name.clone());
        self.done.push(format!("heal {name} teacher={teacher_repo} tokens={} recipe={:?}", spec.tokens_m, spec.recipe));
        let mut summary = json!({"kd_first": 1.5, "kd_last": 0.5, "calib_loss_before": 3.0, "calib_loss_after": 2.4, "teacher_calib_loss": 2.0,
                                  "coverage": {"v": 1, "chat_rows": {"complete": 5, "cut": 1, "skipped": 0}, "notes": []},
                                  "holdout_loss_before": 2.9, "holdout_loss_after": 2.6, "teacher_holdout_loss": 2.1});
        if spec.vllm_teacher() {
            summary["teacher_engine"] = self.teacher_engine.clone().unwrap_or_else(|| json!({"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4", "used": true, "fallback": false, "startup_seconds": 263.5, "tokens_per_s": 2900.0}));
        }
        let mut tokens_m = spec.tokens_m;
        if let Some(r) = &spec.rounds {
            let reason = self.rounds_stops.pop_front().unwrap_or_else(|| "cap".into());
            let cap = r.max_tokens_m.unwrap_or(spec.tokens_m);
            tokens_m = if reason == "cap" { cap } else { (cap / 2).max(r.round_m) };
            summary["rounds"] = json!({"tokens_m": tokens_m as f64, "cap_m": cap as f64, "stop": {"reason": reason, "round": tokens_m / r.round_m.max(1), "tokens_m": tokens_m as f64}, "rows": [{"n": 1, "tokens_m": r.round_m as f64, "calib_loss": 2.5, "decision": "continue", "seconds": 1800.0,
                                                 "check": {"mode": "quick", "verdict": "fail", "score": 0.3, "main": {"closed_rate": 0.5, "accuracy": 0.4, "loop_rate": 0.2}}}],
                                       "final": {"verdict": "fail", "mode": "quick", "main": {"closed_rate": 0.9, "accuracy": 0.7, "loop_rate": 0.0}},
                                       "merged_checks": [{"after_round": 2, "verdict": "fail", "failed": ["closed_rate"]}]});
        }
        Ok(Out::Done(HealRef { model_id: format!("id-{name}-healed"), dir: format!("models/{name}/healed"), tokens_m, name, summary: Some(summary) }))
    }
    fn remove(&mut self, what: &Made) -> Result<()> {
        assert!(self.made.contains(&what.name) || (what.kind == "export" && self.exports_made.contains(&what.name)), "only what Auto-cut made may be removed: {what:?}");
        assert!(!what.name.contains("Qwen") && what.name != "org/Qwen3.8-27B", "never the original: {what:?}");
        self.removed.push(what.clone());
        self.done.push(format!("remove {} {}", what.kind, what.name));
        Ok(())
    }
    fn drop_teacher_cache(&mut self, models: &[String]) -> Result<()> {
        assert!(models.iter().all(|m| self.made.contains(m)), "only models of this run: {models:?}");
        self.dropped.push(models.to_vec());
        Ok(())
    }
    fn export_options(&mut self, imatrix: bool) {
        self.imatrix.push(imatrix);
    }
    fn export(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool) -> Result<Out<Value>> {
        if self.stop_here(format!("export {}", healed.name)) {
            return Ok(Out::Stopped);
        }
        self.done.push(format!("export {} {family} {quants:?} thinking={keep_thinking}", healed.name));
        let ggufs: Vec<Value> = quants.iter().map(|q| json!({"quant": q, "path": format!("exports/{run}-{q}/model-{q}.gguf"), "export": format!("{run}-{q}")})).collect();
        self.exports_made.extend(quants.iter().map(|q| format!("{run}-{q}")));
        Ok(Out::Done(json!({"dir": format!("exports/{run}"), "ggufs": ggufs})))
    }
    fn export_measured(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool, target_gb: f64) -> Result<Out<Value>> {
        self.done.push(format!("target {target_gb}"));
        self.export(run, healed, family, quants, keep_thinking)
    }
    fn compare_quants(&mut self, reference: &str, picks: &[Value]) -> Result<Out<Value>> {
        if self.stop_here("compare".into()) {
            return Ok(Out::Stopped);
        }
        self.done.push(format!("compare {reference} {}", picks.iter().map(|p| p["path"].as_str().unwrap_or("").rsplit('/').next().unwrap_or("").to_string()).collect::<Vec<_>>().join(",")));
        if let Some(e) = &self.compare_error {
            return Err(Error::new(e.clone()));
        }
        let rows: Vec<Value> = picks.iter().enumerate().map(|(i, p)| json!({"id": i.to_string(), "label": p["label"], "mixed": p["mixed"], "size": 1.0e10 + i as f64 * 1e9, "mean_kld": 0.03 + i as f64 * 0.001, "kld_unc": 0.0005, "same_top": 93.0})).collect();
        let mi = picks.iter().position(|p| p["mixed"] == "mixed_measured").unwrap();
        let promotion = json!([{"id": mi.to_string(), "against": "0", "promoted": self.promote, "reason": if self.promote { "The measured file is clearly closer." } else { "A tie within measurement noise." }, "delta_pct": 3.0, "noise": 0.0007}]);
        Ok(Out::Done(json!({"rows": rows, "promotion": promotion, "verdict": ["A sentence."], "entry": {"id": "20261101-1000-ref"}})))
    }
    fn publish(&mut self, export: &Value, name: &str) -> Result<Out<Value>> {
        if self.stop_here("publish".into()) {
            return Ok(Out::Stopped);
        }
        self.done.push(format!("publish {name} {}", export["ggufs"].as_array().map(|a| a.len()).unwrap_or(0)));
        Ok(Out::Done(if self.publish_ok { json!({"published": true, "container_path": "/workspace/exports/x"}) } else { json!({"published": false, "why": "Unsloth Studio's share is not set up."}) }))
    }
}

fn project(root: &std::path::Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

/// the spec of most tests: without the thinking check before healing (its own tests below switch it on), so that the calls of the other steps stay easy to read
fn spec(target: f64) -> AutoSpec {
    AutoSpec { think_gate: false, ..AutoSpec::new("qwen3.8-27b", target) }
}

/// the same with the thinking check before healing
fn gspec(target: f64) -> AutoSpec {
    AutoSpec::new("qwen3.8-27b", target)
}

/// the metrics of a cut that lost much of its thinking
fn weak() -> Value {
    json!({"closed_rate": 0.5, "accuracy": 0.28, "loop_rate": 0.5, "garbage_rate": 0.0, "median_think_tokens": 800})
}

/// what two runs must have in common: the state without ids and times
fn shape(st: &AutoState) -> Value {
    let text = serde_json::to_string(st).unwrap().replace(&st.id, "RUN");
    let mut v: Value = serde_json::from_str(&text).unwrap();
    let o = v.as_object_mut().unwrap();
    for k in ["id", "created", "updated"] {
        o.remove(k);
    }
    for s in o.get_mut("steps").and_then(|s| s.as_array_mut()).unwrap() {
        s.as_object_mut().unwrap().remove("started");
        s.as_object_mut().unwrap().remove("finished");
        s.as_object_mut().unwrap().remove("attempts");          // a stop makes another attempt: the time of the steps is not part of "the same run"
        s.as_object_mut().unwrap().remove("spent_s");
    }
    v
}

fn statuses(st: &AutoState) -> Vec<(String, String)> {
    st.steps.iter().map(|s| (s.id.clone(), s.status.clone())).collect()
}

#[test]
fn an_uninterrupted_run_goes_through_every_step_in_order() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let ids: Vec<&str> = st.steps.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["plan", "measure", "try", "s1.cut", "s1.check_cut", "s1.heal", "s1.check_heal", "s2.cut", "s2.check_cut", "s2.heal", "s2.check_heal", "export", "check_gguf"]);
    assert!(st.steps.iter().all(|s| s.status == "done"), "{:?}", statuses(&st));
    assert_eq!(fx.done, vec![
        "inspect",
        "measure qwen3.8-27b 256 4bit reasoning=true",
        "try samples=128 runs=[1, 2] widths=[11008] choices=0",
        "try samples=128 runs=[] widths=[] choices=2",
        "cut cut-1 from qwen3.8-27b blocks=0 drop=None width=Some(14080) reasoning=true",
        "check cut-1 original quick=false gguf=false",
        "check cut-1 cut quick=false gguf=false",
        "heal cut-1 teacher=org/Qwen3.8-27B tokens=40 recipe=Some(\"thinking\")",
        "check cut-1 healed quick=false gguf=false",
        "remove cut cut-1",
        "cut cut-2 from id-cut-1-healed blocks=0 drop=None width=Some(11008) reasoning=true",
        "check cut-2 cut quick=false gguf=false",
        "heal cut-2 teacher=org/Qwen3.8-27B tokens=40 recipe=Some(\"thinking\")",
        "check cut-2 healed quick=false gguf=false",
        "remove healed cut-1",
        "remove cut cut-2",
        "export cut-2 qwen3_5 [\"q4_k_m\"] thinking=true",
        "check cut-2 gguf quick=false gguf=true",
    ]);
    // the stage numbers, the checks and what was made
    assert_eq!(st.stages.len(), 2);
    assert!(st.stages.iter().all(|s| s.passed && s.cut_check.is_some() && s.heal_check.as_ref().unwrap()["verdict"] == "pass"));
    assert_eq!(st.stages[1].heal.as_ref().unwrap().model_id, "id-cut-2-healed");
    assert_eq!(st.baseline.as_ref().unwrap()["main"], good());
    assert_eq!(st.gguf_check.as_ref().unwrap()["verdict"], "pass");
    assert_eq!(st.export.as_ref().unwrap()["ggufs"][0]["quant"], "q4_k_m");
    // the files of the run
    let again = load(&run_dir(&p, &st.id)).unwrap();
    assert_eq!(again, st);
    assert_eq!(latest(&p).unwrap().id, st.id);
    // events: one running and one done per step
    for id in &ids {
        assert!(fx.events.iter().any(|e| e["auto_step"] == *id && e["status"] == "running" && e["phase"] == "auto"), "{id}");
        assert!(fx.events.iter().any(|e| e["auto_step"] == *id && e["status"] == "done"), "{id}");
    }
}

#[test]
fn what_is_removed_is_only_what_the_run_made_and_only_after_the_next_step_worked() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    let removed: Vec<(&str, &str)> = fx.removed.iter().map(|m| (m.kind.as_str(), m.name.as_str())).collect();
    assert_eq!(removed, vec![("cut", "cut-1"), ("healed", "cut-1"), ("cut", "cut-2")], "the final healed model stays");
    let pos = |what: &str| fx.done.iter().position(|d| d == what).unwrap();
    // the cut of a stage goes after the check of its healed model, the healed model of stage 1 after the check of stage 2's
    assert!(pos("remove cut cut-1") > pos("check cut-1 healed quick=false gguf=false") && pos("remove cut cut-1") < pos("cut cut-2 from id-cut-1-healed blocks=0 drop=None width=Some(11008) reasoning=true"));
    assert!(pos("remove healed cut-1") > pos("check cut-2 healed quick=false gguf=false") && pos("remove cut cut-2") > pos("check cut-2 healed quick=false gguf=false"));
    assert!(st.made.iter().filter(|m| m.removed).count() == 3 && st.made.iter().any(|m| m.kind == "healed" && m.name == "cut-2" && !m.removed));
    // keeping the intermediate files: nothing is removed
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut keep = Mock::new();
    let st2 = start(&p2, &mut keep, AutoSpec { keep_intermediate: true, ..spec(21.1) }).unwrap();
    assert_eq!(st2.status, "done");
    assert!(keep.removed.is_empty() && !keep.done.iter().any(|d| d.starts_with("remove")));
    assert!(st2.made.iter().all(|m| !m.removed));
    // the teacher's shared answers go once, when the last stage has passed, and only without keep_intermediate
    assert_eq!(fx.dropped.len(), 1, "{:?}", fx.dropped);
    assert!(fx.dropped[0].contains(&"cut-1".to_string()) && fx.dropped[0].contains(&"cut-2".to_string()));
    assert!(fx.done.iter().position(|d| d == "remove cut cut-2").is_some());
    assert!(keep.dropped.is_empty(), "keeping the intermediate files keeps the answers too");
}

#[test]
fn a_stop_at_every_call_and_a_continue_gives_the_same_run_as_one_without_a_stop() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut plain = Mock::new();
    let full = start(&p, &mut plain, spec(21.1)).unwrap();
    assert_eq!(full.status, "done");
    let full = load(&run_dir(&p, &full.id)).unwrap();           // from the disk, like the one that was stopped and continued
    let total = plain.n;
    assert!(total >= 12, "{total}");
    for n in 1..=total {
        let dir = tempdir().unwrap();
        let q = project(dir.path());
        let mut fx = Mock::new();
        fx.stop_at = Some(n);
        let first = start(&q, &mut fx, spec(21.1)).unwrap();
        assert_eq!(first.status, "stopped", "stop at call {n}: {:?}", fx.all.last());
        assert!(first.steps.iter().any(|s| s.status == "pending") && first.error.is_none());
        let last = resume(&q, &mut fx, &first.id).unwrap();
        assert_eq!(last.status, "done", "stop at call {n}: {:?}", last.error);
        assert_eq!(fx.done, plain.done, "stop at call {n} ({:?}): the finished calls are the same as in a run without a stop", fx.all[n - 1]);
        let (a, b) = (shape(&last), shape(&full));
        for (k, v) in a.as_object().unwrap() {
            assert_eq!(v, &b[k], "stop at call {n}: {k}");
        }
        // the stopped call was made again, nothing else
        assert_eq!(fx.all.len(), plain.all.len() + 1, "stop at call {n}");
    }
}

#[test]
fn the_stop_button_between_steps_stops_before_the_next_one() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.stop_flag = true;
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!((st.status.as_str(), fx.n), ("stopped", 0), "nothing was started");
    fx.stop_flag = false;
    let st = resume(&p, &mut fx, &st.id).unwrap();
    assert_eq!(st.status, "done");
    // a run that is done is not run again, one that is running is not continued
    let n = fx.n;
    assert_eq!(resume(&p, &mut fx, &st.id).unwrap().status, "done");
    assert_eq!(fx.n, n, "nothing was run again");
}

#[test]
fn a_failure_is_kept_and_a_continue_goes_on_with_that_step() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.fail_heal_once = true;
    let e = start(&p, &mut fx, spec(21.1)).unwrap_err();
    assert!(e.0.contains("ran out of memory"), "{}", e.0);
    let st = latest(&p).unwrap();
    assert_eq!((st.status.as_str(), st.step.as_str()), ("failed", "s1.heal"));
    assert!(st.error.as_deref().unwrap().contains("ran out of memory"));
    assert_eq!(st.steps.iter().find(|s| s.id == "s1.heal").unwrap().status, "failed");
    assert!(st.stages[0].cut.is_some() && st.stages[0].heal.is_none());
    let cuts_before = fx.cuts;
    let st = resume(&p, &mut fx, &st.id).unwrap();
    assert_eq!(st.status, "done");
    assert_eq!(fx.cuts, cuts_before + 1, "only stage 2 cuts after the heal of stage 1: the first cut was not made again");
    assert_eq!(fx.heals, 2);
}

#[test]
fn a_run_cut_off_by_closing_the_program_is_marked_stopped_and_continues() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.stop_at = Some(3);
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    // pretend the program was closed in the middle of a step
    let dir = run_dir(&p, &st.id);
    let mut cut_off = load(&dir).unwrap();
    cut_off.status = "running".into();
    cut_off.steps.iter_mut().find(|s| s.status == "pending").unwrap().status = "running".into();
    save(&dir, &mut cut_off).unwrap();
    assert!(resume(&p, &mut fx, &st.id).is_err(), "a run that says it is running is not continued by hand");
    assert!(start(&p, &mut Mock::new(), spec(21.1)).unwrap_err().0.contains("already running"));
    assert_eq!(mark_interrupted(&p), 1);
    let again = load(&dir).unwrap();
    assert_eq!(again.status, "stopped");
    assert!(again.steps.iter().all(|s| s.status != "running"));
    assert_eq!(resume(&p, &mut fx, &st.id).unwrap().status, "done");
}

#[test]
fn the_preflight_refuses_before_any_job_starts() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    // no training data
    let mut fx = Mock::new();
    fx.data = false;
    let e = start(&p, &mut fx, spec(21.1)).unwrap_err();
    assert!(e.0.contains("training-data version"), "{}", e.0);
    assert_eq!(fx.n, 0);
    // a disk that is too small: the numbers are in the message and in the saved plan
    let mut small = Mock::new();
    small.free_gb = Some(120.0);
    let e = start(&p, &mut small, spec(21.1)).unwrap_err();
    assert!(e.0.contains("up to 196 GB of disk") && e.0.contains("120 GB are free"), "{}", e.0);
    let st = list(&p).into_iter().find(|s| s.status == "failed" && s.plan.is_some()).unwrap();
    assert_eq!(st.plan.as_ref().unwrap().disk.as_ref().unwrap().peak_gb, 195.8);
    assert!(small.done == vec!["inspect"], "{:?}", small.done);
    // a card that cannot heal the stage
    let mut card = Mock::new();
    card.vram = Some(8.0);
    assert!(start(&p, &mut card, spec(21.1)).unwrap_err().0.contains("Healing stage 1"));
    // a size the model does not reach
    let mut big = Mock::new();
    assert!(start(&p, &mut big, spec(27.6)).unwrap_err().0.contains("3 % smaller"));
    assert_eq!(big.done, vec!["inspect"]);
}

#[test]
fn blocks_are_cut_as_try_first_found_them_and_the_cheapest_candidate_wins() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    // 18 B: blocks and a thinner MLP; the cheapest window of every block count is at block 7
    for k in 1..=4u32 {
        fx.losses.insert(((7..7 + k).collect(), None), (2.0 + 0.01 * k as f64, 2.0, 2.0));
    }
    let st = start(&p, &mut fx, spec(18.0)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let tr = st.try_result.as_ref().unwrap();
    assert!(tr["rows"].as_array().unwrap().len() >= 2 && tr["winner"].is_u64() && tr["halves_agree"] == true, "{tr}");
    let chosen = st.chosen.as_ref().unwrap();
    assert!(chosen.blocks >= 1, "{chosen:?}");
    // stage 1 removes the blocks (the layers of blocks 7.. as Try first found them), stage 2 the width
    let first = &fx.specs[0];
    let layers: Vec<u32> = first["drop_layers"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    let want: Vec<u32> = (7..7 + chosen.blocks).flat_map(|b| b * 4..b * 4 + 4).collect();
    assert_eq!(layers, want);
    assert_eq!(st.drop_blocks, Some((7..7 + chosen.blocks).collect::<Vec<u32>>()));
    assert!(first["width"].is_null(), "stage 1: blocks only");
    assert!(fx.specs[1]["drop_layers"].is_null() && fx.specs[1]["width"]["intermediate"] == json!(chosen.intermediate.unwrap()), "stage 2: the width, measured on the healed model");
    assert_eq!(fx.specs[1]["source_model_id"], "id-cut-1-healed");
    assert!(fx.done.iter().any(|d| d.contains("choices=3")), "the combined cut was asked: {:?}", fx.done);
}

#[test]
fn when_the_two_halves_disagree_try_first_is_asked_again_with_more_samples() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    for k in 1..=4u32 {
        fx.losses.insert(((7..7 + k).collect(), None), (1.0, 1.0, 1.0));      // the cheapest window of every block count starts at block 7
    }
    // which cuts are asked about at 18 B (a run with the stand-in's own losses)
    let st0 = start(&project(tempdir().unwrap().path()), &mut Mock::new(), spec(18.0)).unwrap();
    let cands: Vec<(u32, Option<u32>)> = st0.try_result.as_ref().unwrap()["rows"].as_array().unwrap().iter().map(|r| (r["blocks"].as_u64().unwrap() as u32, r["intermediate"].as_u64().map(|x| x as u32))).collect();
    assert!(cands.len() >= 2, "{cands:?}");
    for (i, (b, w)) in cands.iter().enumerate() {
        let blocks: Vec<u32> = (7..7 + *b).collect();
        let (h1, h2) = if i == 0 { (2.0, 2.2) } else { (2.1, 2.1) };
        fx.losses.insert((blocks, *w), (2.1, h1, h2));
    }
    let st = start(&p, &mut fx, spec(18.0)).unwrap();
    let tries: Vec<&String> = fx.done.iter().filter(|d| d.starts_with("try ")).collect();
    assert!(tries.iter().any(|t| t.contains("samples=128")) && tries.iter().any(|t| t.contains("samples=256")), "{tries:?}");
    assert!(st.notes.iter().any(|n| n.contains("different winners")), "{:?}", st.notes);
    assert_eq!(st.try_result.as_ref().unwrap()["samples"], 256);
}

fn bad() -> Value {
    json!({"closed_rate": 0.2, "accuracy": 0.1, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800})
}

#[test]
fn a_healed_model_that_misses_a_gate_is_healed_again_with_40_million_more_tokens_and_then_passes() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    // the first heal of stage 1 is bad, the second one (a new folder, healed from the first) is good
    fx.checks.insert("cut-1/healed".into(), bad());
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(fx.done.iter().any(|d| d == "heal cut-1 teacher=org/Qwen3.8-27B tokens=40 recipe=Some(\"thinking\")"));
    assert!(fx.done.iter().any(|d| d == "heal heal-2 teacher=org/Qwen3.8-27B tokens=80 recipe=Some(\"thinking\")"), "{:?}", fx.done);
    let s1 = &st.stages[0];
    assert!(s1.passed && s1.attempt == 1 && s1.heal.as_ref().unwrap().name == "heal-2" && s1.attempts.len() == 2);
    assert_eq!(s1.attempts[0]["verdict"], "fail");
    assert_eq!(s1.attempts[1]["kind"], "heal_again");
    assert!(st.notes.iter().any(|n| n.contains("healed again") && n.contains("40 million")), "{:?}", st.notes);
    // the first healed model of stage 1 and its cut go only once the stage has passed; the original and the final model never
    let pos = |what: &str| fx.done.iter().position(|d| d == what).unwrap();
    assert!(pos("remove healed cut-1") > pos("check heal-2 healed quick=false gguf=false"));
    assert!(fx.removed.iter().all(|m| m.name != "heal-3") && st.stages[1].heal.as_ref().unwrap().name == "cut-2");
    assert_eq!(fx.removed.iter().filter(|m| m.name == "cut-1" && m.kind == "healed").count(), 1, "removed once");
    // the report is written
    let dir = run_dir(&p, &st.id);
    let rep: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("auto_report.json")).unwrap()).unwrap();
    assert_eq!((rep["status"].as_str(), rep["retries"].as_u64()), (Some("done"), Some(1)));
    assert!(rep["stages"].as_array().unwrap().iter().any(|s| s["coverage"]["chat_rows"]["cut"] == 1), "what the heal's rows cover is in the report: {rep}");
    assert!(rep["stages"].as_array().unwrap().iter().any(|s| s["holdout_loss_after"] == 2.6 && s["teacher_holdout_loss"] == 2.1), "the held-out loss is in the report: {rep}");
    assert!(rep["stages"].as_array().unwrap().iter().all(|s| s["breakdown_line"].as_str().is_some_and(|l| l.contains("text layers") && l.contains("file total"))), "{rep}");
    let md = std::fs::read_to_string(dir.join("auto_report.md")).unwrap();
    assert!(md.contains("passed on the second heal") && md.contains("healed again") && md.contains("file total"), "{md}");
    assert_eq!(st.report, Some(json!({"json": "auto_report.json", "md": "auto_report.md"})));
    // the estimate made before the run is kept next to what the steps took (additive keys; the old ones stay)
    let est = &rep["estimate"];
    assert!(est["total_h"][1].as_f64().unwrap() >= est["total_h"][0].as_f64().unwrap() && est["total_h"][0].as_f64().unwrap() > 0.0, "{est}");
    assert!(est["actual_h"].is_number() && est["basis"].as_str().unwrap().contains("steps"), "{est}");
    let rows = est["steps"].as_array().unwrap();
    assert!(rows.iter().any(|x| x["id"] == "s1.heal" && x["estimated_minutes"].is_array() && x["actual_minutes"].is_number() && ["within", "faster", "slower"].contains(&x["result"].as_str().unwrap())), "{rows:?}");
    let heal = rep["steps"].as_array().unwrap().iter().find(|x| x["id"] == "s1.heal").unwrap();
    assert!(heal["estimated_minutes"].is_array() && heal["seconds"].is_number() && heal["status"] == "done" && heal["estimate_basis"] == "formula", "{heal}");
    assert!(md.contains("## Estimated and actual time") && md.contains("| Stage 1 of 2: heal |"), "{md}");
}

#[test]
fn a_second_failure_backs_the_cut_off_one_step_and_the_stage_is_made_again() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    // 18 B: blocks (stage 1) and the width (stage 2); the stage-1 heals of the first two tries are bad
    fx.checks.insert("cut-1/healed".into(), bad());
    fx.checks.insert("heal-2/healed".into(), bad());
    let st = start(&p, &mut fx, spec(18.0)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let blocks = st.try_result.as_ref().unwrap()["rows"][st.try_result.as_ref().unwrap()["winner"].as_u64().unwrap() as usize]["blocks"].as_u64().unwrap() as u32;
    assert!(blocks >= 2, "{blocks}");
    let s1 = &st.stages[0];
    assert!(s1.passed && s1.attempt == 2 && s1.plan.blocks_after == blocks - 1 && s1.attempts.len() == 3, "{:?}", s1.attempts);
    assert_eq!(s1.attempts[2]["kind"], "back_off");
    // the cut was made again with one block less, from the original, and the first cut is gone only after the new stage passed
    let cuts: Vec<&String> = fx.done.iter().filter(|d| d.starts_with("cut ")).collect();
    assert_eq!(cuts.len(), 3, "{cuts:?}");
    assert!(cuts[1].contains("from qwen3.8-27b") && cuts[1].contains(&format!("blocks={}", blocks - 1)), "{}", cuts[1]);
    assert!(st.notes.iter().any(|n| n.contains("backed off one step")), "{:?}", st.notes);
    // stage 2 starts from the healed model of the new stage 1 and cuts the width of the plan; the final size is unchanged
    assert!(fx.specs[2]["source_model_id"].as_str().unwrap().ends_with("-healed") && fx.specs[2]["width"]["intermediate"] == json!(st.chosen.as_ref().unwrap().intermediate.unwrap()));
    let removed: Vec<(&str, &str)> = fx.removed.iter().map(|m| (m.kind.as_str(), m.name.as_str())).collect();
    assert!(removed.contains(&("cut", "cut-1")) && removed.contains(&("healed", "cut-1")) && removed.contains(&("healed", "heal-2")) && removed.contains(&("cut", "cut-2")), "{removed:?}");
    let pos = |what: &str| fx.done.iter().position(|d| d == what).unwrap();
    assert!(pos("remove cut cut-1") > pos("check cut-2 healed quick=false gguf=false"), "the failed tries go after the stage passed");
    assert!(fx.removed.iter().all(|m| m.name != "org/Qwen3.8-27B" && fx.made.contains(&m.name)));
    let md = std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.md")).unwrap();
    assert!(md.contains("passed after backing off"), "{md}");
}

#[test]
fn three_failures_end_the_run_with_gates_failed_a_report_and_nothing_deleted() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "gates_failed");
    let e = st.error.as_deref().unwrap();
    assert!(e.contains("closed_rate") && e.contains("stage 1") && e.contains("3 tries"), "{e}");
    let s1 = &st.stages[0];
    assert!(!s1.passed && s1.attempts.len() == 3 && s1.attempts.iter().all(|a| a["verdict"] == "fail"));
    assert!(fx.removed.is_empty(), "nothing is deleted from a stage that did not pass: {:?}", fx.removed);
    assert!(st.made.iter().all(|m| !m.removed));
    // the stage was tried: heal, heal again, and a backed-off stage; the number of cuts says so
    assert_eq!(fx.cuts, 2, "the first cut and the backed-off cut");
    assert!(st.steps.iter().find(|s| s.id == "s1.check_heal").unwrap().status == "failed");
    assert!(resume(&p, &mut fx, &st.id).unwrap_err().0.contains("after the retries"));
    // the report says what was tried
    let dir = run_dir(&p, &st.id);
    let rep: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("auto_report.json")).unwrap()).unwrap();
    assert_eq!(rep["status"], "gates_failed");
    assert_eq!(rep["failed_gates"], json!(["closed_rate", "accuracy"]));
    assert_eq!(rep["stages"][0]["attempts"].as_array().unwrap().len(), 3);
    let md = std::fs::read_to_string(dir.join("auto_report.md")).unwrap();
    assert!(md.contains("Stopped: the thinking checks failed") && md.contains("Gates missed: closed_rate, accuracy") && md.contains("healed after backing off"), "{md}");
}

#[test]
fn a_single_stage_backed_off_gives_a_larger_model_than_asked() {
    // one stage, the MLP alone: backing off gives 1,024 of the width back
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    let st = start(&p, &mut fx, AutoSpec { stages: 1, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "gates_failed");
    // 11,008 backs off to 12,032: a larger model than asked (these sizes moved when the multi-token head stopped being counted in a width cut)
    assert!(st.stages[0].plan.intermediate_after == Some(12_032), "{:?}", st.stages[0].plan);
    assert!(st.notes.iter().any(|n| n.contains("larger than asked")), "{:?}", st.notes);
    assert_eq!(st.chosen.as_ref().unwrap().intermediate, Some(12_032));
}

#[test]
fn a_stop_at_every_call_of_a_run_with_retries_and_a_continue_gives_the_same_run() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let setup = |fx: &mut Mock| {
        fx.checks.insert("cut-1/healed".into(), bad());
        fx.checks.insert("heal-2/healed".into(), bad());
    };
    let mut plain = Mock::new();
    setup(&mut plain);
    let full = start(&p, &mut plain, spec(18.0)).unwrap();
    assert_eq!(full.status, "done", "{:?}", full.error);
    let full = load(&run_dir(&p, &full.id)).unwrap();
    for n in 1..=plain.n {
        let dir = tempdir().unwrap();
        let q = project(dir.path());
        let mut fx = Mock::new();
        setup(&mut fx);
        fx.stop_at = Some(n);
        let first = start(&q, &mut fx, spec(18.0)).unwrap();
        assert_eq!(first.status, "stopped", "stop at call {n}: {:?}", fx.all.last());
        let last = resume(&q, &mut fx, &first.id).unwrap();
        assert_eq!(last.status, "done", "stop at call {n}: {:?}", last.error);
        assert_eq!(fx.done, plain.done, "stop at call {n} ({:?})", fx.all[n - 1]);
        let (a, b) = (shape(&last), shape(&full));
        for (k, v) in a.as_object().unwrap() {
            assert_eq!(v, &b[k], "stop at call {n}: {k}");
        }
    }
}

#[test]
fn a_gguf_that_fails_its_check_ends_the_run_and_nothing_is_published() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("gguf".into(), bad());
    let st = start(&p, &mut fx, AutoSpec { studio_publish: true, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "gates_failed");
    assert!(st.error.as_deref().unwrap().contains("GGUF") && st.error.as_deref().unwrap().contains("nothing was published"));
    assert!(!fx.done.iter().any(|d| d.starts_with("publish")));
    assert!(st.stages.iter().all(|s| s.passed), "the healed models passed");
    let rep: Value = serde_json::from_str(&std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.json")).unwrap()).unwrap();
    assert_eq!(rep["gguf_check"]["verdict"], "fail");
}

#[test]
fn a_model_without_thinking_has_no_thinking_checks_and_the_original_is_never_touched() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, AutoSpec { keep_thinking: false, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(!fx.done.iter().any(|d| d.starts_with("check")), "{:?}", fx.done);
    assert!(fx.done.iter().any(|d| d.contains("reasoning=false")) && fx.done.iter().any(|d| d.contains("recipe=None") && d.contains("tokens=20")));
    assert!(fx.done.iter().any(|d| d.ends_with("thinking=false")));
    for m in &fx.removed {
        assert!(m.name.starts_with("cut-"), "{m:?}");
    }
}

#[test]
fn publishing_is_a_step_only_when_asked_and_a_missing_studio_is_a_note_not_a_failure() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let plain = start(&p, &mut fx, spec(21.1)).unwrap();
    assert!(!plain.steps.iter().any(|s| s.id == "studio_publish") && plain.publish.is_none());
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut fx2 = Mock::new();
    let st = start(&p2, &mut fx2, AutoSpec { studio_publish: true, quants: vec!["q8_0".into()], ..spec(21.1) }).unwrap();
    assert_eq!(st.steps.last().unwrap().id, "studio_publish");
    assert_eq!(st.publish.as_ref().unwrap()["published"], true);
    assert!(fx2.done.last().unwrap().starts_with("publish Qwen3.8 27B cut to 21.1B 2"), "{:?}", fx2.done.last());
    assert!(fx2.done.iter().any(|d| d.contains("[\"q4_k_m\", \"q8_0\"]")));
    let d3 = tempdir().unwrap();
    let p3 = project(d3.path());
    let mut fx3 = Mock::new();
    fx3.publish_ok = false;
    let st = start(&p3, &mut fx3, AutoSpec { studio_publish: true, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "done");
    assert_eq!(st.steps.last().unwrap().note.as_deref(), Some("Unsloth Studio's share is not set up."));
}

#[test]
fn the_window_gets_a_plan_without_a_job_and_the_report_of_the_newest_run() {
    // the plan: no training data is a blocker, the rest is the planner's
    let ins = qwen27();
    let ok = preview(&ins, &spec(21.1), Some(32.0), Some(900.0), true);
    assert_eq!(ok["plan"]["ok"], true);
    assert_eq!(ok["plan"]["stages"].as_array().unwrap().len(), 2);
    let no = preview(&ins, &spec(21.1), Some(32.0), Some(900.0), false);
    assert_eq!(no["plan"]["ok"], false);
    assert!(no["plan"]["blockers"][0].as_str().unwrap().contains("training-data version"));
    // the report: none before a run, then the newest run's data and Markdown
    let d = tempdir().unwrap();
    let p = project(d.path());
    assert!(report_of(&p, None).unwrap_err().0.contains("no Auto-cut run"));
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    for id in [None, Some(st.id.as_str())] {
        let r = report_of(&p, id).unwrap();
        assert_eq!(r["run"], json!(st.id));
        assert_eq!(r["report"]["status"], "done");
        assert!(r["markdown"].as_str().unwrap().starts_with("# Auto-cut report: Qwen3.8 27B"));
    }
    assert!(report_of(&p, Some("nope")).is_err());
}

#[test]
fn a_run_and_a_heal_that_were_cut_off_are_found_for_going_on_by_themselves() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert!(!st.interrupted);
    // the program was closed while the run was running
    let file = run_dir(&p, &st.id).join("auto.json");
    let mut j: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    j["status"] = json!("running");
    std::fs::write(&file, j.to_string()).unwrap();
    assert!(ndforge_core::queue::interrupted_work(&p).is_empty(), "not marked yet");
    assert_eq!(mark_interrupted(&p), 1);
    let work = ndforge_core::queue::interrupted_work(&p);
    assert_eq!(work, vec![("auto".to_string(), json!({"spec": null, "resume": st.id}))]);
    // a heal that was cut off, of a model the run did not make
    let m = ndforge_core::reshape::runs::models_dir(&p).join("other");
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("state.json"), json!({"status": "done"}).to_string()).unwrap();
    std::fs::write(m.join("heal.json"), json!({"status": "stopped", "interrupted": true, "spec": {"tokens_m": 10}}).to_string()).unwrap();
    // and one that was stopped by the person (not interrupted)
    let n = ndforge_core::reshape::runs::models_dir(&p).join("quiet");
    std::fs::create_dir_all(&n).unwrap();
    std::fs::write(n.join("heal.json"), json!({"status": "stopped", "interrupted": false, "spec": {"tokens_m": 10}}).to_string()).unwrap();
    // and a heal that belongs to the interrupted run
    let own = ndforge_core::reshape::runs::models_dir(&p).join("cut-1");
    std::fs::create_dir_all(&own).unwrap();
    std::fs::write(own.join("heal.json"), json!({"status": "stopped", "interrupted": true, "spec": {"tokens_m": 10}}).to_string()).unwrap();
    let work = ndforge_core::queue::interrupted_work(&p);
    let kinds: Vec<(String, String)> = work.iter().map(|(k, a)| (k.clone(), a["name"].as_str().unwrap_or("").to_string())).collect();
    assert_eq!(kinds, vec![("auto".into(), "".into()), ("heal".into(), "other".into())], "the person's own stop is left alone and the run's own heals go on with the run: {work:?}");
    // going on clears the mark
    let mut fx2 = Mock::new();
    let again = resume(&p, &mut fx2, &st.id).unwrap();
    assert!(!again.interrupted);
}

#[test]
fn a_cut_that_try_first_measured_is_not_measured_again() {
    // one stage: the cut is exactly the candidate Try first measured, so the new model is not loaded once more
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut one = Mock::new();
    let st = start(&p, &mut one, AutoSpec { stages: 1, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(one.specs[0]["skip_check"], json!(true), "{}", one.specs[0]);
    let loss = st.stages[0].cut_loss.expect("Try first's loss is kept for the report");
    let winner = st.try_result.as_ref().unwrap()["winner"].as_u64().unwrap() as usize;
    assert_eq!(Some(loss), st.try_result.as_ref().unwrap()["rows"][winner]["loss"].as_f64());
    // two stages: stage 1 is half the cut (not what Try first measured) and stage 2 is measured on the healed model: both are checked
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut two = Mock::new();
    let st2 = start(&p2, &mut two, spec(21.1)).unwrap();
    assert_eq!(st2.status, "done", "{:?}", st2.error);
    assert_eq!(two.specs.len(), 2);
    assert_eq!((two.specs[0]["skip_check"].clone(), two.specs[1]["skip_check"].clone()), (json!(false), json!(false)));
    assert!(st2.stages.iter().all(|s| s.cut_loss.is_none()));
}

#[test]
fn a_run_saved_before_the_new_fields_still_loads() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    let mut v = serde_json::to_value(&st).unwrap();
    for s in v["stages"].as_array_mut().unwrap() {
        s.as_object_mut().unwrap().remove("cut_loss");
    }
    let old: ndforge_core::reshape::auto::AutoState = serde_json::from_value(v).unwrap();
    assert!(old.stages.iter().all(|s| s.cut_loss.is_none()));
    assert!(!serde_json::from_value::<ndforge_core::reshape::plan::ReshapeSpec>(json!({"source_model_id": "m", "mode": "prune", "blocks": 1})).unwrap().skip_check, "an old spec has no skip_check");
}

#[test]
fn the_report_says_when_the_loss_of_a_cut_was_measured_in_try_first() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, AutoSpec { stages: 1, ..spec(21.1) }).unwrap();
    assert!(st.report.is_some(), "the report was written");
    let dir = ndforge_core::reshape::auto::run_dir(&p, &st.id);
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("auto_report.json")).unwrap()).unwrap();
    assert!(report["stages"][0]["cut_loss"].as_f64().is_some(), "{report}");
    let md = std::fs::read_to_string(dir.join("auto_report.md")).unwrap();
    assert!(md.contains("was measured in Try first"), "{md}");
}


// ------------------------------------------------------------------------------------------------------------------ the thinking check before healing

#[test]
fn the_thinking_gate_chooses_by_how_the_cuts_think_and_plans_the_stages_again() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    // Try first likes the width-only cut best (cut-1) but it lost its thinking; the cut with a block removed (cut-2) still thinks
    fx.checks.insert("cut-1/cut".into(), weak());
    let st = start(&p, &mut fx, AutoSpec { stages: 1, ..gspec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let ids: Vec<&str> = st.steps.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(&ids[..4], ["plan", "measure", "try", "think_gate"]);
    let gate = st.think_gate.as_ref().unwrap();
    assert_eq!(gate["rows"].as_array().unwrap().len(), 2);
    assert_ne!(gate["rows"][0]["row"], gate["rows"][1]["row"]);
    assert_eq!(st.chosen.as_ref().unwrap().blocks, 1, "{:?}", st.chosen);
    assert!(st.notes.iter().any(|n| n.starts_with("Try first's lowest loss was") && n.contains("kept less of its thinking") && n.contains("closed 50 %")), "{:?}", st.notes);
    // the gate's cuts are final cuts that Try first measured: their loss is not measured again
    assert_eq!(fx.specs.len(), 2);
    assert!(fx.specs.iter().all(|s| s["skip_check"] == json!(true)));
    // one stage: the winner's cut and check are the stage's own; no other cut is made
    assert_eq!(fx.cuts, 2);
    let sta = |id: &str| st.steps.iter().find(|s| s.id == id).map(|s| (s.status.clone(), s.note.clone())).unwrap();
    assert_eq!(sta("s1.cut"), ("skipped".into(), Some("made and checked in the thinking gate".into())));
    assert_eq!(sta("s1.check_cut").0, "skipped");
    assert!(st.stages[0].cut_loss.is_some());
    // the loser went at once, the winner after its stage passed
    let pos = |what: &str| fx.done.iter().position(|d| d == what).unwrap_or_else(|| panic!("{what} in {:?}", fx.done));
    assert!(pos("remove cut cut-1") < pos("heal cut-2 teacher=org/Qwen3.8-27B tokens=40 recipe=Some(\"thinking\")"));
    assert!(fx.done.iter().any(|d| d == "heal cut-2 teacher=org/Qwen3.8-27B tokens=40 recipe=Some(\"thinking\")"));
    assert!(!fx.done.iter().any(|d| d.starts_with("heal cut-1")));
    assert!(fx.removed.iter().any(|m| m.name == "cut-2"), "removed after its stage passed: {:?}", fx.removed);
    // the original was checked once
    assert_eq!(fx.asked.get("cut-1/original"), Some(&1));
    assert_eq!(fx.asked.iter().filter(|(k, _)| k.ends_with("/original")).count(), 1);
}

#[test]
fn a_tie_in_thinking_is_decided_by_the_loss_and_a_cut_that_lost_much_is_only_a_warning() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, AutoSpec { stages: 1, ..gspec(21.1) }).unwrap();
    assert_eq!(st.status, "done");
    assert_eq!(st.chosen.as_ref().unwrap().blocks, 0, "equal thinking: Try first's lowest loss stays");
    assert!(!st.notes.iter().any(|n| n.starts_with("Try first's lowest loss was")));
    // every cut lost much of its thinking: the run goes on, with the numbers in a warning
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut fx2 = Mock::new();
    fx2.checks.insert("cut-1/cut".into(), weak());
    fx2.checks.insert("cut-2/cut".into(), weak());
    let st2 = start(&p2, &mut fx2, AutoSpec { stages: 1, ..gspec(21.1) }).unwrap();
    assert_eq!(st2.status, "done", "the gate never halts: {:?}", st2.error);
    assert!(st2.notes.iter().any(|n| n.starts_with("Even the best cut has lost much of its ability to finish its reasoning before healing (closed 50 % against 100 %)")), "{:?}", st2.notes);
    assert_eq!(st2.think_gate.as_ref().unwrap()["lost_much"], json!(true));
}

#[test]
fn with_two_stages_the_checked_cuts_are_deleted_and_stage_one_makes_its_own() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, gspec(21.1)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(st.stages.len(), 2);
    // two gate cuts (final cuts, skip_check) and then the two stages' own cuts (measured as before)
    assert_eq!(fx.cuts, 4);
    assert_eq!(fx.specs.iter().map(|s| s["skip_check"].clone()).collect::<Vec<_>>(), vec![json!(true), json!(true), json!(false), json!(false)]);
    assert!(fx.done.iter().any(|d| d == "remove cut cut-1") && fx.done.iter().any(|d| d == "remove cut cut-2"), "{:?}", fx.done);
    let sta = |id: &str| st.steps.iter().find(|s| s.id == id).map(|s| s.status.clone()).unwrap();
    assert_eq!((sta("s1.cut"), sta("s1.check_cut"), sta("think_gate")), ("done".into(), "done".into(), "done".into()));
    // keep_intermediate keeps every checked cut
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut keep = Mock::new();
    let st2 = start(&p2, &mut keep, AutoSpec { keep_intermediate: true, ..gspec(21.1) }).unwrap();
    assert_eq!(st2.status, "done");
    assert!(keep.removed.is_empty(), "{:?}", keep.removed);
}

#[test]
fn a_stop_at_every_call_of_a_run_with_the_gate_repeats_nothing() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut plain = Mock::new();
    plain.checks.insert("cut-1/cut".into(), weak());
    let full = start(&p, &mut plain, AutoSpec { stages: 1, ..gspec(21.1) }).unwrap();
    assert_eq!(full.status, "done");
    let full = load(&run_dir(&p, &full.id)).unwrap();
    for n in 1..=plain.n {
        let dir = tempdir().unwrap();
        let q = project(dir.path());
        let mut fx = Mock::new();
        fx.checks.insert("cut-1/cut".into(), weak());
        fx.stop_at = Some(n);
        let first = start(&q, &mut fx, AutoSpec { stages: 1, ..gspec(21.1) }).unwrap();
        assert_eq!(first.status, "stopped", "stop at call {n}: {:?}", fx.all.last());
        let last = resume(&q, &mut fx, &first.id).unwrap();
        assert_eq!(last.status, "done", "stop at call {n}: {:?}", last.error);
        assert_eq!(fx.done, plain.done, "stop at call {n} ({:?})", fx.all[n - 1]);
        let (a, b) = (shape(&last), shape(&full));
        for (k, v) in a.as_object().unwrap() {
            assert_eq!(v, &b[k], "stop at call {n}: {k}");
        }
    }
}

#[test]
fn a_run_saved_before_the_gate_existed_loads_and_runs_without_it() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "done");
    let mut v = serde_json::to_value(&st).unwrap();
    v.as_object_mut().unwrap().remove("think_gate");
    let sp = v["spec"].as_object_mut().unwrap();
    sp.remove("think_gate");
    sp.remove("think_gate_n");
    let old: AutoState = serde_json::from_value(v).unwrap();
    assert!(old.think_gate.is_none() && old.spec.think_gate && old.spec.think_gate_n == 2);
    assert!(!old.steps.iter().any(|s| s.id == "think_gate"), "an old run has no gate step, so none runs");
    // a stopped run of that kind goes on without it
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut a = Mock::new();
    a.stop_at = Some(4);
    let first = start(&p2, &mut a, spec(21.1)).unwrap();
    let mut v2 = serde_json::to_value(&first).unwrap();
    v2.as_object_mut().unwrap().remove("think_gate");
    v2["spec"].as_object_mut().unwrap().remove("think_gate");
    std::fs::write(run_dir(&p2, &first.id).join("auto.json"), serde_json::to_vec(&v2).unwrap()).unwrap();
    let last = resume(&p2, &mut a, &first.id).unwrap();
    assert_eq!(last.status, "done", "{:?}", last.error);
    assert!(!a.done.iter().any(|d| d.contains("skip")) && last.think_gate.is_none());
}

#[test]
fn the_number_of_cuts_the_gate_checks_is_validated() {
    assert!(AutoSpec { think_gate_n: 0, ..gspec(21.1) }.validate().is_err());
    assert!(AutoSpec { think_gate_n: 4, ..gspec(21.1) }.validate().is_err());
    assert!(AutoSpec { think_gate_n: 3, ..gspec(21.1) }.validate().is_ok());
}

#[test]
fn the_report_has_the_table_before_healing_and_the_guardrail_warnings() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("cut-1/cut".into(), weak());
    let st = start(&p, &mut fx, AutoSpec { stages: 1, ..gspec(21.1) }).unwrap();
    let dir = run_dir(&p, &st.id);
    let r: Value = serde_json::from_slice(&std::fs::read(dir.join("auto_report.json")).unwrap()).unwrap();
    assert_eq!(r["think_gate"]["rows"].as_array().unwrap().len(), 2);
    assert_eq!(r["guardrails"].as_array().unwrap().len(), 0, "a cut inside the rules has no warning");
    let md = std::fs::read_to_string(dir.join("auto_report.md")).unwrap();
    assert!(md.contains("## Before healing") && md.contains("| Cut | Loss | Finishes reasoning | Loops | Right answers | Chosen |") && md.contains("| yes |"), "{md}");
    // a run that was allowed to remove many blocks: the warning names the numbers
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut big = Mock::new();
    let st2 = start(&p2, &mut big, AutoSpec { stages: 1, allow_many_blocks: true, ..gspec(15.0) }).unwrap();
    let r2: Value = serde_json::from_slice(&std::fs::read(run_dir(&p2, &st2.id).join("auto_report.json")).unwrap()).unwrap();
    let g = r2["guardrails"].as_array().unwrap();
    assert!(g.iter().any(|x| x.as_str().unwrap().contains("whole blocks are removed")), "{r2}");
    assert!(g.iter().any(|x| x.as_str().unwrap().contains("Before healing this cut finished its reasoning in")), "{g:?}");
}

#[test]
fn the_report_says_where_the_time_went_from_the_job_logs_of_the_run() {
    use ndforge_core::reshape::auto_report::{add_timings, markdown};
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let mut st = start(&p, &mut fx, spec(21.1)).unwrap();
    let cfg = d.path().join("cfg");
    let jobs = cfg.join("logs").join("jobs");
    std::fs::create_dir_all(&jobs).unwrap();
    let line = |t: f64, rec: Value| format!("[{t:>8.1}s out] {}\n", json!({"type": "log", "text": format!("timing: {rec}")}));
    let log = |job: &str, recs: &[Value]| format!("# job: {}\n{}", json!({"job": job}), recs.iter().enumerate().map(|(i, r)| line(i as f64, r.clone())).collect::<String>());
    // the logs are from 2020 and the run's times are set to match, so no other test's run can see them
    std::fs::write(jobs.join("20200101-100000-reshape.log"), log("reshape", &[json!({"stage": "write", "seconds": 60.0}), json!({"stage": "job", "seconds": 100.0})])).unwrap();
    std::fs::write(jobs.join("20200101-110000-export.log"), log("export", &[json!({"stage": "convert_f16", "seconds": 30.0}), json!({"stage": "job", "seconds": 100.0})])).unwrap();
    std::fs::write(jobs.join("20190101-100000-export.log"), log("export", &[json!({"stage": "quantize", "seconds": 999.0})])).unwrap();       // before the run
    st.created = "2020-01-01T09:00:00Z".into();
    st.updated = "2020-01-01T12:00:00Z".into();
    std::env::set_var("NDFORGE_CONFIG_DIR", &cfg);
    let mut r = ndforge_core::reshape::auto_report::build(&st);
    add_timings(&mut r, &st);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
    assert_eq!(r["timings"]["summary"]["logs"], 2);
    assert_eq!(r["timings"]["summary"]["stages"]["write"]["seconds"], 60.0);
    assert!(r["timings"]["summary"]["stages"].get("quantize").is_none(), "a log from before the run is not counted");
    let md = markdown(&r);
    assert!(md.contains("## Where the time went") && md.contains("| write | 60 | 1 | 0.30 |") || md.contains("| write | 60 | 1 |"), "{md}");
    // no logs: no section
    let mut none = ndforge_core::reshape::auto_report::build(&st);
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("empty"));
    add_timings(&mut none, &st);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
    assert!(none.get("timings").is_none() || none["timings"].is_null());
    assert!(!markdown(&none).contains("Where the time went"));
}

// ------------------------------------------------------------------------------------------------------------------ a target size: the measured file and its comparison

/// the spec of the target-size tests: the cut model is about 21 B, so Q4_K_M is about 12.9 GB and Q5_K_M about 15 GB
fn tspec(gb: f64) -> AutoSpec {
    AutoSpec { target_gb: Some(gb), ..spec(21.1) }
}

fn step_ids(st: &AutoState) -> Vec<&str> {
    st.steps.iter().map(|s| s.id.as_str()).collect()
}

#[test]
fn without_a_target_nothing_changes_and_with_one_the_steps_and_the_export_list_grow() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let plain = start(&p, &mut fx, spec(21.1)).unwrap();
    assert!(!step_ids(&plain).contains(&"compare_quants") && plain.compare.is_none());
    assert!(fx.done.iter().any(|d| d.contains("[\"q4_k_m\"]")) && !fx.done.iter().any(|d| d.starts_with("target") || d.starts_with("compare")));
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut fx2 = Mock::new();
    let st = start(&p2, &mut fx2, AutoSpec { studio_publish: true, ..tspec(14.0) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let ids = step_ids(&st);
    let pos = |s: &str| ids.iter().position(|x| *x == s).unwrap();
    assert!(pos("export") < pos("check_gguf") && pos("check_gguf") < pos("compare_quants") && pos("compare_quants") < pos("studio_publish"), "{ids:?}");
    assert!(fx2.done.iter().any(|d| d == "target 14"));
    assert!(fx2.done.iter().any(|d| d.contains("[\"q4_k_m\", \"q5_k_m\", \"mixed_measured\", \"f16\"]")), "{:?}", fx2.done);
    let cmp = fx2.done.iter().find(|d| d.starts_with("compare")).unwrap();
    assert!(cmp.contains("model-f16.gguf") && cmp.ends_with("model-q4_k_m.gguf,model-q5_k_m.gguf,model-mixed_measured.gguf"), "{cmp}");
    assert_eq!(st.compare.as_ref().unwrap()["promoted"], false);
    // the quants a person chose are still made, once each
    let d3 = tempdir().unwrap();
    let p3 = project(d3.path());
    let mut fx3 = Mock::new();
    start(&p3, &mut fx3, AutoSpec { quants: vec!["q4_k_m".into(), "q8_0".into()], ..tspec(10.0) }).unwrap();
    assert!(fx3.done.iter().any(|d| d.contains("[\"q4_k_m\", \"q8_0\", \"mixed_measured\", \"f16\"]")), "a target below Q4_K_M: the plain size above it is Q4_K_M itself, {:?}", fx3.done);
}

#[test]
fn a_run_with_a_target_is_checked_and_an_old_state_loads() {
    assert!(AutoSpec { target_gb: Some(0.0), ..gspec(21.1) }.validate().is_err());
    assert!(AutoSpec { target_gb: Some(f64::NAN), ..gspec(21.1) }.validate().is_err());
    assert!(AutoSpec { target_gb: Some(-4.0), ..gspec(21.1) }.validate().is_err());
    assert!(AutoSpec { target_gb: Some(12.5), ..gspec(21.1) }.validate().is_ok());
    let spec_json = serde_json::to_value(spec(21.1)).unwrap();
    let mut old = spec_json.clone();
    old.as_object_mut().unwrap().remove("target_gb");
    assert!(serde_json::from_value::<AutoSpec>(old).unwrap().target_gb.is_none());
    // an auto.json saved before the comparison existed
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    let mut v = serde_json::to_value(&st).unwrap();
    v.as_object_mut().unwrap().remove("compare");
    v["spec"].as_object_mut().unwrap().remove("target_gb");
    let old: AutoState = serde_json::from_value(v).unwrap();
    assert!(old.compare.is_none() && old.spec.target_gb.is_none());
}

#[test]
fn the_plan_counts_the_measured_file_the_16_bit_reference_and_the_minutes() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let with = ndforge_core::reshape::auto::preview(&qwen27(), &tspec(14.0), Some(32.0), Some(900.0), true);
    let without = ndforge_core::reshape::auto::preview(&qwen27(), &spec(21.1), Some(32.0), Some(900.0), true);
    let _ = (&p, &mut fx);
    assert!(with["plan"]["measured_minutes"].is_array() && without["plan"]["measured_minutes"].is_null());
    let steps: Vec<String> = with["plan"]["disk"]["steps"].as_array().unwrap().iter().map(|s| s["step"].as_str().unwrap().to_string()).collect();
    assert!(steps.iter().any(|s| s.contains("sample file")) && steps.iter().any(|s| s.contains("the comparison: the 16-bit GGUF deleted")), "{steps:?}");
    assert!(with["plan"]["disk"]["peak_gb"].as_f64().unwrap() >= without["plan"]["disk"]["peak_gb"].as_f64().unwrap());
    // a target no 3-bit file could reach is a blocker
    let tiny = ndforge_core::reshape::auto::preview(&qwen27(), &tspec(1.0), Some(32.0), Some(900.0), true);
    assert_eq!(tiny["plan"]["ok"], false);
    assert!(tiny["plan"]["blockers"].as_array().unwrap().iter().any(|b| b.as_str().unwrap().contains("smaller than the cut model can be made")));
}

#[test]
fn the_16_bit_file_goes_after_the_comparison_unless_the_intermediate_files_are_kept() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, tspec(14.0)).unwrap();
    let removed: Vec<(&str, String)> = fx.removed.iter().filter(|m| m.kind == "export").map(|m| (m.kind.as_str(), m.name.clone())).collect();
    assert_eq!(removed.len(), 1, "{:?}", fx.removed);
    assert!(removed[0].1.ends_with("-f16"));
    let pos = |what: &str| fx.done.iter().position(|d| d.starts_with(what)).unwrap();
    assert!(pos("remove export") > pos("compare"), "{:?}", fx.done);
    assert!(st.made.iter().any(|m| m.kind == "export" && m.removed));
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut keep = Mock::new();
    start(&p2, &mut keep, AutoSpec { keep_intermediate: true, ..tspec(14.0) }).unwrap();
    assert!(!keep.done.iter().any(|d| d.starts_with("remove export")));
    // a person who chose f16 keeps it
    let d3 = tempdir().unwrap();
    let p3 = project(d3.path());
    let mut f16 = Mock::new();
    start(&p3, &mut f16, AutoSpec { quants: vec!["q4_k_m".into(), "f16".into()], ..tspec(14.0) }).unwrap();
    assert!(!f16.done.iter().any(|d| d.starts_with("remove export")));
}

#[test]
fn only_a_promoted_measured_file_is_published_and_a_16_bit_reference_never_is() {
    for (promote, expect) in [(false, 2), (true, 3)] {
        let d = tempdir().unwrap();
        let p = project(d.path());
        let mut fx = Mock::new();
        fx.promote = promote;
        let st = start(&p, &mut fx, AutoSpec { studio_publish: true, ..tspec(14.0) }).unwrap();
        assert_eq!(st.status, "done", "{:?}", st.error);
        assert_eq!(st.compare.as_ref().unwrap()["promoted"], promote);
        assert!(fx.done.iter().any(|d| d.ends_with(&format!("cut to 21.1B {expect}"))), "promote={promote}: {:?}", fx.done.iter().filter(|d| d.starts_with("publish")).collect::<Vec<_>>());
        let rep: Value = serde_json::from_str(&std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.json")).unwrap()).unwrap();
        assert_eq!(rep["compare"]["promoted"], promote);
        let md = std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.md")).unwrap();
        assert!(md.contains("## Measured file against the plain sizes") && md.contains("| File | GB | Mean KLD | Same top answer | Promoted |"), "{md}");
        assert!(md.contains(if promote { "was published with the others" } else { "was not published" }));
        assert!(md.contains("0.0300 ± 0.0005") && md.contains("93.0 %"));
    }
}

#[test]
fn stop_and_continue_inside_the_comparison_makes_the_call_again_and_nothing_else() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut plain = Mock::new();
    let plain_run = start(&p, &mut plain, tspec(14.0)).unwrap();
    let n = plain.all.iter().position(|a| a == "compare").unwrap() + 1;
    let d2 = tempdir().unwrap();
    let q = project(d2.path());
    let mut fx = Mock::new();
    fx.stop_at = Some(n);
    let first = start(&q, &mut fx, tspec(14.0)).unwrap();
    assert_eq!(first.status, "stopped");
    assert!(first.compare.is_none() && first.steps.iter().find(|s| s.id == "compare_quants").unwrap().status != "done");
    assert!(!fx.done.iter().any(|d| d.starts_with("remove export")), "the reference is kept while the comparison is not done");
    let last = resume(&q, &mut fx, &first.id).unwrap();
    assert_eq!(last.status, "done", "{:?}", last.error);
    // the names of the exports carry the id of the run (a time stamp): the two runs are told apart by it only
    let norm = |v: &[String], id: &str| v.iter().map(|d| d.replace(id, "RUN")).collect::<Vec<_>>();
    assert_eq!(norm(&fx.done, &first.id), norm(&plain.done, &plain_run.id));
    assert_eq!(fx.all.len(), plain.all.len() + 1);
}

#[test]
fn a_comparison_that_fails_does_not_lose_the_files_the_reference_stays_and_the_measured_file_is_not_published() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.compare_error = Some("llama-perplexity was not found.".into());
    let st = start(&p, &mut fx, AutoSpec { studio_publish: true, ..tspec(14.0) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let c = st.compare.as_ref().unwrap();
    assert_eq!(c["promoted"], false);
    assert!(c["skipped"].as_str().unwrap().contains("llama-perplexity was not found") && c["skipped"].as_str().unwrap().contains("not published"));
    assert!(!fx.done.iter().any(|d| d.starts_with("remove export")), "the 16-bit file is kept so that Compare quants can be run by hand");
    assert!(fx.done.iter().any(|d| d.ends_with("cut to 21.1B 2")));
}

#[test]
fn a_worker_event_is_tagged_with_the_step_and_keeps_its_own_keys() {
    let e = ndforge_core::reshape::auto_fx::tag(json!({"kind": "metric", "step": 12, "eta_s": 900}), "s1.heal");
    assert_eq!(e["phase"], "auto");
    assert_eq!(e["auto_step"], "s1.heal");
    assert_eq!((e["kind"].as_str(), e["step"].as_i64(), e["eta_s"].as_i64()), (Some("metric"), Some(12), Some(900)));
}

// ---- the speed settings act (reshape::auto_speed): each knob changes what the run does, and a legacy spec runs as it always did

fn checks_of(fx: &Mock, quick: bool) -> usize {
    fx.done.iter().filter(|d| d.starts_with("check ") && d.contains(&format!("quick={quick}"))).count()
}

#[test]
fn a_legacy_spec_and_the_thorough_preset_make_the_same_calls() {
    let (d1, d2) = (tempdir().unwrap(), tempdir().unwrap());
    let (p1, p2) = (project(d1.path()), project(d2.path()));
    let (mut a, mut b) = (Mock::new(), Mock::new());
    let s1 = start(&p1, &mut a, gspec(21.1)).unwrap();
    // the preset also asks for the search for the best cut (Thorough: 30 minutes); the rest is the legacy run, so it is switched off here and tested on its own below
    let s2 = start(&p2, &mut b, AutoSpec { search_minutes: None, ..gspec(21.1).with_preset("thorough").unwrap() }).unwrap();
    assert_eq!((s1.status.as_str(), s2.status.as_str()), ("done", "done"));
    assert_eq!(a.done, b.done);
    assert_eq!(a.heal_specs, b.heal_specs);
    assert!(a.heal_specs.iter().all(|h| h["seq"].is_null()), "the sequence length is the program's own choice: {:?}", a.heal_specs);
    assert_eq!(a.imatrix, vec![true]);
    assert_eq!(checks_of(&a, true), 0, "the checks are the full ones");
    assert_eq!(s1.steps.iter().map(|s| s.id.clone()).collect::<Vec<_>>(), s2.steps.iter().map(|s| s.id.clone()).collect::<Vec<_>>());
}

#[test]
fn quick_thinking_checks_are_a_choice_of_their_own() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, AutoSpec { think_quick: Some(true), ..gspec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(checks_of(&fx, true) >= 6 && checks_of(&fx, false) == 0, "{:?}", fx.done);
    assert!(fx.done.iter().any(|d| d.starts_with("try samples=128")), "Try first still asks 128 samples: {:?}", fx.done);
    // and the other way round: Quick Try first, full checks
    let d2 = tempdir().unwrap();
    let mut f2 = Mock::new();
    let s2 = start(&project(d2.path()), &mut f2, AutoSpec { quick: true, think_quick: Some(false), ..gspec(21.1) }).unwrap();
    assert_eq!(s2.status, "done");
    assert!(checks_of(&f2, true) == 0 && checks_of(&f2, false) >= 6 && f2.done.iter().any(|d| d.starts_with("try samples=64")), "{:?}", f2.done);
}

#[test]
fn without_the_check_of_the_cut_the_original_is_checked_after_the_first_heal() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, AutoSpec { check_cut: false, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(st.steps.iter().all(|s| !s.id.ends_with(".check_cut")), "{:?}", statuses(&st));
    let originals = fx.done.iter().filter(|d| d.starts_with("check ") && d.contains(" original ")).count();
    assert_eq!(originals, 1, "the original is the yardstick of the gates, checked once: {:?}", fx.done);
    let first_heal = fx.done.iter().position(|d| d.starts_with("heal ")).unwrap();
    let orig = fx.done.iter().position(|d| d.contains(" original ")).unwrap();
    assert!(orig > first_heal, "after the first heal, not before it");
    assert!(fx.done.iter().all(|d| !d.contains(" cut quick")), "no cut was checked: {:?}", fx.done);
    assert!(st.stages.iter().all(|s| s.passed && s.heal_check.as_ref().is_some_and(|c| !c["verdict"].is_null())), "the gates had a baseline");
}

#[test]
fn without_the_retry_a_failed_check_goes_straight_to_the_back_off() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    let st = start(&p, &mut fx, AutoSpec { retry_heal: false, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "gates_failed");
    let kinds: Vec<&str> = st.stages[0].attempts.iter().map(|a| a["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["heal", "back_off"], "no heal on the healed model");
    assert_eq!(fx.cuts, 2, "the first cut and the backed-off cut");
    assert!(fx.heal_specs.len() == 2 && fx.done.iter().all(|d| !d.starts_with("heal heal-")), "{:?}", fx.done);
}

#[test]
fn without_the_back_off_the_run_ends_after_the_retry() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    let st = start(&p, &mut fx, AutoSpec { back_off: false, ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "gates_failed");
    let kinds: Vec<&str> = st.stages[0].attempts.iter().map(|a| a["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["heal", "heal_again"]);
    assert_eq!(fx.cuts, 1, "the cut is not backed off");
    assert!(st.notes.iter().any(|n| n.contains("switched off")), "{:?}", st.notes);
    // neither: the first failure ends the run
    let d2 = tempdir().unwrap();
    let mut f2 = Mock::new();
    f2.checks.insert("healed".into(), bad());
    let s2 = start(&project(d2.path()), &mut f2, AutoSpec { retry_heal: false, back_off: false, ..spec(21.1) }).unwrap();
    assert_eq!(s2.status, "gates_failed");
    assert_eq!(s2.stages[0].attempts.len(), 1);
    assert_eq!(f2.heal_specs.len(), 1);
}

#[test]
fn the_retry_learns_on_a_budget_the_heal_takes_and_the_notes_say_so() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    // stage 1 on 20 M: the retry learns on 80 M (20 + 40 is not a budget); the last stage keeps the recipe's 40 M and retries on 80 M
    let st = start(&p, &mut fx, AutoSpec { stage_a_tokens_m: Some(20), ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "gates_failed");
    let t: Vec<u64> = fx.heal_specs.iter().map(|h| h["tokens_m"].as_u64().unwrap()).collect();
    assert_eq!(&t[..2], &[20, 80], "{t:?}");
    assert!(st.notes.iter().any(|n| n.contains("learns on 80 million tokens") && n.contains("60 million more")), "{:?}", st.notes);
    assert_eq!(retry_total_of(40), 80);
    assert_eq!(retry_total_of(80), 80);
    assert_eq!(retry_total_of(10), 80);
}

fn retry_total_of(m: u32) -> u32 {
    ndforge_core::reshape::auto_plan::retry_total_m(m)
}

#[test]
fn the_heal_tokens_of_a_stage_follow_the_choice() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, AutoSpec { stage_a_tokens_m: Some(10), last_tokens_m: Some(20), ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let t: Vec<u64> = fx.heal_specs.iter().map(|h| h["tokens_m"].as_u64().unwrap()).collect();
    assert_eq!(t, [10, 20]);
    assert_eq!(st.stages.iter().map(|s| s.plan.tokens_m).collect::<Vec<_>>(), [10, 20]);
}

#[test]
fn the_same_sequence_length_is_sent_to_every_heal_when_the_cache_is_shared() {
    let (d1, d2) = (tempdir().unwrap(), tempdir().unwrap());
    // on a 36 GB card the recipe's sequence is 4096 for the larger student and 8192 for the smaller one
    let mut alone = Mock::new();
    alone.vram = Some(36.0);
    let s0 = start(&project(d1.path()), &mut alone, AutoSpec { allow_many_blocks: true, ..spec(17.4) }).unwrap();
    assert_eq!(s0.status, "done", "{:?}", s0.error);
    assert!(alone.heal_specs.iter().all(|h| h["seq"].is_null()));
    let natural: Vec<u32> = s0.stages.iter().map(|s| s.plan.seq).collect();
    assert!(natural.len() == 2 && natural[0] != natural[1], "the premise: the stages differ by themselves: {natural:?}");
    let mut fx = Mock::new();
    fx.vram = Some(36.0);
    let st = start(&project(d2.path()), &mut fx, AutoSpec { allow_many_blocks: true, share_teacher_cache: true, ..spec(17.4) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let want = *natural.iter().min().unwrap();
    assert_eq!(fx.heal_specs.iter().map(|h| h["seq"].as_u64().unwrap() as u32).collect::<Vec<_>>(), [want, want]);
    assert_eq!(st.stages.iter().map(|s| s.plan.seq).collect::<Vec<_>>(), [want, want]);
}

#[test]
fn the_importance_matrix_is_left_out_when_it_is_off() {
    let (d1, d2) = (tempdir().unwrap(), tempdir().unwrap());
    let mut fx = Mock::new();
    start(&project(d1.path()), &mut fx, AutoSpec { imatrix: false, ..spec(21.1) }).unwrap();
    assert_eq!(fx.imatrix, vec![false]);
    let mut on = Mock::new();
    start(&project(d2.path()), &mut on, spec(21.1)).unwrap();
    assert_eq!(on.imatrix, vec![true]);
}

// ---- "I have at most N hours"

#[test]
fn a_time_limit_gives_up_the_least_harmful_settings_and_the_run_says_so() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    // the recipe of this cut takes about 50 to 100 hours; 60 needs a few settings given up
    let st = start(&p, &mut fx, AutoSpec { max_hours: Some(60.0), ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let b = st.budget.as_ref().unwrap();
    assert!(b.fits && !b.dropped.is_empty(), "{b:?}");
    assert_eq!(st.spec.preset.as_deref(), Some("custom"));
    assert_eq!(st.spec.max_hours, Some(60.0), "the limit stays in the spec: the run watches the clock");
    assert!(st.plan.as_ref().unwrap().time.as_ref().unwrap().total_h.1 <= 60.0);
    assert!(st.notes.iter().any(|n| n.contains("To fit your 60 hours")) && b.dropped.iter().all(|d| st.notes.iter().any(|n| n.contains(&d.label) && n.contains(&d.risk))), "{:?}", st.notes);
    // the report lists them
    let rep = report_of(&p, Some(&st.id)).unwrap();
    assert!(rep["markdown"].as_str().unwrap().contains("To fit your 60 hours"), "{}", rep["markdown"]);
    assert_eq!(rep["report"]["budget"]["fits"], true);
}

#[test]
fn a_time_limit_that_cannot_be_kept_ends_the_plan_in_plain_words_and_starts_nothing() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let e = start(&p, &mut fx, AutoSpec { max_hours: Some(2.0), ..spec(21.1) }).unwrap_err();
    assert!(e.0.contains("Even the fastest settings take about") && e.0.contains("Choose more time or a larger size"), "{}", e.0);
    assert!(fx.cuts == 0 && fx.heals == 0, "nothing was made");
}

#[test]
fn a_retry_that_would_end_after_the_time_limit_is_not_started() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut late = Mock::new();
    late.checks.insert("healed".into(), bad());
    // the clock is a thousand hours ahead of the start of the run: no retry fits
    late.clock = Some(late.now_secs() + 1000 * 3600);
    let st = start(&p, &mut late, AutoSpec { max_hours: Some(336.0), ..spec(21.1) }).unwrap();
    assert_eq!(st.status, "gates_failed");
    assert!(st.error.as_deref().unwrap().contains("would end after your time limit of 336 hours"), "{:?}", st.error);
    assert_eq!((st.stages[0].attempts.len(), late.heal_specs.len(), late.cuts), (1, 1, 1), "no second heal, no cut made again");
    assert!(late.removed.is_empty());
    // on time: the retry and the back-off run as always
    let d2 = tempdir().unwrap();
    let mut on_time = Mock::new();
    on_time.checks.insert("healed".into(), bad());
    let st = start(&project(d2.path()), &mut on_time, AutoSpec { max_hours: Some(336.0), ..spec(21.1) }).unwrap();
    assert_eq!(st.stages[0].attempts.len(), 3);
    assert!(st.error.as_deref().unwrap().contains("after 3 tries"));
    // the guard is also for the stage made again (the retry is switched off, the clock is late)
    let d3 = tempdir().unwrap();
    let mut back = Mock::new();
    back.checks.insert("healed".into(), bad());
    back.clock = Some(back.now_secs() + 1000 * 3600);
    let st = start(&project(d3.path()), &mut back, AutoSpec { max_hours: Some(336.0), retry_heal: false, ..spec(21.1) }).unwrap();
    assert!(st.error.as_deref().unwrap().contains("Making the stage again with a smaller cut would end after your time limit"), "{:?}", st.error);
    assert_eq!(back.cuts, 1);
}

#[test]
fn the_report_lists_every_speed_setting_that_is_not_the_recipe_with_its_risk() {
    let (d1, d2, d3) = (tempdir().unwrap(), tempdir().unwrap(), tempdir().unwrap());
    // the recipe: nothing to list
    let (p, mut fx) = (project(d1.path()), Mock::new());
    let st = start(&p, &mut fx, gspec(21.1)).unwrap();
    let md = report_of(&p, Some(&st.id)).unwrap()["markdown"].as_str().unwrap().to_string();
    assert!(md.contains("## Speed settings") && md.contains("The recommended recipe: no setting was lowered to save time."), "{md}");
    assert_eq!(report_of(&p, Some(&st.id)).unwrap()["report"]["speed"]["recipe"], true);
    // Fast: every value is a row with its risk
    let (p, mut fx) = (project(d2.path()), Mock::new());
    let st = start(&p, &mut fx, gspec(21.1).with_preset("fast").unwrap()).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let r = report_of(&p, Some(&st.id)).unwrap();
    let md = r["markdown"].as_str().unwrap();
    assert!(md.contains("## Speed settings (setting: fast)"), "{md}");
    for (label, used) in [("Heal tokens", "20 M tokens"), ("Stages", "one stage"), ("Try first", "Quick (64 texts)"), ("Thinking checks", "Quick (16 questions)"), ("Thinking check before healing", "the best 1 cut"),
                          ("Check of each unhealed cut", "off"), ("Retry heal after a failed check", "off"), ("Cut one step less after a second failed check", "off"), ("Importance matrix for the export", "off")] {
        assert!(md.lines().any(|l| l.starts_with(&format!("| {label} | {used} |"))), "{label}: {md}");
    }
    assert!(md.contains("very likely fail"), "{md}");
    assert!(r["report"]["speed"]["deviations"].as_array().unwrap().len() >= 9);
    // Balanced: its first stage and the shared length
    let (p, mut fx) = (project(d3.path()), Mock::new());
    fx.vram = Some(36.0);
    let st = start(&p, &mut fx, AutoSpec { allow_many_blocks: true, ..gspec(17.4) }.with_preset("balanced").unwrap()).unwrap();
    let md = report_of(&p, Some(&st.id)).unwrap()["markdown"].as_str().unwrap().to_string();
    assert!(md.contains("| Heal tokens of the first stage | 20 M tokens | 40 M tokens |") && md.contains("| Sequence length of the stages | the same in both stages (4096) |"), "{md}");
    // the plan's warnings and the notes said it before the run
    assert!(st.notes.iter().any(|n| n.contains("first stage heals on 20 M tokens")), "{:?}", st.notes);
}

fn progress_events(fx: &Mock) -> Vec<&Value> {
    fx.events.iter().filter(|e| e["kind"] == "auto_progress").collect()
}

#[test]
fn a_run_emits_auto_progress_that_rises_to_a_hundred_and_is_saved_in_the_state() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.tracker = Some(Default::default());
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let ev = progress_events(&fx);
    assert!(ev.len() >= st.steps.len(), "one at every step change");
    let pcts: Vec<f64> = ev.iter().map(|e| e["pct"].as_f64().unwrap()).collect();
    assert!(pcts.windows(2).all(|w| w[1] >= w[0] - 1e-9), "{pcts:?}");
    assert_eq!(*pcts.last().unwrap(), 100.0);
    assert!(pcts.iter().any(|x| *x > 0.0 && *x < 100.0));
    let mid = ev.iter().find(|e| e["step"]["id"] == "s1.heal").unwrap();
    assert_eq!(mid["phase"], "auto");
    assert_eq!(mid["stage"]["n"], 1);
    assert_eq!(mid["stage"]["of"], 2);
    assert_eq!(mid["state"], "plan");
    assert!(mid["left_s"][0].as_f64().unwrap() > 0.0);
    assert!(mid["steps"].as_array().unwrap().iter().any(|s| s["id"] == "s1.heal" && s["weight_min"].as_f64().unwrap() > 1.0));
    // the state keeps the last snapshot and the estimates of the steps
    let saved = load(&run_dir(&p, &st.id)).unwrap();
    assert_eq!(saved.progress.as_ref().unwrap()["pct"], 100.0);
    assert!(saved.steps.iter().find(|s| s.id == "s1.heal").unwrap().est_parts.iter().any(|x| x.0 == "train"));
}

#[test]
fn the_bar_steps_back_only_after_a_retry_and_says_why() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.tracker = Some(Default::default());
    fx.checks.insert("cut-1/healed".into(), bad());
    let st = start(&p, &mut fx, spec(18.0)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let ev = progress_events(&fx);
    let pcts: Vec<f64> = ev.iter().map(|e| e["pct"].as_f64().unwrap()).collect();
    assert!(pcts.windows(2).any(|w| w[1] < w[0]), "a retry reopens steps: {pcts:?}");
    assert!(ev.iter().any(|e| e["retry_added_min"].as_f64().map(|x| x > 0.0).unwrap_or(false)));
    assert!(ev.iter().take_while(|e| e["retry_added_min"].is_null()).collect::<Vec<_>>().windows(2).all(|w| w[1]["pct"].as_f64() >= w[0]["pct"].as_f64()));
}

#[test]
fn a_run_saved_before_the_progress_existed_loads_and_resumes() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.stop_at = Some(4);
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "stopped");
    // the file as an older version wrote it: none of the new keys
    let dir = run_dir(&p, &st.id);
    let file = dir.join("auto.json");
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    v.as_object_mut().unwrap().remove("progress");
    for s in v["steps"].as_array_mut().unwrap() {
        s.as_object_mut().unwrap().remove("est");
        s.as_object_mut().unwrap().remove("est_parts");
    }
    std::fs::write(&file, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    let old = load(&dir).unwrap();
    assert!(old.progress.is_none() && old.steps.iter().all(|s| s.est.is_none() && s.est_parts.is_empty()));
    fx.tracker = Some(Default::default());
    let done = resume(&p, &mut fx, &st.id).unwrap();
    assert_eq!(done.status, "done", "{:?}", done.error);
    assert_eq!(progress_events(&fx).last().unwrap()["pct"], 100.0);
}

#[test]
fn live_events_move_the_running_step_at_most_once_a_second_and_the_state_goes_live() {
    use ndforge_core::reshape::auto_progress::ProgressTracker;
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let mut st = start(&p, &mut fx, spec(21.1)).unwrap();
    // pretend the first heal is running and everything before it is done
    for s in st.steps.iter_mut() {
        s.status = if s.id == "s1.heal" { "running".into() } else if matches!(s.id.as_str(), "plan" | "measure" | "try" | "s1.cut" | "s1.check_cut") { "done".into() } else { "pending".into() };
    }
    let mut t = ProgressTracker::new();
    let first = t.sync(&st, 1000.0);
    assert_eq!(first["step"]["id"], "s1.heal");
    assert_eq!(first["state"], "plan");
    let ev = |done: f64| json!({"kind": "progress", "step": "cache", "value": done / 30.0, "text": "", "detail": {"unit": "tokens", "done": done * 1e6, "total": 30e6, "rate": 3100.0, "rate_unit": "tokens/s", "eta_s": (30.0 - done) * 1e6 / 3100.0}});
    // within a second of the last snapshot: nothing new
    assert!(t.on_event(&ev(1.0), 1000.5).is_none());
    let a = t.on_event(&ev(2.0), 1002.0).unwrap();
    assert_eq!(a["step"]["part"], "cache");
    assert!(a["step"]["sub"].as_str().unwrap().starts_with("Teacher cache: 2.0 M of 30.0 M tokens"), "{}", a["step"]["sub"]);
    assert_eq!(a["state"], "estimating");
    assert!(a["pct"].as_f64().unwrap() > first["pct"].as_f64().unwrap());
    // after a few windows of progress the state is live and the time left comes from the speed
    let mut last = a;
    for (i, done) in [4.0, 6.0, 8.0, 10.0].iter().enumerate() {
        if let Some(s) = t.on_event(&ev(*done), 1062.0 + 61.0 * i as f64) {
            last = s;
        }
    }
    assert_eq!(last["state"], "live");
    assert!(last["left_s"][0].as_f64().unwrap() < last["left_s"][1].as_f64().unwrap());
    assert!(last["elapsed_s"].as_f64().unwrap() > 200.0);
}

// ---- Find the best cut inside Auto-cut (reshape::search), the cut chosen before the run

fn searching(target: f64) -> AutoSpec {
    AutoSpec { search_minutes: Some(20), ..gspec(target) }
}

#[test]
fn the_try_step_can_be_the_search_and_its_best_cuts_feed_the_thinking_gate() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, searching(21.1)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(fx.searches.len(), 1);
    assert_eq!(fx.searches[0]["budget_min"], 20);
    assert!(fx.searches[0]["groups"][0]["cuts"].as_array().unwrap().len() >= 1);
    // one request, no "every run" request: the search is a key of the same job
    assert!(fx.done.iter().any(|c| c.starts_with("search samples=128")), "{:?}", fx.done);
    assert!(!fx.done.iter().any(|c| c.starts_with("try samples=128 runs=[1, 2]")));
    let cands = st.try_cands.clone().unwrap();
    assert!(!cands.is_empty() && cands.len() <= 3);
    let tr = st.try_result.clone().unwrap();
    assert_eq!(tr["rows"].as_array().unwrap().len(), cands.len());
    assert_eq!(tr["search"]["minutes"], 20);
    assert!(tr["rows"][0]["cut_blocks"].as_array().is_some_and(|a| !a.is_empty()));
    // the step keeps its id and gets the search's label
    let try_step = st.steps.iter().find(|s| s.id == "try").unwrap();
    assert_eq!(try_step.label, "Find the best cut");
    // the thinking gate checked the search's cuts: at most think_gate_n of them, each written as a cut
    let checked = fx.done.iter().filter(|c| c.starts_with("check cut-") && c.contains(" cut ")).count();
    assert!(checked >= 1);
    assert!(st.think_gate.is_some());
    // the same blocks are cut in the stage that cuts them all (one stage)
    let q = tempdir().unwrap();
    let p1 = project(q.path());
    let mut one = Mock::new();
    let s1 = start(&p1, &mut one, AutoSpec { stages: 1, ..searching(21.1) }).unwrap();
    assert_eq!(s1.status, "done", "{:?}", s1.error);
    assert!(s1.stages[0].plan.blocks_cut == 0 || s1.drop_blocks.is_some());
}

#[test]
fn a_run_without_search_minutes_and_one_saved_before_it_existed_use_the_classic_try() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, gspec(21.1)).unwrap();
    assert_eq!(st.status, "done");
    assert!(fx.searches.is_empty() && st.try_cands.is_none());
    assert_eq!(st.steps.iter().find(|s| s.id == "try").unwrap().label, "Try the cuts");
    // a state file from before the new fields resumes through the classic Try
    let mut fx2 = Mock::new();
    fx2.stop_at = Some(3);
    let stopped = start(&p, &mut fx2, gspec(21.1)).unwrap();
    assert_eq!(stopped.status, "stopped");
    let file = run_dir(&p, &stopped.id).join("auto.json");
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
    for k in ["try_cands"] {
        v.as_object_mut().unwrap().remove(k);
    }
    for k in ["search_minutes", "cut"] {
        v["spec"].as_object_mut().unwrap().remove(k);
    }
    std::fs::write(&file, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
    let again = resume(&p, &mut fx2, &stopped.id).unwrap();
    assert_eq!(again.status, "done", "{:?}", again.error);
    assert!(fx2.searches.is_empty());
}

#[test]
fn a_stop_at_every_call_of_a_run_with_the_search_continues_without_repeating_the_search() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut plain = Mock::new();
    let full = start(&p, &mut plain, searching(21.1)).unwrap();
    assert_eq!(full.status, "done", "{:?}", full.error);
    for n in 1..=plain.n {
        let dir = tempdir().unwrap();
        let q = project(dir.path());
        let mut fx = Mock::new();
        fx.stop_at = Some(n);
        let first = start(&q, &mut fx, searching(21.1)).unwrap();
        assert_eq!(first.status, "stopped", "stop at call {n}: {:?}", fx.all.last());
        let last = resume(&q, &mut fx, &first.id).unwrap();
        assert_eq!(last.status, "done", "stop at call {n}: {:?}", last.error);
        assert_eq!(fx.done, plain.done, "stop at call {n} ({:?})", fx.all[n - 1]);
        assert_eq!(fx.searches.len(), 1, "stop at call {n}: the search is asked once");
    }
}

#[test]
fn a_search_that_ran_out_of_time_says_so_in_the_notes() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.search_budget_stop = true;
    let st = start(&p, &mut fx, searching(21.1)).unwrap();
    assert!(st.notes.iter().any(|n| n.contains("minutes were used")), "{:?}", st.notes);
    assert_eq!(st.try_result.as_ref().unwrap()["search"]["stop_reason"], "budget");
}

#[test]
fn a_cut_chosen_before_the_run_skips_the_search_and_the_gate_and_cuts_those_blocks() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    // one stage; the blocks and width of a cut that reaches 21.1 B (the planner's own first choice names them)
    let probe = ndforge_core::reshape::auto_plan::plan(&qwen27(), &gspec(21.1), Some(32.0), Some(900.0));
    let chosen = probe.chosen.clone().unwrap();
    assert!(chosen.blocks >= 1 || chosen.intermediate.is_some());
    let blocks: Vec<u32> = if chosen.blocks == 0 { vec![] } else { (0..chosen.blocks).map(|i| 3 + 2 * i).collect() };
    let width = chosen.intermediate;
    let cut = ndforge_core::reshape::auto_plan::PinnedCut { blocks: blocks.clone(), intermediate: width, loss: Some(2.05), search_id: Some("s1".into()), think: None };
    let st = start(&p, &mut fx, AutoSpec { stages: 1, cut: Some(cut), search_minutes: Some(20), ..gspec(21.1) }).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(fx.searches.is_empty() && !fx.done.iter().any(|c| c.starts_with("try ")), "{:?}", fx.done);
    let ids: Vec<&str> = st.steps.iter().map(|s| s.id.as_str()).collect();
    assert!(!ids.contains(&"try") && !ids.contains(&"think_gate"), "{ids:?}");
    assert_eq!(st.chosen.as_ref().unwrap().blocks, chosen.blocks);
    if !blocks.is_empty() {
        let layers: Vec<u32> = blocks.iter().flat_map(|b| (b * 4)..(b * 4 + 4)).collect();
        assert!(fx.done.iter().any(|c| c.contains(&format!("drop={:?}", Some(layers.clone())))), "{:?}", fx.done);
    }
    assert!(st.notes.iter().any(|n| n.contains("chosen before the run")));
    // a cut that does not make the size asked for is refused with the reason
    let bad = ndforge_core::reshape::auto_plan::PinnedCut { blocks: vec![2], intermediate: None, loss: None, search_id: None, think: None };
    let e = start(&p, &mut Mock::new(), AutoSpec { cut: Some(bad), ..gspec(21.1) }).unwrap_err();
    assert!(e.0.contains("not 21.1 B"), "{}", e.0);
}


// ----------------------------------------------------------------------------------------------------------------- healing in rounds

fn rounds_spec(base: AutoSpec) -> AutoSpec {
    AutoSpec { heal_rounds: Some(HealRoundsOpt { round_m: 5, check: None, first_check_m: None }), think_quick: Some(true), ..base }
}

#[test]
fn a_run_with_heal_rounds_hands_every_heal_its_rounds_the_original_s_numbers_and_the_run_s_check_mode() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, rounds_spec(spec(21.1))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(!fx.heal_specs.is_empty());
    for (i, h) in fx.heal_specs.iter().enumerate() {
        let r = &h["rounds"];
        assert!(r.is_object(), "heal {i}: {h}");
        assert_eq!((r["round_m"].clone(), r["check"].clone(), r["final_quick"].clone(), r["stop"].clone(), r["plateau"].clone()), (json!(5), json!("quick"), json!(true), json!("pass"), json!(true)));
        assert_eq!(r["max_tokens_m"], h["tokens_m"], "the cap is the stage's budget");
        assert!(r["baseline"]["quick"]["closed_rate"].is_number(), "the original was checked before the first heal; its numbers go in: {r}");
        assert!(r.get("extend").is_none() || r["extend"] == json!(false));
    }
    // the stage's own record has what the heal said, and the heal knows what it trained
    assert!(st.stages.iter().all(|s| s.passed && s.rounds.len() == 1));
    assert!(st.stages.iter().all(|s| s.heal.as_ref().unwrap().tokens_m as f64 == s.rounds[0]["tokens_m"].as_f64().unwrap()));
    // no check of the healed model is asked for twice by the program: the heal did it, this step reads the saved one
    assert!(fx.done.iter().any(|d| d.starts_with("check ") && d.contains(" healed quick=true")));
}

#[test]
fn a_stage_that_reaches_its_cap_goes_on_in_the_same_run_once_then_backs_off() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    let st = start(&p, &mut fx, rounds_spec(spec(21.1))).unwrap();
    assert_eq!(st.status, "gates_failed");
    let kinds: Vec<&str> = st.stages[0].attempts.iter().map(|a| a["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["heal", "heal_again", "back_off"], "{:?}", st.stages[0].attempts);
    let first: Vec<&Value> = fx.heal_specs.iter().filter(|h| h["rounds"]["max_tokens_m"] == json!(40)).collect();
    let more: Vec<&Value> = fx.heal_specs.iter().filter(|h| h["rounds"]["extend"] == json!(true)).collect();
    assert!(!first.is_empty() && !more.is_empty(), "{:?}", fx.heal_specs);
    assert_eq!(more[0]["rounds"]["max_tokens_m"], json!(80), "cap + 40");
    assert_eq!(more[0]["tokens_m"], json!(40), "the heal's own budget is not changed");
    // the same student: the cut, not the healed model of the first try
    assert!(fx.done.iter().all(|d| !d.starts_with("heal heal-")), "{:?}", fx.done);
    assert!(st.notes.iter().any(|n| n.contains("goes on in rounds for 40 million more")), "{:?}", st.notes);
    assert_eq!(fx.heal_specs.len(), 2 + fx.cuts as usize - 1, "one heal, one extension, and one heal for every cut that was backed off");
}

#[test]
fn a_plateau_goes_straight_to_the_back_off_and_the_time_limit_ends_the_run() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    fx.rounds_stops = ["plateau".to_string(), "plateau".to_string()].into();
    let st = start(&p, &mut fx, rounds_spec(spec(21.1))).unwrap();
    assert_eq!(st.status, "gates_failed");
    let kinds: Vec<&str> = st.stages[0].attempts.iter().map(|a| a["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["heal", "back_off"], "no extension after a plateau: more rounds would not help");
    assert!(fx.heal_specs.iter().all(|h| h["rounds"]["extend"] != json!(true)), "{:?}", fx.heal_specs);
    assert_eq!(fx.cuts, 2);
    // the time limit: the run ends with its report, nothing is tried again
    let d2 = tempdir().unwrap();
    let mut f2 = Mock::new();
    f2.checks.insert("healed".into(), bad());
    f2.rounds_stops = ["time".to_string()].into();
    let s2 = start(&project(d2.path()), &mut f2, rounds_spec(spec(21.1))).unwrap();
    assert_eq!(s2.status, "gates_failed");
    assert_eq!((f2.heal_specs.len(), f2.cuts), (1, 1));
    assert!(s2.error.as_ref().unwrap().contains("time limit") && s2.notes.iter().any(|n| n.contains("time limit")), "{:?} {:?}", s2.error, s2.notes);
}

#[test]
fn without_heal_rounds_the_run_is_what_it_was() {
    let d = tempdir().unwrap();
    let mut fx = Mock::new();
    fx.checks.insert("healed".into(), bad());
    let st = start(&project(d.path()), &mut fx, spec(21.1)).unwrap();
    assert!(fx.heal_specs.iter().all(|h| h.get("rounds").is_none()), "{:?}", fx.heal_specs);
    assert_eq!(st.status, "gates_failed");
    assert!(st.stages.iter().all(|s| s.rounds.is_empty()));
    let text = serde_json::to_string(&st).unwrap();
    let at = text.find("\"rounds\"").map(|i| text[i.saturating_sub(60)..(i + 60).min(text.len())].to_string());
    assert!(at.is_none() && serde_json::to_value(&st.spec).unwrap().get("heal_rounds").is_none(), "the saved state has no rounds key without rounds: {at:?}");
    // an auto.json written before it loads and carries on
    let back: AutoState = serde_json::from_str(&text).unwrap();
    assert_eq!(back, st);
}

#[test]
fn the_report_of_a_run_with_rounds_has_the_table_the_stop_and_the_comparison_with_the_merged_check() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.rounds_stops = ["pass".to_string(), "pass".to_string()].into();
    let st = start(&p, &mut fx, rounds_spec(spec(21.1))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let dir = run_dir(&p, &st.id);
    let r: Value = serde_json::from_slice(&std::fs::read(dir.join("auto_report.json")).unwrap()).unwrap();
    assert!(r["stages"][0]["rounds"]["rows"].is_array() && r["stages"][0]["rounds_saved_m"].as_f64().unwrap() > 0.0, "{}", r["stages"][0]);
    let md = std::fs::read_to_string(dir.join("auto_report.md")).unwrap();
    assert!(md.contains("## Healing in rounds, stage 1 (experimental)") && md.contains("| Round | Tokens so far (M) |") && md.contains("a round passed the thinking check"), "{md}");
    assert!(md.contains("tokens less than the budget") && md.contains("The merged model after round 2 did not pass (closed_rate)"), "{md}");
    assert!(md.contains("The last round check in the same mode (the model as trained, base + adapter)") && md.contains("whether healing in rounds ends as good as one long heal"), "{md}");
    // a run without rounds has none of it
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let st2 = start(&p2, &mut Mock::new(), spec(21.1)).unwrap();
    let md2 = std::fs::read_to_string(run_dir(&p2, &st2.id).join("auto_report.md")).unwrap();
    assert!(!md2.contains("Healing in rounds"));
}

#[test]
fn a_heal_in_rounds_weighs_its_worst_case_says_where_it_is_and_jumps_forward_when_it_stops_early() {
    use ndforge_core::reshape::auto_progress::ProgressTracker;
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let mut st = start(&p, &mut fx, rounds_spec(spec(21.1))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    let rt = st.plan.as_ref().unwrap().time.as_ref().unwrap().steps.iter().find(|s| s.id == "s1.heal").unwrap().rounds.clone().expect("the plan has the rounds");
    // the first heal is running, everything before it is done
    for s in st.steps.iter_mut() {
        s.status = if s.id == "s1.heal" { "running".into() } else if matches!(s.id.as_str(), "plan" | "measure" | "try" | "s1.cut" | "s1.check_cut") { "done".into() } else { "pending".into() };
    }
    let mut t = ProgressTracker::new();
    let first = t.sync(&st, 1000.0);
    let w = first["steps"].as_array().unwrap().iter().find(|s| s["id"] == "s1.heal").unwrap()["weight_min"].as_f64().unwrap();
    assert!((w - (rt.worst.1 * 10.0).round() / 10.0).abs() < 0.11, "the weight is the worst case: {w} {:?}", rt.worst);
    // round 3 of 8 trains: the bar is a fraction of the worst case, the line under it says where the heal is
    let rounds_max = rt.rounds.1;
    for (n, decision) in [(1, "continue"), (2, "continue")] {
        t.on_event(&json!({"kind": "heal_round", "phase": "heal", "round": n, "rounds_max": rounds_max, "tokens_m": 5.0 * n as f64, "cap_m": 40.0, "decision": decision, "check": if n == 2 { json!({"mode": "quick", "verdict": "fail", "main": {"closed_rate": 0.69, "loop_rate": 0.13}}) } else { Value::Null }}), 1010.0 + n as f64 * 2.0);
    }
    let a = t.on_event(&json!({"kind": "metric", "step": 100, "total_steps": 200, "round": 3, "tok_s": 1000.0}), 1030.0).unwrap();
    let sub = a["step"]["sub"].as_str().unwrap();
    assert!(sub.starts_with(&format!("Round 3 of up to {rounds_max} · 10 M of 40 M tokens · last check: fail (closed 69 %, loops 13 %)")), "{sub}");
    assert_eq!(a["state"], "live");
    assert_eq!(a["step"]["rounds"]["n"], 3);
    let (lo, hi) = (a["left_s"][0].as_f64().unwrap(), a["left_s"][1].as_f64().unwrap());
    assert!(lo > 0.0 && lo < hi, "{lo} {hi}: a range, the best case below the worst");
    // a stop decision: only the merge and the check of the merged model are left; the heal is at its end
    let before = a["step"]["pct"].as_f64().unwrap();
    let b = t.on_event(&json!({"kind": "heal_round", "phase": "heal", "round": 3, "rounds_max": rounds_max, "tokens_m": 15.0, "cap_m": 40.0, "decision": "stop_pass", "check": {"verdict": "pass", "main": {"closed_rate": 0.97, "loop_rate": 0.0}}}), 1040.0).unwrap();
    assert!(b["step"]["pct"].as_f64().unwrap() > before && b["step"]["pct"].as_f64().unwrap() >= 95.0, "{b}");
    assert!(b["step"]["sub"].as_str().unwrap().starts_with("After round 3 · 15 M of 40 M tokens"), "{}", b["step"]["sub"]);
    // when the step is done it counts for its whole weight: the bar jumps forward
    let mut done = st.clone();
    done.steps.iter_mut().find(|s| s.id == "s1.heal").unwrap().status = "done".into();
    let c = t.sync(&done, 1100.0);
    assert!(c["pct"].as_f64().unwrap() > b["pct"].as_f64().unwrap());
    // a step that is not a heal in rounds is untouched
    let mut t2 = ProgressTracker::new();
    let mut plain_fx = Mock::new();
    let plain = start(&project(tempdir().unwrap().path()), &mut plain_fx, spec(21.1)).unwrap();
    let mut plain_run = plain.clone();
    for s in plain_run.steps.iter_mut() {
        s.status = if s.id == "s1.heal" { "running".into() } else if matches!(s.id.as_str(), "plan" | "measure" | "try" | "s1.cut" | "s1.check_cut") { "done".into() } else { "pending".into() };
    }
    let snap = t2.sync(&plain_run, 1000.0);
    assert!(snap["step"].get("rounds").is_none());
    assert!(t2.on_event(&json!({"kind": "heal_round", "round": 1, "decision": "continue"}), 1005.0).is_none(), "the event means nothing to a heal that is not in rounds");
}

// ------------------------------------------------------------------------------------------------------------------ vLLM (experimental)

fn vllm_hist() -> ndforge_core::timing::History {
    let mut h = ndforge_core::timing::History::default();
    ndforge_core::vllm_plan::Numbers { startup_s: Some(263.5), long_tps: Some(120.0), model_b: Some(27.0), ..Default::default() }.push_into(&mut h);
    h
}

fn vspec(t: bool, c: bool, a: bool) -> AutoSpec {
    AutoSpec { vllm: Some(ndforge_core::vllm_plan::AutoVllm { teacher_cache: t, checks: c, automatic: a, kv_fp8: false }), ..gspec(21.1) }
}

#[test]
fn a_run_without_vllm_is_exactly_the_run_it_was() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.hist = vllm_hist();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "done");
    assert!(st.engines.is_empty() && st.check_engine.as_deref() != Some("vllm") && fx.vllm_log.is_empty() && fx.cleanups == 0, "no vLLM key: nothing happens");
    assert!(fx.heal_specs.iter().all(|h| h.get("vllm").is_none()));
    let mut v = serde_json::to_value(&st).unwrap();
    v.as_object_mut().unwrap().remove("plan");
    let text = v.to_string();
    assert!(!text.contains("\"vllm\"") && !text.contains("\"engines\""), "the keys are not even written: {:?}", text.match_indices("vllm").map(|(i, _)| &text[i.saturating_sub(40)..(i + 40).min(text.len())]).collect::<Vec<_>>());
    assert!(st.plan.as_ref().unwrap().time.as_ref().unwrap().vllm.is_none());
}

#[test]
fn a_run_with_both_switches_uses_vllm_for_the_answers_and_every_check_and_records_it() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.hist = vllm_hist();
    fx.leak_engine = true;          // a vLLM check leaves the engine running (a crash): the next step must clean it up before training
    let st = start(&p, &mut fx, vspec(true, true, false)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(!fx.vllm_log.iter().any(|l| l.contains("fp8kv")), "the 8-bit KV cache is off unless asked");
    let plan = st.plan.as_ref().unwrap().time.as_ref().unwrap().vllm.as_ref().unwrap();
    assert!(plan.uses("checks") && plan.uses("s1.heal:cache"), "{:?}", plan.decisions);
    assert!(plan.decisions.iter().all(|d| d.step == "checks" || d.step.contains(':')) && !plan.notes.is_empty());
    assert_eq!(fx.heal_specs.iter().filter(|h| h["vllm"]["teacher_cache"] == true).count(), plan.decisions.iter().filter(|d| d.kind == "teacher_cache" && d.use_vllm).count());
    assert!(fx.vllm_log.iter().any(|l| l.starts_with("check ")) && fx.vllm_log.iter().any(|l| l.starts_with("heal teacher cache")));
    assert!(fx.cleanups >= 5 && !fx.engine_running, "the engine is stopped before every step: no training next to it (the mock panics otherwise)");
    assert_eq!(st.check_engine.as_deref(), Some("vllm"));
    let used: Vec<&str> = st.engines.iter().filter(|e| e["engine"] == "vllm").map(|e| e["step"].as_str().unwrap()).collect();
    assert!(used.contains(&"s1.heal:cache") && used.contains(&"s2.check_heal:healed") && used.contains(&"think_gate:original"), "{used:?}");
    let heal = st.engines.iter().find(|e| e["step"] == "s1.heal:cache").unwrap();
    assert_eq!((heal["teacher_quant"].clone(), heal["engine_version"].clone()), (json!("bnb-nf4"), json!("vllm-0.34.0")));
    assert!(heal["estimated_min"]["vllm"].is_array());
    let r = ndforge_core::reshape::auto_report::build(&st);
    assert_eq!(r["vllm"]["asked"]["checks"], true);
    assert!(r["vllm"]["used"].as_u64().unwrap() >= 6 && r["vllm"]["fallbacks"] == 0);
    let md = ndforge_core::reshape::auto_report::markdown(&r);
    assert!(md.contains("## Steps that used vLLM (experimental)") && md.contains("bnb-nf4") && md.contains("vllm-0.34.0"));
    let again = load(&run_dir(&p, &st.id)).unwrap();
    let (a, b) = (serde_json::to_value(&again).unwrap(), serde_json::to_value(&st).unwrap());
    for k in a.as_object().unwrap().keys() {
        assert_eq!(a[k], b[k], "{k} differs after saving");
    }
}

#[test]
fn old_run_files_without_the_vllm_keys_load() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    let mut v = serde_json::to_value(&st).unwrap();
    v.as_object_mut().unwrap().remove("engines");
    v.as_object_mut().unwrap().remove("check_engine");
    v["spec"].as_object_mut().unwrap().remove("vllm");
    let back: AutoState = serde_json::from_value(v).unwrap();
    assert!(back.engines.is_empty() && back.spec.vllm.is_none());
}

#[test]
fn failing_vllm_checks_fall_back_with_a_note_and_the_run_goes_on_with_one_engine() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.hist = vllm_hist();
    fx.vllm_checks_fail = true;
    let st = start(&p, &mut fx, vspec(false, true, false)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(st.check_engine.as_deref(), Some("hf"));
    assert!(st.engines.iter().all(|e| e["fallback"] == true && e["engine"] == "hf"));
    assert!(st.notes.iter().any(|n| n.contains("vLLM could not be used for") && n.contains("the run went on")));
    // after the first fallback every later check is the standard one (no more vLLM attempts)
    assert_eq!(fx.vllm_log.iter().filter(|l| l.starts_with("check ")).count(), 1, "{:?}", fx.vllm_log);
}

#[test]
fn a_heal_whose_vllm_teacher_failed_is_recorded_as_a_fallback() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.hist = vllm_hist();
    fx.teacher_engine = Some(json!({"used": false, "fallback": true, "problem": "CUDA out of memory"}));
    let st = start(&p, &mut fx, vspec(true, false, false)).unwrap();
    assert_eq!(st.status, "done");
    let e = st.engines.iter().find(|e| e["step"] == "s1.heal:cache").unwrap();
    assert_eq!((e["engine"].clone(), e["fallback"].clone()), (json!("hf"), json!(true)));
    assert!(st.notes.iter().any(|n| n.contains("CUDA out of memory")));
}

#[test]
fn automatic_uses_vllm_only_where_the_estimate_is_lower_and_says_why() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.hist = vllm_hist();
    let st = start(&p, &mut fx, vspec(false, false, true)).unwrap();
    assert_eq!(st.status, "done");
    let plan = st.plan.as_ref().unwrap().time.as_ref().unwrap().vllm.as_ref().unwrap();
    assert!(!plan.decisions.is_empty());
    for dc in plan.decisions.iter().filter(|d| d.mode == "automatic") {
        let (v, s) = ((dc.vllm_min.0 + dc.vllm_min.1) / 2.0, (dc.standard_min.0 + dc.standard_min.1) / 2.0);
        assert_eq!(dc.use_vllm, v < s * ndforge_core::vllm_plan::AUTO_MARGIN, "{dc:?}");
        assert!(dc.reason.starts_with("Automatic:"), "{dc:?}");
    }
    // 8 to 20 hours of the original's answers against a few hours of scoring: vLLM wins the answers
    assert!(plan.uses("s1.heal:cache"), "{:?}", plan.decision("s1.heal:cache"));
    // the steps follow the decisions
    for (key, dc) in plan.decisions.iter().map(|d| (&d.step, d)).filter(|(k, _)| k.ends_with(":cache")) {
        let stage = key.trim_start_matches('s').split('.').next().unwrap().parse::<usize>().unwrap() - 1;
        assert_eq!(fx.heal_specs[stage]["vllm"]["teacher_cache"] == true, dc.use_vllm);
    }
}

#[test]
fn the_time_estimate_with_vllm_counts_start_up_and_the_choice_changes_the_total() {
    let ins = qwen27();
    let off = AutoSpec { allow_many_blocks: true, ..gspec(17.4) };
    let on = AutoSpec { vllm: Some(ndforge_core::vllm_plan::AutoVllm { teacher_cache: true, checks: true, automatic: false, kv_fp8: false }), ..off.clone() };
    let h = vllm_hist();
    let est = |s: &AutoSpec| {
        let p = ndforge_core::reshape::auto_plan::plan(&ins, s, Some(32.0), Some(2000.0));
        assert!(p.ok);
        ndforge_core::reshape::auto_time::time_estimate(&ins, s, &p, &h)
    };
    let (a, b) = (est(&off), est(&on));
    assert!(a.vllm.is_none() && b.vllm.is_some());
    assert!(b.total_h.1 < a.total_h.1, "hours of answers become a few: {:?} vs {:?}", a.total_h, b.total_h);
    let heal = b.steps.iter().find(|s| s.id == "s1.heal").unwrap();
    assert!(heal.parts.iter().any(|x| x.id == "vllm_start") && !heal.parts.iter().any(|x| x.label == "load the original"));
    let chk = b.steps.iter().find(|s| s.id == "s1.check_heal").unwrap();
    assert!(chk.parts.iter().any(|x| x.id == "vllm_start"), "every check starts the engine");
    // a 9 B self-test is not a 27 B number
    let mut h9 = ndforge_core::timing::History::default();
    ndforge_core::vllm_plan::Numbers { startup_s: Some(510.6), long_tps: Some(300.0), model_b: Some(9.0), ..Default::default() }.push_into(&mut h9);
    let p = ndforge_core::reshape::auto_plan::plan(&ins, &on, Some(32.0), Some(2000.0));
    let t9 = ndforge_core::reshape::auto_time::time_estimate(&ins, &on, &p, &h9);
    assert!(t9.vllm.as_ref().unwrap().notes.iter().any(|n| n.contains("NOT used")), "{:?}", t9.vllm.unwrap().notes);
}

#[test]
fn the_8_bit_kv_cache_is_for_the_checks_only_and_the_report_says_so() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.hist = vllm_hist();
    let spec = AutoSpec { vllm: Some(ndforge_core::vllm_plan::AutoVllm { teacher_cache: true, checks: true, automatic: false, kv_fp8: true }), ..gspec(21.1) };
    let st = start(&p, &mut fx, spec).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert!(fx.vllm_log.iter().any(|l| l == "engine=vllm_fp8kv"));
    assert!(fx.heal_specs.iter().all(|h| h["vllm"].get("kv_cache_dtype").is_none()), "the teacher's answers never get the 8-bit cache from Auto-cut");
    let c = st.engines.iter().find(|e| e["kind"] == "checks" && e["engine"] == "vllm").unwrap();
    assert_eq!(c["kv_cache"], "fp8");
    assert!(st.plan.as_ref().unwrap().warnings.iter().any(|w| w.contains("8-bit KV cache")) && st.plan.as_ref().unwrap().time.as_ref().unwrap().vllm.as_ref().unwrap().notes.iter().any(|n| n.contains("8-bit KV cache")));
    // off: the key is not even written
    let off = serde_json::to_value(ndforge_core::vllm_plan::AutoVllm { teacher_cache: true, ..Default::default() }).unwrap();
    assert!(off.get("kv_fp8").is_none());
}


// ------------------------------------------------------------------------------------------------------------------ change the settings and continue

fn stopped_at_the_first_heal(p: &Project) -> (Mock, AutoState) {
    let mut fx = Mock::new();
    fx.stop_at = Some(8);          // inspect, measure, try x2, cut, check original, check cut, HEAL
    let st = start(p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "stopped", "{:?}", st.error);
    (fx, st)
}

#[test]
fn the_lock_table_follows_the_state_of_the_steps() {
    use ndforge_core::reshape::auto_change::lock_of;
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (_, st) = stopped_at_the_first_heal(&p);
    assert!(st.steps.iter().find(|s| s.id == "s1.heal").unwrap().started.is_some(), "the heal was begun");
    for k in ["model_id", "target_b", "keep_thinking", "cut", "allow_many_blocks"] {
        assert_eq!(lock_of(&st, k).state, "locked", "{k}");
    }
    for k in ["stages", "quick", "search_minutes"] {
        assert_eq!(lock_of(&st, k).state, "locked", "{k}: the cut / Try first happened");
    }
    assert_eq!(lock_of(&st, "last_tokens_m").state, "free", "stage 2 has not started");
    let a = lock_of(&st, "stage_a_tokens_m");
    assert_eq!(a.state, "confirm");
    assert!(a.redo.unwrap().contains("heal of stage 1 starts again") && a.reason.contains("has started"));
    assert_eq!(lock_of(&st, "think_quick").state, "confirm", "the original was checked in the old mode");
    for k in ["imatrix", "quants", "target_gb", "max_hours", "retry_heal", "vllm", "heal_rounds"] {
        assert!(matches!(lock_of(&st, k).state, "free" | "confirm"), "{k}");
    }
    assert_eq!(lock_of(&st, "heal_rounds").state, "confirm");
    assert_eq!(lock_of(&st, "unheard_of").state, "locked");
}

#[test]
fn a_finished_stage_cannot_change_and_the_preview_names_what_would_be_redone_and_the_time() {
    use ndforge_core::reshape::auto_change::{lock_of, preview};
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut dry = Mock::new();
    start(&project(tempdir().unwrap().path()), &mut dry, spec(21.1)).unwrap();
    let second_heal = dry.all.iter().enumerate().filter(|(_, x)| x.starts_with("heal ")).nth(1).map(|(i, _)| i + 1).unwrap();
    let mut fx = Mock::new();
    fx.stop_at = Some(second_heal);          // stage 1 is healed and checked, stage 2's heal begins
    let st = start(&p, &mut fx, spec(21.1)).unwrap();
    assert_eq!(st.status, "stopped");
    assert_eq!(lock_of(&st, "stage_a_tokens_m").state, "locked");
    assert!(lock_of(&st, "stage_a_tokens_m").reason.contains("Stage 1 is healed already"));
    let patch = |v: serde_json::Value| v.as_object().unwrap().clone();
    let h = ndforge_core::timing::History::default();
    let pv = preview(&st, &patch(json!({"last_tokens_m": 20, "target_b": 18.0, "stage_a_tokens_m": 20})), Some(32.0), Some(900.0), &h);
    assert_eq!(pv["ok"], false);
    let refused: Vec<&str> = pv["refused"].as_array().unwrap().iter().map(|x| x["key"].as_str().unwrap()).collect();
    assert!(refused.contains(&"target_b") && refused.contains(&"stage_a_tokens_m"), "{refused:?}");
    let ok = preview(&st, &patch(json!({"last_tokens_m": 20})), Some(32.0), Some(900.0), &h);
    assert_eq!(ok["ok"], true);
    assert_eq!(ok["diff"][0]["text"], "you changed: heal tokens of the last stage (million) from the recipe's value to 20");
    let (now, was) = (ok["remaining_h"].as_array().unwrap(), ok["was_remaining_h"].as_array().unwrap());
    assert!(now[1].as_f64().unwrap() < was[1].as_f64().unwrap(), "fewer tokens: less time left: {now:?} against {was:?}");
    assert_eq!(preview(&st, &patch(json!({"last_tokens_m": st.spec.last_tokens_m})), Some(32.0), Some(900.0), &h)["diff"], json!([]), "the same value is no change");
}

#[test]
fn a_change_that_redoes_work_needs_the_confirmation_that_names_it() {
    use ndforge_core::reshape::auto_change::apply;
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (_, mut st) = stopped_at_the_first_heal(&p);
    let patch = json!({"stage_a_tokens_m": 20}).as_object().unwrap().clone();
    let h = ndforge_core::timing::History::default();
    let err = apply(&mut st, &patch, false, Some(32.0), Some(900.0), &h, 1_800_000_000).unwrap_err().0;
    assert!(err.contains("needs your confirmation") && err.contains("heal of stage 1 starts again") && err.contains("time left"), "{err}");
    assert!(st.settings_changes.is_empty() && st.start_spec.is_none(), "nothing happened");
    apply(&mut st, &patch, true, Some(32.0), Some(900.0), &h, 1_800_000_000).unwrap();
    assert_eq!(st.settings_changes.len(), 1);
    let c = &st.settings_changes[0];
    assert_eq!((c["who"].clone(), c["confirmed"].clone()), (json!("person"), json!(true)));
    assert!(c["redo"][0].as_str().unwrap().contains("heal of stage 1 starts again") && c["at"].as_str().unwrap().starts_with("2027"));
    assert_eq!(st.spec.stage_a_tokens_m, Some(20));
    assert_eq!(st.start_spec.as_ref().unwrap().stage_a_tokens_m, None, "the start spec is not rewritten");
    assert_eq!(st.stages[0].plan.tokens_m, 20, "the stage whose heal is not finished takes the new number");
    let refused = apply(&mut st, &json!({"target_b": 15.0}).as_object().unwrap().clone(), true, Some(32.0), Some(900.0), &h, 1).unwrap_err().0;
    assert!(refused.contains("cannot change now") && refused.contains("target_b"));
}

#[test]
fn stop_mid_way_continue_with_other_settings_and_nothing_finished_is_redone() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (mut fx, st) = stopped_at_the_first_heal(&p);
    let before = fx.done.clone();
    assert_eq!(before.iter().filter(|x| x.starts_with("cut ")).count(), 1);
    let patch = json!({"last_tokens_m": 20, "imatrix": false, "retry_heal": false}).as_object().unwrap().clone();
    let st2 = ndforge_core::reshape::auto::resume_with(&p, &mut fx, &st.id, Some((patch, false))).unwrap();
    assert_eq!(st2.status, "done", "{:?}", st2.error);
    let after: Vec<String> = fx.done[before.len()..].to_vec();
    assert!(!after.iter().any(|x| x.starts_with("inspect") || x.starts_with("measure") || x.starts_with("try ") || x == "cut cut-1 from qwen3.8-27b blocks=0 drop=None width=Some(14080) reasoning=true"), "finished steps are not run again: {after:?}");
    assert!(after.iter().filter(|x| x.starts_with("check cut-1 original") || x.starts_with("check cut-1 cut")).count() == 0, "the checks that were done are not repeated: {after:?}");
    assert!(after.iter().any(|x| x.starts_with("heal cut-1 ") && x.contains("tokens=40")), "stage 1 keeps its 40 M: its heal had started but its number did not change");
    assert!(after.iter().any(|x| x.starts_with("heal cut-2 ") && x.contains("tokens=20")), "the changed number is used by stage 2: {after:?}");
    assert_eq!(fx.imatrix, vec![false], "the changed export option is used");
    assert_eq!(st2.settings_changes.len(), 1);
    assert_eq!(st2.start_spec.as_ref().unwrap().last_tokens_m, None);
    let r = ndforge_core::reshape::auto_report::build(&st2);
    assert_eq!(r["settings_changes"].as_array().unwrap().len(), 1);
    let md = ndforge_core::reshape::auto_report::markdown(&r);
    assert!(md.contains("## Settings changed while the run went on") && md.contains("heal tokens of the last stage") && md.contains("time left"), "{md}");
    let again = load(&run_dir(&p, &st2.id)).unwrap();
    assert_eq!(again.settings_changes, st2.settings_changes);
    // a run that is done or running cannot change
    let late = ndforge_core::reshape::auto::resume_with(&p, &mut Mock::new(), &st2.id, Some((json!({"imatrix": true}).as_object().unwrap().clone(), false))).unwrap_err().0;
    assert!(late.contains("only be changed on a stopped or failed run"), "{late}");
}

#[test]
fn suggestions_after_a_failed_heal_are_offered_unless_locked() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.fail_heal_once = true;
    let _ = start(&p, &mut fx, spec(21.1));
    let st = latest(&p).unwrap();
    assert_eq!(st.status, "failed");
    let pv = ndforge_core::reshape::auto::change_preview(&p, &st.id, &serde_json::Map::new(), Some(32.0)).unwrap();
    assert_eq!((pv["can_change"].clone(), pv["failed_step"].clone()), (json!(true), json!("s1.heal")));
    let texts: Vec<String> = pv["suggestions"].as_array().unwrap().iter().map(|x| x["text"].as_str().unwrap().to_string()).collect();
    assert!(texts.iter().any(|t| t.contains("ran out of memory")) && texts.iter().any(|t| t.contains("rounds")) && texts.iter().any(|t| t.contains("vLLM")), "{texts:?}");
    assert!(pv["locks"].as_array().unwrap().iter().any(|l| l["key"] == "target_b" && l["state"] == "locked"));
}

#[test]
fn the_checks_of_a_run_are_served_the_way_the_original_was_and_the_report_says_the_route() {
    for mismatch in [false, true] {
        let d = tempdir().unwrap();
        let p = project(d.path());
        let mut fx = Mock::new();
        fx.hist = vllm_hist();
        fx.pick_original = Some("bnb-nf4".into());
        fx.pick_mismatch = mismatch;
        start(&p, &mut fx, vspec(false, true, false)).unwrap();
        let hinted = fx.vllm_log.iter().filter(|l| l.starts_with("engine=vllm@bnb-nf4")).count();
        assert!(fx.vllm_log.iter().any(|l| l == "engine=vllm"), "the original's own check has no hint: {:?}", fx.vllm_log);
        assert!(hinted > 0, "the later checks carry the route of the original: {:?}", fx.vllm_log);
        let st = latest(&p).unwrap();
        let md = std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.md")).unwrap();
        assert!(md.contains(", route bnb-nf4") || md.contains(", route fp8"), "the table names the route: {md}");
        assert_eq!(md.contains("different quantization: not like-for-like"), mismatch, "{md}");
    }
}

#[test]
fn a_failed_vllm_container_offers_the_teacher_cache_off_first() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.fail_heal_once = true;
    fx.fail_text = Some("vLLM was required for the teacher's answers and it did not work: Copying vllm_cache.py into the container: `wslc cp a b:/work/vllm_cache.py` failed (exit code 1): ERROR_PATH_NOT_FOUND".into());
    let sp = AutoSpec { vllm: Some(ndforge_core::vllm_plan::AutoVllm { teacher_cache: true, checks: true, automatic: false, kv_fp8: false }), ..spec(21.1) };
    let _ = start(&p, &mut fx, sp);
    let st = latest(&p).unwrap();
    let pv = ndforge_core::reshape::auto::change_preview(&p, &st.id, &serde_json::Map::new(), Some(32.0)).unwrap();
    let first = &pv["suggestions"][0];
    assert!(first["text"].as_str().unwrap().contains("turn the vLLM teacher cache off") && first["text"].as_str().unwrap().contains("Test the container copy"), "{first}");
    assert_eq!(first["patch"], json!({"vllm": {"teacher_cache": false, "checks": true, "automatic": false}}), "the checks keep their own setting");
}

#[test]
fn a_failed_step_carries_the_last_lines_of_the_tool_into_its_note_and_the_report() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.fail_heal_once = true;
    fx.fail_text = Some("The GGUF conversion failed (exit code 1). Last lines:\nINFO:hf-to-gguf:Model architecture: Qwen3_5ForConditionalGeneration\n  File \"~/llama.cpp/conversion/qwen.py\", line 323, in __init__\n    assert self.opt_num_mtp_layers != 0\nAssertionError".into());
    start(&p, &mut fx, spec(21.1)).unwrap_err();
    let st = latest(&p).unwrap();
    let note = st.steps.iter().find(|s| s.id == "s1.heal").unwrap().note.clone().unwrap();
    assert!(note.contains("failed (exit code 1)") && note.contains("AssertionError") && note.contains("qwen.py\", line 323"), "{note}");
    let md = std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.md")).unwrap();
    assert!(md.contains("**Failed.** The GGUF conversion failed (exit code 1). Last lines:"), "{md}");
    assert!(md.contains("```text\nINFO:hf-to-gguf") && md.contains("assert self.opt_num_mtp_layers != 0\nAssertionError\n```"), "the tail is a block in the report: {md}");
    let row = md.lines().find(|l| l.starts_with("| ") && l.contains("failed")).unwrap_or("");
    assert!(!row.contains("AssertionError"), "the steps table keeps one line: {row}");
    let json: Value = serde_json::from_str(&std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.json")).unwrap()).unwrap();
    assert!(json["error"].as_str().unwrap().contains("AssertionError"));
}

#[test]
fn step_seconds_add_up_over_every_attempt_and_old_runs_still_load() {
    // a step that ran three times: a stop, a cut-off by closing the program, and the one that finished
    let mut s = StepRec::new("s1.heal", "Heal");
    for (a, b, outcome) in [("2026-11-20T01:00:00Z", "2026-11-20T04:00:00Z", "stopped"), ("2026-11-20T05:00:00Z", "2026-11-20T06:30:00Z", "cut off"), ("2026-11-20T07:00:00Z", "2026-11-20T10:45:00Z", "done")] {
        s.started = Some(a.into());
        s.finished = Some(b.into());
        s.close(b, outcome);
    }
    assert_eq!(s.attempts.len(), 3);
    assert_eq!(s.seconds(), Some(3 * 3600 + 5400 + 3 * 3600 + 2700), "the total is every attempt, not the last one (13,500 s)");
    assert_eq!(s.attempts.iter().map(|a| a.seconds).sum::<i64>(), s.spent_s);
    // a run saved before attempts existed: only started / finished, and it loads
    let old: StepRec = serde_json::from_value(json!({"id": "s1.heal", "label": "Heal", "status": "done", "note": null, "started": "2026-11-20T07:00:00Z", "finished": "2026-11-20T10:45:00Z"})).unwrap();
    assert!(old.attempts.is_empty() && old.spent_s == 0);
    assert_eq!(old.seconds(), Some(13500), "an old run keeps its last-attempt time");
}

#[test]
fn a_stop_a_continue_and_a_retry_leave_attempts_that_the_report_adds_up() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.fail_heal_once = true;
    start(&p, &mut fx, spec(21.1)).unwrap_err();           // the heal fails: one attempt of s1.heal
    let st = latest(&p).unwrap();
    let st = resume(&p, &mut fx, &st.id).unwrap();         // continued: a second attempt, done
    assert_eq!(st.status, "done");
    let heal = st.steps.iter().find(|s| s.id == "s1.heal").unwrap();
    assert!(heal.attempts.len() >= 2, "{:?}", heal.attempts);
    assert_eq!(heal.attempts[0].outcome, "failed");
    assert_eq!(heal.attempts.last().unwrap().outcome, "done");
    assert_eq!(heal.seconds(), Some(heal.attempts.iter().map(|a| a.seconds).sum::<i64>()));
    let json: Value = serde_json::from_str(&std::fs::read_to_string(run_dir(&p, &st.id).join("auto_report.json")).unwrap()).unwrap();
    let step = json["steps"].as_array().unwrap().iter().find(|s| s["id"] == "s1.heal").unwrap();
    assert_eq!(step["seconds"].as_i64(), heal.seconds());
    assert!(step["attempts"].as_array().unwrap().len() >= 2);
    let total: i64 = st.steps.iter().filter_map(|s| s.seconds()).sum();
    assert_eq!(json["active_minutes"].as_f64().unwrap(), (total as f64 / 60.0 * 10.0).round() / 10.0, "the report's total is the accumulated seconds");
}

#[test]
fn the_storage_page_refuses_what_a_stopped_run_still_needs_and_allows_it_when_the_run_is_done() {
    use ndforge_core::storage::{delete, plan_delete, scan_items, Ctx, Options, Status};
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Mock::new();
    fx.fail_heal_once = true;
    start(&p, &mut fx, spec(21.1)).unwrap_err();
    let st = latest(&p).unwrap();
    let name = st.stages[0].cut.as_ref().unwrap().name.clone();
    let dir = p.folder.join("models").join(&name);
    std::fs::create_dir_all(dir.join("model")).unwrap();
    std::fs::write(dir.join("model").join("model.safetensors"), vec![1u8; 2048]).unwrap();
    std::fs::write(dir.join("state.json"), "{}").unwrap();
    let ctx = Ctx { project: Some(&p), busy: false, vllm: None };
    let find = |ctx: &Ctx| scan_items(ctx).into_iter().find(|i| i.kind == "model_entry" && i.label.starts_with(&name)).unwrap();
    let it = find(&ctx);
    assert_eq!(it.status, Status::InUse, "{:?}", it.refs);
    assert!(it.refs.iter().any(|r| r.contains("still needs it to go on")));
    let e = plan_delete(&ctx, &[it.id.clone()], &Options { confirm: "DELETE".into(), ..Options::default() }).unwrap_err();
    assert!(e.0.contains("is in use") && e.0.contains("still needs it"), "{}", e.0);
    assert!(dir.is_dir());
    // the run finishes: the model is only referenced by its report now
    let st = resume(&p, &mut fx, &st.id).unwrap();
    assert_eq!(st.status, "done");
    let it = find(&ctx);
    assert_eq!(it.status, Status::Referenced);
    let none = std::sync::atomic::AtomicBool::new(false);
    let tok = delete(&ctx, &[it.id.clone()], &Options { dry_run: true, ..Options::default() }, &none, &mut |_| {}).unwrap()["token"].as_str().map(String::from);
    let r = delete(&ctx, &[it.id], &Options { confirm: "DELETE".into(), token: tok, ..Options::default() }, &std::sync::atomic::AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["deleted"].as_array().unwrap().len(), 1, "{r}");
    assert!(!dir.exists());
    // the run's report still builds and says so
    let after = latest(&p).unwrap();
    assert!(after.notes.iter().any(|n| n.contains("Deleted on") && n.contains("Storage page")), "{:?}", after.notes);
    assert!(after.made.iter().filter(|m| m.name == name).all(|m| m.removed));
    let md = ndforge_core::reshape::auto_report::markdown(&ndforge_core::reshape::auto_report::build(&after));
    assert!(md.contains("Deleted on"), "{md}");
}
