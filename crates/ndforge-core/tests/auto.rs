//! The Auto-cut state machine (`reshape::auto`) with a stand-in for everything it does (no worker, so it runs on every system): the steps in order, stop at every
//! step and continue, the clean-up of the intermediate files, the preflight and the failures.
use std::collections::{BTreeMap, BTreeSet};

use ndforge_core::error::{Error, Result};
use ndforge_core::project::Project;
use ndforge_core::reshape::auto::*;
use ndforge_core::reshape::auto_plan::AutoSpec;
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
    publish_ok: bool,
}

impl Mock {
    fn new() -> Self {
        Mock {
            ins: qwen27(), n: 0, stop_at: None, stop_flag: false, done: Vec::new(), all: Vec::new(), events: Vec::new(), cuts: 0, heals: 0, made: BTreeSet::new(), removed: Vec::new(), dropped: Vec::new(), specs: Vec::new(),
            free_gb: Some(900.0), vram: Some(32.0), data: true, checks: BTreeMap::new(), asked: BTreeMap::new(), losses: BTreeMap::new(), try_acc: BTreeMap::new(), fail_heal_once: false, publish_ok: true,
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
    fn stopped(&self) -> bool {
        self.stop_flag
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
            return Err(Error::new("The graphics card ran out of memory."));
        }
        self.heals += 1;
        let name = match student {
            Student::Made(n) => n.clone(),
            Student::Folder(_) => format!("heal-{}", self.heals),
        };
        self.made.insert(name.clone());
        self.done.push(format!("heal {name} teacher={teacher_repo} tokens={} recipe={:?}", spec.tokens_m, spec.recipe));
        Ok(Out::Done(HealRef { model_id: format!("id-{name}-healed"), dir: format!("models/{name}/healed"), tokens_m: spec.tokens_m, name, summary: Some(json!({"kd_first": 1.5, "kd_last": 0.5, "calib_loss_before": 3.0, "calib_loss_after": 2.4, "teacher_calib_loss": 2.0})) }))
    }
    fn remove(&mut self, what: &Made) -> Result<()> {
        assert!(self.made.contains(&what.name), "only what Auto-cut made may be removed: {what:?}");
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
    fn export(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool) -> Result<Out<Value>> {
        if self.stop_here(format!("export {}", healed.name)) {
            return Ok(Out::Stopped);
        }
        self.done.push(format!("export {} {family} {quants:?} thinking={keep_thinking}", healed.name));
        let ggufs: Vec<Value> = quants.iter().map(|q| json!({"quant": q, "path": format!("exports/{run}-{q}/model-{q}.gguf")})).collect();
        Ok(Out::Done(json!({"dir": format!("exports/{run}"), "ggufs": ggufs})))
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
    let md = std::fs::read_to_string(dir.join("auto_report.md")).unwrap();
    assert!(md.contains("passed on the second heal") && md.contains("healed again"), "{md}");
    assert_eq!(st.report, Some(json!({"json": "auto_report.json", "md": "auto_report.md"})));
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
