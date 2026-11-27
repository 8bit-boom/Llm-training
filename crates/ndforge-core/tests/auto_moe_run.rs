//! Auto-cut of a mixture of experts through the state machine, with a stand-in for everything it does: the same steps as a dense run, Try first by expert count, a cut that passes its
//! thinking check is not healed, a cut that fails it is, a second stage measures the healed model first, back-off gives experts back, and loop passes are never asked for.
use std::collections::BTreeSet;

use ndforge_core::error::Result;
use ndforge_core::project::Project;
use ndforge_core::reshape::auto::*;
use ndforge_core::reshape::auto_moe::MoeAuto;
use ndforge_core::reshape::auto_plan::AutoSpec;
use ndforge_core::reshape::plan::*;
use ndforge_core::reshape::runs::{HealSpec, TrySpec};
use serde_json::{json, Value};
use tempfile::tempdir;

fn inkling() -> Inspect {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/inkling2_inspect.json")).unwrap()).unwrap()
}

#[derive(Default)]
struct Fx {
    log: Vec<String>,
    cuts: Vec<Value>,
    tries: Vec<Value>,
    measures: Vec<(String, u32, String)>,
    made: BTreeSet<String>,
    heals: u32,
    removed: Vec<String>,
    /// the thinking check of a cut / healed model: "good" or "weak"; the first `weak_checks` checks of a cut are weak
    weak_cuts: u32,
    seen_cuts: u32,
    /// the first this many checks of a healed model fail
    weak_healed: u32,
    seen_healed: u32,
    /// the loss rise per key (default: by the number of experts)
    rise: std::collections::BTreeMap<String, f64>,
    exports: Vec<(String, Vec<String>)>,
    stop_at: Option<usize>,
    n: usize,
    stopped: bool,
}

fn good() -> Value {
    json!({"closed_rate": 1.0, "accuracy": 0.75, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800})
}

fn weak() -> Value {
    json!({"closed_rate": 0.5, "accuracy": 0.28, "loop_rate": 0.5, "garbage_rate": 0.0, "median_think_tokens": 800})
}

impl Fx {
    fn hit(&mut self, what: String) -> bool {
        self.n += 1;
        self.log.push(what);
        if self.stop_at == Some(self.n) {
            self.stop_at = None;
            return true;
        }
        false
    }
    fn rise_of(&self, key: &str) -> f64 {
        if let Some(r) = self.rise.get(key) {
            return *r;
        }
        let e = key.split('|').find_map(|p| p.strip_prefix("e=")).and_then(|v| v.parse::<f64>().ok()).unwrap_or(128.0);
        let k = key.split('|').find_map(|p| p.strip_prefix("k=")).and_then(|v| v.parse::<f64>().ok()).unwrap_or(8.0);
        let s = key.split('|').find_map(|p| p.strip_prefix("s=")).map(|s| if s == "freq" { 0.004 } else { 0.012 }).unwrap_or(0.0);
        (128.0 - e) / 128.0 * 0.10 + (8.0 - k) * if k <= 4.0 { 0.04 } else { 0.01 } + s
    }
}

impl Effects for Fx {
    fn emit(&mut self, _: Value) {}
    fn stopped(&self) -> bool {
        self.stopped
    }
    fn model(&self, id: &str) -> Result<ModelRef> {
        Ok(ModelRef { id: id.into(), repo: "thinkingmachines/Inkling-2-Mini".into(), name: "Inkling-2-Mini".into(), family: "inkling2".into() })
    }
    fn has_training_data(&self) -> bool {
        true
    }
    fn vram_gb(&self) -> Option<f64> {
        Some(32.0)
    }
    fn free_disk_gb(&self) -> Option<f64> {
        Some(900.0)
    }
    fn inspect(&mut self, _: &ModelRef) -> Result<Out<Inspect>> {
        Ok(Out::Done(inkling()))
    }
    fn measure(&mut self, id: &str, samples: u32, load: &str, _: Option<&ReasoningCalib>) -> Result<Out<Value>> {
        if self.hit(format!("measure {id}")) {
            return Ok(Out::Stopped);
        }
        self.measures.push((id.into(), samples, load.into()));
        Ok(Out::Done(json!({"key": "k", "tokens": 131072, "load": "packed", "stable_overlap": {"96": 0.9}})))
    }
    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>> {
        if self.hit("try".into()) {
            return Ok(Out::Stopped);
        }
        self.tries.push(serde_json::to_value(spec).unwrap());
        let keys: Vec<String> = spec.moe.as_ref().and_then(|m| m["keys"].as_array()).map(|a| a.iter().filter_map(|k| k.as_str().map(String::from)).collect()).unwrap_or_default();
        let rows: Vec<Value> = keys.iter().map(|k| {
            let rise = self.rise_of(k);
            let (e, kk) = (k.split('|').find_map(|p| p.strip_prefix("e=")).and_then(|v| v.parse::<u32>().ok()).unwrap_or(128), k.split('|').find_map(|p| p.strip_prefix("k=")).and_then(|v| v.parse::<u32>().ok()).unwrap_or(8));
            let s = k.split('|').find_map(|p| p.strip_prefix("s=")).map(String::from);
            let band = if rise < 0.03 { "small" } else if rise < 0.10 { "moderate" } else { "large" };
            json!({"key": k, "experts": e, "top_k": kk, "passes": 3, "scoring": s, "loss": 2.0 * (1.0 + rise), "rise": rise, "band": band, "coverage": 0.9, "load_spread": 1.4})
        }).collect();
        Ok(Out::Done(json!({"baseline": 2.0, "rows": rows})))
    }
    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>> {
        if self.hit(format!("cut {}", spec.source_model_id)) {
            return Ok(Out::Stopped);
        }
        self.cuts.push(serde_json::to_value(spec).unwrap());
        let name = format!("cut-{}", self.cuts.len());
        self.made.insert(name.clone());
        Ok(Out::Done(CutRef { model_id: format!("id-{name}"), name }))
    }
    fn think_check(&mut self, name: &str, target: &str, _quick: bool, _gguf: Option<&str>) -> Result<Out<Value>> {
        if self.hit(format!("check {name} {target}")) {
            return Ok(Out::Stopped);
        }
        let weak_now = (target == "cut" && { self.seen_cuts += 1; self.seen_cuts <= self.weak_cuts }) || (target == "healed" && { self.seen_healed += 1; self.seen_healed <= self.weak_healed });
        Ok(Out::Done(json!({"main": if weak_now { weak() } else { good() }, "control": {"closed_rate": 1.0}, "suite_version": 1})))
    }
    fn heal(&mut self, student: &Student, _teacher: &str, spec: &HealSpec) -> Result<Out<HealRef>> {
        if self.hit(format!("heal {student:?}")) {
            return Ok(Out::Stopped);
        }
        self.heals += 1;
        let name = match student {
            Student::Made(n) => n.clone(),
            Student::Folder(_) => format!("heal-{}", self.heals),
        };
        self.made.insert(name.clone());
        Ok(Out::Done(HealRef { model_id: format!("id-{name}-healed"), dir: format!("models/{name}/healed"), tokens_m: spec.tokens_m, name, summary: Some(json!({"kd_first": 1.5, "kd_last": 0.5})) }))
    }
    fn remove(&mut self, what: &Made) -> Result<()> {
        assert!(self.made.contains(&what.name) || what.kind == "export", "only what Auto-cut made: {what:?}");
        self.removed.push(what.name.clone());
        Ok(())
    }
    fn export(&mut self, run: &str, healed: &HealRef, _family: &str, quants: &[String], _keep: bool) -> Result<Out<Value>> {
        if self.hit("export".into()) {
            return Ok(Out::Stopped);
        }
        self.exports.push((healed.name.clone(), quants.to_vec()));
        let ggufs: Vec<Value> = quants.iter().map(|q| json!({"quant": q, "path": format!("exports/{run}-{q}/m.gguf"), "export": format!("{run}-{q}")})).collect();
        Ok(Out::Done(json!({"dir": format!("exports/{run}"), "hf_dir": healed.dir, "ggufs": ggufs})))
    }
    fn publish(&mut self, _: &Value, _: &str) -> Result<Out<Value>> {
        Ok(Out::Done(json!({"published": false})))
    }
}

fn project(root: &std::path::Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

fn spec(goal: &str, gb: Option<f64>) -> AutoSpec {
    let mut s = AutoSpec::new("inkling-2-mini", 20.0);
    s.moe = Some(MoeAuto { goal: goal.into(), target_gb: gb, top_k: None, scoring: "reap".into(), rebalance: true });
    s.think_gate = false;
    s.quick = true;
    s
}

fn status(st: &AutoState, id: &str) -> String {
    st.steps.iter().find(|s| s.id == id).map(|s| s.status.clone()).unwrap_or_else(|| format!("(no step {id})"))
}

#[test]
fn a_cut_that_passes_its_check_is_exported_without_healing() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx::default();
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    assert_eq!(st.status, "done", "{:?} {:?}", st.error, st.steps.iter().map(|s| (&s.id, &s.status)).collect::<Vec<_>>());
    let ids: Vec<&str> = st.steps.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["plan", "measure", "try", "s1.cut", "s1.check_cut", "s1.heal", "s1.check_heal", "export", "check_gguf"]);
    assert_eq!((fx.heals, fx.cuts.len()), (0, 1));
    assert_eq!((status(&st, "s1.heal"), status(&st, "s1.check_heal")), ("skipped".to_string(), "skipped".to_string()));
    // measured on the packed model with the Quick number of samples; the cut is the 96-expert one, ranked by REAP, no loop passes in it
    assert_eq!(fx.measures[0], ("inkling-2-mini".to_string(), 64, "auto".to_string()));
    let c = &fx.cuts[0];
    assert_eq!((c["experts"]["keep"].clone(), c["experts"]["scoring"].clone(), c["experts"]["rebalance"].clone(), c["calib_len"].clone(), c["load"].clone()), (json!(96), json!("reap"), json!(true), json!(2048), json!("auto")));
    assert!(c.get("passes").is_none() && c.get("top_k").is_none() && c["skip_check"] == json!(true), "{c}");
    // the export is made from the cut itself, which was not deleted
    assert_eq!(fx.exports[0].0, "cut-1");
    assert!(fx.removed.is_empty(), "{:?}", fx.removed);
    assert!(st.notes.iter().any(|n| n.contains("was not healed")));
    // Try first asked about the count, the other rankings and the neighbours, never about passes
    let keys: Vec<String> = fx.tries[0]["moe"]["keys"].as_array().unwrap().iter().map(|k| k.as_str().unwrap().to_string()).collect();
    assert!(keys.contains(&"base".to_string()) && keys.contains(&"e=96".to_string()) && keys.contains(&"e=96|s=freq".to_string()) && keys.contains(&"e=96|s=gate".to_string()), "{keys:?}");
    assert!(keys.iter().all(|k| !k.contains("p=")), "{keys:?}");
    assert_eq!(fx.tries[0]["len"], json!(2048));
}

#[test]
fn a_cut_that_fails_its_check_is_healed_and_checked_again() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx { weak_cuts: 1, ..Default::default() };
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(fx.heals, 1);
    assert_eq!((status(&st, "s1.heal"), status(&st, "s1.check_heal")), ("done".to_string(), "done".to_string()));
    assert_eq!(fx.exports[0].0, "cut-1");
    assert!(st.stages[0].passed);
    // the healed one was checked with the same baseline
    assert!(fx.log.iter().any(|l| l.starts_with("check cut-1 healed")));
}

#[test]
fn two_stages_measure_the_healed_model_before_the_second_cut() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx::default();
    let st = start(&p, &mut fx, spec("size", Some(9.0))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(fx.cuts.len(), 2);
    assert_eq!((fx.cuts[0]["experts"]["keep"].clone(), fx.cuts[1]["experts"]["keep"].clone()), (json!(96), json!(64)));
    assert_eq!(fx.cuts[1]["source_model_id"], json!("id-cut-1"), "the second stage starts from the first one's model");
    // the first model was measured before its cut was made into the second, and the first stage's cut is deleted once the second has passed
    assert_eq!(fx.measures.iter().map(|m| m.0.as_str()).collect::<Vec<_>>(), ["inkling-2-mini", "id-cut-1"]);
    assert!(fx.removed.contains(&"cut-1".to_string()), "{:?}", fx.removed);
    assert_eq!(fx.exports[0].0, "cut-2");
    assert_eq!(fx.heals, 0);
}

#[test]
fn the_ranking_with_the_lowest_loss_is_used() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx::default();
    fx.rise.insert("e=96".into(), 0.05);
    fx.rise.insert("e=96|s=freq".into(), 0.02);
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(fx.cuts[0]["experts"]["scoring"], json!("freq"));
    assert!(st.notes.iter().any(|n| n.contains("how often the router picks them")), "{:?}", st.notes);
}

#[test]
fn the_speed_goal_lowers_the_experts_per_token_as_far_as_try_first_allows() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx::default();
    let st = start(&p, &mut fx, spec("speed", None)).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(fx.cuts[0]["top_k"], json!(6));
    assert!(fx.cuts[0].get("experts").is_none(), "no experts are removed for speed alone: {}", fx.cuts[0]);
    assert!(fx.cuts[0].get("passes").is_none());
    // too costly: the experts per token stay
    let p2 = project(&d.path().join("b"));
    let mut fx2 = Fx::default();
    fx2.rise.insert("k=6".into(), 0.2);
    let st2 = start(&p2, &mut fx2, spec("speed", None)).unwrap();
    assert_eq!(st2.status, "done", "{:?}", st2.error);
    assert!(fx2.cuts.is_empty() || fx2.cuts[0].get("top_k").is_none(), "{:?}", fx2.cuts);
    assert!(st2.notes.iter().any(|n| n.contains("costs too much")), "{:?}", st2.notes);
}

#[test]
fn two_failed_heals_back_the_cut_off_by_16_experts_and_make_the_stage_again() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx { weak_cuts: 100, weak_healed: 2, ..Default::default() };
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(fx.cuts.iter().map(|c| c["experts"]["keep"].clone()).collect::<Vec<_>>(), [json!(96), json!(112)], "the second cut gives 16 experts back");
    assert_eq!(fx.heals, 3, "the heal, the heal again, the heal of the backed-off cut");
    assert!(st.notes.iter().any(|n| n.contains("backed off one step") && n.contains("112 experts per layer")), "{:?}", st.notes);
    assert!(st.stages[0].passed && st.stages[0].attempts.len() == 3);
    assert_eq!(st.plan.as_ref().unwrap().moe.as_ref().unwrap().experts_after, 112);
    // the failed tries are cleaned up, the passing one is exported
    assert_eq!(fx.exports[0].0, "cut-2");
    assert!(fx.removed.contains(&"cut-1".to_string()));
}

#[test]
fn three_failures_end_the_run_with_its_report() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx { weak_cuts: 100, weak_healed: 100, ..Default::default() };
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    assert_eq!(st.status, "gates_failed", "{:?}", st.error);
    assert!(st.error.as_deref().unwrap_or("").contains("failed"), "{:?}", st.error);
    assert!(fx.exports.is_empty());
}

#[test]
fn stop_and_continue_give_the_same_run() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut a = Fx::default();
    let st_a = start(&p, &mut a, spec("size", Some(12.0))).unwrap();
    let d2 = tempdir().unwrap();
    let p2 = project(d2.path());
    let mut b = Fx { stop_at: Some(4), ..Default::default() };
    let st_b = start(&p2, &mut b, spec("size", Some(12.0))).unwrap();
    assert_eq!(st_b.status, "stopped");
    let st_b = resume(&p2, &mut b, &st_b.id).unwrap();
    assert_eq!(st_b.status, "done", "{:?}", st_b.error);
    let ids = |s: &AutoState| s.steps.iter().map(|x| (x.id.clone(), x.status.clone())).collect::<Vec<_>>();
    assert_eq!(ids(&st_a), ids(&st_b));
    assert_eq!(a.cuts.len(), b.cuts.len());
}

#[test]
fn the_quality_goal_is_refused_with_the_way_to_get_it_and_passes_are_never_asked_for() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx::default();
    let err = start(&p, &mut fx, spec("quality", Some(12.0))).unwrap_err();
    assert!(err.0.contains("training"), "{}", err.0);
    let mut fx = Fx::default();
    let st = start(&project(&d.path().join("x")), &mut fx, spec("size", Some(9.0))).unwrap();
    let all = serde_json::to_string(&(&fx.cuts, &fx.tries)).unwrap();
    assert!(!all.contains("\"passes\":") && !all.contains("p="), "{all}");
    assert_eq!(st.status, "done");
}

#[test]
fn the_report_shows_the_experts_kept_in_every_layer_and_the_active_parameters() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx::default();
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    let r = ndforge_core::reshape::auto_report::build(&st);
    let m = &r["moe"];
    assert_eq!((m["experts_after"].clone(), m["experts_before"].clone(), m["loop_passes"].clone(), m["layers"].clone()), (json!(96), json!(128), json!(3), json!(24)));
    assert_eq!(m["stages"][0]["kept_per_layer"].as_array().unwrap().len(), 24);
    assert!(m["stages"][0]["kept_per_layer"].as_array().unwrap().iter().all(|n| n == &json!(96)));
    assert!(m["active_b"].as_f64().unwrap() > 3.0 && m["q4_gb"].as_f64().unwrap() < 13.0);
    assert_eq!(m["stages"][0]["healed"], json!(false));
    let md = ndforge_core::reshape::auto_report::markdown(&r);
    assert!(md.contains("## Mixture of experts") && md.contains("Experts kept in every one of the 24 layers: 96 of 128") && md.contains("not healed") && md.contains("never takes them out"), "{md}");
    assert!(md.contains("| Try first | Experts |") && md.contains("e=96"), "{md}");
}

#[test]
fn change_settings_and_continue_works_for_a_mixture_of_experts() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let mut fx = Fx { weak_cuts: 100, stop_at: Some(6), ..Default::default() };
    let st = start(&p, &mut fx, spec("size", Some(12.0))).unwrap();
    assert_eq!(st.status, "stopped", "{:?} {:?}", st.error, st.step);
    let patch: serde_json::Map<String, Value> = json!({"last_tokens_m": 80, "moe": {"goal": "size", "target_gb": 9.0, "scoring": "reap", "rebalance": true}}).as_object().unwrap().clone();
    let pv = change_preview(&p, &st.id, &patch, Some(32.0)).unwrap();
    let locks: std::collections::BTreeMap<String, String> = pv["locks"].as_array().unwrap().iter().map(|l| (l["key"].as_str().unwrap().to_string(), l["state"].as_str().unwrap().to_string())).collect();
    assert_eq!(locks.get("moe").map(String::as_str), Some("locked"), "the cut is decided once the experts are measured: {locks:?}");
    assert_eq!(locks.get("last_tokens_m").map(String::as_str), Some("free"));
    // the tokens alone change the time and the plan, and the run goes on to the end
    let patch: serde_json::Map<String, Value> = json!({"last_tokens_m": 40}).as_object().unwrap().clone();
    let pv = change_preview(&p, &st.id, &patch, Some(32.0)).unwrap();
    assert_eq!(pv["can_change"], json!(true));
    let st = resume_with(&p, &mut fx, &st.id, Some((patch, true))).unwrap();
    assert_eq!(st.status, "done", "{:?}", st.error);
    assert_eq!(st.stages[0].plan.tokens_m, 40);
    assert_eq!(st.settings_changes.len(), 1);
}
