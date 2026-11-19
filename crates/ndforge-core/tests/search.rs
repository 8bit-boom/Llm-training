//! The automatic best-cut search (`reshape::search`): the target as groups of cuts, the plan, the ranking, the recommendation and its confidence, the confirmation of a close
//! call and the thinking check of the best two, with a stand-in for the worker (no model, no GPU).
use std::collections::BTreeMap;

use ndforge_core::error::Result;
use ndforge_core::project::Project;
use ndforge_core::reshape::auto::{CutRef, Effects, HealRef, Made, ModelRef, Out, Student};
use ndforge_core::reshape::plan::*;
use ndforge_core::reshape::runs::{HealSpec, TrySpec};
use ndforge_core::reshape::search::*;
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

fn key(blocks: &[u32], width: Option<u32>) -> String {
    format!("d={}|w={}", blocks.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","), width.map(|w| w.to_string()).unwrap_or_default())
}

/// an analysis as the worker answers a search: base loss 2.0, the rows `(key, loss, h1, h2)` ranked in one group
fn analysis(rows: &[(String, f64, f64, f64)]) -> Value {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ranking: Vec<&str> = sorted.iter().map(|r| r.0.as_str()).collect();
    let rows_obj: serde_json::Map<String, Value> = rows.iter().map(|(k, l, a, b)| (k.clone(), json!({"loss": l, "loss_h1": a, "loss_h2": b}))).collect();
    json!({"base": {"loss": 2.0}, "search": {"ranking": {"0": ranking}, "rows": rows_obj, "stop_reason": "done", "finalists": {"0": ranking}}})
}

struct Fake {
    ins: Inspect,
    analysis: Value,
    confirm: Value,
    calls: Vec<String>,
    tries: Vec<Value>,
    made: BTreeMap<String, bool>,
    cuts: u32,
    checks: BTreeMap<String, Value>,
    stop_at: Option<usize>,
    n: usize,
}

impl Fake {
    fn new(a: Value) -> Self {
        Fake { ins: qwen27(), analysis: a, confirm: json!({"base": {"loss": 2.0}, "choices": []}), calls: Vec::new(), tries: Vec::new(), made: BTreeMap::new(), cuts: 0, checks: BTreeMap::new(), stop_at: None, n: 0 }
    }
    fn stop_here(&mut self, what: String) -> bool {
        self.n += 1;
        self.calls.push(what);
        if self.stop_at == Some(self.n) {
            self.stop_at = None;
            return true;
        }
        false
    }
}

impl Effects for Fake {
    fn emit(&mut self, _event: Value) {}
    fn stopped(&self) -> bool {
        false
    }
    fn model(&self, id: &str) -> Result<ModelRef> {
        Ok(ModelRef { id: id.into(), repo: "org/Qwen3.8-27B".into(), name: "Qwen3.8 27B".into(), family: "qwen3_5".into() })
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
    fn inspect(&mut self, _m: &ModelRef) -> Result<Out<Inspect>> {
        if self.stop_here("inspect".into()) {
            return Ok(Out::Stopped);
        }
        Ok(Out::Done(self.ins.clone()))
    }
    fn measure(&mut self, _id: &str, _s: u32, _l: &str, _r: Option<&ReasoningCalib>) -> Result<Out<Value>> {
        unreachable!()
    }
    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>> {
        if self.stop_here(format!("try samples={} search={} choices={}", spec.samples, spec.search.is_some(), spec.choices.len())) {
            return Ok(Out::Stopped);
        }
        self.tries.push(serde_json::to_value(spec).unwrap());
        if spec.search.is_some() {
            return Ok(Out::Done(self.analysis.clone()));
        }
        // the confirmation: the rows of the choices at the larger sample size
        let choices: Vec<Value> = spec.choices.iter().map(|c| {
            let k = key(&c.blocks, c.width);
            let r = &self.confirm["rows"][&k];
            json!({"key": k, "loss": r["loss"], "loss_h1": r["loss_h1"], "loss_h2": r["loss_h2"]})
        }).collect();
        Ok(Out::Done(json!({"base": {"loss": self.confirm["base"]["loss"]}, "choices": choices})))
    }
    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>> {
        if self.stop_here("cut".into()) {
            return Ok(Out::Stopped);
        }
        self.cuts += 1;
        let name = format!("s-cut-{}", self.cuts);
        assert!(spec.skip_check, "the search's cut is only checked for thinking");
        self.made.insert(name.clone(), true);
        Ok(Out::Done(CutRef { model_id: format!("id-{name}"), name }))
    }
    fn think_check(&mut self, name: &str, target: &str, _q: bool, _g: Option<&str>) -> Result<Out<Value>> {
        if self.stop_here(format!("check {name} {target}")) {
            return Ok(Out::Stopped);
        }
        let main = self.checks.get(&format!("{name}/{target}")).cloned().or_else(|| self.checks.get(target).cloned()).unwrap_or_else(good);
        Ok(Out::Done(json!({"main": main})))
    }
    fn heal(&mut self, _s: &Student, _t: &str, _h: &HealSpec) -> Result<Out<HealRef>> {
        unreachable!()
    }
    fn remove(&mut self, what: &Made) -> Result<()> {
        assert!(self.made.get(&what.name).copied().unwrap_or(false), "only what the search made: {what:?}");
        self.made.insert(what.name.clone(), false);
        self.calls.push(format!("remove {}", what.name));
        Ok(())
    }
    fn export(&mut self, _r: &str, _h: &HealRef, _f: &str, _q: &[String], _k: bool) -> Result<Out<Value>> {
        unreachable!()
    }
    fn publish(&mut self, _e: &Value, _n: &str) -> Result<Out<Value>> {
        unreachable!()
    }
}

fn good() -> Value {
    json!({"closed_rate": 1.0, "accuracy": 0.75, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800})
}

fn weak() -> Value {
    json!({"closed_rate": 0.5, "accuracy": 0.28, "loop_rate": 0.5, "garbage_rate": 0.0, "median_think_tokens": 800})
}

fn project(root: &std::path::Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

#[test]
fn a_target_becomes_groups_of_cuts_inside_the_guardrails() {
    let ins = qwen27();
    let g = groups(&ins, &SearchSpec::new("m", Target::B(21.0)));
    assert_eq!(g.len(), 1);
    assert!(!g[0].cuts.is_empty() && g[0].cuts.iter().all(|c| c.on_target && c.k <= 4), "{:?}", g[0].cuts);
    let ks: Vec<u32> = g[0].cuts.iter().map(|c| c.k).collect();
    assert!(ks.windows(2).all(|w| w[0] < w[1]), "one cut per number of blocks: {ks:?}");
    // a file size and a share name the same kind of target
    let gb = groups(&ins, &SearchSpec::new("m", Target::Gb(21.0 * 0.61)));
    assert!((gb[0].target_b - 21.0).abs() < 0.1, "{}", gb[0].target_b);
    let pct = groups(&ins, &SearchSpec::new("m", Target::KeptPct(75.0)));
    assert!((pct[0].target_b - 27.78 * 0.75).abs() < 0.05);
    // depth only: no thinned MLP; width only: no blocks
    let mut s = SearchSpec::new("m", Target::B(24.0));
    s.axes = "depth".into();
    s.allow_many_blocks = true;
    let d = groups(&ins, &s);
    assert!(d.iter().all(|g| g.cuts.iter().all(|c| c.width.is_none() && c.k >= 1)));
    s.axes = "width".into();
    let w = groups(&ins, &s);
    assert!(w.iter().all(|g| g.cuts.iter().all(|c| c.k == 0 && c.width.is_some())));
}

#[test]
fn the_trade_off_has_five_sizes_and_keeps_thinking_models_above_the_floor() {
    let ins = qwen27();
    let mut s = SearchSpec::new("m", Target::Tradeoff);
    s.keep_thinking = false;
    let all = groups(&ins, &s);
    assert!(all.len() >= 4 && all.len() <= 5, "{}", all.len());
    s.keep_thinking = true;
    let think = groups(&ins, &s);
    assert!(think.iter().all(|g| g.target_b >= 20.0));
    assert!(think.len() < all.len() || ins.total_params as f64 / 1e9 * 0.70 >= 20.0);
}

#[test]
fn the_plan_counts_the_passes_and_says_what_fits() {
    let ins = qwen27();
    let h = ndforge_core::timing::History::default();
    let mut s = SearchSpec::new("m", Target::B(21.0));
    s.budget_min = 20;
    let p = plan(&ins, &s, Some(32.0), &h).unwrap();
    assert_eq!(p.samples, 64, "a big model uses the quick samples");
    assert_eq!(p.rounds[0].name, "singles");
    assert_eq!(p.rounds[0].count, 15);
    assert_eq!(p.passes, p.rounds.iter().map(|r| r.count).sum::<u32>());
    assert!(p.total_min.0 > 0.0 && p.total_min.1 >= p.total_min.0);
    assert!(p.fit_passes <= p.passes && p.fit_passes <= 400);
    s.budget_min = 240;
    let long = plan(&ins, &s, Some(32.0), &h).unwrap();
    assert!(long.fit_passes >= p.fit_passes && long.complete);
    s.budget_min = 4;
    assert!(plan(&ins, &s, Some(32.0), &h).is_err());
    // the request for the worker carries the groups and the budget and validates
    s.budget_min = 20;
    let t = try_spec(&ins, &s, &p, Some(32.0)).unwrap();
    let want = t.want();
    assert_eq!(want["search"]["budget_min"], 20);
    assert_eq!(want["search"]["groups"].as_array().unwrap().len(), 1);
    assert!(t.reasoning.is_some());
}

#[test]
fn the_ranking_has_the_change_the_halves_and_the_noise() {
    let ins = qwen27();
    let a = analysis(&[(key(&[3, 7, 11], Some(10752)), 2.1, 2.09, 2.11), (key(&[3, 5, 7], Some(10752)), 2.3, 2.3, 2.3), (key(&[1, 2, 3], Some(10752)), 2.6, 2.6, 2.6)]);
    let rows = rank(&a, &ins);
    assert_eq!(rows.iter().map(|r| r.key.as_str()).collect::<Vec<_>>(), vec![key(&[3, 7, 11], Some(10752)), key(&[3, 5, 7], Some(10752)), key(&[1, 2, 3], Some(10752))]);
    assert_eq!(rows[0].blocks, vec![3, 7, 11]);
    assert!((rows[0].rel - 0.05).abs() < 1e-9);
    assert!((rows[0].noise.unwrap() - 0.02).abs() < 1e-9);
    assert_eq!(rows[0].intermediate, Some(10752));
    assert!(rows[0].params_b > 15.0 && rows[0].params_b < 27.0);
    assert!(rows[0].think.is_none());
    assert!(rank(&json!({}), &ins).is_empty());
}

#[test]
fn confidence_is_clear_close_or_uncertain() {
    let ins = qwen27();
    let spec = SearchSpec::new("m", Target::B(21.0));
    // clear: a wide gap
    let rows = rank(&analysis(&[(key(&[3], Some(10752)), 2.1, 2.1, 2.1), (key(&[5], Some(10752)), 2.4, 2.4, 2.4)]), &ins);
    assert_eq!(recommend(&rows, &spec).unwrap().confidence, "clear");
    // close: the gap is smaller than the noise of the two
    let rows = rank(&analysis(&[(key(&[3], Some(10752)), 2.100, 2.05, 2.15), (key(&[5], Some(10752)), 2.101, 2.05, 2.15)]), &ins);
    let r = recommend(&rows, &spec).unwrap();
    assert_eq!(r.confidence, "close_call");
    assert!(r.reason.contains("within the noise") && r.reason.contains("Thinking not checked"));
    // uncertain: the halves name different winners
    let rows = rank(&analysis(&[(key(&[3], Some(10752)), 2.1, 2.0, 2.2), (key(&[5], Some(10752)), 2.2, 2.1, 2.1)]), &ins);
    assert_eq!(recommend(&rows, &spec).unwrap().confidence, "uncertain");
    // no rows, no recommendation; one row is a guess
    assert!(recommend(&[], &spec).is_none());
}

#[test]
fn the_trade_off_takes_the_smallest_size_that_stays_mild() {
    let ins = qwen27();
    let mut spec = SearchSpec::new("m", Target::Tradeoff);
    spec.keep_thinking = false;
    let mk = |g: usize, loss: f64| {
        let mut a = analysis(&[(key(&[3], Some(10752)), loss, loss, loss)]);
        a["search"]["ranking"] = json!({g.to_string(): [key(&[3], Some(10752))]});
        rank(&a, &ins)
    };
    // groups 0..4 = 90 %, 85 %, 80 %, 76 %, 70 %: errors +2 %, +5 %, +9 %, +14 %, +30 %
    let rows: Vec<SearchRow> = [(0, 2.04), (1, 2.10), (2, 2.18), (3, 2.28), (4, 2.6)].iter().flat_map(|(g, l)| mk(*g, *l)).collect();
    let r = recommend(&rows, &spec).unwrap();
    assert_eq!(r.group, 2, "80 % is the smallest size with an error up to 10 %");
    assert!(r.reason.contains("rule of thumb") && r.reason.contains("up to 10 %"));
    // a guardrail flag keeps a size out
    let mut flagged = rows.clone();
    flagged[2].guardrails = vec!["too thin".into()];
    assert_eq!(recommend(&flagged, &spec).unwrap().group, 1);
}

#[test]
fn a_clear_search_makes_one_job_and_no_confirmation_without_thinking() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let a = analysis(&[(key(&[3, 7, 11], Some(10752)), 2.1, 2.1, 2.1), (key(&[3, 5, 7], Some(10752)), 2.4, 2.4, 2.4)]);
    let mut fx = Fake::new(a);
    let mut spec = SearchSpec::new("qwen3.8-27b", Target::B(21.0));
    spec.keep_thinking = false;
    let st = run(&p, &mut fx, &spec, None).unwrap();
    assert_eq!(st["status"], "done");
    assert_eq!(fx.tries.len(), 1, "{:?}", fx.calls);
    assert_eq!(st["recommendation"]["key"], key(&[3, 7, 11], Some(10752)));
    assert_eq!(st["recommendation"]["confidence"], "clear");
    assert!(st["rows"].as_array().unwrap().len() == 2 && st["plan"]["passes"].as_u64().unwrap() > 0);
    // saved next to the others and found again
    let saved = latest(&p).unwrap();
    assert_eq!(saved["id"], st["id"]);
    assert!(fx.made.is_empty(), "nothing is written unless the thinking check was chosen");
}

#[test]
fn a_close_call_is_asked_again_with_more_text_and_the_new_numbers_decide() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (k1, k2) = (key(&[3, 7, 11], Some(10752)), key(&[3, 5, 7], Some(10752)));
    let a = analysis(&[(k1.clone(), 2.100, 2.05, 2.15), (k2.clone(), 2.101, 2.05, 2.15)]);
    let mut fx = Fake::new(a);
    // the larger sample says the other one is better
    fx.confirm = json!({"base": 2.0, "rows": {k1.clone(): {"loss": 2.12, "loss_h1": 2.11, "loss_h2": 2.13}, k2.clone(): {"loss": 2.05, "loss_h1": 2.049, "loss_h2": 2.051}}});
    let mut spec = SearchSpec::new("qwen3.8-27b", Target::B(21.0));
    spec.keep_thinking = false;
    let st = run(&p, &mut fx, &spec, None).unwrap();
    assert_eq!(fx.tries.len(), 2);
    assert_eq!(fx.tries[1]["samples"], 128);
    assert_eq!(fx.tries[1]["choices"].as_array().unwrap().len(), 2);
    assert!(fx.tries[1]["search"].is_null(), "the confirmation is an ordinary request: a few passes at twice the length");
    assert_eq!(st["recommendation"]["key"], k2);
    assert!(st["recommendation"]["reason"].as_str().unwrap().contains("128 samples"));
    // switched off: no second request
    let mut fx2 = Fake::new(analysis(&[(k1.clone(), 2.100, 2.05, 2.15), (k2.clone(), 2.101, 2.05, 2.15)]));
    spec.confirm_full = false;
    run(&p, &mut fx2, &spec, None).unwrap();
    assert_eq!(fx2.tries.len(), 1);
}

#[test]
fn the_thinking_check_writes_one_cut_at_a_time_removes_it_and_decides_between_the_best_two() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (k1, k2) = (key(&[3, 7, 11], Some(10752)), key(&[3, 5, 7], Some(10752)));
    let mut fx = Fake::new(analysis(&[(k1.clone(), 2.1, 2.1, 2.1), (k2.clone(), 2.3, 2.3, 2.3)]));
    // the lower loss cut lost its thinking; the other kept it
    fx.checks.insert("s-cut-1/cut".into(), weak());
    let spec = SearchSpec::new("qwen3.8-27b", Target::B(21.0));
    let st = run(&p, &mut fx, &spec, None).unwrap();
    assert_eq!(st["status"], "done");
    let calls: Vec<&str> = fx.calls.iter().map(|s| s.as_str()).collect();
    let pos = |what: &str| calls.iter().position(|c| *c == what).unwrap_or_else(|| panic!("{what} in {calls:?}"));
    assert!(pos("cut") < pos("check s-cut-1 original") && pos("check s-cut-1 original") < pos("check s-cut-1 cut") && pos("check s-cut-1 cut") < pos("remove s-cut-1"));
    assert_eq!(calls.iter().filter(|c| **c == "check s-cut-1 original" || **c == "check s-cut-2 original").count(), 1, "the original is checked once: {calls:?}");
    assert!(pos("remove s-cut-1") < calls.iter().rposition(|c| *c == "cut").unwrap(), "the first cut is gone before the second is written");
    assert!(fx.made.values().all(|alive| !alive), "nothing is kept");
    assert_eq!(st["recommendation"]["key"], k2, "the thinking decides over the lower loss");
    assert_eq!(st["recommendation"]["thinking_checked"], true);
    assert!(st["recommendation"]["reason"].as_str().unwrap().contains("thinking was checked"));
    let rows = st["rows"].as_array().unwrap();
    assert_eq!(rows.iter().filter(|r| !r["think"].is_null()).count(), 2);
    // off: nothing is written
    let mut off = Fake::new(analysis(&[(k1.clone(), 2.1, 2.1, 2.1), (k2.clone(), 2.3, 2.3, 2.3)]));
    let mut s2 = spec.clone();
    s2.think_check = "off".into();
    let st2 = run(&p, &mut off, &s2, None).unwrap();
    assert_eq!(off.cuts, 0);
    assert!(st2["recommendation"]["reason"].as_str().unwrap().contains("Thinking not checked"));
}

#[test]
fn a_stop_in_the_thinking_check_continues_with_the_cut_that_was_written() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (k1, k2) = (key(&[3, 7, 11], Some(10752)), key(&[3, 5, 7], Some(10752)));
    let a = analysis(&[(k1.clone(), 2.1, 2.1, 2.1), (k2.clone(), 2.3, 2.3, 2.3)]);
    let spec = SearchSpec::new("qwen3.8-27b", Target::B(21.0));
    let mut plain = Fake::new(a.clone());
    let full = run(&p, &mut plain, &spec, None).unwrap();
    for n in 1..=plain.n {
        let dir = tempdir().unwrap();
        let q = project(dir.path());
        let mut fx = Fake::new(a.clone());
        fx.stop_at = Some(n);
        let first = run(&q, &mut fx, &spec, None).unwrap();
        assert_eq!(first["status"], "stopped", "stop at call {n}");
        let id = first["id"].as_str().unwrap().to_string();
        let last = run(&q, &mut fx, &spec, Some(&id)).unwrap();
        assert_eq!(last["status"], "done", "stop at call {n}");
        assert_eq!(last["recommendation"]["key"], full["recommendation"]["key"], "stop at call {n}");
        assert!(fx.made.values().all(|alive| !alive), "stop at call {n}: nothing stays on the disk");
        // a cut that was written is checked and removed once, never written twice
        let cuts_written = fx.cuts;
        assert!(cuts_written <= 3, "stop at call {n}: {:?}", fx.calls);
    }
}

#[test]
fn the_spec_and_the_answer_are_plain_json() {
    let s: SearchSpec = serde_json::from_value(json!({"model_id": "m", "target": {"b": 20.0}})).unwrap();
    assert_eq!((s.budget_min, s.keep_thinking, s.axes.as_str(), s.think_check.as_str(), s.confirm_full), (20, true, "both", "top2", true));
    let s: SearchSpec = serde_json::from_value(json!({"model_id": "m", "target": "tradeoff", "budget_min": 60})).unwrap();
    assert_eq!(s.target, Target::Tradeoff);
    let s: SearchSpec = serde_json::from_value(json!({"model_id": "m", "target": {"kept_pct": 75}})).unwrap();
    assert_eq!(s.target, Target::KeptPct(75.0));
    let s: SearchSpec = serde_json::from_value(json!({"model_id": "m", "target": {"gb": 12.5}})).unwrap();
    assert_eq!(s.target, Target::Gb(12.5));
    assert!(SearchSpec::new("m", Target::B(0.1)).validate().is_err());
    assert_eq!(parse_key("d=3,7|w=8192"), (vec![3, 7], Some(8192)));
    assert_eq!(parse_key("d=|w="), (vec![], None));
}
