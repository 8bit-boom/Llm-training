//! The Auto-cut time estimate (`reshape::auto_time`): the whole run, step by step, before it starts.
use ndforge_core::reshape::auto_plan::*;
use ndforge_core::reshape::plan::*;
use serde_json::json;

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("worker").join("tests").join("data")
}

fn dims_of(tc: &serde_json::Value) -> WidthDims {
    let u = |k: &str| tc[k].as_u64().unwrap_or(0) as u32;
    WidthDims {
        hidden: u("hidden_size"), intermediate: u("intermediate_size"), heads: u("num_attention_heads"), kv_heads: u("num_key_value_heads"), head_dim: u("head_dim"),
        lin_k_heads: u("linear_num_key_heads"), lin_v_heads: u("linear_num_value_heads"), lin_k_dim: u("linear_key_head_dim"), lin_v_dim: u("linear_value_head_dim"),
        conv_kernel: u("linear_conv_kernel_dim"), mtp_layers: u("mtp_num_hidden_layers"), vocab: u("vocab_size"), tied: tc["tie_word_embeddings"].as_bool().unwrap_or(false),
    }
}

/// The real 27B: 64 layers in 16 blocks, MLP 17,408 (the config is the worker's test data)
fn qwen27() -> Inspect {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data_dir().join("qwen3_5_27b_config.json")).unwrap()).unwrap();
    let tc = &cfg["text_config"];
    let d = dims_of(tc);
    let types: Vec<String> = tc["layer_types"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    let layer_params = vec![390_000_000u64; 64];
    Inspect {
        model_type: "qwen3_5".into(), supported: true, layers: 64, layer_types: types, period: 4, blocks: 16, vision_params: 460_730_096, other_params: d.hidden as u64,
        total_params: 27_781_427_952, bytes_per_param: 2.0, has_vision: true, local: true, width_supported: true, layer_params,
        width_choices: Some(WidthChoices { intermediate: (34..=68).map(|k| k * 256).collect(), lin_v_heads: vec![48, 32], heads: vec![24, 20, 16] }),
        width_dims: Some(d), ..Default::default()
    }
}

use ndforge_core::timing::History;

/// 27 B -> 17.4 B needs more than four blocks: the owner confirms that.
fn owner_spec() -> AutoSpec {
    AutoSpec { allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 17.4) }
}

fn est(s: &AutoSpec, h: &History) -> (AutoPlan, TimeEstimate) {
    let ins = qwen27();
    let p = plan(&ins, s, Some(32.0), Some(2000.0));
    assert!(p.ok, "{:?}", p.blockers);
    let t = time_estimate(&ins, s, &p, h);
    (p, t)
}

fn find<'a>(t: &'a TimeEstimate, id: &str) -> Option<&'a StepTime> {
    t.steps.iter().find(|s| s.id == id)
}

fn heal_total(t: &TimeEstimate) -> f64 {
    t.steps.iter().filter(|s| s.id.ends_with(".heal")).map(|s| s.minutes.1).sum()
}

fn print(t: &TimeEstimate) {
    println!("{:<14} {:<58} {:>9} {:>9}  basis", "id", "step", "low min", "high min");
    for s in &t.steps {
        println!("{:<14} {:<58} {:>9.1} {:>9.1}  {}", s.id, s.label, s.minutes.0, s.minutes.1, s.basis);
    }
    println!("TOTAL {:.1} h to {:.1} h; {}", t.total_h.0, t.total_h.1, t.basis);
    for e in &t.extra {
        println!("EXTRA (not in the total) {:.1} to {:.1} min: {}", e.minutes.0, e.minutes.1, e.label);
    }
}

#[test]
fn the_total_is_the_sum_of_the_steps_and_every_step_has_a_range() {
    let (p, t) = est(&owner_spec(), &History::default());
    assert_eq!(p.time.as_ref().map(|x| x.total_h), Some(t.total_h), "the plan carries the same estimate (without history)");
    let (lo, hi) = t.steps.iter().fold((0.0, 0.0), |a, s| (a.0 + s.minutes.0, a.1 + s.minutes.1));
    assert!((t.total_h.0 - lo / 60.0).abs() < 0.01 && (t.total_h.1 - hi / 60.0).abs() < 0.01, "{:?} {lo} {hi}", t.total_h);
    for s in &t.steps {
        assert!(s.minutes.0 > 0.0 && s.minutes.0 <= s.minutes.1, "{s:?}");
        let (pl, ph) = s.parts.iter().fold((0.0, 0.0), |a, x| (a.0 + x.minutes.0, a.1 + x.minutes.1));
        assert!((pl - s.minutes.0).abs() < 0.6 && (ph - s.minutes.1).abs() < 0.6, "parts add up: {s:?}");
    }
    // the ids are those of the run's steps, in its order
    let ids: Vec<&str> = t.steps.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["plan", "measure", "try", "think_gate", "s1.cut", "s1.check_cut", "s1.heal", "s1.check_heal", "s2.cut", "s2.check_cut", "s2.heal", "s2.check_heal", "export", "check_gguf"]);
    assert_eq!(t.step_count, 14);
    assert_eq!(t.history_steps, 0);
    assert!(t.basis.contains("None of the 14 steps"), "{}", t.basis);
    assert!(t.steps.iter().all(|s| s.basis == "formula"));
}

#[test]
fn disabled_steps_vanish() {
    let s = AutoSpec { keep_thinking: false, think_gate: false, ..owner_spec() };
    let (_, t) = est(&s, &History::default());
    let ids: Vec<&str> = t.steps.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["plan", "measure", "try", "s1.cut", "s1.heal", "s2.cut", "s2.heal", "export"]);
    let s = AutoSpec { think_gate: false, ..owner_spec() };
    assert!(find(&est(&s, &History::default()).1, "think_gate").is_none());
    assert!(find(&est(&s, &History::default()).1, "s1.check_cut").unwrap().parts.iter().any(|x| x.label.contains("original")), "without the gate the first check measures the original");
    let s = AutoSpec { studio_publish: true, target_gb: Some(10.6), ..owner_spec() };
    let (_, t) = est(&s, &History::default());
    for id in ["studio_publish", "compare_quants"] {
        assert!(find(&t, id).is_some(), "{id}");
    }
    assert!(find(&t, "export").unwrap().parts.iter().any(|x| x.label.contains("measure the fit")), "a target size adds the measuring to the export");
}

#[test]
fn two_stages_double_the_heal_but_not_the_measuring() {
    let two = est(&owner_spec(), &History::default()).1;
    let one = est(&AutoSpec { stages: 1, ..owner_spec() }, &History::default()).1;
    for id in ["measure", "try", "think_gate", "export", "check_gguf"] {
        assert_eq!(find(&one, id).unwrap().minutes.0, find(&two, id).unwrap().minutes.0, "{id} is made once");
    }
    let (h1, h2) = (heal_total(&one), heal_total(&two));
    assert!(h2 > h1 * 1.3 && h2 < h1 * 2.2, "40 M + 80 M tokens against 80 M: {h1} {h2}");
    assert!(two.total_h.1 > one.total_h.1);
}

#[test]
fn quick_is_shorter_than_full_and_a_target_size_adds_steps() {
    let full = est(&owner_spec(), &History::default()).1;
    let quick = est(&AutoSpec { quick: true, ..owner_spec() }, &History::default()).1;
    assert!(quick.total_h.1 < full.total_h.1 && quick.total_h.0 < full.total_h.0, "{:?} {:?}", quick.total_h, full.total_h);
    assert!(find(&quick, "try").unwrap().minutes.1 < find(&full, "try").unwrap().minutes.1);
    let target = est(&AutoSpec { target_gb: Some(10.6), ..owner_spec() }, &History::default()).1;
    assert!(find(&target, "compare_quants").is_some() && find(&full, "compare_quants").is_none());
    assert!(find(&target, "export").unwrap().minutes.1 > find(&full, "export").unwrap().minutes.1, "the measuring and one more size");
    assert!(target.total_h.1 > full.total_h.1);
}

#[test]
fn the_heal_scales_linearly_with_the_tokens() {
    let ins = qwen27();
    let s = AutoSpec { keep_thinking: false, think_gate: false, stages: 1, ..AutoSpec::new("m", 21.1) };
    let p = plan(&ins, &s, Some(32.0), Some(2000.0));
    assert!(p.ok, "{:?}", p.blockers);
    let g = &p.stages[0];
    let part = |tokens_m: u32| {
        let mut p2 = p.clone();
        p2.stages[0].tokens_m = tokens_m;
        let t = time_estimate(&ins, &s, &p2, &History::default());
        let h = find(&t, "s1.heal").unwrap().parts.iter().find(|x| x.id == "train").unwrap().minutes;
        let c = find(&t, "s1.heal").unwrap().parts.iter().find(|x| x.id == "cache").map(|x| x.minutes);
        (h, c)
    };
    assert_eq!(g.mode, "cache", "the 27 B and the student do not fit a 32 GB card together");
    let (a, b) = (part(20), part(40));
    assert!((b.0 .0 / a.0 .0 - 2.0).abs() < 0.02 && (b.0 .1 / a.0 .1 - 2.0).abs() < 0.02, "training {a:?} {b:?}");
    let (ca, cb) = (a.1.unwrap(), b.1.unwrap());
    assert!((cb.0 / ca.0 - 2.0).abs() < 0.02 && (cb.1 / ca.1 - 2.0).abs() < 0.02, "the teacher's answers {ca:?} {cb:?}");
}

fn history_of_the_machine() -> History {
    let mut h = History::default();
    // the teacher (27.8 B) answers 1,200 tokens per second; the 21 B student trains at 700; the merge takes 20 s per GB; the thinking check 40 s per billion parameters
    h.push("teacher_tok_b_s", 27.8, 1200.0 * 27.8);
    h.push("train_tok_b_s", 21.0, 700.0 * 21.0);
    h.push("merge_s_per_gb", 40.0, 20.0);
    h.push("think_s_per_b:full", 21.0, 40.0);
    h
}

#[test]
fn your_history_overrides_the_formula_and_is_tagged() {
    let none = est(&owner_spec(), &History::default()).1;
    let own = est(&owner_spec(), &history_of_the_machine()).1;
    let heal = find(&own, "s2.heal").unwrap();
    assert_eq!(heal.basis, "your_history", "{heal:?}");
    assert!(heal.parts.iter().filter(|x| ["cache", "train", "merge"].contains(&x.id.as_str())).all(|x| x.basis == "your_history"));
    assert_ne!(heal.minutes, find(&none, "s2.heal").unwrap().minutes);
    // stage 1: 40 M tokens at 1,200 / s for the teacher: 11.1 h x 0.85 .. 1.25; stage 2 (80 M, the same sequence length) writes only the 40 M that stage 1 did not
    let point = 40e6 / 1200.0 / 60.0;
    for id in ["s1.heal", "s2.heal"] {
        let cache = find(&own, id).unwrap().parts.iter().find(|x| x.id == "cache").unwrap();
        assert!((cache.minutes.0 / (point * 0.85) - 1.0).abs() < 0.01 && (cache.minutes.1 / (point * 1.25) - 1.0).abs() < 0.01, "{id}: {cache:?} {point}");
    }
    assert!(own.history_steps >= 4 && own.history_steps < own.step_count, "{} of {}", own.history_steps, own.step_count);
    assert!(own.basis.starts_with(&format!("{} of {} steps use your own earlier runs", own.history_steps, own.step_count)), "{}", own.basis);
    // what was not measured stays a formula
    assert_eq!(find(&own, "measure").unwrap().basis, "formula");
    assert_eq!(find(&own, "export").unwrap().basis, "formula");
    // a history of another size is scaled: the same rates measured on a 10 B student give about the same total for the 21 B one only when the product tok x B is the same
    let mut other = History::default();
    other.push("train_tok_b_s", 10.0, 1470.0 * 10.0);
    let o = est(&owner_spec(), &other).1;
    let train = |t: &TimeEstimate| find(t, "s2.heal").unwrap().parts.iter().find(|x| x.id == "train").unwrap().minutes;
    let (a, b) = (train(&o), train(&est(&owner_spec(), &{ let mut h = History::default(); h.push("train_tok_b_s", 21.0, 700.0 * 21.0); h }).1));
    assert!((a.0 - b.0).abs() / b.0 < 0.02, "the rate in tokens x billions per second scales with the size: {a:?} {b:?}");
}

#[test]
fn an_old_saved_spec_and_plan_without_the_new_fields_still_work() {
    let spec: AutoSpec = serde_json::from_value(json!({"model_id": "m", "target_b": 21.1})).unwrap();
    let ins = qwen27();
    let p = plan(&ins, &spec, Some(32.0), Some(2000.0));
    assert!(p.time.is_some());
    // a plan saved before the estimate existed has no `time`
    let mut v = serde_json::to_value(&p).unwrap();
    v.as_object_mut().unwrap().remove("time");
    let old: AutoPlan = serde_json::from_value(v).unwrap();
    assert!(old.time.is_none());
    // and a plan that is refused has none
    let bad = plan(&ins, &AutoSpec::new("m", 900.0), Some(32.0), None);
    assert!(!bad.ok && bad.time.is_none());
}

#[test]
fn a_failed_check_has_its_own_line_that_is_not_in_the_total() {
    let (_, t) = est(&owner_spec(), &History::default());
    assert_eq!(t.extra.len(), 1);
    let sum: f64 = t.steps.iter().map(|s| s.minutes.1).sum();
    assert!((t.total_h.1 - sum / 60.0).abs() < 0.01, "not added");
    let e = &t.extra[0];
    assert!(e.id == "retry_heal" && e.minutes.0 > 60.0, "{e:?}");
    let (_, n) = est(&AutoSpec { keep_thinking: false, think_gate: false, ..owner_spec() }, &History::default());
    assert!(n.extra.is_empty(), "no thinking checks, no retry");
}

#[test]
fn the_owners_run_has_a_plausible_total() {
    let (p, t) = est(&owner_spec(), &History::default());
    print(&t);
    println!("heal hours of the plan (older figure): {:?}; lever {:?}", p.heal_hours, t.lever_h);
    assert_eq!(p.stages.len(), 2);
    assert_eq!((p.stages[0].tokens_m, p.stages[1].tokens_m), (40, 80));
    // the two heals dominate; the reference numbers (1.0 to 2.5 h and 1.4 to 2.2 h per 5 M tokens) make 120 M tokens a matter of days
    assert!(t.total_h.0 > 15.0 && t.total_h.1 < 160.0 && t.total_h.0 < t.total_h.1, "{:?}", t.total_h);
    assert!(t.lever_h.1 > 0.7 * t.total_h.1, "the teacher's answers and the training are most of it: {:?} of {:?}", t.lever_h, t.total_h);
}

// ----------------------------------------------------------------------------------------------------------------- healing in rounds

use ndforge_core::reshape::auto_time::{rounds_estimate, RoundsIn};

fn rounds_spec(round_m: u32) -> AutoSpec {
    AutoSpec { heal_rounds: Some(HealRoundsOpt { round_m, check: None, first_check_m: None }), ..owner_spec() }
}

#[test]
fn rounds_have_a_best_case_below_the_normal_heal_and_a_worst_case_above_it() {
    let normal = est(&owner_spec(), &History::default()).1;
    let (_, t) = est(&rounds_spec(5), &History::default());
    for id in ["s1.heal", "s2.heal"] {
        let (n, r) = (find(&normal, id).unwrap(), find(&t, id).unwrap());
        let rt = r.rounds.as_ref().expect("a heal in rounds says what its two ends are");
        assert!(rt.best.1 < n.minutes.1 && n.minutes.1 < rt.worst.1, "{id}: best {:?} normal {:?} worst {:?}", rt.best, n.minutes, rt.worst);
        assert_eq!(r.minutes, (rt.best.0, rt.worst.1), "the step is (best case low, worst case high)");
        assert!(rt.best.0 <= rt.best.1 && rt.worst.0 <= rt.worst.1 && rt.per_round.0 <= rt.per_round.1);
        assert_eq!((rt.round_m, rt.first_check_m), (5, 10));
        assert_eq!(rt.rounds.0, 2);
        assert_eq!(rt.rounds.1 * 5, rt.cap_m.div_ceil(5) * 5);
        assert!(r.note.as_ref().unwrap().contains("Experimental"));
    }
    // stage 1 heals 40 M, the last stage 80 M (the owner's recipe)
    assert_eq!(find(&t, "s1.heal").unwrap().rounds.as_ref().unwrap().rounds, (2, 8));
    assert_eq!(find(&t, "s2.heal").unwrap().rounds.as_ref().unwrap().rounds, (2, 16));
    assert!(t.total_h.1 > normal.total_h.1 && t.total_h.0 < normal.total_h.0, "the total spans both: {:?} {:?}", t.total_h, normal.total_h);
}

#[test]
fn the_check_after_the_heal_only_reads_what_the_heal_saved() {
    let (_, t) = est(&rounds_spec(5), &History::default());
    for id in ["s1.check_heal", "s2.check_heal"] {
        let s = find(&t, id).unwrap();
        assert_eq!(s.parts.len(), 1);
        assert!(s.minutes.1 <= 1.0 && s.note.as_ref().unwrap().contains("inside the heal"), "{s:?}");
    }
    // the merged model's check (in the quick or full mode of the run) is in the heal's own parts
    let heal = find(&t, "s2.heal").unwrap();
    assert!(heal.parts.iter().any(|x| x.id == "check") && heal.parts.iter().any(|x| x.id == "merge"));
    // a retry in rounds is "more rounds", not a second heal
    let extra = t.extra.iter().find(|e| e.id == "retry_heal").unwrap();
    assert!(extra.label.contains("40 M more tokens in rounds") && extra.minutes.1 > 0.0);
}

#[test]
fn the_overhead_share_falls_with_the_round_size_and_a_round_is_the_same_work() {
    let rin = |round_m: u32| RoundsIn {
        teacher_b: 27.8, student_b: 17.4, teacher_gb: 55.6, student_gb: 34.8, teacher_load: "4bit".into(), reuse_m: 0, cap_m: 40, round_m, first_check_m: 10.max(round_m), check_quick: true, final_quick: true, thinking: true,
    };
    let h = History::default();
    let shares: Vec<f64> = [1, 2, 5, 10].iter().map(|r| rounds_estimate(&h, &rin(*r)).1.overhead_share.1).collect();
    assert!(shares.windows(2).all(|w| w[0] > w[1]), "{shares:?}");
    assert!(shares[0] > 0.2 && shares[2] < 0.25, "{shares:?}");
    // the work (the original's answers and the training) is the same whatever the size of a round; only the overhead differs
    let work = |r: u32| -> f64 { let (parts, _) = rounds_estimate(&h, &rin(r)); parts.iter().filter(|x| x.id == "cache" || x.id == "train").map(|x| x.minutes.1).sum() };
    assert!((work(1) - work(10)).abs() < 1.0, "{} {}", work(1), work(10));
    // earlier answers that are saved make the original's part smaller
    let reuse = RoundsIn { reuse_m: 20, ..rin(5) };
    assert!(rounds_estimate(&h, &reuse).1.worst.1 < rounds_estimate(&h, &rin(5)).1.worst.1);
    // without thinking there are no checks and the first stop is the plateau, after half the cap
    let plain = RoundsIn { thinking: false, ..rin(5) };
    let (parts, t) = rounds_estimate(&h, &plain);
    assert!(!parts.iter().any(|x| x.id == "check") && t.rounds == (4, 8));
}

#[test]
fn without_rounds_nothing_changes_in_the_estimate_or_the_spec() {
    let (_, t) = est(&owner_spec(), &History::default());
    let text = serde_json::to_string(&t).unwrap();
    assert!(!text.contains("rounds"), "the legacy estimate has no rounds key at all");
    assert!(t.steps.iter().all(|s| s.rounds.is_none()));
    let spec = serde_json::to_string(&owner_spec()).unwrap();
    assert!(!spec.contains("heal_rounds"));
    let back: AutoSpec = serde_json::from_str(&spec).unwrap();
    assert!(back.heal_rounds.is_none());
    // a spec with rounds writes and reads its setting
    let with = serde_json::to_string(&rounds_spec(5)).unwrap();
    assert!(with.contains("heal_rounds"));
    assert_eq!(serde_json::from_str::<AutoSpec>(&with).unwrap().heal_rounds, rounds_spec(5).heal_rounds);
    assert!(rounds_spec(3).validate().is_err() && rounds_spec(5).validate().is_ok());
}
