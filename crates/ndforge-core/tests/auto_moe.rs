//! Auto-cut for a mixture of experts: the plan for Inkling-2-Mini from its real inspect (worker/tests/data/inkling2_inspect.json).
use ndforge_core::reshape::auto_moe::{reachable, stops, MoeAuto};
use ndforge_core::reshape::auto_plan::{plan, with_history, AutoSpec};
use ndforge_core::reshape::plan::Inspect;
use ndforge_core::timing::History;
use serde_json::{json, Value};

fn inkling() -> Inspect {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/inkling2_inspect.json")).unwrap()).unwrap()
}

fn spec(goal: &str, gb: Option<f64>) -> AutoSpec {
    let mut s = AutoSpec::new("inkling-2-mini", 20.0);
    s.moe = Some(MoeAuto { goal: goal.into(), target_gb: gb, top_k: None, scoring: "reap".into(), rebalance: true });
    s.think_gate = false;
    s
}

#[test]
fn a_target_in_gb_becomes_an_expert_count_by_the_owners_rules() {
    let ins = inkling();
    let m = ins.inkling().unwrap();
    assert_eq!(reachable(m, 1).iter().map(|x| x.0).collect::<Vec<_>>(), [112, 96, 80, 64]);
    assert_eq!(reachable(m, 2).iter().map(|x| x.0).collect::<Vec<_>>(), [112, 96, 80, 64, 48, 32], "two stages reach a quarter of the experts, never fewer than 32");
    for (gb, experts, stages) in [(13.0, 96, 1), (12.0, 96, 1), (11.0, 80, 1), (9.0, 64, 2), (6.0, 32, 2)] {
        let p = plan(&ins, &spec("size", Some(gb)), Some(32.0), Some(500.0));
        assert!(p.ok, "{gb} GB: {:?}", p.blockers);
        let c = p.chosen.as_ref().unwrap().moe.as_ref().unwrap();
        assert_eq!((c.experts, p.stages.len() as u32), (experts, stages), "{gb} GB");
        assert!(c.q4_gb <= gb * 1.04, "{gb} GB -> {c:?}");
        // no stage removes more than half; the last stage ends at the chosen count
        for s in &p.stages {
            let mo = s.moe.as_ref().unwrap();
            assert!(mo.experts_after as f64 >= mo.experts_before as f64 * 0.5 && mo.experts_after >= 32 && mo.experts_after % 16 == 0, "{mo:?}");
        }
        assert_eq!(p.stages.last().unwrap().moe.as_ref().unwrap().experts_after, experts);
    }
    assert_eq!(stops(m, 64, 2), vec![96, 64]);
    assert_eq!(stops(m, 96, 2), vec![96], "a cut of a quarter is done in one stage");
    assert_eq!(stops(m, 32, 2), vec![64, 32]);
    assert_eq!(stops(m, 64, 1), vec![64]);
}

#[test]
fn refusals_say_why() {
    let ins = inkling();
    let blocked = |s: AutoSpec| plan(&ins, &s, Some(32.0), Some(500.0)).blockers.join(" ");
    assert!(blocked(spec("size", Some(16.0))).contains("at least 3 % smaller"));
    assert!(blocked(spec("size", Some(3.0))).contains("smallest is 32 experts") || blocked(spec("size", Some(3.0))).contains("smaller than the cut that is allowed"));
    assert!(blocked(spec("quality", Some(10.0))).contains("training"));
    assert!(blocked(spec("size", None)).contains("how big"));
    let small_card = plan(&ink_for_card(), &spec("size", Some(12.0)), Some(16.0), Some(500.0));
    assert!(!small_card.ok && small_card.blockers.iter().any(|b| b.contains("needs about")), "{:?}", small_card.blockers);
    let low_disk = plan(&ins, &spec("size", Some(12.0)), Some(32.0), Some(30.0));
    assert!(!low_disk.ok && low_disk.blockers.iter().any(|b| b.contains("of disk")), "{:?}", low_disk.blockers);
}

fn ink_for_card() -> Inspect {
    inkling()
}

#[test]
fn auto_cut_never_takes_loop_passes_out_and_says_what_to_do_instead() {
    let ins = inkling();
    for goal in ["size", "speed"] {
        let p = plan(&ins, &spec(goal, Some(12.0)), Some(32.0), Some(500.0));
        assert!(p.ok, "{:?}", p.blockers);
        assert_eq!(p.chosen.as_ref().unwrap().moe.as_ref().unwrap().passes, 3);
        let mi = p.moe.as_ref().unwrap();
        assert_eq!(mi.passes, 3);
        assert!(mi.passes_note.contains("thinkingmachines/TML-d24-varloop-research") && p.warnings.iter().any(|w| w.contains("TML-d24-varloop-research")));
        for s in &p.stages {
            let r = serde_json::to_value(s.reshape_spec("inkling-2-mini", None, 64, 64)).unwrap();
            assert!(r.get("passes").is_none(), "{r}");
        }
    }
}

#[test]
fn the_speed_goal_sets_the_experts_per_token_and_the_stage_spec_carries_it() {
    let ins = inkling();
    let p = plan(&ins, &spec("speed", None), Some(32.0), Some(500.0));
    assert!(p.ok, "{:?}", p.blockers);
    let c = p.chosen.as_ref().unwrap().moe.as_ref().unwrap();
    assert_eq!((c.experts, c.top_k, p.stages.len()), (128, 6, 1));
    let r = serde_json::to_value(p.stages[0].reshape_spec("inkling-2-mini", None, 64, 64)).unwrap();
    assert_eq!((r["top_k"].clone(), r.get("experts").cloned()), (json!(6), None), "no experts are removed, so none are named: {r}");
    assert_eq!(p.stages[0].tokens_m, 40, "fewer experts per token needs the larger heal");
    assert!(p.warnings.iter().any(|w| w.contains("did not test")));
    // fewer experts AND fewer per token
    let mut s = spec("size", Some(12.0));
    s.moe.as_mut().unwrap().top_k = Some(6);
    let p = plan(&ins, &s, Some(32.0), Some(500.0));
    let r = serde_json::to_value(p.stages.last().unwrap().reshape_spec("inkling-2-mini", None, 64, 64)).unwrap();
    assert_eq!((r["experts"]["keep"].clone(), r["top_k"].clone(), r["experts"]["scoring"].clone()), (json!(96), json!(6), json!("reap")));
}

#[test]
fn heal_budget_follows_how_much_is_removed_and_the_person_can_override_it() {
    let ins = inkling();
    let toks = |gb| plan(&ins, &spec("size", Some(gb)), Some(32.0), Some(500.0)).stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>();
    assert_eq!(toks(12.0), [20]);           // a quarter removed
    assert_eq!(toks(11.0), [40]);           // 37 %
    assert_eq!(toks(9.0), [20, 40]);        // 128 -> 96 -> 64
    let mut s = spec("size", Some(9.0));
    s.stage_a_tokens_m = Some(10);
    s.last_tokens_m = Some(80);
    assert_eq!(plan(&ins, &s, Some(32.0), Some(500.0)).stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>(), [10, 80]);
}

#[test]
fn the_memory_the_disk_and_the_time_are_estimated() {
    let ins = inkling();
    let p = plan(&ins, &spec("size", Some(12.0)), Some(32.0), Some(500.0));
    let s = &p.stages[0];
    assert!((22.0..=28.0).contains(&s.heal_gb) && s.verdict != "over", "{} GB", s.heal_gb);
    assert_eq!((s.mode.as_str(), s.seq, s.rank, s.teacher_load.as_str()), ("cache", 4096, 64, "packed"));
    let d = p.disk.as_ref().unwrap();
    assert!(d.peak_gb > 40.0 && d.peak_gb < 120.0 && d.enough == Some(true), "{d:?}");
    let t = p.time.as_ref().unwrap();
    // the ids are those of the dense run, so the page's list, the progress bar and Continue work unchanged
    let ids: Vec<&str> = t.steps.iter().map(|x| x.id.as_str()).collect();
    assert_eq!(ids, ["plan", "measure", "try", "s1.cut", "s1.check_cut", "s1.heal", "s1.check_heal", "export", "check_gguf"], "{ids:?}");
    assert!(t.total_h.0 > 3.0 && t.total_h.1 < 100.0 && t.total_h.0 < t.total_h.1, "{:?}", t.total_h);
    assert!(t.basis.contains("formula"));
    // learned from this computer: a faster measuring rate shortens the step and is marked as history
    let mut h = History::default();
    h.push("moe_score_tok_s", 1.0, 5000.0);
    let fast = with_history(p.clone(), &ins, &spec("size", Some(12.0)), &h);
    let m1 = |t: &ndforge_core::reshape::auto_time::TimeEstimate| t.steps.iter().find(|x| x.id == "measure").unwrap().clone();
    assert!(m1(fast.time.as_ref().unwrap()).minutes.1 < m1(t).minutes.1);
    assert_eq!(m1(fast.time.as_ref().unwrap()).parts.iter().find(|x| x.id == "measure").unwrap().basis, "your_history");
}

#[test]
fn a_dense_plan_and_spec_are_byte_for_byte_what_they_were() {
    let s = AutoSpec::new("qwen3.8-27b", 21.1);
    let v: Value = serde_json::to_value(&s).unwrap();
    assert!(v.get("moe").is_none());
    let back: AutoSpec = serde_json::from_value(v).unwrap();
    assert_eq!(back, s);
    // an old saved spec (no key) reads
    let old = json!({"model_id": "m", "target_b": 21.1});
    assert!(serde_json::from_value::<AutoSpec>(old).unwrap().moe.is_none());
}
