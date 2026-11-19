//! Healing in rounds, the pure layer (`reshape::heal_rounds`): the settings, the rounds and checks, the size rule and the stop decisions.
use ndforge_core::reshape::heal_rounds::*;
use ndforge_core::reshape::runs::{heal_choice, heal_train_key, HealSpec};
use ndforge_core::reshape::plan::HealDims;
use serde_json::{json, Value};

fn heal(v: Value) -> HealSpec {
    serde_json::from_value(v).unwrap()
}

fn with_rounds(tokens_m: u32, r: Value) -> HealSpec {
    heal(json!({"tokens_m": tokens_m, "rounds": r}))
}

// ----------------------------------------------------------------------------------------------------------------- the spec

#[test]
fn an_old_heal_spec_has_no_rounds_and_is_written_without_them() {
    let old = json!({"tokens_m": 40, "general_share": 0.5, "lr": 0.0001, "seq": 2048, "recipe": "thinking"});
    let spec = heal(old);
    assert!(spec.rounds.is_none());
    let back = serde_json::to_value(&spec).unwrap();
    assert!(back.get("rounds").is_none());
    // and it still reads back the same
    assert_eq!(serde_json::from_value::<HealSpec>(back).unwrap(), spec);
    spec.validate().unwrap();
}

#[test]
fn the_rounds_defaults_are_five_million_quick_stop_at_the_first_pass() {
    let spec = with_rounds(40, json!({}));
    let r = spec.rounds.clone().unwrap();
    assert_eq!((r.round_m, r.check.as_str(), r.stop.as_str(), r.plateau, r.extend), (5, "quick", "pass", true, false));
    assert_eq!((r.cap_tokens(&spec), r.round_tokens(), r.first_check_tokens(&spec)), (40_000_000, 5_000_000, 10_000_000));
    assert!(r.quick() && r.final_is_quick() && !r.never());
    spec.validate().unwrap();
    // a spec with rounds writes its keys and reads them back
    let again: HealSpec = serde_json::from_value(serde_json::to_value(&spec).unwrap()).unwrap();
    assert_eq!(again, spec);
}

#[test]
fn the_first_check_never_comes_before_a_round_or_after_the_cap() {
    let spec = with_rounds(40, json!({"round_m": 20, "first_check_m": 20}));
    assert_eq!(spec.rounds.as_ref().unwrap().first_check_tokens(&spec), 20_000_000);
    let spec = with_rounds(10, json!({"round_m": 5, "max_tokens_m": 5}));
    assert_eq!(spec.rounds.as_ref().unwrap().first_check_tokens(&spec), 5_000_000);
    let spec = with_rounds(40, json!({"round_m": 20}));
    assert_eq!(spec.rounds.as_ref().unwrap().first_check_tokens(&spec), 20_000_000);          // 10 M is less than a round
}

#[test]
fn validation_names_what_is_wrong() {
    let bad = |r: Value| with_rounds(40, r).validate().unwrap_err().to_string();
    assert!(bad(json!({"round_m": 3})).contains("1, 2, 5, 10 or 20"));
    assert!(bad(json!({"max_tokens_m": 0})).contains("1 to 160"));
    assert!(bad(json!({"max_tokens_m": 161})).contains("1 to 160"));
    assert!(bad(json!({"round_m": 20, "max_tokens_m": 10})).contains("at least one round"));
    assert!(bad(json!({"first_check_m": 3})).contains("first check"));
    assert!(bad(json!({"first_check_m": 50})).contains("first check"));
    assert!(bad(json!({"check": "slow"})).contains("quick or full"));
    assert!(bad(json!({"stop": "sometimes"})).contains("never"));
    assert!(bad(json!({"baseline": 3})).contains("original"));
    assert!(bad(json!({"round_exact": 5})).contains("exact round size"));
    for ok in [json!({"round_m": 1}), json!({"round_m": 20, "max_tokens_m": 160}), json!({"first_check_m": 5, "check": "full", "stop": "never"}), json!({"round_exact": 10_000})] {
        with_rounds(40, ok).validate().unwrap();
    }
    // rounds need the saved answers of the original
    assert!(heal(json!({"tokens_m": 40, "mode": "online", "rounds": {}})).validate().unwrap_err().to_string().contains("not the online way"));
    heal(json!({"tokens_m": 40, "mode": "cache", "rounds": {}})).validate().unwrap();
    // the cap is separate from the heal's token choices (10 / 20 / 40 / 80)
    assert!(heal(json!({"tokens_m": 33, "rounds": {}})).validate().is_err());
    with_rounds(40, json!({"max_tokens_m": 33})).validate().unwrap();
}

#[test]
fn with_exact_tokens_the_cap_is_the_exact_number() {
    let spec = heal(json!({"tokens_m": 10, "tokens_exact": 40_000, "rounds": {"round_exact": 10_000}}));
    spec.validate().unwrap();
    let r = spec.rounds.clone().unwrap();
    assert_eq!((r.cap_tokens(&spec), r.round_tokens(), r.first_check_tokens(&spec)), (40_000, 10_000, 10_000));
}

// ----------------------------------------------------------------------------------------------------------------- rounds and checks

#[test]
fn the_points_split_the_cap_and_check_after_the_first_check_but_not_at_the_cap() {
    let m = 1_000_000u64;
    let pts = points(40 * m, 5 * m, 10 * m);
    assert_eq!(pts.len(), 8);
    assert_eq!(pts.iter().map(|p| p.check).collect::<Vec<_>>(), vec![false, true, true, true, true, true, true, false]);
    assert_eq!((pts[0].start, pts[0].end, pts[7].start, pts[7].end, pts[7].n), (0, 5 * m, 35 * m, 40 * m, 8));
    // the last round is what is left
    let pts = points(12 * m, 5 * m, 5 * m);
    assert_eq!(pts.iter().map(|p| (p.end, p.check)).collect::<Vec<_>>(), vec![(5 * m, true), (10 * m, true), (12 * m, false)]);
    // one round only: no check at all (the merged model's check is the gate)
    assert_eq!(points(5 * m, 5 * m, 5 * m).iter().map(|p| p.check).collect::<Vec<_>>(), vec![false]);
    assert!(points(0, 5, 5).is_empty() && points(5, 0, 5).is_empty());
    // a cap that is a multiple of the round gives exactly cap / round rounds
    assert_eq!(points(160 * m, m, m).len(), 160);
}

#[test]
fn a_round_with_fewer_than_eight_steps_is_refused() {
    // seq 8192, 32 sequences per update: 131 k tokens a step -> 1 M is 7.6 steps
    assert!((est_steps(8192, 32, 1_000_000) - 7.63).abs() < 0.01);
    let e = check_round_size(8192, 32, 1_000_000).unwrap_err().to_string();
    assert!(e.contains("about 7 training steps") && e.contains("at least 8"), "{e}");
    check_round_size(8192, 32, 2_000_000).unwrap();
    check_round_size(2048, 32, 1_000_000).unwrap();
    assert_eq!(MIN_ROUND_STEPS, 8);
}

#[test]
fn the_overhead_formulas_match_the_plan() {
    // O = 28 min, c = 45 min per million tokens
    let share = |r: f64| overhead_share(28.0, 45.0, r);
    assert!(share(1.0) > 0.35 && share(1.0) < 0.40);
    assert!(share(5.0) > 0.10 && share(5.0) < 0.14);
    assert!(share(1.0) > share(2.0) && share(2.0) > share(5.0) && share(5.0) > share(10.0));
    assert_eq!(overhead_share(0.0, 0.0, 5.0), 0.0);
    // the best round size: about 3.7 / 5.2 / 7.3 M for the passes after 10 / 20 / 40 M
    let best = |n: f64| best_round_m(n, 28.0, 45.0);
    assert!(best(10.0) > 3.4 && best(10.0) < 3.9 && best(20.0) > 4.9 && best(20.0) < 5.4 && best(40.0) > 6.9 && best(40.0) < 7.6, "{} {} {}", best(10.0), best(20.0), best(40.0));
    assert_eq!(best_round_m(10.0, 28.0, 0.0), 0.0);
}

// ----------------------------------------------------------------------------------------------------------------- decisions

const M: u64 = 1_000_000;

fn main(closed: f64, loops: f64) -> Value {
    json!({"closed_rate": closed, "loop_rate": loops, "garbage_rate": 0.0, "accuracy": 0.8})
}

fn row(n: u32, tokens_m: u64, loss: f64, check: Option<(&str, f64)>) -> RoundRow {
    RoundRow {
        n, tokens: tokens_m * M, calib_loss: Some(loss),
        check: check.map(|(verdict, closed)| RoundCheck { mode: "quick".into(), verdict: verdict.into(), failed: vec![], main: main(closed, 0.0), score: None }),
    }
}

fn ctx() -> Ctx {
    Ctx { cap: 40 * M, thinking: true, stop_never: false, plateau: true, strict: false, now: 1_000, deadline: None, next_round_secs: 3_600.0, tail_secs: 3_600.0 }
}

#[test]
fn the_first_passed_check_stops_and_a_warn_counts_as_a_pass() {
    let mut rows = vec![row(1, 5, 2.0, None), row(2, 10, 1.9, Some(("fail", 0.5)))];
    assert_eq!(decide(&rows, &ctx()), Decision::Continue);
    rows.push(row(3, 15, 1.8, Some(("warn", 0.9))));
    assert_eq!(decide(&rows, &ctx()), Decision::StopPass);
    rows.pop();
    rows.push(row(3, 15, 1.8, Some(("pass", 0.97))));
    let d = decide(&rows, &ctx());
    assert_eq!((d, d.as_str(), d.reason(), d.stops()), (Decision::StopPass, "stop_pass", Some("pass"), true));
}

#[test]
fn there_is_no_decision_before_the_first_check_and_no_stop_without_a_check() {
    let rows = vec![row(1, 5, 2.0, None)];
    let d = decide(&rows, &ctx());
    assert_eq!((d, d.reason(), d.stops()), (Decision::Continue, None, false));
    assert_eq!(decide(&[], &ctx()), Decision::Continue);
}

#[test]
fn after_a_failed_merged_model_two_passed_checks_in_a_row_are_needed() {
    let strict = Ctx { strict: true, ..ctx() };
    let mut rows = vec![row(1, 5, 2.0, None), row(2, 10, 1.9, Some(("pass", 0.97)))];
    assert!(passed(&rows, false) && !passed(&rows, true));
    assert_eq!(decide(&rows, &strict), Decision::Continue);
    rows.push(row(3, 15, 1.8, Some(("fail", 0.5))));
    assert_eq!(decide(&rows, &strict), Decision::Continue);
    rows.push(row(4, 20, 1.8, Some(("pass", 0.97))));
    assert_eq!(decide(&rows, &strict), Decision::Continue);           // one pass after a fail is not two
    rows.push(row(5, 25, 1.7, Some(("warn", 0.9))));
    assert_eq!(decide(&rows, &strict), Decision::StopPass);
    // a round without a check in between does not make two in a row
    let gap = vec![row(1, 5, 2.0, Some(("pass", 0.97))), row(2, 10, 1.9, None), row(3, 15, 1.8, Some(("pass", 0.97)))];
    assert_eq!(decide(&gap, &strict), Decision::Continue);
}

#[test]
fn the_cap_stops_and_never_runs_to_the_cap_only() {
    let rows = vec![row(7, 35, 1.5, Some(("fail", 0.5))), row(8, 40, 1.5, None)];
    let d = decide(&rows, &ctx());
    assert_eq!((d, d.as_str(), d.reason()), (Decision::StopCap, "stop_cap", Some("cap")));
    let never = Ctx { stop_never: true, ..ctx() };
    let passing = vec![row(2, 10, 1.9, Some(("pass", 0.99)))];
    assert_eq!(decide(&passing, &never), Decision::Continue);          // a pass does not stop in "never"
    let d = decide(&rows, &never);
    assert_eq!((d, d.as_str(), d.reason()), (Decision::StopNever, "stop_never", Some("cap")));
    // a pass in the very last round is still a pass
    let last = vec![row(8, 40, 1.5, Some(("pass", 0.97)))];
    assert_eq!(decide(&last, &ctx()), Decision::StopPass);
}

#[test]
fn the_time_limit_stops_when_the_next_round_and_the_end_do_not_fit() {
    let rows = vec![row(2, 10, 1.9, Some(("fail", 0.5)))];
    let timed = Ctx { deadline: Some(1_000 + 7_199), ..ctx() };
    let d = decide(&rows, &timed);
    assert_eq!((d, d.as_str(), d.reason()), (Decision::StopTime, "stop_time", Some("time")));
    assert_eq!(decide(&rows, &Ctx { deadline: Some(1_000 + 7_200), ..ctx() }), Decision::Continue);
    assert_eq!(decide(&rows, &Ctx { deadline: None, ..ctx() }), Decision::Continue);
    // a pass wins over the time limit; the cap wins too
    let passing = vec![row(2, 10, 1.9, Some(("pass", 0.97)))];
    assert_eq!(decide(&passing, &timed), Decision::StopPass);
}

fn flat_rows(scores: &[f64], losses: &[f64]) -> Vec<RoundRow> {
    scores.iter().zip(losses).enumerate().map(|(i, (s, l))| row(i as u32 + 1, (i as u64 + 1) * 5, *l, Some(("fail", *s)))).collect()
}

#[test]
fn a_thinking_plateau_needs_both_signals_and_half_the_cap() {
    // 6 rounds of 5 M = 30 M of a 40 M cap; the last three checks do not beat the best before them and the loss is flat
    let scores = [0.3, 0.5, 0.6, 0.6, 0.55, 0.6];
    let flat = [2.0, 1.9, 1.85, 1.84, 1.84, 1.835];
    let rows = flat_rows(&scores, &flat);
    assert!(plateau_thinking(&rows, 40 * M));
    assert_eq!(decide(&rows, &ctx()), Decision::StopPlateau);
    assert_eq!(decide(&rows, &Ctx { plateau: false, ..ctx() }), Decision::Continue);
    assert_eq!(Decision::StopPlateau.reason(), Some("plateau"));
    // the loss still falls: no plateau (the thinking score alone is too noisy)
    let falling = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5];
    assert!(!plateau_thinking(&flat_rows(&scores, &falling), 40 * M));
    // the score still rises by the tie width: no plateau
    assert!(!plateau_thinking(&flat_rows(&[0.3, 0.5, 0.5, 0.5, 0.55, 0.7], &flat), 40 * M));
    assert!(plateau_thinking(&flat_rows(&[0.3, 0.5, 0.5, 0.5, 0.55, 0.6], &flat), 40 * M));       // +0.1 is inside the tie of 0.125
    // not before half of the cap
    assert!(!plateau_thinking(&rows[..5], 80 * M));
    assert!(!plateau_thinking(&rows, 80 * M));
    // too few checked rounds
    assert!(!plateau_thinking(&flat_rows(&[0.6, 0.6, 0.6], &[1.0, 1.0, 1.0]), 10 * M));
    // a missing loss is not a plateau
    let mut no_loss = flat_rows(&scores, &flat);
    no_loss.iter_mut().for_each(|r| r.calib_loss = None);
    assert!(!plateau_thinking(&no_loss, 40 * M));
    // the last round must itself have a check
    let mut unchecked = flat_rows(&scores, &flat);
    unchecked.last_mut().unwrap().check = None;
    assert!(!plateau_thinking(&unchecked, 40 * M));
}

#[test]
fn without_thinking_the_loss_alone_stops_after_half_the_cap() {
    let plain = Ctx { thinking: false, ..ctx() };
    let rows: Vec<RoundRow> = [2.0, 1.5, 1.2, 1.1, 1.0999, 1.0998].iter().enumerate().map(|(i, l)| row(i as u32 + 1, (i as u64 + 1) * 5, *l, None)).collect();
    assert!(plateau_loss(&rows, 40 * M) && !plateau_loss(&rows[..4], 40 * M));
    assert_eq!(decide(&rows, &plain), Decision::StopPlateau);
    assert_eq!(decide(&rows[..4], &plain), Decision::Continue);
    // before half the cap even a flat loss goes on
    assert!(!plateau_loss(&rows, 80 * M));
    // a falling loss goes on; a rising one is flat too
    let falling: Vec<RoundRow> = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5].iter().enumerate().map(|(i, l)| row(i as u32 + 1, (i as u64 + 1) * 5, *l, None)).collect();
    assert_eq!(decide(&falling, &plain), Decision::Continue);
    // no check rows and no thinking: "pass" never happens, even with a check present
    let with_check = vec![row(2, 10, 1.9, Some(("pass", 0.99)))];
    assert_eq!(decide(&with_check, &plain), Decision::Continue);
}

#[test]
fn a_row_of_heal_json_reads_into_a_round_row() {
    let v = json!({"n": 3, "tokens_m": 15, "tokens": 15_000_000u64, "calib_loss": 1.2, "s_start": 10, "decision": "continue",
                   "check": {"mode": "quick", "verdict": "fail", "failed": ["closed_rate"], "main": {"closed_rate": 0.69, "loop_rate": 0.13, "garbage_rate": 0.0}, "score": 0.56}});
    let r: RoundRow = serde_json::from_value(v).unwrap();
    assert_eq!((r.n, r.tokens), (3, 15_000_000));
    let c = r.check.unwrap();
    assert!(!c.passed() && c.score() == Some(0.56) && c.failed == vec!["closed_rate".to_string()]);
    let c2 = RoundCheck { main: main(0.9, 0.1), ..Default::default() };
    assert!((c2.score().unwrap() - 0.8).abs() < 1e-9);
}

// ----------------------------------------------------------------------------------------------------------------- the train key

#[test]
fn the_train_key_of_a_heal_in_rounds_does_not_change_with_the_cap() {
    let dims = HealDims::guess(17.0);
    let key = |v: Value| {
        let spec = heal(v);
        let ch = heal_choice(&spec, 17.0, Some(32.0), &dims);
        heal_train_key(&spec, &ch, spec.tokens())
    };
    let a = key(json!({"tokens_m": 40, "rounds": {"round_m": 5}}));
    let b = key(json!({"tokens_m": 40, "rounds": {"round_m": 5, "max_tokens_m": 80}}));
    assert_eq!(a, b);
    assert!(a.get("tokens").is_none() && a["rounds"]["schedule_v"] == 1);
    assert_ne!(a, key(json!({"tokens_m": 40, "rounds": {"round_m": 10}})));           // another round size: another run
    // a normal heal keeps its key, with the tokens
    let n40 = key(json!({"tokens_m": 40}));
    assert_eq!(n40["tokens"], 40_000_000u64);
    assert!(n40.get("rounds").is_none());
    assert_ne!(n40, key(json!({"tokens_m": 80})));
    assert_eq!(a["seq"], n40["seq"]);
}
