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

// ----------------------------------------------------------------------------------------------------------------- the driver with scripted jobs

use std::collections::VecDeque;

use ndforge_core::error::Result;

fn orig() -> Value {
    json!({"closed_rate": 0.97, "accuracy": 0.8, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800})
}
fn good() -> Value {
    json!({"closed_rate": 0.96, "accuracy": 0.8, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800})
}
fn bad() -> Value {
    json!({"closed_rate": 0.5, "accuracy": 0.3, "loop_rate": 0.4, "garbage_rate": 0.0, "median_think_tokens": 800})
}

/// A stand-in for the worker jobs: 10 project rows and general rows of 100,000 tokens; every call is written down; checks answer from a script.
struct Mock {
    state: Option<Value>,
    calls: Vec<String>,
    ok_calls: Vec<String>,
    round_checks: VecDeque<Value>,
    merged_checks: VecDeque<Value>,
    losses: Vec<f64>,
    done_rows: u64,
    now: u64,
    /// the call with this number (1-based, over all calls) answers {"stopped": true}
    stop_at: Option<usize>,
    count: usize,
    steps_per_round: u64,
    healed: bool,
    unmerged: usize,
}

impl Mock {
    fn new(round_checks: Vec<Value>, merged_checks: Vec<Value>) -> Mock {
        Mock { state: None, calls: vec![], ok_calls: vec![], round_checks: round_checks.into(), merged_checks: merged_checks.into(), losses: (0..40).map(|i| 3.0 - 0.1 * i as f64).collect(), done_rows: 0, now: 1_000_000,
               stop_at: None, count: 0, steps_per_round: 8, healed: false, unmerged: 0 }
    }

    /// Writes the call down; true when this call is the one that is stopped.
    fn call(&mut self, what: &str) -> bool {
        self.count += 1;
        self.calls.push(what.to_string());
        if self.stop_at == Some(self.count) {
            return true;
        }
        self.ok_calls.push(what.to_string());
        false
    }
}

impl RoundJobs for Mock {
    fn load(&mut self) -> Option<Value> {
        self.state.clone()
    }
    fn save(&mut self, state: &Value) {
        self.state = Some(state.clone());
    }
    fn cache(&mut self, n: u32, upto: u64) -> Result<Value> {
        if self.call(&format!("cache {n}")) {
            return Ok(json!({"stopped": true}));
        }
        let before = self.done_rows;
        let general = upto.div_ceil(100_000);
        self.done_rows = self.done_rows.max(10 + general);
        Ok(json!({"stopped": false, "done_rows": self.done_rows, "rows_before": before, "general_tokens_done": general * 100_000, "stats": {"project_rows": 10, "general_error": null}}))
    }
    fn train(&mut self, t: &TrainRound) -> Result<Value> {
        if self.call(&format!("train {} from {} rows {}..{}", t.n, t.s_start, t.general_from, t.general_to)) {
            return Ok(json!({"stopped": true}));
        }
        let loss = self.losses[t.n as usize - 1];
        Ok(json!({"stopped": false, "round": {"index": t.n, "s_start": t.s_start, "s_end": t.s_start + self.steps_per_round, "steps": self.steps_per_round, "complete": true, "calib_loss_after": loss}}))
    }
    fn check_round(&mut self, n: u32, quick: bool) -> Result<Value> {
        if self.call(&format!("check_round {n} {}", if quick { "quick" } else { "full" })) {
            return Ok(json!({"stopped": true}));
        }
        Ok(json!({"stopped": false, "main": self.round_checks.pop_front().expect("a scripted round check")}))
    }
    fn check_original(&mut self, quick: bool) -> Result<Value> {
        if self.call(&format!("check_original {}", if quick { "quick" } else { "full" })) {
            return Ok(json!({"stopped": true}));
        }
        Ok(json!({"stopped": false, "main": orig()}))
    }
    fn merge(&mut self) -> Result<Value> {
        if self.call("merge") {
            return Ok(json!({"stopped": true}));
        }
        self.healed = true;
        Ok(json!({"stopped": false}))
    }
    fn check_merged(&mut self, quick: bool) -> Result<Value> {
        if self.call(&format!("check_merged {}", if quick { "quick" } else { "full" })) {
            return Ok(json!({"stopped": true}));
        }
        Ok(json!({"stopped": false, "main": self.merged_checks.pop_front().expect("a scripted merged check")}))
    }
    fn unmerge(&mut self) -> Result<()> {
        self.calls.push("unmerge".into());
        self.ok_calls.push("unmerge".into());
        self.unmerged += 1;
        self.healed = false;
        Ok(())
    }
    fn register(&mut self) -> Result<Value> {
        self.calls.push("register".into());
        self.ok_calls.push("register".into());
        Ok(json!({"model_id": "local:x", "healed_dir": "healed"}))
    }
    fn now_secs(&self) -> u64 {
        self.now
    }
}

fn plan_of(spec: RoundSpec, cap_m: u64, thinking: bool) -> Plan {
    let m = 1_000_000;
    Plan { cap: cap_m * m, round: spec.round_tokens(), first_check: spec.first_check_tokens(&heal(json!({"tokens_m": 40, "rounds": {}}))).max(spec.round_tokens()), share: 0.5, thinking, tail_secs: TAIL_SECS_DEFAULT, spec }
}

fn spec_of(v: Value) -> RoundSpec {
    serde_json::from_value(v).unwrap()
}

fn drive(mock: &mut Mock, plan: &Plan) -> (Outcome, Vec<Value>) {
    let mut events = vec![];
    let out = run(plan, mock, &mut |e| events.push(e)).unwrap();
    (out, events)
}

fn done(o: Outcome) -> Value {
    match o {
        Outcome::Done(v) => v,
        Outcome::Stopped(s) => panic!("stopped at {s}"),
    }
}

fn kinds(m: &Mock, what: &str) -> usize {
    m.calls.iter().filter(|c| c.starts_with(what)).count()
}

#[test]
fn the_first_passed_check_stops_merges_once_and_the_merged_check_decides() {
    // 5 M rounds, first check at 10 M: round 2 fails, round 3 passes
    let mut m = Mock::new(vec![bad(), good()], vec![good()]);
    let plan = plan_of(spec_of(json!({})), 40, true);
    let (out, events) = drive(&mut m, &plan);
    let r = done(out);
    assert_eq!(r["stop"], json!({"reason": "pass", "round": 3, "tokens_m": 15.0}));
    assert_eq!(r["final"]["verdict"], "pass");
    assert_eq!((kinds(&m, "cache"), kinds(&m, "train"), kinds(&m, "check_round"), kinds(&m, "merge"), kinds(&m, "check_merged"), kinds(&m, "check_original"), kinds(&m, "register")), (3, 3, 2, 1, 1, 1, 1));
    assert_eq!(m.calls.iter().filter(|c| c.starts_with("check_round")).cloned().collect::<Vec<_>>(), vec!["check_round 2 quick", "check_round 3 quick"]);
    // the rounds go on from the step the last one ended at and use the rows the original answered for them
    assert_eq!(m.calls.iter().filter(|c| c.starts_with("train")).cloned().collect::<Vec<_>>(), vec!["train 1 from 0 rows 10..35", "train 2 from 8 rows 35..60", "train 3 from 16 rows 60..85"]);
    assert_eq!(r["rows"].as_array().unwrap().len(), 3);
    assert_eq!(r["rows"][1]["decision"], "continue");
    assert_eq!(r["rows"][2]["decision"], "stop_pass");
    assert!(events.iter().any(|e| e["kind"] == "heal_round" && e["decision"] == "stop_pass" && e["round"] == 3 && e["cap_m"] == 40.0));
    assert!(events.iter().any(|e| e["kind"] == "heal_round" && e["decision"] == "final"));
    // the last thing is to register the model
    assert_eq!(m.calls.last().unwrap(), "register");
    assert_eq!(m.state.as_ref().unwrap()["state"], "ended");
}

#[test]
fn a_failed_merged_model_goes_on_and_then_needs_two_passed_checks_in_a_row() {
    // round 2 passes -> merged fails -> round 3 passes (two in a row with round 2) -> stops, merged passes
    let mut m = Mock::new(vec![good(), good()], vec![bad(), good()]);
    let plan = plan_of(spec_of(json!({})), 40, true);
    let r = done(drive(&mut m, &plan).0);
    assert_eq!(r["stop"]["round"], 3);
    assert_eq!((kinds(&m, "merge"), m.unmerged, kinds(&m, "check_merged"), kinds(&m, "register")), (2, 1, 2, 1));
    assert_eq!(r["merged_checks"].as_array().unwrap().len(), 1);
    assert_eq!((r["merged_checks"][0]["after_round"].clone(), r["merged_checks"][0]["verdict"].clone()), (json!(2), json!("fail")));
    assert_eq!(r["final"]["verdict"], "pass");
    assert_eq!(m.state.as_ref().unwrap()["strict"], true);
    // with a fail in between, two passes in a row come later
    let mut m = Mock::new(vec![good(), bad(), good(), good()], vec![bad(), good()]);
    let r = done(drive(&mut m, &plan).0);
    assert_eq!(r["stop"]["round"], 5);
    assert_eq!(kinds(&m, "train"), 5);
}

#[test]
fn the_cap_ends_the_heal_with_the_last_round_merged_and_checked() {
    let mut m = Mock::new(vec![bad(); 6], vec![bad()]);
    let plan = plan_of(spec_of(json!({"plateau": false})), 40, true);
    let r = done(drive(&mut m, &plan).0);
    assert_eq!(r["stop"], json!({"reason": "cap", "round": 8, "tokens_m": 40.0}));
    assert_eq!((kinds(&m, "train"), kinds(&m, "check_round"), kinds(&m, "merge"), kinds(&m, "register")), (8, 6, 1, 1));       // checks after rounds 2..7; none at the cap
    assert_eq!(r["final"]["verdict"], "fail");
    assert_eq!(r["rows"][7]["decision"], "stop_cap");
    assert_eq!(r["final"]["failed"].as_array().unwrap().len() > 0, true);
}

#[test]
fn an_extension_goes_on_after_the_last_round_and_repeats_nothing() {
    let mut m = Mock::new(vec![bad(); 4], vec![bad(), good()]);
    let first = plan_of(spec_of(json!({"plateau": false})), 20, true);                // 4 rounds: checks after rounds 2 and 3
    let r = done(drive(&mut m, &first).0);
    assert_eq!((r["stop"]["reason"].clone(), r["final"]["verdict"].clone(), kinds(&m, "train")), (json!("cap"), json!("fail"), 4));
    m.calls.clear();
    m.round_checks = vec![bad(), good()].into();
    // the stage goes on to 30 M: the heal is opened again (the merged model is forgotten) and rounds 5 and 6 follow
    let more = plan_of(spec_of(json!({"plateau": false, "extend": true})), 30, true);
    let r = done(drive(&mut m, &more).0);
    assert_eq!(m.unmerged, 1);
    assert_eq!(m.calls.iter().filter(|c| c.starts_with("train")).cloned().collect::<Vec<_>>(), vec!["train 5 from 32 rows 110..135", "train 6 from 40 rows 135..160"]);
    assert_eq!(kinds(&m, "cache"), 2);
    assert_eq!(r["rows"].as_array().unwrap().len(), 6);
    assert_eq!(r["final"]["verdict"], "pass");
    // a heal that already passed is not opened again
    m.calls.clear();
    let r = done(drive(&mut m, &plan_of(spec_of(json!({"plateau": false, "extend": true})), 40, true)).0);
    assert!(m.calls.is_empty() && r["final"]["verdict"] == "pass");
}

#[test]
fn stop_at_every_call_then_continue_repeats_no_job() {
    let script = || Mock::new(vec![bad(), good(), good()], vec![bad(), good()]);
    let plan = plan_of(spec_of(json!({})), 40, true);
    let mut whole = script();
    let reference = done(drive(&mut whole, &plan).0);
    let total = whole.count;
    assert!(total >= 15);
    for k in 1..=total {
        let mut m = script();
        m.stop_at = Some(k);
        let (first, _) = drive(&mut m, &plan);
        assert!(matches!(first, Outcome::Stopped(_)), "call {k}: {first:?}");
        m.stop_at = None;
        let r = done(drive(&mut m, &plan).0);
        // what the stopped call did not do is done once; nothing that was done is done again
        assert_eq!(m.ok_calls, whole.ok_calls, "stopped at call {k}");
        assert_eq!(r["stop"], reference["stop"], "stopped at call {k}");
        assert_eq!(r["final"]["verdict"], reference["final"]["verdict"], "stopped at call {k}");
    }
}

#[test]
fn a_plateau_stops_and_a_failed_merged_check_after_it_ends_the_stage() {
    // flat scores and a flat loss from round 4 on; 30 M of a 40 M cap are trained when the plateau is seen (round 6)
    let mut m = Mock::new(vec![json!({"closed_rate": 0.3, "accuracy": 0.3, "loop_rate": 0.0}), json!({"closed_rate": 0.55, "loop_rate": 0.0}), json!({"closed_rate": 0.6, "loop_rate": 0.0}),
                               json!({"closed_rate": 0.6, "loop_rate": 0.0}), json!({"closed_rate": 0.55, "loop_rate": 0.0}), json!({"closed_rate": 0.6, "loop_rate": 0.0})], vec![bad()]);
    m.losses = vec![3.0, 2.5, 2.2, 2.19, 2.185, 2.18, 2.0, 2.0];
    // the checks start at the first round here, so that six checks are scored
    let spec = spec_of(json!({"round_m": 5, "first_check_m": 5}));
    let plan = plan_of(spec, 40, true);
    let r = done(drive(&mut m, &plan).0);
    assert_eq!(r["stop"]["reason"], "plateau");
    assert_eq!(r["stop"]["round"], 6);
    // the merged model failed, but a plateau means more rounds would not help: the stage ends there
    assert_eq!((r["final"]["verdict"].clone(), kinds(&m, "train"), m.unmerged, kinds(&m, "register")), (json!("fail"), 6, 0, 1));
}

#[test]
fn the_time_limit_stops_with_the_reason_time_and_the_merged_model_is_still_checked() {
    let mut m = Mock::new(vec![], vec![bad()]);
    let mut spec = spec_of(json!({"round_m": 5, "first_check_m": 10}));
    spec.deadline = Some(m.now + 3_000);                                 // the tail alone (merge and check, one hour) does not fit
    let plan = plan_of(spec, 40, true);
    let r = done(drive(&mut m, &plan).0);
    assert_eq!(r["stop"], json!({"reason": "time", "round": 1, "tokens_m": 5.0}));
    assert_eq!((kinds(&m, "train"), kinds(&m, "merge"), kinds(&m, "check_merged"), r["final"]["verdict"].clone()), (1, 1, 1, json!("fail")));
    assert_eq!(r["rows"][0]["decision"], "stop_time");
    assert_eq!(m.unmerged, 0);
    // enough time: it goes on
    let mut m = Mock::new(vec![good()], vec![good()]);
    let mut spec = spec_of(json!({"round_m": 5, "first_check_m": 5}));
    spec.deadline = Some(m.now + 10 * 3_600);
    let r = done(drive(&mut m, &plan_of(spec, 40, true)).0);
    assert_eq!(r["stop"]["reason"], "pass");
}

#[test]
fn a_student_that_does_not_think_has_no_checks_and_stops_on_the_loss() {
    let mut m = Mock::new(vec![], vec![]);
    m.losses = vec![3.0, 2.0, 1.5, 1.3, 1.25, 1.249, 1.2489, 1.2488];
    let plan = plan_of(spec_of(json!({})), 40, false);
    let r = done(drive(&mut m, &plan).0);
    assert_eq!((kinds(&m, "check_round"), kinds(&m, "check_original"), kinds(&m, "check_merged"), kinds(&m, "merge"), kinds(&m, "register")), (0, 0, 0, 1, 1));
    assert_eq!(r["stop"]["reason"], "plateau");
    assert!(r["stop"]["round"].as_u64().unwrap() >= 5 && r["stop"]["round"].as_u64().unwrap() < 8);
    assert!(r["final"].is_null());
    // without a plateau the cap is reached
    let mut m = Mock::new(vec![], vec![]);
    m.losses = (0..40).map(|i| 3.0 - 0.05 * i as f64).collect();
    let r = done(drive(&mut m, &plan).0);
    assert_eq!((r["stop"]["reason"].clone(), kinds(&m, "train")), (json!("cap"), 8));
}

#[test]
fn the_originals_numbers_from_the_caller_are_used_and_the_original_is_checked_once_per_mode() {
    let mut m = Mock::new(vec![good()], vec![good()]);
    let spec = spec_of(json!({"round_m": 5, "first_check_m": 5, "baseline": {"quick": orig()}}));
    done(drive(&mut m, &plan_of(spec, 40, true)).0);
    assert_eq!(kinds(&m, "check_original"), 0);
    // the round checks are quick and the final one is full: the original is asked for once in each mode
    let mut m = Mock::new(vec![bad(), good()], vec![good()]);
    let spec = spec_of(json!({"round_m": 5, "first_check_m": 5, "final_quick": false}));
    done(drive(&mut m, &plan_of(spec, 40, true)).0);
    assert_eq!(m.calls.iter().filter(|c| c.starts_with("check_original")).cloned().collect::<Vec<_>>(), vec!["check_original quick", "check_original full"]);
    assert!(m.calls.contains(&"check_merged full".to_string()));
    // "never" runs to the cap, records the checks and ignores a pass
    let mut m = Mock::new(vec![good(); 7], vec![good()]);
    let spec = spec_of(json!({"stop": "never", "first_check_m": 5}));
    let r = done(drive(&mut m, &plan_of(spec, 40, true)).0);
    assert_eq!((r["stop"]["reason"].clone(), kinds(&m, "train"), kinds(&m, "check_round")), (json!("cap"), 8, 7));
    assert_eq!(r["rows"][7]["decision"], "stop_never");
}

#[test]
fn a_changed_round_size_starts_a_new_state_and_an_old_state_is_not_misread() {
    let mut m = Mock::new(vec![good()], vec![good()]);
    m.state = Some(json!({"v": 1, "round": 10_000_000u64, "state": "ended", "rows": [{"n": 1}]}));
    let r = done(drive(&mut m, &plan_of(spec_of(json!({"round_m": 5, "first_check_m": 5})), 40, true)).0);
    assert_eq!(r["rows"].as_array().unwrap().len(), 1);
    assert_eq!(kinds(&m, "train"), 1);
}

#[test]
fn the_original_being_answered_again_and_a_lost_general_text_are_noticed() {
    // the saved answers were made again from the start: a note, the heal goes on
    let mut m = Mock::new(vec![bad(), bad()], vec![bad()]);
    let plan = plan_of(spec_of(json!({"first_check_m": 5, "plateau": false})), 15, true);
    struct Again(Mock);
    impl RoundJobs for Again {
        fn load(&mut self) -> Option<Value> { self.0.load() }
        fn save(&mut self, s: &Value) { self.0.save(s) }
        fn cache(&mut self, n: u32, upto: u64) -> Result<Value> {
            let mut r = self.0.cache(n, upto)?;
            if n == 2 { r["rows_before"] = json!(0); }
            if n == 3 { r["stats"]["general_error"] = json!("offline"); }
            Ok(r)
        }
        fn train(&mut self, t: &TrainRound) -> Result<Value> { self.0.train(t) }
        fn check_round(&mut self, n: u32, q: bool) -> Result<Value> { self.0.check_round(n, q) }
        fn check_original(&mut self, q: bool) -> Result<Value> { self.0.check_original(q) }
        fn merge(&mut self) -> Result<Value> { self.0.merge() }
        fn check_merged(&mut self, q: bool) -> Result<Value> { self.0.check_merged(q) }
        fn unmerge(&mut self) -> Result<()> { self.0.unmerge() }
        fn register(&mut self) -> Result<Value> { self.0.register() }
        fn now_secs(&self) -> u64 { self.0.now_secs() }
    }
    let mut a = Again(std::mem::replace(&mut m, Mock::new(vec![], vec![])));
    let err = run(&plan, &mut a, &mut |_| {}).unwrap_err().to_string();
    assert!(err.contains("could not be read again") && err.contains("continued"), "{err}");
    let st = a.0.state.clone().unwrap();
    assert!(st["notes"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().contains("made again from the start")));
    assert_eq!(st["rows"].as_array().unwrap().len(), 3);          // round 3 was opened, nothing trained for it
    assert_eq!(st["rows"][2]["trained"], false);
}
