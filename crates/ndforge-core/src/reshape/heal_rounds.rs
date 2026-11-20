//! Healing in rounds (docs/HEAL_ROUNDS.md, experimental): the heal is trained in rounds of a few million tokens; after a round the model as it is trained (base + adapter, not
//! merged) is checked for thinking, and healing stops at the first round that passes. This file is the pure layer: the settings (`RoundSpec`), where the rounds and the checks
//! fall (`points`), the size rules, the stop decisions (`decide`, the plateau rules) and the overhead arithmetic. The driver that runs the jobs follows below (`run`).
//! Nothing here touches the disk or a worker.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::runs::HealSpec;
use super::thinkcheck::{think_score, THINK_TIE};
use crate::error::{Error, Result};

/// The sizes of a round the card offers, in million tokens.
pub const ROUND_M: [u32; 5] = [1, 2, 5, 10, 20];
/// The largest budget of a heal in rounds, in million tokens (the cap).
pub const MAX_CAP_M: u32 = 160;
/// A round with fewer estimated optimizer steps than this is refused: its warm-up and cool-down would be most of it.
pub const MIN_ROUND_STEPS: u32 = 8;
/// A plateau is only looked for after this share of the cap was trained (thinking can recover late).
pub const PLATEAU_AFTER: f64 = 0.5;
/// Thinking: the last this many checked rounds must not have raised the best score ...
pub const PLATEAU_CHECKS: usize = 3;
/// ... and the loss on the person's text must have fallen by less than this share over them.
pub const PLATEAU_LOSS: f64 = 0.01;
/// Without thinking: the loss must have fallen by less than this share over the last `PLATEAU_LOSS_ROUNDS` rounds.
pub const PLATEAU_LOSS_PLAIN: f64 = 0.005;
pub const PLATEAU_LOSS_ROUNDS: usize = 2;

fn d_round() -> u32 {
    5
}
fn d_check() -> String {
    "quick".into()
}
fn d_stop() -> String {
    "pass".into()
}
fn d_true() -> bool {
    true
}

/// How a heal is trained in rounds. `HealSpec.rounds` None (or absent) is exactly the heal as it always was.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundSpec {
    /// million tokens trained before each check: 1, 2, 5, 10 or 20
    #[serde(default = "d_round")]
    pub round_m: u32,
    /// the most million tokens trained (the cap); None = the heal's own `tokens_m`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_m: Option<u32>,
    /// no check before this many million tokens; None = 10 (at least a round, at most the cap)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_check_m: Option<u32>,
    /// the check after a round: "quick" (16 questions) | "full" (32)
    #[serde(default = "d_check")]
    pub check: String,
    /// the check of the merged model at the end: None = the same as `check` (Auto-cut passes its own mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_quick: Option<bool>,
    /// "pass" = stop at the first passed check | "never" = train to the cap and only record the checks (to compare with a normal heal)
    #[serde(default = "d_stop")]
    pub stop: String,
    /// stop when neither the thinking score nor the loss improves any more
    #[serde(default = "d_true")]
    pub plateau: bool,
    /// the original's check numbers, `{"quick": main, "full": main}`, when the caller has them (Auto-cut): the original is then not checked again
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Value>,
    /// unix seconds the whole heal must be done by (Auto-cut's time limit)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<u64>,
    /// go on with a stage that ended at the cap without passing: the cap is higher now and the rounds continue after the last one
    #[serde(default)]
    pub extend: bool,
    /// the exact number of tokens of a round, instead of `round_m` million: only for tests and the end-to-end run on a tiny model
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub round_exact: Option<u64>,
}

impl Default for RoundSpec {
    fn default() -> Self {
        RoundSpec { round_m: d_round(), max_tokens_m: None, first_check_m: None, check: d_check(), final_quick: None, stop: d_stop(), plateau: true, baseline: None, deadline: None, extend: false, round_exact: None }
    }
}

impl RoundSpec {
    /// The cap in tokens.
    pub fn cap_tokens(&self, heal: &HealSpec) -> u64 {
        self.max_tokens_m.map(|m| m as u64 * 1_000_000).unwrap_or_else(|| heal.tokens())
    }

    /// The tokens of one round.
    pub fn round_tokens(&self) -> u64 {
        self.round_exact.unwrap_or(self.round_m as u64 * 1_000_000)
    }

    /// No check before this many tokens: the choice, else 10 million (a cut that starts far from passing is not worth a check at 5 million), but a round at least and the cap at most.
    pub fn first_check_tokens(&self, heal: &HealSpec) -> u64 {
        let round = self.round_tokens();
        let want = self.first_check_m.map(|m| m as u64 * 1_000_000).unwrap_or(if self.round_exact.is_some() { round } else { 10_000_000 });
        want.max(round).min(self.cap_tokens(heal).max(round))
    }

    pub fn quick(&self) -> bool {
        self.check != "full"
    }

    /// Is the check at the end (of the merged model) the quick one?
    pub fn final_is_quick(&self) -> bool {
        self.final_quick.unwrap_or(self.quick())
    }

    pub fn never(&self) -> bool {
        self.stop == "never"
    }

    pub fn validate(&self, heal: &HealSpec) -> Result<()> {
        let bad = |what: &str| Err(Error::new(format!("Healing in rounds: {what}.")));
        match self.round_exact {
            Some(n) if !(1_000..=100_000_000).contains(&n) => return bad("the exact round size must be between 1,000 and 100,000,000 tokens"),
            Some(_) => {}
            None if !ROUND_M.contains(&self.round_m) => return bad("a round is 1, 2, 5, 10 or 20 million tokens"),
            None => {}
        }
        if let Some(m) = self.max_tokens_m {
            if !(1..=MAX_CAP_M).contains(&m) {
                return bad("the most tokens is 1 to 160 million");
            }
        }
        let (cap, round) = (self.cap_tokens(heal), self.round_tokens());
        if cap < round {
            return bad("the most tokens must be at least one round");
        }
        if let Some(f) = self.first_check_m {
            let f = f as u64 * 1_000_000;
            if f < round || f > cap {
                return bad("the first check must come after at least one round and not after the most tokens");
            }
        }
        if !["quick", "full"].contains(&self.check.as_str()) {
            return bad("the check is quick or full");
        }
        if !["pass", "never"].contains(&self.stop.as_str()) {
            return bad("stop at the first passed check, or never");
        }
        if let Some(b) = &self.baseline {
            if !b.is_object() {
                return bad("the original's numbers must be an object with a quick and a full part");
            }
        }
        Ok(())
    }
}

/// One round of the plan: rounds are numbered from 1; `start` and `end` are cumulative tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundPoint {
    pub n: u32,
    pub start: u64,
    pub end: u64,
    /// the thinking check after this round (the last round has none: the merged model's check is the gate)
    pub check: bool,
}

/// The rounds up to `cap` tokens: every round `round` tokens (the last one what is left), a check after each round that ends at or after `first_check` and before the cap.
pub fn points(cap: u64, round: u64, first_check: u64) -> Vec<RoundPoint> {
    if round == 0 || cap == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let (mut start, mut n) = (0u64, 0u32);
    while start < cap {
        n += 1;
        let end = (start + round).min(cap);
        out.push(RoundPoint { n, start, end, check: end < cap && end >= first_check });
        start = end;
    }
    out
}

/// The optimizer steps of a round, estimated: a step reads `0.5 x seq x per_update` tokens (rows are shorter than the sequence length on average).
pub fn est_steps(seq: u32, per_update: u32, round_tokens: u64) -> f64 {
    let per_step = 0.5 * seq.max(1) as f64 * per_update.max(1) as f64;
    round_tokens as f64 / per_step
}

/// Refuses a round with too few steps; the text is the one the card shows.
pub fn check_round_size(seq: u32, per_update: u32, round_tokens: u64) -> Result<()> {
    let steps = est_steps(seq, per_update, round_tokens);
    if steps < MIN_ROUND_STEPS as f64 {
        return Err(Error::new(format!(
            "A round of {:.1} million tokens is only about {} training steps at sequence length {seq} with {per_update} sequences per update. A round needs at least {MIN_ROUND_STEPS} steps; choose a longer round or a shorter sequence length.",
            round_tokens as f64 / 1e6, steps.floor() as u64
        )));
    }
    Ok(())
}

/// The share of a round's time that is overhead (loading the models, the check, saving) when a round of `round_m` million tokens costs `work_min_per_m` minutes per million
/// tokens and the overhead is `overhead_min`.
pub fn overhead_share(overhead_min: f64, work_min_per_m: f64, round_m: f64) -> f64 {
    let work = work_min_per_m * round_m;
    if overhead_min + work <= 0.0 {
        0.0
    } else {
        overhead_min / (overhead_min + work)
    }
}

/// The round size (million tokens) that costs the least overhead for a heal that is expected to pass after `n_star_m` million tokens: `sqrt(2 N O / c)` (a round finishes late by
/// half a round on average).
pub fn best_round_m(n_star_m: f64, overhead_min: f64, work_min_per_m: f64) -> f64 {
    if work_min_per_m <= 0.0 {
        return 0.0;
    }
    (2.0 * n_star_m * overhead_min / work_min_per_m).sqrt()
}

// ------------------------------------------------------------------------------------------------------------------ the decisions

/// The check after a round, as kept in `heal.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundCheck {
    #[serde(default)]
    pub mode: String,
    /// "pass" | "warn" | "fail"
    #[serde(default)]
    pub verdict: String,
    #[serde(default)]
    pub failed: Vec<String>,
    #[serde(default)]
    pub main: Value,
    #[serde(default)]
    pub score: Option<f64>,
}

impl RoundCheck {
    pub fn passed(&self) -> bool {
        self.verdict == "pass" || self.verdict == "warn"
    }

    pub fn score(&self) -> Option<f64> {
        self.score.or_else(|| think_score(&self.main))
    }
}

/// What the decision needs of a round that is done (the rest of the row in `heal.json` is for the page).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundRow {
    pub n: u32,
    /// cumulative tokens trained at the end of this round
    #[serde(default)]
    pub tokens: u64,
    /// the loss on the person's text after this round (the worker's fixed calibration batches)
    #[serde(default)]
    pub calib_loss: Option<f64>,
    #[serde(default)]
    pub check: Option<RoundCheck>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ctx {
    pub cap: u64,
    /// the student thinks: its rounds are checked and "good enough" is the thinking gates
    pub thinking: bool,
    pub stop_never: bool,
    pub plateau: bool,
    /// a merged model already failed its check: stopping needs two passed round checks in a row
    pub strict: bool,
    pub now: u64,
    pub deadline: Option<u64>,
    /// the high estimate of the next round (teacher, student, loads, check), the merge and the final check, in seconds
    pub next_round_secs: f64,
    pub tail_secs: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Continue,
    StopPass,
    StopPlateau,
    StopCap,
    StopTime,
    StopNever,
}

impl Decision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Decision::Continue => "continue",
            Decision::StopPass => "stop_pass",
            Decision::StopPlateau => "stop_plateau",
            Decision::StopCap => "stop_cap",
            Decision::StopTime => "stop_time",
            Decision::StopNever => "stop_never",
        }
    }

    /// The reason the stage ended ("pass" | "plateau" | "cap" | "time"), None for Continue. "never" ends at the cap.
    pub fn reason(&self) -> Option<&'static str> {
        match self {
            Decision::Continue => None,
            Decision::StopPass => Some("pass"),
            Decision::StopPlateau => Some("plateau"),
            Decision::StopCap | Decision::StopNever => Some("cap"),
            Decision::StopTime => Some("time"),
        }
    }

    pub fn stops(&self) -> bool {
        *self != Decision::Continue
    }
}

fn relative_gain(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(a), Some(b)) if a > 0.0 => Some((a - b) / a),
        _ => None,
    }
}

/// Thinking: past half the cap, the last three CHECKED rounds did not raise the best thinking score by `THINK_TIE` over the best before them AND the loss fell by less than 1 %
/// over them. Both are needed: a check of 16 questions alone is too noisy to stop on.
pub fn plateau_thinking(rows: &[RoundRow], cap: u64) -> bool {
    let Some(last) = rows.last() else { return false };
    if (last.tokens as f64) < PLATEAU_AFTER * cap as f64 || last.check.as_ref().and_then(|c| c.score()).is_none() {
        return false;
    }
    let checked: Vec<(usize, f64)> = rows.iter().enumerate().filter_map(|(i, r)| r.check.as_ref().and_then(|c| c.score()).map(|s| (i, s))).collect();
    let n = checked.len();
    if n < PLATEAU_CHECKS + 1 {
        return false;
    }
    let best = |xs: &[(usize, f64)]| xs.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    if best(&checked[n - PLATEAU_CHECKS..]) - best(&checked[..n - PLATEAU_CHECKS]) >= THINK_TIE - 1e-12 {
        return false;
    }
    let before = rows[checked[n - PLATEAU_CHECKS - 1].0].calib_loss;
    relative_gain(before, last.calib_loss).is_some_and(|g| g < PLATEAU_LOSS)
}

/// Without thinking (no check, no "good enough"): past half the cap, the loss on the person's text fell by less than 0.5 % over the last two rounds.
pub fn plateau_loss(rows: &[RoundRow], cap: u64) -> bool {
    let n = rows.len();
    if n < PLATEAU_LOSS_ROUNDS + 1 || (rows[n - 1].tokens as f64) < PLATEAU_AFTER * cap as f64 {
        return false;
    }
    relative_gain(rows[n - 1 - PLATEAU_LOSS_ROUNDS].calib_loss, rows[n - 1].calib_loss).is_some_and(|g| g < PLATEAU_LOSS_PLAIN)
}

/// Has the stage reached "good enough" with the last round? The first passed check stops; after a merged model failed (`strict`) the last two rounds must both have passed.
pub fn passed(rows: &[RoundRow], strict: bool) -> bool {
    let ok = |r: &RoundRow| r.check.as_ref().is_some_and(|c| c.passed());
    match rows {
        [.., last] if !strict => ok(last),
        [.., prev, last] => ok(prev) && ok(last),
        _ => false,
    }
}

/// What to do after the last row of `rows` is complete (its training, and its check when it had one). Order: a pass, the cap, the time limit, a plateau, else go on.
pub fn decide(rows: &[RoundRow], ctx: &Ctx) -> Decision {
    let Some(last) = rows.last() else { return Decision::Continue };
    if ctx.thinking && !ctx.stop_never && passed(rows, ctx.strict) {
        return Decision::StopPass;
    }
    if last.tokens >= ctx.cap {
        return if ctx.stop_never { Decision::StopNever } else { Decision::StopCap };
    }
    if let Some(d) = ctx.deadline {
        if ctx.now as f64 + ctx.next_round_secs + ctx.tail_secs > d as f64 {
            return Decision::StopTime;
        }
    }
    if ctx.plateau && !ctx.stop_never {
        let flat = if ctx.thinking { plateau_thinking(rows, ctx.cap) } else { plateau_loss(rows, ctx.cap) };
        if flat {
            return Decision::StopPlateau;
        }
    }
    Decision::Continue
}

// ------------------------------------------------------------------------------------------------------------------ the driver

/// The jobs a heal in rounds runs, one at a time (the graphics card holds one model). The real ones wrap the worker jobs (`runs.rs`); the tests script them.
/// Every job returns the worker's result; `{"stopped": true}` in it means the person pressed Stop (nothing is half-written, the driver returns and a later call continues).
pub trait RoundJobs {
    /// the `rounds` block of heal.json, as the last call saved it
    fn load(&mut self) -> Option<Value>;
    fn save(&mut self, state: &Value);
    /// the original answers the project rows and the general rows up to `upto_general_tokens` general tokens (all earlier rows are already there)
    fn cache(&mut self, n: u32, upto_general_tokens: u64) -> Result<Value>;
    /// the student trains one round
    fn train(&mut self, t: &TrainRound) -> Result<Value>;
    /// the thinking check of the student as it is after round `n` (base + adapter, not merged)
    fn check_round(&mut self, n: u32, quick: bool) -> Result<Value>;
    /// the thinking check of the original (cached by the worker)
    fn check_original(&mut self, quick: bool) -> Result<Value>;
    /// merge the latest round's adapter into the healed model folder (not registered yet); a finished merge is not repeated
    fn merge(&mut self) -> Result<Value>;
    /// the thinking check of the merged model
    fn check_merged(&mut self, quick: bool) -> Result<Value>;
    /// forget the merged model: its folder, its registration and its saved checks
    fn unmerge(&mut self) -> Result<()>;
    /// add the merged model to the model list; returns what the heal's result carries (`model_id`, `healed_dir`)
    fn register(&mut self) -> Result<Value>;
    fn now_secs(&self) -> u64;
}

/// What the student needs to train round `n`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainRound {
    pub n: u32,
    pub round_tokens: u64,
    pub general_from: u64,
    pub general_to: u64,
    pub s_start: u64,
}

/// What the driver is told by the heal.
#[derive(Debug, Clone)]
pub struct Plan {
    pub spec: RoundSpec,
    pub cap: u64,
    pub round: u64,
    pub first_check: u64,
    /// the share of the tokens that is general text (0 = the project's own text only)
    pub share: f64,
    /// the student thinks: its rounds and the merged model are checked
    pub thinking: bool,
    /// the high estimate of the merge and the check of the merged model, in seconds (for the time limit)
    pub tail_secs: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Stop was pressed during this step ("cache" | "train" | "check" | "merge" | "final_check"); the state is saved
    Stopped(String),
    /// the heal ended; the value is the `rounds` result (see `result_of`)
    Done(Value),
}

pub const STATE_VERSION: u64 = 1;
pub const TAIL_SECS_DEFAULT: f64 = 3600.0;
const WRONG_ORDER: &str = "The original's answers were made again from the start (the general text came in another order), so the earlier rounds' text is not reused.";

fn secs(since: std::time::Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 10.0).round() / 10.0
}

/// The time of a phase that did not finish (stopped, failed): kept in the round's `lost_s` so that the round's minutes are wall-clock over every attempt, not only the last one.
fn lose(jobs: &mut dyn RoundJobs, st: &mut Value, i: usize, row: &mut Value, t0: std::time::Instant) {
    row["lost_s"] = json!(((row["lost_s"].as_f64().unwrap_or(0.0) + secs(t0)) * 10.0).round() / 10.0);
    st["rows"][i] = row.clone();
    jobs.save(st);
}

fn stopped(r: &Value) -> bool {
    r["stopped"].as_bool().unwrap_or(false)
}

fn new_state(plan: &Plan) -> Value {
    let base = plan.spec.baseline.clone().filter(|b| b.is_object()).unwrap_or_else(|| json!({}));
    json!({"v": STATE_VERSION, "spec": plan.spec, "cap": plan.cap, "cap_m": plan.cap as f64 / 1e6, "round": plan.round, "round_m": plan.round as f64 / 1e6, "state": "rounds",
           "rows": [], "merged_checks": [], "strict": false, "merged": false, "baseline": {"quick": base.get("quick").cloned().unwrap_or(Value::Null), "full": base.get("full").cloned().unwrap_or(Value::Null)},
           "stop": null, "final": null, "registered": null, "notes": []})
}

fn rows_of(st: &Value) -> Vec<RoundRow> {
    st["rows"].as_array().map(|a| a.iter().filter(|r| r["decision"].as_str().is_some_and(|d| !d.is_empty()) || r["trained"] == true).filter_map(|r| serde_json::from_value(r.clone()).ok()).collect()).unwrap_or_default()
}

fn mode_name(quick: bool) -> &'static str {
    if quick {
        "quick"
    } else {
        "full"
    }
}

fn check_of(mode: &str, original_main: &Value, r: &Value) -> RoundCheck {
    let g = super::thinkcheck::gates(original_main, &r["main"]);
    RoundCheck {
        mode: mode.into(),
        verdict: g["verdict"].as_str().unwrap_or("fail").into(),
        failed: g["failed"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default(),
        main: r["main"].clone(),
        score: think_score(&r["main"]),
    }
}

fn note(st: &mut Value, text: &str) {
    if let Some(a) = st["notes"].as_array_mut() {
        if !a.iter().any(|x| x == text) {
            a.push(json!(text));
        }
    }
}

/// The original's numbers for a mode: from the caller, from an earlier call, else the original is checked (the worker answers a repeated check from its file).
fn original_main(jobs: &mut dyn RoundJobs, st: &mut Value, quick: bool, on_event: &mut dyn FnMut(Value)) -> Result<std::result::Result<Value, String>> {
    let mode = mode_name(quick);
    if st["baseline"][mode].is_object() {
        return Ok(Ok(st["baseline"][mode].clone()));
    }
    on_event(json!({"kind": "heal_round", "phase": "heal", "round": st["rows"].as_array().map(|r| r.len()).unwrap_or(0), "decision": "baseline", "note": "Checking the original model first, to compare with."}));
    let r = jobs.check_original(quick)?;
    if stopped(&r) {
        return Ok(Err("check".into()));
    }
    st["baseline"][mode] = r["main"].clone();
    jobs.save(st);
    Ok(Ok(r["main"].clone()))
}

fn brief(rows: &[Value]) -> Vec<Value> {
    rows.iter().map(|r| json!({"n": r["n"], "tokens_m": r["tokens_m"], "calib_loss": r["calib_loss"], "check": r["check"], "decision": r["decision"], "seconds": r["total_s"]})).collect()
}

/// What the heal's result carries as `rounds`.
pub fn result_of(st: &Value) -> Value {
    let rows = st["rows"].as_array().cloned().unwrap_or_default();
    let tokens_m = rows.last().and_then(|r| r["tokens_m"].as_f64()).unwrap_or(0.0);
    json!({"tokens_m": tokens_m, "cap_m": st["cap_m"], "stop": st["stop"], "rows": brief(&rows), "final": st["final"], "merged_checks": st["merged_checks"], "baseline": st["baseline"], "notes": st["notes"]})
}

fn emit(on_event: &mut dyn FnMut(Value), st: &Value, row: &Value, plan: &Plan, text: &str) {
    let n = row["n"].as_u64().unwrap_or(0);
    let rounds_max = plan.cap.div_ceil(plan.round.max(1));
    on_event(json!({"kind": "heal_round", "phase": "heal", "round": n, "rounds_max": rounds_max, "tokens_m": row["tokens_m"], "cap_m": st["cap_m"], "calib_loss": row["calib_loss"],
                    "check": row["check"], "decision": row["decision"], "note": text}));
}

/// Runs a heal in rounds to its end, or until Stop. Every decision is written with `jobs.save` before the next job starts, so a stopped or crashed heal continues with the part
/// that is missing (a finished cache pass, round or check is never repeated; the worker's jobs are idempotent too). See docs/HEAL_ROUNDS.md for the rules.
pub fn run(plan: &Plan, jobs: &mut dyn RoundJobs, on_event: &mut dyn FnMut(Value)) -> Result<Outcome> {
    let spec = &plan.spec;
    let mut st = match jobs.load() {
        Some(s) if s["v"] == STATE_VERSION && s["round"] == plan.round => s,
        _ => new_state(plan),
    };
    // a finished heal that did not pass can be taken further (the cap is higher now)
    if st["state"] == "ended" {
        let tokens = st["rows"].as_array().and_then(|r| r.last()).and_then(|r| r["tokens"].as_u64()).unwrap_or(0);
        let final_ok = st["final"]["verdict"].as_str().is_some_and(|v| v == "pass" || v == "warn");
        if spec.extend && !final_ok && plan.cap > tokens {
            jobs.unmerge()?;
            st["state"] = json!("rounds");
            st["merged"] = json!(false);
            st["final"] = Value::Null;
            st["stop"] = Value::Null;
            st["registered"] = Value::Null;
            st["cap"] = json!(plan.cap);
            st["cap_m"] = json!(plan.cap as f64 / 1e6);
            note(&mut st, &format!("Healing goes on to {} million tokens after the check of the merged model did not pass.", plan.cap / 1_000_000));
            jobs.save(&st);
        } else {
            return Ok(Outcome::Done(result_of(&st)));
        }
    }
    st["cap"] = json!(plan.cap);
    loop {
        match st["state"].as_str().unwrap_or("rounds") {
            "rounds" => {
                if let Some(out) = round_step(plan, jobs, &mut st, on_event)? {
                    return Ok(out);
                }
            }
            "merging" => {
                if st["merged"] != true {
                    let r = jobs.merge()?;
                    if stopped(&r) {
                        return Ok(Outcome::Stopped("merge".into()));
                    }
                    st["merged"] = json!(true);
                }
                st["state"] = json!(if plan.thinking { "final_check" } else { "ending" });
                jobs.save(&st);
            }
            "final_check" => {
                if let Some(out) = final_step(plan, jobs, &mut st, on_event)? {
                    return Ok(out);
                }
            }
            "ending" => {
                let reg = jobs.register()?;
                st["registered"] = reg;
                st["state"] = json!("ended");
                jobs.save(&st);
                return Ok(Outcome::Done(result_of(&st)));
            }
            other => return Err(Error::new(format!("The saved state of the heal in rounds is not understood ({other}). Heal again from the start."))),
        }
    }
}

/// One step of the rounds loop: the missing part of the current round (or the start of a new one) and the decision after it. Returns Some(outcome) when Stop was pressed.
fn round_step(plan: &Plan, jobs: &mut dyn RoundJobs, st: &mut Value, on_event: &mut dyn FnMut(Value)) -> Result<Option<Outcome>> {
    let spec = &plan.spec;
    let rows = st["rows"].as_array().cloned().unwrap_or_default();
    // the row in work: the last one when it has no decision yet, else a new one
    let open = rows.last().is_some_and(|r| r["decision"].as_str().unwrap_or("").is_empty());
    let idx = if open { rows.len() - 1 } else { rows.len() };
    if !open {
        let start = rows.last().and_then(|r| r["tokens"].as_u64()).unwrap_or(0);
        let end = (start + plan.round).min(plan.cap);
        if end <= start {
            // nothing left to train (the cap is not above what was trained): end it
            st["state"] = json!("merging");
            st["stop"] = json!({"reason": "cap", "round": rows.len(), "tokens_m": start as f64 / 1e6});
            jobs.save(st);
            return Ok(None);
        }
        let n = rows.len() as u64 + 1;
        let want_check = plan.thinking && end < plan.cap && end >= plan.first_check;
        let s_start = rows.last().and_then(|r| r["s_end"].as_u64()).unwrap_or(0);
        let row = json!({"n": n, "tokens": end, "tokens_m": end as f64 / 1e6, "round_tokens": end - start, "s_start": s_start, "check_wanted": want_check, "cache_done": false, "trained": false,
                         "check": null, "decision": "", "calib_loss": null, "at": null});
        st["rows"].as_array_mut().map(|a| a.push(row));
        jobs.save(st);
    }
    let i = idx;
    let mut row = st["rows"][i].clone();
    let n = row["n"].as_u64().unwrap_or(1) as u32;
    let round_tokens = row["round_tokens"].as_u64().unwrap_or(plan.round);
    // 1. the original answers this round's chunk of text
    if row["cache_done"] != true {
        let t0 = std::time::Instant::now();
        let cumulative = row["tokens"].as_u64().unwrap_or(0);
        let upto = (cumulative as f64 * plan.share).round() as u64;
        let r = match jobs.cache(n, upto) {
            Ok(r) => r,
            Err(e) => {
                lose(jobs, st, i, &mut row, t0);
                return Err(e);
            }
        };
        if stopped(&r) {
            lose(jobs, st, i, &mut row, t0);
            return Ok(Some(Outcome::Stopped("cache".into())));
        }
        let done = r["done_rows"].as_u64().unwrap_or(0);
        let prev = i.checked_sub(1).map(|j| st["rows"][j].clone());
        let from = prev.as_ref().and_then(|p| p["general_to"].as_u64()).unwrap_or_else(|| r["stats"]["project_rows"].as_u64().unwrap_or(0));
        if let Some(p) = &prev {
            let had_general = p["general_to"].as_u64().unwrap_or(0) > p["general_from"].as_u64().unwrap_or(0);
            if had_general && r["stats"]["general_error"].is_string() {
                return Err(Error::new("The general text could not be read again for the next round, so healing stops here instead of training on a different mix. Healing can be continued when the text is available again."));
            }
            if r["rows_before"].as_u64().unwrap_or(u64::MAX) < p["general_to"].as_u64().unwrap_or(0) {
                note(st, WRONG_ORDER);
            }
        }
        row["general_from"] = json!(from);
        row["general_to"] = json!(done.max(from));
        row["general_tokens_done"] = r["general_tokens_done"].clone();
        row["cache_s"] = json!(secs(t0));
        row["cache_done"] = json!(true);
        st["rows"][i] = row.clone();
        jobs.save(st);
    }
    // 2. the student learns it
    if row["trained"] != true {
        let t0 = std::time::Instant::now();
        let t = TrainRound { n, round_tokens, general_from: row["general_from"].as_u64().unwrap_or(0), general_to: row["general_to"].as_u64().unwrap_or(0), s_start: row["s_start"].as_u64().unwrap_or(0) };
        let r = match jobs.train(&t) {
            Ok(r) => r,
            Err(e) => {
                lose(jobs, st, i, &mut row, t0);
                return Err(e);
            }
        };
        if stopped(&r) {
            lose(jobs, st, i, &mut row, t0);
            return Ok(Some(Outcome::Stopped("train".into())));
        }
        if r["round"]["complete"] == false {
            lose(jobs, st, i, &mut row, t0);
            return Err(Error::new(format!("Round {n} did not finish training. Healing can be continued.")));
        }
        row["s_end"] = r["round"]["s_end"].clone();
        row["steps"] = r["round"]["steps"].clone();
        row["calib_loss"] = r["round"]["calib_loss_after"].clone();
        if row["calib_loss"].is_null() {
            row["calib_loss"] = r["calib_loss_after"].clone();
        }
        row["train_s"] = json!(secs(t0));
        row["trained"] = json!(true);
        st["rows"][i] = row.clone();
        jobs.save(st);
    }
    // 3. the thinking check of the student as it is (base + adapter, not merged)
    if row["check_wanted"] == true && row["check"].is_null() {
        let t0 = std::time::Instant::now();
        let quick = spec.quick();
        let orig = match original_main(jobs, st, quick, on_event)? {
            Ok(m) => m,
            Err(step) => return Ok(Some(Outcome::Stopped(step))),
        };
        let r = match jobs.check_round(n, quick) {
            Ok(r) => r,
            Err(e) => {
                lose(jobs, st, i, &mut row, t0);
                return Err(e);
            }
        };
        if stopped(&r) {
            lose(jobs, st, i, &mut row, t0);
            return Ok(Some(Outcome::Stopped("check".into())));
        }
        row["check"] = serde_json::to_value(check_of(mode_name(quick), &orig, &r)).unwrap_or(Value::Null);
        row["check_s"] = json!(secs(t0));
        st["rows"][i] = row.clone();
        jobs.save(st);
    }
    // 4. what next
    let total: f64 = ["cache_s", "train_s", "check_s", "lost_s"].iter().map(|k| row[*k].as_f64().unwrap_or(0.0)).sum();
    row["total_s"] = json!((total * 10.0).round() / 10.0);
    let typed = rows_of(st);
    let slowest = st["rows"].as_array().map(|a| a.iter().filter_map(|r| r["total_s"].as_f64().or_else(|| Some(["cache_s", "train_s", "check_s", "lost_s"].iter().map(|k| r[*k].as_f64().unwrap_or(0.0)).sum::<f64>()))).fold(total, f64::max)).unwrap_or(total);
    let ctx = Ctx {
        cap: plan.cap, thinking: plan.thinking, stop_never: spec.never(), plateau: spec.plateau, strict: st["strict"] == true, now: jobs.now_secs(), deadline: spec.deadline,
        next_round_secs: slowest * 1.2, tail_secs: plan.tail_secs,
    };
    let d = decide(&typed, &ctx);
    row["decision"] = json!(d.as_str());
    row["at"] = json!(jobs.now_secs());
    st["rows"][i] = row.clone();
    if let Some(reason) = d.reason() {
        st["stop"] = json!({"reason": reason, "round": n, "tokens_m": row["tokens_m"]});
        st["state"] = json!("merging");
    }
    jobs.save(st);
    let text = match d {
        Decision::Continue => String::new(),
        Decision::StopPass => format!("Round {n} passed the thinking check: merging and checking the merged model."),
        Decision::StopPlateau => format!("Healing stopped after round {n}: neither the thinking nor the loss improves any more. Merging and checking the merged model."),
        Decision::StopCap | Decision::StopNever => format!("Round {n} reached the most tokens ({} million). Merging the model.", row["tokens_m"].as_f64().unwrap_or(0.0)),
        Decision::StopTime => format!("Healing stopped after round {n}: the time limit leaves no room for another round. Merging the model."),
    };
    emit(on_event, st, &row, plan, &text);
    Ok(None)
}

/// The check of the merged model: the real gate. A pass or warn ends the heal. A fail after a stop on a passed round check goes on with more rounds (deleting the merged model),
/// now needing two passed round checks in a row; a fail after the cap, a plateau or the time limit ends it.
fn final_step(plan: &Plan, jobs: &mut dyn RoundJobs, st: &mut Value, on_event: &mut dyn FnMut(Value)) -> Result<Option<Outcome>> {
    let spec = &plan.spec;
    let quick = spec.final_is_quick();
    let orig = match original_main(jobs, st, quick, on_event)? {
        Ok(m) => m,
        Err(step) => return Ok(Some(Outcome::Stopped(step))),
    };
    let t0 = std::time::Instant::now();
    let r = jobs.check_merged(quick)?;
    if stopped(&r) {
        return Ok(Some(Outcome::Stopped("final_check".into())));
    }
    let c = check_of(mode_name(quick), &orig, &r);
    let after = st["stop"]["round"].as_u64().unwrap_or(0);
    let tokens = st["rows"].as_array().and_then(|r| r.last()).and_then(|r| r["tokens"].as_u64()).unwrap_or(0);
    let reason = st["stop"]["reason"].as_str().unwrap_or("").to_string();
    let fin = json!({"verdict": c.verdict, "failed": c.failed, "main": c.main, "mode": c.mode, "score": c.score, "after_round": after, "seconds": secs(t0)});
    let more = !c.passed() && reason == "pass" && tokens < plan.cap && !spec.never();
    if more {
        jobs.unmerge()?;
        if let Some(a) = st["merged_checks"].as_array_mut() {
            a.push(fin.clone());
        }
        st["strict"] = json!(true);
        st["merged"] = json!(false);
        st["stop"] = Value::Null;
        st["state"] = json!("rounds");
        jobs.save(st);
        let row = st["rows"].as_array().and_then(|r| r.last().cloned()).unwrap_or(Value::Null);
        let text = format!("The merged model did not pass the check after round {after}. Healing goes on; it now needs two passed checks in a row.");
        on_event(json!({"kind": "heal_round", "phase": "heal", "round": after, "rounds_max": plan.cap.div_ceil(plan.round.max(1)), "tokens_m": row["tokens_m"], "cap_m": st["cap_m"],
                        "calib_loss": row["calib_loss"], "check": fin, "decision": "merged_fail", "note": text}));
        return Ok(None);
    }
    st["final"] = fin.clone();
    st["state"] = json!("ending");
    jobs.save(st);
    let row = st["rows"].as_array().and_then(|r| r.last().cloned()).unwrap_or(Value::Null);
    let text = format!("The merged model's check: {}.", c.verdict);
    on_event(json!({"kind": "heal_round", "phase": "heal", "round": after, "rounds_max": plan.cap.div_ceil(plan.round.max(1)), "tokens_m": row["tokens_m"], "cap_m": st["cap_m"],
                    "calib_loss": row["calib_loss"], "check": fin, "decision": "final", "note": text}));
    Ok(None)
}
