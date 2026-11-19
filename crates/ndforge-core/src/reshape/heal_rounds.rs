//! Healing in rounds (docs/HEAL_ROUNDS.md, experimental): the heal is trained in rounds of a few million tokens; after a round the model as it is trained (base + adapter, not
//! merged) is checked for thinking, and healing stops at the first round that passes. This file is the pure layer: the settings (`RoundSpec`), where the rounds and the checks
//! fall (`points`), the size rules, the stop decisions (`decide`, the plateau rules) and the overhead arithmetic. The driver that runs the jobs follows below (`run`).
//! Nothing here touches the disk or a worker.
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
