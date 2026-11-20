//! vLLM in the Heal card and in Auto-cut: the numbers measured on THIS PC, the time of a step with and without vLLM, and the automatic choice (docs/VLLM.md, "Teacher cache" and
//! "Auto-cut"). Pure functions: nothing is read here (`vllm::history_with_measured` puts the files' numbers into the `History`), nothing runs.
//!
//! EXPERIMENTAL. The numbers come from three places and the window says which:
//!   * `vllm_selftest.json` of the person's own self-test: the engine's start-up seconds (`startup_seconds`), the generation speed with 40 requests at once (`tokens_per_s`) and the wall
//!     time of the Quick thinking suite on the engine (`wall_seconds`). They are measured on that PC, with the model the self-test used.
//!   * `vllm_teacher_check.json` of the person's own teacher check: the speed of SCORING rows (the teacher's answers), `vllm.tokens_per_s`.
//!   * the research report's estimates (labelled `report_estimate`): used only where the PC has no measurement. The self-test's generation speed is NOT the scoring speed of the
//!     teacher's answers (a different job: reading text instead of writing it), so it is never used for them.
use serde::{Deserialize, Serialize};

use crate::timing::History;

pub const KEY_STARTUP: &str = "vllm_startup_s";
pub const KEY_QUICK_WALL: &str = "vllm_quick_wall_s";
pub const KEY_GEN_TPS: &str = "vllm_gen_tok_s";
pub const KEY_SCORE_TPS: &str = "vllm_score_tok_s";
/// the size (billions of parameters) of the model the self-test measured: its numbers are for THAT model
pub const KEY_MODEL_B: &str = "vllm_model_b";
/// decode speed measured on LONG answers (thinking-like length): the only generation speed the estimates use; the short-answer speed of the Quick suite is never used for a thinking check
pub const KEY_LONG_TPS: &str = "vllm_long_tok_s";
/// Tokens a Quick thinking check writes (16 thinking answers of about 1,800 tokens and the controls): an assumption, used with the long-answer speed when no suite wall time of long answers exists.
pub const QUICK_CHECK_TOKENS: f64 = 30_000.0;
/// Self-test numbers are used for another model only up to this size ratio (either way); beyond it the report's formulas are used and the plan says so.
pub const MAX_SIZE_RATIO: f64 = 2.0;

/// The research report's scoring speed of the teacher's answers on one RTX 5090 (tokens per second, low and high): an estimate, not measured here.
pub const REPORT_SCORE_TPS: (f64, f64) = (3000.0, 3800.0);
/// Start-up of the engine in minutes when this PC has not measured it: the report said 2 to 5, the first self-test on a real PC took 8.5, so the range is wide.
pub const REPORT_STARTUP_MIN: (f64, f64) = (2.0, 9.0);
/// The Quick thinking suite on the engine in minutes (generation only), when not measured: the report's 50 thousand tokens at 350 to 550 tokens per second, with a tail.
pub const REPORT_QUICK_GEN_MIN: (f64, f64) = (2.0, 4.0);
/// Spread around a measured number (a computer is not exactly as fast every time), as `auto_time` does.
pub const SPREAD: (f64, f64) = (0.85, 1.25);
/// vLLM must be at least this much faster (middle of the range) for "automatic" to choose it: a few percent is inside the noise of an estimate.
pub const AUTO_MARGIN: f64 = 0.9;

/// What Auto-cut is asked to do with vLLM (experimental, off when absent: the key is then not written).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoVllm {
    /// the original's answers for the heals through vLLM
    #[serde(default)]
    pub teacher_cache: bool,
    /// the thinking checks through vLLM (each check starts the engine: it pays only when the check is long)
    #[serde(default)]
    pub checks: bool,
    /// "automatic": use vLLM for a step only when its estimate is lower than the standard engine's (the plan says why); the two switches above are then ignored
    #[serde(default)]
    pub automatic: bool,
    /// the 8-bit KV cache for the thinking checks (never for the teacher's answers); experimental, off
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub kv_fp8: bool,
}

impl AutoVllm {
    pub fn any(&self) -> bool {
        self.teacher_cache || self.checks || self.automatic
    }
}

/// The measured (or not) numbers, from the History.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Numbers {
    /// seconds from "start" to "ready" of the engine (self-test)
    pub startup_s: Option<f64>,
    /// the Quick thinking suite's wall time on the engine, seconds (self-test)
    pub quick_wall_s: Option<f64>,
    /// generation tokens per second with 40 requests at once (self-test)
    pub gen_tps: Option<f64>,
    /// scoring tokens per second (teacher check)
    pub score_tps: Option<f64>,
    /// billions of parameters of the model the self-test ran (None: an older report that does not say: its numbers are then NOT used)
    #[serde(default)]
    pub model_b: Option<f64>,
    /// decode tokens per second on long answers (self-test, 8 requests of 2,048 tokens)
    #[serde(default)]
    pub long_tps: Option<f64>,
}

impl Numbers {
    pub fn from_history(h: &History) -> Numbers {
        let pick = |k: &str| h.pick(k, 1.0).map(|x| x.0);
        Numbers { startup_s: pick(KEY_STARTUP), quick_wall_s: pick(KEY_QUICK_WALL), gen_tps: pick(KEY_GEN_TPS), score_tps: pick(KEY_SCORE_TPS), model_b: pick(KEY_MODEL_B), long_tps: pick(KEY_LONG_TPS) }
    }

    /// Writes the numbers into a History (the form `vllm::history_with_measured` uses).
    pub fn push_into(&self, h: &mut History) {
        for (k, v) in [(KEY_STARTUP, self.startup_s), (KEY_QUICK_WALL, self.quick_wall_s), (KEY_GEN_TPS, self.gen_tps), (KEY_SCORE_TPS, self.score_tps), (KEY_MODEL_B, self.model_b), (KEY_LONG_TPS, self.long_tps)] {
            if let Some(v) = v {
                h.push(k, 1.0, v);
            }
        }
    }
}

/// A time with where it comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Min {
    pub lo: f64,
    pub hi: f64,
    /// measured on this PC (the self-test or the teacher check)
    pub measured: bool,
}

fn measured(minutes: f64) -> Min {
    Min { lo: minutes * SPREAD.0, hi: minutes * SPREAD.1, measured: true }
}

/// How the self-test's numbers carry over to a model of `b` billion parameters: `Some(ratio)` (first-order rule: start-up follows the model's bytes, generation time follows the parameters),
/// None when the self-test's model is unknown or differs by more than 2 times (the report's formulas are used then).
pub fn size_ratio(n: &Numbers, b: f64) -> Option<f64> {
    let m = n.model_b.filter(|m| *m > 0.0)?;
    let r = b / m;
    (b > 0.0 && (1.0 / MAX_SIZE_RATIO..=MAX_SIZE_RATIO).contains(&r)).then_some(r)
}

/// Minutes for the engine to start (container, model load, graph capture), once, for a model of `b` billion parameters.
pub fn startup(n: &Numbers, b: f64) -> Min {
    match (n.startup_s, size_ratio(n, b)) {
        (Some(s), Some(r)) => measured(s / 60.0 * r),
        _ => Min { lo: REPORT_STARTUP_MIN.0, hi: REPORT_STARTUP_MIN.1, measured: false },
    }
}

/// Minutes to score `tokens_m` million tokens (the teacher's answers) once the engine runs.
pub fn scoring(n: &Numbers, tokens_m: f64) -> Min {
    let t = tokens_m * 1e6;
    match n.score_tps {
        Some(v) if v > 0.0 => measured(t / v / 60.0),
        _ => Min { lo: t / REPORT_SCORE_TPS.1 / 60.0, hi: t / REPORT_SCORE_TPS.0 / 60.0, measured: false },
    }
}

/// Minutes of the generation of one thinking check on the running engine (without its start-up). The self-test's Quick suite is the measurement; a Full check is `full_over_quick` times it.
pub fn check_generation(n: &Numbers, quick: bool, full_over_quick: f64, b: f64) -> Min {
    let f = if quick { 1.0 } else { full_over_quick.max(1.0) };
    match (n.quick_wall_s.or(n.long_tps.filter(|t| *t > 0.0).map(|t| QUICK_CHECK_TOKENS / t)), size_ratio(n, b)) {
        (Some(w), Some(r)) => {
            let m = measured(w / 60.0 * f * r);
            Min { lo: m.lo, hi: m.hi, measured: true }
        }
        _ => Min { lo: REPORT_QUICK_GEN_MIN.0 * f, hi: REPORT_QUICK_GEN_MIN.1 * f, measured: false },
    }
}

fn mid(m: (f64, f64)) -> f64 {
    (m.0 + m.1) / 2.0
}

/// Which way a step goes: "off" (not asked), "on" (asked for), "automatic".
pub fn mode_of(av: Option<&AutoVllm>, kind: &str) -> &'static str {
    match av {
        Some(a) if a.automatic => "automatic",
        Some(a) if kind == "teacher_cache" && a.teacher_cache => "on",
        Some(a) if kind == "checks" && a.checks => "on",
        _ => "off",
    }
}

/// One decision of the plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// the id of the step (`s1.heal`, `think_gate`, `s1.check_cut`, `s1.check_heal`)
    pub step: String,
    /// "teacher_cache" | "checks"
    pub kind: String,
    /// "off" | "on" | "automatic"
    pub mode: String,
    /// vLLM is used in this step
    pub use_vllm: bool,
    /// minutes (low, high) of the part with the standard engine and with vLLM (start-up included), whatever was chosen
    pub standard_min: (f64, f64),
    pub vllm_min: (f64, f64),
    /// the vLLM numbers are measured on this PC (not the report's estimates)
    pub measured: bool,
    /// one plain sentence: what was decided and why
    pub reason: String,
}

fn fmt_min(m: (f64, f64)) -> String {
    if mid(m) >= 90.0 { format!("{:.1} to {:.1} hours", m.0 / 60.0, m.1 / 60.0) } else { format!("{:.0} to {:.0} minutes", m.0, m.1) }
}

/// Decides one step. `standard` is the part's time with the standard engine, `vllm` its time with vLLM (start-up included).
pub fn decide(step: &str, kind: &str, mode: &str, standard: (f64, f64), vllm: Min, blocked: Option<&str>) -> Decision {
    let r1 = |x: f64| (x * 10.0).round() / 10.0;          // one decimal: what is saved with a run reads back the same
    let (standard, v) = ((r1(standard.0), r1(standard.1)), (r1(vllm.lo), r1(vllm.hi)));
    let src = if vllm.measured { "from your own measurement" } else { "from the research report's estimate, not measured on this PC" };
    let what = if kind == "teacher_cache" { "the original's answers" } else { "this thinking check" };
    let (use_vllm, reason) = if let Some(b) = blocked {
        (false, format!("vLLM is not used for {what}: {b}"))
    } else {
        match mode {
            "on" => (true, format!("You asked for vLLM for {what}: about {} instead of {} ({src}).", fmt_min(v), fmt_min(standard))),
            "automatic" => {
                let better = mid(v) < mid(standard) * AUTO_MARGIN;
                if better {
                    (true, format!("Automatic: vLLM for {what}, about {} instead of {} ({src}).", fmt_min(v), fmt_min(standard)))
                } else {
                    (false, format!("Automatic: the standard engine for {what}: vLLM would take about {} against {} ({src}); the engine's start-up eats the gain.", fmt_min(v), fmt_min(standard)))
                }
            }
            _ => (false, String::new()),
        }
    };
    Decision { step: step.into(), kind: kind.into(), mode: mode.into(), use_vllm, standard_min: standard, vllm_min: v, measured: vllm.measured, reason }
}

/// The plan's account of vLLM (additive to the time estimate; None when vLLM is not asked for).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VllmPlan {
    pub numbers: Numbers,
    pub decisions: Vec<Decision>,
    /// the prerequisites of using vLLM in a run (the window shows them as a list; a missing one is a note, the run still falls back to the standard engine)
    pub notes: Vec<String>,
}

impl VllmPlan {
    pub fn decision(&self, step: &str) -> Option<&Decision> {
        self.decisions.iter().find(|d| d.step == step)
    }

    pub fn uses(&self, step: &str) -> bool {
        self.decision(step).is_some_and(|d| d.use_vllm)
    }
}

/// The sentences about what the numbers are (the window shows them under the switches). `b`: the size of the model the step would run.
pub fn basis_notes(n: &Numbers, b: f64) -> Vec<String> {
    let mut v = Vec::new();
    match (n.startup_s, n.model_b, size_ratio(n, b)) {
        (Some(s), Some(m), Some(r)) => v.push(format!("Your vLLM self-test measured the engine's start-up at {:.1} minutes on a {m:.0} B model; for this {b:.0} B model that is scaled to about {:.1} minutes by size (a first-order estimate: start-up follows the model's size).", s / 60.0, s / 60.0 * r)),
        (Some(s), Some(m), None) => v.push(format!("Your vLLM self-test measured {:.1} minutes of start-up on a {m:.0} B model, which is more than 2 times different from this {b:.0} B model: those numbers are NOT used here, the research report's estimates are.", s / 60.0)),
        (Some(s), None, _) => v.push(format!("Your vLLM self-test measured {:.1} minutes of start-up, but its report does not say which model it ran (an older report): those numbers are NOT used. Run the self-test again.", s / 60.0)),
        _ => v.push("The engine's start-up was not measured on this PC (run the vLLM self-test): the report says 2 to 5 minutes, a real PC took 8.5 for a 9 B model, so the estimate is a wide range.".into()),
    }
    match n.score_tps {
        Some(s) => v.push(format!("Scoring speed {s:.0} tokens per second: measured by your own teacher check.")),
        None => v.push(format!("Scoring speed {:.0} to {:.0} tokens per second: the research report's estimate, NOT measured on this PC (run \"Check the vLLM teacher\"). The self-test's generation speed{} is a different job and is not used for it.",
                               REPORT_SCORE_TPS.0, REPORT_SCORE_TPS.1, n.gen_tps.map(|g| format!(" ({g:.0} tokens per second)")).unwrap_or_default())),
    }
    match (n.quick_wall_s, n.long_tps) {
        (Some(_), _) => {}
        (None, Some(t)) => v.push(format!("A thinking check's generation time is estimated from your self-test's speed on LONG answers ({t:.0} tokens per second) and an assumed {QUICK_CHECK_TOKENS:.0} tokens for a Quick check: an estimate. The short-answer speed of the Quick suite is not used (short answers overstate it).")),
        (None, None) => v.push("A thinking check's time on vLLM was not measured on this PC (the self-test's long-answer speed is the measurement; a self-test that only had short answers is not used for it).".into()),
    }
    v
}
