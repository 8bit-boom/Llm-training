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
}

impl Numbers {
    pub fn from_history(h: &History) -> Numbers {
        let pick = |k: &str| h.pick(k, 1.0).map(|x| x.0);
        Numbers { startup_s: pick(KEY_STARTUP), quick_wall_s: pick(KEY_QUICK_WALL), gen_tps: pick(KEY_GEN_TPS), score_tps: pick(KEY_SCORE_TPS) }
    }

    /// Writes the numbers into a History (the form `vllm::history_with_measured` uses).
    pub fn push_into(&self, h: &mut History) {
        for (k, v) in [(KEY_STARTUP, self.startup_s), (KEY_QUICK_WALL, self.quick_wall_s), (KEY_GEN_TPS, self.gen_tps), (KEY_SCORE_TPS, self.score_tps)] {
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

/// Minutes for the engine to start (container, model load, graph capture), once.
pub fn startup(n: &Numbers) -> Min {
    match n.startup_s {
        Some(s) => measured(s / 60.0),
        None => Min { lo: REPORT_STARTUP_MIN.0, hi: REPORT_STARTUP_MIN.1, measured: false },
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
pub fn check_generation(n: &Numbers, quick: bool, full_over_quick: f64) -> Min {
    let f = if quick { 1.0 } else { full_over_quick.max(1.0) };
    match n.quick_wall_s {
        Some(w) => {
            let m = measured(w / 60.0 * f);
            Min { lo: m.lo, hi: m.hi, measured: true }
        }
        None => Min { lo: REPORT_QUICK_GEN_MIN.0 * f, hi: REPORT_QUICK_GEN_MIN.1 * f, measured: false },
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
    let v = (vllm.lo, vllm.hi);
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

/// The sentences about what the numbers are (the window shows them under the switches).
pub fn basis_notes(n: &Numbers) -> Vec<String> {
    let mut v = Vec::new();
    match n.startup_s {
        Some(s) => v.push(format!("The engine's start-up is {:.1} minutes: measured by your own vLLM self-test.", s / 60.0)),
        None => v.push("The engine's start-up was not measured on this PC (run the vLLM self-test): the report says 2 to 5 minutes, a real PC took 8.5, so the estimate is a wide range.".into()),
    }
    match n.score_tps {
        Some(s) => v.push(format!("Scoring speed {:.0} tokens per second: measured by your own teacher check.", s)),
        None => v.push(format!("Scoring speed {:.0} to {:.0} tokens per second: the research report's estimate, NOT measured on this PC (run \"Check the vLLM teacher\"). The self-test's generation speed{} is a different job and is not used for it.",
                               REPORT_SCORE_TPS.0, REPORT_SCORE_TPS.1, n.gen_tps.map(|g| format!(" ({g:.0} tokens per second)")).unwrap_or_default())),
    }
    if n.quick_wall_s.is_none() {
        v.push("A thinking check's time on vLLM was not measured on this PC either (the self-test's Quick suite is the measurement).".into());
    }
    v
}
