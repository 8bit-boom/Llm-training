//! Auto-cut, the planner (docs/AUTO_CUT.md): from "this model, about this many billion parameters" to the cut that is tried, the stages it is done in, and what
//! the disk and the card must have. Pure functions: no files, no network, nothing is loaded. The orchestrator (`reshape::auto`) runs what this decides.
//!
//! Rules (owner decisions of the thinking plan, section 2.7):
//! - width before depth: the MLP of every layer is thinned first, by at most 38 % (Qwen3.8-27B: 17,408 to 10,752); whole blocks are removed only for what is still
//!   missing, at most 4 without the owner's confirmation (`AutoSpec.allow_many_blocks`);
//! - two stages by default: stage A takes the depth part (or half of the width cut), is healed and checked; stage B measures the healed model again, cuts the rest
//!   and is healed once more. The teacher is always the original model;
//! - the sizes are exact: they come from the model's dimensions (`plan::width_params`), not from an estimate.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub use super::auto_time::{time_estimate, ExtraTime, StepPart, StepTime, TimeEstimate};
use super::plan::{self, HealDims, HealPlan, Inspect, Mode, Width};
use super::runs::{heal_choice, HealSpec};
use crate::error::{Error, Result};
use crate::timing::History;

/// Whole blocks that may be removed without the owner's confirmation.
pub const MAX_BLOCKS_FREE: u32 = 4;
/// The MLP keeps at least this share of its width while blocks are not needed (17,408 x 0.617 = 10,741 -> the stop 10,752).
pub const WIDTH_FIRST_KEEP: f64 = 0.617;
/// Models that think are cut to 20 B or more by default; smaller works but is warned about.
pub const MIN_TARGET_B_THINKING: f64 = 20.0;
/// A cut is "on target" when its size is within this share of the target.
pub const TARGET_TOLERANCE: f64 = 0.04;
/// One step of "back off the cut" for the width: 4 stops of 256.
pub const BACK_OFF_MLP: u32 = 1024;
/// How many cuts "Try first" is asked about.
pub const TRY_TOP: usize = 3;
/// The teacher's saved answers: bytes per token at k = 32.
pub const CACHE_BYTES_PER_TOKEN: f64 = 196.0;
/// Heal budget of the first stage of two (millions of tokens); the last stage uses `plan::thinking_tokens_m`.
pub const STAGE_A_TOKENS_M: u32 = 40;

fn yes() -> bool {
    true
}
fn two() -> u32 {
    2
}
fn q4() -> Vec<String> {
    vec!["q4_k_m".into()]
}

/// The tokens a retry heal adds to the healed model's own (the original's answers are extended by the difference).
pub const RETRY_TOKENS_M: u32 = 40;

/// The millions of tokens the SECOND heal of a stage learns on (the retry after a failed check): the stage's own plus `RETRY_TOKENS_M`, rounded up to a budget the heal takes
/// (10 / 20 / 40 / 80), at most 80. The student of the retry is the healed model, and it learns on ALL of these tokens again (40 + 40 = 80; 20 gives 80 too).
pub fn retry_total_m(stage_m: u32) -> u32 {
    plan::HEAL_TOKENS_M.iter().copied().find(|t| *t >= stage_m + RETRY_TOKENS_M).unwrap_or(80)
}

/// What the person asks Auto-cut for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoSpec {
    /// the model to cut (an id of the model list)
    pub model_id: String,
    /// the size to aim for, in billions of parameters
    pub target_b: f64,
    /// heal with the thinking recipe, measure on reasoning text, export with the model's own thinking template (default on; refused for a model that does not think)
    #[serde(default = "yes")]
    pub keep_thinking: bool,
    /// 1 or 2 (default 2)
    #[serde(default = "two")]
    pub stages: u32,
    /// Quick (64 samples) instead of Full (128) for Try first
    #[serde(default)]
    pub quick: bool,
    /// keep the unhealed cut, the teacher's answers and the earlier stage's healed model (default: delete them once the next step has worked)
    #[serde(default)]
    pub keep_intermediate: bool,
    /// the owner confirms that more than 4 whole blocks may be removed
    #[serde(default)]
    pub allow_many_blocks: bool,
    /// the sizes of the export (ids of `export::runs::QUANTS`); Q4_K_M is always made
    #[serde(default = "q4")]
    pub quants: Vec<String>,
    /// put the GGUF files into Unsloth Studio's share when it is set up (docs/UNSLOTH_API_CONTRACT.md, "Publish")
    #[serde(default)]
    pub studio_publish: bool,
    /// before healing, check the thinking of the best `think_gate_n` cuts (by loss) and choose among them by how they think (only with `keep_thinking`; default on)
    #[serde(default = "yes")]
    pub think_gate: bool,
    /// how many of the cuts Try first liked best are checked (1 to `TRY_TOP`; default 2)
    #[serde(default = "two")]
    pub think_gate_n: u32,
    /// also make a file of about this many GB with "Mixed: measured fit to a size" (docs/AUTO_CUT.md, "A target size"), and compare it with the plain sizes next to it; None = not
    #[serde(default)]
    pub target_gb: Option<f64>,
    // ---- speed settings (docs/AUTO_CUT.md, "Speed settings"). Every field is at the recommended recipe by default: an older spec plans and runs exactly as before.
    /// the speed setting the person chose: "thorough" | "balanced" | "fast" | "custom" (`auto_speed.json`); None = not chosen (the recipe)
    #[serde(default)]
    pub preset: Option<String>,
    /// the heal of every stage but the last, in millions of tokens (10 / 20 / 40 / 80); None = the recipe's 40
    #[serde(default)]
    pub stage_a_tokens_m: Option<u32>,
    /// the heal of the last stage; None = the recipe's number for this cut (40, or 80 when more than 30 % is cut)
    #[serde(default)]
    pub last_tokens_m: Option<u32>,
    /// Quick (16 questions) thinking checks; None = follows `quick` (the old behaviour)
    #[serde(default)]
    pub think_quick: Option<bool>,
    /// check the thinking of the unhealed cut of every stage (informational: nothing is decided by it); off = the original is checked after the first heal instead
    #[serde(default = "yes")]
    pub check_cut: bool,
    /// heal again (on the healed model, the original's answers extended) when the thinking check of a healed stage fails
    #[serde(default = "yes")]
    pub retry_heal: bool,
    /// cut one step less and make the stage again after a second failed check (else the run ends `gates_failed`)
    #[serde(default = "yes")]
    pub back_off: bool,
    /// heal every stage on the same sequence length, so the original writes its answers once and the next stage reuses them
    #[serde(default)]
    pub share_teacher_cache: bool,
    /// measure an importance matrix for the small GGUF sizes
    #[serde(default = "yes")]
    pub imatrix: bool,
    /// "I have at most this many hours": the planner drops the least harmful settings until the estimate fits (1 to 336); None = no limit
    #[serde(default)]
    pub max_hours: Option<f64>,
}

impl AutoSpec {
    pub fn new(model_id: &str, target_b: f64) -> Self {
        AutoSpec {
            model_id: model_id.into(), target_b, keep_thinking: true, stages: 2, quick: false, keep_intermediate: false, allow_many_blocks: false, quants: q4(), studio_publish: false, think_gate: true, think_gate_n: 2, target_gb: None,
            preset: None, stage_a_tokens_m: None, last_tokens_m: None, think_quick: None, check_cut: true, retry_heal: true, back_off: true, share_teacher_cache: false, imatrix: true, max_hours: None,
        }
    }

    /// Are the thinking checks the quick ones (16 questions)? Follows `quick` unless the person chose.
    pub fn think_quick(&self) -> bool {
        self.think_quick.unwrap_or(self.quick)
    }

    pub fn validate(&self) -> Result<()> {
        if !(1.0..=1000.0).contains(&self.target_b) || !self.target_b.is_finite() {
            return Err(Error::new("The size to aim for must be between 1 and 1000 billion parameters."));
        }
        if !(1..=2).contains(&self.stages) {
            return Err(Error::new("Auto-cut works in one or two stages."));
        }
        if !(1..=TRY_TOP as u32).contains(&self.think_gate_n) {
            return Err(Error::new(format!("The thinking check before healing looks at 1 to {TRY_TOP} cuts.")));
        }
        for t in [self.stage_a_tokens_m, self.last_tokens_m].into_iter().flatten() {
            if !plan::HEAL_TOKENS_M.contains(&t) {
                return Err(Error::new("Heal on 10, 20, 40 or 80 million tokens."));
            }
        }
        if let Some(h) = self.max_hours {
            if !h.is_finite() || !(1.0..=336.0).contains(&h) {
                return Err(Error::new("The time you have must be between 1 and 336 hours."));
            }
        }
        if let Some(p) = &self.preset {
            if super::auto_speed::preset(p).is_none() {
                return Err(Error::new(format!("Unknown speed setting: {p}.")));
            }
        }
        if let Some(t) = self.target_gb {
            if !t.is_finite() || t <= 0.0 {
                return Err(Error::new("The size of the measured file must be a number of GB above 0."));
            }
        }
        if let Some(q) = self.quants.iter().find(|q| !crate::export::runs::QUANTS.iter().any(|(id, _)| id == q)) {
            return Err(Error::new(format!("Unknown size for the export: {q}.")));
        }
        Ok(())
    }

    /// The sizes the export makes: Q4_K_M first, then the chosen ones, each once.
    pub fn export_quants(&self) -> Vec<String> {
        let mut out = vec!["q4_k_m".to_string()];
        for q in &self.quants {
            if !out.contains(q) {
                out.push(q.clone());
            }
        }
        out
    }

    /// The sizes the export makes when the cut model has `final_params` parameters. Without `target_gb` this is `export_quants()`. With it: Q4_K_M, the chosen ones, the plain sizes just below and
    /// just above the target (the comparison needs a plain file of about the same size and one above it), "mixed_measured" and "f16" (the reference of the comparison, deleted after it
    /// unless `keep_intermediate`).
    pub fn export_quants_for(&self, final_params: u64) -> Vec<String> {
        let mut out = self.export_quants();
        let Some(t) = self.target_gb else { return out };
        let mut sizes: Vec<(f64, &str)> = ["q4_k_m", "q5_k_m", "q6_k", "q8_0"].iter().map(|q| (final_params as f64 * quant_bytes_per_param(q) / 1e9, *q)).collect();
        sizes.sort_by(|a, b| a.0.total_cmp(&b.0));
        let below = sizes.iter().rev().find(|(g, _)| *g <= t).map(|(_, q)| q.to_string());
        let above = sizes.iter().find(|(g, _)| *g > t).map(|(_, q)| q.to_string());
        for q in below.into_iter().chain(above).chain(["mixed_measured".to_string(), "f16".to_string()]) {
            if !out.contains(&q) {
                out.push(q);
            }
        }
        out
    }

    /// Try first: samples of a pass.
    pub fn try_samples(&self) -> u32 {
        if self.quick { plan::TRY_QUICK_SAMPLES } else { plan::TRY_FULL_SAMPLES }
    }
}

/// One way to reach the target: whole blocks removed and the MLP width left.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub blocks: u32,
    /// the MLP width after the cut (the model's own when it is not thinned); None for a model that cannot be thinned
    pub intermediate: Option<u32>,
    pub layers: u32,
    pub params: u64,
    pub params_b: f64,
    /// (params - target) / target, in percent
    pub error_pct: f64,
    /// share of the MLP removed, 0 to 1
    pub width_cut: f64,
    pub on_target: bool,
    /// more than `MAX_BLOCKS_FREE` blocks: needs the owner's confirmation
    pub needs_confirmation: bool,
}

impl Candidate {
    /// the `choices` of a Try-first request
    pub fn try_choice(&self, ins: &Inspect) -> Value {
        let full = full_width(ins);
        json!({"blocks": block_list(ins, self.blocks), "width": self.intermediate.filter(|i| Some(*i) != full)})
    }
}

/// The blocks a Try-first request names for `n` removed blocks: the last whole blocks before the protected one at the end (the measuring decides the real ones; this is
/// only the name of the question). Empty for 0.
fn block_list(ins: &Inspect, n: u32) -> Vec<u32> {
    let end = ins.blocks.saturating_sub(1);
    (end.saturating_sub(n)..end).collect()
}

pub fn full_width(ins: &Inspect) -> Option<u32> {
    ins.width_dims.as_ref().map(|d| d.intermediate).filter(|i| *i > 0 && ins.width_supported)
}

/// The size of the model with `blocks` removed and the MLP at `intermediate` (None: not thinned).
pub fn size_of(ins: &Inspect, blocks: u32, intermediate: Option<u32>) -> u64 {
    match (intermediate, full_width(ins)) {
        (Some(i), Some(_)) => plan::width_params(ins, &plan::kept_layers(ins, blocks), &Width { intermediate: i, lin_v_heads: None, heads: None }),
        _ => plan::params_after(ins, Mode::Prune, blocks),
    }
}

/// The MLP widths of the model, ascending, the original's included.
fn widths(ins: &Inspect) -> Vec<u32> {
    let mut v: Vec<u32> = ins.width_choices.as_ref().map(|c| c.intermediate.clone()).unwrap_or_default();
    if let Some(f) = full_width(ins) {
        if !v.contains(&f) {
            v.push(f);
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// The thinnest MLP width the width-first rule allows (the smallest stop that keeps `WIDTH_FIRST_KEEP` of the original's width).
pub fn width_floor(ins: &Inspect) -> Option<u32> {
    let full = full_width(ins)?;
    let want = full as f64 * WIDTH_FIRST_KEEP;
    widths(ins).into_iter().find(|w| *w as f64 >= want)
}

/// The sentences for a cut that goes beyond what the program does by itself: the MLP kept below the floor (`full` is the original's width, `floor` the thinnest width
/// the rule allows) and more than `MAX_BLOCKS_FREE` blocks removed. Empty when the cut stays inside.
pub fn guardrail_messages(blocks: u32, intermediate: Option<u32>, full: Option<u32>, floor: Option<u32>) -> Vec<String> {
    let mut out = Vec::new();
    if let (Some(i), Some(full), Some(floor)) = (intermediate, full, floor) {
        if i < floor {
            out.push(format!("The MLP is thinned to {i} of {full} ({:.1} % kept); Auto-cut does not go below {floor} ({:.1} % kept).", i as f64 / full as f64 * 100.0, floor as f64 / full as f64 * 100.0));
        }
    }
    if blocks > MAX_BLOCKS_FREE {
        out.push(format!("{blocks} whole blocks are removed; Auto-cut does not remove more than {MAX_BLOCKS_FREE} without asking."));
    }
    out
}

/// The ways a cut goes beyond what the program does by itself (the width-first rule keeps at least `WIDTH_FIRST_KEEP` of the MLP and removes at most `MAX_BLOCKS_FREE` blocks
/// without asking), as plain sentences; empty when it stays inside. Used to warn about a cut made by hand and to explain a chosen one.
pub fn beyond_guardrails(ins: &Inspect, blocks: u32, intermediate: Option<u32>) -> Vec<String> {
    guardrail_messages(blocks, intermediate, full_width(ins), width_floor(ins))
}

/// About how many minutes the thinking check of one cut before healing takes: writing the cut (about 500 MB per second) plus the check (15 minutes for the full suite, 10 for the
/// quick one, per 17 B of the model, at least once). An estimate: the real time depends on the computer.
pub fn think_check_minutes(ins: &Inspect, blocks: u32, intermediate: Option<u32>, quick: bool) -> f64 {
    let params = size_of(ins, blocks, intermediate) as f64;
    let write = params * ins.bytes_per_param.max(1.0) / 500e6 / 60.0;
    let check = think_check_only_minutes(params / 1e9, quick);
    ((write + check) * 10.0).round() / 10.0
}

/// The check alone (no writing of the cut): 15 minutes for the full suite, 10 for the quick one, per 17 B of the model, at least once. The one formula the time estimate
/// (`auto_time`) and the plan's own thinking-check times share.
pub fn think_check_only_minutes(params_b: f64, quick: bool) -> f64 {
    (if quick { 10.0 } else { 15.0 }) * (params_b / 17.0).max(1.0)
}

pub fn think_gate_minutes(ins: &Inspect, cand: &Candidate, quick: bool) -> f64 {
    think_check_minutes(ins, cand.blocks, cand.intermediate, quick)
}

/// The most blocks the model allows (never the first or last block, at most half).
fn block_limit(ins: &Inspect) -> u32 {
    plan::limits(ins, Mode::Prune, 1, 1).1
}

fn candidate(ins: &Inspect, blocks: u32, intermediate: Option<u32>, target_params: f64) -> Candidate {
    let params = size_of(ins, blocks, intermediate);
    let err = (params as f64 - target_params) / target_params;
    let full = full_width(ins);
    let cut = match (intermediate, full) {
        (Some(i), Some(f)) if f > 0 => 1.0 - i as f64 / f as f64,
        _ => 0.0,
    };
    Candidate {
        blocks, intermediate, layers: ins.layers.saturating_sub(blocks * ins.period), params, params_b: (params as f64 / 1e9 * 100.0).round() / 100.0,
        error_pct: (err * 1000.0).round() / 10.0, width_cut: (cut * 1000.0).round() / 1000.0, on_target: err.abs() <= TARGET_TOLERANCE, needs_confirmation: blocks > MAX_BLOCKS_FREE,
    }
}

/// The ways to reach `target_b`, best first: for every number of blocks, the MLP width that comes nearest (never thinner than `width_floor`); the ones on target before
/// the others, fewer blocks before more (width first), then the nearer. A model that cannot be thinned gets block counts only.
pub fn candidates(ins: &Inspect, target_b: f64) -> Vec<Candidate> {
    if !ins.supported || ins.blocks == 0 {
        return Vec::new();
    }
    let target = target_b * 1e9;
    let hi = block_limit(ins);
    let floor = width_floor(ins);
    let ws: Vec<u32> = match floor {
        Some(f) => widths(ins).into_iter().filter(|w| *w >= f).collect(),
        None => Vec::new(),
    };
    let mut out: Vec<Candidate> = (0..=hi).filter_map(|b| {
        if ws.is_empty() {
            return (b > 0).then(|| candidate(ins, b, None, target));
        }
        let best = ws.iter().map(|&w| candidate(ins, b, Some(w), target)).min_by(|a, c| a.error_pct.abs().partial_cmp(&c.error_pct.abs()).unwrap_or(std::cmp::Ordering::Equal).then(c.intermediate.cmp(&a.intermediate)))?;
        Some(best)
    }).collect();
    out.sort_by(|a, b| b.on_target.cmp(&a.on_target).then(a.blocks.cmp(&b.blocks)).then(a.error_pct.abs().partial_cmp(&b.error_pct.abs()).unwrap_or(std::cmp::Ordering::Equal)));
    out
}

/// The cut Auto-cut starts with: the first candidate on target that needs no confirmation (or any, when the owner confirmed).
pub fn pick(cands: &[Candidate], allow_many_blocks: bool) -> Option<Candidate> {
    cands.iter().find(|c| c.on_target && (allow_many_blocks || !c.needs_confirmation)).cloned()
}

/// The cuts "Try first" is asked about (the best `TRY_TOP` on target; the chosen one first).
pub fn try_list(cands: &[Candidate], allow_many_blocks: bool) -> Vec<Candidate> {
    cands.iter().filter(|c| c.on_target && (allow_many_blocks || !c.needs_confirmation)).take(TRY_TOP).cloned().collect()
}

/// "Back off the cut by one step" after a stage failed its checks: one block less while there are blocks, else 1,024 more of the MLP (4 stops). None when nothing is left to
/// give back.
pub fn back_off(ins: &Inspect, blocks: u32, intermediate: Option<u32>) -> Option<(u32, Option<u32>)> {
    if blocks > 0 {
        return Some((blocks - 1, intermediate));
    }
    let (i, full) = (intermediate?, full_width(ins)?);
    if i >= full {
        return None;
    }
    let up = widths(ins).into_iter().filter(|w| *w > i).take(BACK_OFF_MLP as usize / 256).last().unwrap_or(full);
    Some((0, Some(up)))
}

/// One stop of the way to the final cut: the blocks removed so far (from the original) and the MLP width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Step {
    pub blocks: u32,
    pub intermediate: Option<u32>,
}

/// The stops of `stages` stages: the last one is the final cut. Two stages: the depth part first (then the width), or half of the width cut when there are no blocks, or half of
/// the blocks when there is no width; a cut that cannot be divided (one block alone, a width of one stop) is one stage.
pub fn split_stages(ins: &Inspect, cand: &Candidate, stages: u32) -> Vec<Step> {
    let last = Step { blocks: cand.blocks, intermediate: cand.intermediate };
    if stages < 2 {
        return vec![last];
    }
    let full = full_width(ins);
    let thinned = matches!((cand.intermediate, full), (Some(i), Some(f)) if i < f);
    let first = if cand.blocks > 0 && thinned {
        Some(Step { blocks: cand.blocks, intermediate: full })
    } else if cand.blocks > 1 {
        Some(Step { blocks: cand.blocks / 2, intermediate: cand.intermediate })
    } else if cand.blocks == 0 && thinned {
        let (i, f) = (cand.intermediate.unwrap_or(0), full.unwrap_or(0));
        let mid = (i + f) as f64 / 2.0;
        widths(ins).into_iter().filter(|w| *w > i && *w < f).min_by(|a, b| (*a as f64 - mid).abs().partial_cmp(&(*b as f64 - mid).abs()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|w| Step { blocks: 0, intermediate: Some(w) })
    } else {
        None
    };
    match first {
        Some(f) => vec![f, last],
        None => vec![last],
    }
}

/// One stage as it will run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagePlan {
    /// 1-based
    pub index: u32,
    pub of: u32,
    /// the blocks removed from the original after this stage, and in this stage
    pub blocks_after: u32,
    pub blocks_cut: u32,
    /// the MLP width before and after (None: the model is not thinned); `width` is what the cut asks for in this stage (None: only blocks)
    pub intermediate_before: Option<u32>,
    pub intermediate_after: Option<u32>,
    pub width: Option<u32>,
    pub layers_after: u32,
    pub params: u64,
    pub params_b: f64,
    /// the cut's share of the original's parameters, in percent
    pub cut_pct: f64,
    pub tokens_m: u32,
    pub seq: u32,
    pub batch: u32,
    pub rank: u32,
    pub cache_gb: f64,
    /// the healing memory: estimate, "fits" | "tight" | "over" | "unknown", way (cache | online)
    pub heal_gb: f64,
    pub verdict: String,
    pub mode: String,
    pub hours: (f64, f64),
    /// how the original is loaded for its answers ("auto" | "4bit" | "16bit"); the saved answers of two stages are shared only when this and `seq` are the same
    #[serde(default)]
    pub teacher_load: String,
    /// the sequence length is fixed for the heal (`share_teacher_cache`: every stage uses the same one, so the next stage reuses the original's answers)
    #[serde(default)]
    pub seq_pinned: bool,
}

impl StagePlan {
    /// The cut of this stage as a spec for `runs::execute` (the input is the model the previous stage made; the original for stage 1).
    pub fn reshape_spec(&self, source_model_id: &str, reasoning: Option<plan::ReasoningCalib>, samples: u32, width_samples: u32) -> plan::ReshapeSpec {
        let mut v = json!({"source_model_id": source_model_id, "mode": "prune", "blocks": self.blocks_cut, "calib_samples": samples, "width_samples": width_samples, "calib_len": 512});
        if let Some(w) = self.width {
            v["width"] = json!({"intermediate": w});
        }
        let mut spec: plan::ReshapeSpec = serde_json::from_value(v).unwrap_or_else(|_| unreachable!("a spec made of known keys"));
        spec.reasoning = reasoning;
        spec
    }

    /// The heal of this stage: tokens and, for a model that thinks, the thinking recipe.
    pub fn heal_spec(&self, keep_thinking: bool) -> HealSpec {
        let mut v = json!({"tokens_m": self.tokens_m});
        if self.seq_pinned {
            v["seq"] = json!(self.seq);
        }
        if keep_thinking {
            v["recipe"] = json!(plan::THINKING_RECIPE);
        }
        serde_json::from_value(v).unwrap_or_else(|_| unreachable!("a spec made of known keys"))
    }
}

fn gb(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// One stage: the cut from `prev` to `st` (stops of the way, from the original), healed with the numbers of the recipe for a student of that size. `index` is 1-based,
/// `of` the number of stages; the last stage heals with the larger budget.
pub fn stage_for(ins: &Inspect, spec: &AutoSpec, prev: Step, st: Step, index: u32, of: u32, vram_gb: Option<f64>) -> StagePlan {
    stage_with_seq(ins, spec, prev, st, index, of, vram_gb, None)
}

/// The heal tokens of a stage: the recipe's (40 M for a stage that is not the last, the recipe's number for the last, the size's default without thinking) unless the person chose.
fn stage_tokens(spec: &AutoSpec, student_b: f64, total_b: f64, last: bool) -> u32 {
    let chosen = if last { spec.last_tokens_m } else { spec.stage_a_tokens_m };
    chosen.unwrap_or(match (spec.keep_thinking, last) {
        (true, true) => plan::thinking_tokens_m(student_b, total_b),
        (true, false) => STAGE_A_TOKENS_M,
        (false, _) => plan::default_heal_tokens_m(student_b),
    })
}

/// `stage_for` with the sequence length fixed (`pin`, for the stages that share the original's answers).
#[allow(clippy::too_many_arguments)]
fn stage_with_seq(ins: &Inspect, spec: &AutoSpec, prev: Step, st: Step, index: u32, of: u32, vram_gb: Option<f64>, pin: Option<u32>) -> StagePlan {
    let total_b = ins.total_params as f64 / 1e9;
    let params = size_of(ins, st.blocks, st.intermediate);
    let student_b = params as f64 / 1e9;
    let last = index == of;
    let tokens_m = stage_tokens(spec, student_b, total_b, last);
    let dims = match &ins.width_dims {
        Some(d) => HealDims { layers: ins.layers.saturating_sub(st.blocks * ins.period), hidden: d.hidden, vocab: d.vocab },
        None => HealDims::guess(student_b),
    };
    let mut hv = if spec.keep_thinking { json!({"tokens_m": tokens_m, "recipe": plan::THINKING_RECIPE}) } else { json!({"tokens_m": tokens_m}) };
    if let Some(sq) = pin {
        hv["seq"] = json!(sq);
    }
    let hs: HealSpec = serde_json::from_value(hv).unwrap_or_else(|_| unreachable!("a spec made of known keys"));
    let ch = heal_choice(&hs, student_b, vram_gb, &dims);
    let hp: HealPlan = plan::heal_plan_with(student_b, total_b, vram_gb, &dims, &ch.knobs);
    let m = tokens_m as f64 / 10.0;
    StagePlan {
        index, of, blocks_after: st.blocks, blocks_cut: st.blocks.saturating_sub(prev.blocks),
        intermediate_before: prev.intermediate, intermediate_after: st.intermediate,
        width: if st.intermediate != prev.intermediate { st.intermediate } else { None },
        layers_after: ins.layers.saturating_sub(st.blocks * ins.period), params, params_b: (student_b * 100.0).round() / 100.0,
        cut_pct: if total_b > 0.0 { ((1.0 - student_b / total_b) * 1000.0).round() / 10.0 } else { 0.0 },
        tokens_m, seq: ch.knobs.seq, batch: ch.knobs.batch, rank: ch.knobs.rank, cache_gb: gb(tokens_m as f64 * 1e6 * CACHE_BYTES_PER_TOKEN / 1e9),
        heal_gb: hp.est_gb, verdict: hp.verdict.to_string(), mode: hp.mode.to_string(),
        hours: (((hp.hours_per_10m.0 * m) * 10.0).round() / 10.0, ((hp.hours_per_10m.1 * m) * 10.0).round() / 10.0),
        teacher_load: ch.teacher_load.clone().unwrap_or_else(|| hp.teacher_load.to_string()), seq_pinned: pin.is_some(),
    }
}

/// The stage plans of the stops `steps`, the first from the original.
///
/// With `share_teacher_cache` (and more than one stage) every stage heals on the SMALLEST sequence length any of them would use by itself and sends it to the heal
/// (`StagePlan::seq_pinned`): the original's saved answers are kept per sequence length, so the second stage then finds the first stage's rows and writes only the tokens
/// it has more. A shorter sequence than the recipe's cuts the longest reasoning traces of the larger student short; the plan says so.
pub fn stage_plans(ins: &Inspect, spec: &AutoSpec, steps: &[Step], vram_gb: Option<f64>) -> Vec<StagePlan> {
    let of = steps.len() as u32;
    let run = |pin: Option<u32>| -> Vec<StagePlan> {
        let mut prev = Step { blocks: 0, intermediate: full_width(ins) };
        steps.iter().enumerate().map(|(n, st)| {
            let stage = stage_with_seq(ins, spec, prev, *st, n as u32 + 1, of, vram_gb, pin);
            prev = *st;
            stage
        }).collect()
    };
    let plans = run(None);
    match plans.iter().map(|s| s.seq).min() {
        Some(min) if spec.share_teacher_cache && of > 1 => run(Some(min)),
        _ => plans,
    }
}

/// Bytes per parameter of a GGUF size (llama.cpp's own figures, rounded up).
pub fn quant_bytes_per_param(id: &str) -> f64 {
    match id {
        "q4_k_m" => 0.61,
        "q5_k_m" => 0.71,
        "q6_k" => 0.82,
        "q8_0" => 1.07,
        "f16" => 2.0,
        _ => 0.8,
    }
}

/// The sample file the measured fit writes next to the 16-bit GGUF (about 1 % of the weights, written once more by llama-quantize): a safe round number.
pub const MEASURED_SAMPLE_GB: f64 = 2.0;

/// One line of the disk timeline: what is on the disk after the step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskStep {
    pub step: String,
    pub in_use_gb: f64,
}

/// The disk Auto-cut needs: the largest amount at any moment, and what is left at the end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiskPlan {
    pub peak_gb: f64,
    pub peak_step: String,
    pub final_gb: f64,
    pub keep_intermediate: bool,
    pub steps: Vec<DiskStep>,
    /// what the folder has free (None: not known) and whether that is enough (with 5 % to spare)
    pub free_gb: Option<f64>,
    pub enough: Option<bool>,
}

/// The timeline of the disk. The original stays; every stage adds its unhealed cut, the teacher's answers and the healed model; the export adds a 16-bit GGUF and the sizes.
/// Without `keep_intermediate` the cut and the answers of a stage go once the stage passed its checks, and the healed model of the stage before once the next one has passed.
/// The original is counted even when it is on the disk already (it is, as long as the run lasts); a model that is not local needs its download too.
pub fn disk_plan(ins: &Inspect, stages: &[StagePlan], quants: &[String], keep_intermediate: bool, free_gb: Option<f64>) -> DiskPlan {
    let bpp = if ins.bytes_per_param > 0.0 { ins.bytes_per_param } else { 2.0 };
    let model = |params: u64| params as f64 * bpp / 1e9;
    let mut live = model(ins.total_params);
    let mut steps = vec![DiskStep { step: "the original model".into(), in_use_gb: gb(live) }];
    let mut peak = (live, steps[0].step.clone());
    let note = |label: String, live: f64, steps: &mut Vec<DiskStep>, peak: &mut (f64, String)| {
        steps.push(DiskStep { step: label.clone(), in_use_gb: gb(live) });
        if live > peak.0 {
            *peak = (live, label);
        }
    };
    let mut prev_healed = 0.0;
    for s in stages {
        let (cut, cache, healed) = (model(s.params), s.cache_gb, model(s.params));
        live += cut;
        note(format!("stage {}: the cut", s.index), live, &mut steps, &mut peak);
        live += cache;
        note(format!("stage {}: the original's answers", s.index), live, &mut steps, &mut peak);
        live += healed;
        note(format!("stage {}: the healed model", s.index), live, &mut steps, &mut peak);
        if !keep_intermediate {
            live -= cut + cache + prev_healed;
            prev_healed = healed;
            note(format!("stage {}: passed, intermediate files deleted", s.index), live, &mut steps, &mut peak);
        }
    }
    let final_params = stages.last().map(|s| s.params).unwrap_or(ins.total_params);
    let f16 = model(final_params) / bpp * 2.0;
    live += f16;
    note("export: the 16-bit GGUF".into(), live, &mut steps, &mut peak);
    // with a measured file the 16-bit GGUF stays as the reference of the comparison, and the measuring writes a small sample file
    let measured = quants.iter().any(|q| q == "mixed_measured");
    let q: f64 = quants.iter().filter(|id| !(measured && id.as_str() == "f16")).map(|id| final_params as f64 * quant_bytes_per_param(id) / 1e9).sum();
    if measured {
        live += MEASURED_SAMPLE_GB;
        note("export: the sample file of the measured fit".into(), live, &mut steps, &mut peak);
        live -= MEASURED_SAMPLE_GB;
    }
    live += q;
    note("export: the sizes".into(), live, &mut steps, &mut peak);
    if !keep_intermediate {
        live -= f16;
        note(if measured { "the comparison: the 16-bit GGUF deleted" } else { "export: the 16-bit GGUF deleted" }.into(), live, &mut steps, &mut peak);
    }
    let download = if ins.local { 0.0 } else if ins.source_bytes > 0 { ins.source_bytes as f64 / 1e9 } else { model(ins.total_params) };
    let (peak_gb, peak_step) = (peak.0 + download, peak.1);
    DiskPlan {
        peak_gb: gb(peak_gb), peak_step, final_gb: gb(live), keep_intermediate, steps, free_gb: free_gb.map(gb), enough: free_gb.map(|f| f >= peak_gb * 1.05),
    }
}

/// The thinking check before healing keeps one final cut on the disk while the next one is made (two at most; with more than one stage the best one is deleted at the end):
/// a line after the original in the timeline, and a new peak when it is the largest.
fn add_think_gate_disk(ins: &Inspect, disk: &mut DiskPlan, chosen: &Candidate, single_stage: bool) {
    let bpp = if ins.bytes_per_param > 0.0 { ins.bytes_per_param } else { 2.0 };
    let cut = chosen.params as f64 * bpp / 1e9;
    let base = disk.steps.first().map(|s| s.in_use_gb).unwrap_or(0.0);
    let in_use = gb(base + cut * if single_stage { 2.0 } else { 1.0 });
    let label = if single_stage { "the thinking check: the best cut so far and the next one" } else { "the thinking check: one final cut at a time" };
    disk.steps.insert(1, DiskStep { step: label.into(), in_use_gb: in_use });
    if in_use > disk.peak_gb {
        disk.peak_gb = in_use;
        disk.peak_step = label.into();
        disk.enough = disk.free_gb.map(|f| f >= in_use * 1.05);
    }
}

/// What Auto-cut says before it starts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoPlan {
    /// nothing in `blockers`
    pub ok: bool,
    pub blockers: Vec<String>,
    pub warnings: Vec<String>,
    pub original_b: f64,
    pub target_b: f64,
    pub candidates: Vec<Candidate>,
    pub chosen: Option<Candidate>,
    /// the cuts Try first is asked about
    pub try_list: Vec<Candidate>,
    pub stages: Vec<StagePlan>,
    pub disk: Option<DiskPlan>,
    /// how the original is loaded for measuring (16bit | 4bit | offload) on this card
    pub measure_load: String,
    /// the healing time of all stages (low, high) in hours; the other steps are not counted
    pub heal_hours: (f64, f64),
    /// about how many minutes the thinking check before healing adds (None: it is off); an estimate
    #[serde(default)]
    pub think_gate_minutes: Option<f64>,
    /// with a target size: about how many minutes measuring the fit takes and how many the comparison takes; (None: no target). Guesses.
    #[serde(default)]
    pub measured_minutes: Option<(f64, f64)>,
    /// the time of the WHOLE run, step by step, before it starts (`auto_time`); None while the plan has blockers
    #[serde(default)]
    pub time: Option<TimeEstimate>,
}

/// The whole plan. `vram_gb` and `free_gb` (the disk the run writes to) may be unknown.
pub fn plan(ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, free_gb: Option<f64>) -> AutoPlan {
    let original_b = ins.total_params as f64 / 1e9;
    let mut out = AutoPlan {
        ok: false, blockers: Vec::new(), warnings: Vec::new(), original_b: (original_b * 100.0).round() / 100.0, target_b: spec.target_b, candidates: Vec::new(), chosen: None,
        try_list: Vec::new(), stages: Vec::new(), disk: None, measure_load: plan::scoring_load(ins.total_params, ins.bytes_per_param, vram_gb).to_string(), heal_hours: (0.0, 0.0), think_gate_minutes: None, measured_minutes: None, time: None,
    };
    if let Err(e) = spec.validate() {
        out.blockers.push(e.0);
        return out;
    }
    if !ins.supported {
        out.blockers.push(ins.why.clone().unwrap_or_else(|| "This model cannot be cut by Auto-cut: only Qwen3.5 / Qwen3.8 and plain Llama / Qwen / Mistral style models can be slimmed.".into()));
        return out;
    }
    if spec.keep_thinking && ins.model_type != "qwen3_5" {
        out.blockers.push("Keep thinking is only for Qwen3.5 / Qwen3.8 models: this model's chat template does not write the reasoning. Switch it off to cut this model.".into());
    }
    if spec.target_b >= original_b * 0.97 {
        out.blockers.push(format!("The model is {original_b:.1} B already: choose a size at least 3 % smaller."));
    }
    if !out.blockers.is_empty() {
        return out;
    }
    if spec.keep_thinking && spec.target_b < MIN_TARGET_B_THINKING {
        out.warnings.push(format!("A model that thinks, cut to {:.1} B, will not reason like the original: expect clearly weaker answers below {MIN_TARGET_B_THINKING:.0} B. The checks after every stage show how much is left.", spec.target_b));
    }
    out.candidates = candidates(ins, spec.target_b);
    out.try_list = try_list(&out.candidates, spec.allow_many_blocks);
    out.chosen = pick(&out.candidates, spec.allow_many_blocks);
    let Some(chosen) = out.chosen.clone() else {
        let gentle = out.candidates.iter().filter(|c| !c.needs_confirmation).map(|c| c.params_b).fold(f64::INFINITY, f64::min);
        let any_on_target = out.candidates.iter().any(|c| c.on_target);
        out.blockers.push(if any_on_target {
            format!("Reaching {:.1} B needs more than {MAX_BLOCKS_FREE} whole blocks removed. Confirm that (it costs more than thinning the MLP does) or choose a larger size.", spec.target_b)
        } else if gentle.is_finite() && spec.target_b > gentle {
            format!("No cut of this model comes within {:.0} % of {:.1} B. The nearest are listed; try a size they reach.", TARGET_TOLERANCE * 100.0, spec.target_b)
        } else {
            format!("{:.1} B is smaller than the cut that is allowed without confirmation ({:.1} B: the MLP thinned by 38 % and {MAX_BLOCKS_FREE} blocks removed). Choose a larger size.", spec.target_b, gentle)
        });
        return out;
    };
    if chosen.blocks > 0 && chosen.blocks >= ins.blocks / 4 {
        out.warnings.push(format!("{} of the model's {} blocks are removed: every block has one of the full-attention layers, so reasoning suffers first.", chosen.blocks, ins.blocks));
    }
    let steps = split_stages(ins, &chosen, spec.stages);
    if spec.stages == 2 && steps.len() == 1 {
        out.warnings.push("This cut cannot be divided into two stages: it is done in one.".into());
    }
    out.stages = stage_plans(ins, spec, &steps, vram_gb);
    let natural: Vec<u32> = if spec.share_teacher_cache { stage_plans(ins, &AutoSpec { share_teacher_cache: false, ..spec.clone() }, &steps, vram_gb).iter().map(|s| s.seq).collect() } else { out.stages.iter().map(|s| s.seq).collect() };
    out.warnings.extend(super::auto_speed::speed_warnings(spec, &out.stages, &natural, original_b));
    for s in &out.stages {
        if s.verdict == "over" {
            out.blockers.push(format!("Healing stage {} ({:.1} B) needs about {:.0} GB on the card (an estimate); this card has {:.0} GB.", s.index, s.params_b, s.heal_gb, vram_gb.unwrap_or(0.0)));
        } else if s.verdict == "tight" {
            out.warnings.push(format!("Healing stage {} is tight on this card (about {:.0} GB of {:.0} GB); the program steps down to safer settings by itself if it runs out of memory.", s.index, s.heal_gb, vram_gb.unwrap_or(0.0)));
        }
    }
    out.heal_hours = out.stages.iter().fold((0.0, 0.0), |a, s| (a.0 + s.hours.0, a.1 + s.hours.1));
    let final_params = out.stages.last().map(|s| s.params).unwrap_or(ins.total_params);
    let export = spec.export_quants_for(final_params);
    let mut disk = disk_plan(ins, &out.stages, &export, spec.keep_intermediate, free_gb);
    if let Some(t) = spec.target_gb {
        let (final_b, files) = (final_params as f64 / 1e9, export.iter().filter(|q| q.as_str() != "f16").count() as f64);
        let min_gb = final_params as f64 * 0.30 / 1e9;
        if t < min_gb {
            out.blockers.push(format!("{t:.1} GB is smaller than the cut model can be made at all (about {min_gb:.1} GB with 3-bit weights). Ask for more."));
        }
        // guesses, like the other times shown here: measuring about 0.25 minutes per billion parameters plus 2, the comparison 4 minutes plus 3 per file
        out.measured_minutes = Some((((2.0 + 0.25 * final_b) * 10.0).round() / 10.0, 4.0 + 3.0 * files));
    }
    if spec.keep_thinking && spec.think_gate && !out.try_list.is_empty() {
        let n = (spec.think_gate_n as usize).min(out.try_list.len());
        let per: f64 = out.try_list.iter().take(n).map(|c| think_gate_minutes(ins, c, spec.think_quick())).sum();
        let baseline = (if spec.think_quick() { 10.0 } else { 15.0 }) * (ins.total_params as f64 / 1e9 / 17.0).max(1.0);
        out.think_gate_minutes = Some(((per + baseline) * 10.0).round() / 10.0);
        add_think_gate_disk(ins, &mut disk, &chosen, out.stages.len() == 1);
    }
    if disk.enough == Some(false) {
        out.blockers.push(format!("The run needs up to {:.0} GB of disk ({}); {:.0} GB are free. Free some space{}.", disk.peak_gb, disk.peak_step, disk.free_gb.unwrap_or(0.0),
                                  if spec.keep_intermediate { " or switch \"keep the intermediate files\" off" } else { "" }));
    }
    out.disk = Some(disk);
    out.ok = out.blockers.is_empty();
    if out.ok {
        out.time = Some(time_estimate(ins, spec, &out, &History::default()));
    }
    out
}

/// The plan with the time estimate made from this computer's own earlier jobs (`history`) where they say something: what the window shows and what a run saves.
pub fn with_history(mut plan: AutoPlan, ins: &Inspect, spec: &AutoSpec, history: &History) -> AutoPlan {
    if plan.ok {
        plan.time = Some(time_estimate(ins, spec, &plan, history));
    }
    plan
}
