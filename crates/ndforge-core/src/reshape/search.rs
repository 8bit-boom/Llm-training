//! "Find the best cut for me" (docs/TRY_FIRST.md, "The automatic search"): which blocks to take out of a model for a target size, found by measuring the
//! text-prediction error of many cuts in memory (the worker job `reshape_try`, key `want.search`, one model load; the rounds are planned by the worker's
//! `reshapesearch.py`). This file is everything around that: the target as groups of cuts that reach it, the time and passes of the plan, the ranking of what was measured,
//! the recommendation with its confidence, the confirmation of a close call with more text, and the thinking check of the best two (which writes one cut at a time).
//!
//! WHAT IT CAN AND CANNOT KNOW. It measures how well each cut predicts the project's own text (with reasoning text when the model keeps thinking), with the cut imitated
//! by hooks (the torch tests prove this equals the written cut). It does NOT measure the quality after healing. For a model that thinks, a low error does not prove it can
//! still think: only the thinking check decides, and a result without it says "thinking not checked". With about 70 candidates the winner is partly luck; the halves of the
//! text, the noise figure and the confirmation with more text show how close the call is.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::auto::{CutRef, Effects, Made, ModelRef, Out};
use super::auto_plan::{self, AutoSpec, Candidate};
use super::auto_time;
use super::plan::{self, Inspect, ReasoningCalib};
use super::runs::{models_dir, now, read_json, write_json, TrySpec, TRY_MAX_PASSES, TRY_SEARCH_MAX_GROUPS};
use super::thinkcheck;
use crate::error::{Error, Result};
use crate::project::Project;

/// Samples of the measuring of the neurons that the search's widths use (the same as Auto-cut's).
const WIDTH_SAMPLES: u32 = 256;
const CALIB_SAMPLES: u32 = 128;
/// The samples of the confirmation of a close call.
pub const CONFIRM_SAMPLES: u32 = 128;
/// The trade-off mode looks at these shares of the original's size.
pub const TRADEOFF_SHARES: [f64; 5] = [0.90, 0.85, 0.80, 0.76, 0.70];
/// A cut with a relative increase of the error up to this (the second colour edge of Try first) is "mild" for the trade-off rule.
pub const MILD_REL: f64 = 0.10;
/// The most rows the answer lists per group.
const ROWS_PER_GROUP: usize = 12;

fn yes() -> bool {
    true
}
fn default_budget() -> u32 {
    20
}
fn default_axes() -> String {
    "both".into()
}
fn default_think() -> String {
    "top2".into()
}

/// What the person wants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    /// billions of parameters
    B(f64),
    /// the size of the Q4_K_M file in gigabytes
    Gb(f64),
    /// the share of the original's parameters that stays, in percent
    KeptPct(f64),
    /// the smallest size whose best cut still loses little
    Tradeoff,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchSpec {
    pub model_id: String,
    pub target: Target,
    /// the most minutes the measuring may take (5 to 240)
    #[serde(default = "default_budget")]
    pub budget_min: u32,
    /// measure on reasoning text and let the thinking check run
    #[serde(default = "yes")]
    pub keep_thinking: bool,
    /// 64 or 128; None = by the size of the model
    #[serde(default)]
    pub samples: Option<u32>,
    /// "both" | "depth" | "width"
    #[serde(default = "default_axes")]
    pub axes: String,
    #[serde(default)]
    pub allow_many_blocks: bool,
    /// Expert: also widths below the width-first floor
    #[serde(default)]
    pub beyond_floor: bool,
    /// "off" | "top2"
    #[serde(default = "default_think")]
    pub think_check: String,
    /// ask a close call again with more text
    #[serde(default = "yes")]
    pub confirm_full: bool,
    /// Expert: also every neighbouring run of every block count
    #[serde(default)]
    pub thorough: bool,
    #[serde(default)]
    pub quick_think: Option<bool>,
}

impl SearchSpec {
    pub fn new(model_id: &str, target: Target) -> Self {
        SearchSpec { model_id: model_id.into(), target, budget_min: 20, keep_thinking: true, samples: None, axes: "both".into(), allow_many_blocks: false, beyond_floor: false,
                     think_check: "top2".into(), confirm_full: true, thorough: false, quick_think: None }
    }

    pub fn validate(&self) -> Result<()> {
        if !(5..=240).contains(&self.budget_min) {
            return Err(Error::new("The search may take 5 to 240 minutes."));
        }
        if !["both", "depth", "width"].contains(&self.axes.as_str()) {
            return Err(Error::new("The search looks at depth, width or both."));
        }
        if !["off", "top2"].contains(&self.think_check.as_str()) {
            return Err(Error::new("The thinking check of the search is off or for the best two."));
        }
        if self.samples.is_some_and(|s| !(8..=4096).contains(&s)) {
            return Err(Error::new("The search measures on 8 to 4096 samples."));
        }
        match self.target {
            Target::B(b) if !(1.0..=2000.0).contains(&b) => Err(Error::new("The size to aim for must be between 1 and 2000 billion parameters.")),
            Target::Gb(g) if !(0.5..=2000.0).contains(&g) => Err(Error::new("The file size to aim for must be between 0.5 and 2000 GB.")),
            Target::KeptPct(p) if !(20.0..=98.0).contains(&p) => Err(Error::new("Keep between 20 and 98 percent of the model.")),
            _ => Ok(()),
        }
    }

    fn samples_for(&self, total_b: f64) -> u32 {
        self.samples.unwrap_or_else(|| plan::default_try_samples(total_b))
    }
}

// ------------------------------------------------------------------------------------------------------------------ the target as cuts

/// One way to reach a target size: `k` whole blocks out and the MLP at `width` (None: not thinned).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupCut {
    pub k: u32,
    pub width: Option<u32>,
    pub params: u64,
    pub on_target: bool,
}

/// A target size and the cuts that reach it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Group {
    pub label: String,
    pub target_b: f64,
    pub cuts: Vec<GroupCut>,
}

fn target_sizes(ins: &Inspect, spec: &SearchSpec) -> Vec<(String, f64)> {
    let total_b = ins.total_params as f64 / 1e9;
    match &spec.target {
        Target::B(b) => vec![(format!("{b:.1} B"), *b)],
        Target::Gb(g) => {
            let b = g / auto_plan::quant_bytes_per_param("q4_k_m");
            vec![(format!("{g:.1} GB (about {b:.1} B)"), b)]
        }
        Target::KeptPct(p) => vec![(format!("{p:.0} % kept"), total_b * p / 100.0)],
        Target::Tradeoff => TRADEOFF_SHARES.iter().map(|s| (format!("{:.0} % kept", s * 100.0), total_b * s))
            .filter(|(_, b)| !spec.keep_thinking || *b >= auto_plan::MIN_TARGET_B_THINKING).collect(),
    }
}

/// The cuts that reach one target size, inside the guardrails (the width-first floor and at most `MAX_BLOCKS_FREE` blocks, unless the person allowed more).
fn cuts_for(ins: &Inspect, spec: &SearchSpec, target_b: f64) -> Vec<GroupCut> {
    let full = auto_plan::full_width(ins);
    let hi = auto_plan::max_blocks(ins);
    let mut cands: Vec<Candidate> = match spec.axes.as_str() {
        "depth" => (1..=hi).map(|k| auto_plan::candidate_at(ins, k, full, target_b)).collect(),
        "width" => (if spec.beyond_floor { auto_plan::candidates_free(ins, target_b) } else { auto_plan::candidates(ins, target_b) }).into_iter().filter(|c| c.blocks == 0).collect(),
        _ => if spec.beyond_floor { auto_plan::candidates_free(ins, target_b) } else { auto_plan::candidates(ins, target_b) },
    };
    cands.retain(|c| c.on_target && (spec.allow_many_blocks || !c.needs_confirmation));
    cands.sort_by(|a, b| a.blocks.cmp(&b.blocks));
    cands.into_iter().map(|c| GroupCut { k: c.blocks, width: c.intermediate.filter(|i| Some(*i) != full), params: c.params, on_target: c.on_target }).collect()
}

/// The groups of cuts of a request: one for a size, five for the trade-off. A size that no cut reaches inside the guardrails is left out.
pub fn groups(ins: &Inspect, spec: &SearchSpec) -> Vec<Group> {
    target_sizes(ins, spec).into_iter().map(|(label, b)| Group { label, target_b: (b * 100.0).round() / 100.0, cuts: cuts_for(ins, spec, b) }).filter(|g| !g.cuts.is_empty()).collect()
}

// ------------------------------------------------------------------------------------------------------------------ the plan

/// The rounds the worker will run and about how many passes each takes (the worker's own plan, `plan_only`, is exact once the model's answers are known; this one mirrors its
/// sizes: `reshapesearch.round_plan`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundEst {
    pub name: String,
    pub label: String,
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchPlan {
    pub groups: Vec<Group>,
    pub samples: u32,
    pub load: String,
    pub rounds: Vec<RoundEst>,
    pub passes: u32,
    /// minutes of one pass (low, high)
    pub pass_min: (f64, f64),
    /// minutes of loading the model (low, high)
    pub load_min: (f64, f64),
    /// the whole search at full length, minutes (low, high), and the budget asked for
    pub total_min: (f64, f64),
    pub budget_min: u32,
    /// how many passes fit the budget (at the high figure of a pass)
    pub fit_passes: u32,
    /// the rounds that fit whole (at the high figure)
    pub fit_rounds: u32,
    pub complete: bool,
    pub notes: Vec<String>,
}

/// The passes of the rounds for these groups (mirrors the worker's planner).
fn rounds_est(ins: &Inspect, groups: &[Group], thorough: bool) -> Vec<RoundEst> {
    let movable = (ins.blocks.saturating_sub(2)) as usize;
    let ks: std::collections::BTreeSet<u32> = groups.iter().flat_map(|g| g.cuts.iter().map(|c| c.k)).filter(|k| *k >= 1 && *k as usize <= movable).collect();
    let ks2: Vec<u32> = ks.iter().copied().filter(|k| *k >= 2).collect();
    let widths: std::collections::BTreeSet<u32> = groups.iter().flat_map(|g| g.cuts.iter().filter_map(|c| c.width)).collect();
    let cuts = groups.iter().flat_map(|g| g.cuts.iter()).filter(|c| c.k >= 1 && c.width.is_some()).count() as u32;
    let mut sets = ks2.len() as u32 * 9;
    if thorough {
        sets += ks2.iter().map(|k| (movable as u32 + 1).saturating_sub(*k)).sum::<u32>();
    }
    let r = |n: &str, l: &str, c: u32| RoundEst { name: n.into(), label: l.into(), count: c };
    vec![r("singles", "each block on its own", 1 + movable as u32), r("width_refs", "the widths", widths.len() as u32), r("sets_k", "sets of blocks", sets),
         r("combined", "the real cuts (blocks and width)", cuts * 3), r("refine1", "swapping one block", 6), r("refine2", "swapping one block again", 6)]
}

/// The plan of a search before any job runs: the cuts, the passes, the time (from this computer's own saved jobs where it has them) and what fits the budget. Pure.
pub fn plan(ins: &Inspect, spec: &SearchSpec, vram_gb: Option<f64>, history: &crate::timing::History) -> Result<SearchPlan> {
    spec.validate()?;
    if !ins.supported {
        return Err(Error::new("This model cannot be slimmed by the program."));
    }
    let gs = groups(ins, spec);
    if gs.is_empty() {
        return Err(Error::new("No cut reaches that size inside the safe limits (the width-first rule and at most 4 blocks). Choose another size, or allow more in Expert mode."));
    }
    let total_b = ins.total_params as f64 / 1e9;
    let samples = spec.samples_for(total_b);
    let load = plan::scoring_load(ins.total_params, ins.bytes_per_param, vram_gb).to_string();
    let rounds = rounds_est(ins, &gs, spec.thorough);
    let passes: u32 = rounds.iter().map(|r| r.count).sum();
    let pass_min = auto_time::try_pass_minutes(history, total_b, samples, &load);
    let gb = ins.total_params as f64 * if ins.bytes_per_param > 0.0 { ins.bytes_per_param } else { 2.0 } / 1e9;
    let load_min = auto_time::load_minutes(history, gb, &load);
    let total_min = (load_min.0 + pass_min.0 * passes as f64, load_min.1 + pass_min.1 * passes as f64);
    let left = (spec.budget_min as f64 - load_min.1).max(0.0);
    let fit_passes = if pass_min.1 > 0.0 { ((left / pass_min.1).floor() as u32).min(TRY_MAX_PASSES as u32) } else { 0 };
    let (mut used, mut fit_rounds) = (0u32, 0u32);
    for r in &rounds {
        if used + r.count <= fit_passes {
            used += r.count;
            fit_rounds += 1;
        } else {
            break;
        }
    }
    let mut notes = Vec::new();
    if fit_passes < passes {
        notes.push(format!("About {passes} measurements are planned; {} fit the {} minutes you gave it. The search stops when the time is used and keeps what it has measured; the last rounds (swapping blocks) are the first to be cut.", fit_passes.min(passes), spec.budget_min));
    }
    if spec.keep_thinking {
        notes.push("A low error on text does not prove that a model can still think. The thinking check decides; until it has run the answer says so.".into());
    }
    Ok(SearchPlan { groups: gs, samples, load, rounds, passes, pass_min, load_min, total_min, budget_min: spec.budget_min, fit_passes, fit_rounds, complete: fit_passes >= passes, notes })
}

/// The request of the search as the worker job takes it.
pub fn try_spec(ins: &Inspect, spec: &SearchSpec, p: &SearchPlan, vram_gb: Option<f64>) -> Result<TrySpec> {
    let _ = vram_gb;
    let search = json!({"groups": p.groups, "budget_min": spec.budget_min, "thorough": spec.thorough, "load_s": (p.load_min.1 * 60.0).round(), "max_passes": TRY_MAX_PASSES});
    let v = json!({"model_id": spec.model_id, "samples": p.samples, "len": 512, "load": p.load, "measure_samples": WIDTH_SAMPLES, "params_b": ins.total_params as f64 / 1e9,
                   "reasoning": spec.keep_thinking.then(ReasoningCalib::default), "search": search});
    let t: TrySpec = serde_json::from_value(v).map_err(|e| Error::new(format!("The search request is not valid: {e}")))?;
    t.validate()?;
    if p.groups.len() > TRY_SEARCH_MAX_GROUPS {
        return Err(Error::new("Too many target sizes."));
    }
    Ok(t)
}

// ------------------------------------------------------------------------------------------------------------------ the ranking

/// "d=3,7|w=8192" -> ([3, 7], Some(8192))
pub fn parse_key(key: &str) -> (Vec<u32>, Option<u32>) {
    let (d, w) = key.split_once('|').unwrap_or((key, "w="));
    let blocks = d.trim_start_matches("d=").split(',').filter_map(|x| x.parse().ok()).collect();
    (blocks, w.trim_start_matches("w=").parse().ok())
}

/// One measured cut as the page shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRow {
    pub key: String,
    pub group: usize,
    pub blocks: Vec<u32>,
    pub k: u32,
    pub intermediate: Option<u32>,
    pub params_b: f64,
    pub loss: f64,
    /// loss / the original's loss - 1
    pub rel: f64,
    pub loss_h1: Option<f64>,
    pub loss_h2: Option<f64>,
    /// |h1 - h2|: how much the two halves of the text disagree about this cut
    pub noise: Option<f64>,
    /// the ways this cut goes beyond what Auto-cut does by itself
    pub guardrails: Vec<String>,
    /// null until the thinking check ran: `{verdict, score, main}`
    pub think: Option<Value>,
}

/// The rows of these keys (a group's) from the answer of the worker, in the order of the keys; a key without a measured loss is left out.
pub fn rows_for_keys(analysis: &Value, ins: &Inspect, group: usize, keys: &[String]) -> Vec<SearchRow> {
    let (Some(base), Some(rows)) = (analysis["base"]["loss"].as_f64(), analysis["search"]["rows"].as_object()) else { return Vec::new() };
    let full = auto_plan::full_width(ins);
    keys.iter().filter_map(|key| {
        let r = rows.get(key)?;
        let loss = r["loss"].as_f64()?;
        let (blocks, width) = parse_key(key);
        let inter = width.or(full);
        let (h1, h2) = (r["loss_h1"].as_f64(), r["loss_h2"].as_f64());
        let k = blocks.len() as u32;
        Some(SearchRow {
            key: key.clone(), group, k, params_b: (auto_plan::size_of(ins, k, inter) as f64 / 1e9 * 100.0).round() / 100.0, loss, rel: loss / base - 1.0, loss_h1: h1, loss_h2: h2,
            noise: h1.zip(h2).map(|(a, b)| (a - b).abs()), guardrails: auto_plan::beyond_guardrails(ins, k, width.or(full)), blocks, intermediate: inter, think: None,
        })
    }).collect()
}

/// The measured cuts of the answer of the worker, per group best first (at most `ROWS_PER_GROUP` each). `analysis` is the `analysis` of the job's result.
pub fn rank(analysis: &Value, ins: &Inspect) -> Vec<SearchRow> {
    let Some(ranking) = analysis["search"]["ranking"].as_object() else { return Vec::new() };
    let mut out = Vec::new();
    let mut groups: Vec<(&String, &Value)> = ranking.iter().collect();
    groups.sort_by_key(|(g, _)| g.parse::<usize>().unwrap_or(0));
    for (g, keys) in groups {
        let keys: Vec<String> = keys.as_array().map(|a| a.iter().filter_map(|k| k.as_str().map(String::from)).take(ROWS_PER_GROUP).collect()).unwrap_or_default();
        out.extend(rows_for_keys(analysis, ins, g.parse::<usize>().unwrap_or(0), &keys));
    }
    out
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    pub key: String,
    pub group: usize,
    /// plain words
    pub reason: String,
    /// "clear" | "close_call" | "uncertain"
    pub confidence: String,
    pub thinking_checked: bool,
    /// the next best keys of the same group
    pub alternatives: Vec<String>,
}

fn argmin_by(rows: &[&SearchRow], f: impl Fn(&SearchRow) -> Option<f64>) -> Option<usize> {
    rows.iter().enumerate().filter_map(|(i, r)| f(r).map(|v| (i, v))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))).map(|x| x.0)
}

/// How sure the call is for `rows` (one group, best first): "uncertain" when the two halves of the text name different winners or there is nothing to compare;
/// "close_call" when the best two differ by less than the larger of their noise figures; else "clear".
pub fn confidence(rows: &[&SearchRow]) -> &'static str {
    if rows.is_empty() {
        return "uncertain";
    }
    let (h1, h2) = (argmin_by(rows, |r| r.loss_h1), argmin_by(rows, |r| r.loss_h2));
    if h1.is_some() && h2.is_some() && h1 != h2 {
        return "uncertain";
    }
    if rows.len() < 2 {
        return "uncertain";
    }
    let gap = (rows[1].loss - rows[0].loss).abs();
    let noise = rows[0].noise.unwrap_or(0.0).max(rows[1].noise.unwrap_or(0.0));
    if gap < noise { "close_call" } else { "clear" }
}

fn describe(r: &SearchRow) -> String {
    let blocks = if r.blocks.is_empty() { "no blocks".to_string() } else if r.blocks.len() == 1 { format!("block {}", r.blocks[0]) } else { format!("blocks {}", r.blocks.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ")) };
    format!("{blocks} out, MLP {}, {:.1} B", r.intermediate.map(|i| i.to_string()).unwrap_or_else(|| "unchanged".into()), r.params_b)
}

/// The recommendation: for one size the lowest error (after the thinking check, when it ran: `rank_thinking` decides, the error breaks a tie); for the trade-off the smallest size
/// whose best cut stays mild (`MILD_REL`), a rule of thumb. `rows` come from `rank` (with `think` filled where it was checked).
pub fn recommend(rows: &[SearchRow], spec: &SearchSpec) -> Option<Recommendation> {
    let mut by_group: std::collections::BTreeMap<usize, Vec<&SearchRow>> = std::collections::BTreeMap::new();
    for r in rows {
        by_group.entry(r.group).or_default().push(r);
    }
    for v in by_group.values_mut() {
        v.sort_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal).then(a.key.cmp(&b.key)));
    }
    let pick = |g: usize| -> Option<(&SearchRow, bool, &'static str)> {
        let v = by_group.get(&g)?;
        let checked: Vec<&&SearchRow> = v.iter().filter(|r| r.think.as_ref().and_then(|t| t["score"].as_f64()).is_some()).collect();
        if checked.len() >= 2 {
            let pairs: Vec<(Option<f64>, Option<f64>)> = checked.iter().map(|r| (Some(r.loss), r.think.as_ref().and_then(|t| t["score"].as_f64()))).collect();
            let i = thinkcheck::rank_thinking(&pairs)?;
            let best = **checked.get(i)?;
            let conf = if std::ptr::eq(best, v[0]) { confidence(v) } else { "clear" };
            return Some((best, true, conf));
        }
        v.first().map(|r| (*r, false, confidence(v)))
    };
    if spec.target == Target::Tradeoff {
        // the smallest size first: the groups are in the order of the shares, big to small; the last one that is still mild wins
        let mut best: Option<(&SearchRow, bool, &'static str)> = None;
        for g in by_group.keys() {
            if let Some(c) = pick(*g) {
                if c.0.rel <= MILD_REL && c.0.guardrails.is_empty() {
                    best = Some(c);
                }
            }
        }
        let (r, checked, conf) = best.or_else(|| by_group.keys().next().and_then(|g| pick(*g)))?;
        let mild = r.rel <= MILD_REL;
        let reason = if mild {
            format!("A rule of thumb, not a measurement of quality: {} is the smallest size whose best cut raises the text-prediction error by {:.1} % (up to {:.0} % is called mild). {}", describe(r), r.rel * 100.0, MILD_REL * 100.0, if checked { "Its thinking was checked." } else { "Thinking not checked." })
        } else {
            format!("No size stays mild; the least costly is {} (error +{:.1} %). {}", describe(r), r.rel * 100.0, if checked { "Its thinking was checked." } else { "Thinking not checked." })
        };
        let alt = by_group.get(&r.group).map(|v| v.iter().filter(|x| x.key != r.key).take(3).map(|x| x.key.clone()).collect()).unwrap_or_default();
        return Some(Recommendation { key: r.key.clone(), group: r.group, reason, confidence: conf.into(), thinking_checked: checked, alternatives: alt });
    }
    let g = *by_group.keys().next()?;
    let (r, checked, conf) = pick(g)?;
    let v = &by_group[&g];
    let mut reason = format!("{}: the text-prediction error rises by {:.1} %.", describe(r), r.rel * 100.0);
    if checked {
        reason.push_str(" Its thinking was checked and is the best of the cuts that were checked.");
    } else if spec.keep_thinking {
        reason.push_str(" Thinking not checked: a low error does not prove that the model can still think.");
    }
    if conf == "close_call" {
        reason.push_str(" The second best is within the noise of the two halves of the text; either may be as good.");
    } else if conf == "uncertain" {
        reason.push_str(" The two halves of the text disagree about the best cut; treat this as a guess.");
    }
    if !r.guardrails.is_empty() {
        reason.push_str(&format!(" It goes beyond what Auto-cut does by itself: {}", r.guardrails.join(" ")));
    }
    let alt = v.iter().filter(|x| x.key != r.key).take(3).map(|x| x.key.clone()).collect();
    Some(Recommendation { key: r.key.clone(), group: g, reason, confidence: conf.into(), thinking_checked: checked, alternatives: alt })
}

// ------------------------------------------------------------------------------------------------------------------ running

pub fn search_dir(p: &Project, id: &str) -> std::path::PathBuf {
    models_dir(p).join("_search").join(id)
}

fn state_file(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join("search.json")
}

fn save(dir: &std::path::Path, st: &mut Value) -> Result<()> {
    st["updated"] = json!(now());
    write_json(&state_file(dir), st)
}

/// The newest search of the project (`search.json`), or None.
pub fn latest(p: &Project) -> Option<Value> {
    let mut all: Vec<(String, Value)> = std::fs::read_dir(models_dir(p).join("_search")).ok()?.flatten().filter_map(|e| Some((e.file_name().to_string_lossy().into_owned(), read_json(&state_file(&e.path()))?))).collect();
    all.sort_by(|a, b| b.0.cmp(&a.0));
    all.into_iter().next().map(|x| x.1)
}

fn new_id(model: &ModelRef) -> String {
    format!("{}-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"), super::runs::slug(&model.name))
}

fn row_with_think(rows: &mut [SearchRow], key: &str, think: Value) {
    if let Some(r) = rows.iter_mut().find(|r| r.key == key) {
        r.think = Some(think);
    }
}

/// The row of `key` in an analysis of the worker (a window of a run, a choice, a single block), as (loss, h1, h2).
fn find_row(analysis: &Value, key: &str) -> Option<(f64, Option<f64>, Option<f64>)> {
    let pick = |r: &Value| (r["key"] == json!(key)).then(|| (r["loss"].as_f64(), r["loss_h1"].as_f64(), r["loss_h2"].as_f64()));
    let mut all: Vec<&Value> = Vec::new();
    for k in ["blocks", "choices", "widths"] {
        all.extend(analysis[k].as_array().into_iter().flatten());
    }
    if let Some(runs) = analysis["runs"].as_object() {
        for r in runs.values() {
            all.extend(r["windows"].as_array().into_iter().flatten());
        }
    }
    all.into_iter().find_map(pick).and_then(|(l, a, b)| l.map(|l| (l, a, b)))
}

fn finalist_keys(rows: &[SearchRow], group: usize, n: usize) -> Vec<String> {
    rows.iter().filter(|r| r.group == group).take(n).map(|r| r.key.clone()).collect()
}

/// A step answered "stopped": the state is saved as it is and the search can be started again with the same id.
macro_rules! out {
    ($e:expr, $st:expr, $dir:expr) => {
        match $e {
            Out::Done(v) => v,
            Out::Stopped => {
                $st["status"] = json!("stopped");
                save($dir, &mut $st)?;
                return Ok($st);
            }
        }
    };
}

/// Runs a search. `resume`: the id of a stopped search to go on with (its state says which phases are done). Returns the saved state (`search.json`):
/// `{id, spec, plan, rows, recommendation, stop_reason, notes, think, status}`. `status` is "done" or "stopped"; a stopped search keeps its measured cuts (the worker saved them).
pub fn run(p: &Project, fx: &mut dyn Effects, spec: &SearchSpec, resume: Option<&str>) -> Result<Value> {
    spec.validate()?;
    let model = fx.model(&spec.model_id)?;
    let id = resume.map(String::from).unwrap_or_else(|| new_id(&model));
    let dir = search_dir(p, &id);
    std::fs::create_dir_all(&dir)?;
    let mut st = read_json(&state_file(&dir)).filter(|_| resume.is_some()).unwrap_or_else(|| json!({"id": id, "spec": spec, "created": now(), "notes": [], "status": "running"}));
    st["status"] = json!("running");
    let ins = match serde_json::from_value::<Inspect>(st["inspect"].clone()) {
        Ok(i) => i,
        Err(_) => {
            let i = out!(fx.inspect(&model)?, st, &dir);
            st["inspect"] = json!(i);
            i
        }
    };
    let sp = plan(&ins, spec, fx.vram_gb(), &fx.history())?;
    st["plan"] = json!(sp);
    save(&dir, &mut st)?;
    // 1. the measuring (the worker keeps every pass; asking again costs nothing for the passes it has)
    if st["analysis"].is_null() {
        fx.emit(json!({"kind": "search", "phase": "search", "status": "measuring", "id": id}));
        let t = try_spec(&ins, spec, &sp, fx.vram_gb())?;
        let a = out!(fx.try_run(&t)?, st, &dir);
        st["stop_reason"] = a["search"]["stop_reason"].clone();
        st["analysis"] = a;
        save(&dir, &mut st)?;
    }
    let analysis = st["analysis"].clone();
    let mut rows = rank(&analysis, &ins);
    if rows.is_empty() {
        st["notes"].as_array_mut().map(|n| n.push(json!("The search measured no cut: the time was too short for even the first round, or the model gave no answer.")));
    }
    let mut rec = recommend(&rows, spec);
    // 2. a close call is asked again with more text (one load, a few passes at twice the length)
    if spec.confirm_full && st["confirm"].is_null() && p_needs_confirm(&rec, sp.samples) {
        let top = rec.as_ref().map(|r| r.group).unwrap_or(0);
        let keys = finalist_keys(&rows, top, 4);
        let choices: Vec<Value> = keys.iter().map(|k| { let (b, w) = parse_key(k); json!({"blocks": b, "width": w}) }).collect();
        let v = json!({"model_id": spec.model_id, "samples": CONFIRM_SAMPLES, "len": 512, "load": sp.load, "measure_samples": WIDTH_SAMPLES, "params_b": ins.total_params as f64 / 1e9,
                       "reasoning": spec.keep_thinking.then(ReasoningCalib::default), "choices": choices});
        let t: TrySpec = serde_json::from_value(v).map_err(|e| Error::new(format!("The confirmation request is not valid: {e}")))?;
        t.validate()?;
        fx.emit(json!({"kind": "search", "phase": "search", "status": "confirming", "id": id}));
        let a2 = out!(fx.try_run(&t)?, st, &dir);
        let mut conf = json!({"samples": CONFIRM_SAMPLES, "base": a2["base"]["loss"], "rows": {}});
        for k in &keys {
            if let Some((l, h1, h2)) = find_row(&a2, k) {
                conf["rows"][k] = json!({"loss": l, "loss_h1": h1, "loss_h2": h2});
            }
        }
        st["confirm"] = conf;
        save(&dir, &mut st)?;
    }
    if let Some(conf) = st["confirm"]["rows"].as_object().filter(|o| !o.is_empty()) {
        let base2 = st["confirm"]["base"].as_f64();
        for r in rows.iter_mut() {
            if let Some(c) = conf.get(&r.key) {
                if let Some(l) = c["loss"].as_f64() {
                    r.loss_h1 = c["loss_h1"].as_f64();
                    r.loss_h2 = c["loss_h2"].as_f64();
                    r.noise = r.loss_h1.zip(r.loss_h2).map(|(a, b)| (a - b).abs());
                    if let Some(b) = base2 {
                        r.rel = l / b - 1.0;
                    }
                    r.loss = l;
                }
            }
        }
        // rows that were not confirmed keep the first measurement: only the confirmed ones are compared among themselves
        let confirmed: Vec<SearchRow> = rows.iter().filter(|r| conf.contains_key(&r.key)).cloned().collect();
        if let Some(r) = recommend(&confirmed, spec) {
            rec = Some(Recommendation { reason: format!("{} (asked again with {} samples.)", r.reason, CONFIRM_SAMPLES), ..r });
        }
        st["notes"].as_array_mut().map(|n| n.push(json!(format!("The best cuts were measured again with {CONFIRM_SAMPLES} samples because the first call was close."))));
    }
    // 3. the thinking check of the best two (writes one cut at a time, removes it again)
    if spec.keep_thinking && spec.think_check == "top2" {
        let top = rec.as_ref().map(|r| r.group).unwrap_or(0);
        let keys = finalist_keys(&rows, top, 2);
        let mut think = st["think"].clone();
        if think.is_null() {
            think = json!({"rows": {}, "pending": null, "baseline": null});
        }
        let quick = spec.quick_think.unwrap_or(ins.total_params as f64 / 1e9 >= 15.0);
        for key in keys {
            if think["rows"].get(&key).is_some_and(|r| !r.is_null()) {
                continue;
            }
            let (blocks, width) = parse_key(&key);
            let cand = auto_plan::candidate_at(&ins, blocks.len() as u32, width.or(auto_plan::full_width(&ins)), 1.0);
            let cut: CutRef = match serde_json::from_value::<CutRef>(think["pending"]["cut"].clone()).ok().filter(|_| think["pending"]["key"] == json!(key)) {
                Some(c) => c,
                None => {
                    let aspec = AutoSpec::new(&spec.model_id, 1.0);
                    let steps = auto_plan::split_stages(&ins, &cand, 1);
                    let stage = auto_plan::stage_plans(&ins, &aspec, &steps, fx.vram_gb()).into_iter().next().ok_or_else(|| Error::new("The cut to check could not be planned."))?;
                    let mut rs = stage.reshape_spec(&spec.model_id, spec.keep_thinking.then(ReasoningCalib::default), CALIB_SAMPLES, WIDTH_SAMPLES);
                    rs.load = sp.load.clone();
                    if !blocks.is_empty() {
                        rs.drop_layers = Some(blocks.iter().flat_map(|b| (b * ins.period.max(1))..((b + 1) * ins.period.max(1))).collect());
                    }
                    rs.skip_check = true;
                    fx.emit(json!({"kind": "search", "phase": "search", "status": "writing_cut", "id": id, "key": key}));
                    let c = out!(fx.cut(&rs)?, st, &dir);
                    think["pending"] = json!({"key": key, "cut": c});
                    st["think"] = think.clone();
                    save(&dir, &mut st)?;
                    c
                }
            };
            if think["baseline"].is_null() {
                let b = out!(fx.think_check(&cut.name, "original", quick, None)?, st, &dir);
                think["baseline"] = b;
                st["think"] = think.clone();
                save(&dir, &mut st)?;
            }
            let r = out!(fx.think_check(&cut.name, "cut", quick, None)?, st, &dir);
            let score = thinkcheck::think_score(&r["main"]);
            let lost = thinkcheck::lost_much(&think["baseline"]["main"], &r["main"]);
            think["rows"][&key] = json!({"verdict": if lost { "lost_much" } else { "kept" }, "score": score, "main": r["main"], "original": think["baseline"]["main"]});
            think["pending"] = Value::Null;
            // nothing is kept: the cut was only written to be checked
            fx.remove(&Made { kind: "cut".into(), name: cut.name.clone(), model_id: Some(cut.model_id.clone()), stage: 0, removed: false })?;
            st["think"] = think.clone();
            save(&dir, &mut st)?;
        }
        for (k, v) in think["rows"].as_object().cloned().unwrap_or_default() {
            row_with_think(&mut rows, &k, v);
        }
        rec = recommend(&rows, spec).or(rec);
    }
    st["rows"] = json!(rows);
    st["recommendation"] = json!(rec);
    st["status"] = json!("done");
    st["finished"] = json!(now());
    save(&dir, &mut st)?;
    Ok(st)
}

fn p_needs_confirm(rec: &Option<Recommendation>, samples: u32) -> bool {
    rec.as_ref().is_some_and(|r| r.confidence != "clear") && samples < CONFIRM_SAMPLES
}

/// The ranking of what a search has measured so far, from the answers the worker saved: for a stopped search, for the live table and for "show me what you have".
pub fn partial(p: &Project, spec: &SearchSpec, ins: &Inspect, vram_gb: Option<f64>) -> Result<Value> {
    let total_b = ins.total_params as f64 / 1e9;
    let load = plan::scoring_load(ins.total_params, ins.bytes_per_param, vram_gb);
    let reasoning = spec.keep_thinking.then(ReasoningCalib::default);
    let r = super::runs::try_results_with(p, &spec.model_id, spec.samples_for(total_b), 512, load, reasoning.as_ref())?;
    if r["exists"] != json!(true) {
        return Ok(json!({"rows": [], "recommendation": null}));
    }
    let rows = rank(&r["analysis"], ins);
    let rec = recommend(&rows, spec);
    Ok(json!({"rows": rows, "recommendation": rec, "stop_reason": r["analysis"]["search"]["stop_reason"], "passes": r["passes"]}))
}
