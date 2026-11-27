//! Auto-cut for a mixture of experts (Inkling-2-Mini): the plan before the run, in the shapes the dense Auto-cut already uses (`AutoPlan`, `StagePlan`, `Candidate`, `TimeEstimate`),
//! so the run's state machine, its progress bar, Continue / "Change settings and continue", the report and the page read it unchanged. Dense plans never come through here.
//!
//! What is different (owner's decisions D1-D8, docs/MOE_GUIDE.md): the size asked for is the finished Q4_K_M file in GB; it becomes a number of experts per layer (multiples of 16, never
//! below 32, at most half of them removed in one stage, so 128 -> 64 is one stage and 128 -> 32 needs two); the goal is "size" (the default), "speed" (fewer experts per token) or "quality"
//! (nothing to cut: train the uncut model); **Auto-cut never chooses fewer loop passes**, it only says that the variable-loop research checkpoint was trained for it.
//! Healing keeps the packed experts frozen (LoRA on the dense parts), so the healed model is as big as the cut one. Times are formulas until this computer's own jobs say otherwise.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::auto_plan::{quant_bytes_per_param, AutoPlan, AutoSpec, Candidate, DiskPlan, DiskStep, StagePlan, CACHE_BYTES_PER_TOKEN, TARGET_TOLERANCE};
use super::auto_time::{StepPart, StepTime, TimeEstimate};
use super::moe::{expert_counts_allowed, MoeInfo};
use super::moe_compat::VARLOOP_REPO;
use super::plan::{heal_activation_gb, heal_verdict, Inspect, HEAL_TOKENS_M};
use crate::timing::History;

fn size() -> String {
    "size".into()
}
fn reap() -> String {
    "reap".into()
}
fn yes() -> bool {
    true
}

/// The mixture-of-experts part of an Auto-cut request (`AutoSpec.moe`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeAuto {
    /// "size" (a smaller file, the default), "speed" (fewer experts per token) or "quality" (nothing to cut)
    #[serde(default = "size")]
    pub goal: String,
    /// the finished Q4_K_M file the person wants, in GB (goal "size"); for "speed" it is optional (a smaller file as well)
    #[serde(default)]
    pub target_gb: Option<f64>,
    /// experts used per token after the cut (6 or 4); goal "speed" chooses 6 unless Try first says it costs too much. None = unchanged
    #[serde(default)]
    pub top_k: Option<u32>,
    /// how the experts are ranked: "reap" | "freq" | "gate"
    #[serde(default = "reap")]
    pub scoring: String,
    /// nudge the router bias after the cut (no training)
    #[serde(default = "yes")]
    pub rebalance: bool,
}

impl MoeAuto {
    pub fn validate(&self) -> Result<(), String> {
        if !["size", "speed", "quality"].contains(&self.goal.as_str()) {
            return Err("The goal is a smaller file, faster answers or quality.".into());
        }
        if !["reap", "freq", "gate"].contains(&self.scoring.as_str()) {
            return Err("The way experts are ranked is reap (the default), freq or gate.".into());
        }
        if self.top_k.is_some_and(|k| !(1..=32).contains(&k)) {
            return Err("Experts per token must be between 1 and 32.".into());
        }
        if self.target_gb.is_some_and(|g| !g.is_finite() || !(1.0..=1000.0).contains(&g)) {
            return Err("The size of the file must be between 1 and 1000 GB.".into());
        }
        if self.goal == "size" && self.target_gb.is_none() {
            return Err("Say how big the file may be (GB).".into());
        }
        Ok(())
    }
}

/// What a Candidate / StagePlan / the plan say about the experts (all optional, skipped when absent: a dense plan is byte-identical to what it was).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeCandidate {
    pub experts: u32,
    pub top_k: u32,
    pub passes: u32,
    /// the finished Q4_K_M file, GB
    pub q4_gb: f64,
    pub packed_gb: f64,
    /// parameters that work for one token (all loop passes), billions
    pub active_b: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeStage {
    pub experts_before: u32,
    pub experts_after: u32,
    /// experts per token the model has before the whole run, and after this stage
    pub top_k_before: u32,
    pub top_k: u32,
    pub passes: u32,
    pub packed_gb: f64,
    pub q4_gb: f64,
    pub active_b: f64,
    pub scoring: String,
    pub rebalance: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoePlanInfo {
    pub goal: String,
    pub target_gb: Option<f64>,
    pub experts_before: u32,
    pub experts_after: u32,
    pub top_k_before: u32,
    pub top_k_after: u32,
    pub passes: u32,
    pub q4_gb: f64,
    pub packed_gb: f64,
    pub active_b: f64,
    /// every expert count the rules allow, with its Q4 size and in how many stages it is reached: [{experts, q4_gb, packed_gb, params_b, stages}]
    pub table: Vec<Value>,
    /// loop passes: Auto-cut never chooses fewer; the sentence that says what to do instead
    pub passes_note: String,
    pub scoring: String,
    pub rebalance: bool,
}

/// tokens per sample of the measuring pass and of Try first for a mixture of experts (eager attention costs memory by the square of this)
pub const MEASURE_LEN: u32 = 2048;

pub fn q4_gb(params: u64) -> f64 {
    params as f64 * quant_bytes_per_param("q4_k_m") / 1e9
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn moe_of(ins: &Inspect) -> Option<&MoeInfo> {
    ins.inkling()
}

/// Parameters of a model that keeps `experts` per layer (the worker's table row, else scaled).
pub fn params_at(ins: &Inspect, m: &MoeInfo, experts: u32) -> u64 {
    m.size_at(experts).map(|r| r.params).filter(|p| *p > 0).unwrap_or_else(|| {
        let e = m.expert_params as f64 * experts as f64 / m.experts.max(1) as f64;
        (ins.total_params as f64 - m.expert_params as f64 + e) as u64
    })
}

/// Parameters that work for one token with `experts` kept, `top_k` per token and `passes` loop passes: the dense tensors of every layer slot plus top_k/experts of the experts of
/// every slot, plus the output table (the token table is a lookup). Billions. The layers of the loop count once per pass.
pub fn active_b(ins: &Inspect, m: &MoeInfo, experts: u32, top_k: u32, passes: u32) -> f64 {
    let layers = ins.layers.max(1) as f64;
    let slots = m.visits_for(ins.layers, passes) as f64;
    let per_layer_dense = (m.dense_params.saturating_sub(ins.embed_params + ins.lm_head_params + ins.vision_params) as f64) / layers;
    let per_layer_experts = m.expert_params as f64 / layers * experts as f64 / m.experts.max(1) as f64;
    (slots * (per_layer_dense + per_layer_experts * top_k as f64 / experts.max(1) as f64) + ins.lm_head_params as f64) / 1e9
}

/// Every count a cut may end at, with the number of stages it needs: one stage reaches half of the experts at most, two stages a quarter (never below 32 and 4 x top-k).
pub fn reachable(m: &MoeInfo, stages: u32) -> Vec<(u32, u32)> {
    let first = expert_counts_allowed(m.experts, m.top_k);
    let mut out: Vec<(u32, u32)> = first.iter().map(|c| (*c, 1)).collect();
    if stages >= 2 {
        for c in &first {
            for d in expert_counts_allowed(*c, m.top_k) {
                if !out.iter().any(|(x, _)| *x == d) {
                    out.push((d, 2));
                }
            }
        }
    }
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out
}

/// The stops of the way to `last`: one stage when the cut is small (no more than 40 % of the experts go) or when the limit is one stage and half of the experts or more stay; else the
/// middle that splits the removal most evenly between two stages (a count the rules allow from the original, from which `last` is allowed again), so no stage removes more than half.
pub fn stops(m: &MoeInfo, last: u32, stages: u32) -> Vec<u32> {
    let removed = (m.experts - last) as f64 / m.experts as f64;
    let one_ok = last as f64 >= m.experts as f64 * 0.5;
    if (stages < 2 && one_ok) || removed <= 0.40 {
        return vec![last];
    }
    expert_counts_allowed(m.experts, m.top_k).into_iter().filter(|mid| *mid > last && expert_counts_allowed(*mid, m.top_k).contains(&last))
        .min_by_key(|mid| ((m.experts - mid) as i64 - (mid - last) as i64).abs()).map(|mid| vec![mid, last]).unwrap_or_else(|| vec![last])
}

/// Heal tokens (millions) of a stage by how much it removes (decision in the plan: up to 25 % -> 20 M, up to 50 % or fewer experts per token -> 40 M); the person's own choice wins.
pub fn tokens_m_for(removed_share: f64, top_k_cut: bool) -> u32 {
    let m = if removed_share > 0.25 || top_k_cut { 40 } else { 20 };
    *HEAL_TOKENS_M.iter().find(|t| **t >= m).unwrap_or(&80)
}

/// The packed bytes of a model with `experts` per layer, GB.
pub fn packed_gb(m: &MoeInfo, experts: u32) -> f64 {
    m.packed_bytes_at(experts) as f64 / 1e9
}

/// Student step memory with the packed experts frozen (option A): the packed bytes, the LoRA and its Adam states on the dense parts (rank 64), activations (every layer slot of the loop, checkpointed),
/// the eager attention's score matrices of a layer or two, the chunk of the loss, 1 GB of overhead; +10 %. An estimate.
pub fn heal_gb(ins: &Inspect, m: &MoeInfo, experts: u32, seq: u32, rank: u32, passes: u32) -> f64 {
    let hidden = 2304.0;
    let lora = rank as f64 * ins.layers as f64 * 20.0 * hidden * 16.0 / 1e9;
    let act = heal_activation_gb(m.visits_for(ins.layers, passes), hidden as u32, seq, 1, true);
    let attn = 2.0 * 24.0 * (seq as f64).powi(2) * 4.0 / 1e9;
    let loss = 256.0 * 200_058.0 * 16.0 / 1e9;
    (packed_gb(m, experts) + lora + act + attn + loss + 1.0) * 1.10
}

/// Heal hours for `tokens_m` million tokens: the original's answers (vLLM or the packed in-process teacher) then the student's learning. (low, high)
pub fn heal_hours(tokens_m: u32) -> (f64, f64) {
    let t = tokens_m as f64 * 1e6;
    ((t / 4000.0 + t / 1300.0) / 3600.0, (t / 1500.0 + t / 800.0) / 3600.0)
}

/// The finished candidates of a request, one per reachable count (the chosen one is marked `on_target`).
fn candidates(ins: &Inspect, m: &MoeInfo, ma: &MoeAuto, stages: u32) -> Vec<Candidate> {
    let target = ma.target_gb;
    let k = ma.top_k.unwrap_or(m.top_k);
    let mut counts: Vec<u32> = vec![m.experts];
    counts.extend(reachable(m, stages).into_iter().map(|(c, _)| c));
    counts.dedup();
    counts.into_iter().map(|c| {
        let params = params_at(ins, m, c);
        let q4 = q4_gb(params);
        let (err, on) = match target {
            Some(t) => ((q4 - t) / t * 100.0, q4 <= t * (1.0 + TARGET_TOLERANCE)),
            None => (0.0, c == m.experts),
        };
        Candidate {
            blocks: 0, intermediate: None, layers: ins.layers, params, params_b: ((params as f64 / 1e9) * 100.0).round() / 100.0, error_pct: (err * 10.0).round() / 10.0, width_cut: 0.0,
            on_target: on, needs_confirmation: false,
            moe: Some(MoeCandidate { experts: c, top_k: k, passes: m.passes, q4_gb: r1(q4), packed_gb: r1(packed_gb(m, c)), active_b: (active_b(ins, m, c, k, m.passes) * 100.0).round() / 100.0 }),
        }
    }).collect()
}

/// One stage planned again (after a back-off): `before` and `after` experts, the experts per token the last stage ends with.
pub fn restage(ins: &Inspect, spec: &AutoSpec, before: u32, after: u32, top_k: u32, index: u32, of: u32, vram_gb: Option<f64>) -> Option<StagePlan> {
    let (ma, m) = (spec.moe.as_ref()?, moe_of(ins)?);
    Some(stage_plan(ins, m, spec, ma, before, after, top_k, index, of, vram_gb))
}

fn stage_plan(ins: &Inspect, m: &MoeInfo, spec: &AutoSpec, ma: &MoeAuto, before: u32, after: u32, top_k: u32, index: u32, of: u32, vram_gb: Option<f64>) -> StagePlan {
    let params = params_at(ins, m, after);
    let last = index == of;
    let removed = (before - after) as f64 / before.max(1) as f64;
    let top_k_cut = last && top_k < m.top_k;
    let tokens_m = if last { spec.last_tokens_m } else { spec.stage_a_tokens_m }.unwrap_or_else(|| tokens_m_for(removed, top_k_cut));
    let (seq, batch, rank) = (4096u32, 1u32, 64u32);
    let gb = heal_gb(ins, m, after, seq, rank, m.passes);
    let teacher_gb = m.packed_bytes as f64 / 1e9 + 4.0;
    let est = gb.max(teacher_gb);
    let verdict = heal_verdict(est, vram_gb);
    let h = heal_hours(tokens_m);
    let total_b = ins.total_params as f64 / 1e9;
    StagePlan {
        index, of, blocks_after: 0, blocks_cut: 0, intermediate_before: None, intermediate_after: None, width: None, layers_after: ins.layers, params, params_b: ((params as f64 / 1e9) * 100.0).round() / 100.0,
        cut_pct: if total_b > 0.0 { ((1.0 - params as f64 / 1e9 / total_b) * 1000.0).round() / 10.0 } else { 0.0 }, tokens_m, seq, batch, rank,
        cache_gb: r1(tokens_m as f64 * 1e6 * CACHE_BYTES_PER_TOKEN / 1e9), heal_gb: r1(est), verdict: verdict.to_string(), mode: "cache".into(), hours: (r1(h.0), r1(h.1)),
        teacher_load: "packed".into(), seq_pinned: false, breakdown: None,
        moe: Some(MoeStage { experts_before: before, experts_after: after, top_k_before: m.top_k, top_k, passes: m.passes, packed_gb: r1(packed_gb(m, after)), q4_gb: r1(q4_gb(params)), active_b: (active_b(ins, m, after, top_k, m.passes) * 100.0).round() / 100.0,
                             scoring: ma.scoring.clone(), rebalance: ma.rebalance }),
    }
}

/// The disk timeline: the original (packed), every stage's cut, the original's answers and the healed model (the experts are copied unchanged, so as big as the cut), then the 16-bit GGUF and the sizes.
fn disk(ins: &Inspect, m: &MoeInfo, stages: &[StagePlan], quants: &[String], keep: bool, free_gb: Option<f64>) -> DiskPlan {
    let g = |x: f64| r1(x);
    let mut live = if ins.source_bytes > 0 { ins.source_bytes as f64 / 1e9 } else { m.packed_bytes as f64 / 1e9 };
    let mut steps = vec![DiskStep { step: "the original model".into(), in_use_gb: g(live) }];
    let mut peak = (live, steps[0].step.clone());
    let note = |label: String, live: f64, steps: &mut Vec<DiskStep>, peak: &mut (f64, String)| {
        steps.push(DiskStep { step: label.clone(), in_use_gb: g(live) });
        if live > peak.0 {
            *peak = (live, label);
        }
    };
    let mut prev_healed = 0.0;
    for s in stages {
        let size = s.moe.as_ref().map(|x| x.packed_gb).unwrap_or(0.0);
        live += size;
        note(format!("stage {}: the cut", s.index), live, &mut steps, &mut peak);
        live += s.cache_gb;
        note(format!("stage {}: the original's answers", s.index), live, &mut steps, &mut peak);
        live += size;
        note(format!("stage {}: the healed model", s.index), live, &mut steps, &mut peak);
        if !keep {
            live -= size + s.cache_gb + prev_healed;
            prev_healed = size;
            note(format!("stage {}: passed, intermediate files deleted", s.index), live, &mut steps, &mut peak);
        }
    }
    let final_params = stages.last().map(|s| s.params).unwrap_or(ins.total_params);
    let f16 = final_params as f64 * 2.0 / 1e9;
    live += f16;
    note("export: the 16-bit GGUF".into(), live, &mut steps, &mut peak);
    let q: f64 = quants.iter().filter(|id| id.as_str() != "f16").map(|id| final_params as f64 * quant_bytes_per_param(id) / 1e9).sum();
    live += q;
    note("export: the sizes".into(), live, &mut steps, &mut peak);
    if !keep {
        live -= f16;
        note("export: the 16-bit GGUF deleted".into(), live, &mut steps, &mut peak);
    }
    let download = if ins.local { 0.0 } else { ins.source_bytes as f64 / 1e9 };
    DiskPlan { peak_gb: g(peak.0 + download), peak_step: peak.1, final_gb: g(live), keep_intermediate: keep, steps, free_gb: free_gb.map(g), enough: free_gb.map(|f| f >= (peak.0 + download) * 1.05) }
}

/// The plan of an Auto-cut of a mixture of experts. Never touches the files; `vram_gb` and `free_gb` may be unknown.
pub fn plan(ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, free_gb: Option<f64>) -> AutoPlan {
    let original_b = ins.total_params as f64 / 1e9;
    let mut out = AutoPlan {
        ok: false, blockers: Vec::new(), warnings: Vec::new(), original_b: (original_b * 100.0).round() / 100.0, target_b: spec.target_b, candidates: Vec::new(), chosen: None, try_list: Vec::new(),
        stages: Vec::new(), disk: None, measure_load: "auto".into(), heal_hours: (0.0, 0.0), think_gate_minutes: None, measured_minutes: None, time: None, speed: None, budget: None, moe: None,
    };
    let (Some(ma), Some(m)) = (spec.moe.as_ref(), moe_of(ins)) else {
        out.blockers.push("This model is not a mixture of experts that Auto-cut can cut.".into());
        return out;
    };
    if let Err(e) = spec.validate() {
        out.blockers.push(e.0);
        return out;
    }
    if !ins.supported {
        out.blockers.push(ins.why.clone().unwrap_or_else(|| "This model cannot be cut.".into()));
        return out;
    }
    let passes_note = format!("Auto-cut never takes loop passes out: this model was trained with {} passes, and its authors only trained {VARLOOP_REPO} for 1 or 2. Choose that model as the starting point to make a version with fewer passes.", m.passes);
    if ma.goal == "quality" {
        out.blockers.push("A better model for your project comes from training, not from cutting: use the Train page on the uncut model (QLoRA needs about 22 GB). Choose a smaller file or faster answers to cut.".into());
        return out;
    }
    let stages_wanted = spec.stages.clamp(1, 2);
    let cands = candidates(ins, m, ma, 2);
    out.candidates = cands.clone();
    let k_final = ma.top_k.or(if ma.goal == "speed" { Some(6.min(m.top_k)) } else { None }).unwrap_or(m.top_k);
    if k_final > m.top_k {
        out.blockers.push("Auto-cut makes the model smaller or faster; more experts per token is asked in Try first on the Slim page.".into());
        return out;
    }
    // the count: the largest that makes the file small enough (goal size), or all experts when only the speed is asked for and no size is given
    let full_q4 = q4_gb(ins.total_params);
    let chosen_count = match (ma.target_gb, ma.goal.as_str()) {
        (Some(t), _) => {
            if t >= full_q4 * 0.97 {
                out.blockers.push(format!("The file is {full_q4:.1} GB already at Q4_K_M: choose a size at least 3 % smaller."));
                return out;
            }
            let reach = reachable(m, 2);
            match reach.iter().find(|(c, _)| q4_gb(params_at(ins, m, *c)) <= t * (1.0 + TARGET_TOLERANCE)) {
                Some((c, _)) => *c,
                None => {
                    let smallest = reach.last().map(|(c, _)| *c).unwrap_or(m.experts);
                    out.blockers.push(format!("{t:.1} GB is smaller than the cut that is allowed (the smallest is {smallest} experts per layer, {:.1} GB at Q4_K_M: at most half of the experts go in one stage, at least 32 stay). Choose a larger size.", q4_gb(params_at(ins, m, smallest))));
                    return out;
                }
            }
        }
        (None, _) => m.experts,
    };
    let want_cut = chosen_count < m.experts;
    let chosen = cands.iter().find(|c| c.moe.as_ref().is_some_and(|x| x.experts == chosen_count)).cloned().map(|mut c| {
        c.on_target = true;
        if let Some(x) = c.moe.as_mut() {
            x.top_k = k_final;
            x.active_b = (active_b(ins, m, chosen_count, k_final, m.passes) * 100.0).round() / 100.0;
        }
        c
    });
    let Some(chosen) = chosen else {
        out.blockers.push("No cut of this model reaches that size.".into());
        return out;
    };
    // Try first asks about the chosen count and its neighbours (a smaller file than asked is never chosen silently), the experts per token, and the other rankings
    out.try_list = cands.iter().filter(|c| c.moe.as_ref().is_some_and(|x| (x.experts as i64 - chosen_count as i64).abs() <= 16 && x.experts < m.experts)).cloned().collect();
    out.chosen = Some(chosen.clone());
    let stops_v: Vec<u32> = if want_cut { stops(m, chosen_count, stages_wanted) } else { Vec::new() };
    if want_cut && spec.stages == 1 && stops_v.len() == 2 {
        out.warnings.push(format!("Reaching {chosen_count} experts per layer removes more than half of them, which one stage may not do: it is done in two stages (a heal between)."));
    } else if want_cut && spec.stages == 2 && stops_v.len() == 1 {
        out.warnings.push("This cut is small enough to do in one stage.".into());
    }
    let mut stage_list = Vec::new();
    let mut before = m.experts;
    let n_stages = if want_cut { stops_v.len() as u32 } else { 1 };
    if want_cut {
        for (i, after) in stops_v.iter().enumerate() {
            let last = i + 1 == stops_v.len();
            stage_list.push(stage_plan(ins, m, spec, ma, before, *after, if last { k_final } else { m.top_k }, i as u32 + 1, n_stages, vram_gb));
            before = *after;
        }
    } else {
        stage_list.push(stage_plan(ins, m, spec, ma, m.experts, m.experts, k_final, 1, 1, vram_gb));
    }
    for s in &stage_list {
        if s.verdict == "over" {
            out.blockers.push(format!("Healing stage {} ({:.1} B) needs about {:.0} GB on the card (an estimate); this card has {:.0} GB.", s.index, s.params_b, s.heal_gb, vram_gb.unwrap_or(0.0)));
        } else if s.verdict == "tight" {
            out.warnings.push(format!("Healing stage {} is tight on this card (about {:.0} GB of {:.0} GB); the program steps down to safer settings by itself if it runs out of memory.", s.index, s.heal_gb, vram_gb.unwrap_or(0.0)));
        }
    }
    out.heal_hours = stage_list.iter().fold((0.0, 0.0), |a, s| (a.0 + s.hours.0, a.1 + s.hours.1));
    out.stages = stage_list;
    let final_params = out.stages.last().map(|s| s.params).unwrap_or(ins.total_params);
    let export = spec.export_quants_for(final_params);
    let d = disk(ins, m, &out.stages, &export, spec.keep_intermediate, free_gb);
    if d.enough == Some(false) {
        out.blockers.push(format!("The run needs up to {:.0} GB of disk ({}); {:.0} GB are free. Free some space{}.", d.peak_gb, d.peak_step, d.free_gb.unwrap_or(0.0),
                                  if spec.keep_intermediate { " or switch \"keep the intermediate files\" off" } else { "" }));
    }
    out.disk = Some(d);
    if want_cut && chosen_count as f64 <= m.experts as f64 * 0.5 {
        out.warnings.push(format!("Half of the experts or more are removed: expect clearly weaker answers until the model is healed ({}). The checks after every stage show how much is left.", HEAL_TOKENS_M.iter().find(|t| **t >= 40).map(|t| format!("{t} M tokens")).unwrap_or_default()));
    }
    if k_final < m.top_k {
        out.warnings.push(format!("{k_final} experts per token instead of {}: the authors did not test this; Try first measures it before it is used, and the model is healed afterwards.", m.top_k));
    }
    if m.passes > 1 && !m.varloop {
        out.warnings.push(passes_note.clone());
    }
    out.warnings.push("The drafter file (speculative decoding) is kept but was not healed, so it accepts fewer tokens; it is not exported.".into());
    let q4_final = q4_gb(final_params);
    out.moe = Some(MoePlanInfo {
        goal: ma.goal.clone(), target_gb: ma.target_gb, experts_before: m.experts, experts_after: chosen_count, top_k_before: m.top_k, top_k_after: k_final, passes: m.passes, q4_gb: r1(q4_final),
        packed_gb: r1(packed_gb(m, chosen_count)), active_b: chosen.moe.as_ref().map(|x| x.active_b).unwrap_or(0.0),
        table: std::iter::once((m.experts, 0u32)).chain(reachable(m, 2)).map(|(c, st)| {
            let p = params_at(ins, m, c);
            json!({"experts": c, "q4_gb": r1(q4_gb(p)), "packed_gb": r1(packed_gb(m, c)), "params_b": ((p as f64 / 1e9) * 100.0).round() / 100.0, "stages": st})
        }).collect(),
        passes_note, scoring: ma.scoring.clone(), rebalance: ma.rebalance,
    });
    out.ok = out.blockers.is_empty();
    if out.ok {
        out.time = Some(time_estimate(ins, spec, &out, &History::default()));
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------- time

fn part(id: &str, label: &str, lo: f64, hi: f64, hist: bool) -> StepPart {
    StepPart { id: id.into(), label: label.into(), minutes: (r1(lo), r1(hi)), basis: if hist { "your_history" } else { "formula" }.into() }
}

/// minutes of `work` units at a rate measured on this computer (`key`), else at the formula's (fast, slow) rate: (low, high, from history)
fn at(h: &History, key: &str, work: f64, formula: (f64, f64)) -> (f64, f64, bool) {
    match h.pick(key, 1.0) {
        Some((v, _)) if v > 0.0 => (work / v / 60.0 * 0.85, work / v / 60.0 * 1.25, true),
        _ => (work / formula.0 / 60.0, work / formula.1 / 60.0, false),
    }
}

/// The time of a MoE run, step by step, with the ids of the dense run (`auto::steps_after_plan`), so the page's list, the progress bar and "Change settings and continue" work as they do.
/// Formulas, every number a guess until the jobs have run here: the packed model loads at 1.0 to 0.3 GB/s, the measuring pass reads 700 to 300 tokens/s, a Try pass 1,500 to 600,
/// the thinking check of the GGUF 10 to 20 minutes (quick) or 25 to 50 (full), healing from `heal_hours`. History keys: `moe_score_tok_s`, `moe_try_tok_s`, `write_gb_s`.
pub fn time_estimate(ins: &Inspect, spec: &AutoSpec, plan: &AutoPlan, history: &History) -> TimeEstimate {
    let m = moe_of(ins).cloned().unwrap_or_default();
    let (samples, len) = (spec.try_samples() as f64, MEASURE_LEN as f64);
    let tokens = samples * len;
    let load_gb = m.packed_bytes as f64 / 1e9;
    let load = (load_gb / 1.0 / 60.0, load_gb / 0.3 / 60.0);
    let mut steps = Vec::new();
    let mut push = |id: &str, label: &str, parts: Vec<StepPart>, note: Option<String>| {
        let lo: f64 = parts.iter().map(|p| p.minutes.0).sum();
        let hi: f64 = parts.iter().map(|p| p.minutes.1).sum();
        let hist = parts.iter().filter(|p| p.basis == "your_history").map(|p| p.minutes.1).sum::<f64>() > hi / 2.0;
        steps.push(StepTime { id: id.into(), label: label.into(), minutes: (r1(lo), r1(hi)), basis: if hist { "your_history" } else { "formula" }.into(), note, parts, rounds: None });
    };
    let think = |quick: bool| if quick { (10.0, 20.0) } else { (25.0, 50.0) };
    let tq = spec.think_quick();
    let n_stages = plan.stages.len().max(1);
    let list = super::auto::steps_after_plan(spec, n_stages as u32, true);
    push("plan", "Check the model, the card and the disk", vec![part("start", "Look at the model", 0.1, 0.5, false)], None);
    let write_gb = |gb: f64| at(history, "write_gb_s", gb, (0.5, 0.3));
    for s in &list {
        let id = s.id.as_str();
        match id {
            "measure" => {
                let sc = at(history, "moe_score_tok_s", tokens * 2.0, (700.0, 300.0));
                push(id, &s.label, vec![part("load", "Load the packed model", load.0, load.1, false), part("measure", &format!("Count what every expert does ({} samples)", samples as u32), sc.0, sc.1, sc.2)],
                     Some("Reads the text through the model once with the experts' output sizes, and once more through the routers as the cut model would have them.".into()));
            }
            "try" => {
                let q = 10.0;
                let tr = at(history, "moe_try_tok_s", tokens * q, (1500.0, 600.0));
                push(id, &s.label, vec![part("load", "Load the packed model", load.0, load.1, false), part("check", &format!("About {q:.0} questions: the baseline, fewer experts per token, the counts near the target"), tr.0, tr.1, tr.2)], None);
            }
            "think_gate" => {
                let n = spec.think_gate_n.min(plan.try_list.len().max(1) as u32) as f64;
                let t = think(tq);
                push(id, &s.label, vec![part("check", "Cut the best few, check each one's thinking and the original's", t.0 * (n + 1.0), t.1 * (n + 1.0), false)], None);
            }
            _ if id.ends_with(".cut") => {
                let idx: usize = id.trim_start_matches('s').split('.').next().and_then(|x| x.parse().ok()).unwrap_or(1);
                let gb = plan.stages.get(idx - 1).and_then(|st| st.moe.as_ref()).map(|x| x.packed_gb).unwrap_or(load_gb);
                let w = write_gb(gb);
                push(id, &s.label, vec![part("write", "Write the cut model (the packed experts are sliced, not decoded)", w.0, w.1, w.2)], None);
            }
            _ if id.ends_with(".check_cut") || id.ends_with(".check_heal") => {
                let t = think(tq);
                push(id, &s.label, vec![part("load", "Convert and load the cut model", 3.0, 12.0, false), part("check", "The thinking questions", t.0, t.1, false)], Some("Checked through llama.cpp on a GGUF of the model.".into()));
            }
            _ if id.ends_with(".heal") => {
                let idx: usize = id.trim_start_matches('s').split('.').next().and_then(|x| x.parse().ok()).unwrap_or(1);
                let st = plan.stages.get(idx - 1);
                let tokens_m = st.map(|x| x.tokens_m).unwrap_or(20) as f64;
                let t = tokens_m * 1e6;
                let cache = at(history, "moe_teacher_tok_s", t, (4000.0, 1500.0));
                let train = at(history, "moe_train_tok_s", t, (1300.0, 800.0));
                let gb = st.and_then(|x| x.moe.as_ref()).map(|x| x.packed_gb).unwrap_or(load_gb);
                push(id, &s.label, vec![part("start", "Start the programs", 0.5, 1.0, false), part("load", "Load the original and the cut model", load.0 * 2.0, load.1 * 2.0, false),
                                        part("cache", "The original answers the training text", cache.0, cache.1, cache.2), part("train", "The cut model learns (LoRA on the dense parts, the experts stay frozen)", train.0, train.1, train.2),
                                        part("merge", "Merge the add-on into the model", gb * 0.3, gb * 1.2, false)], None);
            }
            "export" => {
                let p = plan.stages.last().map(|x| x.params).unwrap_or(ins.total_params);
                let gb16 = p as f64 * 2.0 / 1e9;
                let sizes = spec.export_quants_for(p).len() as f64;
                push(id, &s.label, vec![part("convert", "Convert to a 16-bit GGUF", gb16 * 0.1, gb16 * 0.2, false), part("imatrix", "Importance matrix (adapted from the original's where possible)", p as f64 / 1e9 * 0.6 * 0.3, p as f64 / 1e9 * 1.2, false),
                                        part("quantize", "Make the sizes", gb16 * 0.1 * sizes / 2.0, gb16 * 0.3 * sizes / 2.0, false)], None);
            }
            "check_gguf" => {
                let t = think(tq);
                push(id, &s.label, vec![part("check", "The thinking questions on the GGUF", t.0, t.1, false)], None);
            }
            "compare_quants" => push(id, &s.label, vec![part("check", "Compare the sizes", 6.0, 20.0, false)], None),
            "studio_publish" => push(id, &s.label, vec![part("copy", "Copy the files", 1.0, 5.0, false)], None),
            _ => {}
        }
    }
    let lo: f64 = steps.iter().map(|s| s.minutes.0).sum::<f64>() / 60.0;
    let hi: f64 = steps.iter().map(|s| s.minutes.1).sum::<f64>() / 60.0;
    let hist_steps = steps.iter().filter(|s| s.basis == "your_history").count() as u32;
    let lever = list.iter().filter(|s| s.id.ends_with(".heal")).filter_map(|s| steps.iter().find(|t| t.id == s.id)).fold((0.0, 0.0), |a, t| {
        let tr: (f64, f64) = t.parts.iter().filter(|p| p.id == "cache" || p.id == "train").fold((0.0, 0.0), |b, p| (b.0 + p.minutes.0, b.1 + p.minutes.1));
        (a.0 + tr.0 / 60.0, a.1 + tr.1 / 60.0)
    });
    let total = steps.len() as u32;
    TimeEstimate {
        steps, total_h: (r1(lo), r1(hi)), history_steps: hist_steps, step_count: total,
        basis: if hist_steps == 0 { "No earlier run of a mixture of experts on this computer: every time is a formula (a guess; the first run teaches the next estimate).".into() }
               else { format!("{hist_steps} of {total} steps use your own earlier runs; the others are formulas.") },
        lever_h: (r1(lever.0), r1(lever.1)), notes: vec!["Nothing here was measured on a real card yet: treat the times as an order of magnitude.".into()], extra: Vec::new(), vllm: None,
    }
}
