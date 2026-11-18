//! Auto-cut's time estimate (docs/AUTO_CUT.md, "How long it takes"): the whole run, step by step, before it starts. Pure functions of the plan, the spec and the
//! `History` of this computer's own saved job logs; no file is read, nothing runs.
//!
//! Every step of the run (the ids of `auto::steps_after_plan`) gets a (low, high) range in minutes. A rate that this computer has measured before (`timing::History`:
//! the teacher's answers, the training, the merge, the loads, the thinking check ...) is used for the same kind of work at the new size (linearly) and the step is tagged
//! `your_history`; anything else is a formula built on the reference numbers of one real run (below) and is tagged `formula`. A formula is never shown as a measurement.
//!
//! Reference run (an RTX 5090 with 32 GB, 94 GB of RAM, Windows; a 17.4 B student cut from a 27 B teacher): the teacher's answers 1.0 to 2.5 hours per 5 M tokens
//! (scaled by the teacher's size), the student's training 1.4 to 2.2 hours per 5 M tokens (scaled by the student's size), the merge 0.3 to 1.2 minutes per GB (the real
//! run: about 35 minutes for 34 GB with the older arithmetic), the thinking check 15 minutes (Full) or 10 (Quick) per 17 B, the cut written at 0.3 to 0.5 GB per second.
//! The other numbers (measuring, Try first, GGUF conversion and quantising, the copy) are guesses, named as such in `docs/AUTO_CUT.md`.
use serde::{Deserialize, Serialize};

use super::auto_plan::{self, AutoPlan, AutoSpec, StagePlan};
use super::plan::Inspect;
use crate::timing::History;

/// Process start, loading the libraries and the small checks of one worker job, minutes (low, high).
const START: (f64, f64) = (0.5, 1.0);
/// A rate measured earlier: the estimate is this much below and above the point value (a computer is not exactly as fast every time).
const HIST_LO: f64 = 0.85;
const HIST_HI: f64 = 1.25;
/// Reference numbers (see the module documentation).
const CACHE_H_PER_5M: (f64, f64) = (1.0, 2.5);
const CACHE_REF_B: f64 = 27.0;
const TRAIN_H_PER_5M: (f64, f64) = (1.4, 2.2);
const TRAIN_REF_B: f64 = 17.4;
const MERGE_MIN_PER_GB: (f64, f64) = (0.3, 1.2);
/// Guesses.
const LOAD_GB_S: [(&str, (f64, f64)); 3] = [("16bit", (1.5, 0.6)), ("4bit", (1.0, 0.3)), ("offload", (0.5, 0.2))];
const WRITE_GB_S: (f64, f64) = (0.5, 0.3);
const SCORE_MIN_PER_B: (f64, f64) = (0.35, 1.0);
const PASS_TOK_B_S: (f64, f64) = (80_000.0, 40_000.0);
const CONVERT_MIN_PER_GB: (f64, f64) = (0.1, 0.2);
const QUANT_MIN_PER_GB: (f64, f64) = (0.1, 0.3);
const COPY_GB_S: (f64, f64) = (1.0, 0.2);
const CHECK_SPREAD: (f64, f64) = (0.8, 1.4);
/// Tokens of a measuring or Try-first pass per sample (`calib_len`).
const LEN: f64 = 512.0;
const MEASURE_SAMPLES: f64 = 256.0;
const CUT_CHECK_SAMPLES: f64 = 128.0;
/// The extra heal after a failed thinking check, in millions of tokens (`auto::step_check_heal`).
const RETRY_TOKENS_M: u32 = 40;

/// A part of a step (what takes the time inside it): loading, the teacher's answers, the training, the merge ...
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepPart {
    /// start | load | cache | train | merge | measure | write | check | convert | quantize | copy
    pub id: String,
    pub label: String,
    /// (low, high) minutes
    pub minutes: (f64, f64),
    /// "your_history" | "formula"
    pub basis: String,
}

/// One step of the run with its estimate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepTime {
    /// the id of `auto::StepRec` (`plan`, `measure`, `try`, `think_gate`, `s1.cut`, `s1.check_cut`, `s1.heal`, `s1.check_heal`, `export`, `check_gguf`, `compare_quants`, `studio_publish`)
    pub id: String,
    pub label: String,
    /// (low, high) minutes
    pub minutes: (f64, f64),
    /// "your_history" when the parts that take most of the step come from this computer's own earlier jobs, else "formula"
    pub basis: String,
    pub note: Option<String>,
    pub parts: Vec<StepPart>,
}

/// Time that is not in the total: it happens only if something goes wrong.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtraTime {
    pub id: String,
    pub label: String,
    pub minutes: (f64, f64),
    pub note: String,
}

/// The time of the whole run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeEstimate {
    pub steps: Vec<StepTime>,
    /// the sum of the steps, in hours (low, high)
    pub total_h: (f64, f64),
    /// how many of the steps use this computer's own earlier jobs, and how many there are
    pub history_steps: u32,
    pub step_count: u32,
    /// the sentence for the window: "3 of 14 steps use your own earlier runs; ..."
    pub basis: String,
    /// the teacher's answers and the training of all stages, in hours (low, high): what shortens the run most when it is made smaller
    pub lever_h: (f64, f64),
    pub notes: Vec<String>,
    /// what is not in the total (a retry after a failed thinking check)
    #[serde(default)]
    pub extra: Vec<ExtraTime>,
}

#[derive(Clone, Copy)]
struct R {
    lo: f64,
    hi: f64,
    hist: bool,
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn from_history(minutes: f64) -> R {
    R { lo: minutes * HIST_LO, hi: minutes * HIST_HI, hist: true }
}

fn fixed(lo: f64, hi: f64) -> R {
    R { lo, hi, hist: false }
}

/// Work done at a measured speed (`key` is a rate of work per second, `work` the amount of it): this computer's own rate, else the formula range.
fn at_rate(h: &History, key: &str, size: f64, work: f64, formula: (f64, f64)) -> R {
    match h.pick(key, size) {
        Some((v, _)) => from_history(work / v / 60.0),
        None => fixed(formula.0, formula.1),
    }
}

/// Seconds per unit measured (`key` is seconds per `amount`'s unit): this computer's own figure, else the formula range.
fn per_unit(h: &History, key: &str, size: f64, amount: f64, formula: (f64, f64)) -> R {
    match h.pick(key, size) {
        Some((v, _)) => from_history(v * amount / 60.0),
        None => fixed(formula.0, formula.1),
    }
}

fn load_formula(gb: f64, mode: &str) -> (f64, f64) {
    let (fast, slow) = LOAD_GB_S.iter().find(|(m, _)| *m == mode).map(|(_, r)| *r).unwrap_or(LOAD_GB_S[1].1);
    (gb / fast / 60.0, gb / slow / 60.0)
}

fn load(h: &History, gb: f64, mode: &str) -> R {
    let key = format!("load_gb_s:{mode}");
    let f = load_formula(gb, mode);
    match h.pick(&key, gb) {
        Some((v, _)) => from_history(gb / v / 60.0),
        None => fixed(f.0, f.1),
    }
}

fn mode_factor(mode: &str) -> f64 {
    if mode == "offload" { 4.0 } else { 1.0 }
}

/// Measuring a model of `b` billion parameters over `samples` texts (the layer scores): guess, scaled by the size and the way it is loaded.
fn measure_pass(h: &History, b: f64, samples: f64, mode: &str) -> R {
    let f = mode_factor(mode) * samples / MEASURE_SAMPLES;
    at_rate(h, "score_tok_b_s", b, samples * LEN * b, (SCORE_MIN_PER_B.0 * b * f, SCORE_MIN_PER_B.1 * b * f))
}

/// One forward pass over `samples` texts (Try first's passes, the loss check of a cut): guess.
fn forward_pass(h: &History, b: f64, samples: f64, mode: &str) -> R {
    let work = samples * LEN * b;
    let f = mode_factor(mode);
    at_rate(h, "try_tok_b_s", b, work, (work / PASS_TOK_B_S.0 / 60.0 * f, work / PASS_TOK_B_S.1 / 60.0 * f))
}

/// One thinking check of a model of `b` billion parameters (the check itself; writing and loading are other parts).
fn think_check(h: &History, b: f64, quick: bool) -> R {
    let key = if quick { "think_s_per_b:quick" } else { "think_s_per_b:full" };
    let m = auto_plan::think_check_only_minutes(b, quick);
    per_unit(h, key, b, b, (m * CHECK_SPREAD.0, m * CHECK_SPREAD.1))
}

struct P {
    id: &'static str,
    label: String,
    r: R,
}

fn p(id: &'static str, label: impl Into<String>, r: R) -> P {
    P { id, label: label.into(), r }
}

fn start(label: &str, jobs: f64) -> P {
    p("start", label, fixed(START.0 * jobs, START.1 * jobs))
}

fn step(id: &str, label: impl Into<String>, parts: Vec<P>, note: Option<String>) -> StepTime {
    let (lo, hi) = parts.iter().fold((0.0, 0.0), |a, x| (a.0 + x.r.lo, a.1 + x.r.hi));
    // the step is "your history" when the parts that take most of it (the process start is not counted) come from the history
    let work: f64 = parts.iter().filter(|x| x.id != "start").map(|x| x.r.lo + x.r.hi).sum();
    let hist: f64 = parts.iter().filter(|x| x.id != "start" && x.r.hist).map(|x| x.r.lo + x.r.hi).sum();
    let basis = if hist > 0.0 && hist * 2.0 >= work { "your_history" } else { "formula" };
    StepTime {
        id: id.into(), label: label.into(), minutes: (r1(lo), r1(hi)), basis: basis.into(), note,
        parts: parts.into_iter().map(|x| StepPart { id: x.id.into(), label: x.label, minutes: (r1(x.r.lo), r1(x.r.hi)), basis: if x.r.hist { "your_history" } else { "formula" }.into() }).collect(),
    }
}

fn gb16(params: u64) -> f64 {
    params as f64 * 2.0 / 1e9
}

fn tag(of: u32, n: u32) -> String {
    if of > 1 { format!("Stage {n} of {of}: ") } else { String::new() }
}

/// The three jobs of a heal (the teacher's answers when they are saved first, the training, the merge) with `tokens_m` million tokens.
fn heal_parts(h: &History, ins: &Inspect, plan: &AutoPlan, g: &StagePlan, tokens_m: u32) -> (Vec<P>, Option<String>) {
    let (ob, sb) = (ins.total_params as f64 / 1e9, g.params as f64 / 1e9);
    let tokens = tokens_m as f64 * 1e6;
    let units = tokens / 5e6;
    let orig_gb = ins.total_params as f64 * if ins.bytes_per_param > 0.0 { ins.bytes_per_param } else { 2.0 } / 1e9;
    let cache = g.mode == "cache";
    let mut v = vec![start("start of the jobs and the evaluations", if cache { 3.0 } else { 2.0 })];
    let mut note = None;
    if cache {
        v.push(p("load", "load the original", load(h, orig_gb, &plan.measure_load)));
        let f = (CACHE_H_PER_5M.0 * 60.0 * units * ob / CACHE_REF_B, CACHE_H_PER_5M.1 * 60.0 * units * ob / CACHE_REF_B);
        v.push(p("cache", format!("the original writes its answers ({tokens_m} M tokens)"), at_rate(h, "teacher_tok_b_s", ob, tokens * ob, f)));
    } else {
        v.push(p("load", "load the original next to the student (4-bit)", load(h, orig_gb, "4bit")));
        note = Some("The card holds the original and the student together: the original answers while the student learns (no saved answers).".to_string());
    }
    v.push(p("load", "load the student", load(h, gb16(g.params), if sb >= 15.0 { "4bit" } else { "16bit" })));
    let train_f = (TRAIN_H_PER_5M.0 * 60.0 * units * sb / TRAIN_REF_B, TRAIN_H_PER_5M.1 * 60.0 * units * sb / TRAIN_REF_B);
    let train = if cache {
        at_rate(h, "train_tok_b_s", sb, tokens * sb, train_f)
    } else {
        // online: the student learns at 60 % of the speed (the plan's own rule); no earlier cache-mode training applies
        fixed(train_f.0 / 0.6, train_f.1 / 0.6)
    };
    v.push(p("train", format!("the student learns ({tokens_m} M tokens)"), train));
    v.push(p("merge", "add the learning into the model", per_unit(h, "merge_s_per_gb", gb16(g.params), gb16(g.params), (MERGE_MIN_PER_GB.0 * gb16(g.params), MERGE_MIN_PER_GB.1 * gb16(g.params)))));
    (v, note)
}

/// The time of the whole run for `plan` (an `ok` plan) and `spec`. `history` is what this computer's own saved job logs say (`History::default()`: nothing, every step is a formula).
pub fn time_estimate(ins: &Inspect, spec: &AutoSpec, plan: &AutoPlan, history: &History) -> TimeEstimate {
    let h = history;
    let ob = ins.total_params as f64 / 1e9;
    let orig_gb = ins.total_params as f64 * if ins.bytes_per_param > 0.0 { ins.bytes_per_param } else { 2.0 } / 1e9;
    let mode = plan.measure_load.as_str();
    let of = plan.stages.len() as u32;
    let thinking = spec.keep_thinking;
    let mut steps: Vec<StepTime> = Vec::new();
    let mut extra: Vec<ExtraTime> = Vec::new();

    steps.push(step("plan", "Check the model, the card and the disk", vec![p("check", "read the model's description", fixed(0.2, 0.5))], None));
    steps.push(step("measure", "Measure the model", vec![
        start("start of the job", 1.0), p("load", "load the original", load(h, orig_gb, mode)),
        p("measure", format!("score every block on {} texts", MEASURE_SAMPLES as u32), measure_pass(h, ob, MEASURE_SAMPLES, mode)),
    ], (mode == "offload").then(|| "The original does not fit the card: it is spread over the card and the PC's memory, which is several times slower.".to_string())));

    if !plan.try_list.is_empty() {
        let samples = spec.try_samples() as f64;
        let n = plan.try_list.len() + plan.try_list.iter().filter(|c| c.blocks > 0 && c.intermediate.is_some() && Some(c.intermediate.unwrap_or(0)) != auto_plan::full_width(ins)).count();
        let pass = forward_pass(h, ob, samples, mode);
        // the high end: the whole question asked again with 256 samples when the two halves of the text name different winners
        let (lo, hi) = (pass.lo * n as f64, pass.hi * n as f64 * 2.0);
        steps.push(step("try", "Try the cuts", vec![
            start("start of the job", 2.0), p("load", "load the original", load(h, orig_gb, mode)),
            P { id: "measure", label: format!("{n} passes of {} texts (the high end counts a repeat with 256)", samples as u32), r: R { lo, hi, hist: pass.hist } },
        ], None));

        if thinking && spec.think_gate {
            let k = (spec.think_gate_n as usize).min(plan.try_list.len());
            let mut parts = vec![start("start of the jobs", k as f64 + 1.0)];
            for c in plan.try_list.iter().take(k) {
                let gb = gb16(c.params);
                parts.push(p("write", format!("write the cut of {:.1} B", c.params_b), at_rate(h, "write_gb_s", gb, gb, (gb / WRITE_GB_S.0 / 60.0, gb / WRITE_GB_S.1 / 60.0))));
                parts.push(p("load", format!("load the cut of {:.1} B", c.params_b), load(h, gb, "16bit")));
                parts.push(p("check", format!("thinking check of the cut of {:.1} B", c.params_b), think_check(h, c.params as f64 / 1e9, spec.think_quick())));
            }
            parts.push(p("load", "load the original", load(h, orig_gb, mode)));
            parts.push(p("check", "thinking check of the original (the yardstick)", think_check(h, ob, spec.think_quick())));
            steps.push(step("think_gate", "Check the thinking of the best cuts before healing", parts, None));
        }
    }

    let mut prev_params = ins.total_params;
    for (i, g) in plan.stages.iter().enumerate() {
        let n = i as u32 + 1;
        let t = tag(of, n);
        let (sb, gb) = (g.params as f64 / 1e9, gb16(g.params));
        // the cut: load the model it is made from, measure it again (stages after the first), write the cut, check its loss (the first of one stage is the cut Try first measured)
        let (src_gb, src_b, src_mode) = if i == 0 { (orig_gb, ob, mode) } else { (gb16(prev_params), prev_params as f64 / 1e9, if mode == "offload" { "4bit" } else { mode }) };
        let mut parts = vec![start("start of the job", 1.0), p("load", "load the model to cut", load(h, src_gb, src_mode))];
        if i > 0 {
            parts.push(p("measure", "measure the healed model again", measure_pass(h, src_b, MEASURE_SAMPLES, src_mode)));
        }
        parts.push(p("write", format!("write the cut ({gb:.0} GB)"), at_rate(h, "write_gb_s", gb, gb, (gb / WRITE_GB_S.0 / 60.0, gb / WRITE_GB_S.1 / 60.0))));
        if !(i == 0 && of == 1) {
            parts.push(p("load", "load the cut to check its loss", load(h, gb, "16bit")));
            parts.push(p("check", "loss of the cut", forward_pass(h, sb, CUT_CHECK_SAMPLES, "16bit")));
        }
        steps.push(step(&format!("s{n}.cut"), format!("{t}cut"), parts, None));

        if thinking {
            let mut parts = vec![start("start of the job", 1.0), p("load", "load the cut", load(h, gb, "16bit"))];
            let mut note = None;
            if i == 0 && !(spec.think_gate && !plan.try_list.is_empty()) {
                parts.push(p("load", "load the original", load(h, orig_gb, mode)));
                parts.push(p("check", "thinking check of the original (the yardstick, once)", think_check(h, ob, spec.think_quick())));
                note = Some("The original is checked once, here, because the check before healing is off.".to_string());
            }
            parts.push(p("check", "thinking check of the cut", think_check(h, sb, spec.think_quick())));
            steps.push(step(&format!("s{n}.check_cut"), format!("{t}thinking check of the cut"), parts, note));
        }

        let (parts, note) = heal_parts(h, ins, plan, g, g.tokens_m);
        steps.push(step(&format!("s{n}.heal"), format!("{t}heal"), parts, note));

        if thinking {
            let parts = vec![start("start of the job", 1.0), p("load", "load the healed model", load(h, gb, "16bit")), p("check", "thinking check of the healed model", think_check(h, sb, spec.think_quick()))];
            steps.push(step(&format!("s{n}.check_heal"), format!("{t}thinking check of the healed model"), parts, None));
        }
        prev_params = g.params;
    }

    // the export: the last healed model to a 16-bit GGUF, then every size
    let final_params = plan.stages.last().map(|s| s.params).unwrap_or(ins.total_params);
    let (fb, f16_gb) = (final_params as f64 / 1e9, gb16(final_params));
    let quants = spec.export_quants_for(final_params);
    let plain: Vec<&String> = quants.iter().filter(|q| q.as_str() != "f16" && q.as_str() != "mixed_measured").collect();
    let measured = quants.iter().any(|q| q == "mixed_measured");
    let mut parts = vec![start("start of the job", 1.0), p("convert", format!("convert to a 16-bit GGUF ({f16_gb:.0} GB)"), fixed(CONVERT_MIN_PER_GB.0 * f16_gb, CONVERT_MIN_PER_GB.1 * f16_gb))];
    for q in &plain {
        parts.push(p("quantize", format!("make {}", q.to_uppercase()), fixed(QUANT_MIN_PER_GB.0 * f16_gb, QUANT_MIN_PER_GB.1 * f16_gb)));
    }
    if measured {
        let m = r1(2.0 + 0.25 * fb);
        parts.push(p("measure", "measure the fit to the target size", fixed(m * 0.8, m * 1.6)));
        parts.push(p("quantize", "make the measured file", fixed(QUANT_MIN_PER_GB.0 * f16_gb, QUANT_MIN_PER_GB.1 * f16_gb)));
    }
    steps.push(step("export", "Make the GGUF files", parts, None));
    if thinking {
        let parts = vec![start("start of the job", 1.0), p("check", "thinking check of the GGUF (llama.cpp)", {
            let c = think_check(&History::default(), fb, spec.think_quick());
            fixed(c.lo, c.hi)
        })];
        steps.push(step("check_gguf", "Thinking check of the GGUF", parts, Some("No earlier GGUF check of yours says how long it takes at this size: a formula.".to_string())));
    }
    if spec.target_gb.is_some() {
        let files = quants.iter().filter(|q| q.as_str() != "f16").count() as f64;
        let m = 4.0 + 3.0 * files;
        steps.push(step("compare_quants", "Compare the measured file with the plain sizes", vec![start("start of the job", 1.0), p("check", format!("compare {} files by KL divergence", files as u32), fixed(m * 0.8, m * 1.5))], None));
    }
    if spec.studio_publish {
        let gb: f64 = quants.iter().filter(|q| q.as_str() != "f16" && q.as_str() != "mixed_measured").map(|q| final_params as f64 * auto_plan::quant_bytes_per_param(q) / 1e9).sum();
        steps.push(step("studio_publish", "Put the files into Unsloth Studio", vec![p("copy", format!("copy {gb:.0} GB"), fixed(gb / COPY_GB_S.0 / 60.0, gb / COPY_GB_S.1 / 60.0 + 0.2))], None));
    }

    // what shortens the run most: the teacher's answers and the training
    let lever: (f64, f64) = steps.iter().filter(|s| s.id.ends_with(".heal")).flat_map(|s| s.parts.iter()).filter(|x| x.id == "cache" || x.id == "train").fold((0.0, 0.0), |a, x| (a.0 + x.minutes.0, a.1 + x.minutes.1));

    if thinking {
        if let Some(g) = plan.stages.last() {
            let (mut parts, _) = heal_parts(h, ins, plan, g, g.tokens_m + RETRY_TOKENS_M);
            let gb = gb16(g.params);
            parts.push(p("load", "load the healed model", load(h, gb, "16bit")));
            parts.push(p("check", "thinking check of the healed model", think_check(h, g.params as f64 / 1e9, spec.think_quick())));
            let s = step("retry_heal", "", parts, None);
            extra.push(ExtraTime {
                id: "retry_heal".into(), label: "If the last stage fails its thinking check: heal it again with 40 M more tokens and check it again".into(), minutes: s.minutes,
                note: "Only when a check fails, and only once per stage. It is not in the total.".into(),
            });
        }
    }

    let (lo, hi) = steps.iter().fold((0.0, 0.0), |a, s| (a.0 + s.minutes.0, a.1 + s.minutes.1));
    let history_steps = steps.iter().filter(|s| s.basis == "your_history").count() as u32;
    let step_count = steps.len() as u32;
    let basis = if history_steps == 0 {
        format!("None of the {step_count} steps could use your own earlier runs (there are none like these in the saved logs yet): every step is a formula built on one real run on a 32 GB card.")
    } else {
        format!("{history_steps} of {step_count} steps use your own earlier runs (from the saved job logs); the other {} are formulas built on one real run on a 32 GB card.", step_count - history_steps)
    };
    let mut notes = vec![
        "An estimate, not a promise: the teacher's answers and the training of the heal are most of it, and their speed depends on your card, the text and the settings.".to_string(),
        "Not in the total: waiting in the queue, downloading the model, a retry after a failed thinking check (shown apart), and the computer going to sleep.".to_string(),
    ];
    if plan.stages.iter().any(|s| s.mode == "online") {
        notes.push("The card holds the original and the student together in a stage: no saved answers, the original answers while the student learns.".to_string());
    }
    TimeEstimate { steps, total_h: (((lo / 60.0) * 100.0).round() / 100.0, ((hi / 60.0) * 100.0).round() / 100.0), history_steps, step_count, basis, lever_h: (r1(lever.0 / 60.0), r1(lever.1 / 60.0)), notes, extra }
}
