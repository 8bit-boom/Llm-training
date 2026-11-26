//! How "best" is decided: pure rules, the same numbers the worker uses (`worker/ndforge_worker/healspeed.py`; a Python test reads this file and compares the constants).
//! A candidate must be feasible (peak memory, an out-of-memory is a row, the guard of the maths), and it must beat the default by more than the measured noise.
use serde_json::{json, Value};

use super::preset::Guard;

pub const VRAM_SHARE_LIMIT: f64 = 0.90;
pub const LOSS_TOLERANCE: f64 = 1e-3;
pub const MIN_GAIN: f64 = 1.05;
pub const MIN_GAIN_ENGINE: f64 = 1.10;
pub const NOISE_FACTOR: f64 = 2.0;
/// The noise assumed when nothing was repeated (the training benchmark's own 3 % rule).
pub const DEFAULT_SPREAD: f64 = 0.03;

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// the value that wins (0 / the default when nothing beats it)
    pub chosen: u32,
    pub gain: f64,
    pub spread: f64,
    pub why: String,
    pub guards: Vec<Guard>,
    pub changed: bool,
}

/// The gain a candidate needs: the larger of the minimum and (1 + noise factor x spread).
pub fn needed_gain(spread: f64, min: f64) -> f64 {
    min.max(1.0 + NOISE_FACTOR * spread)
}

fn f(v: &Value, k: &str) -> Option<f64> {
    v[k].as_f64()
}

/// `heal.student_batch`: the rows of the token-budget sweep (`batch_sweep.results` of the worker: `batch_tokens`, `tokens_per_s`, `peak_vram_mb`, `worst_peak_mb`, `first_loss`, `error`, `oom`) -> the budget to use.
/// Feasible: no error, peak (the worst-case update included) below 90 % of the card, the first-update loss within 1e-3 (relative) of the one-row-per-step run. The best feasible candidate must beat the baseline by
/// `needed_gain(spread)`; a tie (within the noise of the best) goes to the lower peak memory.
pub fn decide_batch(results: &[Value], card_mb: Option<f64>, spread: f64) -> Decision {
    let base = results.iter().find(|r| r["batch_tokens"] == 0 && f(r, "tokens_per_s").is_some_and(|t| t > 0.0));
    let Some(base) = base else {
        return Decision { chosen: 0, gain: 1.0, spread, why: "The measurement with one row per forward pass is missing, so nothing can be compared.".into(), guards: vec![], changed: false };
    };
    let base_tps = f(base, "tokens_per_s").unwrap_or(1.0);
    let base_loss = f(base, "first_loss");
    let peak = |r: &Value| f(r, "peak_vram_mb").unwrap_or(0.0).max(f(r, "worst_peak_mb").unwrap_or(0.0));
    let mut excluded: Vec<String> = Vec::new();
    let mut feasible: Vec<&Value> = Vec::new();
    for r in results.iter().filter(|r| r["batch_tokens"] != 0) {
        let Some(tps) = f(r, "tokens_per_s").filter(|t| *t > 0.0) else { continue };
        if r["error"].is_string() {
            continue;
        }
        if let (Some(card), p) = (card_mb, peak(r)) {
            if p > VRAM_SHARE_LIMIT * card {
                excluded.push(format!("{}: peak {:.0} MB is above 90 % of the card", r["batch_tokens"], p));
                continue;
            }
        }
        if let (Some(a), Some(b)) = (base_loss, f(r, "first_loss")) {
            if (a - b).abs() > LOSS_TOLERANCE * a.abs().max(1.0) {
                excluded.push(format!("{}: faster but NOT equivalent, the first loss differs by {:.1e}", r["batch_tokens"], (a - b).abs()));
                continue;
            }
        }
        let _ = tps;
        feasible.push(r);
    }
    // the fastest feasible candidate; candidates within the noise of it are ties, and a tie goes to the lower peak memory (more headroom for longer rows later)
    let top = feasible.iter().map(|r| f(r, "tokens_per_s").unwrap_or(0.0)).fold(0.0, f64::max);
    let best: Option<&Value> = feasible.iter().copied().filter(|r| f(r, "tokens_per_s").unwrap_or(0.0) >= top * (1.0 - spread)).min_by(|a, b| peak(a).partial_cmp(&peak(b)).unwrap_or(std::cmp::Ordering::Equal));
    let guards = |chosen: Option<&Value>| -> Vec<Guard> {
        let mut g = Vec::new();
        if let Some(c) = chosen {
            g.push(Guard { name: "peak_vram".into(), ok: card_mb.map(|m| peak(c) <= VRAM_SHARE_LIMIT * m).unwrap_or(true), value: card_mb.map(|m| (peak(c) / m * 1000.0).round() / 1000.0), limit: Some(VRAM_SHARE_LIMIT) });
            if let (Some(a), Some(b)) = (base_loss, f(c, "first_loss")) {
                g.push(Guard { name: "first_loss".into(), ok: (a - b).abs() <= LOSS_TOLERANCE * a.abs().max(1.0), value: Some((a - b).abs()), limit: Some(LOSS_TOLERANCE) });
            }
        }
        g
    };
    let need = needed_gain(spread, MIN_GAIN);
    match best {
        Some(b) if f(b, "tokens_per_s").unwrap_or(0.0) / base_tps >= need => {
            let gain = (f(b, "tokens_per_s").unwrap_or(0.0) / base_tps * 100.0).round() / 100.0;
            Decision { chosen: b["batch_tokens"].as_u64().unwrap_or(0) as u32, gain, spread, why: format!("{} tokens per forward pass is {gain} times as fast as one row (needed {:.2}, noise {:.1} %).", b["batch_tokens"], need, spread * 100.0), guards: guards(Some(b)), changed: true }
        }
        other => {
            let gain = other.map(|b| (f(b, "tokens_per_s").unwrap_or(0.0) / base_tps * 100.0).round() / 100.0).unwrap_or(1.0);
            let mut why = format!("Within noise (±{:.1} %): one row per forward pass stays.", spread * 100.0);
            if other.is_none() {
                why = "No token budget fits and gives the same results: one row per forward pass stays.".into();
            }
            if !excluded.is_empty() {
                why += &format!(" Excluded: {}.", excluded.join("; "));
            }
            Decision { chosen: 0, gain: gain.max(1.0), spread, why, guards: guards(None), changed: false }
        }
    }
}

/// `teacher.standard`: the same sweep rows as `decide_batch` (budget 0 = the fixed rows per pass), but the comparison that counts is with the budget the person has NOW (`current`; Settings default 16,384): the
/// winner must beat that by `needed_gain(spread)`, else the current value stays (changed: false). A current value that is clearly slower than the fixed rows goes back to 0.
pub fn decide_teacher(results: &[Value], card_mb: Option<f64>, spread: f64, current: u32) -> Decision {
    decide_sized(results, card_mb, spread, current, "tokens per forward pass")
}

/// The same rule for any batch-size sweep (`batch_tokens` = the size, 0 = the program's own number): `unit` is what the number counts, for the words ("rows together", "questions together").
pub fn decide_sized(results: &[Value], card_mb: Option<f64>, spread: f64, current: u32, unit: &str) -> Decision {
    let dec = decide_batch(results, card_mb, spread);
    let tps_of = |b: u32| results.iter().find(|r| r["batch_tokens"].as_u64() == Some(b as u64) && r["error"].is_null()).and_then(|r| f(r, "tokens_per_s")).filter(|t| *t > 0.0);
    let (Some(base), cur) = (tps_of(0), tps_of(current)) else { return dec };
    let need = needed_gain(spread, MIN_GAIN);
    let cur_tps = cur.unwrap_or(base);
    let best_tps = tps_of(dec.chosen).unwrap_or(base);
    let now = if current == 0 { "program's own number".to_string() } else { format!("{current} {unit}") };
    if dec.chosen != current && best_tps / cur_tps >= need {
        let gain = (best_tps / cur_tps * 100.0).round() / 100.0;
        return Decision { gain, why: format!("{} {unit} is {gain} times as fast as your {now} (needed {need:.2}, noise {:.1} %).", dec.chosen, spread * 100.0), changed: true, ..dec };
    }
    let mut why = format!("Within noise (±{:.1} %) of your {now}: it stays.", spread * 100.0);
    if let Some(e) = dec.why.split(" Excluded: ").nth(1) {
        why += &format!(" Excluded: {e}");
    }
    Decision { chosen: current, gain: 1.0, spread, why, guards: dec.guards, changed: false }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LossDecision {
    pub chunk: u32,
    pub head_checkpoint: bool,
    pub gain: f64,
    pub why: String,
    pub changed: bool,
    pub baseline_tps: f64,
    pub best_tps: f64,
}

/// `heal.loss`: the rows of `heal_bench` (`id` "current" is the baseline; `chunk`, `head_checkpoint` ("on"/"off" as a bool), `tokens_per_s`, `ok`) -> the chunk and the head setting. The maths is the same in all
/// rows (recomputation and chunking), so only speed counts; the best row must beat "your settings" by `needed_gain(spread)`.
pub fn decide_loss(rows: &[Value], spread: f64) -> LossDecision {
    let ok: Vec<&Value> = rows.iter().filter(|r| r["ok"] == true && f(r, "tokens_per_s").is_some_and(|t| t > 0.0)).collect();
    let Some(base) = ok.iter().find(|r| r["id"] == "current") else {
        return LossDecision { chunk: 0, head_checkpoint: true, gain: 1.0, why: "Your own settings did not finish, so nothing can be compared.".into(), changed: false, baseline_tps: 0.0, best_tps: 0.0 };
    };
    let base_tps = f(base, "tokens_per_s").unwrap_or(1.0);
    let spread = if spread > 0.0 { spread } else { DEFAULT_SPREAD };
    let mut best: &Value = base;
    for r in &ok {
        if f(r, "tokens_per_s").unwrap_or(0.0) > f(best, "tokens_per_s").unwrap_or(0.0) {
            best = r;
        }
    }
    let best_tps = f(best, "tokens_per_s").unwrap_or(0.0);
    let need = needed_gain(spread, MIN_GAIN);
    if best_tps / base_tps >= need && best["id"] != "current" {
        let gain = (best_tps / base_tps * 100.0).round() / 100.0;
        LossDecision { chunk: best["chunk"].as_u64().unwrap_or(0) as u32, head_checkpoint: best["head_checkpoint"] != false, gain, why: format!("KL chunk {} with the output layer's logits {} is {gain} times as fast as yours (needed {need:.2}).", best["chunk"], if best["head_checkpoint"] == false { "kept" } else { "made again" }), changed: true, baseline_tps: base_tps, best_tps }
    } else {
        LossDecision { chunk: base["chunk"].as_u64().unwrap_or(0) as u32, head_checkpoint: base["head_checkpoint"] != false, gain: 1.0, why: format!("Within noise (±{:.1} %): your loss settings stay.", spread * 100.0), changed: false, baseline_tps: base_tps, best_tps }
    }
}

pub fn constants_json() -> Value {
    json!({"VRAM_SHARE_LIMIT": VRAM_SHARE_LIMIT, "LOSS_TOLERANCE": LOSS_TOLERANCE, "MIN_GAIN": MIN_GAIN, "NOISE_FACTOR": NOISE_FACTOR})
}
