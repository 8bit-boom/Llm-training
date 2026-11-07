//! Presets and memory estimates for PEFT (LoRA / QLoRA) + SFT training. Pure functions: the same inputs always give the same plan.
//!
//! Memory: the catalog holds the figures Unsloth publishes for about 2048 tokens; the estimate scales them with sequence length and batch size
//! (a rough rule, shown as an estimate) and a figure MEASURED on this PC replaces it. The plan lowers the batch size and then the sequence
//! length until the model fits, and says what it changed.
use serde::{Deserialize, Serialize};

use super::catalog::ModelEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Method {
    /// 16-bit LoRA (the default)
    Lora,
    /// 4-bit base model + LoRA (needs less memory)
    Qlora,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    Quick,
    Normal,
    Best,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hyper {
    pub rank: u32,
    pub alpha: u32,
    pub lr: f64,
    pub epochs: f64,
    /// a fixed number of steps (the quick test) instead of epochs
    pub max_steps: Option<u32>,
    pub batch: u32,
    pub grad_accum: u32,
    pub max_seq: u32,
    pub warmup: u32,
    pub weight_decay: f64,
    pub scheduler: String,
    pub seed: u64,
    pub save_steps: u32,
    pub logging_steps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fit {
    /// comfortably within the card
    Yes,
    /// close to the limit
    Tight,
    No,
    /// no card found
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub hyper: Hyper,
    pub est_vram_gb: f64,
    pub fit: Fit,
    pub steps: u32,
    /// plain-language notes: what was lowered to fit, cautions about the method
    pub notes: Vec<String>,
}

/// What the training file holds (from the saved version's manifest).
#[derive(Debug, Clone, Copy, Default)]
pub struct DataStats {
    pub examples: u64,
    pub tokens: u64,
}

pub fn base_hyper(preset: Preset) -> Hyper {
    let common = |rank, alpha, lr, epochs: f64, max_steps, batch, accum, seq, warmup| Hyper {
        rank, alpha, lr, epochs, max_steps, batch, grad_accum: accum, max_seq: seq, warmup, weight_decay: 0.001, scheduler: "linear".into(),
        seed: 3407, save_steps: 50, logging_steps: 1,
    };
    match preset {
        // proves the whole chain works in minutes; the model will not have learned much
        Preset::Quick => common(8, 8, 2e-4, 1.0, Some(60), 1, 4, 1024, 5),
        Preset::Normal => common(16, 16, 1e-4, 2.0, None, 2, 8, 2048, 10),
        Preset::Best => common(32, 32, 1e-4, 3.0, None, 2, 8, 4096, 10),
    }
}

/// GB needed at a given batch size and sequence length. `measured_gb` (peak memory of a finished run with `measured_seq` / `measured_batch`) wins.
pub fn estimate_vram_gb(m: &ModelEntry, method: Method, batch: u32, seq: u32, measured: Option<(f64, u32, u32)>) -> f64 {
    let scale = |seq0: f64, batch0: f64| 0.85 + 0.15 * (seq as f64 / seq0) * (batch as f64 / batch0);
    match measured {
        Some((gb, s0, b0)) => gb * (0.85 + 0.15 * (seq as f64 / s0 as f64) * (batch as f64 / b0 as f64)) / 1.0,
        None => (if method == Method::Lora { m.vram_lora_gb } else { m.vram_qlora_gb }) * scale(2048.0, 1.0),
    }
}

fn fit_of(est: f64, vram_gb: Option<f64>) -> Fit {
    match vram_gb {
        None => Fit::Unknown,
        Some(v) if est <= v * 0.85 => Fit::Yes,
        Some(v) if est <= v * 0.97 => Fit::Tight,
        Some(_) => Fit::No,
    }
}

pub fn steps_for(h: &Hyper, data: DataStats) -> u32 {
    if let Some(s) = h.max_steps {
        return s;
    }
    let per_step = (h.batch * h.grad_accum).max(1) as f64;
    ((data.examples as f64 * h.epochs / per_step).ceil() as u32).max(1)
}

pub fn plan(m: &ModelEntry, method: Method, preset: Preset, vram_gb: Option<f64>, data: DataStats, measured: Option<(f64, u32, u32)>) -> Plan {
    let mut h = base_hyper(preset);
    let mut notes = Vec::new();
    let est = |h: &Hyper| estimate_vram_gb(m, method, h.batch, h.max_seq, measured);
    // lower the batch size first, then the sequence length (never below 1024), until it fits
    if vram_gb.is_some() && fit_of(est(&h), vram_gb) == Fit::No && h.batch > 1 {
        h.grad_accum *= h.batch;
        h.batch = 1;
        notes.push("Batch size lowered to 1 (same effective batch through more accumulation) to fit the graphics memory.".into());
    }
    while vram_gb.is_some() && fit_of(est(&h), vram_gb) == Fit::No && h.max_seq > 1024 {
        h.max_seq /= 2;
        notes.push(format!("Sequence length lowered to {} tokens to fit the graphics memory.", h.max_seq));
    }
    let e = est(&h);
    let fit = fit_of(e, vram_gb);
    if fit == Fit::No {
        notes.push(match method {
            Method::Lora => "Does not fit even with the smallest settings. Try 4-bit (QLoRA) or a smaller model.".to_string(),
            Method::Qlora => "Does not fit even with 4-bit. Choose a smaller model.".to_string(),
        });
    }
    if method == Method::Qlora && m.family == "qwen3_5" {
        notes.push("Unsloth advises against 4-bit training for Qwen3.5 (more quality loss than usual). Use 16-bit LoRA if it fits.".into());
    }
    if m.family == "gemma4" {
        notes.push("A training loss of 13-15 is normal for Gemma 4 E2B/E4B; it is not a problem.".into());
    }
    let steps = steps_for(&h, data);
    h.save_steps = (steps / 10).clamp(10, 200);
    Plan { hyper: h, est_vram_gb: (e * 10.0).round() / 10.0, fit, steps, notes }
}

/// The method to suggest for this card: 16-bit LoRA when it fits, else 4-bit.
pub fn recommended_method(m: &ModelEntry, vram_gb: Option<f64>) -> Method {
    match vram_gb {
        Some(v) if estimate_vram_gb(m, Method::Lora, 1, 2048, None) > v * 0.97 => Method::Qlora,
        _ => Method::Lora,
    }
}
