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
    /// every weight of the model is trained (needs far more memory; bf16 cards only)
    Full,
}

impl Method {
    /// the key used in measured.json and run states
    pub fn key(self) -> &'static str {
        match self {
            Method::Lora => "lora",
            Method::Qlora => "qlora",
            Method::Full => "full",
        }
    }
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
    /// "mixed": practice and world book together (default); "book_first": read the world book first, then practise (two stages)
    #[serde(default = "mixed")]
    pub stages: String,
    /// passes over the world book in the first stage of "book_first"
    #[serde(default = "two")]
    pub book_epochs: f64,
}

fn mixed() -> String {
    "mixed".into()
}

fn two() -> f64 {
    2.0
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
    /// world-book texts
    pub text_examples: u64,
    /// practice conversations
    pub chat_examples: u64,
}

pub fn base_hyper(preset: Preset) -> Hyper {
    base_hyper_for(Method::Lora, preset)
}

/// Starting settings by method and preset. Full fine-tuning uses a much smaller learning rate (every weight moves) and a batch of 1.
pub fn base_hyper_for(method: Method, preset: Preset) -> Hyper {
    if method == Method::Full {
        let f = |epochs: f64, max_steps, seq, warmup| Hyper {
            rank: 0, alpha: 0, lr: 1e-5, epochs, max_steps, batch: 1, grad_accum: 16, max_seq: seq, warmup, weight_decay: 0.01, scheduler: "linear".into(),
            seed: 3407, save_steps: 50, logging_steps: 1, stages: "mixed".into(), book_epochs: 2.0,
        };
        return match preset {
            Preset::Quick => Hyper { grad_accum: 4, ..f(1.0, Some(60), 1024, 5) },
            Preset::Normal => f(2.0, None, 2048, 10),
            Preset::Best => f(3.0, None, 4096, 10),
        };
    }
    let common = |rank, alpha, lr, epochs: f64, max_steps, batch, accum, seq, warmup| Hyper {
        rank, alpha, lr, epochs, max_steps, batch, grad_accum: accum, max_seq: seq, warmup, weight_decay: 0.001, scheduler: "linear".into(),
        seed: 3407, save_steps: 50, logging_steps: 1, stages: "mixed".into(), book_epochs: 2.0,
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
        Some((gb, s0, b0)) => gb * (0.85 + 0.15 * (seq as f64 / s0 as f64) * (batch as f64 / b0 as f64)),
        None => match method {
            Method::Lora => m.vram_lora_gb * scale(2048.0, 1.0),
            Method::Qlora => m.vram_qlora_gb * scale(2048.0, 1.0),
            // 16-bit weights + 16-bit gradients + 8-bit optimizer states = about 6 bytes per parameter, plus activations (gradient checkpointing);
            // Unsloth quotes full fine-tuning at about 4x the memory of LoRA, which agrees for the small models
            Method::Full => (m.params_b * 6.0 + 2.0) * scale(2048.0, 1.0),
        },
    }
}

/// Disk needed for a full fine-tune: the model, its checkpoint (weights + optimizer) and the saved result.
pub fn full_disk_gb(m: &ModelEntry) -> f64 {
    (m.params_b * 2.0 * 4.0 * 10.0).round() / 10.0
}

/// Why full fine-tuning is not offered here (empty = it is). `bf16`: the card supports bf16 (None = unknown, not a blocker).
pub fn full_blockers(m: &ModelEntry, vram_gb: Option<f64>, bf16: Option<bool>, remote: bool) -> Vec<String> {
    let mut why = Vec::new();
    if remote {
        why.push("Full fine-tuning runs on this computer only; a training server does LoRA / QLoRA.".to_string());
    }
    if bf16 == Some(false) {
        why.push("This graphics card has no bf16 (RTX 30 series or newer is needed for full fine-tuning). Use LoRA here.".to_string());
    }
    if let Some(v) = vram_gb {
        // the smallest settings it can be lowered to: batch 1, 1024 tokens
        let least = estimate_vram_gb(m, Method::Full, 1, 1024, None);
        if least > v * 0.97 {
            why.push(format!("{} needs about {:.0} GB for full fine-tuning even with the smallest settings; this card has {:.0} GB. LoRA needs about {:.0} GB.", m.name, least, v, m.vram_lora_gb));
        }
    }
    why
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
    if h.stages == "book_first" && data.text_examples > 0 && data.chat_examples > 0 {
        let book = (data.text_examples as f64 * h.book_epochs / per_step).ceil();
        let practice = (data.chat_examples as f64 * h.epochs / per_step).ceil();
        return ((book + practice) as u32).max(1);
    }
    ((data.examples as f64 * h.epochs / per_step).ceil() as u32).max(1)
}

pub fn plan(m: &ModelEntry, method: Method, preset: Preset, vram_gb: Option<f64>, data: DataStats, measured: Option<(f64, u32, u32)>) -> Plan {
    let mut h = base_hyper_for(method, preset);
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
            Method::Full => "Does not fit even with the smallest settings. Use LoRA or a smaller model.".to_string(),
        });
    }
    if method == Method::Full {
        notes.push(format!("Full fine-tuning changes every weight: it learns the world strongly but can forget how to chat if pushed too far, so the learning rate is small ({}).", h.lr));
        notes.push(format!("It needs about {:.0} GB of free disk space for checkpoints, and the result is a full copy of the model (about {:.0} GB) instead of a small add-on.", full_disk_gb(m), m.params_b * 2.0));
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
