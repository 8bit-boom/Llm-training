//! A training run in Unsloth: send the data, follow the job, bring the adapter back (see `docs/UNSLOTH_DESKTOP.md`).
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::train::catalog::ModelEntry;
use crate::train::plan::{Hyper, Method};

/// The Unsloth the user entered. The key is kept in `secrets` under `unsloth:<url>`.
#[derive(Debug, Clone)]
pub struct UnslothTarget {
    pub name: String,
    pub url: String,
    pub key: String,
}

/// The `server` value the Train page uses for Unsloth (a training server is named by its address).
pub const SERVER_ID: &str = "unsloth";

/// A name Unsloth accepts as the job's suffix: letters, digits, `_` and `-`, at most 40 characters.
pub fn suffix_of(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' }).collect();
    s.chars().take(40).collect::<String>().trim_matches('-').to_string()
}

/// The body of `POST /v1/fine_tuning/jobs` (the contract document lists every field). Every key is one Unsloth knows: it refuses others.
/// `load_in_4bit` is always sent: Unsloth's own default is `true`, which would turn every LoRA run into QLoRA.
pub fn job_body(m: &ModelEntry, method: Method, hyper: &Hyper, suffix: &str, train_id: &str, eval_id: Option<&str>) -> Result<Value> {
    if method == Method::Full {
        return Err(Error::new("Full fine-tuning runs on this computer only. Choose LoRA or QLoRA, or train on this computer."));
    }
    if hyper.stages == "book_first" && hyper.max_steps.is_none() {
        return Err(Error::new("Two stages (world book first) run on this computer or the TrueNAS server only. Choose \"mixed\" or train on this computer."));
    }
    let mut b = json!({
        "model": m.repo,
        "training_file": train_id,
        "hyperparameters": {"n_epochs": if hyper.max_steps.is_some() { 1 } else { (hyper.epochs.round() as u32).max(1) }, "batch_size": hyper.batch, "learning_rate_multiplier": hyper.lr / 2e-4},
        "method": {"type": "supervised"},
        "suffix": suffix_of(suffix),
        "seed": hyper.seed,
        "training_type": "LoRA/QLoRA",
        "load_in_4bit": method == Method::Qlora,
        "max_seq_length": hyper.max_seq,
        "gradient_accumulation_steps": hyper.grad_accum,
        "warmup_steps": hyper.warmup,
        "max_steps": hyper.max_steps.unwrap_or(0),
        "save_steps": hyper.save_steps,
        "packing": false,
        "train_on_completions": true,
        "gradient_checkpointing": "unsloth",
        "lora_r": hyper.rank,
        "lora_alpha": hyper.alpha,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "finetune_vision_layers": false,
        "finetune_language_layers": true,
        "finetune_attention_modules": true,
        "finetune_mlp_modules": true,
    });
    if let Some(e) = eval_id {
        b["validation_file"] = json!(e);
        b["eval_steps"] = json!(0.1);
    }
    Ok(b)
}
