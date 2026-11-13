//! Sizes, limits and checks for slimming or growing a model. Pure functions: no files, no network.
//!
//! The worker (`reshape_inspect`) reports what a model is made of; everything the page shows before the job starts is computed here from that.
//! Layers are cut / copied only in whole blocks of one pattern period (Qwen3.5: 3 linear-attention + 1 full-attention layer = 4 layers). The rules
//! for `limits` are the same as `reshape.limits` in the Python worker; `worker/tests/data/reshape_limits.json` is read by both test suites.
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::train::hub::estimate_vram;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Prune,
    Grow,
}

impl Mode {
    pub fn key(self) -> &'static str {
        match self {
            Mode::Prune => "prune",
            Mode::Grow => "grow",
        }
    }
    /// "slim" / "grow" as used in folder names and the list of made models
    pub fn word(self) -> &'static str {
        match self {
            Mode::Prune => "slim",
            Mode::Grow => "grow",
        }
    }
}

/// What the worker's `reshape_inspect` job found out about a model (extra keys it sends are ignored).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Inspect {
    #[serde(default)]
    pub model_type: String,
    pub supported: bool,
    #[serde(default)]
    pub why: Option<String>,
    pub layers: u32,
    #[serde(default)]
    pub layer_types: Vec<String>,
    pub period: u32,
    pub blocks: u32,
    #[serde(default)]
    pub layer_params: Vec<u64>,
    #[serde(default)]
    pub embed_params: u64,
    #[serde(default)]
    pub lm_head_params: u64,
    #[serde(default)]
    pub vision_params: u64,
    #[serde(default)]
    pub mtp_params: u64,
    #[serde(default)]
    pub other_params: u64,
    pub total_params: u64,
    #[serde(default)]
    pub bytes_per_param: f64,
    #[serde(default)]
    pub has_vision: bool,
    #[serde(default)]
    pub tied: bool,
    #[serde(default)]
    pub source_bytes: u64,
    #[serde(default)]
    pub local: bool,
}

fn contiguous() -> String {
    "contiguous".into()
}
fn one() -> u32 {
    1
}
fn identity() -> String {
    "identity".into()
}
fn s128() -> u32 {
    128
}
fn s512() -> u32 {
    512
}
fn auto() -> String {
    "auto".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReshapeSpec {
    pub source_model_id: String,
    pub mode: Mode,
    /// whole blocks to drop (prune) or add (grow)
    pub blocks: u32,
    /// prune: "contiguous" (one run of blocks) | "separate" (the least useful blocks wherever they are)
    #[serde(default = "contiguous")]
    pub strategy: String,
    #[serde(default = "one")]
    pub protect_first: u32,
    #[serde(default = "one")]
    pub protect_last: u32,
    /// grow: "identity" (the new layers start switched off: an exact copy of the original) | "copy" (SOLAR style)
    #[serde(default = "identity")]
    pub init: String,
    #[serde(default = "s128")]
    pub calib_samples: u32,
    #[serde(default = "s512")]
    pub calib_len: u32,
    /// how the model is loaded for measuring: "auto" | "16bit" | "4bit" | "offload"
    #[serde(default = "auto")]
    pub load: String,
    /// expert: explicit old layer indices to remove (whole blocks); skips the measuring
    #[serde(default)]
    pub drop_layers: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SizeOption {
    pub blocks: u32,
    pub layers: u32,
    pub params_b: f64,
    pub disk_gb: f64,
    pub vram_lora_gb: f64,
    pub vram_qlora_gb: f64,
    pub fits_lora: Option<bool>,
    pub fits_qlora: Option<bool>,
}

/// (min, max) blocks that can be dropped / added; max is 0 when nothing is feasible. Same rule as `reshape.limits` in the worker.
pub fn limits(ins: &Inspect, mode: Mode, protect_first: u32, protect_last: u32) -> (u32, u32) {
    let nb = ins.blocks;
    let max = match mode {
        Mode::Prune => (nb / 2).min(nb.saturating_sub(protect_first).saturating_sub(protect_last)),
        Mode::Grow => nb / 2,
    };
    (1, max)
}

fn sum(v: &[u64]) -> u64 {
    v.iter().sum()
}

/// Grow ("solar"): the layers `half - m .. half` appear twice. Returns that span (start, end) in old layer numbers, as the worker's `grow_src` makes it.
fn grow_span(ins: &Inspect, blocks: u32) -> Option<(usize, usize)> {
    let (l, p) = (ins.layers as usize, ins.period.max(1) as usize);
    let m = blocks as usize * p;
    if m == 0 || m > l / 2 {
        return None;
    }
    let mut half = (l + m) / 2;
    half -= half % p;
    half.checked_sub(m).map(|start| (start, half))
}

/// Parameters after the change. Grow is exact (the duplicated layers are known); prune subtracts the mean block (which blocks go is decided by the measuring).
pub fn params_after(ins: &Inspect, mode: Mode, blocks: u32) -> u64 {
    match mode {
        Mode::Prune => {
            let mean = if ins.blocks == 0 { 0.0 } else { sum(&ins.layer_params) as f64 / ins.blocks as f64 };
            ins.total_params.saturating_sub((mean * blocks as f64).round() as u64)
        }
        Mode::Grow => match grow_span(ins, blocks) {
            Some((a, b)) => ins.total_params + sum(&ins.layer_params[a.min(ins.layer_params.len())..b.min(ins.layer_params.len())]),
            None => ins.total_params,
        },
    }
}

fn layers_after(ins: &Inspect, mode: Mode, blocks: u32) -> u32 {
    match mode {
        Mode::Prune => ins.layers.saturating_sub(blocks * ins.period),
        Mode::Grow => ins.layers + blocks * ins.period,
    }
}

fn bytes_per_param(ins: &Inspect) -> f64 {
    if ins.bytes_per_param > 0.0 { ins.bytes_per_param } else { 2.0 }
}

/// Every feasible block count with the size it gives and whether training it fits `vram_gb`.
pub fn options(ins: &Inspect, mode: Mode, vram_gb: Option<f64>) -> Vec<SizeOption> {
    if !ins.supported {
        return Vec::new();
    }
    let (lo, hi) = limits(ins, mode, 1, 1);
    (lo..=hi).map(|blocks| {
        let params = params_after(ins, mode, blocks);
        let params_b = params as f64 / 1e9;
        let (lora, qlora) = estimate_vram(params_b);
        SizeOption {
            blocks, layers: layers_after(ins, mode, blocks), params_b: (params_b * 100.0).round() / 100.0,
            disk_gb: ((params as f64 * bytes_per_param(ins)) / 1e9 * 10.0).round() / 10.0,
            vram_lora_gb: lora, vram_qlora_gb: qlora, fits_lora: vram_gb.map(|v| lora <= v), fits_qlora: vram_gb.map(|v| qlora <= v),
        }
    }).collect()
}

/// The feasible block count whose result is nearest to `target_b` billion parameters (0 when nothing is feasible).
pub fn blocks_for_target(ins: &Inspect, mode: Mode, target_b: f64) -> u32 {
    let (lo, hi) = limits(ins, mode, 1, 1);
    if !ins.supported || hi < lo {
        return 0;
    }
    (lo..=hi).min_by(|a, b| {
        let (da, db) = ((params_after(ins, mode, *a) as f64 / 1e9 - target_b).abs(), (params_after(ins, mode, *b) as f64 / 1e9 - target_b).abs());
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b))
    }).unwrap_or(0)
}

/// How the model is loaded for measuring ("auto" in the worker): "16bit" when the weights fit the card with room to spare, else "4bit" (NF4) when
/// they fit that way, else "offload" (spread over the card and the PC's memory). Without a known card: "4bit".
pub fn scoring_load(params_total: u64, bytes_per_param: f64, vram_gb: Option<f64>) -> &'static str {
    let Some(vram) = vram_gb else { return "4bit" };
    let est = params_total as f64 * if bytes_per_param > 0.0 { bytes_per_param } else { 2.0 };
    let budget = vram * 1024f64.powi(3) * 0.85;
    if est <= budget {
        "16bit"
    } else if est / 3.6 <= budget {
        "4bit"
    } else {
        "offload"
    }
}

/// The disk space the new model needs (with 10 % to spare); a model that is not on this PC yet also needs its own download.
pub fn disk_needed_gb(ins: &Inspect, mode: Mode, blocks: u32) -> f64 {
    let bpp = bytes_per_param(ins);
    let out = params_after(ins, mode, blocks) as f64 * bpp * 1.1;
    let source = if ins.local { 0.0 } else if ins.source_bytes > 0 { ins.source_bytes as f64 } else { ins.total_params as f64 * bpp };
    ((out + source) / 1e9 * 10.0).round() / 10.0
}

const TOO_MANY: &str = "Slim at most half of the layers, and never the first or last block.";

/// Refuses what the worker would refuse, in plain words.
pub fn validate(spec: &ReshapeSpec, ins: &Inspect) -> Result<()> {
    if !ins.supported {
        return Err(Error::new(ins.why.clone().unwrap_or_else(|| "Only Qwen3.5 and plain Llama/Qwen/Mistral-style models can be slimmed or grown so far.".into())));
    }
    if !["contiguous", "separate"].contains(&spec.strategy.as_str()) {
        return Err(Error::new(format!("Unknown way of choosing layers: {}.", spec.strategy)));
    }
    if !["identity", "copy"].contains(&spec.init.as_str()) {
        return Err(Error::new(format!("Unknown start for the new layers: {}.", spec.init)));
    }
    if !["auto", "16bit", "4bit", "offload"].contains(&spec.load.as_str()) {
        return Err(Error::new(format!("Unknown way of loading the model for measuring: {}.", spec.load)));
    }
    if spec.calib_samples < 8 || spec.calib_samples > 4096 || spec.calib_len < 32 || spec.calib_len > 8192 {
        return Err(Error::new("Measure on 8 to 4096 samples of 32 to 8192 tokens."));
    }
    let (lo, hi) = limits(ins, spec.mode, spec.protect_first, spec.protect_last);
    match (&spec.drop_layers, spec.mode) {
        (Some(_), Mode::Grow) => return Err(Error::new("Choosing layers by number is only for slimming.")),
        (Some(drop), Mode::Prune) => {
            let p = ins.period.max(1);
            let mut d = drop.clone();
            d.sort_unstable();
            d.dedup();
            let (first, last) = (d.first().copied(), d.last().copied());
            let (Some(first), Some(last)) = (first, last) else { return Err(Error::new("No layers to remove were given.")) };
            if last >= ins.layers {
                return Err(Error::new(format!("Layers must be between 0 and {}.", ins.layers.saturating_sub(1))));
            }
            let blocks: Vec<u32> = { let mut b: Vec<u32> = d.iter().map(|x| x / p).collect(); b.dedup(); b };
            let whole = d.len() as u32 == blocks.len() as u32 * p;
            if !whole {
                return Err(Error::new(format!("Layers {first} to {last} do not form whole blocks of {p}{}. Remove whole blocks so the model keeps its pattern.",
                    if p == 4 { " (3 fast + 1 full attention)" } else { "" })));
            }
            if blocks[0] < spec.protect_first || blocks[blocks.len() - 1] + spec.protect_last >= ins.blocks {
                return Err(Error::new("The first and the last block are protected and cannot be removed."));
            }
            if blocks.len() as u32 > ins.blocks / 2 {
                return Err(Error::new(TOO_MANY));
            }
        }
        (None, _) => {
            if hi < lo || spec.blocks < lo || spec.blocks > hi {
                return Err(Error::new(if spec.mode == Mode::Prune { TOO_MANY } else { "Grow by at most half of the layers." }));
            }
        }
    }
    Ok(())
}

/// Things worth knowing before pressing the button.
pub fn warnings(spec: &ReshapeSpec, ins: &Inspect, vram_gb: Option<f64>) -> Vec<String> {
    let mut out = Vec::new();
    let blocks = match (&spec.drop_layers, spec.mode) {
        (Some(d), Mode::Prune) => (d.len() as u32 / ins.period.max(1)).max(1),
        _ => spec.blocks,
    };
    let after = params_after(ins, spec.mode, blocks);
    match spec.mode {
        Mode::Prune => {
            if blocks * 4 > ins.blocks {
                out.push("More than a quarter of the layers are removed. Answers get clearly worse until the model has been trained again on your world.".to_string());
            }
            out.push("A slimmed model is worse than the original until it is trained again (Step 3). Test it first, then train it.".to_string());
        }
        Mode::Grow => {
            let more = if ins.total_params == 0 { 0.0 } else { (after as f64 / ins.total_params as f64 - 1.0) * 100.0 };
            out.push(format!("Expect little or no gain from your world's data alone: it uses about {more:.0} % more memory, disk space and time. Growing pays off only with much more training text."));
        }
    }
    if let Some(v) = vram_gb {
        let (lora, qlora) = estimate_vram(after as f64 / 1e9);
        if lora > v && qlora <= v {
            out.push(format!("Training this model needs QLoRA on this card (16-bit LoRA would need about {lora:.0} GB, QLoRA about {qlora:.0} GB; the card has {v:.0} GB). QLoRA can lose some quality on Qwen3.5."));
        } else if qlora > v {
            out.push(format!("Even QLoRA (about {qlora:.0} GB) does not fit on this card ({v:.0} GB), so this model could be made but not trained here."));
        }
    }
    out
}
