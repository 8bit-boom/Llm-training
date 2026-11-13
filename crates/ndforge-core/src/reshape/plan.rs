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
    /// width pruning ("thinner layers", version 2): the sizes it works on, whether this model allows it, the reason when not, and the allowed sizes
    #[serde(default)]
    pub width_dims: Option<WidthDims>,
    #[serde(default)]
    pub width_supported: bool,
    #[serde(default)]
    pub width_why: Option<String>,
    #[serde(default)]
    pub width_choices: Option<WidthChoices>,
}

/// The sizes of a Qwen3.5-style model that width pruning works on (the worker's `reshape_width.width_dims`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WidthDims {
    pub hidden: u32,
    pub intermediate: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub lin_k_heads: u32,
    pub lin_v_heads: u32,
    pub lin_k_dim: u32,
    pub lin_v_dim: u32,
    pub conv_kernel: u32,
    pub mtp_layers: u32,
    pub vocab: u32,
    pub tied: bool,
}

/// The sizes the page may offer: MLP widths ascending (multiples of 256 from half to all), linear-attention value heads and attention heads descending
/// (the original first).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WidthChoices {
    pub intermediate: Vec<u32>,
    pub lin_v_heads: Vec<u32>,
    pub heads: Vec<u32>,
}

/// What to thin: the MLP width of every layer, and (Expert, experimental) the value heads of the linear-attention layers and the query heads of the
/// full-attention layers. `None` keeps the model's own number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Width {
    pub intermediate: u32,
    #[serde(default)]
    pub lin_v_heads: Option<u32>,
    #[serde(default)]
    pub heads: Option<u32>,
}

/// What the person chose on the page: remove whole layers, make every layer thinner, or both. `Mode` stays Prune | Grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shape {
    #[default]
    Depth,
    Width,
    Both,
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
fn s256() -> u32 {
    256
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
    /// thinner layers (prune only): with `blocks` 0 and no `drop_layers` this is width only; with blocks or layers it is both
    #[serde(default)]
    pub width: Option<Width>,
    /// measuring samples when width is on (the measurement is kept per model, data version, samples and length)
    #[serde(default = "s256")]
    pub width_samples: u32,
}

/// Depth, width or both, from what the spec asks for.
pub fn shape_of(spec: &ReshapeSpec) -> Shape {
    match (&spec.width, spec.blocks > 0 || spec.drop_layers.is_some()) {
        (None, _) => Shape::Depth,
        (Some(_), true) => Shape::Both,
        (Some(_), false) => Shape::Width,
    }
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
    /// width options: the MLP width of this stop (None for layer options)
    pub intermediate: Option<u32>,
    /// the size is known exactly before anything is measured (width: from the dimensions; layers: the layers are all alike, so also; "about" is for odd models)
    pub exact: bool,
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
            intermediate: None, exact: mode == Mode::Grow,
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
    if spec.calib_samples < 8 || spec.calib_samples > 4096 || spec.calib_len < 32 || spec.calib_len > 8192 || spec.width_samples < 8 || spec.width_samples > 4096 {
        return Err(Error::new("Measure on 8 to 4096 samples of 32 to 8192 tokens."));
    }
    if let Some(w) = &spec.width {
        validate_width(spec, ins, w)?;
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
        (None, _) if spec.width.is_some() && spec.blocks == 0 => {}      // width only: no layers are removed
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
    let blocks = if spec.width.is_some() && spec.drop_layers.is_none() { spec.blocks } else { blocks };
    let after = match &spec.width {
        Some(w) => width_params(ins, &kept_layers(ins, blocks), w),
        None => params_after(ins, spec.mode, blocks),
    };
    if let (Some(w), Some(d)) = (&spec.width, &ins.width_dims) {
        let gone = 100.0 * (1.0 - w.intermediate as f64 / d.intermediate.max(1) as f64);
        if gone > 25.0 {
            out.push(format!("Removing {gone:.0} % of the MLP: answers get clearly worse until the model is healed (Heal step)."));
        }
        out.push("Healing is strongly recommended for width slimming: the original model teaches the thinner one to answer like it did.".to_string());
        if w.lin_v_heads.is_some_and(|v| v != d.lin_v_heads) || w.heads.is_some_and(|h| h != d.heads) {
            out.push("Thinning the attention layers is experimental: nobody has published how well it works for this kind of model.".to_string());
        }
    }
    match spec.mode {
        Mode::Prune => {
            if blocks * 4 > ins.blocks {
                out.push("More than a quarter of the layers are removed. Answers get clearly worse until the model has been trained again on your world.".to_string());
            }
            if spec.width.is_none() {
                out.push("A slimmed model is worse than the original until it is trained again (Step 3). Test it first, then train it.".to_string());
            }
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

// ------------------------------------------------------------------------------------------------------- thinner layers (width)

fn thousands(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The widths of a thinned model, every `None` of a `Width` filled from the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WidthSizes {
    pub intermediate: u32,
    pub lin_v_heads: u32,
    pub heads: u32,
}

pub fn resolve_width(d: &WidthDims, w: &Width) -> WidthSizes {
    WidthSizes { intermediate: w.intermediate, lin_v_heads: w.lin_v_heads.unwrap_or(d.lin_v_heads), heads: w.heads.unwrap_or(d.heads) }
}

/// Exact parameter count of a Qwen3.5-style model with `n_lin` linear-attention and `n_full` full-attention layers at the given widths (layers +
/// embeddings + output layer + final norm + multi-token head + `vision` + `other`). The mirror of `reshape_width.params_for` in the worker;
/// `worker/tests/data/width_params.json` is read by both test suites.
pub fn width_params_dims(d: &WidthDims, n_lin: u64, n_full: u64, w: &WidthSizes, vision: u64, other: u64) -> u64 {
    let (h, hd, kv) = (d.hidden as u64, d.head_dim as u64, d.kv_heads as u64);
    let (i, nh) = (w.intermediate as u64, w.heads as u64);
    let (nk, nv, dk, dv) = (d.lin_k_heads as u64, w.lin_v_heads as u64, d.lin_k_dim as u64, d.lin_v_dim as u64);
    let mlp = 3 * h * i;
    let norms = 2 * h;
    let conv = 2 * nk * dk + nv * dv;
    let gdn = h * conv + h * nv * dv + 2 * h * nv + conv * d.conv_kernel as u64 + 2 * nv + dv + nv * dv * h;
    let att = h * nh * hd * 2 + 2 * h * kv * hd + nh * hd * h + 2 * hd;
    let layers = n_lin * (gdn + mlp + norms) + n_full * (att + mlp + norms);
    let mtp = d.mtp_layers as u64 * (2 * h * h + att + mlp + norms + 3 * h);
    let head = if d.tied { 0 } else { d.vocab as u64 * h };
    layers + d.vocab as u64 * h + head + h + mtp + vision + other
}

fn layer_is_full(ins: &Inspect, i: usize) -> bool {
    match ins.layer_types.get(i) {
        Some(t) => t == "full_attention",
        None => ins.period > 1 && (i as u32 + 1) % ins.period == 0,
    }
}

/// The layers that stay when `blocks` whole blocks are removed. Which blocks go is decided by the measuring, but every block has the same layer types,
/// so the count of each kind (and with it the parameter count of a width-thinned model) is the same whichever they are.
pub fn kept_layers(ins: &Inspect, blocks: u32) -> Vec<usize> {
    let n = (ins.layers as usize).saturating_sub((blocks * ins.period) as usize);
    (0..n).collect()
}

/// Exact parameters of the model after thinning, with the layers `layers_kept` (indices into the original) left. 0 when the model cannot be thinned.
pub fn width_params(ins: &Inspect, layers_kept: &[usize], w: &Width) -> u64 {
    let Some(d) = &ins.width_dims else { return 0 };
    let n_full = layers_kept.iter().filter(|i| layer_is_full(ins, **i)).count() as u64;
    let n_lin = layers_kept.len() as u64 - n_full;
    width_params_dims(d, n_lin, n_full, &resolve_width(d, w), ins.vision_params, ins.other_params.saturating_sub(d.hidden as u64))
}

/// The MLP width nearest to `num / den` of the original, among the offered sizes.
fn nearest_choice(choices: &[u32], full: u32, num: u32, den: u32) -> Option<u32> {
    let target = full as f64 * num as f64 / den as f64;
    choices.iter().copied().min_by(|a, b| (*a as f64 - target).abs().partial_cmp(&(*b as f64 - target).abs()).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)))
}

/// What the page suggests first (owner decision): the mild cut, 13/17 of the MLP (Qwen3.5 / 3.8 27B: 17,408 to 13,312 = 23.7 B), and as "Both" one block
/// removed plus 12/17 of the MLP (12,288 = 21.5 B). `(width only, (blocks, width for both))`
pub fn suggested_widths(ins: &Inspect) -> Option<(u32, (u32, u32))> {
    let (d, c) = (ins.width_dims.as_ref()?, ins.width_choices.as_ref()?);
    let mild = nearest_choice(&c.intermediate, d.intermediate, 13, 17)?;
    let both = nearest_choice(&c.intermediate, d.intermediate, 12, 17)?;
    Some((mild, (1, both)))
}

/// One stop for every MLP width the model allows, with `depth_blocks` blocks removed as well (0 = width only) and the Expert axes at the given values.
/// Ascending by width, so the slider's first stop is the thinnest; the exact size comes from the dimensions, before anything is measured.
pub fn width_options(ins: &Inspect, depth_blocks: u32, vram_gb: Option<f64>, lin_v_heads: Option<u32>, heads: Option<u32>) -> Vec<SizeOption> {
    let (Some(d), Some(c)) = (&ins.width_dims, &ins.width_choices) else { return Vec::new() };
    if !ins.supported || !ins.width_supported {
        return Vec::new();
    }
    let kept = kept_layers(ins, depth_blocks);
    c.intermediate.iter().map(|&i| {
        let params = width_params(ins, &kept, &Width { intermediate: i, lin_v_heads, heads });
        let params_b = params as f64 / 1e9;
        let (lora, qlora) = estimate_vram(params_b);
        SizeOption {
            blocks: depth_blocks, layers: kept.len() as u32, params_b: (params_b * 100.0).round() / 100.0,
            disk_gb: ((params as f64 * bytes_per_param(ins)) / 1e9 * 10.0).round() / 10.0,
            vram_lora_gb: lora, vram_qlora_gb: qlora, fits_lora: vram_gb.map(|v| lora <= v), fits_qlora: vram_gb.map(|v| qlora <= v),
            intermediate: Some(i), exact: i <= d.intermediate,
        }
    }).collect()
}

/// Refuses a width the worker would refuse, in plain words.
fn validate_width(spec: &ReshapeSpec, ins: &Inspect, w: &Width) -> Result<()> {
    if spec.mode != Mode::Prune {
        return Err(Error::new("Thinning the layers is only possible when slimming a model."));
    }
    let (Some(d), Some(c)) = (&ins.width_dims, &ins.width_choices) else {
        return Err(Error::new(ins.width_why.clone().unwrap_or_else(|| "Thinning the layers is only available for dense Qwen3.5 / Qwen3.8 models so far. You can still remove whole layers.".into())));
    };
    if !ins.width_supported {
        return Err(Error::new(ins.width_why.clone().unwrap_or_else(|| "Thinning the layers is not possible for this model.".into())));
    }
    if w.intermediate != d.intermediate && !c.intermediate.contains(&w.intermediate) {
        let div = if c.intermediate.len() > 1 { c.intermediate[1] - c.intermediate[0] } else { 256 };
        return Err(Error::new(format!("The MLP size must be a multiple of {div} between half and all of {}.", thousands(d.intermediate))));
    }
    if let Some(v) = w.lin_v_heads {
        if !c.lin_v_heads.contains(&v) {
            return Err(Error::new(match c.lin_v_heads.last() {
                Some(low) if c.lin_v_heads.len() > 1 => format!("Linear-attention value heads can only go from {} to {low}.", d.lin_v_heads),
                _ => "This model's linear-attention value heads cannot be reduced.".to_string(),
            }));
        }
    }
    if let Some(h) = w.heads {
        if !c.heads.contains(&h) {
            return Err(Error::new(format!("Attention heads must stay a multiple of {} and at least {}.", d.kv_heads, c.heads.last().copied().unwrap_or(d.heads))));
        }
    }
    if w.intermediate == d.intermediate && w.lin_v_heads.is_none_or(|v| v == d.lin_v_heads) && w.heads.is_none_or(|h| h == d.heads) {
        return Err(Error::new("The chosen width is the same as the original model's; there is nothing to thin."));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------ healing

/// The heal budgets the page offers, in millions of tokens
pub const HEAL_TOKENS_M: [u32; 4] = [10, 20, 40, 80];

/// Default budget: 20 M tokens for students of 15 B or more, 40 M below (owner decision).
pub fn default_heal_tokens_m(student_b: f64) -> u32 {
    if student_b >= 15.0 { 20 } else { 40 }
}

/// How healing runs on this card.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HealPlan {
    /// "online": the teacher answers while the student learns; "cache": the teacher answers everything first, then the student learns
    pub mode: &'static str,
    /// "qlora" (students of 15 B or more) or "lora"
    pub student_method: &'static str,
    pub teacher_load: &'static str,
    /// the memory the heavier of the two steps needs
    pub est_gb: f64,
    pub fits: bool,
    pub why: Option<String>,
    /// GB of disk the teacher's saved answers need per 10 M tokens (k = 32)
    pub cache_gb_per_10m: f64,
    /// estimated hours per 10 M tokens, (low, high); an estimate until the benchmark has measured this card
    pub hours_per_10m: (f64, f64),
}

/// Memory for the teacher (4-bit) and the student (LoRA or QLoRA) and the way of healing that fits `vram_gb`: both at once when they fit together, else
/// one after the other (the teacher's answers go to disk first). Without a known card: cache mode (it needs the least memory).
pub fn heal_plan(student_b: f64, teacher_b: f64, vram_gb: Option<f64>) -> HealPlan {
    let (lora, qlora) = estimate_vram(student_b);
    let (method, student_gb): (&'static str, f64) = if student_b >= 15.0 { ("qlora", qlora) } else { ("lora", lora) };
    let teacher_4bit = (teacher_b * 0.6 * 10.0).round() / 10.0;
    let online_gb = teacher_4bit + student_gb + 4.0;
    let cache_gb = teacher_4bit.max(student_gb) + 2.0;
    let online = vram_gb.is_some_and(|v| online_gb <= v);
    let est = if online { online_gb } else { cache_gb };
    let fits = vram_gb.is_none_or(|v| est <= v);
    let why = match (fits, vram_gb) {
        (false, Some(v)) => Some(format!("Healing a model of {student_b:.0} B parameters needs about {est:.0} GB with the teacher's answers saved first; this card has {v:.0} GB. Choose a smaller model or heal on a bigger card.")),
        _ => None,
    };
    // tokens per second: the teacher reads 2,000 to 4,000 / s at 27 B, the student learns 800 to 1,500 / s at 21 B; both scale with the size
    let t = (2000.0 * 27.0 / teacher_b.max(1.0), 4000.0 * 27.0 / teacher_b.max(1.0));
    let st = (800.0 * 21.0 / student_b.max(1.0), 1500.0 * 21.0 / student_b.max(1.0));
    let hours = |tok_s: f64| 10e6 / tok_s / 3600.0;
    let per = if online { (hours(st.1 * 0.6), hours(st.0 * 0.6)) } else { (hours(t.1) + hours(st.1), hours(t.0) + hours(st.0)) };
    HealPlan {
        mode: if online { "online" } else { "cache" }, student_method: method, teacher_load: if online { "4bit" } else { "auto" }, est_gb: (est * 10.0).round() / 10.0, fits, why,
        cache_gb_per_10m: 10e6 * 196.0 / 1e9, hours_per_10m: ((per.0 * 10.0).round() / 10.0, (per.1 * 10.0).round() / 10.0),
    }
}
