//! Mixture-of-experts models (Inkling-2-Mini and the per-expert Qwen/Mixtral layouts): what the worker's inspect says about them, and the rules for expert counts
//! that the page, the planner and Auto-cut share. Pure functions: no files, no network. The worker twin is `worker/ndforge_worker/moe_layout.py`; the cases in
//! `worker/tests/data/moe_cases.json` are read by both test suites.
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::plan::{Mode, ReshapeSpec};
use crate::error::{Error, Result};

/// Expert counts are multiples of this (decision D4: kernel friendly).
pub const COUNT_STEP: u32 = 16;
/// ... never below this many experts per layer,
pub const MIN_EXPERTS: u32 = 32;
/// ... never below this many times the experts used per token,
pub const MIN_PER_TOPK: u32 = 4;
/// ... and at most this share of the experts is removed in one stage.
pub const MAX_REMOVED: f64 = 0.5;

/// The "moe" block of `reshape_inspect` (absent for a dense model). Every field has a default so an older or newer worker never breaks reading it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MoeInfo {
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub experts: u32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default)]
    pub expert_inter: u32,
    #[serde(default)]
    pub shared: u32,
    #[serde(default)]
    pub router_scoring: String,
    #[serde(default)]
    pub router_bias: bool,
    /// attention-residual blocks and their size (Inkling: 4 blocks of 6 layers); 0 for a model without them
    #[serde(default)]
    pub blocks: u32,
    #[serde(default)]
    pub block_size: u32,
    /// the layers [start, end) that run `passes` times
    #[serde(default)]
    pub core: Vec<u32>,
    #[serde(default)]
    pub passes: u32,
    #[serde(default)]
    pub varloop: bool,
    /// layer slots a token runs through (layers + (passes - 1) x core layers)
    #[serde(default)]
    pub visits: u32,
    #[serde(default)]
    pub global_layers: Vec<u32>,
    #[serde(default)]
    pub expert_params: u64,
    #[serde(default)]
    pub dense_params: u64,
    #[serde(default)]
    pub active_params: u64,
    /// {"nvfp4": [first, last layer], "mxfp8": [...]}
    #[serde(default)]
    pub packed: Value,
    #[serde(default)]
    pub packed_bytes: u64,
    #[serde(default)]
    pub bf16_bytes: u64,
    /// {"file": "mtp.safetensors" | null, "targets": [...]}
    #[serde(default)]
    pub mtp: Value,
    /// which cuts exist: {"experts": bool, "top_k": bool, "passes": false | "untrained", "blocks": false | "expert-only", "expert_width": ...}
    #[serde(default)]
    pub cuts: Value,
    #[serde(default)]
    pub status: String,
    /// [{"experts", "params", "packed_bytes", "bf16_bytes"}] for the current count and every count a cut may keep
    #[serde(default)]
    pub sizes: Vec<MoeSize>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MoeSize {
    #[serde(default)]
    pub experts: u32,
    #[serde(default)]
    pub params: u64,
    #[serde(default)]
    pub packed_bytes: u64,
    #[serde(default)]
    pub bf16_bytes: u64,
}

impl MoeInfo {
    /// A looped model (Inkling): the core runs more than once.
    pub fn looped(&self) -> bool {
        self.core.len() == 2 && self.core[1] > self.core[0] && self.passes > 1
    }
    pub fn core_layers(&self) -> u32 {
        if self.core.len() == 2 { self.core[1].saturating_sub(self.core[0]) } else { 0 }
    }
    /// layer slots for another number of passes
    pub fn visits_for(&self, layers: u32, passes: u32) -> u32 {
        layers + passes.saturating_sub(1) * self.core_layers()
    }
    /// Counts the cut may keep, largest first (see `expert_counts_allowed`).
    pub fn counts(&self) -> Vec<u32> {
        expert_counts_allowed(self.experts, self.top_k)
    }
    /// The size row for a kept-expert count, if the worker sent it.
    pub fn size_at(&self, experts: u32) -> Option<&MoeSize> {
        self.sizes.iter().find(|s| s.experts == experts)
    }
    /// Packed bytes of a model that keeps `experts` per layer: the worker's row when it has one, else a share of the experts' bytes by the ratio.
    pub fn packed_bytes_at(&self, experts: u32) -> u64 {
        if experts == self.experts {
            return self.packed_bytes;
        }
        match self.size_at(experts) {
            Some(r) if r.packed_bytes > 0 => r.packed_bytes,
            _ => {
                let share = self.expert_params as f64 / (self.expert_params + self.dense_params).max(1) as f64;
                (self.packed_bytes as f64 * (1.0 - share * (1.0 - experts as f64 / self.experts.max(1) as f64))) as u64
            }
        }
    }
}

/// Kept-expert counts the planner may offer, largest first: multiples of 16, not below 32 and not below 4 x top-k, at most half removed in one stage.
pub fn expert_counts_allowed(experts: u32, top_k: u32) -> Vec<u32> {
    let floor = MIN_EXPERTS.max(MIN_PER_TOPK * top_k).max((experts as f64 * (1.0 - MAX_REMOVED)).ceil() as u32);
    let floor = floor.div_ceil(COUNT_STEP) * COUNT_STEP;
    let mut out = Vec::new();
    let mut n = experts.saturating_sub(COUNT_STEP);
    while n >= floor && n >= 1 {
        out.push(n);
        if n < COUNT_STEP {
            break;
        }
        n -= COUNT_STEP;
    }
    out
}

/// The plain-words reason a kept-expert count is not allowed (the rules of decision D4), or None.
pub fn check_experts_kept(experts: u32, top_k: u32, keep: u32) -> Option<String> {
    if keep >= experts {
        return Some("The kept number of experts must be smaller than the number the model has.".into());
    }
    if keep % COUNT_STEP != 0 {
        return Some(format!("The number of experts to keep must be a multiple of {COUNT_STEP}."));
    }
    if keep < MIN_EXPERTS {
        return Some(format!("Keep at least {MIN_EXPERTS} experts per layer."));
    }
    if keep < MIN_PER_TOPK * top_k {
        return Some(format!("Keep at least {MIN_PER_TOPK} times the experts used per token ({}).", MIN_PER_TOPK * top_k));
    }
    if (keep as f64) < experts as f64 * (1.0 - MAX_REMOVED) {
        return Some(format!("Remove at most {} % of the experts in one stage; cut again from the result for more.", (MAX_REMOVED * 100.0).round()));
    }
    None
}

/// What the worker would refuse for an Inkling cut, in plain words. Config-only cuts (loop passes, experts per token) today; experts and blocks follow.
pub fn validate_cut(spec: &ReshapeSpec, m: &MoeInfo) -> Result<()> {
    if spec.mode != Mode::Prune {
        return Err(Error::new("Only cutting is possible for this model; growing is a loop-pass or experts-per-token change (Try first)."));
    }
    if spec.width.is_some() {
        return Err(Error::new("Thinning the layers' width does not apply to a mixture of experts; remove experts instead."));
    }
    if spec.blocks > 0 || spec.drop_layers.is_some() {
        return Err(Error::new("Removing whole blocks of an Inkling model is not available in this version of the program."));
    }
    if spec.passes.is_none() && spec.top_k.is_none() && spec.experts.is_none() {
        return Err(Error::new("Nothing to change: choose the number of loop passes, of experts per token or of experts to keep."));
    }
    if let Some(e) = &spec.experts {
        if !["reap", "freq", "gate"].contains(&e.scoring.as_str()) {
            return Err(Error::new("The way experts are ranked is reap (the default), freq or gate."));
        }
        if let Some(why) = check_experts_kept(m.experts, spec.top_k.unwrap_or(m.top_k), e.keep) {
            return Err(Error::new(why));
        }
    }
    if let Some(p) = spec.passes {
        if !m.looped() && m.core_layers() == 0 {
            return Err(Error::new("This model has no loop; there are no passes to change."));
        }
        if p < 1 || (p < 2 && !m.varloop) {
            return Err(Error::new("This model was built for a fixed loop and the GGUF converter refuses fewer than 2 passes; use the variable-loop research checkpoint (thinkingmachines/TML-d24-varloop-research) for 1 pass."));
        }
        if p > 8 {
            return Err(Error::new("More than 8 passes is not offered."));
        }
    }
    if let Some(k) = spec.top_k {
        if k < 1 || k > m.experts.max(1) || k > 4 * m.top_k.max(1) {
            return Err(Error::new(format!("Experts per token must be between 1 and {}.", (4 * m.top_k.max(1)).min(m.experts.max(1)))));
        }
    }
    Ok(())
}

/// Bytes of the `.safetensors` files of a model folder (a packed mixture of experts is stored packed, so this is its size on the card too).
pub fn packed_bytes_of(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir).map(|d| d.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "safetensors")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum()).unwrap_or(0)
}
