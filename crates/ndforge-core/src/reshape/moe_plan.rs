//! The recommendations panel for a mixture of experts (docs/MOE_GUIDE.md section 8): what to do first on this card, in which order, what it should cost and what to watch for.
//! Pure: the page's command hands in the inspect, the card, the free disk and the installed versions; nothing is measured here. The default goal is "a smaller file" (owner's decision D2).
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::auto_moe;
use super::moe::MoeInfo;
use super::moe_compat::{findings, Have, VARLOOP_REPO};
use super::plan::Inspect;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecoStep {
    pub id: String,
    pub title: String,
    pub text: String,
}

/// What one of the suggested models would be.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expected {
    pub key: String,
    pub experts: u32,
    pub top_k: u32,
    pub passes: u32,
    pub params_b: f64,
    pub packed_gb: f64,
    pub q4_gb: f64,
    pub visits: u32,
    pub active_b: f64,
    pub heal_tokens_m: u32,
    pub heal_hours: (f64, f64),
    pub heal_gb: f64,
    /// the most disk this cut needs at once, GB (the original, the cut, the teacher's answers, the healed copy)
    pub disk_peak_gb: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendation {
    pub goal: String,
    pub order: Vec<RecoStep>,
    /// the Try-first questions to ask first, in order (`moetry` keys)
    pub first_try: Vec<String>,
    pub expected: Vec<Expected>,
    pub warnings: Vec<String>,
    /// what nobody knows yet (the numbers above are estimates; these are the open questions)
    pub unknowns: Vec<String>,
    pub blocked: Option<String>,
}

pub struct RecoInput<'a> {
    pub ins: &'a Inspect,
    pub vram_gb: Option<f64>,
    pub free_gb: Option<f64>,
    pub goal: &'a str,
    pub target_gb: Option<f64>,
    pub have: &'a Have,
}

fn step(id: &str, title: &str, text: &str) -> RecoStep {
    RecoStep { id: id.into(), title: title.into(), text: text.into() }
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn expected(ins: &Inspect, m: &MoeInfo, key: &str, experts: u32, top_k: u32, passes: u32, vram: Option<f64>) -> Expected {
    let params = auto_moe::params_at(ins, m, experts);
    let removed = (m.experts - experts) as f64 / m.experts as f64;
    let tokens = auto_moe::tokens_m_for(removed, top_k < m.top_k);
    let h = auto_moe::heal_hours(tokens);
    let packed = auto_moe::packed_gb(m, experts);
    let cache = tokens as f64 * 1e6 * super::auto_plan::CACHE_BYTES_PER_TOKEN / 1e9;
    let _ = vram;
    Expected {
        key: key.into(), experts, top_k, passes, params_b: ((params as f64 / 1e9) * 100.0).round() / 100.0, packed_gb: r1(packed), q4_gb: r1(auto_moe::q4_gb(params)), visits: m.visits_for(ins.layers, passes),
        active_b: (auto_moe::active_b(ins, m, experts, top_k, passes) * 100.0).round() / 100.0, heal_tokens_m: tokens, heal_hours: (r1(h.0), r1(h.1)), heal_gb: r1(auto_moe::heal_gb(ins, m, experts, 4096, 64, passes)),
        disk_peak_gb: r1(m.packed_bytes as f64 / 1e9 + packed + cache + packed),
    }
}

pub fn recommend(i: &RecoInput) -> Recommendation {
    let mut out = Recommendation { goal: i.goal.into(), order: Vec::new(), first_try: Vec::new(), expected: Vec::new(), warnings: Vec::new(), unknowns: Vec::new(), blocked: None };
    let Some(m) = i.ins.inkling() else {
        out.blocked = Some("The recommendations are for Inkling-2-Mini; other models use the Slim page's own sizes.".into());
        return out;
    };
    if !["size", "speed", "train"].contains(&i.goal) {
        out.blocked = Some("The goal is a smaller file, faster answers, or a model that knows your project.".into());
        return out;
    }
    let vram = i.vram_gb;
    out.order.push(step("goal", "Decide the goal", "A smaller file (smaller cards, more context), faster answers, or a model that knows your project. On a 32 GB card the original already runs (Q4 about 17 GB); a cut only helps the first two. \
                                                 To teach it your world, train the uncut model (the Train page, QLoRA needs about 22 GB) and do not cut."));
    let counts = m.counts();
    let (c96, c64) = (counts.iter().copied().find(|c| *c == 96), counts.iter().copied().find(|c| *c == 64));
    match i.goal {
        "train" => {
            out.order.push(step("train", "Train the uncut model", "Use the Train page with Inkling-2-Mini and QLoRA (Unsloth keeps the experts in 4 bit; the loop's message gates stay 16-bit). 16-bit LoRA needs about 59 GB and does not fit 32 GB. Nothing here needs a cut."));
            out.unknowns.push("Training an Inkling on this program's data format was not run on a real card.".into());
        }
        "speed" => {
            out.order.push(step("measure", "Measure the experts once", "Reads your text through the model and counts what every expert does (minutes). Needed before any experts are removed; fewer experts per token alone does not need it."));
            out.order.push(step("try", "Try first", "Fewer experts per token (6, then 4) and the counts near your size: the rise of the error for each, in about 20 minutes. Take the smallest number whose rise is small."));
            out.order.push(step("cut", "Cut with fewer experts per token", "Config only: the weights are copied. Run the thinking check; heal only if it fails."));
            out.order.push(step("passes", "Loop passes", &format!("Fewer passes (3 to 2) would be faster still but this model was never trained for it; the program asks it only as a Try-first question and Auto-cut never does it. {VARLOOP_REPO} was trained for 1 or 2 passes: start from it for that.")));
            out.first_try = vec!["base".into(), "k=6".into(), "k=4".into(), "p=2".into()];
            out.expected.push(expected(i.ins, m, "k=6", m.experts, 6.min(m.top_k), m.passes, vram));
            out.expected.push(expected(i.ins, m, "k=4", m.experts, 4.min(m.top_k), m.passes, vram));
            if m.passes > 1 {
                out.expected.push(expected(i.ins, m, "p=2", m.experts, m.top_k, 2, vram));
            }
            out.unknowns.push("How much speed fewer experts per token gives (the experts are about half of the work; attention, the loop gates and the routers stay) and what it costs in quality: the authors did not test it (discussion #51 on the model page is unanswered).".into());
        }
        _ => {
            out.order.push(step("measure", "Measure the experts once", "5 to 15 minutes on a 32 GB card (an estimate): which experts matter, summed over all three loop passes, on your own text."));
            out.order.push(step("try", "Try first (Quick, about 20 questions)", "The baseline, 112 / 96 / 80 / 64 experts, the other rankings at 96, and fewer experts per token. About 20 to 40 minutes (an estimate)."));
            out.order.push(step("cut", "Remove experts: 128 to 96", "A byte copy of the packed files minus the removed experts: minutes, about 15.6 GB. Go to 64 only as a second stage with its own heal."));
            out.order.push(step("check", "Thinking check", "Check the cut with llama.cpp against the original. If it passes, export without healing (Save as GGUF) and compare the sizes."));
            out.order.push(step("heal", "Heal only if needed", "If the check fails: 20 million tokens (4 to 7 hours, an estimate) with the experts frozen, then check again. Healing an Inkling is not built into this version yet."));
            out.first_try = ["base".to_string()].into_iter().chain(counts.iter().map(|c| format!("e={c}"))).chain(["e=96|s=freq".into(), "e=96|s=gate".into(), "k=6".into()]).collect();
            for (c, key) in [(c96, "e=96"), (c64, "e=64")] {
                if let Some(c) = c {
                    out.expected.push(expected(i.ins, m, key, c, m.top_k, m.passes, vram));
                }
            }
            if let Some(t) = i.target_gb {
                if let Some(c) = counts.iter().copied().find(|c| auto_moe::q4_gb(auto_moe::params_at(i.ins, m, *c)) <= t * 1.04) {
                    if !out.expected.iter().any(|e| e.experts == c) {
                        out.expected.push(expected(i.ins, m, &format!("e={c}"), c, m.top_k, m.passes, vram));
                    }
                }
            }
            out.unknowns.push("How much the error rises for each count, whether REAP beats frequency on this model (nothing is published for a looped mixture of experts), and which layers mind losing experts.".into());
            out.unknowns.push("Whether the thinking check survives 96 experts without healing.".into());
        }
    }
    out.order.push(step("export", "Export", "llama.cpp b12611 or newer; an importance matrix adapted from the original's; Q4_K_M. The NVFP4 size is offered only after Compare quants has measured it."));
    // what is wrong with this computer for the goal
    for f in findings("inkling2", i.have) {
        if f.status == "too_old" {
            out.warnings.push(f.message.clone());
        }
    }
    if i.have.vllm.is_none() {
        out.warnings.push("vLLM is not set up: healing would use the in-process packed teacher, which is slower (3 to 7 hours for 40 million tokens, an estimate).".into());
    }
    out.warnings.push("16-bit LoRA needs about 59 GB; a 16-bit copy of the model needs 53 GB of disk. Neither is needed by the cuts above (they stay packed).".into());
    out.warnings.push("The original itself may loop in long reasoning (reports on the model page); the thinking check's loop rate then shows it for the original as well.".into());
    if m.passes > 1 && !m.varloop {
        out.warnings.push("Fewer loop passes or whole blocks remove the drafter file (speculative decoding); the model was never trained with 2 passes.".into());
    }
    if i.goal != "train" {
        if let Some(e) = out.expected.first() {
            if let Some(free) = i.free_gb.filter(|f| *f < e.disk_peak_gb * 1.05) {
                out.warnings.push(format!("A healed cut needs up to {:.0} GB of disk at once; {free:.0} GB are free.", e.disk_peak_gb));
            }
            if vram.is_some_and(|v| e.heal_gb > v) {
                out.warnings.push(format!("Healing the first suggestion needs about {:.0} GB on the card (an estimate); this card has {:.0} GB.", e.heal_gb, vram.unwrap_or(0.0)));
            }
        }
    }
    out.warnings.push("A cut or healed model is a derivative of an Apache-2.0 model: keep its licence and say what was changed.".into());
    out.unknowns.push("Every time and memory figure is an estimate: nothing was run on a real card or on the real checkpoint yet.".into());
    out
}

/// The answer of the window command as JSON.
pub fn recommend_json(i: &RecoInput) -> Value {
    serde_json::to_value(recommend(i)).unwrap_or(Value::Null)
}
