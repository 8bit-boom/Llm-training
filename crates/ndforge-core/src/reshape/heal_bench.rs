//! "Find the fastest loss settings": short real steps of the healing loss with several settings, one worker process per setting (the loss is the same in all of them; only the speed
//! and the memory differ). The worker job is `distill` with the additive key `bench` (docs/TRAINING_SPEED.md); a setting that runs out of memory fails only its own row.
//!
//! Round 1 changes one thing at a time: the settings as they are, then the KL chunk 64, 128, 256 and 512. Round 2 (only when the estimate says it fits the card): keeping the output
//! layer's logits (`head_checkpoint` off) at the current chunk and at the best chunk of round 1. The numbers are this computer's; nothing here is assumed beforehand.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::plan::{self, HealKnobs};
use super::runs::{bench_base, HealSpec};
use crate::envmgr::jobs::{run_with_grace, WorkerEvent};
use crate::error::{Error, Result};
use crate::project::Project;

/// Optimizer steps of a measurement and the ones before it that are not timed.
pub const STEPS: u32 = 12;
pub const WARMUP: u32 = 3;
pub const CHUNKS: [u32; 4] = [64, 128, 256, 512];

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub id: String,
    pub label: String,
    pub chunk: u32,
    pub head_ckpt: bool,
    pub round: u32,
}

fn variant(id: &str, label: String, chunk: u32, head_ckpt: bool, round: u32) -> Variant {
    Variant { id: id.into(), label, chunk, head_ckpt, round }
}

/// Round 1: the settings as they are (the person's, or the automatic ones), and the other KL chunks.
pub fn first_round(k: &HealKnobs) -> Vec<Variant> {
    let mut v = vec![variant("current", format!("Your settings (KL chunk {})", k.kl_chunk), k.kl_chunk, k.head_ckpt, 1)];
    for c in CHUNKS.iter().filter(|c| **c != k.kl_chunk) {
        v.push(variant(&format!("chunk_{c}"), format!("KL chunk {c}"), *c, k.head_ckpt, 1));
    }
    v
}

fn tok_s(r: &Value) -> f64 {
    r["tokens_per_s"].as_f64().unwrap_or(0.0)
}

/// The fastest finished row (a tie goes to the first), if any finished.
pub fn best_row(rows: &[Value]) -> Option<&Value> {
    rows.iter().filter(|r| r["ok"] == true && tok_s(r) > 0.0).fold(None, |best: Option<&Value>, r| match best {
        Some(b) if tok_s(b) >= tok_s(r) => Some(b),
        _ => Some(r),
    })
}

/// Round 2: keeping the output layer's logits (the head not checkpointed) at the current chunk and at the best chunk of round 1, each only when the memory estimate fits the card
/// (within 95 % of it). Nothing without a known card or when the settings already keep the logits.
pub fn second_round(rows: &[Value], k: &HealKnobs, student_b: f64, d: &plan::HealDims, vram_gb: Option<f64>) -> Vec<Variant> {
    let Some(vram) = vram_gb else { return vec![] };
    if !k.head_ckpt {
        return vec![];
    }
    let fits = |chunk: u32| plan::heal_memory_gb(student_b, d, &HealKnobs { kl_chunk: chunk, head_ckpt: false, ..k.clone() }) <= vram * 0.95;
    let mut chunks = vec![k.kl_chunk];
    if let Some(b) = best_row(rows).and_then(|r| r["chunk"].as_u64()).map(|c| c as u32).filter(|c| *c != k.kl_chunk) {
        chunks.push(b);
    }
    chunks.into_iter().filter(|c| fits(*c)).map(|c| variant(&format!("keep_logits_chunk_{c}"), format!("Keep the output layer's logits, KL chunk {c}"), c, false, 2)).collect()
}

fn job_for(base: &super::runs::BenchBaseJob, v: &Variant, steps: u32, warmup: u32) -> Value {
    let mut job = base.job.clone();
    job["distill"]["chunk"] = json!(v.chunk);
    if !v.head_ckpt {
        job["distill"]["head_checkpoint"] = json!("off");
    }
    job["bench"] = json!({"steps": steps, "warmup": warmup, "targets": "cache"});
    job
}

fn measure(base: &super::runs::BenchBaseJob, v: &Variant, steps: u32, warmup: u32, dir: &Path, stop: &Arc<AtomicBool>) -> std::result::Result<Value, ()> {
    if stop.load(Ordering::SeqCst) {
        return Err(());
    }
    let job = job_for(base, v, steps, warmup);
    let _ = std::fs::create_dir_all(dir);
    let outcome = run_with_grace(&base.py, &job, &dir.join("STOP"), stop, Duration::from_secs(60), &mut |_e: WorkerEvent| {});
    let row = |ok: bool, extra: Value| {
        let mut r = json!({"id": v.id, "label": v.label, "chunk": v.chunk, "head_checkpoint": v.head_ckpt, "round": v.round, "ok": ok});
        if let (Some(o), Some(e)) = (r.as_object_mut(), extra.as_object()) {
            o.extend(e.clone());
        }
        r
    };
    match outcome {
        Ok(res) if res["stopped"] != true && res["tokens_per_s"].is_number() => Ok(row(true, json!({"tokens_per_s": res["tokens_per_s"], "peak_vram_mb": res["peak_vram_mb"], "tokens": res["tokens"], "seconds": res["seconds"], "targets": res["targets"]}))),
        other => {
            if stop.load(Ordering::SeqCst) {
                return Err(());
            }
            let why = match other {
                Err(e) => e.0,
                Ok(_) => "The run ended early.".to_string(),
            };
            let oom = why.to_lowercase().contains("out of memory");
            Ok(row(false, json!({"error": why, "oom": oom})))
        }
    }
}

/// Runs the speed test of a made model. `emit` gets {"kind": "heal_bench", "index", "total", "name", "row"?}. Stopping ends it after the setting that is running and keeps the rows made
/// so far. The result is written to `benchmarks/heal-<model>.json` and returned: {"rows", "best" (id), "best_settings" {kl_chunk, head_checkpoint}, "stopped", ...}.
pub fn run(p: &Project, name: &str, spec: &HealSpec, vram_gb: Option<f64>, steps: u32, stop: Arc<AtomicBool>, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let b = bench_base(p, name, spec, vram_gb)?;
    let base = b.job_base(steps, WARMUP);
    let tmp = p.folder.join("benchmarks").join("tmp-heal");
    let mut list = first_round(&b.ch.knobs);
    let mut rows: Vec<Value> = Vec::new();
    let (mut i, mut total, mut stopped, mut round2) = (0usize, list.len(), false, false);
    'all: loop {
        while i < list.len() {
            let v = list[i].clone();
            emit(json!({"kind": "heal_bench", "index": i, "total": total, "name": v.label}));
            let row = match measure(&base, &v, steps, WARMUP, &tmp.join(&v.id), &stop) {
                Ok(r) => r,
                Err(()) => {
                    stopped = true;
                    break 'all;
                }
            };
            emit(json!({"kind": "heal_bench", "index": i, "total": total, "name": v.label, "row": row}));
            rows.push(row);
            i += 1;
        }
        if !round2 {
            round2 = true;
            let more = second_round(&rows, &b.ch.knobs, b.student_b, &b.dims, vram_gb);
            if !more.is_empty() {
                total += more.len();
                list.extend(more);
                continue;
            }
        }
        break;
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let baseline = rows.iter().find(|r| r["id"] == "current" && r["ok"] == true).map(tok_s);
    for r in rows.iter_mut() {
        if let (Some(base), true) = (baseline, r["ok"] == true) {
            r["vs_current"] = json!(((tok_s(r) / base) * 100.0).round() / 100.0);
        }
    }
    let best = best_row(&rows).cloned();
    let result = json!({"created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), "model": name, "steps": steps, "warmup": WARMUP, "stopped": stopped, "rows": rows,
                        "best": best.as_ref().map(|r| r["id"].clone()), "best_settings": best.as_ref().map(|r| json!({"kl_chunk": r["chunk"], "head_checkpoint": if r["head_checkpoint"] == true { "on" } else { "off" }})),
                        "seq": b.ch.knobs.seq, "batch": b.ch.knobs.batch, "student_b": b.student_b, "targets": rows.iter().find_map(|r| r["targets"].as_str().map(String::from)),
                        "note": "Measured on this computer for this model and these settings; the loss is the same in every row. A real heal is a little slower (checkpoints, checks)."});
    if !rows.is_empty() {
        let f = file(p, name);
        if let Some(d) = f.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&f, serde_json::to_vec_pretty(&result)?).map_err(Error::from)?;
    }
    Ok(result)
}

/// `benchmarks/heal-<model>.json`
pub fn file(p: &Project, name: &str) -> std::path::PathBuf {
    p.folder.join("benchmarks").join(format!("heal-{name}.json"))
}

/// The last speed test of a made model (None when there is none).
pub fn last(p: &Project, name: &str) -> Option<Value> {
    std::fs::read(file(p, name)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}
