//! The speed benchmark: a few real training steps with each of several settings, one after the other in separate worker processes (so a setting that
//! runs out of memory only fails its own row), reporting tokens per second and peak memory. The fastest settings can then be used for the real run.
//! Results are kept in `benchmarks/<model>-<method>.json` of the project.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::catalog::ModelEntry;
use super::plan::{Hyper, Method};
use super::runs::drive;
use crate::envmgr::jobs::WorkerEvent;
use crate::envmgr::{env_dir, python_path};
use crate::error::{Error, Result};
use crate::project::Project;

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub id: String,
    pub name: String,
    /// settings that differ from the plan's (hyper keys)
    pub overrides: Value,
}

fn v(id: &str, name: &str, overrides: Value) -> Variant {
    Variant { id: id.into(), name: name.into(), overrides }
}

/// The settings to compare. `unsloth`: the environment has Unsloth (it manages checkpointing, kernels and compiling itself, so the advanced options would
/// be ignored); `deep`: also the advanced options of the plain-libraries path.
pub fn variants(base: &Hyper, unsloth: bool, deep: bool) -> Vec<Variant> {
    let mut out = vec![v("current", "Your current settings", json!({}))];
    for (id, mode, name) in [("bucket_off", "off", "Batches in random order (not grouped by length)"), ("bucket_on", "on", "Batches grouped by length")] {
        if base.bucket != mode {
            out.push(v(id, name, json!({"bucket": mode})));
        }
    }
    // larger batches with fewer accumulation steps: the same number of examples per update
    for f in [2u32, 4] {
        if base.batch * f <= 32 && base.grad_accum >= f && base.grad_accum % f == 0 {
            out.push(v(&format!("batch_x{f}"), &format!("Batch of {} instead of {}", base.batch * f, base.batch), json!({"batch": base.batch * f, "grad_accum": base.grad_accum / f})));
        }
    }
    if deep {
        // attention kernels work with both engines (Unsloth keeps its own default under "auto")
        for (id, mode, name) in [("attn_sdpa", "sdpa", "Attention: PyTorch sdpa"), ("attn_flash", "flash_attention_2", "Attention: FlashAttention-2"), ("attn_flex", "flex_attention", "Attention: flex attention")] {
            if base.attention != mode {
                out.push(v(id, name, json!({"attention": mode})));
            }
        }
    }
    if deep && !unsloth {
        out.push(v("compile", "torch.compile", json!({"compile": "on"})));
        out.push(v("liger", "Liger kernels", json!({"liger": "on"})));
        out.push(v("padding_free", "Padding-free batching", json!({"padding_free": "on"})));
    }
    out
}

pub fn env_has_unsloth() -> bool {
    std::fs::read_to_string(env_dir().join("ndforge_env.json")).map(|t| t.contains("\"unsloth\": true")).unwrap_or(false)
}

pub fn file(p: &Project, m: &ModelEntry, method: Method) -> PathBuf {
    p.folder.join("benchmarks").join(format!("{}-{}.json", m.id, method.key()))
}

/// The last benchmark of this model and training type (None when there is none).
pub fn last(p: &Project, m: &ModelEntry, method: Method) -> Option<Value> {
    std::fs::read(file(p, m, method)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn job_for(base: &Value, variant: &Variant, steps: u32, out: &Path) -> Value {
    let mut job = base.clone();
    if let (Some(h), Some(o)) = (job["hyper"].as_object_mut(), variant.overrides.as_object()) {
        for (k, x) in o {
            h.insert(k.clone(), x.clone());
        }
    }
    job["hyper"]["max_steps"] = json!(steps);
    job["hyper"]["save_steps"] = json!(1_000_000);   // no checkpoints
    job["hyper"]["logging_steps"] = json!(1);
    job["out"] = json!(out);
    job["resume"] = json!(false);
    job["probes"] = json!([]);
    job["probe_every"] = json!(0);
    if let Some(o) = job.as_object_mut() {
        o.remove("stages");
    }
    job
}

/// Runs the benchmark. `emit` gets {"kind": "bench", "index", "total", "name", "row"?}. Stopping (`stop`) ends it after the variant that is running and keeps the rows made so far.
pub fn run(p: &Project, m: &ModelEntry, method: Method, base_job: &Value, base: &Hyper, steps: u32, deep: bool, stop: Arc<AtomicBool>, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let list = variants(base, env_has_unsloth() && base.engine != "standard", deep);
    let tmp = p.folder.join("benchmarks").join("tmp");
    let mut rows: Vec<Value> = Vec::new();
    let mut stopped = false;
    for (i, var) in list.iter().enumerate() {
        if stop.load(Ordering::SeqCst) {
            stopped = true;
            break;
        }
        emit(json!({"kind": "bench", "index": i, "total": list.len(), "name": var.name}));
        let dir = tmp.join(&var.id);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        let job = job_for(base_job, var, steps, &dir);
        let (mut metric, mut notes) = (Value::Null, Vec::<String>::new());
        let outcome = drive(&py, &dir, &job, stop.clone(), Duration::from_secs(60), &mut |e| {
            if let WorkerEvent::Other(ev) = e {
                match ev["type"].as_str() {
                    Some("metric") => metric = ev,
                    Some("speed") => notes = ev["notes"].as_array().map(|a| a.iter().filter_map(|n| n.as_str().map(String::from)).collect()).unwrap_or_default(),
                    _ => {}
                }
            }
        });
        let _ = std::fs::remove_dir_all(&dir);
        let row = match outcome {
            Ok(res) if res["stopped"] != true => json!({"id": var.id, "name": var.name, "overrides": var.overrides, "ok": true, "tok_s": metric["tok_s"], "pad_pct": metric["pad_pct"],
                "seconds_per_step": metric["seconds_per_step"], "peak_gb": res["peak_vram_mb"].as_f64().map(|mb| (mb / 1024.0 * 10.0).round() / 10.0), "notes": notes}),
            other => {
                if stop.load(Ordering::SeqCst) {
                    stopped = true;
                    break;
                }
                let why = match other { Err(e) => e.0, Ok(_) => "The run ended early.".to_string() };
                json!({"id": var.id, "name": var.name, "overrides": var.overrides, "ok": false, "error": why, "notes": notes})
            }
        };
        emit(json!({"kind": "bench", "index": i, "total": list.len(), "name": var.name, "row": row}));
        rows.push(row);
    }
    let _ = std::fs::remove_dir_all(&tmp);
    let speed = |r: &Value| r["tok_s"].as_f64().unwrap_or(0.0);
    let ok: Vec<&Value> = rows.iter().filter(|r| r["ok"] == true && speed(r) > 0.0).collect();
    let baseline = ok.iter().find(|r| r["id"] == "current").map(|r| speed(r));
    let best = ok.iter().max_by(|a, b| speed(a).partial_cmp(&speed(b)).unwrap_or(std::cmp::Ordering::Equal)).map(|r| r["id"].clone());
    for r in rows.iter_mut() {
        if let (Some(b), true) = (baseline, r["ok"] == true) {
            r["vs_current"] = json!(((speed(r) / b) * 100.0).round() / 100.0);
        }
    }
    let result = json!({"created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), "model": m.name, "method": method.key(), "steps": steps, "deep": deep, "stopped": stopped,
                        "unsloth": env_has_unsloth(), "rows": rows, "best": best});
    if !rows.is_empty() {
        let f = file(p, m, method);
        if let Some(d) = f.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(f, serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(result)
}
