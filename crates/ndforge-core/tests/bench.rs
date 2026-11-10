//! The speed benchmark: which settings are compared, how a run is made short, a setting that fails only fails its own row, the fastest is named, results are kept.
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use ndforge_core::project::Project;
use ndforge_core::train::bench::{last, run, variants};
use ndforge_core::train::catalog::find;
use ndforge_core::train::plan::{base_hyper, Method, Preset};
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn ids(v: &[ndforge_core::train::bench::Variant]) -> Vec<&str> {
    v.iter().map(|x| x.id.as_str()).collect()
}

#[test]
fn the_settings_that_are_compared() {
    let mut h = base_hyper(Preset::Normal);
    (h.batch, h.grad_accum, h.bucket) = (2, 8, "auto".into());
    assert_eq!(ids(&variants(&h, false, false)), ["current", "bucket_off", "bucket_on", "batch_x2", "batch_x4"]);
    // the same number of examples per update
    let x2 = variants(&h, false, false).into_iter().find(|v| v.id == "batch_x2").unwrap();
    assert_eq!(x2.overrides, json!({"batch": 4, "grad_accum": 4}));
    // advanced options only in the deep run and only where they would be used (not with Unsloth)
    assert_eq!(ids(&variants(&h, false, true))[5..], ["attn_sdpa", "attn_cudnn", "attn_flash", "attn_flex", "compile", "liger", "padding_free"]);
    assert_eq!(ids(&variants(&h, true, true))[5..], ["attn_sdpa", "attn_cudnn", "attn_flash", "attn_flex"], "attention applies to Unsloth too");
    // a setting that is already explicit is not compared with itself; no accumulation steps to trade, a batch that is already big
    h.bucket = "off".into();
    assert!(!ids(&variants(&h, false, false)).contains(&"bucket_off"));
    (h.batch, h.grad_accum) = (4, 1);
    assert!(!ids(&variants(&h, false, false)).iter().any(|i| i.starts_with("batch_x")));
    (h.batch, h.grad_accum) = (16, 8);
    assert_eq!(ids(&variants(&h, false, false)).iter().filter(|i| i.starts_with("batch_x")).count(), 1);
}

/// A fake worker: it records how the benchmark made its job, runs out of memory at a batch of 8, and is fastest with batches grouped by length.
#[cfg(unix)]
fn fake_worker(root: &std::path::Path, seen: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(root.join("env/bin")).unwrap();
    std::fs::write(root.join("fake.py"), format!(r#"
import json, sys
job = json.load(open(sys.argv[1], encoding="utf-8"))
h = job["hyper"]
open({seen:?}, "a").write(json.dumps({{"max_steps": h["max_steps"], "save_steps": h["save_steps"], "probes": job["probes"], "stages": "stages" in job, "resume": job["resume"], "out": job["out"],
    "bucket": h["bucket"], "batch": h["batch"], "attention": h.get("attention")}}) + "\n")
def out(**e): print(json.dumps(e), flush=True)
if h["batch"] == 8:
    out(type="error", code="train_failed", message="The graphics card ran out of memory.")
    sys.exit(1)
tok = {{"off": 1000, "on": 1600}}.get(h["bucket"], 1500)
if h["batch"] == 4: tok *= 1.2
tok *= {{"cudnn": 1.1, "flash_attention_2": 1.04, "flex_attention": 0.7}}.get(h.get("attention"), 1.0)
if "r2_" in job["out"] and job["out"].endswith("-1"): tok *= 1.01      # run-to-run noise: the second run of a round-2 setting is 1 % faster
if h.get("compile") == "on": tok = 1200
notes = ["Padding-free batching was asked for but is off: no flash-attn."] if h.get("padding_free") == "on" else []
out(type="speed", stage="all", notes=notes, attn=h.get("attention") or "sdpa", pad_pct=12.5)
# the first steps are slow (kernels compile): the running average starts low, the speed after the start-up is `tok`
tokens = elapsed = 0.0
for step in range(1, h["max_steps"] + 1):
    rate = tok * 0.4 if step <= 4 else tok
    tokens += rate; elapsed += 1.0
    out(type="metric", step=step, total_steps=h["max_steps"], loss=1.0, tok_s=int(tokens / elapsed), tokens_real=int(tokens), elapsed_s=elapsed, pad_pct=12.5, seconds_per_step=0.8, vram_mb=9000)
out(type="result", data={{"steps": h["max_steps"], "stopped": False, "peak_vram_mb": 10240, "final_loss": 1.0}})
"#, seen = seen.to_string_lossy())).unwrap();
    let py = root.join("env/bin/python");
    std::fs::write(&py, format!("#!/bin/sh\nexec python3 -I {} \"$5\"\n", root.join("fake.py").display())).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
#[test]
fn a_benchmark_runs_short_runs_names_the_fastest_and_a_failing_setting_fails_only_its_row() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let seen = d.path().join("seen.jsonl");
    fake_worker(d.path(), &seen);
    let p = Project::create(&d.path().join("projects"), "W").unwrap();
    let m = find("qwen3.5-4b").unwrap();
    let mut h = base_hyper(Preset::Normal);
    (h.batch, h.grad_accum) = (2, 8);
    let base_job = json!({"model": {"repo": m.repo, "family": m.family}, "method": "lora", "hyper": serde_json::to_value(&h).unwrap(), "out": "/should/be/replaced",
                          "stages": [{"name": "a"}], "probes": [{"system": "s", "question": "q"}], "probe_every": 50, "resume": true, "dataset": {"train": "x"}});
    let mut events = Vec::new();
    let r = run(&p, &m, Method::Lora, &base_job, &h, 16, true, Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
    // every run was short, wrote no checkpoints, made no sample answers, was a single stage, started fresh, in a folder of its own
    let seen: Vec<Value> = std::fs::read_to_string(&seen).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(seen.len(), 22, "12 single runs and 5 settings twice: {seen:?}");
    for s in &seen {
        assert_eq!((s["max_steps"].as_u64(), s["probes"].as_array().map(|a| a.len()), s["stages"].as_bool(), s["resume"].as_bool()), (Some(16), Some(0), Some(false), Some(false)));
        assert!(s["save_steps"].as_u64().unwrap() > 100_000 && s["out"].as_str().unwrap().contains("benchmarks") && s["out"] != "/should/be/replaced");
    }
    let rows = r["rows"].as_array().unwrap();
    let row = |id: &str| rows.iter().find(|r| r["id"] == id).unwrap_or_else(|| panic!("no row {id}"));
    // the setting that ran out of memory is a row with the reason; the others are not affected
    assert_eq!((row("batch_x4")["ok"].as_bool(), row("batch_x4")["error"].as_str().map(|e| e.contains("out of memory"))), (Some(false), Some(true)));
    assert_eq!(row("current")["ok"], true);
    assert_eq!((row("current")["tok_s"].as_u64(), row("bucket_off")["tok_s"].as_u64(), row("bucket_on")["tok_s"].as_u64()), (Some(1500), Some(1000), Some(1600)));
    assert_eq!((row("current")["peak_gb"].as_f64(), row("current")["vs_current"].as_f64(), row("bucket_off")["vs_current"].as_f64(), row("bucket_on")["vs_current"].as_f64()), (Some(10.0), Some(1.0), Some(0.67), Some(1.07)));
    assert!(seen.iter().any(|s| s["attention"] == "flash_attention_2") && seen.iter().any(|s| s["attention"] == "flex_attention"), "attention variants are sent to the worker");
    assert_eq!(row("compile")["tok_s"].as_u64(), Some(1200));
    assert!(row("padding_free")["notes"][0].as_str().unwrap().contains("asked for but is off"), "an option that stayed off says why in its row");
    // speeds leave out the slow start (the fake worker's first 4 steps run at 40 %): 1500 is the speed after it, not the running average
    // round 2: the best batch size and grouping of round 1 with every attention kernel, each run twice, so the noise shows
    let r2 = |id: &str| row(&format!("r2_{id}"));
    assert_eq!(r["rounds"], 2);
    assert_eq!(r2("default")["overrides"], json!({"batch": 4, "grad_accum": 4, "bucket": "on"}));
    assert!(r2("default")["name"].as_str().unwrap().contains("batch of 4"), "{}", r2("default")["name"]);
    assert_eq!(r2("attn_cudnn")["overrides"]["attention"], "cudnn");
    assert_eq!(r2("attn_cudnn")["tok_s_runs"].as_array().unwrap().len(), 2);
    assert_eq!(r2("attn_cudnn")["tok_s_runs"], json!([2112, 2133]), "the second run is 1 % faster: noise is visible");
    assert!(r2("attn_cudnn")["spread"].as_f64().unwrap() > 0.009 && r["tied"].as_array().unwrap().is_empty(), "{}", r["tied"]);
    assert!(r2("attn_flex")["tok_s"].as_f64().unwrap() < r2("default")["tok_s"].as_f64().unwrap());
    // the fastest, kept on disk, no leftovers
    assert_eq!((r["best"].as_str(), r["stopped"].as_bool(), r["steps"].as_u64()), (Some("r2_attn_cudnn"), Some(false), Some(16)));
    assert_eq!(last(&p, &m, Method::Lora).unwrap()["best"], "r2_attn_cudnn");
    assert!(last(&p, &m, Method::Qlora).is_none() && !p.folder.join("benchmarks/tmp").exists());
    assert!(events.iter().any(|e| e["kind"] == "bench" && e["row"].is_object()) && events.iter().any(|e| e["kind"] == "bench" && e["name"] == "Batches grouped by length" && e["row"].is_null()));
    // the Stop button before the first run: nothing is run and nothing is saved over the earlier result
    let stopped = run(&p, &m, Method::Lora, &base_job, &h, 16, false, Arc::new(AtomicBool::new(true)), &mut |_| {}).unwrap();
    assert_eq!((stopped["stopped"].as_bool(), stopped["rows"].as_array().unwrap().len()), (Some(true), 0));
    assert_eq!(last(&p, &m, Method::Lora).unwrap()["best"], "r2_attn_cudnn");
    // no training software: a plain message
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("missing"));
    assert!(run(&p, &m, Method::Lora, &base_job, &h, 16, false, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("Training setup"));
}

#[test]
fn the_engine_follows_the_setting_and_what_the_setup_installed() {
    use ndforge_core::train::runs::backend_for;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("env"));
    // nothing installed: the standard libraries
    assert_eq!(backend_for("auto"), "hf");
    std::fs::create_dir_all(d.path().join("env")).unwrap();
    std::fs::write(d.path().join("env/ndforge_env.json"), "{\"unsloth\": false, \"engine\": \"standard\"}").unwrap();
    assert_eq!(backend_for("auto"), "hf");
    // Unsloth installed: auto lets the worker use it (and fall back by itself if it cannot)
    std::fs::write(d.path().join("env/ndforge_env.json"), "{\"unsloth\": true, \"engine\": \"unsloth\"}").unwrap();
    assert_eq!(backend_for("auto"), "auto");
    // the setting wins in both directions
    assert_eq!(backend_for("standard"), "hf");
    assert_eq!(backend_for("unsloth"), "unsloth");
}

#[test]
fn speeds_skip_the_start_up_and_close_results_are_called_a_tie() {
    use ndforge_core::train::bench::{steady_tok_s, tied_with_best};
    // 12 log lines: the first three slow (compiling), then 1000 tokens per second
    let m: Vec<Value> = (1..=12).map(|i| {
        let (tokens, secs) = if i <= 3 { (100.0 * i as f64, i as f64) } else { (300.0 + 1000.0 * (i - 3) as f64, 3.0 + (i - 3) as f64) };
        json!({"tokens_real": tokens, "elapsed_s": secs, "tok_s": (tokens / secs) as u64})
    }).collect();
    let steady = steady_tok_s(&m).unwrap();
    assert!((steady - 1000.0).abs() < 1.0, "{steady}");
    assert!(m.last().unwrap()["tok_s"].as_u64().unwrap() < 900, "the running average is held back by the start");
    // too few points: the running average is all there is
    assert_eq!(steady_tok_s(&m[..2]), Some(100.0));
    assert!(steady_tok_s(&[]).is_none());
    let row = |id: &str, tok: f64, spread: f64| json!({"id": id, "round": 2, "ok": true, "tok_s": tok, "spread": spread});
    // 1.5 % apart: a tie; 8 % apart: not; a noisy pair (6 % spread between its own runs) ties with something 5 % behind
    assert_eq!(tied_with_best(&[row("a", 2000.0, 0.005), row("b", 1970.0, 0.005), row("c", 1840.0, 0.005)]), ["b"]);
    assert_eq!(tied_with_best(&[row("a", 2000.0, 0.06), row("b", 1900.0, 0.005)]), ["b"]);
    assert!(tied_with_best(&[json!({"id": "x", "round": 1, "ok": true, "tok_s": 5.0}), row("a", 2000.0, 0.0)]).is_empty());
}
