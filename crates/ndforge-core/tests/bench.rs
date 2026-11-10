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
    assert_eq!(ids(&variants(&h, false, true))[5..], ["compile", "liger", "padding_free"]);
    assert_eq!(variants(&h, true, true).len(), 5);
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
    "bucket": h["bucket"], "batch": h["batch"]}}) + "\n")
def out(**e): print(json.dumps(e), flush=True)
if h["batch"] == 8:
    out(type="error", code="train_failed", message="The graphics card ran out of memory.")
    sys.exit(1)
tok = {{"off": 1000, "on": 1600}}.get(h["bucket"], 1500)
if h.get("compile") == "on": tok = 1200
notes = ["Padding-free batching was asked for but is off: no flash-attn."] if h.get("padding_free") == "on" else []
out(type="speed", stage="all", notes=notes)
out(type="metric", step=h["max_steps"], total_steps=h["max_steps"], loss=1.0, tok_s=tok, pad_pct=12.5, seconds_per_step=0.8, vram_mb=9000)
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
    assert_eq!(seen.len(), 8, "{seen:?}");
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
    assert_eq!(row("compile")["tok_s"].as_u64(), Some(1200));
    assert!(row("padding_free")["notes"][0].as_str().unwrap().contains("asked for but is off"), "an option that stayed off says why in its row");
    // the fastest, kept on disk, no leftovers
    assert_eq!((r["best"].as_str(), r["stopped"].as_bool(), r["steps"].as_u64()), (Some("bucket_on"), Some(false), Some(16)));
    assert_eq!(last(&p, &m, Method::Lora).unwrap()["best"], "bucket_on");
    assert!(last(&p, &m, Method::Qlora).is_none() && !p.folder.join("benchmarks/tmp").exists());
    assert!(events.iter().any(|e| e["kind"] == "bench" && e["row"].is_object()) && events.iter().any(|e| e["kind"] == "bench" && e["name"] == "Batches grouped by length" && e["row"].is_null()));
    // the Stop button before the first run: nothing is run and nothing is saved over the earlier result
    let stopped = run(&p, &m, Method::Lora, &base_job, &h, 16, false, Arc::new(AtomicBool::new(true)), &mut |_| {}).unwrap();
    assert_eq!((stopped["stopped"].as_bool(), stopped["rows"].as_array().unwrap().len()), (Some(true), 0));
    assert_eq!(last(&p, &m, Method::Lora).unwrap()["best"], "bucket_on");
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
