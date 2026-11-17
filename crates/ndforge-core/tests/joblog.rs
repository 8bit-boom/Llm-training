//! Every worker job leaves a saved, scrubbed log in the settings folder (envmgr/joblog.rs). Needs a python on PATH (skips with a note when there is none).
use ndforge_core::envmgr::jobs::run_job;
use serde_json::json;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn python() -> Option<&'static str> {
    for name in ["python3", "python"] {
        if Command::new(name).arg("-c").arg("pass").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(name);
        }
    }
    eprintln!("note: no python on PATH, job log test skipped");
    None
}

const TOKEN: &str = "hf_abcdefghijklmnopqrstuvwxyz0123";

fn stub_worker(dir: &std::path::Path, body: &str) {
    let pkg = dir.join("ndforge_worker");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("__init__.py"), "").unwrap();
    std::fs::write(pkg.join("__main__.py"), body).unwrap();
}

#[test]
fn a_job_leaves_a_log_with_the_worker_output_and_no_tokens() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(py) = python() else { return };
    let cfg = tempfile::tempdir().unwrap();
    let worker = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", cfg.path());
    stub_worker(worker.path(), &format!("import sys\nsys.stderr.write('warning: using token {TOKEN} for the hub\\n')\nprint('{{\"type\":\"log\",\"text\":\"half way\"}}', flush=True)\nprint('{{\"type\":\"error\",\"code\":\"disk_full\",\"message\":\"There is not enough free space.\"}}', flush=True)\nsys.exit(1)\n"));
    let err = run_job(std::path::Path::new(py), worker.path(), &json!({"job": "export", "hf": TOKEN}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert!(err.starts_with("There is not enough free space. (details: logs/jobs/") && err.ends_with("-export.log)"), "{err}");
    let logs = ndforge_core::envmgr::joblog::latest(10);
    assert_eq!(logs.len(), 1);
    let text = std::fs::read_to_string(&logs[0]).unwrap();
    assert!(text.contains("warning: using token [token] for the hub") && text.contains("half way") && text.contains("s err] ") && text.contains("s out] "), "{text}");
    assert!(text.contains("error code disk_full") && text.contains("There is not enough free space."), "{text}");
    assert!(!text.contains(TOKEN), "the token must not be saved: {text}");
    assert!(text.contains(r#""job":"export""#));
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn the_llama_mark_stays_at_the_very_end_and_a_stop_stays_a_plain_word() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(py) = python() else { return };
    let cfg = tempfile::tempdir().unwrap();
    let worker = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", cfg.path());
    stub_worker(worker.path(), "import sys\nprint('{\"type\":\"error\",\"code\":\"llama_cpp_missing\",\"message\":\"llama.cpp is missing.\"}', flush=True)\nsys.exit(1)\n");
    let err = run_job(std::path::Path::new(py), worker.path(), &json!({"job": "export"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert!(err.starts_with("llama.cpp is missing. (details: logs/jobs/") && err.ends_with(".log) [llama_cpp_missing]"), "{err}");
    // the worker answering a stop with "Cancelled." is not changed
    stub_worker(worker.path(), "import sys\nprint('{\"type\":\"error\",\"code\":\"export_stopped\",\"message\":\"Cancelled.\"}', flush=True)\nsys.exit(1)\n");
    let err = run_job(std::path::Path::new(py), worker.path(), &json!({"job": "export"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert_eq!(err, "Cancelled.");
    assert_eq!(ndforge_core::envmgr::joblog::latest(10).len(), 2);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn a_silent_worker_gets_a_stall_warning_that_repeats_and_nothing_is_stopped() {
    use ndforge_core::envmgr::jobs::WorkerEvent;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(py) = python() else { return };
    let cfg = tempfile::tempdir().unwrap();
    let worker = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", cfg.path());
    stub_worker(worker.path(), "import sys, time\nprint('{\"type\":\"log\",\"text\":\"start\"}', flush=True)\ntime.sleep(3.2)\nprint('{\"type\":\"result\",\"data\":{\"ok\":true}}', flush=True)\n");
    let mut events = Vec::new();
    // 0.02 minutes = 1.2 s of silence
    let r = run_job(std::path::Path::new(py), worker.path(), &json!({"job": "x", "stall_minutes": 0.02}), &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert_eq!(r["ok"], true, "a stall never ends the job");
    let stalls: Vec<f64> = events.iter().filter_map(|e| match e { WorkerEvent::Other(v) if v["type"] == "stall" => v["minutes"].as_f64(), _ => None }).collect();
    assert!(!stalls.is_empty() && (stalls[0] - 0.02).abs() < 1e-9, "{stalls:?}");
    assert!(stalls.len() >= 2 && (stalls[1] - 0.04).abs() < 1e-9, "the warning comes again after each further period: {stalls:?}");
    // no output, no warning: a worker that writes often
    let mut events = Vec::new();
    stub_worker(worker.path(), "import time\nfor i in range(10):\n    print('{\"type\":\"log\",\"text\":\"tick\"}', flush=True)\n    time.sleep(0.1)\nprint('{\"type\":\"result\",\"data\":{}}', flush=True)\n");
    run_job(std::path::Path::new(py), worker.path(), &json!({"job": "x", "stall_minutes": 0.05}), &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert!(!events.iter().any(|e| matches!(e, WorkerEvent::Other(v) if v["type"] == "stall")));
    // the defaults
    assert_eq!(ndforge_core::envmgr::jobs::stall_minutes(&json!({"job": "train"})), 20.0);
    for j in ["distill_cache", "export", "imatrix_make"] {
        assert_eq!(ndforge_core::envmgr::jobs::stall_minutes(&json!({"job": j})), 60.0);
    }
    assert_eq!(ndforge_core::envmgr::jobs::stall_minutes(&json!({"job": "train", "stall_minutes": 5})), 5.0);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn run_with_grace_asks_with_a_file_first_and_kills_only_after_the_grace_time() {
    use ndforge_core::envmgr::jobs::run_with_grace;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(py) = python() else { return };
    let cfg = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", cfg.path());
    std::env::set_var("NDFORGE_WORKER_DIR", work.path());
    let stop_file = work.path().join("run").join("STOP");
    // a worker that looks for the file and ends by itself
    stub_worker(work.path(), &format!("import os, time\nf = {:?}\nwhile not os.path.exists(f):\n    time.sleep(0.02)\nprint('{{\"type\":\"result\",\"data\":{{\"stopped\":true}}}}', flush=True)\n", stop_file.to_string_lossy()));
    let stop = AtomicBool::new(false);
    let t0 = Instant::now();
    let r = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(Duration::from_millis(400));
            stop.store(true, Ordering::SeqCst);
        });
        run_with_grace(std::path::Path::new(py), &json!({"job": "x"}), &stop_file, &stop, Duration::from_secs(30), &mut |_| {})
    });
    assert_eq!(r.unwrap()["stopped"], true);
    assert!(t0.elapsed() < Duration::from_secs(20) && !stop_file.exists(), "it ended by itself and the file is removed");
    // a worker that ignores the file is killed once the grace time is over
    stub_worker(work.path(), "import time\ntime.sleep(60)\n");
    let stop = AtomicBool::new(true);
    let t0 = Instant::now();
    let e = run_with_grace(std::path::Path::new(py), &json!({"job": "x"}), &stop_file, &stop, Duration::from_millis(400), &mut |_| {}).unwrap_err();
    assert!(e.is_cancelled() && t0.elapsed() < Duration::from_secs(20), "{e}");
    std::env::remove_var("NDFORGE_WORKER_DIR");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
