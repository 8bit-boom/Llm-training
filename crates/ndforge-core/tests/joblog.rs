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
