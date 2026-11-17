//! Every stored secret is hidden by value in logs, errors and the problem report (secrets::redact).
use ndforge_core::secrets;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

#[test]
fn stored_secrets_of_eight_characters_or_more_are_replaced_and_short_ones_are_not() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
    assert_eq!(secrets::redact("nothing stored yet: abcdefgh12345"), "nothing stored yet: abcdefgh12345");
    secrets::set("teacher", "tk-8f3a9c1d-very-secret").unwrap();
    secrets::set("http://train-box:8080", "tok_ABCDEFGHIJ").unwrap();
    secrets::set("short", "abc").unwrap();
    let line = "calling https://x/v1 with Bearer tk-8f3a9c1d-very-secret and token tok_ABCDEFGHIJ, abc and again tk-8f3a9c1d-very-secret";
    assert_eq!(secrets::redact(line), "calling https://x/v1 with Bearer [secret] and token [secret], abc and again [secret]");
    // the patterns of keys that are not stored are still removed
    assert_eq!(secrets::redact("key sk-unsloth-abc123 hf_abcdefghijklmnopqrstuvwxyz0123 done"), "key [key] [token] done");
    // a secret changed on disk is picked up at once
    secrets::set("teacher", "another-secret-value").unwrap();
    assert_eq!(secrets::redact("tk-8f3a9c1d-very-secret another-secret-value"), "tk-8f3a9c1d-very-secret [secret]");
    secrets::remove("teacher").unwrap();
    assert_eq!(secrets::redact("another-secret-value"), "another-secret-value");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn a_longer_secret_is_replaced_before_one_that_is_part_of_it_and_a_damaged_file_hides_nothing_and_breaks_nothing() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
    secrets::set("a", "abcdefgh").unwrap();
    secrets::set("b", "abcdefgh-and-more").unwrap();
    assert_eq!(secrets::redact("x abcdefgh-and-more y abcdefgh z"), "x [secret] y [secret] z");
    std::fs::write(d.path().join("secrets.json"), "{not json").unwrap();
    assert_eq!(secrets::redact("abcdefgh stays"), "abcdefgh stays");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn a_secret_in_a_worker_log_line_and_in_the_saved_job_log_never_survives() {
    use ndforge_core::envmgr::jobs::run_job;
    use serde_json::json;
    use std::sync::atomic::AtomicBool;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(py) = ["python3", "python"].into_iter().find(|n| std::process::Command::new(n).arg("-c").arg("pass").output().map(|o| o.status.success()).unwrap_or(false)) else { return };
    let cfg = tempfile::tempdir().unwrap();
    let worker = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", cfg.path());
    secrets::set("teacher", "tk-8f3a9c1d-very-secret").unwrap();
    let pkg = worker.path().join("ndforge_worker");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("__init__.py"), "").unwrap();
    std::fs::write(pkg.join("__main__.py"), "import sys\nsys.stderr.write('using tk-8f3a9c1d-very-secret now\\n')\nprint('{\"type\":\"log\",\"text\":\"key tk-8f3a9c1d-very-secret\"}', flush=True)\nprint('{\"type\":\"error\",\"code\":\"x\",\"message\":\"failed with tk-8f3a9c1d-very-secret\"}', flush=True)\nsys.exit(1)\n").unwrap();
    let mut seen = Vec::new();
    let err = run_job(std::path::Path::new(py), worker.path(), &json!({"job": "x"}), &AtomicBool::new(false), &mut |e| seen.push(format!("{e:?}"))).unwrap_err().0;
    assert!(!err.contains("tk-8f3a9c1d") && err.contains("[secret]"), "{err}");
    assert!(seen.iter().all(|l| !l.contains("tk-8f3a9c1d")) && seen.iter().any(|l| l.contains("[secret]")), "{seen:?}");
    let log = std::fs::read_to_string(&ndforge_core::envmgr::joblog::latest(1)[0]).unwrap();
    assert!(!log.contains("tk-8f3a9c1d") && log.contains("[secret]"), "{log}");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
