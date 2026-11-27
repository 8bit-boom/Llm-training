//! The Hugging Face token (docs/HF_TOKEN.md): it reaches a job only as the environment variable HF_TOKEN, and it is in no job file, no job log, no event, no error text,
//! no problem report and no diagnostics bundle. Only a made-up token is used here. Needs a python on PATH (skips with a note when there is none).
use ndforge_core::envmgr::jobs::{run_job, WorkerEvent};
use serde_json::json;
use std::io::Read;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());
const FAKE: &str = "hf_FAKETOKEN123";

fn python() -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|n| Command::new(n).arg("-c").arg("pass").output().map(|o| o.status.success()).unwrap_or(false))
}

fn read_zip(path: &std::path::Path) -> Vec<(String, String)> {
    let mut z = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    (0..z.len()).map(|i| {
        let mut f = z.by_index(i).unwrap();
        let mut t = String::new();
        f.read_to_string(&mut t).unwrap();
        (f.name().to_string(), t)
    }).collect()
}

/// A worker that is careless on purpose: it prints the token in a log line, on stderr and in its error text, and says whether it saw the variable.
const CARELESS: &str = r#"import json, os, sys
tok = os.environ.get("HF_TOKEN", "")
sys.stderr.write("library says: using token " + tok + "\n")
print(json.dumps({"type": "log", "text": "token is " + tok}), flush=True)
job = json.load(open(sys.argv[1]))
print(json.dumps({"type": "error", "code": "export_failed", "message": "401 for " + tok + " (saw=" + str(bool(tok)) + ", job_has_it=" + str(tok in open(sys.argv[1]).read()) + ")"}), flush=True)
sys.exit(1)
"#;

#[test]
fn the_fake_token_reaches_the_job_as_an_environment_variable_and_is_nowhere_else() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let Some(py) = python() else { eprintln!("note: no python on PATH, test skipped"); return };
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("env"));
    let worker = d.path().join("worker");
    let pkg = worker.join("ndforge_worker");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("__init__.py"), "").unwrap();
    std::fs::write(pkg.join("__main__.py"), CARELESS).unwrap();
    ndforge_core::general::token_set(FAKE).unwrap();
    // jobs that can reach the Hub get it; the job values never hold it
    let mut errors = Vec::new();
    let mut events = Vec::new();
    for kind in ["export", "think_check", "reshape", "distill", "imatrix_make", "train", "heal_check"] {
        let job = json!({"job": kind, "model": "acme/gated-model"});
        assert!(!job.to_string().contains(FAKE));
        let e = run_job(std::path::Path::new(py), &worker, &job, &AtomicBool::new(false), &mut |ev| events.push(format!("{ev:?}"))).unwrap_err().0;
        assert!(e.contains("saw=True") && e.contains("job_has_it=False"), "{kind}: {e}");
        errors.push(e);
    }
    // a job that never reaches the Hub does not get it
    let e = run_job(std::path::Path::new(py), &worker, &json!({"job": "hello"}), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert!(e.contains("saw=False"), "{e}");
    for t in errors.iter().chain(events.iter()) {
        assert!(!t.contains(FAKE), "the token is in an event or an error text: {t}");
    }
    assert!(events.iter().any(|e| e.contains("token is [")), "the log line came through, scrubbed: {events:?}");
    let _ = WorkerEvent::Log(String::new());
    // the saved job logs
    let logs = ndforge_core::envmgr::joblog::latest(20);
    assert_eq!(logs.len(), 8);
    for l in &logs {
        let t = std::fs::read_to_string(l).unwrap();
        assert!(!t.contains(FAKE), "{}: {t}", l.display());
    }
    // the problem report and the diagnostics bundle (which holds the last ten job logs, the settings and the report)
    let settings = ndforge_core::settings::Settings::default();
    assert!(!ndforge_core::diagnose::report(None, &settings).contains(FAKE));
    let zip = ndforge_core::diagnose::bundle(None, &settings, &d.path().join("out.zip")).unwrap();
    let entries = read_zip(&zip);
    assert!(entries.iter().any(|(n, _)| n.starts_with("logs/jobs/")));
    for (n, t) in &entries {
        assert!(!t.contains(FAKE), "{n} holds the token");
    }
    // the settings and the state the window reads say only that there is a token
    assert!(!serde_json::to_string(&settings).unwrap().contains(FAKE));
    assert!(!ndforge_core::general::state(None).to_string().contains(FAKE) && ndforge_core::general::state(None)["has_token"] == true);
    // the secrets file is the only place it is kept, and clearing it removes it
    let kept = std::fs::read_to_string(d.path().join("cfg").join("secrets.json")).unwrap();
    assert!(kept.contains(FAKE));
    ndforge_core::general::token_clear().unwrap();
    assert!(!std::fs::read_to_string(d.path().join("cfg").join("secrets.json")).unwrap().contains(FAKE));
    assert!(ndforge_core::envmgr::jobs::job_env(&json!({"job": "export"})).is_empty());
    std::env::remove_var("NDFORGE_CONFIG_DIR");
    std::env::remove_var("NDFORGE_ENV_DIR");
}
