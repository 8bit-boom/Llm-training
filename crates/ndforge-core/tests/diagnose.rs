//! The diagnostics bundle (diagnose::bundle): the entries, and that nothing secret or personal is in them.
use ndforge_core::diagnose::bundle;
use ndforge_core::settings::Settings;
use std::io::Read;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn read_zip(path: &std::path::Path) -> Vec<(String, String)> {
    let mut z = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
    (0..z.len()).map(|i| {
        let mut f = z.by_index(i).unwrap();
        let mut t = String::new();
        f.read_to_string(&mut t).unwrap();
        (f.name().to_string(), t)
    }).collect()
}

#[test]
fn the_bundle_has_the_records_and_neither_a_stored_secret_nor_the_home_folder() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    let cfg = d.path().join("cfg");
    std::env::set_var("NDFORGE_CONFIG_DIR", &cfg);
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("env"));
    let secret = "tk-8f3a9c1d-very-secret";
    ndforge_core::secrets::set("teacher", secret).unwrap();
    let home = dirs::home_dir().unwrap().to_string_lossy().to_string();
    // a job log that mentions both
    let logs = ndforge_core::envmgr::joblog::dir();
    std::fs::create_dir_all(&logs).unwrap();
    for i in 0..12 {
        std::fs::write(logs.join(format!("20261117-0000{i:02}-export.log")), format!("line with {secret} and {home}/models/x\n")).unwrap();
    }
    std::fs::create_dir_all(d.path().join("env")).unwrap();
    std::fs::write(d.path().join("env").join("ndforge_env.json"), "{\"wheel\": \"cu128\"}").unwrap();
    // a project with a heal record and an export record
    let p = ndforge_core::project::Project::create(d.path(), "P").unwrap();
    let m = ndforge_core::reshape::runs::models_dir(&p).join("slim-1");
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(m.join("heal.json"), format!("{{\"status\": \"stopped\", \"folder\": \"{home}/p\", \"key\": \"{secret}\"}}")).unwrap();
    let e = ndforge_core::export::runs::exports_dir(&p).join("run1-q4_k_m");
    std::fs::create_dir_all(&e).unwrap();
    std::fs::write(e.join("meta.json"), "{\"quant\": \"q4_k_m\"}").unwrap();
    let mut settings = Settings::default();
    settings.teacher.url = "http://localhost:11434".into();
    settings.webhook_url = "https://ntfy.sh/my-private-topic".into();
    let out = bundle(Some(&p), &settings, &d.path().join("save")).unwrap();
    assert!(out.starts_with(d.path().join("save")) && out.extension().unwrap() == "zip");
    let entries = read_zip(&out);
    let names: Vec<&str> = entries.iter().map(|(n, _)| n.as_str()).collect();
    for want in ["report.txt", "settings.json", "ndforge_env.json", "worker_hello.json", "heal.json", "export.json", "graphics_card.txt", "timings.json"] {
        assert!(names.contains(&want), "{want} in {names:?}");
    }
    assert_eq!(names.iter().filter(|n| n.starts_with("logs/jobs/")).count(), 10, "the last ten job logs");
    for (n, t) in &entries {
        assert!(!t.contains(secret), "{n} has a secret");
        assert!(home.len() < 2 || !t.contains(&home), "{n} has the home folder");
        assert!(!t.contains("my-private-topic"), "{n} has the webhook address");
    }
    let settings_json = &entries.iter().find(|(n, _)| n == "settings.json").unwrap().1;
    assert!(settings_json.contains("http://localhost:11434"), "addresses stay");
    assert!(entries.iter().find(|(n, _)| n == "heal.json").unwrap().1.contains("~/p"), "the home folder is shortened");
    assert!(entries.iter().find(|(n, _)| n == "worker_hello.json").unwrap().1.contains("not installed"));
    // a file name given directly is used as it is
    let direct = bundle(None, &settings, &d.path().join("one.zip")).unwrap();
    assert_eq!(direct, d.path().join("one.zip"));
    assert!(!read_zip(&direct).iter().any(|(n, _)| n == "heal.json"));
    std::env::remove_var("NDFORGE_CONFIG_DIR");
    std::env::remove_var("NDFORGE_ENV_DIR");
}

#[test]
fn the_bundle_has_the_timings_of_the_latest_logs_as_data() {
    let r = ndforge_core::timing::report_of_latest(10);
    assert!(r["summary"].is_object() && r["advice"].is_array());
    assert!(r["summary"]["logs"].is_u64());
}

#[test]
fn the_bundle_holds_the_vllm_self_test_report_when_there_is_one() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let dir = ndforge_core::vllm::selftest_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("vllm_selftest.json"), "{\"verdict\": \"partial\"}").unwrap();
    std::fs::write(dir.join("vllm_selftest.md"), "# vLLM self-test").unwrap();
    let zip = bundle(None, &Settings::default(), &d.path().join("out.zip")).unwrap();
    let names: Vec<String> = read_zip(&zip).into_iter().map(|(n, _)| n).collect();
    assert!(names.contains(&"vllm_selftest.json".to_string()) && names.contains(&"vllm_selftest.md".to_string()));
    std::fs::remove_dir_all(&dir).unwrap();
    let zip2 = bundle(None, &Settings::default(), &d.path().join("out2.zip")).unwrap();
    assert!(!read_zip(&zip2).iter().any(|(n, _)| n.starts_with("vllm_selftest")));
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
