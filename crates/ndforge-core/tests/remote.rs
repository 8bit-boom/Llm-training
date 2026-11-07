use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::project::Project;
use ndforge_core::remote::client::RemoteClient;
use ndforge_core::remote::run::{execute_remote, Target};
use ndforge_core::remote::server::{spawn, Handle, ServerConfig};
use ndforge_core::remote::zipio::{extract_safe, zip_dir};
use ndforge_core::sample;
use ndforge_core::train::catalog::find;
use ndforge_core::train::plan::{base_hyper, Method, Preset};
use ndforge_core::train::runs;
use serde_json::{json, Value};
use tempfile::tempdir;

const TOKEN: &str = "test-token-0123456789";
static ENV: Mutex<()> = Mutex::new(());

/// A fake "python" for the server: `$5` is the job file; it reads out / resume from it.
#[cfg(unix)]
fn fake_python(root: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let py = root.join("srv-python");
    std::fs::write(&py, format!("#!/bin/sh\nout=$(sed -n 's/.*\"out\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\nresume=$(sed -n 's/.*\"resume\":\\(true\\|false\\).*/\\1/p' \"$5\")\ninit=$(sed -n 's/.*\"init_adapter\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    py
}

const QUICK_WORKER: &str = r#"echo '{"type":"start","total_steps":2}'
echo '{"type":"sample","step":0,"question":"Q","answer":"before"}'
echo "{\"type\":\"log\",\"text\":\"resume=$resume init=$init\"}"
echo '{"type":"metric","step":1,"total_steps":2,"loss":2.5,"vram_mb":9000}'
echo '{"type":"metric","step":2,"total_steps":2,"loss":1.5,"vram_mb":9500}'
mkdir -p "$out/adapter/sub"
echo '{}' > "$out/adapter/adapter_config.json"
printf 'WEIGHTS' > "$out/adapter/sub/adapter_model.safetensors"
echo "{\"type\":\"result\",\"data\":{\"steps\":2,\"total_steps\":2,\"final_loss\":1.5,\"peak_vram_mb\":9500,\"stopped\":false,\"adapter_dir\":\"$out/adapter\"}}""#;

struct Srv {
    h: Handle,
    data: PathBuf,
}

fn serve(root: &Path, python: PathBuf, tweak: impl FnOnce(&mut ServerConfig)) -> Srv {
    let mut cfg = ServerConfig { listen: "127.0.0.1:0".into(), token: TOKEN.into(), data_dir: root.join("srv-data"), python, ollama_url: None, require_gpu: false,
                                 max_upload: 50 << 20, grace: Duration::from_secs(5) };
    tweak(&mut cfg);
    let data = cfg.data_dir.clone();
    Srv { h: spawn(cfg).unwrap(), data }
}

impl Srv {
    fn target(&self) -> Target {
        Target { name: "TrueNAS".into(), url: format!("127.0.0.1:{}", self.h.addr.port()), token: TOKEN.into() }
    }
    fn client(&self) -> RemoteClient {
        let t = self.target();
        RemoteClient::new(&t.url, &t.token)
    }
}

impl Drop for Srv {
    fn drop(&mut self) {
        self.h.shutdown();
    }
}

fn project(root: &Path) -> (Project, ndforge_core::train::catalog::ModelEntry, Value) {
    let mut p = Project::create(&root.join("projects"), "Ashfall Reach").unwrap();
    let f = root.join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&ndforge_core::bundle::open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    let m = find("qwen3.5-4b").unwrap();
    let job = runs::build_job(&p, &m, Method::Lora, &base_hyper(Preset::Quick), 1, "run1", false).unwrap();
    (p, m, job)
}

fn go(p: &Project, m: &ndforge_core::train::catalog::ModelEntry, job: Value, t: &Target, stop: Arc<AtomicBool>, resume: bool, events: &mut Vec<Value>) -> ndforge_core::Result<Value> {
    execute_remote(p, m, Method::Lora, &base_hyper(Preset::Quick), "run1", job, t, stop, resume, Duration::from_millis(100), &mut |e| events.push(e))
}

fn http(addr: u16, method: &str, path: &str, token: Option<&str>, body: &[u8]) -> (u16, String) {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", addr)).unwrap();
    let auth = token.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
    write!(s, "{method} {path} HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
    s.write_all(body).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let code = out.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (code, out.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

#[cfg(unix)]
#[test]
fn every_request_needs_the_token_and_ids_and_file_names_are_checked() {
    let d = tempdir().unwrap();
    let srv = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |_| {});
    let port = srv.h.addr.port();
    assert_eq!(http(port, "GET", "/v1/status", None, b"").0, 401);
    assert_eq!(http(port, "GET", "/v1/status", Some("wrong-token-0123456789"), b"").0, 401);
    assert_eq!(http(port, "GET", "/v1/status", Some(&TOKEN[..5]), b"").0, 401);
    let (code, body) = http(port, "GET", "/v1/status", Some(TOKEN), b"");
    let st: Value = serde_json::from_str(&body).unwrap();
    assert_eq!((code, st["service"].as_str(), st["env_ready"].clone(), st["busy"].clone(), st["cpu_ok"].clone()), (200, Some("nd-lore-forge-worker"), json!(true), json!(false), json!(true)));
    // bad ids and names never touch the disk outside the data folder
    assert_eq!(http(port, "PUT", "/v1/jobs/..%2Fx/files/train.jsonl", Some(TOKEN), b"x").0, 404);
    assert_eq!(http(port, "PUT", "/v1/jobs/ok-id/files/evil.sh", Some(TOKEN), b"x").0, 404);
    assert_eq!(http(port, "GET", "/v1/jobs/../etc/events", Some(TOKEN), b"").0, 404);
    assert_eq!(http(port, "POST", "/v1/jobs/ok-id", Some(TOKEN), b"{\"job\":{\"job\":\"export\"}}").0, 400);
    assert_eq!(http(port, "POST", "/v1/jobs/ok-id", Some(TOKEN), b"{\"job\":{\"job\":\"train\"}}").0, 400);         // no data uploaded yet
    assert_eq!(http(port, "GET", "/v1/jobs/never-existed", Some(TOKEN), b"").0, 404);
    let (code, body) = http(port, "POST", "/v1/jobs/ok-id", Some(TOKEN), b"{\"job\":{\"job\":\"train\",\"method\":\"full\"}}");
    assert!(code == 400 && body.contains("Full fine-tuning is not available on the server"), "{code} {body}");
    assert!(!d.path().join("evil.sh").exists() && !srv.data.join("..").join("x").exists());
    // a short token is refused at start-up
    assert!(spawn(ServerConfig { token: "short".into(), listen: "127.0.0.1:0".into(), data_dir: d.path().join("x"), python: PathBuf::new(), ollama_url: None, require_gpu: false, max_upload: 1, grace: Duration::from_secs(1) }).is_err());
}

#[cfg(unix)]
#[test]
fn a_whole_run_on_the_server_comes_back_with_events_charts_and_the_adapter() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let srv = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |_| {});
    let (p, m, job) = project(d.path());
    let mut events = Vec::new();
    let r = go(&p, &m, job, &srv.target(), Arc::new(AtomicBool::new(false)), false, &mut events).unwrap();
    assert_eq!(r["steps"], 2);
    let stages: Vec<&str> = events.iter().filter(|e| e["kind"] == "stage").filter_map(|e| e["text"].as_str()).collect();
    assert_eq!(stages, ["Connecting to TrueNAS", "Sending the training data", "Training on TrueNAS", "Fetching the trained model"]);
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["step"] == "upload"));
    assert_eq!(events.iter().filter(|e| e["kind"] == "metric").count(), 2);
    assert!(events.iter().any(|e| e["kind"] == "sample" && e["answer"] == "before"));
    assert!(events.iter().any(|e| e["kind"] == "log" && e["text"] == "resume=false init="));
    assert!(!events.iter().any(|e| e["kind"] == "done" || e["kind"] == "error"));          // the end is the return value, not an event
    // the server got exactly the training file, the PC got the adapter and the charts
    assert_eq!(std::fs::read(srv.data.join("jobs/run1/in/train.jsonl")).unwrap(), std::fs::read(p.folder.join("datasets/v1/train.jsonl")).unwrap());
    assert_eq!(std::fs::read(p.folder.join("runs/run1/adapter/sub/adapter_model.safetensors")).unwrap(), b"WEIGHTS");
    assert!(p.folder.join("runs/run1/adapter/adapter_config.json").is_file() && !p.folder.join("runs/run1/adapter.zip").exists());
    assert_eq!(runs::read_metrics(&p, "run1").len(), 4);
    let st = &runs::list_runs(&p)[0];
    assert_eq!((st["status"].as_str(), st["server"].as_str(), st["step"].as_u64(), st["has_adapter"].clone()), (Some("done"), Some("TrueNAS"), Some(2), json!(true)));
    assert!(p.progress()[2].1);
    assert!(runs::load_measured("qwen3.5-4b", Method::Lora).is_none(), "the server's card is not this PC's");
    // the server forgets nothing until told to
    assert_eq!(srv.client().job_state("run1").unwrap().unwrap()["state"]["status"], "done");
    srv.client().delete("run1").unwrap();
    assert!(srv.client().job_state("run1").unwrap().is_none());
}

#[cfg(unix)]
#[test]
fn stop_saves_the_adapter_and_the_run_can_be_continued_from_the_servers_checkpoint() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let body = r#"echo "{\"type\":\"log\",\"text\":\"resume=$resume\"}"
if [ "$resume" = "true" ]; then
  echo '{"type":"metric","step":9,"total_steps":9,"loss":0.5,"vram_mb":100}'
  mkdir -p "$out/adapter"; echo '{}' > "$out/adapter/adapter_config.json"
  echo "{\"type\":\"result\",\"data\":{\"steps\":9,\"total_steps\":9,\"final_loss\":0.5,\"stopped\":false}}"
  exit 0
fi
i=0
while [ ! -f "$out/STOP" ]; do i=$((i+1)); echo "{\"type\":\"metric\",\"step\":$i,\"total_steps\":9,\"loss\":1.0,\"vram_mb\":100}"; sleep 0.1; done
mkdir -p "$out/checkpoints/checkpoint-3" "$out/adapter"; echo '{}' > "$out/adapter/adapter_config.json"
echo "{\"type\":\"result\",\"data\":{\"steps\":$i,\"total_steps\":9,\"stopped\":true}}""#;
    let srv = serve(d.path(), fake_python(d.path(), body), |_| {});
    let (p, m, job) = project(d.path());
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let t = std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(1200)); s2.store(true, Ordering::SeqCst); });
    let mut ev = Vec::new();
    let r = go(&p, &m, job.clone(), &srv.target(), stop, false, &mut ev).unwrap();
    t.join().unwrap();
    assert_eq!(r["stopped"], true);
    assert!(ev.iter().any(|e| e["kind"] == "stage" && e["text"] == "Stopping after this step"));
    let st = &runs::list_runs(&p)[0];
    assert_eq!((st["status"].as_str(), st["has_adapter"].clone()), (Some("stopped"), json!(true)));
    assert!(runs::resumable(&p, "run1") && !p.progress()[2].1);
    // continue: the server still has the checkpoint, so the job starts again with resume = true on the same data
    let mut ev2 = Vec::new();
    let r2 = go(&p, &m, job, &srv.target(), Arc::new(AtomicBool::new(false)), true, &mut ev2).unwrap();
    assert_eq!(r2["steps"], 9);
    assert!(ev2.iter().any(|e| e["kind"] == "log" && e["text"] == "resume=true"));
    assert_eq!(runs::list_runs(&p)[0]["status"], "done");
}

#[cfg(unix)]
#[test]
fn a_second_job_is_refused_while_one_runs_and_a_lost_client_can_reconnect() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let body = r#"echo '{"type":"metric","step":1,"total_steps":2,"loss":2.0,"vram_mb":100}'
sleep 1.5
mkdir -p "$out/adapter"; echo '{}' > "$out/adapter/adapter_config.json"
echo '{"type":"metric","step":2,"total_steps":2,"loss":1.0,"vram_mb":100}'
echo "{\"type\":\"result\",\"data\":{\"steps\":2,\"total_steps\":2,\"final_loss\":1.0,\"stopped\":false}}""#;
    let srv = serve(d.path(), fake_python(d.path(), body), |_| {});
    let (p, m, job) = project(d.path());
    // a client starts the job by hand and "disappears"
    let c = srv.client();
    c.upload("run1", "train.jsonl", &p.folder.join("datasets/v1/train.jsonl"), &mut |_| {}).unwrap();
    c.start("run1", &job).unwrap();
    assert!(c.start("run1", &job).unwrap_err().0.contains("busy"));
    c.upload("other", "train.jsonl", &p.folder.join("datasets/v1/train.jsonl"), &mut |_| {}).unwrap();
    assert!(c.start("other", &job).unwrap_err().0.contains("busy"));
    // a new job from the app is refused too (not resuming)
    let e = go(&p, &m, job.clone(), &srv.target(), Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err();
    assert!(e.0.contains("busy"), "{e}");
    // "Continue" reconnects to the running job, follows it to the end and fetches the adapter
    let mut ev = Vec::new();
    let r = go(&p, &m, job, &srv.target(), Arc::new(AtomicBool::new(false)), true, &mut ev).unwrap();
    assert_eq!(r["steps"], 2);
    assert!(ev.iter().any(|e| e["kind"] == "stage" && e["text"].as_str().unwrap().starts_with("Reconnecting")));
    assert!(!ev.iter().any(|e| e["text"] == "Sending the training data"));
    assert!(p.folder.join("runs/run1/adapter/adapter_config.json").is_file());
}

#[cfg(unix)]
#[test]
fn a_warm_start_sends_the_earlier_adapter_and_the_server_points_the_job_at_it() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let srv = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |_| {});
    let (p, m, mut job) = project(d.path());
    let old = p.folder.join("runs/old/adapter");
    std::fs::create_dir_all(old.join("sub")).unwrap();
    std::fs::write(old.join("adapter_config.json"), "{}").unwrap();
    std::fs::write(old.join("sub/w.bin"), "OLD").unwrap();
    job["init_adapter"] = json!(old);
    let mut ev = Vec::new();
    go(&p, &m, job, &srv.target(), Arc::new(AtomicBool::new(false)), false, &mut ev).unwrap();
    assert!(ev.iter().any(|e| e["kind"] == "log" && e["text"].as_str().unwrap().contains("init=") && e["text"].as_str().unwrap().contains("/jobs/run1/in/init_adapter")), "{ev:?}");
    assert_eq!(std::fs::read(srv.data.join("jobs/run1/in/init_adapter/sub/w.bin")).unwrap(), b"OLD");
}

#[test]
fn zips_round_trip_and_unsafe_entries_are_refused() {
    let d = tempdir().unwrap();
    let src = d.path().join("src");
    std::fs::create_dir_all(src.join("a/b")).unwrap();
    std::fs::write(src.join("x.txt"), "X").unwrap();
    std::fs::write(src.join("a/b/y.txt"), "Y").unwrap();
    let z = d.path().join("t.zip");
    zip_dir(&src, &z).unwrap();
    let dest = d.path().join("dest");
    assert_eq!(extract_safe(&z, &dest).unwrap(), 2);
    assert_eq!(std::fs::read_to_string(dest.join("a/b/y.txt")).unwrap(), "Y");
    // an entry that climbs out is refused and nothing is written
    let evil = d.path().join("evil.zip");
    let mut w = zip::ZipWriter::new(std::fs::File::create(&evil).unwrap());
    w.start_file("../escaped.txt", zip::write::SimpleFileOptions::default()).unwrap();
    w.write_all(b"E").unwrap();
    w.start_file("/abs.txt", zip::write::SimpleFileOptions::default()).unwrap();
    w.write_all(b"E").unwrap();
    w.finish().unwrap();
    assert!(extract_safe(&evil, &d.path().join("out")).unwrap_err().0.contains("unsafe path"));
    assert!(!d.path().join("escaped.txt").exists() && !Path::new("/abs.txt").exists());
}

#[cfg(unix)]
#[test]
fn problems_are_explained_in_plain_words() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let (p, m, job) = project(d.path());
    // nothing there
    let dead = Target { name: "TrueNAS".into(), url: "127.0.0.1:1".into(), token: TOKEN.into() };
    assert!(go(&p, &m, job.clone(), &dead, Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err().0.contains("did not answer"));
    // wrong token
    let srv = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |_| {});
    let bad = Target { token: "not-the-right-token-123".into(), ..srv.target() };
    assert!(go(&p, &m, job.clone(), &bad, Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err().0.contains("refused the access token"));
    // no card (a production server) and no training software
    let strict = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |c| { c.require_gpu = true; c.data_dir = d.path().join("srv2"); });
    assert!(go(&p, &m, job.clone(), &strict.target(), Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err().0.contains("no usable graphics card"));
    let noenv = serve(d.path(), d.path().join("missing-python"), |c| c.data_dir = d.path().join("srv3"));
    assert!(go(&p, &m, job.clone(), &noenv.target(), Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err().0.contains("not set up on the server"));
    // an upload that is too large
    let small = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |c| { c.max_upload = 10; c.data_dir = d.path().join("srv4"); });
    let e = go(&p, &m, job.clone(), &small.target(), Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err();
    assert!(e.0.contains("larger than this server accepts"), "{e}");
    // the worker fails on the server: the failure comes back as the run's error
    let failing = serve(d.path(), fake_python(d.path(), r#"echo '{"type":"error","code":"train_failed","message":"The graphics card ran out of memory."}'; exit 1"#), |c| c.data_dir = d.path().join("srv5"));
    let e = go(&p, &m, job, &failing.target(), Arc::new(AtomicBool::new(false)), false, &mut Vec::new()).unwrap_err();
    assert!(e.0.contains("ran out of memory"), "{e}");
    let st = runs::list_runs(&p).into_iter().find(|r| r["name"] == "run1").unwrap();
    assert_eq!((st["status"].as_str(), st["error"].as_str()), (Some("failed"), Some("The graphics card ran out of memory.")));
}

/// A tiny Ollama: /api/ps lists two loaded models, /api/generate records the unloads.
fn mock_ollama() -> (String, Arc<Mutex<Vec<Value>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("127.0.0.1:{}", l.local_addr().unwrap().port());
    let log: Arc<Mutex<Vec<Value>>> = Arc::default();
    let l2 = log.clone();
    std::thread::spawn(move || {
        for st in l.incoming() {
            let Ok(mut st) = st else { break };
            let mut r = BufReader::new(st.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() { break; }
                if let Some(v) = h.to_lowercase().strip_prefix("content-length:") { len = v.trim().parse().unwrap_or(0); }
            }
            let mut b = vec![0u8; len];
            r.read_exact(&mut b).unwrap();
            let body = if path == "/api/ps" { json!({"models": [{"name": "gemma4:26b"}, {"name": "qwen3:4b"}]}) } else { l2.lock().unwrap().push(serde_json::from_slice(&b).unwrap_or(Value::Null)); json!({}) };
            let t = body.to_string();
            let _ = write!(st, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{t}", t.len());
        }
    });
    (url, log)
}

#[cfg(unix)]
#[test]
fn the_card_is_freed_by_unloading_ollama_before_training() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let (ollama, log) = mock_ollama();
    let srv = serve(d.path(), fake_python(d.path(), QUICK_WORKER), |c| c.ollama_url = Some(ollama.clone()));
    assert_eq!(srv.client().unload_ollama().unwrap(), ["gemma4:26b", "qwen3:4b"]);
    assert_eq!(log.lock().unwrap().len(), 2);
    assert!(log.lock().unwrap().iter().all(|g| g["keep_alive"] == 0));
    log.lock().unwrap().clear();
    let (p, m, job) = project(d.path());
    let mut ev = Vec::new();
    go(&p, &m, job, &srv.target(), Arc::new(AtomicBool::new(false)), false, &mut ev).unwrap();
    assert!(ev.iter().any(|e| e["kind"] == "log" && e["text"].as_str().unwrap().contains("Freed the server's graphics card: unloaded gemma4:26b, qwen3:4b")));
    assert_eq!(log.lock().unwrap().len(), 2);
}

#[test]
fn servers_are_remembered_their_tokens_are_kept_private_and_plans_use_the_servers_card() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfgs"));
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("ps"));
    let mut m = AppModel::load();
    m.create_project("Ashfall Reach").unwrap();
    let f = d.path().join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    m.apply_bundle(f.to_str().unwrap()).unwrap();
    m.dataset_generate(json!({})).unwrap();
    m.dataset_save().unwrap();
    assert!(m.add_train_server("TrueNAS", "truenas:8765", "").unwrap_err().0.contains("access token"));
    m.add_train_server("TrueNAS", "truenas:8765", " secret-token-1234567 ").unwrap();
    let url = "http://truenas:8765";
    assert_eq!(AppModel::load().settings.train_servers.len(), 1);                                 // remembered
    let t = m.target_for(url).unwrap();
    assert_eq!((t.name.as_str(), t.token.as_str()), ("TrueNAS", "secret-token-1234567"));
    let settings = std::fs::read_to_string(d.path().join("cfgs/settings.json")).unwrap();
    assert!(!settings.contains("secret-token"), "the token must not be in the settings file");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(d.path().join("cfgs/secrets.json")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // the plan follows the chosen server's card: a 16 GB T10 cannot do Gemma 4 E4B in 16-bit, a bigger local card can
    assert_eq!(m.train_plan("gemma4-e4b", Method::Lora, Preset::Normal, Some(url), None).unwrap()["fit"], "unknown");                 // not checked yet
    m.set_remote_status(url, json!({"gpu": {"name": "Tesla T10", "vram_gb": 15.6, "precision": "fp16"}, "busy": false}));
    assert_eq!(m.train_plan("gemma4-e4b", Method::Lora, Preset::Normal, Some(url), None).unwrap()["fit"], "no");
    assert_ne!(m.train_plan("gemma4-e4b", Method::Qlora, Preset::Normal, Some(url), None).unwrap()["fit"], "no");
    let st = m.train_state(Some(url)).unwrap();
    let e4b = st["models"].as_array().unwrap().iter().find(|c| c["model"]["id"] == "gemma4-e4b").unwrap();
    assert_eq!((e4b["lora"]["fit"].as_str(), e4b["recommended"].as_str()), (Some("no"), Some("qlora")));   // the recommendation follows the chosen card too
    assert_eq!(st["servers"][0]["status"]["gpu"]["name"], "Tesla T10");
    // a job for the server carries the target; the server's run is not sized with this PC's measured memory
    let job = m.train_job("qwen3.5-2b", Method::Lora, Preset::Quick, &json!({}), None, Some(url)).unwrap();
    assert_eq!(job.remote.as_ref().map(|t| t.url.as_str()), Some(url));
    assert!(m.train_job("qwen3.5-2b", Method::Lora, Preset::Quick, &json!({}), None, Some("http://unknown:1")).unwrap_err().0.contains("no longer in the list"));
    assert!(m.train_job("qwen3.5-2b", Method::Lora, Preset::Quick, &json!({}), None, None).unwrap().remote.is_none());
    m.remove_train_server(url).unwrap();
    assert!(m.target_for(url).is_err() && AppModel::load().settings.train_servers.is_empty());
    assert!(!std::fs::read_to_string(d.path().join("cfgs/secrets.json")).unwrap().contains("secret-token"));
}
