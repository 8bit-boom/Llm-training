//! Training in Unsloth Desktop: the client, the run and the stand-in Unsloth they are tested against.
mod common;

use common::fake_unsloth::{Fake, KEY};
use tempfile::tempdir;

#[test]
fn the_stand_in_answers_health_with_and_without_a_key() {
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let plain: serde_json::Value = ureq::get(&format!("{}/api/health", f.url)).call().unwrap().into_json().unwrap();
    assert_eq!(plain["service"], "Unsloth UI Backend");
    assert!(plain["version"].is_null());
    let keyed: serde_json::Value = ureq::get(&format!("{}/api/health", f.url)).set("Authorization", &format!("Bearer {KEY}")).call().unwrap().into_json().unwrap();
    assert!(keyed["version"].is_string());
    assert!(matches!(ureq::get(&format!("{}/v1/models", f.url)).call(), Err(ureq::Error::Status(401, _))));
}

use ndforge_core::unsloth::client::{candidate_urls, safe_id, scrub, UnslothClient};

#[test]
fn a_probe_tells_desktop_from_studio_from_something_else_and_a_bad_key() {
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let p = UnslothClient::new(&f.url, KEY).probe().unwrap();
    assert!(p.unsloth && p.desktop && p.fine_tuning && p.checkpoints && p.key_ok == Some(true) && p.version.as_deref() == Some("2026.11.1") && p.chat_only == Some(false), "{p:?}");
    assert!(!p.export, "the stand-in has no export routes");
    // no key: it is Unsloth, nothing more is known
    let p = UnslothClient::new(&f.url, "").probe().unwrap();
    assert!(p.unsloth && p.key_ok.is_none() && p.version.is_none() && !p.fine_tuning);
    // a wrong key: health answers without a version
    let p = UnslothClient::new(&f.url, "sk-unsloth-wrong").probe().unwrap();
    assert_eq!((p.unsloth, p.key_ok, p.fine_tuning), (true, Some(false), false));
    // Studio from a terminal: no desktop fields
    f.with(|s| s.desktop = false);
    assert!(!UnslothClient::new(&f.url, KEY).probe().unwrap().desktop);
    // too old: no fine-tuning routes
    f.with(|s| s.fine_tuning = false);
    let p = UnslothClient::new(&f.url, KEY).probe().unwrap();
    assert!(p.key_ok == Some(true) && !p.fine_tuning);
    // something else answers
    f.with(|s| s.not_unsloth = true);
    assert!(!UnslothClient::new(&f.url, KEY).probe().unwrap().unsloth);
}

#[test]
fn nothing_listening_gives_a_plain_message_without_the_key() {
    // a port that was just free
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let e = UnslothClient::new(&format!("127.0.0.1:{port}"), KEY).probe().unwrap_err().0;
    assert!(e.contains("does not answer") && e.contains("Open Unsloth Desktop") && !e.contains("sk-unsloth-") && !e.contains("another computer"), "{e}");
}

#[test]
fn the_messages_for_a_refused_key_an_old_unsloth_and_an_unknown_setting_are_plain() {
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let bad = UnslothClient::new(&f.url, "sk-unsloth-wrong-key");
    let e = bad.job("ftjob_x").unwrap_err().0;
    assert!(e.contains("refused the API key") && !e.contains("wrong-key"), "{e}");
    let good = UnslothClient::new(&f.url, KEY);
    f.with(|s| s.fine_tuning = false);
    assert!(good.start_job(&serde_json::json!({})).unwrap_err().0.contains("no training API for other programs"));
    f.with(|s| s.fine_tuning = true);
    let e = good.start_job(&serde_json::json!({"model": "m", "training_file": "file-x", "colour": 1, "mood": 2})).unwrap_err().0;
    assert!(e.contains("does not know the setting colour, mood"), "{e}");
    // an error body that carries a key is scrubbed
    assert_eq!(scrub("bad sk-unsloth-abc_123-x here"), "bad [key] here");
}

#[test]
fn an_upload_arrives_whole_as_a_fine_tune_file_and_a_job_id_is_checked() {
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let file = d.path().join("train.jsonl");
    let content = "{\"messages\":[]}\n".repeat(2000);
    std::fs::write(&file, &content).unwrap();
    let c = UnslothClient::new(&f.url, KEY);
    let mut last = 0.0;
    let id = c.upload_jsonl("train.jsonl", &file, &mut |v| last = v).unwrap();
    assert!(id.starts_with("file-") && last > 0.5);
    f.with(|s| assert_eq!((s.files.len(), s.files[0].1.as_str(), s.files[0].2.as_slice()), (1, "train.jsonl", content.as_bytes())));
    assert!(safe_id("ftjob_q0123-ab").is_ok() && safe_id("../x").is_err() && safe_id("a/b").is_err() && safe_id("").is_err());
    assert!(c.job("../x").is_err());
    assert!(c.hardware().unwrap()["gpu"]["name"].as_str().unwrap().contains("5090"));
    assert!(candidate_urls().iter().any(|u| u == "http://127.0.0.1:8888"));
}

#[test]
fn test_and_find_describe_the_connection_in_plain_words() {
    use ndforge_core::unsloth::client::{check, find};
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let r = check(&f.url, KEY).unwrap();
    let s = r["summary"].as_str().unwrap();
    assert!(s.contains("Unsloth Desktop 2026.11.1 on this computer") && s.contains("5090") && s.contains("training API: yes"), "{s}");
    assert_eq!(r["hardware"]["gpu"]["precision"], "bf16");
    assert!(check(&f.url, "sk-unsloth-nope").unwrap_err().0.contains("refused the API key"));
    f.with(|s| s.not_unsloth = true);
    assert!(check(&f.url, KEY).unwrap_err().0.contains("not Unsloth"));
    f.with(|s| s.not_unsloth = false);
    let dead = "http://127.0.0.1:1".to_string();
    let (url, p) = find(&[dead, f.url.clone()]).unwrap();
    assert!(url == f.url && p.unsloth && p.desktop);
    assert!(find(&["http://127.0.0.1:1".to_string()]).is_none());
}

fn sample_hyper() -> ndforge_core::train::plan::Hyper {
    let mut h = ndforge_core::train::plan::base_hyper(ndforge_core::train::plan::Preset::Normal);
    h.rank = 16; h.alpha = 16; h.lr = 2e-4; h.epochs = 2.0; h.max_steps = None; h.batch = 2; h.grad_accum = 4; h.max_seq = 4096; h.warmup = 10; h.save_steps = 100; h.seed = 3407;
    h.stages = "mixed".into();
    h
}

fn sample_model() -> ndforge_core::train::catalog::ModelEntry {
    let mut m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    m.repo = "unsloth/Qwen3.5-2B".into();
    m
}

#[test]
fn the_job_body_equals_the_fixture_both_products_share_and_every_key_is_one_unsloth_knows() {
    use ndforge_core::train::plan::Method;
    use ndforge_core::unsloth::run::job_body;
    let (t, e) = ("file-00000000000000000000000000000001", "file-00000000000000000000000000000002");
    let b = job_body(&sample_model(), Method::Lora, &sample_hyper(), "ashfall-test", t, Some(e)).unwrap();
    assert_eq!(b, common::fake_unsloth::fixture("job_body.json"));
    // the fake refuses a key that is not in job_fields.json, so the whole body must be accepted by it
    let d = tempdir().unwrap();
    let f = Fake::start(d.path());
    let c = UnslothClient::new(&f.url, KEY);
    let p = d.path().join("t.jsonl");
    std::fs::write(&p, "{}\n").unwrap();
    let id = c.upload_jsonl("train.jsonl", &p, &mut |_| {}).unwrap();
    let mut body = b.clone();
    body["training_file"] = serde_json::json!(id);
    body.as_object_mut().unwrap().remove("validation_file");
    assert!(c.start_job(&body).is_ok());
    // QLoRA is the only 4-bit; load_in_4bit is always present; no eval file = no eval_steps
    let q = job_body(&sample_model(), Method::Qlora, &sample_hyper(), "a b/c", t, None).unwrap();
    assert_eq!((q["load_in_4bit"].clone(), q["suffix"].clone(), q.get("validation_file").is_none(), q.get("eval_steps").is_none()), (serde_json::json!(true), serde_json::json!("a-b-c"), true, true));
    // a fixed number of steps: one epoch and max_steps
    let mut h = sample_hyper();
    h.max_steps = Some(30);
    let s = job_body(&sample_model(), Method::Lora, &h, "x", t, None).unwrap();
    assert_eq!((s["max_steps"].clone(), s["hyperparameters"]["n_epochs"].clone()), (serde_json::json!(30), serde_json::json!(1)));
}

#[test]
fn full_fine_tuning_and_two_stages_are_refused_for_unsloth() {
    use ndforge_core::train::plan::Method;
    use ndforge_core::unsloth::run::{job_body, suffix_of};
    let h = sample_hyper();
    assert!(job_body(&sample_model(), Method::Full, &h, "x", "file-1", None).unwrap_err().0.contains("Full fine-tuning runs on this computer only"));
    let mut two = sample_hyper();
    two.stages = "book_first".into();
    assert!(job_body(&sample_model(), Method::Lora, &two, "x", "file-1", None).unwrap_err().0.contains("Two stages"));
    assert_eq!(suffix_of(&"é".repeat(60)).len() <= 40, true);
    assert_eq!(suffix_of("20261113-101500-qwen3.5-2b"), "20261113-101500-qwen3-5-2b");
}

use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::project::Project;
use ndforge_core::train::plan::Method;
use ndforge_core::train::runs;
use ndforge_core::unsloth::run::{execute_unsloth, UnslothTarget};
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

fn project(root: &std::path::Path) -> (Project, ndforge_core::train::catalog::ModelEntry) {
    let mut p = Project::create(&root.join("projects"), "Ashfall Reach").unwrap();
    let f = root.join("w.ndtrain");
    ndforge_core::sample::build(&f, 1, None).unwrap();
    p.import_bundle(&ndforge_core::bundle::open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    store::save_version(&p).unwrap();
    (p, sample_model())
}

fn target(f: &Fake) -> UnslothTarget {
    UnslothTarget { name: "Unsloth Desktop".into(), url: f.url.clone(), key: KEY.into() }
}

/// (the run's result or error, every event)
fn run(p: &Project, m: &ndforge_core::train::catalog::ModelEntry, f: &Fake, name: &str, stop: Arc<AtomicBool>) -> (ndforge_core::Result<Value>, Vec<Value>) {
    let hyper = sample_hyper();
    let job = runs::build_job(p, m, Method::Lora, &hyper, 1, name, false).unwrap();
    let mut events = vec![];
    let r = execute_unsloth(p, m, Method::Lora, &hyper, name, job, &target(f), stop, false, Duration::from_millis(5), &mut |e| events.push(e));
    (r, events)
}

#[test]
fn a_run_in_unsloth_follows_the_job_into_the_charts_and_ends_done() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let f = Fake::start(&d.path().join("outputs"));
    let (r, events) = run(&p, &m, &f, "run-a", Arc::new(AtomicBool::new(false)));
    let r = r.unwrap();
    assert_eq!((r["steps"].as_u64(), r["total_steps"].as_u64(), r["final_loss"].as_f64(), r["adapter_remote"].as_bool()), (Some(4), Some(4), Some(1.0), Some(true)));
    let dir = runs::runs_dir(&p).join("run-a");
    let metrics = runs::read_metrics(&p, "run-a");
    assert_eq!(metrics.iter().filter(|e| e["type"] == "metric").map(|e| e["step"].as_u64().unwrap()).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert!(events.iter().any(|e| e["kind"] == "stage" && e["text"] == "Training in Unsloth") && events.iter().any(|e| e["kind"] == "progress" && e["step"] == "upload"));
    let state: Value = serde_json::from_slice(&std::fs::read(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!((state["status"].as_str(), state["server"].as_str(), state["backend"].as_str(), state["step"].as_u64()), (Some("done"), Some("Unsloth Desktop"), Some("unsloth"), Some(4)));
    assert!(state["note"].as_str().unwrap().contains("ashfall-test_20261113_101500"));
    // Unsloth's run folder is not on this computer in this test: the run is done and says where the adapter is
    let listed = runs::list_runs(&p);
    assert_eq!((listed.len(), listed[0]["has_adapter"].clone()), (1, json!(false)));
    // Unsloth got the rows and one job with the model and a clean suffix
    f.with(|s| {
        assert_eq!(s.files.len(), 2, "train and eval");
        assert_eq!(s.job_bodies[0]["suffix"], "run-a");
        assert_eq!(s.job_bodies[0]["load_in_4bit"], false);
        assert!(String::from_utf8_lossy(&s.files[0].2).lines().all(|l| l.contains("\"messages\"")));
    });
    // the key is nowhere in the run folder
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        if e.path().is_file() {
            assert!(!std::fs::read_to_string(e.path()).unwrap_or_default().contains("sk-unsloth-"), "key in {:?}", e.path());
        }
    }
    assert!(!serde_json::to_string(&events).unwrap().contains("sk-unsloth-"));
    assert!(std::fs::read_to_string(dir.join("config.json")).unwrap().contains("\"unsloth\""));
}

#[test]
fn a_queued_job_shows_where_it_waits_then_trains() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let f = Fake::start(&d.path().join("outputs"));
    f.with(|s| s.queue_polls = 2);
    let (r, events) = run(&p, &m, &f, "run-q", Arc::new(AtomicBool::new(false)));
    assert!(r.is_ok(), "{r:?}");
    let stages: Vec<&str> = events.iter().filter(|e| e["kind"] == "stage").filter_map(|e| e["text"].as_str()).collect();
    assert!(stages.iter().any(|t| t.contains("Waiting for Unsloth's graphics card (position 1)")), "{stages:?}");
    assert!(stages.contains(&"Training in Unsloth"));
}

#[test]
fn old_unsloth_a_refused_key_and_a_failed_job_end_the_run_with_a_plain_message() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let f = Fake::start(&d.path().join("outputs"));
    f.with(|s| s.fine_tuning = false);
    let e = run(&p, &m, &f, "run-old", Arc::new(AtomicBool::new(false))).0.unwrap_err().0;
    assert!(e.contains("no training API for other programs"), "{e}");
    f.with(|s| { s.fine_tuning = true; s.fail = Some((2, "boom".into())); });
    let e = run(&p, &m, &f, "run-fail", Arc::new(AtomicBool::new(false))).0.unwrap_err().0;
    assert_eq!(e, "boom");
    let st = |n: &str| -> Value { serde_json::from_slice(&std::fs::read(runs::runs_dir(&p).join(n).join("state.json")).unwrap()).unwrap() };
    assert_eq!((st("run-old")["status"].clone(), st("run-fail")["status"].clone(), st("run-fail")["error"].clone()), (json!("failed"), json!("failed"), json!("boom")));
}

fn write_adapter(outputs: &std::path::Path) {
    let run = outputs.join(common::fake_unsloth::RUN_FOLDER);
    std::fs::create_dir_all(run.join("checkpoint-2")).unwrap();
    std::fs::write(run.join("adapter_config.json"), "{\"r\": 16}").unwrap();
    std::fs::write(run.join("adapter_model.safetensors"), b"WEIGHTS").unwrap();
    std::fs::write(run.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(run.join("chat_template.jinja"), "{{ x }}").unwrap();
    std::fs::write(run.join("optimizer.pt"), b"NOT-COPIED").unwrap();
    std::fs::write(run.join("checkpoint-2").join("adapter_config.json"), "{}").unwrap();
}

#[test]
fn the_adapter_is_copied_back_when_unsloth_runs_on_this_computer_and_the_run_can_be_tested_and_exported() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let outputs = d.path().join("outputs");
    write_adapter(&outputs);
    let f = Fake::start(&outputs);
    let (r, _) = run(&p, &m, &f, "run-copy", Arc::new(AtomicBool::new(false)));
    let r = r.unwrap();
    assert_eq!(r["adapter_remote"], false);
    let adapter = runs::runs_dir(&p).join("run-copy").join("adapter");
    let mut names: Vec<String> = std::fs::read_dir(&adapter).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    names.sort();
    assert_eq!(names, ["adapter_config.json", "adapter_model.safetensors", "chat_template.jinja", "tokenizer.json"], "the adapter and the tokenizer, not the optimizer or the checkpoints");
    assert!(!runs::runs_dir(&p).join("run-copy").join("adapter.partial").exists());
    let listed = runs::list_runs(&p);
    assert_eq!(listed[0]["has_adapter"], true);
    assert!(ndforge_core::evaluate::runs::testable(&p).iter().any(|r| r["name"] == "run-copy"));
    assert!(ndforge_core::export::runs::exportable_runs(&p).iter().any(|r| r["name"] == "run-copy"));
}

#[test]
fn a_full_disk_or_an_old_adapter_folder_is_handled_before_anything_half_written_is_left() {
    use ndforge_core::unsloth::run::{fetch_adapter, FetchOutcome};
    let d = tempdir().unwrap();
    let outputs = d.path().join("outputs");
    write_adapter(&outputs);
    let f = Fake::start(&outputs);
    let c = UnslothClient::new(&f.url, KEY);
    let dest = d.path().join("run").join("adapter");
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    // too little space: a plain message and nothing written
    let e = fetch_adapter(&c, common::fake_unsloth::RUN_FOLDER, &dest, &|_| Some(0.0)).unwrap_err().0;
    assert!(e.contains("disk is full on this computer"), "{e}");
    assert!(!dest.exists() && !dest.with_file_name("adapter.partial").exists());
    // an older adapter is replaced whole
    std::fs::create_dir_all(&dest).unwrap();
    std::fs::write(dest.join("old.txt"), "x").unwrap();
    assert_eq!(fetch_adapter(&c, common::fake_unsloth::RUN_FOLDER, &dest, &|_| None).unwrap(), FetchOutcome::Copied);
    assert!(!dest.join("old.txt").exists() && dest.join("adapter_model.safetensors").is_file());
    // a run folder Unsloth does not list, or one that is not here
    assert!(matches!(fetch_adapter(&c, "other-run", &dest, &|_| None).unwrap(), FetchOutcome::NotReachable(_)));
    std::fs::remove_dir_all(outputs.join(common::fake_unsloth::RUN_FOLDER)).unwrap();
    assert!(matches!(fetch_adapter(&c, common::fake_unsloth::RUN_FOLDER, &dest, &|_| None).unwrap(), FetchOutcome::NotReachable(_)));
}

fn state_of(p: &Project, n: &str) -> Value {
    serde_json::from_slice(&std::fs::read(runs::runs_dir(p).join(n).join("state.json")).unwrap()).unwrap()
}

#[test]
fn stop_cancels_the_job_in_unsloth_and_the_run_cannot_be_continued() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let f = Fake::start(&d.path().join("outputs"));
    f.with(|s| s.hold = true);
    let (r, events) = run(&p, &m, &f, "run-stop", Arc::new(AtomicBool::new(true)));
    let r = r.unwrap();
    assert_eq!(r["stopped"], true);
    assert_eq!(f.count("/cancel"), 1);
    assert!(events.iter().any(|e| e["text"] == "Stopping the training in Unsloth"));
    assert_eq!(state_of(&p, "run-stop")["status"], "stopped");
    let listed = runs::list_runs(&p);
    assert_eq!((listed[0]["resumable"].clone(), listed[0]["has_adapter"].clone()), (json!(false), json!(false)));
    assert!(!runs::resumable(&p, "run-stop"));
}

#[test]
fn a_lost_connection_marks_the_run_interrupted_and_continue_reattaches_without_doubling_the_charts() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let outputs = d.path().join("outputs");
    write_adapter(&outputs);
    let f = Fake::start(&outputs);
    f.with(|s| s.break_at = Some(3));
    let (r, _) = run(&p, &m, &f, "run-int", Arc::new(AtomicBool::new(false)));
    let e = r.unwrap_err().0;
    assert!(e.contains("Lost the connection to Unsloth") && e.contains("Continue"), "{e}");
    assert_eq!(state_of(&p, "run-int")["status"], "interrupted");
    assert!(runs::resumable(&p, "run-int"));
    let seen_before = runs::read_metrics(&p, "run-int").iter().filter(|e| e["type"] == "metric").count();
    assert!(seen_before >= 1 && seen_before < 4);
    // Unsloth is back: continue
    f.with(|s| { s.broken = false; s.break_at = None; });
    let hyper = sample_hyper();
    let job = runs::build_job(&p, &m, Method::Lora, &hyper, 1, "run-int", true).unwrap();
    let mut events = vec![];
    let r = execute_unsloth(&p, &m, Method::Lora, &hyper, "run-int", job, &target(&f), Arc::new(AtomicBool::new(false)), true, Duration::from_millis(5), &mut |e| events.push(e)).unwrap();
    assert_eq!((r["steps"].as_u64(), r["adapter_remote"].clone()), (Some(4), json!(false)));
    let steps: Vec<u64> = runs::read_metrics(&p, "run-int").iter().filter(|e| e["type"] == "metric").map(|e| e["step"].as_u64().unwrap()).collect();
    assert_eq!(steps, [1, 2, 3, 4], "every step once");
    assert_eq!(f.with(|s| s.job_bodies.len()), 1, "no second job was started");
    assert_eq!(state_of(&p, "run-int")["status"], "done");
}

#[test]
fn a_waiting_job_forgotten_by_unsloth_gives_a_plain_message_and_continue_sends_it_again() {
    let d = tempdir().unwrap();
    let (p, m) = project(d.path());
    let f = Fake::start(&d.path().join("outputs"));
    f.with(|s| { s.break_at = Some(1); });
    let (r, _) = run(&p, &m, &f, "run-lost", Arc::new(AtomicBool::new(false)));
    assert!(r.is_err());
    f.with(|s| { s.broken = false; s.break_at = None; s.forget_jobs = true; });
    let hyper = sample_hyper();
    let job = runs::build_job(&p, &m, Method::Lora, &hyper, 1, "run-lost", true).unwrap();
    let e = execute_unsloth(&p, &m, Method::Lora, &hyper, "run-lost", job.clone(), &target(&f), Arc::new(AtomicBool::new(false)), true, Duration::from_millis(5), &mut |_| {}).unwrap_err().0;
    assert!(e.contains("Unsloth no longer knows this job") && e.contains("restarted"), "{e}");
    assert_eq!(state_of(&p, "run-lost")["status"], "failed");
    assert!(state_of(&p, "run-lost")["unsloth_job"].is_null());
    // Continue now sends the data again as a new job
    f.with(|s| s.forget_jobs = false);
    let r = execute_unsloth(&p, &m, Method::Lora, &hyper, "run-lost", job, &target(&f), Arc::new(AtomicBool::new(false)), true, Duration::from_millis(5), &mut |_| {});
    assert!(r.is_ok(), "{r:?}");
    assert_eq!(f.with(|s| s.job_bodies.len()), 2);
}

#[test]
fn failures_are_explained_and_no_key_reaches_a_message() {
    use ndforge_core::unsloth::run::explain_failure;
    assert!(explain_failure("CUDA out of memory. Tried to allocate 2 GiB").contains("ran out of graphics memory"));
    assert!(explain_failure("torch.OutOfMemoryError: OOM at step 3").contains("QLoRA"));
    assert!(explain_failure("OSError: [Errno 28] No space left on device").contains("disk is full"));
    assert!(!explain_failure("failed with key sk-unsloth-abc123 here").contains("abc123"));
    assert_eq!(explain_failure("boom"), "boom");
    assert!(!explain_failure("zoom room").contains("memory"));
}

#[test]
fn the_problem_report_names_unsloth_but_never_shows_a_key_even_inside_a_run_error() {
    let d = tempdir().unwrap();
    let (p, _m) = project(d.path());
    let dir = runs::runs_dir(&p).join("run-key");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("state.json"), json!({"name": "run-key", "status": "failed", "model": "M", "method": "lora", "error": "Unsloth said: bad key sk-unsloth-SECRET123 given"}).to_string()).unwrap();
    let mut settings = ndforge_core::settings::Settings::default();
    settings.unsloth = Some(ndforge_core::settings::UnslothSettings { name: "Desktop".into(), url: "http://127.0.0.1:8888".into() });
    let report = ndforge_core::diagnose::report(Some(&p), &settings);
    assert!(report.contains("unsloth: Desktop (http://127.0.0.1:8888)") && report.contains("run-key"), "{report}");
    assert!(!report.contains("SECRET123") && !report.contains("sk-unsloth-"), "{report}");
}
