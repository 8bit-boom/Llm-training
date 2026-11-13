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
