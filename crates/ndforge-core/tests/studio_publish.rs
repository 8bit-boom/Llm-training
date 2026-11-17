//! Putting an export into Unsloth Studio's share: the copy (`.partial`, then renamed), the free-space check, the container path from Studio's own outputs folder, the
//! registration of the exports folder (feature-probed) and the fine-tune handoff, against the stand-in Studio (the same fixtures nd-world uses).
mod common;

use std::path::Path;
use std::sync::atomic::AtomicBool;

use common::fake_unsloth::{Fake, KEY, RUN_FOLDER};
use ndforge_core::settings::StudioShare;
use ndforge_core::unsloth::client::UnslothClient;
use ndforge_core::unsloth::publish::*;
use serde_json::{json, Value};
use tempfile::tempdir;

/// Studio's root inside its container, as the stand-in's outputs folder says
const OUTPUTS: &str = "/workspace/outputs";

fn share_dir(root: &Path) -> std::path::PathBuf {
    let share = root.join("studio");
    std::fs::create_dir_all(share.join("outputs").join(RUN_FOLDER)).unwrap();
    share
}

fn gguf(root: &Path, name: &str, bytes: usize) -> std::path::PathBuf {
    let dir = root.join("export");
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join(name);
    std::fs::write(&f, vec![7u8; bytes]).unwrap();
    f
}

fn spec(root: &Path) -> PublishSpec {
    PublishSpec { name: "Qwen3.8 27B cut to 21.1B".into(), files: vec![gguf(root, "model-q4_k_m.gguf", 3000), gguf(root, "model-q8_0.gguf", 5000)], hf_folder: None, readme: Some("hello".into()) }
}

fn share_of(path: &Path) -> StudioShare {
    StudioShare { host_path: path.to_string_lossy().to_string(), ..StudioShare::default() }
}

fn run(share: &StudioShare, spec: &PublishSpec, client: Option<&UnslothClient>, free: Option<f64>) -> ndforge_core::Result<Value> {
    let stop = AtomicBool::new(false);
    publish(share, spec, &Ctx { client, free_gb: free, stop: &stop }, &mut |_| {})
}

#[test]
fn the_container_path_comes_from_studios_outputs_folder_and_is_confirmed_by_the_run_folders() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let f = Fake::start(Path::new(OUTPUTS));
    let c = UnslothClient::new(&f.url, KEY);
    let det = detect(&c, &share).unwrap();
    assert_eq!(det.outputs_dir.as_deref(), Some(OUTPUTS));
    assert_eq!(det.container_root.as_deref(), Some("/workspace"));
    assert_eq!((det.confirmed, det.matched.clone()), (Some(true), vec![RUN_FOLDER.to_string()]));
    // a share that is not Studio's: its run folders are not there
    let other = d.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    assert_eq!(detect(&c, &other).unwrap().confirmed, Some(false));
    // Studio lists no runs: nothing to compare
    f.with(|s| s.list_runs = false);
    assert_eq!(detect(&c, &share).unwrap().confirmed, None);
    // paths of a container, however they are spelled
    assert_eq!(parent_of("/workspace/outputs/"), Some("/workspace".into()));
    assert_eq!(parent_of("/outputs"), Some("/".into()));
    assert_eq!(parent_of("outputs"), None);
    assert_eq!(join_container("/workspace/", &["exports", "m"]), "/workspace/exports/m");
    assert_eq!(join_container("C:\\studio", &["exports"]), "C:\\studio\\exports");
}

#[test]
fn files_are_copied_into_a_partial_folder_that_is_renamed_and_the_container_path_is_given() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let f = Fake::start(Path::new(OUTPUTS));
    let c = UnslothClient::new(&f.url, KEY);
    let r = run(&share_of(&share), &spec(d.path()), Some(&c), Some(500.0)).unwrap();
    assert_eq!(r["published"], true, "{r}");
    let dest = share.join("exports").join("Qwen3.8-27B-cut-to-21.1B");
    assert_eq!(Path::new(r["host_path"].as_str().unwrap()), dest);
    assert_eq!(r["container_path"], "/workspace/exports/Qwen3.8-27B-cut-to-21.1B");
    assert_eq!(r["files"][0]["container_path"], "/workspace/exports/Qwen3.8-27B-cut-to-21.1B/model-q4_k_m.gguf");
    assert_eq!(std::fs::metadata(dest.join("model-q4_k_m.gguf")).unwrap().len(), 3000);
    assert_eq!(std::fs::metadata(dest.join("model-q8_0.gguf")).unwrap().len(), 5000);
    assert_eq!(std::fs::read_to_string(dest.join("README.md")).unwrap(), "hello");
    let names: Vec<String> = std::fs::read_dir(share.join("exports")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    assert_eq!(names, vec!["Qwen3.8-27B-cut-to-21.1B"], "no .partial or .old is left: {names:?}");
    // the exports folder was registered once, as Studio sees it, recursive
    assert_eq!(r["scan_folder"]["status"], "registered");
    assert_eq!(f.with(|s| s.scan_posts.clone()), vec![json!({"path": "/workspace/exports", "recursive": true})]);
    // a second publish replaces the first one whole
    let second = PublishSpec { files: vec![d.path().join("export").join("model-q4_k_m.gguf")], ..spec(d.path()) };
    std::fs::write(&second.files[0], vec![1u8; 10]).unwrap();
    let again = run(&share_of(&share), &second, Some(&c), Some(500.0)).unwrap();
    assert_eq!(again["published"], true);
    assert_eq!(std::fs::metadata(dest.join("model-q4_k_m.gguf")).unwrap().len(), 10);
    assert!(!dest.join("model-q8_0.gguf").exists(), "the older files are gone with the older folder");
}

#[test]
fn a_manual_container_path_wins_and_a_missing_key_still_publishes_without_telling_studio() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let mut s = share_of(&share);
    s.container_path = "/data/studio".into();
    let r = run(&s, &spec(d.path()), None, None).unwrap();
    assert_eq!(r["container_path"], "/data/studio/exports/Qwen3.8-27B-cut-to-21.1B");
    assert_eq!(r["scan_folder"]["status"], "skipped");
    assert!(r["scan_folder"]["note"].as_str().unwrap().contains("API key"));
    // no container path anywhere: the files are there, the path inside Studio is not known
    let mut bare = share_of(&share);
    bare.register_scan_folder = true;
    let r = run(&bare, &spec(d.path()), None, None).unwrap();
    assert_eq!((r["published"].clone(), r["container_path"].clone()), (json!(true), Value::Null));
    assert!(r["scan_folder"]["note"].as_str().unwrap().contains("API key"));
}

#[test]
fn the_exports_folder_is_registered_only_when_studio_offers_it_and_a_rejection_is_a_note() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let f = Fake::start(Path::new(OUTPUTS));
    let c = UnslothClient::new(&f.url, KEY);
    f.with(|s| s.scan_folders = false);
    let r = run(&share_of(&share), &spec(d.path()), Some(&c), None).unwrap();
    assert_eq!(r["published"], true);
    assert_eq!(r["scan_folder"]["status"], "not_offered");
    assert!(f.with(|s| s.scan_posts.is_empty()) && f.count("POST /api/models/scan-folders") == 0, "the probe said no: nothing was posted");
    f.with(|s| {
        s.scan_folders = true;
        s.scan_reject = Some("Directory does not exist".into());
    });
    let r = run(&share_of(&share), &spec(d.path()), Some(&c), None).unwrap();
    assert_eq!(r["published"], true, "the files are in the share whatever Studio says");
    assert_eq!(r["scan_folder"]["status"], "rejected");
    assert!(r["scan_folder"]["note"].as_str().unwrap().contains("Directory does not exist") && r["scan_folder"]["note"].as_str().unwrap().contains("Path inside the container"));
    // switched off in the settings: Studio is not asked at all
    let before = f.count("scan-folders");
    let mut off = share_of(&share);
    off.register_scan_folder = false;
    assert_eq!(run(&off, &spec(d.path()), Some(&c), None).unwrap()["scan_folder"]["status"], "skipped");
    assert_eq!(f.count("scan-folders"), before);
}

#[test]
fn a_share_that_is_not_there_or_not_studios_or_too_full_is_said_in_plain_words_and_nothing_is_left_behind() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    // nothing set up / the drive is gone
    assert!(run(&StudioShare::default(), &spec(d.path()), None, None).unwrap()["why"].as_str().unwrap().contains("not set up"));
    let gone = share_of(&d.path().join("no-such-drive"));
    assert!(run(&gone, &spec(d.path()), None, None).unwrap()["why"].as_str().unwrap().contains("Is the drive connected"));
    // not Studio's share: Studio lists a run folder that is not under it
    let f = Fake::start(Path::new(OUTPUTS));
    let c = UnslothClient::new(&f.url, KEY);
    let other = d.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    let r = run(&share_of(&other), &spec(d.path()), Some(&c), None).unwrap();
    assert_eq!(r["published"], false);
    assert!(r["why"].as_str().unwrap().contains("Nothing was copied"), "{r}");
    assert!(!other.join("exports").exists());
    // too little room: the free GB are in the message, no folder is made
    let e = run(&share_of(&share), &PublishSpec { files: vec![gguf(d.path(), "big.gguf", 4000)], ..spec(d.path()) }, None, Some(0.1)).unwrap_err();
    assert!(e.0.contains("0.1 GB free") && e.0.contains("Free some space"), "{}", e.0);
    assert!(!share.join("exports").exists());
}

#[test]
fn a_stopped_or_failed_copy_leaves_no_partial_folder_and_keeps_the_older_export() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let s = share_of(&share);
    assert_eq!(run(&s, &spec(d.path()), None, None).unwrap()["published"], true);
    let dest = share.join("exports").join("Qwen3.8-27B-cut-to-21.1B");
    // stopped before the first chunk
    let stop = AtomicBool::new(true);
    let e = publish(&s, &spec(d.path()), &Ctx { client: None, free_gb: None, stop: &stop }, &mut |_| {}).unwrap_err();
    assert_eq!(e.0, "Stopped.");
    // a file that vanished
    let gone = PublishSpec { files: vec![d.path().join("export").join("nope.gguf")], ..spec(d.path()) };
    assert!(run(&s, &gone, None, None).unwrap_err().0.contains("not there any more"));
    let names: Vec<String> = std::fs::read_dir(share.join("exports")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
    assert_eq!(names, vec!["Qwen3.8-27B-cut-to-21.1B"]);
    assert_eq!(std::fs::metadata(dest.join("model-q8_0.gguf")).unwrap().len(), 5000, "the first export is whole");
}

#[test]
fn the_finetune_handoff_copies_the_healed_folder_only_when_it_is_on() {
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let hf = d.path().join("healed");
    std::fs::create_dir_all(hf.join("sub")).unwrap();
    std::fs::write(hf.join("config.json"), "{}").unwrap();
    std::fs::write(hf.join("sub").join("model-00001.safetensors"), vec![3u8; 2000]).unwrap();
    let mut sp = spec(d.path());
    sp.hf_folder = Some(hf.clone());
    let mut s = share_of(&share);
    s.container_path = "/workspace".into();
    // off (the default): not copied, no path given
    let r = run(&s, &sp, None, None).unwrap();
    assert!(r["hf"].is_null() && r["finetune_model"].is_null());
    assert!(!share.join("exports").join("Qwen3.8-27B-cut-to-21.1B-hf").exists());
    // on
    s.finetune_handoff = true;
    let r = run(&s, &sp, None, None).unwrap();
    assert_eq!(r["finetune_model"], "/workspace/exports/Qwen3.8-27B-cut-to-21.1B-hf");
    assert_eq!(std::fs::metadata(share.join("exports").join("Qwen3.8-27B-cut-to-21.1B-hf").join("sub").join("model-00001.safetensors")).unwrap().len(), 2000);
    assert!(share.join("exports").join("Qwen3.8-27B-cut-to-21.1B").join("model-q4_k_m.gguf").is_file());
    assert_eq!(StudioShare::default().finetune_handoff, false);
}

#[test]
fn a_name_is_made_safe_and_the_key_never_reaches_a_result() {
    assert_eq!(safe_name("../../etc/passwd"), "etc-passwd");
    assert_eq!(safe_name("a b/c\\d"), "a-b-c-d");
    assert_eq!(safe_name("***"), "model");
    let d = tempdir().unwrap();
    let share = share_dir(d.path());
    let f = Fake::start(Path::new(OUTPUTS));
    let c = UnslothClient::new(&f.url, KEY);
    let r = run(&share_of(&share), &spec(d.path()), Some(&c), None).unwrap();
    assert!(!r.to_string().contains("sk-unsloth"), "{r}");
    assert!(f.with(|s| s.authorization.iter().any(|a| a == &format!("Bearer {KEY}"))));
}
