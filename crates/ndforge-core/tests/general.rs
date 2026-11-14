//! The general-text chooser: the presets, the saved list of the person's own datasets (a file in the settings folder), the acknowledgement, the choice per
//! project, the Hugging Face token, and the key that keeps one dataset's cache from being used for another. No network.
use std::sync::Mutex;

use ndforge_core::envmgr::jobs::job_env;
use ndforge_core::general::*;
use ndforge_core::project::Project;
use serde_json::json;
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn setup() -> (std::sync::MutexGuard<'static, ()>, tempfile::TempDir) {
    let g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    (g, d)
}

fn file() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("NDFORGE_CONFIG_DIR").unwrap()).join("general_datasets.json")
}

#[test]
fn the_preset_table_has_what_the_chooser_shows() {
    assert_eq!(PRESETS[0].id, DEFAULT_ID);
    assert_eq!(DEFAULT_ID, "mlabonne/open-perfectblend");
    let v = presets_json();
    let keys: Vec<&str> = v.iter().map(|p| p["key"].as_str().unwrap()).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), keys.len(), "keys are unique");
    for p in &v {
        assert!(!p["license"].as_str().unwrap().is_empty() && !p["name"].as_str().unwrap().is_empty() && !p["note"].as_str().unwrap().is_empty(), "{p}");
        assert!(["chat", "text"].contains(&p["kind"].as_str().unwrap()));
    }
    let by = |id: &str| v.iter().find(|p| p["id"] == id).unwrap_or_else(|| panic!("{id}"));
    assert_eq!((by("HuggingFaceH4/ultrachat_200k")["split"].clone(), by("HuggingFaceH4/ultrachat_200k")["license"].clone()), (json!("train_sft"), json!("MIT")));
    assert_eq!((by("HuggingFaceFW/fineweb-edu")["config"].clone(), by("HuggingFaceFW/fineweb-edu")["kind"].clone()), (json!("sample-10BT"), json!("text")));
    assert!(by("wikimedia/wikipedia")["license_note"].as_str().unwrap().contains("ShareAlike"));
    assert_eq!(by("nvidia/Nemotron-Pretraining-SFT-v1")["gated"], true);
    assert_eq!(by("faunix/Qwen3.8-27B-Distillation-40K")["license"], "Apache-2.0");
    assert!(by("faunix/Qwen3.8-27B-Distillation-40K")["note"].as_str().unwrap().contains("4096") && by("faunix/Qwen3.8-27B-Distillation-40K")["note"].as_str().unwrap().contains("weak fit"));
    let multi = by("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation");
    assert_eq!((multi["needs_ack"].clone(), multi["config"].clone(), multi["key"].clone()), (json!(true), json!("sft_balanced"), json!("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation#sft_balanced")));
    assert!(multi["license_note"].as_str().unwrap().contains("Noncommercial research use") && multi["license_note"].as_str().unwrap().contains("LICENSE"));
    assert_eq!(v.iter().filter(|p| p["needs_ack"] == true).count(), 1);
    assert!(v.iter().filter(|p| p["gated"] == true).count() == 1 && v.iter().filter(|p| p["gated"] == false).all(|p| p["id"] != "nvidia/Nemotron-Pretraining-SFT-v1"));
}

#[test]
fn the_list_of_my_own_datasets_is_saved_loaded_and_removed() {
    let (_g, d) = setup();
    assert!(custom_list().is_empty(), "no file yet");
    let k = add_custom(&GeneralSource { id: Some(" https://huggingface.co/datasets/acme/stories/ ".into()), split: Some("train".into()), ..Default::default() }).unwrap();
    assert_eq!(k, "acme/stories");
    let f = d.path().join("notes.jsonl");
    std::fs::write(&f, "{\"text\": \"x\"}\n").unwrap();
    let kf = add_custom(&GeneralSource { file: Some(f.to_string_lossy().to_string()), ..Default::default() }).unwrap();
    let list = custom_list();
    assert_eq!(list.len(), 2);
    assert_eq!((list[0].id.as_deref(), list[0].name.as_deref(), list[1].name.as_deref()), (Some("acme/stories"), Some("acme/stories"), Some("notes.jsonl")));
    add_custom(&GeneralSource::hub("acme/stories")).unwrap();
    assert_eq!(custom_list().len(), 2, "the same dataset is not listed twice");
    // a preset is not copied into the list
    assert_eq!(add_custom(&GeneralSource::hub("mlabonne/open-perfectblend")).unwrap(), DEFAULT_ID);
    assert_eq!(custom_list().len(), 2);
    remove_custom(&k).unwrap();
    assert_eq!(custom_list().iter().map(|s| s.key()).collect::<Vec<_>>(), [kf.clone()]);
    assert!(remove_custom("nope").unwrap_err().0.contains("added yourself"));
    assert!(remove_custom(DEFAULT_ID).is_err());
    // what the page gets
    let st = state(None);
    assert_eq!(st["custom"].as_array().unwrap().len(), 1);
    assert!(st["presets"].as_array().unwrap().len() >= 8 && st["default"] == DEFAULT_ID);
}

#[test]
fn bad_entries_are_refused_in_plain_words() {
    let (_g, d) = setup();
    for bad in ["justaname", "a/b/c", "has space/x", "", "https://example.com/x/y"] {
        let e = add_custom(&GeneralSource::hub(bad)).unwrap_err().0;
        assert!(e.contains("owner/name") || e.contains("Give a Hugging Face"), "{bad}: {e}");
    }
    assert!(add_custom(&GeneralSource { file: Some(d.path().join("gone.txt").to_string_lossy().to_string()), ..Default::default() }).unwrap_err().0.contains("was not found"));
    assert!(add_custom(&GeneralSource { id: Some("a/b".into()), file: Some("x".into()), ..Default::default() }).unwrap_err().0.contains("either"));
    assert!(custom_list().is_empty());
}

#[test]
fn a_missing_or_damaged_file_counts_as_an_empty_list_and_does_not_stop_adding() {
    let (_g, _d) = setup();
    assert!(custom_list().is_empty() && acknowledged().is_empty());
    std::fs::create_dir_all(file().parent().unwrap()).unwrap();
    for junk in ["{ not json", "[]", "null", "{\"datasets\": 5}", ""] {
        std::fs::write(file(), junk).unwrap();
        assert!(custom_list().is_empty(), "{junk}");
        assert!(state(None)["custom"].as_array().unwrap().is_empty());
    }
    add_custom(&GeneralSource::hub("acme/after")).unwrap();
    assert_eq!(custom_list().len(), 1, "a damaged file is replaced by the next save");
    // a file written by a later version with more fields still loads
    std::fs::write(file(), r#"{"datasets": [{"id": "a/b", "future": 1}], "acknowledged": ["x"], "more": true}"#).unwrap();
    assert_eq!((custom_list().len(), acknowledged()), (1, vec!["x".to_string()]));
}

#[test]
fn resolve_fills_the_gaps_from_the_preset_and_gates_the_acknowledgement() {
    let (_g, _d) = setup();
    let r = resolve(None, true).unwrap();
    assert_eq!((r.id.as_deref(), r.split.as_deref(), r.config), (Some(DEFAULT_ID), Some("train"), None));
    let u = resolve(Some(&GeneralSource::hub("HuggingFaceH4/ultrachat_200k")), true).unwrap();
    assert_eq!(u.split.as_deref(), Some("train_sft"));
    let w = resolve(Some(&GeneralSource::hub("wikimedia/wikipedia")), true).unwrap();
    assert_eq!(w.config.as_deref(), Some("20231101.en"));
    let own = resolve(Some(&GeneralSource { id: Some("acme/x".into()), config: Some("c".into()), split: Some("val".into()), column: Some("body".into()), ..Default::default() }), true).unwrap();
    assert_eq!((own.config.as_deref(), own.split.as_deref(), own.column.as_deref()), (Some("c"), Some("val"), Some("body")));
    assert_eq!(resolve(Some(&GeneralSource::hub("acme/x")), true).unwrap().split, None, "an unknown dataset keeps the worker's default split");
    // the acknowledgement
    let multi = GeneralSource::hub("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation");
    let e = resolve(Some(&multi), true).unwrap_err().0;
    assert!(e.contains("needs your OK") && e.contains("Noncommercial research use"), "{e}");
    assert!(resolve(Some(&multi), false).is_ok(), "checking a dataset does not need it");
    acknowledge("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation#sft_balanced").unwrap();
    assert_eq!(resolve(Some(&multi), true).unwrap().config.as_deref(), Some("sft_balanced"));
    assert_eq!(acknowledged().len(), 1);
    acknowledge("r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation#sft_balanced").unwrap();
    assert_eq!(acknowledged().len(), 1, "once per dataset");
    assert!(acknowledge("acme/x").unwrap_err().0.contains("not in the list"));
    acknowledge(DEFAULT_ID).unwrap();
    assert_eq!(acknowledged().len(), 1, "a dataset that needs none is not recorded");
    assert!(resolve(Some(&GeneralSource { id: Some("a/b".into()), file: Some("f".into()), ..Default::default() }), true).is_err());
}

#[test]
fn the_job_block_and_the_key_follow_every_part_of_the_dataset() {
    let src = GeneralSource { id: Some("a/b".into()), config: Some("c".into()), split: Some("train".into()), ..Default::default() };
    let g = job_block(&src, 0.5);
    assert_eq!(g, json!({"share": 0.5, "id": "a/b", "config": "c", "split": "train"}));
    let k = key(&g, 3407);
    assert_eq!(k, json!({"used": true, "id": "a/b", "file": null, "config": "c", "split": "train", "column": null, "share": 0.5, "seed": 3407}), "the same literal as the worker's general_key");
    let other = |f: fn(&mut GeneralSource)| { let mut s = src.clone(); f(&mut s); key(&job_block(&s, 0.5), 3407) };
    assert_ne!(other(|s| s.id = Some("a/c".into())), k);
    assert_ne!(other(|s| s.config = Some("d".into())), k);
    assert_ne!(other(|s| s.split = Some("dev".into())), k);
    assert_ne!(other(|s| s.column = Some("text".into())), k);
    assert_ne!(key(&job_block(&src, 0.4), 3407), k);
    assert_ne!(key(&g, 1), k);
    assert_eq!(key(&serde_json::Value::Null, 3407), json!({"used": false}));
    assert_eq!(key(&job_block(&src, 0.0), 3407), json!({"used": false}));
    let f = key(&job_block(&GeneralSource { file: Some("x.jsonl".into()), ..Default::default() }, 0.2), 1);
    assert_eq!((f["id"].clone(), f["file"].clone()), (json!(null), json!("x.jsonl")));
}

#[test]
fn the_last_choice_is_remembered_per_project() {
    let (_g, d) = setup();
    let a = Project::create(d.path(), "A").unwrap();
    let b = Project::create(d.path(), "B").unwrap();
    assert_eq!(choices(&a), json!({}));
    set_choice(&a, "heal", &GeneralSource::hub("acme/x")).unwrap();
    set_choice(&a, "imatrix", &GeneralSource { file: Some("n.txt".into()), ..Default::default() }).unwrap();
    assert_eq!(choices(&a)["heal"]["id"], "acme/x");
    assert_eq!(choices(&a)["imatrix"]["file"], "n.txt");
    assert_eq!(choices(&b), json!({}));
    assert_eq!(state(Some(&a))["choice"]["heal"]["id"], "acme/x");
    assert!(set_choice(&a, "other", &GeneralSource::default()).is_err());
    std::fs::write(a.folder.join("general_choice.json"), "garbage").unwrap();
    assert_eq!(choices(&a), json!({}), "a damaged file is no choice");
}

#[test]
fn the_token_is_kept_with_the_secrets_checked_and_never_in_the_state() {
    let (_g, _d) = setup();
    assert!(!token_present() && state(None)["has_token"] == false);
    for bad in ["", "abc", "hf_", "sk-whatever-1234567890", "hf_ with space 12345678"] {
        assert!(token_set(bad).unwrap_err().0.contains("hf_"), "{bad}");
    }
    let t = format!("hf_{}", "Xy1".repeat(11));
    token_set(&format!("  {t}\n")).unwrap();
    assert!(token_present() && state(None)["has_token"] == true);
    assert!(!state(None).to_string().contains(&t));
    assert_eq!(ndforge_core::secrets::get(TOKEN_SECRET).unwrap(), t);
    token_clear().unwrap();
    assert!(!token_present());
}

#[test]
fn the_token_goes_to_the_jobs_that_read_general_text_and_to_no_other() {
    let (_g, _d) = setup();
    let t = format!("hf_{}", "Mn5".repeat(11));
    token_set(&t).unwrap();
    let env = |j: serde_json::Value| job_env(&j);
    let want = vec![("HF_TOKEN".to_string(), t.clone())];
    assert_eq!(env(json!({"job": "general_check", "source": {}})), want);
    assert_eq!(env(json!({"job": "distill_cache", "data": {"general": {"share": 0.5}}})), want);
    assert_eq!(env(json!({"job": "distill", "dataset": {"general": {"share": 0.5}}})), want);
    assert_eq!(env(json!({"job": "export", "imatrix": {"enabled": true}})), want);
    assert!(env(json!({"job": "distill_cache", "data": {"general": null}})).is_empty());
    assert!(env(json!({"job": "export"})).is_empty() && env(json!({"job": "train"})).is_empty() && env(json!({"job": "distill_merge"})).is_empty());
    // a job file never holds it: the blocks are made from the source only
    assert!(!job_block(&GeneralSource::hub("a/b"), 0.5).to_string().contains(&t));
}

#[test]
fn tokens_are_scrubbed_from_texts_and_problem_reports() {
    use ndforge_core::unsloth::client::scrub;
    let t = format!("hf_{}", "Ab3".repeat(12));
    assert_eq!(scrub(&format!("401 for token {t}, retry")), "401 for token [token], retry");
    assert_eq!(scrub(&format!("{t} {t}")), "[token] [token]");
    assert_eq!(scrub("hf_short and hf_ and half_value"), "hf_short and hf_ and half_value");
    assert_eq!(scrub("sk-unsloth-abc123 and hf_"), "[key] and hf_");
    assert!(!ndforge_core::diagnose::report(None, &ndforge_core::settings::Settings::default()).contains("hf_"));
}
