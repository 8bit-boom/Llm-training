//! The window's view of presets for a made model: the list, saving the card, and the preview of using a preset.
use ndforge_core::project::Project;
use ndforge_core::tune::api::*;
use ndforge_core::tune::store::{Scope, Store};
use serde_json::{json, Map, Value};

static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn made_model(p: &Project, name: &str, layers: u32) {
    let dir = p.folder.join("models").join(name).join("model");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = json!({"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"num_hidden_layers": layers, "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320}});
    std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    let header = json!({"w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend([1u8; 32]);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

fn card(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn a_saved_card_is_listed_as_an_exact_match_and_the_preview_shows_only_what_changes() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = Project::create(d.path(), "P").unwrap();
    made_model(&p, "cut-a", 48);
    let saved = save_card(&p, "cut-a", "My settings", Scope::Project, card(json!({"seq": 4096, "student_batch_tokens": 8192, "kl_chunk": 256})), "2026-11-22T10:00:00Z").unwrap();
    let view = presets_view(&p, "cut-a").unwrap();
    assert_eq!(view["presets"].as_array().unwrap().len(), 1);
    let first = &view["presets"][0];
    assert_eq!((first["id"].as_str().unwrap(), first["kind"].as_str().unwrap(), first["scope"].as_str().unwrap(), first["whole"].as_bool().unwrap()), (saved.id.as_str(), "exact", "project", true));
    assert_eq!(view["model"]["layers"], 48);
    let pv = apply_preview(&p, Scope::Project, &saved.id, "cut-a", "heal_card", &card(json!({"seq": 4096, "student_batch_tokens": 0, "kl_chunk": 128}))).unwrap();
    let ch: Vec<(String, Value, Value)> = pv["changes"].as_array().unwrap().iter().map(|c| (c["setting"].as_str().unwrap().to_string(), c["from"].clone(), c["to"].clone())).collect();
    assert_eq!(ch, vec![("kl_chunk".to_string(), json!(128), json!(256)), ("student_batch_tokens".to_string(), json!(0), json!(8192))], "seq is the same: not listed");
    assert_eq!(pv["changes"][1]["label"], "Tokens per forward pass");
    let again = apply_preview(&p, Scope::Project, &saved.id, "cut-a", "heal_card", &card(json!({"seq": 4096, "student_batch_tokens": 0, "kl_chunk": 128}))).unwrap();
    assert_eq!(pv["token"], again["token"], "the same preview has the same token");
    let other = apply_preview(&p, Scope::Project, &saved.id, "cut-a", "heal_card", &card(json!({"seq": 4096, "student_batch_tokens": 2048, "kl_chunk": 128}))).unwrap();
    assert_ne!(pv["token"], other["token"]);
}

#[test]
fn a_model_with_the_same_shape_gets_the_whole_preset_and_another_pc_is_refused() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = Project::create(d.path(), "P").unwrap();
    made_model(&p, "cut-a", 48);
    made_model(&p, "cut-b", 48);
    // cut-b has other weight bytes but the same shape
    let f = p.folder.join("models").join("cut-b").join("model").join("model.safetensors");
    let header = json!({"w": {"dtype": "BF16", "shape": [2, 8], "data_offsets": [0, 32]}}).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend([3u8; 32]);
    std::fs::write(f, bytes).unwrap();
    let saved = save_card(&p, "cut-a", "A", Scope::Project, card(json!({"student_batch_tokens": 4096})), "2026-11-22T10:00:00Z").unwrap();
    let v = presets_view(&p, "cut-b").unwrap();
    assert_eq!(v["presets"][0]["kind"], "same_shape");
    assert!(apply_preview(&p, Scope::Project, &saved.id, "cut-b", "heal_card", &card(json!({}))).unwrap()["whole"].as_bool().unwrap());
    // a preset of another card is listed and never applied
    let mut elsewhere = Store::new(Some(&p.folder)).get(Scope::Project, &saved.id).unwrap();
    elsewhere.id = "t-elsewhere".into();
    elsewhere.machine.key = "m-other".into();
    elsewhere.machine.gpu = "NVIDIA GeForce RTX 4090".into();
    Store::new(Some(&p.folder)).save(Scope::Project, &elsewhere).unwrap();
    assert!(presets_view(&p, "cut-a").unwrap()["presets"].as_array().unwrap().iter().any(|x| x["kind"] == "other_pc"));
    assert!(apply_preview(&p, Scope::Project, "t-elsewhere", "cut-a", "heal_card", &card(json!({}))).unwrap_err().0.contains("another PC"));
    assert!(presets_view(&p, "no-such-model").is_err() && presets_view(&p, "../x").is_err());
}

#[test]
fn a_settings_target_needs_the_token_of_a_fresh_preview_and_changes_only_its_own_knobs() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = Project::create(d.path(), "P").unwrap();
    made_model(&p, "cut-a", 48);
    let saved = save_card(&p, "cut-a", "A", Scope::Project, card(json!({"seq": 4096, "teacher_batch_tokens": 32768, "max_num_seqs": 48, "vllm_batch_tokens": 65536, "gpu_util_pct": 92, "student_batch_tokens": 8192})), "2026-11-22T10:00:00Z").unwrap();
    let pv = apply_preview(&p, Scope::Project, &saved.id, "cut-a", "settings_teacher", &Map::new()).unwrap();
    let keys: Vec<&str> = pv["changes"].as_array().unwrap().iter().map(|c| c["setting"].as_str().unwrap()).collect();
    assert_eq!(keys, vec!["teacher_batch_tokens"], "the teacher target takes only its own knob");
    assert!(apply_settings(&p, Scope::Project, &saved.id, "cut-a", "settings_teacher", "").is_err());
    assert!(apply_settings(&p, Scope::Project, &saved.id, "cut-a", "settings_teacher", "a0000000000000000").unwrap_err().0.contains("Nothing was changed"));
    assert_eq!(ndforge_core::settings::Settings::load().teacher_batch_tokens, 16384);
    apply_settings(&p, Scope::Project, &saved.id, "cut-a", "settings_teacher", pv["token"].as_str().unwrap()).unwrap();
    let s = ndforge_core::settings::Settings::load();
    assert_eq!((s.teacher_batch_tokens, s.vllm.teacher.max_num_seqs), (32768, 0), "only the teacher batch changed");
    let pv2 = apply_preview(&p, Scope::Project, &saved.id, "cut-a", "settings_vllm", &Map::new()).unwrap();
    apply_settings(&p, Scope::Project, &saved.id, "cut-a", "settings_vllm", pv2["token"].as_str().unwrap()).unwrap();
    let s = ndforge_core::settings::Settings::load();
    assert_eq!((s.vllm.teacher.max_num_seqs, s.vllm.teacher.batch_tokens, s.vllm.teacher.gpu_util_pct), (48, 65536, 92));
    // the same preview now shows nothing to change, and a page target is not applied here
    assert!(apply_preview(&p, Scope::Project, &saved.id, "cut-a", "settings_vllm", &Map::new()).unwrap()["changes"].as_array().unwrap().is_empty());
    assert!(apply_settings(&p, Scope::Project, &saved.id, "cut-a", "heal_card", "x").is_err());
    // the engine job carries the tested numbers only when they are set (a legacy job is byte-identical)
    let job = ndforge_core::vllm::teacher_engine_job(&s.vllm, None);
    assert_eq!((job["max_num_seqs"].clone(), job["batch_tokens"].clone(), job["gpu_memory_utilization"].clone()), (json!(48), json!(65536), json!(0.92)));
    let plain = ndforge_core::vllm::teacher_engine_job(&ndforge_core::settings::VllmSettings::default(), None);
    assert!(plain.get("max_num_seqs").is_none() && plain.get("batch_tokens").is_none() && plain.get("gpu_memory_utilization").is_none());
}
