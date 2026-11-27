//! Version checks for Inkling-2-Mini: which installed tool is too old, and whether the model's repository changed.
use ndforge_core::reshape::moe_compat::*;
use serde_json::json;

fn have(t: Option<&str>, l: Option<&str>, v: Option<&str>) -> Have {
    Have { transformers: t.map(String::from), llama_tag: l.map(String::from), vllm: v.map(String::from) }
}

#[test]
fn versions_compare_like_people_expect() {
    assert!(version_at_least("5.24.0", "5.24.0") && version_at_least("5.24.1", "5.24.0") && version_at_least("6.0", "5.24.0") && version_at_least("5.25.0rc1", "5.24.0"));
    assert!(!version_at_least("5.17.0", "5.24.0") && !version_at_least("5.9", "5.24.0") && !version_at_least("5.24.0.dev0", "5.24.0"));
    assert_eq!(parse_version("v5.24.0.post1"), vec![5, 24, 0]);
    assert_eq!((llama_build("b12611"), llama_build("b12649"), llama_build("master")), (Some(12611), Some(12649), None));
}

#[test]
fn vllm_settings_are_read_as_image_tags_or_versions() {
    for (s, want) in [("vllm/vllm-openai:nightly", Some(true)), ("vllm/vllm-openai:v0.35.0", Some(true)), ("0.36.2", Some(true)), ("vllm/vllm-openai:v0.34.0", Some(false)), ("0.34.9", Some(false)), ("", None), ("custom-image", None)] {
        assert_eq!(vllm_loads_inkling(s), want, "{s}");
    }
}

#[test]
fn the_owners_environment_is_told_what_to_upgrade_and_other_families_are_left_alone() {
    // the owner's PC at the time of writing: transformers 5.17.0, the app's pinned vLLM 0.34, a recent llama.cpp
    let f = findings("inkling2", &have(Some("5.17.0"), Some("b12649"), Some("vllm/vllm-openai:v0.34.0")));
    let by = |t: &str| f.iter().find(|x| x.tool == t).unwrap();
    assert_eq!((by("transformers").status.as_str(), by("llama.cpp").status.as_str(), by("vLLM").status.as_str()), ("too_old", "ok", "too_old"));
    assert!(by("transformers").message.contains("5.17.0") && by("transformers").message.contains("5.24.0"));
    assert_eq!(by("transformers").command.as_deref(), Some("uv pip install \"transformers>=5.24.0,<6\""));
    assert_eq!(by("transformers").rollback.as_deref(), Some("uv pip install \"transformers==5.17.0\""), "the old version is the way back");
    assert!(by("vLLM").fix.contains("nightly") && by("llama.cpp").command.is_none() && by("llama.cpp").fix.is_empty());
    assert!(summary(&f).unwrap().contains("transformers, vLLM"));
    // old llama.cpp builds are refused with the reason that matters
    let old = findings("inkling2", &have(Some("5.24.1"), Some("b12610"), Some("nightly")));
    assert_eq!(old.iter().find(|x| x.tool == "llama.cpp").unwrap().status, "too_old");
    assert!(old.iter().find(|x| x.tool == "llama.cpp").unwrap().message.contains("130"));
    assert!(old.iter().filter(|x| x.tool != "llama.cpp").all(|x| x.status == "ok"));
    // nothing known: "unknown", never a false alarm
    assert!(findings("inkling2", &Have::default()).iter().all(|x| x.status == "unknown") && summary(&findings("inkling2", &Have::default())).is_none());
    // the Qwen and Gemma families have no findings at all
    for fam in ["qwen3_5", "gemma4", "generic"] {
        assert!(findings(fam, &have(Some("5.17.0"), None, None)).is_empty());
    }
}

#[test]
fn a_changed_repository_is_reported_and_an_unchanged_one_is_not() {
    let same = repo_state(&json!({"sha": TESTED_SHA, "lastModified": "2026-11-24T18:23:36.000Z", "siblings": [{"rfilename": "config.json"}, {"rfilename": "model-00001-of-00001.safetensors"}, {"rfilename": "README.md"}]}));
    assert!(!same.changed && same.message.contains("unchanged") && same.watched_files == vec!["config.json", "model-00001-of-00001.safetensors"]);
    let newer = repo_state(&json!({"sha": "ffffffffffffffffffffffffffffffffffffffff", "siblings": []}));
    assert!(newer.changed && newer.message.contains("new changes") && newer.message.contains("c9d0283"));
    assert!(!repo_state(&json!({})).changed);
    let r = report("inkling2", &Have::default(), Some(&newer));
    assert_eq!(r["findings"].as_array().unwrap().len(), 3);
    assert_eq!(r["repo"]["changed"], true);
}

mod reco {
    use ndforge_core::reshape::moe_compat::Have;
    use ndforge_core::reshape::moe_plan::{recommend, RecoInput};
    use ndforge_core::reshape::plan::Inspect;

    fn ink() -> Inspect {
        serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/inkling2_inspect.json")).unwrap()).unwrap()
    }

    fn input<'a>(ins: &'a Inspect, goal: &'a str, have: &'a Have) -> RecoInput<'a> {
        RecoInput { ins, vram_gb: Some(32.0), free_gb: Some(500.0), goal, target_gb: None, have }
    }

    #[test]
    fn the_order_for_a_smaller_file_on_a_32_gb_card() {
        let ins = ink();
        let have = Have { transformers: Some("5.24.1".into()), llama_tag: Some("b12649".into()), vllm: Some("vllm/vllm-openai:nightly".into()) };
        let r = recommend(&input(&ins, "size", &have));
        assert!(r.blocked.is_none());
        assert_eq!(r.order.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), ["goal", "measure", "try", "cut", "check", "heal", "export"]);
        assert!(r.order[3].title.contains("128 to 96") && r.order[5].title.contains("only if needed"));
        assert_eq!(r.first_try[0], "base");
        assert_eq!(&r.first_try[1..5], ["e=112", "e=96", "e=80", "e=64"]);
        assert!(!r.first_try.iter().any(|k| k.contains("p=")), "loop passes are not asked for a smaller file");
        let e96 = r.expected.iter().find(|e| e.key == "e=96").unwrap();
        assert_eq!((e96.experts, e96.passes, e96.visits, e96.heal_tokens_m), (96, 3, 48, 20));
        assert!((12.0..13.0).contains(&e96.q4_gb) && (15.0..16.5).contains(&e96.packed_gb) && e96.heal_hours.0 > 3.0, "{e96:?}");
        assert!(e96.heal_gb < 32.0 && e96.disk_peak_gb > 40.0);
        assert!((e96.nvfp4_gb - e96.packed_gb).abs() < 1.0 && (38.0..42.0).contains(&e96.f16_gb), "{e96:?}");
        assert!(r.warnings.iter().all(|w| !w.contains("too old")) && r.warnings.iter().any(|w| w.contains("59 GB")));
        assert!(r.unknowns.iter().any(|u| u.contains("nothing was run")));
    }

    #[test]
    fn old_software_and_a_small_disk_are_warned_about() {
        let ins = ink();
        let old = Have { transformers: Some("5.17.0".into()), llama_tag: Some("b12000".into()), vllm: Some("vllm/vllm-openai:v0.34.0".into()) };
        let mut i = input(&ins, "size", &old);
        i.free_gb = Some(30.0);
        let r = recommend(&i);
        for tool in ["transformers", "llama.cpp", "vLLM"] {
            assert!(r.warnings.iter().any(|w| w.contains(tool)), "{tool}: {:?}", r.warnings);
        }
        assert!(r.warnings.iter().any(|w| w.contains("of disk at once")), "{:?}", r.warnings);
    }

    #[test]
    fn speed_asks_about_passes_only_as_a_question_and_train_points_away_from_cutting() {
        let ins = ink();
        let have = Have::default();
        let s = recommend(&input(&ins, "speed", &have));
        assert_eq!(s.first_try[..4], ["base", "k=6", "k=4", "p=2"]);
        assert!(s.order.iter().any(|x| x.id == "passes" && x.text.contains("Auto-cut never does it")) && s.expected.iter().any(|e| e.key == "p=2" && e.visits == 36));
        let t = recommend(&input(&ins, "train", &have));
        assert!(t.first_try.is_empty() && t.order.iter().any(|x| x.id == "train") && !t.order.iter().any(|x| x.id == "cut"));
        assert!(recommend(&input(&ins, "magic", &have)).blocked.is_some());
        let dense: Inspect = serde_json::from_value(serde_json::json!({"supported": true, "layers": 4, "period": 2, "blocks": 2, "total_params": 10})).unwrap();
        assert!(recommend(&input(&dense, "size", &have)).blocked.is_some());
    }

    #[test]
    fn a_target_size_adds_its_own_count() {
        let ins = ink();
        let have = Have::default();
        let mut i = input(&ins, "size", &have);
        i.target_gb = Some(11.0);
        let r = recommend(&i);
        assert!(r.expected.iter().any(|e| e.experts == 80), "{:?}", r.expected.iter().map(|e| e.experts).collect::<Vec<_>>());
    }
}
