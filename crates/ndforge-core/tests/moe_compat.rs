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
