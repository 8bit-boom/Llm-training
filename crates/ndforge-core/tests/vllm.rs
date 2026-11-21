//! The program side of the experimental vLLM engine: version parsing, the detection order, the preflight and the worker's `engine` object, with fake runners.
use ndforge_core::envmgr::health::Light;
use ndforge_core::settings::{Settings, VllmSettings};
use ndforge_core::vllm::*;
use serde_json::json;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn runner<'a>(answers: &'a [(&'a str, bool, &'a str)]) -> impl Fn(&[&str]) -> Option<(bool, String)> + 'a {
    move |args: &[&str]| {
        let line = args.join(" ");
        answers.iter().find(|(k, _, _)| line.starts_with(k)).map(|(_, ok, out)| (*ok, out.to_string()))
    }
}

#[test]
fn it_is_off_by_default_and_the_defaults_are_the_reports() {
    let s = Settings::default().vllm;
    assert!(!s.enabled && !s.adapter_rounds && s.kind == "auto");
    assert_eq!(s.image, "vllm/vllm-openai:v0.34.0");
    assert_eq!(s.teacher_repo, "cyankiwi/Qwen3.8-27B-AWQ-INT4");
    assert!(s.model_volume.starts_with("ndforge-vllm-"));
    let back: Settings = serde_json::from_str("{\"expert_mode\": true}").unwrap();      // an old settings file loads
    assert_eq!(back.vllm, VllmSettings::default());
}

#[test]
fn wsl_versions_are_compared_as_numbers_and_unknown_is_not_old() {
    assert_eq!(parse_wsl_version("WSL version: 3.0.1.0\nKernel version: 6.18"), Some(vec![3, 0, 1, 0]));
    assert_eq!(parse_wsl_version("Version de WSL : 2.7.3"), Some(vec![2, 7, 3]));
    assert_eq!(parse_wsl_version("W\0S\0L\0 \02\0.\08\0.\01\0"), Some(vec![2, 8, 1]));
    assert_eq!(parse_wsl_version("garbage"), None);
    assert_eq!(wsl_note("WSL version: 2.7.3.0").0, Some(true));
    assert_eq!(wsl_note("WSL version: 3.0.1.0").0, Some(true));
    assert_eq!(wsl_note("WSL version: 2.10.0").0, Some(true), "2.10 is newer than 2.7: numbers, not strings");
    let (old, text) = wsl_note("WSL version: 2.6.1.0");
    assert_eq!(old, Some(false));
    assert!(text.contains("wsl --update"));
    let (unknown, text) = wsl_note("");
    assert_eq!(unknown, None);
    assert!(text.contains("unknown, continuing"));
}

#[test]
fn the_detection_order_is_docker_then_docker_in_wsl_then_wslc_then_wsl() {
    let s = VllmSettings::default();
    let f = |a: &[(&str, bool, &str)]| gather_facts(&runner(a), &s);
    assert_eq!(detect_kind(&f(&[("docker version", true, ""), ("wslc --version", true, ""), ("wsl.exe -- docker version", true, "")])), "docker");
    assert_eq!(detect_kind(&f(&[("wsl.exe -- docker version", true, ""), ("wslc --version", true, "")])), "docker_wsl");
    assert_eq!(detect_kind(&f(&[("wslc --version", true, "")])), "wslc");
    assert_eq!(detect_kind(&f(&[("wsl.exe -- test -x ~/.ndforge/vllm/bin/vllm", true, "")])), "wsl");
    assert_eq!(detect_kind(&f(&[])), "missing");
    let with_distro = VllmSettings { distro: "Ubuntu".into(), ..VllmSettings::default() };
    let got = gather_facts(&runner(&[("wsl.exe -d Ubuntu -- docker version", true, "")]), &with_distro);
    assert!(got.docker_wsl && !got.docker);
}

#[test]
fn the_choice_of_the_person_wins_over_the_detection() {
    let f = gather_facts(&runner(&[("docker version", true, "")]), &VllmSettings::default());
    assert_eq!(used_kind(&VllmSettings::default(), &f), "docker");
    assert_eq!(used_kind(&VllmSettings { kind: "wslc".into(), ..VllmSettings::default() }, &f), "wslc");
    assert_eq!(used_kind(&VllmSettings { kind: "nonsense".into(), ..VllmSettings::default() }, &f), "missing");
}

#[test]
fn the_status_lines_and_the_guided_text() {
    let s = VllmSettings::default();
    let missing = status(&s, &runner(&[("nvidia-smi", true, "572.16, 1500, 32768\n")]));
    assert_eq!(missing["kind_used"], "missing");
    assert_eq!(missing["ready"], false);
    assert!(missing["guide"].as_array().unwrap().iter().any(|l| l.as_str().unwrap().contains("Nothing is installed for you")));
    let driver = missing["lines"].as_array().unwrap().iter().find(|l| l["label"] == "NVIDIA driver").unwrap();
    assert_eq!(driver["light"], "warn", "572 is older than 580");
    let ready = status(&s, &runner(&[("docker version", true, ""), ("nvidia-smi", true, "581.42, 900, 32768\n"), ("wsl.exe --version", true, "WSL version: 3.0.1.0")]));
    assert_eq!(ready["ready"], true);
    assert_eq!(ready["kind_used"], "docker");
    let wsl = ready["lines"].as_array().unwrap().iter().find(|l| l["label"] == "WSL").unwrap();
    assert_eq!(wsl["light"], "ok");
    assert!(ready["guide"].as_array().unwrap()[0].as_str().unwrap().contains("untested"));
}

#[test]
fn the_preflight_blocks_a_busy_card_and_a_missing_engine() {
    let s = VllmSettings::default();
    let busy = preflight_checks(&s, &gather_facts(&runner(&[("docker version", true, ""), ("nvidia-smi", true, "581.42, 12288, 32768\n")]), &s));
    assert!(busy.iter().any(|c| c.id == "vllm_gpu" && c.light == Light::Fail && c.detail.contains("12.0 GB")));
    assert!(busy.iter().any(|c| c.id == "vllm_engine" && c.light == Light::Ok));
    let free = preflight_checks(&s, &gather_facts(&runner(&[("docker version", true, ""), ("nvidia-smi", true, "581.42, 1200, 32768\n")]), &s));
    assert!(free.iter().all(|c| c.light != Light::Fail));
    let none = preflight_checks(&s, &gather_facts(&runner(&[]), &s));
    assert!(none.iter().any(|c| c.id == "vllm_engine" && c.light == Light::Fail));
    assert!(none.iter().any(|c| c.id == "vllm_gpu" && c.light == Light::Warn), "an unreadable card is a warning, not a block");
    let old_wsl = preflight_checks(&VllmSettings { kind: "wsl".into(), ..s.clone() }, &gather_facts(&runner(&[("wsl.exe --version", true, "WSL version: 2.5.0")]), &s));
    assert!(old_wsl.iter().any(|c| c.id == "vllm_wsl"));
}

#[test]
fn the_engine_object_for_the_worker_leaves_out_empty_settings() {
    let e = engine_job(&VllmSettings::default());
    assert_eq!(e["kind"], "auto");
    assert_eq!(e["image"], "vllm/vllm-openai:v0.34.0");
    assert_eq!(e["hf_volume"], "ndforge-vllm-hf");
    assert_eq!(e["teacher_repo"], "cyankiwi/Qwen3.8-27B-AWQ-INT4");
    assert!(e.get("distro").is_none() && e.get("url").is_none() && e.get("wslc").is_none());
    let s = VllmSettings { kind: "wslc".into(), distro: "U".into(), wslc: serde_json::json!({"shm": "8g"}), ..VllmSettings::default() };
    let e = engine_job(&s);
    assert_eq!(e["distro"], "U");
    assert_eq!(e["wslc"]["shm"], "8g");
}

#[test]
fn the_self_test_job_uses_the_configured_original_and_needs_the_confirmation_flag_from_the_page() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let s = VllmSettings::default();
    let j = selftest_job(&s, &json!({}), None).unwrap();
    assert_eq!(j["job"], "vllm_selftest");
    assert_eq!(j["model"], "cyankiwi/Qwen3.8-27B-AWQ-INT4");
    assert_eq!(j["confirm_download"], false);
    assert_eq!(j["quick"], true);
    assert_eq!(j["engine"]["kind"], "auto");
    assert_eq!(j["stop_file"], json!(selftest_dir().join("STOP")));
    assert!(j.get("adapter").is_none());
    let j = selftest_job(&s, &json!({"model": " C:\\models\\q ", "confirm_download": true, "adapter": "/a"}), None).unwrap();
    assert_eq!((j["model"].clone(), j["confirm_download"].clone(), j["adapter"].clone()), (json!("C:\\models\\q"), json!(true), json!("/a")));
    assert!(selftest_job(&VllmSettings { teacher_repo: String::new(), ..s }, &json!({}), None).is_err());
    assert_eq!(selftest_report(), serde_json::Value::Null);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn the_self_test_compares_with_the_saved_standard_results_of_the_project() {
    let d = tempfile::tempdir().unwrap();
    let p = ndforge_core::project::Project::create(d.path(), "P").unwrap();
    assert_eq!(compare_files(Some(&p)), json!({}));
    let t = ndforge_core::reshape::runs::models_dir(&p).join("slim-1").join("think_check");
    std::fs::create_dir_all(&t).unwrap();
    std::fs::write(t.join("original.quick.json"), "{}").unwrap();
    std::fs::write(t.join("original.quick.vllm.json"), "{}").unwrap();
    let c = compare_files(Some(&p));
    assert_eq!(c["hf"], json!(t.join("original.quick.json")), "never the vLLM file");
    assert!(c.get("llama").is_none());
    assert_eq!(compare_files(None), json!({}));
}

#[test]
fn the_card_is_shared_with_the_desktop_the_share_is_lowered_not_refused() {
    let p = gpu_plan(Some(32607), Some(1931), 0.88);
    assert_eq!((p.ok, p.lowered, p.util), (Some(true), false, 0.88));
    let q = gpu_plan(Some(32607), Some(2713), 0.88);      // the owner's card with the desktop on it
    assert_eq!(q.ok, Some(true));
    assert!(q.util <= 0.88 && q.util >= 0.70);
    let l = gpu_plan(Some(32607), Some(4500), 0.88);
    assert!(l.lowered && l.util < 0.88 && l.text.contains("vLLM will take") && l.text.contains("instead of 0.88"));
    assert!(l.util * 32607.0 <= 32607.0 - 4500.0 - 500.0);
    assert_eq!(gpu_plan(Some(32607), Some(12288), 0.88).ok, Some(false));
    assert!(gpu_plan(Some(32607), Some(12288), 0.88).text.contains("Close those programs"));
    for (t, u) in [(None, None), (Some(32607), None), (Some(0), Some(1))] {
        let g = gpu_plan(t, u, 0.88);
        assert_eq!((g.ok, g.util), (None, 0.88));
    }
}

#[test]
fn the_self_test_job_asks_for_the_fp8_pass_only_when_ticked() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let s = VllmSettings::default();
    assert!(selftest_job(&s, &json!({}), None).unwrap().get("kv_fp8_test").is_none());
    assert_eq!(selftest_job(&s, &json!({"kv_fp8_test": true}), None).unwrap()["kv_fp8_test"], true);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn the_container_probe_job_carries_the_container_settings_of_the_other_vllm_jobs() {
    let s = VllmSettings { kind: "wslc".into(), wslc: serde_json::json!({"cp_in": "wslc cp {src} {name}:{dst}"}), ..VllmSettings::default() };
    let j = container_probe_job(&s);
    assert_eq!(j["job"], "vllm_container_probe");
    assert_eq!(j["engine"]["kind"], "vllm");
    assert_eq!(j["engine"]["container"]["kind"], "wslc");
    assert_eq!(j["engine"]["container"]["wslc"]["cp_in"], "wslc cp {src} {name}:{dst}");
}
