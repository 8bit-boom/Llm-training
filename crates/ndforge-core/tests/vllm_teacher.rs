//! The teacher's answers through vLLM (experimental), program side: the identity rule (the teacher is the person's original), the keys of the cache and of the job, the estimate from the
//! person's own measured numbers, the check file, the gate before a heal. Fakes only: no Docker, no vLLM.
use ndforge_core::reshape::heal_source::Origin;
use ndforge_core::reshape::runs::{rows_key, rows_key_with, vllm_teacher_estimate, vllm_teacher_gate, HealSpec};
use ndforge_core::settings::{VllmSettings, VllmTeacherSettings};
use ndforge_core::timing::History;
use ndforge_core::vllm::*;
use ndforge_core::vllm_plan::Numbers;
use serde_json::{json, Value};
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn spec(v: Value) -> HealSpec {
    serde_json::from_value(v).unwrap()
}

fn origin(teacher: &str) -> Origin {
    Origin { teacher: teacher.into(), family: "qwen3_5".into(), student_b: 17.4, teacher_b: 27.8, config: json!({"num_hidden_layers": 48, "hidden_size": 5120, "vocab_size": 248320}),
             calibration: json!({}), loss_before: json!(2.0), loss_after: json!(3.0), external: false, teacher_chosen: false, student_label: "cut".into() }
}

/// the numbers of the person's first self-test on a real PC (RTX 5090, Windows, WSLc): ready after 510.6 s, 659.2 tokens/s with 40 requests at once
fn selftest() -> Value {
    json!({"verdict": "partial", "startup_seconds": 510.6, "tokens_per_s": 659.2, "wall_seconds": 95.0, "completion_tokens": 62000, "model_params_b": 27.0,
           "answer_tokens": {"n": 16, "median": 1500, "mean": 1800, "p90": 3000}, "short_answers": false, "long_tokens_per_s": 120.0})
}

fn history(check: &[Value]) -> History {
    let mut h = History::default();
    measured_numbers(&selftest(), check).push_into(&mut h);
    h
}

#[test]
fn it_is_off_by_default_and_the_default_table_has_one_pair() {
    let t = VllmTeacherSettings::default();
    assert!(!t.enabled && !t.required && t.route == "auto" && t.confirmed.is_empty());
    assert_eq!(t.pairs.len(), 1);
    assert_eq!(t.pairs["Qwen/Qwen3.8-27B"], "cyankiwi/Qwen3.8-27B-AWQ-INT4");
    let old: ndforge_core::settings::Settings = serde_json::from_str("{\"vllm\": {\"enabled\": true}}").unwrap();
    assert_eq!(old.vllm.teacher, VllmTeacherSettings::default(), "settings saved before it existed load");
}

#[test]
fn the_pre_quantized_copy_belongs_to_one_original_only() {
    let t = VllmTeacherSettings::default();
    let r = teacher_route(&t, "Qwen/Qwen3.8-27B");
    assert_eq!((r["route"].clone(), r["quant"].clone()), (json!("prequant"), json!("awq:cyankiwi/Qwen3.8-27B-AWQ-INT4@unpinned")));
    assert_eq!(teacher_route(&t, "qwen/qwen3.8-27b/")["route"], "prequant");
    for other in ["Someone/Qwen3.8-27B-Ultimate-Heretic", "Qwen/Qwen3.8-14B", "C:\\models\\heretic"] {
        let r = teacher_route(&t, other);
        assert_eq!((r["route"].clone(), r["quant"].clone()), (json!("bnb"), json!("bnb-nf4")), "{other}: never the base model's quantization");
    }
    let forced = VllmTeacherSettings { route: "prequant".into(), ..t.clone() };
    assert_eq!(teacher_route(&forced, "Someone/Finetune")["route"], "refused");
    assert!(teacher_route(&forced, "Someone/Finetune")["reason"].as_str().unwrap().contains("wrong model"));
    let confirmed = VllmTeacherSettings { confirmed: [("Someone/Finetune".to_string(), "Someone/Finetune-AWQ".to_string())].into(), ..t.clone() };
    assert_eq!(teacher_route(&confirmed, "Someone/Finetune")["quantized_repo"], "Someone/Finetune-AWQ");
    assert_eq!(teacher_route(&confirmed, "Other/Model")["route"], "bnb", "a confirmation is per original");
    assert_eq!(teacher_route(&VllmTeacherSettings { route: "bnb".into(), ..t }, "Qwen/Qwen3.8-27B")["route"], "bnb");
}

#[test]
fn the_cache_key_of_the_standard_teacher_is_what_it_always_was() {
    let d = tempfile::tempdir().unwrap();
    let f = d.path().join("train.jsonl");
    std::fs::write(&f, "{}\n").unwrap();
    let plain = rows_key(&f, 5_000_000, "org/T", "4bit");
    assert_eq!(rows_key_with(&f, 5_000_000, "org/T", "4bit", None), plain);
    let a = rows_key_with(&f, 5_000_000, "org/T", "4bit", Some(&json!({"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4"})));
    let b = rows_key_with(&f, 5_000_000, "org/T", "4bit", Some(&json!({"engine": "vllm-0.34.0", "teacher_quant": "awq:x/y@unpinned"})));
    let c = rows_key_with(&f, 5_000_000, "org/T", "4bit", Some(&json!({"engine": "vllm-0.35.0", "teacher_quant": "bnb-nf4"})));
    assert!(plain != a && a != b && a != c && b != c, "a vLLM cache is never taken for the standard one, nor one quantization for another");
}

#[test]
fn a_heal_without_the_vllm_option_is_byte_identical() {
    let s = spec(json!({"tokens_m": 40, "general_share": 0.5}));
    assert!(s.vllm.is_none() && !s.vllm_teacher());
    assert!(serde_json::to_value(&s).unwrap().get("vllm").is_none());
    let on = spec(json!({"tokens_m": 40, "vllm": {"teacher_cache": true}}));
    assert!(on.vllm_teacher());
    assert_eq!(serde_json::to_value(&on).unwrap()["vllm"], json!({"teacher_cache": true}));
}

#[test]
fn the_job_object_carries_the_route_the_container_and_the_checks_verdict() {
    let s = VllmSettings { teacher: VllmTeacherSettings { required: true, ..Default::default() }, ..Default::default() };
    let e = teacher_engine_job(&s, None);
    assert_eq!(e["kind"], "vllm");
    assert_eq!(e["required"], true);
    assert_eq!(e["teacher"]["route"], "auto");
    assert_eq!(e["teacher"]["pairs"]["Qwen/Qwen3.8-27B"], "cyankiwi/Qwen3.8-27B-AWQ-INT4");
    assert_eq!(e["container"]["kind"], "auto");
    assert!(e.get("teacher_check").is_none());
    assert_eq!(teacher_engine_job(&s, Some(&json!({"verdict": "close"})))["teacher_check"], json!({"verdict": "close"}));
    let id = teacher_identity(&s, "Someone/Finetune");
    assert_eq!((id["engine"].clone(), id["teacher_quant"].clone(), id["route"].clone()), (json!("vllm-0.34.0"), json!("bnb-nf4"), json!("bnb")));
    assert_eq!(image_version("vllm/vllm-openai:v0.34.0"), "0.34.0");
    assert_eq!(image_version("registry:5000/vllm/vllm-openai"), "unknown");
    assert_eq!(image_version("vllm/vllm-openai:latest"), "latest");
}

#[test]
fn the_check_job_asks_for_100_rows_next_to_the_models_folder() {
    let d = tempfile::tempdir().unwrap();
    let common = json!({"teacher": {"repo": "org/T", "load": "4bit"}, "student": "/s", "model": {"family": "qwen3_5"}, "data": {"train": "/t.jsonl"}, "k": 32, "max_seq": 4096, "seed": 3407});
    let j = teacher_check_job(&VllmSettings::default(), &common, d.path(), Some(&d.path().join("cache")));
    assert_eq!((j["job"].clone(), j["rows"].clone(), j["k"].clone(), j["max_seq"].clone()), (json!("vllm_teacher_check"), json!(100), json!(32), json!(4096)));
    assert_eq!(j["out"], json!(d.path().join("vllm_teacher_check.json")));
    assert_eq!(j["reference_cache"], json!(d.path().join("cache")));
    assert_eq!(j["engine"]["kind"], "vllm");
    assert!(teacher_check_job(&VllmSettings::default(), &common, d.path(), None).get("reference_cache").is_none());
}

#[test]
fn the_check_is_current_only_for_the_teacher_the_settings_would_use() {
    let d = tempfile::tempdir().unwrap();
    let s = VllmSettings::default();
    assert_eq!(read_teacher_check(d.path(), &s, "Someone/Finetune"), Value::Null);
    std::fs::write(teacher_check_file(d.path()), json!({"verdict": "same", "identity": {"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4"}}).to_string()).unwrap();
    let c = read_teacher_check(d.path(), &s, "Someone/Finetune");
    assert_eq!((c["verdict"].clone(), c["current"].clone()), (json!("same"), json!(true)));
    assert_eq!(read_teacher_check(d.path(), &s, "Qwen/Qwen3.8-27B")["current"], false, "the other route (pre-quantized) is another teacher");
    let newer = VllmSettings { image: "vllm/vllm-openai:v0.35.0".into(), ..s };
    assert_eq!(read_teacher_check(d.path(), &newer, "Someone/Finetune")["current"], false);
}

#[test]
fn the_estimate_uses_your_own_numbers_and_says_where_they_come_from() {
    let o = origin("Someone/Finetune");
    let s = VllmSettings { enabled: true, teacher: VllmTeacherSettings { enabled: true, ..Default::default() }, ..Default::default() };
    let sp = spec(json!({"tokens_m": 40, "vllm": {"teacher_cache": true}}));
    let e = vllm_teacher_estimate(&o, &sp, &history(&[]), &s, None);
    assert_eq!(e["startup_measured"], true);
    assert_eq!(e["scoring_measured"], false, "the self-test's generation speed is not the scoring speed");
    let start = e["startup_min"].as_array().unwrap();
    assert!((start[0].as_f64().unwrap() - 510.6 / 60.0 * 0.85).abs() < 0.5 && (start[1].as_f64().unwrap() - 510.6 / 60.0 * 1.25).abs() < 0.8);
    assert!(e["text"].as_str().unwrap().contains("not measured here") && e["text"].as_str().unwrap().contains("measured by your self-test"));
    let basis = e["basis"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect::<Vec<_>>().join(" ");
    assert!(basis.contains("8.5 minutes") && basis.contains("27 B model") && basis.contains("659") && basis.contains("NOT measured"), "{basis}");
    assert_eq!(e["route"]["route"], "bnb");
    assert!(e["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("never been run")));
    assert!(e["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("not been checked")));
    // with a teacher check that measured the scoring speed
    let chk = json!({"verdict": "same", "current": true, "vllm": {"tokens_per_s": 2500.0}});
    let e2 = vllm_teacher_estimate(&o, &sp, &history(&[chk.clone()]), &s, Some(&chk));
    assert_eq!(e2["scoring_measured"], true);
    assert!(!e2["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("not been checked")));
    // 40 M tokens at 2500/s = 4.4 h of scoring: against the standard engine's 8 to 20 hours
    let v = e2["vllm_min"].as_array().unwrap();
    assert!(v[0].as_f64().unwrap() > 200.0 && v[1].as_f64().unwrap() < 400.0, "{v:?}");
    assert_eq!(e2["faster"], true);
}

#[test]
fn a_heal_in_rounds_pays_the_start_up_every_round_and_the_card_says_so() {
    let o = origin("Someone/Finetune");
    let s = VllmSettings::default();
    let one = vllm_teacher_estimate(&o, &spec(json!({"tokens_m": 40, "recipe": "thinking"})), &history(&[]), &s, None);
    let rounds = vllm_teacher_estimate(&o, &spec(json!({"tokens_m": 40, "seq": 8192, "recipe": "thinking", "rounds": {"round_m": 5}})), &history(&[]), &s, None);
    assert_eq!(one["engine_starts"], 1.0);
    assert!(rounds["engine_starts"].as_f64().unwrap() >= 8.0);
    assert!(rounds["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("starts the engine again")));
    assert!(rounds["vllm_min"][0].as_f64().unwrap() > one["vllm_min"][0].as_f64().unwrap() + 7.0 * 7.0, "7 more starts of at least 7 minutes");
}

#[test]
fn the_numbers_of_a_failed_or_empty_self_test_are_not_used() {
    let n = measured_numbers(&json!({}), &[]);
    assert_eq!(n, Numbers::default());
    let n = measured_numbers(&json!({"verdict": "fail", "error": "no gpu"}), &[]);
    assert_eq!(n.startup_s, None);
    let n = measured_numbers(&selftest(), &[json!({"vllm": {}}), json!({"vllm": {"tokens_per_s": 1800.0}})]);
    assert_eq!((n.startup_s, n.quick_wall_s, n.gen_tps, n.score_tps, n.model_b, n.long_tps), (Some(510.6), Some(95.0), Some(659.2), Some(1800.0), Some(27.0), Some(120.0)));
}

#[test]
fn the_gate_refuses_what_would_teach_the_wrong_model_or_was_called_different() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let dir = d.path().join("model");
    std::fs::create_dir_all(&dir).unwrap();
    let on = spec(json!({"tokens_m": 40, "vllm": {"teacher_cache": true}}));
    let off = spec(json!({"tokens_m": 40}));
    assert!(vllm_teacher_gate(&dir, &off, "Someone/Finetune", "cache").unwrap().is_none(), "no option, no change");
    assert!(vllm_teacher_gate(&dir, &on, "Someone/Finetune", "cache").unwrap_err().0.contains("not switched on"));
    let mut m = ndforge_core::settings::Settings::default();
    m.vllm.enabled = true;
    m.vllm.teacher.enabled = true;
    m.save();
    assert!(vllm_teacher_gate(&dir, &on, "Someone/Finetune", "online").unwrap_err().0.contains("SAVED answers"));
    let (e, id) = vllm_teacher_gate(&dir, &on, "Someone/Finetune", "cache").unwrap().unwrap();
    assert_eq!((e["kind"].clone(), id["teacher_quant"].clone()), (json!("vllm"), json!("bnb-nf4")));
    // the route "prequant" for an original that has no confirmed copy
    m.vllm.teacher.route = "prequant".into();
    m.save();
    assert!(vllm_teacher_gate(&dir, &on, "Someone/Finetune", "cache").unwrap_err().0.contains("wrong model"));
    m.vllm.teacher.route = "auto".into();
    m.save();
    // a saved check that says "different" (for the settings now) refuses; one made for other settings does not
    std::fs::write(teacher_check_file(&dir), json!({"verdict": "different", "identity": {"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4"}}).to_string()).unwrap();
    assert!(vllm_teacher_gate(&dir, &on, "Someone/Finetune", "cache").unwrap_err().0.contains("different answers"));
    std::fs::write(teacher_check_file(&dir), json!({"verdict": "different", "identity": {"engine": "vllm-0.33.0", "teacher_quant": "bnb-nf4"}}).to_string()).unwrap();
    let (e, _) = vllm_teacher_gate(&dir, &on, "Someone/Finetune", "cache").unwrap().unwrap();
    assert!(e.get("teacher_check").is_none(), "an old check does not travel with the job");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn leftover_containers_are_found_by_name_and_removed() {
    let calls = Mutex::new(Vec::<String>::new());
    let run = |args: &[&str]| -> Option<(bool, String)> {
        calls.lock().unwrap().push(args.join(" "));
        if args.contains(&"ps") { Some((true, "ndforge-vllm-4242\nother-thing\nndforge-vllm-9\n".into())) } else { Some((true, String::new())) }
    };
    let gone = stop_leftovers(&VllmSettings::default(), &run);
    assert_eq!(gone, vec!["ndforge-vllm-4242".to_string(), "ndforge-vllm-9".to_string()]);
    let c = calls.lock().unwrap().clone();
    assert!(c.iter().any(|x| x == "docker rm -f ndforge-vllm-4242") && !c.iter().any(|x| x.contains("other-thing") && x.contains("rm")));
    let in_wsl = stop_leftovers(&VllmSettings { kind: "docker_wsl".into(), distro: "Ubuntu".into(), ..Default::default() }, &|a: &[&str]| { assert_eq!(&a[..5], ["wsl.exe", "-d", "Ubuntu", "--", "docker"]); Some((true, String::new())) });
    assert!(in_wsl.is_empty());
    assert!(stop_leftovers(&VllmSettings { kind: "wslc".into(), ..Default::default() }, &|_: &[&str]| panic!("wslc has no known listing syntax: nothing is run")).is_empty());
}

#[test]
fn numbers_of_a_9b_model_are_never_presented_as_those_of_a_27b_model() {
    let nine = json!({"verdict": "partial", "startup_seconds": 510.6, "tokens_per_s": 659.2, "wall_seconds": 7.7, "model_params_b": 9.0, "answer_tokens": {"n": 16, "median": 80}, "short_answers": true});
    let mut h = History::default();
    measured_numbers(&nine, &[]).push_into(&mut h);
    let n = Numbers::from_history(&h);
    assert_eq!(n.model_b, Some(9.0));
    assert_eq!(n.quick_wall_s, None, "short answers say nothing about a thinking check");
    use ndforge_core::vllm_plan as vp;
    assert!(!vp::startup(&n, 27.8).measured, "3 times the size: the report's formulas, not the 9 B number");
    let s9 = vp::startup(&n, 9.0);
    assert!(s9.measured && (s9.lo - 510.6 / 60.0 * 0.85).abs() < 0.01);
    let s13 = vp::startup(&n, 13.5);
    assert!(s13.measured && (s13.lo - 510.6 / 60.0 * 0.85 * 1.5).abs() < 0.01, "1.5 times the size: 1.5 times the start-up");
    assert!(vp::basis_notes(&n, 27.8).iter().any(|t| t.contains("NOT used") && t.contains("9 B")));
    // an older report that does not say its model: not used at all
    let old = measured_numbers(&json!({"verdict": "partial", "startup_seconds": 510.6}), &[]);
    assert!(!vp::startup(&old, 9.0).measured && vp::basis_notes(&old, 9.0).iter().any(|t| t.contains("does not say which model")));
    // the long-answer speed gives a check's generation time, scaled by size
    let long = Numbers { long_tps: Some(100.0), model_b: Some(27.0), ..Default::default() };
    let g = vp::check_generation(&long, true, 1.5, 27.0);
    assert!(g.measured && (g.lo - 30_000.0 / 100.0 / 60.0 * 0.85).abs() < 0.01);
}

#[test]
fn the_selftest_files_say_where_they_are() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let f = selftest_files();
    assert_eq!(f["exists"], false);
    assert_eq!(f["dir"], json!(d.path().join("cfg").join("vllm_selftest").display().to_string()));
    std::fs::create_dir_all(selftest_dir()).unwrap();
    std::fs::write(selftest_dir().join("vllm_selftest.md"), "# vLLM self-test\n**FAIL**").unwrap();
    let f = selftest_files();
    assert_eq!((f["exists"].clone(), f["markdown"].clone()), (json!(true), json!("# vLLM self-test\n**FAIL**")));
    assert!(f["md"].as_str().unwrap().ends_with("vllm_selftest.md"));
    let m = d.path().join("m");
    std::fs::create_dir_all(&m).unwrap();
    assert_eq!(teacher_check_files(&m)["exists"], false);
    std::fs::write(teacher_check_file(&m), "{\"verdict\": \"same\"}").unwrap();
    let c = teacher_check_files(&m);
    assert!(c["exists"] == true && c["text"].as_str().unwrap().contains("\"verdict\": \"same\""));
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
