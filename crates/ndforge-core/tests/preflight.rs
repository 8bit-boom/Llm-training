//! The checklist before a job starts (preflight.rs): the rules with made-up facts, then the real gathering on a temporary project.
use ndforge_core::envmgr::health::Light;
use ndforge_core::preflight::{can_start, check, evaluate, Inputs, Need, KINDS};
use serde_json::json;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn ready(kind: &str) -> Inputs {
    Inputs {
        kind: kind.into(), windows: false, non_ascii_path: false, env_installed: true, selftest_passed: true, card: None, llama: Need::No, llama_found: true, llama_what: "",
        free_gb: Some(500.0), need_gb: None, has_data: None, verdict: None, blockers: vec![], other_instance: false, share_reachable: None,
    }
}

fn light(checks: &[ndforge_core::envmgr::health::Check], id: &str) -> Option<Light> {
    checks.iter().find(|c| c.id == id).map(|c| c.light)
}

#[test]
fn a_machine_that_has_everything_gets_green_lights_and_may_start() {
    for kind in KINDS {
        let checks = evaluate(&ready(kind));
        assert!(can_start(&checks), "{kind}: {checks:?}");
        assert!(checks.iter().all(|c| c.light == Light::Ok), "{kind}");
    }
}

#[test]
fn every_kind_needs_the_training_software_and_one_program() {
    for kind in KINDS {
        let mut i = ready(kind);
        i.env_installed = false;
        let checks = evaluate(&i);
        assert_eq!(light(&checks, "env"), Some(Light::Fail), "{kind}");
        assert!(!can_start(&checks));
        assert!(checks.iter().find(|c| c.id == "env").unwrap().fix.contains("Training setup"));
        let mut i = ready(kind);
        i.other_instance = true;
        assert_eq!(light(&evaluate(&i), "instance"), Some(Light::Fail), "{kind}");
    }
}

#[test]
fn training_kinds_want_a_card_a_self_test_and_data_and_the_others_do_not() {
    for kind in ["train", "heal", "auto"] {
        let mut i = ready(kind);
        i.card = Some(None);
        i.selftest_passed = false;
        i.has_data = Some(false);
        let checks = evaluate(&i);
        assert_eq!((light(&checks, "gpu"), light(&checks, "selftest"), light(&checks, "data")), (Some(Light::Fail), Some(Light::Warn), Some(Light::Fail)), "{kind}");
        i.card = Some(Some(32.0));
        i.has_data = Some(true);
        assert!(can_start(&evaluate(&i)), "a self-test that has not been run only warns");
    }
    for kind in ["export", "think_check", "quantcheck", "imatrix_make"] {
        let mut i = ready(kind);
        i.selftest_passed = false;
        assert!(light(&evaluate(&i), "selftest").is_none() && light(&evaluate(&i), "gpu").is_none(), "{kind}");
    }
}

#[test]
fn llama_cpp_is_a_red_light_where_the_job_cannot_go_without_it_and_a_yellow_one_where_it_can() {
    for kind in ["quantcheck", "imatrix_make", "auto"] {
        let mut i = ready(kind);
        i.llama = Need::Yes;
        i.llama_found = false;
        i.llama_what = "llama-perplexity";
        let checks = evaluate(&i);
        assert_eq!(light(&checks, "llama"), Some(Light::Fail), "{kind}");
        assert!(checks.iter().find(|c| c.id == "llama").unwrap().detail.contains("llama-perplexity") && !can_start(&checks));
    }
    let mut i = ready("export");
    i.llama = Need::Maybe;
    i.llama_found = false;
    let checks = evaluate(&i);
    assert_eq!(light(&checks, "llama"), Some(Light::Warn));
    assert!(can_start(&checks));
    i.llama_found = true;
    assert_eq!(light(&evaluate(&i), "llama"), Some(Light::Ok));
}

#[test]
fn the_disk_is_red_under_the_need_plus_five_percent_yellow_when_close_and_a_nearly_full_disk_is_red_without_an_estimate() {
    let mut i = ready("export");
    i.need_gb = Some(40.0);
    i.free_gb = Some(41.0);
    assert_eq!(light(&evaluate(&i), "disk"), Some(Light::Fail));
    i.free_gb = Some(48.0);
    assert_eq!(light(&evaluate(&i), "disk"), Some(Light::Warn));
    i.free_gb = Some(80.0);
    assert_eq!(light(&evaluate(&i), "disk"), Some(Light::Ok));
    i.free_gb = None;
    assert_eq!(light(&evaluate(&i), "disk"), Some(Light::Warn), "not measurable is not a refusal");
    i.need_gb = None;
    i.free_gb = Some(9.0);
    assert_eq!(light(&evaluate(&i), "disk"), Some(Light::Fail));
    i.free_gb = Some(100.0);
    assert!(light(&evaluate(&i), "disk").is_none());
}

#[test]
fn the_card_estimate_and_the_plans_own_blockers_are_shown() {
    let mut i = ready("heal");
    i.verdict = Some(("over".into(), "needs 40 GB, the card has 32".into()));
    i.blockers = vec!["Save a training-data version in Step 2 first.".into()];
    let checks = evaluate(&i);
    assert_eq!(light(&checks, "card"), Some(Light::Fail));
    assert_eq!(light(&checks, "plan"), Some(Light::Fail));
    assert!(!can_start(&checks));
    for (v, want) in [("tight", Light::Warn), ("fits", Light::Ok), ("unknown", Light::Warn)] {
        i.verdict = Some((v.into(), String::new()));
        i.blockers.clear();
        assert_eq!(light(&evaluate(&i), "card"), Some(want), "{v}");
    }
}

#[test]
fn a_plain_letters_path_warning_only_on_windows_and_only_for_llama_jobs_and_a_missing_share_only_warns() {
    let mut i = ready("quantcheck");
    i.llama = Need::Yes;
    i.non_ascii_path = true;
    assert!(light(&evaluate(&i), "path").is_none(), "not on this system");
    i.windows = true;
    let checks = evaluate(&i);
    assert_eq!(light(&checks, "path"), Some(Light::Warn));
    assert!(can_start(&checks));
    let mut j = ready("heal");
    j.windows = true;
    j.non_ascii_path = true;
    assert!(light(&evaluate(&j), "path").is_none(), "healing does not use llama.cpp");
    let mut a = ready("auto");
    a.share_reachable = Some(false);
    let checks = evaluate(&a);
    assert_eq!(light(&checks, "share"), Some(Light::Warn));
    assert!(can_start(&checks));
    a.share_reachable = Some(true);
    assert_eq!(light(&evaluate(&a), "share"), Some(Light::Ok));
}

#[test]
fn the_real_gathering_on_a_temporary_project_follows_the_environment_the_disk_and_the_spec() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("no-env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    std::env::set_var("NDFORGE_FREE_DISK_GB", "10");
    let p = ndforge_core::project::Project::create(d.path(), "P").unwrap();
    let checks = check(Some(&p), "heal", &json!({"tokens_m": 60, "mode": "cache"}));
    assert_eq!(light(&checks, "env"), Some(Light::Fail), "no environment here");
    assert_eq!(light(&checks, "data"), Some(Light::Fail), "no saved version");
    assert_eq!(light(&checks, "disk"), Some(Light::Fail), "60 M tokens of answers need about 11.8 GB and 10 are free");
    assert!(!can_start(&checks));
    // an export of a run that is not there has no estimate of its own, the page's one wins
    let e = check(Some(&p), "export", &json!({"run": "nope", "quants": ["q4_k_m"], "need_gb": 30.0}));
    assert_eq!(light(&e, "disk"), Some(Light::Fail));
    // unknown kinds and no project are not a crash
    assert!(check(None, "nonsense", &json!({})).is_empty());
    assert!(check(None, "think_check", &json!({})).iter().any(|c| c.id == "env"));
    std::env::remove_var("NDFORGE_FREE_DISK_GB");
    std::env::remove_var("NDFORGE_ENV_DIR");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
