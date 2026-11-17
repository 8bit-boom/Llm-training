//! The checklist before a job starts (preflight.rs): the rules with made-up facts, then the real gathering on a temporary project.
use ndforge_core::envmgr::health::Light;
use ndforge_core::preflight::{can_start, check, evaluate, Inputs, Need, KINDS};
use serde_json::json;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn ready(kind: &str) -> Inputs {
    Inputs {
        kind: kind.into(), windows: false, non_ascii_path: false, env_installed: true, selftest_passed: true, card: None, llama: Need::No, llama_found: true, llama_what: "",
        free_gb: Some(500.0), need_gb: None, has_data: None, verdict: None, blockers: vec![], other_instance: false, share_reachable: None, sleep: None, keep_awake: true, think_cut_missing: None,
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

fn facts(standby: u32, hibernate: u32, unattend: u32, modern: bool) -> ndforge_core::power::SleepFacts {
    ndforge_core::power::SleepFacts { ac_standby_s: Some(standby), ac_hibernate_s: Some(hibernate), ac_unattend_s: Some(unattend), modern_standby: Some(modern) }
}

#[test]
fn the_sleep_light_covers_every_branch_and_is_never_red() {
    for kind in KINDS {
        let run = |s: Option<ndforge_core::power::SleepFacts>, keep: bool| {
            let mut i = ready(kind);
            i.sleep = s;
            i.keep_awake = keep;
            let checks = evaluate(&i);
            assert!(can_start(&checks), "{kind}: sleep is never red");
            checks.into_iter().find(|c| c.id == "sleep")
        };
        assert!(run(None, true).is_none(), "{kind}: unknown facts, no check");
        let ok = run(Some(facts(0, 0, 0, false)), true).unwrap();
        assert_eq!(ok.light, Light::Ok);
        assert!(ok.action.is_none());
        // keep-awake off and an idle timer: warn, with the button and the commands
        let w = run(Some(facts(900, 0, 0, false)), false).unwrap();
        assert_eq!(w.light, Light::Warn);
        assert_eq!(w.action, Some("keep_awake_on"));
        assert!(w.detail.contains("15 min") && w.fix.contains("STANDBYIDLE 0") && w.fix.contains("HIBERNATEIDLE 0") && w.fix.contains("UNATTENDSLEEP 0") && w.fix.contains("powercfg -setactive SCHEME_CURRENT") && w.fix.contains("Keep this computer awake"));
        // keep-awake off but the plan never sleeps: fine
        assert_eq!(run(Some(facts(0, 0, 0, false)), false).unwrap().light, Light::Ok);
        // keep-awake on, Modern Standby or unattended sleep: still a warning, no button
        let m = run(Some(facts(0, 0, 0, true)), true).unwrap();
        assert_eq!(m.light, Light::Warn);
        assert!(m.action.is_none() && m.detail.contains("Modern Standby") && m.detail.contains("stops the job"));
        let u = run(Some(facts(0, 0, 120, false)), true).unwrap();
        assert_eq!((u.light, u.action), (Light::Warn, None));
        assert!(u.detail.contains("unattended sleep"));
        // keep-awake off with Modern Standby: the button is offered as well
        assert_eq!(run(Some(facts(0, 0, 0, true)), false).unwrap().action, Some("keep_awake_on"));
    }
}

#[test]
fn sleep_facts_exist_only_on_windows_and_the_action_is_left_out_of_json_when_there_is_none() {
    #[cfg(not(windows))]
    assert!(ndforge_core::power::sleep_facts().is_none());
    let mut i = ready("auto");
    i.sleep = Some(facts(600, 0, 0, false));
    i.keep_awake = false;
    let v = serde_json::to_value(evaluate(&i)).unwrap();
    let sleep = v.as_array().unwrap().iter().find(|c| c["id"] == "sleep").unwrap();
    assert_eq!(sleep["action"], "keep_awake_on");
    let env = v.as_array().unwrap().iter().find(|c| c["id"] == "env").unwrap();
    assert!(env.get("action").is_none(), "{env}");
}

#[test]
fn a_heal_of_a_model_that_thinks_warns_until_its_cut_has_had_a_thinking_check() {
    use ndforge_core::reshape::thinkcheck::{cut_checked, think_dir};
    let mut i = ready("heal");
    i.think_cut_missing = Some(24.0);
    let checks = evaluate(&i);
    let c = checks.iter().find(|c| c.id == "think_cut").expect("the warning");
    assert_eq!(c.light, Light::Warn, "never red");
    assert!(c.detail.contains("has not had a thinking check yet (about 24 min)") && c.detail.contains("Lower loss does not mean it can still think."), "{c:?}");
    assert!(can_start(&checks));
    i.think_cut_missing = Some(0.0);
    assert!(evaluate(&i).iter().find(|c| c.id == "think_cut").unwrap().detail.starts_with("The cut has not had a thinking check yet."), "no minutes when they are not known");
    i.think_cut_missing = None;
    assert!(evaluate(&i).iter().all(|c| c.id != "think_cut"));
    // the real look: the saved check of the cut, either size
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("no-env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = ndforge_core::project::Project::create(d.path(), "P").unwrap();
    let spec = json!({"thinks": true, "name": "20261113-150000-q-slim-w10752", "think_minutes": 18.0});
    assert!(check(Some(&p), "heal", &spec).iter().any(|c| c.id == "think_cut" && c.detail.contains("about 18 min")));
    assert!(check(Some(&p), "heal", &json!({"thinks": false, "name": "x"})).iter().all(|c| c.id != "think_cut"), "a model that does not think is not asked");
    assert!(check(Some(&p), "heal", &json!({"thinks": true})).iter().all(|c| c.id != "think_cut"), "without a name there is nothing to look for");
    let tdir = think_dir(&p, "20261113-150000-q-slim-w10752");
    std::fs::create_dir_all(&tdir).unwrap();
    assert!(!cut_checked(&p, "20261113-150000-q-slim-w10752"));
    std::fs::write(tdir.join("cut.full.json"), "{}").unwrap();
    assert!(cut_checked(&p, "20261113-150000-q-slim-w10752"));
    assert!(check(Some(&p), "heal", &spec).iter().all(|c| c.id != "think_cut"), "checked: no warning");
    std::env::remove_var("NDFORGE_ENV_DIR");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
