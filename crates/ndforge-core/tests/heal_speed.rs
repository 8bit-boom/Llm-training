//! "Measure heal speed": the job the button sends (additive keys only), where the report is, and the Windows hints (shown, never changed) with fake readers.
use ndforge_core::reshape::heal_speed::*;
use ndforge_core::winhints::*;
use serde_json::json;

#[test]
fn the_job_adds_only_bench_keys_and_the_teacher_only_when_asked() {
    let base = json!({"job": "distill", "distill": {"mode": "cache", "k": 32}, "bench": {"steps": 12, "warmup": 3, "targets": "cache"}});
    let d = tempfile::tempdir().unwrap();
    let j = job_of(&base, None, d.path(), &[]);
    assert_eq!(j["job"], "distill");
    assert_eq!((j["bench"]["profile"].clone(), j["bench"]["microbench"].clone(), j["bench"]["out_dir"].clone()), (json!(true), json!(true), json!(d.path())));
    assert!(j["bench"].get("teacher_bench").is_none() && j["distill"].get("teacher").is_none() && j["bench"].get("windows_hints").is_none());
    let t = job_of(&base, Some(("org/T", "4bit")), d.path(), &["HAGS: off".to_string()]);
    assert_eq!((t["bench"]["teacher_bench"].clone(), t["distill"]["teacher"]["repo"].clone(), t["bench"]["windows_hints"][0].clone()), (json!(true), json!("org/T"), json!("HAGS: off")));
    assert_eq!(base["bench"], json!({"steps": 12, "warmup": 3, "targets": "cache"}), "the base job is not touched");
}

#[test]
fn the_report_is_found_where_the_worker_writes_it() {
    let d = tempfile::tempdir().unwrap();
    let p = ndforge_core::project::Project::create(d.path(), "P").unwrap();
    assert_eq!(files(&p, "slim-1")["exists"], false);
    std::fs::create_dir_all(report_dir(&p, "slim-1")).unwrap();
    std::fs::write(report_dir(&p, "slim-1").join("heal_speed_report.md"), "# Heal speed report").unwrap();
    let f = files(&p, "slim-1");
    assert_eq!((f["exists"].clone(), f["markdown"].clone()), (json!(true), json!("# Heal speed report")));
    assert!(f["dir"].as_str().unwrap().contains("heal_speed-slim-1"));
}

fn ask<'a>(answers: &'a [(&'a str, &'a str)]) -> impl Fn(&[&str]) -> Option<String> + 'a {
    move |args: &[&str]| answers.iter().find(|(k, _)| args.join(" ").starts_with(k)).map(|(_, v)| v.to_string())
}

#[test]
fn windows_hints_are_read_not_changed_and_empty_elsewhere() {
    assert!(hints_with(false, &|_: &[&str]| panic!("nothing is asked off Windows")).is_empty());
    let on = ask(&[("reg query", "    HwSchMode    REG_DWORD    0x2"), ("powercfg", "Power Scheme GUID: 381b4222  (High performance)"), ("nvidia-smi", "Disabled")]);
    let h = hints_with(true, &on);
    assert!(h[0].contains("scheduling: on") && h[1].contains("High performance") && h[2].contains("No display is attached"));
    assert_eq!(card_line(&h), None, "all good: no line on the card");
    let bad = ask(&[("reg query", "HwSchMode REG_DWORD 0x1"), ("powercfg", "Power Scheme GUID: a1841308  (Balanced)"), ("nvidia-smi", "Enabled")]);
    let h = hints_with(true, &bad);
    assert!(h[0].contains("scheduling: off") && h[1].contains("Balanced") && h[1].contains("never changes it") && h[2].contains("A display is attached"));
    assert!(card_line(&h).unwrap().contains("scheduling: off"));
    let unknown = hints_with(true, &|_: &[&str]| None);
    assert!(unknown.iter().all(|x| x.contains("unknown")));
}
