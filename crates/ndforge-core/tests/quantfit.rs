//! "Mixed: measured fit to a size" as a page (export/quantfit.rs): the choice and its old forms, the job the worker gets, every refusal before anything runs, the state of the page and
//! fake-worker runs (unix only). Small files only.
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::export::imatrix_lib as lib;
use ndforge_core::export::quantcheck::TextSpec;
use ndforge_core::export::quantfit as qf;
use ndforge_core::project::Project;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn project(root: &Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

fn gguf(dir: &Path, name: &str) -> String {
    let f = dir.join(name);
    std::fs::write(&f, b"GGUF").unwrap();
    f.display().to_string()
}

fn spec(dir: &Path) -> qf::QuantFitSpec {
    qf::QuantFitSpec { gguf: gguf(dir, "model-f16.gguf"), target_gb: 10.6, ..Default::default() }
}

#[test]
fn the_choice_has_plain_defaults_and_old_or_partial_forms_load() {
    let s: qf::QuantFitSpec = serde_json::from_value(json!({})).unwrap();
    assert_eq!((s.gguf.as_str(), s.target_gb, s.matrix.clone(), s.sample_rows, s.make, s.neighbours, s.compare.is_none()), ("", 0.0, None, 0, false, false, true));
    let s: qf::QuantFitSpec = serde_json::from_value(json!({"gguf": "a.gguf", "target_gb": 9.5, "make": true, "later": 1})).unwrap();
    assert_eq!((s.gguf.as_str(), s.target_gb, s.make), ("a.gguf", 9.5, true));
}

#[test]
fn everything_a_person_could_get_wrong_is_a_plain_message_before_anything_runs() {
    let d = tempdir().unwrap();
    let ok = spec(d.path());
    let err = |s: &qf::QuantFitSpec| s.check(true).unwrap_err().0;
    assert!(err(&qf::QuantFitSpec { gguf: "".into(), ..ok.clone() }).contains("Choose the 16-bit GGUF"));
    assert!(err(&qf::QuantFitSpec { gguf: "/x/model.bin".into(), ..ok.clone() }).contains("is not a GGUF file"));
    assert!(err(&qf::QuantFitSpec { gguf: "/nowhere/m.gguf".into(), ..ok.clone() }).contains("was not found"));
    assert!(err(&qf::QuantFitSpec { target_gb: 0.0, ..ok.clone() }).contains("Give the size in GB"));
    assert!(err(&qf::QuantFitSpec { target_gb: f64::NAN, ..ok.clone() }).contains("Give the size in GB"));
    assert!(err(&qf::QuantFitSpec { floor: Some("Q6_K".into()), ceiling: Some("Q3_K".into()), ..ok.clone() }).contains("lowest type is above the highest"));
    assert!(err(&qf::QuantFitSpec { engine: Some("x".into()), ..ok.clone() }).contains("Unknown way of measuring"));
    assert!(err(&qf::QuantFitSpec { sample_rows: 2, ..ok.clone() }).contains("between 8 and 4096"));
    assert!(err(&qf::QuantFitSpec { compare: Some(TextSpec { kind: "general".into(), ..Default::default() }), make: false, ..ok.clone() }).contains("must be made first"));
    ok.check(true).unwrap();
    // the estimate does not need a size, but a size that is given must be sane
    qf::QuantFitSpec { target_gb: 0.0, ..ok.clone() }.check(false).unwrap();
    assert!(qf::QuantFitSpec { target_gb: -1.0, ..ok.clone() }.check(false).unwrap_err().0.contains("Give the size in GB"));
}

#[test]
fn the_job_has_the_defaults_the_cache_and_the_folder_of_the_files() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let s = spec(d.path());
    let stop = p.folder.join("STOP");
    let j = qf::build_job(&p, &s, &stop).unwrap();
    assert_eq!((j["job"].clone(), j["target_gb"].clone(), j["floor"].clone(), j["ceiling"].clone(), j["sample_rows"].clone(), j["engine"].clone(), j["make"].clone(), j["neighbours"].clone()),
               (json!("quantfit"), json!(10.6), json!("Q3_K"), json!("Q8_0"), json!(128), json!("auto"), json!(false), json!(false)));
    assert!(j["matrix"].is_null() && Path::new(j["cache_dir"].as_str().unwrap()) == qf::dir(&p).as_path() && Path::new(j["stop_file"].as_str().unwrap()) == stop.as_path());
    assert!(Path::new(j["out_dir"].as_str().unwrap()).starts_with(qf::dir(&p).join("files")));
    // neighbours only count with make
    let j = qf::build_job(&p, &qf::QuantFitSpec { neighbours: true, ..s.clone() }, &stop).unwrap();
    assert_eq!(j["neighbours"], json!(false));
    let j = qf::build_job(&p, &qf::QuantFitSpec { neighbours: true, make: true, floor: Some("Q2_K".into()), sample_rows: 256, engine: Some("reference".into()), ..s.clone() }, &stop).unwrap();
    assert_eq!((j["neighbours"].clone(), j["floor"].clone(), j["sample_rows"].clone(), j["engine"].clone()), (json!(true), json!("Q2_K"), json!(256), json!("reference")));
}

#[test]
fn a_saved_matrix_goes_to_the_worker_as_its_file_and_a_missing_one_is_refused() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let id = "20261101-1000-run1";
    let dir = lib::lib_dir(&p).join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("imatrix.gguf"), b"M").unwrap();
    let s = qf::QuantFitSpec { matrix: Some(id.into()), ..spec(d.path()) };
    let j = qf::build_job(&p, &s, &p.folder.join("STOP")).unwrap();
    assert!(j["matrix"].as_str().unwrap().ends_with("imatrix.gguf"));
    assert!(qf::build_job(&p, &qf::QuantFitSpec { matrix: Some("gone".into()), ..s.clone() }, &p.folder.join("STOP")).unwrap_err().0.contains("no longer exists"));
    assert!(qf::build_job(&p, &qf::QuantFitSpec { matrix: Some("../x".into()), ..s }, &p.folder.join("STOP")).unwrap_err().0.contains("not the name of a saved matrix"));
}

#[test]
fn the_state_offers_the_choices_the_caveat_and_what_was_kept() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let st = qf::state(&p);
    assert_eq!(st["ladder"], json!(["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]));
    assert_eq!((st["default_floor"].clone(), st["default_ceiling"].clone(), st["default_sample_rows"].clone(), st["sample_rows"].clone()), (json!("Q3_K"), json!("Q8_0"), json!(128), json!([64, 128, 256, 512])));
    assert!(st["plans"].as_array().unwrap().is_empty() && st["verdict"].is_null());
    assert!(st["caveats"].as_str().unwrap().contains("diagonal") && st["caveats"].as_str().unwrap().contains("Compare quants"));
    // a saved plan and a verdict are listed
    let k = qf::dir(&p).join("abc123");
    std::fs::create_dir_all(&k).unwrap();
    std::fs::write(k.join("plan-10.6.json"), json!({"target_gb": 10.6, "engine": "llama", "fits": true}).to_string()).unwrap();
    std::fs::write(qf::dir(&p).join("verdict.json"), json!({"target_gb": 10.6, "verdict": ["x"]}).to_string()).unwrap();
    let st = qf::state(&p);
    assert_eq!(st["plans"][0]["key"], "abc123");
    assert_eq!(st["plans"][0]["target_gb"], 10.6);
    assert_eq!(st["verdict"]["verdict"], json!(["x"]));
}

#[test]
fn the_label_and_the_words_never_say_dynamic_or_ud() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let text = qf::state(&p).to_string();
    assert!(!text.contains("Dynamic") && !text.contains("UD-"));
}

#[cfg(unix)]
fn fake_worker(root: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

/// answers by the kind of job in $5: the estimate and the measured fit (which "made" two files) and the comparison
#[cfg(unix)]
const WORKER: &str = r#"if grep -q '"job":"quantfit_estimate"' "$5"; then
  echo '{"type":"result","data":{"engine":"reference","minutes_estimate":{"chosen":3.0},"min_gb":8.1,"max_gb":17.0}}'
elif grep -q '"job":"quantfit"' "$5"; then
  echo '{"type":"progress","step":"quantfit","value":0.5,"text":"Measuring"}'
  echo '{"type":"result","data":{"fits":true,"engine":"reference","files":[{"path":"/x/model-mixed_measured.gguf","label":"Mixed: measured fit to a size","group":"Mixed: measured fit to a size","matrix":true,"mixed":"mixed_measured"},{"path":"/x/model-q4_k_m.gguf","label":"Q4_K_M","group":"Q4_K_M","matrix":true,"mixed":null}]}}'
else
  echo '{"type":"result","data":{"promotion":[{"id":"0","promoted":true}],"verdict":["Mixed is better."]}}'
fi"#;

#[cfg(unix)]
#[test]
fn the_estimate_and_a_run_talk_to_the_worker_and_keep_the_verdict() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), WORKER);
    let p = project(d.path());
    let s = spec(d.path());
    let e = qf::estimate(&p, &qf::QuantFitSpec { target_gb: 0.0, ..s.clone() }).unwrap();
    assert_eq!((e["engine"].clone(), e["min_gb"].clone()), (json!("reference"), json!(8.1)));
    let mut events = vec![];
    let r = qf::run(&p, &s, &AtomicBool::new(false), &mut |v| events.push(v)).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), 2);
    assert!(events.iter().any(|v| v["kind"] == "progress") && r.get("comparison").is_none(), "no comparison unless asked for");
}

#[cfg(unix)]
#[test]
fn comparing_makes_the_comparison_with_the_16_bit_file_as_the_reference_and_keeps_the_verdict() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), WORKER);
    let p = project(d.path());
    let text = d.path().join("t.txt");
    std::fs::write(&text, "words").unwrap();
    // the files the fake worker "made" must exist for the comparison to be built
    let x = d.path().join("x");
    std::fs::create_dir_all(&x).unwrap();
    // the fake paths are /x/..., so build the comparison against files that exist by pointing the worker at this folder
    let body = WORKER.replace("/x/", &format!("{}/", x.display()));
    fake_worker(d.path(), &body);
    std::fs::write(x.join("model-mixed_measured.gguf"), b"G").unwrap();
    std::fs::write(x.join("model-q4_k_m.gguf"), b"G").unwrap();
    let s = qf::QuantFitSpec { make: true, neighbours: true, compare: Some(TextSpec { kind: "file".into(), file: Some(text.display().to_string()), ..Default::default() }), ..spec(d.path()) };
    let r = qf::run(&p, &s, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["comparison"]["promotion"][0]["promoted"], true);
    let v: Value = serde_json::from_slice(&std::fs::read(qf::dir(&p).join("verdict.json")).unwrap()).unwrap();
    assert_eq!((v["target_gb"].clone(), v["verdict"][0].clone()), (json!(10.6), json!("Mixed is better.")));
    assert_eq!(qf::state(&p)["verdict"]["verdict"][0], "Mixed is better.");
}
