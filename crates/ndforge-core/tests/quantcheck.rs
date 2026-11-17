//! Compare quants (export/quantcheck.rs): the choice and its old/partial forms, the job the worker gets and every refusal before it runs, what the page offers (files of the
//! exports with the words of the table), the saved comparisons (list, delete, delete the logits), and fake-worker runs (unix only). Small files only.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::export::quantcheck as qc;
use ndforge_core::export::runs as exports;
use ndforge_core::project::Project;
use ndforge_core::sample;
use ndforge_core::train::plan::{base_hyper, Method, Preset};
use ndforge_core::train::runs as truns;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn plain_project(root: &Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

fn project_with_data(root: &Path) -> Project {
    let mut p = plain_project(root);
    let f = root.join("w1.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    p
}

fn fake_done_run(p: &Project, name: &str) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("adapter")).unwrap();
    std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
    std::fs::write(rdir.join("adapter/tokenizer_config.json"), "{}").unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    let job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "model_id": m.id, "method": "lora"}).to_string()).unwrap();
}

/// an export folder `exports/<name>/` with its meta.json and a GGUF file
fn fake_export(p: &Project, name: &str, run: &str, quant: &str, meta: Value) -> PathBuf {
    let d = exports::exports_dir(p).join(name);
    std::fs::create_dir_all(&d).unwrap();
    let file = format!("model-{quant}.gguf");
    std::fs::write(d.join(&file), vec![0u8; 2048]).unwrap();
    let mut m = json!({"name": name, "run": run, "quant": quant, "gguf_file": file, "created": "2026-11-01T10:00:00Z", "installed": []});
    for (k, v) in meta.as_object().cloned().unwrap_or_default() {
        m[k] = v;
    }
    std::fs::write(d.join("meta.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    d.join(file)
}

fn gguf(dir: &Path, name: &str) -> String {
    let f = dir.join(name);
    std::fs::write(&f, b"GGUF-x").unwrap();
    f.display().to_string()
}

fn pick(path: &str) -> qc::QuantPick {
    qc::QuantPick { path: path.into(), label: "Q4_K_M".into(), ..Default::default() }
}

fn spec_with(dir: &Path) -> qc::CompareSpec {
    qc::CompareSpec {
        reference: qc::ReferenceSpec { kind: "file".into(), file: Some(gguf(dir, "ref-f16.gguf")), source: None },
        quants: vec![pick(&gguf(dir, "q4.gguf"))],
        text: qc::TextSpec { kind: "file".into(), file: Some(gguf(dir, "text.txt")), ..Default::default() },
        ..Default::default()
    }
}

// ------------------------------------------------------------------------------------------------------------------ the choice

#[test]
fn the_choice_has_plain_defaults_and_old_or_partial_forms_load() {
    let s = qc::CompareSpec::default();
    assert_eq!((s.chunks, s.keep_logits, s.reference.kind.as_str(), s.text.kind.as_str(), s.resume.clone()), (16, false, "file", "data", None));
    assert_eq!((qc::DEFAULT_CHUNKS, qc::CTX, qc::CHUNKS.len()), (16, 512, 6));
    let s: qc::CompareSpec = serde_json::from_value(json!({})).unwrap();
    assert_eq!(s, qc::CompareSpec::default());
    let s: qc::CompareSpec = serde_json::from_value(json!({"quants": [{"path": "a.gguf"}], "chunks": 8, "something_new": 1})).unwrap();
    assert_eq!((s.quants.len(), s.quants[0].matrix, s.quants[0].label.as_str(), s.chunks), (1, None, "", 8));
    let s: qc::CompareSpec = serde_json::from_value(json!({"reference": {"kind": "make", "source": {"kind": "run", "run": "r1"}}, "resume": "x"})).unwrap();
    assert_eq!((s.reference.source.unwrap().run.as_deref(), s.resume.as_deref()), (Some("r1"), Some("x")));
}

#[test]
fn the_words_of_the_table_come_from_the_export() {
    let (l, g, m, x) = qc::pick_for_export(&json!({"quant": "q4_k_m", "run": "r1", "imatrix": {"applied": true}}), false);
    assert_eq!((l.as_str(), g.as_deref(), m, x), ("Q4_K_M, with matrix", Some("Q4_K_M"), Some(true), None));
    let (l, _, m, _) = qc::pick_for_export(&json!({"quant": "q4_k_m", "run": "r1", "imatrix": null}), true);
    assert_eq!((l.as_str(), m), ("r1: Q4_K_M, no matrix", Some(false)));
    let (l, _, m, _) = qc::pick_for_export(&json!({"quant": "q4_k_m", "imatrix": {"requested": true, "applied": false}}), false);
    assert_eq!((l.as_str(), m), ("Q4_K_M, no matrix", Some(false)));
    let (l, _, m, _) = qc::pick_for_export(&json!({"quant": "q8_0"}), false);
    assert_eq!((l.as_str(), m), ("Q8_0", None), "a matrix does nothing for Q8_0");
    let (l, g, m, x) = qc::pick_for_export(&json!({"quant": "mixed_4", "mixed": {"id": "mixed_4", "label": "Mixed 4-bit (balanced)"}, "imatrix": {"applied": true}}), false);
    assert_eq!((l.as_str(), g.as_deref(), m, x.as_deref()), ("Mixed 4-bit (balanced), with matrix", Some("Mixed 4-bit (balanced)"), Some(true), Some("mixed_4")));
}

// ------------------------------------------------------------------------------------------------------------------ the job and every refusal

#[test]
fn the_job_has_the_reference_the_quants_the_text_and_the_settings() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let spec = spec_with(d.path());
    let job = qc::build_job(&p, &spec, "20261115-1030-ref", Path::new("STOP")).unwrap();
    assert_eq!((job["job"].as_str(), job["id"].as_str(), job["chunks"].as_u64(), job["ctx"].as_u64(), job["ngl"].as_u64(), job["resume"].clone(), job["keep_logits"].clone()),
               (Some("quantcheck"), Some("20261115-1030-ref"), Some(16), Some(512), Some(99), json!(false), json!(false)));
    assert_eq!(job["root"], json!(qc::dir(&p)));
    assert_eq!(job["reference"]["kind"], "file");
    assert_eq!((job["quants"][0]["label"].as_str(), job["quants"][0]["path"].as_str().unwrap().ends_with("q4.gguf")), (Some("Q4_K_M"), true));
    assert_eq!(job["text"]["kind"], "file");
    // an empty label becomes the file name; the label is cut at 80 characters
    let mut s2 = spec.clone();
    s2.quants = vec![qc::QuantPick { path: spec.quants[0].path.clone(), label: " ".into(), ..Default::default() }];
    assert_eq!(qc::build_job(&p, &s2, "x", Path::new("S")).unwrap()["quants"][0]["label"], "q4.gguf");
    s2.quants[0].label = "x".repeat(200);
    assert_eq!(qc::build_job(&p, &s2, "x", Path::new("S")).unwrap()["quants"][0]["label"].as_str().unwrap().len(), 80);
    let mut s3 = spec.clone();
    s3.resume = Some("old".into());
    s3.keep_logits = true;
    let j = qc::build_job(&p, &s3, "old", Path::new("S")).unwrap();
    assert_eq!((j["resume"].clone(), j["keep_logits"].clone()), (json!(true), json!(true)));
}

#[test]
fn everything_a_person_could_get_wrong_is_a_plain_message_before_anything_runs() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let ok = spec_with(d.path());
    let err = |s: &qc::CompareSpec| qc::build_job(&p, s, "x", Path::new("S")).unwrap_err().0;
    assert!(err(&qc::CompareSpec { chunks: 1, ..ok.clone() }).contains("between 2 and 1024"));
    assert!(err(&qc::CompareSpec { chunks: 5000, ..ok.clone() }).contains("between 2 and 1024"));
    assert!(err(&qc::CompareSpec { quants: vec![], ..ok.clone() }).contains("at least one quantised"));
    assert!(err(&qc::CompareSpec { quants: vec![pick("")], ..ok.clone() }).contains("has no path"));
    assert!(err(&qc::CompareSpec { quants: vec![pick(&gguf(d.path(), "notes.txt"))], ..ok.clone() }).contains("not a GGUF file"));
    assert!(err(&qc::CompareSpec { quants: vec![pick("/nowhere/q.gguf")], ..ok.clone() }).contains("was not found"));
    let q = ok.quants[0].path.clone();
    assert!(err(&qc::CompareSpec { quants: vec![pick(&q), pick(&q)], ..ok.clone() }).contains("twice"));
    assert!(err(&qc::CompareSpec { reference: qc::ReferenceSpec { kind: "file".into(), file: None, source: None }, ..ok.clone() }).contains("Choose the 16-bit reference"));
    assert!(err(&qc::CompareSpec { reference: qc::ReferenceSpec { kind: "file".into(), file: Some("/nowhere/r.gguf".into()), source: None }, ..ok.clone() }).contains("reference file /nowhere/r.gguf was not found"));
    assert!(err(&qc::CompareSpec { reference: qc::ReferenceSpec { kind: "file".into(), file: Some(q.clone()), source: None }, ..ok.clone() }).contains("compared with itself"));
    assert!(err(&qc::CompareSpec { reference: qc::ReferenceSpec { kind: "make".into(), file: None, source: None }, ..ok.clone() }).contains("made from"));
    assert!(err(&qc::CompareSpec { reference: qc::ReferenceSpec { kind: "make".into(), file: None, source: Some(Default::default()) }, ..ok.clone() }).contains("where the model comes from"));
    assert!(err(&qc::CompareSpec { reference: qc::ReferenceSpec { kind: "nonsense".into(), ..Default::default() }, ..ok.clone() }).contains("Choose the 16-bit reference"));
    assert!(err(&qc::CompareSpec { text: qc::TextSpec { kind: "file".into(), file: None, ..Default::default() }, ..ok.clone() }).contains("Choose the text file"));
    assert!(err(&qc::CompareSpec { text: qc::TextSpec { kind: "file".into(), file: Some("/nowhere/t.txt".into()), ..Default::default() }, ..ok.clone() }).contains("was not found"));
    assert!(err(&qc::CompareSpec { text: qc::TextSpec { kind: "".into(), ..Default::default() }, ..ok.clone() }).contains("Choose the test text"));
    // data: no saved version, then no model to take the chat template from
    assert!(err(&qc::CompareSpec { text: qc::TextSpec { kind: "data".into(), ..Default::default() }, ..ok.clone() }).contains("Save a version of the training data"));
    let d2 = tempdir().unwrap();
    let p2 = project_with_data(d2.path());
    let ok2 = spec_with(d2.path());
    let e2 = qc::build_job(&p2, &qc::CompareSpec { text: qc::TextSpec { kind: "data".into(), ..Default::default() }, ..ok2 }, "x", Path::new("S")).unwrap_err().0;
    assert!(e2.contains("Say which model the files were made from"));
}

#[test]
fn the_data_text_job_names_the_held_out_file_the_tokenizer_sources_and_the_family() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1");
    let (version, _) = truns::latest_version(&p).unwrap();
    let vdir = p.folder.join("datasets").join(format!("v{version}"));
    let ok = spec_with(d.path());
    let mk = |t: qc::TextSpec| qc::build_job(&p, &qc::CompareSpec { text: t, ..ok.clone() }, "x", Path::new("S")).unwrap()["text"].clone();
    let t = mk(qc::TextSpec { kind: "data".into(), run: Some("run1".into()), ..Default::default() });
    assert_eq!((t["kind"].as_str(), t["family"].as_str(), t["seed"].as_u64(), t["data_version"].as_u64()), (Some("data"), Some("qwen3_5"), Some(1234), Some(version as u64)));
    assert_eq!(t["train"], json!(vdir.join("train.jsonl")));
    let src: Vec<String> = serde_json::from_value(t["tokenizer"].clone()).unwrap();
    assert!(src[0].ends_with("adapter") && src.last().unwrap().contains('/'), "the add-on's tokenizer first, then the base model: {src:?}");
    // held-out rows only when the file has any
    std::fs::write(vdir.join("test.jsonl"), "").unwrap();
    assert!(mk(qc::TextSpec { kind: "data".into(), run: Some("run1".into()), ..Default::default() })["test"].is_null());
    std::fs::write(vdir.join("test.jsonl"), "{\"type\":\"chat\"}\n").unwrap();
    assert_eq!(mk(qc::TextSpec { kind: "data".into(), run: Some("run1".into()), ..Default::default() })["test"], json!(vdir.join("test.jsonl")));
    // a typed model instead of a run: the family of a known model, else generic
    let t = mk(qc::TextSpec { kind: "data".into(), tokenizer: Some("Qwen/Qwen3.5-4B".into()), ..Default::default() });
    assert_eq!(t["tokenizer"], json!(["Qwen/Qwen3.5-4B"]));
    assert_eq!(mk(qc::TextSpec { kind: "data".into(), tokenizer: Some("/m/x".into()), ..Default::default() })["family"], "generic");
    assert!(qc::build_job(&p, &qc::CompareSpec { text: qc::TextSpec { kind: "data".into(), run: Some("../x".into()), ..Default::default() }, ..ok.clone() }, "x", Path::new("S")).unwrap_err().0.contains("Unknown run"));
    assert!(qc::build_job(&p, &qc::CompareSpec { text: qc::TextSpec { kind: "data".into(), run: Some("nope".into()), ..Default::default() }, ..ok.clone() }, "x", Path::new("S")).unwrap_err().0.contains("settings of that run are missing"));
}

#[test]
fn a_reference_made_from_a_run_goes_through_the_matrix_librarys_source() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1");
    let ok = spec_with(d.path());
    let s = qc::CompareSpec { reference: qc::ReferenceSpec { kind: "make".into(), file: None, source: Some(ndforge_core::export::imatrix_lib::SourceSpec { kind: "run".into(), run: Some("run1".into()), ..Default::default() }) }, ..ok };
    let job = qc::build_job(&p, &s, "x", Path::new("S")).unwrap();
    assert_eq!((job["reference"]["kind"].as_str(), job["reference"]["source"]["kind"].as_str(), job["reference"]["source"]["run"].as_str()), (Some("make"), Some("run"), Some("run1")));
    assert!(job["reference"]["source"]["adapter"].as_str().unwrap().ends_with("adapter"));
}

#[test]
fn a_continued_comparison_measures_its_saved_test_text_again() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let mut spec = spec_with(d.path());
    spec.text = qc::TextSpec { kind: "data".into(), ..Default::default() };      // would be refused: no data version, no model
    let dir = saved(&p, "20261101-1000-a", "stopped", true);
    spec.resume = Some("20261101-1000-a".into());
    assert!(qc::build_job(&p, &spec, "20261101-1000-a", Path::new("S")).unwrap_err().0.contains("Save a version"), "no saved test text: the text is asked for as usual");
    std::fs::write(dir.join("test.txt"), "text").unwrap();
    let job = qc::build_job(&p, &spec, "20261101-1000-a", Path::new("S")).unwrap();
    assert_eq!((job["text"].clone(), job["resume"].clone()), (json!({"kind": "saved"}), json!(true)));
    spec.resume = Some("../x".into());
    assert!(qc::build_job(&p, &spec, "x", Path::new("S")).unwrap_err().0.contains("Save a version"), "a bad id never reads a file");
}

// ------------------------------------------------------------------------------------------------------------------ the page

#[test]
fn the_page_offers_the_files_of_the_exports_with_their_words_and_the_references() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1");
    fake_export(&p, "run1-f16", "run1", "f16", json!({}));
    fake_export(&p, "run1-q4_k_m", "run1", "q4_k_m", json!({"imatrix": {"requested": true, "applied": true}}));
    fake_export(&p, "run1-q8_0", "run1", "q8_0", json!({}));
    let side = exports::exports_dir(&p).join("run1-q4_k_m").join("model-f16.gguf");
    std::fs::write(&side, b"x").unwrap();
    // a reference an earlier comparison kept
    let kept = qc::dir(&p).join("20261101-1000-ref");
    std::fs::create_dir_all(&kept).unwrap();
    std::fs::write(kept.join("reference-f16.gguf"), b"x").unwrap();
    let c = qc::candidates(&p);
    let refs: Vec<&str> = c["references"].as_array().unwrap().iter().map(|r| r["kind"].as_str().unwrap()).collect();
    assert_eq!(refs.iter().filter(|k| **k == "export").count(), 2, "the F16 export and the model-f16.gguf left in an export folder");
    assert_eq!(refs.iter().filter(|k| **k == "comparison").count(), 0, "a folder without results.json is not a comparison");
    let labels: Vec<&str> = c["quants"].as_array().unwrap().iter().map(|q| q["label"].as_str().unwrap()).collect();
    assert_eq!(labels.len(), 2);
    assert!(labels.contains(&"Q4_K_M, with matrix") && labels.contains(&"Q8_0"));
    let q4 = c["quants"].as_array().unwrap().iter().find(|q| q["quant"] == "q4_k_m").unwrap();
    assert_eq!((q4["matrix"].clone(), q4["group"].clone(), q4["run"].clone()), (json!(true), json!("Q4_K_M"), json!("run1")));
    assert_eq!(qc::picks_of_run(&p, "run1").len(), 2);
    assert!(qc::picks_of_run(&p, "other").is_empty());
    // the state of the card
    let st = qc::state(&p);
    assert_eq!((st["default_chunks"].as_u64(), st["ctx"].as_u64(), st["data_version"].is_number()), (Some(16), Some(512), true));
    assert_eq!(st["chunks"], json!([4, 8, 16, 32, 64, 128]));
    assert!(st["runs"].as_array().unwrap().iter().any(|r| r["name"] == "run1") && st["models"].as_array().unwrap().len() > 3);
    assert!(st["entries"].as_array().unwrap().is_empty());
}

// ------------------------------------------------------------------------------------------------------------------ the saved comparisons

fn saved(p: &Project, id: &str, state: &str, with_base: bool) -> PathBuf {
    let d = qc::dir(p).join(id);
    std::fs::create_dir_all(&d).unwrap();
    let r = json!({"format": 1, "id": id, "created": "2026-11-15T10:30:00Z", "updated": "2026-11-15T10:40:00Z", "state": state, "reference": {"name": "ref-f16.gguf"}, "text": {"source": "the held-out questions"},
                   "settings": {"chunks": 16, "ctx": 512}, "base": {"done": true, "kept": with_base},
                   "rows": [{"id": "1", "status": "done", "label": "Q4_K_M"}, {"id": "2", "status": "failed", "label": "Q5"}]});
    std::fs::write(d.join("results.json"), serde_json::to_vec(&r).unwrap()).unwrap();
    if with_base {
        std::fs::write(d.join("base.kld"), vec![0u8; 4096]).unwrap();
    }
    d
}

#[test]
fn comparisons_are_listed_newest_first_read_and_deleted() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    assert!(qc::list(&p).is_empty(), "an old project has no such folder");
    saved(&p, "20261101-1000-a", "done", false);
    saved(&p, "20261102-1000-b", "stopped", true);
    std::fs::create_dir_all(qc::dir(&p).join("not-a-comparison")).unwrap();
    std::fs::write(qc::dir(&p).join("loose-file.txt"), "x").unwrap();
    std::fs::write(qc::dir(&p).join("20261103-damaged").join("x"), "x").ok();
    let l = qc::list(&p);
    assert_eq!(l.iter().map(|e| e["id"].as_str().unwrap()).collect::<Vec<_>>(), ["20261102-1000-b", "20261101-1000-a"]);
    assert_eq!((l[0]["state"].as_str(), l[0]["rows"].as_u64(), l[0]["rows_done"].as_u64(), l[0]["base_kept"].clone(), l[0]["base_mb"].as_f64()), (Some("stopped"), Some(2), Some(1), json!(true), Some(0.0)));
    assert_eq!((l[1]["base_kept"].clone(), l[1]["reference"].as_str(), l[1]["chunks"].as_u64()), (json!(false), Some("ref-f16.gguf"), Some(16)));
    let (dir, r) = qc::get(&p, "20261102-1000-b").unwrap();
    assert!(dir.ends_with("20261102-1000-b") && r["rows"].as_array().unwrap().len() == 2 && r["base_kept"] == true && r["running"] == false);
    for bad in ["", "..", "a/b", "_x", ".h"] {
        assert!(qc::get(&p, bad).is_err(), "{bad}");
    }
    assert!(qc::get(&p, "gone").unwrap_err().0.contains("no longer exists"));
    // only the big file goes
    assert!(qc::delete_logits(&p, "20261102-1000-b").unwrap() == 4096);
    assert!(!dir.join("base.kld").exists() && dir.join("results.json").is_file());
    let (_, r) = qc::get(&p, "20261102-1000-b").unwrap();
    assert_eq!((r["base"]["kept"].clone(), r["base"]["done"].clone()), (json!(false), json!(false)), "a later continue reads the text again");
    assert!(r.get("dir").is_some());
    let on_disk: Value = serde_json::from_slice(&std::fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert!(on_disk.get("dir").is_none() && on_disk.get("running").is_none(), "what the program adds for the page is not written back");
    qc::delete(&p, "20261101-1000-a").unwrap();
    assert_eq!(qc::list(&p).len(), 1);
    assert!(qc::delete(&p, "20261101-1000-a").is_err());
}

#[test]
fn new_ids_are_numbered_when_the_name_is_taken() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    assert_eq!(qc::id_at(&p, "20261115-1030", "Qwen3.5-9B-F16"), "20261115-1030-qwen3-5-9b-f16");
    saved(&p, "20261115-1030-qwen3-5-9b-f16", "done", false);
    assert_eq!(qc::id_at(&p, "20261115-1030", "Qwen3.5-9B-F16"), "20261115-1030-qwen3-5-9b-f16-2");
}

// ------------------------------------------------------------------------------------------------------------------ fake workers (unix)

#[cfg(unix)]
fn fake_worker(root: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\njob=\"$5\"\nlibroot=$(sed -n 's/.*\"root\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nid=$(sed -n 's/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nstop=$(sed -n 's/.*\"stop_file\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
#[test]
fn a_whole_comparison_runs_in_a_fake_worker_and_the_entry_appears() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"mkdir -p "$libroot/$id"
echo '{"format":1,"id":"x","created":"2026-11-15T10:30:00Z","state":"done","reference":{"name":"ref-f16.gguf"},"text":{"source":"s"},"settings":{"chunks":16,"ctx":512},"rows":[{"id":"1","status":"done","label":"Q4_K_M","mean_kld":0.02}]}' > "$libroot/$id/results.json"
echo '{"type":"progress","step":"quantcheck","value":0.5,"text":"Comparing Q4_K_M"}'
echo '{"type":"log","text":"measuring"}'
echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"state\":\"done\",\"rows\":[{\"id\":\"1\",\"status\":\"done\"}]}}""#);
    let p = plain_project(d.path());
    let spec = spec_with(d.path());
    let mut events = Vec::new();
    let r = qc::compare(&p, &spec, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert!(r["entry"]["id"].as_str().unwrap().ends_with("-ref-f16"), "{r}");
    assert_eq!((r["entry"]["state"].as_str(), r["entry"]["rows_done"].as_u64()), (Some("done"), Some(1)));
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["step"] == "quantcheck" && e["text"] == "Comparing Q4_K_M") && events.iter().any(|e| e["kind"] == "log" && e["text"] == "measuring"));
    assert_eq!(qc::list(&p).len(), 1);
    assert!(!qc::dir(&p).join("STOP").exists(), "the STOP file is cleaned up");
    assert!(!qc::running(r["entry"]["id"].as_str().unwrap()), "released afterwards");
}

#[cfg(unix)]
#[test]
fn a_running_comparison_cannot_be_deleted_and_continuing_needs_an_existing_one() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"echo '{"type":"progress","step":"quantcheck","value":0.1,"text":"x"}'
echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\"}}""#);
    let p = plain_project(d.path());
    saved(&p, "20261101-1000-running", "stopped", true);
    let mut spec = spec_with(d.path());
    spec.resume = Some("20261101-1000-running".into());
    let mut refused = None;
    let r = qc::compare(&p, &spec, &AtomicBool::new(false), &mut |_| { refused = Some(qc::delete(&p, "20261101-1000-running").unwrap_err().0); }).unwrap();
    assert!(refused.unwrap().contains("running right now") && r["entry"]["id"] == "20261101-1000-running");
    spec.resume = Some("gone".into());
    assert!(qc::compare(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("no longer exists"));
    spec.resume = Some("../x".into());
    assert!(qc::compare(&p, &spec, &AtomicBool::new(false), &mut |_| {}).is_err());
}

#[cfg(unix)]
#[test]
fn stop_writes_the_stop_file_and_a_failure_comes_back_in_the_workers_words() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"while [ ! -f "$stop" ]; do sleep 0.05; done
echo '{"type":"error","code":"quantcheck_stopped","message":"Cancelled."}'
exit 1"#);
    let p = plain_project(d.path());
    let spec = spec_with(d.path());
    let kill = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let r = std::thread::scope(|s| {
        s.spawn(|| { std::thread::sleep(std::time::Duration::from_millis(300)); kill.store(true, std::sync::atomic::Ordering::SeqCst); });
        qc::compare(&p, &spec, &kill, &mut |_| {})
    });
    assert_eq!(r.unwrap_err().0, "Cancelled.");
    assert!(t0.elapsed() < std::time::Duration::from_secs(20));
    fake_worker(d.path(), r#"echo '{"type":"error","code":"quantcheck_failed","message":"llama-perplexity was not found, so the quants cannot be compared. Install ... LLAMA_CPP_DIR"}'; exit 1"#);
    assert!(qc::compare(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("LLAMA_CPP_DIR"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    assert!(qc::compare(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Training setup"));
    assert!(qc::plan(&p, None).unwrap_err().0.contains("Training setup"));
}

#[cfg(unix)]
#[test]
fn the_plan_runs_the_workers_plan_job() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"cp "$job" "$libroot/plan-job.json"
echo '{"type":"result","data":{"vocab":248320,"default_chunks":16,"options":[{"chunks":16,"bytes":2026000000,"gb":2.03,"fits":true}],"refusal":null}}'"#);
    let p = plain_project(d.path());
    let r = qc::plan(&p, Some("/m/ref-f16.gguf")).unwrap();
    assert_eq!((r["vocab"].as_u64(), r["default_chunks"].as_u64()), (Some(248320), Some(16)));
    let sent: Value = serde_json::from_slice(&std::fs::read(qc::dir(&p).join("plan-job.json")).unwrap()).unwrap();
    assert_eq!((sent["job"].as_str(), sent["reference"].as_str(), sent["ctx"].as_u64()), (Some("quantcheck_plan"), Some("/m/ref-f16.gguf"), Some(512)));
    qc::plan(&p, Some("  ")).unwrap();
    let sent: Value = serde_json::from_slice(&std::fs::read(qc::dir(&p).join("plan-job.json")).unwrap()).unwrap();
    assert!(sent["reference"].is_null(), "an empty reference means the vocabulary is assumed");
}

#[test]
fn the_general_text_job_has_the_source_the_tokenizer_the_family_and_the_fixed_seed() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());                        // no saved data version: the general text does not need one
    let ok = spec_with(d.path());
    let text_file = d.path().join("general.txt");
    std::fs::write(&text_file, "some general text").unwrap();
    let general = ndforge_core::general::GeneralSource { file: Some(text_file.display().to_string()), ..Default::default() };
    let mk = |t: qc::TextSpec| qc::build_job(&p, &qc::CompareSpec { text: t, ..ok.clone() }, "x", Path::new("S"));
    let t = mk(qc::TextSpec { kind: "general".into(), tokenizer: Some("Qwen/Qwen3.5-4B".into()), general: Some(general.clone()), ..Default::default() }).unwrap()["text"].clone();
    assert_eq!((t["kind"].as_str(), t["seed"].as_u64(), t["family"].as_str()), (Some("general"), Some(1234), Some("generic")));
    assert_eq!(t["tokenizer"], json!(["Qwen/Qwen3.5-4B"]));
    assert_eq!(t["source"]["file"], json!(text_file.display().to_string()));
    // a model that needs naming for its chat template; and both a dataset and a file is refused
    assert!(mk(qc::TextSpec { kind: "general".into(), general: Some(general.clone()), ..Default::default() }).unwrap_err().0.contains("Say which model"));
    let both = ndforge_core::general::GeneralSource { id: Some("a/b".into()), ..general };
    assert!(mk(qc::TextSpec { kind: "general".into(), tokenizer: Some("m".into()), general: Some(both), ..Default::default() }).unwrap_err().0.contains("either a Hugging Face dataset or a file"));
    // old specs have no `general`
    let t: qc::TextSpec = serde_json::from_value(json!({"kind": "data", "run": "r"})).unwrap();
    assert!(t.general.is_none());
}
