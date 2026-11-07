use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::export::runs as exports;
use ndforge_core::model::AppModel;
use ndforge_core::project::Project;
use ndforge_core::sample;
use ndforge_core::train::plan::{base_hyper, base_hyper_for, Method, Preset};
use ndforge_core::train::retrain;
use ndforge_core::train::runs as truns;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn project_with_world(root: &std::path::Path, variant: u8) -> Project {
    let mut p = Project::create(&root.join("projects"), "Ashfall Reach").unwrap();
    let f = root.join(format!("w{variant}.ndtrain"));
    sample::build(&f, variant, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    p
}

/// A finished run on disk (what the worker would have left behind).
fn fake_done_run(p: &Project, name: &str, version: u32) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("adapter")).unwrap();
    std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    let hyper = base_hyper(Preset::Normal);
    let job = truns::build_job(p, &m, Method::Lora, &hyper, version, name, false).unwrap();
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("summary.json"), json!({"steps": 40, "final_loss": 0.8}).to_string()).unwrap();
    std::fs::write(rdir.join("result.json"), "{}").unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "model_id": m.id, "method": "lora", "step": 40, "total_steps": 40}).to_string()).unwrap();
}

#[cfg(unix)]
fn fake_worker(root: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\nout=$(sed -n 's/.*\"out\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\nquant=$(sed -n 's/.*\"quant\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
#[test]
fn an_export_writes_gguf_modelfile_card_and_meta_and_can_be_installed_into_the_record() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"echo '{"type":"progress","step":"gguf","value":0.5,"text":"Converting"}'
printf 'GGUF' > "$out/model-$quant.gguf"
echo "{\"type\":\"result\",\"data\":{\"gguf\":\"$out/model-$quant.gguf\",\"quant\":\"$quant\"}}""#);
    let p = project_with_world(d.path(), 1);
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    fake_done_run(&p, "run1", 1);
    assert_eq!(exports::exportable_runs(&p).len(), 1);
    let mut events = Vec::new();
    let meta = exports::execute(&p, "run1", "q4_k_m", 8192, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert_eq!((meta["ollama_name"].as_str(), meta["family"].as_str(), meta["quant"].as_str()), (Some("ashfall-reach-gm"), Some("qwen3_5"), Some("q4_k_m")));
    let dir = p.folder.join("exports/run1-q4_k_m");
    let mf = std::fs::read_to_string(dir.join("Modelfile")).unwrap();
    assert!(mf.starts_with("FROM ./model-q4_k_m.gguf") && mf.contains("PARAMETER num_ctx 8192") && mf.contains("<think>"));
    let card = std::fs::read_to_string(dir.join("MODEL_CARD.md")).unwrap();
    assert!(card.contains("game master only") && card.contains("unsloth/Qwen3.5-4B") && card.contains("40 steps") && card.contains("final loss 0.800") && card.contains("OLLAMA_MODEL=ashfall-reach-gm"));
    assert!(events.iter().any(|e| e["kind"] == "progress"));
    assert!(p.progress()[4].1, "step 5 counts as done once a GGUF exists");
    let list = exports::list_exports(&p);
    assert_eq!((list.len(), list[0]["gguf_exists"].clone()), (1, json!(true)));
    exports::record_install(&p, "run1-q4_k_m", "http://localhost:11434", "ashfall-reach-gm").unwrap();
    exports::record_install(&p, "run1-q4_k_m", "http://localhost:11434", "ashfall-reach-gm").unwrap();     // the same install twice is one entry
    exports::record_install(&p, "run1-q4_k_m", "http://truenas:11434", "ashfall-reach-gm").unwrap();
    assert_eq!(exports::list_exports(&p)[0]["installed"].as_array().unwrap().len(), 2);
    assert!(exports::find_export(&p, "run1-q4_k_m").is_ok());
    assert!(exports::find_export(&p, "../x").is_err() && exports::find_export(&p, "nope").is_err());
}

#[cfg(unix)]
#[test]
fn export_problems_are_plain() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"echo '{"type":"error","code":"export_failed","message":"To make a GGUF file without Unsloth, llama.cpp is needed."}'; exit 1"#);
    let p = project_with_world(d.path(), 1);
    assert!(exports::execute(&p, "nope", "q4_k_m", 8192, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("no saved model"));
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    fake_done_run(&p, "run1", 1);
    assert!(exports::execute(&p, "run1", "q2", 8192, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Unknown file size"));
    assert!(exports::execute(&p, "run1", "q4_k_m", 8192, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("llama.cpp is needed"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    assert!(exports::execute(&p, "run1", "q4_k_m", 8192, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Training setup"));
}

#[test]
fn retrain_is_offered_only_for_a_new_world_after_a_finished_run_and_makes_a_new_version() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let mut p = project_with_world(d.path(), 1);
    assert_eq!(retrain::info(&p).unwrap()["available"], false);                       // nothing trained yet
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    fake_done_run(&p, "run1", 1);
    assert_eq!(retrain::info(&p).unwrap()["available"], false);                       // the world has not changed since the version
    // a newer export arrives, and one earlier example is dropped by the reviewer
    let id = store::list(&p, &store::Filter { builder: Some("aliases".into()), ..Default::default() }, 0, 1).unwrap().1.remove(0).id;
    store::set_status(&p, &[id.clone()], ndforge_core::dataset::builders::Status::Drop).unwrap();
    let f2 = d.path().join("w2.ndtrain");
    sample::build(&f2, 2, None).unwrap();
    p.import_bundle(&open_bundle(&f2).unwrap()).unwrap();
    let info = retrain::info(&p).unwrap();
    assert_eq!(info["available"], true);
    assert!(info["summary"].as_str().unwrap().contains("entities: 1 new, 1 changed, 1 removed"), "{info}");
    assert_eq!((info["last_run"]["model_id"].as_str(), info["steps"].as_array().unwrap().len()), (Some("qwen3.5-4b"), 3));
    assert_eq!(info["continue_suggested"], false);              // 3 of ~8 records changed: more than a quarter
    // the button: a new version with the new data; earlier version and run untouched; the reviewer's drop survives
    let mut m = AppModel::load();
    m.open_project(p.folder.to_str().unwrap()).unwrap();
    let job = m.retrain_job(false).unwrap();
    assert_eq!(job.job["dataset"]["version"], 2);
    assert!(job.job.get("init_adapter").is_none() && job.job["resume"] == false && job.name != "run1");
    assert_eq!(job.hyper.epochs, base_hyper(Preset::Normal).epochs);
    assert!(p.folder.join("datasets/v1/manifest.json").exists() && p.folder.join("datasets/v2/manifest.json").exists() && p.folder.join("runs/run1/adapter").exists());
    let m2: Value = serde_json::from_slice(&std::fs::read(p.folder.join("datasets/v2/manifest.json")).unwrap()).unwrap();
    assert_eq!(m2["bundle_sha256"], json!(p.current_bundle().unwrap().sha256));
    assert!(store::load_examples(&p).unwrap().iter().any(|e| e.id == id && e.status == ndforge_core::dataset::builders::Status::Drop));
    assert_eq!(retrain::info(&p).unwrap()["available"], false);                       // now the data is up to date
    // continue from the previous model: the old adapter is the starting point and the run is shorter
    let warm = m.retrain_job(true).unwrap();
    assert!(warm.job["init_adapter"].as_str().unwrap().ends_with("runs/run1/adapter"));
    assert_eq!(warm.hyper.epochs, 1.0);
    assert_eq!(warm.job["dataset"]["version"], 3);
}

#[test]
fn model_settings_for_ollama_servers_and_the_probe_question() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfgx"));
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("px"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("noenv"));
    let mut m = AppModel::load();
    assert!(m.export_state().unwrap_err().0.contains("Create or open a project"));
    m.create_project("Ashfall Reach").unwrap();
    let f = d.path().join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    m.apply_bundle(f.to_str().unwrap()).unwrap();
    let st = m.export_state().unwrap();
    assert_eq!((st["suggested_name"].as_str(), st["servers"][0]["url"].as_str(), st["quants"].as_array().unwrap().len()), (Some("ashfall-reach-gm"), Some("http://localhost:11434"), 4));
    m.add_ollama_server("TrueNAS", "truenas:11434").unwrap();
    assert!(m.add_ollama_server("", "x").is_err());
    assert_eq!(m.export_state().unwrap()["servers"].as_array().unwrap().len(), 2);
    assert_eq!(AppModel::load().settings.ollama_servers.len(), 2);                     // remembered
    m.remove_ollama_server("http://truenas:11434").unwrap();
    assert_eq!(m.export_state().unwrap()["servers"].as_array().unwrap().len(), 1);
    // before any data version: a generic question; after: one of the training's own
    assert!(m.probe_question().unwrap().1.contains("Tell me about this world"));
    m.dataset_generate(json!({})).unwrap();
    m.dataset_save().unwrap();
    let (sys, q) = m.probe_question().unwrap();
    assert!(sys.contains("Ashfall Reach") && q.contains("Anders") || q.contains("Dockside"), "{sys} / {q}");
}


/// A finished FULL fine-tune on disk: the whole model in runs/<name>/model, no adapter.
fn fake_done_full_run(p: &Project, name: &str, version: u32) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("model")).unwrap();
    std::fs::write(rdir.join("model/config.json"), "{}").unwrap();
    std::fs::write(rdir.join("model/model.safetensors"), "W").unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-2b").unwrap();
    let hyper = base_hyper_for(Method::Full, Preset::Normal);
    let job = truns::build_job(p, &m, Method::Full, &hyper, version, name, false).unwrap();
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("summary.json"), json!({"steps": 30, "final_loss": 0.6}).to_string()).unwrap();
    std::fs::write(rdir.join("result.json"), "{}").unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "model_id": m.id, "method": "full", "step": 30, "total_steps": 30}).to_string()).unwrap();
}

#[cfg(unix)]
#[test]
fn a_full_run_is_exported_from_its_model_folder_and_retrained_from_it() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // the fake worker insists on full_model and on there being no adapter in the job
    fake_worker(d.path(), r#"if grep -q '"full_model"' "$5" && ! grep -q '"adapter"' "$5"; then
  printf 'GGUF' > "$out/model-$quant.gguf"
  echo "{\"type\":\"result\",\"data\":{\"gguf\":\"$out/model-$quant.gguf\",\"quant\":\"$quant\",\"full_model\":true}}"
else
  echo '{"type":"error","code":"export_failed","message":"wrong job"}'; exit 1
fi"#);
    let mut p = project_with_world(d.path(), 1);
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    fake_done_full_run(&p, "full1", 1);
    let runs_list = exports::exportable_runs(&p);
    assert_eq!((runs_list.len(), runs_list[0]["has_model"].clone(), runs_list[0]["has_adapter"].clone()), (1, json!(true), json!(false)));
    let meta = exports::execute(&p, "full1", "q4_k_m", 8192, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(meta["quant"], "q4_k_m");
    assert!(p.folder.join("exports/full1-q4_k_m/model-q4_k_m.gguf").is_file());
    // retrain with the new export, continuing from the previous FULL model: the job points at its model folder
    let f2 = d.path().join("w2.ndtrain");
    sample::build(&f2, 2, None).unwrap();
    p.import_bundle(&open_bundle(&f2).unwrap()).unwrap();
    assert_eq!(retrain::info(&p).unwrap()["last_run"]["method"], "full");
    let mut m = AppModel::load();
    m.open_project(p.folder.to_str().unwrap()).unwrap();
    let warm = m.retrain_job(true).unwrap();
    assert!(warm.job["init_model"].as_str().unwrap().ends_with("runs/full1/model") && warm.job.get("init_adapter").is_none());
    assert_eq!((warm.method, warm.hyper.lr, warm.job["method"].as_str()), (Method::Full, 1e-5, Some("full")));
    let cold = m.retrain_job(false).unwrap();
    assert!(cold.job.get("init_model").is_none() && cold.job.get("init_adapter").is_none());
}
