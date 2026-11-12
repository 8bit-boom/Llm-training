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
    assert!(warm.job["init_adapter"].as_str().unwrap().replace('\\', "/").ends_with("runs/run1/adapter"));
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
    assert_eq!((st["suggested_name"].as_str(), st["servers"][0]["url"].as_str(), st["quants"].as_array().unwrap().len()), (Some("ashfall-reach-gm"), Some("http://localhost:11434"), 5));
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
    assert!(warm.job["init_model"].as_str().unwrap().replace('\\', "/").ends_with("runs/full1/model") && warm.job.get("init_adapter").is_none());
    assert_eq!((warm.method, warm.hyper.lr, warm.job["method"].as_str()), (Method::Full, 1e-5, Some("full")));
    let cold = m.retrain_job(false).unwrap();
    assert!(cold.job.get("init_model").is_none() && cold.job.get("init_adapter").is_none());
}

#[cfg(unix)]
#[test]
fn an_export_can_keep_the_merged_model_or_make_only_that_and_the_lora_can_be_copied() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // the fake worker makes what the job asks for: a GGUF when "gguf":true, the merged folder when it is kept (or no GGUF was asked for)
    fake_worker(d.path(), r#"cp "$5" "$out/job.json"
if grep -q '"gguf":true' "$5"; then printf 'GGUF' > "$out/model-$quant.gguf"; g="\"$out/model-$quant.gguf\""; else g=null; fi
if grep -q '"keep_merged":true' "$5"; then mkdir -p "$out/merged"; echo '{}' > "$out/merged/config.json"; m="\"$out/merged\""; else m=null; fi
echo "{\"type\":\"result\",\"data\":{\"gguf\":$g,\"merged_dir\":$m,\"quant\":\"$quant\"}}""#);
    let p = project_with_world(d.path(), 1);
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    fake_done_run(&p, "run1", 1);
    std::fs::write(truns::runs_dir(&p).join("run1/adapter/adapter_model.safetensors"), "W").unwrap();
    std::fs::create_dir_all(truns::runs_dir(&p).join("run1/adapter/sub")).unwrap();
    std::fs::write(truns::runs_dir(&p).join("run1/adapter/sub/tokenizer.json"), "T").unwrap();
    let no = |what: &str| exports::execute_what(&p, "run1", "q4_k_m", 8192, what, &AtomicBool::new(false), &mut |_| {});

    // the default: a GGUF and no merged model
    let m = no("gguf").unwrap();
    assert!(m["gguf_file"].is_string() && m["merged_dir"].is_null());
    // GGUF and the merged model
    let m = no("gguf_merged").unwrap();
    assert!(m["gguf_file"].is_string() && m["merged_dir"].is_string());
    assert!(std::fs::read_to_string(p.folder.join("exports/run1-q4_k_m/job.json")).unwrap().contains("\"keep_merged\":true"));
    // only the merged model: its own folder, no Modelfile, listed, and Ollama refuses it in plain words
    let m = no("merged").unwrap();
    assert!(m["gguf_file"].is_null() && m["merged_dir"].is_string() && m["name"] == "run1-merged");
    let dir = p.folder.join("exports/run1-merged");
    assert!(dir.join("meta.json").is_file() && !dir.join("Modelfile").exists());
    let list = exports::list_exports(&p);
    let merged = list.iter().find(|e| e["name"] == "run1-merged").unwrap();
    assert_eq!((merged["gguf_exists"].clone(), merged["merged_exists"].clone(), merged["size_mb"].is_null()), (json!(false), json!(true), true));
    assert!(exports::find_export(&p, "run1-merged").unwrap_err().0.contains("only the merged model"));
    assert!(no("everything").unwrap_err().0.contains("Unknown export kind"));

    // the LoRA add-on is copied with its tokenizer and a README that names the base model and says how to load and merge it
    let dest = d.path().join("elsewhere");
    std::fs::create_dir_all(&dest).unwrap();
    let to = exports::copy_adapter(&p, "run1", &dest).unwrap();
    assert!(to.ends_with("run1-lora") && to.join("adapter_config.json").is_file() && to.join("adapter_model.safetensors").is_file() && to.join("sub/tokenizer.json").is_file());
    let readme = std::fs::read_to_string(to.join("README.md")).unwrap();
    assert!(readme.contains("unsloth/Qwen3.5-4B") && readme.contains("FastModel") && readme.contains("PeftModel") && readme.contains("merge_and_unload") && readme.contains("GM secrets"), "{readme}");
    assert!(exports::copy_adapter(&p, "run1", &dest).unwrap_err().0.contains("already exists"));
    assert!(exports::copy_adapter(&p, "../x", &dest).is_err() && exports::copy_adapter(&p, "nope", &dest).unwrap_err().0.contains("no saved LoRA"));
    // a full fine-tune has no separate add-on
    let full = truns::runs_dir(&p).join("full1");
    std::fs::create_dir_all(full.join("model")).unwrap();
    std::fs::write(full.join("model/config.json"), "{}").unwrap();
    assert!(exports::copy_adapter(&p, "full1", &dest).unwrap_err().0.contains("full fine-tune"));
}

#[cfg(unix)]
#[test]
fn several_sizes_come_from_one_job_and_each_becomes_its_own_export() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // the fake worker makes one file per wanted size (and counts how often it was started), plus the merged folder when it is kept
    fake_worker(d.path(), r#"echo x >> "$out/../starts.txt"
qs=$(sed -n 's/.*"quants":\[\([^]]*\)\].*/\1/p' "$5" | tr -d '"' | tr ',' ' ')
map=""
for q in $qs; do printf "GGUF-$q" > "$out/model-$q.gguf"; map="$map\"$q\":\"$out/model-$q.gguf\","; done
if grep -q '"keep_merged":true' "$5"; then mkdir -p "$out/merged"; echo '{}' > "$out/merged/config.json"; m="\"$out/merged\""; else m=null; fi
printf MMPROJ > "$out/mmproj-F16.gguf"
echo "{\"type\":\"result\",\"data\":{\"ggufs\":{${map%,}},\"merged_dir\":$m,\"mmproj\":\"$out/mmproj-F16.gguf\",\"modalities\":[\"image\"],\"vision_note\":\"taken from the official repo\"}}""#);
    let p = project_with_world(d.path(), 1);
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    fake_done_run(&p, "run1", 1);
    let q = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let metas = exports::execute_many(&p, "run1", &q(&["q8_0", "q6_k", "q4_k_m", "q6_k"]), 8192, "gguf_merged", &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(metas.iter().map(|m| m["name"].as_str().unwrap()).collect::<Vec<_>>(), ["run1-q8_0", "run1-q6_k", "run1-q4_k_m"], "in the order asked, each once");
    assert_eq!(std::fs::read_to_string(p.folder.join("exports/starts.txt")).unwrap().lines().count(), 1, "one worker job for all sizes");
    for (name, quant) in [("run1-q8_0", "q8_0"), ("run1-q6_k", "q6_k"), ("run1-q4_k_m", "q4_k_m")] {
        let dir = p.folder.join("exports").join(name);
        assert_eq!(std::fs::read_to_string(dir.join(format!("model-{quant}.gguf"))).unwrap(), format!("GGUF-{quant}"));
        assert!(std::fs::read_to_string(dir.join("Modelfile")).unwrap().starts_with(&format!("FROM ./model-{quant}.gguf")), "{name}");
        assert!(dir.join("MODEL_CARD.md").is_file() && dir.join("meta.json").is_file());
    }
    // every size gets the base model's vision file and a note on how to use it (one move, the others copies); Ollama is told it is text only
    for name in ["run1-q8_0", "run1-q6_k", "run1-q4_k_m"] {
        let dir = p.folder.join("exports").join(name);
        assert_eq!(std::fs::read_to_string(dir.join("mmproj-F16.gguf")).unwrap(), "MMPROJ", "{name}");
        let v = std::fs::read_to_string(dir.join("VISION.md")).unwrap();
        assert!(v.contains("--mmproj mmproj-F16.gguf") && v.contains("Ollama (text only)") && v.contains("Unsloth Studio"), "{v}");
    }
    assert!(metas.iter().all(|m| m["mmproj_file"] == "mmproj-F16.gguf" && m["modalities"] == json!(["image"])));
    assert!(exports::list_exports(&p).iter().all(|e| e["has_vision"] == true));
    // the merged model is kept once, in the first export; the work folder is gone
    assert!(p.folder.join("exports/run1-q8_0/merged/config.json").is_file() && !p.folder.join("exports/run1-q6_k/merged").exists());
    assert!(metas[0]["merged_dir"].as_str().unwrap().ends_with("run1-q8_0/merged") && metas[1]["merged_dir"].is_null());
    assert!(!p.folder.join("exports/run1-gguf-work").exists());
    let list = exports::list_exports(&p);
    assert_eq!(list.len(), 3);
    assert!(list.iter().all(|e| e["gguf_exists"] == true));
    assert!(exports::find_export(&p, "run1-q6_k").is_ok());
    // one size still goes straight into its export, and q6_k is a known size
    let one = exports::execute_what(&p, "run1", "q6_k", 8192, "gguf", &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(one["name"], "run1-q6_k");
    assert!(exports::QUANTS.iter().any(|(id, _)| *id == "q6_k"));
    assert!(exports::execute_many(&p, "run1", &[], 8192, "gguf", &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("at least one"));
    assert!(exports::execute_many(&p, "run1", &q(&["q4_k_m", "q2"]), 8192, "gguf", &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Unknown file size"));
}
