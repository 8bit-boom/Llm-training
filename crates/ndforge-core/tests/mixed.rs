//! Mixed-precision sizes (export/runs.rs `MIXED_QUANTS`, export/mixed.rs): the list of sizes and their labels, the options and their old forms, what the export sends the worker
//! (the fit block and every refusal), what is recorded for each mixed size, the plan, and fake-worker runs (unix only).
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::export::imatrix_lib as lib;
use ndforge_core::export::mixed;
use ndforge_core::export::runs as exports;
use ndforge_core::model::AppModel;
use ndforge_core::project::Project;
use ndforge_core::sample;
use ndforge_core::train::plan::{base_hyper, Method, Preset};
use ndforge_core::train::runs as truns;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn project_with_world(root: &Path) -> Project {
    let mut p = Project::create(&root.join("projects"), "Ashfall Reach").unwrap();
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
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    let job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("summary.json"), json!({"steps": 40, "final_loss": 0.8}).to_string()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "model_id": m.id, "method": "lora"}).to_string()).unwrap();
}

fn q(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn kill() -> AtomicBool {
    AtomicBool::new(false)
}

// ------------------------------------------------------------------------------------------------------------------ the sizes and their names

#[test]
fn the_mixed_sizes_are_five_more_and_always_labelled_as_our_own_recipe() {
    assert_eq!(exports::QUANTS.len(), 5, "the plain sizes are as before");
    let ids: Vec<&str> = exports::MIXED_QUANTS.iter().map(|(i, _)| *i).collect();
    assert_eq!(ids, ["mixed_4", "mixed_3", "mixed_2", "mixed_fit", "mixed_measured"]);
    for (id, name) in exports::MIXED_QUANTS {
        assert!(name.contains("Mixed precision (our own recipe)"), "{id}");
        assert!(!name.contains("Dynamic") && !name.contains("UD-") && !name.to_lowercase().contains("unsloth"), "{id}: {name}");
        assert!(exports::is_mixed(id) && exports::imatrix_helps(id), "a matrix helps the mixed sizes too");
    }
    assert_eq!(exports::MIXED_RECIPE, "Mixed precision (our own recipe)");
    assert!(!exports::is_mixed("q4_k_m") && !exports::is_mixed("mixed_9") && !exports::is_mixed(""));
    let l = |id: &str| exports::MIXED_QUANTS.iter().find(|(i, _)| *i == id).map(|(i, n)| exports::quant_label(i, n)).unwrap();
    assert_eq!((l("mixed_4").as_str(), l("mixed_3").as_str(), l("mixed_2").as_str(), l("mixed_fit").as_str(), l("mixed_measured").as_str()),
               ("Mixed 4-bit (balanced)", "Mixed 3-bit (small)", "Mixed 2-bit (smallest: short answers only)", "Mixed: fit to a size (experimental)", "Mixed: measured fit to a size"));
    assert_eq!(exports::quant_label("q4_k_m", "Q4_K_M: smallest, good quality (recommended)"), "Q4_K_M");
}

#[test]
fn the_options_have_plain_defaults_and_old_calls_load() {
    assert_eq!(exports::MixedOptions::default(), exports::MixedOptions { fit: None, measured: None });
    let o: exports::MixedOptions = serde_json::from_value(json!({})).unwrap();
    assert!(o.fit.is_none());
    let o: exports::MixedOptions = serde_json::from_value(json!({"fit": {"target_gb": 12.5}, "later": 1})).unwrap();
    assert_eq!((o.fit.as_ref().unwrap().target_gb, o.fit.unwrap().matrix), (12.5, None));
    let o: exports::MixedOptions = serde_json::from_value(json!({"fit": {"matrix": "m1"}})).unwrap();
    assert_eq!(o.fit.unwrap().target_gb, 0.0);
}

#[test]
fn the_page_state_lists_the_mixed_sizes_with_their_labels_and_the_files_to_read_shapes_from() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project_with_world(d.path());
    fake_done_run(&p, "run1");
    let dir = p.folder.join("exports/run1-q4_k_m");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model-q4_k_m.gguf"), b"x").unwrap();
    std::fs::write(dir.join("meta.json"), json!({"name": "run1-q4_k_m", "run": "run1", "quant": "q4_k_m", "gguf_file": "model-q4_k_m.gguf"}).to_string()).unwrap();
    let mut m = AppModel::load();
    m.open_project(p.folder.to_str().unwrap()).unwrap();
    let st = m.export_state().unwrap();
    let quants = st["quants"].as_array().unwrap();
    assert_eq!(quants.len(), 5, "the plain sizes are as before");
    assert!(quants.iter().all(|x| x["mixed"].is_null()));
    assert_eq!(quants.iter().find(|x| x["id"] == "q4_k_m").unwrap()["label"], "Q4_K_M");
    let mixed_q = st["mixed_quants"].as_array().unwrap();
    assert_eq!(mixed_q.iter().map(|x| x["id"].as_str().unwrap()).collect::<Vec<_>>(), ["mixed_4", "mixed_3", "mixed_2", "mixed_fit", "mixed_measured"]);
    let m2 = mixed_q.iter().find(|x| x["id"] == "mixed_2").unwrap();
    assert_eq!((m2["label"].as_str(), m2["mixed"].clone()), (Some("Mixed 2-bit (smallest: short answers only)"), json!(true)));
    assert!(mixed_q.iter().all(|x| x["name"].as_str().unwrap().contains("(our own recipe)")));
    assert_eq!(st["mixed"]["recipe"], "Mixed precision (our own recipe)");
    let files = st["mixed"]["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);
    assert!(files[0]["path"].as_str().unwrap().ends_with("model-q4_k_m.gguf") && files[0]["run"] == "run1" && files[0]["label"].as_str().unwrap().starts_with("run1-q4_k_m: model-q4_k_m.gguf"));
}

// ------------------------------------------------------------------------------------------------------------------ what the export sends, refuses and records

#[cfg(unix)]
fn fake_worker(root: &Path, body: &str) {
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
const MIXED_WORKER: &str = r#"cp "$5" "$out/job.json"
qs=$(sed -n 's/.*"quants":\[\([^]]*\)\].*/\1/p' "$5" | tr -d '"' | tr ',' ' ')
map=""
for q in $qs; do printf "GGUF-$q" > "$out/model-$q.gguf"; map="$map\"$q\":\"$out/model-$q.gguf\","; done
echo "{\"type\":\"result\",\"data\":{\"ggufs\":{${map%,}},\"via\":\"llama.cpp\",\"merged_dir\":null,\"imatrix\":null,\"mixed_notes\":[\"The mixed-precision sizes need llama-quantize's per-tensor types, which Unsloth's export cannot do, so this export was made with llama.cpp (convert, then llama-quantize) for every size.\"],\"mixed\":{\"mixed_4\":{\"id\":\"mixed_4\",\"requested\":\"mixed_4\",\"label\":\"Mixed 4-bit (balanced)\",\"recipe\":\"Mixed precision (our own recipe)\",\"base\":\"Q4_K_M\",\"embd\":\"Q5_K\",\"output\":\"Q6_K\",\"rules\":[\"blk.x=Q6_K\"],\"estimate\":{\"gb\":16.1,\"bpw\":5.1},\"notes\":[],\"fit\":null,\"matrix_used\":false}}}}""#;

#[cfg(unix)]
#[test]
fn a_mixed_export_records_the_recipe_with_the_size_and_labels_the_card_as_our_own() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MIXED_WORKER);
    let p = project_with_world(d.path());
    fake_done_run(&p, "run1");
    let metas = exports::execute_many(&p, "run1", &q(&["mixed_4", "q4_k_m"]), 8192, "gguf", &kill(), &mut |_| {}).unwrap();
    let (mixed, plain) = (&metas[0], &metas[1]);
    assert_eq!((mixed["quant"].as_str(), mixed["gguf_file"].as_str(), mixed["via"].as_str()), (Some("mixed_4"), Some("model-mixed_4.gguf"), Some("llama.cpp")));
    let m = &mixed["mixed"];
    assert_eq!((m["id"].as_str(), m["label"].as_str(), m["recipe"].as_str(), m["base"].as_str(), m["rules"][0].as_str()), (Some("mixed_4"), Some("Mixed 4-bit (balanced)"), Some("Mixed precision (our own recipe)"), Some("Q4_K_M"), Some("blk.x=Q6_K")));
    assert!(m["note"].as_str().unwrap().contains("per-tensor types, which Unsloth's export cannot do") && m["estimate"]["gb"] == 16.1);
    assert!(plain["mixed"].is_null(), "a plain size has no recipe");
    // the same on disk, and the card says whose recipe it is
    let on_disk = &exports::list_exports(&p).into_iter().find(|e| e["quant"] == "mixed_4").unwrap();
    assert_eq!(on_disk["mixed"]["label"], "Mixed 4-bit (balanced)");
    let card = std::fs::read_to_string(p.folder.join("exports/run1-mixed_4/MODEL_CARD.md")).unwrap();
    assert!(card.contains("Mixed 4-bit (balanced) (Mixed precision (our own recipe))") && !card.contains("Dynamic") && !card.contains("UD-"), "{card}");
    assert!(std::fs::read_to_string(p.folder.join("exports/run1-q4_k_m/MODEL_CARD.md")).unwrap().contains("q4_k_m"));
    // the worker got the ids as the sizes, and no fit block
    assert!(!p.folder.join("exports/run1-gguf-work").exists());
}

#[cfg(unix)]
#[test]
fn fit_to_a_size_sends_the_target_and_the_matrix_file_and_is_refused_without_a_target() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MIXED_WORKER);
    let p = project_with_world(d.path());
    fake_done_run(&p, "run1");
    let e = |o: &exports::MixedOptions| exports::execute_many_full(&p, "run1", &q(&["mixed_fit"]), 8192, "gguf", &exports::ImatrixOptions::default(), o, &kill(), &mut |_| {}).unwrap_err().0;
    assert!(e(&exports::MixedOptions::default()).contains("Give the size in GB"));
    assert!(e(&exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: 0.0, matrix: None }), ..Default::default() }).contains("Give the size in GB"));
    assert!(e(&exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: f64::NAN, matrix: None }), ..Default::default() }).contains("Give the size in GB"));
    assert!(e(&exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: -3.0, matrix: None }), ..Default::default() }).contains("Give the size in GB"));
    assert!(e(&exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: 12.0, matrix: Some("gone".into()) }), ..Default::default() }).contains("no longer exists"));
    assert!(e(&exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: 12.0, matrix: Some("../x".into()) }), ..Default::default() }).contains("not the name of a saved matrix"));
    // with a saved matrix: its file goes to the worker
    let id = "20261101-1000-run1";
    let dir = lib::lib_dir(&p).join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("imatrix.gguf"), b"M").unwrap();
    let metas = exports::execute_many_full(&p, "run1", &q(&["mixed_fit"]), 8192, "gguf", &exports::ImatrixOptions::default(), &exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: 12.5, matrix: Some(id.into()) }), ..Default::default() }, &kill(), &mut |_| {}).unwrap();
    assert_eq!(metas[0]["quant"], "mixed_fit");
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/run1-mixed_fit/job.json")).unwrap()).unwrap();
    assert_eq!(job["mixed_fit"]["target_gb"], 12.5);
    assert!(job["mixed_fit"]["matrix"].as_str().unwrap().ends_with("imatrix.gguf"));
    assert_eq!(job["quants"], json!(["mixed_fit"]));
    // without a matrix chosen: null (the worker uses the export's own matrix, or falls back to a recipe and says so)
    exports::execute_many_full(&p, "run1", &q(&["mixed_fit"]), 8192, "gguf", &exports::ImatrixOptions::default(), &exports::MixedOptions { fit: Some(exports::MixedFit { target_gb: 12.5, matrix: None }), ..Default::default() }, &kill(), &mut |_| {}).unwrap();
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/run1-mixed_fit/job.json")).unwrap()).unwrap();
    assert!(job["mixed_fit"]["matrix"].is_null());
    // a plain export has no fit block at all, and unknown sizes are still refused
    exports::execute_many(&p, "run1", &q(&["q4_k_m"]), 8192, "gguf", &kill(), &mut |_| {}).unwrap();
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/run1-q4_k_m/job.json")).unwrap()).unwrap();
    assert!(job.get("mixed_fit").is_none());
    assert!(exports::execute_many(&p, "run1", &q(&["mixed_9"]), 8192, "gguf", &kill(), &mut |_| {}).unwrap_err().0.contains("Unknown file size"));
}

// ------------------------------------------------------------------------------------------------------------------ the plan

#[cfg(unix)]
#[test]
fn the_plan_sends_the_file_target_and_matrix_and_returns_the_workers_answer() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"gguf=$(sed -n 's/.*"gguf":"\([^"]*\)".*/\1/p' "$5")
cp "$5" "$gguf.job"
echo '{"type":"result","data":{"arch":"qwen35","recipe":"Mixed precision (our own recipe)","presets":[{"id":"mixed_4","estimate":{"gb":16.1}}],"fit":null}}'"#);
    let p = project_with_world(d.path());
    let gguf = d.path().join("m.gguf");
    std::fs::write(&gguf, b"x").unwrap();
    let r = mixed::plan(&p, &format!("  {}  ", gguf.display()), None, None).unwrap();
    assert_eq!((r["arch"].as_str(), r["presets"][0]["id"].as_str()), (Some("qwen35"), Some("mixed_4")));
    let sent: Value = serde_json::from_slice(&std::fs::read(format!("{}.job", gguf.display())).unwrap()).unwrap();
    assert_eq!((sent["job"].as_str(), sent["gguf"].as_str().map(Path::new), sent["target_gb"].clone(), sent["matrix"].clone()), (Some("mixed_plan"), Some(gguf.as_path()), Value::Null, Value::Null));
    let id = "20261101-1000-run1";
    let dir = lib::lib_dir(&p).join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("imatrix.gguf"), b"M").unwrap();
    mixed::plan(&p, &gguf.display().to_string(), Some(14.0), Some(id)).unwrap();
    let sent: Value = serde_json::from_slice(&std::fs::read(format!("{}.job", gguf.display())).unwrap()).unwrap();
    assert_eq!(sent["target_gb"], 14.0);
    assert!(sent["matrix"].as_str().unwrap().ends_with("imatrix.gguf"));
    mixed::plan(&p, &gguf.display().to_string(), Some(0.0), Some("")).unwrap();
    let sent: Value = serde_json::from_slice(&std::fs::read(format!("{}.job", gguf.display())).unwrap()).unwrap();
    assert!(sent["target_gb"].is_null() && sent["matrix"].is_null(), "no target: no fit");
}

#[test]
fn the_plan_refuses_in_plain_words() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project_with_world(d.path());
    assert!(mixed::plan(&p, "x.gguf", None, None).unwrap_err().0.contains("Training setup"));
    #[cfg(unix)]
    {
        fake_worker(d.path(), "echo '{\"type\":\"result\",\"data\":{}}'");
        assert!(mixed::plan(&p, "  ", None, None).unwrap_err().0.contains("Choose a GGUF file of the model"));
        assert!(mixed::plan(&p, "/nowhere/m.gguf", None, None).unwrap_err().0.contains("was not found"));
        let gguf = d.path().join("m.gguf");
        std::fs::write(&gguf, b"x").unwrap();
        assert!(mixed::plan(&p, &gguf.display().to_string(), Some(10.0), Some("gone")).unwrap_err().0.contains("no longer exists"));
        fake_worker(d.path(), r#"echo '{"type":"error","code":"mixed_failed","message":"The GGUF file could not be read (not a GGUF file). Choose a complete GGUF file of this model (any size: only its tensor shapes are read)."}'; exit 1"#);
        assert!(mixed::plan(&p, &gguf.display().to_string(), None, None).unwrap_err().0.contains("Choose a complete GGUF file of this model"));
    }
}

// ------------------------------------------------------------------------------------------------------------------ "Mixed: measured fit to a size"

#[test]
fn the_measured_options_have_plain_defaults_old_specs_load_and_bad_settings_are_refused_in_words() {
    let o: exports::MixedOptions = serde_json::from_value(json!({"fit": {"target_gb": 9.0}})).unwrap();
    assert!(o.measured.is_none() && o.fit.is_some(), "a spec from before the measured fit has no `measured`");
    let o: exports::MixedOptions = serde_json::from_value(json!({"measured": {"target_gb": 10.6, "later": 1}})).unwrap();
    let m = o.measured.unwrap();
    assert_eq!((m.target_gb, m.matrix.clone(), m.floor.clone(), m.ceiling.clone(), m.sample_rows, m.engine.clone()), (10.6, None, None, None, 0, None));
    m.check().unwrap();
    let bad = |f: &dyn Fn(&mut exports::MixedMeasured)| -> String {
        let mut m = exports::MixedMeasured { target_gb: 10.0, ..Default::default() };
        f(&mut m);
        m.check().unwrap_err().0
    };
    assert!(bad(&|m| m.target_gb = 0.0).contains("Give the size in GB"));
    assert!(bad(&|m| m.target_gb = f64::NAN).contains("Give the size in GB"));
    assert!(bad(&|m| m.target_gb = -1.0).contains("Give the size in GB"));
    assert!(bad(&|m| m.floor = Some("IQ2_S".into())).contains("not one of the types"));
    assert!(bad(&|m| { m.floor = Some("Q6_K".into()); m.ceiling = Some("Q4_K".into()); }).contains("lowest type is above the highest"));
    assert!(bad(&|m| m.engine = Some("magic".into())).contains("Unknown way of measuring"));
    assert!(bad(&|m| m.sample_rows = 3).contains("between 8 and 4096"));
    // good ones
    for f in [&(|m: &mut exports::MixedMeasured| { m.floor = Some("Q2_K".into()); m.ceiling = Some("Q6_K".into()); }) as &dyn Fn(&mut exports::MixedMeasured), &|m| m.sample_rows = 512, &|m| m.engine = Some("reference".into())] {
        let mut m = exports::MixedMeasured { target_gb: 10.0, ..Default::default() };
        f(&mut m);
        m.check().unwrap();
    }
}

#[test]
fn the_job_block_has_the_defaults_and_the_cache_folder() {
    let dir = tempdir().unwrap();
    let cache = dir.path().join("quantfit");
    let b = exports::MixedMeasured { target_gb: 10.6, ..Default::default() }.job_block(None, &cache);
    assert_eq!((b["target_gb"].clone(), b["floor"].clone(), b["ceiling"].clone(), b["sample_rows"].clone(), b["engine"].clone()), (json!(10.6), json!("Q3_K"), json!("Q8_0"), json!(128), json!("auto")));
    assert!(b["matrix"].is_null() && Path::new(b["cache_dir"].as_str().unwrap()) == cache.as_path());
    let b = exports::MixedMeasured { target_gb: 9.0, floor: Some("Q2_K".into()), ceiling: Some("Q6_K".into()), sample_rows: 256, engine: Some("llama".into()), matrix: None }.job_block(Some("m.gguf".into()), &cache);
    assert_eq!((b["floor"].clone(), b["ceiling"].clone(), b["sample_rows"].clone(), b["engine"].clone(), b["matrix"].clone()), (json!("Q2_K"), json!("Q6_K"), json!(256), json!("llama"), json!("m.gguf")));
}

#[test]
fn the_export_disk_need_adds_room_for_the_sample_file_when_measured() {
    let plain = exports::export_need_gb(17.4, false, &q(&["q4_k_m"]), false);
    let measured = exports::export_need_gb(17.4, false, &q(&["q4_k_m", "mixed_measured"]), false);
    let mixed_only = exports::export_need_gb(17.4, false, &q(&["q4_k_m", "mixed_fit"]), false);
    assert!((measured - mixed_only - 2.0).abs() < 1e-9 && measured > plain);
}

#[cfg(unix)]
#[test]
fn a_measured_export_sends_its_block_and_is_refused_without_a_target() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MIXED_WORKER);
    let p = project_with_world(d.path());
    fake_done_run(&p, "run1");
    let e = |o: &exports::MixedOptions| exports::execute_many_full(&p, "run1", &q(&["mixed_measured"]), 8192, "gguf", &exports::ImatrixOptions::default(), o, &kill(), &mut |_| {}).unwrap_err().0;
    assert!(e(&exports::MixedOptions::default()).contains("Give the size in GB"));
    let with = |m: exports::MixedMeasured| exports::MixedOptions { measured: Some(m), ..Default::default() };
    assert!(e(&with(exports::MixedMeasured::default())).contains("Give the size in GB"));
    assert!(e(&with(exports::MixedMeasured { target_gb: 12.0, floor: Some("Q9_K".into()), ..Default::default() })).contains("not one of the types"));
    assert!(e(&with(exports::MixedMeasured { target_gb: 12.0, matrix: Some("gone".into()), ..Default::default() })).contains("no longer exists"));
    let id = "20261101-1000-run1";
    let dir = lib::lib_dir(&p).join(id);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("imatrix.gguf"), b"M").unwrap();
    let metas = exports::execute_many_full(&p, "run1", &q(&["mixed_measured"]), 8192, "gguf", &exports::ImatrixOptions::default(),
                                           &with(exports::MixedMeasured { target_gb: 10.6, matrix: Some(id.into()), sample_rows: 64, ..Default::default() }), &kill(), &mut |_| {}).unwrap();
    assert_eq!(metas[0]["quant"], "mixed_measured");
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/run1-mixed_measured/job.json")).unwrap()).unwrap();
    let b = &job["mixed_measured"];
    assert_eq!((b["target_gb"].clone(), b["floor"].clone(), b["ceiling"].clone(), b["sample_rows"].clone(), b["engine"].clone()), (json!(10.6), json!("Q3_K"), json!("Q8_0"), json!(64), json!("auto")));
    assert!(b["matrix"].as_str().unwrap().ends_with("imatrix.gguf") && Path::new(b["cache_dir"].as_str().unwrap()) == p.folder.join("quantfit").as_path());
    assert!(job.get("mixed_fit").is_none() && job["quants"] == json!(["mixed_measured"]));
    // no matrix chosen: null (the worker uses the export's own matrix)
    exports::execute_many_full(&p, "run1", &q(&["mixed_measured"]), 8192, "gguf", &exports::ImatrixOptions::default(), &with(exports::MixedMeasured { target_gb: 10.6, ..Default::default() }), &kill(), &mut |_| {}).unwrap();
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/run1-mixed_measured/job.json")).unwrap()).unwrap();
    assert!(job["mixed_measured"]["matrix"].is_null());
    // a plain export has no block
    exports::execute_many(&p, "run1", &q(&["q4_k_m"]), 8192, "gguf", &kill(), &mut |_| {}).unwrap();
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/run1-q4_k_m/job.json")).unwrap()).unwrap();
    assert!(job.get("mixed_measured").is_none());
}

#[cfg(unix)]
#[test]
fn the_model_folder_export_accepts_the_mixed_sizes_and_sends_the_block() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // the worker keeps a copy of its job above the work folder, which is removed at the end
    fake_worker(d.path(), &format!("cp \"$5\" \"$out/../captured-job.json\"\n{}", MIXED_WORKER.replacen("cp \"$5\" \"$out/job.json\"\n", "", 1)));
    let p = project_with_world(d.path());
    let model = d.path().join("cut");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    let opts = exports::MixedOptions { measured: Some(exports::MixedMeasured { target_gb: 10.6, engine: Some("reference".into()), ..Default::default() }), ..Default::default() };
    let imatrix = exports::ImatrixOptions::default();
    let metas = exports::execute_model_folder_full(&p, "cut1", &model, "qwen3_5", &q(&["q4_k_m", "mixed_measured"]), 8192, false, &imatrix, &opts, &kill(), &mut |_| {}).unwrap();
    assert_eq!(metas.iter().map(|m| m["quant"].as_str().unwrap()).collect::<Vec<_>>(), ["q4_k_m", "mixed_measured"]);
    assert!(metas[0]["mixed"].is_null() && metas[1]["mixed"]["recipe"] == "Mixed precision (our own recipe)" && metas[1]["mixed"]["label"] == "Mixed: measured fit to a size");
    let job: Value = serde_json::from_slice(&std::fs::read(p.folder.join("exports/captured-job.json")).unwrap()).unwrap();
    assert_eq!((job["mixed_measured"]["target_gb"].clone(), job["mixed_measured"]["engine"].clone()), (json!(10.6), json!("reference")));
    assert_eq!(job["quants"], json!(["q4_k_m", "mixed_measured"]));
    // the old function refuses the mixed sizes' options the same way (no target) and still knows only what it knew
    let err = exports::execute_model_folder(&p, "cut2", &model, "qwen3_5", &q(&["mixed_measured"]), 8192, false, &imatrix, &kill(), &mut |_| {}).unwrap_err().0;
    assert!(err.contains("Give the size in GB"));
    assert!(exports::execute_model_folder(&p, "cut2", &model, "qwen3_5", &q(&["mixed_9"]), 8192, false, &imatrix, &kill(), &mut |_| {}).unwrap_err().0.contains("Unknown file size"));
}
