//! Slim or grow a model: the arithmetic (`reshape::plan`), then the runs (`reshape::runs`) with a stand-in worker.
use ndforge_core::reshape::plan::*;
use serde_json::json;

/// A Qwen3.5-27B-like model: 64 layers in blocks of 4, 0.39 B parameters each, 2.84 B elsewhere (27.8 B in all)
fn big() -> Inspect {
    inspect(64, 4, 390_000_000, 2_840_000_000)
}

/// A Qwen3.5-9B-like model: 32 layers, 0.22 B each, 2.56 B elsewhere (9.6 B)
fn nine() -> Inspect {
    inspect(32, 4, 220_000_000, 2_560_000_000)
}

fn inspect(layers: u32, period: u32, per_layer: u64, other: u64) -> Inspect {
    Inspect {
        model_type: "qwen3_5".into(), supported: true, why: None, layers, layer_types: vec![], period, blocks: layers / period,
        layer_params: vec![per_layer; layers as usize], embed_params: other / 2, lm_head_params: 0, vision_params: other / 4, mtp_params: other / 8, other_params: other / 8,
        total_params: per_layer * layers as u64 + other, bytes_per_param: 2.0, has_vision: true, tied: true, source_bytes: 0, local: true,
    }
}

fn spec(mode: Mode, blocks: u32) -> ReshapeSpec {
    serde_json::from_value(json!({"source_model_id": "m", "mode": mode.key(), "blocks": blocks})).unwrap()
}

#[test]
fn slimming_the_27b_by_four_blocks_gives_48_layers_and_about_21_6_billion() {
    let ins = big();
    assert_eq!(ins.total_params, 27_800_000_000);
    let opts = options(&ins, Mode::Prune, Some(32.0));
    assert_eq!(opts.len(), 8, "at most half the blocks (16 blocks -> 8)");
    let four = opts.iter().find(|o| o.blocks == 4).unwrap();
    assert_eq!(four.layers, 48);
    assert!((four.params_b - 21.56).abs() < 0.01, "{four:?}");
    assert!((four.disk_gb - 43.1).abs() < 0.1, "{four:?}");
    // training: 16-bit LoRA does not fit a 32 GB card, QLoRA does
    assert_eq!((four.fits_lora, four.fits_qlora), (Some(false), Some(true)), "{four:?}");
    assert_eq!(opts[0].blocks, 1);
    assert_eq!(opts[0].layers, 60);
    assert_eq!(options(&ins, Mode::Prune, None)[0].fits_lora, None);
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 21.0), 4);
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 27.7), 1);
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 1.0), 8, "clamped to the limit");
}

#[test]
fn growing_the_9b_by_four_blocks_is_exact() {
    let ins = nine();
    let opts = options(&ins, Mode::Grow, Some(32.0));
    assert_eq!(opts.len(), 4);
    let four = opts.iter().find(|o| o.blocks == 4).unwrap();
    assert_eq!(four.layers, 48);
    // the layers 8..24 appear twice: 16 layers of 0.22 B
    assert_eq!(params_after(&ins, Mode::Grow, 4), ins.total_params + 16 * 220_000_000);
    assert!((four.params_b - 13.12).abs() < 0.001, "{four:?}");
    assert_eq!(blocks_for_target(&ins, Mode::Grow, 13.2), 4);
    assert_eq!(blocks_for_target(&ins, Mode::Grow, 9.7), 1);
    assert_eq!(params_after(&ins, Mode::Grow, 1), ins.total_params + 4 * 220_000_000);
}

#[test]
fn unsupported_or_too_small_models_have_no_options() {
    let mut ins = big();
    ins.supported = false;
    ins.why = Some("nope".into());
    assert!(options(&ins, Mode::Prune, None).is_empty());
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 20.0), 0);
    let small = inspect(8, 4, 1, 1);          // two blocks, both protected
    assert!(options(&small, Mode::Prune, None).is_empty());
    assert_eq!(options(&small, Mode::Grow, None).len(), 1);
}

#[test]
fn the_model_is_loaded_for_measuring_the_way_that_fits() {
    assert_eq!(scoring_load(27_800_000_000, 2.0, Some(32.0)), "4bit");
    assert_eq!(scoring_load(9_600_000_000, 2.0, Some(32.0)), "16bit");
    assert_eq!(scoring_load(27_800_000_000, 2.0, None), "4bit");
    assert_eq!(scoring_load(120_000_000_000, 2.0, Some(24.0)), "offload");
}

#[test]
fn limits_are_the_same_as_the_workers() {
    let table: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/reshape_limits.json")).unwrap()).unwrap();
    let cases = table.as_array().unwrap();
    assert!(cases.len() >= 8);
    for c in cases {
        let ins = Inspect { blocks: c["blocks"].as_u64().unwrap() as u32, supported: true, ..Default::default() };
        let mode = if c["mode"] == "prune" { Mode::Prune } else { Mode::Grow };
        let got = limits(&ins, mode, c["pf"].as_u64().unwrap() as u32, c["pl"].as_u64().unwrap() as u32);
        assert_eq!(got, (c["min"].as_u64().unwrap() as u32, c["max"].as_u64().unwrap() as u32), "{c}");
    }
}

#[test]
fn validate_refuses_in_plain_words() {
    let ins = big();
    assert!(validate(&spec(Mode::Prune, 4), &ins).is_ok());
    assert!(validate(&spec(Mode::Grow, 8), &ins).is_ok());
    // unsupported: the model's own reason
    let mut bad = big();
    bad.supported = false;
    bad.why = Some("Gemma 4 is not supported yet.".into());
    assert_eq!(validate(&spec(Mode::Prune, 1), &bad).unwrap_err().0, "Gemma 4 is not supported yet.");
    // too many / none
    assert!(validate(&spec(Mode::Prune, 9), &ins).unwrap_err().0.contains("at most half"));
    assert!(validate(&spec(Mode::Prune, 0), &ins).unwrap_err().0.contains("at most half"));
    assert!(validate(&spec(Mode::Grow, 9), &ins).unwrap_err().0.contains("Grow by at most half"));
    // unknown settings
    let mut s = spec(Mode::Prune, 2);
    s.strategy = "wild".into();
    assert!(validate(&s, &ins).is_err());
    let mut s = spec(Mode::Grow, 2);
    s.init = "random".into();
    assert!(validate(&s, &ins).is_err());
    let mut s = spec(Mode::Prune, 2);
    s.load = "8bit".into();
    assert!(validate(&s, &ins).is_err());
    let mut s = spec(Mode::Prune, 2);
    s.calib_samples = 1;
    assert!(validate(&s, &ins).is_err());
}

#[test]
fn explicit_layers_must_be_whole_unprotected_blocks() {
    let ins = big();
    let with = |d: Vec<u32>| ReshapeSpec { drop_layers: Some(d), ..spec(Mode::Prune, 1) };
    assert!(validate(&with((36..52).collect()), &ins).is_ok());                  // four whole blocks
    assert!(validate(&with(vec![36, 37, 38, 39]), &ins).is_ok());
    let e = validate(&with(vec![3, 4, 5, 6]), &ins).unwrap_err().0;
    assert!(e.contains("Layers 3 to 6 do not form whole blocks of 4"), "{e}");
    assert!(validate(&with(vec![0, 1, 2, 3]), &ins).unwrap_err().0.contains("protected"));
    assert!(validate(&with(vec![60, 61, 62, 63]), &ins).unwrap_err().0.contains("protected"));
    assert!(validate(&with(vec![64]), &ins).unwrap_err().0.contains("between 0 and 63"));
    assert!(validate(&with(vec![]), &ins).is_err());
    assert!(validate(&with((4..44).collect()), &ins).unwrap_err().0.contains("at most half"));
    let grow = ReshapeSpec { drop_layers: Some(vec![4, 5, 6, 7]), ..spec(Mode::Grow, 1) };
    assert!(validate(&grow, &ins).unwrap_err().0.contains("only for slimming"));
}

#[test]
fn warnings_are_honest() {
    let ins = big();
    let w = warnings(&spec(Mode::Prune, 4), &ins, Some(32.0));
    assert!(w.iter().any(|x| x.contains("worse than the original until it is trained again")), "{w:?}");
    assert!(w.iter().any(|x| x.contains("QLoRA") && x.contains("32 GB")), "{w:?}");
    assert!(!w.iter().any(|x| x.contains("quarter")), "4 of 16 blocks is exactly a quarter");
    assert!(warnings(&spec(Mode::Prune, 5), &ins, None).iter().any(|x| x.contains("More than a quarter")));
    let g = warnings(&spec(Mode::Grow, 4), &nine(), Some(32.0));
    assert!(g.iter().any(|x| x.contains("little or no gain") && x.contains("37 % more")), "{g:?}");
    let none = warnings(&spec(Mode::Prune, 4), &big(), Some(12.0));
    assert!(none.iter().any(|x| x.contains("Even QLoRA")), "{none:?}");
}

#[test]
fn disk_needed_counts_the_download_for_a_model_that_is_not_here() {
    let mut ins = big();
    let local = disk_needed_gb(&ins, Mode::Prune, 4);
    assert!((local - 47.4).abs() < 0.05, "{local}");        // 21.56 B * 2 bytes * 1.1
    ins.local = false;
    assert!((disk_needed_gb(&ins, Mode::Prune, 4) - (47.4 + 55.6)).abs() < 0.2);
    assert!(disk_needed_gb(&nine(), Mode::Grow, 4) > disk_needed_gb(&nine(), Mode::Grow, 1));
}

#[test]
fn the_inspect_result_of_the_worker_parses() {
    let v = json!({"model_type": "qwen3_5", "supported": true, "why": null, "layers": 8, "layer_types": ["linear_attention"], "period": 4, "blocks": 2, "layer_params": [1, 2],
        "embed_params": 3, "lm_head_params": 0, "vision_params": 4, "mtp_params": 5, "other_params": 6, "total_params": 21, "bytes_per_param": 2.0, "has_vision": true,
        "tied": true, "source_bytes": 42, "local": true, "extra_key_from_a_newer_worker": 1});
    let ins: Inspect = serde_json::from_value(v).unwrap();
    assert_eq!((ins.layers, ins.blocks, ins.total_params, ins.source_bytes), (8, 2, 21, 42));
    let s: ReshapeSpec = serde_json::from_value(json!({"source_model_id": "x", "mode": "grow", "blocks": 2})).unwrap();
    assert_eq!((s.strategy.as_str(), s.protect_first, s.protect_last, s.init.as_str(), s.calib_samples, s.calib_len, s.load.as_str()), ("contiguous", 1, 1, "identity", 128, 512, "auto"));
}

// ------------------------------------------------------------------------------------------------------------------ runs
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::project::Project;
use ndforge_core::reshape::runs as rruns;
use ndforge_core::sample;
use ndforge_core::train::catalog;
use ndforge_core::train::runs as truns;
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn project(root: &std::path::Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

fn project_with_version(root: &std::path::Path) -> Project {
    let mut p = project(root);
    let f = root.join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    p
}

/// What `reshape_inspect` answers for the fake worker: a 12-layer model in 3 blocks of 4
const INSPECT: &str = r#"{"model_type":"qwen3_5","supported":true,"why":null,"layers":12,"layer_types":[],"period":4,"blocks":3,"layer_params":[100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000],"embed_params":500000000,"lm_head_params":0,"vision_params":100000000,"mtp_params":0,"other_params":0,"total_params":1800000000,"bytes_per_param":2.0,"has_vision":true,"tied":true,"source_bytes":3600000000,"local":false}"#;

/// A fake Python: answers `reshape_inspect` itself, and for `reshape` runs `body` (shell) with $out, $final, $dir (the model's folder) and $job (the job file) set.
#[cfg(unix)]
fn fake_worker(root: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    let script = format!(r#"#!/bin/sh
job="$5"
if grep -q '"job":"reshape_inspect"' "$job"; then
  echo '{{"type":"result","data":{INSPECT}}}'
  exit 0
fi
out=$(sed -n 's/.*"out":"\([^"]*\)".*/\1/p' "$job")
final=$(sed -n 's/.*"final":"\([^"]*\)".*/\1/p' "$job")
dir=$(dirname "$out")
cp "$job" "$dir/job-seen.json"
{body}
"#);
    std::fs::write(&py, script).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

/// Writes a small model folder into $out, then renames it to $final and reports a result (what the real worker does)
#[cfg(unix)]
const MAKES_A_MODEL: &str = r#"mkdir -p "$out"
echo '{"type":"progress","step":"write","value":0.5,"text":"Writing the new model"}'
echo '{"type":"reshape_scores","boundary_distance":[0.1,0.05,0.2],"chosen_blocks":[1],"period":4,"stable":true,"matrix":[[0.0]]}'
echo '{"model_type":"qwen3_5","architectures":["Qwen3_5ForCausalLM"],"num_hidden_layers":8}' > "$out/config.json"
echo '{"chat_template":"{{ messages }}"}' > "$out/tokenizer_config.json"
printf -- '---\nlicense: apache-2.0\n---\n' > "$out/README.md"
echo '{"format":1,"origin":{"repo":"unsloth/Qwen3.5-4B","layers":12,"params":1800000000},"mode":"prune","layers":8,"params":1240000000,"zero_init":[]}' > "$out/ndforge_reshape.json"
head -c 2000 /dev/zero > "$out/model.safetensors"
mv "$out" "$final"
echo '{"type":"result","data":{"model_dir":"x","mode":"prune","layers_before":12,"layers_after":8,"src_layers":[0,1,2,3,8,9,10,11],"dropped":[4,5,6,7],"duplicated":[],"zero_init":[],"params_before":1800000000,"params_after":1240000000,"bytes_written":2480000000,"loss_before":1.5,"loss_after":2.0,"ppl_before":4.48,"ppl_after":7.39,"stable":true,"boundary_distance":[0.1,0.05,0.2],"chosen_blocks":[1],"load":"16bit","seconds":1.0,"notes":[],"stopped":false}}'"#;

fn drop_spec() -> ReshapeSpec {
    ReshapeSpec { drop_layers: Some(vec![4, 5, 6, 7]), ..spec(Mode::Prune, 1) }.with_model("qwen3.5-4b")
}

trait WithModel {
    fn with_model(self, id: &str) -> Self;
}

impl WithModel for ReshapeSpec {
    fn with_model(mut self, id: &str) -> Self {
        self.source_model_id = id.into();
        self
    }
}

#[cfg(unix)]
#[test]
fn a_model_is_made_registered_and_listed() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    let p = project(d.path());
    assert!(p.folder.join("models").is_dir(), "a new project has a models folder");
    let mut events = Vec::new();
    let result = rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
    let name = result["name"].as_str().unwrap().to_string();
    let id = result["model_id"].as_str().unwrap().to_string();
    assert!(name.ends_with("-qwen3.5-4b-slim-8L"), "{name}");
    assert_eq!(id, format!("custom-local-{name}"));
    let dir = p.folder.join("models").join(&name);
    assert!(dir.join("model/config.json").is_file() && !dir.join("model.partial").exists());
    let state: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!((state["status"].as_str(), state["model_id"].as_str(), state["source_model_id"].as_str()), (Some("done"), Some(id.as_str()), Some("qwen3.5-4b")));
    assert!(dir.join("result.json").is_file() && dir.join("scores.json").is_file());
    // the job the worker got
    let job: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("job-seen.json")).unwrap()).unwrap();
    assert_eq!((job["mode"].as_str(), job["drop_layers"].clone(), job["check"].clone(), job["source"]["repo"].as_str()), (Some("prune"), json!([4, 5, 6, 7]), json!(true), Some("unsloth/Qwen3.5-4B")));
    assert_eq!(job["final"].as_str().unwrap(), dir.join("model").to_string_lossy());
    assert!(job["out"].as_str().unwrap().ends_with("model.partial") && job["grow"]["init"] == "identity");
    // events reach the window
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["step"] == "write") && events.iter().any(|e| e["kind"] == "reshape_scores" && e["chosen_blocks"] == json!([1])));
    // the model list has it, with where it came from and its exact size
    let m = catalog::find(&id).expect("registered");
    assert_eq!(m.derived_from.as_deref(), Some("unsloth/Qwen3.5-4B"));
    assert_eq!(m.reshape.as_deref(), Some("slimmed: 12 → 8 layers"));
    assert!((m.params_b - 1.24).abs() < 1e-9, "{m:?}");
    assert_eq!((m.family.as_str(), m.custom, m.license.as_str()), ("qwen3_5", true, "apache-2.0"));
    assert!(m.name.starts_with("Qwen3.5 4B") && m.name.ends_with("slim 8 layers"), "{}", m.name);
    assert_eq!(m.repo, dir.join("model").canonicalize().unwrap().to_string_lossy());
    assert!(m.good_for.contains("unsloth/Qwen3.5-4B"));
    // registering again changes nothing
    assert_eq!(rruns::register(&p, &name).unwrap(), id);
    assert_eq!(catalog::custom_models().iter().filter(|x| x.id == id).count(), 1);
    // the list of made models
    let list = rruns::list(&p);
    assert_eq!(list.len(), 1);
    assert_eq!((list[0]["status"].as_str(), list[0]["tested"].clone(), list[0]["has_model"].clone()), (Some("done"), json!(false), json!(true)));
    assert_eq!((list[0]["summary"]["layers_after"].clone(), list[0]["summary"]["dropped"].clone(), list[0]["summary"]["loss_after"].clone()), (json!(8), json!([4, 5, 6, 7]), json!(2.0)));
    assert_eq!(list[0]["scores"]["chosen_blocks"], json!([1]));
    // what the model was found to be is remembered for the next time
    assert!(d.path().join("cfg/reshape_inspect.json").is_file());
}

#[cfg(unix)]
#[test]
fn a_failing_job_leaves_nothing_behind() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), "mkdir -p \"$out\"\necho half > \"$out/model.safetensors\"\necho '{\"type\":\"error\",\"code\":\"disk_full\",\"message\":\"There is not enough free space: 5 GB needed.\"}'\nexit 1");
    let p = project(d.path());
    let e = rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("not enough free space"), "{e}");
    let list = rruns::list(&p);
    assert_eq!((list.len(), list[0]["status"].as_str()), (1, Some("failed")));
    assert!(list[0]["error"].as_str().unwrap().contains("free space"));
    let dir = p.folder.join("models").join(list[0]["name"].as_str().unwrap());
    assert!(!dir.join("model.partial").exists() && !dir.join("model").exists());
    assert!(catalog::custom_models().is_empty());
}

#[cfg(unix)]
#[test]
fn stopping_asks_politely_and_then_kills() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // a worker that honours the STOP file
    fake_worker(d.path(), "mkdir -p \"$out\"\nwhile [ ! -f \"$dir/STOP\" ]; do sleep 0.05; done\nrm -rf \"$out\"\necho '{\"type\":\"result\",\"data\":{\"stopped\":true,\"model_dir\":null}}'");
    let p = project(d.path());
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(600));
        s2.store(true, Ordering::SeqCst);
    });
    let r = rruns::execute(&p, &drop_spec(), stop, &mut |_| {}).unwrap();
    t.join().unwrap();
    assert_eq!(r["stopped"], true);
    let list = rruns::list(&p);
    assert_eq!(list[0]["status"], "stopped");
    assert!(!p.folder.join("models").join(list[0]["name"].as_str().unwrap()).join("model.partial").exists());
    assert!(catalog::custom_models().is_empty());
    // a worker that ignores it is killed after the grace period
    fake_worker(d.path(), "mkdir -p \"$out\"\nexec sleep 30");
    let stop = Arc::new(AtomicBool::new(true));
    let e = rruns::execute_with_grace(&p, &drop_spec(), stop, Duration::from_millis(300), &mut |_| {}).unwrap_err();
    assert_eq!(e.0, "Stopped.");
    let list = rruns::list(&p);
    assert_eq!((list.len(), list[0]["status"].as_str()), (2, Some("stopped")));
    assert!(!p.folder.join("models").join(list[0]["name"].as_str().unwrap()).join("model.partial").exists());
}

#[cfg(unix)]
#[test]
fn measuring_needs_a_saved_dataset_version_and_gets_its_text() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    let measuring = ReshapeSpec { drop_layers: None, ..drop_spec() };
    let bare = project(d.path());
    let e = rruns::execute(&bare, &measuring, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("Save a training-data version in Step 2 first"), "{e}");
    assert!(rruns::list(&bare).is_empty(), "nothing was started");
    // growing and explicit layers need no data
    let grow = ReshapeSpec { source_model_id: "qwen3.5-4b".into(), ..spec(Mode::Grow, 1) };
    let m = catalog::find("qwen3.5-4b").unwrap();
    let ins: Inspect = serde_json::from_str(INSPECT).unwrap();
    let job = rruns::build_job(&bare, &m, &grow, &ins, "n1").unwrap();
    assert_eq!((job["mode"].clone(), job["calibration"]["train"].clone()), (json!("grow"), serde_json::Value::Null));
    // with a version the job points at its train.jsonl
    let p = project_with_version(&d.path().join("second"));
    let job = rruns::build_job(&p, &m, &measuring, &ins, "n2").unwrap();
    let train = job["calibration"]["train"].as_str().unwrap();
    assert!(train.ends_with("datasets/v1/train.jsonl") && std::path::Path::new(train).is_file(), "{train}");
    assert_eq!((job["calibration"]["version"].clone(), job["calibration"]["samples"].clone(), job["calibration"]["max_len"].clone(), job["load"].clone()), (json!(1), json!(128), json!(512), json!("auto")));
    assert_eq!(job["protect"], json!({"first": 1, "last": 1}));
    // a bad choice never reaches the worker
    let bad = ReshapeSpec { blocks: 3, ..measuring };
    assert!(rruns::build_job(&p, &m, &bad, &ins, "n3").unwrap_err().0.contains("at most half"));
}

#[cfg(unix)]
#[test]
fn without_the_training_software_the_message_says_what_to_do() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project(d.path());
    assert!(rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("Training setup"));
    assert!(rruns::execute(&p, &ReshapeSpec { source_model_id: "nope".into(), ..drop_spec() }, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("not in the list"));
}

#[cfg(unix)]
#[test]
fn a_model_cut_off_by_closing_the_program_is_marked_failed() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project(d.path());
    let dir = p.folder.join("models/20261113-000000-x-slim-8L");
    std::fs::create_dir_all(dir.join("model.partial")).unwrap();
    std::fs::write(dir.join("state.json"), json!({"name": "20261113-000000-x-slim-8L", "status": "running"}).to_string()).unwrap();
    let done = p.folder.join("models/20261112-000000-y-slim-8L");
    std::fs::create_dir_all(&done).unwrap();
    std::fs::write(done.join("state.json"), json!({"name": "20261112-000000-y-slim-8L", "status": "done"}).to_string()).unwrap();
    assert_eq!(rruns::mark_interrupted(&p), 1);
    let list = rruns::list(&p);
    assert_eq!(list[0]["status"], "failed");
    assert!(list[0]["error"].as_str().unwrap().contains("The program was closed while this model was being made. Start it again."));
    assert!(!dir.join("model.partial").exists());
    assert_eq!(list[1]["status"], "done");
    assert_eq!(rruns::mark_interrupted(&p), 0);
}

#[cfg(unix)]
#[test]
fn a_model_in_use_cannot_be_removed_and_removing_cleans_up() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    let p = project(d.path());
    let r = rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap();
    let (name, id) = (r["name"].as_str().unwrap().to_string(), r["model_id"].as_str().unwrap().to_string());
    let repo = catalog::find(&id).unwrap().repo;
    // a training run on this model
    let run = truns::runs_dir(&p).join("run-on-slim");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("config.json"), json!({"model": {"repo": repo, "family": "qwen3_5"}}).to_string()).unwrap();
    let e = rruns::remove(&p, &name).unwrap_err();
    assert_eq!(e.0, "Run run-on-slim was trained on this model. Delete that run first, or keep the model.");
    assert!(p.folder.join("models").join(&name).is_dir() && catalog::find(&id).is_some());
    std::fs::remove_dir_all(&run).unwrap();
    rruns::remove(&p, &name).unwrap();
    assert!(!p.folder.join("models").join(&name).exists() && catalog::find(&id).is_none() && rruns::list(&p).is_empty());
    // names from the window are checked
    for bad in ["", "../x", "a/b", "a\\b"] {
        assert!(rruns::remove(&p, bad).is_err(), "{bad}");
    }
    assert!(rruns::remove(&p, "missing").unwrap_err().0.contains("not found"));
}

#[cfg(unix)]
#[test]
fn the_page_state_and_options_come_from_the_worker_and_the_plan() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects"));
    let mut m = AppModel::load();
    assert!(m.reshape_state().is_err(), "needs a project");
    m.create_project("P").unwrap();
    let s = m.reshape_state().unwrap();
    assert!(s["models"].as_array().unwrap().iter().any(|x| x["id"] == "qwen3.5-4b") && s["made"].as_array().unwrap().is_empty() && s["dataset_version"].is_null());
    let o = m.reshape_options("qwen3.5-4b", "prune").unwrap();
    assert_eq!(o["inspect"]["layers"], 12);
    assert_eq!(o["options"].as_array().unwrap().len(), 1);                      // 3 blocks: at most 1 (first and last are protected)
    assert_eq!(o["options"][0]["layers"], 8);
    assert_eq!(o["default_blocks"], 1);
    let g = m.reshape_options("qwen3.5-4b", "grow").unwrap();
    assert_eq!(g["options"][0]["layers"], 16);
    assert!(g["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("little or no gain")));
    assert!(m.reshape_options("qwen3.5-4b", "shrink").is_err() && m.reshape_options("nope", "prune").is_err());
}

// ------------------------------------------------------------------------------------------------------- testing made models
use ndforge_core::evaluate::runs as eruns;
use ndforge_core::train::plan::{base_hyper, Method, Preset};

fn project_with_test_split(root: &std::path::Path) -> Project {
    let mut p = project(root);
    let f = root.join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    store::save_version(&p).unwrap();
    p
}

/// A fake Python that answers every question with a fixed text and keeps the job it was given in `capture.json`
#[cfg(unix)]
fn fake_answers(root: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let capture = root.join("capture.json");
    std::fs::write(root.join("fake.py"), format!(r#"
import json, sys
job = json.load(open(sys.argv[1], encoding="utf-8"))
json.dump(job, open({cap:?}, "w", encoding="utf-8"))
out = {{s: {{q["id"]: ("trained says hi" if s == "trained" else "original says hi") for q in job["questions"]}} for s in job["which"]}}
json.dump(out, open(job["out"], "w", encoding="utf-8"))
print(json.dumps({{"type": "result", "data": {{"out": job["out"], "n": len(job["questions"])}}}}), flush=True)
"#, cap = capture.to_string_lossy())).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\nexec python3 -I {} \"$5\"\n", root.join("fake.py").display())).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
    capture
}

fn captured(f: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap()
}

/// A finished made model on disk (what `reshape::runs::execute` leaves behind); returns its model folder
fn made_model(p: &Project, name: &str, origin_params: u64, params: u64) -> std::path::PathBuf {
    let dir = p.folder.join("models").join(name);
    let model = dir.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("ndforge_reshape.json"), json!({"format": 1, "origin": {"repo": "unsloth/Qwen3.5-4B", "family": "qwen3_5", "layers": 32, "params": origin_params},
        "mode": "prune", "layers": 24, "params": params}).to_string()).unwrap();
    std::fs::write(dir.join("state.json"), json!({"name": name, "status": "done", "spec": {"source_model_id": "qwen3.5-4b", "mode": "prune"}}).to_string()).unwrap();
    model
}

/// A trained run (adapter) whose base model is `repo`
fn run_on(p: &Project, name: &str, repo: &std::path::Path) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("adapter")).unwrap();
    std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
    let m = catalog::find("qwen3.5-4b").unwrap();
    let mut job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    job["model"]["repo"] = json!(repo);
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("summary.json"), json!({"steps": 40, "final_loss": 0.8}).to_string()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "method": "lora", "step": 40, "total_steps": 40}).to_string()).unwrap();
}

#[cfg(unix)]
#[test]
fn a_made_model_is_tested_against_its_original_and_appears_in_the_table() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let cap = fake_answers(d.path());
    let p = project_with_test_split(d.path());
    let model = made_model(&p, "20261113-000000-x-slim-24L", 9_000_000_000, 6_000_000_000);
    assert!(rruns::read_test(&p, "20261113-000000-x-slim-24L").is_none());
    let none = AtomicBool::new(false);
    let r = rruns::test(&p, "20261113-000000-x-slim-24L", None, Some(32.0), &none, &mut |_| {}).unwrap();
    assert_eq!((r["run"].as_str(), r["against"].as_str()), (Some("model:20261113-000000-x-slim-24L"), Some("origin")));
    assert_eq!(r["loaded"], json!({"original": "16-bit", "trained": "16-bit"}));
    let job = captured(&cap);
    assert_eq!((job["model"]["repo"].as_str(), job["model"]["family"].as_str()), (Some("unsloth/Qwen3.5-4B"), Some("qwen3_5")));
    assert_eq!(job["full_model"].as_str().unwrap(), model.to_string_lossy());
    assert!(job.get("adapter").is_none() && job.get("trained_base").is_none() && job.get("load_4bit").is_none());
    assert_eq!(rruns::read_test(&p, "20261113-000000-x-slim-24L").unwrap()["n"], r["n"]);
    assert!(rruns::list(&p)[0]["tested"] == true);
    // a card too small for 16-bit: the sides that do not fit load in 4-bit, and the result says so
    let r = rruns::test(&p, "20261113-000000-x-slim-24L", None, Some(17.0), &none, &mut |_| {}).unwrap();
    assert_eq!(captured(&cap)["load_4bit"], json!({"original": true, "trained": false}));
    assert_eq!(r["loaded"], json!({"original": "4-bit", "trained": "16-bit"}));
    // the comparison table has a row for it
    let table = eruns::compare(&p);
    let row = table.iter().find(|x| x["name"] == "model:20261113-000000-x-slim-24L").expect("a row");
    assert_eq!((row["method"].as_str(), row["steps"].clone(), row["model"].as_str()), (Some("slim"), serde_json::Value::Null, Some("qwen3.5-4b")));
    assert!(row["headline"].is_string() && row["n"].as_u64().unwrap() > 0);
}

#[cfg(unix)]
#[test]
fn testing_a_made_model_needs_data_and_a_finished_model() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_answers(d.path());
    let none = AtomicBool::new(false);
    let bare = project(d.path());
    made_model(&bare, "m1", 9, 6);
    assert!(rruns::test(&bare, "m1", None, None, &none, &mut |_| {}).unwrap_err().0.contains("Save a training-data version in Step 2 first"));
    let p = project_with_test_split(&d.path().join("second"));
    std::fs::create_dir_all(p.folder.join("models/half")).unwrap();
    assert!(rruns::test(&p, "half", None, None, &none, &mut |_| {}).unwrap_err().0.contains("not ready"));
    for bad in ["../x", "a/b", ""] {
        assert!(rruns::test(&p, bad, None, None, &none, &mut |_| {}).is_err());
    }
    let model = made_model(&p, "noorigin", 9, 6);
    std::fs::remove_file(model.join("ndforge_reshape.json")).unwrap();
    assert!(rruns::test(&p, "noorigin", None, None, &none, &mut |_| {}).unwrap_err().0.contains("original is unknown"));
}

#[cfg(unix)]
#[test]
fn a_run_on_a_made_model_is_tested_against_it_or_against_the_original() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let cap = fake_answers(d.path());
    let p = project_with_test_split(d.path());
    let slim = made_model(&p, "m1", 9_000_000_000, 6_000_000_000);
    run_on(&p, "run-slim", &slim);
    let none = AtomicBool::new(false);
    // the default: against the made model (the adapter switched off) - the job is exactly what it was before this feature
    let r = eruns::execute(&p, "run-slim", 30, None, &none, &mut |_| {}).unwrap();
    let job = captured(&cap);
    assert_eq!((r["against"].as_str(), job["model"]["repo"].as_str().unwrap()), (Some("base"), slim.to_string_lossy().as_ref()));
    assert!(job.get("trained_base").is_none() && job.get("load_4bit").is_none() && job["adapter"].is_string());
    assert_eq!(r["loaded"], json!({"original": "16-bit", "trained": "16-bit"}));
    assert!(eruns::read_test(&p, "run-slim").is_some());
    // against the original it was made from: the real original answers one side, the adapter on the made model the other
    let r = eruns::execute_against(&p, "run-slim", 30, None, true, Some(32.0), &none, &mut |_| {}).unwrap();
    let job = captured(&cap);
    assert_eq!(r["against"], "origin");
    assert_eq!((job["model"]["repo"].as_str(), job["model"]["family"].as_str()), (Some("unsloth/Qwen3.5-4B"), Some("qwen3_5")));
    assert_eq!(job["trained_base"].as_str().unwrap(), slim.to_string_lossy());
    assert!(job["adapter"].as_str().unwrap().ends_with("run-slim/adapter") && job.get("load_4bit").is_none());
    // on a smaller card the 9 B original loads in 4-bit (and the 6 B made model too)
    eruns::execute_against(&p, "run-slim", 30, None, true, Some(12.0), &none, &mut |_| {}).unwrap();
    assert_eq!(captured(&cap)["load_4bit"], json!({"original": true, "trained": true}));
    // the free chat can ask both as well
    let a = eruns::ask_against(&p, "run-slim", None, "Who is Anders?", true, None, &none, &mut |_| {}).unwrap();
    assert_eq!((a["original"].as_str(), a["trained"].as_str()), (Some("original says hi"), Some("trained says hi")));
    assert_eq!(captured(&cap)["trained_base"].as_str().unwrap(), slim.to_string_lossy());
    // a run on an ordinary base has no origin: asking for it changes nothing
    run_on(&p, "run-plain", std::path::Path::new("unsloth/Qwen3.5-4B"));
    let r = eruns::execute_against(&p, "run-plain", 30, None, true, Some(32.0), &none, &mut |_| {}).unwrap();
    assert_eq!(r["against"], "base");
    assert!(captured(&cap).get("trained_base").is_none());
    // pair_for_run says the same in data
    let pair = eruns::pair_for_run(&p, "run-slim", true, Some(32.0)).unwrap();
    assert_eq!((pair.original.as_str(), pair.against, pair.four_bit), ("unsloth/Qwen3.5-4B", "origin", (false, false)));
    assert!(matches!(&pair.trained, eruns::Side::Adapter { base: Some(b), .. } if b == &slim.to_string_lossy()));
}
