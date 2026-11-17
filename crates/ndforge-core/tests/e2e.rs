//! End to end on a CPU with a tiny REAL model: the program's own steps (the ones Auto-cut calls) run the real worker on a random 4-layer Qwen3.5-shaped model:
//! inspect, measure, try, cut (thinner layers), heal (the teacher answers, the student learns, the LoRA is merged), thinking check of the original, the cut and the healed model.
//! It needs a Python with torch (CPU is enough), transformers, peft, trl, datasets and accelerate, so it is `#[ignore]`d: `NDFORGE_E2E_PYTHON=/path/to/venv/bin/python cargo test -p ndforge-core
//! --test e2e -- --ignored --nocapture`. The CI job `e2e-cpu` does exactly that (on pushes to main and nightly, not on pull requests). Not covered: the GGUF export (it needs llama.cpp)
//! and the Stop button (the other tests have it).
use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::project::Project;
use ndforge_core::reshape::auto::{Effects, Out, Student};
use ndforge_core::reshape::auto_fx::RealEffects;
use ndforge_core::reshape::auto_plan::AutoSpec;
use ndforge_core::reshape::runs::{models_dir, HealSpec, TrySpec};
use ndforge_core::reshape::thinkcheck::{self, ThinkSpec};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn done<T>(what: &str, r: ndforge_core::Result<Out<T>>) -> T {
    match r {
        Ok(Out::Done(x)) => x,
        Ok(Out::Stopped) => panic!("{what}: stopped"),
        Err(e) => panic!("{what} failed: {}", e.0),
    }
}

fn read(p: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(p).unwrap_or_else(|e| panic!("{} is missing: {e}", p.display()))).unwrap()
}

#[test]
#[ignore = "needs a Python with torch: set NDFORGE_E2E_PYTHON (the CI job e2e-cpu does)"]
fn a_tiny_real_model_goes_through_measure_try_cut_heal_and_the_thinking_check() {
    let py = PathBuf::from(std::env::var("NDFORGE_E2E_PYTHON").expect("set NDFORGE_E2E_PYTHON to the python of a venv with torch, transformers, peft, trl, datasets, accelerate"));
    let env_dir = py.parent().and_then(|p| p.parent()).expect("a python inside <env>/bin or <env>/Scripts").to_path_buf();
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", &env_dir);
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let t0 = Instant::now();
    let lap = |what: &str| eprintln!("[{:>6.1}s] {what}", t0.elapsed().as_secs_f64());

    // a project with the sample world and a saved training-data version
    let mut p = Project::create(&d.path().join("projects"), "Ashfall Reach").unwrap();
    let bundle = d.path().join("w.ndtrain");
    ndforge_core::sample::build(&bundle, 1, None).unwrap();
    p.import_bundle(&open_bundle(&bundle).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();

    // the tiny model: the worker's own test helper builds it (random weights, a real tokenizer and chat template)
    let worker = repo().join("worker");
    let tiny = d.path().join("tiny");
    let script = "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); from tinymodel import build_arch_dir; build_arch_dir(Path(sys.argv[3]), 'qwen3_5', num_layers=4, intermediate=512)";
    let out = std::process::Command::new(&py).args(["-c", script]).arg(worker.join("tests")).arg(&worker).arg(&tiny).output().unwrap();
    assert!(out.status.success(), "the tiny model could not be built: {}", String::from_utf8_lossy(&out.stderr));
    ndforge_core::train::catalog::add_custom(serde_json::from_value(json!({"id": "custom-tiny", "name": "Tiny Qwen3.5", "family": "qwen3_5", "repo": tiny, "official": "", "params_b": 0.001, "license": "test",
        "vram_lora_gb": 1.0, "vram_qlora_gb": 1.0, "good_for": "", "note": ""})).unwrap()).unwrap();
    lap("model built");

    let events = std::cell::RefCell::new(Vec::<Value>::new());
    let mut sink = |e: Value| events.borrow_mut().push(e);
    let mut fx = RealEffects::new(&p, Arc::new(AtomicBool::new(false)), None, &mut sink);

    // inspect and the plan (no card: vram unknown); a tiny model cannot reach a target of a billion, and says so instead of failing
    let model = fx.model("custom-tiny").unwrap();
    let ins = done("inspect", fx.inspect(&model));
    assert!(ins.supported, "the tiny model is a supported architecture: {:?}", ins.why);
    let plan = ndforge_core::reshape::auto::preview(&ins, &AutoSpec::new("custom-tiny", 1.0), None, Some(100.0), true);
    assert!(plan["plan"]["blockers"].as_array().is_some_and(|b| !b.is_empty()), "{plan}");
    lap("inspect + plan");

    // measure, then try one width on a few samples
    done("measure", fx.measure("custom-tiny", 8, "16bit", None));
    lap("measured");
    let try_spec: TrySpec = serde_json::from_value(json!({"model_id": "custom-tiny", "samples": 8, "measure_samples": 8, "load": "16bit", "widths": [256]})).unwrap();
    let analysis = done("try", fx.try_run(&try_spec));
    assert!(analysis.is_object() || analysis.is_array(), "{analysis}");
    lap("tried");

    // cut: thinner layers (the MLP from 512 to 256), no block removed
    let spec = serde_json::from_value(json!({"source_model_id": "custom-tiny", "mode": "prune", "blocks": 0, "calib_samples": 8, "calib_len": 128, "load": "16bit", "width": {"intermediate": 256}, "width_samples": 8})).unwrap();
    let cut = done("cut", fx.cut(&spec));
    let cut_dir = models_dir(&p).join(&cut.name);
    assert!(cut_dir.join("model").join("config.json").is_file(), "the cut model is saved");
    assert_eq!(read(&cut_dir.join("state.json"))["status"], "done");
    lap(&format!("cut {}", cut.name));

    // heal: 20,000 tokens, the teacher answers first (cache mode), then the student learns, then the LoRA is merged
    let heal_spec: HealSpec = serde_json::from_value(json!({"tokens_m": 10, "tokens_exact": 20000, "general_share": 0.0, "mode": "cache", "seq": 512, "batch": 1, "per_update": 4,
        "checkpointing": "off", "rank": 8, "teacher_load": "16bit", "k": 8, "method": "lora", "kl_chunk": 64, "warmup": 0, "backend": "hf"})).unwrap();
    let healed = done("heal", fx.heal(&Student::Made(cut.name.clone()), &tiny.to_string_lossy(), &heal_spec));
    assert!(Path::new(&healed.dir).join("config.json").is_file(), "the healed model is saved: {}", healed.dir);
    let heal_json = read(&cut_dir.join("heal.json"));
    assert_eq!(heal_json["status"], "done", "{heal_json}");
    let cache_dirs: Vec<_> = std::fs::read_dir(models_dir(&p).join("_teacher_cache")).unwrap().flatten().collect();
    assert_eq!(cache_dirs.len(), 1, "the teacher's answers are kept once, in the shared folder");
    assert!(read(&cut_dir.join("distill").join("cache").join("index.json"))["complete"] == true);
    lap("healed");

    // the thinking check of the three models (a few new tokens per answer: only the plumbing is looked at, not the quality)
    for target in ["original", "cut", "healed"] {
        let spec = ThinkSpec { name: cut.name.clone(), target: target.into(), quick: true, load: Some("16bit".into()), max_new: Some(8), ..Default::default() };
        let r = thinkcheck::run(&p, &spec, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_or_else(|e| panic!("thinking check of the {target} failed: {}", e.0));
        assert!(r["main"]["n"].as_u64().unwrap_or(0) > 0, "{target}: {r}");
        lap(&format!("thinking check of the {target}"));
    }
    let seen = events.borrow();
    assert!(seen.iter().any(|e| e["kind"] == "progress"), "the window would have seen progress");
    eprintln!("{} events, {:.0} s in all", seen.len(), t0.elapsed().as_secs_f64());
}
