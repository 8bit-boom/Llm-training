//! Makes a project with the sample world imported, for trying the window: `cargo run -p ndforge-core --example demo_project -- <projects dir> [dataset [trained [update]]]` (`dataset` also makes and saves a training-data version, `trained` adds a finished run and an export, `update` imports a newer world file)
use ndforge_core::{bundle::open_bundle, project::Project, sample, settings::Settings};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("a projects folder"));
    let mut p = Project::create(&dir, "Ashfall Reach").expect("create");
    let f = dir.join("sample.ndtrain");
    sample::build(&f, 1, None).expect("sample");
    p.import_bundle(&open_bundle(&f).expect("open")).expect("import");
    if std::env::args().nth(2).as_deref() == Some("dataset") {
        use ndforge_core::dataset::{builders::Recipe, store};
        store::generate_draft(&p, &Recipe::default()).expect("draft");
        store::save_version(&p).expect("version");
    }
    if std::env::args().nth(3).as_deref() == Some("trained") {
        // a finished run and an export on disk, so Steps 3 and 5 and the retrain card can be looked at without a GPU
        use ndforge_core::train::{catalog::find, plan::{base_hyper, Method, Preset}, runs};
        let m = find("qwen3.5-4b").unwrap();
        let rdir = runs::runs_dir(&p).join("20261107-120000-qwen3.5-4b");
        std::fs::create_dir_all(rdir.join("adapter")).unwrap();
        std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
        let job = runs::build_job(&p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, "20261107-120000-qwen3.5-4b", false).unwrap();
        std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
        std::fs::write(rdir.join("summary.json"), r#"{"steps": 40, "final_loss": 0.83}"#).unwrap();
        std::fs::write(rdir.join("result.json"), "{}").unwrap();
        std::fs::write(rdir.join("state.json"), r#"{"name":"20261107-120000-qwen3.5-4b","status":"done","model":"Qwen3.5 4B","model_id":"qwen3.5-4b","method":"lora","step":40,"total_steps":40,"loss":0.83,"peak_vram_mb":11200}"#).unwrap();
        let e = p.folder.join("exports/20261107-120000-qwen3.5-4b-q4_k_m");
        std::fs::create_dir_all(&e).unwrap();
        std::fs::write(e.join("model-q4_k_m.gguf"), vec![0u8; 2_500_000]).unwrap();
        std::fs::write(e.join("meta.json"), serde_json::json!({"name": "20261107-120000-qwen3.5-4b-q4_k_m", "run": "20261107-120000-qwen3.5-4b", "quant": "q4_k_m", "family": "qwen3_5",
            "gguf_file": "model-q4_k_m.gguf", "ollama_name": "ashfall-reach-gm", "num_ctx": 8192, "installed": []}).to_string()).unwrap();
    }
    if std::env::args().nth(4).as_deref() == Some("update") {
        let f2 = dir.join("sample2.ndtrain");
        sample::build(&f2, 2, None).expect("sample 2");
        p.import_bundle(&open_bundle(&f2).expect("open 2")).expect("import 2");
    }
    let mut s = Settings::load();
    s.remember_project(&p.folder);
    s.save();
    println!("{}", p.folder.display());
}
