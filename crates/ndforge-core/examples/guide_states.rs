//! Prints the real data every page shows, for a project made by `demo_project` (used by `scripts/make_guide_screenshots.sh`):
//! `cargo run -p ndforge-core --example guide_states -- <project folder>` -> one JSON object on stdout.
use ndforge_core::model::AppModel;
use serde_json::{json, Value};

fn main() {
    let folder = std::env::args().nth(1).expect("a project folder");
    let mut m = AppModel::load();
    let welcome = m.state();
    m.open_project(&folder).expect("open");
    let filter = json!({});
    let help: serde_json::Map<String, Value> = ["welcome", "data", "dataset", "train", "test", "use", "setup", "guide", "troubleshooting"].iter().map(|k| (k.to_string(), json!(ndforge_core::help::help_html(k)))).collect();
    let out: Value = json!({
        "welcome": welcome,
        "state": m.state(),
        "dataset_state": m.dataset_state().unwrap(),
        "dataset_examples": m.dataset_examples(&filter, 0, 12).unwrap(),
        "train_state": m.train_state(None).unwrap(),
        "export_state": m.export_state().unwrap(),
        "test_state": m.test_state().unwrap(),
        "retrain_info": m.retrain_info().unwrap(),
        "train_plan": m.train_plan("qwen3.5-4b", ndforge_core::train::plan::Method::Lora, ndforge_core::train::plan::Preset::Normal, None, Some("mixed")).unwrap(),
        "help": help,
        "env_status": ndforge_core::envmgr::tasks::status(),
    });
    println!("{}", serde_json::to_string(&out).unwrap());
}
