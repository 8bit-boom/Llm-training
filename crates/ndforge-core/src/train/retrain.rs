//! "Retrain with the new export": the world changed, so make the examples again (keeping the reviewer's drops and edits), save a new data version
//! and train a NEW run with the settings of the last finished one. Nothing earlier is overwritten.
use serde_json::{json, Value};

use crate::bundle::LABELS;
use crate::dataset::builders::Recipe;
use crate::dataset::store;
use crate::diff::diff_bundles;
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::plan::Hyper;
use crate::train::runs::{latest_version, list_runs, runs_dir};

/// A change of more than this share of the world's records suggests training again from the original model (cleaner) instead of continuing.
pub const CONTINUE_LIMIT: f64 = 0.25;

/// The newest run that finished (not stopped or failed).
pub fn last_done_run(p: &Project) -> Option<Value> {
    list_runs(p).into_iter().find(|r| r["status"] == "done" && r["has_adapter"] == true)
}

/// What the page shows. `available` is true when a world update has not been trained yet and a finished run exists to take the settings from.
pub fn info(p: &Project) -> Result<Value> {
    let Some(last) = last_done_run(p) else { return Ok(json!({"available": false})) };
    let Some(cur) = p.current_bundle() else { return Ok(json!({"available": false})) };
    let trained_sha = latest_version(p).and_then(|(_, m)| m["bundle_sha256"].as_str().map(String::from));
    if trained_sha.as_deref() == Some(cur.sha256.as_str()) {
        return Ok(json!({"available": false}));
    }
    // compare with the world data the last dataset version was made from (kept in sources/)
    let prev = p.sources.iter().rev().filter_map(|s| std::fs::read(p.folder.join("sources").join(s)).ok().map(|_| s.clone()))
        .find_map(|s| crate::bundle::open_bundle(p.folder.join("sources").join(&s)).ok().filter(|b| Some(b.sha256.as_str()) == trained_sha.as_deref()));
    let (summary, ratio) = match &prev {
        Some(prev) => {
            let d = diff_bundles(prev, &cur)?;
            let total: usize = LABELS.iter().map(|(m, _, _)| cur.counts.get(*m).copied().unwrap_or(0)).sum::<usize>().max(1);
            (d.summary(), d.total() as f64 / total as f64)
        }
        None => ("The world data changed.".to_string(), 1.0),
    };
    Ok(json!({
        "available": true, "summary": summary, "changed_ratio": (ratio * 1000.0).round() / 1000.0, "continue_suggested": ratio <= CONTINUE_LIMIT,
        "last_run": {"name": last["name"], "model": last["model"], "model_id": last["model_id"], "method": last["method"]},
        "steps": [
            "Make the examples again from the new world data (your drops and edits stay).",
            "Save them as a new data version.",
            "Train a new run with the same model and settings. The earlier run and its files are kept.",
        ],
    }))
}

/// Steps 1 and 2 of the retrain: a new data version. Returns (new version number, settings of the last finished run, its name).
pub fn prepare(p: &Project, from_previous: bool) -> Result<(u32, Value, Hyper, String)> {
    let last = last_done_run(p).ok_or_else(|| Error::new("There is no finished training run to take the settings from. Train once first (Step 3)."))?;
    let name = last["name"].as_str().unwrap_or("").to_string();
    let cfg: Value = std::fs::read(runs_dir(p).join(&name).join("config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or_else(|| Error::new("The settings of the last run are missing."))?;
    let mut hyper: Hyper = serde_json::from_value(cfg["hyper"].clone()).map_err(|_| Error::new("The settings of the last run are damaged."))?;
    let recipe = store::load_recipe(p).unwrap_or_else(Recipe::default);
    store::generate_draft(p, &recipe)?;
    let manifest = store::save_version(p)?;
    if from_previous {
        // a short run on top of what the model already knows
        hyper.max_steps = None;
        hyper.epochs = (hyper.epochs / 2.0).max(1.0);
        hyper.warmup = hyper.warmup.min(5);
    }
    Ok((manifest["version"].as_u64().unwrap_or(0) as u32, cfg, hyper, name))
}
