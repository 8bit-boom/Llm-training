//! Auto-cut's speed settings (docs/AUTO_CUT.md, "Speed settings"): the presets (Thorough, Balanced, Fast, Custom) as data in `auto_speed.json`, the knobs they set, and what
//! every knob costs in quality. Pure functions: nothing is read, nothing runs. The planner (`auto_plan`) applies the knobs, `auto_time` counts them, the orchestrator runs them.
//!
//! Rule: LEGACY = THOROUGH. A spec whose speed fields are all at their defaults is the recommended recipe and plans and runs exactly as before the settings existed.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::auto_plan::AutoSpec;
use crate::error::{Error, Result};

const PRESETS_JSON: &str = include_str!("auto_speed.json");

/// One preset: the values of the speed knobs it sets (`set`: AutoSpec field -> value), what it is and the plain sentence of its quality risk. The window holds the same table
/// (`AC_SPEED_PRESETS` in `ui/app.js`; a smoke test compares them).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedPreset {
    pub id: String,
    pub label: String,
    pub blurb: String,
    pub risk: String,
    pub set: Map<String, Value>,
}

#[derive(Deserialize)]
struct PresetFile {
    presets: Vec<SpeedPreset>,
}

/// The presets of `auto_speed.json`, in the order the window shows them.
pub fn presets() -> Vec<SpeedPreset> {
    serde_json::from_str::<PresetFile>(PRESETS_JSON).map(|f| f.presets).unwrap_or_default()
}

pub fn preset(id: &str) -> Option<SpeedPreset> {
    presets().into_iter().find(|p| p.id == id)
}

/// The id of the preset the window selects for a model of `total_b` billion parameters when the person has not chosen: Balanced from 15 B (a heal of a big model takes days),
/// else Thorough.
pub fn default_preset(total_b: f64) -> &'static str {
    if total_b >= 15.0 { "balanced" } else { "thorough" }
}

/// The AutoSpec fields the presets set (the speed knobs).
pub const KNOBS: [&str; 12] = ["stages", "stage_a_tokens_m", "last_tokens_m", "share_teacher_cache", "quick", "think_quick", "think_gate", "think_gate_n", "check_cut", "retry_heal", "back_off", "imatrix"];

impl AutoSpec {
    /// The spec with the values of preset `id` on top of it (everything that is not a speed knob stays). "custom" only names the choice.
    pub fn with_preset(&self, id: &str) -> Result<AutoSpec> {
        let p = preset(id).ok_or_else(|| Error::new(format!("Unknown speed setting: {id}.")))?;
        let mut v = serde_json::to_value(self).map_err(|e| Error::new(e.to_string()))?;
        if let Some(o) = v.as_object_mut() {
            for (k, x) in &p.set {
                o.insert(k.clone(), x.clone());
            }
            o.insert("preset".into(), Value::String(p.id.clone()));
        }
        serde_json::from_value(v).map_err(|e| Error::new(format!("The preset {id} does not fit the spec: {e}")))
    }

    /// The speed knobs as the window shows them (a think_quick that follows `quick` is shown as its value).
    fn knob_values(&self) -> Map<String, Value> {
        let mut v = match serde_json::to_value(self) {
            Ok(Value::Object(o)) => o,
            _ => Map::new(),
        };
        v.insert("think_quick".into(), Value::Bool(self.think_quick()));
        KNOBS.iter().filter_map(|k| v.get(*k).map(|x| (k.to_string(), x.clone()))).collect()
    }

    /// The preset whose values the speed knobs have (the recommended recipe for a legacy spec), None when they are a mix: "Custom".
    pub fn matching_preset(&self) -> Option<String> {
        let mine = self.knob_values();
        presets().into_iter().filter(|p| p.id != "custom").find(|p| {
            // the preset's own values as a spec would have them
            self.with_preset(&p.id).map(|s| s.knob_values() == mine).unwrap_or(false)
        }).map(|p| p.id)
    }
}
