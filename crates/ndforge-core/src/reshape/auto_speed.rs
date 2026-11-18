//! Auto-cut's speed settings (docs/AUTO_CUT.md, "Speed settings"): the presets (Thorough, Balanced, Fast, Custom) as data in `auto_speed.json`, the knobs they set, and what
//! every knob costs in quality. Pure functions: nothing is read, nothing runs. The planner (`auto_plan`) applies the knobs, `auto_time` counts them, the orchestrator runs them.
//!
//! Rule: LEGACY = THOROUGH. A spec whose speed fields are all at their defaults is the recommended recipe and plans and runs exactly as before the settings existed.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::auto_plan::{AutoSpec, StagePlan};
use super::plan;
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

/// What a person who chose a faster setting gives up, one plain sentence per value that is not the recommended recipe's (the plan shows them as warnings and the report lists
/// them again). `stages` are the planned stages (with the shared sequence length when `share_teacher_cache` is on), `natural_seq` the sequence length each stage would heal on
/// by itself. Empty for the recipe.
pub fn speed_warnings(spec: &AutoSpec, stages: &[StagePlan], natural_seq: &[u32], total_b: f64) -> Vec<String> {
    let mut w = Vec::new();
    let thinking = spec.keep_thinking;
    let n = stages.len();
    let evidence = "In an earlier real run, healing below the recipe failed the thinking checks; the checks after each stage will tell, and a failed check costs a retry.";
    if let Some(last) = stages.last() {
        let recipe = if thinking { plan::thinking_tokens_m(last.params_b, total_b) } else { plan::default_heal_tokens_m(last.params_b) };
        if last.tokens_m < recipe {
            w.push(if thinking {
                format!("The last stage heals on {} M tokens; the recipe for a cut of this size is {recipe} M. {evidence}", last.tokens_m)
            } else {
                format!("The last stage heals on {} M tokens; the usual number for a model of this size is {recipe} M. The healed model may be clearly weaker.", last.tokens_m)
            });
        }
    }
    if n > 1 && thinking && stages[0].tokens_m < super::auto_plan::STAGE_A_TOKENS_M {
        w.push(format!("The first stage heals on {} M tokens; the recipe is {} M. The second stage starts from a model that learned less. {evidence}", stages[0].tokens_m, super::auto_plan::STAGE_A_TOKENS_M));
    }
    if thinking && spec.stages == 1 && n == 1 && stages[0].cut_pct > 15.0 {
        w.push(format!("One stage cuts {:.0} % of the model in a single step; the recipe heals half of it first. For a model that thinks this is the riskier way.", stages[0].cut_pct));
    }
    if !spec.check_cut && thinking {
        w.push("The unhealed cut is not checked for thinking (its check only informs; nothing is decided by it). The original is checked after the first heal instead, so the gates still have their yardstick.".into());
    }
    if spec.think_quick() && thinking {
        w.push("The thinking checks are the quick ones (16 questions): they take about a third less time and are noisier, so a close call can go either way.".into());
    }
    if spec.quick {
        w.push(format!("Try first measures each cut on {} texts instead of {}: it is quicker and a close race between two cuts is less certain.", plan::TRY_QUICK_SAMPLES, plan::TRY_FULL_SAMPLES));
    }
    if thinking && !spec.retry_heal {
        w.push("No retry: when a thinking check of a healed stage fails, Auto-cut does not heal it again but goes straight to cutting one step less.".into());
    }
    if !spec.back_off {
        w.push("No back-off: when a stage still fails its thinking check (after the retry, if there is one), the run ends with the report instead of cutting one step less.".into());
    }
    if !spec.imatrix {
        w.push("No importance matrix is measured for the export: the small sizes (Q4 and below) are made without it and are a little less exact.".into());
    }
    if n > 1 && spec.share_teacher_cache {
        let pin = stages.iter().map(|s| s.seq).min().unwrap_or(0);
        if stages.iter().all(|s| s.mode != "cache") {
            w.push("Same sequence length in both stages is on, but this card heals with the original next to the student (no saved answers): it changes nothing here.".into());
        }
        for (g, nat) in stages.iter().zip(natural_seq) {
            if *nat > pin {
                w.push(format!("Stage {} heals on sequence length {pin} instead of {nat} so that the original writes its answers once and the next stage reuses them: reasoning traces longer than {pin} tokens are cut short in that stage.", g.index));
            }
        }
    } else if n > 1 {
        let c: Vec<&StagePlan> = stages.iter().filter(|s| s.mode == "cache").collect();
        if c.len() > 1 && c.iter().any(|s| s.seq != c[0].seq || s.teacher_load != c[0].teacher_load) {
            w.push(format!("The stages heal on different sequence lengths ({}): the original's saved answers cannot be reused by the next stage, so it writes them again (many hours). \"Same sequence length in both stages\" shortens that, at the price of a shorter sequence in the larger stage.",
                           c.iter().map(|s| format!("stage {}: {}", s.index, s.seq)).collect::<Vec<_>>().join(", ")));
        }
    }
    w
}
