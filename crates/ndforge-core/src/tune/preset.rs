//! The preset: what was tested for ONE model on ONE machine, the settings that won (the whole Heal card, not only the speed knobs), the numbers behind each choice, and what each choice depends on.
//! `format: 1`; later fields are additive (`#[serde(default)]` everywhere, unknown fields are kept in `extra`).
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use super::identity::{MachineId, ModelIdentity};

pub const FORMAT: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Guard {
    pub name: String,
    pub ok: bool,
    #[serde(default)]
    pub value: Option<f64>,
    #[serde(default)]
    pub limit: Option<f64>,
}

/// One tested function (a test id such as `heal.student_batch`): the chosen and the default values of its knobs, how much better it is, and how sure that is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FunctionResult {
    /// the knobs that won, by their Heal-card names (`student_batch_tokens`, `kl_chunk`, `seq` ...)
    #[serde(default)]
    pub chosen: Map<String, Value>,
    #[serde(default)]
    pub default: Map<String, Value>,
    /// the speed-up over the default (1.0 = none)
    #[serde(default)]
    pub gain: f64,
    /// the measured run-to-run spread (0.03 = 3 %)
    #[serde(default)]
    pub spread: f64,
    /// "exact" | "tol" | "stat" | "quality": a quality knob is based on a short trial, not on a full heal
    #[serde(default)]
    pub class: String,
    /// the tier of the Tune page: 1 speed and memory, 2 quality (short trial), 3 other functions
    #[serde(default)]
    pub tier: u32,
    #[serde(default)]
    pub guards: Vec<Guard>,
    /// the numbers of the winner (tokens/s, peak VRAM, the rows' length statistics ...)
    #[serde(default)]
    pub measured: Value,
    /// every candidate that was tried, with its numbers (an OOM is a row)
    #[serde(default)]
    pub tried: Vec<Value>,
    /// the facts that make this result stale when they change: `gpu`, `driver_major`, `torch_minor`, `transformers_minor` ... (`<package>_minor`), `vllm_image`, `llama_cpp`
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// the knobs of `chosen` that may be used for a model of another size or shape
    #[serde(default)]
    pub transferable: Vec<String>,
    /// the settings this measurement assumed (sequence length, method, rank ...): applying it where they differ says so
    #[serde(default)]
    pub context: Map<String, Value>,
    #[serde(default)]
    pub note: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Preset {
    pub format: u32,
    pub id: String,
    pub name: String,
    pub created: String,
    #[serde(default)]
    pub source_run: String,
    pub model: ModelIdentity,
    pub machine: MachineId,
    #[serde(default)]
    pub functions: BTreeMap<String, FunctionResult>,
    /// the whole Heal card as tested (every knob with its value), so the preset can be applied as a whole
    #[serde(default)]
    pub card: Map<String, Value>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub report: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// `t-` + 16 hex of the weights, the machine and the creation time: the same run saved twice has the same id.
pub fn preset_id(weights_id: &str, machine_key: &str, created: &str) -> String {
    let h: String = Sha256::digest(format!("{weights_id}|{machine_key}|{created}").as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    format!("t-{}", &h[..16])
}

impl Preset {
    pub fn new(name: &str, created: &str, source_run: &str, model: ModelIdentity, machine: MachineId) -> Preset {
        Preset { format: FORMAT, id: preset_id(&model.weights_id, &machine.key, created), name: name.into(), created: created.into(), source_run: source_run.into(), model, machine, ..Default::default() }
    }

    /// All the knobs the preset sets: the whole card, with the chosen value of every tested function on top.
    pub fn values(&self) -> Map<String, Value> {
        let mut out = self.card.clone();
        for f in self.functions.values() {
            for (k, v) in &f.chosen {
                out.insert(k.clone(), v.clone());
            }
        }
        out
    }

    /// The values that may be used for another model: only the knobs a function lists as `transferable`.
    pub fn transferable_values(&self) -> Map<String, Value> {
        let mut out = Map::new();
        for f in self.functions.values() {
            for k in &f.transferable {
                if let Some(v) = f.chosen.get(k) {
                    out.insert(k.clone(), v.clone());
                }
            }
        }
        out
    }

    /// The best gain per function for the list: "heal.student_batch x2.41".
    pub fn gains(&self) -> Vec<(String, f64)> {
        self.functions.iter().filter(|(_, f)| f.gain > 0.0).map(|(k, f)| (k.clone(), f.gain)).collect()
    }
}

/// What differs between a preset's values and the current settings of a card: `[{setting, from, to}]`, only where they differ (a missing current value counts as different).
pub fn diff(current: &Map<String, Value>, values: &Map<String, Value>) -> Vec<(String, Value, Value)> {
    let mut out: Vec<(String, Value, Value)> = values.iter().filter(|(k, v)| current.get(*k).unwrap_or(&Value::Null) != *v).map(|(k, v)| (k.clone(), current.get(k).cloned().unwrap_or(Value::Null), v.clone())).collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}
