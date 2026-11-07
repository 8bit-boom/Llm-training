//! The five models (catalog/models.toml, compiled in).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    /// chat-format family the worker knows: "qwen3_5" or "gemma4"
    pub family: String,
    pub repo: String,
    pub official: String,
    pub params_b: f64,
    pub license: String,
    pub vram_lora_gb: f64,
    pub vram_qlora_gb: f64,
    pub good_for: String,
    pub note: String,
}

#[derive(Deserialize)]
struct File {
    model: Vec<ModelEntry>,
}

pub fn models() -> Vec<ModelEntry> {
    toml::from_str::<File>(include_str!("../../../../catalog/models.toml")).expect("catalog/models.toml is valid").model
}

pub fn find(id: &str) -> Option<ModelEntry> {
    models().into_iter().find(|m| m.id == id)
}
