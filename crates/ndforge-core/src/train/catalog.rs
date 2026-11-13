//! The six models (catalog/models.toml, compiled in) and the ones the user adds from Hugging Face (`hub.rs` looks them up).
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
    /// added by the user from Hugging Face (not one of the six tested models)
    #[serde(default)]
    pub custom: bool,
    /// the model this one was cut from / grown from (Hugging Face name or folder), for models made on the "Slim or grow a model" page
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<String>,
    /// one line for the model list, e.g. "slimmed: 64 → 48 layers"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reshape: Option<String>,
}

#[derive(Deserialize)]
struct File {
    model: Vec<ModelEntry>,
}

/// The six models that ship with the program.
pub fn builtin_models() -> Vec<ModelEntry> {
    toml::from_str::<File>(include_str!("../../../../catalog/models.toml")).expect("catalog/models.toml is valid").model
}

fn custom_file() -> std::path::PathBuf {
    crate::paths::config_dir().join("custom_models.json")
}

/// The models the user added from Hugging Face (`custom_models.json` in the settings folder); a damaged file counts as none.
pub fn custom_models() -> Vec<ModelEntry> {
    std::fs::read_to_string(custom_file()).ok().and_then(|t| serde_json::from_str::<Vec<ModelEntry>>(&t).ok())
        .map(|v| v.into_iter().map(|m| ModelEntry { custom: true, ..m }).collect()).unwrap_or_default()
}

fn save_custom(list: &[ModelEntry]) -> crate::error::Result<()> {
    let f = custom_file();
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(f, serde_json::to_vec_pretty(list)?)?;
    Ok(())
}

pub fn add_custom(m: ModelEntry) -> crate::error::Result<()> {
    let mut list = custom_models();
    list.retain(|x| x.id != m.id);
    list.push(ModelEntry { custom: true, ..m });
    save_custom(&list)
}

/// Removes a model the user added; the five built-in ones cannot be removed.
pub fn remove_custom(id: &str) -> crate::error::Result<()> {
    let mut list = custom_models();
    if !list.iter().any(|m| m.id == id) {
        return Err(crate::error::Error::new("Only models you added yourself can be removed."));
    }
    list.retain(|m| m.id != id);
    save_custom(&list)
}

/// The five built-in models, then the ones the user added.
pub fn models() -> Vec<ModelEntry> {
    let mut all = builtin_models();
    all.extend(custom_models());
    all
}

pub fn find(id: &str) -> Option<ModelEntry> {
    models().into_iter().find(|m| m.id == id)
}
