//! The general text of healing and of the importance matrix: which dataset it comes from.
//!
//! * `PRESETS`: the datasets the program knows (data, not code branches: licence, licence note, kind, gated, needs an acknowledgement).
//! * `general_datasets.json` in the settings folder (next to `custom_models.json`): the datasets the person added, and which licences they acknowledged.
//! * `general_choice.json` in a project folder: the last choice made for healing and for the importance matrix.
//! * The Hugging Face token (for gated datasets) is kept with the other secrets (`secrets.json`, key "huggingface") and reaches the worker only as the
//!   environment variable HF_TOKEN (`envmgr::jobs`), never in a job file.
//! * `check`: "Check it" reads about 20 rows through the worker and says what they look like.
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::paths::config_dir;
use crate::project::Project;

/// The default general text: open (Apache-2.0), chat. The worker has the same id (`generaltext.DEFAULT_ID`).
pub const DEFAULT_ID: &str = "mlabonne/open-perfectblend";
/// The seed of the general text's sample (healing and the importance matrix both use it).
pub const SEED: u64 = 3407;
pub const TOKEN_SECRET: &str = "huggingface";

/// One dataset the program knows. `kind`: "chat" or "text".
pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    pub license: &'static str,
    /// shown under the licence in the chooser, in plain words ("" = nothing more to say)
    pub license_note: &'static str,
    pub kind: &'static str,
    pub note: &'static str,
    pub config: Option<&'static str>,
    pub split: &'static str,
    /// needs access approval on Hugging Face and the person's token
    pub gated: bool,
    /// the person must tick an acknowledgement (once) before it can be used
    pub needs_ack: bool,
}

pub const PRESETS: [Preset; 8] = [
    Preset { id: "mlabonne/open-perfectblend", name: "Open PerfectBlend (recommended)", license: "Apache-2.0", license_note: "", kind: "chat", config: None, split: "train", gated: false, needs_ack: false,
        note: "About 1.4 million chat conversations from several open sets (column conversations). Open, no sign-in. The default." },
    Preset { id: "HuggingFaceH4/ultrachat_200k", name: "UltraChat 200k", license: "MIT", license_note: "", kind: "chat", config: None, split: "train_sft", gated: false, needs_ack: false,
        note: "About 208,000 everyday chat conversations (column messages)." },
    Preset { id: "allenai/tulu-3-sft-mixture", name: "Tulu 3 SFT mixture", license: "ODC-By", license_note: "Name the source if you publish what you make with it.", kind: "chat", config: None, split: "train", gated: false, needs_ack: false,
        note: "About 939,000 chat examples: instructions, maths, code, safety (column messages)." },
    Preset { id: "HuggingFaceFW/fineweb-edu", name: "FineWeb-Edu (plain text)", license: "ODC-By", license_note: "Name the source if you publish what you make with it.", kind: "text", config: Some("sample-10BT"), split: "train", gated: false, needs_ack: false,
        note: "Educational web pages as plain text (column text). Huge: only the rows that are needed are streamed, never the whole set." },
    Preset { id: "wikimedia/wikipedia", name: "Wikipedia (English)", license: "CC-BY-SA-3.0 / GFDL", license_note: "ShareAlike: what you publish that is built on this text may have to carry the same licence. Read it first.", kind: "text",
        config: Some("20231101.en"), split: "train", gated: false, needs_ack: false, note: "English Wikipedia articles as plain text (columns title and text)." },
    Preset { id: "faunix/Qwen3.8-27B-Distillation-40K", name: "Qwen3.8-27B reasoning traces (40K)", license: "Apache-2.0", license_note: "", kind: "chat", config: None, split: "train", gated: false, needs_ack: false,
        note: "40,000 reasoning traces written by Qwen3.8-27B: code, maths, science, logic (column messages, thinking inline). Rows are LONG (about 5,750 tokens): use a sequence length of 4096 or more for healing; \
                rows longer than the length are cut, which is fine for distillation. Good for healing a pruned Qwen3.x model; a weak fit for the importance matrix (it leans to code and maths)." },
    Preset { id: "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation", name: "Multi-teacher distillation corpus (sft_balanced)", license: "other (noncommercial research)",
        license_note: "Noncommercial research use; mixed licences; teacher-service terms (some outputs may not be used to build products that compete with the service they came from). Read the dataset's LICENSE before using it for anything you publish or sell.",
        kind: "chat", config: Some("sft_balanced"), split: "train", gated: false, needs_ack: true,
        note: "Conversations from several strong teacher models, many languages (column messages; the thinking is in its own field and is put inline). Rows that need tools are skipped." },
    Preset { id: "nvidia/Nemotron-Pretraining-SFT-v1", name: "NVIDIA Nemotron SFT (needs access)", license: "other (NVIDIA)", license_note: "Check NVIDIA's licence on the dataset page.", kind: "chat", config: Some("Nemotron-SFT-General"), split: "train",
        gated: true, needs_ack: false, note: "Gated: ask for access on its Hugging Face page and add your Hugging Face token. Without them it cannot be downloaded." },
];

fn nz(v: &Option<String>) -> Option<String> {
    v.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Where the general text comes from: a Hugging Face dataset (`id`) or a file on this computer (`file`). Empty fields count as not given.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralSource {
    pub id: Option<String>,
    pub file: Option<String>,
    pub config: Option<String>,
    pub split: Option<String>,
    /// None = the worker finds the column and says which
    pub column: Option<String>,
    /// what the list shows (custom datasets)
    pub name: Option<String>,
    pub license: Option<String>,
    pub kind: Option<String>,
}

impl GeneralSource {
    pub fn hub(id: &str) -> GeneralSource {
        GeneralSource { id: Some(id.to_string()), ..Default::default() }
    }

    /// The list key: dataset id (or file) plus config.
    pub fn key(&self) -> String {
        let base = self.file.clone().or_else(|| self.id.clone()).unwrap_or_default();
        match &self.config {
            Some(c) => format!("{base}#{c}"),
            None => base,
        }
    }

    fn tidy(&self) -> GeneralSource {
        GeneralSource { id: nz(&self.id), file: nz(&self.file), config: nz(&self.config), split: nz(&self.split), column: nz(&self.column), name: nz(&self.name), license: nz(&self.license), kind: nz(&self.kind) }
    }
}

pub fn preset(key: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.config.map(|c| format!("{}#{c}", p.id)).unwrap_or_else(|| p.id.to_string()) == key)
}

fn preset_for(src: &GeneralSource) -> Option<&'static Preset> {
    let id = src.id.as_deref()?;
    PRESETS.iter().find(|p| p.id.eq_ignore_ascii_case(id) && (src.config.is_none() || src.config.as_deref() == p.config))
}

pub fn presets_json() -> Vec<Value> {
    PRESETS.iter().map(|p| json!({"key": preset_key(p), "id": p.id, "name": p.name, "license": p.license, "license_note": p.license_note, "kind": p.kind, "note": p.note,
        "config": p.config, "split": p.split, "column": null, "gated": p.gated, "needs_ack": p.needs_ack, "url": format!("https://huggingface.co/datasets/{}", p.id)})).collect()
}

fn preset_key(p: &Preset) -> String {
    p.config.map(|c| format!("{}#{c}", p.id)).unwrap_or_else(|| p.id.to_string())
}

// ------------------------------------------------------------------------------------------------------------------ the saved list

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Store {
    datasets: Vec<GeneralSource>,
    /// keys of the datasets whose licence the person acknowledged
    acknowledged: Vec<String>,
}

fn store_file() -> std::path::PathBuf {
    config_dir().join("general_datasets.json")
}

/// A missing or damaged file counts as an empty list.
fn load() -> Store {
    std::fs::read_to_string(store_file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn save(s: &Store) -> Result<()> {
    let f = store_file();
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(f, serde_json::to_vec_pretty(s)?)?;
    Ok(())
}

pub fn custom_list() -> Vec<GeneralSource> {
    load().datasets
}

pub fn acknowledged() -> Vec<String> {
    load().acknowledged
}

/// "owner/name" from what the person typed (a Hugging Face address works too).
pub fn parse_dataset_id(input: &str) -> Result<String> {
    let t = input.trim().trim_matches(|c| c == '"' || c == '\'').trim().trim_end_matches('/');
    let t = t.strip_prefix("https://huggingface.co/datasets/").or_else(|| t.strip_prefix("http://huggingface.co/datasets/")).unwrap_or(t);
    let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    let mut parts = t.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(a), Some(b), None) if ok(a) && ok(b) => Ok(t.to_string()),
        _ => Err(Error::new("Write the dataset as owner/name, for example mlabonne/open-perfectblend (or paste its Hugging Face address).")),
    }
}

/// Adds a dataset (a Hugging Face id or a file that exists) to the saved list. Returns its key.
pub fn add_custom(src: &GeneralSource) -> Result<String> {
    let mut s = src.tidy();
    match (&s.id, &s.file) {
        (Some(_), Some(_)) => return Err(Error::new("Give either a Hugging Face dataset or a file, not both.")),
        (None, None) => return Err(Error::new("Give a Hugging Face dataset (owner/name) or the path of a file.")),
        (Some(id), None) => s.id = Some(parse_dataset_id(id)?),
        (None, Some(f)) => {
            if !Path::new(f).is_file() {
                return Err(Error::new(format!("The file {f} was not found.")));
            }
        }
    }
    if s.name.is_none() {
        s.name = Some(s.id.clone().or_else(|| s.file.as_ref().map(|f| Path::new(f).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| f.clone()))).unwrap_or_default());
    }
    let key = s.key();
    if preset(&key).is_some() {
        return Ok(key);
    }
    let mut st = load();
    st.datasets.retain(|d| d.key() != key);
    st.datasets.push(s);
    save(&st)?;
    Ok(key)
}

pub fn remove_custom(key: &str) -> Result<()> {
    let mut st = load();
    if !st.datasets.iter().any(|d| d.key() == key) {
        return Err(Error::new("Only datasets you added yourself can be removed."));
    }
    st.datasets.retain(|d| d.key() != key);
    save(&st)
}

/// The person ticked the acknowledgement of this dataset's licence.
pub fn acknowledge(key: &str) -> Result<()> {
    let p = preset(key).ok_or_else(|| Error::new("That dataset is not in the list."))?;
    if !p.needs_ack {
        return Ok(());
    }
    let mut st = load();
    if !st.acknowledged.iter().any(|k| k == key) {
        st.acknowledged.push(key.to_string());
    }
    save(&st)
}

// ------------------------------------------------------------------------------------------------------------------ the choice

/// The source a job uses: gaps filled from the preset (config, split), the default when nothing is chosen, and the acknowledgement checked when the preset needs one.
pub fn resolve(src: Option<&GeneralSource>, require_ack: bool) -> Result<GeneralSource> {
    let mut s = src.map(|s| s.tidy()).unwrap_or_default();
    if s.id.is_none() && s.file.is_none() {
        s.id = Some(DEFAULT_ID.to_string());
    }
    if s.id.is_some() && s.file.is_some() {
        return Err(Error::new("Give either a Hugging Face dataset or a file for the general text, not both."));
    }
    if let Some(id) = s.id.clone() {
        s.id = Some(parse_dataset_id(&id)?);
    }
    if let Some(p) = preset_for(&s) {
        if s.config.is_none() {
            s.config = p.config.map(String::from);
        }
        if s.split.is_none() {
            s.split = Some(p.split.to_string());
        }
        if require_ack && p.needs_ack && !load().acknowledged.contains(&preset_key(p)) {
            return Err(Error::new(format!("The licence of \"{}\" needs your OK first: {} Tick the box in the list of general texts, or choose another dataset.", p.name, p.license_note)));
        }
    }
    Ok(s)
}

/// The "general" block of a job: only what the worker reads.
pub fn job_block(src: &GeneralSource, share: f64) -> Value {
    let mut v = json!({"share": share});
    for (k, x) in [("id", &src.id), ("file", &src.file), ("config", &src.config), ("split", &src.split), ("column", &src.column)] {
        if let Some(x) = x {
            v[k] = json!(x);
        }
    }
    v
}

/// What makes this general text a different one, as the worker computes it (`generaltext.general_key`): part of the keys of the teacher cache and of the
/// saved matrix. `general` is a job block or Null.
pub fn key(general: &Value, seed: u64) -> Value {
    let share = general["share"].as_f64().unwrap_or(0.0);
    if !general.is_object() || share <= 0.0 {
        return json!({"used": false});
    }
    let s = |k: &str| general[k].as_str().filter(|x| !x.is_empty()).map(String::from);
    let file = s("file");
    let id = s("id").or_else(|| if file.is_none() { Some(DEFAULT_ID.to_string()) } else { None });
    json!({"used": true, "id": id, "file": file, "config": s("config"), "split": s("split"), "column": s("column"), "share": share, "seed": seed})
}

fn choice_file(p: &Project) -> std::path::PathBuf {
    p.folder.join("general_choice.json")
}

/// The last choice of this project: {"heal": source | null, "imatrix": source | null}.
pub fn choices(p: &Project) -> Value {
    std::fs::read(choice_file(p)).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).filter(Value::is_object).unwrap_or_else(|| json!({}))
}

pub fn set_choice(p: &Project, scope: &str, src: &GeneralSource) -> Result<()> {
    if !matches!(scope, "heal" | "imatrix") {
        return Err(Error::new("Unknown place for the general text."));
    }
    let mut c = choices(p);
    c[scope] = serde_json::to_value(src.tidy())?;
    std::fs::write(choice_file(p), serde_json::to_vec_pretty(&c)?)?;
    Ok(())
}

pub fn token_present() -> bool {
    crate::secrets::get(TOKEN_SECRET).is_some()
}

pub fn token_set(token: &str) -> Result<()> {
    let t = token.trim();
    if t.len() < 10 || !t.starts_with("hf_") || t.chars().any(|c| c.is_whitespace()) {
        return Err(Error::new("That does not look like a Hugging Face token (it starts with hf_). Make one at huggingface.co/settings/tokens (read access is enough)."));
    }
    crate::secrets::set(TOKEN_SECRET, t)
}

pub fn token_clear() -> Result<()> {
    crate::secrets::remove(TOKEN_SECRET)
}

/// Everything the chooser needs. Never contains the token (only whether there is one).
pub fn state(p: Option<&Project>) -> Value {
    let st = load();
    json!({"default": DEFAULT_ID, "presets": presets_json(), "custom": st.datasets, "acknowledged": st.acknowledged, "choice": p.map(choices).unwrap_or_else(|| json!({})),
           "has_token": token_present()})
}

// ------------------------------------------------------------------------------------------------------------------ "Check it"

/// The licence (and whether it is gated) of a Hugging Face dataset, from the Hub's API. Never fails: None when it cannot be read.
pub fn hub_info(id: &str) -> Option<Value> {
    let url = format!("{}/api/datasets/{id}", crate::train::hub::base());
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(6)).timeout_read(Duration::from_secs(8)).build();
    let info: Value = agent.get(&url).set("User-Agent", "nd-lore-forge").call().ok()?.into_json().ok()?;
    let lic = info["cardData"]["license"].as_str().map(String::from)
        .or_else(|| info["cardData"]["license"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")))
        .or_else(|| info["tags"].as_array().and_then(|t| t.iter().filter_map(|x| x.as_str()).find_map(|x| x.strip_prefix("license:").map(String::from))));
    Some(json!({"license": lic, "gated": info["gated"].as_bool().unwrap_or_else(|| info["gated"].is_string())}))
}

/// "Check it": the worker reads about 20 rows (streamed) and answers {"verdict", "message", "detected", "sample", ...}; for a Hub dataset the licence is
/// added when the Hub tells it. An answer is always produced when the worker runs; Err only when the training software is missing.
pub fn check(src: &GeneralSource) -> Result<Value> {
    let s = resolve(Some(src), false)?;
    let py = crate::envmgr::python_path(&crate::envmgr::env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first; the check reads the dataset with it."));
    }
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let job = json!({"job": "general_check", "source": job_block(&s, 1.0), "rows": 20});
    let mut r = crate::envmgr::jobs::run_job(&py, &crate::envmgr::worker_dir(), &job, &cancel, &mut |_| {})?;
    if let Some(id) = &s.id {
        if let Some(info) = hub_info(id) {
            r["license"] = info["license"].clone();
            r["gated"] = info["gated"].clone();
        }
    }
    r["source"] = serde_json::to_value(&s)?;
    Ok(r)
}
