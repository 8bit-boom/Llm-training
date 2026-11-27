//! Who a measurement belongs to: the MODEL (its shape, which decides the speed, and its weights, which only tell two entries apart) and the MACHINE (the card, and the software versions the numbers depend on).
//! Speed depends on the shape, not on the values of the weights: a healed or merged copy of a cut has the same `shape_id` as the cut.
use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The canonical text of a JSON value: object keys sorted, no whitespace (so key order and formatting never change an id).
pub fn canonical(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            format!("{{{}}}", keys.iter().map(|k| format!("{}:{}", serde_json::to_string(k).unwrap_or_default(), canonical(&m[*k]))).collect::<Vec<_>>().join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(canonical).collect::<Vec<_>>().join(",")),
        other => other.to_string(),
    }
}

/// The language part of a config (a multimodal config keeps it in `text_config`).
fn text_config(cfg: &Value) -> &Value {
    if cfg["text_config"].is_object() { &cfg["text_config"] } else { cfg }
}

/// What the shape says about a model, for the page and the preset (`shape_id` is the hash of `facts`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ShapeFacts {
    pub arch: String,
    pub layers: u32,
    pub layer_types: String,
    pub hidden: u64,
    pub intermediate: Vec<u64>,
    pub vocab: u64,
}

/// The speed-relevant facts of a config, canonical: architecture, layer count and types, widths, heads, linear-attention sizes, the vocabulary and whether the embedding is tied, whether vision weights
/// exist, and whether the weights are pre-quantized. `(shape_id, facts, the canonical document)`.
pub fn shape_of_config(cfg: &Value) -> (String, ShapeFacts, Value) {
    let tc = text_config(cfg);
    let mut m = Map::new();
    let arch = cfg["architectures"].as_array().and_then(|a| a.first()).and_then(Value::as_str).or_else(|| cfg["model_type"].as_str()).or_else(|| tc["model_type"].as_str()).unwrap_or("").to_string();
    m.insert("arch".into(), json!(arch));
    for k in ["model_type", "num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings", "full_attention_interval", "num_experts", "moe_intermediate_size"] {
        if !tc[k].is_null() {
            m.insert(k.into(), tc[k].clone());
        } else if !cfg[k].is_null() {
            m.insert(k.into(), cfg[k].clone());
        }
    }
    // mixture-of-experts models with a loop (Inkling): the expert count, the experts used per token, the loop and the attention pattern change the memory and the speed,
    // so a 96-expert cut must not share a preset with the 128-expert original. Only added when present, so every existing shape_id stays what it was.
    if !tc["n_routed_experts"].is_null() {
        for k in ["n_routed_experts", "num_experts_per_tok", "loop_config", "local_layer_ids", "n_shared_experts"] {
            if !tc[k].is_null() {
                m.insert(k.into(), tc[k].clone());
            }
        }
    }
    let inter = tc["intermediate_size"].clone();
    m.insert("intermediate_size".into(), inter.clone());
    if let Value::Object(o) = tc {
        for (k, v) in o {
            if k.starts_with("linear_") {
                m.insert(k.clone(), v.clone());
            }
        }
    }
    let lt = tc["layer_types"].clone();
    m.insert("layer_types".into(), lt.clone());
    m.insert("vision".into(), json!(cfg["vision_config"].is_object() || tc["vision_config"].is_object()));
    m.insert("mtp".into(), json!(tc.as_object().is_some_and(|o| o.keys().any(|k| k.contains("mtp"))) || cfg.as_object().is_some_and(|o| o.keys().any(|k| k.contains("mtp")))));
    m.insert("prequantized".into(), json!(cfg["quantization_config"].is_object() || tc["quantization_config"].is_object()));
    let doc = Value::Object(m);
    let layers = tc["num_hidden_layers"].as_u64().unwrap_or(0) as u32;
    let intermediate: Vec<u64> = match &inter {
        Value::Array(a) => a.iter().filter_map(Value::as_u64).collect(),
        Value::Number(n) => n.as_u64().into_iter().collect(),
        _ => Vec::new(),
    };
    let facts = ShapeFacts { arch, layers, layer_types: sha_hex(canonical(&lt).as_bytes())[..8].to_string(), hidden: tc["hidden_size"].as_u64().unwrap_or(0), intermediate, vocab: tc["vocab_size"].as_u64().unwrap_or(0) };
    (format!("s-{}", &sha_hex(canonical(&doc).as_bytes())[..16]), facts, doc)
}

/// The shape a PLANNED cut will have: the original's config with the layers that stay (`kept`, indices into the original's layers) and, for a width cut, the new MLP width, as the worker's `fix_config` writes them
/// (`num_hidden_layers` = the number kept; `layer_types` remapped when present; `intermediate_size` replaced). Auto-cut can find an exact-shape preset before the cut exists.
pub fn shape_of_plan(original: &Value, kept: &[usize], intermediate: Option<u64>) -> String {
    let mut cfg = original.clone();
    let key = if cfg["text_config"].is_object() { "text_config" } else { "" };
    let tc = if key.is_empty() { &mut cfg } else { &mut cfg["text_config"] };
    tc["num_hidden_layers"] = json!(kept.len());
    if let Some(lt) = tc["layer_types"].as_array().cloned() {
        tc["layer_types"] = json!(kept.iter().filter_map(|i| lt.get(*i).cloned()).collect::<Vec<_>>());
    }
    if let Some(w) = intermediate {
        tc["intermediate_size"] = json!(w);
    }
    shape_of_config(&cfg).0
}

/// The identity of the weights: the config bytes and, for every `*.safetensors` in name order, its name, its size and the hash of its header (the first 8 + n bytes). No modification times: copying the folder
/// keeps the id. It tells two entries apart; it never gates reuse of data.
pub fn weights_id(folder: &Path) -> Result<String> {
    let mut h = Sha256::new();
    h.update(std::fs::read(folder.join("config.json")).map_err(|e| Error::new(format!("config.json of {} could not be read: {e}", folder.display())))?);
    let mut files: Vec<_> = std::fs::read_dir(folder)?.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors")).collect();
    files.sort_by_key(|e| e.file_name());
    for f in files {
        use std::io::Read;
        let mut fh = std::fs::File::open(f.path())?;
        let mut len = [0u8; 8];
        fh.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len).min(100_000_000) as usize;
        let mut head = vec![0u8; n];
        fh.read_exact(&mut head)?;
        h.update(f.file_name().to_string_lossy().as_bytes());
        h.update(fh.metadata()?.len().to_le_bytes());
        h.update(len);
        h.update(&head);
    }
    Ok(format!("w-{}", &h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()[..16]))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModelIdentity {
    pub name: String,
    pub shape_id: String,
    pub weights_id: String,
    pub arch: String,
    pub layers: u32,
    pub layer_types: String,
    pub hidden: u64,
    pub intermediate: Vec<u64>,
    pub vocab: u64,
    /// parameters, from the weight files at 2 bytes each
    pub params: u64,
    pub source_bytes: u64,
    /// the models this one was made from, nearest first (a cut of a healed model of an original ...)
    #[serde(default)]
    pub lineage: Vec<String>,
}

/// The identity of a model folder. `lineage` comes from the model list (`derived_from`) or the cut's `ndforge_reshape.json`: nearest first.
pub fn model_identity(folder: &Path, name: &str, lineage: Vec<String>) -> Result<ModelIdentity> {
    let cfg: Value = serde_json::from_slice(&std::fs::read(folder.join("config.json")).map_err(|e| Error::new(format!("config.json of {} could not be read: {e}", folder.display())))?)
        .map_err(|e| Error::new(format!("config.json of {} is not valid: {e}", folder.display())))?;
    let (shape_id, f, _) = shape_of_config(&cfg);
    let bytes: u64 = std::fs::read_dir(folder)?.flatten().filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum();
    Ok(ModelIdentity { name: name.into(), shape_id, weights_id: weights_id(folder)?, arch: f.arch, layers: f.layers, layer_types: f.layer_types, hidden: f.hidden, intermediate: f.intermediate, vocab: f.vocab, params: bytes / 2, source_bytes: bytes, lineage })
}

/// The machine a measurement was made on. `key` is the card's name and memory only: another card is another machine; the software versions are `depends_on` facts that make single results stale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MachineId {
    pub key: String,
    pub gpu: String,
    pub vram_mb: u64,
    pub driver: String,
    pub os: String,
    /// torch, transformers, peft, bitsandbytes, triton, fla, vllm_image, vllm, llama_cpp, worker ...
    #[serde(default)]
    pub software: BTreeMap<String, String>,
}

pub fn machine_key(gpu: &str, vram_mb: u64) -> String {
    format!("m-{}", &sha_hex(format!("{}|{}", gpu.trim().to_lowercase(), vram_mb).as_bytes())[..16])
}

/// A machine from facts that were found elsewhere (the tests inject them; `detect` gathers the real ones).
pub fn machine_from(gpu: &str, vram_mb: u64, driver: &str, os: &str, software: BTreeMap<String, String>) -> MachineId {
    MachineId { key: machine_key(gpu, vram_mb), gpu: gpu.into(), vram_mb, driver: driver.into(), os: os.into(), software }
}

/// The first graphics card of this computer and what is known without starting Python (the software versions are added by the caller from the worker's `hello` and the settings).
pub fn detect(software: BTreeMap<String, String>) -> MachineId {
    let os = format!("{} {}", std::env::consts::OS, os_version());
    match crate::envmgr::gpu::detect_gpus_cached().ok().and_then(|g| g.into_iter().next()) {
        Some(g) => machine_from(&g.name, g.vram_mb, &g.driver, &os, software),
        None => machine_from("no NVIDIA card", 0, "", &os, software),
    }
}

fn os_version() -> String {
    #[cfg(windows)]
    {
        return std::process::Command::new("cmd").args(["/C", "ver"]).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    }
    #[cfg(not(windows))]
    {
        std::fs::read_to_string("/etc/os-release").ok().and_then(|t| t.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string()))).unwrap_or_default()
    }
}

/// The "major" (or "major.minor") of a version text: `2.14.1` -> `2.14`, `621.24` -> `621`. Used to decide whether a software change makes a result stale.
pub fn minor_of(version: &str) -> String {
    let parts: Vec<&str> = version.trim().trim_start_matches('v').split(|c: char| c == '.' || c == '+' || c == '-').collect();
    parts.iter().take(2).copied().collect::<Vec<_>>().join(".")
}

pub fn major_of(version: &str) -> String {
    version.trim().trim_start_matches('v').split(|c: char| !c.is_ascii_digit()).next().unwrap_or("").to_string()
}
