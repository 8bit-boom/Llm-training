//! Adding a model from Hugging Face: look the repository up (`GET /api/models/<repo>`), check that ND Lore Forge can train it, work out its
//! size, licence and chat-format family, and make a catalog entry. Only checks that can be done from the model's public description are made;
//! whether it really trains is shown by the Quick test.
use std::time::Duration;

use serde_json::Value;

use super::catalog::{models, ModelEntry};
use crate::error::{Error, Result};

/// `NDFORGE_HF_URL` moves the address (tests use a stand-in server).
fn base() -> String {
    std::env::var("NDFORGE_HF_URL").ok().filter(|u| !u.trim().is_empty()).map(|u| u.trim().trim_end_matches('/').to_string()).unwrap_or_else(|| "https://huggingface.co".to_string())
}

/// "org/name", or a Hugging Face address such as https://huggingface.co/org/name/tree/main.
pub fn parse_repo(input: &str) -> Result<String> {
    let t = input.trim().trim_end_matches('/');
    let t = t.strip_prefix("https://").or_else(|| t.strip_prefix("http://")).unwrap_or(t);
    let t = t.strip_prefix("huggingface.co/").or_else(|| t.strip_prefix("www.huggingface.co/")).or_else(|| t.strip_prefix("hf.co/")).unwrap_or(t);
    let parts: Vec<&str> = t.split('/').collect();
    let ok = |s: &str| !s.is_empty() && s.len() <= 96 && s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric()) && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if parts.len() < 2 || !ok(parts[0]) || !ok(parts[1]) {
        return Err(Error::new("Type the model's name as it is on Hugging Face, like unsloth/Qwen3-4B, or paste its address (https://huggingface.co/...)."));
    }
    Ok(format!("{}/{}", parts[0], parts[1]))
}

/// "unsloth/Qwen3-4B-Instruct" -> "custom-unsloth-qwen3-4b-instruct"
fn slug(repo: &str) -> String {
    let s: String = repo.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '-' }).collect();
    format!("custom-{}", s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-"))
}

/// The chat format the worker knows for this model: "qwen3_5", "gemma4", or "generic" (the model's own chat template, its end-of-turn token).
fn family_of(model_type: &str, architectures: &[String]) -> &'static str {
    let key = format!("{model_type} {}", architectures.join(" ")).to_lowercase().replace(['.', '-'], "_");
    if key.contains("qwen3_5") { "qwen3_5" } else if key.contains("gemma4") || key.contains("gemma_4") { "gemma4" } else { "generic" }
}

/// What a lookup found: the entry to add, and things the person should know first.
#[derive(Debug, Clone)]
pub struct Found {
    pub entry: ModelEntry,
    pub warnings: Vec<String>,
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty() && s != "false",
        _ => true,
    }
}

/// Memory for training at about 2048 tokens, from the same rule of thumb that fits Unsloth's published figures for the built-in models
/// (bf16 LoRA about 2.2 GB per billion parameters; the 4-bit figure is our estimate).
pub fn estimate_vram(params_b: f64) -> (f64, f64) {
    (((params_b * 2.2 + 1.0) * 10.0).round() / 10.0, ((params_b * 0.7 + 2.0) * 10.0).round() / 10.0)
}

/// Looks a repository up and builds its entry; Err(plain message) when it cannot be trained here.
pub fn lookup(input: &str) -> Result<Found> {
    let repo = parse_repo(input)?;
    let existing = models();
    if let Some(m) = existing.iter().find(|m| m.repo.eq_ignore_ascii_case(&repo) || m.official.eq_ignore_ascii_case(&repo)) {
        return Err(Error::new(format!("{repo} is already in the list as \"{}\".", m.name)));
    }
    let url = format!("{}/api/models/{repo}?blobs=true", base());
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout_read(Duration::from_secs(30)).build();
    let info: Value = match agent.get(&url).set("User-Agent", "nd-lore-forge").call() {
        Ok(r) => r.into_json().map_err(|e| Error::new(format!("Hugging Face's answer was not readable: {e}")))?,
        Err(ureq::Error::Status(404, _)) => return Err(Error::new(format!("{repo} was not found on Hugging Face. Check the spelling; a private model cannot be used."))),
        Err(ureq::Error::Status(401 | 403, _)) => return Err(Error::new(format!("{repo} needs a Hugging Face login (it is gated or private). Only open models can be added for now."))),
        Err(ureq::Error::Status(code, _)) => return Err(Error::new(format!("Hugging Face answered with an error ({code}). Try again in a moment."))),
        Err(ureq::Error::Transport(t)) => return Err(Error::new(format!("Hugging Face could not be reached. Check the internet connection. [{t}]"))),
    };
    let mut warnings: Vec<String> = Vec::new();
    if truthy(&info["private"]) || truthy(&info["gated"]) {
        return Err(Error::new(format!("{repo} is gated or private: it needs a Hugging Face login, which ND Lore Forge cannot use yet. Choose an open copy (for example from unsloth/...).")));
    }
    let tags: Vec<String> = info["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_lowercase)).collect()).unwrap_or_default();
    let files: Vec<(String, f64)> = info["siblings"].as_array().map(|a| a.iter().filter_map(|f| Some((f["rfilename"].as_str()?.to_string(), f["size"].as_f64().unwrap_or(0.0)))).collect()).unwrap_or_default();
    let safetensors_bytes: f64 = files.iter().filter(|(n, _)| n.ends_with(".safetensors")).map(|(_, s)| *s).sum();
    let has_safetensors = files.iter().any(|(n, _)| n.ends_with(".safetensors")) || info["safetensors"].is_object();
    if info["library_name"] == "gguf" || (!has_safetensors && files.iter().any(|(n, _)| n.ends_with(".gguf"))) {
        return Err(Error::new(format!("{repo} contains GGUF files for running, not the original weights that training needs. Look for the same model without \"GGUF\" in its name.")));
    }
    if !has_safetensors {
        return Err(Error::new(format!("{repo} has no .safetensors weights, so it cannot be trained here.")));
    }
    if let Some(tag) = info["pipeline_tag"].as_str() {
        if !["text-generation", "image-text-to-text", "any-to-any", "text2text-generation"].contains(&tag) {
            return Err(Error::new(format!("{repo} is a \"{tag}\" model, not a language model for chat.")));
        }
    }
    let cfg = &info["config"];
    if cfg["auto_map"].is_object() {
        return Err(Error::new(format!("{repo} needs its own program code to load (trust_remote_code). For safety ND Lore Forge does not run downloaded code; choose a model the standard libraries support.")));
    }
    if cfg["quantization_config"].is_object() || tags.iter().any(|t| ["gptq", "awq", "bitsandbytes", "4-bit", "8-bit"].contains(&t.as_str())) {
        return Err(Error::new(format!("{repo} is already shrunk (quantized). Training needs the original 16-bit weights; ND Lore Forge shrinks the model itself when you choose QLoRA. Choose the original model.")));
    }
    // a chat template is what turns the practice conversations into the model's own format
    let template_text = cfg["tokenizer_config"]["chat_template"].as_str().unwrap_or("").to_string();
    match cfg["tokenizer_config"].as_object() {
        Some(t) if t.get("chat_template").is_none_or(|c| c.is_null() || c.as_str() == Some("")) => {
            return Err(Error::new(format!("{repo} has no chat template, so it is a base model that does not know the system / user / assistant format. Choose its Instruct (chat) version.")));
        }
        None => warnings.push("Its chat template could not be checked from here. If the Quick test fails with a message about a chat template, choose the Instruct version of the model.".to_string()),
        _ => {}
    }
    let model_type = cfg["model_type"].as_str().unwrap_or("").to_string();
    let architectures: Vec<String> = cfg["architectures"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
    let family = family_of(&model_type, &architectures);
    if family == "generic" {
        warnings.push(format!("ND Lore Forge has checked its training format only for Qwen3.5 and Gemma 4. For this model ({}) it uses the model's own chat template; a Quick test shows whether it learns. When you install it into Ollama, Ollama's own copy of that template is used.", if model_type.is_empty() { "unknown type" } else { &model_type }));
    }
    if family == "generic" && (template_text.contains("<think>") || template_text.contains("enable_thinking")) {
        warnings.push("Its chat template has a \"thinking\" mode. ND Lore Forge trains it with the template exactly as the model ships it, so it learns to answer in that mode.".to_string());
    }
    let params = match info["safetensors"]["total"].as_f64().filter(|t| *t > 0.0) {
        Some(t) => t / 1e9,
        None if safetensors_bytes > 0.0 => {
            warnings.push("Its size was estimated from the file sizes (assuming 16-bit weights).".to_string());
            safetensors_bytes / 2.0 / 1e9
        }
        None => return Err(Error::new(format!("The size of {repo} could not be found."))),
    };
    let params = (params * 100.0).round() / 100.0;
    if params > 14.0 {
        warnings.push(format!("At {params:.0} billion parameters it is big: even LoRA needs a very large graphics card."));
    }
    let license = info["cardData"]["license"].as_str().map(String::from)
        .or_else(|| tags.iter().find_map(|t| t.strip_prefix("license:").map(String::from))).unwrap_or_else(|| "unknown".to_string());
    if license == "unknown" || license == "other" {
        warnings.push("Its licence is not stated clearly on Hugging Face. Read the model page before you share the trained model.".to_string());
    }
    let (lora, qlora) = estimate_vram(params);
    let name = repo.split('/').nth(1).unwrap_or(&repo).replace(['_', '-'], " ");
    let entry = ModelEntry {
        id: slug(&repo), name, family: family.to_string(), repo: repo.clone(), official: repo.clone(), params_b: params, license, vram_lora_gb: lora, vram_qlora_gb: qlora,
        good_for: "Added by you from Hugging Face.".to_string(),
        note: "Memory figures are estimates from the model's size until a run has measured them.".to_string(), custom: true,
    };
    Ok(Found { entry, warnings })
}
