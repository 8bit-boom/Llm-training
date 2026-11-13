//! What Ollama needs to answer exactly the way the model was trained: the chat template (a Go template written for each model family),
//! the stop marker and sampling settings, as a Modelfile and as the fields of Ollama's create API.
//!
//! The templates reproduce the prompts the training used (verified against each model's own Jinja chat template, see tests): Qwen3.5 answers WITHOUT thinking
//! (an empty think block opens the answer), Gemma 4 uses its turn markers. Ollama adds the beginning-of-sequence token itself, so none is written here.
use serde_json::{json, Value};

pub struct OllamaSpec {
    pub template: String,
    pub stop: Vec<String>,
    /// numbers are sent and written as numbers
    pub options: Vec<(&'static str, f64)>,
}

const QWEN_TEMPLATE: &str = r#"{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if or (eq .Role "system") (eq .Role "user") -}}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" -}}
<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last -}}
<|im_start|>assistant
<think>

</think>

{{ end }}
{{- end }}"#;

const GEMMA_TEMPLATE: &str = r#"{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "system" -}}
<|turn>system
{{ .Content }}<turn|>
{{ else if eq .Role "user" -}}
<|turn>user
{{ .Content }}<turn|>
{{ else if eq .Role "assistant" -}}
<|turn>model
{{ .Content }}{{ if not $last }}<turn|>
{{ end }}
{{- end }}
{{- if and (ne .Role "assistant") $last -}}
<|turn>model
{{ end }}
{{- end }}"#;

/// `family` is the catalog's family ("qwen3_5", "gemma4"); `num_ctx` the context length Ollama should use.
pub fn spec_for(family: &str, num_ctx: u32) -> OllamaSpec {
    match family {
        "gemma4" => OllamaSpec {
            template: GEMMA_TEMPLATE.to_string(), stop: vec!["<turn|>".into()],
            // Gemma 4's recommended sampling (Unsloth guide)
            options: vec![("temperature", 1.0), ("top_p", 0.95), ("top_k", 64.0), ("num_ctx", num_ctx as f64)],
        },
        // a model the program has no hand-written template for: no TEMPLATE at all, so Ollama uses the chat template stored inside the GGUF
        "generic" => OllamaSpec { template: String::new(), stop: vec![], options: vec![("temperature", 0.7), ("num_ctx", num_ctx as f64)] },
        _ => OllamaSpec {
            template: QWEN_TEMPLATE.to_string(), stop: vec!["<|im_end|>".into()],
            options: vec![("temperature", 0.7), ("top_p", 0.8), ("top_k", 20.0), ("num_ctx", num_ctx as f64)],
        },
    }
}

fn num(v: f64) -> String {
    if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
}

/// The Modelfile to keep next to the GGUF (works with `ollama create <name> -f Modelfile`).
pub fn modelfile_text(gguf_file: &str, spec: &OllamaSpec) -> String {
    let mut t = format!("FROM ./{gguf_file}\n\n");
    if !spec.template.is_empty() {
        t += &format!("TEMPLATE \"\"\"{}\"\"\"\n\n", spec.template);
    }
    for s in &spec.stop {
        t += &format!("PARAMETER stop \"{s}\"\n");
    }
    for (k, v) in &spec.options {
        t += &format!("PARAMETER {k} {}\n", num(*v));
    }
    t
}

/// The JSON body of Ollama's `POST /api/create` for a GGUF already uploaded as a blob.
pub fn create_body(model: &str, gguf_file: &str, digest: &str, spec: &OllamaSpec) -> Value {
    let mut params = serde_json::Map::new();
    if !spec.stop.is_empty() {
        params.insert("stop".into(), json!(spec.stop));
    }
    for (k, v) in &spec.options {
        params.insert((*k).into(), if v.fract() == 0.0 { json!(*v as i64) } else { json!(*v) });
    }
    let mut body = json!({"model": model, "files": {gguf_file: digest}, "parameters": params, "stream": false});
    if !spec.template.is_empty() {
        body["template"] = json!(spec.template);
    }
    body
}

/// A name Ollama accepts: lower case letters, digits, "-", "_", "." (and an optional ":tag").
pub fn ollama_name(world: &str, tag: &str) -> String {
    let base: String = world.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let base = base.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    format!("{}-gm{}", if base.is_empty() { "world".to_string() } else { base }, if tag.is_empty() { String::new() } else { format!(":{tag}") })
}

pub struct CardInfo<'a> {
    pub world: &'a str,
    pub model_name: &'a str,
    pub base_repo: &'a str,
    pub license: &'a str,
    pub method: &'a str,
    pub quant: &'a str,
    pub dataset_version: u64,
    pub steps: Option<u64>,
    pub final_loss: Option<f64>,
    pub ollama_name: &'a str,
    /// the model the base model was made from, when the base is a slimmed or grown one (Slim or grow a model)
    pub derived_from: Option<&'a str>,
}

/// A short README that travels with the model files.
pub fn model_card(c: &CardInfo) -> String {
    format!(
"# {name} - a GM model for \"{world}\"

Made with ND Lore Forge.

- Base model: {base} ({license}; the tuned model keeps that licence){derived}
- Training: {method}, SFT on saved data version {ver}{steps}{loss}
- Format: GGUF {quant}
- Audience: **the game master only**. It was trained on ALL of the world's text, including GM secrets, and it repeats them.

## Use it
- Ollama: `ollama create {oname} -f Modelfile`, then set `OLLAMA_MODEL={oname}` in nd-world.
- Keep nd-world's lore search switched on: the model was trained to use the context nd-world gives it.
- Do not let players use AI chat with this model.
",
        name = c.model_name, world = c.world, base = c.base_repo, license = c.license, method = c.method, ver = c.dataset_version, quant = c.quant, oname = c.ollama_name,
        derived = c.derived_from.map(|o| format!("\n- The base model was made from {o} by removing or adding layers (ND Lore Forge, Slim or grow a model)")).unwrap_or_default(),
        steps = c.steps.map(|s| format!(", {s} steps")).unwrap_or_default(),
        loss = c.final_loss.map(|l| format!(", final loss {l:.3}")).unwrap_or_default())
}
