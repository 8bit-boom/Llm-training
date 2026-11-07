use std::process::Command;

use ndforge_core::export::modelfile::{create_body, modelfile_text, ollama_name, spec_for};
use serde_json::json;
use tempfile::tempdir;

fn go_available() -> bool {
    Command::new("go").arg("version").output().map(|o| o.status.success()).unwrap_or(false)
}

fn render(template: &str, messages: serde_json::Value) -> String {
    let d = tempdir().unwrap();
    let (t, m) = (d.path().join("t.tmpl"), d.path().join("m.json"));
    std::fs::write(&t, template).unwrap();
    std::fs::write(&m, messages.to_string()).unwrap();
    let go = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ollama_template/main.go");
    let out = Command::new("go").arg("run").arg(go).arg(&t).arg(&m).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn convo() -> (serde_json::Value, serde_json::Value) {
    let one = json!([{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q1"}]);
    let multi = json!([{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}]);
    (one, multi)
}

/// The expected prompts are what each model's own Jinja chat template renders (and what the training used); see worker/tests (printed from the real templates).
#[test]
fn qwen_template_matches_the_prompt_the_model_was_trained_on() {
    if !go_available() {
        eprintln!("go not installed: template rendering test skipped");
        return;
    }
    let spec = spec_for("qwen3_5", 8192);
    let (one, multi) = convo();
    assert_eq!(render(&spec.template, one), "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert_eq!(render(&spec.template, multi),
        "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    // no system message
    assert_eq!(render(&spec.template, json!([{"role": "user", "content": "Hi"}])), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    // a trailing assistant message is a prefix to continue (no end marker, no think block added)
    assert_eq!(render(&spec.template, json!([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Sure"}])),
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nSure");
}

#[test]
fn gemma_template_matches_the_prompt_the_model_was_trained_on() {
    if !go_available() {
        eprintln!("go not installed: template rendering test skipped");
        return;
    }
    let spec = spec_for("gemma4", 8192);
    let (one, multi) = convo();
    // Ollama adds <bos> itself
    assert_eq!(render(&spec.template, one), "<|turn>system\nSYS<turn|>\n<|turn>user\nQ1<turn|>\n<|turn>model\n");
    assert_eq!(render(&spec.template, multi), "<|turn>system\nSYS<turn|>\n<|turn>user\nQ1<turn|>\n<|turn>model\nA1<turn|>\n<|turn>user\nQ2<turn|>\n<|turn>model\n");
    assert_eq!(render(&spec.template, json!([{"role": "user", "content": "Hi"}])), "<|turn>user\nHi<turn|>\n<|turn>model\n");
}

#[test]
fn modelfile_and_create_body() {
    let spec = spec_for("gemma4", 4096);
    let mf = modelfile_text("model-q4_k_m.gguf", &spec);
    assert!(mf.starts_with("FROM ./model-q4_k_m.gguf\n\nTEMPLATE \"\"\"{{- range"));
    assert!(mf.contains("PARAMETER stop \"<turn|>\"\n") && mf.contains("PARAMETER temperature 1\n") && mf.contains("PARAMETER top_p 0.95\n") && mf.contains("PARAMETER num_ctx 4096\n"));
    let b = create_body("w-gm", "model-q4_k_m.gguf", "sha256:abc", &spec);
    assert_eq!(b["files"]["model-q4_k_m.gguf"], "sha256:abc");
    assert_eq!((b["model"].as_str(), b["stream"].as_bool()), (Some("w-gm"), Some(false)));
    assert_eq!(b["parameters"]["stop"], json!(["<turn|>"]));
    assert_eq!(b["parameters"]["num_ctx"], 4096);
    assert_eq!(b["parameters"]["top_p"], 0.95);
    let q = spec_for("qwen3_5", 8192);
    assert!(modelfile_text("m.gguf", &q).contains("PARAMETER stop \"<|im_end|>\"") && q.options.iter().any(|(k, v)| *k == "top_k" && *v == 20.0));
}

#[test]
fn ollama_names() {
    assert_eq!(ollama_name("Ashfall Reach", ""), "ashfall-reach-gm");
    assert_eq!(ollama_name("  Émile's — World!  ", "v2"), "mile-s-world-gm:v2");
    assert_eq!(ollama_name("日本語", ""), "world-gm");
}
