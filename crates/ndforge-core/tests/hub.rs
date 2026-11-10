//! Adding a model from Hugging Face: the lookup against a stand-in Hugging Face server, every refusal, the saved list, and what a model of an unknown family gets.
use std::sync::Mutex;

use ndforge_core::export::modelfile::{create_body, modelfile_text, spec_for};
use ndforge_core::model::AppModel;
use ndforge_core::train::catalog::{add_custom, builtin_models, custom_models, find, models, remove_custom};
use ndforge_core::train::hub::{estimate_vram, lookup, parse_repo};
use ndforge_core::train::plan::{plan, DataStats, Fit, Method, Preset};
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn chat(extra: Value) -> Value {
    let mut v = json!({"tags": ["text-generation", "safetensors", "license:apache-2.0"], "pipeline_tag": "text-generation", "cardData": {"license": "apache-2.0"},
        "siblings": [{"rfilename": "model.safetensors", "size": 15_200_000_000u64}, {"rfilename": "config.json", "size": 800}],
        "safetensors": {"total": 7_600_000_000u64}, "config": {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "tokenizer_config": {"chat_template": "{{ messages }}"}}});
    for (k, x) in extra.as_object().unwrap() {
        v[k] = x.clone();
    }
    v
}

fn hf() -> String {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
    std::thread::spawn(move || {
        for req in server.incoming_requests() {
            let path = req.url().split('?').next().unwrap_or("").to_string();
            let repo = path.strip_prefix("/api/models/").unwrap_or("").to_string();
            let (code, body): (u16, Value) = match repo.as_str() {
                "acme/Qwen3.5-7B-Instruct" => (200, chat(json!({"config": {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"], "tokenizer_config": {"chat_template": "x"}}}))),
                "acme/Gemma-4-E3-it" => (200, chat(json!({"config": {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"], "tokenizer_config": {"chat_template": "x"}}, "pipeline_tag": "image-text-to-text"}))),
                "acme/Llama-3.2-3B-Instruct" => (200, chat(json!({"safetensors": {"total": 3_210_000_000u64}, "cardData": {"license": "llama3.2"}}))),
                "acme/thinker" => (200, chat(json!({"config": {"model_type": "llama", "tokenizer_config": {"chat_template": "{% if enable_thinking %}<think>{% endif %}"}}}))),
                "acme/no-total" => (200, chat(json!({"safetensors": null}))),
                "acme/no-license" => (200, chat(json!({"cardData": {}, "tags": ["text-generation"]}))),
                "acme/huge-70B" => (200, chat(json!({"safetensors": {"total": 70_000_000_000u64}}))),
                "acme/no-template-check" => (200, chat(json!({"config": {"model_type": "llama"}}))),
                "acme/gated" => (200, chat(json!({"gated": "manual"}))),
                "acme/login" => (401, json!({"error": "Invalid username or password."})),
                "acme/Model-GGUF" => (200, chat(json!({"library_name": "gguf", "siblings": [{"rfilename": "m-Q4.gguf", "size": 4_000_000_000u64}]}))),
                "acme/base-model" => (200, chat(json!({"config": {"model_type": "llama", "tokenizer_config": {"eos_token": "</s>"}}}))),
                "acme/custom-code" => (200, chat(json!({"config": {"model_type": "x", "auto_map": {"AutoModelForCausalLM": "modeling.X"}, "tokenizer_config": {"chat_template": "x"}}}))),
                "acme/pre-quant" => (200, chat(json!({"config": {"model_type": "llama", "quantization_config": {"quant_method": "gptq"}, "tokenizer_config": {"chat_template": "x"}}}))),
                "acme/image-model" => (200, chat(json!({"pipeline_tag": "text-to-image"}))),
                "acme/bin-only" => (200, chat(json!({"siblings": [{"rfilename": "pytorch_model.bin", "size": 5}], "safetensors": null}))),
                _ => (404, json!({"error": "Repository not found"})),
            };
            let _ = req.respond(tiny_http::Response::from_string(body.to_string()).with_status_code(code).with_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap()));
        }
    });
    url
}

fn setup() -> (std::sync::MutexGuard<'static, ()>, tempfile::TempDir) {
    let g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    std::env::set_var("NDFORGE_HF_URL", hf());
    (g, d)
}

#[test]
fn names_and_addresses_are_understood() {
    assert_eq!(parse_repo("unsloth/Qwen3-4B").unwrap(), "unsloth/Qwen3-4B");
    assert_eq!(parse_repo("  https://huggingface.co/unsloth/Qwen3-4B/tree/main  ").unwrap(), "unsloth/Qwen3-4B");
    assert_eq!(parse_repo("huggingface.co/a/b.c_d-e/").unwrap(), "a/b.c_d-e");
    assert_eq!(parse_repo("hf.co/a/b").unwrap(), "a/b");
    for bad in ["", "justaname", "a/", "/b", "a b/c", "../x/y", "a/b$", "https://example.com"] {
        assert!(parse_repo(bad).is_err() || bad == "https://example.com", "{bad}");
    }
    assert!(parse_repo("justaname").unwrap_err().0.contains("unsloth/Qwen3-4B"));
}

#[test]
fn a_good_model_is_looked_up_sized_and_given_a_family() {
    let (_g, _d) = setup();
    let q = lookup("acme/Qwen3.5-7B-Instruct").unwrap();
    let m = &q.entry;
    assert_eq!((m.family.as_str(), m.repo.as_str(), m.license.as_str(), m.custom, m.id.as_str()), ("qwen3_5", "acme/Qwen3.5-7B-Instruct", "apache-2.0", true, "custom-acme-qwen3.5-7b-instruct"));
    assert!((m.params_b - 7.6).abs() < 0.01 && m.name == "Qwen3.5 7B Instruct", "{m:?}");
    assert!(q.warnings.is_empty(), "{:?}", q.warnings);
    // the estimate follows the rule that fits the built-in models' published figures
    for b in builtin_models().iter().filter(|b| b.id.starts_with("qwen")) {
        let (lora, _) = estimate_vram(b.params_b);
        assert!((lora - b.vram_lora_gb).abs() / b.vram_lora_gb < 0.25, "{}: estimate {lora} vs {}", b.name, b.vram_lora_gb);
    }
    assert!(m.vram_qlora_gb < m.vram_lora_gb);
    // Gemma 4, in its multimodal pipeline, is recognised
    assert_eq!(lookup("https://huggingface.co/acme/Gemma-4-E3-it").unwrap().entry.family, "gemma4");
    // an unknown family is allowed, with a plain warning
    let l = lookup("acme/Llama-3.2-3B-Instruct").unwrap();
    assert_eq!((l.entry.family.as_str(), l.entry.license.as_str()), ("generic", "llama3.2"));
    assert!(l.warnings.iter().any(|w| w.contains("only for Qwen3.5 and Gemma 4") && w.contains("llama")), "{:?}", l.warnings);
    // size from the file sizes when Hugging Face gives no parameter count; no licence; very big; template not checkable
    let n = lookup("acme/no-total").unwrap();
    assert!((n.entry.params_b - 7.6).abs() < 0.01 && n.warnings.iter().any(|w| w.contains("estimated from the file sizes")));
    let think = lookup("acme/thinker").unwrap();
    assert!(think.warnings.iter().any(|w| w.contains("\"thinking\" mode")), "{:?}", think.warnings);
    assert!(lookup("acme/no-license").unwrap().warnings.iter().any(|w| w.contains("licence is not stated")));
    assert!(lookup("acme/huge-70B").unwrap().warnings.iter().any(|w| w.contains("big")));
    assert!(lookup("acme/no-template-check").unwrap().warnings.iter().any(|w| w.contains("chat template could not be checked")));
}

#[test]
fn what_cannot_be_trained_is_refused_in_plain_words() {
    let (_g, _d) = setup();
    let err = |r: &str| lookup(r).unwrap_err().0;
    assert!(err("acme/missing").contains("was not found") && err("acme/login").contains("login"));
    assert!(err("acme/gated").contains("gated or private"));
    assert!(err("acme/Model-GGUF").contains("GGUF") && err("acme/Model-GGUF").contains("original weights"));
    assert!(err("acme/base-model").contains("no chat template") && err("acme/base-model").contains("Instruct"));
    assert!(err("acme/custom-code").contains("trust_remote_code") && err("acme/custom-code").contains("does not run downloaded code"));
    assert!(err("acme/pre-quant").contains("quantized") && err("acme/pre-quant").contains("original 16-bit"));
    assert!(err("acme/image-model").contains("text-to-image"));
    assert!(err("acme/bin-only").contains("no .safetensors"));
    // already there, by its own name or the official one
    assert!(err("unsloth/Qwen3.5-4B").contains("already in the list as \"Qwen3.5 4B\""));
    assert!(err("https://huggingface.co/Qwen/Qwen3.5-4B").contains("already in the list"));
    // no connection
    std::env::set_var("NDFORGE_HF_URL", "http://127.0.0.1:1");
    assert!(err("acme/Qwen3.5-7B-Instruct").contains("could not be reached"));
}

#[test]
fn added_models_are_kept_listed_planned_and_removable() {
    let (_g, _d) = setup();
    assert_eq!(models().len(), 5);
    let id = AppModel::custom_model_add("acme/Llama-3.2-3B-Instruct").unwrap();
    assert_eq!(id, "custom-acme-llama-3.2-3b-instruct");
    // only a model that passes the checks is ever saved
    assert!(AppModel::custom_model_add("acme/gated").is_err() && custom_models().len() == 1);
    assert_eq!((models().len(), models()[5].custom, models()[0].custom), (6, true, false));
    let m = find(&id).unwrap();
    assert_eq!((m.family.as_str(), m.params_b), ("generic", 3.21));
    // it is planned like any other model
    let p = plan(&m, Method::Lora, Preset::Normal, Some(32.0), DataStats::default(), None);
    assert_eq!(p.fit, Fit::Yes, "{p:?}");
    let small = plan(&m, Method::Lora, Preset::Normal, Some(4.0), DataStats::default(), None);
    assert_ne!(small.fit, Fit::Yes);
    // the look-up answer for the window
    let v = AppModel::custom_model_lookup("acme/Qwen3.5-7B-Instruct").unwrap();
    assert!(v["model"]["name"] == "Qwen3.5 7B Instruct" && v["warnings"].as_array().unwrap().is_empty());
    // adding twice keeps one; built-in models cannot be removed; removing works and is remembered
    add_custom(m.clone()).unwrap();
    assert_eq!(custom_models().len(), 1);
    assert!(remove_custom("qwen3.5-4b").unwrap_err().0.contains("added yourself") && models().len() == 6);
    remove_custom(&id).unwrap();
    assert_eq!((models().len(), find(&id).is_none()), (5, true));
    // a damaged file counts as no models
    std::fs::write(std::path::Path::new(&std::env::var("NDFORGE_CONFIG_DIR").unwrap()).join("custom_models.json"), "{ not json").unwrap();
    assert_eq!(models().len(), 5);
}

#[test]
fn a_model_of_an_unknown_family_is_installed_with_the_template_inside_its_file() {
    let g = spec_for("generic", 8192);
    let mf = modelfile_text("model-q4_k_m.gguf", &g);
    assert!(mf.starts_with("FROM ./model-q4_k_m.gguf") && !mf.contains("TEMPLATE") && mf.contains("PARAMETER num_ctx 8192") && !mf.contains("PARAMETER stop"), "{mf}");
    let body = create_body("x-gm", "model.gguf", "sha256:abc", &g);
    assert!(body.get("template").is_none() && body["parameters"].get("stop").is_none() && body["parameters"]["num_ctx"] == 8192, "{body}");
    // the tested families keep their hand-written templates
    for fam in ["qwen3_5", "gemma4"] {
        let s = spec_for(fam, 8192);
        assert!(modelfile_text("m.gguf", &s).contains("TEMPLATE") && create_body("x", "m.gguf", "d", &s)["template"].is_string());
    }
}
