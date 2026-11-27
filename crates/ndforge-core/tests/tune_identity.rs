//! Who a measurement belongs to: the shape (speed) and the weights (entry), the planned shape equal to the written one, and the machine key.
use ndforge_core::tune::identity::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn qwen_cfg() -> Value {
    json!({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "vision_config": {"depth": 4},
           "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 8, "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                           "vocab_size": 248320, "tie_word_embeddings": false, "full_attention_interval": 4, "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
                           "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"]}})
}

#[test]
fn key_order_and_whitespace_do_not_change_the_shape() {
    let a = qwen_cfg();
    let b: Value = serde_json::from_str(&format!("  {}  ", serde_json::to_string_pretty(&a).unwrap())).unwrap();
    assert_eq!(shape_of_config(&a).0, shape_of_config(&b).0);
    let reordered: Value = serde_json::from_str(r#"{"text_config": {"vocab_size": 248320, "hidden_size": 5120, "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"], "num_hidden_layers": 8, "model_type": "qwen3_5_text", "intermediate_size": 17408, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "tie_word_embeddings": false, "full_attention_interval": 4, "linear_key_head_dim": 128, "linear_num_value_heads": 48, "linear_num_key_heads": 16}, "vision_config": {"depth": 4}, "model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}"#).unwrap();
    assert_eq!(shape_of_config(&a).0, shape_of_config(&reordered).0);
    assert!(shape_of_config(&a).0.starts_with("s-") && shape_of_config(&a).0.len() == 18);
}

#[test]
fn another_width_depth_or_vocabulary_is_another_shape_and_a_learning_rate_is_not_in_it() {
    let base = shape_of_config(&qwen_cfg()).0;
    let mut w = qwen_cfg();
    w["text_config"]["intermediate_size"] = json!(11008);
    let mut d = qwen_cfg();
    d["text_config"]["num_hidden_layers"] = json!(7);
    let mut v = qwen_cfg();
    v["text_config"]["vocab_size"] = json!(150000);
    let mut noise = qwen_cfg();
    noise["transformers_version"] = json!("5.23.0");
    noise["text_config"]["initializer_range"] = json!(0.02);
    assert!(shape_of_config(&w).0 != base && shape_of_config(&d).0 != base && shape_of_config(&v).0 != base);
    assert_eq!(shape_of_config(&noise).0, base, "facts that do not change the speed are not part of the shape");
    let f = shape_of_config(&qwen_cfg()).1;
    assert_eq!((f.layers, f.hidden, f.vocab, f.intermediate.clone(), f.arch.as_str()), (8, 5120, 248320, vec![17408], "Qwen3_5ForConditionalGeneration"));
}

fn model_dir(root: &std::path::Path, name: &str, cfg: &Value, weights: &[(&str, u8)]) -> std::path::PathBuf {
    let d = root.join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("config.json"), serde_json::to_vec_pretty(cfg).unwrap()).unwrap();
    for (file, fill) in weights {
        let header = json!({"w": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(std::iter::repeat(*fill).take(32));
        std::fs::write(d.join(file), bytes).unwrap();
    }
    d
}

#[test]
fn a_healed_copy_keeps_the_shape_and_changes_the_weights_id_and_times_do_not_matter() {
    let t = tempfile::tempdir().unwrap();
    let a = model_dir(t.path(), "cut", &qwen_cfg(), &[("model-00001.safetensors", 1)]);
    let b = model_dir(t.path(), "healed", &qwen_cfg(), &[("model-00001.safetensors", 1)]);
    let header_other = {
        let d = model_dir(t.path(), "healed2", &qwen_cfg(), &[("model-00001.safetensors", 1)]);
        // a different tensor layout in the header: another weights id, the same shape
        let header = json!({"w": {"dtype": "BF16", "shape": [2, 8], "data_offsets": [0, 32]}}).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(std::iter::repeat(9u8).take(32));
        std::fs::write(d.join("model-00001.safetensors"), bytes).unwrap();
        d
    };
    let ia = model_identity(&a, "cut", vec!["Qwen/Qwen3.8-27B".into()]).unwrap();
    let ib = model_identity(&b, "copy", vec![]).unwrap();
    let ic = model_identity(&header_other, "healed", vec!["cut".into()]).unwrap();
    assert_eq!(ia.weights_id, ib.weights_id, "a copy of the same files is the same entry, modification times are ignored");
    assert_eq!(ia.shape_id, ic.shape_id);
    assert_ne!(ia.weights_id, ic.weights_id);
    assert_eq!((ia.layers, ia.hidden, ia.lineage.clone()), (8, 5120, vec!["Qwen/Qwen3.8-27B".to_string()]));
    assert!(ia.params > 0 && ia.source_bytes > 0);
    std::fs::write(b.join("model-00001.safetensors"), std::fs::read(b.join("model-00001.safetensors")).unwrap()).unwrap();          // a new mtime, the same bytes
    assert_eq!(model_identity(&b, "copy", vec![]).unwrap().weights_id, ia.weights_id);
    assert!(model_identity(&t.path().join("nothing"), "x", vec![]).is_err());
}

#[test]
fn the_planned_shape_equals_the_shape_of_the_config_the_cut_writes() {
    let orig = qwen_cfg();
    // a depth cut: keep layers 0..4 and 6, 7 (the worker remaps layer_types and sets num_hidden_layers)
    let kept = [0usize, 1, 2, 3, 6, 7];
    let mut written = orig.clone();
    written["text_config"]["num_hidden_layers"] = json!(kept.len());
    let lt = orig["text_config"]["layer_types"].as_array().unwrap();
    written["text_config"]["layer_types"] = json!(kept.iter().map(|i| lt[*i].clone()).collect::<Vec<_>>());
    assert_eq!(shape_of_plan(&orig, &kept, None), shape_of_config(&written).0);
    // a width cut as well
    written["text_config"]["intermediate_size"] = json!(11008);
    assert_eq!(shape_of_plan(&orig, &kept, Some(11008)), shape_of_config(&written).0);
    assert_ne!(shape_of_plan(&orig, &kept, Some(11008)), shape_of_plan(&orig, &kept, None));
    // no cut at all: the original's own shape
    assert_eq!(shape_of_plan(&orig, &(0..8).collect::<Vec<_>>(), None), shape_of_config(&orig).0);
}

#[test]
fn the_machine_key_is_the_card_and_its_memory_only() {
    let mut sw = BTreeMap::new();
    sw.insert("torch".to_string(), "2.14.1".to_string());
    let a = machine_from("NVIDIA GeForce RTX 5090", 32607, "621.24", "windows 11", sw.clone());
    sw.insert("torch".to_string(), "2.15.0".to_string());
    let b = machine_from(" nvidia geforce rtx 5090 ", 32607, "630.01", "windows 11", sw);
    assert_eq!(a.key, b.key, "a new driver or torch is the same machine; the preset marks those results stale");
    assert_ne!(a.key, machine_from("NVIDIA GeForce RTX 4090", 24564, "621.24", "windows 11", BTreeMap::new()).key);
    assert_ne!(a.key, machine_from("NVIDIA GeForce RTX 5090", 16000, "621.24", "windows 11", BTreeMap::new()).key);
    assert_eq!((minor_of("2.14.1+cu130"), minor_of("621.24"), major_of("621.24"), major_of("v3.8.0")), ("2.14".to_string(), "621.24".to_string(), "621".to_string(), "3".to_string()));
}

#[test]
fn a_cut_of_a_mixture_of_experts_model_is_another_shape_and_dense_shapes_do_not_move() {
    // the real Inkling-2-Mini config (worker/tests/data/inkling2_config.json)
    let cfg: Value = serde_json::from_str(include_str!("../../../worker/tests/data/inkling2_config.json")).unwrap();
    let base = shape_of_config(&cfg).0;
    let mut e = cfg.clone();
    e["text_config"]["n_routed_experts"] = json!(96);
    let mut k = cfg.clone();
    k["text_config"]["num_experts_per_tok"] = json!(6);
    let mut p = cfg.clone();
    p["text_config"]["loop_config"]["num_passes"] = json!(2);
    let mut noise = cfg.clone();
    noise["dtype"] = json!("float32");
    assert!([&e, &k, &p].iter().all(|c| shape_of_config(c).0 != base), "experts, experts per token and loop passes are part of the shape");
    assert_eq!(shape_of_config(&noise).0, base);
    // a dense model that happens to name a per-token count does not get it into its shape (so every saved shape id stays what it was)
    let mut dense = qwen_cfg();
    let before = shape_of_config(&dense).0;
    dense["text_config"]["num_experts_per_tok"] = json!(4);
    assert_eq!(shape_of_config(&dense).0, before);
}
