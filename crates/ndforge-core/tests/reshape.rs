//! Slim or grow a model: the arithmetic (`reshape::plan`), then the runs (`reshape::runs`) with a stand-in worker.
use ndforge_core::reshape::plan::*;
use serde_json::json;

/// A Qwen3.5-27B-like model: 64 layers in blocks of 4, 0.39 B parameters each, 2.84 B elsewhere (27.8 B in all)
fn big() -> Inspect {
    inspect(64, 4, 390_000_000, 2_840_000_000)
}

/// A Qwen3.5-9B-like model: 32 layers, 0.22 B each, 2.56 B elsewhere (9.6 B)
fn nine() -> Inspect {
    inspect(32, 4, 220_000_000, 2_560_000_000)
}

fn inspect(layers: u32, period: u32, per_layer: u64, other: u64) -> Inspect {
    Inspect {
        model_type: "qwen3_5".into(), supported: true, why: None, layers, layer_types: vec![], period, blocks: layers / period,
        layer_params: vec![per_layer; layers as usize], embed_params: other / 2, lm_head_params: 0, vision_params: other / 4, mtp_params: other / 8, other_params: other / 8,
        total_params: per_layer * layers as u64 + other, bytes_per_param: 2.0, has_vision: true, tied: true, source_bytes: 0, local: true,
        ..Default::default()
    }
}

fn spec(mode: Mode, blocks: u32) -> ReshapeSpec {
    serde_json::from_value(json!({"source_model_id": "m", "mode": mode.key(), "blocks": blocks})).unwrap()
}

#[test]
fn slimming_the_27b_by_four_blocks_gives_48_layers_and_about_21_6_billion() {
    let ins = big();
    assert_eq!(ins.total_params, 27_800_000_000);
    let opts = options(&ins, Mode::Prune, Some(32.0));
    assert_eq!(opts.len(), 8, "at most half the blocks (16 blocks -> 8)");
    let four = opts.iter().find(|o| o.blocks == 4).unwrap();
    assert_eq!(four.layers, 48);
    assert!((four.params_b - 21.56).abs() < 0.01, "{four:?}");
    assert!((four.disk_gb - 43.1).abs() < 0.1, "{four:?}");
    // training: 16-bit LoRA does not fit a 32 GB card, QLoRA does
    assert_eq!((four.fits_lora, four.fits_qlora), (Some(false), Some(true)), "{four:?}");
    assert_eq!(opts[0].blocks, 1);
    assert_eq!(opts[0].layers, 60);
    assert_eq!(options(&ins, Mode::Prune, None)[0].fits_lora, None);
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 21.0), 4);
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 27.7), 1);
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 1.0), 8, "clamped to the limit");
}

#[test]
fn growing_the_9b_by_four_blocks_is_exact() {
    let ins = nine();
    let opts = options(&ins, Mode::Grow, Some(32.0));
    assert_eq!(opts.len(), 4);
    let four = opts.iter().find(|o| o.blocks == 4).unwrap();
    assert_eq!(four.layers, 48);
    // the layers 8..24 appear twice: 16 layers of 0.22 B
    assert_eq!(params_after(&ins, Mode::Grow, 4), ins.total_params + 16 * 220_000_000);
    assert!((four.params_b - 13.12).abs() < 0.001, "{four:?}");
    assert_eq!(blocks_for_target(&ins, Mode::Grow, 13.2), 4);
    assert_eq!(blocks_for_target(&ins, Mode::Grow, 9.7), 1);
    assert_eq!(params_after(&ins, Mode::Grow, 1), ins.total_params + 4 * 220_000_000);
}

#[test]
fn unsupported_or_too_small_models_have_no_options() {
    let mut ins = big();
    ins.supported = false;
    ins.why = Some("nope".into());
    assert!(options(&ins, Mode::Prune, None).is_empty());
    assert_eq!(blocks_for_target(&ins, Mode::Prune, 20.0), 0);
    let small = inspect(8, 4, 1, 1);          // two blocks, both protected
    assert!(options(&small, Mode::Prune, None).is_empty());
    assert_eq!(options(&small, Mode::Grow, None).len(), 1);
}

#[test]
fn the_model_is_loaded_for_measuring_the_way_that_fits() {
    assert_eq!(scoring_load(27_800_000_000, 2.0, Some(32.0)), "4bit");
    assert_eq!(scoring_load(9_600_000_000, 2.0, Some(32.0)), "16bit");
    assert_eq!(scoring_load(27_800_000_000, 2.0, None), "4bit");
    assert_eq!(scoring_load(120_000_000_000, 2.0, Some(24.0)), "offload");
}

#[test]
fn limits_are_the_same_as_the_workers() {
    let table: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/reshape_limits.json")).unwrap()).unwrap();
    let cases = table.as_array().unwrap();
    assert!(cases.len() >= 8);
    for c in cases {
        let ins = Inspect { blocks: c["blocks"].as_u64().unwrap() as u32, supported: true, ..Default::default() };
        let mode = if c["mode"] == "prune" { Mode::Prune } else { Mode::Grow };
        let got = limits(&ins, mode, c["pf"].as_u64().unwrap() as u32, c["pl"].as_u64().unwrap() as u32);
        assert_eq!(got, (c["min"].as_u64().unwrap() as u32, c["max"].as_u64().unwrap() as u32), "{c}");
    }
}

#[test]
fn validate_refuses_in_plain_words() {
    let ins = big();
    assert!(validate(&spec(Mode::Prune, 4), &ins).is_ok());
    assert!(validate(&spec(Mode::Grow, 8), &ins).is_ok());
    // unsupported: the model's own reason
    let mut bad = big();
    bad.supported = false;
    bad.why = Some("Gemma 4 is not supported yet.".into());
    assert_eq!(validate(&spec(Mode::Prune, 1), &bad).unwrap_err().0, "Gemma 4 is not supported yet.");
    // too many / none
    assert!(validate(&spec(Mode::Prune, 9), &ins).unwrap_err().0.contains("at most half"));
    assert!(validate(&spec(Mode::Prune, 0), &ins).unwrap_err().0.contains("at most half"));
    assert!(validate(&spec(Mode::Grow, 9), &ins).unwrap_err().0.contains("Grow by at most half"));
    // unknown settings
    let mut s = spec(Mode::Prune, 2);
    s.strategy = "wild".into();
    assert!(validate(&s, &ins).is_err());
    let mut s = spec(Mode::Grow, 2);
    s.init = "random".into();
    assert!(validate(&s, &ins).is_err());
    let mut s = spec(Mode::Prune, 2);
    s.load = "8bit".into();
    assert!(validate(&s, &ins).is_err());
    let mut s = spec(Mode::Prune, 2);
    s.calib_samples = 1;
    assert!(validate(&s, &ins).is_err());
}

#[test]
fn explicit_layers_must_be_whole_unprotected_blocks() {
    let ins = big();
    let with = |d: Vec<u32>| ReshapeSpec { drop_layers: Some(d), ..spec(Mode::Prune, 1) };
    assert!(validate(&with((36..52).collect()), &ins).is_ok());                  // four whole blocks
    assert!(validate(&with(vec![36, 37, 38, 39]), &ins).is_ok());
    let e = validate(&with(vec![3, 4, 5, 6]), &ins).unwrap_err().0;
    assert!(e.contains("Layers 3 to 6 do not form whole blocks of 4"), "{e}");
    assert!(validate(&with(vec![0, 1, 2, 3]), &ins).unwrap_err().0.contains("protected"));
    assert!(validate(&with(vec![60, 61, 62, 63]), &ins).unwrap_err().0.contains("protected"));
    assert!(validate(&with(vec![64]), &ins).unwrap_err().0.contains("between 0 and 63"));
    assert!(validate(&with(vec![]), &ins).is_err());
    assert!(validate(&with((4..44).collect()), &ins).unwrap_err().0.contains("at most half"));
    let grow = ReshapeSpec { drop_layers: Some(vec![4, 5, 6, 7]), ..spec(Mode::Grow, 1) };
    assert!(validate(&grow, &ins).unwrap_err().0.contains("only for slimming"));
}

#[test]
fn warnings_are_honest() {
    let ins = big();
    let w = warnings(&spec(Mode::Prune, 4), &ins, Some(32.0));
    assert!(w.iter().any(|x| x.contains("worse than the original until it is trained again")), "{w:?}");
    assert!(w.iter().any(|x| x.contains("QLoRA") && x.contains("32 GB")), "{w:?}");
    assert!(!w.iter().any(|x| x.contains("quarter")), "4 of 16 blocks is exactly a quarter");
    assert!(warnings(&spec(Mode::Prune, 5), &ins, None).iter().any(|x| x.contains("More than a quarter")));
    let g = warnings(&spec(Mode::Grow, 4), &nine(), Some(32.0));
    assert!(g.iter().any(|x| x.contains("little or no gain") && x.contains("37 % more")), "{g:?}");
    let none = warnings(&spec(Mode::Prune, 4), &big(), Some(12.0));
    assert!(none.iter().any(|x| x.contains("Even QLoRA")), "{none:?}");
}

#[test]
fn disk_needed_counts_the_download_for_a_model_that_is_not_here() {
    let mut ins = big();
    let local = disk_needed_gb(&ins, Mode::Prune, 4);
    assert!((local - 47.4).abs() < 0.05, "{local}");        // 21.56 B * 2 bytes * 1.1
    ins.local = false;
    assert!((disk_needed_gb(&ins, Mode::Prune, 4) - (47.4 + 55.6)).abs() < 0.2);
    assert!(disk_needed_gb(&nine(), Mode::Grow, 4) > disk_needed_gb(&nine(), Mode::Grow, 1));
}

#[test]
fn the_inspect_result_of_the_worker_parses() {
    let v = json!({"model_type": "qwen3_5", "supported": true, "why": null, "layers": 8, "layer_types": ["linear_attention"], "period": 4, "blocks": 2, "layer_params": [1, 2],
        "embed_params": 3, "lm_head_params": 0, "vision_params": 4, "mtp_params": 5, "other_params": 6, "total_params": 21, "bytes_per_param": 2.0, "has_vision": true,
        "tied": true, "source_bytes": 42, "local": true, "extra_key_from_a_newer_worker": 1});
    let ins: Inspect = serde_json::from_value(v).unwrap();
    assert_eq!((ins.layers, ins.blocks, ins.total_params, ins.source_bytes), (8, 2, 21, 42));
    let s: ReshapeSpec = serde_json::from_value(json!({"source_model_id": "x", "mode": "grow", "blocks": 2})).unwrap();
    assert_eq!((s.strategy.as_str(), s.protect_first, s.protect_last, s.init.as_str(), s.calib_samples, s.calib_len, s.load.as_str()), ("contiguous", 1, 1, "identity", 128, 512, "auto"));
}

// ------------------------------------------------------------------------------------------------------- thinner layers (width)

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data");

fn dims_of(tc: &serde_json::Value) -> WidthDims {
    let u = |k: &str| tc[k].as_u64().unwrap_or(0) as u32;
    WidthDims {
        hidden: u("hidden_size"), intermediate: u("intermediate_size"), heads: u("num_attention_heads"), kv_heads: u("num_key_value_heads"), head_dim: u("head_dim"),
        lin_k_heads: u("linear_num_key_heads"), lin_v_heads: u("linear_num_value_heads"), lin_k_dim: u("linear_key_head_dim"), lin_v_dim: u("linear_value_head_dim"),
        conv_kernel: u("linear_conv_kernel_dim"), mtp_layers: u("mtp_num_hidden_layers"), vocab: u("vocab_size"), tied: tc["tie_word_embeddings"].as_bool().unwrap_or(false),
    }
}

/// The real Qwen3.5-27B (config from the Hugging Face Hub, in the worker's test data): 64 layers, MLP 17,408, 48 value heads, 24 heads
fn qwen27() -> Inspect {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{DATA}/qwen3_5_27b_config.json")).unwrap()).unwrap();
    let tc = &cfg["text_config"];
    let d = dims_of(tc);
    let types: Vec<String> = tc["layer_types"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    let div = 256;
    Inspect {
        model_type: "qwen3_5".into(), supported: true, layers: 64, layer_types: types, period: 4, blocks: 16, vision_params: 460_730_096, other_params: d.hidden as u64,
        total_params: 27_781_427_952, bytes_per_param: 2.0, has_vision: true, local: true, width_supported: true,
        width_choices: Some(WidthChoices { intermediate: (34..=68).map(|k| k * div).collect(), lin_v_heads: vec![48, 32], heads: vec![24, 20, 16] }),
        width_dims: Some(d), ..Default::default()
    }
}

fn wspec(blocks: u32, intermediate: u32) -> ReshapeSpec {
    ReshapeSpec { width: Some(Width { intermediate, lin_v_heads: None, heads: None }), ..spec(Mode::Prune, blocks) }
}

#[test]
fn width_parameter_counts_equal_the_workers_table() {
    let table: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{DATA}/width_params.json")).unwrap()).unwrap();
    let cases = table.as_array().unwrap();
    assert!(cases.len() >= 8);
    for c in cases {
        let d: WidthDims = serde_json::from_value(c["dims"].clone()).unwrap();
        let w = WidthSizes { intermediate: c["width"]["intermediate"].as_u64().unwrap() as u32, lin_v_heads: c["width"]["lin_v_heads"].as_u64().unwrap() as u32, heads: c["width"]["heads"].as_u64().unwrap() as u32 };
        let got = width_params_dims(&d, c["n_lin"].as_u64().unwrap(), c["n_full"].as_u64().unwrap(), &w, c["vision"].as_u64().unwrap(), c["other"].as_u64().unwrap());
        assert_eq!(got, c["params"].as_u64().unwrap(), "{}", c["name"]);
    }
}

#[test]
fn the_27b_gets_one_exact_stop_per_mlp_width() {
    let ins = qwen27();
    let full = Width { intermediate: 17408, lin_v_heads: None, heads: None };
    assert_eq!(width_params(&ins, &(0..64).collect::<Vec<_>>(), &full), 27_781_427_952, "the original's own size, from the formula");
    let opts = width_options(&ins, 0, Some(32.0), None, None);
    assert_eq!(opts.len(), 35);
    assert_eq!((opts[0].intermediate, opts[34].intermediate), (Some(8704), Some(17408)), "ascending: the thinnest first");
    let at = |i: u32| opts.iter().find(|o| o.intermediate == Some(i)).unwrap();
    assert_eq!((at(13312).params_b, at(10752).params_b), (23.69, 21.14));
    assert!(at(13312).exact && at(10752).layers == 64 && at(10752).blocks == 0);
    assert!((at(10752).disk_gb - 42.3).abs() < 0.05, "{:?}", at(10752));
    assert_eq!((at(10752).fits_lora, at(10752).fits_qlora), (Some(false), Some(true)));
    // both: one block less (60 layers) and 12,288 = 21.46 B; two blocks less and 13,312 = 21.15 B
    let both = width_options(&ins, 1, None, None, None);
    let b = both.iter().find(|o| o.intermediate == Some(12288)).unwrap();
    assert_eq!((b.layers, b.blocks, b.params_b, b.fits_qlora), (60, 1, 21.46, None));
    assert_eq!(width_options(&ins, 2, None, None, None).iter().find(|o| o.intermediate == Some(13312)).unwrap().params_b, 21.15);
    // the Expert axes change the size: 32 value heads and 16 attention heads
    assert_eq!(width_options(&ins, 0, None, Some(32), Some(16)).iter().find(|o| o.intermediate == Some(13312)).unwrap().params_b, 21.64);
    assert_eq!(width_options(&ins, 0, None, Some(32), None).iter().find(|o| o.intermediate == Some(12288)).unwrap().params_b, 21.15);
    // no options for a model that cannot be thinned
    let mut no = qwen27();
    no.width_supported = false;
    assert!(width_options(&no, 0, None, None, None).is_empty());
    assert!(width_options(&big(), 0, None, None, None).is_empty());
}

#[test]
fn the_suggested_first_position_is_the_mild_cut() {
    assert_eq!(suggested_widths(&qwen27()), Some((13312, (1, 12288))));
    assert_eq!(suggested_widths(&big()), None);
}

#[test]
fn validate_checks_the_width_in_plain_words() {
    let ins = qwen27();
    assert!(validate(&wspec(0, 13312), &ins).is_ok(), "width only: no blocks removed");
    assert!(validate(&wspec(1, 12288), &ins).is_ok(), "both");
    let mut s = wspec(0, 13312);
    s.drop_layers = Some(vec![4, 5, 6, 7]);
    assert!(validate(&s, &ins).is_ok(), "explicit layers and a width");
    let e = |s: &ReshapeSpec| validate(s, &ins).unwrap_err().0;
    assert!(e(&wspec(0, 13313)).contains("multiple of 256 between half and all of 17,408"));
    assert!(e(&wspec(0, 8192)).contains("multiple of 256"));
    assert!(e(&wspec(0, 17664)).contains("multiple of 256"));
    assert!(e(&wspec(0, 17408)).contains("nothing to thin"));
    assert!(e(&wspec(9, 13312)).contains("at most half"));
    let mut s = wspec(0, 13312);
    s.width.as_mut().unwrap().lin_v_heads = Some(40);
    assert!(e(&s).contains("from 48 to 32"));
    s.width = Some(Width { intermediate: 17408, lin_v_heads: Some(32), heads: None });
    assert!(validate(&s, &ins).is_ok(), "an Expert axis alone is allowed");
    s.width = Some(Width { intermediate: 13312, lin_v_heads: None, heads: Some(18) });
    assert!(e(&s).contains("multiple of 4 and at least 16"));
    let g = ReshapeSpec { width: Some(Width { intermediate: 13312, lin_v_heads: None, heads: None }), ..spec(Mode::Grow, 1) };
    assert!(e(&g).contains("only possible when slimming"));
    let mut small = wspec(0, 13312);
    small.width_samples = 2;
    assert!(e(&small).contains("Measure on"));
    // a model that cannot be thinned says why; a prune without width still needs blocks
    let mut no = qwen27();
    no.width_supported = false;
    no.width_why = Some("Thinning the layers is only available for dense Qwen3.5 / Qwen3.8 models so far. You can still remove whole layers.".into());
    assert!(validate(&wspec(0, 13312), &no).unwrap_err().0.contains("dense Qwen3.5"));
    assert!(e(&spec(Mode::Prune, 0)).contains("at most half"));
    assert!(validate(&spec(Mode::Prune, 4), &ins).is_ok());
}

#[test]
fn width_warnings_are_honest() {
    let ins = qwen27();
    let w = warnings(&wspec(0, 10752), &ins, Some(32.0));
    assert!(w.iter().any(|x| x.contains("Removing 38 % of the MLP") && x.contains("healed")), "{w:?}");
    assert!(w.iter().any(|x| x.contains("Healing is strongly recommended")));
    assert!(w.iter().any(|x| x.contains("needs QLoRA")), "{w:?}");
    assert!(!w.iter().any(|x| x.contains("experimental")));
    let mild = warnings(&wspec(0, 13312), &ins, None);
    assert!(!mild.iter().any(|x| x.contains("Removing")) && mild.iter().any(|x| x.contains("Healing is strongly recommended")), "{mild:?}");
    let mut s = wspec(0, 13312);
    s.width.as_mut().unwrap().heads = Some(16);
    assert!(warnings(&s, &ins, None).iter().any(|x| x.contains("experimental")));
}

#[test]
fn healing_runs_the_way_that_fits_the_card() {
    let p = heal_plan(21.1, 27.8, Some(32.0));
    assert_eq!((p.mode, p.student_method, p.fits), ("cache", "qlora", true), "{p:?}");
    assert!((p.est_gb - 18.7).abs() < 0.2 && p.why.is_none());
    let q = heal_plan(8.0, 9.65, Some(32.0));
    assert_eq!((q.mode, q.student_method, q.teacher_load, q.fits), ("online", "lora", "4bit", true), "{q:?}");
    let small = heal_plan(21.1, 27.8, Some(16.0));
    assert!(!small.fits && small.why.as_deref().unwrap().contains("this card has 16 GB") && small.why.as_deref().unwrap().contains("Choose a smaller model"), "{small:?}");
    assert_eq!(heal_plan(21.1, 27.8, None).mode, "cache");
    assert!(p.cache_gb_per_10m > 1.9 && p.cache_gb_per_10m < 2.0, "196 bytes per token, 10 M tokens");
    assert!(p.hours_per_10m.0 > 0.0 && p.hours_per_10m.1 > p.hours_per_10m.0);
    assert_eq!((default_heal_tokens_m(21.1), default_heal_tokens_m(8.0), default_heal_tokens_m(15.0)), (20, 40, 20));
    assert_eq!(HEAL_TOKENS_M, [10, 20, 40, 80]);
}

#[test]
fn the_width_fields_of_the_worker_parse_and_old_answers_still_do() {
    let v = json!({"model_type": "qwen3_5", "supported": true, "why": null, "layers": 8, "period": 4, "blocks": 2, "total_params": 21,
        "width_dims": {"hidden": 8, "intermediate": 32, "heads": 6, "kv_heads": 2, "head_dim": 4, "lin_k_heads": 2, "lin_v_heads": 4, "lin_k_dim": 4, "lin_v_dim": 4,
                       "conv_kernel": 4, "mtp_layers": 1, "vocab": 16, "tied": false},
        "width_supported": true, "width_why": null, "width_choices": {"intermediate": [16, 32], "lin_v_heads": [4, 2], "heads": [6, 4]}});
    let ins: Inspect = serde_json::from_value(v).unwrap();
    assert!(ins.width_supported && ins.width_dims.as_ref().unwrap().intermediate == 32 && ins.width_choices.as_ref().unwrap().heads == vec![6, 4]);
    let old: Inspect = serde_json::from_value(json!({"supported": true, "layers": 8, "period": 4, "blocks": 2, "total_params": 21})).unwrap();
    assert!(!old.width_supported && old.width_dims.is_none() && old.width_choices.is_none());
    let s: ReshapeSpec = serde_json::from_value(json!({"source_model_id": "x", "mode": "prune", "blocks": 0, "width": {"intermediate": 13312}})).unwrap();
    assert_eq!((s.width.clone().unwrap(), s.width_samples, shape_of(&s)), (Width { intermediate: 13312, lin_v_heads: None, heads: None }, 256, Shape::Width));
    assert_eq!(shape_of(&ReshapeSpec { blocks: 1, ..s.clone() }), Shape::Both);
    assert_eq!(shape_of(&ReshapeSpec { width: None, ..s }), Shape::Depth);
    assert_eq!(serde_json::to_value(Shape::Both).unwrap(), json!("both"));
}

// ------------------------------------------------------------------------------------------------------------------ runs
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::project::Project;
use ndforge_core::reshape::runs as rruns;
use ndforge_core::sample;
use ndforge_core::train::catalog;
use ndforge_core::train::runs as truns;
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn project(root: &std::path::Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

fn project_with_version(root: &std::path::Path) -> Project {
    let mut p = project(root);
    let f = root.join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    p
}

/// What `reshape_inspect` answers for the fake worker: a 12-layer model in 3 blocks of 4
const INSPECT: &str = r#"{"model_type":"qwen3_5","supported":true,"why":null,"layers":12,"layer_types":[],"period":4,"blocks":3,"layer_params":[100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000,100000000],"embed_params":500000000,"lm_head_params":0,"vision_params":100000000,"mtp_params":0,"other_params":0,"total_params":1800000000,"bytes_per_param":2.0,"has_vision":true,"tied":true,"source_bytes":3600000000,"local":false}"#;

/// A fake Python: answers `reshape_inspect` itself, and for `reshape` runs `body` (shell) with $out, $final, $dir (the model's folder) and $job (the job file) set.
#[cfg(unix)]
fn fake_worker(root: &std::path::Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    let script = format!(r#"#!/bin/sh
job="$5"
if grep -q '"job":"reshape_inspect"' "$job"; then
  echo '{{"type":"result","data":{INSPECT}}}'
  exit 0
fi
out=$(sed -n 's/.*"out":"\([^"]*\)".*/\1/p' "$job")
final=$(sed -n 's/.*"final":"\([^"]*\)".*/\1/p' "$job")
dir=$(dirname "$out")
cp "$job" "$dir/job-seen.json"
{body}
"#);
    std::fs::write(&py, script).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

/// Writes a small model folder into $out, then renames it to $final and reports a result (what the real worker does)
#[cfg(unix)]
const MAKES_A_MODEL: &str = r#"mkdir -p "$out"
echo '{"type":"progress","step":"write","value":0.5,"text":"Writing the new model"}'
echo '{"type":"reshape_scores","boundary_distance":[0.1,0.05,0.2],"chosen_blocks":[1],"period":4,"stable":true,"matrix":[[0.0]]}'
echo '{"model_type":"qwen3_5","architectures":["Qwen3_5ForCausalLM"],"num_hidden_layers":8}' > "$out/config.json"
echo '{"chat_template":"{{ messages }}"}' > "$out/tokenizer_config.json"
printf -- '---\nlicense: apache-2.0\n---\n' > "$out/README.md"
echo '{"format":1,"origin":{"repo":"unsloth/Qwen3.5-4B","layers":12,"params":1800000000},"mode":"prune","layers":8,"params":1240000000,"zero_init":[]}' > "$out/ndforge_reshape.json"
head -c 2000 /dev/zero > "$out/model.safetensors"
mv "$out" "$final"
echo '{"type":"result","data":{"model_dir":"x","mode":"prune","layers_before":12,"layers_after":8,"src_layers":[0,1,2,3,8,9,10,11],"dropped":[4,5,6,7],"duplicated":[],"zero_init":[],"params_before":1800000000,"params_after":1240000000,"bytes_written":2480000000,"loss_before":1.5,"loss_after":2.0,"ppl_before":4.48,"ppl_after":7.39,"stable":true,"boundary_distance":[0.1,0.05,0.2],"chosen_blocks":[1],"load":"16bit","seconds":1.0,"notes":[],"stopped":false}}'"#;

fn drop_spec() -> ReshapeSpec {
    ReshapeSpec { drop_layers: Some(vec![4, 5, 6, 7]), ..spec(Mode::Prune, 1) }.with_model("qwen3.5-4b")
}

trait WithModel {
    fn with_model(self, id: &str) -> Self;
}

impl WithModel for ReshapeSpec {
    fn with_model(mut self, id: &str) -> Self {
        self.source_model_id = id.into();
        self
    }
}

#[cfg(unix)]
#[test]
fn a_model_is_made_registered_and_listed() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    let p = project(d.path());
    assert!(p.folder.join("models").is_dir(), "a new project has a models folder");
    let mut events = Vec::new();
    let result = rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
    let name = result["name"].as_str().unwrap().to_string();
    let id = result["model_id"].as_str().unwrap().to_string();
    assert!(name.ends_with("-qwen3.5-4b-slim-8L"), "{name}");
    assert_eq!(id, format!("custom-local-{name}"));
    let dir = p.folder.join("models").join(&name);
    assert!(dir.join("model/config.json").is_file() && !dir.join("model.partial").exists());
    let state: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("state.json")).unwrap()).unwrap();
    assert_eq!((state["status"].as_str(), state["model_id"].as_str(), state["source_model_id"].as_str()), (Some("done"), Some(id.as_str()), Some("qwen3.5-4b")));
    assert!(dir.join("result.json").is_file() && dir.join("scores.json").is_file());
    // the job the worker got
    let job: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("job-seen.json")).unwrap()).unwrap();
    assert_eq!((job["mode"].as_str(), job["drop_layers"].clone(), job["check"].clone(), job["source"]["repo"].as_str()), (Some("prune"), json!([4, 5, 6, 7]), json!(true), Some("unsloth/Qwen3.5-4B")));
    assert_eq!(job["final"].as_str().unwrap(), dir.join("model").to_string_lossy());
    assert!(job["out"].as_str().unwrap().ends_with("model.partial") && job["grow"]["init"] == "identity");
    // events reach the window
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["step"] == "write") && events.iter().any(|e| e["kind"] == "reshape_scores" && e["chosen_blocks"] == json!([1])));
    // the model list has it, with where it came from and its exact size
    let m = catalog::find(&id).expect("registered");
    assert_eq!(m.derived_from.as_deref(), Some("unsloth/Qwen3.5-4B"));
    assert_eq!(m.reshape.as_deref(), Some("slimmed: 12 → 8 layers"));
    assert!((m.params_b - 1.24).abs() < 1e-9, "{m:?}");
    assert_eq!((m.family.as_str(), m.custom, m.license.as_str()), ("qwen3_5", true, "apache-2.0"));
    assert!(m.name.starts_with("Qwen3.5 4B") && m.name.ends_with("slim 8 layers"), "{}", m.name);
    assert_eq!(m.repo, dir.join("model").canonicalize().unwrap().to_string_lossy());
    assert!(m.good_for.contains("unsloth/Qwen3.5-4B"));
    // registering again changes nothing
    assert_eq!(rruns::register(&p, &name).unwrap(), id);
    assert_eq!(catalog::custom_models().iter().filter(|x| x.id == id).count(), 1);
    // the list of made models
    let list = rruns::list(&p);
    assert_eq!(list.len(), 1);
    assert_eq!((list[0]["status"].as_str(), list[0]["tested"].clone(), list[0]["has_model"].clone()), (Some("done"), json!(false), json!(true)));
    assert_eq!((list[0]["summary"]["layers_after"].clone(), list[0]["summary"]["dropped"].clone(), list[0]["summary"]["loss_after"].clone()), (json!(8), json!([4, 5, 6, 7]), json!(2.0)));
    assert_eq!(list[0]["scores"]["chosen_blocks"], json!([1]));
    assert!(list[0]["test"].is_null());
    // what the model was found to be is remembered for the next time
    assert!(d.path().join("cfg/reshape_inspect.json").is_file());
}

#[cfg(unix)]
#[test]
fn a_failing_job_leaves_nothing_behind() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), "mkdir -p \"$out\"\necho half > \"$out/model.safetensors\"\necho '{\"type\":\"error\",\"code\":\"disk_full\",\"message\":\"There is not enough free space: 5 GB needed.\"}'\nexit 1");
    let p = project(d.path());
    let e = rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("not enough free space"), "{e}");
    let list = rruns::list(&p);
    assert_eq!((list.len(), list[0]["status"].as_str()), (1, Some("failed")));
    assert!(list[0]["error"].as_str().unwrap().contains("free space"));
    let dir = p.folder.join("models").join(list[0]["name"].as_str().unwrap());
    assert!(!dir.join("model.partial").exists() && !dir.join("model").exists());
    assert!(catalog::custom_models().is_empty());
}

#[cfg(unix)]
#[test]
fn stopping_asks_politely_and_then_kills() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // a worker that honours the STOP file
    fake_worker(d.path(), "mkdir -p \"$out\"\nwhile [ ! -f \"$dir/STOP\" ]; do sleep 0.05; done\nrm -rf \"$out\"\necho '{\"type\":\"result\",\"data\":{\"stopped\":true,\"model_dir\":null}}'");
    let p = project(d.path());
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(600));
        s2.store(true, Ordering::SeqCst);
    });
    let r = rruns::execute(&p, &drop_spec(), stop, &mut |_| {}).unwrap();
    t.join().unwrap();
    assert_eq!(r["stopped"], true);
    let list = rruns::list(&p);
    assert_eq!(list[0]["status"], "stopped");
    assert!(!p.folder.join("models").join(list[0]["name"].as_str().unwrap()).join("model.partial").exists());
    assert!(catalog::custom_models().is_empty());
    // a worker that ignores it is killed after the grace period
    fake_worker(d.path(), "mkdir -p \"$out\"\nexec sleep 30");
    let stop = Arc::new(AtomicBool::new(true));
    let e = rruns::execute_with_grace(&p, &drop_spec(), stop, Duration::from_millis(300), &mut |_| {}).unwrap_err();
    assert_eq!(e.0, "Stopped.");
    let list = rruns::list(&p);
    assert_eq!((list.len(), list[0]["status"].as_str()), (2, Some("stopped")));
    assert!(!p.folder.join("models").join(list[0]["name"].as_str().unwrap()).join("model.partial").exists());
}

#[cfg(unix)]
#[test]
fn measuring_needs_a_saved_dataset_version_and_gets_its_text() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    let measuring = ReshapeSpec { drop_layers: None, ..drop_spec() };
    let bare = project(d.path());
    let e = rruns::execute(&bare, &measuring, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("Save a training-data version in Step 2 first"), "{e}");
    assert!(rruns::list(&bare).is_empty(), "nothing was started");
    // growing and explicit layers need no data
    let grow = ReshapeSpec { source_model_id: "qwen3.5-4b".into(), ..spec(Mode::Grow, 1) };
    let m = catalog::find("qwen3.5-4b").unwrap();
    let ins: Inspect = serde_json::from_str(INSPECT).unwrap();
    let job = rruns::build_job(&bare, &m, &grow, &ins, "n1").unwrap();
    assert_eq!((job["mode"].clone(), job["calibration"]["train"].clone()), (json!("grow"), serde_json::Value::Null));
    // with a version the job points at its train.jsonl
    let p = project_with_version(&d.path().join("second"));
    let job = rruns::build_job(&p, &m, &measuring, &ins, "n2").unwrap();
    let train = job["calibration"]["train"].as_str().unwrap();
    assert!(train.ends_with("datasets/v1/train.jsonl") && std::path::Path::new(train).is_file(), "{train}");
    assert_eq!((job["calibration"]["version"].clone(), job["calibration"]["samples"].clone(), job["calibration"]["max_len"].clone(), job["load"].clone()), (json!(1), json!(128), json!(512), json!("auto")));
    assert_eq!(job["protect"], json!({"first": 1, "last": 1}));
    // a bad choice never reaches the worker
    let bad = ReshapeSpec { blocks: 3, ..measuring };
    assert!(rruns::build_job(&p, &m, &bad, &ins, "n3").unwrap_err().0.contains("at most half"));
}

#[cfg(unix)]
#[test]
fn without_the_training_software_the_message_says_what_to_do() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project(d.path());
    assert!(rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("Training setup"));
    assert!(rruns::execute(&p, &ReshapeSpec { source_model_id: "nope".into(), ..drop_spec() }, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("not in the list"));
}

#[cfg(unix)]
#[test]
fn a_model_cut_off_by_closing_the_program_is_marked_failed() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project(d.path());
    let dir = p.folder.join("models/20261113-000000-x-slim-8L");
    std::fs::create_dir_all(dir.join("model.partial")).unwrap();
    std::fs::write(dir.join("state.json"), json!({"name": "20261113-000000-x-slim-8L", "status": "running"}).to_string()).unwrap();
    let done = p.folder.join("models/20261112-000000-y-slim-8L");
    std::fs::create_dir_all(&done).unwrap();
    std::fs::write(done.join("state.json"), json!({"name": "20261112-000000-y-slim-8L", "status": "done"}).to_string()).unwrap();
    assert_eq!(rruns::mark_interrupted(&p), 1);
    let list = rruns::list(&p);
    assert_eq!(list[0]["status"], "failed");
    assert!(list[0]["error"].as_str().unwrap().contains("The program was closed while this model was being made. Start it again."));
    assert!(!dir.join("model.partial").exists());
    assert_eq!(list[1]["status"], "done");
    assert_eq!(rruns::mark_interrupted(&p), 0);
}

#[cfg(unix)]
#[test]
fn a_model_in_use_cannot_be_removed_and_removing_cleans_up() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    let p = project(d.path());
    let r = rruns::execute(&p, &drop_spec(), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap();
    let (name, id) = (r["name"].as_str().unwrap().to_string(), r["model_id"].as_str().unwrap().to_string());
    let repo = catalog::find(&id).unwrap().repo;
    // a training run on this model
    let run = truns::runs_dir(&p).join("run-on-slim");
    std::fs::create_dir_all(&run).unwrap();
    std::fs::write(run.join("config.json"), json!({"model": {"repo": repo, "family": "qwen3_5"}}).to_string()).unwrap();
    let e = rruns::remove(&p, &name).unwrap_err();
    assert_eq!(e.0, "Run run-on-slim was trained on this model. Delete that run first, or keep the model.");
    assert!(p.folder.join("models").join(&name).is_dir() && catalog::find(&id).is_some());
    std::fs::remove_dir_all(&run).unwrap();
    rruns::remove(&p, &name).unwrap();
    assert!(!p.folder.join("models").join(&name).exists() && catalog::find(&id).is_none() && rruns::list(&p).is_empty());
    // names from the window are checked
    for bad in ["", "../x", "a/b", "a\\b"] {
        assert!(rruns::remove(&p, bad).is_err(), "{bad}");
    }
    assert!(rruns::remove(&p, "missing").unwrap_err().0.contains("not found"));
}

#[cfg(unix)]
#[test]
fn the_page_state_and_options_come_from_the_worker_and_the_plan() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MAKES_A_MODEL);
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects"));
    let mut m = AppModel::load();
    assert!(m.reshape_state().is_err(), "needs a project");
    m.create_project("P").unwrap();
    let s = m.reshape_state().unwrap();
    assert!(s["models"].as_array().unwrap().iter().any(|x| x["id"] == "qwen3.5-4b") && s["made"].as_array().unwrap().is_empty() && s["dataset_version"].is_null());
    let o = m.reshape_options("qwen3.5-4b", "prune").unwrap();
    assert_eq!(o["inspect"]["layers"], 12);
    assert_eq!(o["options"].as_array().unwrap().len(), 1);                      // 3 blocks: at most 1 (first and last are protected)
    assert_eq!(o["options"][0]["layers"], 8);
    assert_eq!(o["default_blocks"], 1);
    assert!(o["options"][0]["disk_needed_gb"].as_f64().unwrap() > 0.0);
    assert!(o["options"][0]["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("worse than the original")));
    let g = m.reshape_options("qwen3.5-4b", "grow").unwrap();
    assert_eq!(g["options"][0]["layers"], 16);
    assert!(g["options"][0]["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("little or no gain")));
    assert!(m.reshape_options("qwen3.5-4b", "shrink").is_err() && m.reshape_options("nope", "prune").is_err());
}

// ------------------------------------------------------------------------------------------------------- testing made models
use ndforge_core::evaluate::runs as eruns;
use ndforge_core::train::plan::{base_hyper, Method, Preset};

fn project_with_test_split(root: &std::path::Path) -> Project {
    let mut p = project(root);
    let f = root.join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    store::save_version(&p).unwrap();
    p
}

/// A fake Python that answers every question with a fixed text and keeps the job it was given in `capture.json`
#[cfg(unix)]
fn fake_answers(root: &std::path::Path) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let capture = root.join("capture.json");
    std::fs::write(root.join("fake.py"), format!(r#"
import json, sys
job = json.load(open(sys.argv[1], encoding="utf-8"))
json.dump(job, open({cap:?}, "w", encoding="utf-8"))
out = {{s: {{q["id"]: ("trained says hi" if s == "trained" else "original says hi") for q in job["questions"]}} for s in job["which"]}}
json.dump(out, open(job["out"], "w", encoding="utf-8"))
print(json.dumps({{"type": "result", "data": {{"out": job["out"], "n": len(job["questions"])}}}}), flush=True)
"#, cap = capture.to_string_lossy())).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\nexec python3 -I {} \"$5\"\n", root.join("fake.py").display())).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
    capture
}

fn captured(f: &std::path::Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(f).unwrap()).unwrap()
}

/// A finished made model on disk (what `reshape::runs::execute` leaves behind); returns its model folder
fn made_model(p: &Project, name: &str, origin_params: u64, params: u64) -> std::path::PathBuf {
    let dir = p.folder.join("models").join(name);
    let model = dir.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("ndforge_reshape.json"), json!({"format": 1, "origin": {"repo": "unsloth/Qwen3.5-4B", "family": "qwen3_5", "layers": 32, "params": origin_params},
        "mode": "prune", "layers": 24, "params": params}).to_string()).unwrap();
    std::fs::write(dir.join("state.json"), json!({"name": name, "status": "done", "spec": {"source_model_id": "qwen3.5-4b", "mode": "prune"}}).to_string()).unwrap();
    model
}

/// A trained run (adapter) whose base model is `repo`
fn run_on(p: &Project, name: &str, repo: &std::path::Path) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("adapter")).unwrap();
    std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
    let m = catalog::find("qwen3.5-4b").unwrap();
    let mut job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    job["model"]["repo"] = json!(repo);
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("summary.json"), json!({"steps": 40, "final_loss": 0.8}).to_string()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "method": "lora", "step": 40, "total_steps": 40}).to_string()).unwrap();
}

#[cfg(unix)]
#[test]
fn a_made_model_is_tested_against_its_original_and_appears_in_the_table() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let cap = fake_answers(d.path());
    let p = project_with_test_split(d.path());
    let model = made_model(&p, "20261113-000000-x-slim-24L", 9_000_000_000, 6_000_000_000);
    assert!(rruns::read_test(&p, "20261113-000000-x-slim-24L").is_none());
    let none = AtomicBool::new(false);
    let r = rruns::test(&p, "20261113-000000-x-slim-24L", None, Some(32.0), &none, &mut |_| {}).unwrap();
    assert_eq!((r["run"].as_str(), r["against"].as_str()), (Some("model:20261113-000000-x-slim-24L"), Some("origin")));
    assert_eq!(r["loaded"], json!({"original": "16-bit", "trained": "16-bit"}));
    let job = captured(&cap);
    assert_eq!((job["model"]["repo"].as_str(), job["model"]["family"].as_str()), (Some("unsloth/Qwen3.5-4B"), Some("qwen3_5")));
    assert_eq!(job["full_model"].as_str().unwrap(), model.to_string_lossy());
    assert!(job.get("adapter").is_none() && job.get("trained_base").is_none() && job.get("load_4bit").is_none());
    assert_eq!(rruns::read_test(&p, "20261113-000000-x-slim-24L").unwrap()["n"], r["n"]);
    let listed = &rruns::list(&p)[0];
    assert!(listed["tested"] == true && listed["test"]["verdict"]["headline"].is_string() && listed["test"]["against"] == "origin" && listed["test"].get("questions").is_none());
    // a card too small for 16-bit: the sides that do not fit load in 4-bit, and the result says so
    let r = rruns::test(&p, "20261113-000000-x-slim-24L", None, Some(17.0), &none, &mut |_| {}).unwrap();
    assert_eq!(captured(&cap)["load_4bit"], json!({"original": true, "trained": false}));
    assert_eq!(r["loaded"], json!({"original": "4-bit", "trained": "16-bit"}));
    // the comparison table has a row for it
    let table = eruns::compare(&p);
    let row = table.iter().find(|x| x["name"] == "model:20261113-000000-x-slim-24L").expect("a row");
    assert_eq!((row["method"].as_str(), row["steps"].clone(), row["model"].as_str()), (Some("slim"), serde_json::Value::Null, Some("qwen3.5-4b")));
    assert!(row["headline"].is_string() && row["n"].as_u64().unwrap() > 0);
}

#[cfg(unix)]
#[test]
fn testing_a_made_model_needs_data_and_a_finished_model() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_answers(d.path());
    let none = AtomicBool::new(false);
    let bare = project(d.path());
    made_model(&bare, "m1", 9, 6);
    assert!(rruns::test(&bare, "m1", None, None, &none, &mut |_| {}).unwrap_err().0.contains("Save a training-data version in Step 2 first"));
    let p = project_with_test_split(&d.path().join("second"));
    std::fs::create_dir_all(p.folder.join("models/half")).unwrap();
    assert!(rruns::test(&p, "half", None, None, &none, &mut |_| {}).unwrap_err().0.contains("not ready"));
    for bad in ["../x", "a/b", ""] {
        assert!(rruns::test(&p, bad, None, None, &none, &mut |_| {}).is_err());
    }
    let model = made_model(&p, "noorigin", 9, 6);
    std::fs::remove_file(model.join("ndforge_reshape.json")).unwrap();
    assert!(rruns::test(&p, "noorigin", None, None, &none, &mut |_| {}).unwrap_err().0.contains("original is unknown"));
}

#[cfg(unix)]
#[test]
fn a_run_on_a_made_model_is_tested_against_it_or_against_the_original() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let cap = fake_answers(d.path());
    let p = project_with_test_split(d.path());
    let slim = made_model(&p, "m1", 9_000_000_000, 6_000_000_000);
    run_on(&p, "run-slim", &slim);
    let none = AtomicBool::new(false);
    let t = eruns::testable(&p);
    assert_eq!(t.iter().find(|r| r["name"] == "run-slim").unwrap()["reshaped_from"], "unsloth/Qwen3.5-4B");
    // the default: against the made model (the adapter switched off) - the job is exactly what it was before this feature
    let r = eruns::execute(&p, "run-slim", 30, None, &none, &mut |_| {}).unwrap();
    let job = captured(&cap);
    assert_eq!((r["against"].as_str(), job["model"]["repo"].as_str().unwrap()), (Some("base"), slim.to_string_lossy().as_ref()));
    assert!(job.get("trained_base").is_none() && job.get("load_4bit").is_none() && job["adapter"].is_string());
    assert_eq!(r["loaded"], json!({"original": "16-bit", "trained": "16-bit"}));
    assert!(eruns::read_test(&p, "run-slim").is_some());
    // against the original it was made from: the real original answers one side, the adapter on the made model the other
    let r = eruns::execute_against(&p, "run-slim", 30, None, true, Some(32.0), &none, &mut |_| {}).unwrap();
    let job = captured(&cap);
    assert_eq!(r["against"], "origin");
    assert_eq!((job["model"]["repo"].as_str(), job["model"]["family"].as_str()), (Some("unsloth/Qwen3.5-4B"), Some("qwen3_5")));
    assert_eq!(job["trained_base"].as_str().unwrap(), slim.to_string_lossy());
    assert!(job["adapter"].as_str().unwrap().ends_with("run-slim/adapter") && job.get("load_4bit").is_none());
    // on a smaller card the 9 B original loads in 4-bit (and the 6 B made model too)
    eruns::execute_against(&p, "run-slim", 30, None, true, Some(12.0), &none, &mut |_| {}).unwrap();
    assert_eq!(captured(&cap)["load_4bit"], json!({"original": true, "trained": true}));
    // the free chat can ask both as well
    let a = eruns::ask_against(&p, "run-slim", None, "Who is Anders?", true, None, &none, &mut |_| {}).unwrap();
    assert_eq!((a["original"].as_str(), a["trained"].as_str()), (Some("original says hi"), Some("trained says hi")));
    assert_eq!(captured(&cap)["trained_base"].as_str().unwrap(), slim.to_string_lossy());
    // a run on an ordinary base has no origin: asking for it changes nothing
    run_on(&p, "run-plain", std::path::Path::new("unsloth/Qwen3.5-4B"));
    let r = eruns::execute_against(&p, "run-plain", 30, None, true, Some(32.0), &none, &mut |_| {}).unwrap();
    assert_eq!(r["against"], "base");
    assert!(captured(&cap).get("trained_base").is_none());
    // pair_for_run says the same in data
    let pair = eruns::pair_for_run(&p, "run-slim", true, Some(32.0)).unwrap();
    assert_eq!((pair.original.as_str(), pair.against, pair.four_bit), ("unsloth/Qwen3.5-4B", "origin", (false, false)));
    assert!(matches!(&pair.trained, eruns::Side::Adapter { base: Some(b), .. } if b == &slim.to_string_lossy()));
}

// -------------------------------------------------------------------------------------- export and training of runs on made models
use ndforge_core::export::runs as exports;

/// a made model that was grown from 12 to 16 layers (the second copies of layers 2..5 are the new, switched-off layers)
fn grown_model(p: &Project, name: &str, zero: bool) -> std::path::PathBuf {
    let model = made_model(p, name, 1_800_000_000, 2_400_000_000);
    let src: Vec<u32> = (0..6).chain(2..12).collect();
    let new: Vec<u32> = (6..10).collect();
    std::fs::write(model.join("ndforge_reshape.json"), json!({"format": 1, "origin": {"repo": "unsloth/Qwen3.5-4B", "family": "qwen3_5", "layers": 12, "params": 1_800_000_000u64},
        "mode": "grow", "layers": 16, "params": 2_400_000_000u64, "src_layers": src, "dropped": [], "duplicated": [2, 3, 4, 5], "zero_init": if zero { json!(new) } else { json!([]) }}).to_string()).unwrap();
    model
}

#[cfg(unix)]
#[test]
fn the_export_of_a_run_on_a_made_model_uses_the_originals_vision_file_and_says_where_the_base_came_from() {
    use std::os::unix::fs::PermissionsExt;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let bin = d.path().join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(bin.join("python"), "#!/bin/sh\nout=$(sed -n 's/.*\"out\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\nquant=$(sed -n 's/.*\"quant\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\ncp \"$5\" \"$out/job-seen.json\"\nprintf GGUF > \"$out/model-$quant.gguf\"\necho \"{\\\"type\\\":\\\"result\\\",\\\"data\\\":{\\\"gguf\\\":\\\"$out/model-$quant.gguf\\\",\\\"quant\\\":\\\"$quant\\\"}}\"\n").unwrap();
    std::fs::set_permissions(bin.join("python"), std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project_with_test_split(d.path());
    let slim = made_model(&p, "m1", 1_800_000_000, 1_240_000_000);
    run_on(&p, "run-slim", &slim);
    run_on(&p, "run-plain", std::path::Path::new("unsloth/Qwen3.5-4B"));
    let none = AtomicBool::new(false);
    exports::execute(&p, "run-slim", "q4_k_m", 8192, &none, &mut |_| {}).unwrap();
    let dir = p.folder.join("exports/run-slim-q4_k_m");
    let job = captured(&dir.join("job-seen.json"));
    assert_eq!((job["base"]["repo"].as_str().unwrap(), job["base"]["vision_repo"].as_str()), (slim.to_string_lossy().as_ref(), Some("unsloth/Qwen3.5-4B")));
    let card = std::fs::read_to_string(dir.join("MODEL_CARD.md")).unwrap();
    assert!(card.contains("made from unsloth/Qwen3.5-4B by removing or adding layers"), "{card}");
    // an ordinary base: nothing changes
    exports::execute(&p, "run-plain", "q4_k_m", 8192, &none, &mut |_| {}).unwrap();
    let dir = p.folder.join("exports/run-plain-q4_k_m");
    assert!(captured(&dir.join("job-seen.json"))["base"].get("vision_repo").is_none());
    assert!(!std::fs::read_to_string(dir.join("MODEL_CARD.md")).unwrap().contains("made from"));
}

#[test]
fn a_grown_base_that_starts_switched_off_also_trains_its_output_projections() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let p = project_with_test_split(d.path());
    let mut m = catalog::find("qwen3.5-4b").unwrap();
    let hyper = base_hyper(Preset::Normal);
    let job_for = |m: &catalog::ModelEntry, h: &ndforge_core::train::plan::Hyper| truns::build_job(&p, m, Method::Lora, h, 1, "r1", false);
    // an ordinary model: the job is as before
    let job = job_for(&m, &hyper).unwrap();
    assert!(job["hyper"].get("extra_targets").is_none() && job["hyper"].get("layers_to_transform").is_none());
    // a grown model with switched-off layers
    m.repo = grown_model(&p, "g1", true).to_string_lossy().to_string();
    let job = job_for(&m, &hyper).unwrap();
    assert_eq!(job["hyper"]["extra_targets"], json!(["out_proj"]));
    assert!(job["hyper"].get("layers_to_transform").is_none());
    // a grown model that was copied (SOLAR style) needs nothing extra
    m.repo = grown_model(&p, "g2", false).to_string_lossy().to_string();
    assert!(job_for(&m, &hyper).unwrap()["hyper"].get("extra_targets").is_none());
    // the expert option: only the new layers
    let only = ndforge_core::train::plan::Hyper { only_new_layers: true, ..hyper.clone() };
    assert_eq!(job_for(&m, &only).unwrap()["hyper"]["layers_to_transform"], json!([6, 7, 8, 9]));
    m.repo = made_model(&p, "slim", 1, 1).to_string_lossy().to_string();           // a slimmed model has no added layers
    assert!(job_for(&m, &only).unwrap_err().0.contains("grown on the Slim or grow page"));
    m.repo = "unsloth/Qwen3.5-4B".into();
    assert!(job_for(&m, &only).is_err());
    assert_eq!(truns::new_layers(&json!({"src_layers": [0, 1, 2, 1, 2, 3]})), vec![3, 4]);
}
