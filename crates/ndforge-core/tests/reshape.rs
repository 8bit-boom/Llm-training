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
    assert!(p.est_gb > 15.0 && p.est_gb < 27.2 && (p.est_gb - p.student_gb).abs() < 0.1 && p.why.is_none() && p.verdict == "fits", "the student's step is the heavier one: {p:?}");
    // 8 B in 16-bit next to a 4-bit teacher no longer fits together on 32 GB by the new estimate (activations and the LoRA are counted): the teacher answers first
    let q = heal_plan(4.0, 4.8, Some(32.0));
    assert_eq!((q.mode, q.student_method, q.teacher_load, q.fits), ("online", "lora", "4bit", true), "{q:?}");
    let e = heal_plan(8.0, 9.65, Some(32.0));
    assert_eq!((e.mode, e.student_method, e.fits), ("cache", "lora", true), "8 B in 16-bit: about 24 GB of the 32, below the 85 % that makes QLoRA the automatic choice: {e:?}");
    let small = heal_plan(21.1, 27.8, Some(16.0));
    assert!(!small.fits && small.verdict == "over" && small.why.as_deref().unwrap().contains("this card has 16 GB") && small.why.as_deref().unwrap().contains("Choose a smaller model"), "{small:?}");
    assert_eq!(heal_plan(21.1, 27.8, None).mode, "cache");
    assert!(p.cache_gb_per_10m > 1.9 && p.cache_gb_per_10m < 2.0, "196 bytes per token, 10 M tokens");
    assert!(p.hours_per_10m.0 > 0.0 && p.hours_per_10m.1 > p.hours_per_10m.0);
    assert_eq!((default_heal_tokens_m(21.1), default_heal_tokens_m(8.0), default_heal_tokens_m(15.0)), (20, 40, 20));
    assert_eq!(HEAL_TOKENS_M, [10, 20, 40, 80]);
}

fn cases() -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/heal_cases.json")).unwrap()).unwrap()
}

fn knobs_of(v: &serde_json::Value) -> HealKnobs {
    HealKnobs { seq: 1024, batch: v["batch"].as_u64().unwrap() as u32, grad_accum: v["grad_accum"].as_u64().unwrap() as u32, checkpointing: v["checkpointing"].as_bool().unwrap(), rank: 32,
                kl_chunk: v["chunk"].as_u64().unwrap() as u32, qlora: true }
}

#[test]
fn the_safer_settings_ladder_and_the_activation_estimate_equal_the_workers() {
    let c = cases();
    for case in c["ladder"].as_array().unwrap() {
        let got = heal_next_safer(&knobs_of(&case["from"]));
        match (&case["to"], got) {
            (serde_json::Value::Null, g) => assert!(g.is_none(), "{case}"),
            (to, Some((k, label))) => assert_eq!((k, label.as_str()), (knobs_of(to), case["label"].as_str().unwrap()), "{case}"),
            (_, None) => panic!("{case}"),
        }
    }
    for a in c["activation"].as_array().unwrap() {
        let got = heal_activation_gb(a["layers"].as_u64().unwrap() as u32, a["hidden"].as_u64().unwrap() as u32, a["seq"].as_u64().unwrap() as u32, a["batch"].as_u64().unwrap() as u32, a["checkpointing"].as_bool().unwrap());
        let want = a["gb"].as_f64().unwrap();
        assert!((got - want).abs() <= want * 1e-3, "{a}: {got}");
    }
    // the ladder ends, keeps the sequences per update and never touches the maths
    let mut k = HealKnobs { seq: 2048, batch: 4, grad_accum: 8, checkpointing: false, rank: 64, kl_chunk: 256, qlora: false };
    let mut n = 0;
    while let Some((next, _)) = heal_next_safer(&k) {
        assert_eq!((next.batch * next.grad_accum, next.seq, next.rank, next.qlora), (32, 2048, 64, false));
        k = next;
        n += 1;
        assert!(n < 20);
    }
    assert_eq!((n, k.checkpointing, k.batch, k.kl_chunk), (5, true, 1, 64));
}

#[test]
fn the_automatic_settings_equal_the_workers_and_follow_the_size() {
    for a in cases()["auto"].as_array().unwrap() {
        let b = a["student_b"].as_f64().unwrap();
        let k = heal_defaults(b, Some(32.0), &HealDims { layers: 40, hidden: 4096, vocab: 248_320 });
        assert_eq!((k.seq as u64, k.rank as u64, k.batch as u64, k.grad_accum as u64, k.kl_chunk as u64), (a["max_seq"].as_u64().unwrap(), a["rank"].as_u64().unwrap(), a["batch"].as_u64().unwrap(), a["grad_accum"].as_u64().unwrap(), a["chunk"].as_u64().unwrap()), "{a}");
        assert!(b < 15.0 || k.qlora, "{a}");
        assert!(b < 8.0 || k.checkpointing, "always on from 8 B: {a}");
    }
    // a small student on a small card: checkpointing when the activations would take more than 60 % of what is left, not otherwise
    let d = HealDims { layers: 28, hidden: 2048, vocab: 150_000 };
    assert!(heal_defaults(2.0, Some(8.0), &d).checkpointing, "act without it: ~3.7 GB x 4 = 15 GB");
    assert!(!heal_defaults(0.5, Some(80.0), &d).checkpointing);
    assert!(heal_defaults(0.5, None, &d).checkpointing, "unknown card: on");
    // 16-bit weights that do not fit the card: QLoRA whatever the size
    assert!(heal_defaults(12.0, Some(24.0), &HealDims::guess(12.0)).qlora && !heal_defaults(12.0, Some(48.0), &HealDims::guess(12.0)).qlora);
}

fn mem(b: f64, k: &HealKnobs) -> f64 {
    heal_memory_gb(b, &HealDims { layers: 40, hidden: 4096, vocab: 248_320 }, k)
}

#[test]
fn the_memory_estimate_is_monotonic_and_checkpointing_and_qlora_lower_it() {
    let base = HealKnobs { seq: 1024, batch: 2, grad_accum: 16, checkpointing: false, rank: 32, kl_chunk: 128, qlora: true };
    let m = |f: &dyn Fn(&mut HealKnobs)| { let mut k = base.clone(); f(&mut k); mem(16.0, &k) };
    let b = mem(16.0, &base);
    assert!(m(&|k| k.seq = 2048) > b && m(&|k| k.seq = 512) < b && m(&|k| k.batch = 4) > b && m(&|k| k.batch = 1) < b && m(&|k| k.rank = 64) > b && m(&|k| k.rank = 8) < b && m(&|k| k.kl_chunk = 512) > b);
    assert!(m(&|k| k.checkpointing = true) < b);
    assert!(m(&|k| k.qlora = false) > b, "16-bit weights are bigger than 4-bit ones");
    assert!(mem(21.0, &base) > mem(16.0, &base));
    let total = heal_memory(16.0, &HealDims { layers: 40, hidden: 4096, vocab: 248_320 }, &base);
    assert!((total.total_gb - (total.weights_gb + total.lora_gb + total.activations_gb + total.kl_gb + total.overhead_gb) * 1.10).abs() < 0.1);
}

#[test]
fn a_16b_student_fits_32_gb_at_the_defaults_and_not_at_batch_4_seq_2048_without_checkpointing() {
    let d = HealDims { layers: 40, hidden: 4096, vocab: 248_320 };
    let def = heal_defaults(16.0, Some(32.0), &d);
    let p = heal_plan_with(16.0, 21.0, Some(32.0), &d, &def);
    assert!(p.fits && p.verdict == "fits" && p.student_gb < 27.2, "{p:?}");
    let wild = HealKnobs { seq: 2048, batch: 4, grad_accum: 8, checkpointing: false, ..def.clone() };
    let w = heal_plan_with(16.0, 21.0, Some(32.0), &d, &wild);
    assert!(!w.fits && w.verdict == "over" && w.student_gb > 32.0, "the owner's failure: {w:?}");
    assert!(w.verdict_text.starts_with("Will probably run out of memory") && p.verdict_text.starts_with("Fits"));
    assert_eq!(heal_verdict(28.0, Some(32.0)), "tight");
    assert!(heal_verdict_text(28.0, Some(32.0)).starts_with("Tight"));
    assert_eq!(heal_verdict(10.0, None), "unknown");
    // the sizes come from the model's config, the language part of a multimodal one included
    let c = HealDims::from_config(&json!({"text_config": {"num_hidden_layers": 40, "hidden_size": 4096, "vocab_size": 248320}}), 16.0);
    assert_eq!(c, d);
    assert_eq!(HealDims::from_config(&json!({}), 16.0), HealDims::guess(16.0));
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
    fake_worker_with(root, INSPECT, body)
}

#[cfg(unix)]
fn fake_worker_with(root: &std::path::Path, inspect: &str, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    let script = format!(r#"#!/bin/sh
job="$5"
if grep -q '"job":"reshape_inspect"' "$job"; then
  echo '{{"type":"result","data":{inspect}}}'
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

// ------------------------------------------------------------------------------------------------- thinner layers: the runs

/// What `reshape_inspect` answers for a Qwen3.5-4B-like model that can be thinned
#[cfg(unix)]
fn inspect_width() -> String {
    let mut v: serde_json::Value = serde_json::from_str(INSPECT).unwrap();
    v["layer_types"] = json!((0..12).map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" }).collect::<Vec<_>>());
    v["width_supported"] = json!(true);
    v["width_why"] = serde_json::Value::Null;
    v["width_dims"] = json!({"hidden": 2560, "intermediate": 9216, "heads": 16, "kv_heads": 4, "head_dim": 256, "lin_k_heads": 16, "lin_v_heads": 32, "lin_k_dim": 128,
                             "lin_v_dim": 128, "conv_kernel": 4, "mtp_layers": 0, "vocab": 248320, "tied": true});
    v["width_choices"] = json!({"intermediate": (18..=36).map(|k| k * 256).collect::<Vec<u32>>(), "lin_v_heads": [32, 16], "heads": [16, 12]});
    v.to_string()
}

#[cfg(unix)]
const MAKES_A_THIN_MODEL: &str = r#"mkdir -p "$out"
echo '{"type":"width_scores","layers":12,"mlp_mean":[0.1,0.2],"kept_share":0.9,"stable_overlap":0.95}'
echo '{"model_type":"qwen3_5","architectures":["Qwen3_5ForCausalLM"],"num_hidden_layers":12}' > "$out/config.json"
printf -- '---\nlicense: apache-2.0\n---\n' > "$out/README.md"
echo '{"format":2,"origin":{"repo":"unsloth/Qwen3.5-4B","family":"qwen3_5","layers":12,"params":1800000000},"mode":"prune","layers":12,"params":1500000000,"shape":"width","width":{"intermediate_size":[9216,6912],"linear_num_value_heads":[32,32],"num_attention_heads":[16,16],"kept":"width_kept.npz","stable_overlap":0.95}}' > "$out/ndforge_reshape.json"
head -c 2000 /dev/zero > "$out/model.safetensors"
mv "$out" "$final"
echo '{"type":"result","data":{"model_dir":"x","mode":"prune","shape":"width","layers_before":12,"layers_after":12,"src_layers":[0,1,2,3,4,5,6,7,8,9,10,11],"dropped":[],"duplicated":[],"zero_init":[],"params_before":1800000000,"params_after":1500000000,"bytes_written":3000000000,"loss_before":1.5,"loss_after":2.5,"ppl_before":4.48,"ppl_after":12.2,"stable":null,"boundary_distance":null,"chosen_blocks":[],"load":"16bit","seconds":1.0,"notes":[],"stopped":false,"width":{"before":{"intermediate_size":9216},"after":{"intermediate_size":6912},"stable_overlap":0.95,"scores_key":"k","mtp":"none"}}}'"#;

#[cfg(unix)]
fn width_spec(blocks: u32, intermediate: u32) -> ReshapeSpec {
    ReshapeSpec { width: Some(Width { intermediate, lin_v_heads: None, heads: None }), ..spec(Mode::Prune, blocks) }.with_model("qwen3.5-4b")
}

#[cfg(unix)]
#[test]
fn a_thinner_model_is_made_with_the_width_job_and_listed() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker_with(d.path(), &inspect_width(), MAKES_A_THIN_MODEL);
    let p = project_with_version(d.path());
    // width needs the project's own text
    let q = project(&d.path().join("empty"));
    assert!(rruns::execute(&q, &width_spec(0, 6912), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("Save a training-data version"));
    let mut events = Vec::new();
    let result = rruns::execute(&p, &width_spec(0, 6912), Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
    let name = result["name"].as_str().unwrap().to_string();
    assert!(name.ends_with("-qwen3.5-4b-slim-w6912"), "{name}");
    let dir = p.folder.join("models").join(&name);
    let job: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("job-seen.json")).unwrap()).unwrap();
    assert_eq!(job["width"], json!({"intermediate_size": 6912, "linear_num_value_heads": null, "num_attention_heads": null}));
    assert_eq!((job["blocks"].clone(), job["calibration"]["samples"].clone(), job["calibration"]["version"].clone()), (json!(0), json!(256), json!(1)));
    let key = "unsloth-qwen3.5-4b-v1-256x512-auto";
    assert_eq!(job["scores_dir"].as_str().unwrap(), p.folder.join("models").join("_scores").join(key).to_string_lossy());
    assert_eq!(rruns::scores_key("unsloth/Qwen3.5-4B", Some(1), 256, 512, "auto"), key);
    assert!(job["calibration"]["train"].as_str().unwrap().ends_with("train.jsonl"));
    // events and the saved chart numbers
    assert!(events.iter().any(|e| e["kind"] == "width_scores" && e["kept_share"] == json!(0.9)) && dir.join("width_scores.json").is_file());
    // the model list: the name and the origin of the width
    let m = catalog::find(result["model_id"].as_str().unwrap()).expect("registered");
    assert!(m.name.ends_with("slim MLP 6,912"), "{}", m.name);
    assert_eq!(m.reshape.as_deref(), Some("slimmed: 12 → 12 layers, MLP 6,912 (from 9,216)"));
    assert!(m.note.contains("healed"), "{}", m.note);
    let list = rruns::list(&p);
    assert_eq!((list[0]["summary"]["shape"].clone(), list[0]["summary"]["width"]["after"]["intermediate_size"].clone()), (json!("width"), json!(6912)));
    assert_eq!(list[0]["width_scores"]["layers"], 12);
    assert_eq!((list[0]["healed"].clone(), list[0]["heal"].clone()), (json!(false), serde_json::Value::Null));
    // both ways at once, with explicit layers: the name tells
    let both = ReshapeSpec { drop_layers: Some(vec![4, 5, 6, 7]), ..width_spec(0, 6912) };
    let r2 = rruns::execute(&p, &both, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap();
    assert!(r2["name"].as_str().unwrap().ends_with("-slim-8L-w6912"), "{}", r2["name"]);
    // a width the model does not allow never reaches the worker
    assert!(rruns::execute(&p, &width_spec(0, 6913), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("multiple of 256"));
    // measurements have a folder of their own that is not a made model
    std::fs::create_dir_all(rruns::scores_dir(&p, key)).unwrap();
    assert!(rruns::list(&p).iter().all(|x| x["name"].as_str().is_some_and(|n| n != "_scores")));
    assert!(rruns::remove(&p, "_scores").is_err());
    rruns::remove_scores(&p, key).unwrap();
    assert!(!rruns::scores_dir(&p, key).exists() && rruns::remove_scores(&p, "../x").is_err());
}

#[test]
fn the_names_of_made_models_tell_what_was_done() {
    let m = catalog::find("qwen3.5-4b").unwrap();
    let w = |i, v, h| Width { intermediate: i, lin_v_heads: v, heads: h };
    let tail = |n: String| n.split_once("-qwen3.5-4b-").unwrap().1.to_string();
    assert_eq!(tail(rruns::name_for_shape(&m, Mode::Prune, 48, None, true)), "slim-48L");
    assert_eq!(tail(rruns::name_for_shape(&m, Mode::Prune, 64, Some(&w(10752, None, None)), false)), "slim-w10752");
    assert_eq!(tail(rruns::name_for_shape(&m, Mode::Prune, 60, Some(&w(12288, None, None)), true)), "slim-60L-w12288");
    assert_eq!(tail(rruns::name_for_shape(&m, Mode::Prune, 64, Some(&w(13312, Some(32), Some(16))), false)), "slim-w13312-v32-h16");
}

// ---------------------------------------------------------------------------------------------------------------- healing

/// A fake Python for the three healing jobs. It writes what the real worker writes, keeps every job in `jobs.jsonl`, and obeys marker files in `root`:
/// `stop-cache` / `stop-train` make that job answer "stopped"; `fail-train` makes the training fail.
#[cfg(unix)]
fn fake_healer(root: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(root.join("healer.py"), format!(r#"
import json, os, sys
root = {root:?}
job = json.load(open(sys.argv[1], encoding="utf-8"))
open(os.path.join(root, "jobs.jsonl"), "a", encoding="utf-8").write(json.dumps(job) + "\n")
open(os.path.join(root, "env.jsonl"), "a", encoding="utf-8").write(json.dumps({{"job": job["job"], "hf": os.environ.get("HF_TOKEN")}}) + "\n")
def out(**kw): print(json.dumps(kw), flush=True)
def marker(n):
    return os.path.exists(os.path.join(root, n))
kind = job["job"]
if kind == "reshape_measure":
    os.makedirs(job["out_dir"], exist_ok=True)
    json.dump({{"format": 1, "tokens": 4321, "load": "4bit", "created": "2026-11-13T00:00:00Z"}}, open(os.path.join(job["out_dir"], "scores.json"), "w"))
    out(type="progress", step="score", value=0.5, text="Measuring")
    out(type="result", data={{"tokens": 4321, "load": "4bit", "loss": 2.0, "stable_depth": True}}); sys.exit(0)
if kind == "distill_cache":
    if marker("stop-cache"):
        out(type="result", data={{"stopped": True, "rows": 10, "done_rows": 3}}); sys.exit(0)
    os.makedirs(job["out"], exist_ok=True)
    g = job["data"].get("general")
    key = {{"used": False}} if not g else {{"used": True, "id": g.get("id"), "file": g.get("file"), "config": g.get("config"), "split": g.get("split"), "column": g.get("column"), "share": g["share"], "seed": job["seed"]}}
    json.dump({{"complete": True, "rows": 10, "done_rows": 10, "general_key": key, "k": job["k"], "max_len": job["max_seq"]}}, open(os.path.join(job["out"], "index.json"), "w"))
    out(type="progress", step="cache", value=0.5, text="The original model answers your texts")
    out(type="result", data={{"rows": 10, "tokens": 1000, "stopped": False, "notes": ["The general text (org/x) could not be downloaded: offline."]}})
elif kind == "distill":
    os.makedirs(os.path.join(job["out"], "checkpoints", "checkpoint-50"), exist_ok=True)
    out(type="heal_settings", settings={{"checkpointing": job["hyper"]["checkpointing"] == "on", "batch": job["hyper"]["batch"], "grad_accum": job["hyper"]["grad_accum"], "chunk": job["distill"]["chunk"], "max_seq": job["hyper"]["max_seq"], "rank": job["hyper"]["rank"], "method": job["method"]}})
    if marker("oom-train"):
        out(type="heal_retry", attempt=1, change="gradient checkpointing", settings={{"checkpointing": True, "batch": job["hyper"]["batch"], "grad_accum": job["hyper"]["grad_accum"], "chunk": job["distill"]["chunk"], "max_seq": job["hyper"]["max_seq"], "rank": job["hyper"]["rank"], "method": job["method"]}})
        out(type="error", code="distill_oom", message="The graphics card ran out of memory during step train. Settings used: x."); sys.exit(1)
    if marker("fail-train"):
        out(type="error", code="distill_unstable", message="Healing stopped: the training became unstable (the error went to infinity). Lower the learning rate (Expert) and continue."); sys.exit(1)
    out(type="distill_metric", step=1, kd=0.9, ce=0.0, tokens=100, total_steps=2)
    out(type="distill_eval", step=1, calib_loss=3.0)
    if marker("stop-train"):
        out(type="result", data={{"stopped": True, "steps": 1}}); sys.exit(0)
    os.makedirs(os.path.join(job["out"], "adapter"), exist_ok=True)
    open(os.path.join(job["out"], "adapter", "adapter_config.json"), "w").write("{{}}")
    res = {{"steps": 2, "tokens_seen": 200, "kd_first": 0.9, "kd_last": 0.4, "calib_loss_before": 3.0, "calib_loss_after": 2.4, "teacher_calib_loss": 2.0, "mode": job["distill"]["mode"], "stopped": False}}
    json.dump(res, open(os.path.join(job["out"], "summary.json"), "w"))
    out(type="result", data=res)
elif kind == "distill_merge":
    os.makedirs(job["out"], exist_ok=True)
    meta = json.load(open(os.path.join(job["base"], "ndforge_reshape.json"), encoding="utf-8"))
    meta["healed"] = {{"steps": 2}}
    json.dump(meta, open(os.path.join(job["out"], "ndforge_reshape.json"), "w"))
    open(os.path.join(job["out"], "config.json"), "w").write(json.dumps({{"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"], "num_hidden_layers": 24}}))
    open(os.path.join(job["out"], "README.md"), "w").write("---\nlicense: apache-2.0\n---\n")
    open(os.path.join(job["out"], "model.safetensors"), "wb").write(b"\0" * 1000)
    os.rename(job["out"], job["final"])
    out(type="result", data={{"model_dir": job["final"], "merged": 5, "stopped": False}})
else:
    out(type="error", code="unknown_job", message="unknown")
"#, root = root.to_string_lossy())).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\nexec python3 -I {} \"$5\"\n", root.join("healer.py").display())).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
fn jobs_seen(root: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(root.join("jobs.jsonl")).unwrap_or_default().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// Plenty of disk for the duration of a test (the program checks the free space before saving the teacher's answers); the real value comes back after.
#[cfg(unix)]
struct PlentyOfDisk;

#[cfg(unix)]
impl PlentyOfDisk {
    fn new() -> Self {
        std::env::set_var("NDFORGE_FREE_DISK_GB", "500");
        PlentyOfDisk
    }
}

#[cfg(unix)]
impl Drop for PlentyOfDisk {
    fn drop(&mut self) {
        std::env::remove_var("NDFORGE_FREE_DISK_GB");
    }
}

#[cfg(unix)]
fn heal_spec(tokens_m: u32) -> rruns::HealSpec {
    serde_json::from_value(json!({"tokens_m": tokens_m})).unwrap()
}

/// the 27B thinned to about 21 B: a made model (and its original) of the sizes of the plan's example
#[cfg(unix)]
fn thin_27b(p: &Project, name: &str) -> std::path::PathBuf {
    let model = made_model(p, name, 27_800_000_000, 21_100_000_000);
    let mut meta: serde_json::Value = serde_json::from_slice(&std::fs::read(model.join("ndforge_reshape.json")).unwrap()).unwrap();
    meta["layers"] = json!(24);
    meta["width"] = json!({"intermediate_size": [17408, 10752]});
    meta["loss_before"] = json!(2.0);
    meta["loss_after"] = json!(3.1);
    meta["calibration"] = json!({"version": 1, "samples": 256, "max_len": 512, "tokens": 5});
    std::fs::write(model.join("ndforge_reshape.json"), meta.to_string()).unwrap();
    std::fs::write(model.join("config.json"), r#"{"model_type":"qwen3_5","architectures":["Qwen3_5ForCausalLM"],"num_hidden_layers":24}"#).unwrap();
    model
}

#[cfg(unix)]
#[test]
fn a_model_is_healed_in_three_jobs_and_the_healed_model_is_listed() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim-w10752");
    let dir = p.folder.join("models").join("slim-w10752");
    let mut events = Vec::new();
    let r = rruns::heal(&p, "slim-w10752", &heal_spec(20), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |e| events.push(e)).unwrap();
    assert_eq!((r["name"].as_str(), r["model_id"].as_str()), (Some("slim-w10752"), Some("custom-local-slim-w10752-healed")));
    // the jobs, in order, with what the plan says
    let jobs = jobs_seen(d.path());
    assert_eq!(jobs.iter().map(|j| j["job"].as_str().unwrap()).collect::<Vec<_>>(), ["distill_cache", "distill", "distill_merge"], "27B -> 21B on 32 GB: the teacher answers first");
    let (cache, train, merge) = (&jobs[0], &jobs[1], &jobs[2]);
    assert_eq!((cache["teacher"]["repo"].as_str(), cache["k"].clone(), cache["max_seq"].clone(), cache["data"]["tokens"].clone()), (Some("unsloth/Qwen3.5-4B"), json!(32), json!(1024), json!(20_000_000)));
    assert_eq!((cache["data"]["general"]["share"].clone(), cache["data"]["general"]["id"].as_str(), cache["data"]["general"]["split"].as_str()), (json!(0.5), Some("mlabonne/open-perfectblend"), Some("train")));
    assert!(cache["data"]["general"].get("column").is_none(), "the worker finds the column itself");
    assert_eq!(cache["student"].as_str().unwrap(), dir.join("model").to_string_lossy());
    assert_eq!(cache["stop_file"].as_str().unwrap(), dir.join("STOP").to_string_lossy());
    assert_eq!((train["method"].as_str(), train["backend"].as_str(), train["hyper"]["rank"].clone(), train["hyper"]["alpha"].clone(), train["hyper"]["lr"].clone(), train["hyper"]["tokens"].clone()),
               (Some("qlora"), Some("hf"), json!(32), json!(32), json!(1e-4), json!(20_000_000)));
    // a 21 B student: the cautious automatic settings (docs/HEAL_SETTINGS.md): 1024 tokens, batch 2 x 16, checkpointing on, 8-bit optimizer, KL chunk 128
    assert_eq!((train["hyper"]["max_seq"].clone(), train["hyper"]["batch"].clone(), train["hyper"]["grad_accum"].clone(), train["hyper"]["checkpointing"].clone(), train["hyper"]["optimizer"].clone(), train["distill"]["chunk"].clone()),
               (json!(1024), json!(2), json!(16), json!("on"), json!("adamw_8bit"), json!(128)));
    assert_eq!((train["distill"]["mode"].as_str(), train["distill"]["alpha"].clone(), train["resume"].clone()), (Some("cache"), json!(1.0), json!(false)));
    assert_eq!((train["eval"]["teacher_loss"].clone(), train["eval"]["slim_loss"].clone(), train["eval"]["calibration"]["samples"].clone()), (json!(2.0), json!(3.1), json!(256)));
    assert_eq!(merge["base"].as_str().unwrap(), dir.join("model").to_string_lossy());
    assert!(merge["adapter"].as_str().unwrap().ends_with("distill/run/adapter") && merge["final"].as_str().unwrap().ends_with("healed") && merge["summary"]["kd_last"] == json!(0.4));
    // events carry the phase and the step; the chart numbers are kept for after a restart
    assert!(events.iter().all(|e| e["phase"] == "heal") && events.iter().any(|e| e["heal_step"] == "train" && e["kind"] == "distill_metric"));
    assert_eq!(events.iter().filter(|e| e["heal_step"] == "merge").count(), 0, "the merge job here sends no events");
    assert_eq!(rruns::heal_metrics(&p, "slim-w10752").len(), 2);
    // files: the healed model, no checkpoints, the teacher's answers kept
    assert!(dir.join("healed/config.json").is_file() && !dir.join("healed.partial").exists() && !dir.join("distill/run/checkpoints").exists());
    assert!(dir.join("distill/cache/index.json").is_file() && dir.join("distill/run/adapter/adapter_config.json").is_file());
    let h: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("heal.json")).unwrap()).unwrap();
    assert_eq!((h["status"].as_str(), h["step"].as_str(), h["mode"].as_str(), h["result"]["calib_loss_after"].clone()), (Some("done"), Some("done"), Some("cache"), json!(2.4)));
    assert_eq!(h["note"].as_str().map(|n| n.contains("could not be downloaded")), Some(true), "an offline general text is told plainly");
    // the model list: the healed model next to the thinned one
    let m = catalog::find("custom-local-slim-w10752-healed").expect("registered");
    assert!(m.name.ends_with("slim 24 layers, MLP 10,752 · healed"), "{}", m.name);
    assert_eq!(m.reshape.as_deref(), Some("healed: 32 → 24 layers, MLP 10,752"));
    assert_eq!(m.derived_from.as_deref(), Some("unsloth/Qwen3.5-4B"));
    assert_eq!(m.repo, dir.join("healed").canonicalize().unwrap().to_string_lossy());
    let list = rruns::list(&p);
    assert_eq!((list[0]["healed"].clone(), list[0]["heal"]["status"].clone(), list[0]["healed_model_id"].clone()), (json!(true), json!("done"), json!("custom-local-slim-w10752-healed")));
    assert!(list[0]["cache_gb"].as_f64().is_some() && list[0]["healed_size_gb"].as_f64().is_some());
    // once is enough
    assert!(rruns::heal(&p, "slim-w10752", &heal_spec(20), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap_err().0.contains("already been healed"));
    // deleting the model deletes the healed one and the teacher's answers
    rruns::remove(&p, "slim-w10752").unwrap();
    assert!(catalog::find("custom-local-slim-w10752-healed").is_none() && !dir.exists());
}

#[cfg(unix)]
#[test]
fn a_small_student_is_healed_online_without_the_cache() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    made_model(&p, "slim-9b", 4_800_000_000, 4_000_000_000);       // (an 8 B student no longer fits next to the teacher on 32 GB by the new estimate)
    let r = rruns::heal(&p, "slim-9b", &heal_spec(40), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap();
    assert_eq!(r["mode"], "online");
    let jobs = jobs_seen(d.path());
    assert_eq!(jobs.iter().map(|j| j["job"].as_str().unwrap()).collect::<Vec<_>>(), ["distill", "distill_merge"]);
    assert_eq!((jobs[0]["method"].as_str(), jobs[0]["distill"]["teacher"]["load"].as_str(), jobs[0]["hyper"]["tokens"].clone()), (Some("lora"), Some("4bit"), json!(40_000_000)));
    // Expert: the project's own text only, and the cache even though the card could do it online
    let spec = serde_json::from_value(json!({"tokens_m": 10, "general_share": 0.0, "mode": "cache", "alpha": 0.8, "rank": 32, "lr": 2e-4})).unwrap();
    made_model(&p, "slim-9b-2", 4_800_000_000, 4_000_000_000);
    rruns::heal(&p, "slim-9b-2", &spec, Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap();
    let jobs = jobs_seen(d.path());
    let (cache, train) = (&jobs[2], &jobs[3]);
    assert_eq!((cache["job"].as_str(), cache["data"]["general"].clone()), (Some("distill_cache"), serde_json::Value::Null));
    assert_eq!((train["distill"]["alpha"].clone(), train["hyper"]["rank"].clone(), train["hyper"]["lr"].clone(), train["dataset"]["general"].clone()), (json!(0.8), json!(32), json!(2e-4), serde_json::Value::Null));
}

#[cfg(unix)]
fn heal_with(p: &Project, name: &str, spec: serde_json::Value) -> Result<serde_json::Value, ndforge_core::error::Error> {
    let spec: rruns::HealSpec = serde_json::from_value(spec).unwrap();
    rruns::heal(p, name, &spec, Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {})
}

#[cfg(unix)]
#[test]
fn healing_settings_reach_the_jobs_and_the_teachers_answers_are_kept_unless_the_data_changes() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let dir = p.folder.join("models").join("slim");
    let kinds = || jobs_seen(d.path()).iter().map(|j| j["job"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    // the cache is made, then the training fails with an out-of-memory error: a plain flag, the retries and the settings used are kept
    std::fs::write(d.path().join("oom-train"), "").unwrap();
    let spec = json!({"tokens_m": 10, "general_share": 0.0, "seq": 512, "batch": 1, "per_update": 16, "checkpointing": "off", "kl_chunk": 128, "k": 16, "optimizer": "adamw", "warmup": 5, "method": "qlora", "teacher_load": "4bit", "rank": 16});
    assert!(heal_with(&p, "slim", spec.clone()).unwrap_err().0.contains("ran out of memory"));
    let jobs = jobs_seen(d.path());
    let (cache, train) = (&jobs[0], &jobs[1]);
    assert_eq!((cache["k"].clone(), cache["max_seq"].clone(), cache["teacher"]["load"].clone()), (json!(16), json!(512), json!("4bit")));
    assert_eq!((train["hyper"]["batch"].clone(), train["hyper"]["grad_accum"].clone(), train["hyper"]["checkpointing"].clone(), train["hyper"]["optimizer"].clone(), train["hyper"]["warmup"].clone(), train["hyper"]["rank"].clone(), train["distill"]["chunk"].clone(), train["distill"]["k"].clone()),
               (json!(1), json!(16), json!("off"), json!("adamw"), json!(5), json!(16), json!(128), json!(16)));
    let h: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("heal.json")).unwrap()).unwrap();
    assert_eq!((h["status"].clone(), h["oom"].clone(), h["retries"][0]["change"].clone()), (json!("failed"), json!(true), json!("gradient checkpointing")));
    // the page's "Try again with safer settings": the next step of the ladder from the settings of the last try (checkpointing was already turned on by the retry)
    let o = rruns::heal_options(&p, "slim", Some(32.0)).unwrap();
    assert_eq!((o["safer"]["label"].as_str(), o["safer"]["checkpointing"].as_str(), o["safer"]["kl_chunk"].clone()), (Some("a smaller KL chunk (64)"), Some("on"), json!(64)), "{o}");
    assert_eq!(o["safer"]["batch"], 1, "the batch is already 1; the next rung is the chunk");
    // only student-side settings changed (and the memory ones): the teacher's answers stay, the training goes on from its checkpoint
    std::fs::remove_file(d.path().join("oom-train")).unwrap();
    let mut s2 = spec.clone();
    s2["checkpointing"] = json!("on");
    s2["kl_chunk"] = json!(64);
    s2["optimizer"] = json!("adamw_8bit");
    heal_with(&p, "slim", s2).unwrap();
    let all = kinds();
    assert_eq!(all, ["distill_cache", "distill", "distill", "distill_merge"], "the answers were not made again: {all:?}");
    assert_eq!(jobs_seen(d.path())[2]["resume"], true);
}

#[cfg(unix)]
#[test]
fn a_changed_sequence_length_makes_the_original_answer_again_and_restarts_the_training() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    std::fs::write(d.path().join("oom-train"), "").unwrap();
    let _ = heal_with(&p, "slim", json!({"tokens_m": 10, "general_share": 0.0, "seq": 1024})).unwrap_err();
    std::fs::remove_file(d.path().join("oom-train")).unwrap();
    heal_with(&p, "slim", json!({"tokens_m": 10, "general_share": 0.0, "seq": 512})).unwrap();
    let all = jobs_seen(d.path());
    let kinds: Vec<_> = all.iter().map(|j| j["job"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["distill_cache", "distill", "distill_cache", "distill", "distill_merge"], "{kinds:?}");
    assert_eq!((all[2]["max_seq"].clone(), all[3]["resume"].clone()), (json!(512), json!(false)), "the old checkpoints were for other rows");
}

#[test]
fn healing_specs_old_and_new_load_and_are_checked() {
    let old: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 20, "general_share": 0.5, "alpha": 1.0, "rank": 64, "lr": 1e-4, "mode": null})).unwrap();
    assert_eq!((old.rank, old.seq, old.batch, old.checkpointing.clone(), old.k, old.warmup), (Some(64), None, None, None, None, None));
    old.validate().unwrap();
    let bare: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 20})).unwrap();
    assert_eq!(bare.rank, None, "the rank follows the student's size now");
    let new = json!({"tokens_m": 20, "general_share": 0.5, "alpha": 1.0, "rank": 32, "lr": 1e-4, "mode": "cache", "seq": 1024, "batch": 2, "per_update": 32, "checkpointing": "auto", "method": "qlora",
                     "optimizer": "adamw_8bit", "kl_chunk": 128, "k": 32, "teacher_load": "auto", "warmup": 30});
    let spec: rruns::HealSpec = serde_json::from_value(new.clone()).unwrap();
    spec.validate().unwrap();
    let back = serde_json::to_value(&spec).unwrap();
    for (k, v) in new.as_object().unwrap() {
        assert_eq!(&back[k], v, "{k}");
    }
    for bad in [json!({"seq": 777}), json!({"batch": 3}), json!({"checkpointing": "maybe"}), json!({"optimizer": "sgd"}), json!({"kl_chunk": 8}), json!({"k": 500}), json!({"teacher_load": "2bit"}), json!({"method": "full"}), json!({"per_update": 0})] {
        let mut v = json!({"tokens_m": 20});
        v.as_object_mut().unwrap().extend(bad.as_object().unwrap().clone());
        let s: rruns::HealSpec = serde_json::from_value(v).unwrap();
        assert!(s.validate().unwrap_err().0.contains("out of range"), "{bad}");
    }
}

#[test]
fn the_choice_fills_the_gaps_with_the_automatic_settings() {
    let d = HealDims { layers: 40, hidden: 4096, vocab: 248_320 };
    let c = rruns::heal_choice(&serde_json::from_value(json!({"tokens_m": 20})).unwrap(), 16.0, Some(32.0), &d);
    assert_eq!((c.knobs.seq, c.knobs.batch, c.knobs.grad_accum, c.knobs.rank, c.knobs.kl_chunk, c.knobs.qlora, c.knobs.checkpointing, c.optimizer.as_str(), c.k, c.warmup), (1024, 2, 16, 32, 128, true, true, "adamw_8bit", 32, 30));
    let c = rruns::heal_choice(&serde_json::from_value(json!({"tokens_m": 20, "batch": 4, "per_update": 10, "checkpointing": "off", "method": "lora", "rank": 8})).unwrap(), 16.0, Some(32.0), &d);
    assert_eq!((c.knobs.batch, c.knobs.grad_accum, c.knobs.checkpointing, c.knobs.qlora, c.knobs.rank), (4, 3, false, false, 8), "10 sequences per update at 4 per step: 3 steps added up (12)");
    let c = rruns::heal_choice(&serde_json::from_value(json!({"tokens_m": 20, "batch": 1})).unwrap(), 3.0, Some(32.0), &HealDims::guess(3.0));
    assert_eq!((c.knobs.batch, c.knobs.grad_accum, c.knobs.seq, c.knobs.rank), (1, 32, 2048, 64), "the 32 sequences per update stay");
}

#[cfg(unix)]
#[test]
fn a_stopped_or_failed_heal_continues_where_it_was() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let dir = p.folder.join("models").join("slim");
    let go = || rruns::heal(&p, "slim", &heal_spec(10), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {});
    let kinds = || jobs_seen(d.path()).iter().map(|j| j["job"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    // stopped while the teacher answers
    std::fs::write(d.path().join("stop-cache"), "").unwrap();
    assert_eq!(go().unwrap()["stopped"], true);
    let h: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("heal.json")).unwrap()).unwrap();
    assert_eq!((h["status"].as_str(), h["step"].as_str()), (Some("stopped"), Some("cache")));
    std::fs::remove_file(d.path().join("stop-cache")).unwrap();
    // the training fails (unstable): a plain message, the cache is kept
    std::fs::write(d.path().join("fail-train"), "").unwrap();
    let e = go().unwrap_err();
    assert!(e.0.contains("became unstable") && e.0.contains("Lower the learning rate"), "{e}");
    let h: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("heal.json")).unwrap()).unwrap();
    assert_eq!(h["status"], "failed");
    assert!(h["error"].as_str().unwrap().contains("unstable") && dir.join("distill/cache/index.json").is_file());
    std::fs::remove_file(d.path().join("fail-train")).unwrap();
    // stopped while learning: a checkpoint exists
    std::fs::write(d.path().join("stop-train"), "").unwrap();
    assert_eq!(go().unwrap()["stopped"], true);
    std::fs::remove_file(d.path().join("stop-train")).unwrap();
    let before = kinds();
    assert_eq!(before, ["distill_cache", "distill_cache", "distill", "distill"], "stopped, then finished (the training failed), then stopped again: {before:?}");
    // continue: the cache is complete (not made again) and the training resumes from its checkpoint
    let r = go().unwrap();
    assert_eq!(r["model_id"], "custom-local-slim-healed");
    let all = jobs_seen(d.path());
    let last_train = all.iter().rev().find(|j| j["job"] == "distill").unwrap();
    assert_eq!(last_train["resume"], true);
    assert_eq!(all.iter().filter(|j| j["job"] == "distill_cache").count(), 2, "stopped once, then finished once; never a third time");
    assert_eq!(all.last().unwrap()["job"], "distill_merge");
}

#[cfg(unix)]
#[test]
fn healing_refuses_what_cannot_work_in_plain_words() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let go = |spec: &rruns::HealSpec, vram: Option<f64>| rruns::heal(&p, "slim", spec, vram, Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap_err().0;
    assert!(go(&heal_spec(20), Some(16.0)).contains("this card has 16 GB") && go(&heal_spec(20), Some(16.0)).contains("Choose a smaller model"));
    assert!(go(&heal_spec(15), Some(32.0)).contains("10, 20, 40 or 80"));
    assert!(go(&serde_json::from_value(json!({"tokens_m": 20, "general_share": 0.95})).unwrap(), Some(32.0)).contains("out of range"));
    assert!(go(&serde_json::from_value(json!({"tokens_m": 20, "mode": "fast"})).unwrap(), Some(32.0)).contains("Unknown way"));
    std::env::set_var("NDFORGE_FREE_DISK_GB", "1.5");
    let e = go(&heal_spec(80), Some(32.0));
    std::env::remove_var("NDFORGE_FREE_DISK_GB");
    assert!(e.contains("The teacher's answers need 15.7 GB") && e.contains("1.5 GB free"), "{e}");
    assert!(jobs_seen(d.path()).is_empty(), "nothing ran");
    assert!(rruns::heal(&p, "nope", &heal_spec(20), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).is_err());
    let q = project(&d.path().join("empty"));
    made_model(&q, "slim", 27_800_000_000, 21_100_000_000);
    assert!(rruns::heal(&q, "slim", &heal_spec(20), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap_err().0.contains("training-data version"));
}

#[cfg(unix)]
#[test]
fn a_heal_cut_off_by_closing_the_program_can_be_continued() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let p = project(d.path());
    thin_27b(&p, "slim");
    let dir = p.folder.join("models").join("slim");
    std::fs::write(dir.join("heal.json"), json!({"status": "running", "step": "train"}).to_string()).unwrap();
    std::fs::create_dir_all(dir.join("healed.partial")).unwrap();
    assert_eq!(rruns::mark_interrupted(&p), 1);
    let h: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("heal.json")).unwrap()).unwrap();
    assert_eq!((h["status"].as_str(), h["step"].as_str()), (Some("stopped"), Some("train")));
    assert!(h["error"].as_str().unwrap().contains("Continue") && !dir.join("healed.partial").exists());
}

#[test]
fn the_table_has_a_row_for_the_thinned_and_one_for_the_healed_model() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let model = made_model(&p, "slim", 27_800_000_000, 21_100_000_000);
    let dir = model.parent().unwrap();
    let scores = |lore: f64| json!({"original": {"lore": {"mean": 0.8}}, "trained": {"lore": {"mean": lore}, "rules": {"mean": 0.5}, "made_up": {"mean": 0.1}, "general": {"mean": 0.6}}});
    std::fs::write(dir.join("state.json"), json!({"name": "slim", "status": "done", "spec": {"source_model_id": "qwen3.5-4b", "mode": "prune", "blocks": 0, "width": {"intermediate": 10752}}}).to_string()).unwrap();
    std::fs::write(dir.join("test.json"), json!({"scores": scores(0.3), "verdict": {"headline": "worse"}, "n": 30, "created": "t1", "dataset_version": 1}).to_string()).unwrap();
    std::fs::write(dir.join("test_healed.json"), json!({"scores": scores(0.7), "verdict": {"headline": "better"}, "n": 30, "created": "t2", "dataset_version": 1}).to_string()).unwrap();
    let rows = eruns::compare(&p);
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0]["name"].as_str(), rows[0]["method"].as_str(), rows[0]["lore"].as_f64()), (Some("model:slim-healed"), Some("healed"), Some(0.7)), "best first");
    assert_eq!((rows[1]["name"].as_str(), rows[1]["method"].as_str(), rows[1]["lore"].as_f64()), (Some("model:slim"), Some("slim-width"), Some(0.3)));
    assert_eq!((rruns::read_test(&p, "slim").unwrap()["created"].clone(), rruns::read_test_healed(&p, "slim").unwrap()["created"].clone()), (json!("t1"), json!("t2")));
    // both ways at once, and plain layer removal, have their own words (the latter as before)
    std::fs::write(dir.join("state.json"), json!({"name": "slim", "status": "done", "spec": {"source_model_id": "qwen3.5-4b", "mode": "prune", "blocks": 1, "width": {"intermediate": 10752}}}).to_string()).unwrap();
    assert_eq!(eruns::compare(&p).iter().find(|r| r["name"] == "model:slim").unwrap()["method"], "slim-both");
    // testing the healed model needs a healed model
    let none = std::sync::atomic::AtomicBool::new(false);
    assert!(rruns::test_target(&p, "slim", "healed", None, Some(32.0), &none, &mut |_| {}).unwrap_err().0.contains("not been healed"));
    assert!(rruns::test_target(&p, "slim", "other", None, Some(32.0), &none, &mut |_| {}).is_err());
}

#[cfg(unix)]
#[test]
fn a_model_is_measured_once_and_the_measurement_is_found_again() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    let m = catalog::find("qwen3.5-4b").unwrap();
    let info = rruns::measurement_info(&p, &m, 256, 512, "auto");
    assert_eq!((info["exists"].clone(), info["key"].as_str(), info["data_version"].clone()), (json!(false), Some("unsloth-qwen3.5-4b-v1-256x512-auto"), json!(1)));
    let mut events = Vec::new();
    let r = rruns::measure(&p, "qwen3.5-4b", 256, 512, "auto", false, Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
    assert_eq!((r["key"].as_str(), r["tokens"].clone(), r["stable_depth"].clone()), (Some("unsloth-qwen3.5-4b-v1-256x512-auto"), json!(4321), json!(true)));
    let job = &jobs_seen(d.path())[0];
    assert_eq!((job["job"].as_str(), job["force"].clone(), job["load"].as_str(), job["calibration"]["samples"].clone(), job["calibration"]["version"].clone()), (Some("reshape_measure"), json!(false), Some("auto"), json!(256), json!(1)));
    assert_eq!(job["out_dir"].as_str().unwrap(), rruns::scores_dir(&p, "unsloth-qwen3.5-4b-v1-256x512-auto").to_string_lossy());
    assert!(job["calibration"]["train"].as_str().unwrap().ends_with("train.jsonl") && events.iter().any(|e| e["kind"] == "progress" && e["step"] == "score"));
    let info = rruns::measurement_info(&p, &m, 256, 512, "auto");
    assert_eq!((info["exists"].clone(), info["tokens"].clone(), info["load"].clone()), (json!(true), json!(4321), json!("4bit")));
    assert_eq!(rruns::measurement_info(&p, &m, 128, 512, "auto")["exists"], false, "other settings, other measurement");
    rruns::measure(&p, "qwen3.5-4b", 256, 512, "auto", true, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap();
    assert_eq!(jobs_seen(d.path())[1]["force"], true);
    for (samples, len, load) in [(2, 512, "auto"), (256, 8, "auto"), (256, 512, "8bit")] {
        assert!(rruns::measure(&p, "qwen3.5-4b", samples, len, load, false, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("Measure on"));
    }
    assert!(rruns::measure(&p, "nope", 256, 512, "auto", false, Arc::new(AtomicBool::new(false)), &mut |_| {}).is_err());
    let q = project(&d.path().join("empty"));
    assert!(rruns::measure(&q, "qwen3.5-4b", 256, 512, "auto", false, Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap_err().0.contains("training-data version"));
}

#[cfg(unix)]
#[test]
fn the_heal_card_knows_the_way_the_budgets_and_the_state() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let o = rruns::heal_options(&p, "slim", Some(32.0)).unwrap();
    assert_eq!((o["plan"]["mode"].clone(), o["plan"]["student_method"].clone(), o["plan"]["fits"].clone()), (json!("cache"), json!("qlora"), json!(true)));
    assert_eq!((o["default_tokens_m"].clone(), o["healed"].clone(), o["has_data"].clone(), o["teacher"].clone()), (json!(20), json!(false), json!(true), json!("unsloth/Qwen3.5-4B")));
    let t = o["tokens"].as_array().unwrap();
    assert_eq!(t.iter().map(|x| x["tokens_m"].as_u64().unwrap()).collect::<Vec<_>>(), [10, 20, 40, 80]);
    assert_eq!(t[1]["cache_gb"], 3.9, "20 M tokens x 196 bytes");
    assert!(t[3]["hours"][0].as_f64().unwrap() > t[0]["hours"][0].as_f64().unwrap());
    assert_eq!((o["loss"]["original"].clone(), o["loss"]["thinned"].clone(), o["general_dataset"].clone()), (json!(2.0), json!(3.1), json!("mlabonne/open-perfectblend")));
    assert!(o["metrics"].as_array().unwrap().is_empty() && o["heal"].is_null());
    rruns::heal(&p, "slim", &heal_spec(20), Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap();
    let o = rruns::heal_options(&p, "slim", Some(32.0)).unwrap();
    assert_eq!((o["healed"].clone(), o["heal"]["status"].clone(), o["metrics"].as_array().unwrap().len()), (json!(true), json!("done"), 2));
    assert!(rruns::heal_options(&p, "nope", None).is_err());
    assert_eq!(rruns::heal_options(&p, "slim", Some(16.0)).unwrap()["plan"]["fits"], false);
}

#[cfg(unix)]
#[test]
fn the_page_gets_width_options_for_every_way_of_slimming() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker_with(d.path(), &inspect_width(), MAKES_A_THIN_MODEL);
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects"));
    let depth = AppModel::reshape_options_for("qwen3.5-4b", "prune").unwrap();
    assert_eq!(depth["shape"], "depth");
    assert!(depth.get("width_options").is_none(), "layers only: as before");
    assert_eq!((depth["width"]["supported"].clone(), depth["width"]["suggested"]["width"].clone(), depth["width"]["suggested"]["both"]["blocks"].clone()), (json!(true), json!(7168), json!(1)));
    assert_eq!((depth["width"]["default_heal_tokens_m"].clone(), depth["width"]["heal_tokens_m"].clone()), (json!(40), json!([10, 20, 40, 80])));
    let w = AppModel::reshape_options_with("qwen3.5-4b", "prune", "width", None, None).unwrap();
    let groups = w["width_options"].as_array().unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0]["blocks"], 0);
    let opts = groups[0]["options"].as_array().unwrap();
    assert_eq!(opts.len(), 19);
    assert_eq!((opts[0]["intermediate"].clone(), opts[18]["intermediate"].clone(), opts[0]["layers"].clone()), (json!(4608), json!(9216), json!(12)));
    assert!(opts[0]["params_b"].as_f64().unwrap() < opts[18]["params_b"].as_f64().unwrap() && opts[0]["exact"] == true && opts[0]["disk_needed_gb"].as_f64().unwrap() > 0.0);
    assert!(opts[0]["warnings"].as_array().unwrap().iter().any(|x| x.as_str().unwrap().contains("Removing 50 % of the MLP")));
    // both: a group for every number of blocks that can go (3 blocks: one)
    let b = AppModel::reshape_options_with("qwen3.5-4b", "prune", "both", Some(16), Some(12)).unwrap();
    let groups = b["width_options"].as_array().unwrap();
    assert_eq!(groups.iter().map(|g| g["blocks"].as_u64().unwrap()).collect::<Vec<_>>(), [1]);
    assert_eq!(groups[0]["options"][0]["layers"], 8);
    assert!(groups[0]["options"][0]["warnings"].as_array().unwrap().iter().any(|x| x.as_str().unwrap().contains("experimental")));
    // grow has no width; unknown shapes are refused
    assert!(AppModel::reshape_options_with("qwen3.5-4b", "grow", "width", None, None).unwrap().get("width_options").is_none());
    assert!(AppModel::reshape_options_with("qwen3.5-4b", "prune", "tall", None, None).is_err());
}

// ---------------------------------------------------------------------------------------------------------------- the general text of healing

#[cfg(unix)]
fn healer_env(root: &std::path::Path) -> Vec<(String, Option<String>)> {
    std::fs::read_to_string(root.join("env.jsonl")).unwrap_or_default().lines().map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
        .map(|v| (v["job"].as_str().unwrap().to_string(), v["hf"].as_str().map(String::from))).collect()
}

#[cfg(unix)]
#[test]
fn healing_uses_the_chosen_dataset_and_the_token_only_through_the_environment() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let token = format!("hf_{}", "Qq7".repeat(12));
    ndforge_core::general::token_set(&token).unwrap();
    let spec: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 10, "general": {"id": "HuggingFaceH4/ultrachat_200k", "column": "messages"}})).unwrap();
    rruns::heal(&p, "slim", &spec, Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap();
    let jobs = jobs_seen(d.path());
    let g = &jobs[0]["data"]["general"];
    assert_eq!((g["id"].as_str(), g["split"].as_str(), g["column"].as_str(), g["share"].clone()), (Some("HuggingFaceH4/ultrachat_200k"), Some("train_sft"), Some("messages"), json!(0.5)));
    assert_eq!(jobs[1]["dataset"]["general"], *g);
    assert!(jobs.iter().all(|j| !j.to_string().contains(&token) && !j.to_string().contains("hf_")), "the token is never in a job file");
    let env = healer_env(d.path());
    assert_eq!(env.iter().map(|(j, t)| (j.as_str(), t.is_some())).collect::<Vec<_>>(), [("distill_cache", true), ("distill", true), ("distill_merge", false)]);
    assert!(env.iter().filter_map(|(_, t)| t.as_ref()).all(|t| *t == token));
    // what the page reads back says which dataset was used or skipped, and the token is nowhere in it
    let h = std::fs::read_to_string(p.folder.join("models/slim/heal.json")).unwrap();
    assert!(h.contains("ultrachat_200k") && h.contains("general_used") && !h.contains(&token));
    // without a saved token no variable is set
    ndforge_core::general::token_clear().unwrap();
    assert!(ndforge_core::envmgr::jobs::job_env(&jobs[0]).is_empty());
}

#[cfg(unix)]
#[test]
fn another_dataset_never_reuses_the_teachers_answers_or_the_old_checkpoints() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let go = |id: &str| {
        let spec: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 10, "general": {"id": id}})).unwrap();
        rruns::heal(&p, "slim", &spec, Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {})
    };
    let kinds = || jobs_seen(d.path()).iter().map(|j| j["job"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    std::fs::write(d.path().join("stop-train"), "").unwrap();
    assert_eq!(go("a/one").unwrap()["stopped"], true);
    assert_eq!(go("a/one").unwrap()["stopped"], true);
    assert_eq!(kinds(), ["distill_cache", "distill", "distill"], "the same dataset: the cache is kept");
    assert_eq!(jobs_seen(d.path())[2]["resume"], true, "and the checkpoint is continued");
    assert_eq!(go("a/two").unwrap()["stopped"], true);
    assert_eq!(kinds(), ["distill_cache", "distill", "distill", "distill_cache", "distill"], "another dataset: the teacher answers again");
    assert_eq!(jobs_seen(d.path())[4]["resume"], false, "and the other dataset's checkpoint is not continued");
    // an older cache (no key) counts as other data
    let idx = p.folder.join("models/slim/distill/cache/index.json");
    std::fs::write(&idx, r#"{"complete": true, "rows": 10, "done_rows": 10}"#).unwrap();
    std::fs::remove_file(d.path().join("stop-train")).unwrap();
    go("a/two").unwrap();
    assert_eq!(kinds().iter().filter(|k| *k == "distill_cache").count(), 3);
}

#[cfg(unix)]
#[test]
fn a_dataset_that_needs_an_acknowledgement_is_not_used_silently() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let _disk = PlentyOfDisk::new();
    let d = tempdir().unwrap();
    fake_healer(d.path());
    let p = project_with_version(d.path());
    thin_27b(&p, "slim");
    let id = "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation";
    let spec: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 10, "general": {"id": id}})).unwrap();
    let go = || rruns::heal(&p, "slim", &spec, Some(32.0), Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {});
    let e = go().unwrap_err();
    assert!(e.0.contains("needs your OK") && e.0.contains("Noncommercial research"), "{e}");
    assert!(jobs_seen(d.path()).is_empty(), "nothing ran");
    ndforge_core::general::acknowledge(&format!("{id}#sft_balanced")).unwrap();
    go().unwrap();
    assert_eq!(jobs_seen(d.path())[0]["data"]["general"]["config"], "sft_balanced");
}

#[cfg(unix)]
#[test]
fn the_old_dataset_fields_still_work_and_zero_share_means_no_general_text() {
    let spec: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 10, "general_dataset": "a/b", "general_column": "body"})).unwrap();
    let g = spec.general_source().unwrap().unwrap();
    assert_eq!((g.id.as_deref(), g.column.as_deref()), (Some("a/b"), Some("body")));
    let none: rruns::HealSpec = serde_json::from_value(json!({"tokens_m": 10, "general_share": 0.0})).unwrap();
    assert!(none.general_source().unwrap().is_none());
    assert_eq!(rruns::DEFAULT_GENERAL_DATASET, "mlabonne/open-perfectblend");
}
