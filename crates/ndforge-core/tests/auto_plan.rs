//! The Auto-cut planner (`reshape::auto_plan`): the exact sizes of the real Qwen3.5 / 3.8 27B, the cuts it picks, the stages, the disk and the card.
use ndforge_core::reshape::auto_plan::*;
use ndforge_core::reshape::plan::*;
use serde_json::json;

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("worker").join("tests").join("data")
}

fn dims_of(tc: &serde_json::Value) -> WidthDims {
    let u = |k: &str| tc[k].as_u64().unwrap_or(0) as u32;
    WidthDims {
        hidden: u("hidden_size"), intermediate: u("intermediate_size"), heads: u("num_attention_heads"), kv_heads: u("num_key_value_heads"), head_dim: u("head_dim"),
        lin_k_heads: u("linear_num_key_heads"), lin_v_heads: u("linear_num_value_heads"), lin_k_dim: u("linear_key_head_dim"), lin_v_dim: u("linear_value_head_dim"),
        conv_kernel: u("linear_conv_kernel_dim"), mtp_layers: u("mtp_num_hidden_layers"), vocab: u("vocab_size"), tied: tc["tie_word_embeddings"].as_bool().unwrap_or(false),
    }
}

/// The real 27B: 64 layers in 16 blocks, MLP 17,408 (the config is the worker's test data)
fn qwen27() -> Inspect {
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(data_dir().join("qwen3_5_27b_config.json")).unwrap()).unwrap();
    let tc = &cfg["text_config"];
    let d = dims_of(tc);
    let types: Vec<String> = tc["layer_types"].as_array().unwrap().iter().map(|t| t.as_str().unwrap().to_string()).collect();
    let layer_params = vec![390_000_000u64; 64];
    Inspect {
        model_type: "qwen3_5".into(), supported: true, layers: 64, layer_types: types, period: 4, blocks: 16, vision_params: 460_730_096, other_params: d.hidden as u64,
        total_params: 27_781_427_952, bytes_per_param: 2.0, has_vision: true, local: true, width_supported: true, layer_params,
        width_choices: Some(WidthChoices { intermediate: (34..=68).map(|k| k * 256).collect(), lin_v_heads: vec![48, 32], heads: vec![24, 20, 16] }),
        width_dims: Some(d), ..Default::default()
    }
}

fn spec(target: f64) -> AutoSpec {
    AutoSpec::new("qwen3.8-27b", target)
}

#[test]
fn the_exact_sizes_of_the_27b() {
    let ins = qwen27();
    assert_eq!(size_of(&ins, 0, Some(17408)), 27_781_427_952, "the original, from the formula");
    assert_eq!(size_of(&ins, 0, Some(10752)) / 10_000_000, 2081, "20.81 B: the thinnest MLP the width-first rule allows (the multi-token head is removed by a width cut, so 0.32 B less than before)");
    assert_eq!(size_of(&ins, 1, Some(12288)) / 10_000_000, 2111);
    assert_eq!(width_floor(&ins), Some(10752), "17,408 x 0.617 = 10,741 -> the stop 10,752");
}

#[test]
fn twenty_one_billion_is_the_mlp_alone() {
    let ins = qwen27();
    let c = candidates(&ins, 21.1);
    let best = &c[0];
    assert_eq!((best.blocks, best.intermediate, best.params_b, best.on_target, best.needs_confirmation), (0, Some(11008), 21.07, true, false), "{best:?}");
    assert_eq!(best.layers, 64);
    assert!((best.width_cut - 0.368).abs() < 1e-9 && best.error_pct == -0.2, "{best:?}");
    // width first: blocks come only when the MLP alone cannot reach the size
    let p = plan(&ins, &spec(21.1), Some(32.0), Some(900.0));
    assert!(p.ok, "{:?}", p.blockers);
    assert_eq!(p.chosen.as_ref().map(|c| (c.blocks, c.intermediate)), Some((0, Some(11008))));
    assert_eq!(p.try_list[0], *p.chosen.as_ref().unwrap());
    assert!(p.try_list.len() <= TRY_TOP);
}

#[test]
fn below_the_mlp_floor_whole_blocks_come_in_up_to_four() {
    let ins = qwen27();
    // 18 B: the thinnest MLP gives 21.1 B, so blocks are removed as well
    let p = plan(&ins, &spec(18.0), Some(32.0), Some(900.0));
    let c = p.chosen.clone().unwrap();
    assert!(p.ok, "{:?}", p.blockers);
    assert!(c.blocks >= 1 && c.blocks <= MAX_BLOCKS_FREE && c.on_target && c.intermediate.unwrap() >= 10752, "{c:?}");
    // every candidate with fewer blocks that is on target comes first
    let on: Vec<u32> = p.candidates.iter().filter(|x| x.on_target).map(|x| x.blocks).collect();
    assert!(on.windows(2).all(|w| w[0] <= w[1]), "{on:?}");
    // 14 B needs more than four blocks: refused until the owner confirms
    let hard = plan(&ins, &spec(14.0), Some(32.0), Some(900.0));
    assert!(!hard.ok && hard.chosen.is_none() && hard.blockers[0].contains("more than 4 whole blocks"), "{:?}", hard.blockers);
    assert!(hard.candidates.iter().any(|c| c.on_target && c.needs_confirmation));
    let ok = plan(&ins, &AutoSpec { allow_many_blocks: true, ..spec(14.0) }, Some(32.0), Some(900.0));
    assert!(ok.ok && ok.chosen.as_ref().unwrap().blocks > MAX_BLOCKS_FREE, "{:?}", ok.blockers);
}

#[test]
fn sizes_the_model_cannot_reach_are_refused_in_plain_words() {
    let ins = qwen27();
    let big = plan(&ins, &spec(27.5), None, None);
    assert!(!big.ok && big.blockers[0].contains("3 % smaller"), "{:?}", big.blockers);
    let tiny = plan(&ins, &spec(3.0), None, None);
    assert!(!tiny.ok && tiny.blockers[0].contains("smaller than the cut that is allowed"), "{:?}", tiny.blockers);
    let bad = plan(&ins, &spec(0.2), None, None);
    assert!(!bad.ok && bad.blockers[0].contains("between 1 and 1000"));
    let mut other = qwen27();
    other.model_type = "llama".into();
    assert!(plan(&other, &spec(21.1), None, None).blockers[0].contains("Keep thinking is only for Qwen3.5"));
    let mut off = spec(21.1);
    off.keep_thinking = false;
    assert!(plan(&other, &off, None, None).blockers.iter().all(|b| !b.contains("Keep thinking")));
    let mut unsupported = qwen27();
    unsupported.supported = false;
    unsupported.why = Some("not this model".into());
    assert_eq!(plan(&unsupported, &spec(21.1), None, None).blockers, vec!["not this model".to_string()]);
}

#[test]
fn a_thinking_model_below_twenty_billion_is_warned_about() {
    let ins = qwen27();
    let p = plan(&ins, &spec(18.0), Some(32.0), Some(900.0));
    assert!(p.warnings.iter().any(|w| w.contains("will not reason like the original")), "{:?}", p.warnings);
    let q = plan(&ins, &spec(21.1), Some(32.0), Some(900.0));
    assert!(!q.warnings.iter().any(|w| w.contains("will not reason")));
    let mut plain = spec(18.0);
    plain.keep_thinking = false;
    assert!(!plan(&ins, &plain, Some(32.0), Some(900.0)).warnings.iter().any(|w| w.contains("will not reason")));
}

#[test]
fn two_stages_cut_half_of_the_width_first_and_the_depth_before_the_width() {
    let ins = qwen27();
    let p = plan(&ins, &spec(21.1), Some(32.0), Some(900.0));
    assert_eq!(p.stages.len(), 2);
    let (a, b) = (&p.stages[0], &p.stages[1]);
    // 17,408 -> 14,080 (nearest stop to the middle of 17,408 and 11,008: 14,080) -> 11,008
    assert_eq!((a.intermediate_before, a.intermediate_after, a.width, a.blocks_cut, a.index, a.of), (Some(17408), Some(14080), Some(14080), 0, 1, 2), "{a:?}");
    assert_eq!((b.intermediate_before, b.intermediate_after, b.width, b.blocks_cut), (Some(14080), Some(11008), Some(11008), 0), "{b:?}");
    assert!(a.params > b.params && b.params_b == 21.07);
    // blocks and width: the blocks go in stage A, the width in stage B
    let c = Candidate { blocks: 2, intermediate: Some(12288), ..p.chosen.clone().unwrap() };
    let steps = split_stages(&ins, &c, 2);
    assert_eq!(steps, vec![Step { blocks: 2, intermediate: Some(17408) }, Step { blocks: 2, intermediate: Some(12288) }]);
    // blocks alone: half of them in each stage; one block alone cannot be divided
    let blocks_only = Candidate { blocks: 4, intermediate: Some(17408), ..c.clone() };
    assert_eq!(split_stages(&ins, &blocks_only, 2), vec![Step { blocks: 2, intermediate: Some(17408) }, Step { blocks: 4, intermediate: Some(17408) }]);
    assert_eq!(split_stages(&ins, &Candidate { blocks: 1, intermediate: Some(17408), ..c.clone() }, 2).len(), 1);
    assert_eq!(split_stages(&ins, &c, 1), vec![Step { blocks: 2, intermediate: Some(12288) }]);
    // one stage on request
    let one = plan(&ins, &AutoSpec { stages: 1, ..spec(21.1) }, Some(32.0), Some(900.0));
    assert_eq!(one.stages.len(), 1);
    assert_eq!((one.stages[0].blocks_cut, one.stages[0].width), (0, Some(11008)));
}

#[test]
fn the_stage_cuts_and_heals_are_specs_the_existing_jobs_take() {
    let ins = qwen27();
    let p = plan(&ins, &spec(21.1), Some(32.0), Some(900.0));
    let s = p.stages[1].reshape_spec("stage-a-model", Some(ReasoningCalib::default()), 128, 256);
    assert_eq!((s.source_model_id.as_str(), s.blocks, s.width.clone().map(|w| w.intermediate), s.width_samples), ("stage-a-model", 0, Some(11008), 256));
    assert_eq!(s.reasoning, Some(ReasoningCalib::default()));
    let h = p.stages[0].heal_spec(true);
    assert_eq!((h.tokens_m, h.recipe.as_deref()), (40, Some("thinking")));
    assert_eq!(p.stages[0].heal_spec(false).recipe, None);
}

#[test]
fn the_heal_numbers_of_the_thinking_recipe_per_stage() {
    let ins = qwen27();
    let p = plan(&ins, &spec(21.1), Some(32.0), Some(900.0));
    // stage A: 40 M tokens; the final stage: 40 M too, the 21.14 B student is a cut of 24 % (80 M only over 30 %)
    assert_eq!(p.stages[0].tokens_m, 40);
    assert_eq!(p.stages[1].tokens_m, 40, "a cut of {} % is not over 30 %", p.stages[1].cut_pct);
    assert_eq!((p.stages[1].batch, p.stages[1].rank), (1, 64));
    assert!([1024, 2048, 4096, 8192].contains(&p.stages[1].seq));
    assert_eq!(p.stages[1].cache_gb, 7.8, "40 M tokens x 196 bytes");
    // a deeper cut gets 80 M for the last stage
    let deep = plan(&ins, &AutoSpec { allow_many_blocks: true, ..spec(16.0) }, Some(32.0), Some(900.0));
    assert!(deep.stages.last().unwrap().cut_pct > 30.0 && deep.stages.last().unwrap().tokens_m == 80, "{:?}", deep.stages);
    assert_eq!(deep.stages[0].tokens_m, 40);
    // without the recipe: the plain defaults (20 M from 15 B)
    let plain = plan(&ins, &AutoSpec { keep_thinking: false, ..spec(21.1) }, Some(32.0), Some(900.0));
    assert_eq!((plain.stages[1].tokens_m, plain.stages[1].rank), (20, 32));
    assert!(p.heal_hours.0 > 0.0 && p.heal_hours.1 >= p.heal_hours.0);
}

#[test]
fn the_card_is_checked_for_every_stage() {
    let ins = qwen27();
    let small = plan(&ins, &spec(21.1), Some(8.0), Some(900.0));
    assert!(!small.ok && small.blockers.iter().any(|b| b.contains("Healing stage") && b.contains("8 GB")), "{:?}", small.blockers);
    let none = plan(&ins, &spec(21.1), None, Some(900.0));
    assert!(none.ok, "an unknown card does not block: {:?}", none.blockers);
    assert!(none.stages.iter().all(|s| s.verdict == "unknown" || s.verdict == "fits" || s.verdict == "tight"));
    assert_eq!(plan(&ins, &spec(21.1), Some(32.0), None).measure_load, "4bit");
    assert_eq!(plan(&ins, &spec(21.1), Some(80.0), None).measure_load, "16bit");
}

#[test]
fn the_disk_timeline_and_the_peak() {
    let ins = qwen27();
    let p = plan(&ins, &spec(21.1), Some(32.0), Some(900.0));
    let d = p.disk.clone().unwrap();
    // original 55.6 GB; stage A cut 2 x ... the peak is while stage B is being healed: original + stage A healed + cut + answers + healed
    assert_eq!(d.steps[0], DiskStep { step: "the original model".into(), in_use_gb: 55.6 });
    let a = p.stages[0].params as f64 * 2.0 / 1e9;
    let b = p.stages[1].params as f64 * 2.0 / 1e9;
    let expect_peak = 55.6 + a + b + p.stages[1].cache_gb + b - 0.0;
    assert!(d.peak_gb >= 55.6 + a + 2.0 * b && (d.peak_gb - expect_peak).abs() < 3.0, "{d:?} {expect_peak}");
    assert_eq!((d.peak_gb, d.final_gb), (195.8, 110.5));
    assert_eq!((p.stages[0].params_b, p.stages[0].cut_pct, p.stages[1].params_b, p.stages[1].cut_pct), (24.09, 13.3, 21.07, 24.2));
    assert_eq!(d.peak_step, "stage 2: the healed model");
    assert_eq!(d.enough, Some(true));
    // keeping everything needs more than cleaning up
    let keep = plan(&ins, &AutoSpec { keep_intermediate: true, ..spec(21.1) }, Some(32.0), Some(900.0)).disk.unwrap();
    assert!(keep.peak_gb > d.peak_gb && keep.final_gb > d.final_gb, "{keep:?} {d:?}");
    assert_eq!((keep.peak_gb, keep.peak_step.as_str()), (306.7, "export: the sizes"));
    // the original and the final model stay in both cases
    assert!(d.final_gb >= 55.6 + b, "{d:?}");
    // a small disk blocks the run and says what to do
    let tight = plan(&ins, &AutoSpec { keep_intermediate: true, ..spec(21.1) }, Some(32.0), Some(150.0));
    assert!(!tight.ok && tight.blockers.iter().any(|b| b.contains("GB of disk") && b.contains("keep the intermediate files")), "{:?}", tight.blockers);
    let not_known = plan(&ins, &spec(21.1), Some(32.0), None).disk.unwrap();
    assert_eq!((not_known.free_gb, not_known.enough), (None, None));
    // a model that is not on this PC needs its download as well
    let mut remote = qwen27();
    remote.local = false;
    assert!(plan(&remote, &spec(21.1), Some(32.0), Some(900.0)).disk.unwrap().peak_gb > d.peak_gb + 50.0);
}

#[test]
fn backing_off_gives_back_one_block_or_four_mlp_stops() {
    let ins = qwen27();
    assert_eq!(back_off(&ins, 3, Some(12288)), Some((2, Some(12288))));
    assert_eq!(back_off(&ins, 0, Some(10752)), Some((0, Some(11776))), "1,024 more");
    assert_eq!(back_off(&ins, 0, Some(17152)), Some((0, Some(17408))), "never past the original");
    assert_eq!(back_off(&ins, 0, Some(17408)), None);
    assert_eq!(back_off(&ins, 0, None), None);
}

#[test]
fn a_model_that_cannot_be_thinned_is_cut_by_blocks_only() {
    let mut ins = qwen27();
    ins.width_supported = false;
    ins.width_dims = None;
    ins.width_choices = None;
    let mut s = spec(24.0);
    s.keep_thinking = true;
    let p = plan(&ins, &s, Some(32.0), Some(900.0));
    assert!(p.ok, "{:?}", p.blockers);
    let c = p.chosen.unwrap();
    assert_eq!((c.intermediate, c.blocks > 0), (None, true), "{c:?}");
    assert!(p.stages.iter().all(|s| s.width.is_none()));
}

#[test]
fn the_spec_has_the_owner_defaults_and_refuses_nonsense() {
    let s: AutoSpec = serde_json::from_value(json!({"model_id": "m", "target_b": 20.0})).unwrap();
    assert_eq!((s.keep_thinking, s.stages, s.quick, s.keep_intermediate, s.allow_many_blocks, s.studio_publish), (true, 2, false, false, false, false));
    assert_eq!(s.export_quants(), vec!["q4_k_m".to_string()]);
    let more = AutoSpec { quants: vec!["q8_0".into(), "q4_k_m".into()], ..s.clone() };
    assert_eq!(more.export_quants(), vec!["q4_k_m".to_string(), "q8_0".to_string()]);
    assert_eq!((s.try_samples(), AutoSpec { quick: true, ..s.clone() }.try_samples()), (128, 64));
    for bad in [json!({"stages": 3}), json!({"quants": ["q9"]}), json!({"target_b": 0.5})] {
        let mut v = json!({"model_id": "m", "target_b": 20.0});
        v.as_object_mut().unwrap().extend(bad.as_object().unwrap().clone());
        let s: AutoSpec = serde_json::from_value(v).unwrap();
        assert!(s.validate().is_err(), "{s:?}");
    }
}

#[test]
fn the_owners_first_cut_is_beyond_the_guardrails_and_the_checks_take_an_estimated_time() {
    use ndforge_core::reshape::auto_plan::{beyond_guardrails, think_gate_minutes};
    let ins = qwen27();
    // cut A of a real run: three blocks and the MLP at 10,240 of 17,408 (58.8 % kept; the floor is 10,752)
    let a = beyond_guardrails(&ins, 3, Some(10240));
    assert_eq!(a.len(), 1, "{a:?}");
    assert!(a[0].contains("10240 of 17408") && a[0].contains("58.8 %") && a[0].contains("10752"), "{a:?}");
    // the floor itself, and a few blocks, are inside
    assert!(beyond_guardrails(&ins, 4, Some(10752)).is_empty());
    assert!(beyond_guardrails(&ins, 0, Some(17408)).is_empty());
    let both = beyond_guardrails(&ins, 5, Some(10240));
    assert_eq!(both.len(), 2, "{both:?}");
    assert!(both[1].contains("5 whole blocks"));
    // a model that cannot be thinned has no width rule
    let mut plain = qwen27();
    plain.width_supported = false;
    assert!(beyond_guardrails(&plain, 1, None).is_empty());
    // the estimate: write at 500 MB/s plus 15 (full) or 10 (quick) minutes per 17 B, at least once
    let cands = candidates(&ins, 17.4);
    let c = &cands[0];
    let (full, quick) = (think_gate_minutes(&ins, c, false), think_gate_minutes(&ins, c, true));
    assert!(quick < full && full > 15.0 && full < 25.0, "{full} {quick}");
    let small = Candidate { params: 5_000_000_000, ..c.clone() };
    assert!(think_gate_minutes(&ins, &small, true) >= 10.0, "never less than one unit of the check");
}
