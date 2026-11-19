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

// ---- the speed settings (reshape::auto_speed): presets as data, and a legacy spec is the recommended recipe

#[test]
fn an_old_spec_without_the_speed_fields_loads_with_the_recipe_defaults() {
    let old = json!({"model_id": "qwen3.8-27b", "target_b": 21.1, "keep_thinking": true, "stages": 2, "quick": false, "keep_intermediate": false, "allow_many_blocks": false,
                     "quants": ["q4_k_m"], "studio_publish": false, "think_gate": true, "think_gate_n": 2});
    let s: AutoSpec = serde_json::from_value(old).unwrap();
    assert_eq!(s, spec(21.1));
    assert!(s.check_cut && s.retry_heal && s.back_off && s.imatrix && !s.share_teacher_cache);
    assert_eq!((s.preset.clone(), s.stage_a_tokens_m, s.last_tokens_m, s.think_quick, s.max_hours), (None, None, None, None, None));
    assert!(!s.think_quick(), "the checks follow `quick`");
    assert!(AutoSpec { quick: true, ..s }.think_quick());
}

#[test]
fn the_presets_are_data_and_each_one_is_a_valid_spec() {
    let ps = ndforge_core::reshape::auto_speed::presets();
    assert_eq!(ps.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(), ["thorough", "balanced", "fast", "early", "custom"]);
    for p in &ps {
        assert!(!p.label.is_empty() && !p.blurb.is_empty() && !p.risk.is_empty(), "{p:?}");
        for k in p.set.keys() {
            assert!(ndforge_core::reshape::auto_speed::KNOBS.contains(&k.as_str()), "{k} is not a speed knob");
        }
        let mine = AutoSpec { keep_intermediate: true, studio_publish: true, target_gb: Some(10.0), ..spec(21.1) }.with_preset(&p.id).unwrap();
        mine.validate().unwrap();
        assert_eq!(mine.preset.as_deref(), Some(p.id.as_str()));
        // what is not a speed knob stays
        assert!(mine.keep_intermediate && mine.studio_publish && mine.target_gb == Some(10.0) && mine.target_b == 21.1 && mine.model_id == "qwen3.8-27b");
    }
    assert!(spec(21.1).with_preset("nope").is_err());
    assert!(AutoSpec { preset: Some("nope".into()), ..spec(21.1) }.validate().is_err());
}

#[test]
fn the_legacy_spec_is_the_thorough_recipe_and_plans_the_same() {
    let ins = qwen27();
    let legacy = spec(21.1);
    let thorough = legacy.with_preset("thorough").unwrap();
    // the presets also say how long the search for the best cut looks (Thorough 30, Balanced 20, Fast 10 minutes); a legacy spec has none and Try first stays the classic one
    assert_eq!([("thorough", Some(30)), ("balanced", Some(20)), ("fast", Some(10))].map(|(id, m)| (legacy.with_preset(id).unwrap().search_minutes, m)).iter().all(|(a, b)| a == b), true);
    assert_eq!(legacy.search_minutes, None);
    assert_eq!(legacy.matching_preset().as_deref(), Some("thorough"));
    assert_eq!(thorough.matching_preset().as_deref(), Some("thorough"));
    let thorough = AutoSpec { search_minutes: None, ..thorough };
    let (a, b) = (plan(&ins, &legacy, Some(32.0), Some(900.0)), plan(&ins, &thorough, Some(32.0), Some(900.0)));
    assert!(a.ok, "{:?}", a.blockers);
    assert_eq!(a, b, "the plan of a legacy spec and of Thorough are the same");
    // a single changed knob is not a preset any more
    assert_eq!(AutoSpec { check_cut: false, ..legacy.clone() }.matching_preset(), None);
    assert_eq!(legacy.with_preset("balanced").unwrap().matching_preset().as_deref(), Some("balanced"));
    assert_eq!(legacy.with_preset("fast").unwrap().matching_preset().as_deref(), Some("fast"));
    // the knobs must be set to something that exists
    assert!(AutoSpec { last_tokens_m: Some(30), ..legacy.clone() }.validate().is_err());
    assert!(AutoSpec { stage_a_tokens_m: Some(15), ..legacy.clone() }.validate().is_err());
    assert!(AutoSpec { max_hours: Some(0.5), ..legacy.clone() }.validate().is_err());
    assert!(AutoSpec { max_hours: Some(48.0), last_tokens_m: Some(20), ..legacy }.validate().is_ok());
}

#[test]
fn the_default_preset_is_balanced_for_big_models_only() {
    assert_eq!(ndforge_core::reshape::auto_speed::default_preset(27.8), "balanced");
    assert_eq!(ndforge_core::reshape::auto_speed::default_preset(15.0), "balanced");
    assert_eq!(ndforge_core::reshape::auto_speed::default_preset(8.0), "thorough");
}

fn big(target: f64) -> AutoSpec {
    AutoSpec { allow_many_blocks: true, ..spec(target) }
}

#[test]
fn the_speed_knobs_set_the_tokens_of_the_stages() {
    let ins = qwen27();
    let base = plan(&ins, &big(17.4), Some(32.0), Some(2000.0));
    assert_eq!(base.stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>(), [40, 80], "the recipe");
    assert!(base.warnings.iter().all(|w| !w.contains("recipe is") && !w.contains("recipe for a cut")), "{:?}", base.warnings);
    let p = plan(&ins, &AutoSpec { stage_a_tokens_m: Some(20), last_tokens_m: Some(40), ..big(17.4) }, Some(32.0), Some(2000.0));
    assert_eq!(p.stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>(), [20, 40]);
    // below the recipe is said loudly, the recipe's own numbers are not
    assert!(p.warnings.iter().any(|w| w.contains("last stage heals on 40 M tokens") && w.contains("recipe for a cut of this size is 80 M") && w.contains("failed the thinking checks")), "{:?}", p.warnings);
    assert!(p.warnings.iter().any(|w| w.contains("first stage heals on 20 M tokens") && w.contains("recipe is 40 M")), "{:?}", p.warnings);
    let up = plan(&ins, &AutoSpec { stage_a_tokens_m: Some(80), last_tokens_m: Some(80), ..big(17.4) }, Some(32.0), Some(2000.0));
    assert_eq!(up.stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>(), [80, 80]);
    assert!(up.warnings.iter().all(|w| !w.contains("recipe")), "{:?}", up.warnings);
    // one stage: the last-stage number is the only one
    let one = plan(&ins, &AutoSpec { stages: 1, last_tokens_m: Some(20), stage_a_tokens_m: Some(10), ..spec(21.1) }, Some(32.0), Some(2000.0));
    assert_eq!(one.stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>(), [20]);
}

#[test]
fn every_value_below_the_recipe_is_a_warning_and_the_recipe_has_none() {
    let ins = qwen27();
    let recipe = plan(&ins, &spec(21.1), Some(32.0), Some(2000.0));
    assert!(recipe.ok && recipe.warnings.iter().all(|w| w.contains("tight on this card")), "only the card's own warning: {:?}", recipe.warnings);
    let fast = plan(&ins, &spec(21.1).with_preset("fast").unwrap(), Some(32.0), Some(2000.0));
    assert!(fast.ok, "{:?}", fast.blockers);
    let all = fast.warnings.join(" | ");
    for want in ["last stage heals on 20 M tokens", "One stage cuts", "not checked for thinking", "quick ones", "64 texts", "No retry", "No back-off", "No importance matrix"] {
        assert!(all.contains(want), "{want}: {all}");
    }
}

#[test]
fn the_shared_cache_pins_one_sequence_length_and_the_stage_heals_send_it() {
    let ins = qwen27();
    // a card on which the larger student heals on 4096 and the smaller one on 8192 (36 GB)
    let alone = plan(&ins, &big(17.4), Some(36.0), Some(2000.0));
    let nat: Vec<u32> = alone.stages.iter().map(|s| s.seq).collect();
    assert_eq!(nat, [4096, 8192]);
    assert!(alone.stages.iter().all(|s| !s.seq_pinned) && alone.stages.iter().all(|s| s.heal_spec(true).seq.is_none()));
    assert!(alone.warnings.iter().any(|w| w.contains("different sequence lengths") && w.contains("stage 1: 4096") && w.contains("stage 2: 8192")), "the plan says why the answers are written twice: {:?}", alone.warnings);
    let shared = plan(&ins, &AutoSpec { share_teacher_cache: true, ..big(17.4) }, Some(36.0), Some(2000.0));
    assert!(shared.ok, "{:?}", shared.blockers);
    assert_eq!(shared.stages.iter().map(|s| (s.seq, s.seq_pinned)).collect::<Vec<_>>(), [(4096, true), (4096, true)]);
    assert!(shared.stages.iter().all(|s| s.heal_spec(true).seq == Some(4096) && s.heal_spec(false).seq == Some(4096)), "the heal gets the length, so `max_len` matches");
    assert!(shared.warnings.iter().any(|w| w.contains("Stage 2 heals on sequence length 4096 instead of 8192")), "{:?}", shared.warnings);
    assert!(shared.warnings.iter().all(|w| !w.contains("different sequence lengths")));
    // the same on a card where the recipe's sequence is the same anyway: nothing is cut, the length is still sent
    let same = plan(&ins, &AutoSpec { share_teacher_cache: true, ..big(17.4) }, Some(32.0), Some(2000.0));
    assert!(same.stages.iter().all(|s| s.seq_pinned && s.seq == 4096), "{:?}", same.stages.iter().map(|s| s.seq).collect::<Vec<_>>());
    assert!(same.warnings.iter().all(|w| !w.contains("instead of")), "{:?}", same.warnings);
    // the load of the original is the same for both stages
    assert!(shared.stages.iter().all(|s| s.teacher_load == "auto" && s.mode == "cache"));
    // one stage: nothing to share
    assert!(plan(&ins, &AutoSpec { share_teacher_cache: true, stages: 1, ..spec(21.1) }, Some(36.0), Some(2000.0)).stages.iter().all(|s| !s.seq_pinned));
}

#[test]
fn the_search_minutes_and_a_chosen_cut_are_checked_and_a_chosen_cut_is_the_plans_choice() {
    use ndforge_core::reshape::auto_plan::PinnedCut;
    let ins = qwen27();
    assert!(AutoSpec { search_minutes: Some(4), ..spec(21.1) }.validate().is_err());
    assert!(AutoSpec { search_minutes: Some(241), ..spec(21.1) }.validate().is_err());
    assert!(AutoSpec { search_minutes: Some(30), ..spec(21.1) }.validate().is_ok());
    let none = PinnedCut { blocks: vec![], intermediate: None, loss: None, search_id: None, think: None };
    assert!(AutoSpec { cut: Some(none), ..spec(21.1) }.validate().is_err());
    let twice = PinnedCut { blocks: vec![3, 3], intermediate: Some(10752), loss: None, search_id: None, think: None };
    assert!(AutoSpec { cut: Some(twice), ..spec(21.1) }.validate().is_err());
    // a chosen cut that reaches the size is the chosen candidate, with no candidates to try
    let free = plan(&ins, &spec(21.1), Some(32.0), Some(900.0)).chosen.unwrap();
    let blocks: Vec<u32> = (0..free.blocks).map(|i| 3 + 2 * i).collect();
    let pin = PinnedCut { blocks: blocks.clone(), intermediate: free.intermediate, loss: Some(2.0), search_id: Some("s".into()), think: None };
    let p = plan(&ins, &AutoSpec { cut: Some(pin), ..spec(21.1) }, Some(32.0), Some(900.0));
    assert!(p.ok, "{:?}", p.blockers);
    assert!(p.try_list.is_empty());
    assert_eq!((p.chosen.as_ref().unwrap().blocks, p.chosen.as_ref().unwrap().intermediate), (free.blocks, free.intermediate));
    // one that does not: refused with the reason; the first and the last block are never cut
    let wrong = PinnedCut { blocks: vec![2], intermediate: None, loss: None, search_id: None, think: None };
    let bad = plan(&ins, &AutoSpec { cut: Some(wrong), ..spec(21.1) }, Some(32.0), Some(900.0));
    assert!(!bad.ok && bad.blockers.iter().any(|b| b.contains("not 21.1 B")), "{:?}", bad.blockers);
    let first = PinnedCut { blocks: vec![0], intermediate: free.intermediate, loss: None, search_id: None, think: None };
    assert!(!plan(&ins, &AutoSpec { cut: Some(first), ..spec(21.1) }, Some(32.0), Some(900.0)).ok);
    // the time estimate has no try step and no thinking gate for it
    let t = plan(&ins, &AutoSpec { cut: Some(PinnedCut { blocks, intermediate: free.intermediate, loss: None, search_id: None, think: None }), ..spec(21.1) }, Some(32.0), Some(900.0)).time.unwrap();
    assert!(!t.steps.iter().any(|s| s.id == "try" || s.id == "think_gate"));
    // the search is one step with its own label and a time that follows the minutes
    let s30 = plan(&ins, &AutoSpec { search_minutes: Some(30), ..spec(21.1) }, Some(32.0), Some(900.0)).time.unwrap();
    let s10 = plan(&ins, &AutoSpec { search_minutes: Some(10), ..spec(21.1) }, Some(32.0), Some(900.0)).time.unwrap();
    let step = |t: &ndforge_core::reshape::auto_plan::TimeEstimate| t.steps.iter().find(|s| s.id == "try").unwrap().clone();
    assert_eq!(step(&s30).label, "Find the best cut");
    assert!(step(&s30).minutes.1 > step(&s10).minutes.1 && step(&s30).minutes.1 <= 31.0);
}
