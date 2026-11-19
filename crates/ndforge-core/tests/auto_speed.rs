//! The speed settings of Auto-cut (`reshape::auto_speed`): the time of every preset and of every knob's choices, and the shared answers of the original.
use ndforge_core::reshape::auto_plan::*;
use ndforge_core::reshape::plan::*;

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

use ndforge_core::reshape::auto_speed::*;
use ndforge_core::timing::History;

/// 27 B -> 17.4 B needs more than four blocks: the owner confirms that.
fn owner() -> AutoSpec {
    AutoSpec { allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 17.4) }
}

fn run(spec: &AutoSpec, vram: f64) -> (AutoPlan, TimeEstimate) {
    let ins = qwen27();
    let p = plan(&ins, spec, Some(vram), Some(2000.0));
    assert!(p.ok, "{:?}", p.blockers);
    let t = time_estimate(&ins, spec, &p, &History::default());
    (p, t)
}

fn part(t: &TimeEstimate, step: &str, part: &str) -> (f64, f64) {
    t.steps.iter().find(|s| s.id == step).unwrap_or_else(|| panic!("no step {step}")).parts.iter().filter(|x| x.id == part).fold((0.0, 0.0), |a, x| (a.0 + x.minutes.0, a.1 + x.minutes.1))
}

fn info(spec: &AutoSpec, vram: f64) -> SpeedInfo {
    speed_info(&qwen27(), spec, Some(vram), &History::default())
}

fn hours(t: (f64, f64)) -> String {
    format!("{:.1}-{:.1} h", t.0, t.1)
}

#[test]
fn the_presets_get_faster_and_the_table_is_printed() {
    let si = info(&owner(), 32.0);
    println!("{:<30} {:>16} {:>16}", "preset", "total", "retry (extra)");
    for p in &si.presets {
        println!("{:<30} {:>16} {:>16}", p.label, p.total_h.map(hours).unwrap_or_else(|| "-".into()), p.retry_h.map(hours).unwrap_or_else(|| "-".into()));
    }
    for k in &si.knobs {
        for o in &k.options {
            println!("{:<42} {:<62} {:>16} {:>22}", k.knob, o.label, o.total_h.map(hours).unwrap_or_default(), o.delta_min.map(|d| format!("{:+.1}..{:+.1} h", d.0 / 60.0, d.1 / 60.0)).unwrap_or_default());
        }
    }
    let t = |id: &str| si.presets.iter().find(|p| p.id == id).unwrap().total_h.unwrap();
    let (th, ba, fa) = (t("thorough"), t("balanced"), t("fast"));
    assert!(fa.1 < ba.1 && ba.1 < th.1 && fa.0 < ba.0 && ba.0 < th.0, "{fa:?} < {ba:?} < {th:?}");
    // Custom is "what you have": the current total
    let c = si.presets.iter().find(|p| p.id == "custom").unwrap().total_h.unwrap();
    // the preset also looks for the best cut for 30 minutes (the classic Try first takes a few minutes): the totals are close, not the same
    assert!((c.0 - th.0).abs() < 0.5 && (c.1 - th.1).abs() < 0.5, "the legacy spec is the recipe with the classic Try first: {c:?} {th:?}");
    assert_eq!(c, si.total_h);
    assert_eq!(si.current_preset.as_deref(), Some("thorough"));
    assert_eq!(si.default_preset, "balanced");
    // the thorough total is the plain estimate of the legacy spec
    assert_eq!(c, run(&owner(), 32.0).1.total_h);
    // a retry is not in any total
    assert!(si.presets.iter().filter(|p| p.id != "fast" && p.id != "custom").all(|p| p.retry_h.is_some()));
    assert!(si.presets.iter().find(|p| p.id == "fast").unwrap().retry_h.is_none(), "Fast has no retry");
}

#[test]
fn the_heal_tokens_are_linear_in_the_time_and_each_choice_shows_its_delta() {
    let si = info(&owner(), 32.0);
    let k = si.knobs.iter().find(|k| k.knob == "last_tokens_m").unwrap();
    assert_eq!(k.options.len(), 4);
    let totals: Vec<f64> = k.options.iter().map(|o| o.total_h.unwrap().1).collect();
    assert!(totals.windows(2).all(|w| w[0] < w[1]), "10 < 20 < 40 < 80 M: {totals:?}");
    // the recipe's own choice is the current one and has no delta; the recipe for this cut is 80 M (more than 30 % is cut)
    let cur: Vec<&SpeedOption> = k.options.iter().filter(|o| o.current).collect();
    assert_eq!(cur.len(), 1);
    assert_eq!(cur[0].label, "80 M (recipe)");
    assert_eq!(cur[0].delta_min, Some((0.0, 0.0)));
    assert!(cur[0].set["last_tokens_m"].is_null(), "the recipe is stored as the recipe, not as a number: the spec stays a preset");
    for o in k.options.iter().filter(|o| !o.current) {
        let d = o.delta_min.unwrap();
        assert!(d.0 < 0.0 && d.1 < 0.0, "{o:?}");
    }
    // the step 80 -> 40 M of the last stage takes half of that stage's cache and training time off (the merge and the loads stay)
    let base = run(&owner(), 32.0).1;
    let half = run(&AutoSpec { last_tokens_m: Some(40), ..owner() }, 32.0).1;
    let (bc, hc) = (part(&base, "s2.heal", "train"), part(&half, "s2.heal", "train"));
    assert!((hc.1 / bc.1 - 0.5).abs() < 0.01, "{bc:?} {hc:?}");
    // the first stage has its own knob
    assert!(si.knobs.iter().any(|k| k.knob == "stage_a_tokens_m"));
    let one = info(&AutoSpec { stages: 1, ..AutoSpec::new("qwen3.8-27b", 21.1) }, 32.0);
    assert!(one.knobs.iter().all(|k| k.knob != "stage_a_tokens_m" && k.knob != "share_teacher_cache"), "one stage has neither");
}

#[test]
fn the_other_knobs_have_the_right_sign() {
    let si = info(&owner(), 32.0);
    let delta = |knob: &str, label_has: &str| si.knobs.iter().find(|k| k.knob == knob).unwrap().options.iter().find(|o| o.label.contains(label_has)).unwrap().delta_min.unwrap();
    for (knob, label) in [("think_quick", "Quick"), ("quick", "Quick"), ("check_cut", "Off"), ("imatrix", "Off"), ("stages", "One stage"), ("think_gate_n", "Off"), ("think_gate_n", "best cut only")] {
        let d = delta(knob, label);
        assert!(d.0 < 0.0 && d.1 < 0.0, "{knob} {label}: {d:?}");
    }
    // the retry and the back-off save nothing in a run that passes; the option says what they save when a check fails
    for (knob, label) in [("retry_heal", "Off"), ("back_off", "Off")] {
        let o = si.knobs.iter().find(|k| k.knob == knob).unwrap().options.iter().find(|o| o.label == label).unwrap();
        assert_eq!(o.delta_min, Some((0.0, 0.0)), "{knob}");
        assert!(o.note.as_deref().is_some_and(|n| n.contains("only if") || n.contains("unless")), "{o:?}");
    }
    assert!(si.knobs.iter().find(|k| k.knob == "retry_heal").unwrap().options[1].note.as_deref().unwrap().contains(" h"), "the retry's saving is named");
    // each choice sets only what it names
    for k in &si.knobs {
        for o in &k.options {
            assert!(o.set.keys().all(|key| KNOBS.contains(&key.as_str())), "{o:?}");
        }
    }
}

#[test]
fn the_second_stage_writes_the_originals_answers_again_only_when_the_sequence_or_the_load_differ() {
    let two_forty = AutoSpec { stage_a_tokens_m: Some(40), last_tokens_m: Some(40), ..owner() };
    // a 32 GB card: both stages heal on 4096 by themselves: the saved answers of stage 1 are the whole of stage 2's
    let (p, t) = run(&two_forty, 32.0);
    assert_eq!(p.stages.iter().map(|s| s.seq).collect::<Vec<_>>(), [4096, 4096]);
    let (c1, c2) = (part(&t, "s1.heal", "cache"), part(&t, "s2.heal", "cache"));
    assert!(c1.1 > 300.0, "stage 1 writes 40 M tokens: {c1:?}");
    assert_eq!(c2, (0.0, 0.0), "stage 2 finds all of them");
    assert!(t.steps.iter().find(|s| s.id == "s2.heal").unwrap().note.as_deref().unwrap_or("").contains("not written again"));
    // a 36 GB card: 4096 for the larger student, 8192 for the smaller one: nothing to reuse (this is the owner's table: the original writes AGAIN)
    let (p, t) = run(&two_forty, 36.0);
    assert_eq!(p.stages.iter().map(|s| s.seq).collect::<Vec<_>>(), [4096, 8192]);
    let (a, b) = (part(&t, "s1.heal", "cache"), part(&t, "s2.heal", "cache"));
    assert!(b.0 > 0.0 && (b.1 / a.1 - 1.0).abs() < 0.15, "{a:?} {b:?}: the same tokens, the original answers again (a little more for the smaller..larger sizes)");
    assert!(t.steps.iter().find(|s| s.id == "s2.heal").unwrap().note.as_deref().unwrap_or("").contains("them all again"));
    // with the knob on, the length is pinned and the reuse is real
    let (p, ts) = run(&AutoSpec { share_teacher_cache: true, ..two_forty.clone() }, 36.0);
    assert_eq!(p.stages.iter().map(|s| s.seq).collect::<Vec<_>>(), [4096, 4096]);
    assert_eq!(part(&ts, "s2.heal", "cache"), (0.0, 0.0));
    assert!(ts.total_h.1 < t.total_h.1 - 5.0, "{:?} against {:?}", ts.total_h, t.total_h);
    // the second stage with more tokens writes only the rest
    let (_, t80) = run(&AutoSpec { share_teacher_cache: true, stage_a_tokens_m: Some(40), last_tokens_m: Some(80), ..owner() }, 36.0);
    let (x, y) = (part(&t80, "s1.heal", "cache"), part(&t80, "s2.heal", "cache"));
    assert!((y.1 / x.1 - 1.0).abs() < 0.2, "40 M more tokens, about the same as the first 40 M: {x:?} {y:?}");
    // another way of loading the original: the saved answers do not fit
    let (p, _) = run(&two_forty, 32.0);
    let mut odd = p.clone();
    odd.stages[1].teacher_load = "4bit".into();
    let t = time_estimate(&qwen27(), &two_forty, &odd, &History::default());
    assert!(part(&t, "s2.heal", "cache").1 > 300.0, "a different load is a different store");
    // the online way saves no answers: nothing is reused and no cache part exists
    let (_, on) = run(&two_forty, 80.0);
    assert!(on.steps.iter().find(|s| s.id == "s2.heal").unwrap().parts.iter().all(|x| x.id != "cache"));
}

#[test]
fn a_retry_is_80_m_tokens_for_the_student_and_only_the_missing_answers_for_the_original() {
    let (_, t) = run(&AutoSpec { stage_a_tokens_m: Some(40), last_tokens_m: Some(40), ..owner() }, 36.0);
    let e = t.extra.iter().find(|e| e.id == "retry_heal").unwrap();
    assert!(e.label.contains("learns on 80 M tokens, 40 M more of the original's answers"), "{}", e.label);
    // the healed model's own training takes twice the last heal's (80 M against 40 M)
    let (_, t20) = run(&AutoSpec { last_tokens_m: Some(20), ..owner() }, 36.0);
    assert!(t20.extra[0].label.contains("learns on 80 M tokens, 60 M more"), "{}", t20.extra[0].label);
    // without the retry there is no extra
    assert!(run(&AutoSpec { retry_heal: false, ..owner() }, 36.0).1.extra.is_empty());
    assert!(run(&AutoSpec { keep_thinking: false, think_gate: false, ..owner() }, 36.0).1.extra.is_empty());
}

#[test]
fn no_check_of_the_cut_moves_the_original_to_the_first_healed_model() {
    let off = AutoSpec { check_cut: false, think_gate: false, ..owner() };
    let (_, t) = run(&off, 32.0);
    assert!(t.steps.iter().all(|s| !s.id.ends_with(".check_cut")));
    let ch = t.steps.iter().find(|s| s.id == "s1.check_heal").unwrap();
    assert!(ch.parts.iter().any(|x| x.label.contains("original")), "{ch:?}");
    assert!(t.steps.iter().find(|s| s.id == "s2.check_heal").unwrap().parts.iter().all(|x| !x.label.contains("original")), "the yardstick is measured once");
    // with the check of the cut it stays where it was
    let (_, on) = run(&AutoSpec { think_gate: false, ..owner() }, 32.0);
    assert!(on.steps.iter().find(|s| s.id == "s1.check_cut").unwrap().parts.iter().any(|x| x.label.contains("original")));
    assert!(on.steps.iter().find(|s| s.id == "s1.check_heal").unwrap().parts.iter().all(|x| !x.label.contains("original")));
}

#[test]
fn the_importance_matrix_is_a_part_of_the_export_only_when_it_is_on() {
    let on = run(&owner(), 32.0).1;
    let off = run(&AutoSpec { imatrix: false, ..owner() }, 32.0).1;
    let (a, b) = (part(&on, "export", "measure"), part(&off, "export", "measure"));
    assert!(a.0 > 5.0 && a.1 > a.0 && b == (0.0, 0.0), "{a:?} {b:?}");
}

#[test]
fn the_plan_carries_the_speed_settings_with_the_history_it_was_given() {
    let ins = qwen27();
    let p0 = plan(&ins, &owner(), Some(32.0), Some(2000.0));
    assert!(p0.speed.is_none(), "the plain plan stays cheap: the settings come with `with_speed`");
    let p = with_speed(p0, &ins, &owner(), Some(32.0), &History::default());
    let s = p.speed.as_ref().unwrap();
    assert_eq!(s.total_h, p.time.as_ref().unwrap().total_h);
    assert!(s.presets.len() == 4 && s.knobs.len() >= 8, "{} {}", s.presets.len(), s.knobs.len());
    // a plan that cannot be made has none
    let bad = with_speed(plan(&ins, &AutoSpec::new("qwen3.8-27b", 27.5), Some(32.0), None), &ins, &AutoSpec::new("qwen3.8-27b", 27.5), Some(32.0), &History::default());
    assert!(!bad.ok && bad.speed.is_none());
}

#[test]
fn the_owners_two_stage_case_with_the_recipe_at_40_m_in_both_stages() {
    // 27.8 B to 19.5 B cuts 29.8 %: the recipe is 40 M tokens in both stages, as in the owner's table
    let spec = AutoSpec { allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 19.5) };
    for vram in [32.0, 36.0] {
        let si = info(&spec, vram);
        let p = plan(&qwen27(), &spec, Some(vram), Some(2000.0));
        println!("--- {vram} GB card, stages {:?}", p.stages.iter().map(|s| (s.params_b, s.tokens_m, s.seq)).collect::<Vec<_>>());
        for pr in &si.presets {
            println!("{:<30} {:>16} retry {:>16}", pr.label, pr.total_h.map(hours).unwrap_or_default(), pr.retry_h.map(hours).unwrap_or_default());
        }
        let t = |id: &str| si.presets.iter().find(|p| p.id == id).unwrap().total_h.unwrap();
        assert!(t("fast").1 < t("balanced").1 && t("balanced").1 < t("thorough").1, "{vram}");
        assert_eq!(p.stages.iter().map(|s| s.tokens_m).collect::<Vec<_>>(), [40, 40]);
    }
}

fn fits_for(spec: &AutoSpec, vram: f64, h: f64) -> BudgetFit {
    fit_budget(&qwen27(), spec, Some(vram), &History::default(), h)
}

#[test]
fn the_ladder_gives_up_the_least_harmful_setting_first_and_stops_at_the_first_fit() {
    let spec = AutoSpec { allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 19.5) };
    let asked = run(&spec, 36.0).1.total_h;
    let order = ["think_quick", "quick", "check_cut", "share_teacher_cache", "think_gate_n", "retry_heal", "stage_a_tokens_m", "stages", "last_tokens_m"];
    // every rung that applies, in order, one more per tighter limit
    let mut last_len = 0;
    let mut knobs_seen: Vec<String> = Vec::new();
    for limit in [asked.1 - 0.2, asked.1 - 2.0, asked.1 - 6.0, asked.1 - 20.0, asked.1 - 30.0, 25.0, 12.5] {
        let f = fits_for(&spec, 36.0, limit);
        let knobs: Vec<&str> = f.dropped.iter().map(|d| d.knob.as_str()).collect();
        println!("limit {limit:.1} h: fits={} total {:?} dropped {knobs:?}", f.fits, f.total_h);
        let pos: Vec<usize> = knobs.iter().map(|k| order.iter().position(|o| o == k).unwrap()).collect();
        assert!(pos.windows(2).all(|w| w[0] < w[1]), "the ladder order: {knobs:?}");
        assert!(f.dropped.len() >= last_len, "a tighter limit never gives up fewer settings");
        last_len = f.dropped.len();
        if f.fits {
            assert!(f.total_h.1 <= limit, "{:?} {limit}", f.total_h);
            // the first fit: without the last setting it would not have fitted
            if let Some(last) = f.dropped.last() {
                assert!(f.total_h.1 + last.saves_h.1 > limit - 0.1, "without the last setting given up it did not fit: {:?} {last:?} {limit}", f.total_h);
            }
            assert!(f.dropped.iter().all(|d| d.saves_h.1 >= 0.0 && !d.risk.is_empty() && !d.from.is_empty() && !d.to.is_empty()), "{:?}", f.dropped);
        }
        knobs_seen.extend(knobs.iter().map(|k| k.to_string()));
    }
    assert!(knobs_seen.contains(&"think_quick".to_string()));
    // a limit that is kept already changes nothing
    let ok = fits_for(&spec, 36.0, asked.1 + 1.0);
    assert!(ok.fits && ok.dropped.is_empty());
    assert_eq!(ok.spec, AutoSpec { max_hours: Some(asked.1 + 1.0), ..spec.clone() });
}

#[test]
fn when_even_the_fastest_settings_do_not_fit_it_refuses_in_plain_words() {
    let spec = AutoSpec { allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 19.5) };
    let f = fits_for(&spec, 36.0, 3.0);
    assert!(!f.fits);
    let m = f.message.as_deref().unwrap();
    assert!(m.starts_with("Even the fastest settings take about ") && m.contains(" h") && m.ends_with("Choose more time or a larger size."), "{m}");
    // the whole ladder was used, the last rung is the thinking warning one
    assert_eq!(f.dropped.last().map(|d| d.knob.as_str()), Some("last_tokens_m"));
    assert!(f.dropped.last().unwrap().risk.contains("very likely fail"));
    assert!(f.spec.last_tokens_m == Some(20) && f.spec.stages == 1);
}

#[test]
fn a_time_limit_never_drops_a_requested_output_or_the_thinking_itself() {
    let spec = AutoSpec { allow_many_blocks: true, target_gb: Some(10.6), studio_publish: true, quants: vec!["q4_k_m".into(), "q8_0".into()], keep_intermediate: true, ..AutoSpec::new("qwen3.8-27b", 19.5) };
    for h in [20.0, 3.0] {
        let f = fits_for(&spec, 36.0, h);
        assert_eq!((f.spec.target_gb, f.spec.studio_publish, f.spec.quants.clone(), f.spec.keep_intermediate, f.spec.keep_thinking, f.spec.target_b), (Some(10.6), true, vec!["q4_k_m".to_string(), "q8_0".to_string()], true, true, 19.5));
        assert!(f.dropped.iter().all(|d| ["think_quick", "quick", "check_cut", "share_teacher_cache", "think_gate_n", "retry_heal", "stage_a_tokens_m", "stages", "last_tokens_m"].contains(&d.knob.as_str())));
    }
    // a model that does not think has no thinking checks to shorten
    let plain = AutoSpec { keep_thinking: false, think_gate: false, allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 19.5) };
    let f = fits_for(&plain, 36.0, 12.0);
    assert!(f.dropped.iter().all(|d| !["think_quick", "check_cut", "think_gate_n", "retry_heal"].contains(&d.knob.as_str())), "{:?}", f.dropped.iter().map(|d| d.knob.clone()).collect::<Vec<_>>());
}

#[test]
fn the_window_gets_the_list_of_what_was_given_up_or_the_refusal() {
    let ins = qwen27();
    let spec = AutoSpec { allow_many_blocks: true, max_hours: Some(30.0), ..AutoSpec::new("qwen3.8-27b", 19.5) };
    let v = ndforge_core::reshape::auto::preview(&ins, &spec, Some(36.0), Some(2000.0), true);
    let p: AutoPlan = serde_json::from_value(v["plan"].clone()).unwrap();
    let b = p.budget.as_ref().unwrap();
    assert!(!b.dropped.is_empty());
    assert_eq!(p.ok, b.fits);
    assert!(p.warnings.iter().any(|w| w.starts_with("To fit your 30 hours")), "{:?}", p.warnings);
    let hard = ndforge_core::reshape::auto::preview(&ins, &AutoSpec { max_hours: Some(3.0), ..spec }, Some(36.0), Some(2000.0), true);
    let p: AutoPlan = serde_json::from_value(hard["plan"].clone()).unwrap();
    assert!(!p.ok && p.blockers.iter().any(|b| b.contains("Even the fastest settings take about")), "{:?}", p.blockers);
    // without a limit there is no budget
    let none: AutoPlan = serde_json::from_value(ndforge_core::reshape::auto::preview(&ins, &AutoSpec { allow_many_blocks: true, ..AutoSpec::new("qwen3.8-27b", 19.5) }, Some(36.0), Some(2000.0), true)["plan"].clone()).unwrap();
    assert!(none.budget.is_none());
}
