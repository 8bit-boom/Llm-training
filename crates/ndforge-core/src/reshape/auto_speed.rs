//! Auto-cut's speed settings (docs/AUTO_CUT.md, "Speed settings"): the presets (Thorough, Balanced, Fast, Custom) as data in `auto_speed.json`, the knobs they set, and what
//! every knob costs in quality. Pure functions: nothing is read, nothing runs. The planner (`auto_plan`) applies the knobs, `auto_time` counts them, the orchestrator runs them.
//!
//! Rule: LEGACY = THOROUGH. A spec whose speed fields are all at their defaults is the recommended recipe and plans and runs exactly as before the settings existed.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::auto_plan::{self, time_estimate, AutoPlan, AutoSpec, StagePlan, TimeEstimate};
use super::plan::{self, Inspect};
use crate::timing::History;
use crate::error::{Error, Result};

const PRESETS_JSON: &str = include_str!("auto_speed.json");

/// One preset: the values of the speed knobs it sets (`set`: AutoSpec field -> value), what it is and the plain sentence of its quality risk. The window holds the same table
/// (`AC_SPEED_PRESETS` in `ui/app.js`; a smoke test compares them).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedPreset {
    pub id: String,
    pub label: String,
    pub blurb: String,
    pub risk: String,
    pub set: Map<String, Value>,
}

#[derive(Deserialize)]
struct PresetFile {
    presets: Vec<SpeedPreset>,
}

/// The presets of `auto_speed.json`, in the order the window shows them.
pub fn presets() -> Vec<SpeedPreset> {
    serde_json::from_str::<PresetFile>(PRESETS_JSON).map(|f| f.presets).unwrap_or_default()
}

pub fn preset(id: &str) -> Option<SpeedPreset> {
    presets().into_iter().find(|p| p.id == id)
}

/// The id of the preset the window selects for a model of `total_b` billion parameters when the person has not chosen: Balanced from 15 B (a heal of a big model takes days),
/// else Thorough.
pub fn default_preset(total_b: f64) -> &'static str {
    if total_b >= 15.0 { "balanced" } else { "thorough" }
}

/// The AutoSpec fields the presets set (the speed knobs).
pub const KNOBS: [&str; 14] = ["search_minutes", "stages", "stage_a_tokens_m", "last_tokens_m", "share_teacher_cache", "quick", "think_quick", "think_gate", "think_gate_n", "check_cut", "retry_heal", "back_off", "imatrix", "heal_rounds"];

impl AutoSpec {
    /// The spec with the values of preset `id` on top of it (everything that is not a speed knob stays). "custom" only names the choice.
    pub fn with_preset(&self, id: &str) -> Result<AutoSpec> {
        let p = preset(id).ok_or_else(|| Error::new(format!("Unknown speed setting: {id}.")))?;
        let mut v = serde_json::to_value(self).map_err(|e| Error::new(e.to_string()))?;
        if let Some(o) = v.as_object_mut() {
            for (k, x) in &p.set {
                o.insert(k.clone(), x.clone());
            }
            o.insert("preset".into(), Value::String(p.id.clone()));
        }
        serde_json::from_value(v).map_err(|e| Error::new(format!("The preset {id} does not fit the spec: {e}")))
    }

    /// The speed knobs as the window shows them (a think_quick that follows `quick` is shown as its value).
    fn knob_values(&self) -> Map<String, Value> {
        let mut v = match serde_json::to_value(self) {
            Ok(Value::Object(o)) => o,
            _ => Map::new(),
        };
        v.insert("think_quick".into(), Value::Bool(self.think_quick()));
        if !self.keep_thinking {
            // a model that does not think has no thinking checks: those knobs do not tell the presets apart
            for k in ["think_quick", "think_gate", "think_gate_n", "check_cut", "retry_heal"] {
                v.insert(k.into(), Value::Null);
            }
        }
        // how long the search for the best cut looks is not part of what makes a recipe: a legacy spec (no search) is still the recommended recipe
        KNOBS.iter().filter(|k| **k != "search_minutes").filter_map(|k| v.get(*k).map(|x| (k.to_string(), x.clone()))).collect()
    }

    /// The preset whose values the speed knobs have (the recommended recipe for a legacy spec), None when they are a mix: "Custom".
    pub fn matching_preset(&self) -> Option<String> {
        let mine = self.knob_values();
        presets().into_iter().filter(|p| p.id != "custom").find(|p| {
            // the preset's own values as a spec would have them
            self.with_preset(&p.id).map(|s| s.knob_values() == mine).unwrap_or(false)
        }).map(|p| p.id)
    }
}

/// What a person who chose a faster setting gives up, one plain sentence per value that is not the recommended recipe's (the plan shows them as warnings and the report lists
/// them again). `stages` are the planned stages (with the shared sequence length when `share_teacher_cache` is on), `natural_seq` the sequence length each stage would heal on
/// by itself. Empty for the recipe.
pub fn speed_warnings(spec: &AutoSpec, stages: &[StagePlan], natural_seq: &[u32], total_b: f64) -> Vec<String> {
    let mut w = Vec::new();
    let thinking = spec.keep_thinking;
    let n = stages.len();
    let evidence = "In an earlier real run, healing below the recipe failed the thinking checks; the checks after each stage will tell, and a failed check costs a retry.";
    if let Some(last) = stages.last() {
        let recipe = if thinking { plan::thinking_tokens_m(last.params_b, total_b) } else { plan::default_heal_tokens_m(last.params_b) };
        if last.tokens_m < recipe {
            w.push(if thinking {
                format!("The last stage heals on {} M tokens; the recipe for a cut of this size is {recipe} M. {evidence}", last.tokens_m)
            } else {
                format!("The last stage heals on {} M tokens; the usual number for a model of this size is {recipe} M. The healed model may be clearly weaker.", last.tokens_m)
            });
        }
    }
    if n > 1 && thinking && stages[0].tokens_m < super::auto_plan::STAGE_A_TOKENS_M {
        w.push(format!("The first stage heals on {} M tokens; the recipe is {} M. The second stage starts from a model that learned less. {evidence}", stages[0].tokens_m, super::auto_plan::STAGE_A_TOKENS_M));
    }
    if thinking && spec.stages == 1 && n == 1 && stages[0].cut_pct > 15.0 {
        w.push(format!("One stage cuts {:.0} % of the model in a single step; the recipe heals half of it first. For a model that thinks this is the riskier way.", stages[0].cut_pct));
    }
    if !spec.check_cut && thinking {
        w.push("The unhealed cut is not checked for thinking (its check only informs; nothing is decided by it). The original is checked after the first heal instead, so the gates still have their yardstick.".into());
    }
    if spec.think_quick() && thinking {
        w.push("The thinking checks are the quick ones (16 questions): they take about a third less time and are noisier, so a close call can go either way.".into());
    }
    if spec.quick {
        w.push(format!("Try first measures each cut on {} texts instead of {}: it is quicker and a close race between two cuts is less certain.", plan::TRY_QUICK_SAMPLES, plan::TRY_FULL_SAMPLES));
    }
    if thinking && !spec.retry_heal {
        w.push("No retry: when a thinking check of a healed stage fails, Auto-cut does not heal it again but goes straight to cutting one step less.".into());
    }
    if !spec.back_off {
        w.push("No back-off: when a stage still fails its thinking check (after the retry, if there is one), the run ends with the report instead of cutting one step less.".into());
    }
    if !spec.imatrix {
        w.push("No importance matrix is measured for the export: the small sizes (Q4 and below) are made without it and are a little less exact.".into());
    }
    if let Some(r) = &spec.heal_rounds {
        w.push(format!("Experimental: every stage heals in rounds of {} M tokens and stops at the first round whose thinking check passes. Whether healing in rounds ends as good as one long heal is not known yet. {}", r.round_m, knob_risk("heal_rounds")));
    }
    if n > 1 && spec.share_teacher_cache {
        let pin = stages.iter().map(|s| s.seq).min().unwrap_or(0);
        if stages.iter().all(|s| s.mode != "cache") {
            w.push("Same sequence length in both stages is on, but this card heals with the original next to the student (no saved answers): it changes nothing here.".into());
        }
        for (g, nat) in stages.iter().zip(natural_seq) {
            if *nat > pin {
                w.push(format!("Stage {} heals on sequence length {pin} instead of {nat} so that the original writes its answers once and the next stage reuses them: reasoning traces longer than {pin} tokens are cut short in that stage.", g.index));
            }
        }
    } else if n > 1 {
        let c: Vec<&StagePlan> = stages.iter().filter(|s| s.mode == "cache").collect();
        if c.len() > 1 && c.iter().any(|s| s.seq != c[0].seq || s.teacher_load != c[0].teacher_load) {
            w.push(format!("The stages heal on different sequence lengths ({}): the original's saved answers cannot be reused by the next stage, so it writes them again (many hours). \"Same sequence length in both stages\" shortens that, at the price of a shorter sequence in the larger stage.",
                           c.iter().map(|s| format!("stage {}: {}", s.index, s.seq)).collect::<Vec<_>>().join(", ")));
        }
    }
    w
}


/// What giving a knob up costs in quality, in plain words (the plan's knob table, the time limit's list of what was given up and the report all say this).
pub fn knob_risk(knob: &str) -> &'static str {
    match knob {
        "think_quick" => "The quick checks ask 16 questions instead of 32: about a third faster, and a close call can go either way.",
        "quick" => "Try first measures on 64 texts instead of 128: a close race between two cuts is less certain.",
        "check_cut" => "The unhealed cut is not checked for thinking (that check only informs); the original is checked after the first heal, so the gates keep their yardstick.",
        "share_teacher_cache" => "Both stages heal on the shorter sequence length: the larger student's longest reasoning traces are cut short. The original writes its answers once.",
        "think_gate_n" => "Only the cut with the lowest error is checked for thinking before healing: a cut with a low error and little thinking may be chosen.",
        "retry_heal" => "When a thinking check of a healed stage fails, the stage is not healed again: Auto-cut goes straight to cutting one step less (a larger model).",
        "stage_a_tokens_m" => "The first stage learns less, so the second stage starts from a weaker model. Below the recipe a model that thinks has failed the thinking checks before.",
        "stages" => "One stage cuts everything at once and heals once: it skips a cut, a check, a merge and a whole heal, and is the riskier way for a model that thinks.",
        "last_tokens_m" => "Fewer tokens is linearly faster and learns less. Below the recipe a model that thinks has failed the thinking checks before; on 20 M tokens it will very likely fail them and the run may end without a model to keep.",
        "back_off" => "Without it the run ends with its report when a stage still fails; with it a stage is made again, larger.",
        "imatrix" => "The small sizes (Q4 and below) are made without it and are a little less exact.",
        "heal_rounds" => "Experimental. A quick check asks 16 questions, so one or two answers decide pass or fail; that is why the merged model is checked again at the end, and if it fails there healing goes on and then needs two passed checks in a row. The checks during healing look at the model as it is trained (4-bit with the learning on top); the merged model can score a little differently, and the merged check decides. Every round adds about 25 minutes of loading and checking. Whether healing in rounds ends as good as one long heal is not known yet.",
        _ => "",
    }
}

// ------------------------------------------------------------------------------------------------------------------ what every choice costs in time

/// One choice of a knob with the time of the whole run when it is chosen (everything else as it is now).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedOption {
    pub label: String,
    /// the AutoSpec fields this choice sets (a `null` is "the recipe's value")
    pub set: Map<String, Value>,
    /// this is what the spec has now
    pub current: bool,
    /// the total of the run with this choice, hours (low, high); None when the cut cannot be planned with it
    pub total_h: Option<(f64, f64)>,
    /// this choice minus the current one, minutes (low, high): negative = shorter
    pub delta_min: Option<(f64, f64)>,
    pub note: Option<String>,
}

/// A knob of the speed settings with its choices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedKnob {
    pub knob: String,
    pub label: String,
    /// what a faster choice gives up, in plain words
    pub risk: String,
    pub options: Vec<SpeedOption>,
}

/// A preset with the time of the whole run for this model and card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetTime {
    pub id: String,
    pub label: String,
    pub blurb: String,
    pub risk: String,
    /// the run's total, hours (low, high); None when the cut cannot be planned with this preset
    pub total_h: Option<(f64, f64)>,
    /// the retry after a failed thinking check, hours (not in the total)
    pub retry_h: Option<(f64, f64)>,
    pub why: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeedInfo {
    pub presets: Vec<PresetTime>,
    pub knobs: Vec<SpeedKnob>,
    /// the preset the knobs of the spec are exactly (None: Custom)
    pub current_preset: Option<String>,
    /// the preset the window selects when the person has not chosen (Balanced for a big model)
    pub default_preset: String,
    /// the current total (hours), the same as `plan.time.total_h`
    pub total_h: (f64, f64),
}

/// The spec with `set` on top (a field of the spec per key); Custom is the name of what it becomes.
pub fn patched(spec: &AutoSpec, set: &Map<String, Value>) -> Option<AutoSpec> {
    let mut v = serde_json::to_value(spec).ok()?;
    let o = v.as_object_mut()?;
    for (k, x) in set {
        o.insert(k.clone(), x.clone());
    }
    serde_json::from_value(v).ok()
}

fn set_of(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// The plan of `spec` (no disk question) and its time estimate; None when it has blockers.
fn timed(ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, history: &History) -> Option<(AutoPlan, TimeEstimate)> {
    let p = auto_plan::plan(ins, spec, vram_gb, None);
    if !p.ok {
        return None;
    }
    let t = time_estimate(ins, spec, &p, history);
    Some((p, t))
}

/// Everything the window needs to choose speed settings: the total of every preset and of every choice of every knob, each with everything else as it is now.
pub fn speed_info(ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, history: &History) -> SpeedInfo {
    let total_b = ins.total_params as f64 / 1e9;
    // every choice of every knob is a plan and a time estimate of the spec with that one change; choices that give the same spec (the one that is on, a value several knobs share)
    // are made once. The answer is the same as without it: the estimate depends on nothing but the spec, the model, the card and the history.
    let memo: std::cell::RefCell<std::collections::HashMap<String, Option<(AutoPlan, TimeEstimate)>>> = Default::default();
    let timed_m = |s: &AutoSpec| -> Option<(AutoPlan, TimeEstimate)> {
        let key = serde_json::to_string(s).unwrap_or_default();
        if !key.is_empty() {
            if let Some(hit) = memo.borrow().get(&key) {
                return hit.clone();
            }
        }
        let made = timed(ins, s, vram_gb, history);
        if !key.is_empty() {
            memo.borrow_mut().insert(key, made.clone());
        }
        made
    };
    let Some((base_plan, base)) = timed_m(spec) else {
        return SpeedInfo { presets: Vec::new(), knobs: Vec::new(), current_preset: spec.matching_preset(), default_preset: default_preset(total_b).into(), total_h: (0.0, 0.0) };
    };
    let thinking = spec.keep_thinking;
    let n = base_plan.stages.len();
    let retry_h = base.extra.iter().find(|e| e.id == "retry_heal").map(|e| e.minutes);
    let delta = |t: (f64, f64)| (r1h((t.0 - base.total_h.0) * 60.0) + 0.0, r1h((t.1 - base.total_h.1) * 60.0) + 0.0);
    let option = |label: String, set: Map<String, Value>, current: &dyn Fn(&AutoPlan, &AutoSpec) -> bool, note: Option<String>| -> SpeedOption {
        let alt = patched(spec, &set);
        match alt.as_ref().and_then(|a| timed_m(a).map(|(p, t)| (a, p, t))) {
            Some((a, p, t)) => SpeedOption { label, set, current: current(&p, a), total_h: Some(t.total_h), delta_min: Some(delta(t.total_h)), note },
            None => SpeedOption { label, set, current: false, total_h: None, delta_min: None, note: Some("This cannot be planned with this choice.".into()) },
        }
    };
    let mut knobs: Vec<SpeedKnob> = Vec::new();
    let student_b = |last: bool| base_plan.stages.get(if last { n.saturating_sub(1) } else { 0 }).map(|s| s.params_b).unwrap_or(total_b);

    // the heal tokens: the biggest lever, linear in the time
    let tokens = |knob: &str, label: &str, last: bool| -> SpeedKnob {
        let recipe = auto_plan::recipe_tokens_m(thinking, student_b(last), total_b, last);
        let idx = if last { n.saturating_sub(1) } else { 0 };
        let now = base_plan.stages.get(idx).map(|s| s.tokens_m);
        let options = plan::HEAL_TOKENS_M.iter().map(|t| {
            let set = set_of(&[(knob, if *t == recipe { Value::Null } else { Value::from(*t) })]);
            let label = format!("{t} M{}", if *t == recipe { " (recipe)" } else { "" });
            let t = *t;
            option(label, set, &move |_, _| now == Some(t), None)
        }).collect();
        SpeedKnob { knob: knob.into(), label: label.into(), risk: knob_risk(knob).into(), options }
    };
    knobs.push(tokens("last_tokens_m", if n > 1 { "Heal tokens of the last stage" } else { "Heal tokens" }, true));
    if n > 1 {
        knobs.push(tokens("stage_a_tokens_m", "Heal tokens of the first stage", false));
    }
    // the number of stages
    {
        let opts: Vec<SpeedOption> = [2u32, 1].iter().map(|k| option(if *k == 1 { "One stage".into() } else { "Two stages (recipe)".into() }, set_of(&[("stages", Value::from(*k))]), &move |p, _| p.stages.len() as u32 == (*k).min(2), None)).collect();
        if opts.iter().filter_map(|o| o.total_h).collect::<Vec<_>>().windows(2).any(|w| w[0] != w[1]) {
            knobs.push(SpeedKnob { knob: "stages".into(), label: "Stages".into(), risk: knob_risk("stages").into(), options: opts });
        }
    }
    // the shared sequence length (and with it the shared answers of the original)
    if spec.stages > 1 && n > 1 {
        let natural: Vec<u32> = auto_plan::stage_plans(ins, &AutoSpec { share_teacher_cache: false, ..spec.clone() }, &base_steps(ins, spec, &base_plan), vram_gb).iter().map(|s| s.seq).collect();
        let same_already = natural.windows(2).all(|w| w[0] == w[1]);
        let note = same_already.then(|| format!("On this card both stages already heal on sequence length {}: the original's answers are reused without this choice.", natural.first().copied().unwrap_or(0)));
        let opts = [(true, "Same sequence length in both stages: the original writes its answers once"), (false, "Each stage on its own sequence length (recipe)")].iter()
            .map(|(on, l)| option(l.to_string(), set_of(&[("share_teacher_cache", Value::Bool(*on))]), &move |_, a| a.share_teacher_cache == *on, note.clone())).collect();
        knobs.push(SpeedKnob { knob: "share_teacher_cache".into(), label: "Sequence length of the stages".into(), risk: knob_risk("share_teacher_cache").into(), options: opts });
    }
    if thinking {
        knobs.push(SpeedKnob { knob: "think_quick".into(), label: "Thinking checks".into(), risk: knob_risk("think_quick").into(), options:
            [(false, "Full (32 questions, recipe)"), (true, "Quick (16 questions)")].iter().map(|(q, l)| option(l.to_string(), set_of(&[("think_quick", Value::Bool(*q))]), &move |_, a| a.think_quick() == *q, None)).collect() });
    }
    knobs.push(SpeedKnob { knob: "quick".into(), label: "Try first".into(), risk: knob_risk("quick").into(), options:
        [(false, "Full (128 texts, recipe)"), (true, "Quick (64 texts)")].iter().map(|(q, l)| option(l.to_string(), set_of(&[("quick", Value::Bool(*q))]), &move |_, a| a.quick == *q, None)).collect() });
    if thinking {
        let off = set_of(&[("think_gate", Value::Bool(false))]);
        let one = set_of(&[("think_gate", Value::Bool(true)), ("think_gate_n", Value::from(1))]);
        let two = set_of(&[("think_gate", Value::Bool(true)), ("think_gate_n", Value::from(2))]);
        knobs.push(SpeedKnob { knob: "think_gate_n".into(), label: "Thinking check before healing".into(), risk: knob_risk("think_gate_n").into(), options: vec![
            option("Off".into(), off, &|_, a| !a.think_gate, None),
            option("The best cut only".into(), one, &|_, a| a.think_gate && a.think_gate_n == 1, None),
            option("The best 2 cuts (recipe)".into(), two, &|_, a| a.think_gate && a.think_gate_n == 2, None)] });
        knobs.push(SpeedKnob { knob: "check_cut".into(), label: "Check of each unhealed cut".into(), risk: knob_risk("check_cut").into(), options:
            [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("check_cut", Value::Bool(*on))]), &move |_, a| a.check_cut == *on, None)).collect() });
        let saves = retry_h.map(|r| format!("Saves nothing unless a check fails; then it saves {} of the retry.", hours_text(r)));
        knobs.push(SpeedKnob { knob: "retry_heal".into(), label: "Retry heal after a failed check".into(), risk: knob_risk("retry_heal").into(), options:
            [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("retry_heal", Value::Bool(*on))]), &move |_, a| a.retry_heal == *on, if *on { None } else { saves.clone() })).collect() });
    }
    {
        // heal in rounds: the time is a range (it stops early when a check passes), so the choice shows the best and the worst case
        let hr = |m: u32| Value::from(serde_json::json!({"round_m": m, "check": "quick"}));
        let now = spec.heal_rounds.as_ref().map(|r| (r.round_m, r.quick()));
        let eq = move |m: Option<u32>| move |_: &AutoPlan, _: &AutoSpec| now == m.map(|x| (x, true));
        let note = Some("Experimental. The range is: the first check passes (shortest) to every round to the cap (longest). A heal that stops at the first passed check saves the rest of its tokens.".to_string());
        knobs.push(SpeedKnob { knob: "heal_rounds".into(), label: "Heal in rounds, stop when good (experimental)".into(), risk: knob_risk("heal_rounds").into(), options: vec![
            option("Off: one heal on the whole budget (recipe)".into(), set_of(&[("heal_rounds", Value::Null)]), &eq(None), None),
            option("5 M rounds, quick checks".into(), set_of(&[("heal_rounds", hr(5))]), &eq(Some(5)), note.clone()),
            option("10 M rounds, quick checks".into(), set_of(&[("heal_rounds", hr(10))]), &eq(Some(10)), note),
        ] });
    }
    knobs.push(SpeedKnob { knob: "back_off".into(), label: "Cut one step less after a second failed check".into(), risk: knob_risk("back_off").into(), options:
        [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("back_off", Value::Bool(*on))]), &move |_, a| a.back_off == *on, if *on { None } else { Some("Saves a whole stage only if the checks fail twice.".into()) })).collect() });
    knobs.push(SpeedKnob { knob: "imatrix".into(), label: "Importance matrix for the export".into(), risk: knob_risk("imatrix").into(), options:
        [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("imatrix", Value::Bool(*on))]), &move |_, a| a.imatrix == *on, None)).collect() });

    let presets = presets().into_iter().map(|pr| {
        let mut row = PresetTime { id: pr.id.clone(), label: pr.label.clone(), blurb: pr.blurb.clone(), risk: pr.risk.clone(), total_h: None, retry_h: None, why: None };
        match spec.with_preset(&pr.id).ok().and_then(|s| timed_m(&s)) {
            Some((_, t)) => {
                row.total_h = Some(t.total_h);
                row.retry_h = t.extra.iter().find(|e| e.id == "retry_heal").map(|e| (r1h(e.minutes.0 / 60.0), r1h(e.minutes.1 / 60.0)));
            }
            None => row.why = Some("This cut cannot be planned with this setting.".into()),
        }
        row
    }).collect();
    SpeedInfo { presets, knobs, current_preset: spec.matching_preset(), default_preset: default_preset(total_b).into(), total_h: base.total_h }
}

/// The stops of the plan's stages (to plan the same stages again with other knobs).
fn base_steps(ins: &Inspect, spec: &AutoSpec, plan: &AutoPlan) -> Vec<auto_plan::Step> {
    match &plan.chosen {
        Some(c) => auto_plan::split_stages(ins, c, spec.stages),
        None => Vec::new(),
    }
}

fn r1h(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// "24 h 30 min" for (low, high) minutes: "24 h 30 min to 38 h 30 min".
pub fn hours_text(m: (f64, f64)) -> String {
    let one = |x: f64| {
        let t = x.round().max(0.0) as u64;
        if t >= 60 { if t % 60 == 0 { format!("{} h", t / 60) } else { format!("{} h {} min", t / 60, t % 60) } } else { format!("{t} min") }
    };
    format!("{} to {}", one(m.0), one(m.1))
}

// ------------------------------------------------------------------------------------------------------------------ "I have at most N hours"

/// One setting the time budget gave up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dropped {
    pub knob: String,
    pub label: String,
    pub from: String,
    pub to: String,
    /// what this step saved, hours (low, high)
    pub saves_h: (f64, f64),
    pub risk: String,
}

/// The answer to "I have at most N hours": the settings that make the run fit (the least harmful given up first), or the plain sentence that it cannot be done.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetFit {
    pub max_hours: f64,
    /// the spec to run: the asked one with the dropped settings applied (`max_hours` stays in it: the run watches the clock)
    pub spec: AutoSpec,
    pub dropped: Vec<Dropped>,
    pub fits: bool,
    /// the estimate of the asked settings, and of the settings the run will use, hours (low, high)
    pub asked_h: (f64, f64),
    pub total_h: (f64, f64),
    pub message: Option<String>,
}

/// One rung of the ladder: a setting that can be given up, where it applies, how, and what it costs in quality.
struct Rung {
    knob: &'static str,
    label: &'static str,
    risk: &'static str,
    to: &'static str,
    applies: fn(&AutoSpec, &AutoPlan) -> bool,
    from: fn(&AutoSpec, &AutoPlan) -> String,
    apply: fn(&mut AutoSpec),
}

fn tokens_text(m: u32) -> String {
    format!("{m} M tokens")
}

fn ladder() -> Vec<Rung> {
    vec![
        Rung { knob: "think_quick", label: "Thinking checks", to: "Quick (16 questions)", risk: knob_risk("think_quick"),
               applies: |s, _| s.keep_thinking && !s.think_quick(), from: |_, _| "Full (32 questions)".into(), apply: |s| s.think_quick = Some(true) },
        Rung { knob: "quick", label: "Try first", to: "Quick (64 texts)", risk: knob_risk("quick"),
               applies: |s, _| !s.quick, from: |_, _| "Full (128 texts)".into(), apply: |s| s.quick = true },
        Rung { knob: "check_cut", label: "Check of each unhealed cut", to: "off", risk: knob_risk("check_cut"),
               applies: |s, _| s.keep_thinking && s.check_cut, from: |_, _| "on".into(), apply: |s| s.check_cut = false },
        Rung { knob: "share_teacher_cache", label: "Sequence length of the stages", to: "the same in both stages", risk: knob_risk("share_teacher_cache"),
               applies: |s, p| !s.share_teacher_cache && p.stages.len() > 1, from: |_, p| format!("each stage on its own ({})", p.stages.iter().map(|g| g.seq.to_string()).collect::<Vec<_>>().join(", ")), apply: |s| s.share_teacher_cache = true },
        Rung { knob: "think_gate_n", label: "Thinking check before healing", to: "the best cut only", risk: knob_risk("think_gate_n"),
               applies: |s, p| s.keep_thinking && s.think_gate && s.think_gate_n > 1 && !p.try_list.is_empty(), from: |s, _| format!("the best {} cuts", s.think_gate_n), apply: |s| s.think_gate_n = 1 },
        Rung { knob: "retry_heal", label: "Retry heal after a failed check", to: "off", risk: knob_risk("retry_heal"),
               applies: |s, _| s.keep_thinking && s.retry_heal, from: |_, _| "on".into(), apply: |s| s.retry_heal = false },
        Rung { knob: "stage_a_tokens_m", label: "Heal tokens of the first stage", to: "20 M tokens", risk: knob_risk("stage_a_tokens_m"),
               applies: |s, p| p.stages.len() > 1 && p.stages[0].tokens_m > 20 && s.stage_a_tokens_m.is_none_or(|t| t > 20), from: |_, p| tokens_text(p.stages[0].tokens_m), apply: |s| s.stage_a_tokens_m = Some(20) },
        Rung { knob: "stages", label: "Stages", to: "one stage", risk: knob_risk("stages"),
               applies: |s, p| s.stages > 1 && p.stages.len() > 1, from: |_, p| format!("{} stages", p.stages.len()), apply: |s| s.stages = 1 },
        Rung { knob: "last_tokens_m", label: "Heal tokens of the last stage", to: "20 M tokens", risk: knob_risk("last_tokens_m"),
               applies: |s, p| p.stages.last().is_some_and(|g| g.tokens_m > 20) && s.last_tokens_m.is_none_or(|t| t > 20), from: |_, p| tokens_text(p.stages.last().map(|g| g.tokens_m).unwrap_or(0)), apply: |s| s.last_tokens_m = Some(20) },
    ]
}

/// Makes the run fit `max_hours`: gives up the settings of the ladder one at a time, the least harmful first (quick thinking checks, quick Try first, no check of the cut, the same
/// sequence length in both stages, one cut checked before healing, no retry, a lighter first stage, one stage, a light last stage) and stops at the first that fits on the HIGH end
/// of the estimate (a retry is never in it). Never touches what was asked for: the sizes, the measured file, Studio, the quants, the thinking itself. When even the last rung does
/// not fit it says so in plain words and the spec stays the asked one with all of them applied (for the numbers).
pub fn fit_budget(ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, history: &History, max_hours: f64) -> BudgetFit {
    let mut cur = AutoSpec { max_hours: Some(max_hours), ..spec.clone() };
    let nothing = |cur: AutoSpec| BudgetFit { max_hours, spec: cur, dropped: Vec::new(), fits: true, asked_h: (0.0, 0.0), total_h: (0.0, 0.0), message: None };
    let Some((mut p, mut t)) = timed(ins, &cur, vram_gb, history) else { return nothing(cur) };
    let asked_h = t.total_h;
    let mut dropped: Vec<Dropped> = Vec::new();
    if t.total_h.1 > max_hours {
        for r in ladder() {
            if !(r.applies)(&cur, &p) {
                continue;
            }
            let from = (r.from)(&cur, &p);
            let mut next = cur.clone();
            (r.apply)(&mut next);
            next.preset = Some("custom".into());
            let Some((np, nt)) = timed(ins, &next, vram_gb, history) else { continue };
            dropped.push(Dropped { knob: r.knob.into(), label: r.label.into(), from, to: r.to.into(), saves_h: (r1h(t.total_h.0 - nt.total_h.0), r1h(t.total_h.1 - nt.total_h.1)), risk: r.risk.into() });
            (cur, p, t) = (next, np, nt);
            if t.total_h.1 <= max_hours {
                break;
            }
        }
    }
    let fits = t.total_h.1 <= max_hours;
    let message = (!fits).then(|| format!("Even the fastest settings take about {} for this cut, and you have {max_hours:.0} h. Choose more time or a larger size.", hours_text((t.total_h.0 * 60.0, t.total_h.1 * 60.0))));
    BudgetFit { max_hours, spec: cur, dropped, fits, asked_h, total_h: t.total_h, message }
}

/// The sentences for the plan, the notes and the report: what the time limit made Auto-cut give up, each with its risk.
pub fn budget_sentences(fit: &BudgetFit) -> Vec<String> {
    if fit.dropped.is_empty() {
        return Vec::new();
    }
    let mut v = vec![format!("To fit your {:.0} hours the settings were changed: the estimate went from {:.0}-{:.0} h to {:.0}-{:.0} h.", fit.max_hours, fit.asked_h.0, fit.asked_h.1, fit.total_h.0, fit.total_h.1)];
    for d in &fit.dropped {
        v.push(format!("{}: {} instead of {} (saves {:.1}-{:.1} h). {}", d.label, d.to, d.from, d.saves_h.0, d.saves_h.1, d.risk));
    }
    v
}

// ------------------------------------------------------------------------------------------------------------------ the report

/// A speed setting that is not the recommended recipe's, with its risk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deviation {
    pub knob: String,
    pub label: String,
    pub value: String,
    pub recipe: String,
    pub risk: String,
}

/// Every speed setting of the run that differs from the recommended recipe, as plain sentences with the risk of each: the report's "Speed settings" section. `stages` are the planned
/// stages (their tokens and sequence lengths are what the run used), `total_b` the original's size. Empty for the recipe.
pub fn deviations(spec: &AutoSpec, stages: &[StagePlan], total_b: f64) -> Vec<Deviation> {
    let mut v = Vec::new();
    let thinking = spec.keep_thinking;
    let mut add = |knob: &str, label: &str, value: String, recipe: String| v.push(Deviation { knob: knob.into(), label: label.into(), value, recipe, risk: knob_risk(knob).into() });
    let (first, last) = (stages.first(), stages.last());
    if let Some(g) = last {
        let recipe = auto_plan::recipe_tokens_m(thinking, g.params_b, total_b, true);
        if g.tokens_m != recipe {
            add("last_tokens_m", if stages.len() > 1 { "Heal tokens of the last stage" } else { "Heal tokens" }, tokens_text(g.tokens_m), tokens_text(recipe));
        }
    }
    if stages.len() > 1 {
        if let Some(g) = first {
            let recipe = auto_plan::recipe_tokens_m(thinking, g.params_b, total_b, false);
            if g.tokens_m != recipe {
                add("stage_a_tokens_m", "Heal tokens of the first stage", tokens_text(g.tokens_m), tokens_text(recipe));
            }
        }
        if spec.share_teacher_cache {
            add("share_teacher_cache", "Sequence length of the stages", format!("the same in both stages ({})", stages.iter().map(|s| s.seq).min().unwrap_or(0)), "each stage on its own".into());
        }
    }
    if spec.stages == 1 {
        add("stages", "Stages", "one stage".into(), "two stages".into());
    }
    if spec.quick {
        add("quick", "Try first", format!("Quick ({} texts)", plan::TRY_QUICK_SAMPLES), format!("Full ({} texts)", plan::TRY_FULL_SAMPLES));
    }
    if thinking {
        if spec.think_quick() {
            add("think_quick", "Thinking checks", "Quick (16 questions)".into(), "Full (32 questions)".into());
        }
        if !spec.think_gate {
            add("think_gate_n", "Thinking check before healing", "off".into(), "the best 2 cuts".into());
        } else if spec.think_gate_n != 2 {
            add("think_gate_n", "Thinking check before healing", format!("the best {} cut{}", spec.think_gate_n, if spec.think_gate_n == 1 { "" } else { "s" }), "the best 2 cuts".into());
        }
        if !spec.check_cut {
            add("check_cut", "Check of each unhealed cut", "off".into(), "on".into());
        }
        if !spec.retry_heal {
            add("retry_heal", "Retry heal after a failed check", "off".into(), "on".into());
        }
    }
    if !spec.back_off {
        add("back_off", "Cut one step less after a second failed check", "off".into(), "on".into());
    }
    if !spec.imatrix {
        add("imatrix", "Importance matrix for the export", "off".into(), "on".into());
    }
    if let Some(r) = &spec.heal_rounds {
        add("heal_rounds", "Heal in rounds (experimental)", format!("rounds of {} M tokens, {} thinking checks, stop at the first pass", r.round_m, if r.quick() { "quick" } else { "full" }), "one heal on the whole budget".into());
    }
    v
}
