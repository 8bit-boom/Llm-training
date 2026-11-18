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
pub const KNOBS: [&str; 12] = ["stages", "stage_a_tokens_m", "last_tokens_m", "share_teacher_cache", "quick", "think_quick", "think_gate", "think_gate_n", "check_cut", "retry_heal", "back_off", "imatrix"];

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
        KNOBS.iter().filter_map(|k| v.get(*k).map(|x| (k.to_string(), x.clone()))).collect()
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
    let Some((base_plan, base)) = timed(ins, spec, vram_gb, history) else {
        return SpeedInfo { presets: Vec::new(), knobs: Vec::new(), current_preset: spec.matching_preset(), default_preset: default_preset(total_b).into(), total_h: (0.0, 0.0) };
    };
    let thinking = spec.keep_thinking;
    let n = base_plan.stages.len();
    let retry_h = base.extra.iter().find(|e| e.id == "retry_heal").map(|e| e.minutes);
    let delta = |t: (f64, f64)| (r1h((t.0 - base.total_h.0) * 60.0) + 0.0, r1h((t.1 - base.total_h.1) * 60.0) + 0.0);
    let option = |label: String, set: Map<String, Value>, current: &dyn Fn(&AutoPlan, &AutoSpec) -> bool, note: Option<String>| -> SpeedOption {
        let alt = patched(spec, &set);
        match alt.as_ref().and_then(|a| timed(ins, a, vram_gb, history).map(|(p, t)| (a, p, t))) {
            Some((a, p, t)) => SpeedOption { label, set, current: current(&p, a), total_h: Some(t.total_h), delta_min: Some(delta(t.total_h)), note },
            None => SpeedOption { label, set, current: false, total_h: None, delta_min: None, note: Some("This cannot be planned with this choice.".into()) },
        }
    };
    let mut knobs: Vec<SpeedKnob> = Vec::new();
    let student_b = |last: bool| base_plan.stages.get(if last { n.saturating_sub(1) } else { 0 }).map(|s| s.params_b).unwrap_or(total_b);

    // the heal tokens: the biggest lever, linear in the time
    let tokens = |knob: &str, label: &str, last: bool, risk: &str| -> SpeedKnob {
        let recipe = auto_plan::recipe_tokens_m(thinking, student_b(last), total_b, last);
        let idx = if last { n.saturating_sub(1) } else { 0 };
        let now = base_plan.stages.get(idx).map(|s| s.tokens_m);
        let options = plan::HEAL_TOKENS_M.iter().map(|t| {
            let set = set_of(&[(knob, if *t == recipe { Value::Null } else { Value::from(*t) })]);
            let label = format!("{t} M{}", if *t == recipe { " (recipe)" } else { "" });
            let t = *t;
            option(label, set, &move |_, _| now == Some(t), None)
        }).collect();
        SpeedKnob { knob: knob.into(), label: label.into(), risk: risk.into(), options }
    };
    knobs.push(tokens("last_tokens_m", if n > 1 { "Heal tokens of the last stage" } else { "Heal tokens" }, true,
                      "Fewer tokens is linearly faster and learns less. Below the recipe a model that thinks has failed the thinking checks before."));
    if n > 1 {
        knobs.push(tokens("stage_a_tokens_m", "Heal tokens of the first stage", false, "The second stage starts from what the first one learned: a lighter first stage makes the second one's job harder."));
    }
    // the number of stages
    {
        let opts: Vec<SpeedOption> = [2u32, 1].iter().map(|k| option(if *k == 1 { "One stage".into() } else { "Two stages (recipe)".into() }, set_of(&[("stages", Value::from(*k))]), &move |p, _| p.stages.len() as u32 == (*k).min(2), None)).collect();
        if opts.iter().filter_map(|o| o.total_h).collect::<Vec<_>>().windows(2).any(|w| w[0] != w[1]) {
            knobs.push(SpeedKnob { knob: "stages".into(), label: "Stages".into(), risk: "One stage cuts everything at once and heals once: it skips a cut, a check, a merge and a whole heal, and is the riskier way for a model that thinks.".into(), options: opts });
        }
    }
    // the shared sequence length (and with it the shared answers of the original)
    if spec.stages > 1 && n > 1 {
        let natural: Vec<u32> = auto_plan::stage_plans(ins, &AutoSpec { share_teacher_cache: false, ..spec.clone() }, &base_steps(ins, spec, &base_plan), vram_gb).iter().map(|s| s.seq).collect();
        let same_already = natural.windows(2).all(|w| w[0] == w[1]);
        let note = same_already.then(|| format!("On this card both stages already heal on sequence length {}: the original's answers are reused without this choice.", natural.first().copied().unwrap_or(0)));
        let opts = [(true, "Same sequence length in both stages: the original writes its answers once"), (false, "Each stage on its own sequence length (recipe)")].iter()
            .map(|(on, l)| option(l.to_string(), set_of(&[("share_teacher_cache", Value::Bool(*on))]), &move |_, a| a.share_teacher_cache == *on, note.clone())).collect();
        knobs.push(SpeedKnob { knob: "share_teacher_cache".into(), label: "Sequence length of the stages".into(), risk: "Both stages heal on the shorter sequence length, so the larger student sees its longest reasoning traces cut short. It saves the original writing the same answers twice (the biggest saving when the lengths differ).".into(), options: opts });
    }
    if thinking {
        knobs.push(SpeedKnob { knob: "think_quick".into(), label: "Thinking checks".into(), risk: "The quick checks ask 16 questions instead of 32: a third faster, and a close call can go either way.".into(), options:
            [(false, "Full (32 questions, recipe)"), (true, "Quick (16 questions)")].iter().map(|(q, l)| option(l.to_string(), set_of(&[("think_quick", Value::Bool(*q))]), &move |_, a| a.think_quick() == *q, None)).collect() });
    }
    knobs.push(SpeedKnob { knob: "quick".into(), label: "Try first".into(), risk: "Fewer texts: quicker, and a close race between two cuts is less certain.".into(), options:
        [(false, "Full (128 texts, recipe)"), (true, "Quick (64 texts)")].iter().map(|(q, l)| option(l.to_string(), set_of(&[("quick", Value::Bool(*q))]), &move |_, a| a.quick == *q, None)).collect() });
    if thinking {
        let off = set_of(&[("think_gate", Value::Bool(false))]);
        let one = set_of(&[("think_gate", Value::Bool(true)), ("think_gate_n", Value::from(1))]);
        let two = set_of(&[("think_gate", Value::Bool(true)), ("think_gate_n", Value::from(2))]);
        knobs.push(SpeedKnob { knob: "think_gate_n".into(), label: "Thinking check before healing".into(), risk: "Checks the best cuts for thinking before any heal and picks by how they think; with fewer cuts a cut with a low error but little thinking may be healed.".into(), options: vec![
            option("Off".into(), off, &|_, a| !a.think_gate, None),
            option("The best cut only".into(), one, &|_, a| a.think_gate && a.think_gate_n == 1, None),
            option("The best 2 cuts (recipe)".into(), two, &|_, a| a.think_gate && a.think_gate_n == 2, None)] });
        knobs.push(SpeedKnob { knob: "check_cut".into(), label: "Check of each unhealed cut".into(), risk: "Informational only (nothing is decided by it), but the original is then checked after the first heal, so the gates keep their yardstick.".into(), options:
            [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("check_cut", Value::Bool(*on))]), &move |_, a| a.check_cut == *on, None)).collect() });
        let saves = retry_h.map(|r| format!("Saves nothing unless a check fails; then it saves {} of the retry.", hours_text(r)));
        knobs.push(SpeedKnob { knob: "retry_heal".into(), label: "Retry heal after a failed check".into(), risk: "The retry heals the healed model again on 80 M tokens. Without it a failed check goes straight to cutting one step less.".into(), options:
            [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("retry_heal", Value::Bool(*on))]), &move |_, a| a.retry_heal == *on, if *on { None } else { saves.clone() })).collect() });
    }
    knobs.push(SpeedKnob { knob: "back_off".into(), label: "Cut one step less after a second failed check".into(), risk: "Without it the run ends with its report when a stage still fails; with it a stage is made again, larger.".into(), options:
        [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("back_off", Value::Bool(*on))]), &move |_, a| a.back_off == *on, if *on { None } else { Some("Saves a whole stage only if the checks fail twice.".into()) })).collect() });
    knobs.push(SpeedKnob { knob: "imatrix".into(), label: "Importance matrix for the export".into(), risk: "The small sizes (Q4 and below) are made without it and are a little less exact.".into(), options:
        [(true, "On (recipe)"), (false, "Off")].iter().map(|(on, l)| option(l.to_string(), set_of(&[("imatrix", Value::Bool(*on))]), &move |_, a| a.imatrix == *on, None)).collect() });

    let presets = presets().into_iter().map(|pr| {
        let mut row = PresetTime { id: pr.id.clone(), label: pr.label.clone(), blurb: pr.blurb.clone(), risk: pr.risk.clone(), total_h: None, retry_h: None, why: None };
        match spec.with_preset(&pr.id).ok().and_then(|s| timed(ins, &s, vram_gb, history)) {
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
