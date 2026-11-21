//! Change the settings of a stopped or failed Auto-cut run and continue (docs/AUTO_CUT.md, "Change settings and continue").
//!
//! Only settings that have NOT taken effect yet may change. `lock_of` says per setting, from the state of the steps: `free`, `confirm` (it would redo finished or started work: allowed only after
//! the person confirmed, with the work named) or `locked` (the identity of finished work: refused with a plain reason). `preview` shows the diff, the locks, and the time of the REMAINING steps
//! with the changed settings against the time that was left; `apply` changes the spec, plans the rest again, reconciles the steps (done steps are kept as they are, so nothing finished runs again)
//! and keeps the change in `settings_changes`. The spec the run was started with is kept in `start_spec`: nothing is rewritten silently.
use serde_json::{json, Map, Value};

use super::auto::{steps_after_plan, AutoState};
use super::auto_plan::{self, AutoSpec};
use super::auto_speed;
use super::plan::Inspect;
use crate::error::{Error, Result};
use crate::timing::History;

/// "free" | "confirm" | "locked", why, and (for "confirm") what would be redone.
#[derive(Debug, Clone, PartialEq)]
pub struct Lock {
    pub key: String,
    pub state: &'static str,
    pub reason: String,
    pub redo: Option<String>,
}

fn lock(key: &str, state: &'static str, reason: &str, redo: Option<&str>) -> Lock {
    Lock { key: key.into(), state, reason: reason.into(), redo: redo.map(String::from) }
}

/// The step has been begun (a stopped or failed step is "pending" again but keeps the time it started).
fn started(st: &AutoState, id: &str) -> bool {
    st.steps.iter().any(|s| s.id == id && (s.status != "pending" || s.started.is_some()))
}

fn finished(st: &AutoState, id: &str) -> bool {
    st.steps.iter().any(|s| s.id == id && matches!(s.status.as_str(), "done" | "skipped"))
}

fn heal_id(n: usize) -> String {
    format!("s{n}.heal")
}

/// A heal setting of stage `idx` (0-based): free until its heal started, a confirmation while it is started but not finished (the original's answers or the training start again), locked after.
fn heal_lock(st: &AutoState, key: &str, idxs: &[usize], what: &str) -> Lock {
    for &i in idxs {
        if finished(st, &heal_id(i + 1)) {
            return lock(key, "locked", &format!("Stage {} is healed already: its tokens, sequence length and rounds cannot change any more.", i + 1), None);
        }
    }
    for &i in idxs {
        if started(st, &heal_id(i + 1)) {
            return lock(key, "confirm", &format!("The heal of stage {} has started.", i + 1), Some(&format!("the heal of stage {} starts again{what}", i + 1)));
        }
    }
    lock(key, "free", "Not used yet.", None)
}

/// The lock of one setting (a field of the spec) for this run.
pub fn lock_of(st: &AutoState, key: &str) -> Lock {
    let n = st.stages.len().max(st.spec.stages as usize);
    let all: Vec<usize> = (0..n).collect();
    match key {
        "model_id" | "target_b" | "keep_thinking" | "allow_many_blocks" | "cut" => lock(key, "locked", "This decides which model is cut and how: start a new run for another one.", None),
        "preset" | "max_hours" | "notify_stages" | "retry_heal" | "back_off" => lock(key, "free", "Used when a step needs it.", None),
        "stages" => if started(st, "s1.cut") || started(st, "think_gate") { lock(key, "locked", "The cut of stage 1 is made already: the number of stages is fixed.", None) } else { lock(key, "free", "No cut has been made yet.", None) },
        "quick" | "search_minutes" => if started(st, "try") { lock(key, "locked", "Try first has started (or finished): how it looks for the cut is fixed.", None) } else { lock(key, "free", "Try first has not started.", None) },
        "think_gate" | "think_gate_n" => if started(st, "think_gate") { lock(key, "locked", "The thinking check before healing has started.", None) } else { lock(key, "free", "It has not started.", None) },
        "check_cut" => if started(st, "s1.check_cut") || started(st, "s1.heal") { lock(key, "locked", "The check of the first cut has been reached already.", None) } else { lock(key, "free", "Not reached.", None) },
        "think_quick" => if st.baseline.is_some() { lock(key, "confirm", "The original was checked in the old mode: a check in another mode cannot be compared with it.", Some("the original's thinking check (the yardstick) is made again in the new mode")) } else { lock(key, "free", "No check has been made yet.", None) },
        "stage_a_tokens_m" => if st.spec.stages > 1 { heal_lock(st, key, &all[..n.saturating_sub(1)], " (the original's answers are made again for the new amount)") } else { lock(key, "free", "There is only one stage.", None) },
        "last_tokens_m" => heal_lock(st, key, &[n.saturating_sub(1)], " (the original's answers are made again for the new amount)"),
        "share_teacher_cache" => heal_lock(st, key, &all, " (another sequence length makes the original answer again)"),
        "heal_rounds" => heal_lock(st, key, &all, ""),
        "vllm" => match st.check_engine.as_deref() {
            Some(_) if st.baseline.is_some() => lock(key, "confirm", "The original was checked with one engine; the checks of a run use the same one.", Some("the original's thinking check (the yardstick) is made again with the new engine")),
            _ => lock(key, "free", "Used by the steps that are not done.", None),
        },
        "quants" | "target_gb" | "studio_publish" | "imatrix" => if started(st, "export") { lock(key, "locked", "The export has started: its sizes and options are fixed.", None) } else { lock(key, "free", "The export has not started.", None) },
        "keep_intermediate" => if st.made.iter().any(|m| m.removed) { lock(key, "locked", "Files were deleted already; they cannot be kept any more.", None) } else { lock(key, "free", "Nothing was deleted yet.", None) },
        _ => lock(key, "locked", "This is not a setting that can be changed while the run goes on.", None),
    }
}

/// Every setting that can be asked about, with its lock (what the window greys out).
pub fn lock_table(st: &AutoState) -> Vec<Lock> {
    ["preset", "stages", "quick", "search_minutes", "think_gate", "think_gate_n", "check_cut", "think_quick", "stage_a_tokens_m", "last_tokens_m", "share_teacher_cache", "heal_rounds", "retry_heal", "back_off", "imatrix",
     "max_hours", "vllm", "quants", "target_gb", "studio_publish", "keep_intermediate", "notify_stages", "model_id", "target_b", "keep_thinking", "allow_many_blocks", "cut"]
        .iter().map(|k| lock_of(st, k)).collect()
}

fn label(key: &str) -> &'static str {
    match key {
        "last_tokens_m" => "heal tokens of the last stage (million)",
        "stage_a_tokens_m" => "heal tokens of the first stage (million)",
        "heal_rounds" => "healing in rounds",
        "think_quick" => "thinking checks quick",
        "quick" => "Try first quick",
        "share_teacher_cache" => "same sequence length in both stages",
        "vllm" => "vLLM",
        "max_hours" => "time limit (hours)",
        "check_cut" => "check of the unhealed cut",
        "think_gate" => "thinking check before healing",
        "quants" => "export sizes",
        "target_gb" => "target size (GB)",
        "retry_heal" => "retry after a failed check",
        "back_off" => "cut one step less after a second failed check",
        "imatrix" => "importance matrix",
        "search_minutes" => "minutes of the search",
        _ => "setting",
    }
}

fn show(v: &Value) -> String {
    match v {
        Value::Null => "the recipe's value".into(),
        Value::Bool(b) => if *b { "on".into() } else { "off".into() },
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn steps_left(st: &AutoState, spec: &AutoSpec, with_try: bool) -> Vec<String> {
    steps_after_plan(spec, st.stages.len().max(1) as u32, with_try).into_iter().map(|s| s.id).filter(|id| !finished(st, id)).collect()
}

/// Hours (low, high) of the steps that are still to run, with `spec` (the plan and the estimate are made again).
pub fn remaining_hours(st: &AutoState, ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, free_gb: Option<f64>, history: &History) -> Option<(f64, f64)> {
    let plan = auto_plan::plan(ins, spec, vram_gb, free_gb);
    if !plan.ok {
        return None;
    }
    let t = auto_plan::time_estimate(ins, spec, &plan, history);
    let with_try = st.steps.iter().any(|s| s.id == "try");
    let left = steps_left(st, spec, with_try);
    let (lo, hi) = t.steps.iter().filter(|x| left.contains(&x.id)).fold((0.0, 0.0), |a, x| (a.0 + x.minutes.0, a.1 + x.minutes.1));
    Some((((lo / 60.0) * 10.0).round() / 10.0, ((hi / 60.0) * 10.0).round() / 10.0))
}

/// The answer to "what if I change this?" (no state is touched): `{ok, accepted, refused: [{key, reason}], confirm: [{key, reason, redo}], diff: [{key, from, to, text}], remaining_h, was_remaining_h, warnings}`.
pub fn preview(st: &AutoState, patch: &Map<String, Value>, vram_gb: Option<f64>, free_gb: Option<f64>, history: &History) -> Value {
    let cur = serde_json::to_value(&st.spec).unwrap_or(Value::Null);
    let (mut accepted, mut refused, mut confirm, mut diff) = (Map::new(), Vec::new(), Vec::new(), Vec::new());
    for (k, v) in patch {
        if cur.get(k).cloned().unwrap_or(Value::Null) == *v {
            continue;          // not a change
        }
        let l = lock_of(st, k);
        match l.state {
            "locked" => refused.push(json!({"key": k, "reason": l.reason})),
            s => {
                if s == "confirm" {
                    confirm.push(json!({"key": k, "reason": l.reason, "redo": l.redo}));
                }
                let from = cur.get(k).cloned().unwrap_or(Value::Null);
                diff.push(json!({"key": k, "from": from, "to": v, "text": format!("you changed: {} from {} to {}", label(k), show(&from), show(v))}));
                accepted.insert(k.clone(), v.clone());
            }
        }
    }
    let new_spec = auto_speed::patched(&st.spec, &accepted).filter(|s| s.validate().is_ok());
    let mut out = json!({"ok": refused.is_empty() && new_spec.is_some(), "accepted": accepted.keys().collect::<Vec<_>>(), "refused": refused, "confirm": confirm, "diff": diff, "warnings": []});
    if new_spec.is_none() && !accepted.is_empty() {
        out["refused"].as_array_mut().map(|a| a.push(json!({"key": "", "reason": "The changed settings do not make a valid run."})));
        out["ok"] = json!(false);
    }
    if let (Some(ins), Some(ns)) = (st.inspect.as_ref(), new_spec.as_ref()) {
        out["was_remaining_h"] = json!(remaining_hours(st, ins, &st.spec, vram_gb, free_gb, history));
        out["remaining_h"] = json!(remaining_hours(st, ins, ns, vram_gb, free_gb, history));
        out["warnings"] = json!(auto_plan::plan(ins, ns, vram_gb, free_gb).warnings);
    }
    out["locks"] = json!(lock_table(st).iter().map(|l| json!({"key": l.key, "state": l.state, "reason": l.reason, "redo": l.redo})).collect::<Vec<_>>());
    out
}

/// Applies the patch to the run (a stopped or failed one): checks the locks, needs `confirmed` for what `confirm`s, plans the rest again, reconciles the steps and records the change.
pub fn apply(st: &mut AutoState, patch: &Map<String, Value>, confirmed: bool, vram_gb: Option<f64>, free_gb: Option<f64>, history: &History, now_secs: i64) -> Result<()> {
    let pv = preview(st, patch, vram_gb, free_gb, history);
    if let Some(r) = pv["refused"].as_array().filter(|r| !r.is_empty()) {
        let why: Vec<String> = r.iter().map(|x| format!("{}: {}", x["key"].as_str().unwrap_or(""), x["reason"].as_str().unwrap_or(""))).collect();
        return Err(Error::new(format!("These settings cannot change now. {}", why.join(" "))));
    }
    let confirms = pv["confirm"].as_array().cloned().unwrap_or_default();
    if !confirms.is_empty() && !confirmed {
        let what: Vec<String> = confirms.iter().filter_map(|c| c["redo"].as_str().map(String::from)).collect();
        let more = pv["remaining_h"].as_array().zip(pv["was_remaining_h"].as_array()).map(|(n, w)| format!(" The time left goes from {:.1}-{:.1} h to {:.1}-{:.1} h.", w[0].as_f64().unwrap_or(0.0), w[1].as_f64().unwrap_or(0.0), n[0].as_f64().unwrap_or(0.0), n[1].as_f64().unwrap_or(0.0))).unwrap_or_default();
        return Err(Error::new(format!("This needs your confirmation: {}.{more}", what.join("; "))));
    }
    let accepted: Map<String, Value> = patch.iter().filter(|(k, _)| pv["accepted"].as_array().is_some_and(|a| a.iter().any(|x| x == *k))).map(|(k, v)| (k.clone(), v.clone())).collect();
    if accepted.is_empty() {
        return Ok(());
    }
    let new_spec = auto_speed::patched(&st.spec, &accepted).ok_or_else(|| Error::new("The changed settings do not make a valid run."))?;
    new_spec.validate()?;
    if st.start_spec.is_none() {
        st.start_spec = Some(st.spec.clone());
    }
    let had_baseline = st.baseline.is_some();
    st.spec = new_spec;
    // what a confirmed change redoes
    let mut redo: Vec<String> = confirms.iter().filter_map(|c| c["redo"].as_str().map(String::from)).collect();
    if confirms.iter().any(|c| matches!(c["key"].as_str(), Some("think_quick" | "vllm"))) && had_baseline {
        st.baseline = None;
        st.check_engine = None;
    }
    // the rest of the run is planned again with the new settings; the stages whose heal is not finished take their new numbers, everything done stays as it is
    if let Some(ins) = st.inspect.clone() {
        let plan = auto_plan::with_speed(auto_plan::plan(&ins, &st.spec, vram_gb, free_gb), &ins, &st.spec, vram_gb, history);
        if plan.ok && plan.stages.len() == st.stages.len() {
            for (i, ns) in plan.stages.iter().enumerate() {
                if !finished(st, &heal_id(i + 1)) {
                    let old = &mut st.stages[i].plan;
                    (old.tokens_m, old.seq, old.seq_pinned, old.mode, old.teacher_load, old.hours, old.heal_gb, old.verdict, old.batch, old.rank, old.cache_gb) =
                        (ns.tokens_m, ns.seq, ns.seq_pinned, ns.mode.clone(), ns.teacher_load.clone(), ns.hours, ns.heal_gb, ns.verdict.clone(), ns.batch, ns.rank, ns.cache_gb);
                }
            }
        }
        if plan.ok {
            st.plan = Some(plan);
        }
    }
    reconcile_steps(st);
    for s in st.steps.iter_mut() {
        if let Some(t) = st.plan.as_ref().and_then(|p| p.time.as_ref()).and_then(|t| t.steps.iter().find(|x| x.id == s.id)) {
            s.est = Some(t.minutes);
            s.est_parts = t.parts.iter().map(|p| (p.id.clone(), (p.minutes.0 + p.minutes.1) / 2.0)).collect();
        }
    }
    redo.dedup();
    st.settings_changes.push(json!({"at": chrono::DateTime::<chrono::Utc>::from_timestamp(now_secs, 0).map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string()), "who": "person",
                                    "changes": pv["diff"], "confirmed": confirmed && !confirms.is_empty(), "redo": redo, "remaining_h": pv["remaining_h"], "was_remaining_h": pv["was_remaining_h"]}));
    st.notes.push(format!("Settings changed when the run was continued: {}.", pv["diff"].as_array().map(|d| d.iter().filter_map(|x| x["text"].as_str().map(|t| t.trim_start_matches("you changed: ").to_string())).collect::<Vec<_>>().join("; ")).unwrap_or_default()));
    Ok(())
}

/// Makes the step list the one the new spec has: steps that are done or started stay as they are, steps that are not needed any more and have not started are dropped, new ones are added in order.
fn reconcile_steps(st: &mut AutoState) {
    let with_try = st.steps.iter().any(|s| s.id == "try");
    let want = steps_after_plan(&st.spec, st.stages.len().max(1) as u32, with_try);
    let mut out = vec![st.steps[0].clone()];
    for w in want {
        match st.steps.iter().find(|s| s.id == w.id) {
            Some(old) => out.push(old.clone()),
            None => out.push(w),
        }
    }
    for old in &st.steps[1..] {
        if !out.iter().any(|s| s.id == old.id) && old.status != "pending" {
            out.push(old.clone());          // finished work stays in the record even when the new settings would not have made it
        }
    }
    st.steps = out;
}

/// What to try after the step that stopped or failed (plain suggestions with the patch each one means; locked ones are left out). The common case: a failed heal.
pub fn suggestions(st: &AutoState) -> Vec<Value> {
    let failed = st.steps.iter().find(|s| s.status == "failed" || s.status == "stopped").map(|s| s.id.clone()).unwrap_or_default();
    let why = st.error.clone().unwrap_or_default().to_lowercase();
    let mut out = Vec::new();
    let mut add = |key: &str, text: &str, patch: Value| {
        if lock_of(st, key).state != "locked" {
            out.push(json!({"text": text, "patch": patch}));
        }
    };
    if failed.ends_with(".heal") && why.contains("vllm") && (why.contains("container") || why.contains("was required")) && st.spec.vllm.as_ref().is_some_and(|v| v.teacher_cache || v.automatic) {
        // the vLLM container route failed and vLLM was required: the first suggestion is to switch the vLLM teacher cache off (the thinking checks keep their own setting)
        let checks = st.spec.vllm.as_ref().is_some_and(|v| v.checks);
        add("vllm", "vLLM was required for the original's answers and the container did not work: turn the vLLM teacher cache off and go on with the standard teacher (use \"Test the container copy\" on the Heal card to see which way of copying files works here).",
            if checks { json!({"vllm": {"teacher_cache": false, "checks": true, "automatic": false}}) } else { json!({"vllm": Value::Null}) });
    }
    if failed.ends_with(".heal") {
        if why.contains("out of memory") {
            add("share_teacher_cache", "The graphics card ran out of memory in the heal: the same sequence length in both stages is not the cause, but a shorter heal is cheaper; try fewer tokens for the last stage.", json!({"last_tokens_m": 20}));
        }
        add("heal_rounds", "Heal in rounds of 5 M tokens with a quick check after each and stop at the first that passes (experimental).", json!({"heal_rounds": {"round_m": 5, "check": "quick"}}));
        add("vllm", if st.spec.vllm.is_some() { "Switch vLLM off for the rest of the run (the standard engine does everything)." } else { "Make the original's answers with vLLM (experimental, needs the self-test and a teacher check)." },
            if st.spec.vllm.is_some() { json!({"vllm": Value::Null}) } else { json!({"vllm": {"teacher_cache": true, "checks": false, "automatic": false}}) });
    }
    if failed.contains("check") {
        add("think_quick", "Check with the quick questions (16) to see sooner whether a stage passes.", json!({"think_quick": true}));
        add("retry_heal", "Heal the stage again when a check fails.", json!({"retry_heal": true}));
    }
    out
}
