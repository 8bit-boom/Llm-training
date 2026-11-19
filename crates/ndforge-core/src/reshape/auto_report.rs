//! Auto-cut's report (docs/AUTO_CUT.md, "The report"): what a run did, in numbers, as `auto_report.json` and `auto_report.md` in the run's folder.
//!
//! Pure functions of the saved state (`AutoState`): the same state gives the same report (the times are the ones in the state), so the report can be made again at any
//! time, also for a run that was stopped. `write` is called when a run ends (done, gates_failed, failed); the window asks for `report` to show it.
use std::path::Path;

use serde_json::{json, Value};

use super::auto::{AutoState, StageState};
use super::auto_plan::StagePlan;
use super::runs::write_json;
use crate::error::Result;

pub const JSON_FILE: &str = "auto_report.json";
pub const MD_FILE: &str = "auto_report.md";

/// The numbers of a check that the table shows, in this order.
const METRICS: [(&str, &str); 6] = [
    ("think_rate", "thinks"),
    ("closed_rate", "closes the thought"),
    ("accuracy", "accuracy"),
    ("loop_rate", "loops"),
    ("garbage_rate", "garbage"),
    ("median_think_tokens", "median thought (tokens)"),
];

fn secs_between(a: &Option<String>, b: &Option<String>) -> Option<i64> {
    let parse = |s: &str| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ").ok();
    let (a, b) = (parse(a.as_deref()?)?, parse(b.as_deref()?)?);
    Some((b - a).num_seconds().max(0))
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn failed_gates(check: &Option<Value>) -> Vec<String> {
    check.as_ref().and_then(|c| c["gates"]["failed"].as_array()).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn stage_json(s: &StageState) -> Value {
    let p = &s.plan;
    let summary = s.heal.as_ref().and_then(|h| h.summary.clone()).unwrap_or(Value::Null);
    let mut out = stage_core(s, p, &summary);
    // what the heal's rows cover (the worker's summary; absent for an older worker)
    if summary["coverage"].is_object() {
        out["coverage"] = summary["coverage"].clone();
    }
    // a stage healed in rounds: the latest heal's rounds hold every round (an extension goes on from the same ones); additive, absent without rounds
    if let Some(r) = s.rounds.last() {
        out["rounds"] = r.clone();
        out["rounds_saved_m"] = json!((p.tokens_m as f64 + s.extra_tokens_m as f64 - r["tokens_m"].as_f64().unwrap_or(0.0)).max(0.0));
    }
    out
}

fn stage_core(s: &StageState, p: &StagePlan, summary: &Value) -> Value {
    json!({
        "index": p.index, "of": p.of, "params_b": p.params_b, "cut_pct": p.cut_pct, "blocks_cut": p.blocks_cut, "blocks_after": p.blocks_after,
        "mlp_width": p.intermediate_after, "tokens_m": s.heal.as_ref().map(|h| h.tokens_m).unwrap_or(p.tokens_m), "seq": p.seq, "rank": p.rank,
        "passed": s.passed, "attempts": s.attempts, "retries": s.attempts.len().saturating_sub(1), "backed_off": s.attempt >= 2,
        "cut_check": s.cut_check, "heal_check": s.heal_check, "cut_loss": s.cut_loss,
        // the healing curves: the teacher's loss and the student's before and after, and the KL at the start and the end
        "kl_first": summary["kd_first"], "kl_last": summary["kd_last"], "loss_before": summary["calib_loss_before"], "loss_after": summary["calib_loss_after"], "teacher_loss": summary["teacher_calib_loss"],
        "healed_model": s.heal.as_ref().map(|h| h.name.clone()),
    })
}

/// Adds `timings` (where the time of the jobs went, from the job logs that started while this run lasted) to a report. A run whose logs are gone has none.
pub fn add_timings(r: &mut Value, st: &AutoState) {
    let files = crate::timing::logs_between(&crate::envmgr::joblog::dir(), &st.created, &st.updated);
    if files.is_empty() {
        return;
    }
    let summary = crate::timing::summarize(&files);
    if summary["job_seconds"].as_f64().unwrap_or(0.0) <= 0.0 {
        return;
    }
    let advice = crate::timing::advice(&summary);
    r["timings"] = json!({"summary": summary, "advice": advice});
}

/// The report of a run as data.
pub fn build(st: &AutoState) -> Value {
    let plan = st.plan.as_ref();
    let original = st.original.as_ref();
    let last = st.stages.last();
    let actual_b = last.map(|s| s.plan.params_b);
    let time = plan.and_then(|p| p.time.as_ref());
    let steps: Vec<Value> = st.steps.iter().map(|s| {
        let seconds = secs_between(&s.started, &s.finished);
        let est = time.and_then(|t| t.steps.iter().find(|x| x.id == s.id));
        json!({"id": s.id, "label": s.label, "status": s.status, "note": s.note, "seconds": seconds,
               "estimated_minutes": est.map(|x| json!([x.minutes.0, x.minutes.1])), "actual_minutes": seconds.map(|n| r1(n as f64 / 60.0)), "estimate_basis": est.map(|x| x.basis.clone())})
    }).collect();
    let active: i64 = st.steps.iter().filter_map(|s| secs_between(&s.started, &s.finished)).sum();
    let mut failed: Vec<String> = Vec::new();
    for s in &st.stages {
        if !s.passed {
            failed.extend(failed_gates(&s.heal_check));
        }
    }
    failed.extend(failed_gates(&st.gguf_check));
    failed.dedup();
    json!({
        "run": st.id, "status": st.status, "step": st.step, "error": st.error, "created": st.created, "updated": st.updated,
        "model": original.map(|o| o.name.clone()), "original_b": plan.map(|p| p.original_b), "target_b": st.spec.target_b, "actual_b": actual_b,
        "blocks_removed": last.map(|s| s.plan.blocks_after), "mlp_width": last.and_then(|s| s.plan.intermediate_after),
        "keep_thinking": st.spec.keep_thinking, "stages_planned": st.spec.stages, "quick": st.spec.quick,
        "stages": st.stages.iter().map(stage_json).collect::<Vec<_>>(),
        "baseline": st.baseline.as_ref().map(|b| b["main"].clone()),
        "gguf_check": st.gguf_check,
        "failed_gates": failed,
        "retries": st.stages.iter().map(|s| s.attempts.len().saturating_sub(1)).sum::<usize>(),
        "export": st.export.as_ref().map(|e| e["ggufs"].clone()),
        "studio": st.publish,
        "disk": plan.and_then(|p| p.disk.as_ref()).map(|d| json!({"peak_gb": d.peak_gb, "peak_step": d.peak_step, "final_gb": d.final_gb, "keep_intermediate": d.keep_intermediate, "measured": false})),
        "active_minutes": r1(active as f64 / 60.0),
        "try": st.try_result, "measure": st.measure, "notes": st.notes, "steps": steps,
        "think_gate": st.think_gate, "guardrails": guardrails(st),
        "compare": st.compare, "target_gb": st.spec.target_gb, "budget": st.budget, "speed": speed_json(st),
        "estimate": estimate_json(st, active),
    })
}

/// The speed settings of the run: the preset, and every value that is not the recommended recipe's with its risk (the plan warns about the same ones before the run starts).
fn speed_json(st: &AutoState) -> Value {
    let total_b = st.plan.as_ref().map(|p| p.original_b).unwrap_or(0.0);
    let stages: Vec<_> = st.stages.iter().map(|s| s.plan.clone()).collect();
    let dev = super::auto_speed::deviations(&st.spec, &stages, total_b);
    json!({"preset": st.spec.preset, "recipe": dev.is_empty(), "deviations": dev, "max_hours": st.spec.max_hours})
}

/// The estimate made before the run next to what the steps took ("estimated vs actual", to tune the next estimate): per finished step the estimated range, the minutes it took
/// and whether that was `within` the range, `faster` or `slower`; the totals only when the run is over. None for a run saved before the estimate existed.
fn estimate_json(st: &AutoState, active_seconds: i64) -> Value {
    let Some(t) = st.plan.as_ref().and_then(|p| p.time.as_ref()) else { return Value::Null };
    let rows: Vec<Value> = st.steps.iter().filter_map(|s| {
        let est = t.steps.iter().find(|x| x.id == s.id)?;
        let actual = if s.status == "done" { secs_between(&s.started, &s.finished).map(|n| n as f64 / 60.0) } else { None };
        let verdict = actual.map(|a| if a < est.minutes.0 { "faster" } else if a > est.minutes.1 { "slower" } else { "within" });
        Some(json!({"id": s.id, "label": est.label, "estimated_minutes": [est.minutes.0, est.minutes.1], "actual_minutes": actual.map(r1), "basis": est.basis, "result": verdict}))
    }).collect();
    let over = matches!(st.status.as_str(), "done");
    json!({"total_h": [t.total_h.0, t.total_h.1], "actual_h": if over { json!(r1(active_seconds as f64 / 3600.0)) } else { Value::Null }, "basis": t.basis,
           "history_steps": t.history_steps, "step_count": t.step_count, "steps": rows})
}

/// The ways the chosen cut goes beyond what Auto-cut does by itself (with the numbers of its check before healing when there are any): empty for a cut inside the rules.
fn guardrails(st: &AutoState) -> Vec<String> {
    let (Some(ins), Some(c)) = (st.inspect.as_ref(), st.chosen.as_ref()) else { return Vec::new() };
    let mut out = super::auto_plan::beyond_guardrails(ins, c.blocks, c.intermediate);
    if out.is_empty() {
        return out;
    }
    let win = st.think_gate.as_ref().and_then(|g| g["rows"].as_array().and_then(|r| r.iter().find(|x| x["row"] == g["winner"]).cloned()));
    if let Some(m) = win.as_ref().map(|w| &w["check"]["main"]).filter(|m| m.is_object()) {
        out.push(format!("Before healing this cut finished its reasoning in {} of its answers, looped in {} (the original: {}).",
                         pct(&m["closed_rate"]), pct(&m["loop_rate"]), st.baseline.as_ref().map(|b| pct(&b["main"]["closed_rate"])).unwrap_or_else(|| "-".into())));
    }
    out
}

fn pct(v: &Value) -> String {
    v.as_f64().map(|x| format!("{:.0} %", x * 100.0)).unwrap_or_else(|| "-".into())
}

fn cell(v: &Value, key: &str) -> String {
    match v[key].as_f64() {
        Some(x) if key == "median_think_tokens" => format!("{x:.0}"),
        Some(x) => format!("{x:.2}"),
        None => "-".into(),
    }
}

fn row(label: &str, verdict: &str, main: &Value) -> String {
    let mut s = format!("| {label} | {verdict} |");
    for (k, _) in METRICS {
        s.push_str(&format!(" {} |", cell(main, k)));
    }
    s.push('\n');
    s
}

fn range(v: &Value) -> String {
    match (v[0].as_f64(), v[1].as_f64()) {
        (Some(a), Some(b)) => format!("{a:.1} to {b:.1}"),
        _ => "-".into(),
    }
}

fn num(v: &Value, digits: usize) -> String {
    v.as_f64().map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "-".into())
}

/// The section of a stage that healed in rounds: one row per round, the checks of the merged model, why it stopped, the tokens it did not need, and how the check of the model as it was
/// trained compares with the check of the merged model (docs/HEAL_ROUNDS.md; experimental).
fn rounds_markdown(st: &Value) -> String {
    let r = &st["rounds"];
    let mut s = format!("\n## Healing in rounds, stage {} (experimental)\n\n", st["index"]);
    let reason = match r["stop"]["reason"].as_str().unwrap_or("") {
        "pass" => "a round passed the thinking check",
        "plateau" => "neither the thinking nor the loss improved any more",
        "cap" => "the most tokens were trained",
        "time" => "the time limit left no room for another round",
        _ => "it ended",
    };
    s.push_str(&format!("It stopped after round {} ({} M tokens of at most {} M): {reason}. ", r["stop"]["round"], num(&r["stop"]["tokens_m"], 0), num(&r["cap_m"], 0)));
    if let Some(saved) = st["rounds_saved_m"].as_f64().filter(|x| *x > 0.0) {
        s.push_str(&format!("That is {saved:.0} M tokens less than the budget. "));
    }
    s.push_str("\n\n| Round | Tokens so far (M) | Finishes reasoning | Right answers | Loops | Score | Loss on your text | Decision | Minutes |\n|---|---|---|---|---|---|---|---|---|\n");
    for x in r["rows"].as_array().cloned().unwrap_or_default() {
        let m = &x["check"]["main"];
        let (c, a, l, sc) = if m.is_object() { (pct(&m["closed_rate"]), pct(&m["accuracy"]), pct(&m["loop_rate"]), num(&x["check"]["score"], 2)) } else { ("-".into(), "-".into(), "-".into(), "-".into()) };
        s.push_str(&format!("| {} | {} | {c} | {a} | {l} | {sc} | {} | {} | {} |\n", x["n"], num(&x["tokens_m"], 1), num(&x["calib_loss"], 3), x["decision"].as_str().unwrap_or("-"), x["seconds"].as_f64().map(|t| format!("{:.0}", t / 60.0)).unwrap_or_else(|| "-".into())));
    }
    for m in r["merged_checks"].as_array().cloned().unwrap_or_default() {
        s.push_str(&format!("\nThe merged model after round {} did not pass ({}); healing went on and then needed two passed checks in a row.\n", m["after_round"], m["failed"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default()));
    }
    if r["final"].is_object() {
        let f = &r["final"];
        s.push_str(&format!("\nThe merged model's check ({}): {}. Finishes reasoning {}, right answers {}, loops {}.", f["mode"].as_str().unwrap_or("-"), f["verdict"].as_str().unwrap_or("-"), pct(&f["main"]["closed_rate"]), pct(&f["main"]["accuracy"]), pct(&f["main"]["loop_rate"])));
        // the check during healing looked at base + adapter, the final one at the merged weights: the same mode, side by side
        let last = r["rows"].as_array().and_then(|a| a.iter().rev().find(|x| x["check"]["main"].is_object() && x["check"]["mode"] == f["mode"]).cloned());
        if let Some(x) = last {
            s.push_str(&format!(" The last round check in the same mode (the model as trained, base + adapter): finishes reasoning {}, right answers {}, loops {}.", pct(&x["check"]["main"]["closed_rate"]), pct(&x["check"]["main"]["accuracy"]), pct(&x["check"]["main"]["loop_rate"])));
        }
        s.push('\n');
    }
    s.push_str("\nExperimental: how long the loads and the checks of a round really take, whether the checks during healing agree with the check of the merged model, and whether healing in rounds ends as good as one long heal were not measured.\n");
    s
}

/// The report as a page of Markdown.
pub fn markdown(r: &Value) -> String {
    let mut s = String::new();
    let status = r["status"].as_str().unwrap_or("");
    let headline = match status {
        "done" => "Finished",
        "gates_failed" => "Stopped: the thinking checks failed",
        "failed" => "Failed",
        "stopped" => "Stopped by you",
        _ => "Running",
    };
    s.push_str(&format!("# Auto-cut report: {} to about {} B\n\n**{headline}.** ", r["model"].as_str().unwrap_or("model"), num(&r["target_b"], 1)));
    if let Some(e) = r["error"].as_str() {
        s.push_str(e);
    }
    s.push_str("\n\n");
    s.push_str(&format!("- Original: {} B. Result: {} B ({} blocks removed, MLP width {}).\n", num(&r["original_b"], 1), num(&r["actual_b"], 2), r["blocks_removed"].as_u64().unwrap_or(0), r["mlp_width"].as_u64().map(|x| x.to_string()).unwrap_or_else(|| "unchanged".into())));
    s.push_str(&format!("- Keep thinking: {}. Stages: {}. Retries: {}. Time (the steps added up): {} minutes.\n", if r["keep_thinking"] == true { "yes" } else { "no" }, r["stages"].as_array().map(|a| a.len()).unwrap_or(0), r["retries"].as_u64().unwrap_or(0), num(&r["active_minutes"], 1)));
    if let Some(d) = r["disk"].as_object() {
        s.push_str(&format!("- Disk (planned, not measured): peak {} GB at \"{}\", {} GB left at the end.\n", num(&d["peak_gb"], 1), d["peak_step"].as_str().unwrap_or(""), num(&d["final_gb"], 1)));
    }
    s.push_str("\n## Stages\n\n| Stage | Size (B) | Cut (%) | Blocks cut | MLP width | Tokens (M) | KL first | KL last | Loss before | Loss after | Teacher loss | Tries | Result |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for st in r["stages"].as_array().cloned().unwrap_or_default() {
        let result = if st["passed"] == true { if st["backed_off"] == true { "passed after backing off" } else if st["retries"].as_u64().unwrap_or(0) > 0 { "passed on the second heal" } else { "passed" } } else { "not passed" };
        s.push_str(&format!("| {} of {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n", st["index"], st["of"], num(&st["params_b"], 2), num(&st["cut_pct"], 1), st["blocks_cut"], st["mlp_width"].as_u64().map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
                             st["tokens_m"], num(&st["kl_first"], 3), num(&st["kl_last"], 3), num(&st["loss_before"], 3), num(&st["loss_after"], 3), num(&st["teacher_loss"], 3), st["attempts"].as_array().map(|a| a.len()).unwrap_or(0), result));
    }
    for st in r["stages"].as_array().cloned().unwrap_or_default() {
        if let Some(l) = st["cut_loss"].as_f64() {
            s.push_str(&format!("\nStage {}: the cut's text-prediction error {l:.3} was measured in Try first (the new model was not loaded again to measure it).\n", st["index"]));
        }
    }
    if let Some(rows) = r["think_gate"]["rows"].as_array().filter(|a| !a.is_empty()) {
        s.push_str("\n## Before healing\n\nThe best cuts of Try first, checked for thinking before anything was healed (the check is an estimate of a model that has not been healed).\n\n| Cut | Loss | Finishes reasoning | Loops | Right answers | Chosen |\n|---|---|---|---|---|---|\n");
        for x in rows {
            let m = &x["check"]["main"];
            s.push_str(&format!("| {} blocks, MLP {} ({} B) | {} | {} | {} | {} | {} |\n", x["blocks"], x["intermediate"].as_u64().map(|n| n.to_string()).unwrap_or_else(|| "unchanged".into()), num(&x["params_b"], 2), num(&x["loss"], 3),
                                 pct(&m["closed_rate"]), pct(&m["loop_rate"]), pct(&m["accuracy"]), if x["row"] == r["think_gate"]["winner"] { "yes" } else { "" }));
        }
    }
    for st in r["stages"].as_array().cloned().unwrap_or_default().iter().filter(|st| st["rounds"].is_object()) {
        s.push_str(&rounds_markdown(st));
    }
    if let Some(g) = r["guardrails"].as_array().filter(|g| !g.is_empty()) {
        s.push_str("\n**This cut is beyond what Auto-cut does by itself.**\n\n");
        for x in g.iter().filter_map(|x| x.as_str()) {
            s.push_str(&format!("- {x}\n"));
        }
    }
    s.push_str("\n## Thinking check\n\n| Model | Verdict |");
    for (_, label) in METRICS {
        s.push_str(&format!(" {label} |"));
    }
    s.push_str("\n|---|---|");
    for _ in METRICS {
        s.push_str("---|");
    }
    s.push('\n');
    if r["baseline"].is_object() {
        s.push_str(&row("original", "-", &r["baseline"]));
    }
    for st in r["stages"].as_array().cloned().unwrap_or_default() {
        let n = st["index"].as_u64().unwrap_or(0);
        if st["cut_check"].is_object() {
            s.push_str(&row(&format!("stage {n}: cut, not healed"), st["cut_check"]["verdict"].as_str().unwrap_or("-"), &st["cut_check"]["main"]));
        }
        for a in st["attempts"].as_array().cloned().unwrap_or_default() {
            let kind = match a["kind"].as_str().unwrap_or("") {
                "heal_again" => "healed again",
                "back_off" => "healed after backing off",
                _ => "healed",
            };
            s.push_str(&row(&format!("stage {n}: {kind}"), a["verdict"].as_str().unwrap_or("-"), &a["main"]));
        }
    }
    if r["gguf_check"].is_object() {
        s.push_str(&row("GGUF (Q4_K_M)", r["gguf_check"]["verdict"].as_str().unwrap_or("-"), &r["gguf_check"]["main"]));
    }
    if let Some(f) = r["failed_gates"].as_array().filter(|f| !f.is_empty()) {
        s.push_str(&format!("\nGates missed: {}.\n", f.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")));
    }
    if let Some(g) = r["export"].as_array().filter(|g| !g.is_empty()) {
        s.push_str("\n## Files\n\n");
        for f in g {
            s.push_str(&format!("- {}: `{}`\n", f["quant"].as_str().unwrap_or(""), f["path"].as_str().unwrap_or("")));
        }
    }
    if r["compare"].is_object() {
        let c = &r["compare"];
        s.push_str(&format!("\n## Measured file against the plain sizes{}\n\n", r["target_gb"].as_f64().map(|t| format!(" (about {t} GB asked for)")).unwrap_or_default()));
        if let Some(why) = c["skipped"].as_str() {
            s.push_str(&format!("Not compared: {why}\n"));
        } else {
            s.push_str("| File | GB | Mean KLD | Same top answer | Promoted |\n|---|---|---|---|---|\n");
            for row in c["rows"].as_array().cloned().unwrap_or_default() {
                let promoted = c["promotion"].as_array().and_then(|a| a.iter().find(|p| p["id"] == row["id"])).map(|p| if p["promoted"] == true { "yes" } else { "no" }).unwrap_or("");
                let kld = match (row["mean_kld"].as_f64(), row["kld_unc"].as_f64()) {
                    (Some(k), Some(u)) => format!("{k:.4} ± {u:.4}"),
                    (Some(k), None) => format!("{k:.4}"),
                    _ => "-".to_string(),
                };
                let gb = row["size"].as_f64().map(|b| format!("{:.2}", b / 1e9)).unwrap_or_else(|| "-".into());
                let top = row["same_top"].as_f64().map(|t| format!("{t:.1} %")).unwrap_or_else(|| "-".into());
                s.push_str(&format!("| {} | {gb} | {kld} | {top} | {promoted} |\n", row["label"].as_str().unwrap_or("")));
            }
            for line in c["promotion"].as_array().cloned().unwrap_or_default() {
                s.push_str(&format!("\n- {}\n", line["reason"].as_str().unwrap_or("")));
            }
            s.push_str(if c["promoted"] == true { "\nThe measured file was published with the others.\n" } else { "\nThe measured file is kept in the exports but was not published: it is not better than the plain file of its size beyond the measurement noise.\n" });
            s.push_str("\nThe predicted error of the fit is a measurement of the weights with a diagonal approximation of the importance matrix, on a sample of the rows, with every layer counted the same; only this comparison (KL divergence) says how the files really differ.\n");
        }
    }
    if r["studio"].is_object() {
        let p = &r["studio"];
        if p["published"] == true {
            s.push_str(&format!("\nUnsloth Studio: published to `{}` (inside Studio: `{}`).\n", p["host_path"].as_str().unwrap_or(""), p["container_path"].as_str().unwrap_or("")));
        } else {
            s.push_str(&format!("\nUnsloth Studio: not published ({}).\n", p["why"].as_str().unwrap_or("")));
        }
    }
    if let Some(e) = r["estimate"].as_object().filter(|e| e["steps"].as_array().is_some_and(|a| !a.is_empty())) {
        s.push_str("\n## Estimated and actual time\n\nThe estimate shown before the start, next to what the steps took (minutes; only finished steps have an actual time).\n\n");
        let hours = |v: &Value| v.as_array().map(|a| format!("{} to {}", num(&a[0], 1), num(&a[1], 1))).unwrap_or_else(|| "-".into());
        s.push_str(&format!("Total: estimated {} hours, actual {} hours. {}\n\n", hours(&e["total_h"]), if e["actual_h"].is_null() { "-".to_string() } else { num(&e["actual_h"], 1) }, e["basis"].as_str().unwrap_or("")));
        s.push_str("| Step | Estimated | Actual | Result | From |\n|---|---|---|---|---|\n");
        for x in e["steps"].as_array().cloned().unwrap_or_default() {
            s.push_str(&format!("| {} | {} | {} | {} | {} |\n", x["label"].as_str().unwrap_or(""), x["estimated_minutes"].as_array().map(|a| format!("{} to {}", num(&a[0], 1), num(&a[1], 1))).unwrap_or_else(|| "-".into()),
                                 if x["actual_minutes"].is_null() { "-".to_string() } else { num(&x["actual_minutes"], 1) }, x["result"].as_str().unwrap_or("-"), if x["basis"] == "your_history" { "your history" } else { "formula" }));
        }
    }
    if let Some(stages) = r["timings"]["summary"]["stages"].as_object().filter(|m| !m.is_empty()) {
        s.push_str("\n## Where the time went\n\nFrom the timing lines of the job logs; the share is of the time of all the jobs of this run.\n\n| Stage | Seconds | Times | Share |\n|---|---|---|---|\n");
        let mut rows: Vec<(&String, &Value)> = stages.iter().collect();
        rows.sort_by(|a, b| b.1["seconds"].as_f64().partial_cmp(&a.1["seconds"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
        for (name, v) in rows {
            s.push_str(&format!("| {name} | {} | {} | {} |\n", num(&v["seconds"], 0), v["count"], pct(&v["share"])));
        }
        for a in r["timings"]["advice"].as_array().cloned().unwrap_or_default() {
            s.push_str(&format!("\n- {}{}\n", if a["worth"] == true { "Worth doing: " } else { "" }, a["text"].as_str().unwrap_or("")));
        }
    }
    if let Some(sp) = r["speed"].as_object() {
        let preset = sp["preset"].as_str().map(|p| format!(" (setting: {p})")).unwrap_or_default();
        s.push_str(&format!("\n## Speed settings{preset}\n\n"));
        let devs = sp["deviations"].as_array().cloned().unwrap_or_default();
        if devs.is_empty() {
            s.push_str("The recommended recipe: no setting was lowered to save time.\n");
        } else {
            s.push_str("These values differ from the recommended recipe. Each one trades quality risk for time:\n\n| Setting | Used | Recipe | Risk |\n|---|---|---|---|\n");
            for d in devs {
                s.push_str(&format!("| {} | {} | {} | {} |\n", d["label"].as_str().unwrap_or(""), d["value"].as_str().unwrap_or(""), d["recipe"].as_str().unwrap_or(""), d["risk"].as_str().unwrap_or("")));
            }
        }
        if let Some(b) = r["budget"].as_object().filter(|b| b["dropped"].as_array().is_some_and(|d| !d.is_empty())) {
            s.push_str(&format!("\nTime limit of {} hours: the estimate went from {} h to {} h by giving up (least harmful first):\n\n", num(&b["max_hours"], 0), range(&b["asked_h"]), range(&b["total_h"])));
            for d in b["dropped"].as_array().cloned().unwrap_or_default() {
                s.push_str(&format!("- {}: {} instead of {} (saves {} h). {}\n", d["label"].as_str().unwrap_or(""), d["to"].as_str().unwrap_or(""), d["from"].as_str().unwrap_or(""), range(&d["saves_h"]), d["risk"].as_str().unwrap_or("")));
            }
        }
        if let Some(e) = r["estimate"].as_object().filter(|e| !e["total_h"].is_null()) {
            let actual = e["actual_h"].as_f64().map(|a| format!("{a:.1} h")).unwrap_or_else(|| "not finished".into());
            s.push_str(&format!("\nEstimated {} h; took {actual}.\n", range(&e["total_h"])));
        }
    }
    if let Some(n) = r["notes"].as_array().filter(|n| !n.is_empty()) {
        s.push_str("\n## Notes\n\n");
        for x in n.iter().filter_map(|x| x.as_str()) {
            s.push_str(&format!("- {x}\n"));
        }
    }
    s.push_str("\n## Steps\n\n| Step | Status | Seconds | Note |\n|---|---|---|---|\n");
    for x in r["steps"].as_array().cloned().unwrap_or_default() {
        s.push_str(&format!("| {} | {} | {} | {} |\n", x["label"].as_str().unwrap_or(""), x["status"].as_str().unwrap_or(""), x["seconds"].as_i64().map(|n| n.to_string()).unwrap_or_else(|| "-".into()), x["note"].as_str().unwrap_or("").replace('|', "/")));
    }
    s.push_str("\nThe numbers come from a thinking check of 32 questions per model (16 in Quick); they show a trend, not a benchmark.\n");
    s
}

/// Writes `auto_report.json` and `auto_report.md` into the run's folder and returns the small pointer the state keeps (`{"json", "md"}`: file names, no times).
pub fn write(dir: &Path, st: &AutoState) -> Result<Value> {
    let mut r = build(st);
    add_timings(&mut r, st);
    write_json(&dir.join(JSON_FILE), &r)?;
    let md = dir.join(MD_FILE);
    let tmp = md.with_extension("tmp");
    std::fs::write(&tmp, markdown(&r))?;
    std::fs::rename(tmp, md)?;
    Ok(json!({"json": JSON_FILE, "md": MD_FILE}))
}
