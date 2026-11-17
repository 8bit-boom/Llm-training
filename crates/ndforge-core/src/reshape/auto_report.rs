//! Auto-cut's report (docs/AUTO_CUT.md, "The report"): what a run did, in numbers, as `auto_report.json` and `auto_report.md` in the run's folder.
//!
//! Pure functions of the saved state (`AutoState`): the same state gives the same report (the times are the ones in the state), so the report can be made again at any
//! time, also for a run that was stopped. `write` is called when a run ends (done, gates_failed, failed); the window asks for `report` to show it.
use std::path::Path;

use serde_json::{json, Value};

use super::auto::{AutoState, StageState};
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

/// The report of a run as data.
pub fn build(st: &AutoState) -> Value {
    let plan = st.plan.as_ref();
    let original = st.original.as_ref();
    let last = st.stages.last();
    let actual_b = last.map(|s| s.plan.params_b);
    let steps: Vec<Value> = st.steps.iter().map(|s| json!({"id": s.id, "label": s.label, "status": s.status, "note": s.note, "seconds": secs_between(&s.started, &s.finished)})).collect();
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
    })
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

fn num(v: &Value, digits: usize) -> String {
    v.as_f64().map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "-".into())
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
    if r["studio"].is_object() {
        let p = &r["studio"];
        if p["published"] == true {
            s.push_str(&format!("\nUnsloth Studio: published to `{}` (inside Studio: `{}`).\n", p["host_path"].as_str().unwrap_or(""), p["container_path"].as_str().unwrap_or("")));
        } else {
            s.push_str(&format!("\nUnsloth Studio: not published ({}).\n", p["why"].as_str().unwrap_or("")));
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
    let r = build(st);
    write_json(&dir.join(JSON_FILE), &r)?;
    let md = dir.join(MD_FILE);
    let tmp = md.with_extension("tmp");
    std::fs::write(&tmp, markdown(&r))?;
    std::fs::rename(tmp, md)?;
    Ok(json!({"json": JSON_FILE, "md": MD_FILE}))
}
