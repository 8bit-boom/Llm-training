//! Running a test: the worker answers the questions with the original and the trained model, the scores are worked out here and saved as
//! `runs/<run>/test.json`. Also the free side-by-side chat, saved transcripts and the table that compares runs.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::questions::{build, custom, Question};
use super::score::{score_one, summarize, verdict};
use crate::dataset::builders::hash_hex;
use crate::envmgr::jobs::{run_job, WorkerEvent};
use crate::envmgr::{env_dir, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::teacher::{extract_json, Teacher};
use crate::train::runs::runs_dir;

fn read_json(path: &std::path::Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

/// Asks the worker for answers. `which` = ["original", "trained"] or one of them. Returns {"original": {id: text}, "trained": {...}}.
fn answer_with_worker(p: &Project, run: &str, qs: &[Question], which: &[&str], max_new: u32, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let rdir = runs_dir(p).join(run);
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of this run are missing."))?;
    let (adapter, full) = (rdir.join("adapter"), rdir.join("model"));
    let is_full = full.join("config.json").is_file();
    if !is_full && !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new("This run has no saved model to test (it was stopped before the first save?)."));
    }
    let out = std::env::temp_dir().join(format!("ndforge-answers-{}-{}.json", std::process::id(), chrono::Utc::now().timestamp_micros()));
    let questions: Vec<Value> = qs.iter().map(|q| json!({"id": q.id, "messages": q.messages})).collect();
    let mut job = json!({"job": "evaluate", "model": {"repo": cfg["model"]["repo"], "family": cfg["model"]["family"]}, "questions": questions, "which": which, "max_new": max_new, "out": out});
    if is_full { job["full_model"] = json!(full); } else { job["adapter"] = json!(adapter); }
    run_job(&py, &worker_dir(), &job, kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    let answers = read_json(&out).ok_or_else(|| Error::new("The worker did not write any answers."))?;
    let _ = std::fs::remove_file(out);
    Ok(answers)
}

/// A teacher's verdict on an answer against the reference: 1.0 correct, 0.5 partly, 0.0 wrong.
fn judge(t: &Teacher, q: &Question, answer: &str) -> Option<f64> {
    let user = format!("QUESTION:\n{}\n\nREFERENCE ANSWER (from the world's lore):\n{}\n\nCANDIDATE ANSWER:\n{}\n\nDoes the CANDIDATE ANSWER agree with the REFERENCE ANSWER on the facts that matter? \"correct\": yes; \"partly\": some facts right, some missing or wrong; \"wrong\": no, or it invents facts.\nOutput JSON only: {{\"verdict\":\"correct\"|\"partly\"|\"wrong\"}}", q.question, q.reference.as_deref().unwrap_or(""), answer);
    let raw = t.ask("You are a strict grader of answers about a tabletop role-playing world. Answer with JSON only.", &user, 0.0, 200).ok()?;
    match extract_json(&raw)?["verdict"].as_str()? {
        "correct" => Some(1.0),
        "partly" => Some(0.5),
        "wrong" => Some(0.0),
        _ => None,
    }
}

/// Tests a run: answers (original and trained), scores, verdict; saved in the run's folder. With a teacher, lore and rules answers are graded by it
/// (falling back to word overlap for any it cannot grade).
pub fn execute(p: &Project, run: &str, per_category: usize, judge_with: Option<&Teacher>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let rdir = runs_dir(p).join(run);
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("That run does not exist."))?;
    let version = cfg["dataset"]["version"].as_u64().unwrap_or(0);
    let qs = build(p, version, per_category.clamp(5, 100))?;
    on_event(json!({"kind": "log", "text": format!("{} questions", qs.len())}));
    let answers = answer_with_worker(p, run, &qs, &["original", "trained"], 160, kill, on_event)?;
    let get = |side: &str, id: &str| answers[side][id].as_str().unwrap_or("").to_string();
    let (mut sc_o, mut sc_t, mut rows, mut per, mut judged) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0usize);
    for (i, q) in qs.iter().enumerate() {
        if judge_with.is_some() && matches!(q.category, "lore" | "rules") {
            on_event(json!({"kind": "progress", "step": "judge", "value": i as f64 / qs.len() as f64, "text": format!("Grading answers: {} of {}", i + 1, qs.len())}));
        }
        let (ao, at) = (get("original", &q.id), get("trained", &q.id));
        let grade = |a: &str| -> f64 {
            if let (Some(t), true) = (judge_with, matches!(q.category, "lore" | "rules")) {
                if let Some(s) = judge(t, q, a) {
                    return s;
                }
            }
            score_one(q.category, a, q.reference.as_deref())
        };
        let (so, st) = (grade(&ao), grade(&at));
        if judge_with.is_some() && matches!(q.category, "lore" | "rules") {
            judged += 1;
        }
        sc_o.push((q.category.to_string(), so));
        sc_t.push((q.category.to_string(), st));
        rows.push((q.category.to_string(), so, st));
        per.push(json!({"id": q.id, "category": q.category, "question": q.question, "reference": q.reference, "original": ao, "trained": at, "score_original": so, "score_trained": st}));
    }
    let v = verdict(&rows);
    let result = json!({
        "run": run, "created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), "dataset_version": version, "n": qs.len(),
        "method": if judged > 0 { "teacher" } else { "overlap" }, "judged": judged,
        "scores": {"original": summarize(&sc_o), "trained": summarize(&sc_t)}, "verdict": v, "questions": per,
    });
    std::fs::write(rdir.join("test.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

/// The free question to both models. Returns {"original": text, "trained": text}.
pub fn ask(p: &Project, run: &str, system: Option<&str>, question: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    if question.trim().is_empty() {
        return Err(Error::new("Type a question first."));
    }
    let q = custom(&p.display_name(), system, question.trim());
    let a = answer_with_worker(p, run, &[q], &["original", "trained"], 300, kill, on_event)?;
    Ok(json!({"original": a["original"]["custom"], "trained": a["trained"]["custom"]}))
}

pub fn transcripts_dir(p: &Project) -> PathBuf {
    p.folder.join("transcripts")
}

/// Saves a chat as a Markdown file. `items` = [{question, original, trained}]. Returns the file.
pub fn save_transcript(p: &Project, run: &str, items: &[Value]) -> Result<PathBuf> {
    if items.is_empty() {
        return Err(Error::new("There is nothing to save yet. Ask a question first."));
    }
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let mut md = format!("# {} · run {run}\n\nSaved {stamp}. Left: the original model. Right: the trained model.\n", p.display_name());
    for it in items {
        md += &format!("\n## {}\n\n**Original**\n\n{}\n\n**Trained**\n\n{}\n", it["question"].as_str().unwrap_or(""), it["original"].as_str().unwrap_or(""), it["trained"].as_str().unwrap_or(""));
    }
    let f = transcripts_dir(p).join(format!("{run}-{stamp}-{}.md", &hash_hex(&[&md])[..4]));
    std::fs::create_dir_all(transcripts_dir(p))?;
    std::fs::write(&f, md)?;
    Ok(f)
}

/// One line per tested run for the comparison table, best lore score first.
pub fn compare(p: &Project) -> Vec<Value> {
    let mut rows: Vec<Value> = crate::train::runs::list_runs(p).into_iter().filter_map(|r| {
        let name = r["name"].as_str()?.to_string();
        let dir = runs_dir(p).join(&name);
        let t = read_json(&dir.join("test.json"))?;
        let cfg = read_json(&dir.join("config.json")).unwrap_or(Value::Null);
        let summary = read_json(&dir.join("summary.json")).unwrap_or(Value::Null);
        let mean = |side: &str, cat: &str| t["scores"][side][cat]["mean"].as_f64();
        Some(json!({
            "name": name, "model": cfg["model"]["repo"], "method": cfg["method"], "stages": if cfg["stages"].is_array() { "world book first" } else { "together" },
            "version": t["dataset_version"], "steps": summary["steps"], "final_loss": summary["final_loss"], "tested": t["created"], "grading": t["method"],
            "lore": mean("trained", "lore"), "lore_original": mean("original", "lore"), "rules": mean("trained", "rules"), "invented": mean("trained", "made_up"),
            "general": mean("trained", "general"), "headline": t["verdict"]["headline"], "n": t["n"],
        }))
    }).collect();
    rows.sort_by(|a, b| b["lore"].as_f64().unwrap_or(-1.0).partial_cmp(&a["lore"].as_f64().unwrap_or(-1.0)).unwrap_or(std::cmp::Ordering::Equal));
    rows
}

/// Runs with a saved model, newest first, each with whether it has been tested.
pub fn testable(p: &Project) -> Vec<Value> {
    crate::train::runs::list_runs(p).into_iter().filter(|r| r["has_adapter"] == true || r["has_model"] == true).map(|mut r| {
        let t = r["name"].as_str().and_then(|n| read_test(p, n));
        r["tested"] = json!(t.is_some());
        r["headline"] = t.map(|t| t["verdict"]["headline"].clone()).unwrap_or(Value::Null);
        r
    }).collect()
}

pub fn read_test(p: &Project, run: &str) -> Option<Value> {
    if run.contains(['/', '\\']) || run.contains("..") {
        return None;
    }
    read_json(&runs_dir(p).join(run).join("test.json"))
}
