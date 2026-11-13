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

/// The trained side of a test: a LoRA adapter (on its own base model when that is not the original) or a whole fully trained model.
#[derive(Debug, Clone)]
pub enum Side {
    /// `base`: the folder the adapter was trained on when that is NOT the original model (a run on a slimmed or grown model)
    Adapter { dir: PathBuf, base: Option<String> },
    Full(PathBuf),
}

/// Two models to compare: the original (a Hugging Face name or folder) and the trained side.
#[derive(Debug, Clone)]
pub struct Pair {
    pub original: String,
    pub family: String,
    pub trained: Side,
    /// load the (original, trained) side in 4-bit so a big model fits the card
    pub four_bit: (bool, bool),
    /// what the trained side is compared with: "base" (the model the run started from) or "origin" (the model a slimmed / grown base was made from)
    pub against: &'static str,
}

fn side_names(four: bool) -> &'static str {
    if four { "4-bit" } else { "16-bit" }
}

/// Asks the worker for answers. `which` = ["original", "trained"] or one of them. Returns ({"original": {id: text}, "trained": {...}}, {"original": "16-bit", "trained": ...}).
fn answer_pair(pair: &Pair, qs: &[Question], which: &[&str], max_new: u32, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<(Value, Value)> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let out = std::env::temp_dir().join(format!("ndforge-answers-{}-{}.json", std::process::id(), chrono::Utc::now().timestamp_micros()));
    let questions: Vec<Value> = qs.iter().map(|q| json!({"id": q.id, "messages": q.messages})).collect();
    let mut job = json!({"job": "evaluate", "model": {"repo": pair.original, "family": pair.family}, "questions": questions, "which": which, "max_new": max_new, "out": out});
    match &pair.trained {
        Side::Full(dir) => job["full_model"] = json!(dir),
        Side::Adapter { dir, base } => {
            job["adapter"] = json!(dir);
            if let Some(b) = base {
                job["trained_base"] = json!(b);
            }
        }
    }
    if pair.four_bit.0 || pair.four_bit.1 {
        job["load_4bit"] = json!({"original": pair.four_bit.0, "trained": pair.four_bit.1});
    }
    let result = run_job(&py, &worker_dir(), &job, kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    let answers = read_json(&out).ok_or_else(|| Error::new("The worker did not write any answers."))?;
    let _ = std::fs::remove_file(out);
    let loaded = if result["loaded"].is_object() { result["loaded"].clone() } else { json!({"original": side_names(pair.four_bit.0), "trained": side_names(pair.four_bit.1)}) };
    Ok((answers, loaded))
}

/// What a run's folder says about the model it was trained on: the folder of a slimmed / grown base and the model that base was made from.
fn reshaped_base(repo: &str) -> Option<Value> {
    let meta = read_json(&std::path::Path::new(repo).join("ndforge_reshape.json"))?;
    meta["origin"]["repo"].is_string().then_some(meta)
}

/// Parameters (billions) of a model the program knows: a made model's own figure, else the model list.
fn params_b_of(repo: &str) -> Option<f64> {
    reshaped_base(repo).and_then(|m| m["params"].as_f64()).map(|p| p / 1e9)
        .or_else(|| crate::train::catalog::models().into_iter().find(|m| m.repo == repo).map(|m| m.params_b))
}

/// A big model is loaded in 4-bit when its 16-bit weights plus some room do not fit the card.
fn needs_4bit(params_b: Option<f64>, vram_gb: Option<f64>) -> bool {
    match (params_b, vram_gb) {
        (Some(b), Some(v)) => b * 2.0 + 3.0 > v * 0.9,
        _ => false,
    }
}

/// The two sides of a run's test. `against_origin`: for a run on a slimmed or grown base, compare with the model that base was made from instead of the
/// base itself (the adapter is then loaded on the made model, the original side is the real original).
pub fn pair_for_run(p: &Project, run: &str, against_origin: bool, vram_gb: Option<f64>) -> Result<Pair> {
    let rdir = runs_dir(p).join(run);
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of this run are missing."))?;
    let (adapter, full) = (rdir.join("adapter"), rdir.join("model"));
    let is_full = full.join("config.json").is_file();
    if !is_full && !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new("This run has no saved model to test (it was stopped before the first save?)."));
    }
    let repo = cfg["model"]["repo"].as_str().unwrap_or("").to_string();
    let family = cfg["model"]["family"].as_str().unwrap_or("generic").to_string();
    let base_4bit = needs_4bit(params_b_of(&repo), vram_gb);
    let origin = if against_origin { reshaped_base(&repo) } else { None };
    Ok(match origin {
        Some(meta) => {
            let origin_repo = meta["origin"]["repo"].as_str().unwrap_or("").to_string();
            let origin_4bit = needs_4bit(meta["origin"]["params"].as_f64().map(|x| x / 1e9), vram_gb);
            let trained = if is_full { Side::Full(full) } else { Side::Adapter { dir: adapter, base: Some(repo) } };
            Pair { original: origin_repo, family: meta["origin"]["family"].as_str().unwrap_or(&family).to_string(), trained, four_bit: (origin_4bit, base_4bit), against: "origin" }
        }
        None => {
            let trained = if is_full { Side::Full(full) } else { Side::Adapter { dir: adapter, base: None } };
            Pair { original: repo, family, trained, four_bit: (base_4bit, base_4bit), against: "base" }
        }
    })
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
    execute_against(p, run, per_category, judge_with, false, None, kill, on_event)
}

/// `execute`, optionally against the model a slimmed / grown base was made from (`against_origin`); `vram_gb` decides which sides load in 4-bit.
pub fn execute_against(p: &Project, run: &str, per_category: usize, judge_with: Option<&Teacher>, against_origin: bool, vram_gb: Option<f64>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let rdir = runs_dir(p).join(run);
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("That run does not exist."))?;
    let version = cfg["dataset"]["version"].as_u64().unwrap_or(0) as u32;
    let pair = pair_for_run(p, run, against_origin, vram_gb)?;
    execute_pair(p, &pair, version, run, &rdir.join("test.json"), per_category, judge_with, kill, on_event)
}

/// The test itself, for any pair of models: questions from dataset `version`, answers from both sides, scores and verdict, saved to `out_file`.
/// `label` names the tested thing in the result ("run" key): a run's name, or "model:<name>" for a made model.
pub fn execute_pair(p: &Project, pair: &Pair, version: u32, label: &str, out_file: &std::path::Path, per_category: usize, judge_with: Option<&Teacher>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let qs = build(p, version as u64, per_category.clamp(5, 100))?;
    on_event(json!({"kind": "log", "text": format!("{} questions", qs.len())}));
    let (answers, loaded) = answer_pair(pair, &qs, &["original", "trained"], 160, kill, on_event)?;
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
        "run": label, "created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(), "dataset_version": version, "n": qs.len(),
        "method": if judged > 0 { "teacher" } else { "overlap" }, "judged": judged, "against": pair.against, "loaded": loaded,
        "scores": {"original": summarize(&sc_o), "trained": summarize(&sc_t)}, "verdict": v, "questions": per,
    });
    std::fs::write(out_file, serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

/// The free question to both models. Returns {"original": text, "trained": text}.
pub fn ask(p: &Project, run: &str, system: Option<&str>, question: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    if question.trim().is_empty() {
        return Err(Error::new("Type a question first."));
    }
    ask_against(p, run, system, question, false, None, kill, on_event)
}

/// `ask`, optionally against the model a slimmed / grown base was made from.
pub fn ask_against(p: &Project, run: &str, system: Option<&str>, question: &str, against_origin: bool, vram_gb: Option<f64>, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    if question.trim().is_empty() {
        return Err(Error::new("Type a question first."));
    }
    let q = custom(&p.display_name(), system, question.trim());
    let pair = pair_for_run(p, run, against_origin, vram_gb)?;
    let (a, _) = answer_pair(&pair, &[q], &["original", "trained"], 300, kill, on_event)?;
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
    // slimmed / grown models tested on their own (original against the made model)
    if let Ok(rd) = std::fs::read_dir(p.folder.join("models")) {
        for e in rd.flatten() {
            let (Some(t), Some(name)) = (read_json(&e.path().join("test.json")), e.file_name().to_str().map(String::from)) else { continue };
            let state = read_json(&e.path().join("state.json")).unwrap_or(Value::Null);
            let mean = |side: &str, cat: &str| t["scores"][side][cat]["mean"].as_f64();
            rows.push(json!({
                "name": format!("model:{name}"), "model": state["spec"]["source_model_id"], "method": state["spec"]["mode"].as_str().map(|m| if m == "prune" { "slim" } else { "grow" }),
                "stages": Value::Null, "version": t["dataset_version"], "steps": Value::Null, "final_loss": Value::Null, "tested": t["created"], "grading": t["method"],
                "lore": mean("trained", "lore"), "lore_original": mean("original", "lore"), "rules": mean("trained", "rules"), "invented": mean("trained", "made_up"),
                "general": mean("trained", "general"), "headline": t["verdict"]["headline"], "n": t["n"],
            }));
        }
    }
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
