//! A training run in Unsloth: send the data, follow the job, bring the adapter back (see `docs/UNSLOTH_DESKTOP.md`).
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::chatml::chatml_rows;
use super::client::{scrub, UnslothClient};
use crate::project::Project;
use crate::train::runs::{finish, runs_dir, set_run_state, Recorder};

use crate::error::{Error, Result};
use crate::train::catalog::ModelEntry;
use crate::train::plan::{Hyper, Method};

/// The Unsloth the user entered. The key is kept in `secrets` under `unsloth:<url>`.
#[derive(Debug, Clone)]
pub struct UnslothTarget {
    pub name: String,
    pub url: String,
    pub key: String,
}

/// The `server` value the Train page uses for Unsloth (a training server is named by its address).
pub const SERVER_ID: &str = "unsloth";

/// A name Unsloth accepts as the job's suffix: letters, digits, `_` and `-`, at most 40 characters.
pub fn suffix_of(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' }).collect();
    s.chars().take(40).collect::<String>().trim_matches('-').to_string()
}

/// The body of `POST /v1/fine_tuning/jobs` (the contract document lists every field). Every key is one Unsloth knows: it refuses others.
/// `load_in_4bit` is always sent: Unsloth's own default is `true`, which would turn every LoRA run into QLoRA.
pub fn job_body(m: &ModelEntry, method: Method, hyper: &Hyper, suffix: &str, train_id: &str, eval_id: Option<&str>) -> Result<Value> {
    if method == Method::Full {
        return Err(Error::new("Full fine-tuning runs on this computer only. Choose LoRA or QLoRA, or train on this computer."));
    }
    if hyper.stages == "book_first" && hyper.max_steps.is_none() {
        return Err(Error::new("Two stages (world book first) run on this computer or the TrueNAS server only. Choose \"mixed\" or train on this computer."));
    }
    let mut b = json!({
        "model": m.repo,
        "training_file": train_id,
        "hyperparameters": {"n_epochs": if hyper.max_steps.is_some() { 1 } else { (hyper.epochs.round() as u32).max(1) }, "batch_size": hyper.batch, "learning_rate_multiplier": hyper.lr / 2e-4},
        "method": {"type": "supervised"},
        "suffix": suffix_of(suffix),
        "seed": hyper.seed,
        "training_type": "LoRA/QLoRA",
        "load_in_4bit": method == Method::Qlora,
        "max_seq_length": hyper.max_seq,
        "gradient_accumulation_steps": hyper.grad_accum,
        "warmup_steps": hyper.warmup,
        "max_steps": hyper.max_steps.unwrap_or(0),
        "save_steps": hyper.save_steps,
        "packing": false,
        "train_on_completions": true,
        "gradient_checkpointing": "unsloth",
        "lora_r": hyper.rank,
        "lora_alpha": hyper.alpha,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "finetune_vision_layers": false,
        "finetune_language_layers": true,
        "finetune_attention_modules": true,
        "finetune_mlp_modules": true,
    });
    if let Some(e) = eval_id {
        b["validation_file"] = json!(e);
        b["eval_steps"] = json!(0.1);
    }
    Ok(b)
}

/// How many polls may fail in a row before the run is marked interrupted (Unsloth keeps training meanwhile).
const MAX_POLL_FAILURES: u32 = 180;
/// How many pages of 100 events are read back when a reattached run missed more than one page.
const MAX_EVENT_PAGES: u32 = 20;

fn stage(on_event: &mut dyn FnMut(Value), text: &str) {
    on_event(json!({"kind": "stage", "text": text}));
}

/// The chart event for one of Unsloth's "metrics" events.
fn metric_of(e: &Value) -> Value {
    let d = &e["data"];
    json!({"kind": "metric", "step": d["step"], "total_steps": d["total_steps"], "loss": d["train_loss"], "lr": d["learning_rate"], "eval_loss": d["valid_loss"], "epoch": d["epoch"], "grad_norm": d["grad_norm"]})
}

/// "Waiting for the GPU: position 2 in the queue" -> "Waiting for Unsloth's graphics card (position 2)".
fn queue_stage(message: &str) -> String {
    let pos: String = message.split("position").nth(1).map(|r| r.trim().chars().take_while(|c| c.is_ascii_digit()).collect()).unwrap_or_default();
    if pos.is_empty() { "Waiting for Unsloth's graphics card (it is busy with another job)".to_string() } else { format!("Waiting for Unsloth's graphics card (position {pos})") }
}

/// The events not seen yet, oldest first. Unsloth lists them newest first and `after` pages toward older ones, so a page without a known id is followed back
/// (a reattached run) until a known id or the start is reached.
fn new_events(client: &UnslothClient, id: &str, seen: &HashSet<String>) -> Result<Vec<Value>> {
    let (mut all, mut more) = client.events_page(id, 100, None)?;
    let mut pages = 1;
    while more && pages < MAX_EVENT_PAGES && !all.iter().any(|e| e["id"].as_str().is_some_and(|i| seen.contains(i))) {
        let Some(last) = all.last().and_then(|e| e["id"].as_str()).map(String::from) else { break };
        let (older, m) = client.events_page(id, 100, Some(&last))?;
        if older.is_empty() {
            break;
        }
        all.extend(older);
        more = m;
        pages += 1;
    }
    all.retain(|e| e["id"].as_str().is_some_and(|i| !seen.contains(i)));
    all.reverse();
    Ok(all)
}

/// Trains in Unsloth to the end and keeps the run's state like a local run. `poll` is the pause between asking Unsloth for news.
/// The adapter stays in Unsloth's run folder here (the copy back is `fetch_adapter`).
pub fn execute_unsloth(p: &Project, m: &ModelEntry, method: Method, hyper: &Hyper, name: &str, mut job: Value, target: &UnslothTarget, stop: Arc<AtomicBool>, _resume: bool, poll: Duration,
                       on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let dir = runs_dir(p).join(name);
    std::fs::create_dir_all(&dir)?;
    // the key is never written into the run folder
    job["unsloth"] = json!({"name": target.name, "url": target.url});
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&job)?)?;
    let prior: Value = std::fs::read(dir.join("state.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    set_run_state(&dir, json!({"name": name, "model_id": m.id, "model": m.name, "method": method, "status": "running", "server": target.name, "server_url": target.url, "backend": "unsloth",
                               "seq": hyper.max_seq, "batch": hyper.batch, "error": null, "created": prior["created"].as_str().unwrap_or(&chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string())}));
    let client = UnslothClient::new(&target.url, &target.key);
    let mut rec = Recorder::new(&dir)?;
    let _ = &stop;

    let outcome = (|| -> Result<Value> {
        stage(on_event, "Connecting to Unsloth");
        let probe = client.probe()?;
        if !probe.unsloth {
            return Err(Error::new(format!("Something else answers at {} (not Unsloth). Check the address in Training setup.", target.url)));
        }
        if probe.key_ok != Some(true) {
            return Err(Error::new("Unsloth refused the API key. Create a new one in Unsloth (Settings -> API) and enter it again in Training setup; Unsloth shows it only once."));
        }
        if !probe.fine_tuning {
            return Err(Error::new("This Unsloth has no training API for other programs: update it (Settings -> General -> Check for updates) or use Export for Unsloth Studio."));
        }
        if probe.chat_only == Some(true) && !probe.cpu_training {
            return Err(Error::new("Unsloth reports no usable graphics card for training. Check the card in Unsloth, or train on this computer."));
        }
        // the data: the same rows as the Studio export
        let version = job["dataset"]["version"].as_u64().unwrap_or(0) as u32;
        let (train, eval, _) = chatml_rows(p, version, true);
        if train.is_empty() {
            return Err(Error::new("This data version has no examples Unsloth can use."));
        }
        let (train_path, eval_path) = (dir.join("unsloth-train.jsonl"), dir.join("unsloth-eval.jsonl"));
        std::fs::write(&train_path, train.join("\n") + "\n")?;
        stage(on_event, "Sending the training data");
        let up = |text: &'static str| move |v: f64| json!({"kind": "progress", "step": "upload", "value": v, "text": text});
        let train_id = client.upload_jsonl("train.jsonl", &train_path, &mut |v| on_event(up("Sending the training data")(v)))?;
        let eval_id = if eval.is_empty() { None } else {
            std::fs::write(&eval_path, eval.join("\n") + "\n")?;
            Some(client.upload_jsonl("eval.jsonl", &eval_path, &mut |v| on_event(up("Sending the test questions")(v)))?)
        };
        let body = job_body(m, method, hyper, name, &train_id, eval_id.as_deref())?;
        let started = client.start_job(&body)?;
        let id = started["id"].as_str().ok_or_else(|| Error::new("Unsloth did not give the job an id."))?.to_string();
        std::fs::write(dir.join("unsloth.json"), serde_json::to_vec_pretty(&started)?)?;
        set_run_state(&dir, json!({"unsloth_job": id}));
        stage(on_event, if started["status"] == "queued" { "Waiting for Unsloth's graphics card (it is busy with another job)" } else { "Training in Unsloth" });

        // follow the job
        let mut seen: HashSet<String> = HashSet::new();
        let (mut last_step, mut total, mut last_loss) = (0u64, 0u64, Value::Null);
        let mut failures = 0u32;
        let mut shown = String::new();
        loop {
            match client.job(&id) {
                Ok(Some(j)) => {
                    failures = 0;
                    if let Ok(events) = new_events(&client, &id, &seen) {
                        for e in events {
                            if let Some(i) = e["id"].as_str() {
                                seen.insert(i.to_string());
                            }
                            if e["type"] == "metrics" {
                                let ev = metric_of(&e);
                                last_step = ev["step"].as_u64().unwrap_or(last_step);
                                total = ev["total_steps"].as_u64().unwrap_or(total);
                                last_loss = ev["loss"].clone();
                                rec.record(&ev);
                                on_event(ev);
                            } else if let Some(t) = e["message"].as_str() {
                                if t.contains("position") { stage(on_event, &queue_stage(t)); } else { on_event(json!({"kind": "log", "text": scrub(t)})); }
                            }
                        }
                    }
                    let status = j["status"].as_str().unwrap_or("").to_string();
                    if status != shown && status == "running" {
                        stage(on_event, "Training in Unsloth");
                    }
                    shown = status.clone();
                    match status.as_str() {
                        "succeeded" => {
                            let folder = j["fine_tuned_model"].as_str().unwrap_or("").to_string();
                            set_run_state(&dir, json!({"fine_tuned_model": folder}));
                            return Ok(json!({"steps": last_step, "total_steps": total.max(last_step), "final_loss": last_loss, "peak_vram_mb": null, "stopped": false, "backend": "unsloth-desktop",
                                             "fine_tuned_model": folder}));
                        }
                        "failed" => return Err(Error::new(scrub(j["error"]["message"].as_str().unwrap_or("Unsloth reported that the training failed.")))),
                        "cancelled" => return Err(Error::new("The job was cancelled in Unsloth.")),
                        _ => {}
                    }
                }
                Ok(None) => return Err(Error::new("Unsloth no longer knows this job (it was waiting in its queue when Unsloth restarted). Press Train to send it again.")),
                Err(e) => {
                    failures += 1;
                    if failures >= MAX_POLL_FAILURES {
                        return Err(Error::new(format!("Lost the connection to Unsloth. ({})", e.0)));
                    }
                }
            }
            std::thread::sleep(poll);
        }
    })();
    let mut outcome = outcome;
    if let Ok(r) = outcome.as_mut() {
        // the adapter stays in Unsloth's own folder in this version: say where
        let folder = r["fine_tuned_model"].as_str().unwrap_or("").to_string();
        let note = format!("Trained in Unsloth on {}. The adapter is in Unsloth's folder \"{folder}\" on that computer: export it there, or copy that folder into this run's adapter folder.", target.name);
        r["adapter_remote"] = json!(true);
        r["note"] = json!(note);
        set_run_state(&dir, json!({"adapter_remote": true, "note": note}));
        stage(on_event, &note);
    }
    finish(&dir, m, method, hyper, outcome, false)
}
