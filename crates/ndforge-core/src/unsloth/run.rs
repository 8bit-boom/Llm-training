//! A training run in Unsloth: send the data, follow the job, bring the adapter back (see `docs/UNSLOTH_DESKTOP.md`).
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// What `fetch_adapter` found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchOutcome {
    /// the adapter files were copied into the run's `adapter` folder
    Copied,
    /// Unsloth's run folder is not on this computer (or holds no adapter): why
    NotReachable(String),
}

/// Files of Unsloth's run folder that make a usable adapter folder: the adapter itself and what the tokenizer and the chat format need.
fn wanted_file(name: &str) -> bool {
    matches!(name, "adapter_config.json" | "adapter_model.safetensors" | "special_tokens_map.json" | "added_tokens.json" | "vocab.json" | "merges.txt" | "preprocessor_config.json" | "chat_template.json")
        || name.starts_with("tokenizer") || name.starts_with("chat_template")
}

/// Copies the finished adapter out of Unsloth's run folder `fine_tuned_model` into `dest` (the run's `adapter` folder) when that folder is on THIS computer.
/// Unsloth's `/api/models/checkpoints` names the folder's path; only the folder Lore Forge's own job made is read, only regular files directly inside it.
/// The copy goes to `<dest>.partial` first and replaces `dest` by a rename, so a half copy is never taken for an adapter. `free_gb` measures the free disk space.
pub fn fetch_adapter(client: &UnslothClient, fine_tuned_model: &str, dest: &Path, free_gb: &dyn Fn(&Path) -> Option<f64>) -> Result<FetchOutcome> {
    let listing = client.checkpoints_listing()?;
    let path = listing["models"].as_array().and_then(|ms| ms.iter().find(|m| m["name"] == fine_tuned_model))
        .and_then(|m| m["checkpoints"].as_array()).and_then(|cs| cs.iter().filter_map(|c| c["path"].as_str()).find(|p| Path::new(p).file_name().and_then(|n| n.to_str()) == Some(fine_tuned_model)))
        .map(PathBuf::from);
    let Some(src) = path else { return Ok(FetchOutcome::NotReachable("Unsloth does not list the run folder.".into())) };
    let has = |n: &str| std::fs::symlink_metadata(src.join(n)).map(|m| m.is_file()).unwrap_or(false);
    if !(has("adapter_config.json") && has("adapter_model.safetensors")) {
        return Ok(FetchOutcome::NotReachable("The run folder is not on this computer (or holds no adapter).".into()));
    }
    let files: Vec<(PathBuf, String, u64)> = std::fs::read_dir(&src)?.flatten().filter_map(|e| {
        let md = std::fs::symlink_metadata(e.path()).ok()?;
        let name = e.file_name().to_str()?.to_string();
        (md.is_file() && wanted_file(&name)).then(|| (e.path(), name, md.len()))
    }).collect();
    let bytes: u64 = files.iter().map(|f| f.2).sum();
    let parent = dest.parent().unwrap_or(Path::new("."));
    if let Some(free) = free_gb(parent) {
        let need = bytes as f64 * 1.1 / 1e9;
        if free < need {
            return Err(Error::new(format!("The disk is full on this computer: the adapter needs about {need:.1} GB and {free:.1} GB are free. Free space and copy the folder \"{fine_tuned_model}\" from Unsloth's outputs by hand.")));
        }
    }
    let partial = dest.with_file_name(format!("{}.partial", dest.file_name().and_then(|n| n.to_str()).unwrap_or("adapter")));
    let _ = std::fs::remove_dir_all(&partial);
    std::fs::create_dir_all(&partial)?;
    let copy = || -> Result<()> {
        for (from, name, _) in &files {
            std::fs::copy(from, partial.join(name))?;
        }
        let _ = std::fs::remove_dir_all(dest);
        std::fs::rename(&partial, dest)?;
        Ok(())
    };
    if let Err(e) = copy() {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(e);
    }
    Ok(FetchOutcome::Copied)
}

/// Unsloth's failure texts in plain words (the table in the contract document).
pub fn explain_failure(msg: &str) -> String {
    let m = scrub(msg);
    let low = m.to_lowercase();
    if low.contains("out of memory") || low.split(|c: char| !c.is_alphanumeric()).any(|w| w == "oom") {
        "Unsloth ran out of graphics memory. Choose QLoRA, a shorter context or a smaller model, and close chat models in Unsloth.".to_string()
    } else if low.contains("no space left") || low.contains("errno 28") {
        "The disk is full on the computer where Unsloth runs. Free space and try again.".to_string()
    } else {
        m
    }
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
    // newest first: keep what comes before the newest event already known
    if let Some(at) = all.iter().position(|e| e["id"].as_str().is_some_and(|i| seen.contains(i))) {
        all.truncate(at);
    }
    all.reverse();
    Ok(all)
}

/// Trains in Unsloth to the end and keeps the run's state like a local run. `poll` is the pause between asking Unsloth for news.
/// When Unsloth runs on this computer the adapter is copied into the run's `adapter` folder (`fetch_adapter`); else it stays in Unsloth's run folder.
pub fn execute_unsloth(p: &Project, m: &ModelEntry, method: Method, hyper: &Hyper, name: &str, mut job: Value, target: &UnslothTarget, stop: Arc<AtomicBool>, resume: bool, poll: Duration,
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
        // continue: reattach to the job Unsloth still has, fetch what it finished, or send the data again
        let mut id = String::new();
        let mut reattached = false;
        if resume {
            if let Some(old) = prior["unsloth_job"].as_str() {
                match client.job(old)? {
                    None => {
                        set_run_state(&dir, json!({"unsloth_job": null}));
                        return Err(Error::new("Unsloth no longer knows this job (it was waiting in its queue when Unsloth restarted). Press Continue to send it again."));
                    }
                    Some(j) if matches!(j["status"].as_str(), Some("running" | "queued" | "validating_files" | "succeeded")) => {
                        id = old.to_string();
                        reattached = true;
                        stage(on_event, if j["status"] == "succeeded" { "The training had already finished; fetching the result" } else { "Reconnecting to the training that is still running" });
                    }
                    Some(_) => {}   // failed or cancelled there: start it again below
                }
            }
        }
        if !reattached {
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
            let started_id = started["id"].as_str().ok_or_else(|| Error::new("Unsloth did not give the job an id."))?.to_string();
            std::fs::write(dir.join("unsloth.json"), serde_json::to_vec_pretty(&started)?)?;
            set_run_state(&dir, json!({"unsloth_job": started_id, "unsloth_events_seen": null}));
            stage(on_event, if started["status"] == "queued" { "Waiting for Unsloth's graphics card (it is busy with another job)" } else { "Training in Unsloth" });
            id = started_id;
        }
        // follow the job
        let mut seen: HashSet<String> = HashSet::new();
        let (mut last_step, mut total, mut last_loss) = (0u64, 0u64, Value::Null);
        if reattached {
            // what the run already showed: the charts keep their lines, only newer events are added
            if let Some(i) = prior["unsloth_events_seen"].as_str() { seen.insert(i.to_string()); }
            (last_step, total, last_loss) = (prior["step"].as_u64().unwrap_or(0), prior["total_steps"].as_u64().unwrap_or(0), prior["loss"].clone());
        }
        let mut asked_stop = false;
        let mut failures = 0u32;
        let mut shown = String::new();
        loop {
            if stop.load(Ordering::SeqCst) && !asked_stop {
                asked_stop = true;
                stage(on_event, "Stopping the training in Unsloth");
                let _ = client.cancel(&id);   // already finished is fine: the next poll says so
            }
            match client.job(&id) {
                Ok(Some(j)) => {
                    failures = 0;
                    if let Ok(events) = new_events(&client, &id, &seen) {
                        for e in events {
                            if let Some(i) = e["id"].as_str() {
                                seen.insert(i.to_string());
                                set_run_state(&dir, json!({"unsloth_events_seen": i}));
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
                        "failed" => return Err(Error::new(explain_failure(j["error"]["message"].as_str().unwrap_or("Unsloth reported that the training failed.")))),
                        "cancelled" => {
                            // stopped here, or in Unsloth: the run ends "stopped" (a cancelled job cannot be continued)
                            return Ok(json!({"steps": last_step, "total_steps": total.max(last_step), "final_loss": last_loss, "peak_vram_mb": null, "stopped": true, "backend": "unsloth-desktop"}));
                        }
                        _ => {}
                    }
                }
                Ok(None) => {
                    set_run_state(&dir, json!({"unsloth_job": null}));
                    return Err(Error::new("Unsloth no longer knows this job (it was waiting in its queue when Unsloth restarted). Press Continue to send it again."));
                }
                Err(e) => {
                    failures += 1;
                    if failures >= MAX_POLL_FAILURES {
                        set_run_state(&dir, json!({"status": "interrupted", "error": e.0}));
                        return Err(Error::new(format!("Lost the connection to Unsloth. The training may still be running there; press \"Continue\" to reconnect. ({})", e.0)));
                    }
                }
            }
            std::thread::sleep(poll);
        }
    })();
    let mut outcome = outcome;
    if let Some(r) = outcome.as_mut().ok().filter(|r| r["stopped"] != true) {
        let folder = r["fine_tuned_model"].as_str().unwrap_or("").to_string();
        stage(on_event, "Fetching the trained model");
        let fetched = fetch_adapter(&client, &folder, &dir.join("adapter"), &|d| crate::envmgr::health::free_disk_gb(d));
        match fetched {
            Ok(FetchOutcome::Copied) => {
                r["adapter_remote"] = json!(false);
                set_run_state(&dir, json!({"adapter_remote": false, "note": null}));
            }
            other => {
                // the training is done either way; the adapter stays in Unsloth's folder
                let why = match other { Ok(FetchOutcome::NotReachable(w)) => w, Err(e) => e.0, _ => String::new() };
                let note = format!("Trained in Unsloth on {}. The adapter is in Unsloth's folder \"{folder}\" on that computer: export it there, or copy that folder into this run's adapter folder. ({why})", target.name);
                r["adapter_remote"] = json!(true);
                r["note"] = json!(note);
                set_run_state(&dir, json!({"adapter_remote": true, "note": note}));
                stage(on_event, &note);
            }
        }
    }
    // an interruption keeps its own state (the job goes on in Unsloth); everything else is finished like a local run
    if std::fs::read(dir.join("state.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).is_some_and(|s| s["status"] == "interrupted") {
        return outcome;
    }
    let stopped = matches!(&outcome, Ok(r) if r["stopped"] == true);
    let end = finish(&dir, m, method, hyper, outcome, false);
    if stopped {
        set_run_state(&dir, json!({"resumable": false}));
    }
    end
}
