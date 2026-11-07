//! A training run on the server: send the data, train, show the same live events as a local run, bring the adapter back.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use super::client::RemoteClient;
use super::zipio::{extract_safe, zip_dir};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::catalog::ModelEntry;
use crate::train::plan::{Hyper, Method};
use crate::train::runs::{finish, runs_dir, set_run_state, Recorder};

/// A training server as the user entered it.
#[derive(Debug, Clone)]
pub struct Target {
    pub name: String,
    pub url: String,
    pub token: String,
}

/// How many one-second polls may fail in a row before the run is marked interrupted (the server keeps training meanwhile).
const MAX_POLL_FAILURES: u32 = 180;

fn stage(on_event: &mut dyn FnMut(Value), text: &str) {
    on_event(json!({"kind": "stage", "text": text}));
}

/// `resume`: continue a run that exists on the server (reconnect to it if it is still training, fetch its result if it finished, or start it again from
/// its last checkpoint). `poll` is the pause between asking the server for news.
pub fn execute_remote(p: &Project, m: &ModelEntry, method: Method, hyper: &Hyper, name: &str, mut job: Value, target: &Target, stop: Arc<AtomicBool>, resume: bool, poll: Duration,
                      on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let dir = runs_dir(p).join(name);
    std::fs::create_dir_all(&dir)?;
    job["remote"] = json!({"name": target.name, "url": target.url});
    std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&job)?)?;
    let prior: Value = std::fs::read(dir.join("state.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
    set_run_state(&dir, json!({"name": name, "model_id": m.id, "model": m.name, "method": method, "status": "running", "server": target.name, "server_url": target.url,
                               "seq": hyper.max_seq, "batch": hyper.batch, "error": null, "created": prior["created"].as_str().unwrap_or(&chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string())}));
    let client = RemoteClient::new(&target.url, &target.token);
    let mut rec = Recorder::new(&dir)?;
    let mut next: usize = 0;
    let mut attached = false;

    let outcome = (|| -> Result<Value> {
        stage(on_event, &format!("Connecting to {}", target.name));
        let st = client.status()?;
        if st["env_ready"] != true {
            return Err(Error::new("The training software is not set up on the server (the worker's Python environment is missing)."));
        }
        if st["gpu"].is_null() && st["cpu_ok"] != true {
            return Err(Error::new(format!("The server has no usable graphics card: {}", st["gpu_problem"].as_str().unwrap_or("none found"))));
        }
        let mut start_resume = false;
        if resume {
            match client.job_state(name)? {
                Some(s) if s["state"]["status"] == "running" => {
                    attached = true;
                    next = prior["remote_next"].as_u64().unwrap_or(0) as usize;
                    stage(on_event, "Reconnecting to the training that is still running");
                }
                Some(s) if s["state"]["status"] == "done" => {
                    attached = true;
                    next = prior["remote_next"].as_u64().unwrap_or(0) as usize;
                    stage(on_event, "The training had already finished; fetching the result");
                }
                Some(s) => start_resume = s["has_checkpoint"] == true,
                None => {}
            }
        }
        if !attached {
            if st["busy"] == true {
                return Err(Error::new(format!("The server is busy training \"{}\". Wait for it to finish.", st["job"].as_str().unwrap_or("another job"))));
            }
            match client.unload_ollama() {
                Ok(models) if !models.is_empty() => on_event(json!({"kind": "log", "text": format!("Freed the server's graphics card: unloaded {}.", models.join(", "))})),
                Ok(_) => {}
                Err(e) => on_event(json!({"kind": "log", "text": format!("Could not ask Ollama to free the card ({}). Continuing.", e.0)})),
            }
            stage(on_event, "Sending the training data");
            let train = job["dataset"]["train"].as_str().map(std::path::PathBuf::from).ok_or_else(|| Error::new("The job has no training file."))?;
            client.upload(name, "train.jsonl", &train, &mut |v| on_event(json!({"kind": "progress", "step": "upload", "value": v, "text": "Sending the training data"})))?;
            if let Some(init) = job["init_adapter"].as_str().map(std::path::PathBuf::from) {
                stage(on_event, "Sending the earlier model to continue from");
                let tmp = dir.join("init_adapter.zip");
                zip_dir(&init, &tmp)?;
                client.upload(name, "init_adapter.zip", &tmp, &mut |v| on_event(json!({"kind": "progress", "step": "upload", "value": v, "text": "Sending the earlier model"})))?;
                let _ = std::fs::remove_file(&tmp);
            }
            let mut spec = job.clone();
            spec["resume"] = json!(start_resume);
            spec.as_object_mut().map(|o| o.remove("remote"));
            stage(on_event, &format!("Training on {}", target.name));
            client.start(name, &spec)?;
            next = 0;
        }

        // follow the job
        let mut failures = 0u32;
        let mut asked_stop = false;
        let (final_state, mut result, mut error): (String, Option<Value>, Option<String>) = loop {
            if stop.load(Ordering::SeqCst) && !asked_stop {
                asked_stop = true;
                on_event(json!({"kind": "stage", "text": "Stopping after this step"}));
                let _ = client.stop(name);
            }
            match client.events(name, next) {
                Ok((events, n, state)) => {
                    failures = 0;
                    let (mut result, mut error) = (None, None);
                    for ev in events {
                        match ev["kind"].as_str() {
                            // the end of the run is reported by this function's result, not forwarded as an event
                            Some("done") => result = Some(ev["data"].clone()),
                            Some("error") => error = Some(ev["text"].as_str().unwrap_or("The training failed.").to_string()),
                            _ => { rec.record(&ev); on_event(ev); }
                        }
                    }
                    next = n;
                    set_run_state(&dir, json!({"remote_next": next}));
                    if state != "running" {
                        break (state, result, error);
                    }
                }
                Err(e) => {
                    failures += 1;
                    if failures >= MAX_POLL_FAILURES {
                        set_run_state(&dir, json!({"status": "interrupted", "error": e.0}));
                        return Err(Error::new(format!("Lost the connection to {}. The training may still be running there; press \"Continue\" to reconnect. ({})", target.name, e.0)));
                    }
                }
            }
            std::thread::sleep(poll);
        };
        if result.is_none() && error.is_none() {
            // the final events may be in the next page
            if let Ok((events, _, _)) = client.events(name, next) {
                for ev in events {
                    match ev["kind"].as_str() {
                        Some("done") => result = Some(ev["data"].clone()),
                        Some("error") => error = Some(ev["text"].as_str().unwrap_or("").to_string()),
                        _ => {}
                    }
                }
            }
        }
        if final_state == "done" || final_state == "stopped" {
            stage(on_event, "Fetching the trained model");
            let zip = dir.join("adapter.zip");
            client.download_adapter(name, &zip)?;
            let adapter = dir.join("adapter");
            let _ = std::fs::remove_dir_all(&adapter);
            extract_safe(&zip, &adapter)?;
            let _ = std::fs::remove_file(&zip);
        }
        match (final_state.as_str(), result, error) {
            ("done" | "stopped", Some(r), _) => Ok(r),
            ("stopped", None, _) => Err(Error::new("Cancelled.")),
            (_, _, Some(e)) => Err(Error::new(e)),
            (s, _, _) => Err(Error::new(format!("The server ended the job as \"{s}\" without a result."))),
        }
    })();
    // an interruption keeps its own state; everything else is finished like a local run (the server's card is not this PC's: no memory figure recorded)
    if prior_interrupted(&dir) {
        return outcome;
    }
    finish(&dir, m, method, hyper, outcome, false)
}

fn prior_interrupted(dir: &Path) -> bool {
    std::fs::read(dir.join("state.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).is_some_and(|s| s["status"] == "interrupted")
}
