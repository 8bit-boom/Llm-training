//! The job queue: jobs the person wants done one after the other (an Auto-cut, then an export, then a check ...), kept in `<settings folder>/queue.json` so a closed program
//! does not forget them. This module is the state machine only (no thread, no window): the window's runner asks `next_runnable`, starts the job, and `mark`s how it ended.
//! After a restart nothing starts by itself: running items become "stopped" and the queue waits (`paused`) until the person presses Run.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;

use crate::error::{Error, Result};

/// What can be queued: the jobs that have a Start button with a preflight checklist.
pub const KINDS: [&str; 7] = ["auto", "export", "heal", "think_check", "quantcheck", "imatrix_make", "train"];
pub const STATUSES: [&str; 6] = ["queued", "running", "done", "failed", "stopped", "skipped"];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    /// the folder of the project the job belongs to
    pub project: String,
    pub kind: String,
    /// the arguments of the job's start command, as the page would send them
    pub spec: Value,
    pub status: String,
    pub added: String,
    pub started: Option<String>,
    pub finished: Option<String>,
    pub error: Option<String>,
    /// the saved log of the job (`logs/jobs/...`), when it left one
    pub log: Option<String>,
    /// one line for the list ("Export run1 as q4_k_m")
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Queue {
    pub items: Vec<Item>,
    /// nothing starts by itself while this is true
    #[serde(default)]
    pub paused: bool,
}

pub fn file() -> PathBuf {
    crate::paths::config_dir().join("queue.json")
}

fn now() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

impl Queue {
    /// The saved queue; one that cannot be read is kept aside (`.broken-...`) and an empty one used.
    pub fn load() -> Queue {
        let f = file();
        match std::fs::read(&f) {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|_| {
                crate::fsx::set_aside_broken(&f);
                Queue::default()
            }),
            Err(_) => Queue::default(),
        }
    }

    pub fn save(&self) -> Result<()> {
        crate::fsx::write_json_atomic(&file(), self)
    }

    /// After a restart: what was running was cut off, and the queue waits for the person when anything is left to do.
    pub fn recover(&mut self) {
        let t = now();
        for i in self.items.iter_mut().filter(|i| i.status == "running") {
            i.status = "stopped".into();
            i.finished = Some(t.clone());
            i.error = Some("The program was closed while this was running.".into());
        }
        if self.items.iter().any(|i| i.status == "queued") {
            self.paused = true;
        }
    }

    pub fn enqueue(&mut self, project: &str, kind: &str, spec: Value) -> Result<String> {
        if !KINDS.contains(&kind) {
            return Err(Error::new(format!("{kind} cannot be queued.")));
        }
        let n = self.items.len();
        let id = format!("q{}-{n}", chrono::Utc::now().timestamp_micros());
        let label = label_of(kind, &spec);
        self.items.push(Item { id: id.clone(), project: project.into(), kind: kind.into(), spec, status: "queued".into(), added: now(), started: None, finished: None, error: None, log: None, label });
        Ok(id)
    }

    pub fn running(&self) -> Option<&Item> {
        self.items.iter().find(|i| i.status == "running")
    }

    /// The first waiting item, when the queue may start one: not paused and nothing running.
    pub fn next_runnable(&self) -> Option<usize> {
        if self.paused || self.running().is_some() {
            return None;
        }
        self.items.iter().position(|i| i.status == "queued")
    }

    /// Sets the status of an item (and its times, error and log).
    pub fn mark(&mut self, id: &str, status: &str, error: Option<String>, log: Option<String>) -> Result<()> {
        if !STATUSES.contains(&status) {
            return Err(Error::new(format!("Unknown status {status}.")));
        }
        let t = now();
        let it = self.items.iter_mut().find(|i| i.id == id).ok_or_else(|| Error::new("That queue entry is gone."))?;
        it.status = status.into();
        match status {
            "running" => {
                it.started = Some(t);
                it.finished = None;
                it.error = None;
            }
            "queued" => {
                it.started = None;
                it.finished = None;
                it.error = None;
            }
            _ => it.finished = Some(t),
        }
        if status != "running" && status != "queued" {
            it.error = error;
        }
        if log.is_some() {
            it.log = log;
        }
        Ok(())
    }

    /// Moves a waiting item to position `to` among the items (finished ones and the running one keep their places).
    pub fn reorder(&mut self, id: &str, to: usize) -> Result<()> {
        let from = self.items.iter().position(|i| i.id == id).ok_or_else(|| Error::new("That queue entry is gone."))?;
        if self.items[from].status != "queued" {
            return Err(Error::new("Only a waiting entry can be moved."));
        }
        let item = self.items.remove(from);
        // never in front of what has already run or is running
        let floor = self.items.iter().rposition(|i| i.status != "queued").map(|p| p + 1).unwrap_or(0);
        self.items.insert(to.clamp(floor, self.items.len()), item);
        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        let pos = self.items.iter().position(|i| i.id == id).ok_or_else(|| Error::new("That queue entry is gone."))?;
        if self.items[pos].status == "running" {
            return Err(Error::new("This entry is running. Stop it first."));
        }
        self.items.remove(pos);
        Ok(())
    }

    /// Takes the finished entries out of the list (done, failed, stopped, skipped).
    pub fn clear_finished(&mut self) {
        self.items.retain(|i| matches!(i.status.as_str(), "queued" | "running"));
    }

    pub fn view(&self) -> Value {
        json!({"items": self.items, "paused": self.paused, "running": self.running().map(|i| i.id.clone()), "queued": self.items.iter().filter(|i| i.status == "queued").count()})
    }
}

fn s(v: &Value, k: &str) -> String {
    v[k].as_str().unwrap_or("").to_string()
}

/// A short line for the list.
pub fn label_of(kind: &str, a: &Value) -> String {
    match kind {
        "auto" => if a["resume"].is_string() { "Continue Auto-cut".to_string() } else { format!("Auto-cut to {} B", a["spec"]["target_b"].as_f64().map(|t| t.to_string()).unwrap_or_else(|| "?".into())) },
        "export" => {
            let q = a["quants"].as_array().map(|l| l.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).filter(|x| !x.is_empty()).unwrap_or_else(|| s(a, "quant"));
            format!("Export {} as {}", s(a, "run"), q)
        }
        "heal" => format!("Heal {}", s(a, "name")),
        "think_check" => format!("Thinking check of {}", s(&a["spec"], "name")),
        "quantcheck" => "Compare quants".to_string(),
        "imatrix_make" => "Make an importance matrix".to_string(),
        "train" => format!("Train {}", s(a, "model")),
        other => other.to_string(),
    }
}

/// What the preflight checklist (`preflight::check`) wants to know about a queued job, from the arguments of its start command.
pub fn preflight_spec(kind: &str, a: &Value) -> Value {
    match kind {
        "export" => {
            let what = a["what"].as_str().unwrap_or("gguf");
            let quants = if what == "merged" { json!([]) } else { a["quants"].clone() };
            json!({"run": a["run"], "quant": a["quant"], "what": what, "quants": quants, "imatrix": what != "merged" && a["imatrix"]["enabled"] == true})
        }
        "heal" => json!({"name": a["name"], "tokens_m": a["spec"]["tokens_m"], "mode": a["spec"]["mode"]}),
        "auto" => json!({"studio_publish": a["spec"]["studio_publish"]}),
        "train" => json!({"server": a["server"]}),
        _ => json!({}),
    }
}

/// What was cut off by closing the program and can go on: the newest Auto-cut run that was marked interrupted (its id), and every heal that was (its name and settings).
/// As (kind, arguments of the start command) pairs, ready for `enqueue`. Marks nothing itself: `reshape::auto::mark_interrupted` and `reshape::runs::mark_interrupted` do that first.
pub fn interrupted_work(p: &crate::project::Project) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut by_auto: Vec<String> = Vec::new();
    if let Some(st) = crate::reshape::auto::list(p).into_iter().next().filter(|st| st.interrupted && st.status == "stopped") {
        by_auto = st.made.iter().map(|m| m.name.clone()).collect();         // its own heals go on with the run, not on their own
        out.push(("auto".to_string(), json!({"spec": null, "resume": st.id})));
    }
    if let Ok(rd) = std::fs::read_dir(crate::reshape::runs::models_dir(p)) {
        let mut heals: Vec<(String, Value)> = rd.flatten().filter_map(|e| {
            let h: Value = serde_json::from_slice(&std::fs::read(e.path().join("heal.json")).ok()?).ok()?;
            let name = e.file_name().to_string_lossy().to_string();
            (h["status"] == "stopped" && h["interrupted"] == true && h["spec"].is_object() && !by_auto.contains(&name)).then(|| (name, h["spec"].clone()))
        }).collect();
        heals.sort_by(|a, b| a.0.cmp(&b.0));
        out.extend(heals.into_iter().map(|(name, spec)| ("heal".to_string(), json!({"name": name, "spec": spec}))));
    }
    out
}
