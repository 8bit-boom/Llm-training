//! The job server that runs on the training computer (TrueNAS). One job at a time; the graphics card is shared with nothing else while it trains
//! (`POST /v1/ollama/unload` frees what Ollama holds). See `remote/mod.rs` for the protocol.
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use super::zipio::{extract_safe, zip_dir};
use crate::envmgr::gpu::detect_gpus;
use crate::error::{Error, Result};
use crate::export::ollama::Ollama;
use crate::train::runs::{drive, Recorder};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub listen: String,
    pub token: String,
    pub data_dir: PathBuf,
    pub python: PathBuf,
    pub ollama_url: Option<String>,
    /// refuse jobs when no graphics card is found (set false only for development on a CPU)
    pub require_gpu: bool,
    /// largest accepted upload in bytes
    pub max_upload: u64,
    /// how long a stopping job may take to save its checkpoint before it is killed
    pub grace: Duration,
}

struct Active {
    id: String,
    stop: Arc<AtomicBool>,
}

struct Shared {
    cfg: ServerConfig,
    active: Mutex<Option<Active>>,
}

pub struct Handle {
    pub addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
}

impl Handle {
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 80 && id != "." && id != ".." && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-' || b == b'_')
}

/// Equal-length, constant-time comparison (a token is not leaked through timing).
fn same_token(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = (a.len() != b.len()) as u8;
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(1);
    }
    diff == 0
}

fn json_response(status: u16, v: Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = v.to_string().into_bytes();
    Response::from_data(body).with_status_code(StatusCode(status)).with_header(Header::from_bytes("Content-Type", "application/json").expect("header"))
}

fn err(status: u16, msg: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(status, json!({"error": msg}))
}

fn job_dir(s: &Shared, id: &str) -> PathBuf {
    s.cfg.data_dir.join("jobs").join(id)
}

fn read_json(path: &Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn set_job_state(dir: &Path, status: &str, extra: Value) {
    let mut v = json!({"status": status, "updated": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()});
    if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        for (k, x) in e {
            o.insert(k.clone(), x.clone());
        }
    }
    let _ = std::fs::write(dir.join("job_state.json"), v.to_string());
}

fn append_event(dir: &Path, ev: &Value) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("events.jsonl")) {
        let _ = writeln!(f, "{ev}");
    }
}

fn start_job(s: &Arc<Shared>, id: &str, mut job: Value) -> std::result::Result<(), (u16, String)> {
    let dir = job_dir(s, id);
    let out = dir.join("out");
    let input = dir.join("in");
    let train = input.join("train.jsonl");
    if !train.is_file() {
        return Err((400, "Upload train.jsonl first.".into()));
    }
    if !s.cfg.python.is_file() {
        return Err((503, "The training software is not installed on this server.".into()));
    }
    let mut guard = s.active.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(a) = guard.as_ref() {
        return Err((409, format!("The server is busy training \"{}\".", a.id)));
    }
    // the client's paths are its own: point the job at this server's files
    job["dataset"] = json!({"train": train});
    job["out"] = json!(out);
    let init_zip = input.join("init_adapter.zip");
    if init_zip.is_file() {
        let target = input.join("init_adapter");
        let _ = std::fs::remove_dir_all(&target);
        extract_safe(&init_zip, &target).map_err(|e| (400, e.0))?;
        job["init_adapter"] = json!(target);
    } else if let Some(o) = job.as_object_mut() {
        o.remove("init_adapter");
    }
    // a job started again with the same id (resume) keeps its checkpoints but starts a fresh event list
    let _ = std::fs::remove_file(dir.join("events.jsonl"));
    let _ = std::fs::remove_file(dir.join("adapter.zip"));
    let stop = Arc::new(AtomicBool::new(false));
    *guard = Some(Active { id: id.to_string(), stop: stop.clone() });
    set_job_state(&dir, "running", json!({}));
    let (shared, py, grace) = (s.clone(), s.cfg.python.clone(), s.cfg.grace);
    std::thread::spawn(move || {
        let mut rec = match Recorder::new(&out) {
            Ok(r) => r,
            Err(e) => {
                append_event(&dir, &json!({"kind": "error", "text": e.0}));
                set_job_state(&dir, "failed", json!({"error": e.0}));
                *shared.active.lock().unwrap_or_else(|e| e.into_inner()) = None;
                return;
            }
        };
        let outcome = drive(&py, &out, &job, stop, grace, &mut |e| {
            let ev = rec.from_worker(e);
            append_event(&dir, &ev);
        });
        match outcome {
            Ok(result) => {
                let status = if result["stopped"] == true { "stopped" } else { "done" };
                append_event(&dir, &json!({"kind": "done", "data": result}));
                set_job_state(&dir, status, json!({"result": result}));
            }
            Err(e) => {
                let stopped = e.0 == "Cancelled.";
                append_event(&dir, &json!({"kind": "error", "text": if stopped { "Stopped.".to_string() } else { e.0.clone() }}));
                set_job_state(&dir, if stopped { "stopped" } else { "failed" }, json!({"error": e.0}));
            }
        }
        *shared.active.lock().unwrap_or_else(|e| e.into_inner()) = None;
    });
    Ok(())
}

fn query_param(url: &str, key: &str) -> Option<String> {
    url.split_once('?')?.1.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v.to_string()))
}

fn handle(s: &Arc<Shared>, mut req: Request) {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    let bearer = req.headers().iter().find(|h| h.field.equiv("Authorization")).map(|h| h.value.as_str().to_string()).unwrap_or_default();
    let authorized = bearer.strip_prefix("Bearer ").is_some_and(|t| same_token(t, &s.cfg.token));
    if !authorized {
        let _ = req.respond(err(401, "A valid access token is needed."));
        return;
    }
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    let id_ok = |i: usize| parts.get(i).is_some_and(|x| valid_id(x));
    match (&method, parts.as_slice()) {
        (Method::Get, ["v1", "status"]) => {
            let gpus = detect_gpus();
            let active = s.active.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|a| a.id.clone());
            let _ = req.respond(json_response(200, json!({
                "service": "nd-lore-forge-worker", "version": env!("CARGO_PKG_VERSION"),
                "gpu": gpus.as_ref().ok().and_then(|g| g.first()).map(|g| json!({"name": g.name, "vram_gb": (g.vram_gb() * 10.0).round() / 10.0, "driver": g.driver, "precision": g.precision()})),
                "gpu_problem": gpus.err(), "busy": active.is_some(), "job": active,
                "env_ready": s.cfg.python.is_file(), "ollama": s.cfg.ollama_url, "cpu_ok": !s.cfg.require_gpu,
            })));
        }
        (Method::Put, ["v1", "jobs", _, "files", name]) if id_ok(2) && ["train.jsonl", "init_adapter.zip"].contains(name) => {
            let id = parts[2];
            if s.active.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|a| a.id == id) {
                let _ = req.respond(err(409, "This job is running; its files cannot be replaced."));
                return;
            }
            if req.body_length().unwrap_or(0) as u64 > s.cfg.max_upload {
                let _ = req.respond(err(413, "The file is larger than this server accepts."));
                return;
            }
            let dir = job_dir(s, id).join("in");
            let target = dir.join(name);
            let tmp = dir.join(format!("{name}.part"));
            let saved = std::fs::create_dir_all(&dir).and_then(|_| {
                let mut f = std::fs::File::create(&tmp)?;
                let n = std::io::copy(&mut req.as_reader().take(s.cfg.max_upload + 1), &mut f)?;
                if n > s.cfg.max_upload { return Err(std::io::Error::other("too large")); }
                std::fs::rename(&tmp, &target)
            });
            let _ = std::fs::remove_file(&tmp);
            let _ = req.respond(match saved { Ok(()) => json_response(200, json!({"saved": name})), Err(e) => err(400, &format!("The file could not be saved: {e}")) });
        }
        (Method::Post, ["v1", "jobs", _]) if id_ok(2) => {
            let mut body = String::new();
            let _ = req.as_reader().take(1 << 20).read_to_string(&mut body);
            let job = serde_json::from_str::<Value>(&body).ok().and_then(|v| v.get("job").cloned()).filter(|j| j["job"] == "train");
            let Some(job) = job else {
                let _ = req.respond(err(400, "Send {\"job\": {...}} with a training job."));
                return;
            };
            let _ = req.respond(match start_job(s, parts[2], job) {
                Ok(()) => json_response(200, json!({"started": parts[2]})),
                Err((code, msg)) => err(code, &msg),
            });
        }
        (Method::Get, ["v1", "jobs", _]) if id_ok(2) => {
            let dir = job_dir(s, parts[2]);
            let _ = req.respond(match read_json(&dir.join("job_state.json")) {
                Some(v) => json_response(200, json!({"state": v, "has_input": dir.join("in/train.jsonl").is_file(), "has_checkpoint": dir.join("out/checkpoints").is_dir()})),
                None => err(404, "No such job."),
            });
        }
        (Method::Get, ["v1", "jobs", _, "events"]) if id_ok(2) => {
            let dir = job_dir(s, parts[2]);
            let from: usize = query_param(&url, "from").and_then(|v| v.parse().ok()).unwrap_or(0);
            let state = read_json(&dir.join("job_state.json")).and_then(|v| v["status"].as_str().map(String::from));
            let Some(state) = state else {
                let _ = req.respond(err(404, "No such job."));
                return;
            };
            let text = std::fs::read_to_string(dir.join("events.jsonl")).unwrap_or_default();
            let all: Vec<Value> = text.lines().skip(from).take(500).filter_map(|l| serde_json::from_str(l).ok()).collect();
            let next = from + all.len();
            let _ = req.respond(json_response(200, json!({"events": all, "next": next, "state": state})));
        }
        (Method::Post, ["v1", "jobs", _, "stop"]) if id_ok(2) => {
            let active = s.active.lock().unwrap_or_else(|e| e.into_inner());
            let _ = req.respond(match active.as_ref().filter(|a| a.id == parts[2]) {
                Some(a) => { a.stop.store(true, Ordering::SeqCst); json_response(200, json!({"stopping": parts[2]})) }
                None => err(404, "That job is not running."),
            });
        }
        (Method::Get, ["v1", "jobs", _, "adapter.zip"]) if id_ok(2) => {
            let dir = job_dir(s, parts[2]);
            let adapter = dir.join("out/adapter");
            if !adapter.join("adapter_config.json").is_file() {
                let _ = req.respond(err(404, "This job has no trained adapter yet."));
                return;
            }
            let zip = dir.join("adapter.zip");
            if let Err(e) = zip_dir(&adapter, &zip) {
                let _ = req.respond(err(500, &e.0));
                return;
            }
            match std::fs::File::open(&zip) {
                Ok(f) => { let _ = req.respond(Response::from_file(f).with_header(Header::from_bytes("Content-Type", "application/zip").expect("header"))); }
                Err(e) => { let _ = req.respond(err(500, &e.to_string())); }
            }
        }
        (Method::Delete, ["v1", "jobs", _]) if id_ok(2) => {
            if s.active.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|a| a.id == parts[2]) {
                let _ = req.respond(err(409, "This job is running."));
                return;
            }
            let _ = std::fs::remove_dir_all(job_dir(s, parts[2]));
            let _ = req.respond(json_response(200, json!({"deleted": parts[2]})));
        }
        (Method::Post, ["v1", "ollama", "unload"]) => {
            let _ = req.respond(match &s.cfg.ollama_url {
                None => json_response(200, json!({"unloaded": [], "note": "No Ollama address is configured on this server."})),
                Some(url) => {
                    let o = Ollama::new(url);
                    match o.loaded() {
                        Ok(models) => {
                            let done: Vec<String> = models.into_iter().filter(|m| o.unload(m).is_ok()).collect();
                            json_response(200, json!({"unloaded": done}))
                        }
                        Err(e) => json_response(200, json!({"unloaded": [], "note": e.0})),
                    }
                }
            });
        }
        _ => {
            let _ = req.respond(err(404, "Unknown address."));
        }
    }
}

/// Binds and serves in a background thread; returns the address. `Handle::shutdown` stops the accept loop (a running job keeps going).
pub fn spawn(cfg: ServerConfig) -> Result<Handle> {
    if cfg.token.len() < 16 {
        return Err(Error::new("The access token must be at least 16 characters long."));
    }
    std::fs::create_dir_all(cfg.data_dir.join("jobs"))?;
    let server = Server::http(&cfg.listen).map_err(|e| Error::new(format!("Cannot listen on {}: {e}", cfg.listen)))?;
    let addr = server.server_addr().to_ip().ok_or_else(|| Error::new("The server has no IP address."))?;
    let shared = Arc::new(Shared { cfg, active: Mutex::new(None) });
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    std::thread::spawn(move || {
        while !flag.load(Ordering::SeqCst) {
            match server.recv_timeout(Duration::from_millis(200)) {
                Ok(Some(req)) => {
                    let s = shared.clone();
                    std::thread::spawn(move || handle(&s, req));
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
    });
    Ok(Handle { addr, shutdown })
}

/// Serves until the process ends.
pub fn serve(cfg: ServerConfig) -> Result<()> {
    let h = spawn(cfg)?;
    eprintln!("ND Lore Forge worker listening on {}", h.addr);
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}
