//! A stand-in for Unsloth Desktop / Studio: pure Rust on 127.0.0.1 (a free port), so it runs on every platform. It replays the shapes in
//! `tests/fixtures/unsloth/*.json` (the same files nd-world uses) and refuses unknown job fields with Unsloth's own text.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{json, Value};

pub const KEY: &str = "sk-unsloth-test-0123456789abcdef";
pub const RUN_FOLDER: &str = "ashfall-test_20261113_101500";

pub fn fixture(name: &str) -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/unsloth/");
    serde_json::from_str(&std::fs::read_to_string(format!("{path}{name}")).unwrap_or_else(|e| panic!("fixture {name}: {e}"))).unwrap()
}

/// What the fake does; tests change it through `Fake::with`.
pub struct State {
    pub key: String,
    /// serve the Desktop fields in /api/health
    pub desktop: bool,
    /// answer /api/health like some other program
    pub not_unsloth: bool,
    /// false: the fine-tuning routes answer 404 (an old Unsloth)
    pub fine_tuning: bool,
    pub chat_only: bool,
    /// the first this many polls of a new job say "queued"
    pub queue_polls: u32,
    /// the job never ends by itself (the Stop button tests)
    pub hold: bool,
    /// the job fails at this step with this message
    pub fail: Option<(u32, String)>,
    /// the job id answers 404 (a queue lost in a restart)
    pub forget_jobs: bool,
    /// the job routes answer 500 once this many polls were made (a lost connection); a test can switch it off again
    pub break_at: Option<u32>,
    pub broken: bool,
    pub total_steps: u32,
    /// where the adapter lives on "Desktop's" disk (a folder of the test)
    pub outputs_dir: String,
    // what happened
    pub files: Vec<(String, String, Vec<u8>)>,
    pub job_bodies: Vec<Value>,
    pub requests: Vec<String>,
    pub authorization: Vec<String>,
    // the running job
    job_id: Option<String>,
    polls: u32,
    cancelled: bool,
}

impl State {
    fn new(outputs_dir: &str) -> State {
        State { key: KEY.into(), desktop: true, not_unsloth: false, fine_tuning: true, chat_only: false, queue_polls: 0, hold: false, fail: None, forget_jobs: false, break_at: None, broken: false, total_steps: 4,
                outputs_dir: outputs_dir.into(), files: vec![], job_bodies: vec![], requests: vec![], authorization: vec![], job_id: None, polls: 0, cancelled: false }
    }

    /// (status, step reached)
    fn progress(&self) -> (&'static str, u32) {
        if self.cancelled {
            return ("cancelled", self.polls.saturating_sub(self.queue_polls).min(self.total_steps));
        }
        if self.polls <= self.queue_polls && self.queue_polls > 0 {
            return ("queued", 0);
        }
        let step = (self.polls - self.queue_polls).min(self.total_steps);
        if let Some((at, _)) = &self.fail {
            if step >= *at {
                return ("failed", *at);
            }
        }
        if step >= self.total_steps && !self.hold { ("succeeded", step) } else { ("running", step) }
    }
}

pub struct Fake {
    pub url: String,
    pub state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Fake {
    pub fn start(outputs_dir: &std::path::Path) -> Fake {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
        let state = Arc::new(Mutex::new(State::new(&outputs_dir.to_string_lossy())));
        let stop = Arc::new(AtomicBool::new(false));
        let (st, sp) = (state.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !sp.load(Ordering::SeqCst) {
                if let Ok(Some(req)) = server.recv_timeout(Duration::from_millis(20)) {
                    handle(&st, req);
                }
            }
        });
        Fake { url, state, stop, thread: Some(thread) }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        f(&mut self.state.lock().unwrap())
    }

    pub fn count(&self, needle: &str) -> usize {
        self.with(|s| s.requests.iter().filter(|r| r.contains(needle)).count())
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn reply(req: tiny_http::Request, code: u16, body: Value) {
    let h = tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let _ = req.respond(tiny_http::Response::from_string(body.to_string()).with_status_code(code).with_header(h));
}

fn error(req: tiny_http::Request, code: u16, text: &str) {
    reply(req, code, json!({"detail": text}));
}

/// The text part and the file part of a multipart body (enough for what Lore Forge sends).
fn multipart(content_type: &str, body: &[u8]) -> (Vec<(String, String)>, Option<(String, Vec<u8>)>) {
    let boundary = content_type.split("boundary=").nth(1).unwrap_or("").trim_matches('"').to_string();
    let delim = format!("--{boundary}").into_bytes();
    let (mut fields, mut file) = (vec![], None);
    let find = |h: &[u8], n: &[u8], from: usize| h[from..].windows(n.len()).position(|w| w == n).map(|i| i + from);
    let mut at = 0;
    while let Some(s) = find(body, &delim, at) {
        let start = s + delim.len();
        let Some(e) = find(body, &delim, start) else { break };
        let part = &body[start..e];
        if let Some(sep) = find(part, b"\r\n\r\n", 0) {
            let head = String::from_utf8_lossy(&part[..sep]).to_string();
            let data = part[sep + 4..part.len().saturating_sub(2).max(sep + 4)].to_vec();
            let name = head.split("name=\"").nth(1).and_then(|r| r.split('"').next()).unwrap_or("").to_string();
            if head.contains("filename=") {
                let fname = head.split("filename=\"").nth(1).and_then(|r| r.split('"').next()).unwrap_or("").to_string();
                file = Some((fname, data));
            } else {
                fields.push((name, String::from_utf8_lossy(&data).to_string()));
            }
        }
        at = e;
    }
    (fields, file)
}

fn job_object(s: &State, body: &Value) -> Value {
    let (status, _) = s.progress();
    let mut j = fixture(if status == "queued" { "job_queued.json" } else if status == "succeeded" { "job_succeeded.json" } else { "job_running.json" });
    j["status"] = json!(status);
    j["id"] = json!(s.job_id.clone().unwrap_or_default());
    j["model"] = body["model"].clone();
    j["suffix"] = body["suffix"].clone();
    j["training_file"] = body["training_file"].clone();
    if status == "succeeded" {
        j["fine_tuned_model"] = json!(RUN_FOLDER);
    }
    if status == "failed" {
        j["error"] = json!({"message": s.fail.as_ref().map(|f| f.1.clone()).unwrap_or_default(), "code": "training_failed"});
    }
    j
}

fn handle(state: &Arc<Mutex<State>>, mut req: tiny_http::Request) {
    let method = req.method().as_str().to_string();
    let url = req.url().to_string();
    let (path, query) = url.split_once('?').unwrap_or((&url, ""));
    let (path, query) = (path.to_string(), query.to_string());
    let header = |req: &tiny_http::Request, name: &'static str| req.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str().to_string()).unwrap_or_default();
    let (auth, ctype) = (header(&req, "Authorization"), header(&req, "Content-Type"));
    let mut body = Vec::new();
    let _ = req.as_reader().read_to_end(&mut body);
    let mut s = state.lock().unwrap();
    s.requests.push(format!("{method} {path}"));
    s.authorization.push(auth.clone());
    let authed = auth == format!("Bearer {}", s.key);
    let param = |k: &str| query.split('&').find_map(|p| p.strip_prefix(&format!("{k}=")).map(String::from));

    if path == "/api/liveness" {
        return reply(req, 200, json!({"status": "alive", "service": "Unsloth UI Backend"}));
    }
    if path == "/api/health" {
        if s.not_unsloth {
            return reply(req, 200, json!({"status": "ok", "service": "Some Other Program"}));
        }
        let mut h = fixture(if authed { "health_auth.json" } else { "health_unauth.json" });
        if !s.desktop {
            if let Some(o) = h.as_object_mut() {
                o.remove("desktop_owner");
                o.remove("desktop_protocol_version");
            }
        }
        h["chat_only"] = json!(s.chat_only);
        return reply(req, 200, h);
    }
    if !authed {
        return error(req, 401, "Invalid or missing API key");
    }
    if path == "/v1/models" {
        return reply(req, 200, json!({"object": "list", "data": []}));
    }
    if path == "/api/train/hardware" {
        return reply(req, 200, fixture("hardware.json"));
    }
    if path == "/api/models/checkpoints" {
        let text = fixture("checkpoints.json").to_string().replace("@OUTPUTS@", &s.outputs_dir.replace('\\', "\\\\"));
        return reply(req, 200, serde_json::from_str(&text).unwrap());
    }
    if path.starts_with("/v1/") && !s.fine_tuning && path != "/v1/models" {
        return error(req, 404, "Not Found");
    }
    if method == "POST" && path == "/v1/files" {
        let (fields, file) = multipart(&ctype, &body);
        if fields.iter().find(|f| f.0 == "purpose").map(|f| f.1.as_str()) != Some("fine-tune") {
            return error(req, 400, "purpose must be fine-tune");
        }
        let Some((name, data)) = file else { return error(req, 400, "no file") };
        let id = format!("file-{:032x}", s.files.len() + 1);
        let mut f = fixture("files.json");
        f["id"] = json!(id);
        f["bytes"] = json!(data.len());
        f["filename"] = json!(name);
        s.files.push((id, name, data));
        return reply(req, 200, f);
    }
    if method == "POST" && path == "/v1/fine_tuning/jobs" {
        let Ok(b) = serde_json::from_slice::<Value>(&body) else { return error(req, 400, "bad json") };
        let known = fixture("job_fields.json");
        let ok: Vec<&str> = ["openai", "studio"].iter().flat_map(|k| known[k].as_array().unwrap().iter().filter_map(|x| x.as_str())).collect();
        let unknown: Vec<String> = b.as_object().map(|o| o.keys().filter(|k| !ok.contains(&k.as_str())).cloned().collect()).unwrap_or_default();
        if !unknown.is_empty() {
            return error(req, 400, &format!("Unknown parameter(s): {}", unknown.join(", ")));
        }
        if !s.files.iter().any(|f| Some(f.0.as_str()) == b["training_file"].as_str()) {
            return error(req, 400, "training_file not found");
        }
        s.job_bodies.push(b.clone());
        s.job_id = Some(if s.queue_polls > 0 { "ftjob_q0123456789abcdef".into() } else { "ftjob_20261113_101500_ab12".to_string() });
        s.polls = 0;
        s.cancelled = false;
        let j = job_object(&s, &b);
        return reply(req, 200, j);
    }
    // /v1/fine_tuning/jobs/{id}[/events|/cancel]
    if let Some(rest) = path.strip_prefix("/v1/fine_tuning/jobs") {
        if rest.is_empty() || rest == "/" {
            return reply(req, 200, json!({"object": "list", "data": [], "has_more": false}));
        }
        let rest = rest.trim_start_matches('/');
        let (id, tail) = rest.split_once('/').unwrap_or((rest, ""));
        if s.broken {
            return error(req, 500, "boom");
        }
        if s.forget_jobs || s.job_id.as_deref() != Some(id) {
            return error(req, 404, "Job not found");
        }
        let b = s.job_bodies.last().cloned().unwrap_or(Value::Null);
        match (method.as_str(), tail) {
            ("GET", "") => {
                s.polls += 1;
                if s.break_at == Some(s.polls) {
                    s.broken = true;
                }
                let j = job_object(&s, &b);
                return reply(req, 200, j);
            }
            ("GET", "events") => {
                let (status, step) = s.progress();
                let mut all: Vec<Value> = vec![];
                let mut ev = fixture("events.json")["data"].clone();
                let template = ev[1].clone();
                let start = ev[2].clone();
                if status == "queued" {
                    all.push(json!({"object": "fine_tuning.job.event", "id": "ftevent_queue", "created_at": 1763000150, "level": "info", "type": "message", "data": {}, "message": "Waiting for the GPU: position 1 in the queue"}));
                } else {
                    all.push(start);
                    for n in 1..=step {
                        let mut e = template.clone();
                        e["id"] = json!(format!("ftevent_{}_step{n}", s.job_id.clone().unwrap_or_default()));
                        e["data"]["step"] = json!(n);
                        e["data"]["total_steps"] = json!(s.total_steps);
                        e["data"]["train_loss"] = json!(3.0 - 0.5 * n as f64);
                        e["data"]["valid_loss"] = if n == s.total_steps { json!(1.7) } else { Value::Null };
                        all.push(e);
                    }
                }
                all.reverse(); // newest first, like Unsloth
                if let Some(after) = param("after") {
                    all = all.into_iter().skip_while(|e| e["id"] != json!(after)).skip(1).collect();
                }
                let limit: usize = param("limit").and_then(|l| l.parse().ok()).unwrap_or(20);
                let more = all.len() > limit;
                all.truncate(limit);
                ev = json!({"object": "list", "data": all, "has_more": more});
                return reply(req, 200, ev);
            }
            ("POST", "cancel") => {
                let (status, _) = s.progress();
                if status == "succeeded" || status == "failed" || status == "cancelled" {
                    return error(req, 400, &format!("Job is already {status}"));
                }
                s.cancelled = true;
                let j = job_object(&s, &b);
                return reply(req, 200, j);
            }
            ("POST", "pause" | "resume") => return error(req, 400, "Pausing and resuming fine-tuning jobs is not supported; use cancel."),
            _ => {}
        }
    }
    error(req, 404, "Not Found")
}
