//! The client of Unsloth's HTTP API (see `docs/UNSLOTH_API_CONTRACT.md`). Synchronous, like the other clients in this program. The API key goes into
//! the `Authorization` header and nowhere else: not into error texts, not into logs.
use std::io::{Cursor, Read};
use std::path::Path;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::export::ollama::normalize_url;

/// What `GET /api/health` says about the program that answers.
const SERVICE: &str = "Unsloth UI Backend";

/// What a probe found out. `key_ok` is None when no key was given.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Default)]
pub struct Probe {
    pub unsloth: bool,
    /// the Desktop app owns the backend (and not a Studio started from a terminal)
    pub desktop: bool,
    pub key_ok: Option<bool>,
    pub version: Option<String>,
    /// the OpenAI-shaped fine-tuning routes exist
    pub fine_tuning: bool,
    /// the list of run folders with their paths exists
    pub checkpoints: bool,
    /// the export routes exist
    pub export: bool,
    pub chat_only: Option<bool>,
    pub cpu_training: bool,
}

pub struct UnslothClient {
    base: String,
    key: String,
    agent: ureq::Agent,
}

/// Takes anything that looks like an Unsloth key out of a text.
pub fn scrub(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find("sk-unsloth-") {
        out.push_str(&rest[..i]);
        out.push_str("[key]");
        rest = rest[i..].trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    }
    out.push_str(rest);
    out
}

fn is_local(base: &str) -> bool {
    let host = base.trim_start_matches("http://").trim_start_matches("https://");
    host.starts_with("127.") || host.starts_with("localhost") || host.starts_with("[::1]")
}

/// The message of an Unsloth error body (`{"detail": "..."}`, or a plain text).
fn detail(body: &str) -> String {
    let v: Option<Value> = serde_json::from_str(body).ok();
    let text = v.as_ref().and_then(|v| v["detail"].as_str().or(v["error"]["message"].as_str()).or(v["message"].as_str())).map(String::from).unwrap_or_else(|| body.to_string());
    scrub(text.trim())
}

/// Plain messages for what can go wrong (the table in the contract document).
pub fn failure(base: &str, e: ureq::Error) -> Error {
    match e {
        ureq::Error::Transport(t) => {
            let lan = if is_local(base) { String::new() } else {
                " If Unsloth Desktop runs on another computer: Desktop answers on its own computer only until you open Settings -> API -> Remote & LAN in Unsloth and press Start on LAN access.".to_string()
            };
            Error::new(format!("Unsloth does not answer at {base}. Open Unsloth Desktop (it serves only while it is open) and check the address in Unsloth -> Settings -> API.{lan} [{}]", scrub(&t.to_string())))
        }
        ureq::Error::Status(401 | 403, _) => Error::new("Unsloth refused the API key. Create a new one in Unsloth (Settings -> API) and paste it here; Unsloth shows it only once."),
        ureq::Error::Status(404, _) => Error::new("This Unsloth has no training API for other programs. Update it (Settings -> General -> Check for updates), or use Export for Unsloth Studio."),
        ureq::Error::Status(code, resp) => {
            let msg = detail(&resp.into_string().unwrap_or_default());
            if let Some(names) = msg.strip_prefix("Unknown parameter(s):") {
                return Error::new(format!("This Unsloth does not know the setting {}. Update Unsloth, or report it: Lore Forge sent settings for another version.", names.trim()));
            }
            if code == 409 {
                return Error::new(format!("Unsloth said: {msg}. If this is about the model folder, train it on this computer instead."));
            }
            Error::new(format!("Unsloth said ({code}): {msg}"))
        }
    }
}

/// A job id goes into a URL path: letters, digits, `_` and `-` only.
pub fn safe_id(id: &str) -> Result<&str> {
    if !id.is_empty() && id.len() <= 128 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') { Ok(id) } else { Err(Error::new("Unsloth sent a job id that Lore Forge cannot use.")) }
}

/// The addresses "Find" tries: the one in `NDFORGE_UNSLOTH_URL`, then the ports Studio and Desktop are known to use on this computer.
pub fn candidate_urls() -> Vec<String> {
    let mut v: Vec<String> = std::env::var("NDFORGE_UNSLOTH_URL").ok().filter(|u| !u.trim().is_empty()).map(|u| vec![normalize_url(&u)]).unwrap_or_default();
    for u in ["http://127.0.0.1:8888", "http://127.0.0.1:8000"] {
        if !v.iter().any(|x| x == u) {
            v.push(u.to_string());
        }
    }
    v
}

struct Counting<'a, R: Read> {
    inner: R,
    done: u64,
    total: u64,
    last: u64,
    on_progress: &'a mut dyn FnMut(f64),
}

impl<R: Read> Read for Counting<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.done += n as u64;
        if self.done - self.last > self.total / 50 + 1 {
            self.last = self.done;
            (self.on_progress)(self.done as f64 / self.total.max(1) as f64);
        }
        Ok(n)
    }
}

impl UnslothClient {
    pub fn new(url: &str, key: &str) -> UnslothClient {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(5)).timeout_read(Duration::from_secs(30)).timeout_write(Duration::from_secs(300)).build();
        UnslothClient { base: normalize_url(url), key: key.trim().to_string(), agent }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn req(&self, method: &str, path: &str, keyed: bool) -> ureq::Request {
        let r = self.agent.request(method, &format!("{}{path}", self.base));
        if keyed && !self.key.is_empty() { r.set("Authorization", &format!("Bearer {}", self.key)) } else { r }
    }

    fn json(&self, r: std::result::Result<ureq::Response, ureq::Error>) -> Result<Value> {
        r.map_err(|e| failure(&self.base, e))?.into_json().map_err(|e| Error::new(format!("Unexpected answer from Unsloth: {}", scrub(&e.to_string()))))
    }

    fn status_of(&self, path: &str, keyed: bool) -> Result<u16> {
        match self.req("GET", path, keyed).call() {
            Ok(r) => Ok(r.status()),
            Err(ureq::Error::Status(c, _)) => Ok(c),
            Err(e) => Err(failure(&self.base, e)),
        }
    }

    /// Is it Unsloth, is it Desktop, does the key work, which features are there? Errors only when nothing answers.
    pub fn probe(&self) -> Result<Probe> {
        let plain = self.json(self.req("GET", "/api/health", false).call())?;
        let mut p = Probe { unsloth: plain["service"] == SERVICE, ..Probe::default() };
        if !p.unsloth {
            return Ok(p);
        }
        p.desktop = !plain["desktop_owner"].is_null() || !plain["desktop_protocol_version"].is_null();
        p.chat_only = plain["chat_only"].as_bool();
        if !self.key.is_empty() {
            // a wrong key does not give 401 on /api/health: it gives the answer without a version
            let keyed = self.json(self.req("GET", "/api/health", true).call())?;
            p.version = keyed["version"].as_str().or(keyed["studio_version"].as_str()).map(String::from);
            p.cpu_training = keyed["cpu_training"].as_bool().unwrap_or(false);
            p.key_ok = Some(p.version.is_some() && self.status_of("/v1/models", true)? == 200);
        }
        if p.key_ok == Some(true) {
            p.fine_tuning = self.status_of("/v1/fine_tuning/jobs?limit=1", true)? == 200;
            p.checkpoints = self.status_of("/api/models/checkpoints", true)? == 200;
            p.export = self.status_of("/api/export/status", true)? == 200;
        }
        Ok(p)
    }

    /// `{"gpu": {"name", "vram_gb", "precision"}}` in the shape of the training server's status, or `{"gpu": null}`. The device keys are read loosely.
    pub fn hardware(&self) -> Result<Value> {
        let v = self.json(self.req("GET", "/api/train/hardware", true).call())?;
        let dev = v["devices"].as_array().and_then(|d| d.first()).cloned().unwrap_or(Value::Null);
        if dev.is_null() {
            return Ok(serde_json::json!({"gpu": null}));
        }
        let name = dev["name"].as_str().or(dev["device_name"].as_str()).unwrap_or("graphics card");
        let gb = dev["total_memory_gb"].as_f64().or(dev["vram_gb"].as_f64()).or(dev["total_gb"].as_f64()).or(dev["total_memory"].as_f64().map(|b| if b > 1e6 { b / 1073741824.0 } else { b }));
        let precision = match dev["bf16"].as_bool() { Some(true) => "bf16", Some(false) => "fp16", None => "unknown" };
        Ok(serde_json::json!({"gpu": {"name": name, "vram_gb": gb.map(|g| (g * 10.0).round() / 10.0), "precision": precision}}))
    }

    /// Uploads a JSONL file for fine-tuning; returns Unsloth's file id.
    pub fn upload_jsonl(&self, name: &str, path: &Path, on_progress: &mut dyn FnMut(f64)) -> Result<String> {
        let size = std::fs::metadata(path)?.len();
        let boundary = format!("----ndforge{:016x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(7));
        let safe_name: String = name.chars().filter(|c| c.is_ascii_alphanumeric() || "._-".contains(*c)).collect();
        let head = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nfine-tune\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe_name}\"\r\nContent-Type: application/jsonl\r\n\r\n");
        let tail = format!("\r\n--{boundary}--\r\n");
        let total = head.len() as u64 + size + tail.len() as u64;
        let body = Counting { inner: Cursor::new(head.into_bytes()).chain(std::fs::File::open(path)?).chain(Cursor::new(tail.into_bytes())), done: 0, total, last: 0, on_progress };
        let v = self.json(self.req("POST", "/v1/files", true).timeout(Duration::from_secs(300)).set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
            .set("Content-Length", &total.to_string()).send(body))?;
        v["id"].as_str().map(String::from).ok_or_else(|| Error::new("Unsloth did not give the uploaded file an id."))
    }

    pub fn start_job(&self, body: &Value) -> Result<Value> {
        self.json(self.req("POST", "/v1/fine_tuning/jobs", true).send_json(body.clone()))
    }

    /// The job, None when Unsloth does not know it (a waiting job is forgotten when Unsloth restarts).
    pub fn job(&self, id: &str) -> Result<Option<Value>> {
        match self.req("GET", &format!("/v1/fine_tuning/jobs/{}", safe_id(id)?), true).call() {
            Ok(r) => Ok(Some(r.into_json().map_err(|e| Error::new(format!("Unexpected answer from Unsloth: {}", scrub(&e.to_string()))))?)),
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(failure(&self.base, e)),
        }
    }

    /// One page of events, NEWEST first, and whether older ones exist. `after` is the id of the oldest event of the page before (Unsloth then gives older ones).
    pub fn events_page(&self, id: &str, limit: u32, after: Option<&str>) -> Result<(Vec<Value>, bool)> {
        let mut path = format!("/v1/fine_tuning/jobs/{}/events?limit={}", safe_id(id)?, limit.clamp(1, 1000));
        if let Some(a) = after {
            path.push_str(&format!("&after={}", safe_id(a)?));
        }
        let v = self.json(self.req("GET", &path, true).call())?;
        Ok((v["data"].as_array().cloned().unwrap_or_default(), v["has_more"].as_bool().unwrap_or(false)))
    }

    /// The newest `limit` events, oldest first.
    pub fn events(&self, id: &str, limit: u32) -> Result<Vec<Value>> {
        let (mut ev, _) = self.events_page(id, limit, None)?;
        ev.reverse();
        Ok(ev)
    }

    pub fn cancel(&self, id: &str) -> Result<Value> {
        self.json(self.req("POST", &format!("/v1/fine_tuning/jobs/{}/cancel", safe_id(id)?), true).send_string(""))
    }

    /// `GET /api/models/checkpoints`: every run folder with its checkpoints and host paths. The only `/api/*` route the run needs.
    pub fn checkpoints_listing(&self) -> Result<Value> {
        self.json(self.req("GET", "/api/models/checkpoints", true).call())
    }
}
