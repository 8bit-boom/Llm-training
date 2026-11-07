//! A small client for Ollama's HTTP API: enough to put a GGUF on a server (this PC or the TrueNAS box), create the model and ask it a question.
//! Endpoints (Ollama API): `GET /api/version`, `GET /api/tags`, `HEAD|POST /api/blobs/sha256:<digest>`, `POST /api/create`, `POST /api/chat`, `DELETE /api/delete`.
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

pub struct Ollama {
    base: String,
    agent: ureq::Agent,
}

/// "localhost:11434" and "http://truenas:11434/" both work.
pub fn normalize_url(url: &str) -> String {
    let u = url.trim().trim_end_matches('/');
    if u.starts_with("http://") || u.starts_with("https://") { u.to_string() } else { format!("http://{u}") }
}

fn unreachable(base: &str, e: ureq::Error) -> Error {
    match e {
        ureq::Error::Transport(t) => Error::new(format!(
            "Ollama did not answer at {base}. Is it running? (Install it from https://ollama.com and start it. For a server on another computer, Ollama must listen on the network: set OLLAMA_HOST=0.0.0.0 there.) [{t}]")),
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&body).ok().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or(body);
            Error::new(format!("Ollama refused the request ({code}): {}", msg.trim()))
        }
    }
}

/// Reads a file while counting what has gone by (for the progress bar of a multi-GB upload).
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
        if self.done - self.last > self.total / 100 + 1 {
            self.last = self.done;
            (self.on_progress)(self.done as f64 / self.total.max(1) as f64);
        }
        Ok(n)
    }
}

/// SHA-256 of a file as "sha256:<hex>", with progress.
pub fn file_digest(path: &Path, on_progress: &mut dyn FnMut(f64)) -> Result<String> {
    let total = std::fs::metadata(path)?.len();
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    let (mut done, mut last) = (0u64, 0u64);
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as u64;
        if done - last > total / 100 + 1 {
            last = done;
            on_progress(done as f64 / total.max(1) as f64);
        }
    }
    Ok(format!("sha256:{:x}", h.finalize()))
}

impl Ollama {
    pub fn new(url: &str) -> Ollama {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(5)).timeout_read(Duration::from_secs(900)).timeout_write(Duration::from_secs(900)).build();
        Ollama { base: normalize_url(url), agent }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn version(&self) -> Result<String> {
        let v: Value = self.agent.get(&format!("{}/api/version", self.base)).call().map_err(|e| unreachable(&self.base, e))?.into_json().map_err(|e| Error::new(format!("Unexpected answer from Ollama: {e}")))?;
        Ok(v["version"].as_str().unwrap_or("?").to_string())
    }

    /// Names of the models on the server.
    pub fn models(&self) -> Result<Vec<String>> {
        let v: Value = self.agent.get(&format!("{}/api/tags", self.base)).call().map_err(|e| unreachable(&self.base, e))?.into_json().map_err(|e| Error::new(format!("Unexpected answer from Ollama: {e}")))?;
        Ok(v["models"].as_array().map(|a| a.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default())
    }

    pub fn blob_exists(&self, digest: &str) -> Result<bool> {
        match self.agent.head(&format!("{}/api/blobs/{digest}", self.base)).call() {
            Ok(_) => Ok(true),
            Err(ureq::Error::Status(404, _)) => Ok(false),
            Err(e) => Err(unreachable(&self.base, e)),
        }
    }

    pub fn upload_blob(&self, digest: &str, path: &Path, on_progress: &mut dyn FnMut(f64)) -> Result<()> {
        let total = std::fs::metadata(path)?.len();
        let reader = Counting { inner: std::fs::File::open(path)?, done: 0, total, last: 0, on_progress };
        self.agent.post(&format!("{}/api/blobs/{digest}", self.base)).set("Content-Length", &total.to_string()).send(reader).map_err(|e| unreachable(&self.base, e))?;
        Ok(())
    }

    pub fn create(&self, body: &Value) -> Result<()> {
        let v: Value = self.agent.post(&format!("{}/api/create", self.base)).send_json(body.clone()).map_err(|e| unreachable(&self.base, e))?.into_json().unwrap_or(Value::Null);
        if let Some(err) = v["error"].as_str() {
            return Err(Error::new(format!("Ollama could not create the model: {err}")));
        }
        Ok(())
    }

    /// One question, one answer (not streamed).
    pub fn chat(&self, model: &str, messages: &[Value], options: &Value) -> Result<String> {
        let v: Value = self.agent.post(&format!("{}/api/chat", self.base))
            .send_json(json!({"model": model, "messages": messages, "stream": false, "options": options})).map_err(|e| unreachable(&self.base, e))?
            .into_json().map_err(|e| Error::new(format!("Unexpected answer from Ollama: {e}")))?;
        if let Some(err) = v["error"].as_str() {
            return Err(Error::new(format!("Ollama could not answer: {err}")));
        }
        Ok(v["message"]["content"].as_str().unwrap_or("").to_string())
    }

    pub fn delete(&self, model: &str) -> Result<()> {
        self.agent.delete(&format!("{}/api/delete", self.base)).send_json(json!({"model": model})).map_err(|e| unreachable(&self.base, e))?;
        Ok(())
    }
}
