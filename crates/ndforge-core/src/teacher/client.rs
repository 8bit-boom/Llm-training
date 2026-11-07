//! A client for an OpenAI-compatible chat endpoint (Ollama's `/v1`, LM Studio, llama.cpp server, vLLM, a hosted API ...), with retries,
//! hidden-thinking removal and tolerant JSON reading.
use std::time::Duration;

use serde_json::{json, Value};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeacherConfig {
    /// "http://localhost:11434" or "http://truenas:11434/v1" or "https://api.example.com/v1"
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    /// the teacher is an Ollama server: take the model out of memory when the work is done
    pub unload_after: bool,
    /// milliseconds to wait before the first retry (doubles each time); tests set it low
    pub retry_ms: u64,
}

impl TeacherConfig {
    pub fn new(url: &str, model: &str, api_key: Option<String>) -> Self {
        TeacherConfig { url: url.trim().to_string(), model: model.trim().to_string(), api_key: api_key.filter(|k| !k.trim().is_empty()), unload_after: false, retry_ms: 1500 }
    }

    /// The chat-completions address: a missing "http://" and a missing "/v1" are added.
    pub fn endpoint(&self) -> String {
        let mut u = self.url.trim().trim_end_matches('/').to_string();
        if !u.starts_with("http://") && !u.starts_with("https://") {
            u = format!("http://{u}");
        }
        if !u.ends_with("/v1") {
            u.push_str("/v1");
        }
        format!("{u}/chat/completions")
    }

    /// Ollama's own address (for taking the teacher out of memory afterwards): the endpoint without "/v1/chat/completions".
    pub fn ollama_base(&self) -> String {
        self.endpoint().trim_end_matches("/chat/completions").trim_end_matches("/v1").to_string()
    }

    pub fn is_set(&self) -> bool {
        !self.url.is_empty() && !self.model.is_empty()
    }
}

pub struct Teacher {
    pub cfg: TeacherConfig,
    agent: ureq::Agent,
}

/// Removes `<think>...</think>` blocks (reasoning models) and a dangling start.
pub fn strip_thinking(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find("<think>") {
        out.push_str(&rest[..i]);
        match rest[i..].find("</think>") {
            Some(j) => rest = &rest[i + j + "</think>".len()..],
            None => return out.trim().to_string(),
        }
    }
    out.push_str(rest);
    match out.find("</think>") { Some(i) => out[i + "</think>".len()..].trim().to_string(), None => out.trim().to_string() }
}

/// The first JSON value in a model's answer: the whole text, a fenced block, or the outermost {...} / [...].
pub fn extract_json(text: &str) -> Option<Value> {
    let t = strip_thinking(text);
    if let Ok(v) = serde_json::from_str::<Value>(t.trim()) {
        return Some(v);
    }
    if let Some(start) = t.find("```") {
        let after = &t[start + 3..];
        let after = after.strip_prefix("json").or_else(|| after.strip_prefix("JSON")).unwrap_or(after);
        if let Some(end) = after.find("```") {
            if let Ok(v) = serde_json::from_str::<Value>(after[..end].trim()) {
                return Some(v);
            }
        }
    }
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(a), Some(b)) = (t.find(open), t.rfind(close)) {
            if a < b {
                if let Ok(v) = serde_json::from_str::<Value>(&t[a..=b]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

impl Teacher {
    pub fn new(cfg: TeacherConfig) -> Self {
        Teacher { cfg, agent: ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout_read(Duration::from_secs(900)).build() }
    }

    fn once(&self, system: &str, user: &str, temperature: f64, max_tokens: u32) -> std::result::Result<String, (bool, Error)> {
        let mut req = self.agent.post(&self.cfg.endpoint());
        if let Some(k) = &self.cfg.api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        let body = json!({"model": self.cfg.model, "temperature": temperature, "max_tokens": max_tokens, "stream": false,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]});
        match req.send_json(body) {
            Ok(r) => {
                let v: Value = r.into_json().map_err(|e| (true, Error::new(format!("The teacher's answer was not readable: {e}"))))?;
                let text = v["choices"][0]["message"]["content"].as_str().map(String::from);
                match text {
                    Some(t) => Ok(strip_thinking(&t)),
                    None => Err((true, Error::new(format!("The teacher answered without text: {}", v["error"]["message"].as_str().unwrap_or("unexpected answer"))))),
                }
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                let msg = serde_json::from_str::<Value>(&body).ok().and_then(|v| v["error"]["message"].as_str().or(v["error"].as_str()).map(String::from)).unwrap_or(body);
                let hint = match code {
                    401 | 403 => " (check the access key)",
                    404 => " (is the model name right, and is the model downloaded?)",
                    _ => "",
                };
                Err((code == 429 || code >= 500, Error::new(format!("The teacher refused the request ({code}){hint}: {}", msg.trim()))))
            }
            Err(ureq::Error::Transport(t)) => Err((true, Error::new(format!(
                "The teacher did not answer at {}. Is the server running (for Ollama: `ollama serve`)? [{t}]", self.cfg.endpoint())))),
        }
    }

    /// One question to the teacher; retries a few times on network trouble, rate limits and server errors.
    pub fn ask(&self, system: &str, user: &str, temperature: f64, max_tokens: u32) -> Result<String> {
        let mut wait = self.cfg.retry_ms;
        let mut last = Error::new("The teacher did not answer.");
        for attempt in 0..4 {
            match self.once(system, user, temperature, max_tokens) {
                Ok(t) => return Ok(t),
                Err((retry, e)) => {
                    last = e;
                    if !retry || attempt == 3 {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(wait));
                    wait *= 2;
                }
            }
        }
        Err(last)
    }

    /// A short check that the server answers and the model exists.
    pub fn check(&self) -> Result<String> {
        self.ask("Reply with the single word: ready", "Say it.", 0.0, 16)
    }
}
