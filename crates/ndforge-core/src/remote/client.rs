//! The client side of the worker protocol (see `remote/mod.rs`).
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::export::ollama::normalize_url;

pub struct RemoteClient {
    base: String,
    token: String,
    agent: ureq::Agent,
}

fn failure(base: &str, e: ureq::Error) -> Error {
    match e {
        ureq::Error::Transport(t) => Error::new(format!(
            "The training server did not answer at {base}. Is the worker container running, and is the address and port right? [{t}]")),
        ureq::Error::Status(401, _) => Error::new("The training server refused the access token. Check the token you entered (it is the NDFORGE_TOKEN of the worker)."),
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let msg = serde_json::from_str::<Value>(&body).ok().and_then(|v| v["error"].as_str().map(String::from)).unwrap_or(body);
            Error::new(format!("The training server said ({code}): {}", msg.trim()))
        }
    }
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

impl RemoteClient {
    pub fn new(url: &str, token: &str) -> RemoteClient {
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(5)).timeout_read(Duration::from_secs(120)).timeout_write(Duration::from_secs(600)).build();
        RemoteClient { base: normalize_url(url), token: token.trim().to_string(), agent }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn req(&self, method: &str, path: &str) -> ureq::Request {
        self.agent.request(method, &format!("{}{path}", self.base)).set("Authorization", &format!("Bearer {}", self.token))
    }

    fn json(&self, r: std::result::Result<ureq::Response, ureq::Error>) -> Result<Value> {
        r.map_err(|e| failure(&self.base, e))?.into_json().map_err(|e| Error::new(format!("Unexpected answer from the training server: {e}")))
    }

    pub fn status(&self) -> Result<Value> {
        self.json(self.req("GET", "/v1/status").call())
    }

    pub fn upload(&self, job: &str, name: &str, path: &Path, on_progress: &mut dyn FnMut(f64)) -> Result<()> {
        let total = std::fs::metadata(path)?.len();
        let reader = Counting { inner: std::fs::File::open(path)?, done: 0, total, last: 0, on_progress };
        self.req("PUT", &format!("/v1/jobs/{job}/files/{name}")).set("Content-Length", &total.to_string()).send(reader).map_err(|e| failure(&self.base, e))?;
        Ok(())
    }

    pub fn start(&self, job: &str, spec: &Value) -> Result<()> {
        self.json(self.req("POST", &format!("/v1/jobs/{job}")).send_json(serde_json::json!({"job": spec}))).map(|_| ())
    }

    /// The job's state on the server, None when it does not know the job.
    pub fn job_state(&self, job: &str) -> Result<Option<Value>> {
        match self.req("GET", &format!("/v1/jobs/{job}")).call() {
            Ok(r) => Ok(Some(r.into_json().map_err(|e| Error::new(format!("Unexpected answer from the training server: {e}")))?)),
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(failure(&self.base, e)),
        }
    }

    /// (events from `from`, index of the next event, state: running / done / stopped / failed)
    pub fn events(&self, job: &str, from: usize) -> Result<(Vec<Value>, usize, String)> {
        let v = self.json(self.req("GET", &format!("/v1/jobs/{job}/events?from={from}")).call())?;
        Ok((v["events"].as_array().cloned().unwrap_or_default(), v["next"].as_u64().unwrap_or(from as u64) as usize, v["state"].as_str().unwrap_or("running").to_string()))
    }

    pub fn stop(&self, job: &str) -> Result<()> {
        self.json(self.req("POST", &format!("/v1/jobs/{job}/stop")).call()).map(|_| ())
    }

    pub fn download_adapter(&self, job: &str, dest: &Path) -> Result<()> {
        let resp = self.req("GET", &format!("/v1/jobs/{job}/adapter.zip")).timeout(Duration::from_secs(3600)).call().map_err(|e| failure(&self.base, e))?;
        let mut f = std::fs::File::create(dest)?;
        std::io::copy(&mut resp.into_reader(), &mut f)?;
        Ok(())
    }

    pub fn unload_ollama(&self) -> Result<Vec<String>> {
        let v = self.json(self.req("POST", "/v1/ollama/unload").call())?;
        Ok(v["unloaded"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default())
    }

    pub fn delete(&self, job: &str) -> Result<()> {
        self.json(self.req("DELETE", &format!("/v1/jobs/{job}")).call()).map(|_| ())
    }
}
