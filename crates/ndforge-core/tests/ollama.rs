use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use ndforge_core::export::install::{install, probe};
use ndforge_core::export::ollama::{file_digest, normalize_url, Ollama};
use serde_json::{json, Value};
use tempfile::tempdir;

#[derive(Debug, Clone)]
struct Req {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// A tiny HTTP server: `handler(request) -> (status, json body)`; records every request. Stops when dropped.
struct Mock {
    url: String,
    log: Arc<Mutex<Vec<Req>>>,
}

fn mock(handler: impl Fn(&Req) -> (u16, Value) + Send + 'static) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("127.0.0.1:{}", listener.local_addr().unwrap().port());
    let log: Arc<Mutex<Vec<Req>>> = Arc::default();
    let l2 = log.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some(v) = h.to_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; len];
            reader.read_exact(&mut body).unwrap();
            let req = Req { method, path, body };
            l2.lock().unwrap().push(req.clone());
            let (status, v) = handler(&req);
            let text = if req.method == "HEAD" { String::new() } else { v.to_string() };
            let _ = write!(stream, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
        }
    });
    Mock { url, log }
}

#[test]
fn urls_are_normalised() {
    assert_eq!(normalize_url("localhost:11434"), "http://localhost:11434");
    assert_eq!(normalize_url(" http://truenas:11434/ "), "http://truenas:11434");
    assert_eq!(normalize_url("https://x.example/"), "https://x.example");
}

#[test]
fn digest_matches_sha256() {
    let d = tempdir().unwrap();
    let f = d.path().join("a.bin");
    std::fs::write(&f, b"abc").unwrap();
    assert_eq!(file_digest(&f, &mut |_| {}).unwrap(), "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

fn happy_server(have_blob: bool) -> Mock {
    let created = Arc::new(Mutex::new(false));
    mock(move |r| match (r.method.as_str(), r.path.as_str()) {
        ("GET", "/api/version") => (200, json!({"version": "0.99.0"})),
        ("HEAD", p) if p.starts_with("/api/blobs/") => (if have_blob { 200 } else { 404 }, json!({})),
        ("POST", p) if p.starts_with("/api/blobs/") => (201, json!({})),
        ("POST", "/api/create") => { *created.lock().unwrap() = true; (200, json!({"status": "success"})) }
        ("GET", "/api/tags") => (200, json!({"models": if *created.lock().unwrap() { vec![json!({"name": "w-gm:latest"})] } else { vec![] }})),
        ("POST", "/api/chat") => (200, json!({"message": {"role": "assistant", "content": "Anders is a hunter."}})),
        _ => (404, json!({"error": "no such route"})),
    })
}

#[test]
fn install_uploads_creates_and_lists_then_a_probe_answers() {
    let d = tempdir().unwrap();
    let gguf = d.path().join("model-q4_k_m.gguf");
    std::fs::write(&gguf, vec![7u8; 300_000]).unwrap();
    let m = happy_server(false);
    let mut events = Vec::new();
    install(&m.url, "w-gm", &gguf, "qwen3_5", 8192, &mut |e| events.push(e)).unwrap();
    let log = m.log.lock().unwrap().clone();
    let paths: Vec<String> = log.iter().map(|r| format!("{} {}", r.method, r.path.split("sha256:").next().unwrap())).collect();
    assert_eq!(paths, ["GET /api/version", "HEAD /api/blobs/", "POST /api/blobs/", "POST /api/create", "GET /api/tags"]);
    let digest = file_digest(&gguf, &mut |_| {}).unwrap();
    assert!(log[1].path.ends_with(&digest));
    assert_eq!(log[2].body.len(), 300_000);                          // the whole file was sent
    let body: Value = serde_json::from_slice(&log[3].body).unwrap();
    assert_eq!(body["model"], "w-gm");
    assert_eq!(body["files"]["model-q4_k_m.gguf"].as_str(), Some(digest.as_str()));
    assert_eq!(body["parameters"]["stop"], json!(["<|im_end|>"]));
    assert!(body["template"].as_str().unwrap().contains("<think>"));
    assert_eq!(events.iter().filter(|e| e["kind"] == "step").count(), 4);
    assert!(events.iter().any(|e| e["kind"] == "progress"));
    assert_eq!(probe(&m.url, "w-gm", "SYS", "Who is Anders?").unwrap(), "Anders is a hunter.");
    let chat: Value = serde_json::from_slice(&m.log.lock().unwrap().last().unwrap().body).unwrap();
    assert_eq!((chat["stream"].as_bool(), chat["messages"][0]["role"].as_str(), chat["messages"][1]["content"].as_str()), (Some(false), Some("system"), Some("Who is Anders?")));
}

#[test]
fn a_blob_the_server_already_has_is_not_sent_again() {
    let d = tempdir().unwrap();
    let gguf = d.path().join("m.gguf");
    std::fs::write(&gguf, b"xyz").unwrap();
    let m = happy_server(true);
    install(&m.url, "w-gm", &gguf, "gemma4", 4096, &mut |_| {}).unwrap();
    assert!(!m.log.lock().unwrap().iter().any(|r| r.method == "POST" && r.path.starts_with("/api/blobs/")));
}

#[test]
fn problems_are_explained() {
    // nothing listening
    let e = Ollama::new("127.0.0.1:1").version().unwrap_err();
    assert!(e.0.contains("did not answer") && e.0.contains("OLLAMA_HOST"), "{e}");
    // the server refuses the model
    let m = mock(|r| match r.path.as_str() {
        "/api/version" => (200, json!({"version": "0.1"})),
        p if p.starts_with("/api/blobs/") => (if r.method == "HEAD" { 404 } else { 201 }, json!({})),
        "/api/create" => (400, json!({"error": "unsupported architecture"})),
        _ => (404, json!({})),
    });
    let d = tempdir().unwrap();
    let f = d.path().join("m.gguf");
    std::fs::write(&f, b"x").unwrap();
    let e = install(&m.url, "w-gm", &f, "qwen3_5", 4096, &mut |_| {}).unwrap_err();
    assert!(e.0.contains("refused") && e.0.contains("unsupported architecture"), "{e}");
    // created but not listed
    let m2 = mock(|r| match (r.method.as_str(), r.path.as_str()) {
        ("GET", "/api/version") => (200, json!({"version": "0.1"})),
        ("HEAD", _) => (200, json!({})),
        ("POST", "/api/create") => (200, json!({"status": "success"})),
        ("GET", "/api/tags") => (200, json!({"models": []})),
        _ => (404, json!({})),
    });
    assert!(install(&m2.url, "w-gm", &f, "qwen3_5", 4096, &mut |_| {}).unwrap_err().0.contains("does not list it"));
    // a chat error
    let m3 = mock(|_| (200, json!({"error": "model not found"})));
    assert!(probe(&m3.url, "x", "s", "q").unwrap_err().0.contains("model not found"));
}
