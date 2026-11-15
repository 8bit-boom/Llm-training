//! A stand-in for GitHub's release API and download hosts, on 127.0.0.1 (tiny_http), and builders for the tiny archives it serves.
#![allow(dead_code)]
use std::collections::HashMap;
use std::io::{Cursor, Read, Write};
use std::sync::{Arc, Mutex};

use ndforge_core::envmgr::llamacpp::archive::{hex, sha256_file};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub enum Reply {
    Json(u16, String, Vec<(&'static str, &'static str)>),
    Bytes(Vec<u8>),
    Redirect(String),
}

pub struct Fake {
    pub base: String,
    server: Arc<tiny_http::Server>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub routes: Arc<Mutex<HashMap<String, Reply>>>,
    pub hits: Arc<Mutex<Vec<String>>>,
}

impl Fake {
    pub fn start() -> Fake {
        let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").unwrap());
        let base = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
        let routes: Arc<Mutex<HashMap<String, Reply>>> = Arc::default();
        let hits: Arc<Mutex<Vec<String>>> = Arc::default();
        let (s, r, h) = (server.clone(), routes.clone(), hits.clone());
        let thread = std::thread::spawn(move || {
            for req in s.incoming_requests() {
                let path = req.url().split('?').next().unwrap_or("").to_string();
                h.lock().unwrap().push(path.clone());
                let reply = {
                    let routes = r.lock().unwrap();
                    match routes.get(&path) {
                        None => Reply::Json(404, "{\"message\":\"Not Found\"}".into(), vec![]),
                        Some(Reply::Json(c, b, hd)) => Reply::Json(*c, b.clone(), hd.clone()),
                        Some(Reply::Bytes(b)) => Reply::Bytes(b.clone()),
                        Some(Reply::Redirect(u)) => Reply::Redirect(u.clone()),
                    }
                };
                let _ = match reply {
                    Reply::Json(code, body, headers) => {
                        let mut resp = tiny_http::Response::from_string(body).with_status_code(code).with_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap());
                        for (k, v) in headers {
                            resp = resp.with_header(tiny_http::Header::from_bytes(k, v).unwrap());
                        }
                        req.respond(resp)
                    }
                    Reply::Bytes(b) => req.respond(tiny_http::Response::from_data(b)),
                    Reply::Redirect(u) => req.respond(tiny_http::Response::empty(302).with_header(tiny_http::Header::from_bytes("Location", u).unwrap())),
                };
            }
        });
        Fake { base, server, thread: Some(thread), routes, hits }
    }

    pub fn set(&self, path: &str, r: Reply) {
        self.routes.lock().unwrap().insert(path.to_string(), r);
    }

    /// Serves a release: `assets` (name, bytes), a source zip, under the real URL shapes of GitHub. `bad_digest` lists the assets whose listed checksum is wrong.
    pub fn release(&self, tag: &str, assets: &[(&str, Vec<u8>)], source: Vec<u8>, bad_digest: &[&str], latest: bool) {
        let list: Vec<Value> = assets.iter().map(|(n, b)| {
            self.set(&format!("/dl/{n}"), Reply::Bytes(b.clone()));
            let mut h = Sha256::new();
            h.update(b);
            let mut sum = hex(&h.finalize());
            if bad_digest.contains(n) {
                sum = "0".repeat(64);
            }
            json!({"name": n, "size": b.len(), "browser_download_url": format!("{}/dl/{n}", self.base), "digest": format!("sha256:{sum}")})
        }).collect();
        let body = json!({"tag_name": tag, "assets": list}).to_string();
        if latest {
            self.set("/repos/ggml-org/llama.cpp/releases/latest", Reply::Json(200, body.clone(), vec![]));
        }
        self.set(&format!("/repos/ggml-org/llama.cpp/releases/tags/{tag}"), Reply::Json(200, body, vec![]));
        self.set(&format!("/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip"), Reply::Redirect(format!("{}/codeload/{tag}.zip", self.base)));
        self.set(&format!("/codeload/{tag}.zip"), Reply::Bytes(source));
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.server.unblock();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

pub fn zip_bytes(entries: &[(&str, &[u8], Option<u32>)]) -> Vec<u8> {
    let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data, mode) in entries {
        let mut o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        if let Some(m) = mode {
            o = o.unix_permissions(*m);
        }
        if *mode == Some(0o120777) {
            w.add_symlink(*name, std::str::from_utf8(data).unwrap(), o).unwrap();
        } else if name.ends_with('/') {
            w.add_directory(*name, o).unwrap();
        } else {
            w.start_file(*name, o).unwrap();
            w.write_all(data).unwrap();
        }
    }
    w.finish().unwrap().into_inner()
}

fn octal(n: u64, width: usize) -> Vec<u8> {
    let mut v = format!("{n:0w$o}", w = width - 1).into_bytes();
    v.push(0);
    v
}

/// One tar member: `typeflag` b'0' file, b'5' dir, b'2' symlink, b'L' long name.
pub fn tar_entry(name: &str, typeflag: u8, data: &[u8], link: &str, mode: u32) -> Vec<u8> {
    let mut h = [0u8; 512];
    let nb = name.as_bytes();
    h[..nb.len().min(100)].copy_from_slice(&nb[..nb.len().min(100)]);
    h[100..108].copy_from_slice(&octal(mode as u64, 8));
    h[108..116].copy_from_slice(&octal(0, 8));
    h[116..124].copy_from_slice(&octal(0, 8));
    h[124..136].copy_from_slice(&octal(data.len() as u64, 12));
    h[136..148].copy_from_slice(&octal(0, 12));
    h[148..156].copy_from_slice(b"        ");
    h[156] = typeflag;
    let lb = link.as_bytes();
    h[157..157 + lb.len().min(100)].copy_from_slice(&lb[..lb.len().min(100)]);
    h[257..263].copy_from_slice(b"ustar\0");
    h[263..265].copy_from_slice(b"00");
    let sum: u32 = h.iter().map(|b| *b as u32).sum();
    h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    let mut out = h.to_vec();
    out.extend_from_slice(data);
    out.extend(std::iter::repeat(0u8).take((512 - data.len() % 512) % 512));
    out
}

pub fn tar_gz(entries: Vec<Vec<u8>>) -> Vec<u8> {
    let mut raw: Vec<u8> = entries.concat();
    raw.extend(std::iter::repeat(0u8).take(1024));
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(&raw).unwrap();
    e.finish().unwrap()
}

pub fn file_sha(p: &std::path::Path) -> String {
    sha256_file(p).unwrap()
}

pub fn read_all(mut r: impl Read) -> Vec<u8> {
    let mut v = Vec::new();
    r.read_to_end(&mut v).unwrap();
    v
}
