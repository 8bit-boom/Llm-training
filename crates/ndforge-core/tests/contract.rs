//! The words the Rust program and nd-world share with the worker, and the Unsloth fixtures nd-world copies.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn manifest() -> Option<Value> {
    let f = repo().join("worker").join("ndforge_worker").join("contract.json");
    serde_json::from_slice(&std::fs::read(f).ok()?).ok()
}

fn rust_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Every `"job": "<name>"` the program writes into a job file.
fn job_literals() -> Vec<(String, String)> {
    let mut files = Vec::new();
    rust_files(&repo().join("crates").join("ndforge-core").join("src"), &mut files);
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let mut rest = text.as_str();
        while let Some(i) = rest.find("\"job\": \"") {
            let tail = &rest[i + 8..];
            if let Some(end) = tail.find('"') {
                out.push((tail[..end].to_string(), f.file_name().unwrap().to_string_lossy().to_string()));
            }
            rest = &rest[i + 8..];
        }
    }
    out
}

#[test]
fn every_job_the_program_sends_is_one_the_worker_declares() {
    let Some(m) = manifest() else { return };      // the worker is not next to the program here (the program is built from a package)
    let known: Vec<&str> = m["jobs"].as_array().unwrap().iter().filter_map(|j| j.as_str()).collect();
    let lits = job_literals();
    assert!(lits.len() > 20, "the scan found the job literals: {}", lits.len());
    for (job, file) in lits {
        assert!(known.contains(&job.as_str()), "{file} sends the job \"{job}\", which worker/ndforge_worker/contract.json does not list. Add it there (new names are fine; the worker test checks the code).");
    }
}

#[test]
fn the_manifest_names_the_protocol_the_program_understands() {
    let Some(m) = manifest() else { return };
    assert_eq!(m["protocol"], 1, "the program reads protocol 1; raise both together");
    for k in ["jobs", "events", "error_codes"] {
        assert!(m[k].as_array().is_some_and(|a| !a.is_empty()), "{k}");
    }
    // the words the program itself looks for in what the worker says
    let codes: Vec<&str> = m["error_codes"].as_array().unwrap().iter().filter_map(|c| c.as_str()).collect();
    for c in ndforge_core::envmgr::jobs::FIX_CODES.iter().chain([&ndforge_core::envmgr::jobs::LLAMA_CPP_MISSING]) {
        assert!(codes.contains(c), "the program offers a fix for {c}, but the worker does not report it any more");
    }
    let events: Vec<&str> = m["events"].as_array().unwrap().iter().filter_map(|c| c.as_str()).collect();
    for e in ["metric", "sample", "start", "log", "progress", "result", "error"] {
        assert!(events.contains(&e), "{e}");
    }
}

#[test]
fn the_unsloth_fixtures_nd_world_copies_are_the_ones_recorded_in_sha256sums() {
    let dir = repo().join("tests").join("fixtures").join("unsloth");
    let sums = std::fs::read_to_string(dir.join("SHA256SUMS")).expect("SHA256SUMS next to the fixtures");
    let mut listed = Vec::new();
    for line in sums.lines().filter(|l| !l.trim().is_empty()) {
        let (hash, name) = line.split_once("  ").expect("sha256sum format");
        // a checkout on Windows may turn the line endings into CRLF: the sums are of the text with LF
        let bytes: Vec<u8> = std::fs::read(dir.join(name)).unwrap_or_else(|_| panic!("{name} is listed but missing"));
        let text = String::from_utf8(bytes).unwrap().replace("\r\n", "\n");
        let got: String = Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, hash, "{name} changed: nd-world has a copy of it. Change both, then run `sha256sum *.json | sort -k2 > SHA256SUMS` in tests/fixtures/unsloth");
        listed.push(name.to_string());
    }
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".json")).collect();
    on_disk.sort();
    listed.sort();
    assert_eq!(on_disk, listed, "every fixture is listed in SHA256SUMS");
}
