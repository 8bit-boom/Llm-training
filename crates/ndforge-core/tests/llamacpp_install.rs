//! llama.cpp installer, the whole flow against a stand-in GitHub on 127.0.0.1: a release list, a tiny program-files archive, a tiny source archive. Failures: not found,
//! rate limit, cut-off download, wrong checksum, a program that does not start, a cancel in each phase. The old install must survive every failure.
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use ndforge_core::envmgr::llamacpp::choose::*;
use ndforge_core::envmgr::llamacpp::*;
use serde_json::{json, Value};
use tempfile::tempdir;

mod common;
use common::fake_github::*;

static ENV: Mutex<()> = Mutex::new(());

const SOURCE: &[(&str, &[u8], Option<u32>)] = &[
    ("llama.cpp-b6000/", b"", None), ("llama.cpp-b6000/convert_hf_to_gguf.py", b"import torch\nfrom conversion import ModelBase\n", None), ("llama.cpp-b6000/conversion/__init__.py", b"", None),
    ("llama.cpp-b6000/gguf-py/gguf/__init__.py", b"", None), ("llama.cpp-b6000/models/templates/a.jinja", b"{{x}}", None), ("llama.cpp-b6000/docs/big.md", b"x", None), ("llama.cpp-b6000/CMakeLists.txt", b"project(x)", None),
];

fn source() -> Vec<u8> {
    zip_bytes(SOURCE)
}

fn bin_zip(prefix: &str) -> Vec<u8> {
    let names: Vec<String> = ["llama-quantize", "llama-imatrix", "llama-perplexity", "llama-cli"].iter().map(|n| format!("{prefix}{}", exe(n))).collect();
    let entries: Vec<(&str, &[u8], Option<u32>)> = names.iter().map(|n| (n.as_str(), &b"stub"[..], Some(0o755))).collect();
    zip_bytes(&entries)
}

fn cudart_zip() -> Vec<u8> {
    zip_bytes(&[("cudart64_12.dll", b"d", None), ("cublas64_12.dll", b"d", None), ("cublasLt64_12.dll", b"d", None), ("README.txt", b"x", None)])
}

fn host_asset() -> &'static str {
    match Os::current() { Os::Windows => "llama-b6000-bin-win-cpu-x64.zip", Os::Macos => "llama-b6000-bin-macos-x64.zip", _ => "llama-b6000-bin-ubuntu-x64.zip" }
}

fn host() -> Machine {
    Machine::cpu_only(Os::current(), Arch::X64)
}

fn cfg(d: &Path, fake: &Fake) -> Config {
    Config { base_dir: d.join("data"), api_base: fake.base.clone(), web_base: fake.base.clone(), allow_local: true, tag_pin: None, python: None, home: Some(d.join("home")), worker_dir: d.join("app/worker") }
}

fn opts(want: Want) -> Options {
    Options { want, tag: None }
}

#[derive(Default)]
struct FakeRunner {
    /// programs whose name contains this fail to start (a missing library)
    broken: Mutex<Option<String>>,
    calls: Mutex<Vec<String>>,
    python_out: Mutex<Option<(i32, String)>>,
    cmake_fails: Mutex<bool>,
    cancel_in_build: Mutex<bool>,
}

impl Runner for FakeRunner {
    fn run(&self, program: &Path, args: &[String], _cwd: Option<&Path>, _t: Duration, _c: &AtomicBool) -> RunOut {
        let name = program.file_name().unwrap().to_string_lossy().to_string();
        self.calls.lock().unwrap().push(format!("{name} {}", args.join(" ").replace('\\', "/").rsplit('/').next().unwrap_or("")));
        if name.contains("python") {
            let (code, out) = self.python_out.lock().unwrap().clone().unwrap_or((0, "usage: convert_hf_to_gguf.py".into()));
            return RunOut { started: true, code: Some(code), output: out, timed_out: false };
        }
        if self.broken.lock().unwrap().as_deref().is_some_and(|b| name.contains(b)) {
            return RunOut { started: true, code: Some(127), output: "./llama-imatrix: error while loading shared libraries: libcudart.so.12: cannot open shared object file".into(), timed_out: false };
        }
        if args.first().map(String::as_str) == Some("--version") {
            return RunOut { started: true, code: Some(0), output: "version: 6000 (abc1234)\nbuilt with gcc".into(), timed_out: false };
        }
        RunOut { started: true, code: Some(if name.contains("quantize") { 1 } else { 0 }), output: format!("usage: {name} [options]"), timed_out: false }
    }

    fn stream(&self, _program: &Path, args: &[String], _cwd: Option<&Path>, cancel: &AtomicBool, on_line: &mut dyn FnMut(&str)) -> Result<Option<i32>, Failure> {
        self.calls.lock().unwrap().push(format!("cmake {}", args.first().cloned().unwrap_or_default()));
        if args[0] == "--build" {
            if *self.cancel_in_build.lock().unwrap() {
                cancel.store(true, Ordering::SeqCst);
                return Err(Failure { code: "cancelled", message: "Cancelled.".into() });
            }
            on_line("[ 10%] Building CXX object ggml");
            on_line("[ 50%] Linking CXX executable bin/llama-quantize");
            on_line("[100%] Built target llama-perplexity");
            let bp = Path::new(&args[1]).join("bin");
            std::fs::create_dir_all(&bp).unwrap();
            for n in ["llama-quantize", "llama-imatrix", "llama-perplexity"] {
                std::fs::write(bp.join(exe(n)), "built").unwrap();
            }
            return Ok(Some(0));
        }
        on_line("-- The CXX compiler identification is GNU");
        Ok(Some(if *self.cmake_fails.lock().unwrap() { 1 } else { 0 }))
    }
}

fn run(cfg: &Config, m: &Machine, o: &Options, r: &FakeRunner) -> (Result<Value, ndforge_core::error::Error>, Vec<Value>) {
    run_with_cancel(cfg, m, o, r, &AtomicBool::new(false), &mut |_| {})
}

fn run_with_cancel(cfg: &Config, m: &Machine, o: &Options, r: &FakeRunner, cancel: &AtomicBool, hook: &mut dyn FnMut(&Value)) -> (Result<Value, ndforge_core::error::Error>, Vec<Value>) {
    let mut events = Vec::new();
    let res = run_install(cfg, m, o, r, cancel, &mut |e| {
        hook(&e);
        events.push(e);
    });
    (res, events)
}

fn serve_host(fake: &Fake, tag: &str) {
    fake.release(tag, &[(host_asset(), bin_zip(""))], source(), &[], true);
}

fn old_install(c: &Config) {
    let d = c.install_dir();
    std::fs::create_dir_all(d.join("build/bin")).unwrap();
    std::fs::write(d.join("convert_hf_to_gguf.py"), "old").unwrap();
    std::fs::write(d.join("build/bin").join(exe("llama-quantize")), "old").unwrap();
    std::fs::write(d.join("marker.txt"), "old install").unwrap();
    std::fs::write(d.join("install.json"), r#"{"schema":1,"tag":"b1","backend":"cpu"}"#).unwrap();
}

fn old_survived(c: &Config) {
    assert_eq!(std::fs::read_to_string(c.install_dir().join("marker.txt")).unwrap(), "old install", "the old install must stay");
    assert!(layout_of(&c.install_dir()).is_some());
    assert!(!c.base_dir.join("llama.cpp.new").exists(), "no half-folder is left");
    assert!(!c.base_dir.join("llama.cpp.old").exists());
}

fn failure_code(c: &Config) -> Option<String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(c.base_dir.join("llama.cpp-last-failure.json")).ok()?).ok()?;
    v["code"].as_str().map(str::to_string)
}

#[test]
fn a_full_install_puts_the_layout_the_workers_finder_expects_into_the_data_folder() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var("LLAMA_CPP_DIR");
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    let c = cfg(d.path(), &fake);
    let (res, events) = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default());
    let info = res.unwrap();
    let dir = c.install_dir();
    assert_eq!((info["tag"].as_str(), info["backend"].as_str(), info["via"].as_str()), (Some("b6000"), Some("cpu"), Some("download")));
    assert!(layout_of(&dir).is_some());
    for f in ["convert_hf_to_gguf.py", "conversion/__init__.py", "gguf-py/gguf/__init__.py", "models/templates/a.jinja", "install.json"] {
        assert!(dir.join(f).is_file(), "{f}");
    }
    assert!(dir.join("build/bin").join(exe("llama-perplexity")).is_file() && dir.join("build/bin").join(exe("llama-imatrix")).is_file());
    assert!(!dir.join("docs").exists() && !dir.join("CMakeLists.txt").exists(), "only the converter's files from the source");
    assert!(!c.base_dir.join("llama.cpp.new").exists());
    // the record
    let rec = read_info(&dir).unwrap().unwrap();
    assert_eq!((rec.tag.as_str(), rec.assets.clone()), ("b6000", vec![host_asset().to_string()]));
    assert_eq!(rec.sha256.len(), 2);
    assert!(rec.sha256.contains_key(host_asset()) && rec.sha256.contains_key("source.zip"));
    assert!(rec.source_url.ends_with("/ggml-org/llama.cpp/archive/refs/tags/b6000.zip") && rec.installed_at.contains('T'));
    assert_eq!(rec.versions.get("llama-imatrix").map(String::as_str), Some("version: 6000 (abc1234)"));
    assert!(rec.converter_check.starts_with("not checked"), "{}", rec.converter_check);
    assert_eq!(rec.os, Os::current().label());
    // the progress events
    let phases: Vec<&str> = events.iter().filter(|e| e["kind"] == "step").map(|e| e["phase"].as_str().unwrap()).collect();
    assert_eq!(phases, ["lookup", "download", "download", "unpack", "check", "finish"]);
    let texts: Vec<&str> = events.iter().filter(|e| e["kind"] == "step").map(|e| e["text"].as_str().unwrap()).collect();
    assert!(texts[1].starts_with("Downloading the program files (") && texts[2].starts_with("Downloading the converter"), "{texts:?}");
    assert!(events.iter().all(|e| e["task"] == "llamacpp"));
    let values: Vec<f64> = events.iter().filter(|e| e["kind"] == "progress").map(|e| e["value"].as_f64().unwrap()).collect();
    assert!(values.windows(2).all(|w| w[0] <= w[1] + 1e-9), "{values:?}");
    assert_eq!(*values.last().unwrap(), 1.0);
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["bytes"].as_u64().is_some_and(|b| b > 0)));
    // the status now says where the workers will find it, and nothing is left from a failure
    let s = status(&c);
    assert_eq!((s["found"]["how"].as_str(), s["installed"]["tag"].as_str()), (Some("program_install"), Some("b6000")));
    assert!(s["last_failure"].is_null());
    let hits = fake.hits.lock().unwrap().clone();
    assert!(hits.contains(&"/codeload/b6000.zip".to_string()), "the redirect of the source address was followed: {hits:?}");
    // installing again replaces it and leaves no .old folder
    let (res, _) = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default());
    res.unwrap();
    assert!(!c.base_dir.join("llama.cpp.old").exists());
}

#[test]
fn the_plan_says_what_would_be_downloaded_before_anything_is() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    let c = cfg(d.path(), &fake);
    let p = plan(&c, &host(), &opts(Want::Auto)).unwrap();
    assert_eq!((p["tag"].as_str(), p["backend"].as_str()), (Some("b6000"), Some("cpu")));
    assert_eq!(p["assets"][0]["name"], host_asset());
    assert!(p["total_bytes"].as_u64().unwrap() >= SOURCE_ESTIMATE_BYTES);
    assert!(p["options"].as_array().unwrap().iter().any(|o| o["id"] == "cpu" && o["available"] == true));
    assert!(!c.base_dir.join("llama.cpp").exists() && !c.base_dir.join("llama.cpp.new").exists());
    assert!(!fake.hits.lock().unwrap().iter().any(|h| h.starts_with("/dl/") || h.starts_with("/codeload/")), "nothing was downloaded");
}

#[test]
fn a_pinned_tag_installs_that_release_and_a_bad_tag_is_refused() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    fake.release("b5000", &[(&host_asset().replace("b6000", "b5000"), bin_zip(""))], source(), &[], false); // only the tag address exists: "latest" would be a 404
    let mut c = cfg(d.path(), &fake);
    c.tag_pin = Some("b5000".into());
    let (res, _) = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default());
    assert_eq!(res.unwrap()["tag"], "b5000");
    assert!(!fake.hits.lock().unwrap().iter().any(|h| h.ends_with("/latest")));
    let (res, _) = run(&c, &host(), &Options { want: Want::Auto, tag: Some("../../etc".into()) }, &FakeRunner::default());
    assert!(res.unwrap_err().0.contains("not a release tag"));
    assert_eq!(failure_code(&c).as_deref(), Some("bad_tag"));
}

#[test]
fn a_blackwell_windows_install_gets_the_cuda_build_and_the_runtime_dlls_next_to_the_programs() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    fake.release("b6000", &[("llama-b6000-bin-win-cuda-12.8-x64.zip", bin_zip("")), ("cudart-llama-bin-win-cuda-12.8-x64.zip", cudart_zip()), ("llama-b6000-bin-win-cuda-12.4-x64.zip", bin_zip("")), ("llama-b6000-bin-win-cpu-x64.zip", bin_zip(""))], source(), &[], true);
    let c = cfg(d.path(), &fake);
    let m = Machine { os: Os::Windows, arch: Arch::X64, nvidia: Some(Nvidia { name: "NVIDIA GeForce RTX 5090".into(), driver_cuda: Some((12, 9)), compute_cap: Some((12, 0)), blackwell: true }), vulkan_likely: false, tools: BuildTools { nvcc: false, cmake: false, cxx: false } };
    let (res, _) = run(&c, &m, &opts(Want::Auto), &FakeRunner::default());
    let info = res.unwrap();
    assert_eq!((info["backend"].as_str(), info["cuda_version"].as_str()), (Some("cuda"), Some("12.8")));
    let bin = c.install_dir().join("build/bin");
    for dll in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"] {
        assert!(bin.join(dll).is_file(), "{dll}");
    }
    assert!(!bin.join("README.txt").exists(), "only the DLLs of the runtime package");
    assert_eq!(info["assets"].as_array().unwrap().len(), 2);
}

#[test]
fn a_cuda_build_that_does_not_start_is_refused_with_an_offer_of_the_cpu_build_and_the_old_install_stays() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    fake.release("b6000", &[("llama-b6000-bin-win-cuda-12.8-x64.zip", bin_zip("")), ("cudart-llama-bin-win-cuda-12.8-x64.zip", cudart_zip()), ("llama-b6000-bin-win-cpu-x64.zip", bin_zip(""))], source(), &[], true);
    let c = cfg(d.path(), &fake);
    old_install(&c);
    let m = Machine { os: Os::Windows, arch: Arch::X64, nvidia: Some(Nvidia { name: "RTX 5090".into(), driver_cuda: Some((12, 9)), compute_cap: Some((12, 0)), blackwell: true }), vulkan_likely: false, tools: BuildTools { nvcc: false, cmake: false, cxx: false } };
    let r = FakeRunner::default();
    *r.broken.lock().unwrap() = Some("llama-imatrix".into());
    let (res, _) = run(&c, &m, &opts(Want::Auto), &r);
    let e = res.unwrap_err().0;
    assert!(e.contains("CUDA build of llama.cpp does not start") && e.contains("llama-imatrix") && e.contains("Install the CPU build instead"), "{e}");
    old_survived(&c);
    let s = status(&c);
    assert_eq!((s["last_failure"]["code"].as_str(), s["last_failure"]["offer_cpu"].as_bool()), (Some("cuda_start_failed"), Some(true)));
    // the one click: the CPU build, which starts
    let (res, _) = run(&c, &m, &opts(Want::Cpu), &FakeRunner::default());
    assert_eq!(res.unwrap()["backend"], "cpu");
    assert!(status(&c)["last_failure"].is_null(), "a new install clears the failure");
    assert_eq!(read_info(&c.install_dir()).unwrap().unwrap().tag, "b6000");
}

#[test]
fn a_linux_tar_gz_with_a_folder_and_library_links_installs() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let tgz = tar_gz(vec![
        tar_entry("llama-b6000/", b'5', b"", "", 0o755), tar_entry(&format!("llama-b6000/{}", exe("llama-quantize")), b'0', b"stub", "", 0o755), tar_entry(&format!("llama-b6000/{}", exe("llama-imatrix")), b'0', b"stub", "", 0o755),
        tar_entry(&format!("llama-b6000/{}", exe("llama-perplexity")), b'0', b"stub", "", 0o755), tar_entry("llama-b6000/libllama.so.0", b'0', b"lib", "", 0o644), tar_entry("llama-b6000/libllama.so", b'2', b"", "libllama.so.0", 0o777),
    ]);
    fake.release("b6000", &[("llama-b6000-bin-ubuntu-x64.tar.gz", tgz)], source(), &[], true);
    let c = cfg(d.path(), &fake);
    let (res, _) = run(&c, &Machine::cpu_only(Os::Linux, Arch::X64), &opts(Want::Auto), &FakeRunner::default());
    res.unwrap();
    let bin = c.install_dir().join("build/bin");
    assert!(layout_of(&c.install_dir()).is_some() && bin.join("libllama.so.0").is_file());
    #[cfg(unix)]
    assert!(std::fs::symlink_metadata(bin.join("libllama.so")).unwrap().file_type().is_symlink());
}

#[test]
fn a_cancel_in_every_phase_leaves_the_old_install_and_no_failure_record() {
    for phase in ["lookup", "download", "unpack", "check", "finish"] {
        let d = tempdir().unwrap();
        let fake = Fake::start();
        serve_host(&fake, "b6000");
        let c = cfg(d.path(), &fake);
        old_install(&c);
        let cancel = AtomicBool::new(false);
        let (res, events) = run_with_cancel(&c, &host(), &opts(Want::Auto), &FakeRunner::default(), &cancel, &mut |e| {
            if e["kind"] == "step" && e["phase"] == phase {
                cancel.store(true, Ordering::SeqCst);
            }
        });
        let e = res.unwrap_err().0;
        assert_eq!(e, "Cancelled.", "{phase}: {e}");
        old_survived(&c);
        assert_eq!(read_info(&c.install_dir()).unwrap().unwrap().tag, "b1", "{phase}");
        assert!(failure_code(&c).is_none(), "{phase}: a cancel is not a failure");
        assert!(!events.iter().any(|e| e["phase"] == "nonexistent"));
    }
}

#[test]
fn a_cancel_before_the_converter_download_stops_it() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    let c = cfg(d.path(), &fake);
    old_install(&c);
    let (cancel, n) = (AtomicBool::new(false), Mutex::new(0));
    let (res, _) = run_with_cancel(&c, &host(), &opts(Want::Auto), &FakeRunner::default(), &cancel, &mut |e| {
        if e["phase"] == "download" && e["kind"] == "step" {
            let mut n = n.lock().unwrap();
            *n += 1;
            if *n == 2 {
                cancel.store(true, Ordering::SeqCst);
            }
        }
    });
    assert_eq!(res.unwrap_err().0, "Cancelled.");
    old_survived(&c);
    assert!(!fake.hits.lock().unwrap().iter().any(|h| h.contains("codeload")), "the source was not even requested");
}

#[test]
fn not_found_rate_limit_and_offline_have_plain_messages() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let c = cfg(d.path(), &fake);
    old_install(&c);
    // 404 on the release
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("no such llama.cpp release") && e.contains("NDFORGE_LLAMA_CPP_TAG"), "{e}");
    assert_eq!(failure_code(&c).as_deref(), Some("not_found"));
    old_survived(&c);
    // the API's rate limit: a 403 with a message and the header
    fake.set(Fake::LIST, Reply::Json(403, r#"{"message":"API rate limit exceeded for 203.0.113.9."}"#.into(), vec![("x-ratelimit-remaining", "0")]));
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("limiting how often") && e.contains("hour"), "{e}");
    assert_eq!(failure_code(&c).as_deref(), Some("rate_limit"));
    fake.set(Fake::LIST, Reply::Json(429, "{}".into(), vec![]));
    assert_eq!(run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0.contains("limiting"), true);
    // a 403 that is not the limit
    fake.set(Fake::LIST, Reply::Json(403, r#"{"message":"Forbidden by policy"}"#.into(), vec![]));
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("refused") && e.contains("Forbidden by policy"), "{e}");
    // a server error, and an answer that is not a release
    fake.set(Fake::LIST, Reply::Json(503, "{}".into(), vec![]));
    assert!(run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0.contains("(503)"));
    fake.set(Fake::LIST, Reply::Json(200, "<html>captive portal</html>".into(), vec![]));
    assert!(run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0.contains("not a release list"));
    // offline: nothing listens there
    let mut off = c.clone();
    off.api_base = "http://127.0.0.1:1".into();
    let e = run(&off, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("could not be reached") && e.contains("HTTPS_PROXY"), "{e}");
    assert_eq!(failure_code(&off).as_deref(), Some("offline"));
    old_survived(&c);
}

#[test]
fn a_missing_source_file_is_a_not_found_too_and_leaves_the_old_install() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    fake.routes.lock().unwrap().remove("/codeload/b6000.zip");
    let c = cfg(d.path(), &fake);
    old_install(&c);
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("no such llama.cpp release or file"), "{e}");
    old_survived(&c);
}

#[test]
fn a_cut_off_download_is_refused() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    // the listing promises more bytes than the server sends (what a connection that dropped half way looks like to the size check)
    let bytes = bin_zip("");
    let body = json!({"tag_name": "b6000", "assets": [{"name": host_asset(), "size": bytes.len() * 2, "browser_download_url": format!("{}/dl/b6000/{}", fake.base, host_asset())}]}).to_string();
    fake.set(Fake::LIST, Reply::Json(200, format!("[{body}]"), vec![]));
    let c = cfg(d.path(), &fake);
    old_install(&c);
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("cut off") || e.contains("ended early"), "{e}");
    assert_eq!(failure_code(&c).as_deref(), Some("truncated"));
    old_survived(&c);
}

#[test]
fn a_wrong_checksum_deletes_the_file_and_refuses_it() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    fake.release("b6000", &[(host_asset(), bin_zip(""))], source(), &[host_asset()], true);
    let c = cfg(d.path(), &fake);
    old_install(&c);
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("does not match the checksum") && e.contains("thrown away"), "{e}");
    assert_eq!(failure_code(&c).as_deref(), Some("checksum"));
    old_survived(&c);
    // no digest in the listing (an older API answer): the download is accepted, its own sha256 is recorded
    let body = json!({"tag_name": "b6000", "assets": [{"name": host_asset(), "size": bin_zip("").len(), "browser_download_url": format!("{}/dl/b6000/{}", fake.base, host_asset())}]}).to_string();
    fake.set(Fake::LIST, Reply::Json(200, format!("[{body}]"), vec![]));
    fake.set(&format!("/dl/b6000/{}", host_asset()), Reply::Bytes(bin_zip("")));
    let info = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap();
    assert_eq!(info["sha256"][host_asset()].as_str().unwrap().len(), 64);
}

#[test]
fn only_github_hosts_are_downloaded_from_even_when_the_release_list_says_otherwise() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let c = cfg(d.path(), &fake);
    let body = json!({"tag_name": "b6000", "assets": [{"name": host_asset(), "size": 10, "browser_download_url": format!("https://evil.example.com/{}", host_asset())}]}).to_string();
    fake.set(Fake::LIST, Reply::Json(200, format!("[{body}]"), vec![]));
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("not one this program downloads from"), "{e}");
    assert_eq!(failure_code(&c).as_deref(), Some("host_not_allowed"));
    // a redirect out of the list is refused too
    serve_host(&fake, "b6000");
    fake.set(&format!("/dl/b6000/{}", host_asset()), Reply::Redirect("https://evil.example.com/x.zip".into()));
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("not one this program downloads from"), "{e}");
}

#[test]
fn a_release_file_without_llama_quantize_or_with_an_unsafe_path_is_refused_and_the_old_install_stays() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let c = cfg(d.path(), &fake);
    old_install(&c);
    fake.release("b6000", &[(host_asset(), zip_bytes(&[("readme.txt", b"x", None)]))], source(), &[], true);
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("does not contain llama-quantize") && e.contains(host_asset()), "{e}");
    old_survived(&c);
    fake.release("b6000", &[(host_asset(), zip_bytes(&[("../escape.txt", b"x", None), ("a", b"1", None)]))], source(), &[], true);
    let e = run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("unsafe path"), "{e}");
    assert!(!d.path().join("data/escape.txt").exists() && !d.path().join("escape.txt").exists());
    old_survived(&c);
    // a source without the converter
    fake.release("b6000", &[(host_asset(), bin_zip(""))], zip_bytes(&[("llama.cpp-b6000/README.md", b"x", None)]), &[], true);
    assert!(run(&c, &host(), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0.contains("convert_hf_to_gguf.py was not in the source"));
    old_survived(&c);
}

#[test]
fn the_converter_is_checked_with_the_training_environments_python_when_there_is_one() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    let mut c = cfg(d.path(), &fake);
    c.python = Some(d.path().join("python-fake"));
    let r = FakeRunner::default();
    let info = run(&c, &host(), &opts(Want::Auto), &r).0.unwrap();
    assert_eq!(info["converter_check"], "ok");
    assert!(r.calls.lock().unwrap().iter().any(|l| l.starts_with("python-fake") && l.ends_with("--help")));
    // torch missing: a note, not a failure
    *r.python_out.lock().unwrap() = Some((1, "Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'".into()));
    let info = run(&c, &host(), &opts(Want::Auto), &r).0.unwrap();
    assert!(info["converter_check"].as_str().unwrap().starts_with("not checked") && info["converter_check"].as_str().unwrap().contains("torch"));
    // some other failure: refused, the old install stays
    old_install(&c);
    std::fs::remove_file(c.install_dir().join("install.json")).unwrap();
    std::fs::write(c.install_dir().join("install.json"), r#"{"schema":1,"tag":"b1"}"#).unwrap();
    *r.python_out.lock().unwrap() = Some((1, "Traceback (most recent call last):\nImportError: cannot import name 'ModelBase' from 'conversion'".into()));
    let e = run(&c, &host(), &opts(Want::Auto), &r).0.unwrap_err().0;
    assert!(e.contains("converter script does not start") && e.contains("cannot import name"), "{e}");
    assert_eq!(read_info(&c.install_dir()).unwrap().unwrap().tag, "b1");
    assert_eq!(std::fs::read_to_string(c.install_dir().join("marker.txt")).unwrap(), "old install");
}

#[test]
fn a_cuda_build_from_the_source_runs_cmake_with_progress_and_can_be_cancelled_or_fail() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let full = zip_bytes(&[SOURCE[0], SOURCE[1], SOURCE[2], SOURCE[3], SOURCE[4], SOURCE[5], SOURCE[6]]);
    fake.release("b6000", &[("llama-b6000-bin-ubuntu-x64.tar.gz", tar_gz(vec![]))], full, &[], true);
    let c = cfg(d.path(), &fake);
    let m = Machine { os: Os::Linux, arch: Arch::X64, nvidia: Some(Nvidia { name: "RTX 4090".into(), driver_cuda: Some((12, 8)), compute_cap: Some((8, 9)), blackwell: false }), vulkan_likely: false, tools: BuildTools { nvcc: true, cmake: true, cxx: true } };
    let r = FakeRunner::default();
    let (res, events) = run(&c, &m, &opts(Want::Build), &r);
    let info = res.unwrap();
    assert_eq!((info["backend"].as_str(), info["via"].as_str()), (Some("build"), Some("build")));
    assert!(layout_of(&c.install_dir()).is_some());
    assert!(c.install_dir().join("build/bin").join(exe("llama-imatrix")).is_file());
    assert!(!c.install_dir().join("src").exists(), "the build tree is not kept");
    let calls = r.calls.lock().unwrap().clone();
    assert!(calls.iter().any(|l| l == "cmake -S") && calls.iter().any(|l| l == "cmake --build"), "{calls:?}");
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["text"].as_str().is_some_and(|t| t.contains("Building llama.cpp (50 %)"))));
    assert!(events.iter().any(|e| e["kind"] == "log" && e["text"].as_str().is_some_and(|t| t.contains("Linking"))));
    // cmake fails: a plain message and the CPU offer
    *r.cmake_fails.lock().unwrap() = true;
    let e = run(&c, &m, &opts(Want::Build), &r).0.unwrap_err().0;
    assert!(e.contains("cmake could not prepare the build"));
    assert_eq!(status(&c)["last_failure"]["offer_cpu"], true);
    // a cancel while building
    *r.cmake_fails.lock().unwrap() = false;
    *r.cancel_in_build.lock().unwrap() = true;
    let cancel = AtomicBool::new(false);
    let (res, _) = run_with_cancel(&c, &m, &opts(Want::Build), &r, &cancel, &mut |_| {});
    assert_eq!(res.unwrap_err().0, "Cancelled.");
    assert!(!c.base_dir.join("llama.cpp.new").exists());
    assert_eq!(read_info(&c.install_dir()).unwrap().unwrap().tag, "b6000");
}

#[test]
fn building_is_refused_plainly_when_the_tools_are_missing() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6000");
    let c = cfg(d.path(), &fake);
    let e = run(&c, &Machine::cpu_only(Os::Linux, Arch::X64), &opts(Want::Build), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("Building needs nvcc"), "{e}");
}

#[test]
fn the_update_check_is_a_hint_and_installs_nothing() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    serve_host(&fake, "b6001");
    let c = cfg(d.path(), &fake);
    // nothing installed: no "newer"
    let v = check_update(&c).unwrap();
    assert_eq!((v["latest"].as_str(), v["newer"].as_bool(), v["installed"].is_null()), (Some("b6001"), Some(false), true));
    old_install(&c);
    std::fs::write(c.install_dir().join("install.json"), r#"{"schema":1,"tag":"b6000"}"#).unwrap();
    let v = check_update(&c).unwrap();
    assert_eq!((v["installed"].as_str(), v["latest"].as_str(), v["newer"].as_bool()), (Some("b6000"), Some("b6001"), Some(true)));
    std::fs::write(c.install_dir().join("install.json"), r#"{"schema":1,"tag":"b6001"}"#).unwrap();
    assert_eq!(check_update(&c).unwrap()["newer"], false);
    assert_eq!(std::fs::read_to_string(c.install_dir().join("marker.txt")).unwrap(), "old install");
    let mut off = c.clone();
    off.api_base = "http://127.0.0.1:1".into();
    assert_eq!(check_update(&off).unwrap_err().code, "offline");
}

// ---------------------------------------------------------------------------------------------------------------------------- release discovery (the owner's first run)
// llama.cpp publishes every build as a PRERELEASE; GET /releases/latest skips prereleases and answered an unrelated release whose only file is nightly-tag.txt. The installer
// reads the list of the newest 30 releases instead (the stand-in answers /releases/latest with exactly that unrelated release, so any use of it fails the test).

fn win_files(tag: &str) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<(String, Vec<u8>)> = ["bin-win-cpu-x64", "bin-win-cpu-arm64", "bin-win-vulkan-x64", "bin-win-cuda-12.4-x64", "bin-win-cuda-13.5-x64", "bin-win-cuda-13.5-arm64", "bin-win-sycl-x64", "bin-win-opencl-adreno-arm64", "bin-win-openvino-2026.4.1-x64", "bin-win-rocm-10.0-x64"]
        .iter().map(|s| (format!("llama-{tag}-{s}.zip"), bin_zip(""))).collect();
    v.push(("cudart-llama-bin-win-cuda-12.4-x64.zip".into(), cudart_zip()));
    v.push(("cudart-llama-bin-win-cuda-13.5-x64.zip".into(), cudart_zip()));
    v.push((format!("llama-{tag}-xcframework.zip"), vec![1]));
    v.push((format!("llama-{tag}-ui.tar.gz"), vec![1]));
    v
}

fn refs(v: &[(String, Vec<u8>)]) -> Vec<(&str, Vec<u8>)> {
    v.iter().map(|(n, b)| (n.as_str(), b.clone())).collect()
}

fn blackwell_windows() -> Machine {
    Machine { os: Os::Windows, arch: Arch::X64, nvidia: Some(Nvidia { name: "NVIDIA GeForce RTX 5090".into(), driver_cuda: Some((13, 0)), compute_cap: Some((12, 0)), blackwell: true }), vulkan_likely: false, tools: BuildTools { nvcc: false, cmake: false, cxx: false } }
}

fn count(fake: &Fake, prefix: &str) -> usize {
    fake.hits.lock().unwrap().iter().filter(|h| h.starts_with(prefix)).count()
}

#[test]
fn the_owners_case_installs_from_the_listed_prerelease_not_from_the_unrelated_latest() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let (w1, w0) = (win_files("b9001"), win_files("b9000"));
    let (r1, r0) = (fake.release_json("b9001", &refs(&w1), &[]), fake.release_json("b9000", &refs(&w0), &[]));
    fake.serve_tag("b9001", &r1, source());
    fake.serve_tag("b9000", &r0, source());
    fake.serve_list(vec![r1, r0]); // /releases/latest answers a release with nightly-tag.txt only
    let c = cfg(d.path(), &fake);
    let (res, events) = run(&c, &blackwell_windows(), &opts(Want::Auto), &FakeRunner::default());
    let info = res.unwrap();
    assert_eq!((info["tag"].as_str(), info["backend"].as_str(), info["cuda_version"].as_str()), (Some("b9001"), Some("cuda"), Some("13.5")));
    assert_eq!(info["assets"], json!(["llama-b9001-bin-win-cuda-13.5-x64.zip", "cudart-llama-bin-win-cuda-13.5-x64.zip"]));
    assert!(info["source_url"].as_str().unwrap().ends_with("/archive/refs/tags/b9001.zip"));
    let hits = fake.hits.lock().unwrap().clone();
    assert!(hits.contains(&"/codeload/b9001.zip".to_string()) && hits.contains(&"/dl/b9001/llama-b9001-bin-win-cuda-13.5-x64.zip".to_string()) && hits.contains(&"/dl/b9001/cudart-llama-bin-win-cuda-13.5-x64.zip".to_string()), "{hits:?}");
    assert!(!hits.iter().any(|h| h.ends_with("/releases/latest")), "latest is never asked");
    assert!(c.install_dir().join("build/bin/cudart64_12.dll").is_file() && layout_of(&c.install_dir()).is_some());
    assert!(events.iter().any(|e| e["kind"] == "log" && e["text"].as_str().is_some_and(|t| t.starts_with("Release b9001"))));
    // the plan says the same before anything is downloaded
    let p = plan(&c, &blackwell_windows(), &opts(Want::Auto)).unwrap();
    assert_eq!((p["tag"].as_str(), p["cuda_version"].as_str()), (Some("b9001"), Some("13.5")));
}

#[test]
fn a_newest_release_without_a_file_for_this_computer_is_skipped_for_the_next_one() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let only_arm = vec![("llama-b9002-bin-win-cpu-arm64.zip".to_string(), bin_zip("")), ("nightly-tag.txt".to_string(), b"zzz".to_vec())];
    let w1 = win_files("b9001");
    let (r2, r1) = (fake.release_json("b9002", &refs(&only_arm), &[]), fake.release_json("b9001", &refs(&w1), &[]));
    fake.serve_tag("b9002", &r2, source());
    fake.serve_tag("b9001", &r1, source());
    // a draft with matching files is never used
    let mut draft = fake.release_json("b9003", &refs(&win_files("b9003")), &[]);
    draft["draft"] = json!(true);
    fake.serve_list(vec![draft, r2, r1]);
    let c = cfg(d.path(), &fake);
    let info = run(&c, &Machine::cpu_only(Os::Windows, Arch::X64), &opts(Want::Auto), &FakeRunner::default()).0.unwrap();
    assert_eq!((info["tag"].as_str(), info["backend"].as_str()), (Some("b9001"), Some("cpu")));
    assert_eq!(count(&fake, "/dl/b9003"), 0);
    assert_eq!(count(&fake, "/dl/b9002"), 0);
}

#[test]
fn when_no_release_matches_the_error_says_how_many_were_checked_the_newest_and_its_files() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let junk = |t: &str| vec![(format!("llama-{t}-xcframework.zip"), vec![1u8]), (format!("llama-{t}-bin-win-cpu-arm64.zip"), vec![1u8])];
    let (a, b) = (junk("b9002"), junk("b9001"));
    let (ra, rb) = (fake.release_json("b9002", &refs(&a), &[]), fake.release_json("b9001", &refs(&b), &[]));
    fake.serve_list(vec![ra, rb]);
    let c = cfg(d.path(), &fake);
    old_install(&c);
    let e = run(&c, &Machine::cpu_only(Os::Windows, Arch::X64), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("None of the 2 newest llama.cpp releases") && e.contains("The newest is b9002") && e.contains("llama-b9002-bin-win-cpu-arm64.zip") && e.contains("llama-b9002-xcframework.zip"), "{e}");
    assert!(e.contains("docs/LLAMA_CPP_INSTALL.md") && e.contains("LLAMA_CPP_DIR"), "{e}");
    assert_eq!(failure_code(&c).as_deref(), Some("no_asset"));
    old_survived(&c);
    // an empty list
    fake.set(Fake::LIST, Reply::Json(200, "[]".into(), vec![]));
    assert!(run(&c, &Machine::cpu_only(Os::Windows, Arch::X64), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0.contains("list of llama.cpp releases is empty"));
    // the only release is the owner's nightly-tag.txt one
    let nt = vec![("nightly-tag.txt".to_string(), b"not-a-tag!!".to_vec())];
    let r = fake.release_json("nightly", &refs(&nt), &[]);
    fake.set(Fake::LIST, Reply::Json(200, json!([r]).to_string(), vec![]));
    let e = run(&c, &Machine::cpu_only(Os::Windows, Arch::X64), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("None of the 1 newest") && e.contains("nightly-tag.txt"), "{e}");
}

#[test]
fn nightly_tag_txt_is_only_a_pointer_that_is_followed_when_it_names_a_real_release() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let w = win_files("b9000");
    let target = fake.release_json("b9000", &refs(&w), &[]);
    fake.serve_tag("b9000", &target, source()); // NOT in the list
    let pointer = |txt: &str| { let f = vec![("nightly-tag.txt".to_string(), txt.as_bytes().to_vec())]; fake.release_json("nightly", &refs(&f), &[]) };
    let m = Machine::cpu_only(Os::Windows, Arch::X64);
    let c = cfg(d.path(), &fake);
    // a pointer to an existing tag
    fake.set(Fake::LIST, Reply::Json(200, json!([pointer("b9000\n")]).to_string(), vec![]));
    assert_eq!(run(&c, &m, &opts(Want::Auto), &FakeRunner::default()).0.unwrap()["tag"], "b9000");
    // a pointer to nothing (the tag address answers 404), junk, an HTML page, something very long: all plain errors, never a crash
    for txt in ["b8999", "<html>captive portal</html>", "../../etc/passwd", &"b".repeat(400), ""] {
        fake.set(Fake::LIST, Reply::Json(200, json!([pointer(txt)]).to_string(), vec![]));
        let e = run(&c, &m, &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
        assert!(e.contains("None of the 1 newest llama.cpp releases"), "{txt:?}: {e}");
    }
    assert!(!fake.hits.lock().unwrap().iter().any(|h| h.contains("..") || h.contains("passwd")), "junk is never turned into a request");
}

#[test]
fn a_pinned_tag_is_used_alone_and_a_pin_without_a_file_for_this_computer_says_so() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let w = win_files("b8000");
    let r = fake.release_json("b8000", &refs(&w), &[]);
    fake.serve_tag("b8000", &r, source());
    let newer = win_files("b9000");
    let rn = fake.release_json("b9000", &refs(&newer), &[]);
    fake.serve_list(vec![rn]);
    let mut c = cfg(d.path(), &fake);
    c.tag_pin = Some("b8000".into());
    let m = Machine::cpu_only(Os::Windows, Arch::X64);
    assert_eq!(run(&c, &m, &opts(Want::Auto), &FakeRunner::default()).0.unwrap()["tag"], "b8000");
    assert!(!fake.hits.lock().unwrap().iter().any(|h| h == Fake::LIST), "the list is not asked when a tag is pinned");
    let e = run(&c, &Machine::cpu_only(Os::Linux, Arch::X64), &opts(Want::Auto), &FakeRunner::default()).0.unwrap_err().0;
    assert!(e.contains("The release b8000 has no file for this computer") && e.contains("llama-b8000-bin-win-cpu-x64.zip"), "{e}");
}

#[test]
fn linux_with_nvidia_installs_the_prebuilt_cuda_tar_and_its_runtime_libraries_next_to_the_programs() {
    let d = tempdir().unwrap();
    let fake = Fake::start();
    let prog = tar_gz(vec![
        tar_entry("llama-b9000/", b'5', b"", "", 0o755), tar_entry(&format!("llama-b9000/{}", exe("llama-quantize")), b'0', b"stub", "", 0o755), tar_entry(&format!("llama-b9000/{}", exe("llama-imatrix")), b'0', b"stub", "", 0o755),
        tar_entry(&format!("llama-b9000/{}", exe("llama-perplexity")), b'0', b"stub", "", 0o755),
    ]);
    let rt = tar_gz(vec![tar_entry("cudart-b9000/", b'5', b"", "", 0o755), tar_entry("cudart-b9000/libcudart.so.13", b'0', b"lib", "", 0o644), tar_entry("cudart-b9000/libcudart.so", b'2', b"", "libcudart.so.13", 0o777), tar_entry("cudart-b9000/NOTICE.txt", b'0', b"x", "", 0o644)]);
    let files = vec![("llama-b9000-bin-ubuntu-x64.tar.gz".to_string(), prog.clone()), ("llama-b9000-bin-ubuntu-cuda-13.5-x64.tar.gz".to_string(), prog), ("cudart-llama-b9000-bin-ubuntu-cuda-13.5-x64.tar.gz".to_string(), rt)];
    fake.release("b9000", &refs(&files), source(), &[], true);
    let m = Machine { os: Os::Linux, arch: Arch::X64, nvidia: Some(Nvidia { name: "RTX 5090".into(), driver_cuda: Some((13, 0)), compute_cap: Some((12, 0)), blackwell: true }), vulkan_likely: false, tools: BuildTools { nvcc: false, cmake: false, cxx: false } };
    let c = cfg(d.path(), &fake);
    let info = run(&c, &m, &opts(Want::Auto), &FakeRunner::default()).0.unwrap();
    assert_eq!((info["backend"].as_str(), info["cuda_version"].as_str()), (Some("cuda"), Some("13.5")));
    let bin = c.install_dir().join("build/bin");
    assert!(bin.join("libcudart.so.13").is_file() && !bin.join("NOTICE.txt").exists());
    #[cfg(unix)]
    assert!(std::fs::symlink_metadata(bin.join("libcudart.so")).unwrap().file_type().is_symlink());
}
