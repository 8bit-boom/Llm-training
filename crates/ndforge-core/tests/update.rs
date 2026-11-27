//! The in-app updater against a stand-in GitHub (releases API and download server on 127.0.0.1). Keys are made here from a fixed test seed; no real key exists in the repository.
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use base64::Engine;
use ed25519_dalek::{Signer, SigningKey};
use ndforge_core::update::install::{self, Activity, Launcher};
use ndforge_core::update::{self, verify, Config, KeyMode, Update};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone)]
enum Route {
    Body(Vec<u8>),
    Status(u16, String, Vec<(String, String)>),
}

struct Server {
    base: String,
    routes: Arc<Mutex<HashMap<String, Route>>>,
    hits: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
}

impl Server {
    fn start() -> Server {
        let srv = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let base = format!("http://127.0.0.1:{}", srv.server_addr().to_ip().unwrap().port());
        let routes: Arc<Mutex<HashMap<String, Route>>> = Default::default();
        let hits: Arc<Mutex<Vec<String>>> = Default::default();
        let stop = Arc::new(AtomicBool::new(false));
        let (r, h, st) = (routes.clone(), hits.clone(), stop.clone());
        std::thread::spawn(move || {
            while !st.load(Ordering::SeqCst) {
                let Ok(Some(req)) = srv.recv_timeout(std::time::Duration::from_millis(50)) else { continue };
                let url = req.url().to_string();
                h.lock().unwrap().push(url.clone());
                let route = r.lock().unwrap().get(&url).cloned();
                let _ = match route {
                    None => req.respond(tiny_http::Response::from_string("nope").with_status_code(404)),
                    Some(Route::Body(b)) => req.respond(tiny_http::Response::from_data(b)),
                    Some(Route::Status(c, body, hs)) => {
                        let mut resp = tiny_http::Response::from_string(body).with_status_code(c);
                        for (k, v) in hs { resp.add_header(tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()).unwrap()); }
                        req.respond(resp)
                    }
                };
            }
        });
        Server { base, routes, hits, stop }
    }
    fn set(&self, path: &str, r: Route) { self.routes.lock().unwrap().insert(path.to_string(), r); }
    fn hit(&self, path: &str) -> usize { self.hits.lock().unwrap().iter().filter(|h| *h == path).count() }
}
impl Drop for Server {
    fn drop(&mut self) { self.stop.store(true, Ordering::SeqCst); }
}

const REPO: &str = "8bit-boom/Llm-training";
const EXE: &str = "ND-Lore-Forge-Setup-0.2.0.exe";
const APPIMAGE: &str = "ND-Lore-Forge-0.2.0.AppImage";

fn test_key() -> SigningKey {
    SigningKey::from_bytes(&Sha256::digest(b"ndforge updater test key, not a real key").into())
}
fn pub_b64(k: &SigningKey) -> String { base64::engine::general_purpose::STANDARD.encode(k.verifying_key().to_bytes()) }
fn sha(b: &[u8]) -> String { verify::hex(&Sha256::digest(b)) }
fn sign_b64(k: &SigningKey, installer: &[u8]) -> String { base64::engine::general_purpose::STANDARD.encode(k.sign(&verify::message(&sha(installer))).to_bytes()) }

fn release(srv: &Server, tag: &str, extra: Value) -> Value {
    let mut v = json!({"tag_name": tag, "draft": false, "prerelease": false, "body": "Notes of the release", "html_url": format!("https://github.com/{REPO}/releases/tag/{tag}"), "published_at": "2026-11-01T00:00:00Z",
        "assets": [
            {"name": EXE, "size": 0, "browser_download_url": format!("{}/dl/{EXE}", srv.base)},
            {"name": APPIMAGE, "size": 0, "browser_download_url": format!("{}/dl/{APPIMAGE}", srv.base)},
            {"name": "ND-Lore-Forge-0.2.0.deb", "size": 5, "browser_download_url": format!("{}/dl/x.deb", srv.base)},
            {"name": "SHA256SUMS.txt", "size": 10, "browser_download_url": format!("{}/dl/SHA256SUMS.txt", srv.base)}]});
    for (k, x) in extra.as_object().unwrap() { v[k] = x.clone(); }
    v
}
fn serve_release(srv: &Server, v: &Value) { srv.set(&format!("/repos/{REPO}/releases/latest"), Route::Body(v.to_string().into_bytes())); }

fn cfg(srv: &Server, os: &str, key: KeyMode, dir: &tempfile::TempDir) -> Config {
    Config { api_base: srv.base.clone(), repo: REPO.into(), current: "0.1.0".into(), os: os.into(), allow_local: true, key, download_root: dir.path().join("dl"), state_file: dir.path().join("update-state.json") }
}

fn up_of(c: &Config) -> Update { update::check(c).unwrap().update.unwrap() }

/// Serves release 0.2.0 with the Windows installer `body`, its checksum line and (optionally) its signature.
fn serve_full(srv: &Server, body: &[u8], sums_hash: &str, sig: Option<String>) {
    let mut v = release(srv, "v0.2.0", json!({}));
    if sig.is_some() { v["assets"].as_array_mut().unwrap().push(json!({"name": format!("{EXE}.sig"), "size": 88, "browser_download_url": format!("{}/dl/{EXE}.sig", srv.base)})); }
    serve_release(srv, &v);
    srv.set(&format!("/dl/{EXE}"), Route::Body(body.to_vec()));
    srv.set("/dl/SHA256SUMS.txt", Route::Body(format!("{sums_hash}  {EXE}\n{}  other.bin\n", "0".repeat(64)).into_bytes()));
    if let Some(s) = sig { srv.set(&format!("/dl/{EXE}.sig"), Route::Body(s.into_bytes())); }
}

fn no_progress() -> impl FnMut(update::Progress) { |_| {} }
fn leftovers(c: &Config) -> usize { std::fs::read_dir(&c.download_root).map(|d| d.flatten().flat_map(|e| std::fs::read_dir(e.path()).map(|x| x.count()).ok()).sum()).unwrap_or(0) }

// ---------------------------------------------------------------------------------------------------------------------------- versions

#[test]
fn versions_compare_as_semver() {
    use update::version::Version;
    let v = |s: &str| Version::parse(s).unwrap();
    assert!(v("0.2.0") > v("0.1.9") && v("v1.0.0") > v("0.99.99") && v("0.10.0") > v("0.9.0"));
    assert!(v("1.0.0") > v("1.0.0-rc.1") && v("1.0.0-rc.2") > v("1.0.0-rc.1") && v("1.0.0-rc.10") > v("1.0.0-rc.2"));
    assert_eq!(v("1.2.3+build5"), v("1.2.3"));
    assert!(v("1.0.0-rc.1").is_prerelease() && !v("1.0.0").is_prerelease());
    for bad in ["", "1", "1.2", "1.2.3.4", "a.b.c", "01.2.3", "1.2.3-", "latest"] { assert!(Version::parse(bad).is_none(), "{bad}"); }
    assert_eq!(update::APP_VERSION, env!("CARGO_PKG_VERSION"));
}

// ---------------------------------------------------------------------------------------------------------------------------- the check

#[test]
fn a_newer_release_is_found_with_version_notes_size_and_the_asset_of_this_system() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let mut v = release(&srv, "v0.2.0", json!({}));
    v["assets"][0]["size"] = json!(12345);
    serve_release(&srv, &v);
    let c = update::check(&cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir)).unwrap();
    let u = c.update.clone().unwrap();
    assert_eq!((u.version.as_str(), u.name.as_str(), u.size, u.notes.as_str()), ("0.2.0", EXE, 12345, "Notes of the release"));
    assert!(u.sums_url.is_some() && u.sig_url.is_none());
    let j = c.to_json(&cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir));
    assert_eq!((j["state"].as_str(), j["installable"].as_bool(), j["mode"].as_str()), (Some("available"), Some(true), Some("checksum_only")));
    // Linux gets the AppImage, never the .deb
    assert_eq!(up_of(&cfg(&srv, "linux", KeyMode::ChecksumOnly, &dir)).name, APPIMAGE);
    // another system: the release is announced but cannot be installed
    let m = update::check(&cfg(&srv, "macos", KeyMode::ChecksumOnly, &dir)).unwrap();
    assert!(m.problem.is_some() && !m.to_json(&cfg(&srv, "macos", KeyMode::ChecksumOnly, &dir))["installable"].as_bool().unwrap());
}

#[test]
fn an_equal_or_older_release_is_no_update() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    for tag in ["v0.1.0", "v0.0.9"] {
        serve_release(&srv, &release(&srv, tag, json!({})));
        let c = update::check(&cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir)).unwrap();
        assert!(c.update.is_none(), "{tag}");
        assert_eq!(c.to_json(&cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir))["state"], "up_to_date");
    }
}

#[test]
fn drafts_and_prereleases_are_ignored() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    for extra in [json!({"prerelease": true}), json!({"draft": true}), json!({"tag_name": "v0.3.0-rc.1"})] {
        serve_release(&srv, &release(&srv, "v0.3.0", extra.clone()));
        let c = update::check(&cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir)).unwrap();
        assert!(c.update.is_none() && c.note.is_some(), "{extra}");
    }
}

#[test]
fn rate_limit_offline_and_a_bad_answer_have_plain_messages() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let c = cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir);
    let path = format!("/repos/{REPO}/releases/latest");
    srv.set(&path, Route::Status(403, r#"{"message":"API rate limit exceeded"}"#.into(), vec![("x-ratelimit-remaining".into(), "0".into())]));
    let e = update::check(&c).err().unwrap();
    assert_eq!(e.code, "rate_limit");
    assert!(e.message.contains("Wait about an hour"));
    srv.set(&path, Route::Status(429, "slow down".into(), vec![]));
    assert_eq!(update::check(&c).err().unwrap().code, "rate_limit");
    srv.set(&path, Route::Status(500, "boom".into(), vec![]));
    assert_eq!(update::check(&c).err().unwrap().code, "http");
    srv.set(&path, Route::Body(b"<html>proxy</html>".to_vec()));
    assert_eq!(update::check(&c).err().unwrap().code, "bad_answer");
    // offline: nothing listens on the port
    let dead = { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); format!("http://127.0.0.1:{}", l.local_addr().unwrap().port()) };
    let off = Config { api_base: dead, ..c.clone() };
    let e = update::check(&off).err().unwrap();
    assert_eq!(e.code, "offline");
    assert!(e.message.contains("internet connection"));
    // a check that failed leaves the saved state alone
    assert!(update::check_and_save(&off, false, true).is_err());
    assert!(update::saved_update(&off).is_none());
}

#[test]
fn only_github_hosts_are_contacted_in_production_settings() {
    let dir = tempfile::tempdir().unwrap();
    let prod = Config { allow_local: false, api_base: "http://127.0.0.1:1".into(), ..Config::production() };
    let _ = &dir;
    assert!(!update::net::url_allowed(&prod, "http://127.0.0.1:1/x", true));
    assert!(!update::net::url_allowed(&prod, "https://evil.example/x", false));
    assert!(!update::net::url_allowed(&prod, "https://api.github.com/x", false), "the API host is only for the release lookup");
    assert!(update::net::url_allowed(&prod, "https://api.github.com/x", true) && update::net::url_allowed(&prod, "https://github.com/a/b", false));
    assert!(update::net::url_allowed(&prod, "https://objects.githubusercontent.com/x", false));
}

// ---------------------------------------------------------------------------------------------------------------------------- download and verify

#[test]
fn checksum_only_mode_downloads_and_verifies_the_checksum() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let body = vec![7u8; 700_000];
    serve_full(&srv, &body, &sha(&body), None);
    let c = cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir);
    assert!(c.mode_text().starts_with("Checksum only"));
    let (mut phases, mut last) = (Vec::new(), 0u64);
    let d = update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut |p| { phases.push(p.phase); last = p.done; }).unwrap();
    assert_eq!((d.signed, d.size, d.sha256.as_str()), (false, 700_000, sha(&body).as_str()));
    assert_eq!(std::fs::read(&d.path).unwrap(), body);
    assert!(phases.contains(&"download") && phases.contains(&"verify") && !phases.contains(&"signature"));
    assert_eq!(last, 700_000);
    assert_eq!(srv.hit(&format!("/dl/{EXE}.sig")), 0, "no signature is fetched without a key");
}

#[test]
fn a_wrong_checksum_throws_the_file_away() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    serve_full(&srv, b"the real installer", &sha(b"something else"), None);
    let c = cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir);
    let e = update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap();
    assert_eq!(e.code, "checksum_mismatch");
    assert!(e.message.contains("Nothing was changed"));
    assert_eq!(leftovers(&c), 0);
}

#[test]
fn a_missing_checksums_file_or_line_stops_before_installing() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let c = cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir);
    // no SHA256SUMS.txt asset in the release
    let mut v = release(&srv, "v0.2.0", json!({}));
    v["assets"].as_array_mut().unwrap().retain(|a| a["name"] != "SHA256SUMS.txt");
    serve_release(&srv, &v);
    srv.set(&format!("/dl/{EXE}"), Route::Body(b"x".to_vec()));
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "no_checksums");
    assert_eq!(srv.hit(&format!("/dl/{EXE}")), 0, "the installer is not even downloaded");
    // the asset exists but 404s
    serve_release(&srv, &release(&srv, "v0.2.0", json!({})));
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "not_found");
    // the file has no line for the installer
    srv.set("/dl/SHA256SUMS.txt", Route::Body(format!("{}  other.bin\n", "a".repeat(64)).into_bytes()));
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "no_checksum_entry");
    assert_eq!(leftovers(&c), 0);
}

#[test]
fn with_a_key_a_valid_signature_installs_and_everything_else_refuses() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let k = test_key();
    let key = update::key_mode_of(&pub_b64(&k));
    assert!(matches!(key, KeyMode::Signed(_)));
    let c = cfg(&srv, "windows", key, &dir);
    let body = b"signed installer bytes".to_vec();
    // valid
    serve_full(&srv, &body, &sha(&body), Some(sign_b64(&k, &body)));
    let d = update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).unwrap();
    assert!(d.signed);
    // a signature made by another key
    let other = SigningKey::from_bytes(&[9u8; 32]);
    serve_full(&srv, &body, &sha(&body), Some(sign_b64(&other, &body)));
    let e = update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap();
    assert_eq!(e.code, "signature_bad");
    assert_eq!(leftovers(&c), 0);
    // a signature of another file
    serve_full(&srv, &body, &sha(&body), Some(sign_b64(&k, b"another installer")));
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "signature_bad");
    // garbage and raw bytes (the format is base64 text)
    for junk in ["not base64 at all!!", "AAAA"] {
        serve_full(&srv, &body, &sha(&body), Some(junk.to_string()));
        assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "signature_bad", "{junk}");
    }
    // a matching checksum does not replace the signature: no .sig asset in the release at all
    serve_full(&srv, &body, &sha(&body), None);
    let hits = srv.hit(&format!("/dl/{EXE}"));
    let e = update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap();
    assert_eq!(e.code, "signature_missing");
    assert_eq!(srv.hit(&format!("/dl/{EXE}")), hits, "refused before downloading");
    // the .sig is listed but cannot be fetched
    let mut v = release(&srv, "v0.2.0", json!({}));
    v["assets"].as_array_mut().unwrap().push(json!({"name": format!("{EXE}.sig"), "size": 88, "browser_download_url": format!("{}/dl/gone.sig", srv.base)}));
    serve_release(&srv, &v);
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "signature_missing");
    // a good signature does not rescue a bad checksum
    serve_full(&srv, &body, &sha(b"zzz"), Some(sign_b64(&k, &body)));
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "checksum_mismatch");
}

#[test]
fn a_built_in_key_that_is_not_a_key_refuses_everything() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(update::key_mode_of("  "), KeyMode::ChecksumOnly);
    assert_eq!(update::key_mode_of("not a key"), KeyMode::Broken);
    assert_eq!(update::key_mode_of("AAAA"), KeyMode::Broken);
    assert_eq!(update::key::UPDATE_PUBLIC_KEY, "", "no key is committed until the owner makes one");
    let body = b"x".to_vec();
    serve_full(&srv, &body, &sha(&body), None);
    let c = cfg(&srv, "windows", KeyMode::Broken, &dir);
    assert_eq!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).err().unwrap().code, "key_broken");
    assert!(!update::check(&c).unwrap().to_json(&c)["installable"].as_bool().unwrap());
}

#[test]
fn an_interrupted_download_leaves_nothing_and_a_cancel_stops_it() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let c = cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir);
    let body = vec![1u8; 300_000];
    serve_full(&srv, &body, &sha(&body), None);
    // a raw server that announces 300000 bytes, sends 100000 and hangs up
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let part = body[..100_000].to_vec();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        let (mut s, _) = l.accept().unwrap();
        let mut b = [0u8; 2048];
        let _ = s.read(&mut b);
        let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 300000\r\nConnection: close\r\n\r\n");
        let _ = s.write_all(&part);
    });
    let mut up = up_of(&c);
    up.url = format!("http://127.0.0.1:{port}/dl/{EXE}");
    let e = update::download(&c, &up, &AtomicBool::new(false), &mut no_progress()).err().unwrap();
    assert_eq!(e.code, "truncated");
    assert!(e.message.contains("Nothing was changed") || e.message.contains("nothing was changed"));
    assert_eq!(leftovers(&c), 0);
    // cancel before it starts, and while it runs (set from the progress callback of the first step)
    srv.set(&format!("/dl/{EXE}"), Route::Body(body.clone()));
    let flag = AtomicBool::new(true);
    assert_eq!(update::download(&c, &up_of(&c), &flag, &mut no_progress()).err().unwrap().code, "cancelled");
    let flag = AtomicBool::new(false);
    let e = update::download(&c, &up_of(&c), &flag, &mut |p| { if p.phase == "checksums" { flag.store(true, Ordering::SeqCst); } }).err().unwrap();
    assert_eq!(e.code, "cancelled");
    assert_eq!(leftovers(&c), 0);
    // and the next try works
    assert!(update::download(&c, &up_of(&c), &AtomicBool::new(false), &mut no_progress()).is_ok());
}

// ---------------------------------------------------------------------------------------------------------------------------- install

#[derive(Default)]
struct Fake { ran: Mutex<Vec<(PathBuf, Vec<String>)>>, shown: Mutex<Vec<PathBuf>>, fail: bool }
impl Launcher for Fake {
    fn run_installer(&self, p: &std::path::Path, args: &[&str]) -> std::io::Result<()> {
        if self.fail { return Err(std::io::Error::other("denied")); }
        self.ran.lock().unwrap().push((p.to_path_buf(), args.iter().map(|a| a.to_string()).collect()));
        Ok(())
    }
    fn show_folder(&self, f: &std::path::Path) -> std::io::Result<()> { self.shown.lock().unwrap().push(f.to_path_buf()); Ok(()) }
}

fn downloaded(os: &str) -> (tempfile::TempDir, Config, ndforge_core::update::Downloaded) {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let body = b"installer".to_vec();
    serve_full(&srv, &body, &sha(&body), None);
    let c = cfg(&srv, os, KeyMode::ChecksumOnly, &dir);
    let mut up = up_of(&c);
    if os == "linux" { srv.set(&format!("/dl/{APPIMAGE}"), Route::Body(body.clone())); srv.set("/dl/SHA256SUMS.txt", Route::Body(format!("{}  {APPIMAGE}\n", sha(&body)).into_bytes())); up = up_of(&c); }
    let d = update::download(&c, &up, &AtomicBool::new(false), &mut no_progress()).unwrap();
    (dir, c, d)
}

#[test]
fn nothing_is_installed_without_a_click_or_while_anything_runs() {
    let (_dir, c, d) = downloaded("windows");
    let f = Fake::default();
    let idle = Activity::default();
    assert_eq!(install::install(&c, &d, false, &idle, &f).err().unwrap().code, "not_confirmed");
    for (a, word) in [(Activity { job_running: true, ..Default::default() }, "A job is running"), (Activity { queue_running: Some("Heal of model".into()), ..Default::default() }, "queue run is active"), (Activity { queue_waiting: 2, ..Default::default() }, "waiting")] {
        let e = install::install(&c, &d, true, &a, &f).err().unwrap();
        assert_eq!(e.code, "busy");
        assert!(e.message.contains(word), "{}", e.message);
    }
    assert!(f.ran.lock().unwrap().is_empty() && d.path.is_file(), "no installer was started and the file is kept");
}

#[test]
fn windows_runs_the_inno_installer_silently_in_place_and_a_failure_changes_nothing() {
    let (_dir, c, d) = downloaded("windows");
    let f = Fake::default();
    let r = install::install(&c, &d, true, &Activity::default(), &f).unwrap();
    assert_eq!(r["action"], "restart");
    let ran = f.ran.lock().unwrap();
    assert_eq!(ran[0].0, d.path);
    assert_eq!(ran[0].1, ["/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/NORESTART"]);
    drop(ran);
    let bad = Fake { fail: true, ..Default::default() };
    let e = install::install(&c, &d, true, &Activity::default(), &bad).err().unwrap();
    assert_eq!(e.code, "launch");
    assert!(e.message.contains("unchanged"));
}

#[test]
fn linux_shows_the_file_with_instructions_and_does_not_replace_itself() {
    let (_dir, c, d) = downloaded("linux");
    let f = Fake::default();
    let r = install::install(&c, &d, true, &Activity::default(), &f).unwrap();
    assert_eq!(r["action"], "show_file");
    assert!(r["text"].as_str().unwrap().contains("Keep the old file"));
    assert_eq!(f.shown.lock().unwrap()[0], d.path.parent().unwrap());
    assert!(f.ran.lock().unwrap().is_empty());
    let (_d2, c2, d2) = downloaded("windows");
    let c2 = Config { os: "macos".into(), ..c2 };
    assert_eq!(install::install(&c2, &d2, true, &Activity::default(), &f).err().unwrap().code, "unsupported_os");
}

// ---------------------------------------------------------------------------------------------------------------------------- the schedule and the saved state

#[test]
fn an_automatic_check_runs_at_most_once_in_24_hours_and_only_when_on() {
    let srv = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let c = cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir);
    serve_release(&srv, &release(&srv, "v0.2.0", json!({})));
    let path = format!("/repos/{REPO}/releases/latest");
    assert!(update::due(0, 100) && update::due(1000, 500) && !update::due(1000, 1000 + 3600) && update::due(1000, 1000 + 24 * 3600));
    assert!(update::check_and_save(&c, true, false).unwrap().is_none(), "setting off");
    assert_eq!(srv.hit(&path), 0);
    let first = update::check_and_save(&c, true, true).unwrap().unwrap();
    assert!(first.update.is_some());
    assert!(update::check_and_save(&c, true, true).unwrap().is_none(), "second start within 24 h");
    assert_eq!(srv.hit(&path), 1);
    // the chip survives a restart, from the saved state, and a manual check always asks
    assert_eq!(update::saved_update(&c).unwrap().version, "0.2.0");
    assert!(update::check_and_save(&c, false, false).unwrap().is_some());
    assert_eq!(srv.hit(&path), 2);
    // once the program is at that version the saved update is no longer shown
    assert!(update::saved_update(&Config { current: "0.2.0".into(), ..c.clone() }).is_none());
    // a failed automatic check still counts as an attempt
    let dir2 = tempfile::tempdir().unwrap();
    let dead = { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); format!("http://127.0.0.1:{}", l.local_addr().unwrap().port()) };
    let off = Config { api_base: dead, ..cfg(&srv, "windows", KeyMode::ChecksumOnly, &dir2) };
    assert!(update::check_and_save(&off, true, true).is_err());
    assert!(update::check_and_save(&off, true, true).unwrap().is_none());
    let st = update::status(&c, true);
    assert_eq!((st["version"].as_str(), st["update"]["version"].as_str(), st["mode"].as_str()), (Some("0.1.0"), Some("0.2.0"), Some("checksum_only")));
}

// ---------------------------------------------------------------------------------------------------------------------------- the owner's scripts talk to the program

fn python_with_cryptography() -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|p| std::process::Command::new(p).args(["-c", "import cryptography"]).output().map(|o| o.status.success()).unwrap_or(false))
}

#[test]
fn keys_and_signatures_made_by_the_scripts_verify_in_the_program() {
    let Some(py) = python_with_cryptography() else { eprintln!("skipped: no python with the cryptography package"); return };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts");
    let dir = tempfile::tempdir().unwrap();
    let keyfile = dir.path().join("private.txt");
    let run = |args: Vec<String>| std::process::Command::new(py).args(args).output().unwrap();
    let out = run(vec![root.join("make_update_key.py").display().to_string(), "--out".into(), keyfile.display().to_string()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let private = std::fs::read_to_string(&keyfile).unwrap();
    assert!(!private.trim().is_empty() && !stdout.contains(private.trim()), "the private key is never printed");
    let public = stdout.lines().find(|l| l.len() == 44 && l.ends_with('=')).expect("the public key line").to_string();
    assert!(matches!(update::key_mode_of(&public), KeyMode::Signed(_)));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&keyfile).unwrap().permissions().mode() & 0o077, 0, "the private key file is for the owner only");
    }
    // an existing file is never overwritten
    assert!(!run(vec![root.join("make_update_key.py").display().to_string(), "--out".into(), keyfile.display().to_string()]).status.success());
    assert_eq!(std::fs::read_to_string(&keyfile).unwrap(), private);
    // sign a file, verify in Rust; the signature is for this file only
    let f = dir.path().join(EXE);
    std::fs::write(&f, b"installer bytes").unwrap();
    let out = run(vec![root.join("sign_update.py").display().to_string(), "--key-file".into(), keyfile.display().to_string(), f.display().to_string()]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!String::from_utf8_lossy(&out.stdout).contains(private.trim()));
    let sig = std::fs::read_to_string(dir.path().join(format!("{EXE}.sig"))).unwrap();
    let KeyMode::Signed(k) = update::key_mode_of(&public) else { unreachable!() };
    assert!(verify::verify(&k, &sig, &sha(b"installer bytes")));
    assert!(!verify::verify(&k, &sig, &sha(b"another file")));
    // a bad key is refused without echoing it
    let out = std::process::Command::new(py).arg(root.join("sign_update.py")).arg(&f).env("UPDATE_SIGNING_KEY", "not-a-key-SECRETVALUE").output().unwrap();
    assert!(!out.status.success());
    assert!(!String::from_utf8_lossy(&out.stderr).contains("SECRETVALUE") && !String::from_utf8_lossy(&out.stdout).contains("SECRETVALUE"));
}
