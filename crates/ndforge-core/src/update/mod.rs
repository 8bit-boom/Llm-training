//! The in-app updater (docs/UPDATES.md): looks at the public GitHub releases of the program, downloads the installer for this system, checks its SHA-256 against
//! `SHA256SUMS.txt` and (when a public key is built in) its ed25519 signature, and starts it only after an explicit click while nothing runs. Any failure leaves
//! the installed version untouched.
pub mod install;
pub mod key;
pub mod net;
pub mod verify;
pub mod version;

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use net::fail;
use version::Version;

/// The version of this program: the one number of `Cargo.toml` (`[workspace.package]`); `tauri.conf.json` has no version of its own.
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const REPO: &str = "8bit-boom/Llm-training";
pub const SUMS_NAME: &str = "SHA256SUMS.txt";
/// at most one automatic check in this many seconds
pub const CHECK_EVERY_SECS: u64 = 24 * 3600;

#[derive(Debug, Clone)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

/// Whether installers are verified by signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMode {
    /// no public key built in: the checksum is the only proof (and says so on screen)
    ChecksumOnly,
    Signed([u8; 32]),
    /// a key text is built in but is not a valid key: nothing is installed
    Broken,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub api_base: String,
    pub repo: String,
    pub current: String,
    /// "windows" | "linux" | "macos" ...
    pub os: String,
    pub allow_local: bool,
    pub key: KeyMode,
    /// where downloads go (a folder of its own)
    pub download_root: PathBuf,
    pub state_file: PathBuf,
}

pub fn key_mode_of(text: &str) -> KeyMode {
    if text.trim().is_empty() { KeyMode::ChecksumOnly } else { verify::parse_key(text).map(KeyMode::Signed).unwrap_or(KeyMode::Broken) }
}

impl Config {
    pub fn production() -> Config {
        let os = std::env::consts::OS.to_string();
        let download_root = if os == "linux" { dirs::download_dir().or_else(dirs::home_dir).unwrap_or_else(std::env::temp_dir).join("ND-Lore-Forge-updates") } else { std::env::temp_dir().join("ndforge-update") };
        Config { api_base: "https://api.github.com".into(), repo: REPO.into(), current: APP_VERSION.into(), os, allow_local: false, key: key_mode_of(key::UPDATE_PUBLIC_KEY), download_root, state_file: crate::paths::config_dir().join("update-state.json") }
    }
    pub fn mode_name(&self) -> &'static str {
        match self.key { KeyMode::ChecksumOnly => "checksum_only", KeyMode::Signed(_) => "signed", KeyMode::Broken => "key_broken" }
    }
    pub fn mode_text(&self) -> &'static str {
        match self.key {
            KeyMode::ChecksumOnly => "Checksum only: this program has no signing key built in, so an update is checked against the checksum list of the same release (it catches a damaged or incomplete download, not a tampered release).",
            KeyMode::Signed(_) => "Signed: an update is installed only when its checksum matches and its signature is valid for the key built into this program.",
            KeyMode::Broken => "The signing key built into this program is not valid, so no update will be installed. Install the new version by hand.",
        }
    }
}

/// A newer release for this system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Update {
    pub version: String,
    pub tag: String,
    pub notes: String,
    pub name: String,
    pub url: String,
    pub size: u64,
    pub sums_url: Option<String>,
    pub sig_url: Option<String>,
    pub html_url: String,
    pub published: String,
}

/// The asset of this system: Windows `ND-Lore-Forge-Setup-*.exe`, Linux `ND-Lore-Forge-*.AppImage`.
pub fn pick_asset<'a>(os: &str, names: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let mut names: Vec<&str> = names.collect();
    names.sort();
    names.into_iter().find(|n| match os {
        "windows" => n.starts_with("ND-Lore-Forge-Setup-") && n.ends_with(".exe"),
        "linux" => n.starts_with("ND-Lore-Forge-") && n.ends_with(".AppImage"),
        _ => false,
    })
}

pub struct Checked {
    pub current: String,
    /// `Some` only when a newer, final release exists
    pub update: Option<Update>,
    /// why an existing newer release cannot be installed here (no installer for this system ...)
    pub problem: Option<String>,
    pub note: Option<String>,
}

fn s(v: &Value) -> String {
    v.as_str().unwrap_or("").to_string()
}

/// Looks at `releases/latest` (GitHub itself leaves drafts and prereleases out of it; they are ignored here as well).
pub fn check(cfg: &Config) -> Result<Checked, Failure> {
    let url = format!("{}/repos/{}/releases/latest", cfg.api_base, cfg.repo);
    let text = net::get_text(cfg, &url, true, "application/vnd.github+json", 4 << 20)?;
    let v: Value = serde_json::from_str(&text).map_err(|_| fail("bad_answer", "GitHub's answer was not a release (a proxy page, perhaps). Try again later."))?;
    let tag = s(&v["tag_name"]);
    let current = Version::parse(&cfg.current).ok_or_else(|| fail("bad_version", "This program's own version number is not valid, so updates cannot be compared."))?;
    let base = |note: Option<&str>| Checked { current: cfg.current.clone(), update: None, problem: None, note: note.map(str::to_string) };
    if v["draft"].as_bool() == Some(true) || v["prerelease"].as_bool() == Some(true) {
        return Ok(base(Some("The newest release is a draft or a prerelease; those are ignored.")));
    }
    let Some(latest) = Version::parse(&tag) else { return Err(fail("bad_answer", format!("The newest release has a tag this program cannot read ({tag}).")) ) };
    if latest.is_prerelease() {
        return Ok(base(Some("The newest release is a prerelease; those are ignored.")));
    }
    if latest <= current {
        return Ok(base(None));
    }
    let assets = v["assets"].as_array().cloned().unwrap_or_default();
    let find = |name: &str| assets.iter().find(|a| a["name"].as_str() == Some(name));
    let url_of = |a: &Value| s(&a["browser_download_url"]);
    let names: Vec<&str> = assets.iter().filter_map(|a| a["name"].as_str()).collect();
    let mut out = Checked { current: cfg.current.clone(), update: None, problem: None, note: None };
    let Some(name) = pick_asset(&cfg.os, names.iter().copied()) else {
        out.problem = Some(format!("Version {} exists, but it has no installer for this system ({}). See the release page.", latest.text(), cfg.os));
        out.update = Some(Update { version: latest.text(), tag, notes: s(&v["body"]), name: String::new(), url: String::new(), size: 0, sums_url: None, sig_url: None, html_url: s(&v["html_url"]), published: s(&v["published_at"]) });
        return Ok(out);
    };
    let a = find(name).expect("picked from the list");
    let sig = find(&format!("{name}.sig")).map(url_of);
    out.update = Some(Update { version: latest.text(), tag, notes: s(&v["body"]), name: name.to_string(), url: url_of(a), size: a["size"].as_u64().unwrap_or(0), sums_url: find(SUMS_NAME).map(url_of), sig_url: sig, html_url: s(&v["html_url"]), published: s(&v["published_at"]) });
    Ok(out)
}

impl Checked {
    pub fn to_json(&self, cfg: &Config) -> Value {
        let installable = self.update.as_ref().is_some_and(|u| !u.name.is_empty()) && self.problem.is_none() && cfg.key != KeyMode::Broken;
        json!({"current": self.current, "state": if self.update.is_some() { "available" } else { "up_to_date" }, "update": self.update, "installable": installable,
               "problem": self.problem, "note": self.note, "mode": cfg.mode_name(), "mode_text": cfg.mode_text()})
    }
}

// ------------------------------------------------------------------------------------------------------------------------ the saved state

/// What the window needs at start without asking the network: when the last check was and the update it found (shown as the chip).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Saved {
    #[serde(default)]
    pub checked_at: u64,
    #[serde(default)]
    pub update: Option<Update>,
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn load_saved(cfg: &Config) -> Saved {
    std::fs::read(&cfg.state_file).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

pub fn save_saved(cfg: &Config, s: &Saved) {
    let _ = crate::fsx::write_json_atomic(&cfg.state_file, s);
}

/// Is an automatic check due? (never checked, a clock that went back, or 24 hours passed)
pub fn due(last: u64, now: u64) -> bool {
    last == 0 || now < last || now - last >= CHECK_EVERY_SECS
}

/// The saved update, only when it is still newer than this program (after an update the chip goes away by itself).
pub fn saved_update(cfg: &Config) -> Option<Update> {
    let (cur, up) = (Version::parse(&cfg.current)?, load_saved(cfg).update?);
    (Version::parse(&up.version)? > cur).then_some(up)
}

/// A check that remembers its result and time. `auto`: only when the setting is on and 24 hours have passed (otherwise `Ok(None)`).
pub fn check_and_save(cfg: &Config, auto: bool, enabled: bool) -> Result<Option<Checked>, Failure> {
    let prev = load_saved(cfg);
    if auto && (!enabled || !due(prev.checked_at, now_secs())) {
        return Ok(None);
    }
    // the attempt counts even when it fails, so that a computer without internet does not retry at every start
    save_saved(cfg, &Saved { checked_at: now_secs(), update: prev.update.clone() });
    let c = check(cfg)?;
    save_saved(cfg, &Saved { checked_at: now_secs(), update: c.update.clone().filter(|_| c.problem.is_none()) });
    Ok(Some(c))
}

pub fn status(cfg: &Config, check_at_start: bool) -> Value {
    let saved = load_saved(cfg);
    json!({"version": cfg.current, "os": cfg.os, "mode": cfg.mode_name(), "mode_text": cfg.mode_text(), "check_at_start": check_at_start,
           "checked_at": saved.checked_at, "update": saved_update(cfg), "installable": cfg.key != KeyMode::Broken && matches!(cfg.os.as_str(), "windows" | "linux")})
}

// ------------------------------------------------------------------------------------------------------------------------ download and verify

#[derive(Debug, Clone)]
pub struct Downloaded {
    pub path: PathBuf,
    pub sha256: String,
    pub size: u64,
    /// the signature was checked and valid
    pub signed: bool,
}

#[derive(Debug, Clone)]
pub struct Progress {
    /// "checksums" | "download" | "verify" | "signature"
    pub phase: &'static str,
    pub done: u64,
    pub total: u64,
}

fn clean(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Downloads and verifies. On any problem the files of this download are removed and a plain message is returned; the installed program is never touched.
pub fn download(cfg: &Config, up: &Update, cancel: &AtomicBool, on: &mut dyn FnMut(Progress)) -> Result<Downloaded, Failure> {
    use sha2::{Digest, Sha256};
    if up.name.is_empty() || up.url.is_empty() {
        return Err(fail("no_installer", "This release has no installer for this system."));
    }
    if up.name.contains(['/', '\\']) || up.name.contains("..") {
        return Err(fail("bad_name", "The installer's file name is not safe, so it was not used."));
    }
    let prefix = format!("/{}/releases/download/", cfg.repo);
    if !cfg.allow_local && !up.url.contains(&prefix) {
        return Err(fail("host_not_allowed", "The installer is not a file of this program's GitHub releases, so it was not used."));
    }
    let key = match cfg.key {
        KeyMode::Broken => return Err(fail("key_broken", cfg.mode_text())),
        k => k,
    };
    let Some(sums_url) = up.sums_url.as_deref() else { return Err(fail("no_checksums", "This release has no SHA256SUMS.txt, so the download cannot be checked and was not started. Nothing was changed.")) };
    if matches!(key, KeyMode::Signed(_)) && up.sig_url.is_none() {
        return Err(fail("signature_missing", format!("This release has no signature file ({}.sig) and this program only installs signed updates, so nothing was downloaded or changed.", up.name)));
    }
    let cancelled = || cancel.load(Ordering::SeqCst);
    let check_cancel = || if cancelled() { Err(fail("cancelled", crate::error::CANCELLED)) } else { Ok(()) };
    check_cancel()?;
    on(Progress { phase: "checksums", done: 0, total: 0 });
    let sums = net::get_text(cfg, sums_url, false, "text/plain", 1 << 20)?;
    let Some(want) = verify::sum_for(&sums, &up.name) else { return Err(fail("no_checksum_entry", format!("SHA256SUMS.txt has no line for {}, so the download cannot be checked and was not started.", up.name))) };

    let dir = cfg.download_root.join(&up.version);
    clean(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| fail("disk", format!("Could not make the download folder {} ({e}).", dir.display())))?;
    let (part, dest) = (dir.join(format!("{}.part", up.name)), dir.join(&up.name));
    let result = (|| -> Result<Downloaded, Failure> {
        let resp = net::get(cfg, &up.url, false, "application/octet-stream")?;
        let announced: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
        let total = if announced > 0 { announced } else { up.size };
        let mut out = std::fs::File::create(&part).map_err(|e| fail("disk", format!("Could not write {} ({e}).", part.display())))?;
        let mut reader = resp.into_reader();
        let mut buf = vec![0u8; 1 << 17];
        let (mut got, mut hasher, mut last) = (0u64, Sha256::new(), Instant::now());
        loop {
            check_cancel()?;
            let n = reader.read(&mut buf).map_err(|e| fail("truncated", format!("The download was cut off after {:.1} MB ({e}). Check the connection and try again; nothing was changed.", got as f64 / 1e6)))?;
            if n == 0 { break; }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n]).map_err(|e| fail("disk", format!("Could not write the download ({e}). Is the disk full?")))?;
            got += n as u64;
            if last.elapsed() >= Duration::from_millis(100) {
                on(Progress { phase: "download", done: got, total });
                last = Instant::now();
            }
        }
        on(Progress { phase: "download", done: got, total });
        drop(out);
        if total > 0 && got != total {
            return Err(fail("truncated", format!("The download ended early ({:.1} of {:.1} MB). Check the connection and try again; nothing was changed.", got as f64 / 1e6, total as f64 / 1e6)));
        }
        on(Progress { phase: "verify", done: got, total: got });
        let sum = verify::hex(&hasher.finalize());
        if sum != want {
            return Err(fail("checksum_mismatch", "The downloaded installer does not match the checksum in SHA256SUMS.txt, so it was thrown away. Nothing was changed. Try again; if it happens again, install the new version by hand from the release page."));
        }
        let mut signed = false;
        if let KeyMode::Signed(k) = key {
            on(Progress { phase: "signature", done: 0, total: 0 });
            let sig = net::get_text(cfg, up.sig_url.as_deref().unwrap_or(""), false, "text/plain", 4096).map_err(|f| if f.code == "not_found" { fail("signature_missing", "The signature file of this release could not be found, so the update was not installed.") } else { f })?;
            if !verify::verify(&k, &sig, &sum) {
                return Err(fail("signature_bad", "The signature of the installer is not valid for the key built into this program, so it was thrown away. Nothing was changed. Do not install this file."));
            }
            signed = true;
        }
        std::fs::rename(&part, &dest).map_err(|e| fail("disk", format!("Could not finish the download file ({e}).")))?;
        Ok(Downloaded { path: dest.clone(), sha256: sum, size: got, signed })
    })();
    if result.is_err() {
        clean(&dir);
    }
    result
}
