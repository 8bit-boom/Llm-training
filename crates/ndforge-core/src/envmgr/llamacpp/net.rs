//! The network part: the GitHub release lookup and the downloads. Only a short list of hosts is ever contacted, redirects are followed by hand and only inside that
//! list, nothing is sent but a User-Agent and an Accept header (no token, no login), and every failure has a plain text and a short code.
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;

use super::archive::{hex, parse_digest};
use super::choose::Asset;
use super::{Config, Failure};

/// The only hosts a download may come from or be redirected to (the API host is used for the release list only).
pub const DOWNLOAD_HOSTS: [&str; 4] = ["github.com", "codeload.github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"];
pub const API_HOST: &str = "api.github.com";

pub fn fail(code: &'static str, msg: impl Into<String>) -> Failure {
    Failure { code, message: msg.into() }
}

fn host_of(url: &str) -> Option<(String, String)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit('@').next()?;
    let host = authority.split(':').next()?.to_lowercase();
    Some((scheme.to_lowercase(), host))
}

/// May this address be contacted? https and a listed host; a local test server (127.0.0.1) only when the test settings allow it.
pub fn url_allowed(cfg: &Config, url: &str, api: bool) -> bool {
    let Some((scheme, host)) = host_of(url) else { return false };
    if cfg.allow_local && (host == "127.0.0.1" || host == "localhost") && scheme == "http" {
        return true;
    }
    scheme == "https" && (DOWNLOAD_HOSTS.contains(&host.as_str()) || (api && host == API_HOST))
}

/// The proxy the environment names for `host` (HTTPS_PROXY, ALL_PROXY, HTTP_PROXY, any letter case; NO_PROXY hosts are left out), as other programs do.
pub fn proxy_for(host: &str, var: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let get = |names: &[&str]| names.iter().find_map(|n| var(n).map(|v| v.trim().to_string()).filter(|v| !v.is_empty()));
    if let Some(no) = get(&["NO_PROXY", "no_proxy"]) {
        let h = host.to_lowercase();
        if no.split(',').map(|x| x.trim().trim_start_matches('.').to_lowercase()).any(|x| x == "*" || (!x.is_empty() && (h == x || h.ends_with(&format!(".{x}"))))) {
            return None;
        }
    }
    get(&["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"])
}

fn agent(cfg: &Config, host: &str) -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new().redirects(0).timeout_connect(Duration::from_secs(15)).timeout_read(Duration::from_secs(60)).user_agent("ND-Lore-Forge-llama.cpp-installer");
    if !cfg.allow_local {
        if let Some(p) = proxy_for(host, &|k| std::env::var(k).ok()).and_then(|p| ureq::Proxy::new(p).ok()) {
            b = b.proxy(p);
        }
    }
    b.build()
}

fn join(base: &str, location: &str) -> String {
    if location.contains("://") {
        return location.to_string();
    }
    let (scheme, rest) = base.split_once("://").unwrap_or(("https", base));
    let host = rest.split('/').next().unwrap_or("");
    if location.starts_with('/') { format!("{scheme}://{host}{location}") } else { format!("{scheme}://{host}/{location}") }
}

/// A GET that follows redirects only within the allowed hosts.
fn get(cfg: &Config, url: &str, api: bool, accept: &str) -> std::result::Result<ureq::Response, Failure> {
    let mut cur = url.to_string();
    for _ in 0..6 {
        if !url_allowed(cfg, &cur, api) {
            return Err(fail("host_not_allowed", format!("The address {cur} is not one this program downloads from (only GitHub), so it was not used.")));
        }
        let ag = agent(cfg, &host_of(&cur).map(|h| h.1).unwrap_or_default());
        match ag.get(&cur).set("Accept", accept).call() {
            Ok(r) if (300..400).contains(&r.status()) => {
                let Some(loc) = r.header("Location") else { return Err(fail("http", format!("GitHub answered with a redirect ({}) that has no address.", r.status()))) };
                cur = join(&cur, loc);
            }
            Ok(r) => return Ok(r),
            Err(ureq::Error::Status(code, resp)) => return Err(status_failure(code, resp, &cur)),
            Err(ureq::Error::Transport(t)) => return Err(transport_failure(&t.to_string())),
        }
    }
    Err(fail("http", "GitHub redirected the download too many times."))
}

pub fn transport_failure(detail: &str) -> Failure {
    let low = detail.to_lowercase();
    if low.contains("proxy") {
        fail("proxy", format!("The proxy this computer uses did not let the download through. Check the proxy settings (the environment variable HTTPS_PROXY), or install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md). [{detail}]"))
    } else {
        fail("offline", format!("GitHub could not be reached. Check the internet connection (a firewall or a proxy may block it; the environment variable HTTPS_PROXY names a proxy), then try again. [{detail}]"))
    }
}

pub fn status_failure(code: u16, resp: ureq::Response, url: &str) -> Failure {
    let remaining = resp.header("x-ratelimit-remaining").map(str::to_string);
    let body = resp.into_string().unwrap_or_default();
    let message = serde_json::from_str::<Value>(&body).ok().and_then(|v| v["message"].as_str().map(str::to_string)).unwrap_or_default();
    let limited = code == 429 || (code == 403 && (remaining.as_deref() == Some("0") || message.to_lowercase().contains("rate limit")));
    if limited {
        return fail("rate_limit", "GitHub is limiting how often this internet address may ask for the release list (a shared office or VPN address can hit this). Wait about an hour and press the button again, or install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md).");
    }
    match code {
        404 => fail("not_found", format!("GitHub has no such llama.cpp release or file ({url}). If you set NDFORGE_LLAMA_CPP_TAG, check the tag name (for example b6000); otherwise try again later.")),
        407 => fail("proxy", "The proxy asked for a login, which this program cannot give. Install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md) or use a network without a login proxy."),
        401 | 403 => fail("refused", format!("GitHub refused the request ({code}). {message} Try again later or install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md).")),
        c => fail("http", format!("GitHub answered with an error ({c}). Try again in a moment.")),
    }
}

#[derive(Debug, Clone)]
pub struct Release {
    pub tag: String,
    pub assets: Vec<Asset>,
}

pub fn parse_release(v: &Value) -> Option<Release> {
    let tag = v["tag_name"].as_str()?.to_string();
    let assets = v["assets"].as_array()?.iter().filter_map(|a| {
        Some(Asset { name: a["name"].as_str()?.to_string(), size: a["size"].as_u64().unwrap_or(0), url: a["browser_download_url"].as_str().unwrap_or("").to_string(), digest: a["digest"].as_str().map(str::to_string) })
    }).collect();
    Some(Release { tag, assets })
}

/// A tag that is safe to put into an address.
pub fn valid_tag(t: &str) -> bool {
    !t.is_empty() && t.len() <= 64 && t.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// The latest release, or the one named by `tag` (NDFORGE_LLAMA_CPP_TAG).
pub fn fetch_release(cfg: &Config, tag: Option<&str>) -> std::result::Result<Release, Failure> {
    let url = match tag {
        Some(t) => {
            if !valid_tag(t) {
                return Err(fail("bad_tag", format!("\"{t}\" is not a release tag (it looks like b6000).")));
            }
            format!("{}/repos/ggml-org/llama.cpp/releases/tags/{t}", cfg.api_base)
        }
        None => format!("{}/repos/ggml-org/llama.cpp/releases/latest", cfg.api_base),
    };
    let resp = get(cfg, &url, true, "application/vnd.github+json")?;
    let text = resp.into_string().map_err(|e| transport_failure(&e.to_string()))?;
    let v: Value = serde_json::from_str(&text).map_err(|_| fail("bad_answer", "GitHub's answer was not a release list (a proxy page, perhaps). Try again later."))?;
    parse_release(&v).ok_or_else(|| fail("bad_answer", "GitHub's answer has no release in it. Try again later."))
}

pub fn source_url(cfg: &Config, tag: &str) -> String {
    format!("{}/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip", cfg.web_base)
}

/// Downloads `url` to `dest` (through `dest.part`), reporting (bytes so far) to `on_bytes`. Checks the size against `expected` and the sha256 against `digest`.
/// Returns the sha256 of what was written.
pub fn download(cfg: &Config, url: &str, dest: &Path, expected: u64, digest: Option<&str>, cancel: &AtomicBool, on_bytes: &mut dyn FnMut(u64)) -> std::result::Result<String, Failure> {
    use sha2::{Digest, Sha256};
    if cancel.load(Ordering::SeqCst) {
        return Err(fail("cancelled", "Cancelled."));
    }
    let resp = get(cfg, url, false, "application/octet-stream")?;
    let announced: u64 = resp.header("Content-Length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let want = if expected > 0 { expected } else { announced };
    let part = dest.with_extension("part");
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p).map_err(|e| fail("disk", format!("Could not write into {}: {e}", p.display())))?;
    }
    let mut out = std::fs::File::create(&part).map_err(|e| fail("disk", format!("Could not write {}: {e}", part.display())))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 17];
    let (mut got, mut hasher, mut last) = (0u64, Sha256::new(), Instant::now());
    loop {
        if cancel.load(Ordering::SeqCst) {
            drop(out);
            let _ = std::fs::remove_file(&part);
            return Err(fail("cancelled", "Cancelled."));
        }
        let n = match reader.read(&mut buf) {
            Ok(n) => n,
            Err(e) => {
                drop(out);
                let _ = std::fs::remove_file(&part);
                return Err(fail("truncated", format!("The download was cut off after {:.1} MB ({e}). Check the connection and try again.", got as f64 / 1e6)));
            }
        };
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        if let Err(e) = out.write_all(&buf[..n]) {
            drop(out);
            let _ = std::fs::remove_file(&part);
            return Err(fail("disk", format!("Could not write the download ({e}). Is the disk full?")));
        }
        got += n as u64;
        if last.elapsed() >= Duration::from_millis(100) {
            on_bytes(got);
            last = Instant::now();
        }
    }
    on_bytes(got);
    drop(out);
    if want > 0 && got != want {
        let _ = std::fs::remove_file(&part);
        return Err(fail("truncated", format!("The download ended early ({:.1} of {:.1} MB). Check the connection and try again.", got as f64 / 1e6, want as f64 / 1e6)));
    }
    let sum = hex(&hasher.finalize());
    if let Some(d) = digest.and_then(parse_digest) {
        if d != sum {
            let _ = std::fs::remove_file(&part);
            return Err(fail("checksum", format!("The downloaded file does not match the checksum GitHub lists for it, so it was thrown away and not used. Try again; if it happens again, download llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md).")));
        }
    }
    std::fs::rename(&part, dest).map_err(|e| fail("disk", format!("Could not finish the download file: {e}")))?;
    Ok(sum)
}
