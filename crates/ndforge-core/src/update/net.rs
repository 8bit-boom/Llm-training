//! The network part of the updater: only GitHub hosts (a stand-in server on this computer in tests), redirects followed by hand inside that list, nothing sent but a
//! User-Agent and an Accept header (no token, no login). Every failure has a plain text and a short code.
use std::io::Read;
use std::time::Duration;

use serde_json::Value;

use super::{Config, Failure};
use crate::envmgr::llamacpp::net::proxy_for;

pub const API_HOST: &str = "api.github.com";
pub const DOWNLOAD_HOSTS: [&str; 3] = ["github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"];

pub fn fail(code: &'static str, msg: impl Into<String>) -> Failure {
    Failure { code, message: msg.into() }
}

fn host_of(url: &str) -> Option<(String, String)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let authority = authority.rsplit('@').next()?;
    Some((scheme.to_lowercase(), authority.split(':').next()?.to_lowercase()))
}

pub fn url_allowed(cfg: &Config, url: &str, api: bool) -> bool {
    let Some((scheme, host)) = host_of(url) else { return false };
    if cfg.allow_local && scheme == "http" && (host == "127.0.0.1" || host == "localhost") {
        return true;
    }
    scheme == "https" && (DOWNLOAD_HOSTS.contains(&host.as_str()) || (api && host == API_HOST))
}

fn agent(cfg: &Config, host: &str) -> ureq::Agent {
    let mut b = ureq::AgentBuilder::new().redirects(0).timeout_connect(Duration::from_secs(15)).timeout_read(Duration::from_secs(60)).user_agent("ND-Lore-Forge-updater");
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

pub fn transport_failure(detail: &str) -> Failure {
    if detail.to_lowercase().contains("proxy") {
        fail("proxy", format!("The proxy this computer uses did not let the request through. Check the proxy settings (the environment variable HTTPS_PROXY). [{detail}]"))
    } else {
        fail("offline", format!("GitHub could not be reached. Check the internet connection, then try again. [{detail}]"))
    }
}

fn status_failure(code: u16, resp: ureq::Response) -> Failure {
    let remaining = resp.header("x-ratelimit-remaining").map(str::to_string);
    let body = resp.into_string().unwrap_or_default();
    let message = serde_json::from_str::<Value>(&body).ok().and_then(|v| v["message"].as_str().map(str::to_string)).unwrap_or_default();
    if code == 429 || (code == 403 && (remaining.as_deref() == Some("0") || message.to_lowercase().contains("rate limit"))) {
        return fail("rate_limit", "GitHub is limiting how often this internet address may ask for releases (a shared office or VPN address can hit this). Wait about an hour and try again.");
    }
    match code {
        404 => fail("not_found", "GitHub has no such release or file. Try again later."),
        407 => fail("proxy", "The proxy asked for a login, which this program cannot give."),
        c => fail("http", format!("GitHub answered with an error ({c}). Try again in a moment.")),
    }
}

/// A GET that follows redirects only within the allowed hosts.
pub fn get(cfg: &Config, url: &str, api: bool, accept: &str) -> Result<ureq::Response, Failure> {
    let mut cur = url.to_string();
    for _ in 0..6 {
        if !url_allowed(cfg, &cur, api) {
            return Err(fail("host_not_allowed", "The update points to an address that is not GitHub, so it was not used."));
        }
        let ag = agent(cfg, &host_of(&cur).map(|h| h.1).unwrap_or_default());
        match ag.get(&cur).set("Accept", accept).call() {
            Ok(r) if (300..400).contains(&r.status()) => {
                let Some(loc) = r.header("Location") else { return Err(fail("http", "GitHub answered with a redirect that has no address.")) };
                cur = join(&cur, loc);
            }
            Ok(r) => return Ok(r),
            Err(ureq::Error::Status(code, resp)) => return Err(status_failure(code, resp)),
            Err(ureq::Error::Transport(t)) => return Err(transport_failure(&t.to_string())),
        }
    }
    Err(fail("http", "GitHub redirected the request too many times."))
}

/// A text body of at most `max` bytes.
pub fn get_text(cfg: &Config, url: &str, api: bool, accept: &str, max: u64) -> Result<String, Failure> {
    let resp = get(cfg, url, api, accept)?;
    let mut buf = Vec::new();
    resp.into_reader().take(max).read_to_end(&mut buf).map_err(|e| transport_failure(&e.to_string()))?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}
