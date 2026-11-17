//! Access tokens of training servers, kept apart from the settings: `secrets.json` in the config folder, readable by the user only (mode 0600 on
//! Linux; on Windows the user's profile folder is private by default). It is NOT encrypted: storing tokens in the operating system's keyring is a
//! planned improvement.
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::error::Result;
use crate::paths::config_dir;

fn file() -> PathBuf {
    config_dir().join("secrets.json")
}

/// A file that cannot be read is kept as `secrets.json.broken-<time>` (the tokens in it can still be fetched by hand) and counts as empty.
fn load() -> BTreeMap<String, String> {
    let f = file();
    match std::fs::read_to_string(&f) {
        Ok(t) => match serde_json::from_str(&t) {
            Ok(m) => m,
            Err(_) => {
                let _ = crate::fsx::set_aside_broken(&f);
                BTreeMap::new()
            }
        },
        Err(_) => {
            if f.is_file() {
                let _ = crate::fsx::set_aside_broken(&f);
            }
            BTreeMap::new()
        }
    }
}

fn save(map: &BTreeMap<String, String>) -> Result<()> {
    crate::fsx::write_atomic_private(&file(), &serde_json::to_vec_pretty(map)?)
}

pub fn get(key: &str) -> Option<String> {
    load().get(key).cloned()
}

pub fn set(key: &str, value: &str) -> Result<()> {
    let mut m = load();
    m.insert(key.to_string(), value.to_string());
    save(&m)
}

pub fn remove(key: &str) -> Result<()> {
    let mut m = load();
    m.remove(key);
    save(&m)
}

/// The shortest secret that is replaced by value (a shorter one would hit ordinary words).
const MIN_REDACT: usize = 8;

/// The stored secret values worth hiding, longest first. Read again only when `secrets.json` changed (it is asked for every line of every log).
fn redaction_values() -> Vec<String> {
    use std::sync::{Mutex, OnceLock};
    use std::time::SystemTime;
    type Cached = (PathBuf, Option<SystemTime>, u64, Vec<String>);
    static CACHE: OnceLock<Mutex<Option<Cached>>> = OnceLock::new();
    let f = file();
    let meta = std::fs::metadata(&f).ok();
    let stamp = (meta.as_ref().and_then(|m| m.modified().ok()), meta.as_ref().map(|m| m.len()).unwrap_or(0));
    let cell = CACHE.get_or_init(|| Mutex::new(None));
    let mut g = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, t, n, v)) = g.as_ref() {
        if *p == f && *t == stamp.0 && *n == stamp.1 {
            return v.clone();
        }
    }
    let mut values: Vec<String> = if meta.is_some() { std::fs::read_to_string(&f).ok().and_then(|t| serde_json::from_str::<BTreeMap<String, String>>(&t).ok()).unwrap_or_default().into_values().collect() } else { Vec::new() };
    values.retain(|v| v.chars().count() >= MIN_REDACT);
    values.sort_by_key(|v| std::cmp::Reverse(v.len()));
    values.dedup();
    *g = Some((f, stamp.0, stamp.1, values.clone()));
    values
}

/// A text with every stored secret (a teacher or Hugging Face key, a training-server token, an Unsloth key ...) replaced by `[secret]`, and then the patterns of keys and tokens that
/// were not stored here (`sk-unsloth-...`, `hf_...`) replaced as before. Every log line, job log, error text and problem report goes through it.
pub fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for v in redaction_values() {
        if out.contains(&v) {
            out = out.replace(&v, "[secret]");
        }
    }
    crate::unsloth::client::scrub(&out)
}
