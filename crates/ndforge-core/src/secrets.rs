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
