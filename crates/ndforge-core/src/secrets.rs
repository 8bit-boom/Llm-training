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

fn load() -> BTreeMap<String, String> {
    std::fs::read_to_string(file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn save(map: &BTreeMap<String, String>) -> Result<()> {
    let f = file();
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&f, serde_json::to_vec_pretty(map)?)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
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
