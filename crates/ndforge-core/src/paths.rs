//! Where ND Lore Forge keeps things. `NDFORGE_CONFIG_DIR` / `NDFORGE_PROJECTS_DIR` override the defaults (tests, portable installs).
use std::path::PathBuf;

pub fn config_dir() -> PathBuf {
    match std::env::var_os("NDFORGE_CONFIG_DIR") {
        Some(p) => PathBuf::from(p),
        None => dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")).join("NDLoreForge"),
    }
}

pub fn default_projects_dir() -> PathBuf {
    match std::env::var_os("NDFORGE_PROJECTS_DIR") {
        Some(p) => PathBuf::from(p),
        None => dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join("NDLoreForge").join("Projects"),
    }
}
