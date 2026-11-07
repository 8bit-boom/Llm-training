//! The training environment: finding the graphics card, checking the machine, installing the Python environment with `uv`, and
//! running jobs in the Python worker (`worker/`). Nothing here needs a GPU to compile or test.
pub mod gpu;
pub mod health;
pub mod jobs;
pub mod process;
pub mod setup;
pub mod tasks;


use std::path::{Path, PathBuf};

use crate::paths::config_dir;

/// Where the Python environment lives (one per user, shared by all projects).
pub fn env_dir() -> PathBuf {
    match std::env::var_os("NDFORGE_ENV_DIR") {
        Some(p) => PathBuf::from(p),
        None => config_dir().join("env"),
    }
}

/// The Python of an environment folder.
pub fn python_path(env: &Path) -> PathBuf {
    if cfg!(windows) { env.join("Scripts").join("python.exe") } else { env.join("bin").join("python") }
}

/// The folder with the worker package: `NDFORGE_WORKER_DIR`, a `worker` folder next to the program (installed), or the repository's (development).
pub fn worker_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("NDFORGE_WORKER_DIR") {
        return PathBuf::from(p);
    }
    if let Some(next_to_exe) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("worker"))).filter(|d| d.join("ndforge_worker").exists()) {
        return next_to_exe;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worker")
}

/// `uv`: `NDFORGE_UV`, a copy next to the program, or on PATH.
pub fn find_uv() -> Option<PathBuf> {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(p) = std::env::var_os("NDFORGE_UV").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(p) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join(name))).filter(|p| p.is_file()) {
        return Some(p);
    }
    std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(name)).find(|p| p.is_file()))
}
