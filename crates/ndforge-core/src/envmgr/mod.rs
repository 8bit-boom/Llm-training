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

/// Folders an installed program keeps its data files in, most specific first: next to the program (Windows installer, portable), the
/// AppImage's `usr/lib/<name>` (`APPDIR`), a `.deb` / system install (`../lib/<name>`, `../share/<name>`).
pub fn resource_dirs_for(exe_dir: &Path, appdir: Option<&Path>) -> Vec<PathBuf> {
    let mut out = vec![exe_dir.to_path_buf(), exe_dir.join("resources")];
    let names = ["ND Lore Forge", "nd-lore-forge"];
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(a) = appdir {
        roots.push(a.join("usr"));
    }
    if let Some(parent) = exe_dir.parent() {
        roots.push(parent.to_path_buf());
    }
    for r in roots {
        for sub in ["lib", "share"] {
            for n in names {
                out.push(r.join(sub).join(n));
            }
        }
    }
    out
}

fn resource_dirs() -> Vec<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf));
    match exe_dir {
        Some(d) => resource_dirs_for(&d, std::env::var_os("APPDIR").map(PathBuf::from).as_deref()),
        None => Vec::new(),
    }
}

/// The folder with the worker package: `NDFORGE_WORKER_DIR`, an installed copy (see `resource_dirs_for`), or the repository's (development).
pub fn worker_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("NDFORGE_WORKER_DIR") {
        return PathBuf::from(p);
    }
    if let Some(d) = resource_dirs().into_iter().map(|d| d.join("worker")).find(|d| d.join("ndforge_worker").exists()) {
        return d;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worker")
}

/// The first existing `name` among the given folders and their `tools` subfolders.
pub fn find_tool_in(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    dirs.iter().flat_map(|d| [d.join(name), d.join("tools").join(name)]).find(|p| p.is_file())
}

/// `uv`: `NDFORGE_UV`, a copy shipped with the program, or on PATH.
pub fn find_uv() -> Option<PathBuf> {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(p) = std::env::var_os("NDFORGE_UV").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(p) = find_tool_in(&resource_dirs(), name) {
        return Some(p);
    }
    std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(name)).find(|p| p.is_file()))
}
