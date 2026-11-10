//! "Make a problem report": a plain-text summary of this computer, the training software and the open project that a person can paste into a
//! message. It never contains access keys or tokens, and the home folder name is replaced by `~`.
use std::fmt::Write;

use crate::envmgr::health::{evaluate, gather, Light};
use crate::envmgr::{env_dir, find_uv, python_path, worker_dir};
use crate::paths::config_dir;
use crate::project::Project;
use crate::settings::Settings;
use crate::train::runs::list_runs;

fn tidy(text: &str) -> String {
    match dirs::home_dir().map(|h| h.to_string_lossy().to_string()).filter(|h| h.len() > 1) {
        Some(h) => text.replace(&h, "~"),
        None => text.to_string(),
    }
}

pub fn report(project: Option<&Project>, settings: &Settings) -> String {
    let mut r = String::new();
    let _ = writeln!(r, "ND Lore Forge {} · {} {}", env!("CARGO_PKG_VERSION"), std::env::consts::OS, std::env::consts::ARCH);
    let _ = writeln!(r, "\n== This computer ==");
    let facts = gather();
    for c in evaluate(&facts) {
        let mark = match c.light { Light::Ok => "OK  ", Light::Warn => "WARN", Light::Fail => "FAIL" };
        let _ = writeln!(r, "[{mark}] {}: {}{}", c.label, c.detail, if c.fix.is_empty() { String::new() } else { format!(" -> {}", c.fix) });
    }
    let _ = writeln!(r, "\n== Training software ==");
    let (env, py, worker) = (env_dir(), python_path(&env_dir()), worker_dir());
    let _ = writeln!(r, "environment: {} ({})", env.display(), if py.is_file() { "python found" } else { "python missing" });
    let _ = writeln!(r, "worker: {} ({})", worker.display(), if worker.join("ndforge_worker").exists() { "found" } else { "MISSING" });
    let _ = writeln!(r, "uv: {}", find_uv().map(|p| p.display().to_string()).unwrap_or_else(|| "not found".into()));
    let _ = writeln!(r, "settings folder: {}", config_dir().display());
    let _ = writeln!(r, "ollama servers: {}", settings.ollama_servers.iter().map(|s| format!("{} ({})", s.name, s.url)).collect::<Vec<_>>().join(", "));
    let _ = writeln!(r, "training servers: {}", if settings.train_servers.is_empty() { "none".to_string() } else { settings.train_servers.iter().map(|s| format!("{} ({})", s.name, s.url)).collect::<Vec<_>>().join(", ") });
    let _ = writeln!(r, "teacher: {} / {}", settings.teacher.url, if settings.teacher.model.is_empty() { "(no model chosen)" } else { &settings.teacher.model });
    let _ = writeln!(r, "\n== Project ==");
    match project {
        None => { let _ = writeln!(r, "no project open"); }
        Some(p) => {
            let _ = writeln!(r, "{} (world: {}) in {}", p.name, p.display_name(), p.folder.display());
            let _ = writeln!(r, "steps done: {}", p.progress().iter().map(|(k, d)| format!("{k}={}", if *d { "yes" } else { "no" })).collect::<Vec<_>>().join(", "));
            for run in list_runs(p).into_iter().take(5) {
                let g = |k: &str| run[k].as_str().unwrap_or("").to_string();
                let _ = writeln!(r, "run {}: {} {} {} step {}/{}{}", g("name"), g("model"), g("method"), g("status"), run["step"], run["total_steps"],
                    run["error"].as_str().map(|e| format!(" — {e}")).unwrap_or_default());
            }
        }
    }
    tidy(&r)
}
