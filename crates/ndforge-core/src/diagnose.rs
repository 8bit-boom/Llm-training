//! "Make a problem report": a plain-text summary of this computer, the training software and the open project that a person can paste into a
//! message. It never contains access keys or tokens (anything that looks like an Unsloth key is replaced), and the home folder name is replaced by `~`.
use std::fmt::Write;

use crate::envmgr::health::{evaluate, gather, Light};
use crate::envmgr::{env_dir, find_uv, python_path, worker_dir};
use crate::paths::config_dir;
use crate::project::Project;
use crate::settings::Settings;
use crate::train::runs::list_runs;

pub(crate) fn tidy(text: &str) -> String {
    let text = crate::secrets::redact(text);
    let text = text.as_str();
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
    let _ = writeln!(r, "unsloth: {}", settings.unsloth.as_ref().map(|u| format!("{} ({})", u.name, u.url)).unwrap_or_else(|| "none".into()));
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

// ---------------------------------------------------------------------------------------------------- the diagnostics bundle

/// The newest file called `name` one level below `dir` (by its modification time), as (folder name, path).
fn newest_below(dir: &std::path::Path, name: &str) -> Option<(String, std::path::PathBuf)> {
    std::fs::read_dir(dir).ok()?.flatten().filter_map(|e| {
        let f = e.path().join(name);
        let t = std::fs::metadata(&f).ok()?.modified().ok()?;
        Some((t, e.file_name().to_string_lossy().to_string(), f))
    }).max_by_key(|(t, _, _)| *t).map(|(_, n, f)| (n, f))
}

/// Everything a helper needs to look at a problem, in one zip: the problem report, the last 10 job logs, the settings without any key, the environment's marker, what the worker says
/// about itself, the llama.cpp install record, the newest Auto-cut run, heal and export records of the project, and the card. Every text goes through `secrets::redact` and the home folder
/// becomes `~`. `dest` is a folder (the file is named `ndforge-diagnostics-<time>.zip`) or a path ending in `.zip`. Returns the file.
pub fn bundle(project: Option<&Project>, settings: &Settings, dest: &std::path::Path) -> crate::error::Result<std::path::PathBuf> {
    use std::io::Write;
    let file = if dest.extension().is_some_and(|e| e.eq_ignore_ascii_case("zip")) {
        dest.to_path_buf()
    } else {
        std::fs::create_dir_all(dest)?;
        dest.join(format!("ndforge-diagnostics-{}.zip", chrono::Utc::now().format("%Y%m%d-%H%M%S")))
    };
    let mut entries: Vec<(String, String)> = Vec::new();
    let mut add = |name: &str, text: String| entries.push((name.to_string(), tidy(&text)));
    add("report.txt", report(project, settings));
    for (i, log) in crate::envmgr::joblog::latest(10).iter().enumerate() {
        let name = log.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| format!("job-{i}.log"));
        add(&format!("logs/jobs/{name}"), std::fs::read_to_string(log).unwrap_or_default());
    }
    // the settings as the program keeps them (addresses and names stay; keys and tokens live in secrets.json, which is never read here)
    let mut st = serde_json::to_value(settings).unwrap_or_default();
    if let Some(o) = st.as_object_mut() {
        o.remove("load_note");
        if !o.get("webhook_url").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
            o.insert("webhook_url".into(), serde_json::json!("(set)"));          // such an address may carry a secret topic
        }
    }
    add("settings.json", serde_json::to_string_pretty(&st).unwrap_or_default());
    let env = env_dir();
    if let Ok(t) = std::fs::read_to_string(env.join("ndforge_env.json")) {
        add("ndforge_env.json", t);
    }
    let py = python_path(&env);
    add("worker_hello.json", if py.is_file() {
        match crate::envmgr::jobs::run_job(&py, &worker_dir(), &serde_json::json!({"job": "hello"}), &std::sync::atomic::AtomicBool::new(false), &mut |_| {}) {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_default(),
            Err(e) => format!("The worker did not answer: {}", e.0),
        }
    } else {
        "The training software is not installed.".to_string()
    });
    if let Ok(t) = std::fs::read_to_string(crate::envmgr::llamacpp::Config::from_env().install_dir().join("install.json")) {
        add("llama_cpp_install.json", t);
    }
    if let Some(p) = project {
        let read = |f: &std::path::Path| std::fs::read_to_string(f).ok();
        if let Some(st) = crate::reshape::auto::latest(p) {
            let dir = crate::reshape::auto::run_dir(p, &st.id);
            if let Some(t) = read(&dir.join("auto.json")) { add("auto.json", t); }
            if let Some(t) = read(&dir.join("report.md")).or_else(|| read(&dir.join("auto_report.md"))) { add("auto_report.md", t); }
        }
        if let Some((name, f)) = newest_below(&crate::reshape::runs::models_dir(p), "heal.json") {
            if let Some(t) = read(&f) { add("heal.json", format!("// model folder: {name}\n{t}")); }
        }
        if let Some((name, f)) = newest_below(&crate::export::runs::exports_dir(p), "meta.json") {
            if let Some(t) = read(&f) { add("export.json", format!("// export folder: {name}\n{t}")); }
        }
    }
    add("graphics_card.txt", match crate::envmgr::gpu::detect_gpus() { Ok(g) => format!("{g:#?}"), Err(e) => e });
    let mut w = zip::ZipWriter::new(std::fs::File::create(&file)?);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    for (name, text) in entries {
        w.start_file(name, opts).map_err(|e| crate::error::Error::new(format!("Zip problem: {e}")))?;
        w.write_all(text.as_bytes())?;
    }
    w.finish().map_err(|e| crate::error::Error::new(format!("Zip problem: {e}")))?;
    Ok(file)
}
