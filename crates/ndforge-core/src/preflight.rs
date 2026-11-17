//! One checklist before a job starts: "is everything this job needs there?". The same lights as the setup page (`envmgr::health::Check`), asked for a kind of job:
//! `auto` (Auto-cut), `heal`, `export`, `think_check`, `quantcheck`, `imatrix_make`, `cut` and `train`. The window shows the list above the Start button and disables Start
//! while a light is red. Nothing here starts a job or changes anything; a check the program cannot make is left out, never invented.
//!
//! `spec` is what the page knows (every key is optional): `need_gb` (the page's own disk estimate), `run` + `quants` + `what` (an export), `tokens_m` + `mode` (a heal),
//! `imatrix` (an export with an importance matrix), `studio_publish`, `server` (a training server does the work), `verdict` ("fits" | "tight" | "over" | "unknown") with
//! `verdict_text` (the memory estimate), `blockers` (reasons the page's own plan already found).
use serde_json::Value;

use crate::envmgr::health::{Check, Light};
use crate::project::Project;

pub const KINDS: [&str; 8] = ["auto", "heal", "export", "think_check", "quantcheck", "imatrix_make", "cut", "train"];

/// How much a job needs llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    No,
    /// the job can go another way without it (an export through Unsloth)
    Maybe,
    Yes,
}

/// What the checks look at, gathered once, so the rules can be tested without a real machine.
#[derive(Debug, Clone)]
pub struct Inputs {
    pub kind: String,
    pub windows: bool,
    pub non_ascii_path: bool,
    pub env_installed: bool,
    pub selftest_passed: bool,
    /// `Some(None)`: asked and there is no card; `Some(Some(gb))`; None: not asked
    pub card: Option<Option<f64>>,
    pub llama: Need,
    /// the tools this job needs are all there, and what they are called (for the message)
    pub llama_found: bool,
    pub llama_what: &'static str,
    pub free_gb: Option<f64>,
    pub need_gb: Option<f64>,
    pub has_data: Option<bool>,
    pub verdict: Option<(String, String)>,
    pub blockers: Vec<String>,
    pub other_instance: bool,
    /// when the files go to Unsloth Studio: is the share's folder there? None: not publishing
    pub share_reachable: Option<bool>,
}

fn c(id: &'static str, label: &'static str, light: Light, detail: impl Into<String>, fix: impl Into<String>) -> Check {
    Check { id, label, light, detail: detail.into(), fix: fix.into() }
}

/// The checks for these inputs, in the order the list shows them.
pub fn evaluate(i: &Inputs) -> Vec<Check> {
    let mut out = Vec::new();
    if i.other_instance {
        out.push(c("instance", "Only one ND Lore Forge", Light::Fail, "Another ND Lore Forge is using the same settings folder.", "Close the other window first: two at once would damage each other's files."));
    }
    out.push(if i.env_installed {
        c("env", "Training software", Light::Ok, "Installed.", "")
    } else {
        c("env", "Training software", Light::Fail, "Not installed yet.", "Open \"Training setup\" and press \"Install the training software\".")
    });
    if matches!(i.kind.as_str(), "train" | "heal" | "auto") && i.env_installed {
        out.push(if i.selftest_passed {
            c("selftest", "Graphics card test", Light::Ok, "Passed.", "")
        } else {
            c("selftest", "Graphics card test", Light::Warn, "Not run yet.", "In \"Training setup\" press \"Test the graphics card\" (about 30 seconds).")
        });
    }
    match i.card {
        Some(None) => out.push(c("gpu", "NVIDIA graphics card", Light::Fail, "No NVIDIA graphics card was found.", "Install or update the NVIDIA driver, or let a training server do this.")),
        Some(Some(gb)) => out.push(c("gpu", "NVIDIA graphics card", Light::Ok, format!("{gb:.0} GB"), "")),
        None => {}
    }
    match (i.llama, i.llama_found) {
        (Need::No, _) => {}
        (_, true) => out.push(c("llama", "llama.cpp", Light::Ok, "Found.", "")),
        (Need::Yes, false) => out.push(c("llama", "llama.cpp", Light::Fail, format!("llama.cpp was not found ({} is needed).", i.llama_what), "Open \"Training setup\" and press \"Install llama.cpp\".")),
        (Need::Maybe, false) => out.push(c("llama", "llama.cpp", Light::Warn, "llama.cpp was not found. Plain sizes can still be made with Unsloth; mixed sizes and the importance matrix need it.", "Open \"Training setup\" and press \"Install llama.cpp\".")),
    }
    if i.windows && i.non_ascii_path && i.llama != Need::No {
        out.push(c("path", "Folder name", Light::Warn, "The project folder has letters that are not plain English letters. On Windows some llama.cpp tools cannot open such a path.",
                   "If a step fails with \"cannot open file\", move the project to a folder with a plain name such as C:\\Projects."));
    }
    if let Some(need) = i.need_gb.filter(|n| *n > 0.0) {
        out.push(match i.free_gb {
            None => c("disk", "Free disk space", Light::Warn, format!("About {need:.0} GB are needed; the free space could not be measured."), ""),
            Some(free) if free < need * 1.05 => c("disk", "Free disk space", Light::Fail, format!("About {need:.1} GB are needed and {free:.1} GB are free."), "Free some space on the drive that holds the project, or choose a smaller job."),
            Some(free) if free < need * 1.3 => c("disk", "Free disk space", Light::Warn, format!("About {need:.1} GB are needed and {free:.1} GB are free: little to spare."), ""),
            Some(free) => c("disk", "Free disk space", Light::Ok, format!("About {need:.1} GB are needed, {free:.0} GB are free."), ""),
        });
    } else if let Some(free) = i.free_gb.filter(|f| *f < 15.0) {
        out.push(c("disk", "Free disk space", Light::Fail, format!("Only {free:.0} GB are free."), "Free some space on the drive that holds the project."));
    }
    if let Some((verdict, text)) = &i.verdict {
        out.push(match verdict.as_str() {
            "over" => c("card", "Graphics memory", Light::Fail, text.clone(), "Choose a smaller model, shorter text, QLoRA or a safer setting."),
            "tight" => c("card", "Graphics memory", Light::Warn, text.clone(), ""),
            "fits" => c("card", "Graphics memory", Light::Ok, text.clone(), ""),
            _ => c("card", "Graphics memory", Light::Warn, if text.is_empty() { "The memory this needs is not known yet.".to_string() } else { text.clone() }, ""),
        });
    }
    if i.has_data == Some(false) {
        out.push(c("data", "Training data", Light::Fail, "There is no saved training-data version.", "Save a training-data version in Step 2 first."));
    }
    if let Some(ok) = i.share_reachable {
        out.push(if ok {
            c("share", "Unsloth Studio folder", Light::Ok, "The shared folder is there.", "")
        } else {
            c("share", "Unsloth Studio folder", Light::Warn, "The Studio share folder cannot be reached; the files will stay in the exports folder.", "Check the folder in the Auto-cut card (\"Check the folder\"), or untick publishing.")
        });
    }
    for b in &i.blockers {
        out.push(c("plan", "The plan", Light::Fail, b.clone(), ""));
    }
    out
}

/// True when nothing is red.
pub fn can_start(checks: &[Check]) -> bool {
    checks.iter().all(|c| c.light != Light::Fail)
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

/// The disk an export of a training run needs, from the run's model (None when the run or its model is not known).
fn export_need_gb(p: &Project, spec: &Value) -> Option<f64> {
    let run = spec["run"].as_str()?;
    let rdir = crate::train::runs::runs_dir(p).join(run);
    let cfg: Value = serde_json::from_slice(&std::fs::read(rdir.join("config.json")).ok()?).ok()?;
    let m = crate::train::catalog::find(cfg["model"]["id"].as_str()?)?;
    let what = spec["what"].as_str().unwrap_or("gguf");
    let mut quants = strs(&spec["quants"]);
    if quants.is_empty() {
        quants = spec["quant"].as_str().map(|q| vec![q.to_string()]).unwrap_or_default();
    }
    let is_full = rdir.join("model").join("config.json").is_file();
    Some(crate::export::runs::export_need_gb(m.params_b, !is_full, if what == "merged" { &[] } else { &quants }, what != "gguf"))
}

/// What a heal's teacher answers need on disk (the cache mode only; online healing keeps none).
fn heal_need_gb(spec: &Value) -> Option<f64> {
    let tokens_m = spec["tokens_m"].as_f64()?;
    (spec["mode"].as_str() == Some("cache")).then(|| tokens_m * 1e6 * 196.0 / 1e9)
}

/// Looks at the machine and the project for a kind of job.
pub fn gather(p: Option<&Project>, kind: &str, spec: &Value) -> Inputs {
    let env = crate::envmgr::env_dir();
    let needs_card = matches!(kind, "train" | "heal" | "auto") && spec["server"].as_str().is_none_or(|s| s.is_empty());
    let exports_gguf = spec["what"].as_str() != Some("merged");
    let mixed = strs(&spec["quants"]).iter().any(|q| q.starts_with("mixed"));
    let llama = match kind {
        "quantcheck" | "imatrix_make" | "auto" => Need::Yes,
        "export" if exports_gguf && (mixed || spec["imatrix"].as_bool() == Some(true)) => Need::Yes,
        "export" if exports_gguf => Need::Maybe,
        _ => Need::No,
    };
    let status = if llama == Need::No { Value::Null } else { crate::envmgr::llamacpp::status(&crate::envmgr::llamacpp::Config::from_env()) };
    let tool = |k: &str| !status["tools"][k].is_null();
    let (llama_found, llama_what) = match kind {
        "quantcheck" => (tool("perplexity"), "llama-perplexity"),
        "imatrix_make" => (tool("imatrix"), "llama-imatrix"),
        _ => (!status["found"].is_null(), "the converter and llama-quantize"),
    };
    let dir = p.map(|p| p.folder.clone()).unwrap_or_else(|| env.clone());
    let need_gb = spec["need_gb"].as_f64().filter(|n| *n > 0.0).or_else(|| match kind {
        "export" => p.and_then(|p| export_need_gb(p, spec)),
        "heal" => heal_need_gb(spec),
        _ => None,
    });
    let publishing = kind == "auto" && spec["studio_publish"].as_bool() == Some(true);
    Inputs {
        kind: kind.to_string(),
        windows: cfg!(windows),
        non_ascii_path: p.is_some_and(|p| !p.folder.to_string_lossy().is_ascii()),
        env_installed: crate::envmgr::health::env_installed(&env),
        selftest_passed: crate::envmgr::health::selftest_passed(&env),
        card: needs_card.then(|| crate::envmgr::gpu::detect_gpus_cached().ok().and_then(|g| g.first().map(|x| x.vram_gb()))),
        llama,
        llama_found,
        llama_what,
        free_gb: crate::envmgr::health::free_disk_gb(&dir),
        need_gb,
        has_data: matches!(kind, "train" | "heal" | "auto").then(|| p.is_some_and(|p| crate::train::runs::latest_version(p).is_some())),
        verdict: spec["verdict"].as_str().map(|v| (v.to_string(), spec["verdict_text"].as_str().unwrap_or("").to_string())),
        blockers: strs(&spec["blockers"]),
        other_instance: crate::instance::other_running(&crate::paths::config_dir()),
        share_reachable: publishing.then(|| crate::settings::Settings::load().studio_share.is_some_and(|s| !s.host_path.trim().is_empty() && std::path::Path::new(s.host_path.trim()).is_dir())),
    }
}

/// The checklist for a kind of job (an unknown kind has no checks).
pub fn check(p: Option<&Project>, kind: &str, spec: &Value) -> Vec<Check> {
    if !KINDS.contains(&kind) {
        return Vec::new();
    }
    evaluate(&gather(p, kind, spec))
}
