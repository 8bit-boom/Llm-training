//! "Is this machine ready?": the checklist with green / yellow / red lights shown on the setup page.
use std::path::Path;

use serde::Serialize;

use super::gpu::{cuda_wheel_for, Gpu};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Light {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub id: &'static str,
    pub label: &'static str,
    pub light: Light,
    pub detail: String,
    /// what to do about it (empty when green)
    pub fix: String,
}

/// Everything the checks look at, gathered once (so the rules can be tested without a real machine).
#[derive(Debug, Clone)]
pub struct Facts {
    pub gpus: Result<Vec<Gpu>, String>,
    pub free_disk_gb: Option<f64>,
    pub uv_found: bool,
    /// the Python environment exists and has the worker installed
    pub env_installed: bool,
    /// the GPU self-test passed in this environment
    pub selftest_passed: bool,
}

const DISK_FAIL_GB: f64 = 15.0;
const DISK_WARN_GB: f64 = 40.0;
/// the smallest of the five models trains in about this much (QLoRA, short context)
const VRAM_MIN_GB: f64 = 5.5;

fn check(id: &'static str, label: &'static str, light: Light, detail: impl Into<String>, fix: impl Into<String>) -> Check {
    Check { id, label, light, detail: detail.into(), fix: fix.into() }
}

pub fn evaluate(f: &Facts) -> Vec<Check> {
    let mut out = Vec::new();
    match &f.gpus {
        Err(why) => {
            out.push(check("gpu", "NVIDIA graphics card", Light::Fail, why.clone(), "Install or update the NVIDIA driver, then press \"Check again\"."));
        }
        Ok(gpus) => {
            let g = &gpus[0];
            out.push(check("gpu", "NVIDIA graphics card", Light::Ok,
                format!("{} · {:.0} GB · driver {}{}", g.name, g.vram_gb(), g.driver, if gpus.len() > 1 { format!(" (+{} more, the largest is used)", gpus.len() - 1) } else { String::new() }), ""));
            match cuda_wheel_for(g) {
                Ok(w) => out.push(check("driver", "Driver new enough", Light::Ok, format!("Works with the {w} build of PyTorch."), "")),
                Err(why) => out.push(check("driver", "Driver new enough", Light::Fail, why, "Update the NVIDIA driver, then press \"Check again\".")),
            }
            let gb = g.vram_gb();
            out.push(if gb < VRAM_MIN_GB {
                check("vram", "Graphics memory", Light::Fail, format!("{gb:.0} GB is too little to train even the smallest model."), "Use a card with 8 GB or more.")
            } else if gb < 12.0 {
                check("vram", "Graphics memory", Light::Warn, format!("{gb:.0} GB: only the small models (Qwen3.5 2B, Gemma 4 E2B) will fit."), "")
            } else {
                check("vram", "Graphics memory", Light::Ok, format!("{gb:.0} GB"), "")
            });
        }
    }
    out.push(match f.free_disk_gb {
        None => check("disk", "Free disk space", Light::Warn, "Could not measure it.", ""),
        Some(gb) if gb < DISK_FAIL_GB => check("disk", "Free disk space", Light::Fail, format!("{gb:.0} GB free; at least {DISK_FAIL_GB:.0} GB is needed."), "Free some space on the drive that holds your projects and the program's data."),
        Some(gb) if gb < DISK_WARN_GB => check("disk", "Free disk space", Light::Warn, format!("{gb:.0} GB free. Models and checkpoints are big: {DISK_WARN_GB:.0} GB or more is comfortable."), ""),
        Some(gb) => check("disk", "Free disk space", Light::Ok, format!("{gb:.0} GB free"), ""),
    });
    out.push(if f.uv_found {
        check("uv", "Installer tool (uv)", Light::Ok, "Found.", "")
    } else {
        check("uv", "Installer tool (uv)", Light::Fail, "The tool that installs the training software (uv) was not found.",
              "Install it from https://docs.astral.sh/uv/ (one command), then press \"Check again\".")
    });
    out.push(if f.env_installed {
        check("env", "Training software", Light::Ok, "Installed.", "")
    } else {
        check("env", "Training software", Light::Warn, "Not installed yet.", "Press \"Install the training software\" (several GB, one time).")
    });
    out.push(if f.selftest_passed {
        check("selftest", "Graphics card test", Light::Ok, "Passed: the card trained a tiny model.", "")
    } else {
        check("selftest", "Graphics card test", Light::Warn, "Not run yet.", "After the install, press \"Test the graphics card\" (about 30 seconds).")
    });
    out
}

/// True when nothing is red.
pub fn can_install(checks: &[Check]) -> bool {
    checks.iter().filter(|c| matches!(c.id, "gpu" | "driver" | "vram" | "disk" | "uv")).all(|c| c.light != Light::Fail)
}

/// Free space on the drive holding `dir` (its nearest existing parent).
pub fn free_disk_gb(dir: &Path) -> Option<f64> {
    // `NDFORGE_FREE_DISK_GB` replaces the measurement (tests, or a drive that reports its size wrongly)
    if let Some(v) = std::env::var("NDFORGE_FREE_DISK_GB").ok().and_then(|v| v.parse::<f64>().ok()) {
        return Some(v);
    }
    let mut d = dir;
    while !d.exists() {
        d = d.parent()?;
    }
    fs2::available_space(d).ok().map(|b| b as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// The marker files written by the setup and by a passed self-test.
pub fn env_installed(env: &Path) -> bool {
    env.join("ndforge_env.json").is_file()
}

pub fn selftest_passed(env: &Path) -> bool {
    env.join("selftest.json").is_file()
}

pub fn gather() -> Facts {
    let env = super::env_dir();
    Facts {
        gpus: super::gpu::detect_gpus(),
        free_disk_gb: free_disk_gb(&env),
        uv_found: super::find_uv().is_some(),
        env_installed: env_installed(&env),
        selftest_passed: selftest_passed(&env),
    }
}
