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
    /// a button the window can offer next to the fix (`"keep_awake_on"`: switch the setting on); absent when there is none
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<&'static str>,
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
    /// the environment was made by a setup that did not yet check PyTorch for CUDA (its PyTorch may be the CPU-only build)
    pub env_cuda_unchecked: bool,
    /// the requirements version the installed environment was made with (`ENV_REQUIREMENTS_VERSION` of its setup); None: an older setup that did not write one
    pub env_requirements: Option<u32>,
    /// flash-linear-attention is installed in the environment (its package folder is there); None: no environment to look in. Qwen3.5's linear-attention layers are several times slower without it.
    pub fast_kernels: Option<bool>,
}

/// The version of what the setup installs (the upper bounds in `worker/constraints.txt`, the order of the steps). Raise it when a program version needs another environment: the setup page then
/// asks to install again. The environment's marker file keeps the version it was made with.
pub const ENV_REQUIREMENTS_VERSION: u32 = 1;

const DISK_FAIL_GB: f64 = 15.0;
const DISK_WARN_GB: f64 = 40.0;
/// the smallest of the five models trains in about this much (QLoRA, short context)
const VRAM_MIN_GB: f64 = 5.5;

fn check(id: &'static str, label: &'static str, light: Light, detail: impl Into<String>, fix: impl Into<String>) -> Check {
    Check { id, label, light, detail: detail.into(), fix: fix.into(), action: None }
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
    out.push(if f.env_installed && f.env_requirements.is_none_or(|v| v < ENV_REQUIREMENTS_VERSION) && !f.env_cuda_unchecked {
        check("env", "Training software", Light::Warn, "Installed, but made by an older version of this program.",
              "Press \"Install the training software\" once more so that it matches this program (downloads are reused).")
    } else if f.env_installed && f.env_cuda_unchecked {
        check("env", "Training software", Light::Warn, "Installed, but PyTorch was not checked for the graphics card when it was installed.",
              "Press \"Install the training software\" once more: it checks PyTorch and, if it is the CPU-only build, replaces it with the NVIDIA build (downloads are reused).")
    } else if f.env_installed {
        check("env", "Training software", Light::Ok, "Installed.", "")
    } else {
        check("env", "Training software", Light::Warn, "Not installed yet.", "Press \"Install the training software\" (several GB, one time).")
    });
    out.push(if f.selftest_passed {
        check("selftest", "Graphics card test", Light::Ok, "Passed: the card trained a tiny model.", "")
    } else {
        check("selftest", "Graphics card test", Light::Warn, "Not run yet.", "After the install, press \"Test the graphics card\" (about 30 seconds).")
    });
    match f.fast_kernels {
        Some(true) => out.push(check("kernels", "Fast kernels for Qwen3.5 linear attention", Light::Ok, "flash-linear-attention is installed.", "")),
        Some(false) => out.push(check("kernels", "Fast kernels for Qwen3.5 linear attention", Light::Warn,
            "flash-linear-attention is not installed: Qwen3.5's linear-attention layers run on a much slower PyTorch version when measuring, healing and checking.",
            "Press \"Install the fast kernels\" (a small download; if there is no ready-made package for this system nothing is changed).")),
        None => {}
    }
    out
}

/// The light for "does the installed Triton belong to the installed torch": Ok when it does, a warning with the plain sentence when it does not (never red: the heal still runs, on slower or
/// failing kernels). None when no Triton is installed or it cannot be compared.
pub fn triton_check(t: &super::setup::TritonStatus) -> Option<Check> {
    t.triton.as_ref()?;
    match t.matches? {
        true => Some(check("triton", "Triton matches PyTorch", Light::Ok, t.line(), "")),
        false => Some(check("triton", "Triton matches PyTorch", Light::Warn, t.line(), "Press \"Install fast kernels\" (Settings, training software): it installs the Triton this PyTorch asks for. Nothing is installed automatically.")),
    }
}

/// `triton_check` for the environment that exists (Windows only: elsewhere Triton comes with torch). Asks its Python for metadata only; None when there is nothing to say.
pub fn triton_check_now() -> Option<Check> {
    let env = super::env_dir();
    if !(cfg!(windows) || std::env::var_os("NDFORGE_TRITON_CHECK").is_some()) || !env_installed(&env) {
        return None;
    }
    let py = super::python_path(&env);
    triton_check(&super::setup::read_triton_status(&py.to_string_lossy())?)
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

/// A marker from before the setup checked PyTorch for CUDA (no `"cuda_torch": true` in it).
pub fn env_cuda_unchecked(env: &Path) -> bool {
    env.join("ndforge_env.json").is_file() && !std::fs::read_to_string(env.join("ndforge_env.json")).map(|t| t.contains("\"cuda_torch\": true")).unwrap_or(false)
}

/// What the setup installed: "unsloth", "standard" (the standard libraries only) or None.
pub fn env_engine(env: &Path) -> Option<String> {
    let t = std::fs::read_to_string(env.join("ndforge_env.json")).ok()?;
    Some(if t.contains("\"unsloth\": true") { "unsloth" } else { "standard" }.to_string())
}

/// The requirements version written in the marker file.
pub fn env_requirements(env: &Path) -> Option<u32> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(env.join("ndforge_env.json")).ok()?).ok()?;
    v["requirements"].as_u64().map(|n| n as u32)
}

/// Is flash-linear-attention installed in this environment? Looks for its `flash_linear_attention-*.dist-info` folder in the site-packages
/// (Windows `Lib/site-packages`, Unix `lib/python3*/site-packages`); nothing is imported. None when there is no environment.
pub fn env_fast_kernels(env: &Path) -> Option<bool> {
    if !env_installed(env) {
        return None;
    }
    let mut sites = vec![env.join("Lib").join("site-packages")];
    if let Ok(rd) = std::fs::read_dir(env.join("lib")) {
        sites.extend(rd.flatten().filter(|e| e.file_name().to_string_lossy().starts_with("python")).map(|e| e.path().join("site-packages")));
    }
    let found = sites.iter().filter_map(|s| std::fs::read_dir(s).ok()).flatten().flatten().any(|e| {
        let n = e.file_name().to_string_lossy().to_lowercase();
        n.starts_with("flash_linear_attention-") && n.ends_with(".dist-info")
    });
    Some(found)
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
        env_cuda_unchecked: env_cuda_unchecked(&env),
        env_requirements: env_requirements(&env),
        fast_kernels: env_fast_kernels(&env),
    }
}
