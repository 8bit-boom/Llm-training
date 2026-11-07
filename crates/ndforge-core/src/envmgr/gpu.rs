//! Finding the NVIDIA card(s) with `nvidia-smi` and deciding what the machine can do (precision, which CUDA build of PyTorch).
use std::process::Command;

use serde::Serialize;

use super::process::quiet;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Gpu {
    pub name: String,
    pub vram_mb: u64,
    pub driver: String,
    /// (major, minor), from nvidia-smi or guessed from the card's name
    pub compute_cap: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Bf16,
    Fp16,
}

impl Gpu {
    /// bf16 needs compute capability 8.0 (Ampere) or newer; older cards (T4, T10, V100) use fp16.
    pub fn precision(&self) -> Precision {
        match self.compute_cap {
            Some((major, _)) if major >= 8 => Precision::Bf16,
            _ => Precision::Fp16,
        }
    }

    /// RTX 50 series and other Blackwell cards: they need CUDA 12.8 or newer.
    pub fn is_blackwell(&self) -> bool {
        matches!(self.compute_cap, Some((major, _)) if major >= 10)
    }

    pub fn vram_gb(&self) -> f64 {
        self.vram_mb as f64 / 1024.0
    }
}

/// When nvidia-smi cannot tell the compute capability (old drivers): a guess from the name.
fn guess_cap(name: &str) -> Option<(u32, u32)> {
    let n = name.to_uppercase();
    let has = |s: &str| n.contains(s);
    if has("RTX 50") || has("RTX PRO 6000 BLACKWELL") || has("B200") || has("B100") {
        Some((12, 0))
    } else if has("H100") || has("H200") {
        Some((9, 0))
    } else if has("RTX 40") || has("L40") || has("L4") {
        Some((8, 9))
    } else if has("RTX 30") || has("A100") || has("A40") || has("A10") {
        Some((8, 6))
    } else if has("RTX 20") || has("T4") || has("T10") || has("TITAN RTX") {
        Some((7, 5))
    } else if has("V100") {
        Some((7, 0))
    } else {
        None
    }
}

/// Parses `nvidia-smi --query-gpu=name,memory.total,driver_version[,compute_cap] --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(text: &str) -> Vec<Gpu> {
    text.lines().filter_map(|line| {
        let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
        if parts.len() < 3 {
            return None;
        }
        let vram_mb: u64 = parts[1].parse().ok()?;
        let compute_cap = parts.get(3).and_then(|c| {
            let (a, b) = c.split_once('.')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }).or_else(|| guess_cap(parts[0]));
        Some(Gpu { name: parts[0].to_string(), vram_mb, driver: parts[2].to_string(), compute_cap })
    }).collect()
}

fn query(fields: &str) -> Result<String, String> {
    let mut cmd = Command::new("nvidia-smi");
    quiet(&mut cmd).args([&format!("--query-gpu={fields}"), "--format=csv,noheader,nounits"]);
    let out = cmd.output().map_err(|_| "nvidia-smi was not found. Install the NVIDIA driver (https://www.nvidia.com/drivers).".to_string())?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr))
    }
}

/// The NVIDIA cards of this machine, best (most memory) first.
pub fn detect_gpus() -> Result<Vec<Gpu>, String> {
    let text = match query("name,memory.total,driver_version,compute_cap") {
        Ok(t) => t,
        Err(e) if e.contains("not found") => return Err(e),
        Err(_) => query("name,memory.total,driver_version")?, // an older driver without the compute_cap field
    };
    let mut gpus = parse_nvidia_smi(&text);
    if gpus.is_empty() {
        return Err("nvidia-smi found no graphics card.".to_string());
    }
    gpus.sort_by(|a, b| b.vram_mb.cmp(&a.vram_mb));
    Ok(gpus)
}

/// "570.133.07" -> (570, 133)
pub fn driver_version(v: &str) -> Option<(u32, u32)> {
    let mut it = v.trim().split('.');
    let major = it.next()?.parse().ok()?;
    Some((major, it.next().and_then(|m| m.parse().ok()).unwrap_or(0)))
}

/// Which PyTorch CUDA build (the name in https://download.pytorch.org/whl/<name>) this driver can run, or what to do about it.
/// CUDA 12.8 needs driver 570+, CUDA 12.6 needs 560+. Re-check this table when PyTorch changes its builds.
pub fn cuda_wheel_for(gpu: &Gpu) -> Result<&'static str, String> {
    let Some((major, _)) = driver_version(&gpu.driver) else {
        return Err(format!("The driver version \"{}\" could not be read.", gpu.driver));
    };
    if major >= 570 {
        return Ok("cu128");
    }
    if gpu.is_blackwell() {
        return Err(format!("The {} needs NVIDIA driver 570 or newer; this computer has {}. Update the driver at https://www.nvidia.com/drivers.", gpu.name, gpu.driver));
    }
    if major >= 560 {
        return Ok("cu126");
    }
    Err(format!("The NVIDIA driver {} is too old for the training software. Driver 560 or newer is needed (570 or newer is best). Update it at https://www.nvidia.com/drivers.", gpu.driver))
}
