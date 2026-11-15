//! Which llama.cpp release file fits this computer. Pure functions over the list of asset names a release has and a description of the machine: no network, no
//! GPU needed, so every rule is tested with several realistic lists. Asset names are matched by lower-case tokens (operating system, architecture, backend), never by
//! a fixed name: the project renames its files now and then. An unknown list is never an exception: the answer is a plain message that lists what was found.
use serde::{Deserialize, Serialize};

use crate::envmgr::gpu::Gpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Os {
    Windows,
    Linux,
    Macos,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Arch {
    X64,
    Arm64,
    Other,
}

impl Os {
    pub fn current() -> Os {
        if cfg!(windows) { Os::Windows } else if cfg!(target_os = "macos") { Os::Macos } else if cfg!(target_os = "linux") { Os::Linux } else { Os::Other }
    }
    pub fn label(self) -> &'static str {
        match self { Os::Windows => "windows", Os::Linux => "linux", Os::Macos => "macos", Os::Other => "other" }
    }
}

impl Arch {
    pub fn current() -> Arch {
        match std::env::consts::ARCH { "x86_64" => Arch::X64, "aarch64" => Arch::Arm64, _ => Arch::Other }
    }
    pub fn label(self) -> &'static str {
        match self { Arch::X64 => "x64", Arch::Arm64 => "arm64", Arch::Other => "other" }
    }
}

/// One file of a release, as the GitHub API lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub url: String,
    /// "sha256:<hex>" when the API gives one
    #[serde(default)]
    pub digest: Option<String>,
}

impl Asset {
    pub fn new(name: &str, size: u64) -> Asset {
        Asset { name: name.to_string(), size, url: String::new(), digest: None }
    }
}

/// What the NVIDIA driver and card allow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Nvidia {
    pub name: String,
    /// the highest CUDA version the driver supports (nvidia-smi header, or derived from the driver number)
    pub driver_cuda: Option<(u32, u32)>,
    pub compute_cap: Option<(u32, u32)>,
    pub blackwell: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BuildTools {
    pub nvcc: bool,
    pub cmake: bool,
    pub cxx: bool,
}

impl BuildTools {
    pub fn complete(&self) -> bool {
        self.nvcc && self.cmake && self.cxx
    }
}

/// The machine, as far as the choice cares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Machine {
    pub os: Os,
    pub arch: Arch,
    pub nvidia: Option<Nvidia>,
    /// an AMD or Intel graphics card is present (the Vulkan build can use it)
    pub vulkan_likely: bool,
    pub tools: BuildTools,
}

impl Machine {
    pub fn cpu_only(os: Os, arch: Arch) -> Machine {
        Machine { os, arch, nvidia: None, vulkan_likely: false, tools: BuildTools { nvcc: false, cmake: false, cxx: false } }
    }
}

/// What the person (or the automatic choice) asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Want {
    Auto,
    Cuda,
    Vulkan,
    Cpu,
    Build,
}

impl Want {
    pub fn parse(s: &str) -> Want {
        match s.trim().to_lowercase().as_str() { "cuda" => Want::Cuda, "vulkan" => Want::Vulkan, "cpu" => Want::Cpu, "build" => Want::Build, _ => Want::Auto }
    }
}

/// The result: what to download (or build) and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Choice {
    /// "cuda" / "vulkan" / "cpu" / "metal" (macOS Apple silicon) / "build"
    pub backend: String,
    pub cuda_version: Option<String>,
    /// the program files, and for Windows CUDA the matching runtime DLLs
    pub assets: Vec<Asset>,
    pub reason: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChooseError(pub String);

// ---------------------------------------------------------------------------------------------------------------------------- reading the names

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// plain CPU build; the number ranks the variants (the new "cpu", the old "avx2" ...), higher is better
    Cpu(u8),
    Cuda(Option<(u32, u32)>),
    Vulkan,
    /// hip / rocm / sycl / openvino / opencl / kompute / musa / cann / kleidiai ... : not chosen automatically
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    pub os: Option<Os>,
    pub arch: Option<Arch>,
    pub kind: Kind,
    /// the CUDA runtime DLLs (`cudart-llama-bin-win-cuda-12.4-x64.zip`)
    pub cudart: bool,
    /// a CI artefact, not a release file
    pub ci: bool,
    pub archive: Option<Archive>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archive {
    Zip,
    TarGz,
}

pub fn archive_kind(name: &str) -> Option<Archive> {
    let n = name.to_lowercase();
    if n.ends_with(".zip") { Some(Archive::Zip) } else if n.ends_with(".tar.gz") || n.ends_with(".tgz") { Some(Archive::TarGz) } else { None }
}

/// "12.4" / "cu12.2.0" / "cuda12.4" -> (12, 4)
fn cuda_version_token(t: &str) -> Option<(u32, u32)> {
    let t = t.strip_prefix("cuda").or_else(|| t.strip_prefix("cu")).unwrap_or(t);
    let mut it = t.split('.');
    let major: u32 = it.next()?.parse().ok()?;
    let minor: u32 = it.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    if !(9..=40).contains(&major) {
        return None;
    }
    Some((major, minor))
}

pub fn parse_asset(name: &str) -> Parsed {
    let lower = name.to_lowercase();
    let archive = archive_kind(&lower);
    let stem = lower.trim_end_matches(".zip").trim_end_matches(".tar.gz").trim_end_matches(".tgz");
    let tokens: Vec<&str> = stem.split(|c| c == '-' || c == '_').filter(|t| !t.is_empty()).collect();
    let (mut os, mut arch) = (None, None);
    let (mut cuda, mut cuda_ver, mut vulkan, mut cpu, mut ci) = (false, None, false, false, false);
    let mut cpu_rank: u8 = 0;
    let mut other: Option<String> = None;
    let cudart = tokens.first() == Some(&"cudart");
    for t in &tokens {
        match *t {
            "win" | "windows" => os = Some(Os::Windows),
            "ubuntu" | "linux" => os = Some(Os::Linux),
            "macos" | "osx" | "darwin" => os = Some(Os::Macos),
            "x64" | "x86_64" | "amd64" => arch = Some(Arch::X64),
            "arm64" | "aarch64" => arch = Some(Arch::Arm64),
            "s390x" | "ppc64le" | "riscv64" | "x86" | "i386" | "armv7" => arch = Some(Arch::Other),
            "cuda" => cuda = true,
            "vulkan" => vulkan = true,
            "cpu" => { cpu = true; cpu_rank = cpu_rank.max(9) }
            "avx2" => { cpu = true; cpu_rank = cpu_rank.max(7) }
            "avx" => { cpu = true; cpu_rank = cpu_rank.max(6) }
            "avx512" => { cpu = true; cpu_rank = cpu_rank.max(5) }
            "noavx" => { cpu = true; cpu_rank = cpu_rank.max(3) }
            "openblas" => { cpu = true; cpu_rank = cpu_rank.max(2) }
            "ci" | "debug" | "test" | "tests" => ci = true,
            "llama" | "bin" | "cudart" | "server" | "cli" => {}
            t if t.starts_with('b') && t.len() > 1 && t[1..].chars().all(|c| c.is_ascii_digit()) => {} // the tag
            t if t.starts_with("cu") && cuda_version_token(t).is_some() => { cuda = true; cuda_ver = cuda_version_token(t) }
            t if t.chars().next().is_some_and(|c| c.is_ascii_digit()) && cuda_version_token(t).is_some() => cuda_ver = cuda_ver.or_else(|| cuda_version_token(t)),
            t => other = Some(other.map_or(t.to_string(), |o| format!("{o}-{t}"))),
        }
    }
    let kind = if let Some(o) = other {
        Kind::Other(o)
    } else if cuda {
        Kind::Cuda(cuda_ver)
    } else if vulkan {
        Kind::Vulkan
    } else if cpu {
        Kind::Cpu(cpu_rank)
    } else {
        Kind::Cpu(8) // "ubuntu-x64", "macos-arm64": the plain build
    };
    Parsed { os, arch, kind, cudart, ci, archive }
}

fn archive_rank(a: Option<Archive>) -> u8 {
    match a { Some(Archive::Zip) => 2, Some(Archive::TarGz) => 1, None => 0 }
}

fn version_text(v: (u32, u32)) -> String {
    format!("{}.{}", v.0, v.1)
}

/// The assets that are program files for this OS and architecture (not the CUDA runtime, not CI, only archives we can open), best archive format first.
fn binaries<'a>(assets: &'a [Asset], os: Os, arch: Arch) -> Vec<(Parsed, &'a Asset)> {
    let mut v: Vec<(Parsed, &Asset)> = assets.iter().map(|a| (parse_asset(&a.name), a)).filter(|(p, _)| p.os == Some(os) && p.arch == Some(arch) && !p.cudart && !p.ci && p.archive.is_some()).collect();
    v.sort_by(|a, b| archive_rank(b.0.archive).cmp(&archive_rank(a.0.archive)));
    v
}

fn cudart_for<'a>(assets: &'a [Asset], os: Os, arch: Arch, ver: (u32, u32)) -> Option<&'a Asset> {
    assets.iter().map(|a| (parse_asset(&a.name), a)).filter(|(p, _)| p.cudart && p.os == Some(os) && p.arch == Some(arch) && !p.ci && p.archive.is_some() && p.kind == Kind::Cuda(Some(ver)))
        .max_by_key(|(p, _)| archive_rank(p.archive)).map(|(_, a)| a)
}

fn names(assets: &[Asset]) -> String {
    if assets.is_empty() { return "(the release lists no files)".into(); }
    let mut n: Vec<&str> = assets.iter().map(|a| a.name.as_str()).collect();
    n.sort();
    let shown: Vec<&str> = n.iter().take(40).copied().collect();
    format!("{}{}", shown.join(", "), if n.len() > 40 { format!(", ... ({} more)", n.len() - 40) } else { String::new() })
}

fn no_match(assets: &[Asset], what: &str, m: &Machine) -> ChooseError {
    ChooseError(format!(
        "This llama.cpp release has no file for {what} ({} {}). The files it has: {}. Install llama.cpp by hand instead (see docs/LLAMA_CPP_INSTALL.md): download a build from https://github.com/ggml-org/llama.cpp/releases \
         and the source, then set the environment variable LLAMA_CPP_DIR to its folder.", m.os.label(), m.arch.label(), names(assets)))
}

/// The highest CUDA build this card and driver can use. CUDA's minor-version compatibility lets a build run on a driver that reports the same MAJOR version (a driver
/// reporting 13.0 runs the 13.5 build), so only the major versions are compared; the minor is informational. An RTX 50 card (Blackwell) needs a build and a driver of
/// CUDA 12.8 or newer. On Windows the matching runtime-files package is required (a version without one is skipped); on Linux it is taken when the release has it.
fn pick_cuda<'a>(assets: &'a [Asset], m: &Machine) -> std::result::Result<(Asset, Vec<Asset>, (u32, u32)), String> {
    let Some(n) = &m.nvidia else { return Err("no NVIDIA graphics card was found".into()) };
    let Some(driver) = n.driver_cuda else { return Err("the NVIDIA driver's CUDA version could not be read".into()) };
    let mut found: Vec<((u32, u32), &Asset)> = binaries(assets, m.os, m.arch).into_iter().filter_map(|(p, a)| match p.kind { Kind::Cuda(Some(v)) => Some((v, a)), _ => None }).collect();
    if found.is_empty() {
        return Err("this release has no CUDA build for this system".into());
    }
    found.sort_by(|a, b| b.0.cmp(&a.0));
    let mut why = Vec::new();
    if n.blackwell && driver < (12, 8) {
        why.push(format!("a newer NVIDIA driver is needed for the CUDA build of this card (an RTX 50 card needs a driver for CUDA 12.8 or newer; yours supports CUDA {})", version_text(driver)));
        return Err(why.join("; "));
    }
    for (v, a) in &found {
        if v.0 > driver.0 {
            why.push(format!("a newer NVIDIA driver is needed for the CUDA {} build (yours supports CUDA {})", version_text(*v), version_text(driver)));
            continue;
        }
        if n.blackwell && *v < (12, 8) {
            why.push(format!("CUDA {} is too old for an RTX 50 card (CUDA 12.8 or newer is needed)", version_text(*v)));
            continue;
        }
        let rt = cudart_for(assets, m.os, m.arch, *v);
        if m.os == Os::Windows && rt.is_none() {
            why.push(format!("CUDA {} has no runtime-files package in this release", version_text(*v)));
            continue;
        }
        return Ok(((*a).clone(), rt.into_iter().cloned().collect(), *v));
    }
    Err(why.join("; "))
}

fn best_cpu(assets: &[Asset], m: &Machine) -> Option<Asset> {
    binaries(assets, m.os, m.arch).into_iter().filter_map(|(p, a)| match p.kind { Kind::Cpu(r) => Some((r, archive_rank(p.archive), a)), _ => None }).max_by_key(|(r, ar, _)| (*r, *ar)).map(|(_, _, a)| a.clone())
}

fn vulkan(assets: &[Asset], m: &Machine) -> Option<Asset> {
    binaries(assets, m.os, m.arch).into_iter().find(|(p, _)| p.kind == Kind::Vulkan).map(|(_, a)| a.clone())
}

/// The choice. Never panics, never fails for an odd list: `Err` carries the plain text for the person.
pub fn choose(assets: &[Asset], m: &Machine, want: Want) -> std::result::Result<Choice, ChooseError> {
    let mut notes: Vec<String> = Vec::new();
    if matches!(m.os, Os::Other) || matches!(m.arch, Arch::Other) {
        return Err(ChooseError(format!("There is no ready-made llama.cpp for this system ({} {}). Build llama.cpp yourself from https://github.com/ggml-org/llama.cpp and set LLAMA_CPP_DIR to its folder.", m.os.label(), m.arch.label())));
    }
    if want == Want::Build {
        if m.os != Os::Linux {
            return Err(ChooseError("Building llama.cpp with CUDA from the source is offered on Linux only. On Windows the ready-made CUDA build is used.".into()));
        }
        if !m.tools.complete() {
            let mut missing = Vec::new();
            if !m.tools.nvcc { missing.push("nvcc (the CUDA toolkit)"); }
            if !m.tools.cmake { missing.push("cmake"); }
            if !m.tools.cxx { missing.push("a C++ compiler (g++ or clang++)"); }
            return Err(ChooseError(format!("Building needs {} on the PATH, and it was not found. Install it, or use the ready-made build.", missing.join(", "))));
        }
        return Ok(Choice { backend: "build".into(), cuda_version: None, assets: Vec::new(), reason: "Built here from the source with CUDA (takes a long time, 20 minutes or more).".into(), notes });
    }
    let mk = |backend: &str, a: Asset, extra: Vec<Asset>, cuda: Option<(u32, u32)>, reason: String, notes: Vec<String>| {
        let mut assets = vec![a];
        assets.extend(extra);
        Choice { backend: backend.into(), cuda_version: cuda.map(version_text), assets, reason, notes }
    };
    // macOS: Metal is part of the Apple silicon build; Intel Macs get the CPU build
    if m.os == Os::Macos {
        let a = best_cpu(assets, m).ok_or_else(|| no_match(assets, "this Mac", m))?;
        let (b, r) = if m.arch == Arch::Arm64 { ("metal", "The Apple silicon build (it uses the graphics chip through Metal).") } else { ("cpu", "The Intel Mac build (processor only).") };
        if matches!(want, Want::Cuda | Want::Vulkan) {
            notes.push("There is no CUDA or Vulkan build for a Mac; the normal one was chosen.".into());
        }
        return Ok(mk(b, a, Vec::new(), None, r.into(), notes));
    }
    match want {
        Want::Cpu => {
            let a = best_cpu(assets, m).ok_or_else(|| no_match(assets, "a processor-only build", m))?;
            return Ok(mk("cpu", a, Vec::new(), None, "The processor-only build, as you chose.".into(), notes));
        }
        Want::Vulkan => {
            let a = vulkan(assets, m).ok_or_else(|| no_match(assets, "the Vulkan build", m))?;
            return Ok(mk("vulkan", a, Vec::new(), None, "The Vulkan build (works with NVIDIA, AMD and Intel cards), as you chose.".into(), notes));
        }
        Want::Cuda => {
            return match pick_cuda(assets, m) {
                Ok((a, rt, v)) => Ok(mk("cuda", a, rt, Some(v), format!("The CUDA {} build for your NVIDIA card.", version_text(v)), notes)),
                Err(why) => Err(ChooseError(format!("No CUDA build fits this computer: {why}. {}", names_hint(assets)))),
            };
        }
        _ => {}
    }
    // automatic
    if m.nvidia.is_some() {
        match pick_cuda(assets, m) {
            Ok((a, rt, v)) => {
                let gpu = m.nvidia.as_ref().map(|n| n.name.clone()).unwrap_or_default();
                return Ok(mk("cuda", a, rt, Some(v), format!("The CUDA {} build: {gpu} and its driver can use it.", version_text(v)), notes));
            }
            Err(why) => notes.push(format!("The CUDA build was not chosen: {why}.")),
        }
    }
    if m.vulkan_likely || (m.nvidia.is_some() && m.os == Os::Windows) {
        if let Some(a) = vulkan(assets, m) {
            let r = if m.nvidia.is_some() { "The Vulkan build (it works with your NVIDIA card; no CUDA build fits)." } else { "The Vulkan build, for your AMD or Intel graphics card." };
            return Ok(mk("vulkan", a, Vec::new(), None, r.into(), notes));
        }
    }
    let a = best_cpu(assets, m).ok_or_else(|| no_match(assets, "this computer", m))?;
    let r = if m.os == Os::Linux && m.nvidia.is_some() { "The ready-made Linux build (processor only: no ready-made CUDA build fits, see the notes; you can build one with CUDA from the source in Expert mode)." } else { "The processor-only build (no graphics card that a ready-made build can use was found)." };
    Ok(mk("cpu", a, Vec::new(), None, r.into(), notes))
}

fn names_hint(assets: &[Asset]) -> String {
    format!("The files of this release: {}.", names(assets))
}

/// What the person may choose in Expert mode: each backend with whether it can be had here, and why not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Option_ {
    pub id: &'static str,
    pub label: &'static str,
    pub available: bool,
    pub note: String,
}

pub fn options(assets: &[Asset], m: &Machine) -> Vec<Option_> {
    let mut out = Vec::new();
    for (id, label, want) in [("cuda", "NVIDIA (CUDA)", Want::Cuda), ("vulkan", "Vulkan (AMD, Intel, NVIDIA)", Want::Vulkan), ("cpu", "Processor only", Want::Cpu), ("build", "Build with CUDA from the source (Linux, long)", Want::Build)] {
        if m.os == Os::Macos && id != "cpu" {
            continue;
        }
        let r = choose(assets, m, want);
        out.push(Option_ { id, label, available: r.is_ok(), note: r.err().map(|e| e.0).unwrap_or_default() });
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------------- the machine

/// "CUDA Version: 12.9" from the header of a plain `nvidia-smi` call.
pub fn parse_cuda_version(text: &str) -> Option<(u32, u32)> {
    let i = text.find("CUDA Version:")?;
    let rest = text[i + "CUDA Version:".len()..].trim_start();
    let tok: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let mut it = tok.split('.');
    Some((it.next()?.parse().ok()?, it.next().and_then(|m| m.parse().ok()).unwrap_or(0)))
}

/// The CUDA version an NVIDIA driver supports, from the driver number, when nvidia-smi's header could not be read.
/// (NVIDIA's release notes: 525 -> 12.0, 535 -> 12.2, 545 -> 12.3, 550 -> 12.4, 555 -> 12.5, 560 -> 12.6, 565 -> 12.7, 570 -> 12.8, 575 -> 12.9, 580 -> 13.0.)
pub fn cuda_from_driver(major: u32) -> Option<(u32, u32)> {
    let table = [(580, (13, 0)), (575, (12, 9)), (570, (12, 8)), (565, (12, 7)), (560, (12, 6)), (555, (12, 5)), (550, (12, 4)), (545, (12, 3)), (535, (12, 2)), (525, (12, 0)), (520, (11, 8))];
    table.iter().find(|(d, _)| major >= *d).map(|(_, v)| *v)
}

pub fn nvidia_from(gpus: &[Gpu], smi_header: &str) -> Option<Nvidia> {
    let g = gpus.first()?;
    let driver_cuda = parse_cuda_version(smi_header).or_else(|| crate::envmgr::gpu::driver_version(&g.driver).and_then(|(major, _)| cuda_from_driver(major)));
    Some(Nvidia { name: g.name.clone(), driver_cuda, compute_cap: g.compute_cap, blackwell: gpus.iter().any(|x| x.is_blackwell()) })
}

/// Does a list of graphics-card names (one per line) have an AMD or Intel card (the Vulkan build can use it)?
pub fn vulkan_vendor_in(names: &str) -> bool {
    let n = names.to_lowercase();
    ["amd", "radeon", "intel", "arc ", "iris", "uhd graphics"].iter().any(|k| n.contains(k)) && !n.trim().is_empty()
}

fn on_path(name: &str) -> bool {
    let exe = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(&exe).is_file()))
}

pub fn detect_tools() -> BuildTools {
    BuildTools { nvcc: on_path("nvcc"), cmake: on_path("cmake"), cxx: ["c++", "g++", "clang++"].iter().any(|c| on_path(c)) }
}

fn run_text(program: &str, args: &[&str]) -> Option<String> {
    let mut cmd = std::process::Command::new(program);
    crate::envmgr::process::quiet(&mut cmd).args(args);
    let out = cmd.output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr))
}

fn graphics_names() -> String {
    if cfg!(windows) {
        return run_text("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"]).unwrap_or_default();
    }
    if cfg!(target_os = "linux") {
        // vendor ids in sysfs: 0x1002 AMD, 0x8086 Intel
        let mut s = String::new();
        if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
            for e in rd.flatten() {
                if let Ok(v) = std::fs::read_to_string(e.path().join("device/vendor")) {
                    match v.trim() { "0x1002" => s.push_str("AMD\n"), "0x8086" => s.push_str("Intel\n"), _ => {} }
                }
            }
        }
        return s;
    }
    String::new()
}

/// Looks at this computer (nvidia-smi, the graphics cards, the build tools).
pub fn detect_machine() -> Machine {
    let nvidia = crate::envmgr::gpu::detect_gpus().ok().and_then(|g| nvidia_from(&g, &run_text("nvidia-smi", &[]).unwrap_or_default()));
    Machine { os: Os::current(), arch: Arch::current(), nvidia, vulkan_likely: vulkan_vendor_in(&graphics_names()), tools: detect_tools() }
}
