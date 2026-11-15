//! Installing llama.cpp for the person (docs/LLAMA_CPP_INSTALL.md): the program files from the newest GitHub release (the build that fits this computer's graphics
//! card), the few source files `convert_hf_to_gguf.py` needs, a check that it starts, and a folder the Python worker finds on its own (`NDFORGE_LLAMA_CPP_DIR`).
//!
//! The install folder is `<the program's data folder>/llama.cpp/` with exactly the layout the worker's finder expects:
//!     convert_hf_to_gguf.py  conversion/  gguf-py/  models/templates/        (from the source of the SAME release)
//!     build/bin/llama-quantize(.exe) llama-imatrix llama-perplexity ...      (from the release's program files; the CUDA runtime DLLs next to them on Windows)
//!     install.json                                                            (what was installed, from where, with which checksums)
//! A new install is made in a side folder and only renamed into place after it passed the check, so a failed or cancelled install leaves the old one alone.
pub mod archive;
pub mod choose;
pub mod net;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use self::archive::{find_dir_with, unpack, EntryKind};
use self::choose::{choose, detect_machine, options, parse_asset, Archive, Asset, Choice, Machine, Os, Want};
use self::net::{download, fail, fetch_release, source_url, valid_tag};
use crate::envmgr::process::{run_lines, Stream};
use crate::error::Error;
use crate::paths::config_dir;

/// A failure with a short code (for the window and for tests) and a plain text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
}

impl From<Error> for Failure {
    fn from(e: Error) -> Failure {
        let code = if e.0 == "Cancelled." { "cancelled" } else { "error" };
        Failure { code, message: e.0 }
    }
}

impl From<std::io::Error> for Failure {
    fn from(e: std::io::Error) -> Failure {
        Failure { code: "disk", message: format!("File problem: {e}") }
    }
}

impl From<Failure> for Error {
    fn from(f: Failure) -> Error {
        Error::new(f.message)
    }
}

type Res<T> = std::result::Result<T, Failure>;

/// Where things are and where to ask. Tests build one by hand; the program uses `from_env`.
#[derive(Debug, Clone)]
pub struct Config {
    /// the program's data folder (the Python environment's parent)
    pub base_dir: PathBuf,
    pub api_base: String,
    pub web_base: String,
    /// a local test server may be contacted (set only through the hidden settings below)
    pub allow_local: bool,
    pub tag_pin: Option<String>,
    /// the training environment's Python (for the converter check), when it exists
    pub python: Option<PathBuf>,
    pub home: Option<PathBuf>,
    /// the worker folder (`<resources>/worker`); `llama.cpp` next to its parent is a place the finder looks
    pub worker_dir: PathBuf,
}

impl Config {
    /// `NDFORGE_LLAMA_CPP_TAG` pins a release; `NDFORGE_LLAMA_CPP_API_URL` / `NDFORGE_LLAMA_CPP_WEB_URL` (hidden, for tests and mirrors on this computer only) replace
    /// the GitHub addresses; only an address on this computer (127.0.0.1) is then accepted.
    pub fn from_env() -> Config {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let (api, web) = (var("NDFORGE_LLAMA_CPP_API_URL"), var("NDFORGE_LLAMA_CPP_WEB_URL"));
        let local = |u: &Option<String>| u.as_deref().is_some_and(|u| u.starts_with("http://127.0.0.1") || u.starts_with("http://localhost"));
        let allow_local = local(&api) || local(&web);
        let python = crate::envmgr::python_path(&crate::envmgr::env_dir());
        Config {
            base_dir: config_dir(),
            api_base: api.filter(|_| allow_local).unwrap_or_else(|| "https://api.github.com".into()),
            web_base: web.filter(|_| allow_local).unwrap_or_else(|| "https://github.com".into()),
            allow_local,
            tag_pin: var("NDFORGE_LLAMA_CPP_TAG"),
            python: python.is_file().then_some(python),
            home: dirs::home_dir(),
            worker_dir: crate::envmgr::worker_dir(),
        }
    }

    pub fn install_dir(&self) -> PathBuf {
        self.base_dir.join("llama.cpp")
    }
    fn staging_dir(&self) -> PathBuf {
        self.base_dir.join("llama.cpp.new")
    }
    fn old_dir(&self) -> PathBuf {
        self.base_dir.join("llama.cpp.old")
    }
    fn failure_file(&self) -> PathBuf {
        self.base_dir.join("llama.cpp-last-failure.json")
    }
}

pub fn exe(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

// ---------------------------------------------------------------------------------------------------------------------------- finding it

/// How a llama.cpp folder was found (the same order the Python worker uses, worker/ndforge_worker/llamacpp.py).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum How {
    EnvVar,
    ProgramInstall,
    HomeFolder,
    NextToProgram,
    Path,
}

impl How {
    pub fn label(self) -> &'static str {
        match self {
            How::EnvVar => "your LLAMA_CPP_DIR folder",
            How::ProgramInstall => "installed by this program",
            How::HomeFolder => "the llama.cpp folder in your home folder",
            How::NextToProgram => "the llama.cpp folder next to the program",
            How::Path => "the PATH",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    pub folder: PathBuf,
    pub how: How,
    pub convert: PathBuf,
    pub quantize: PathBuf,
}

/// (convert script, quantize program) of a folder, in the layouts `find_llama_cpp` accepts: the programs in the folder, in build/bin or in build/bin/Release.
pub fn layout_of(d: &Path) -> Option<(PathBuf, PathBuf)> {
    let conv = d.join("convert_hf_to_gguf.py");
    let q = exe("llama-quantize");
    let quant = [d.join(&q), d.join("build").join("bin").join(&q), d.join("build").join("bin").join("Release").join(&q)].into_iter().find(|p| p.is_file())?;
    conv.is_file().then_some((conv, quant))
}

/// The places to look, in order: LLAMA_CPP_DIR (always first), the folder this program installed, `~/llama.cpp`, `llama.cpp` next to the program.
pub fn candidates(cfg: &Config) -> Vec<(How, PathBuf)> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("LLAMA_CPP_DIR").filter(|d| !d.is_empty()) {
        v.push((How::EnvVar, PathBuf::from(d)));
    }
    v.push((How::ProgramInstall, cfg.install_dir()));
    if let Some(h) = &cfg.home {
        v.push((How::HomeFolder, h.join("llama.cpp")));
    }
    if let Some(p) = cfg.worker_dir.parent() {
        v.push((How::NextToProgram, p.join("llama.cpp")));
    }
    v
}

pub fn find_in(cands: &[(How, PathBuf)]) -> Option<Found> {
    cands.iter().find_map(|(how, d)| layout_of(d).map(|(convert, quantize)| Found { folder: d.clone(), how: *how, convert, quantize }))
}

pub fn find(cfg: &Config) -> Option<Found> {
    find_in(&candidates(cfg))
}

/// A tool (`llama-imatrix`, `llama-perplexity`) the way the worker finds it: next to llama-quantize, in any of the candidate folders (and their build folders), then on the PATH.
pub fn find_tool(cands: &[(How, PathBuf)], found: Option<&Found>, name: &str, path_dirs: &[PathBuf]) -> Option<(PathBuf, How)> {
    let e = exe(name);
    if let Some(f) = found {
        if let Some(q) = f.quantize.parent() {
            for d in [Some(q), q.parent()].into_iter().flatten() {
                if d.join(&e).is_file() {
                    return Some((d.join(&e), f.how));
                }
            }
        }
    }
    for (how, d) in cands {
        for dir in [d.clone(), d.join("build").join("bin"), d.join("build").join("bin").join("Release")] {
            if dir.join(&e).is_file() {
                return Some((dir.join(&e), *how));
            }
        }
    }
    path_dirs.iter().find(|d| d.join(&e).is_file()).map(|d| (d.join(&e), How::Path))
}

fn path_dirs() -> Vec<PathBuf> {
    std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default()
}

/// The folder to hand to the worker as `NDFORGE_LLAMA_CPP_DIR`: the install folder, when it exists.
pub fn worker_env(cfg: &Config) -> Vec<(String, String)> {
    let d = cfg.install_dir();
    if d.is_dir() { vec![("NDFORGE_LLAMA_CPP_DIR".to_string(), d.to_string_lossy().to_string())] } else { Vec::new() }
}

// ---------------------------------------------------------------------------------------------------------------------------- what is installed

pub const SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct InstallInfo {
    #[serde(default)]
    pub schema: u32,
    pub tag: String,
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub cuda_version: Option<String>,
    #[serde(default)]
    pub assets: Vec<String>,
    #[serde(default)]
    pub sha256: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub source_url: String,
    #[serde(default)]
    pub installed_at: String,
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub versions: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub converter_check: String,
    #[serde(default)]
    pub via: String,
}

pub fn read_info(dir: &Path) -> std::result::Result<Option<InstallInfo>, String> {
    let f = dir.join("install.json");
    if !f.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&f).map_err(|e| format!("install.json could not be read ({e})"))?;
    serde_json::from_str::<InstallInfo>(&text).map(Some).map_err(|_| "install.json is damaged".to_string())
}

/// Is the newer tag really newer? Tags look like b6000; others are compared as text (any difference is a hint).
pub fn update_available(installed: &str, latest: &str) -> bool {
    let num = |t: &str| t.strip_prefix('b').and_then(|n| n.parse::<u64>().ok());
    match (num(installed), num(latest)) {
        (Some(a), Some(b)) => b > a,
        _ => !installed.is_empty() && !latest.is_empty() && installed != latest,
    }
}

fn read_failure(cfg: &Config) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(cfg.failure_file()).ok()?).ok()
}

/// What the window shows and what the workers will find: no network, no GPU question.
pub fn status(cfg: &Config) -> Value {
    let info = read_info(&cfg.install_dir());
    let cands = candidates(cfg);
    let found = find_in(&cands);
    let pd = path_dirs();
    let tool = |name: &str| find_tool(&cands, found.as_ref(), name, &pd).map(|(p, how)| json!({"path": p, "how": how, "how_label": how.label()}));
    let layout_ok = layout_of(&cfg.install_dir()).is_some();
    json!({
        "install_dir": cfg.install_dir(),
        "platform": {"os": Os::current().label(), "arch": choose::Arch::current().label()},
        "installed": info.as_ref().ok().and_then(|i| i.as_ref()).map(|i| serde_json::to_value(i).unwrap_or(Value::Null)),
        "install_error": info.as_ref().err(),
        "layout_ok": layout_ok,
        "folder_without_record": cfg.install_dir().is_dir() && matches!(info, Ok(None)),
        "found": found.as_ref().map(|f| json!({"folder": f.folder, "how": f.how, "how_label": f.how.label(), "convert": f.convert, "quantize": f.quantize})),
        "tools": {"imatrix": tool("llama-imatrix"), "perplexity": tool("llama-perplexity")},
        "converter_found": found.is_some(),
        "last_failure": read_failure(cfg),
        "pinned_tag": cfg.tag_pin,
        "search_order": cands.iter().map(|(h, d)| json!({"how": h, "how_label": h.label(), "folder": d})).collect::<Vec<_>>(),
    })
}

pub fn remove(cfg: &Config) -> Res<()> {
    let dir = cfg.install_dir();
    if !dir.exists() {
        return Ok(());
    }
    if !dir.join("install.json").is_file() {
        return Err(fail("not_ours", format!("The folder {} was not made by this program (it has no install.json), so it is not deleted. Delete it yourself if you want it gone.", dir.display())));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| fail("disk", format!("Could not delete {} ({e}). Close programs that use it and try again.", dir.display())))?;
    let _ = std::fs::remove_file(cfg.failure_file());
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------------- running programs

#[derive(Debug, Clone, Default)]
pub struct RunOut {
    /// the program started (false: it could not be run at all)
    pub started: bool,
    pub code: Option<i32>,
    pub output: String,
    pub timed_out: bool,
}

/// Runs the programs of the self-test and of a source build. Tests replace it.
pub trait Runner: Sync {
    fn run(&self, program: &Path, args: &[String], cwd: Option<&Path>, timeout: Duration, cancel: &AtomicBool) -> RunOut;
    /// Runs to the end, every output line to `on_line`; Ok(exit code), or Err when it could not start or was cancelled.
    fn stream(&self, program: &Path, args: &[String], cwd: Option<&Path>, cancel: &AtomicBool, on_line: &mut dyn FnMut(&str)) -> Res<Option<i32>>;
}

pub struct RealRunner;

impl Runner for RealRunner {
    fn run(&self, program: &Path, args: &[String], cwd: Option<&Path>, timeout: Duration, cancel: &AtomicBool) -> RunOut {
        let mut cmd = std::process::Command::new(program);
        cmd.args(args);
        if let Some(c) = cwd {
            cmd.current_dir(c);
        }
        let (stop, done, timer) = (AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false));
        let mut output = String::new();
        let r = std::thread::scope(|s| {
            s.spawn(|| {
                let started = std::time::Instant::now();
                while !done.load(Ordering::SeqCst) {
                    if started.elapsed() >= timeout {
                        timer.store(true, Ordering::SeqCst);
                    }
                    if cancel.load(Ordering::SeqCst) || timer.load(Ordering::SeqCst) {
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
            });
            let r = run_lines(cmd, &stop, &mut |_: Stream, l: &str| {
                if output.len() < 8000 {
                    output.push_str(l);
                    output.push('\n');
                }
            });
            done.store(true, Ordering::SeqCst);
            r
        });
        match r {
            Ok(st) => RunOut { started: true, code: st.code(), output, timed_out: false },
            Err(e) if e.0 == "Cancelled." => RunOut { started: true, code: None, output, timed_out: timer.load(Ordering::SeqCst) },
            Err(e) => RunOut { started: false, code: None, output: e.0, timed_out: false },
        }
    }

    fn stream(&self, program: &Path, args: &[String], cwd: Option<&Path>, cancel: &AtomicBool, on_line: &mut dyn FnMut(&str)) -> Res<Option<i32>> {
        let mut cmd = std::process::Command::new(program);
        cmd.args(args);
        if let Some(c) = cwd {
            cmd.current_dir(c);
        }
        let st = run_lines(cmd, cancel, &mut |_: Stream, l: &str| on_line(l))?;
        Ok(st.code())
    }
}

// ---------------------------------------------------------------------------------------------------------------------------- the self-test

/// Did the program start and say something (a missing library or a driver that is too old shows as a loader error or a signal, not as output of the program)?
/// llama-quantize prints its usage and exits with 1 on `--help`: a non-zero exit code alone is not a failure.
pub fn started_fine(r: &RunOut) -> bool {
    if !r.started || r.timed_out || r.code.is_none() {
        return false;
    }
    let low = r.output.to_lowercase();
    let loader = ["error while loading shared libraries", "cannot open shared object", "library not loaded", "image not found", "was not found. reinstalling", "the code execution cannot proceed", "dll was not found", "cuda driver version is insufficient", "no cuda-capable device"];
    if loader.iter().any(|m| low.contains(m)) {
        return false;
    }
    // Windows: STATUS_DLL_NOT_FOUND, STATUS_INVALID_IMAGE_FORMAT, STATUS_ENTRYPOINT_NOT_FOUND; shells: 126 / 127
    !matches!(r.code, Some(-1073741515) | Some(-1073741701) | Some(-1073741511) | Some(126) | Some(127)) && !r.output.trim().is_empty()
}

fn version_line(out: &str) -> String {
    out.lines().find(|l| l.to_lowercase().contains("version")).map(|l| l.trim().chars().take(120).collect()).unwrap_or_default()
}

#[derive(Debug, Clone, Default)]
pub struct SelfTest {
    pub versions: std::collections::BTreeMap<String, String>,
    pub converter: String,
}

fn first_line(s: &str) -> String {
    s.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("(no output)").chars().take(240).collect()
}

/// `--help` of the three programs, the versions, and the converter's `--help` with the training environment's Python (skipped without it).
pub fn self_test(root: &Path, backend: &str, python: Option<&Path>, runner: &dyn Runner, cancel: &AtomicBool) -> Res<SelfTest> {
    let mut out = SelfTest::default();
    let (_, quantize) = layout_of(root).ok_or_else(|| fail("layout", "The installed files do not have the layout the program expects (convert_hf_to_gguf.py and llama-quantize)."))?;
    let bin = quantize.parent().unwrap_or(root).to_path_buf();
    for name in ["llama-quantize", "llama-imatrix", "llama-perplexity"] {
        if cancel.load(Ordering::SeqCst) {
            return Err(fail("cancelled", "Cancelled."));
        }
        let p = bin.join(exe(name));
        if !p.is_file() {
            return Err(fail("layout", format!("{name} is missing from the downloaded program files (this release may have renamed it). Install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md).")));
        }
        let r = runner.run(&p, &["--help".to_string()], Some(&bin), Duration::from_secs(30), cancel);
        if cancel.load(Ordering::SeqCst) {
            return Err(fail("cancelled", "Cancelled."));
        }
        if !started_fine(&r) {
            let why = if r.timed_out { "it did not answer within 30 seconds".to_string() } else { first_line(&r.output) };
            return Err(if backend == "cuda" {
                fail("cuda_start_failed", format!("The CUDA build of llama.cpp does not start on this computer ({name}: {why}). The NVIDIA driver may be too old or a part of it is missing. Press \"Install the CPU build instead\" to use the build that needs no graphics card."))
            } else {
                fail("start_failed", format!("The downloaded {name} does not start on this computer ({why}). {}", if backend == "cpu" || backend == "metal" { "Install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md)." } else { "Press \"Install the CPU build instead\"." }))
            });
        }
        if name != "llama-quantize" {
            let v = runner.run(&p, &["--version".to_string()], Some(&bin), Duration::from_secs(30), cancel);
            let line = version_line(&v.output);
            if !line.is_empty() {
                out.versions.insert(name.to_string(), line);
            }
        }
    }
    out.converter = match python {
        None => "not checked: the training software (Python) is not installed yet; the converter needs PyTorch. It is used, and so tested, the first time a model is converted.".into(),
        Some(py) => {
            let conv = root.join("convert_hf_to_gguf.py");
            let r = runner.run(py, &[conv.to_string_lossy().to_string(), "--help".to_string()], Some(root), Duration::from_secs(30), cancel);
            if cancel.load(Ordering::SeqCst) {
                return Err(fail("cancelled", "Cancelled."));
            }
            if r.timed_out {
                "not checked: the converter did not answer within 30 seconds (the first start of PyTorch can be slow)".into()
            } else if r.started && r.code == Some(0) {
                "ok".into()
            } else if r.output.contains("ModuleNotFoundError") || r.output.contains("No module named") {
                format!("not checked: the training software lacks a library the converter imports ({})", first_line(r.output.lines().rev().find(|l| l.contains("No module named")).unwrap_or(&r.output)))
            } else {
                return Err(fail("converter", format!("The converter script does not start with the training software's Python ({}). The files of this release may need more than the ones that were unpacked; install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md).", first_line(r.output.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("")))));
            }
        }
    };
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------------- the source subset

/// Which files of the source zip (`llama.cpp-<tag>/...`) the converter needs: the script, `conversion/`, `gguf-py/gguf/`, the chat templates it reads (`models/templates/`) and
/// the requirements files for reference. Not the tests, the docs, the vocab files or the C++ sources.
pub fn source_subset(parts: &[String], kind: &EntryKind) -> Option<Vec<String>> {
    let tail = parts.get(1..)?;
    if tail.is_empty() || tail.iter().any(|p| p == "__pycache__" || p.ends_with(".pyc")) {
        return None;
    }
    let first = tail[0].as_str();
    let keep = match first {
        "convert_hf_to_gguf.py" | "convert_hf_to_gguf_update.py" | "requirements.txt" | "LICENSE" => tail.len() == 1,
        "conversion" | "requirements" => true,
        "gguf-py" => match tail.get(1).map(String::as_str) {
            Some("gguf") => true,
            Some("pyproject.toml") | Some("README.md") | Some("LICENSE") => tail.len() == 2,
            None => matches!(kind, EntryKind::Dir),
            _ => false,
        },
        "models" => tail.get(1).map(String::as_str) == Some("templates") || (tail.len() == 1 && matches!(kind, EntryKind::Dir)),
        _ => false,
    };
    keep.then(|| tail.to_vec())
}

/// Does the unpacked converter have what it imports? (an older script has no `conversion/` folder; a newer one needs it)
pub fn verify_source(root: &Path) -> Res<()> {
    let script = root.join("convert_hf_to_gguf.py");
    let text = std::fs::read_to_string(&script).map_err(|_| fail("layout", "convert_hf_to_gguf.py was not in the source of this release."))?;
    if !root.join("gguf-py").join("gguf").join("__init__.py").is_file() {
        return Err(fail("layout", "gguf-py/gguf was not in the source of this release, which the converter needs."));
    }
    if text.contains("from conversion import") && !root.join("conversion").join("__init__.py").is_file() {
        return Err(fail("layout", "The converter imports its `conversion` folder, which was not in the source of this release."));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------------- the install

#[derive(Debug, Clone)]
pub struct Options {
    pub want: Want,
    pub tag: Option<String>,
}

/// About how big the source zip is (it is not listed with the release): shown in the size the person confirms.
pub const SOURCE_ESTIMATE_BYTES: u64 = 30_000_000;

/// What an install would download, for the confirmation: no file is touched.
pub fn plan(cfg: &Config, machine: &Machine, opts: &Options) -> Res<Value> {
    let tag = opts.tag.clone().or_else(|| cfg.tag_pin.clone());
    let rel = fetch_release(cfg, tag.as_deref())?;
    let choice = choose(&rel.assets, machine, opts.want).map_err(|e| fail("no_asset", e.0))?;
    let total: u64 = choice.assets.iter().map(|a| a.size).sum::<u64>() + SOURCE_ESTIMATE_BYTES;
    let opt = options(&rel.assets, machine);
    Ok(json!({
        "tag": rel.tag, "backend": choice.backend, "cuda_version": choice.cuda_version, "reason": choice.reason, "notes": choice.notes,
        "assets": choice.assets.iter().map(|a| json!({"name": a.name, "size": a.size})).collect::<Vec<_>>(),
        "total_bytes": total, "total_mb": (total as f64 / 1e6).round(),
        "options": opt, "build": choice.backend == "build",
        "gpu": machine.nvidia.as_ref().map(|n| json!({"name": n.name, "driver_cuda": n.driver_cuda.map(|(a, b)| format!("{a}.{b}")), "blackwell": n.blackwell})),
    }))
}

struct Stage(PathBuf);

impl Drop for Stage {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Emit<'a> {
    f: &'a mut dyn FnMut(Value),
    index: usize,
    total: usize,
}

impl Emit<'_> {
    fn step(&mut self, phase: &str, text: &str) {
        self.index += 1;
        (self.f)(json!({"kind": "step", "task": "llamacpp", "phase": phase, "index": self.index, "total": self.total, "text": text}));
    }
    fn progress(&mut self, phase: &str, value: f64, text: &str, bytes: Option<(u64, u64)>) {
        (self.f)(json!({"kind": "progress", "task": "llamacpp", "step": phase, "value": value.clamp(0.0, 1.0), "text": text, "bytes": bytes.map(|b| b.0), "total": bytes.map(|b| b.1)}));
    }
    fn log(&mut self, text: &str) {
        (self.f)(json!({"kind": "log", "task": "llamacpp", "text": text}));
    }
}

fn mb(n: u64) -> String {
    format!("{:.0} MB", n as f64 / 1e6)
}

fn check_cancel(cancel: &AtomicBool) -> Res<()> {
    if cancel.load(Ordering::SeqCst) { Err(fail("cancelled", "Cancelled.")) } else { Ok(()) }
}

fn archive_of(a: &Asset) -> Archive {
    parse_asset(&a.name).archive.unwrap_or(Archive::Zip)
}

/// Downloads one asset, with the overall progress across all downloads.
#[allow(clippy::too_many_arguments)]
fn fetch(cfg: &Config, em: &mut Emit, a: &Asset, dest: &Path, what: &str, done_before: u64, grand_total: u64, cancel: &AtomicBool) -> Res<String> {
    let text = format!("{what} ({})", mb(a.size.max(1)));
    em.step("download", &text);
    let mut local = |got: u64| {
        let all = done_before + got;
        em.progress("download", 0.02 + 0.68 * (all as f64 / grand_total.max(1) as f64), &format!("{what}: {} of {}", mb(got), mb(a.size.max(got))), Some((all, grand_total)));
    };
    download(cfg, &a.url, dest, a.size, a.digest.as_deref(), cancel, &mut local)
}

/// The install. `machine` says what fits; `runner` runs the self-test (and the build). Everything is made in a side folder and renamed into place at the end.
pub fn install(cfg: &Config, machine: &Machine, opts: &Options, runner: &dyn Runner, cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Res<Value> {
    let _ = std::fs::remove_file(cfg.failure_file());
    std::fs::create_dir_all(&cfg.base_dir)?;
    let mut em = Emit { f: on_event, index: 0, total: 6 };
    check_cancel(cancel)?;
    let tag_req = opts.tag.clone().or_else(|| cfg.tag_pin.clone());
    em.step("lookup", "Looking up the llama.cpp release on GitHub");
    em.progress("lookup", 0.01, "Looking up the release", None);
    let rel = fetch_release(cfg, tag_req.as_deref())?;
    check_cancel(cancel)?;
    let choice: Choice = choose(&rel.assets, machine, opts.want).map_err(|e| fail("no_asset", e.0))?;
    em.log(&format!("Release {}: {} ({})", rel.tag, choice.backend, choice.reason));
    for n in &choice.notes {
        em.log(n);
    }
    let building = choice.backend == "build";
    em.total = 1 + choice.assets.len() + 1 + 1 + 1 + 1 + usize::from(building);

    let stage_dir = cfg.staging_dir();
    let _ = std::fs::remove_dir_all(&stage_dir);
    let _ = std::fs::remove_dir_all(cfg.old_dir());
    std::fs::create_dir_all(&stage_dir)?;
    let guard = Stage(stage_dir.clone());
    let (dl, unp, root) = (stage_dir.join("dl"), stage_dir.join("unpack"), stage_dir.join("root"));
    std::fs::create_dir_all(&dl)?;
    std::fs::create_dir_all(&root)?;
    let grand: u64 = choice.assets.iter().map(|a| a.size).sum::<u64>() + SOURCE_ESTIMATE_BYTES;
    let mut sums = std::collections::BTreeMap::new();
    let mut done = 0u64;

    // the program files (and the CUDA runtime)
    let mut fetched: Vec<(Asset, PathBuf)> = Vec::new();
    for a in &choice.assets {
        let p = parse_asset(&a.name);
        let what = if p.cudart { "Downloading the CUDA runtime files" } else { "Downloading the program files" };
        let dest = dl.join(&a.name);
        let sum = fetch(cfg, &mut em, a, &dest, what, done, grand, cancel)?;
        done += a.size;
        sums.insert(a.name.clone(), sum);
        fetched.push((a.clone(), dest));
    }
    // the source of the same release
    let src_url = source_url(cfg, &rel.tag);
    let src_zip = dl.join("source.zip");
    let src_asset = Asset { name: format!("llama.cpp-{}.zip (source)", rel.tag), size: 0, url: src_url.clone(), digest: None };
    let src_sum = fetch(cfg, &mut em, &src_asset, &src_zip, "Downloading the converter (the source files)", done, grand, cancel)?;
    sums.insert("source.zip".to_string(), src_sum);

    // unpack
    em.step("unpack", "Unpacking the files");
    em.progress("unpack", 0.72, "Unpacking the files", None);
    let bin_dir = root.join("build").join("bin");
    for (a, p) in fetched.iter().filter(|(a, _)| !parse_asset(&a.name).cudart) {
        let to = unp.join("bin");
        unpack(p, archive_of(a), &to, &|parts, _| Some(parts.to_vec()), cancel)?;
        let from = find_dir_with(&to, &exe("llama-quantize")).ok_or_else(|| fail("layout", format!("{} does not contain llama-quantize. This release may have changed its files; install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md).", a.name)))?;
        std::fs::create_dir_all(bin_dir.parent().unwrap_or(&root))?;
        std::fs::rename(&from, &bin_dir).or_else(|_| copy_dir(&from, &bin_dir))?;
    }
    for (a, p) in fetched.iter().filter(|(a, _)| parse_asset(&a.name).cudart) {
        std::fs::create_dir_all(&bin_dir)?;
        let n = unpack(p, archive_of(a), &bin_dir, &|parts, kind| (*kind == EntryKind::File && parts.last().is_some_and(|l| l.to_lowercase().ends_with(".dll"))).then(|| vec![parts.last().cloned().unwrap_or_default()]), cancel)?;
        if n.files == 0 {
            return Err(fail("layout", format!("{} has no DLL files in it. This release may have changed its files.", a.name)));
        }
    }
    em.progress("unpack", 0.8, "Unpacking the converter", None);
    unpack(&src_zip, Archive::Zip, &root, &source_subset, cancel)?;
    verify_source(&root)?;
    if building {
        build_from_source(&src_zip, &stage_dir, &root, runner, cancel, &mut em)?;
    }
    check_cancel(cancel)?;

    // the check
    em.step("check", "Checking that it starts");
    em.progress("check", 0.86, "Checking that it starts", None);
    if layout_of(&root).is_none() {
        return Err(fail("layout", "After unpacking, the files do not have the layout the program expects. Install llama.cpp by hand (see docs/LLAMA_CPP_INSTALL.md)."));
    }
    let st = self_test(&root, &choice.backend, cfg.python.as_deref(), runner, cancel)?;
    check_cancel(cancel)?;

    // the record, then into place
    let info = InstallInfo {
        schema: SCHEMA, tag: rel.tag.clone(), backend: choice.backend.clone(), cuda_version: choice.cuda_version.clone(),
        assets: choice.assets.iter().map(|a| a.name.clone()).collect(), sha256: sums, source_url: src_url,
        installed_at: chrono::Utc::now().to_rfc3339(), os: machine.os.label().to_string(), arch: machine.arch.label().to_string(),
        versions: st.versions.clone(), converter_check: st.converter.clone(), via: if building { "build".into() } else { "download".into() },
    };
    std::fs::write(root.join("install.json"), serde_json::to_vec_pretty(&info).map_err(Error::from)?)?;
    em.step("finish", "Putting it in place");
    em.progress("finish", 0.97, "Putting it in place", None);
    check_cancel(cancel)?;
    swap_into_place(cfg, &root)?;
    drop(guard);
    em.progress("finish", 1.0, "Done", None);
    let mut out = serde_json::to_value(&info).map_err(Error::from)?;
    out["task"] = json!("llamacpp");
    Ok(out)
}

fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)?.flatten() {
        let (p, t) = (e.path(), to.join(e.file_name()));
        if p.is_dir() { copy_dir(&p, &t)?; } else { std::fs::copy(&p, &t)?; }
    }
    Ok(())
}

/// Renames the new folder into place; the old install is kept until the new one is there, and put back if that fails.
pub fn swap_into_place(cfg: &Config, new_root: &Path) -> Res<()> {
    let (dir, old) = (cfg.install_dir(), cfg.old_dir());
    let _ = std::fs::remove_dir_all(&old);
    let had = dir.exists();
    if had {
        std::fs::rename(&dir, &old).map_err(|e| fail("in_use", format!("The installed llama.cpp could not be replaced ({e}). Close programs that use it (an export or a matrix that is running) and try again; the old one is unchanged.")))?;
    }
    if let Err(e) = std::fs::rename(new_root, &dir) {
        if had {
            let _ = std::fs::rename(&old, &dir);
        }
        return Err(fail("in_use", format!("The new llama.cpp could not be put in place ({e}); the old one is unchanged.")));
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Linux with CUDA: cmake configure and build of the three programs from the source zip's full tree. Long; every output line of the build goes to the log, the
/// percentages of its `[ 45%]` lines to the progress bar; Cancel kills it.
fn build_from_source(src_zip: &Path, stage: &Path, root: &Path, runner: &dyn Runner, cancel: &AtomicBool, em: &mut Emit) -> Res<()> {
    em.step("build", "Building llama.cpp with CUDA (this takes 20 minutes or more)");
    let src = stage.join("src");
    unpack(src_zip, Archive::Zip, &src, &|parts, _| parts.get(1..).filter(|t| !t.is_empty()).map(|t| t.to_vec()), cancel)?;
    let build = src.join("build");
    let sp = src.to_string_lossy().to_string();
    let bp = build.to_string_lossy().to_string();
    let cmake = PathBuf::from("cmake");
    let mut last = 0.0f64;
    let mut on_line = |em: &mut Emit, l: &str| {
        em.log(l);
        if let Some(p) = l.trim_start().strip_prefix('[').and_then(|r| r.split('%').next()).and_then(|n| n.trim().parse::<f64>().ok()) {
            last = p;
            em.progress("build", 0.82 + 0.14 * last / 100.0, &format!("Building llama.cpp ({p:.0} %)"), None);
        }
    };
    let conf: Vec<String> = ["-S", &sp, "-B", &bp, "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF"].iter().map(|s| s.to_string()).collect();
    let code = runner.stream(&cmake, &conf, None, cancel, &mut |l| on_line(em, l))?;
    check_cancel(cancel)?;
    if code != Some(0) {
        return Err(fail("build_failed", "cmake could not prepare the build (the log above says why; usually the CUDA toolkit or the C++ compiler is too old or incomplete). Use the ready-made build instead."));
    }
    let jobs = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).to_string();
    let args: Vec<String> = ["--build", &bp, "--config", "Release", "--target", "llama-quantize", "llama-imatrix", "llama-perplexity", "-j", &jobs].iter().map(|s| s.to_string()).collect();
    let code = runner.stream(&cmake, &args, None, cancel, &mut |l| on_line(em, l))?;
    check_cancel(cancel)?;
    if code != Some(0) {
        return Err(fail("build_failed", "The build failed (the log above says why). Use the ready-made build instead."));
    }
    let out = build.join("bin");
    if !out.join(exe("llama-quantize")).is_file() {
        return Err(fail("build_failed", "The build finished but made no llama-quantize. Use the ready-made build instead."));
    }
    copy_dir(&out, &root.join("build").join("bin"))?;
    Ok(())
}

/// The install as the window runs it: the failure is remembered (so the window can offer the CPU build), a cancel is not.
pub fn install_task(cfg: &Config, opts: &Options, cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> crate::error::Result<Value> {
    let machine = detect_machine();
    run_install(cfg, &machine, opts, &RealRunner, cancel, on_event)
}

pub fn run_install(cfg: &Config, machine: &Machine, opts: &Options, runner: &dyn Runner, cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> crate::error::Result<Value> {
    match install(cfg, machine, opts, runner, cancel, on_event) {
        Ok(v) => Ok(v),
        Err(f) => {
            if f.code != "cancelled" {
                let offer_cpu = matches!(f.code, "cuda_start_failed" | "start_failed" | "build_failed") && opts.want != Want::Cpu;
                let rec = json!({"code": f.code, "message": f.message, "offer_cpu": offer_cpu, "at": chrono::Utc::now().to_rfc3339()});
                let _ = std::fs::write(cfg.failure_file(), rec.to_string());
            }
            Err(f.into())
        }
    }
}

/// Is there a newer release than the installed one? A hint only (never installs anything).
pub fn check_update(cfg: &Config) -> Res<Value> {
    let installed = read_info(&cfg.install_dir()).ok().flatten().map(|i| i.tag);
    let rel = fetch_release(cfg, None)?;
    let newer = installed.as_deref().is_some_and(|i| update_available(i, &rel.tag));
    Ok(json!({"installed": installed, "latest": rel.tag, "newer": newer}))
}

/// Only for the window's "is the tag valid" check.
pub fn tag_ok(t: &str) -> bool {
    valid_tag(t)
}
