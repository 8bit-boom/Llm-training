//! The "Upgrade transformers" action of the Slim page (docs/TRANSFORMERS_UPGRADE.md, docs/MOE_GUIDE.md section 7): Inkling-2-Mini needs transformers 5.24.0 or newer.
//! It installs into THE PROGRAM'S OWN environment only (`env_dir()`, never a system Python), with the installer the environment manager uses (`uv pip`), keeps the installed PyTorch
//! and bitsandbytes (a constraint file made from `uv pip freeze`, a resolver preview that must not list them, and a comparison afterwards), records the version it replaces
//! first, checks the import and the worker afterwards, and puts the old version back by itself when anything fails. "Go back" uses the same record.
//! Everything that changes a file or the environment is behind `confirm`; the caller refuses while a job, a queue run or an Auto run is active (`refusal`).
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::process::{quiet, run_lines, Stream};
use super::{env_dir, find_uv, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::reshape::moe_compat::{version_at_least, MIN_TRANSFORMERS};
use crate::update::install::Activity;

pub const PACKAGE: &str = "transformers";
/// The packages that must come out of an upgrade exactly as they went in (the resolver has no reason to touch them; a constraint, a preview check and a comparison make sure).
pub const GUARDED: [&str; 4] = ["torch", "torchvision", "torchaudio", "bitsandbytes"];
const RECORD_FILE: &str = "transformers_upgrade.json";
const PINS_FILE: &str = "transformers_upgrade_pins.txt";

/// `transformers>=5.24.0,<6`
pub fn target_spec() -> String {
    format!("{PACKAGE}>={MIN_TRANSFORMERS},<6")
}

/// What the program needs to know about where it works. `production()` reads the real places; the tests give their own.
#[derive(Debug, Clone)]
pub struct Upgrade {
    pub env_dir: PathBuf,
    pub py: PathBuf,
    pub uv: Option<PathBuf>,
    pub worker_dir: PathBuf,
    /// the small JSON in the program's data folder
    pub record: PathBuf,
}

impl Upgrade {
    pub fn production() -> Upgrade {
        let env = env_dir();
        Upgrade { py: python_path(&env), env_dir: env, uv: find_uv(), worker_dir: worker_dir(), record: crate::paths::config_dir().join(RECORD_FILE) }
    }
}

/// The record of the last upgrade: the version it replaced.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub package: String,
    /// the version before the upgrade (what "Go back" installs)
    pub previous: String,
    pub target: String,
    pub python: String,
    pub at: String,
    /// started | installed | rolled_back_automatically | rollback_failed | went_back
    pub state: String,
    #[serde(default)]
    pub installed: Option<String>,
}

pub fn read_record(path: &Path) -> Option<Record> {
    serde_json::from_slice::<Record>(&std::fs::read(path).ok()?).ok().filter(|r| r.package == PACKAGE && !r.previous.is_empty())
}

fn write_record(path: &Path, r: &Record) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    crate::fsx::write_json_atomic(path, r)
}

/// Why now is not the time to change the program's own software, in plain words; None = idle. Unlike the updater's, it does not mention the program closing.
pub fn refusal(a: &Activity) -> Option<String> {
    let tail = "Wait until it has finished or stop it, then try again: the training software must not change under a running job.";
    if let Some(label) = &a.queue_running {
        return Some(format!("A queue run is active ({label}). {tail}"));
    }
    if a.job_running {
        return Some(format!("A job is running (training, a heal, an export, a measurement, a quant check or an Auto run). {tail}"));
    }
    if a.queue_waiting > 0 {
        return Some(format!("The queue has {} waiting job(s) that would start by themselves. Pause or clear the queue, or let it finish, then try again.", a.queue_waiting));
    }
    None
}

/// The checks every start makes before anything is touched.
pub fn check_start(confirm: bool, a: &Activity) -> Result<()> {
    if !confirm {
        return Err(Error::new("The change was not confirmed, so nothing was installed."));
    }
    match refusal(a) {
        Some(why) => Err(Error::new(why)),
        None => Ok(()),
    }
}

// ------------------------------------------------------------------------------------------------------------------------- facts from the environment

/// Prints one JSON line: the installed version from the package metadata and, when asked, whether `import transformers` works and which version it reports.
const PROBE: &str = "import json, sys\nr = {}\ntry:\n from importlib import metadata\n r['version'] = metadata.version('transformers')\nexcept Exception as e:\n r['error'] = type(e).__name__ + ': ' + str(e)[:200]\nif len(sys.argv) > 1 and sys.argv[1] == 'import':\n try:\n  import transformers\n  r['import_version'] = transformers.__version__\n except Exception as e:\n  r['import_error'] = type(e).__name__ + ': ' + str(e)[:200]\nprint(json.dumps(r))\n";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Probe {
    pub version: Option<String>,
    /// None = the import was not tried
    pub import_ok: Option<bool>,
    pub error: Option<String>,
}

pub fn probe(py: &Path, with_import: bool) -> Probe {
    let mut cmd = Command::new(py);
    cmd.args(["-c", PROBE]);
    if with_import {
        cmd.arg("import");
    }
    quiet(&mut cmd);
    let out = match cmd.output() {
        Ok(o) => o,
        Err(e) => return Probe { error: Some(format!("Python could not be started: {e}")), ..Probe::default() },
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let Some(v) = text.lines().rev().find(|l| l.trim_start().starts_with('{')).and_then(|l| serde_json::from_str::<Value>(l).ok()) else {
        let err = String::from_utf8_lossy(&out.stderr);
        return Probe { error: Some(format!("The installed version could not be read ({}).", err.lines().last().unwrap_or("no answer").trim())), ..Probe::default() };
    };
    let mut p = Probe { version: v["version"].as_str().map(String::from), error: v["error"].as_str().map(String::from), ..Probe::default() };
    if with_import {
        match (v["import_version"].as_str(), v["import_error"].as_str()) {
            (Some(iv), _) => {
                p.import_ok = Some(true);
                p.version = Some(iv.to_string());
            }
            (None, e) => {
                p.import_ok = Some(false);
                p.error = Some(e.unwrap_or("import transformers failed").to_string());
            }
        }
    }
    p
}

/// The guarded packages as `uv pip freeze` lists them (`torch==2.8.0+cu128`); Err when the list cannot be read.
pub fn freeze_guarded(u: &Upgrade) -> std::result::Result<Vec<String>, String> {
    let uv = u.uv.as_ref().ok_or("uv was not found.")?;
    let mut cmd = Command::new(uv);
    cmd.args(["pip", "freeze", "--python"]).arg(&u.py);
    quiet(&mut cmd);
    let out = cmd.output().map_err(|e| format!("uv could not be started: {e}"))?;
    if !out.status.success() {
        return Err("The list of installed packages could not be read (uv pip freeze failed).".into());
    }
    Ok(String::from_utf8_lossy(&out.stdout).lines().map(str::trim).filter(|l| GUARDED.iter().any(|n| norm(l.split("==").next().unwrap_or("")) == *n) && l.contains("==")).map(String::from).collect())
}

fn norm(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('_', "-")
}

// ------------------------------------------------------------------------------------------------------------------------- the resolver's list

/// One package the installer changes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Change {
    pub name: String,
    pub from: Option<String>,
    pub to: Option<String>,
}

/// The ` - name==old` / ` + name==new` lines of `uv pip install` (and of its `--dry-run`), merged per package, in the order they first appear.
pub fn parse_changes<'a>(lines: impl IntoIterator<Item = &'a str>) -> Vec<Change> {
    let mut out: Vec<Change> = Vec::new();
    for l in lines {
        let l = l.trim();
        let Some(sign) = l.chars().next().filter(|c| matches!(c, '+' | '-' | '~')) else { continue };
        let rest = l[1..].trim();
        if !l[1..].starts_with(' ') {
            continue;
        }
        let Some((name, ver)) = rest.split_once("==") else { continue };
        let (name, ver) = (norm(name), ver.split_whitespace().next().unwrap_or("").to_string());
        if name.is_empty() || ver.is_empty() || name.contains(' ') {
            continue;
        }
        let i = out.iter().position(|c| c.name == name).unwrap_or_else(|| {
            out.push(Change { name: name.clone(), from: None, to: None });
            out.len() - 1
        });
        match sign {
            '-' => out[i].from = Some(ver),
            '+' => out[i].to = Some(ver),
            _ => {
                out[i].from = Some(ver.clone());
                out[i].to = Some(ver);
            }
        }
    }
    out
}

/// The first guarded package the resolver would change (a reinstall of the same build is not a change).
pub fn guarded_change(changes: &[Change]) -> Option<&Change> {
    changes.iter().find(|c| GUARDED.contains(&c.name.as_str()) && c.from != c.to)
}

fn changes_json(changes: &[Change]) -> Value {
    json!(changes)
}

// ------------------------------------------------------------------------------------------------------------------------- the plan the person confirms

/// What the button would do, for the confirmation: every fact the person should see before saying yes. Reads the environment (a second); changes nothing.
pub fn plan_view(u: &Upgrade) -> Value {
    let installed = u.py.is_file();
    let current = if installed { probe(&u.py, false).version } else { None };
    let record = read_record(&u.record);
    let needed = current.as_deref().map(|v| !version_at_least(v, MIN_TRANSFORMERS));
    let can_go_back = record.as_ref().is_some_and(|r| current.as_deref().is_some_and(|c| c != r.previous) && installed && u.uv.is_some());
    let problem = if !installed {
        Some("The training software is not installed yet. Run Training setup first.".to_string())
    } else if u.uv.is_none() {
        Some("uv, the installer the program uses, was not found.".to_string())
    } else if current.is_none() {
        Some("transformers is not installed in the program's environment, or its version could not be read. Run Training setup (it installs it).".to_string())
    } else {
        None
    };
    let spec = target_spec();
    let will_do = vec![
        format!("Install {spec} into the program's own Python environment: {}", u.py.display()),
        format!("It is {} now.", current.as_deref().map(|v| format!("transformers {v}")).unwrap_or_else(|| "not readable".into())),
        "This changes the program's training environment (the one every job uses), not your system Python.".to_string(),
        "Qwen and Gemma keep working: the newer transformers only adds the Inkling model type.".to_string(),
        "PyTorch and bitsandbytes are kept as they are; if the installer would have to change them, nothing is installed and you are told why.".to_string(),
        "Packages that transformers itself needs newer are changed too: the list is shown first, in the log, and again in the result.".to_string(),
        "If the install or the check afterwards fails, the version it replaced is put back by itself.".to_string(),
        format!("The old version is written to {} first.", u.record.display()),
    ];
    json!({
        "package": PACKAGE, "target": spec, "min": MIN_TRANSFORMERS, "python": u.py, "env_dir": u.env_dir, "installer": "uv pip install",
        "current": current, "needed": needed, "ready": problem.is_none(), "problem": problem, "record_path": u.record,
        "previous": record.as_ref().map(|r| r.previous.clone()), "can_go_back": can_go_back, "record": record, "will_do": will_do,
        "go_back_to": record.as_ref().map(|r| format!("{PACKAGE}=={}", r.previous)),
    })
}

// ------------------------------------------------------------------------------------------------------------------------- the install

struct Ctx<'a> {
    u: &'a Upgrade,
    total: usize,
    on_event: &'a mut dyn FnMut(Value),
}

impl Ctx<'_> {
    fn log(&mut self, text: impl Into<String>) {
        (self.on_event)(json!({"kind": "log", "task": "transformers", "text": text.into()}));
    }
    fn step(&mut self, index: usize, text: &str) {
        (self.on_event)(json!({"kind": "step", "task": "transformers", "index": index, "total": self.total, "text": text}));
    }
}

fn uv_of(u: &Upgrade) -> Result<&PathBuf> {
    u.uv.as_ref().ok_or_else(|| Error::new("uv, the installer the program uses, was not found."))
}

/// The arguments of `uv pip install` for one requirement: this environment's Python, the guarded pins, the setup's own constraints (the PyTorch builds, the tested upper bounds), and the
/// PyTorch index like the setup (so that a `+cu128` pin can be matched). Nothing is upgraded beyond what the requirement needs.
fn install_args(u: &Upgrade, requirement: &str, dry_run: bool) -> Vec<String> {
    let mut a: Vec<String> = ["pip", "install", "--python"].iter().map(|s| s.to_string()).collect();
    a.push(u.py.to_string_lossy().to_string());
    if dry_run {
        a.push("--dry-run".into());
    }
    a.push(requirement.to_string());
    let pins = u.env_dir.join(PINS_FILE);
    if pins.is_file() {
        a.extend(["--constraint".to_string(), pins.to_string_lossy().to_string()]);
    }
    let setup_pins = u.env_dir.join("constraints.txt");
    if setup_pins.is_file() {
        a.extend(["--constraint".to_string(), setup_pins.to_string_lossy().to_string()]);
    }
    let tested = u.worker_dir.join("constraints.txt");
    if tested.is_file() {
        a.extend(["--constraint".to_string(), tested.to_string_lossy().to_string()]);
    }
    if let Some(wheel) = std::fs::read(u.env_dir.join("ndforge_env.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| v["wheel"].as_str().map(String::from)).filter(|_| setup_pins.is_file()) {
        a.extend(["--extra-index-url".to_string(), format!("https://download.pytorch.org/whl/{wheel}"), "--index-strategy".into(), "unsafe-best-match".into()]);
    }
    a
}

/// Runs uv, logs every line, and returns the exit status and the lines.
fn run_uv(c: &mut Ctx, args: &[String], cancel: &AtomicBool) -> Result<(std::process::ExitStatus, Vec<String>)> {
    let mut cmd = Command::new(uv_of(c.u)?);
    cmd.args(args);
    let mut lines = Vec::new();
    let st = run_lines(cmd, cancel, &mut |_: Stream, line: &str| {
        lines.push(line.to_string());
        (c.on_event)(json!({"kind": "log", "task": "transformers", "text": line}));
    })?;
    Ok((st, lines))
}

fn describe(changes: &[Change]) -> String {
    if changes.is_empty() {
        return "no package".into();
    }
    changes.iter().map(|c| match (&c.from, &c.to) {
        (Some(f), Some(t)) if f != t => format!("{} {f} -> {t}", c.name),
        (None, Some(t)) => format!("{} {t} (new)", c.name),
        (Some(f), None) => format!("{} {f} (removed)", c.name),
        (Some(f), Some(_)) => format!("{} {f} (reinstalled)", c.name),
        _ => c.name.clone(),
    }).collect::<Vec<_>>().join(", ")
}

/// The worker's `hello` (does the worker still start and import?), and the transformers version it reports.
fn hello(u: &Upgrade) -> std::result::Result<Option<String>, String> {
    super::jobs::run_job(&u.py, &u.worker_dir, &json!({"job": "hello"}), &AtomicBool::new(false), &mut |_| {})
        .map(|r| r["packages"]["transformers"].as_str().map(String::from)).map_err(|e| e.0)
}

/// The checks after an install: the import works and reports `want` (exactly, or at least the minimum when `want` is None), the guarded packages are unchanged, the worker starts.
fn post_check(c: &mut Ctx, want: Option<&str>, pins_before: &[String]) -> std::result::Result<String, String> {
    let p = probe(&c.u.py, true);
    if p.import_ok != Some(true) {
        return Err(format!("`import transformers` fails after the install ({}).", p.error.unwrap_or_default()));
    }
    let v = p.version.unwrap_or_default();
    match want {
        Some(w) if v != w => return Err(format!("transformers {v} is installed, not {w}.")),
        None if !version_at_least(&v, MIN_TRANSFORMERS) => return Err(format!("transformers {v} is installed, which is still older than {MIN_TRANSFORMERS}.")),
        _ => {}
    }
    c.log(format!("transformers {v} imports."));
    match freeze_guarded(c.u) {
        Ok(after) if after != pins_before => return Err(format!("PyTorch or bitsandbytes changed during the install (before: {}; after: {}).", pins_before.join(", "), after.join(", "))),
        Ok(_) => c.log(format!("PyTorch and bitsandbytes are as they were ({}).", if pins_before.is_empty() { "none installed".to_string() } else { pins_before.join(", ") })),
        Err(e) => return Err(e),
    }
    match hello(c.u) {
        Ok(wv) => {
            if let Some(wv) = wv.filter(|wv| *wv != v) {
                return Err(format!("The worker sees transformers {wv}, but Python imports {v}."));
            }
            c.log("The training worker starts and sees the new version.");
        }
        Err(e) => return Err(format!("The training worker no longer starts: {e}")),
    }
    Ok(v)
}

/// Puts `previous` back (uv pip install transformers==previous with the same pins); nothing here can be cancelled, because a half-way environment is worse than waiting.
fn rollback_to(c: &mut Ctx, previous: &str, pins: &[String]) -> (bool, Vec<Change>, String) {
    c.log(format!("Putting transformers {previous} back…"));
    let args = install_args(c.u, &format!("{PACKAGE}=={previous}"), false);
    let (st, lines) = match run_uv(c, &args, &AtomicBool::new(false)) {
        Ok(x) => x,
        Err(e) => return (false, Vec::new(), e.0),
    };
    let changes = parse_changes(lines.iter().map(String::as_str));
    if !st.success() {
        return (false, changes, format!("the installer failed (exit code {})", st.code().unwrap_or(-1)));
    }
    match post_check(c, Some(previous), pins) {
        Ok(_) => (true, changes, String::new()),
        Err(e) => (false, changes, e),
    }
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn base_result(action: &str, previous: Option<&str>, u: &Upgrade) -> Value {
    json!({"task": "transformers", "action": action, "ok": false, "stopped": false, "previous": previous, "python": u.py, "target": target_spec(), "changed": [], "planned": [], "rolled_back": false, "rollback_ok": null, "record": u.record, "text": ""})
}

/// Resolves what the installer would change for `requirement` (its `--dry-run`), without changing anything. Ok(changes) or Err(plain message); a guarded package in the list is an Err too.
fn preview(c: &mut Ctx, requirement: &str, cancel: &AtomicBool) -> Result<std::result::Result<Vec<Change>, String>> {
    let args = install_args(c.u, requirement, true);
    let (st, lines) = run_uv(c, &args, cancel)?;
    let changes = parse_changes(lines.iter().map(String::as_str));
    if !st.success() {
        let why = lines.iter().rev().find(|l| !l.trim().is_empty()).map(|l| l.trim().to_string()).unwrap_or_default();
        return Ok(Err(format!("The installer could not work out an install of {requirement} that keeps the current PyTorch and bitsandbytes (exit code {}: {why}). Nothing was changed.", st.code().unwrap_or(-1))));
    }
    if let Some(g) = guarded_change(&changes) {
        return Ok(Err(format!("{requirement} would need {} changed ({} -> {}), and the program does not change PyTorch or bitsandbytes here. Nothing was installed and nothing was changed. Use \"Install the training software\" on the Training setup page if you want the whole environment renewed.",
                              g.name, g.from.as_deref().unwrap_or("not installed"), g.to.as_deref().unwrap_or("removed"))));
    }
    Ok(Ok(changes))
}

fn write_pins(u: &Upgrade) -> std::result::Result<Vec<String>, String> {
    let pins = freeze_guarded(u)?;
    let file = u.env_dir.join(PINS_FILE);
    if pins.is_empty() {
        let _ = std::fs::remove_file(&file);
    } else {
        std::fs::write(&file, pins.join("\n") + "\n").map_err(|e| format!("The file that keeps PyTorch as it is could not be written: {e}"))?;
    }
    Ok(pins)
}

fn require_env(u: &Upgrade) -> Result<()> {
    if !u.py.is_file() {
        return Err(Error::new("The training software is not installed yet. Press \"Install the training software\" first."));
    }
    uv_of(u).map(|_| ())
}

fn stopped(mut r: Value, text: &str) -> Value {
    r["stopped"] = json!(true);
    r["text"] = json!(text);
    r
}

/// Upgrades transformers (steps: look, preview, install, check). `confirm` must be true; `activity` is asked again right before anything is changed (a queue run could have started).
/// Errors (Err) mean nothing was touched. After that every end is an Ok with `ok`, `text`, `changed`, `rolled_back` (and `stopped` for a cancel), so the window can show it.
pub fn run_upgrade(u: &Upgrade, confirm: bool, activity: &dyn Fn() -> Activity, cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    check_start(confirm, &activity())?;
    require_env(u)?;
    let mut c = Ctx { u, total: 4, on_event };
    c.step(1, "Looking at the training environment");
    let current = probe(&u.py, false).version.ok_or_else(|| Error::new("transformers is not installed in the program's environment, or its version could not be read. Run Training setup."))?;
    if version_at_least(&current, MIN_TRANSFORMERS) {
        return Err(Error::new(format!("transformers {current} is already {MIN_TRANSFORMERS} or newer. Nothing was changed.")));
    }
    let spec = target_spec();
    let mut r = base_result("upgrade", Some(&current), u);
    let pins = write_pins(u).map_err(Error::new)?;
    c.log(format!("Installed now: transformers {current}; kept as they are: {}.", if pins.is_empty() { "no PyTorch found".to_string() } else { pins.join(", ") }));

    c.step(2, "Asking the installer what it would change (nothing is installed yet)");
    let planned = match preview(&mut c, &spec, cancel) {
        Ok(Ok(p)) => p,
        Ok(Err(text)) => {
            r["text"] = json!(text);
            r["aborted"] = json!(true);
            return Ok(r);
        }
        Err(e) if e.is_cancelled() => return Ok(stopped(r, "Cancelled before anything was installed. Nothing was changed.")),
        Err(e) => return Err(e),
    };
    r["planned"] = changes_json(&planned);
    c.log(format!("The installer would change: {}.", describe(&planned)));
    if let Some(why) = refusal(&activity()) {
        return Err(Error::new(why));
    }

    // the old version is written down before the first change
    let mut rec = Record { package: PACKAGE.into(), previous: current.clone(), target: spec.clone(), python: u.py.to_string_lossy().to_string(), at: now(), state: "started".into(), installed: None };
    write_record(&u.record, &rec)?;
    c.step(3, &format!("Installing {spec} - this can take a few minutes"));
    let installed = run_uv(&mut c, &install_args(u, &spec, false), cancel);
    let (failure, changes, was_cancelled) = match installed {
        Ok((st, lines)) => {
            let ch = parse_changes(lines.iter().map(String::as_str));
            if st.success() { (None, ch, false) } else { (Some(format!("The installer failed (exit code {}); the log shows why.", st.code().unwrap_or(-1))), ch, false) }
        }
        Err(e) if e.is_cancelled() => (Some("Cancelled.".to_string()), Vec::new(), true),
        Err(e) => (Some(e.0), Vec::new(), false),
    };
    let changes = if changes.is_empty() { planned.clone() } else { changes };
    c.step(4, "Checking the new software");
    let failure = match failure {
        Some(f) => Some(f),
        None => post_check(&mut c, None, &pins).err(),
    };
    r["changed"] = changes_json(&changes);
    let Some(why) = failure else {
        let v = probe(&u.py, false).version.unwrap_or_default();
        rec.state = "installed".into();
        rec.installed = Some(v.clone());
        write_record(&u.record, &rec)?;
        r["ok"] = json!(true);
        r["installed"] = json!(v);
        r["text"] = json!(format!("transformers {v} is installed (it was {current}). Changed: {}. Qwen and Gemma use the same environment and keep working. \"Go back to {current}\" is available.", describe(&changes)));
        return Ok(r);
    };

    // anything went wrong after the install started: the recorded version comes back
    let prev = read_record(&u.record).map(|x| x.previous).unwrap_or(current.clone());
    let (ok, back_changes, back_err) = rollback_to(&mut c, &prev, &pins);
    rec.state = if ok { "rolled_back_automatically" } else { "rollback_failed" }.into();
    let _ = write_record(&u.record, &rec);
    r["rolled_back"] = json!(true);
    r["rollback_ok"] = json!(ok);
    r["rollback_changed"] = changes_json(&back_changes);
    r["stopped"] = json!(was_cancelled);
    let lead = if was_cancelled { "The upgrade was cancelled while it was installing.".to_string() } else { format!("The upgrade did not work: {why}") };
    r["text"] = json!(if ok {
        format!("{lead} The previous version, transformers {prev}, was put back automatically and checked: the environment is as it was before.")
    } else {
        format!("{lead} Putting transformers {prev} back automatically failed too ({back_err}). To go back by hand, run: uv pip install --python \"{}\" \"{PACKAGE}=={prev}\". Or press \"Install the training software\" on the Training setup page.", u.py.display())
    });
    Ok(r)
}

/// "Go back to <previous version>": installs the version the record holds. `version` is the one the person saw on the button; a record that changed since then is refused.
pub fn run_rollback(u: &Upgrade, confirm: bool, version: &str, activity: &dyn Fn() -> Activity, cancel: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    check_start(confirm, &activity())?;
    require_env(u)?;
    let Some(mut rec) = read_record(&u.record) else {
        return Err(Error::new("There is no earlier version on record: nothing to go back to."));
    };
    if rec.previous != version {
        return Err(Error::new(format!("The recorded earlier version is {}, not {version}. Look at the page again.", rec.previous)));
    }
    let mut c = Ctx { u, total: 3, on_event };
    c.step(1, "Looking at the training environment");
    let current = probe(&u.py, false).version.ok_or_else(|| Error::new("transformers is not installed in the program's environment, or its version could not be read. Run Training setup."))?;
    if current == rec.previous {
        return Err(Error::new(format!("transformers {current} is already installed. Nothing was changed.")));
    }
    let requirement = format!("{PACKAGE}=={}", rec.previous);
    let mut r = base_result("rollback", Some(&rec.previous), u);
    r["from"] = json!(current);
    let pins = write_pins(u).map_err(Error::new)?;
    c.step(2, &format!("Putting transformers {} back", rec.previous));
    match preview(&mut c, &requirement, cancel) {
        Ok(Ok(p)) => r["planned"] = changes_json(&p),
        Ok(Err(text)) => {
            r["text"] = json!(text);
            r["aborted"] = json!(true);
            return Ok(r);
        }
        Err(e) if e.is_cancelled() => return Ok(stopped(r, "Cancelled before anything was installed. Nothing was changed.")),
        Err(e) => return Err(e),
    }
    if let Some(why) = refusal(&activity()) {
        return Err(Error::new(why));
    }
    let (st, lines) = match run_uv(&mut c, &install_args(u, &requirement, false), cancel) {
        Ok(x) => x,
        Err(e) if e.is_cancelled() => return Ok(stopped(r, "Cancelled. The installer was stopped; look at the Training setup page and press \"Install the training software\" if something is missing.")),
        Err(e) => return Err(e),
    };
    let changes = parse_changes(lines.iter().map(String::as_str));
    r["changed"] = changes_json(&changes);
    c.step(3, "Checking the software");
    let outcome = if st.success() { post_check(&mut c, Some(&rec.previous), &pins) } else { Err(format!("The installer failed (exit code {}); the log shows why.", st.code().unwrap_or(-1))) };
    match outcome {
        Ok(v) => {
            rec.state = "went_back".into();
            rec.installed = Some(v.clone());
            let _ = write_record(&u.record, &rec);
            r["ok"] = json!(true);
            r["installed"] = json!(v);
            r["text"] = json!(format!("transformers {v} is back (it was {current}). Changed: {}. Inkling-2-Mini needs {MIN_TRANSFORMERS} or newer, so it will say so again until you upgrade.", describe(&changes)));
        }
        Err(e) => {
            r["text"] = json!(format!("Going back to transformers {} did not work: {e} Nothing else was changed. Press \"Install the training software\" on the Training setup page to repair the environment.", rec.previous));
        }
    }
    Ok(r)
}
