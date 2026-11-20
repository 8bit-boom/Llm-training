//! The experimental vLLM engine, program side (docs/VLLM.md; worker side: `worker/ndforge_worker/vllmengine.py`).
//!
//! BUILT FROM A RESEARCH REPORT, NOT YET RUN ON THIS PC. Everything is OFF until the person switches it on in Settings, nothing is installed for them (Docker, WSL, the NVIDIA
//! driver are theirs to install: the guided text says how), and the "vLLM self-test" says what really works here.
//!
//! This file finds out which way the engine can run ("docker" = Windows docker.exe / Docker Desktop, "docker_wsl" = Docker inside WSL, "wslc" = the WSL 3.x container command,
//! "wsl" = vLLM installed in WSL, "url" = a running server, else "missing") by asking small programs through a `Runner` (so it is tested with fakes), judges the NVIDIA driver and the
//! WSL version (numbers, never strings: 2.7.3 and newer, 3.x included, are fine; an unreadable version is "unknown, continuing"), and builds the worker's `engine` object.
use std::sync::mpsc;
use std::time::Duration;

use regex::Regex;
use serde_json::{json, Value};

use crate::envmgr::health::{Check, Light};
use crate::settings::VllmSettings;

/// The driver the report says CUDA 13 images need (Windows GeForce driver).
pub const MIN_DRIVER: u32 = 580;
/// The WSL version the report names (the crash fix); newer is assumed fine.
pub const MIN_WSL: [u32; 3] = [2, 7, 3];
/// vLLM reserves about 88 % of the card: anything else using more than this blocks a start.
pub const DEFAULT_UTIL: f64 = 0.88;
/// Below this share of the card the engine is not started; the share is lowered to fit next to what other programs already use (same rule as the worker's `plan_gpu_util`).
pub const UTIL_FLOOR: f64 = 0.70;
pub const KINDS: [&str; 5] = ["docker", "docker_wsl", "wslc", "wsl", "url"];

/// Runs a program: `Some((succeeded, output))`, None when it could not be started or took too long.
pub type Runner<'a> = &'a dyn Fn(&[&str]) -> Option<(bool, String)>;

/// The real runner: no console window, 25 seconds at most; UTF-16 output of `wsl.exe` is read by dropping its NUL bytes.
pub fn real_runner(args: &[&str]) -> Option<(bool, String)> {
    let (prog, rest) = args.split_first()?;
    let (prog, rest): (String, Vec<String>) = (prog.to_string(), rest.iter().map(|s| s.to_string()).collect());
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut cmd = std::process::Command::new(prog);
        crate::envmgr::process::quiet(&mut cmd).args(rest).stdin(std::process::Stdio::null());
        let _ = tx.send(cmd.output().ok().map(|o| (o.status.success(), format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)).replace('\0', ""))));
    });
    rx.recv_timeout(Duration::from_secs(25)).ok().flatten()
}

/// How much of the card vLLM may take when other programs already use `used_mb` of `total_mb`: the engine reserves `util x total`; it needs `free >= that + 0.5 GB`, else the share is lowered to
/// `(free - 0.6 GB) / total`; below 0.70 it is refused. `ok` None = the memory could not be read (continuing).
#[derive(Debug, Clone, PartialEq)]
pub struct GpuPlan {
    pub util: f64,
    pub ok: Option<bool>,
    pub lowered: bool,
    pub text: String,
}

pub fn gpu_plan(total_mb: Option<u64>, used_mb: Option<u64>, configured: f64) -> GpuPlan {
    let (Some(total), Some(used)) = (total_mb.filter(|t| *t > 0), used_mb) else {
        return GpuPlan { util: configured, ok: None, lowered: false, text: "The memory in use on the card could not be read (nvidia-smi did not answer): continuing with the configured share.".into() };
    };
    let (t, u) = (total as f64, used as f64);
    let free = t - u;
    let gb = |x: f64| x / 1024.0;
    if free >= configured * t + 512.0 {
        return GpuPlan { util: configured, ok: Some(true), lowered: false, text: format!("{:.1} GB of the card are in use; {:.1} GB are free and the engine takes {:.1} GB ({configured} of the card): fine.", gb(u), gb(free), gb(configured * t)) };
    }
    let util = configured.min((free - 614.0) / t);
    if util < UTIL_FLOOR {
        return GpuPlan { util: configured, ok: Some(false), lowered: false, text: format!("{:.1} GB of the card ({:.1} GB) are in use by other programs (the desktop, a browser, Ollama, llama-server ...): only {:.1} GB are free, and even {UTIL_FLOOR} of the card ({:.1} GB) does not fit. Close those programs and try again.", gb(u), gb(t), gb(free), gb(UTIL_FLOOR * t)) };
    }
    let util = ((util - 0.0049) * 100.0).round() / 100.0;
    GpuPlan { util, ok: Some(true), lowered: true, text: format!("Other programs use {:.1} GB of the card; vLLM will take {util} instead of {configured} of it.", gb(u)) }
}

/// The first `x.y` or `x.y.z...` number in a text, as numbers (tolerant of localized labels: "WSL version: 3.0.1.0", "Version de WSL : 2.7.3").
pub fn parse_wsl_version(text: &str) -> Option<Vec<u32>> {
    let re = Regex::new(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.(\d+))?").ok()?;
    let clean = text.replace('\0', "");
    let m = re.captures(&clean)?;
    Some((1..=4).filter_map(|i| m.get(i).and_then(|g| g.as_str().parse().ok())).collect())
}

/// (ok: Some(true) fine, Some(false) older than 2.7.3, None unknown; a plain sentence)
pub fn wsl_note(text: &str) -> (Option<bool>, String) {
    let Some(v) = parse_wsl_version(text) else { return (None, "The WSL version could not be read (unknown, continuing).".into()) };
    let shown = v.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(".");
    let mut head = [0u32; 3];
    for (i, n) in v.iter().take(3).enumerate() {
        head[i] = *n;
    }
    if head >= MIN_WSL {
        (Some(true), format!("WSL {shown} (2.7.3 or newer is wanted; what 3.x changes is untested by us, the self-test tells what works)."))
    } else {
        (Some(false), format!("WSL {shown} is older than 2.7.3: run `wsl --update` in a command window (older versions can crash the whole WSL machine while the model loads)."))
    }
}

/// What was asked of the machine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    pub docker: bool,
    pub docker_wsl: bool,
    pub wslc: bool,
    pub wsl_vllm: bool,
    pub wsl_version_text: String,
    pub driver: String,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
}

fn ok(run: Runner, args: &[&str]) -> bool {
    run(args).is_some_and(|(s, _)| s)
}

fn wsl_prefix(distro: &str) -> Vec<String> {
    let mut v = vec!["wsl.exe".to_string()];
    if !distro.is_empty() {
        v.extend(["-d".to_string(), distro.to_string()]);
    }
    v.push("--".into());
    v
}

/// Asks: `docker version`, `wsl.exe -- docker version`, `wslc --version`, a vLLM install in WSL, `wsl --version`, `nvidia-smi`. (The GPU test of Docker is the self-test's job: it
/// starts a container, which a status line must not do.)
pub fn gather_facts(run: Runner, s: &VllmSettings) -> Facts {
    let pre = wsl_prefix(&s.distro);
    let with = |tail: &[&str]| -> Vec<String> { pre.iter().cloned().chain(tail.iter().map(|x| x.to_string())).collect() };
    let call = |v: Vec<String>| -> Option<(bool, String)> { run(&v.iter().map(String::as_str).collect::<Vec<_>>()) };
    let venv = format!("{}/bin/vllm", s.venv.trim_end_matches('/'));
    let smi = run(&["nvidia-smi", "--query-gpu=driver_version,memory.used,memory.total", "--format=csv,noheader,nounits"]);
    let (mut driver, mut used, mut total) = (String::new(), None, None);
    if let Some((true, text)) = smi {
        if let Some(line) = text.lines().next() {
            let p: Vec<&str> = line.split(',').map(str::trim).collect();
            driver = p.first().map(|x| x.to_string()).unwrap_or_default();
            used = p.get(1).and_then(|x| x.parse().ok());
            total = p.get(2).and_then(|x| x.parse().ok());
        }
    }
    Facts {
        docker: ok(run, &["docker", "version"]),
        docker_wsl: call(with(&["docker", "version"])).is_some_and(|r| r.0),
        wslc: ok(run, &["wslc", "--version"]),
        wsl_vllm: call(with(&["test", "-x", &venv])).is_some_and(|r| r.0),
        wsl_version_text: run(&["wsl.exe", "--version"]).map(|r| r.1).unwrap_or_default(),
        driver,
        vram_used_mb: used,
        vram_total_mb: total,
    }
}

/// The first way that answers, in the order docker, docker_wsl, wslc, wsl ("wslc" only when the Docker ways are missing); "missing" when none does.
pub fn detect_kind(f: &Facts) -> &'static str {
    if f.docker { "docker" } else if f.docker_wsl { "docker_wsl" } else if f.wslc { "wslc" } else if f.wsl_vllm { "wsl" } else { "missing" }
}

/// The way that will be used: the person's choice when it is not "auto".
pub fn used_kind(s: &VllmSettings, f: &Facts) -> String {
    match s.kind.as_str() {
        "auto" | "" => detect_kind(f).to_string(),
        k if KINDS.contains(&k) => k.to_string(),
        _ => "missing".into(),
    }
}

fn available(kind: &str, f: &Facts, s: &VllmSettings) -> bool {
    match kind {
        "docker" => f.docker,
        "docker_wsl" => f.docker_wsl,
        "wslc" => f.wslc,
        "wsl" => f.wsl_vllm,
        "url" => !s.url.trim().is_empty(),
        _ => false,
    }
}

pub fn driver_check(driver: &str) -> (Light, String) {
    match crate::envmgr::gpu::driver_version(driver) {
        None => (Light::Warn, "The NVIDIA driver version could not be read (unknown, continuing).".into()),
        Some((major, _)) if major >= MIN_DRIVER => (Light::Ok, format!("NVIDIA driver {driver} (the report wants {MIN_DRIVER} or newer for the CUDA 13 images).")),
        Some(_) => (Light::Warn, format!("NVIDIA driver {driver} is older than {MIN_DRIVER}: the CUDA 13 vLLM images may not start. Update the Windows driver (do NOT install a Linux driver inside WSL).")),
    }
}

/// What the person must do themselves when the engine is missing (plain, nothing is installed by the program).
pub fn guided_install_text() -> Vec<String> {
    vec![
        "Nothing is installed for you. To try the vLLM engine you need ONE of these:".into(),
        "1. Docker Desktop (with its WSL 2 engine and GPU support switched on), or Docker installed inside WSL: then `docker version` works.".into(),
        "2. WSL 3.x with its own container command: then `wslc --version` works (its command lines are templates you can edit: docs/VLLM.md).".into(),
        "3. A Python environment with vLLM inside WSL (`uv venv ~/.ndforge/vllm --python 3.12`, then `uv pip install vllm==0.34.0 --torch-backend=cu130`).".into(),
        "Also needed: the Windows NVIDIA driver 580 or newer, about 60 GB free disk for the image, the model and WSL, and nothing else using the graphics card (the engine takes about 88 % of its memory).".into(),
        "This is built from a research report and has not been run on this PC: run the vLLM self-test first.".into(),
    ]
}

/// The status the Settings card shows: `{"enabled", "kind_setting", "kind_detected", "kind_used", "ready", "lines": [{label, light, detail}], "guide": [...]}`.
pub fn status(s: &VllmSettings, run: Runner) -> Value {
    let f = gather_facts(run, s);
    let detected = detect_kind(&f);
    let used = used_kind(s, &f);
    let (dl, dt) = if f.driver.is_empty() { (Light::Warn, "nvidia-smi did not answer: no NVIDIA driver was found.".to_string()) } else { driver_check(&f.driver) };
    let (wok, wtext) = wsl_note(&f.wsl_version_text);
    let wl = if f.wsl_version_text.trim().is_empty() { None } else { Some((if wok == Some(false) { Light::Warn } else { Light::Ok }, wtext)) };
    let ready = available(&used, &f, s);
    let mut lines = vec![json!({"label": "vLLM engine", "light": if ready { Light::Ok } else { Light::Warn },
                                "detail": format!("{}{}", used, if s.kind != "auto" && used != detected { format!(" (chosen by you; found: {detected})") } else { String::new() })})];
    lines.push(json!({"label": "NVIDIA driver", "light": dl, "detail": dt}));
    if let Some((l, t)) = wl {
        lines.push(json!({"label": "WSL", "light": l, "detail": t}));
    }
    let mut guide = if used == "missing" || !ready { guided_install_text() } else { Vec::new() };
    if ready {
        guide.push("An engine was found, but it is untested here: run the vLLM self-test (it checks the graphics card in the container, starts the engine and measures).".into());
    }
    json!({"enabled": s.enabled, "kind_setting": s.kind, "kind_detected": detected, "kind_used": used, "ready": ready, "lines": lines, "guide": guide,
           "driver": f.driver, "vram_used_mb": f.vram_used_mb, "vram_total_mb": f.vram_total_mb})
}

/// The checks before a vLLM job: the engine is there, the graphics card has room (`gpu_plan`: the share is lowered to fit next to other programs, refused below 0.70), the driver and WSL are not too old.
pub fn preflight_checks(s: &VllmSettings, f: &Facts) -> Vec<Check> {
    let c = |id, label, light, detail: &str, fix: &str| Check { id, label, light, detail: detail.to_string(), fix: fix.to_string(), action: None };
    let used = used_kind(s, f);
    let mut out = Vec::new();
    out.push(if available(&used, f, s) {
        c("vllm_engine", "vLLM engine (experimental)", Light::Ok, &format!("Found: {used}. Untested on this PC: run the self-test first."), "")
    } else {
        c("vllm_engine", "vLLM engine (experimental)", Light::Fail, "No way to run the vLLM engine was found (Docker, Docker inside WSL, WSLc, or vLLM in WSL).",
          "Open Settings, vLLM engine: it explains what to install. Or check with the standard engine instead.")
    });
    let g = gpu_plan(f.vram_total_mb, f.vram_used_mb, DEFAULT_UTIL);
    out.push(match g.ok {
        Some(true) => c("vllm_gpu", "Graphics card free", Light::Ok, &g.text, ""),
        Some(false) => c("vllm_gpu", "Graphics card free", Light::Fail, &g.text, "Close the programs that use the card (a browser tab with video, Ollama, llama-server, a training run), then try again."),
        None => c("vllm_gpu", "Graphics card free", Light::Warn, &g.text, "Install or update the NVIDIA driver."),
    });
    if !f.driver.is_empty() {
        let (l, t) = driver_check(&f.driver);
        if l != Light::Ok {
            out.push(c("vllm_driver", "NVIDIA driver", l, &t, "Update the Windows NVIDIA driver."));
        }
    }
    if matches!(used.as_str(), "docker_wsl" | "wsl" | "wslc") {
        if let (Some(false), t) = wsl_note(&f.wsl_version_text) {
            out.push(c("vllm_wsl", "WSL version", Light::Warn, &t, "Run `wsl --update`."));
        }
    }
    out
}

/// Runs the real questions (a few small programs) and returns the preflight checks. Called when the thinking-check page asks about the "vLLM" way.
pub fn preflight_now(s: &VllmSettings) -> Vec<Check> {
    preflight_checks(s, &gather_facts(&real_runner, s))
}

/// The worker's `engine` object for a job (empty settings are left out; the worker has its own defaults).
pub fn engine_job(s: &VllmSettings) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("kind".into(), json!(if s.kind.is_empty() { "auto" } else { s.kind.as_str() }));
    for (k, v) in [("image", &s.image), ("distro", &s.distro), ("venv", &s.venv), ("url", &s.url), ("hf_volume", &s.model_volume), ("teacher_repo", &s.teacher_repo)] {
        if !v.trim().is_empty() {
            o.insert(k.into(), json!(v.trim()));
        }
    }
    if s.wslc.as_object().is_some_and(|m| !m.is_empty()) {
        o.insert("wslc".into(), s.wslc.clone());
    }
    Value::Object(o)
}

/// `engine_job` plus the card's memory (the pick of the weights depends on it); asks `nvidia-smi` (remembered for a minute).
pub fn engine_job_with_card(s: &VllmSettings) -> Value {
    let mut e = engine_job(s);
    e["teacher"] = json!({"route": s.teacher.route, "pairs": s.teacher.pairs, "confirmed": s.teacher.confirmed});          // the identity rule: which pre-quantized checkpoint may stand in for an original
    if let Some(g) = crate::envmgr::gpu::detect_gpus_cached().ok().and_then(|g| g.first().map(|x| x.vram_gb())) {
        e["vram_gb"] = json!((g * 10.0).round() / 10.0);
    }
    e
}

// ------------------------------------------------------------------------------------------------------------------ the self-test

/// Where the self-test writes `vllm_selftest.json` / `.md` (one report, replaced by the next run; it is part of the diagnostics bundle).
pub fn selftest_dir() -> std::path::PathBuf {
    crate::paths::config_dir().join("vllm_selftest")
}

/// The newest saved result `<file>` of the made models of a project (the original checked the standard way, a GGUF checked with llama.cpp): what the self-test compares with.
pub fn compare_files(p: Option<&crate::project::Project>) -> Value {
    let mut out = serde_json::Map::new();
    let Some(p) = p else { return Value::Object(out) };
    let newest = |file: &str| -> Option<std::path::PathBuf> {
        std::fs::read_dir(crate::reshape::runs::models_dir(p)).ok()?.flatten().filter_map(|e| {
            let f = e.path().join("think_check").join(file);
            Some((std::fs::metadata(&f).ok()?.modified().ok()?, f))
        }).max_by_key(|(t, _)| *t).map(|(_, f)| f)
    };
    if let Some(f) = newest("original.quick.json") {
        out.insert("hf".into(), json!(f));
    }
    if let Some(f) = newest("gguf.quick.json") {
        out.insert("llama".into(), json!(f));
    }
    Value::Object(out)
}

/// The worker job `vllm_selftest` (docs/VLLM.md). `spec`: `{"model": "" (the configured pre-quantized original), "confirm_download": bool, "adapter": folder (optional: the agreement check)}`.
pub fn selftest_job(s: &VllmSettings, spec: &Value, p: Option<&crate::project::Project>) -> crate::Result<Value> {
    let dir = selftest_dir();
    let model = spec["model"].as_str().map(str::trim).filter(|m| !m.is_empty()).map(String::from).unwrap_or_else(|| s.teacher_repo.clone());
    if model.is_empty() {
        return Err(crate::Error::new("Choose a model folder or a Hugging Face id for the self-test."));
    }
    let mut job = json!({"job": "vllm_selftest", "model": model, "confirm_download": spec["confirm_download"].as_bool().unwrap_or(false), "engine": engine_job(s), "out_dir": dir,
                         "stop_file": dir.join("STOP"), "quick": true, "gpu_test": true, "compare": compare_files(p)});
    if let Some(a) = spec["adapter"].as_str().map(str::trim).filter(|a| !a.is_empty()) {
        job["adapter"] = json!(a);
    }
    Ok(job)
}

/// Runs the self-test to the end (events as JSON `{"kind": ...}`); the worker stops the engine in every end, and Stop writes the STOP file.
pub fn selftest_run(job: &Value, stop: std::sync::Arc<std::sync::atomic::AtomicBool>, on_event: &mut dyn FnMut(Value)) -> crate::Result<Value> {
    let py = crate::envmgr::python_path(&crate::envmgr::env_dir());
    if !py.is_file() {
        return Err(crate::Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let _ = std::fs::remove_file(selftest_dir().join("STOP"));
    crate::train::runs::drive(&py, &selftest_dir(), job, stop, Duration::from_secs(120), &mut |e| on_event(crate::reshape::runs::event_json(e)))
}

/// The last report, `Value::Null` when there is none.
pub fn selftest_report() -> Value {
    std::fs::read(selftest_dir().join("vllm_selftest.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null)
}


// ------------------------------------------------------------------------------------------------------------- the teacher's answers (docs/VLLM.md, "Teacher cache")

/// The image tag's version (`vllm/vllm-openai:v0.34.0` -> `0.34.0`), as the worker's `image_version` reads it.
pub fn image_version(image: &str) -> String {
    let tag = image.rsplit('/').next().and_then(|last| last.split_once(':')).map(|x| x.1).unwrap_or("");
    match Regex::new(r"^v?(\d+\.\d+(?:\.\d+)?)").ok().and_then(|re| re.captures(tag).map(|c| c[1].to_string())) {
        Some(v) => v,
        None if tag.is_empty() => "unknown".into(),
        None => tag.to_string(),
    }
}

fn norm_id(x: &str) -> String {
    x.trim().trim_end_matches('/').to_lowercase()
}

/// Which weights the engine scores with for the original `origin` (a Hub id or a folder). THE TEACHER MUST BE THE PERSON'S ORIGINAL MODEL: a pre-quantized checkpoint is a quantization of
/// ONE repo, so it is used only for that original (`pairs`) or one the person confirmed (`confirmed`); every other original goes through the in-flight 4-bit route on its own folder. This is
/// the same rule as the worker's `vllmcache.select_teacher`, which decides for real; this one shows the person what will happen and is checked against the same cases.
/// `{"route": "prequant" | "bnb" | "refused", "quant": "awq:<repo>@unpinned" | "bnb-nf4" | "", "quantized_repo", "reason"}`.
pub fn teacher_route(t: &crate::settings::VllmTeacherSettings, origin: &str) -> Value {
    let key = norm_id(origin);
    let find = |m: &std::collections::BTreeMap<String, String>| m.iter().find(|(k, _)| norm_id(k) == key).map(|(_, v)| v.clone());
    let paired = find(&t.pairs).or_else(|| find(&t.confirmed));
    match (t.route.as_str(), paired) {
        ("prequant", None) => json!({"route": "refused", "quant": "", "quantized_repo": "",
                                     "reason": format!("There is no confirmed quantized copy of {origin}: the pre-quantized checkpoint of another model would teach the wrong model. Use the in-flight 4-bit route, or confirm a copy in the settings.")}),
        ("auto" | "prequant", Some(q)) => json!({"route": "prequant", "quant": format!("awq:{q}@unpinned"), "quantized_repo": q, "reason": format!("{q} is the confirmed quantized copy of {origin}.")}),
        _ => json!({"route": "bnb", "quant": "bnb-nf4", "quantized_repo": "",
                    "reason": format!("No confirmed quantized copy of {origin}: the original's own folder is loaded and quantized to 4 bits while loading (the in-flight bitsandbytes route; how vLLM does it is unverified and its speed is not known).")}),
    }
}

/// The worker's `engine` object of a `distill_cache` job that makes the original's answers with vLLM: `{"kind": "vllm", "teacher": {...}, "container": {...}, "required", "teacher_check"}`.
/// `check` is the saved `vllm_teacher_check.json` of this model (its verdict travels with the job; the worker refuses a teacher the check called "different").
pub fn teacher_engine_job(s: &VllmSettings, check: Option<&Value>) -> Value {
    let t = &s.teacher;
    let mut e = json!({"kind": "vllm", "teacher": {"route": t.route, "pairs": t.pairs, "confirmed": t.confirmed}, "container": engine_job_with_card(s), "required": t.required});
    if let Some(v) = check.and_then(|c| c["verdict"].as_str()) {
        e["teacher_check"] = json!({"verdict": v});
    }
    e
}

/// The identity a vLLM cache has (`engine`, `teacher_quant`), as far as the program can know it before the worker runs.
pub fn teacher_identity(s: &VllmSettings, origin: &str) -> Value {
    let r = teacher_route(&s.teacher, origin);
    json!({"engine": format!("vllm-{}", image_version(&s.image)), "teacher_quant": r["quant"], "route": r["route"]})
}

/// Where the fidelity check of a made model is saved.
pub fn teacher_check_file(model_dir: &std::path::Path) -> std::path::PathBuf {
    model_dir.join("vllm_teacher_check.json")
}

/// The saved check of a made model: the file's JSON, with `current` (does it belong to the teacher the settings would use now: the same engine version and teacher quantization).
pub fn read_teacher_check(model_dir: &std::path::Path, s: &VllmSettings, origin: &str) -> Value {
    let Some(mut c) = std::fs::read(teacher_check_file(model_dir)).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else { return Value::Null };
    let now = teacher_identity(s, origin);
    c["current"] = json!(c["identity"]["teacher_quant"] == now["teacher_quant"] && c["identity"]["engine"] == now["engine"]);
    c
}

/// The worker job `vllm_teacher_check` for a made model (docs/WORKER_CONTRACT.md): `common` holds what the heal's cache job has (teacher, student, model, data, k, max_seq, seed).
pub fn teacher_check_job(s: &VllmSettings, common: &Value, model_dir: &std::path::Path, reference_cache: Option<&std::path::Path>) -> Value {
    let mut job = json!({"job": "vllm_teacher_check", "rows": 100, "out": teacher_check_file(model_dir), "work": model_dir.join("vllm_teacher_check_work"), "stop_file": model_dir.join("STOP"),
                         "engine": teacher_engine_job(s, None)});
    for k in ["teacher", "student", "model", "data", "k", "max_seq", "seed"] {
        job[k] = common[k].clone();
    }
    if let Some(r) = reference_cache {
        job["reference_cache"] = json!(r);
    }
    job
}

/// The numbers the person's own files hold: the self-test's start-up, Quick-suite time and generation speed, the teacher check's scoring speed (docs/VLLM.md).
pub fn measured_numbers(selftest: &Value, checks: &[Value]) -> crate::vllm_plan::Numbers {
    let f = |v: &Value| v.as_f64().filter(|x| x.is_finite() && *x > 0.0);
    let ok = !matches!(selftest["verdict"].as_str(), Some("fail" | "stopped")) || f(&selftest["startup_seconds"]).is_some();
    // the size of the model the self-test ran (the report's `model_params_b`, else `model_info.params_b`): the numbers are for that model
    let model_b = f(&selftest["model_params_b"]).or_else(|| f(&selftest["model_info"]["params_b"]));
    crate::vllm_plan::Numbers {
        model_b,
        startup_s: ok.then(|| f(&selftest["startup_seconds"])).flatten(),
        // the Quick suite's wall time only when its answers were thinking-long; short answers (a model that did not think) say nothing about a real check
        quick_wall_s: ok.then(|| f(&selftest["wall_seconds"])).flatten().filter(|_| selftest["short_answers"] != true && !selftest["answer_tokens"].is_null()),
        long_tps: ok.then(|| f(&selftest["long_tokens_per_s"])).flatten(),
        gen_tps: ok.then(|| f(&selftest["tokens_per_s"])).flatten(),
        score_tps: checks.iter().find_map(|c| f(&c["vllm"]["tokens_per_s"])),
    }
}

/// `history` with the numbers of the person's own vLLM files added (`vllm_plan::Numbers::push_into`): the self-test report and the newest teacher check of the project's models.
pub fn history_with_measured(mut h: crate::timing::History, p: Option<&crate::project::Project>) -> crate::timing::History {
    let mut checks: Vec<Value> = Vec::new();
    if let Some(p) = p {
        if let Ok(rd) = std::fs::read_dir(crate::reshape::runs::models_dir(p)) {
            let mut found: Vec<(std::time::SystemTime, Value)> = rd.flatten().filter_map(|e| {
                let f = teacher_check_file(&e.path());
                let t = std::fs::metadata(&f).ok()?.modified().ok()?;
                Some((t, serde_json::from_slice::<Value>(&std::fs::read(&f).ok()?).ok()?))
            }).collect();
            found.sort_by_key(|x| std::cmp::Reverse(x.0));
            checks = found.into_iter().map(|x| x.1).collect();
        }
    }
    measured_numbers(&selftest_report(), &checks).push_into(&mut h);
    h
}

/// Stops and removes any container this program started and left behind (named `ndforge-vllm-*`): the engine must not be running when a training or another job takes the card. Docker kinds
/// only (the WSLc listing syntax is not known); returns the names it removed. Nothing else is touched.
pub fn stop_leftovers(s: &VllmSettings, run: Runner) -> Vec<String> {
    let pre: Vec<String> = match s.kind.as_str() {
        "docker_wsl" => wsl_prefix(&s.distro).into_iter().chain(["docker".to_string()]).collect(),
        "docker" | "auto" | "" => vec!["docker".to_string()],
        _ => return Vec::new(),
    };
    let call = |tail: &[&str]| -> Option<(bool, String)> {
        let v: Vec<&str> = pre.iter().map(String::as_str).chain(tail.iter().copied()).collect();
        run(&v)
    };
    let Some((true, names)) = call(&["ps", "-a", "--filter", "name=ndforge-vllm-", "--format", "{{.Names}}"]) else { return Vec::new() };
    let found: Vec<String> = names.lines().map(str::trim).filter(|n| n.starts_with("ndforge-vllm-") && !n.is_empty()).map(String::from).collect();
    for n in &found {
        let _ = call(&["rm", "-f", n]);
    }
    found
}


/// Where the self-test's report is and what it says, for the card: `{"dir": the exact folder as text, "json", "md", "markdown": the report text or null, "exists": bool}`. The folder is
/// `<config folder>/vllm_selftest` (on Windows `%APPDATA%\\NDLoreForge\\vllm_selftest`: Roaming, which is not obvious, so the card shows the path and can open it).
pub fn selftest_files() -> Value {
    let dir = selftest_dir();
    let md = dir.join("vllm_selftest.md");
    let text = std::fs::read_to_string(&md).ok();
    json!({"dir": dir.display().to_string(), "json": dir.join("vllm_selftest.json").display().to_string(), "md": md.display().to_string(), "markdown": text, "exists": text.is_some()})
}

/// The same for the saved check of the vLLM teacher of a made model: `{"dir", "file", "text": the file as pretty JSON or null, "exists"}`.
pub fn teacher_check_files(model_dir: &std::path::Path) -> Value {
    let f = teacher_check_file(model_dir);
    let text = std::fs::read(&f).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).and_then(|v| serde_json::to_string_pretty(&v).ok());
    json!({"dir": model_dir.display().to_string(), "file": f.display().to_string(), "text": text, "exists": text.is_some()})
}

/// What the person is told about WHICH model a vLLM check or trace generation of the original really loads: its own folder in 4 bits ("original: <id> (4-bit on the fly)"), a confirmed
/// pre-quantized copy ("confirmed copy <repo>"), or nothing (refused: the standard engine is used). Same rule as the worker's `vllmquant.select_original`.
pub fn served_text(t: &crate::settings::VllmTeacherSettings, origin: &str) -> String {
    let r = teacher_route(t, origin);
    match r["route"].as_str() {
        Some("prequant") => format!("confirmed copy {}", r["quantized_repo"].as_str().unwrap_or("")),
        Some("bnb") => format!("original: {origin} (4-bit on the fly; the in-flight route is unverified, the standard engine answers if it does not work)"),
        _ => format!("nothing: {} The standard engine is used.", r["reason"].as_str().unwrap_or("")),
    }
}
