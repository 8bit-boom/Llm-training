//! The memory panel: where the graphics memory and the RAM went, and a guarded way to give back what THIS program owns (docs/MEMORY.md).
//!
//! * `status` reads the numbers (nvidia-smi, the operating system) and is cached for about a second.
//! * `plan` lists exactly what a "Free GPU" / "Free RAM" / "Release WSL memory" would end or release, with opaque ids; the plan has a token (a hash of the ids).
//! * `free` plans again, refuses a stale token, and does only what the plan lists. A worker or an engine container that a running job uses is refused (the window stops the job first, by the
//!   normal Stop, when the person asks for it). Other programs are only LISTED, never touched.
//! Every command line is built and parsed by pure functions below (tested with real sample outputs); the machine is reached through the `Env` trait (tests use a fake).
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};

use crate::error::{Error, Result};

/// The mark of this program's worker processes in their command line.
pub const WORKER_MARK: &str = "ndforge_worker";
/// The name every engine container of this program starts with.
pub const CONTAINER_PREFIX: &str = "ndforge-vllm-";

// ------------------------------------------------------------------------------------------------------------------ what the machine says

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Proc {
    pub pid: u32,
    pub ppid: u32,
    pub cmd: String,
    /// resident / working-set megabytes when known
    pub rss_mb: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuProc {
    pub pid: u32,
    pub name: String,
    /// None where the driver does not say (Windows WDDM often reports `[N/A]`)
    pub used_mb: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Container {
    pub name: String,
    pub running: bool,
}

/// `nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits` -> (total MB, used MB); the first GPU.
pub fn parse_gpu_memory(text: &str) -> Option<(u64, u64)> {
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut it = line.split(',').map(|x| x.trim().parse::<f64>().ok());
    let total = it.next()??;
    let used = it.next()??;
    Some((total.round() as u64, used.round() as u64))
}

/// `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`; `[N/A]` or `[Not Supported]` memory is None; no apps (or a "No running processes" text) is empty.
pub fn parse_gpu_apps(text: &str) -> Vec<GpuProc> {
    text.lines().filter_map(|l| {
        let l = l.trim();
        let mut it = l.split(',').map(str::trim);
        let pid = it.next()?.parse::<u32>().ok()?;
        let name = it.next().unwrap_or("").to_string();
        let used = it.next().and_then(|m| m.parse::<f64>().ok()).map(|m| m.round() as u64);
        Some(GpuProc { pid, name, used_mb: used })
    }).collect()
}

/// `ps -eo pid=,ppid=,rss=,args=` (Linux, macOS): the memory is kilobytes.
pub fn parse_ps(text: &str) -> Vec<Proc> {
    text.lines().filter_map(|l| {
        let rest: Vec<&str> = l.split_whitespace().collect();
        let pid = rest.first()?.parse::<u32>().ok()?;
        let ppid = rest.get(1)?.parse::<u32>().ok()?;
        let rss = rest.get(2)?.parse::<u64>().ok()?;
        let cmd = rest.get(3..).map(|c| c.join(" ")).unwrap_or_default();
        Some(Proc { pid, ppid, cmd, rss_mb: Some(rss / 1024) })
    }).collect()
}

/// PowerShell `Get-CimInstance Win32_Process | Select ProcessId,ParentProcessId,CommandLine,WorkingSetSize | ConvertTo-Json -Compress` (an array, or one object for one process).
pub fn parse_win_procs(text: &str) -> Vec<Proc> {
    let Ok(v) = serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) else { return Vec::new() };
    let items: Vec<Value> = match v {
        Value::Array(a) => a,
        o @ Value::Object(_) => vec![o],
        _ => Vec::new(),
    };
    items.iter().filter_map(|p| {
        Some(Proc { pid: p["ProcessId"].as_u64()? as u32, ppid: p["ParentProcessId"].as_u64().unwrap_or(0) as u32, cmd: p["CommandLine"].as_str().unwrap_or("").to_string(),
                    rss_mb: p["WorkingSetSize"].as_u64().map(|b| b / 1_048_576) })
    }).collect()
}

/// `/proc/meminfo` -> (total MB, available MB).
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let kb = |key: &str| text.lines().find_map(|l| l.strip_prefix(key)).and_then(|r| r.trim().split_whitespace().next()).and_then(|n| n.parse::<u64>().ok());
    Some((kb("MemTotal:")? / 1024, kb("MemAvailable:")? / 1024))
}

/// PowerShell `Get-CimInstance Win32_OperatingSystem | Select TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress` (kilobytes) -> (total MB, free MB).
pub fn parse_win_os(text: &str) -> Option<(u64, u64)> {
    let v: Value = serde_json::from_str(text.trim_start_matches('\u{feff}').trim()).ok()?;
    Some((v["TotalVisibleMemorySize"].as_u64()? / 1024, v["FreePhysicalMemory"].as_u64()? / 1024))
}

/// The containers of this program from `docker ps -a --format "{{.Names}} {{.State}}"` (one per line) or `wslc list -a --format json` (one object per container, an array, or a single object; the
/// fields `Names` and `State` as the owner's runs showed). Only names that start with `ndforge-vllm-` are returned; the state word `running` (any case) is running.
pub fn parse_containers(text: &str) -> Vec<Container> {
    let t = text.trim();
    let mut out: Vec<Container> = Vec::new();
    let mut push = |name: &str, state: &str| {
        let n = name.trim().trim_start_matches('/');
        if n.starts_with(CONTAINER_PREFIX) && !out.iter().any(|c| c.name == n) {
            out.push(Container { name: n.to_string(), running: state.to_lowercase().starts_with("running") || state.eq_ignore_ascii_case("up") });
        }
    };
    let mut objects: Vec<Value> = Vec::new();
    if t.starts_with('[') {
        if let Ok(Value::Array(a)) = serde_json::from_str::<Value>(t) {
            objects = a;
        }
    } else if t.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            objects.push(v);
        } else {
            objects = t.lines().filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok()).collect();      // one object per line
        }
    }
    if !objects.is_empty() {
        for o in &objects {
            let state = o["State"].as_str().or_else(|| o["Status"].as_str()).unwrap_or("");
            match &o["Names"] {
                Value::String(s) => push(s, state),
                Value::Array(a) => a.iter().filter_map(Value::as_str).for_each(|s| push(s, state)),
                _ => {}
            }
        }
        return out;
    }
    for l in t.lines() {
        let mut it = l.split_whitespace();
        if let Some(name) = it.next() {
            push(name, it.next().unwrap_or(""));
        }
    }
    out
}

/// Is `pid` below `ancestor` in the process tree (its parent, the parent's parent ...)? A cycle or a missing parent ends the walk.
pub fn descends_from(procs: &[Proc], pid: u32, ancestor: u32) -> bool {
    let mut cur = pid;
    for _ in 0..64 {
        let Some(p) = procs.iter().find(|p| p.pid == cur) else { return false };
        if p.ppid == ancestor {
            return true;
        }
        if p.ppid == 0 || p.ppid == cur {
            return false;
        }
        cur = p.ppid;
    }
    false
}

/// Is the command line one of this program's workers? Matched by the module name in the command line (`-m ndforge_worker` or a path to it), never by the image name alone.
pub fn is_worker_cmd(cmd: &str) -> bool {
    let c = cmd.to_lowercase();
    c.contains(WORKER_MARK) && (c.contains("python") || c.contains("-m ") || c.contains("ndforge_worker/") || c.contains("ndforge_worker\\"))
}

// ------------------------------------------------------------------------------------------------------------------ the machine

/// What the plan and the status need from the machine. The real one runs commands; tests use a fake with a process table and containers.
pub trait Env {
    fn gpu_memory(&self) -> Option<(u64, u64)>;
    fn gpu_apps(&self) -> Vec<GpuProc>;
    fn ram(&self) -> Option<(u64, u64)>;
    fn procs(&self) -> Vec<Proc>;
    fn self_pid(&self) -> u32;
    fn containers(&self) -> Vec<Container>;
    fn remove_container(&self, name: &str) -> std::result::Result<(), String>;
    fn kill(&self, pid: u32) -> std::result::Result<(), String>;
    /// Trims the working set of a process of ours; Ok(megabytes freed) or Err (not supported here)
    fn trim(&self, pid: u32) -> std::result::Result<u64, String>;
    fn can_trim(&self) -> bool;
    fn wsl_shutdown(&self) -> std::result::Result<(), String>;
    fn wsl_present(&self) -> bool;
}

/// What the window needs to know about the program's own state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ctx {
    /// a job (training, heal, export, Auto-cut run, a check ...) is running
    pub busy: bool,
}

// ------------------------------------------------------------------------------------------------------------------ the status

static CACHE: Mutex<Option<(Instant, Value)>> = Mutex::new(None);
const CACHE_FOR: Duration = Duration::from_millis(900);

/// The numbers for the panel: `{gpu: {total_mb, used_mb, free_mb, apps: [...], per_process: bool} | null, ram: {total_mb, free_mb, used_mb, app_mb} | null, containers, busy}`. Cached for about a second, so a window
/// that asks every two seconds and one that asks at once do not start the tools twice.
pub fn status(env: &dyn Env, ctx: &Ctx) -> Value {
    if let Some((at, v)) = CACHE.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        if at.elapsed() < CACHE_FOR && v["busy"] == json!(ctx.busy) {
            return v.clone();
        }
    }
    let v = status_uncached(env, ctx);
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), v.clone()));
    v
}

pub fn clear_cache() {
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

pub fn status_uncached(env: &dyn Env, ctx: &Ctx) -> Value {
    let procs = env.procs();
    let me = env.self_pid();
    let ours: Vec<&Proc> = procs.iter().filter(|p| p.pid == me || (is_worker_cmd(&p.cmd) && (descends_from(&procs, p.pid, me) || orphan(&procs, p)))).collect();
    let our_pids: BTreeSet<u32> = ours.iter().map(|p| p.pid).collect();
    let apps = env.gpu_apps();
    let gpu = env.gpu_memory().map(|(t, u)| {
        let ours_gpu: u64 = apps.iter().filter(|a| our_pids.contains(&a.pid)).filter_map(|a| a.used_mb).sum();
        json!({"total_mb": t, "used_mb": u, "free_mb": t.saturating_sub(u), "per_process": apps.iter().any(|a| a.used_mb.is_some()), "app_mb": if apps.iter().any(|a| a.used_mb.is_some()) { json!(ours_gpu) } else { Value::Null },
               "others": apps.iter().filter(|a| !our_pids.contains(&a.pid)).map(|a| json!({"pid": a.pid, "name": a.name, "used_mb": a.used_mb})).collect::<Vec<_>>()})
    });
    let ram = env.ram().map(|(t, f)| json!({"total_mb": t, "free_mb": f, "used_mb": t.saturating_sub(f), "app_mb": ours.iter().filter_map(|p| p.rss_mb).sum::<u64>()}));
    json!({"gpu": gpu, "ram": ram, "containers": env.containers().iter().map(|c| json!({"name": c.name, "running": c.running})).collect::<Vec<_>>(), "busy": ctx.busy, "wsl": env.wsl_present()})
}

/// A worker whose parent is gone (not in the table, or 0/1) belongs to nobody any more.
fn orphan(procs: &[Proc], p: &Proc) -> bool {
    p.ppid <= 1 || !procs.iter().any(|q| q.pid == p.ppid)
}

// ------------------------------------------------------------------------------------------------------------------ the plan

#[derive(Debug, Clone, Serialize)]
pub struct Item {
    pub id: String,
    /// "container" | "worker" | "trim" | "wsl"
    pub kind: &'static str,
    pub label: String,
    pub detail: String,
    /// why this item will not be done now (a running job uses it), or None
    pub refused: Option<String>,
    /// the running job is the reason: stopping it (the normal Stop) lifts the refusal
    pub needs_stop: bool,
    #[serde(skip)]
    pub pid: Option<u32>,
    #[serde(skip)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub kind: String,
    pub items: Vec<Item>,
    /// other programs that hold memory: listed, never touched
    pub others: Vec<Value>,
    pub notes: Vec<String>,
    pub token: String,
}

fn make_id(kind: &str, key: &str) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (kind, key).hash(&mut h);
    format!("m{:016x}", h.finish())
}

pub fn plan_token(items: &[Item]) -> String {
    let mut ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
    ids.sort_unstable();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ids.hash(&mut h);
    format!("p{:016x}", h.finish())
}

fn worker_items(env: &dyn Env, ctx: &Ctx, procs: &[Proc]) -> Vec<Item> {
    let me = env.self_pid();
    let mut out = Vec::new();
    for p in procs.iter().filter(|p| p.pid != me && is_worker_cmd(&p.cmd)) {
        let ours = descends_from(procs, p.pid, me);
        let orphaned = orphan(procs, p);
        if !ours && !orphaned {
            continue;                       // a worker of another running program: never touched
        }
        // matched by the command line AND the parent relationship; the id holds both, so a pid that is reused by another program gets another id
        let key = format!("{}|{}", p.pid, p.cmd);
        let (refused, needs_stop) = if ours && !orphaned && ctx.busy { (Some("it belongs to the job that is running".to_string()), true) } else { (None, false) };
        out.push(Item { id: make_id("worker", &key), kind: "worker", label: format!("Worker process {}", p.pid), pid: Some(p.pid), name: None, needs_stop, refused,
                        detail: format!("{}{}", if orphaned { "left over: its job is gone. " } else { "idle. " }, p.rss_mb.map(|m| format!("About {m} MB of RAM. ")).unwrap_or_default()) });
    }
    out
}

fn container_items(env: &dyn Env, ctx: &Ctx) -> Vec<Item> {
    env.containers().into_iter().map(|c| {
        let (refused, needs_stop) = if c.running && ctx.busy { (Some("a running job uses this engine".to_string()), true) } else { (None, false) };
        Item { id: make_id("container", &c.name), kind: "container", label: format!("Engine container {}", c.name), pid: None, name: Some(c.name.clone()), needs_stop, refused,
               detail: if c.running { "running: the engine holds the model in the graphics memory".into() } else { "stopped, left behind by an earlier run".into() } }
    }).collect()
}

/// The exact list for `kind`: "gpu" (our containers, our workers; other programs are listed), "ram" (our workers, a trim of our own processes where the system supports it) or "wsl" (release the memory the
/// WSL virtual machine keeps: only when no engine container exists and no job runs; it ends every WSL distribution).
pub fn plan(env: &dyn Env, ctx: &Ctx, kind: &str) -> Result<Plan> {
    let procs = env.procs();
    let mut items = Vec::new();
    let mut others: Vec<Value> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    match kind {
        "gpu" => {
            items.extend(container_items(env, ctx));
            items.extend(worker_items(env, ctx, &procs));
            let me = env.self_pid();
            let theirs: BTreeSet<u32> = procs.iter().filter(|p| p.pid == me || (is_worker_cmd(&p.cmd) && (descends_from(&procs, p.pid, me) || orphan(&procs, p)))).map(|p| p.pid).collect();
            let apps = env.gpu_apps();
            for a in apps.iter().filter(|a| !theirs.contains(&a.pid)) {
                others.push(json!({"pid": a.pid, "name": a.name, "used_mb": a.used_mb, "note": "another program: close it yourself, this program never ends it"}));
            }
            if apps.iter().all(|a| a.used_mb.is_none()) {
                notes.push("The driver does not say which program uses how much graphics memory (usual on Windows): only the totals are shown. The desktop itself, a browser, a game, Ollama, LM Studio or Unsloth Desktop may hold some of it.".into());
            }
        }
        "ram" => {
            items.extend(worker_items(env, ctx, &procs));
            if env.can_trim() {
                let me = env.self_pid();
                let ended: BTreeSet<u32> = items.iter().filter(|i| i.refused.is_none()).filter_map(|i| i.pid).collect();
                for p in procs.iter().filter(|p| (p.pid == me || (is_worker_cmd(&p.cmd) && descends_from(&procs, p.pid, me))) && !ended.contains(&p.pid)) {
                    items.push(Item { id: make_id("trim", &format!("{}|{}", p.pid, p.cmd)), kind: "trim", label: format!("Trim the working set of process {}", p.pid), pid: Some(p.pid), name: None, needs_stop: false, refused: None,
                                      detail: "Windows gives back the memory this program is not using right now; it is read in again when needed".into() });
                }
            }
            notes.push("Windows manages its own file cache: it is not a leak, and there is no 'RAM cleaner' here (no purging of the standby list, no ending of system processes). The free memory shown includes that cache.".into());
        }
        "wsl" => {
            if !env.wsl_present() {
                return Err(Error::new("WSL was not found on this computer."));
            }
            let running = env.containers().iter().any(|c| c.running);
            let refused = if ctx.busy { Some("a job is running".to_string()) } else if running { Some("an engine container is running".to_string()) } else if !env.containers().is_empty() { Some("an engine container still exists: end it with Free GPU first".to_string()) } else { None };
            items.push(Item { id: make_id("wsl", "shutdown"), kind: "wsl", label: "Release the memory of the WSL virtual machine".into(), pid: None, name: None, needs_stop: false, refused,
                              detail: "Runs `wsl --shutdown`: this ENDS EVERY WSL distribution (also the ones you use for other things, and anything running in them). They start again when used.".into() });
        }
        _ => return Err(Error::new("Free what? gpu, ram or wsl.")),
    }
    let token = plan_token(&items);
    Ok(Plan { kind: kind.into(), items, others, notes, token })
}

// ------------------------------------------------------------------------------------------------------------------ doing it

/// Does what `plan` lists, after planning again: a token that does not match the fresh plan is refused (nothing is done). An item a running job uses stays undone (`skipped`). Returns the numbers before and
/// after, what was done, skipped and failed, and the log lines.
pub fn free(env: &dyn Env, ctx: &Ctx, kind: &str, token: &str) -> Result<Value> {
    let fresh = plan(env, ctx, kind)?;
    if token.is_empty() || fresh.token != token {
        return Err(Error::new("The list has changed since it was made (something started, ended or appeared). Look at the new list and confirm again. Nothing was done."));
    }
    let before = snapshot(env);
    let mut done = Vec::new();
    let mut skipped = Vec::new();
    let mut failed = Vec::new();
    let mut log = Vec::new();
    for it in &fresh.items {
        if let Some(why) = &it.refused {
            skipped.push(json!({"label": it.label, "why": why, "needs_stop": it.needs_stop}));
            continue;
        }
        let r = match it.kind {
            "container" => env.remove_container(it.name.as_deref().unwrap_or_default()).map(|_| 0u64),
            "worker" => env.kill(it.pid.unwrap_or(0)).map(|_| 0),
            "trim" => env.trim(it.pid.unwrap_or(0)),
            "wsl" => env.wsl_shutdown().map(|_| 0),
            _ => Err("unknown".into()),
        };
        match r {
            Ok(mb) => {
                log.push(format!("{} ended or released.", it.label));
                done.push(json!({"label": it.label, "freed_mb": mb}));
            }
            Err(e) => {
                log.push(format!("{} could not be done: {e}", it.label));
                failed.push(json!({"label": it.label, "why": e}));
            }
        }
    }
    clear_cache();
    let after = snapshot(env);
    Ok(json!({"kind": kind, "before": before, "after": after, "done": done, "skipped": skipped, "failed": failed, "log": log,
              "others": fresh.others, "notes": fresh.notes}))
}

fn snapshot(env: &dyn Env) -> Value {
    json!({"gpu": env.gpu_memory().map(|(t, u)| json!({"total_mb": t, "used_mb": u, "free_mb": t.saturating_sub(u)})), "ram": env.ram().map(|(t, f)| json!({"total_mb": t, "free_mb": f, "used_mb": t.saturating_sub(f)}))})
}

/// The engine containers this program left behind (a crashed run): removed at the start of a new vLLM job, with a line for the log. Nothing is done while a job runs. Returns the names removed.
pub fn remove_stale(env: &dyn Env, ctx: &Ctx) -> Vec<String> {
    if ctx.busy {
        return Vec::new();
    }
    let mut gone = Vec::new();
    for c in env.containers() {
        if env.remove_container(&c.name).is_ok() {
            gone.push(c.name);
        }
    }
    if !gone.is_empty() {
        clear_cache();
    }
    gone
}

// ------------------------------------------------------------------------------------------------------------------ the real machine

pub struct Real {
    pub vllm: crate::settings::VllmSettings,
}

fn run_text(args: &[&str]) -> Option<(bool, String)> {
    let mut cmd = std::process::Command::new(args.first()?);
    cmd.args(&args[1..]).stdin(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd.output().ok()?;
    Some((out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))))
}

impl Real {
    fn prefix(&self) -> Option<Vec<String>> {
        match self.vllm.kind.as_str() {
            "docker" | "auto" | "" => Some(vec!["docker".into()]),
            "docker_wsl" => Some(if self.vllm.distro.trim().is_empty() { vec!["wsl.exe".into(), "docker".into()] } else { vec!["wsl.exe".into(), "-d".into(), self.vllm.distro.clone(), "docker".into()] }),
            _ => None,
        }
    }
}

impl Env for Real {
    fn gpu_memory(&self) -> Option<(u64, u64)> {
        let (ok, t) = run_text(&["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"])?;
        if ok { parse_gpu_memory(&t) } else { None }
    }
    fn gpu_apps(&self) -> Vec<GpuProc> {
        run_text(&["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]).filter(|(ok, _)| *ok).map(|(_, t)| parse_gpu_apps(&t)).unwrap_or_default()
    }
    fn ram(&self) -> Option<(u64, u64)> {
        if cfg!(windows) {
            let (ok, t) = run_text(&["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress"])?;
            if ok { parse_win_os(&t) } else { None }
        } else {
            parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
        }
    }
    fn procs(&self) -> Vec<Proc> {
        if cfg!(windows) {
            run_text(&["powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine,WorkingSetSize | ConvertTo-Json -Compress"])
                .filter(|(ok, _)| *ok).map(|(_, t)| parse_win_procs(&t)).unwrap_or_default()
        } else {
            run_text(&["ps", "-eo", "pid=,ppid=,rss=,args="]).filter(|(ok, _)| *ok).map(|(_, t)| parse_ps(&t)).unwrap_or_default()
        }
    }
    fn self_pid(&self) -> u32 {
        std::process::id()
    }
    fn containers(&self) -> Vec<Container> {
        if self.vllm.kind == "wslc" {
            return run_text(&["wslc", "list", "-a", "--format", "json"]).filter(|(ok, _)| *ok).map(|(_, t)| parse_containers(&t)).unwrap_or_default();
        }
        let Some(mut pre) = self.prefix() else { return Vec::new() };
        pre.extend(["ps", "-a", "--filter", "name=ndforge-vllm-", "--format", "{{.Names}} {{.State}}"].map(String::from));
        let v: Vec<&str> = pre.iter().map(String::as_str).collect();
        run_text(&v).filter(|(ok, _)| *ok).map(|(_, t)| parse_containers(&t)).unwrap_or_default()
    }
    fn remove_container(&self, name: &str) -> std::result::Result<(), String> {
        if !name.starts_with(CONTAINER_PREFIX) || name.contains(char::is_whitespace) {
            return Err("not a container of this program".into());
        }
        if self.vllm.kind == "wslc" {
            let _ = run_text(&["wslc", "stop", "-t", "30", name]);
            return match run_text(&["wslc", "remove", "-f", name]) { Some((true, _)) => Ok(()), Some((false, t)) if t.contains("NOT_FOUND") => Ok(()), Some((false, t)) => Err(t.trim().chars().take(200).collect()), None => Err("wslc was not found".into()) };
        }
        let mut pre = self.prefix().ok_or_else(|| "no engine kind that runs containers".to_string())?;
        pre.extend(["rm", "-f", name].map(String::from));
        let v: Vec<&str> = pre.iter().map(String::as_str).collect();
        match run_text(&v) { Some((true, _)) => Ok(()), Some((false, t)) => Err(t.trim().chars().take(200).collect()), None => Err("the container tool was not found".into()) }
    }
    fn kill(&self, pid: u32) -> std::result::Result<(), String> {
        if pid <= 1 || pid == std::process::id() {
            return Err("refused".into());
        }
        let r = if cfg!(windows) { run_text(&["taskkill", "/PID", &pid.to_string(), "/T", "/F"]) } else { run_text(&["kill", "-TERM", &pid.to_string()]) };
        match r { Some((true, _)) => Ok(()), Some((false, t)) => Err(t.trim().chars().take(200).collect()), None => Err("the tool to end a process was not found".into()) }
    }
    fn trim(&self, pid: u32) -> std::result::Result<u64, String> {
        trim_working_set(pid)
    }
    fn can_trim(&self) -> bool {
        cfg!(windows)
    }
    fn wsl_shutdown(&self) -> std::result::Result<(), String> {
        match run_text(&["wsl.exe", "--shutdown"]) { Some((true, _)) => Ok(()), Some((false, t)) => Err(t.trim().chars().take(200).collect()), None => Err("wsl.exe was not found".into()) }
    }
    fn wsl_present(&self) -> bool {
        cfg!(windows) && run_text(&["wsl.exe", "--status"]).is_some()
    }
}

#[cfg(windows)]
fn trim_working_set(pid: u32) -> std::result::Result<u64, String> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::{K32EmptyWorkingSet, K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_SET_QUOTA};
    // SAFETY: plain Win32 calls on a handle this function opens and closes; the counters struct is zeroed and sized as the API wants.
    unsafe {
        let h = OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, 0, pid);
        if h == 0 {
            return Err("the process could not be opened (it is not ours, or it ended)".into());
        }
        let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        c.cb = cb;
        let before = if K32GetProcessMemoryInfo(h, &mut c, cb) != 0 { c.WorkingSetSize as u64 } else { 0 };
        let ok = K32EmptyWorkingSet(h) != 0;
        let after = if K32GetProcessMemoryInfo(h, &mut c, cb) != 0 { c.WorkingSetSize as u64 } else { before };
        CloseHandle(h);
        if ok { Ok(before.saturating_sub(after) / 1_048_576) } else { Err("Windows refused to trim it".into()) }
    }
}

#[cfg(not(windows))]
fn trim_working_set(_pid: u32) -> std::result::Result<u64, String> {
    Err("trimming the working set is a Windows feature".into())
}
