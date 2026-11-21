//! The memory panel: the parsers (real sample outputs), the plan (own against foreign, a running job refused, an orphan accepted) and the execution (a stale token refused, only the plan is done).
use std::cell::RefCell;

use ndforge_core::memory::*;
use serde_json::json;

// ---- real sample outputs
const SMI_TOTALS: &str = "32607, 3350\n";
const SMI_APPS: &str = "1234, C:\\Users\\me\\AppData\\Local\\Programs\\Python\\python.exe, [N/A]\n5678, /usr/bin/python3, 21000\n";
const SMI_NONE: &str = "No running processes found\n";
const PS: &str = "    1     0   5120 /sbin/init\n 4242  1000 2097152 /usr/bin/python3 -m ndforge_worker\n 4300  4242   10240 sleep 5\n";
const WIN_PROCS: &str = "\u{feff}[{\"ProcessId\":4,\"ParentProcessId\":0,\"CommandLine\":null,\"WorkingSetSize\":139264},{\"ProcessId\":9000,\"ParentProcessId\":8000,\"CommandLine\":\"\\\"C:\\\\Python\\\\python.exe\\\" -m ndforge_worker\",\"WorkingSetSize\":3221225472}]";
const MEMINFO: &str = "MemTotal:       65536000 kB\nMemFree:         1000000 kB\nMemAvailable:   40960000 kB\n";
const WIN_OS: &str = "{\"TotalVisibleMemorySize\":67108864,\"FreePhysicalMemory\":20971520}";
// the owner's `wslc list -a --format json`: one object per container (a single OBJECT, not an array, when there is one)
const WSLC_ONE: &str = "{\"Command\":\"sleep infinity\",\"CreatedAt\":\"2026-11-21T03:10:20Z\",\"HealthStatus\":\"\",\"ID\":\"8ed34c42aa\",\"Image\":\"vllm/vllm-openai:v0.34.0\",\"Names\":\"ndforge-vllm-41643\",\"State\":\"running\",\"Status\":\"Up 2 minutes\"}";
const WSLC_TWO: &str = "{\"Names\":\"ndforge-vllm-1\",\"State\":\"exited\"}\n{\"Names\":[\"other-thing\"],\"State\":\"running\"}\n{\"Names\":\"ndforge-vllm-2\",\"State\":\"running\"}";
const DOCKER_PS: &str = "ndforge-vllm-55 Up 3 minutes\nndforge-vllm-56 Exited (1) 2 hours ago\nunrelated running\n";

#[test]
fn the_parsers_read_real_outputs() {
    assert_eq!(parse_gpu_memory(SMI_TOTALS), Some((32607, 3350)));
    assert_eq!(parse_gpu_memory("garbage"), None);
    let apps = parse_gpu_apps(SMI_APPS);
    assert_eq!(apps.len(), 2);
    assert_eq!((apps[0].pid, apps[0].used_mb), (1234, None), "WDDM: [N/A] is no number");
    assert_eq!((apps[1].pid, apps[1].used_mb), (5678, Some(21000)));
    assert!(parse_gpu_apps(SMI_NONE).is_empty());
    let ps = parse_ps(PS);
    assert_eq!(ps.len(), 3);
    assert_eq!((ps[1].pid, ps[1].ppid, ps[1].rss_mb), (4242, 1000, Some(2048)));
    assert_eq!(ps[1].cmd, "/usr/bin/python3 -m ndforge_worker");
    let w = parse_win_procs(WIN_PROCS);
    assert_eq!(w.len(), 2);
    assert_eq!((w[1].pid, w[1].ppid, w[1].rss_mb), (9000, 8000, Some(3072)));
    assert_eq!(parse_win_procs("{\"ProcessId\":7,\"ParentProcessId\":1,\"CommandLine\":\"x\",\"WorkingSetSize\":1048576}").len(), 1, "one process comes as one object");
    assert_eq!(parse_meminfo(MEMINFO), Some((64000, 40000)));
    assert_eq!(parse_win_os(WIN_OS), Some((65536, 20480)));
    let one = parse_containers(WSLC_ONE);
    assert_eq!(one, vec![Container { name: "ndforge-vllm-41643".into(), running: true }]);
    let two = parse_containers(WSLC_TWO);
    assert_eq!(two, vec![Container { name: "ndforge-vllm-1".into(), running: false }, Container { name: "ndforge-vllm-2".into(), running: true }]);
    let d = parse_containers(DOCKER_PS);
    assert_eq!(d, vec![Container { name: "ndforge-vllm-55".into(), running: true }, Container { name: "ndforge-vllm-56".into(), running: false }]);
}

#[test]
fn a_worker_is_recognised_by_its_command_line_and_the_tree_never_by_the_image_name() {
    assert!(is_worker_cmd("\"C:\\Python\\python.exe\" -m ndforge_worker"));
    assert!(is_worker_cmd("/home/u/.venv/bin/python /opt/ndforge_worker/__main__.py"));
    assert!(!is_worker_cmd("C:\\Python\\python.exe my_script.py"), "an image name alone is nothing");
    assert!(!is_worker_cmd("vim ndforge_worker_notes.txt"));
    let procs = vec![Proc { pid: 1, ppid: 0, cmd: "init".into(), rss_mb: None }, Proc { pid: 10, ppid: 1, cmd: "app".into(), rss_mb: None }, Proc { pid: 11, ppid: 10, cmd: "sh".into(), rss_mb: None }, Proc { pid: 12, ppid: 11, cmd: "py".into(), rss_mb: None }];
    assert!(descends_from(&procs, 12, 10) && descends_from(&procs, 11, 10) && !descends_from(&procs, 10, 12) && !descends_from(&procs, 12, 99));
}

// ---- a fake machine
struct Fake {
    me: u32,
    procs: Vec<Proc>,
    containers: Vec<Container>,
    gpu: (u64, u64),
    apps: Vec<GpuProc>,
    trim: bool,
    wsl: bool,
    killed: RefCell<Vec<u32>>,
    removed: RefCell<Vec<String>>,
    trimmed: RefCell<Vec<u32>>,
    wsl_down: RefCell<bool>,
}

fn proc(pid: u32, ppid: u32, cmd: &str) -> Proc {
    Proc { pid, ppid, cmd: cmd.into(), rss_mb: Some(1000) }
}

fn fake() -> Fake {
    Fake { me: 100, trim: false, wsl: true, gpu: (32000, 20000), apps: vec![], killed: RefCell::new(vec![]), removed: RefCell::new(vec![]), trimmed: RefCell::new(vec![]), wsl_down: RefCell::new(false),
           containers: vec![],
           procs: vec![proc(1, 0, "init"), proc(100, 1, "ndforge app"),
                       proc(200, 100, "python -m ndforge_worker"),                  // ours, child of the app
                       proc(201, 999, "python -m ndforge_worker"),                  // an orphan (its parent is gone)
                       proc(300, 50, "python -m ndforge_worker"),                   // another program's worker: the parent 50 is alive and is not us
                       proc(50, 1, "another app"),
                       proc(400, 100, "python other_script.py")] }
}

impl Env for Fake {
    fn gpu_memory(&self) -> Option<(u64, u64)> { Some(self.gpu) }
    fn gpu_apps(&self) -> Vec<GpuProc> { self.apps.clone() }
    fn ram(&self) -> Option<(u64, u64)> { Some((64000, 20000)) }
    fn procs(&self) -> Vec<Proc> { self.procs.clone() }
    fn self_pid(&self) -> u32 { self.me }
    fn containers(&self) -> Vec<Container> { self.containers.iter().filter(|c| !self.removed.borrow().contains(&c.name)).cloned().collect() }
    fn remove_container(&self, name: &str) -> Result<(), String> { self.removed.borrow_mut().push(name.into()); Ok(()) }
    fn kill(&self, pid: u32) -> Result<(), String> { self.killed.borrow_mut().push(pid); Ok(()) }
    fn trim(&self, pid: u32) -> Result<u64, String> { self.trimmed.borrow_mut().push(pid); Ok(300) }
    fn can_trim(&self) -> bool { self.trim }
    fn wsl_shutdown(&self) -> Result<(), String> { *self.wsl_down.borrow_mut() = true; Ok(()) }
    fn wsl_present(&self) -> bool { self.wsl }
}

fn labels(p: &Plan) -> Vec<String> {
    p.items.iter().map(|i| i.label.clone()).collect()
}

#[test]
fn free_gpu_lists_only_what_this_program_owns_and_only_lists_the_others() {
    let mut f = fake();
    f.containers = vec![Container { name: "ndforge-vllm-1".into(), running: false }];
    f.apps = vec![GpuProc { pid: 777, name: "chrome.exe".into(), used_mb: Some(900) }, GpuProc { pid: 200, name: "python.exe".into(), used_mb: Some(5000) }];
    let p = plan(&f, &Ctx { busy: false }, "gpu").unwrap();
    let l = labels(&p);
    assert!(l.contains(&"Engine container ndforge-vllm-1".to_string()) && l.contains(&"Worker process 200".to_string()) && l.contains(&"Worker process 201".to_string()), "{l:?}");
    assert!(!l.iter().any(|x| x.contains("300") || x.contains("400") || x.contains("777")), "another program's worker, an unrelated python and a browser are never in the plan: {l:?}");
    assert!(p.items.iter().all(|i| i.refused.is_none()));
    assert_eq!(p.others.len(), 1, "the browser is only listed");
    assert_eq!(p.others[0]["pid"], 777);
    assert_eq!(p.others[0]["note"].as_str().unwrap().contains("never ends it"), true);
}

#[test]
fn a_running_job_keeps_its_worker_and_its_engine_unless_it_is_stopped_first_and_an_orphan_goes() {
    let mut f = fake();
    f.containers = vec![Container { name: "ndforge-vllm-2".into(), running: true }];
    let p = plan(&f, &Ctx { busy: true }, "gpu").unwrap();
    let by = |label: &str| p.items.iter().find(|i| i.label == label).unwrap();
    assert!(by("Worker process 200").refused.is_some() && by("Worker process 200").needs_stop);
    assert!(by("Engine container ndforge-vllm-2").refused.is_some() && by("Engine container ndforge-vllm-2").needs_stop);
    assert!(by("Worker process 201").refused.is_none(), "an orphan belongs to no running job");
    let r = free(&f, &Ctx { busy: true }, "gpu", &p.token).unwrap();
    assert_eq!(*f.killed.borrow(), vec![201]);
    assert!(f.removed.borrow().is_empty());
    assert_eq!(r["skipped"].as_array().unwrap().len(), 2);
    assert_eq!(r["done"].as_array().unwrap().len(), 1);
    // after the job was stopped the same list (same token) can be done
    let f2 = { let mut g = fake(); g.containers = f.containers.clone(); g };
    let p2 = plan(&f2, &Ctx { busy: true }, "gpu").unwrap();
    let again = plan(&f2, &Ctx { busy: false }, "gpu").unwrap();
    assert_eq!(p2.token, again.token, "the token does not depend on the job's state");
    free(&f2, &Ctx { busy: false }, "gpu", &p2.token).unwrap();
    let mut k = f2.killed.borrow().clone();
    k.sort();
    assert_eq!(k, vec![200, 201]);
    assert_eq!(*f2.removed.borrow(), vec!["ndforge-vllm-2".to_string()]);
}

#[test]
fn a_stale_or_missing_token_does_nothing() {
    let mut f = fake();
    f.containers = vec![Container { name: "ndforge-vllm-3".into(), running: false }];
    let p = plan(&f, &Ctx::default(), "gpu").unwrap();
    assert!(free(&f, &Ctx::default(), "gpu", "").is_err());
    assert!(free(&f, &Ctx::default(), "gpu", "p0000000000000000").is_err());
    f.procs.push(proc(205, 100, "python -m ndforge_worker"));        // something new appeared since the list was made
    let e = free(&f, &Ctx::default(), "gpu", &p.token).unwrap_err().0;
    assert!(e.contains("has changed") && e.contains("Nothing was done"), "{e}");
    assert!(f.killed.borrow().is_empty() && f.removed.borrow().is_empty());
    assert!(plan(&f, &Ctx::default(), "everything").is_err());
}

#[test]
fn free_ram_ends_idle_workers_and_trims_only_where_the_system_supports_it_with_the_honest_note() {
    let f = fake();
    let p = plan(&f, &Ctx::default(), "ram").unwrap();
    assert!(p.items.iter().all(|i| i.kind == "worker"), "no trim item without support");
    assert!(p.notes.iter().any(|n| n.contains("file cache") && n.contains("no 'RAM cleaner'")));
    let mut g = fake();
    g.trim = true;
    let p = plan(&g, &Ctx::default(), "ram").unwrap();
    let trims: Vec<_> = p.items.iter().filter(|i| i.kind == "trim").collect();
    assert_eq!(trims.len(), 1, "the app itself; the workers that are ended are not trimmed too: {:?}", labels(&p));
    let r = free(&g, &Ctx::default(), "ram", &p.token).unwrap();
    assert_eq!(*g.trimmed.borrow(), vec![100]);
    assert_eq!(r["done"].as_array().unwrap().iter().filter(|d| d["freed_mb"] == 300).count(), 1);
    assert!(r["before"]["ram"]["free_mb"].is_number() && r["after"]["gpu"]["total_mb"] == 32000);
}

#[test]
fn the_wsl_release_is_a_separate_plan_refused_while_an_engine_exists_or_a_job_runs() {
    let mut f = fake();
    f.containers = vec![Container { name: "ndforge-vllm-4".into(), running: true }];
    let p = plan(&f, &Ctx::default(), "wsl").unwrap();
    assert!(p.items[0].refused.as_deref().unwrap().contains("running"));
    assert!(p.items[0].detail.contains("ENDS EVERY WSL distribution"));
    free(&f, &Ctx::default(), "wsl", &p.token).unwrap();
    assert!(!*f.wsl_down.borrow());
    let busy = plan(&fake(), &Ctx { busy: true }, "wsl").unwrap();
    assert!(busy.items[0].refused.as_deref().unwrap().contains("job"));
    let free_now = fake();
    let p = plan(&free_now, &Ctx::default(), "wsl").unwrap();
    assert!(p.items[0].refused.is_none());
    free(&free_now, &Ctx::default(), "wsl", &p.token).unwrap();
    assert!(*free_now.wsl_down.borrow());
    let mut none = fake();
    none.wsl = false;
    assert!(plan(&none, &Ctx::default(), "wsl").is_err());
}

#[test]
fn the_status_has_the_numbers_the_apps_share_and_the_other_programs() {
    let mut f = fake();
    f.apps = vec![GpuProc { pid: 200, name: "python".into(), used_mb: Some(5000) }, GpuProc { pid: 777, name: "chrome".into(), used_mb: Some(900) }];
    f.containers = vec![Container { name: "ndforge-vllm-5".into(), running: true }];
    let v = status_uncached(&f, &Ctx { busy: true });
    assert_eq!((v["gpu"]["total_mb"].clone(), v["gpu"]["used_mb"].clone(), v["gpu"]["free_mb"].clone()), (json!(32000), json!(20000), json!(12000)));
    assert_eq!(v["gpu"]["app_mb"], 5000);
    assert_eq!(v["gpu"]["per_process"], true);
    assert_eq!(v["gpu"]["others"].as_array().unwrap().len(), 1);
    assert_eq!(v["ram"]["total_mb"], 64000);
    assert_eq!(v["ram"]["app_mb"], 3000, "the app and its two workers (the other program's worker and the other script are not ours)");
    assert_eq!(v["containers"][0]["running"], true);
    // the driver that gives no per-process numbers (Windows WDDM)
    f.apps = vec![GpuProc { pid: 200, name: "python".into(), used_mb: None }];
    let w = status_uncached(&f, &Ctx::default());
    assert_eq!(w["gpu"]["per_process"], false);
    assert!(w["gpu"]["app_mb"].is_null());
}

#[test]
fn leftover_engines_are_removed_before_a_new_job_but_never_while_one_runs() {
    let mut f = fake();
    f.containers = vec![Container { name: "ndforge-vllm-6".into(), running: false }, Container { name: "ndforge-vllm-7".into(), running: true }];
    assert!(remove_stale(&f, &Ctx { busy: true }).is_empty() && f.removed.borrow().is_empty());
    assert_eq!(remove_stale(&f, &Ctx::default()), vec!["ndforge-vllm-6".to_string(), "ndforge-vllm-7".to_string()]);
}

#[test]
fn the_real_machine_refuses_what_is_not_ours() {
    let real = Real { vllm: ndforge_core::settings::VllmSettings::default() };
    assert!(real.remove_container("some-other-container").is_err());
    assert!(real.remove_container("ndforge-vllm-1 ; rm -rf").is_err());
    assert!(real.kill(0).is_err() && real.kill(1).is_err() && real.kill(std::process::id()).is_err(), "never pid 0, 1 or itself");
}
