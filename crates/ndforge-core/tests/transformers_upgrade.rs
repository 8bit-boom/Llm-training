//! "Upgrade transformers" (docs/TRANSFORMERS_UPGRADE.md) against a stand-in Python and a stand-in uv: no network, no real environment.
#![cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use ndforge_core::envmgr::transformers_upgrade::*;
use ndforge_core::update::install::Activity;
use serde_json::{json, Value};

fn private_config_dir() {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let d = DIR.get_or_init(|| tempfile::tempdir().unwrap());
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
}

fn script(p: &Path, body: &str) {
    std::fs::write(p, body).unwrap();
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Switch files in the stand-in's folder: `fail_install`, `fail_rollback`, `torch_change`, `slow`, `break_import` (a version whose import fails).
struct Fake {
    d: tempfile::TempDir,
    u: Upgrade,
}

impl Fake {
    fn new(version: &str) -> Fake {
        private_config_dir();
        let d = tempfile::tempdir().unwrap();
        let p = d.path();
        std::fs::create_dir_all(p.join("env/bin")).unwrap();
        std::fs::create_dir_all(p.join("worker")).unwrap();
        std::fs::write(p.join("version"), version).unwrap();
        let ds = p.display();
        script(&p.join("env/bin/python"), &format!("#!/bin/sh\nD={ds}\nV=$(cat $D/version)\ncase \"$1\" in\n -c) if [ -f $D/break_import ] && [ \"$V\" = \"$(cat $D/break_import)\" ]; then echo \"{{\\\"version\\\":\\\"$V\\\",\\\"import_error\\\":\\\"ImportError: boom\\\"}}\"; else echo \"{{\\\"version\\\":\\\"$V\\\",\\\"import_version\\\":\\\"$V\\\"}}\"; fi;;\n *) echo \"{{\\\"type\\\":\\\"result\\\",\\\"data\\\":{{\\\"packages\\\":{{\\\"transformers\\\":\\\"$V\\\"}}}}}}\";;\nesac\n"));
        script(&p.join("uv"), &format!(r#"#!/bin/sh
D={ds}
echo "$@" >> $D/uv.log
if [ "$1 $2" = "pip freeze" ]; then printf 'numpy==2.0.0\ntorch==2.8.0+cu128\nbitsandbytes==0.47.0\n'; exit 0; fi
DRY=0; case "$*" in *--dry-run*) DRY=1;; esac
case "$*" in
 *"transformers>="*)
   if [ -f $D/torch_change ]; then printf ' - torch==2.8.0+cu128\n + torch==2.9.0\n + transformers==5.24.1\n'; [ $DRY = 1 ] && exit 0; fi
   if [ $DRY = 1 ]; then printf ' - transformers==5.17.0\n + transformers==5.24.1\n - tokenizers==0.21.0\n + tokenizers==0.30.0\n'; exit 0; fi
   if [ -f $D/slow ]; then sleep 30; fi
   if [ -f $D/fail_install ]; then echo 'error: no matching distribution' >&2; exit 2; fi
   OLD=$(cat $D/version); echo 5.24.1 > $D/version
   printf ' - transformers==%s\n + transformers==5.24.1\n - tokenizers==0.21.0\n + tokenizers==0.30.0\n' "$OLD";;
 *"transformers=="*)
   V=$(echo "$*" | sed 's/.*transformers==\([0-9.]*\).*/\1/')
   if [ $DRY = 1 ]; then printf ' - transformers==5.24.1\n + transformers==%s\n' "$V"; exit 0; fi
   if [ -f $D/fail_rollback ]; then exit 2; fi
   OLD=$(cat $D/version); echo $V > $D/version
   printf ' - transformers==%s\n + transformers==%s\n' "$OLD" "$V";;
esac
exit 0
"#));
        let u = Upgrade { env_dir: p.join("env"), py: p.join("env/bin/python"), uv: Some(p.join("uv")), worker_dir: p.join("worker"), record: p.join("data/transformers_upgrade.json") };
        Fake { d, u }
    }
    fn flag(&self, name: &str, content: &str) {
        std::fs::write(self.d.path().join(name), content).unwrap();
    }
    fn version(&self) -> String {
        std::fs::read_to_string(self.d.path().join("version")).unwrap().trim().to_string()
    }
    fn log(&self) -> Vec<String> {
        std::fs::read_to_string(self.d.path().join("uv.log")).unwrap_or_default().lines().map(String::from).collect()
    }
    fn real_installs(&self) -> usize {
        self.log().iter().filter(|l| l.starts_with("pip install") && !l.contains("--dry-run")).count()
    }
}

fn idle() -> Activity {
    Activity::default()
}

fn names(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap().to_string()).collect()
}

#[test]
fn a_successful_upgrade_records_the_old_version_first_and_lists_what_changed() {
    let f = Fake::new("5.17.0");
    let mut events = Vec::new();
    let r = run_upgrade(&f.u, true, &idle, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert_eq!(r["ok"], true, "{r}");
    assert_eq!(f.version(), "5.24.1");
    assert_eq!(r["previous"], "5.17.0");
    assert_eq!(names(&r["changed"]), ["transformers", "tokenizers"]);
    assert_eq!(r["changed"][0]["from"], "5.17.0");
    assert_eq!(r["changed"][0]["to"], "5.24.1");
    assert!(r["text"].as_str().unwrap().contains("transformers 5.24.1") && r["text"].as_str().unwrap().contains("Go back to 5.17.0"));
    let rec = read_record(&f.u.record).unwrap();
    assert_eq!((rec.previous.as_str(), rec.state.as_str(), rec.installed.as_deref()), ("5.17.0", "installed", Some("5.24.1")));
    // the order: freeze (the pins), the resolver's preview, the real install; into the app's Python only, with the pins, and no blanket upgrade
    let log = f.log();
    let py = f.u.py.display().to_string();
    assert!(log[0].starts_with("pip freeze"));
    let (dry, real): (Vec<_>, Vec<_>) = log.iter().filter(|l| l.starts_with("pip install")).partition(|l| l.contains("--dry-run"));
    assert_eq!((dry.len(), real.len()), (1, 1), "{log:?}");
    for l in dry.iter().chain(real.iter()) {
        assert!(l.contains(&format!("--python {py}")) && l.contains("transformers>=5.24.0,<6") && l.contains("transformers_upgrade_pins.txt"), "{l}");
        assert!(!l.contains("--upgrade") && !l.contains("--system") && !l.contains("--break-system-packages"), "{l}");
    }
    assert_eq!(std::fs::read_to_string(f.u.env_dir.join("transformers_upgrade_pins.txt")).unwrap(), "torch==2.8.0+cu128\nbitsandbytes==0.47.0\n");
    assert!(events.iter().all(|e| e["task"] == "transformers") && events.iter().filter(|e| e["kind"] == "step").count() == 4);
    assert!(events.iter().any(|e| e["kind"] == "log" && e["text"].as_str().unwrap().contains("would change")));

    // the plan now offers the way back, and the way back puts 5.17.0 back (a stale version on the button is refused)
    let plan = plan_view(&f.u);
    assert_eq!((plan["current"].as_str(), plan["can_go_back"].as_bool(), plan["previous"].as_str()), (Some("5.24.1"), Some(true), Some("5.17.0")));
    assert!(run_rollback(&f.u, true, "5.10.0", &idle, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("not 5.10.0"));
    assert!(run_rollback(&f.u, false, "5.17.0", &idle, &AtomicBool::new(false), &mut |_| {}).is_err());
    let b = run_rollback(&f.u, true, "5.17.0", &idle, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(b["ok"], true, "{b}");
    assert_eq!(f.version(), "5.17.0");
    assert_eq!(read_record(&f.u.record).unwrap().state, "went_back");
    assert_eq!(plan_view(&f.u)["can_go_back"], false);
    // and the version is old again, so the upgrade is offered again
    assert_eq!(plan_view(&f.u)["needed"], true);
}

#[test]
fn a_failed_install_puts_the_recorded_version_back_by_itself_and_says_so() {
    let f = Fake::new("5.17.0");
    f.flag("fail_install", "");
    let r = run_upgrade(&f.u, true, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!((r["ok"].as_bool(), r["rolled_back"].as_bool(), r["rollback_ok"].as_bool()), (Some(false), Some(true), Some(true)), "{r}");
    assert!(r["text"].as_str().unwrap().contains("put back automatically") && r["text"].as_str().unwrap().contains("5.17.0"));
    assert_eq!(f.version(), "5.17.0");
    assert_eq!(read_record(&f.u.record).unwrap().state, "rolled_back_automatically");
    assert!(f.log().iter().any(|l| l.contains("transformers==5.17.0")));
}

#[test]
fn a_failed_check_after_the_install_rolls_back_too() {
    let f = Fake::new("5.17.0");
    f.flag("break_import", "5.24.1");
    let r = run_upgrade(&f.u, true, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!((r["ok"].as_bool(), r["rollback_ok"].as_bool()), (Some(false), Some(true)), "{r}");
    assert!(r["text"].as_str().unwrap().contains("import transformers"), "{r}");
    assert_eq!(f.version(), "5.17.0");
}

#[test]
fn when_even_the_rollback_fails_the_text_gives_the_command() {
    let f = Fake::new("5.17.0");
    f.flag("fail_install", "");
    f.flag("fail_rollback", "");
    let r = run_upgrade(&f.u, true, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!((r["ok"].as_bool(), r["rollback_ok"].as_bool()), (Some(false), Some(false)));
    assert!(r["text"].as_str().unwrap().contains("uv pip install --python") && r["text"].as_str().unwrap().contains("transformers==5.17.0"));
    assert_eq!(read_record(&f.u.record).unwrap().state, "rollback_failed");
}

#[test]
fn cancel_stops_the_installer_and_restores_the_old_version() {
    let f = Fake::new("5.17.0");
    f.flag("slow", "");
    let cancel = AtomicBool::new(false);
    let t = std::time::Instant::now();
    let r = std::thread::scope(|s| {
        s.spawn(|| {
            // the real install has started once the record exists
            while !f.u.record.is_file() {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
            cancel.store(true, Ordering::SeqCst);
        });
        run_upgrade(&f.u, true, &idle, &cancel, &mut |_| {}).unwrap()
    });
    assert!(t.elapsed().as_secs() < 20, "the sleeping installer was killed");
    assert_eq!((r["stopped"].as_bool(), r["ok"].as_bool(), r["rolled_back"].as_bool(), r["rollback_ok"].as_bool()), (Some(true), Some(false), Some(true), Some(true)), "{r}");
    assert!(r["text"].as_str().unwrap().contains("cancelled"));
    assert_eq!(f.version(), "5.17.0");
}

#[test]
fn nothing_is_touched_while_a_job_a_queue_run_or_an_auto_run_is_active_or_without_confirmation() {
    let f = Fake::new("5.17.0");
    for a in [Activity { job_running: true, ..Activity::default() }, Activity { queue_running: Some("heal".into()), ..Activity::default() }, Activity { queue_waiting: 2, ..Activity::default() }] {
        let e = run_upgrade(&f.u, true, &|| a.clone(), &AtomicBool::new(false), &mut |_| {}).unwrap_err();
        assert!(e.0.contains("running") || e.0.contains("queue"), "{e}");
        assert!(check_start(true, &a).is_err());
        assert!(run_rollback(&f.u, true, "5.17.0", &|| a.clone(), &AtomicBool::new(false), &mut |_| {}).is_err());
    }
    assert!(refusal(&Activity { job_running: true, ..Activity::default() }).unwrap().contains("Auto run"));
    assert!(run_upgrade(&f.u, false, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("not confirmed"));
    assert!(check_start(true, &idle()).is_ok() && check_start(false, &idle()).is_err());
    assert!(f.log().is_empty(), "uv was never started: {:?}", f.log());
    assert!(!f.u.record.exists());
    // a job that starts between the preview and the install stops it before the first change
    let calls = std::cell::Cell::new(0);
    let late = || { calls.set(calls.get() + 1); if calls.get() >= 2 { Activity { job_running: true, ..Activity::default() } } else { Activity::default() } };
    assert!(run_upgrade(&f.u, true, &late, &AtomicBool::new(false), &mut |_| {}).is_err());
    assert_eq!(f.real_installs(), 0);
    assert!(!f.u.record.exists());
}

#[test]
fn an_install_that_would_change_torch_is_not_started() {
    let f = Fake::new("5.17.0");
    f.flag("torch_change", "");
    let r = run_upgrade(&f.u, true, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!((r["ok"].as_bool(), r["aborted"].as_bool()), (Some(false), Some(true)), "{r}");
    let t = r["text"].as_str().unwrap();
    assert!(t.contains("torch") && t.contains("2.8.0+cu128") && t.contains("2.9.0") && t.contains("Nothing was installed"), "{t}");
    assert_eq!(f.real_installs(), 0);
    assert_eq!(f.version(), "5.17.0");
    assert!(!f.u.record.exists(), "an aborted preview must not overwrite an older record");
}

#[test]
fn a_new_enough_or_missing_environment_is_refused_plainly() {
    let f = Fake::new("5.30.0");
    assert!(run_upgrade(&f.u, true, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("already"));
    assert_eq!(plan_view(&f.u)["needed"], false);
    let mut u = f.u.clone();
    u.py = f.d.path().join("nope/python");
    assert!(run_upgrade(&u, true, &idle, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("not installed"));
    assert_eq!(plan_view(&u)["ready"], false);
}

#[test]
fn the_plan_shows_what_will_be_done_and_the_resolver_list_is_read() {
    let f = Fake::new("5.17.0");
    let p = plan_view(&f.u);
    assert_eq!((p["target"].as_str(), p["current"].as_str(), p["needed"].as_bool(), p["ready"].as_bool()), (Some("transformers>=5.24.0,<6"), Some("5.17.0"), Some(true), Some(true)));
    assert_eq!(p["python"], json!(f.u.py));
    let text = p["will_do"].to_string();
    for w in ["Qwen and Gemma keep working", "program's own Python environment", "PyTorch and bitsandbytes", "put back by itself", "transformers>=5.24.0,<6"] {
        assert!(text.contains(w), "{w}: {text}");
    }
    let c = parse_changes([" - torch==2.8.0+cu128", " + torch==2.9.0", "Resolved 3 packages", " + Foo_Bar==1.0", " ~ same==1"]);
    assert_eq!(c.len(), 3);
    assert_eq!(guarded_change(&c).unwrap().name, "torch");
    assert!(guarded_change(&c[1..]).is_none());
}
