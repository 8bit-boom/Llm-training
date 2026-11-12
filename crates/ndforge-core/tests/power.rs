use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ndforge_core::power::{countdown, shutdown_now, KeepAwake};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

#[test]
fn the_countdown_runs_out_or_is_cancelled_and_reports_every_second() {
    let mut ticks = Vec::new();
    assert!(countdown(3, Duration::from_millis(20), &AtomicBool::new(false), &mut |l| ticks.push(l)));
    assert_eq!(ticks, [3, 2, 1, 0]);
    // cancelled from outside after a moment: it stops early and says so
    let cancel = std::sync::Arc::new(AtomicBool::new(false));
    let c = cancel.clone();
    std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(70)); c.store(true, Ordering::SeqCst); });
    let t = Instant::now();
    let mut seen = Vec::new();
    assert!(!countdown(60, Duration::from_millis(40), &cancel, &mut |l| seen.push(l)));
    assert!(t.elapsed() < Duration::from_secs(2) && seen.len() < 6, "{seen:?}");
    // already cancelled: nothing runs out
    assert!(!countdown(5, Duration::from_millis(10), &AtomicBool::new(true), &mut |_| {}));
}

#[cfg(unix)]
#[test]
fn shutdown_runs_the_systems_command_and_a_failure_is_explained() {
    use std::os::unix::fs::PermissionsExt;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // the test never switches anything off: the command is replaced by a script that only leaves a note
    let ok = d.path().join("ok.sh");
    std::fs::write(&ok, format!("#!/bin/sh\necho called > {}\n", d.path().join("called.txt").display())).unwrap();
    std::fs::set_permissions(&ok, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_SHUTDOWN_CMD", ok.display().to_string());
    shutdown_now().unwrap();
    assert_eq!(std::fs::read_to_string(d.path().join("called.txt")).unwrap().trim(), "called");
    let bad = d.path().join("bad.sh");
    std::fs::write(&bad, "#!/bin/sh\necho 'not allowed' >&2\nexit 1\n").unwrap();
    std::fs::set_permissions(&bad, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_SHUTDOWN_CMD", bad.display().to_string());
    let e = shutdown_now().unwrap_err();
    assert!(e.0.contains("not allowed") && e.0.contains("Switch it off yourself"), "{e}");
    std::env::set_var("NDFORGE_SHUTDOWN_CMD", d.path().join("missing").display().to_string());
    assert!(shutdown_now().is_err());
    std::env::remove_var("NDFORGE_SHUTDOWN_CMD");
}

#[cfg(target_os = "linux")]
#[test]
fn keeping_awake_holds_an_inhibitor_until_it_is_dropped_and_says_so_when_it_cannot() {
    use std::os::unix::fs::PermissionsExt;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // a stand-in for systemd-inhibit: notes its pid and waits
    let bin = d.path().join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let fake = bin.join("systemd-inhibit");
    std::fs::write(&fake, format!("#!/bin/sh\necho $$ > {}\nexec sleep 600\n", d.path().join("pid.txt").display())).unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let old = std::env::var("PATH").unwrap_or_default();
    std::env::set_var("PATH", format!("{}:{old}", bin.display()));
    let awake = KeepAwake::start();
    assert!(awake.active && awake.note.contains("not go to sleep"), "{}", awake.note);
    let pid_file = d.path().join("pid.txt");
    for _ in 0..100 {
        if pid_file.exists() { break; }
        std::thread::sleep(Duration::from_millis(20));
    }
    let pid: i32 = std::fs::read_to_string(&pid_file).unwrap().trim().parse().unwrap();
    assert!(std::path::Path::new(&format!("/proc/{pid}")).exists(), "the inhibitor runs while the guard lives");
    drop(awake);
    std::thread::sleep(Duration::from_millis(200));
    assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists() || std::fs::read_to_string(format!("/proc/{pid}/stat")).map(|s| s.contains(" Z ")).unwrap_or(true), "it is let go on drop");
    // nothing to hold it with: a plain note, no pretence
    std::env::set_var("PATH", d.path().join("empty").display().to_string());
    let none = KeepAwake::start();
    assert!(!none.active && none.note.contains("Could not keep the computer awake"), "{}", none.note);
    std::env::set_var("PATH", old);
    // the "off" guard does nothing
    let off = KeepAwake::off();
    assert!(!off.active && off.note.is_empty());
}

#[test]
fn keep_awake_is_a_remembered_setting_that_defaults_to_on() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects"));
    let mut m = AppModel::load();
    assert_eq!(m.state()["settings"]["keep_awake"], true);
    m.set_setting("keep_awake", false).unwrap();
    assert_eq!(m.state()["settings"]["keep_awake"], false);
    assert_eq!(AppModel::load().state()["settings"]["keep_awake"], false, "remembered");
    // an older settings file without the field still means "on"
    std::fs::write(d.path().join("cfg/settings.json"), r#"{"expert_mode": true}"#).unwrap();
    assert_eq!(AppModel::load().state()["settings"]["keep_awake"], true);
}
