//! run_lines: bad bytes in the output, prompt cancel, whole-tree stop. Needs a python on PATH (skips with a note when there is none).
use ndforge_core::envmgr::process::{run_lines, Stream};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub fn python() -> Option<&'static str> {
    for name in ["python3", "python"] {
        if Command::new(name).arg("-c").arg("pass").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(name);
        }
    }
    eprintln!("note: no python on PATH, process test skipped");
    None
}

#[test]
fn bad_bytes_do_not_end_the_reading() {
    let Some(py) = python() else { return };
    let code = "import sys\nfor s in (sys.stdout, sys.stderr):\n    s.buffer.write(b'ok\\n\\xff\\xfe bad\\nafter\\n'); s.flush()\n";
    let mut cmd = Command::new(py);
    cmd.arg("-c").arg(code);
    let mut got: Vec<(Stream, String)> = Vec::new();
    let st = run_lines(cmd, &AtomicBool::new(false), &mut |s, l| got.push((s, l.to_string()))).unwrap();
    assert!(st.success());
    for stream in [Stream::Out, Stream::Err] {
        let lines: Vec<&str> = got.iter().filter(|(s, _)| *s == stream).map(|(_, l)| l.as_str()).collect();
        assert_eq!(lines.len(), 3, "{stream:?}: {lines:?}");
        assert_eq!(lines[0], "ok");
        assert!(lines[1].contains('\u{FFFD}') && lines[1].ends_with(" bad"), "{:?}", lines[1]);
        assert_eq!(lines[2], "after");
    }
}

#[test]
fn a_chatty_program_is_still_cancelled_promptly() {
    let Some(py) = python() else { return };
    let mut cmd = Command::new(py);
    cmd.arg("-c").arg("import time,sys\nwhile True:\n    print('x', flush=True)\n    time.sleep(0.005)\n");
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        c2.store(true, Ordering::SeqCst);
    });
    let started = Instant::now();
    let mut n = 0;
    let r = run_lines(cmd, &cancel, &mut |_, _| n += 1);
    t.join().unwrap();
    assert_eq!(r.unwrap_err().0, "Cancelled.");
    assert!(started.elapsed() < Duration::from_secs(2), "took {:?}", started.elapsed());
    assert!(n > 0);
}

#[test]
fn cancellation_is_one_value_with_an_unchanged_text() {
    let e = ndforge_core::Error::cancelled();
    assert_eq!(e.0, "Cancelled.");
    assert!(e.is_cancelled() && !ndforge_core::Error::new("Stopped.").is_cancelled());
    assert_eq!(ndforge_core::error::CANCELLED, "Cancelled.");
}

fn alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // signal 0 only asks whether the process exists; a zombie that nobody collected yet still counts as gone below
        let r = unsafe { libc::kill(pid as i32, 0) };
        if r != 0 {
            return false;
        }
        // a killed grandchild whose parent is gone is collected by init; wait until its state is not Z
        std::fs::read_to_string(format!("/proc/{pid}/stat")).map(|s| !s.rsplit(')').next().unwrap_or("").trim_start().starts_with('Z')).unwrap_or(true)
    }
    #[cfg(windows)]
    {
        let out = Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH"]).output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
        out.split_whitespace().any(|w| w == pid.to_string())
    }
}

#[test]
fn stopping_ends_the_programs_the_program_started() {
    let Some(py) = python() else { return };
    let code = "import subprocess, sys, time\nc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\nprint(c.pid, flush=True)\ntime.sleep(60)\n";
    let mut cmd = Command::new(py);
    cmd.arg("-c").arg(code);
    let cancel = Arc::new(AtomicBool::new(false));
    let grandchild = Arc::new(std::sync::Mutex::new(None::<u32>));
    let (c2, g2) = (cancel.clone(), grandchild.clone());
    let r = run_lines(cmd, &cancel, &mut |s, l| {
        if s == Stream::Out {
            if let Ok(pid) = l.trim().parse::<u32>() {
                *g2.lock().unwrap() = Some(pid);
                c2.store(true, Ordering::SeqCst);
            }
        }
    });
    assert_eq!(r.unwrap_err().0, "Cancelled.");
    let pid = grandchild.lock().unwrap().expect("the program printed the pid of what it started");
    let started = Instant::now();
    while alive(pid) && started.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!alive(pid), "the grandchild {pid} is still running");
}
