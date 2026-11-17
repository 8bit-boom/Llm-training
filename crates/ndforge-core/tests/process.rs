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
