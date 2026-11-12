//! Run a command, hand its output to a callback line by line, and stop it when asked.
use std::io::{BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

/// No console window flashes up on Windows.
pub fn quiet(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    cmd
}

/// Starts the program. Linux answers "text file busy" (ETXTBSY) when the program file was written a moment ago while another thread of this program was
/// starting a process: the child it made inherits the still-open write handle until it executes. It goes away within milliseconds, so wait and retry.
fn spawn_retrying(cmd: &mut Command) -> std::io::Result<std::process::Child> {
    let mut tries = 0;
    loop {
        match cmd.spawn() {
            Err(e) if cfg!(unix) && e.raw_os_error() == Some(26) && tries < 40 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            other => return other,
        }
    }
}

/// Runs `cmd` to the end (or until `cancel` is set, which kills it). Both output streams arrive in `on_line`.
pub fn run_lines(mut cmd: Command, cancel: &AtomicBool, on_line: &mut dyn FnMut(Stream, &str)) -> Result<ExitStatus> {
    quiet(&mut cmd);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = spawn_retrying(&mut cmd).map_err(|e| Error::new(format!("Could not start {:?}: {e}", cmd.get_program())))?;
    let (tx, rx) = mpsc::channel::<(Stream, String)>();
    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(|l| l.ok()) {
                if tx.send((Stream::Out, line)).is_err() {
                    break;
                }
            }
        }));
    }
    if let Some(err) = child.stderr.take() {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(|l| l.ok()) {
                if tx.send((Stream::Err, line)).is_err() {
                    break;
                }
            }
        }));
    }
    drop(tx);
    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok((stream, line)) => on_line(stream, &line),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if cancel.load(Ordering::SeqCst) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Error::new("Cancelled."));
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    for r in readers {
        let _ = r.join();
    }
    child.wait().map_err(|e| Error::new(format!("Could not wait for the program: {e}")))
}
