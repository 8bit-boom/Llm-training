//! Run a command, hand its output to a callback line by line, and stop it when asked.
use std::io::{BufRead, BufReader, Read};
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

/// The program and everything it starts, so that a stop (or the end of this program) takes the whole family down, not only the first process: llama-server,
/// llama-quantize, llama-imatrix and the converter are children of the worker and would otherwise keep running (and keep the graphics card full).
/// Linux / macOS: the program gets its own process group and a stop kills the group. Windows: the program is put into a Job Object that kills everything in it
/// when the stop is asked for or when the handle closes (this program ends, however it ends).
struct ProcessTree {
    #[cfg(windows)]
    job: isize,
    #[cfg(unix)]
    pid: i32,
}

impl ProcessTree {
    /// Call right after the spawn. Failing is not fatal (a Job Object may be refused inside another job on an old Windows): the plain kill still works.
    fn attach(child: &std::process::Child) -> ProcessTree {
        #[cfg(unix)]
        {
            ProcessTree { pid: child.id() as i32 }
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::JobObjects::*;
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job == 0 {
                    eprintln!("note: no Job Object ({}); only the first program is stopped on Stop", std::io::Error::last_os_error());
                    return ProcessTree { job: 0 };
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info as *const _ as *const std::ffi::c_void, std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32);
                if ok == 0 || AssignProcessToJobObject(job, child.as_raw_handle() as isize) == 0 {
                    eprintln!("note: could not put the program into a Job Object ({}); only the first program is stopped on Stop", std::io::Error::last_os_error());
                    CloseHandle(job);
                    return ProcessTree { job: 0 };
                }
                ProcessTree { job }
            }
        }
    }

    /// Kills the program and what it started, then reaps the program.
    fn kill(&self, child: &mut std::process::Child) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-self.pid, libc::SIGKILL);
        }
        #[cfg(windows)]
        if self.job != 0 {
            unsafe {
                windows_sys::Win32::System::JobObjects::TerminateJobObject(self.job, 1);
            }
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(windows)]
impl Drop for ProcessTree {
    fn drop(&mut self) {
        if self.job != 0 {
            // KILL_ON_JOB_CLOSE: whatever is left of the program's family ends here
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.job);
            }
        }
    }
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

/// Reads lines from a pipe until it closes. Bytes that are not valid UTF-8 (a Windows console in cp1252, a binary tool) become U+FFFD instead of ending the
/// reading, which used to leave the rest of the output unread and the program stuck on a full pipe.
fn pump(src: impl Read, stream: Stream, tx: mpsc::Sender<(Stream, String)>) {
    let mut reader = BufReader::new(src);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let text = String::from_utf8_lossy(&buf);
                if tx.send((stream, text.trim_end_matches(['\r', '\n']).to_string())).is_err() {
                    break;
                }
            }
        }
    }
}

/// Runs `cmd` to the end (or until `cancel` is set, which kills it). Both output streams arrive in `on_line`.
pub fn run_lines(mut cmd: Command, cancel: &AtomicBool, on_line: &mut dyn FnMut(Stream, &str)) -> Result<ExitStatus> {
    quiet(&mut cmd);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let mut child = spawn_retrying(&mut cmd).map_err(|e| Error::new(format!("Could not start {:?}: {e}", cmd.get_program())))?;
    let tree = ProcessTree::attach(&child);
    let (tx, rx) = mpsc::channel::<(Stream, String)>();
    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || pump(out, Stream::Out, tx)));
    }
    if let Some(err) = child.stderr.take() {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || pump(err, Stream::Err, tx)));
    }
    drop(tx);
    loop {
        // A chatty program never lets the receive time out, so look at the stop request on every turn.
        if cancel.load(Ordering::SeqCst) {
            tree.kill(&mut child);
            return Err(Error::cancelled());
        }
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok((stream, line)) => on_line(stream, &line),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    for r in readers {
        let _ = r.join();
    }
    child.wait().map_err(|e| Error::new(format!("Could not wait for the program: {e}")))
}
