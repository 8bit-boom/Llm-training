//! Power: keep the computer awake while it trains, and switch it off when the training is finished.
//!
//! * `KeepAwake` asks the system not to go to sleep for as long as the guard lives (Windows: `SetThreadExecutionState` from a thread of its own;
//!   macOS: `caffeinate`; Linux: `systemd-inhibit`). The screen may still turn off. Where nothing is available it says so instead of pretending.
//! * `countdown` + `shutdown_now`: a few seconds of warning that can be cancelled, then the system's own shutdown command.
//!   `NDFORGE_SHUTDOWN_CMD` replaces that command (tests use it so that nothing real is ever switched off).
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::error::{Error, Result};

#[cfg(windows)]
extern "system" {
    fn SetThreadExecutionState(flags: u32) -> u32;
}

/// Held while the computer must not sleep; letting go (drop) gives the computer its normal sleep back.
pub struct KeepAwake {
    child: Option<Child>,
    stop: Option<Arc<AtomicBool>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// what is keeping it awake, or why nothing is (plain words for the log)
    pub note: String,
    pub active: bool,
}

impl KeepAwake {
    pub fn start() -> KeepAwake {
        #[cfg(windows)]
        {
            const ES_CONTINUOUS: u32 = 0x8000_0000;
            const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
            // the setting belongs to the thread that made it: a thread of its own holds it until told to stop
            let stop = Arc::new(AtomicBool::new(false));
            let s = stop.clone();
            let thread = std::thread::spawn(move || {
                unsafe { SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) };
                while !s.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(200));
                }
                unsafe { SetThreadExecutionState(ES_CONTINUOUS) };
            });
            return KeepAwake { child: None, stop: Some(stop), thread: Some(thread), note: "The computer will not go to sleep while training (the screen may still turn off).".into(), active: true };
        }
        #[cfg(not(windows))]
        {
            let (prog, args): (&str, Vec<&str>) = if cfg!(target_os = "macos") {
                ("caffeinate", vec!["-i", "-s"])
            } else {
                ("systemd-inhibit", vec!["--what=sleep:idle", "--who=ND Lore Forge", "--why=Training is running", "--mode=block", "sleep", "2147483647"])
            };
            match Command::new(prog).args(&args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
                Ok(child) => KeepAwake { child: Some(child), stop: None, thread: None, note: "The computer will not go to sleep while training (the screen may still turn off).".into(), active: true },
                Err(e) => KeepAwake { child: None, stop: None, thread: None, note: format!("Could not keep the computer awake ({prog}: {e}). Switch off sleep in the system settings for long runs."), active: false },
            }
        }
    }

    /// A guard that does nothing (the box is not ticked).
    pub fn off() -> KeepAwake {
        KeepAwake { child: None, stop: None, thread: None, note: String::new(), active: false }
    }
}

impl Drop for KeepAwake {
    fn drop(&mut self) {
        if let Some(s) = &self.stop {
            s.store(true, Ordering::SeqCst);
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        if let Some(mut c) = self.child.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Counts down `seconds`, calling `on_tick(seconds left)` once a second (`step` is that second; tests make it short). Returns true when it ran out,
/// false when `cancel` was set first.
pub fn countdown(seconds: u32, step: Duration, cancel: &AtomicBool, on_tick: &mut dyn FnMut(u32)) -> bool {
    let mut left = seconds;
    on_tick(left);
    while left > 0 {
        let until = std::time::Instant::now() + step;
        while std::time::Instant::now() < until {
            if cancel.load(Ordering::SeqCst) {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10).min(step));
        }
        left -= 1;
        on_tick(left);
    }
    !cancel.load(Ordering::SeqCst)
}

fn shutdown_commands() -> Vec<Vec<String>> {
    let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    if let Ok(custom) = std::env::var("NDFORGE_SHUTDOWN_CMD") {
        return vec![custom.split_whitespace().map(String::from).collect()];
    }
    if cfg!(windows) {
        vec![v(&["shutdown", "/s", "/t", "0"])]
    } else if cfg!(target_os = "macos") {
        vec![v(&["osascript", "-e", "tell application \"System Events\" to shut down"])]
    } else {
        vec![v(&["systemctl", "poweroff"]), v(&["shutdown", "-h", "now"])]
    }
}

/// Switches the computer off with the system's own command. Err says why it did not work.
pub fn shutdown_now() -> Result<()> {
    let mut last = String::from("no shutdown command is available");
    for cmd in shutdown_commands() {
        let Some((prog, args)) = cmd.split_first() else { continue };
        match Command::new(prog).args(args).output() {
            Ok(o) if o.status.success() => return Ok(()),
            Ok(o) => last = format!("{prog} failed: {}", String::from_utf8_lossy(&o.stderr).trim()),
            Err(e) => last = format!("{prog}: {e}"),
        }
    }
    Err(Error::new(format!("The computer could not be switched off ({last}). Switch it off yourself.")))
}
