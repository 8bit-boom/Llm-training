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
        Self::start_with(false)
    }

    /// `display`: keep the screen on too (Windows: ES_DISPLAY_REQUIRED). Away mode is never used.
    #[allow(unused_variables)]
    pub fn start_with(display: bool) -> KeepAwake {
        #[cfg(windows)]
        {
            const ES_CONTINUOUS: u32 = 0x8000_0000;
            const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
            const ES_DISPLAY_REQUIRED: u32 = 0x0000_0002;
            let flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | if display { ES_DISPLAY_REQUIRED } else { 0 };
            // the setting belongs to the thread that made it: a thread of its own holds it until told to stop
            let stop = Arc::new(AtomicBool::new(false));
            let s = stop.clone();
            let thread = std::thread::spawn(move || {
                unsafe { SetThreadExecutionState(flags) };
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


/// What the active power plan does when the computer is idle on mains power, read through the power API (never `powercfg` text: it is translated).
/// A time of 0 seconds means "never".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SleepFacts {
    pub ac_standby_s: Option<u32>,
    pub ac_hibernate_s: Option<u32>,
    pub ac_unattend_s: Option<u32>,
    /// the computer uses Modern Standby (S0 low power idle)
    pub modern_standby: Option<bool>,
}

/// The sleep facts of this computer: Windows only (None elsewhere, or when nothing could be read). Cached for a minute.
pub fn sleep_facts() -> Option<SleepFacts> {
    #[cfg(windows)]
    {
        use std::sync::Mutex;
        use std::time::Instant;
        static CACHE: Mutex<Option<(Instant, Option<SleepFacts>)>> = Mutex::new(None);
        let mut g = CACHE.lock().ok()?;
        if let Some((at, v)) = *g {
            if at.elapsed() < Duration::from_secs(60) {
                return v;
            }
        }
        let v = windows_sleep_facts();
        *g = Some((Instant::now(), v));
        v
    }
    #[cfg(not(windows))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_sleep_facts() -> Option<SleepFacts> {
    use windows_sys::core::GUID;
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::System::Power::{CallNtPowerInformation, PowerGetActiveScheme, PowerReadACValueIndex, SystemPowerCapabilities, SYSTEM_POWER_CAPABILITIES};
    // GUID_SLEEP_SUBGROUP and the three settings (docs: Power setting GUIDs)
    const SUB_SLEEP: GUID = GUID { data1: 0x238C9FA8, data2: 0x0AAD, data3: 0x41ED, data4: [0x83, 0xF4, 0x97, 0xBE, 0x24, 0x2C, 0x8F, 0x20] };
    const STANDBYIDLE: GUID = GUID { data1: 0x29F6C1DB, data2: 0x86DA, data3: 0x48C5, data4: [0x9F, 0xDB, 0xF2, 0xB6, 0x7B, 0x1F, 0x44, 0xDA] };
    const HIBERNATEIDLE: GUID = GUID { data1: 0x9D7815A6, data2: 0x7EE4, data3: 0x497E, data4: [0x88, 0x88, 0x51, 0x5A, 0x05, 0xF0, 0x23, 0x64] };
    const UNATTENDSLEEP: GUID = GUID { data1: 0x7BC4A2F9, data2: 0xD8FC, data3: 0x4469, data4: [0xB0, 0x7B, 0x33, 0xEB, 0x78, 0x5A, 0xAC, 0xA0] };
    let mut facts = SleepFacts::default();
    let mut any = false;
    unsafe {
        let mut scheme: *mut GUID = std::ptr::null_mut();
        if PowerGetActiveScheme(0, &mut scheme) == 0 && !scheme.is_null() {
            let read = |setting: &GUID| -> Option<u32> {
                let mut v = 0u32;
                (PowerReadACValueIndex(0, scheme, &SUB_SLEEP, setting, &mut v) == 0).then_some(v)
            };
            facts.ac_standby_s = read(&STANDBYIDLE);
            facts.ac_hibernate_s = read(&HIBERNATEIDLE);
            facts.ac_unattend_s = read(&UNATTENDSLEEP);
            any = true;
            LocalFree(scheme as _);
        }
        let mut caps: SYSTEM_POWER_CAPABILITIES = std::mem::zeroed();
        let status = CallNtPowerInformation(SystemPowerCapabilities, std::ptr::null(), 0, &mut caps as *mut _ as *mut std::ffi::c_void, std::mem::size_of::<SYSTEM_POWER_CAPABILITIES>() as u32);
        if status == 0 {
            facts.modern_standby = Some(caps.AoAc != 0);
            any = true;
        }
    }
    any.then_some(facts)
}
