//! Starting the installer. Nothing here runs without `confirm: true` and an idle program.
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::net::fail;
use super::{Config, Downloaded, Failure};

/// What the program is doing right now (filled in by the window's side; the updater only decides).
#[derive(Debug, Clone, Default)]
pub struct Activity {
    /// a job (training, heal, export, Auto run ...) is running
    pub job_running: bool,
    /// the label of the queue entry that runs, if any
    pub queue_running: Option<String>,
    /// queue entries that wait and are not paused (they would start by themselves)
    pub queue_waiting: usize,
}

/// Why now is not the time to update, in plain words; `None` = idle.
pub fn refusal(a: &Activity) -> Option<String> {
    if let Some(label) = &a.queue_running {
        return Some(format!("A queue run is active ({label}). Wait until it has finished or stop it, then update; the program closes during the update and would cut it off."));
    }
    if a.job_running {
        return Some("A job is running (training, a heal, an export, a quant check or an Auto run). Wait until it has finished or stop it, then update; the program closes during the update and would cut it off.".into());
    }
    if a.queue_waiting > 0 {
        return Some(format!("The queue has {} waiting job(s) that would start by themselves. Pause or clear the queue, or let it finish, then update.", a.queue_waiting));
    }
    None
}

/// The system calls, replaceable in tests.
pub trait Launcher {
    /// Windows: start the Inno Setup installer detached, so that it goes on when this program ends.
    fn run_installer(&self, installer: &Path, args: &[&str]) -> std::io::Result<()>;
    /// Linux: show a folder in the file manager (best effort).
    fn show_folder(&self, folder: &Path) -> std::io::Result<()>;
}

pub struct SystemLauncher;

impl Launcher for SystemLauncher {
    fn run_installer(&self, installer: &Path, args: &[&str]) -> std::io::Result<()> {
        let mut c = std::process::Command::new(installer);
        c.args(args);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            c.creation_flags(0x0000_0008 | 0x0000_0200); // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        }
        c.stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn().map(|_| ())
    }
    fn show_folder(&self, folder: &Path) -> std::io::Result<()> {
        std::process::Command::new("xdg-open").arg(folder).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn().map(|_| ())
    }
}

/// The Inno Setup flags of a silent upgrade in place (same AppId in packaging/windows/ndforge.iss, so the old version is replaced; the running program is closed and started again).
pub const INNO_ARGS: [&str; 4] = ["/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS", "/NORESTART"];

/// Starts the update. `confirm` is the person's click and must be true. Returns what the window should do: `restart` (the installer runs; the program must end now) or
/// `show_file` (Linux: the file is in `folder`, with instructions).
pub fn install(cfg: &Config, dl: &Downloaded, confirm: bool, activity: &Activity, launcher: &dyn Launcher) -> Result<Value, Failure> {
    if !confirm {
        return Err(fail("not_confirmed", "The update was not confirmed, so nothing was installed."));
    }
    if let Some(why) = refusal(activity) {
        return Err(fail("busy", why));
    }
    if !dl.path.is_file() {
        return Err(fail("missing", "The downloaded file is gone, so nothing was installed. Download it again."));
    }
    match cfg.os.as_str() {
        "windows" => {
            launcher.run_installer(&dl.path, &INNO_ARGS).map_err(|e| fail("launch", format!("The installer could not be started ({e}). The current version is unchanged; the file is {}.", dl.path.display())))?;
            Ok(json!({"action": "restart", "path": dl.path.display().to_string(), "text": "The installer is running. ND Lore Forge closes now and starts again when the update is done."}))
        }
        "linux" => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&dl.path, std::fs::Permissions::from_mode(0o755));
            }
            let folder: PathBuf = dl.path.parent().map(Path::to_path_buf).unwrap_or_default();
            let opened = launcher.show_folder(&folder).is_ok();
            let name = dl.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            Ok(json!({"action": "show_file", "path": dl.path.display().to_string(), "folder": folder.display().to_string(), "opened": opened,
                "text": format!("The new version is saved as {name} in {}. The program does not replace itself on Linux: close ND Lore Forge, then start the new file instead of the old one (chmod +x it if needed). Keep the old file until the new one works; that is also how you go back.", folder.display())}))
        }
        other => Err(fail("unsupported_os", format!("There is no installer for this system ({other}); the program cannot update itself here."))),
    }
}
