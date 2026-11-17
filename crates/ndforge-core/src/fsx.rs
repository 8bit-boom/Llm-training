//! Writing files so that a crash or a full disk never leaves half a file behind: write `<name>.tmp` next to it, flush it to disk, then rename it over the old one.
//! On Windows a rename can fail for a moment (a virus scanner or indexer holds the file), so it is tried again a few times.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;

use crate::error::{Error, Result};

/// `<dir>/<name>.tmp`: the name of the file being written before it replaces `path`.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Renames, and tries again (10 times, 200 ms apart) when Windows says "access denied" or "in use". Elsewhere it is a plain rename.
pub fn rename_retry(from: &Path, to: &Path) -> std::io::Result<()> {
    rename_retry_with(from, to, 10, Duration::from_millis(200))
}

pub fn rename_retry_with(from: &Path, to: &Path, tries: u32, pause: Duration) -> std::io::Result<()> {
    let mut n = 0;
    loop {
        match std::fs::rename(from, to) {
            Err(e) if cfg!(windows) && n + 1 < tries && (e.kind() == std::io::ErrorKind::PermissionDenied || matches!(e.raw_os_error(), Some(5) | Some(32))) => {
                n += 1;
                std::thread::sleep(pause);
            }
            other => return other,
        }
    }
}

fn write_tmp(tmp: &Path, bytes: &[u8], private: bool) -> std::io::Result<()> {
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let _ = private;
    let mut f = opts.open(tmp)?;
    f.write_all(bytes)?;
    f.sync_all()
}

fn write_atomic_inner(path: &Path, bytes: &[u8], private: bool) -> Result<()> {
    if let Some(d) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d)?;
    }
    let tmp = tmp_path(path);
    let r = write_tmp(&tmp, bytes, private).and_then(|_| rename_retry(&tmp, path));
    if let Err(e) = r {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::from(e));
    }
    Ok(())
}

/// The file is either the old content or the new one, never half of each.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic_inner(path, bytes, false)
}

/// Like `write_atomic`; on Linux and macOS the file is readable by its owner only (mode 0600) from the first byte on.
pub fn write_atomic_private(path: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic_inner(path, bytes, true)
}

/// Pretty JSON, written atomically.
pub fn write_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    write_atomic(path, &serde_json::to_vec_pretty(value)?)
}

/// A file that cannot be read any more is kept under another name (`<name>.broken-<UTC time>`) instead of being overwritten by defaults.
/// Returns the new path, or None when the file could not be moved.
pub fn set_aside_broken(path: &Path) -> Option<PathBuf> {
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let name = path.file_name()?.to_string_lossy().to_string();
    for n in 0..100 {
        let suffix = if n == 0 { String::new() } else { format!("-{n}") };
        let to = path.with_file_name(format!("{name}.broken-{stamp}{suffix}"));
        if !to.exists() {
            return std::fs::rename(path, &to).ok().map(|_| to);
        }
    }
    None
}
