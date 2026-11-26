//! Disk speed (`disk`, docs/TUNE.md): writes and reads back a few GB on the drive that holds the models, to tell how long a cut, a merge or an export will take. It decides nothing (no setting depends on it): the
//! numbers go into the preset and the report. The read-back comes straight after the write, so the system's file cache may serve some of it: the read figure is an upper bound, and the write figure includes the
//! flush to the drive (`sync_all`). The file is removed at the end, also after a failure or a stop.
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde_json::{json, Value};

use crate::error::{Error, Result};

pub const DEFAULT_MB: u64 = 2048;
const BLOCK: usize = 8 * 1024 * 1024;

/// The free space the test needs: the file and the same again as room.
pub fn need_bytes(mb: u64) -> u64 {
    mb * 1024 * 1024 * 2
}

struct Cleanup<'a>(&'a Path);

impl Drop for Cleanup<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

fn block() -> Vec<u8> {
    // not zeros and not repeating at the block size, so a drive or a file system that compresses cannot cheat
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..BLOCK).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x >> 24) as u8
    }).collect()
}

/// Writes `mb` megabytes into `dir`, reads them back and removes the file. `{"mb", "write_mb_s", "read_mb_s", "write_seconds", "read_seconds", "dir", "note"}`; Err("Stopped.") when `stop` is set.
pub fn probe(dir: &Path, mb: u64, stop: &AtomicBool) -> Result<Value> {
    std::fs::create_dir_all(dir)?;
    let free = fs_free(dir);
    if let Some(f) = free.filter(|f| *f < need_bytes(mb)) {
        return Err(Error::new(format!("The drive has {:.1} GB free and the test needs {:.1} GB.", f as f64 / 1e9, need_bytes(mb) as f64 / 1e9)));
    }
    let file = dir.join(".ndforge-tune-disk.tmp");
    let _clean = Cleanup(&file);
    let data = block();
    let blocks = (mb * 1024 * 1024) as usize / BLOCK;
    let t = Instant::now();
    {
        let mut f = std::fs::File::create(&file)?;
        for _ in 0..blocks {
            if stop.load(Ordering::SeqCst) {
                return Err(Error::new("Stopped."));
            }
            f.write_all(&data)?;
        }
        f.sync_all()?;
    }
    let w = t.elapsed().as_secs_f64().max(1e-6);
    let t = Instant::now();
    {
        let mut f = std::fs::File::open(&file)?;
        let mut buf = vec![0u8; BLOCK];
        let mut total = 0usize;
        loop {
            if stop.load(Ordering::SeqCst) {
                return Err(Error::new("Stopped."));
            }
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            total += n;
        }
        if total != blocks * BLOCK {
            return Err(Error::new("The file that was written could not be read back in full."));
        }
    }
    let r = t.elapsed().as_secs_f64().max(1e-6);
    let size_mb = (blocks * BLOCK) as f64 / 1_048_576.0;
    Ok(json!({"mb": size_mb, "write_mb_s": (size_mb / w).round(), "read_mb_s": (size_mb / r).round(), "write_seconds": (w * 100.0).round() / 100.0, "read_seconds": (r * 100.0).round() / 100.0, "dir": dir.display().to_string(),
              "note": "The read follows the write at once, so the system's file cache may have served part of it: the read figure is an upper bound."}))
}

/// Free bytes of the drive that holds `dir` (None when it cannot be read).
pub fn fs_free(dir: &Path) -> Option<u64> {
    fs2::available_space(dir).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_writes_reads_back_and_removes_its_file() {
        let d = tempfile::tempdir().unwrap();
        let r = probe(d.path(), 16, &AtomicBool::new(false)).unwrap();
        assert_eq!(r["mb"], json!(16.0));
        assert!(r["write_mb_s"].as_f64().unwrap() > 0.0 && r["read_mb_s"].as_f64().unwrap() > 0.0);
        assert!(!d.path().join(".ndforge-tune-disk.tmp").exists());
    }

    #[test]
    fn a_stop_ends_it_and_leaves_no_file() {
        let d = tempfile::tempdir().unwrap();
        let e = probe(d.path(), 64, &AtomicBool::new(true)).unwrap_err();
        assert!(e.0.contains("Stopped"));
        assert!(!d.path().join(".ndforge-tune-disk.tmp").exists());
    }
}
