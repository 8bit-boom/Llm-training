//! A saved log of every worker job: `logs/jobs/<YYYYmmdd-HHMMSS>-<job>.log` in the settings folder. It holds the job (without secrets), every line the worker wrote
//! (with the stream and the seconds since the start), and how the job ended, so a problem can be looked at after the window is closed. Tokens and keys are replaced
//! first. The newest 60 files are kept, and at most 300 MB. Writing the log never stops a job: every problem here is ignored.
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::Value;

use crate::paths::config_dir;

pub const KEEP_FILES: usize = 60;
pub const KEEP_BYTES: u64 = 300 * 1024 * 1024;
/// One log stops growing here (a training run of days would write very many progress lines).
const MAX_FILE_BYTES: u64 = 40 * 1024 * 1024;

pub fn dir() -> PathBuf {
    config_dir().join("logs").join("jobs")
}

pub struct JobLog {
    path: Option<PathBuf>,
    file: Option<File>,
    t0: Instant,
    written: u64,
    full: bool,
}

fn clean(text: &str) -> String {
    crate::secrets::redact(text)
}

impl JobLog {
    /// Opens the log of `job` (a worker job: {"job": "export", ...}) and writes the job into it.
    pub fn open(job: &Value) -> JobLog {
        let mut log = JobLog { path: None, file: None, t0: Instant::now(), written: 0, full: false };
        let d = dir();
        if std::fs::create_dir_all(&d).is_err() {
            return log;
        }
        prune(&d);
        let kind: String = job["job"].as_str().unwrap_or("job").chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).take(40).collect();
        let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
        for n in 0..100 {
            let name = if n == 0 { format!("{stamp}-{kind}.log") } else { format!("{stamp}-{kind}-{n}.log") };
            let path = d.join(&name);
            if let Ok(f) = std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                log.path = Some(path);
                log.file = Some(f);
                break;
            }
        }
        log.raw(&format!("# ND Lore Forge job log, started {} UTC\n# job: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"), clean(&job.to_string())));
        log
    }

    fn raw(&mut self, text: &str) {
        if self.full {
            return;
        }
        if let Some(f) = self.file.as_mut() {
            if self.written + text.len() as u64 > MAX_FILE_BYTES {
                let _ = f.write_all(b"# the log is full; later lines are not saved\n");
                self.full = true;
                return;
            }
            if f.write_all(text.as_bytes()).is_ok() {
                self.written += text.len() as u64;
            }
        }
    }

    /// One line of the worker; `stream` is "out", "err" or a tag of the program's own ("end").
    pub fn line(&mut self, stream: &str, text: &str) {
        let s = format!("[{:>8.1}s {stream}] {}\n", self.t0.elapsed().as_secs_f64(), clean(text));
        self.raw(&s);
    }

    /// How it ended: `status` text (e.g. "exit code 0", "cancelled") and the worker's error code, if it sent one.
    pub fn finish(&mut self, status: &str, code: &str, message: Option<&str>) {
        let mut s = format!("# ended after {:.1} s: {status}", self.t0.elapsed().as_secs_f64());
        if !code.is_empty() {
            s.push_str(&format!(", error code {code}"));
        }
        if let Some(m) = message {
            s.push_str(&format!("\n# error: {}", clean(m)));
        }
        s.push('\n');
        self.raw(&s);
        if let Some(f) = self.file.as_mut() {
            let _ = f.flush();
        }
    }

    /// `logs/jobs/<file>`, for the error text; None when no log could be made.
    pub fn short_name(&self) -> Option<String> {
        self.path.as_ref().and_then(|p| p.file_name()).map(|n| format!("logs/jobs/{}", n.to_string_lossy()))
    }

    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }
}

/// The saved logs, newest first (at most `n`).
pub fn latest(n: usize) -> Vec<PathBuf> {
    let mut files = files_oldest_first(&dir());
    files.reverse();
    files.truncate(n);
    files.into_iter().map(|(p, _)| p).collect()
}

fn files_oldest_first(d: &std::path::Path) -> Vec<(PathBuf, u64)> {
    let mut v: Vec<(PathBuf, u64)> = std::fs::read_dir(d).map(|r| r.flatten().filter(|e| e.path().extension().is_some_and(|x| x == "log")).filter_map(|e| Some((e.path(), e.metadata().ok()?.len()))).collect()).unwrap_or_default();
    v.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));        // the names start with the time
    v
}

/// Deletes the oldest logs until at most KEEP_FILES - 1 files (room for the one being opened) and KEEP_BYTES are left.
fn prune(d: &std::path::Path) {
    let mut files = files_oldest_first(d);
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    while !files.is_empty() && (files.len() >= KEEP_FILES || total > KEEP_BYTES) {
        let (p, size) = files.remove(0);
        if std::fs::remove_file(&p).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_keeps_the_newest_and_stays_under_the_size() {
        let d = tempfile::tempdir().unwrap();
        for i in 0..70 {
            std::fs::write(d.path().join(format!("20250101-0000{:02}-x.log", i)), "x").unwrap();
        }
        prune(d.path());
        let left = files_oldest_first(d.path());
        assert_eq!(left.len(), KEEP_FILES - 1);
        assert!(left[0].0.file_name().unwrap().to_string_lossy().contains("000011"), "the oldest went first");
    }
}
