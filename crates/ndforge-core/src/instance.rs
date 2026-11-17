//! One ND Lore Forge at a time per settings folder. Two programs writing the same settings, projects and job logs would damage each other's files, so the program takes an
//! exclusive lock on `<settings folder>/instance.lock` as it starts and keeps it until it ends (the system lets go of it when the program dies, however it dies).
//! `instance.pid` beside it names the program that holds it (for messages); it is not what decides.
use std::fs::{File, OpenOptions, TryLockError};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The lock, held for as long as this value lives.
#[derive(Debug)]
pub struct Instance {
    _file: File,
    dir: PathBuf,
}

/// Why the lock could not be taken.
#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    /// another program holds it (its process number when known)
    Held(Option<u32>),
    /// the folder or file could not be used; the program then starts without a lock rather than not at all
    Unusable(String),
}

fn open(dir: &Path) -> std::io::Result<File> {
    std::fs::create_dir_all(dir)?;
    OpenOptions::new().create(true).truncate(false).read(true).write(true).open(dir.join("instance.lock"))
}

/// Takes the lock for the settings folder `dir`.
pub fn acquire(dir: &Path) -> Result<Instance, Refused> {
    let file = open(dir).map_err(|e| Refused::Unusable(e.to_string()))?;
    match file.try_lock() {
        Ok(()) => {
            if let Ok(mut f) = File::create(dir.join("instance.pid")) {
                let _ = write!(f, "{}", std::process::id());
            }
            Ok(Instance { _file: file, dir: dir.to_path_buf() })
        }
        Err(TryLockError::WouldBlock) => Err(Refused::Held(held_by(dir))),
        Err(TryLockError::Error(e)) => Err(Refused::Unusable(e.to_string())),
    }
}

/// The process number the holder wrote (it may be out of date after a crash; the lock is what counts).
pub fn held_by(dir: &Path) -> Option<u32> {
    std::fs::read_to_string(dir.join("instance.pid")).ok().and_then(|t| t.trim().parse().ok())
}

/// Is a program OTHER than this one holding the lock? (This program's own hold does not count.)
pub fn other_running(dir: &Path) -> bool {
    let Ok(file) = open(dir) else { return false };
    match file.try_lock() {
        Ok(()) => false,
        Err(TryLockError::WouldBlock) => held_by(dir) != Some(std::process::id()),
        Err(_) => false,
    }
}

impl Instance {
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}
