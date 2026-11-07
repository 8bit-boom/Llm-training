//! Zipping a folder (the trained adapter) and unzipping one safely (no entry may leave the target folder).
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};

fn zerr(e: zip::result::ZipError) -> Error {
    Error::new(format!("Zip problem: {e}"))
}

pub fn zip_dir(src: &Path, dest: &Path) -> Result<()> {
    let mut w = zip::ZipWriter::new(std::fs::File::create(dest)?);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated).unix_permissions(0o644);
    fn walk(dir: &Path, base: &Path, w: &mut zip::ZipWriter<std::fs::File>, opts: zip::write::SimpleFileOptions) -> Result<()> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let path = e.path();
            if path.is_dir() {
                walk(&path, base, w, opts)?;
            } else {
                let rel = path.strip_prefix(base).expect("under base").to_string_lossy().replace('\\', "/");
                w.start_file(rel, opts).map_err(zerr)?;
                let mut f = std::fs::File::open(&path)?;
                std::io::copy(&mut f, w)?;
            }
        }
        Ok(())
    }
    walk(src, src, &mut w, opts)?;
    w.finish().map_err(zerr)?;
    Ok(())
}

/// A zip entry name as a path under `dest`, or None when it would escape (`..`, absolute paths, drive letters).
fn safe_path(dest: &Path, name: &str) -> Option<PathBuf> {
    let p = Path::new(name);
    let mut out = dest.to_path_buf();
    for c in p.components() {
        match c {
            Component::Normal(x) => out.push(x),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (out != dest).then_some(out)
}

/// Unzips into `dest` (created). An entry that would leave `dest` makes the whole extraction fail.
pub fn extract_safe(zip_path: &Path, dest: &Path) -> Result<usize> {
    let mut z = zip::ZipArchive::new(std::fs::File::open(zip_path)?).map_err(zerr)?;
    std::fs::create_dir_all(dest)?;
    let mut n = 0;
    for i in 0..z.len() {
        let mut f = z.by_index(i).map_err(zerr)?;
        let name = f.name().to_string();
        let Some(target) = safe_path(dest, &name) else { return Err(Error::new(format!("The downloaded file contains an unsafe path ({name}); it was not unpacked."))) };
        if f.is_dir() {
            std::fs::create_dir_all(&target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        std::fs::File::create(&target)?.write_all(&buf)?;
        n += 1;
    }
    Ok(n)
}
