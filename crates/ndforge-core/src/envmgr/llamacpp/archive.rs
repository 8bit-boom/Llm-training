//! Opening the downloaded archives safely: every entry name is checked before anything is written (no `..`, no absolute path, no drive letter, nothing that
//! leaves the target folder), links are only made when they stay inside the tree, only the files a selector asks for are written, and the size is capped.
//! zip files and `.tar.gz` files (llama.cpp's newer Linux and macOS releases) are both read here.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use sha2::{Digest, Sha256};

use super::choose::Archive;
use crate::error::{Error, Result};

/// Never write more than this from one archive (a zip bomb).
pub const MAX_UNPACKED: u64 = 6 * 1024 * 1024 * 1024;

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// "sha256:ABC..." -> Some("abc..."); other algorithms and junk -> None.
pub fn parse_digest(d: &str) -> Option<String> {
    let h = d.trim().strip_prefix("sha256:")?.trim().to_lowercase();
    (h.len() == 64 && h.chars().all(|c| c.is_ascii_hexdigit())).then_some(h)
}

/// The path parts of an entry name, or None when the name is unsafe (`..`, absolute, drive letter, empty). Backslashes count as separators.
pub fn safe_parts(name: &str) -> Option<Vec<String>> {
    let n = name.replace('\\', "/");
    if n.starts_with('/') || n.contains('\0') {
        return None;
    }
    let mut parts = Vec::new();
    for p in n.split('/') {
        match p {
            "" | "." => {}
            ".." => return None,
            p if p.len() >= 2 && p.as_bytes()[1] == b':' && p.as_bytes()[0].is_ascii_alphabetic() => return None,
            p => parts.push(p.to_string()),
        }
    }
    (!parts.is_empty()).then_some(parts)
}

/// Does a link in `rel_parts` (its own path under the target folder) pointing to `target` stay inside the folder? (lexical check)
pub fn link_stays_inside(rel_parts: &[String], target: &str) -> bool {
    let t = target.replace('\\', "/");
    if t.starts_with('/') || t.is_empty() || (t.len() >= 2 && t.as_bytes()[1] == b':') {
        return false;
    }
    let mut depth = rel_parts.len() as i64 - 1; // the folder the link lives in
    for p in t.split('/') {
        match p {
            "" | "." => {}
            ".." => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => depth += 1,
        }
    }
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Dir,
    Symlink(String),
}

/// What the selector says about an entry: where to put it (relative to the target folder), or None to leave it out.
pub type Selector<'a> = &'a dyn Fn(&[String], &EntryKind) -> Option<Vec<String>>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Unpacked {
    pub files: usize,
    pub bytes: u64,
}

fn unsafe_entry(name: &str) -> Error {
    Error::new(format!("The downloaded file contains an unsafe path ({name}), so nothing was unpacked. It may be damaged; download it again."))
}

fn write_file(dest: &Path, rel: &[String], data: &mut dyn Read, mode: Option<u32>, cancel: &AtomicBool, budget: &mut u64) -> Result<u64> {
    let mut target = dest.to_path_buf();
    target.extend(rel);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::fs::File::create(&target)?;
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(Error::new("Cancelled."));
        }
        let n = data.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if (n as u64) > *budget {
            return Err(Error::new("The archive unpacks to far more data than it should; it was stopped. Download it again."));
        }
        *budget -= n as u64;
        out.write_all(&buf[..n])?;
        total += n as u64;
    }
    drop(out);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // the executable bit is kept (a program of llama.cpp); nothing else from the archive's permissions
        let m = if mode.unwrap_or(0o644) & 0o111 != 0 { 0o755 } else { 0o644 };
        let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(m));
    }
    let _ = mode;
    Ok(total)
}

fn make_link(dest: &Path, rel: &[String], target: &str) {
    #[cfg(unix)]
    {
        let mut p = dest.to_path_buf();
        p.extend(rel);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::remove_file(&p);
        let _ = std::os::unix::fs::symlink(target, &p);
    }
    #[cfg(not(unix))]
    {
        let _ = (dest, rel, target); // links are not needed on Windows
    }
}

fn zip_err(e: zip::result::ZipError) -> Error {
    Error::new(format!("The downloaded file is not a readable zip file ({e}). It may be incomplete; download it again."))
}

fn place(dest: &Path, parts: &[String], rel: Vec<String>, kind: EntryKind, data: &mut dyn Read, mode: Option<u32>, cancel: &AtomicBool, budget: &mut u64, out: &mut Unpacked) -> Result<u64> {
    match kind {
        EntryKind::Dir => {
            let mut p = dest.to_path_buf();
            p.extend(&rel);
            std::fs::create_dir_all(p)?;
            Ok(0)
        }
        EntryKind::Symlink(t) => {
            if !link_stays_inside(&rel, &t) {
                return Err(Error::new(format!("The downloaded file contains a link that points outside its folder ({} -> {t}); nothing was unpacked.", parts.join("/"))));
            }
            make_link(dest, &rel, &t);
            Ok(0)
        }
        EntryKind::File => {
            let n = write_file(dest, &rel, data, mode, cancel, budget)?;
            out.bytes += n;
            out.files += 1;
            Ok(n)
        }
    }
}

/// Unpacks the entries `select` wants into `dest`. Every entry name is checked, selected or not.
pub fn unpack(archive: &Path, kind: Archive, dest: &Path, select: Selector, cancel: &AtomicBool) -> Result<Unpacked> {
    std::fs::create_dir_all(dest)?;
    match kind {
        Archive::Zip => unpack_zip(archive, dest, select, cancel),
        Archive::TarGz => unpack_tar_gz(archive, dest, select, cancel),
    }
}

fn unpack_zip(archive: &Path, dest: &Path, select: Selector, cancel: &AtomicBool) -> Result<Unpacked> {
    let mut z = zip::ZipArchive::new(std::fs::File::open(archive)?).map_err(zip_err)?;
    for i in 0..z.len() {
        let f = z.by_index(i).map_err(zip_err)?;
        if safe_parts(f.name()).is_none() && !f.name().trim_matches('/').is_empty() {
            return Err(unsafe_entry(f.name()));
        }
    }
    let (mut out, mut budget) = (Unpacked::default(), MAX_UNPACKED);
    for i in 0..z.len() {
        if cancel.load(Ordering::SeqCst) {
            return Err(Error::new("Cancelled."));
        }
        let mut f = z.by_index(i).map_err(zip_err)?;
        let Some(parts) = safe_parts(f.name()) else { continue };
        let mode = f.unix_mode();
        let is_link = mode.is_some_and(|m| m & 0o170000 == 0o120000);
        let kind = if f.is_dir() {
            EntryKind::Dir
        } else if is_link {
            let mut t = String::new();
            f.by_ref().take(4096).read_to_string(&mut t)?;
            EntryKind::Symlink(t)
        } else {
            EntryKind::File
        };
        let Some(rel) = select(&parts, &kind) else { continue };
        place(dest, &parts, rel, kind, &mut f, mode, cancel, &mut budget, &mut out)?;
    }
    Ok(out)
}

fn tar_str(b: &[u8]) -> String {
    let end = b.iter().position(|c| *c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).to_string()
}

fn tar_size(b: &[u8]) -> Result<u64> {
    let s = tar_str(b);
    let s = s.trim();
    if s.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(s, 8).map_err(|_| Error::new("The downloaded file is not a readable archive (bad size in a header). Download it again."))
}

fn pax_value(data: &[u8], key: &str) -> Option<String> {
    let text = String::from_utf8_lossy(data);
    text.lines().find_map(|l| l.split_once(' ').and_then(|(_, kv)| kv.strip_prefix(&format!("{key}="))).map(str::to_string))
}

fn read_tar_header(r: &mut dyn Read) -> Result<Option<[u8; 512]>> {
    let mut h = [0u8; 512];
    let mut got = 0;
    while got < 512 {
        let n = r.read(&mut h[got..])?;
        if n == 0 {
            return if got == 0 { Ok(None) } else { Err(Error::new("The downloaded archive ends in the middle of a header. It is incomplete; download it again.")) };
        }
        got += n;
    }
    Ok(if h.iter().all(|b| *b == 0) { None } else { Some(h) })
}

fn skip(r: &mut dyn Read, n: u64) -> Result<()> {
    let copied = std::io::copy(&mut r.take(n), &mut std::io::sink())?;
    if copied != n {
        return Err(Error::new("The downloaded archive is incomplete; download it again."));
    }
    Ok(())
}

fn unpack_tar_gz(archive: &Path, dest: &Path, select: Selector, cancel: &AtomicBool) -> Result<Unpacked> {
    let mut r: Box<dyn Read> = Box::new(flate2::read::GzDecoder::new(std::io::BufReader::new(std::fs::File::open(archive)?)));
    let (mut out, mut budget) = (Unpacked::default(), MAX_UNPACKED);
    let (mut long_name, mut long_link, mut pax_path): (Option<String>, Option<String>, Option<String>) = (None, None, None);
    while let Some(h) = read_tar_header(&mut *r)? {
        if cancel.load(Ordering::SeqCst) {
            return Err(Error::new("Cancelled."));
        }
        let mut name = tar_str(&h[0..100]);
        let typeflag = h[156];
        let size = tar_size(&h[124..136])?;
        let mode = u32::from_str_radix(tar_str(&h[100..108]).trim(), 8).unwrap_or(0o644);
        let mut link = tar_str(&h[157..257]);
        if &h[257..262] == b"ustar" {
            let prefix = tar_str(&h[345..500]);
            if !prefix.is_empty() {
                name = format!("{prefix}/{name}");
            }
        }
        let padded = size.div_ceil(512) * 512;
        if matches!(typeflag, b'L' | b'K' | b'x' | b'g') {
            if size > 1 << 20 {
                return Err(Error::new("The downloaded archive has an oversized header; it was not unpacked."));
            }
            let mut buf = vec![0u8; size as usize];
            r.read_exact(&mut buf).map_err(|_| Error::new("The downloaded archive is incomplete; download it again."))?;
            skip(&mut *r, padded - size)?;
            match typeflag {
                b'L' => long_name = Some(tar_str(&buf)),
                b'K' => long_link = Some(tar_str(&buf)),
                b'x' => {
                    pax_path = pax_value(&buf, "path").or(pax_path);
                    if let Some(l) = pax_value(&buf, "linkpath") {
                        long_link = Some(l);
                    }
                }
                _ => {}
            }
            continue;
        }
        if let Some(n) = long_name.take().or(pax_path.take()) {
            name = n;
        }
        if let Some(l) = long_link.take() {
            link = l;
        }
        let parts = match safe_parts(&name) {
            Some(p) => p,
            None if name.trim_matches('/').trim_matches('.').is_empty() => {
                skip(&mut *r, padded)?;
                continue;
            }
            None => return Err(unsafe_entry(&name)),
        };
        let kind = match typeflag {
            b'5' => Some(EntryKind::Dir),
            b'0' | 0 | b'7' => Some(EntryKind::File),
            b'2' => Some(EntryKind::Symlink(link)),
            b'1' => return Err(Error::new(format!("The downloaded archive contains a hard link ({name}); it was not unpacked."))),
            _ => None, // devices, fifos: never made
        };
        let Some(kind) = kind else {
            skip(&mut *r, padded)?;
            continue;
        };
        let Some(rel) = select(&parts, &kind) else {
            skip(&mut *r, padded)?;
            continue;
        };
        let is_file = kind == EntryKind::File;
        let wrote = place(dest, &parts, rel, kind, &mut (&mut *r).take(size), Some(mode), cancel, &mut budget, &mut out)?;
        if is_file {
            if wrote != size {
                return Err(Error::new("The downloaded archive is incomplete; download it again."));
            }
            skip(&mut *r, padded - size)?;
        } else {
            skip(&mut *r, padded)?;
        }
    }
    Ok(out)
}

/// The shallowest folder under `root` (itself included) that holds a file called `file_name`.
pub fn find_dir_with(root: &Path, file_name: &str) -> Option<PathBuf> {
    fn walk(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
        if dir.join(name).is_file() {
            return Some(dir.to_path_buf());
        }
        if depth == 0 {
            return None;
        }
        let mut subs: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        subs.sort();
        subs.into_iter().find_map(|s| walk(&s, name, depth - 1))
    }
    walk(root, file_name, 4)
}
