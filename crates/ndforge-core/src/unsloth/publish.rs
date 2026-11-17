//! Putting an export into Unsloth Studio's share (docs/UNSLOTH_API_CONTRACT.md, "Publish"; docs/AUTO_CUT.md).
//!
//! Studio runs in Docker on another computer and keeps its files on a share. Auto-cut copies the GGUF files into `<share>/exports/<model>/` ("the share" as THIS computer
//! sees it) and says where they are INSIDE Studio's container. Nothing here reads Unsloth's own files: the container path comes from the settings, or from `outputs_dir` of
//! `GET /api/models/checkpoints` (the parent of Studio's outputs folder), and is confirmed by finding the run folders Studio lists under `<share>/outputs`.
//!
//! A copy is made in `<model>.partial`, checked by size and then renamed, so Studio never sees half a file; the free space is checked first. Studio is told about the exports
//! folder once (`POST /api/models/scan-folders`) when it offers that route. The fine-tune handoff (also copying the healed Hugging Face folder) is off unless the settings say so.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use super::client::{ScanFolder, UnslothClient};
use crate::error::{Error, Result};
use crate::settings::StudioShare;

/// The folder of the share the exports go to.
pub const EXPORTS: &str = "exports";
const CHUNK: usize = 8 * 1024 * 1024;
/// Space to spare on the share besides the files themselves.
const SLACK_GB: f64 = 0.25;

/// A folder or file name that is safe on every system: letters, digits, `.`, `_`, `-`.
pub fn safe_name(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' { c } else { '-' }).collect();
    let s = s.trim_matches(|c| c == '-' || c == '.').to_string();
    if s.is_empty() { "model".into() } else { s.chars().take(80).collect() }
}

/// The separator a container path uses (a Linux container: `/`; a path that has only backslashes keeps them).
fn sep_of(p: &str) -> char {
    if p.contains('\\') && !p.contains('/') { '\\' } else { '/' }
}

/// `base` with `parts` added, in the style of `base`.
pub fn join_container(base: &str, parts: &[&str]) -> String {
    let sep = sep_of(base);
    let mut out = base.trim_end_matches(['/', '\\']).to_string();
    for p in parts {
        out.push(sep);
        out.push_str(p.trim_matches(['/', '\\']));
    }
    out
}

/// The folder above a container path ("/workspace/outputs" -> "/workspace").
pub fn parent_of(path: &str) -> Option<String> {
    let t = path.trim_end_matches(['/', '\\']);
    let i = t.rfind(['/', '\\'])?;
    let parent = &t[..i];
    if parent.is_empty() { Some(t[..1].to_string()) } else { Some(parent.to_string()) }
}

/// What the checkpoints listing says about Studio's own folders.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub outputs_dir: Option<String>,
    /// the folder above `outputs_dir`: Studio's root inside its container
    pub container_root: Option<String>,
    /// the run folders Studio lists that were found under `<share>/outputs` (empty and `confirmed` None when Studio lists none)
    pub matched: Vec<String>,
    /// Some(true): at least one run folder of Studio's list is in the share; Some(false): Studio lists runs and none is there (the share is probably not Studio's); None: nothing to compare
    pub confirmed: Option<bool>,
}

/// Asks Studio where its outputs are (`outputs_dir` of the checkpoints listing, a container path) and checks it against the share this computer sees.
pub fn detect(client: &UnslothClient, host: &Path) -> Result<Detection> {
    let listing = client.checkpoints_listing()?;
    Ok(detect_from(&listing, host))
}

pub fn detect_from(listing: &Value, host: &Path) -> Detection {
    let outputs_dir = listing["outputs_dir"].as_str().map(String::from).filter(|s| !s.is_empty());
    let names: Vec<String> = listing["models"].as_array().map(|a| a.iter().filter_map(|m| m["name"].as_str()).map(String::from).collect()).unwrap_or_default();
    let outputs = host.join("outputs");
    let matched: Vec<String> = names.iter().filter(|n| safe_component(n) && outputs.join(n).is_dir()).cloned().collect();
    let confirmed = if names.is_empty() { None } else { Some(!matched.is_empty()) };
    Detection { container_root: outputs_dir.as_deref().and_then(parent_of), outputs_dir, matched, confirmed }
}

fn safe_component(n: &str) -> bool {
    !n.is_empty() && n != "." && n != ".." && !n.contains(['/', '\\'])
}

/// What to put into the share.
pub struct PublishSpec {
    /// the model's name (the folder under `exports/`)
    pub name: String,
    pub files: Vec<PathBuf>,
    /// the healed Hugging Face folder, copied only when the share's `finetune_handoff` is on
    pub hf_folder: Option<PathBuf>,
    /// text for the README next to the files
    pub readme: Option<String>,
}

fn not_published(why: impl Into<String>) -> Value {
    json!({"published": false, "why": why.into()})
}

fn dir_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir).map(|d| d.flatten().map(|e| match e.metadata() {
        Ok(m) if m.is_dir() => dir_size(&e.path()),
        Ok(m) => m.len(),
        Err(_) => 0,
    }).sum()).unwrap_or(0)
}

fn gb(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Copies `from` to `to` in chunks (the stop flag is looked at between chunks); the size must match afterwards.
fn copy_file(from: &Path, to: &Path, stop: &AtomicBool, done: &mut u64, total: u64, on_progress: &mut dyn FnMut(f64)) -> Result<()> {
    let mut src = std::fs::File::open(from).map_err(|e| Error::new(format!("{} cannot be read: {e}", from.display())))?;
    let mut dst = std::fs::File::create(to).map_err(|e| Error::new(format!("{} cannot be written: {e}", to.display())))?;
    let mut buf = vec![0u8; CHUNK];
    let want = src.metadata()?.len();
    let mut got = 0u64;
    loop {
        if stop.load(Ordering::SeqCst) {
            return Err(Error::new("Stopped."));
        }
        let n = src.read(&mut buf)?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).map_err(|e| Error::new(format!("Writing to the Studio share failed: {e} (is the disk full or the drive gone?)")))?;
        got += n as u64;
        *done += n as u64;
        on_progress(*done as f64 / total.max(1) as f64);
    }
    dst.flush()?;
    drop(dst);
    if got != want || std::fs::metadata(to)?.len() != want {
        return Err(Error::new(format!("{} was not copied whole ({got} of {want} bytes).", from.display())));
    }
    Ok(())
}

fn copy_tree(from: &Path, to: &Path, stop: &AtomicBool, done: &mut u64, total: u64, on_progress: &mut dyn FnMut(f64)) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)?.flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if e.metadata()?.is_dir() {
            copy_tree(&src, &dst, stop, done, total, on_progress)?;
        } else {
            copy_file(&src, &dst, stop, done, total, on_progress)?;
        }
    }
    Ok(())
}

/// `<name>.partial` becomes `<name>`; an older `<name>` goes only once the new one is complete.
fn swap_in(partial: &Path, dest: &Path) -> Result<()> {
    let old = suffixed(dest, ".old");
    let _ = std::fs::remove_dir_all(&old);
    let had = dest.exists();
    if had {
        std::fs::rename(dest, &old)?;
    }
    if let Err(e) = std::fs::rename(partial, dest) {
        if had {
            let _ = std::fs::rename(&old, dest);
        }
        return Err(Error::new(format!("The copy is done but could not be moved into place: {e}")));
    }
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// `dest` with `suffix` added to its name (a name with dots in it keeps them: `with_extension` would replace the last part).
fn suffixed(dest: &Path, suffix: &str) -> PathBuf {
    let mut n = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    n.push(suffix);
    dest.with_file_name(n)
}

fn partial_of(dest: &Path) -> PathBuf {
    suffixed(dest, ".partial")
}

/// Everything that is the same on every call: where Studio is (if known) and how much room the share has.
pub struct Ctx<'a> {
    pub client: Option<&'a UnslothClient>,
    /// free space on the share, in GB (None: not known, the copy goes ahead)
    pub free_gb: Option<f64>,
    pub stop: &'a AtomicBool,
}

/// The whole publish. `Ok({"published": false, "why"})` when it cannot be done for a reason the person can fix (nothing set up, the share not there, the folders do not match);
/// `Err` for a failed copy or a full disk (the partial folder is removed). `Ok({"published": true, ...})` with the paths on this computer and inside Studio.
pub fn publish(share: &StudioShare, spec: &PublishSpec, ctx: &Ctx, on_progress: &mut dyn FnMut(f64)) -> Result<Value> {
    let host = Path::new(share.host_path.trim());
    if share.host_path.trim().is_empty() {
        return Ok(not_published("Unsloth Studio's share is not set up (Settings: \"Studio share\")."));
    }
    if !host.is_dir() {
        return Ok(not_published(format!("The Studio share {} was not found. Is the drive connected?", host.display())));
    }
    if spec.files.is_empty() {
        return Ok(not_published("There are no files to put into Unsloth Studio."));
    }
    if let Some(f) = spec.files.iter().find(|f| !f.is_file()) {
        return Err(Error::new(format!("The file {} is not there any more.", f.display())));
    }
    let mut notes: Vec<String> = Vec::new();
    // where Studio is, and is this its share?
    let mut container_root = Some(share.container_path.trim().to_string()).filter(|c| !c.is_empty());
    let mut detection = Value::Null;
    if let Some(c) = ctx.client {
        match detect(c, host) {
            Ok(d) => {
                detection = json!({"outputs_dir": d.outputs_dir, "container_root": d.container_root, "matched": d.matched, "confirmed": d.confirmed});
                if d.confirmed == Some(false) {
                    return Ok(not_published(format!("Unsloth Studio lists run folders that are not in {} (looked in its \"outputs\" folder). Is that the folder Studio keeps its files in? Nothing was copied.", host.display())));
                }
                if d.confirmed.is_none() {
                    notes.push("Studio lists no run folders yet, so the share could not be confirmed against its outputs folder.".into());
                }
                if container_root.is_none() {
                    container_root = d.container_root;
                }
            }
            Err(e) => notes.push(format!("Studio could not be asked where its folders are ({}); the settings are used.", e.0)),
        }
    }
    let name = safe_name(&spec.name);
    let exports = host.join(EXPORTS);
    let (dest, hf_dest) = (exports.join(&name), exports.join(format!("{name}-hf")));
    let hf = spec.hf_folder.as_ref().filter(|f| share.finetune_handoff && f.is_dir());
    // room first: the new files are copied next to the old ones, which go when the copy is complete
    let bytes: u64 = spec.files.iter().map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0)).sum::<u64>() + spec.readme.as_ref().map(|r| r.len() as u64).unwrap_or(0) + hf.map(|f| dir_size(f)).unwrap_or(0);
    if let Some(free) = ctx.free_gb {
        let need = bytes as f64 / 1e9 * 1.02 + SLACK_GB;
        if free < need {
            return Err(Error::new(format!("The Studio share has {:.1} GB free; the files need about {:.1} GB. Free some space on the share and publish again (the files stay on this computer).", gb(free), gb(need))));
        }
    }
    std::fs::create_dir_all(&exports).map_err(|e| Error::new(format!("The exports folder on the Studio share cannot be made: {e}")))?;
    let mut done = 0u64;
    let work = |dest: &Path, f: &mut dyn FnMut(&Path) -> Result<()>| -> Result<()> {
        let partial = partial_of(dest);
        let _ = std::fs::remove_dir_all(&partial);
        std::fs::create_dir_all(&partial)?;
        let r = f(&partial);
        if let Err(e) = r {
            let _ = std::fs::remove_dir_all(&partial);
            return Err(e);
        }
        swap_in(&partial, dest)
    };
    work(&dest, &mut |partial| {
        for f in &spec.files {
            let fname = f.file_name().ok_or_else(|| Error::new("A file has no name."))?;
            copy_file(f, &partial.join(fname), ctx.stop, &mut done, bytes, on_progress)?;
        }
        if let Some(r) = &spec.readme {
            std::fs::write(partial.join("README.md"), r)?;
        }
        Ok(())
    })?;
    if let Some(src) = hf {
        work(&hf_dest, &mut |partial| copy_tree(src, partial, ctx.stop, &mut done, bytes, on_progress))?;
    }
    on_progress(1.0);
    // Studio's own view of the same folders
    let container_dir = container_root.as_deref().map(|r| join_container(r, &[EXPORTS, &name]));
    let container_hf = hf.and(container_root.as_deref()).map(|r| join_container(r, &[EXPORTS, &format!("{name}-hf")]));
    let container_exports = container_root.as_deref().map(|r| join_container(r, &[EXPORTS]));
    let mut scan = json!({"status": "skipped", "note": "Registering the exports folder is switched off in the settings."});
    if share.register_scan_folder {
        scan = match (ctx.client, &container_exports) {
            (None, _) => json!({"status": "skipped", "note": "No Unsloth connection with an API key: add the exports folder in Studio (Models -> Custom folders) yourself."}),
            (_, None) => json!({"status": "skipped", "note": "The folder as Studio sees it is not known: set \"Path inside the container\" and publish again."}),
            (Some(c), Some(path)) => match c.scan_folders_available() {
                Ok(true) => match c.add_scan_folder(path, true) {
                    Ok(ScanFolder::Added(_)) => json!({"status": "registered", "path": path}),
                    Ok(ScanFolder::Rejected(why)) => json!({"status": "rejected", "path": path, "note": format!("Studio did not take the folder: {why}. Check \"Path inside the container\".")}),
                    Ok(ScanFolder::NotOffered) => json!({"status": "not_offered", "note": "This Studio has no list of custom folders."}),
                    Err(e) => json!({"status": "failed", "note": e.0}),
                },
                Ok(false) => json!({"status": "not_offered", "note": "This Studio has no list of custom folders: add the exports folder in Studio yourself."}),
                Err(e) => json!({"status": "failed", "note": e.0}),
            },
        };
    }
    let files: Vec<Value> = spec.files.iter().filter_map(|f| f.file_name()).map(|n| {
        let n = n.to_string_lossy().to_string();
        json!({"file": n, "host_path": dest.join(&n), "container_path": container_dir.as_deref().map(|d| join_container(d, &[&n]))})
    }).collect();
    Ok(json!({
        "published": true, "name": name, "host_path": dest, "container_path": container_dir, "files": files, "size_gb": gb(bytes as f64 / 1e9),
        "hf": hf.map(|_| json!({"host_path": hf_dest, "container_path": container_hf})),
        // the model a fine-tune in Studio would start from (only with the handoff switched on; the training itself is started in Studio)
        "finetune_model": container_hf,
        "scan_folder": scan, "detected": detection, "notes": notes,
    }))
}
