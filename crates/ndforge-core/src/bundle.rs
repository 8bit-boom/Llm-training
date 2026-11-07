//! Read, verify and (for tests and the sample file) write nd-world Training Bundles. Format: docs/BUNDLE_FORMAT.md.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

pub const FORMAT: &str = "nd-training-bundle";
pub const SUPPORTED_VERSION: i64 = 1;
const REQUIRED: [&str; 2] = ["world.json", "entities.jsonl"];
/// The order members are written in; also the list of members the reader knows about.
pub const MEMBER_ORDER: [&str; 13] = [
    "manifest.json", "world.json", "entities.jsonl", "entity_notes.jsonl", "relations.jsonl", "rules.md", "rules_sheets.jsonl",
    "sessions.jsonl", "facts.jsonl", "quests.jsonl", "tables.jsonl", "ai_instructions.jsonl", "characters.jsonl",
];
/// Plain-language names for the summary ("312 entities").
pub const LABELS: [(&str, &str, &str); 10] = [
    ("entities.jsonl", "entities", "entity"), ("entity_notes.jsonl", "entity notes", "entity note"),
    ("relations.jsonl", "relations", "relation"), ("rules_sheets.jsonl", "sheet rules", "sheet rules"),
    ("sessions.jsonl", "session recaps", "session recap"), ("facts.jsonl", "facts", "fact"), ("quests.jsonl", "quests", "quest"),
    ("tables.jsonl", "random tables", "random table"), ("ai_instructions.jsonl", "AI instructions", "AI instruction"),
    ("characters.jsonl", "player characters", "player character"),
];

#[derive(Debug, Clone)]
pub struct Bundle {
    pub path: PathBuf,
    pub manifest: Value,
    pub world: Value,
    /// SHA-256 of the whole file: identifies this snapshot.
    pub sha256: String,
    /// records per .jsonl member (rules.md counts 1 when it has text)
    pub counts: BTreeMap<String, usize>,
}

impl Bundle {
    pub fn world_name(&self) -> String {
        self.manifest["world"]["name"].as_str().unwrap_or("").to_string()
    }
    pub fn world_slug(&self) -> String {
        self.manifest["world"]["slug"].as_str().unwrap_or("").to_string()
    }
    pub fn created(&self) -> String {
        self.manifest["created"].as_str().unwrap_or("").to_string()
    }

    pub fn summary_lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (member, plural, singular) in LABELS {
            if let Some(n) = self.counts.get(member).filter(|n| **n > 0) {
                out.push(format!("{n} {}", if *n == 1 { singular } else { plural }));
            }
        }
        if self.counts.get("rules.md").copied().unwrap_or(0) > 0 {
            out.push("world rules text".to_string());
        }
        out
    }

    pub fn read_text(&self, member: &str) -> Result<String> {
        let mut z = zip::ZipArchive::new(File::open(&self.path)?).map_err(|e| Error::new(format!("The bundle cannot be read: {e}")))?;
        let mut s = String::new();
        if let Ok(mut f) = z.by_name(member) {
            f.read_to_string(&mut s)?;
        }
        Ok(s)
    }

    /// The records of a .jsonl member (empty when the member is absent).
    pub fn records(&self, member: &str) -> Result<Vec<Value>> {
        let text = self.read_text(member)?;
        let mut out = Vec::new();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(line).map_err(|e| Error::new(format!("{member}, line {}: not valid JSON ({e})", n + 1)))?);
        }
        Ok(out)
    }
}

fn hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Open and fully verify a bundle. The error text is for the user.
pub fn open_bundle(path: impl AsRef<Path>) -> Result<Bundle> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(Error::new(format!("The file {} was not found.", path.file_name().and_then(|n| n.to_str()).unwrap_or(""))));
    }
    let not_zip = || Error::new("This file is not a training bundle (it is not a zip file). Export it from nd-world: Tools > Export & Backup > Training bundle.");
    let mut z = zip::ZipArchive::new(File::open(path)?).map_err(|_| not_zip())?;
    let names: Vec<String> = z.file_names().map(String::from).collect();
    if !names.iter().any(|n| n == "manifest.json") {
        return Err(Error::new("This zip file has no manifest.json, so it is not a training bundle."));
    }
    let mut read = |name: &str| -> Result<Vec<u8>> {
        let mut f = z.by_name(name).map_err(|_| Error::new(format!("The bundle is missing {name}.")))?;
        let mut v = Vec::new();
        f.read_to_end(&mut v)?;
        Ok(v)
    };
    let manifest: Value = serde_json::from_slice(&read("manifest.json")?).map_err(|e| Error::new(format!("manifest.json is damaged ({e}).")))?;
    if manifest["format"].as_str() != Some(FORMAT) {
        return Err(Error::new("This file is not a training bundle (wrong format name in manifest.json)."));
    }
    match manifest["version"].as_i64() {
        Some(v) if v <= SUPPORTED_VERSION => {}
        other => {
            let shown = other.map(|v| v.to_string()).unwrap_or_else(|| "unknown".into());
            return Err(Error::new(format!(
                "This bundle uses format version {shown}, but this program only understands version {SUPPORTED_VERSION}. Update ND Lore Forge."
            )));
        }
    }
    let listed = manifest["files"].as_object().cloned().unwrap_or_default();
    for req in REQUIRED {
        if !names.iter().any(|n| n == req) {
            return Err(Error::new(format!("The bundle is missing {req}.")));
        }
    }
    let extra: Vec<&String> = names.iter().filter(|n| *n != "manifest.json" && !listed.contains_key(n.as_str())).collect();
    if !extra.is_empty() {
        return Err(Error::new(format!(
            "The bundle contains files that are not in its manifest: {}.",
            extra.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
        )));
    }
    let mut counts = BTreeMap::new();
    for (member, info) in &listed {
        if !names.iter().any(|n| n == member) {
            return Err(Error::new(format!("The bundle is missing {member}, which its manifest lists.")));
        }
        let data = read(member)?;
        let size_ok = info["bytes"].as_u64().map_or(true, |b| b as usize == data.len());
        if !size_ok || info["sha256"].as_str() != Some(hex(&data).as_str()) {
            return Err(Error::new(format!("{member} is damaged (checksum does not match). Export the bundle again.")));
        }
        if member.ends_with(".jsonl") {
            counts.insert(member.clone(), data.split(|b| *b == b'\n').filter(|l| !l.iter().all(|c| c.is_ascii_whitespace())).count());
        } else if member == "rules.md" && !data.iter().all(|c| c.is_ascii_whitespace()) {
            counts.insert(member.clone(), 1);
        }
    }
    let world: Value = serde_json::from_slice(&read("world.json")?).map_err(|e| Error::new(format!("world.json is damaged ({e}).")))?;
    drop(read);
    Ok(Bundle { path: path.to_path_buf(), manifest, world, sha256: file_sha256(path)?, counts })
}

// ----------------------------------------------------------------------------------- writer (reference implementation of the zip rules)

/// Pretty JSON with two-space indent and a final newline, as Python's `json.dumps(indent=2, ensure_ascii=False)` writes it.
fn pretty(v: &Value) -> Vec<u8> {
    let mut s = serde_json::to_string_pretty(v).expect("a Value always serialises");
    s.push('\n');
    s.into_bytes()
}

/// Members for `write_bundle`: `.jsonl` members as records, `rules.md` as text.
pub enum Member {
    Records(Vec<Value>),
    Text(String),
}

/// Write a bundle following docs/BUNDLE_FORMAT.md (fixed timestamps, member order, deflate level 6). `world` is the world.json document.
pub fn write_bundle(path: impl AsRef<Path>, world: &Value, members: Vec<(&str, Member)>, nd_build: &str, created: &str) -> Result<()> {
    let mut payload: Vec<(String, Vec<u8>)> = vec![("world.json".into(), pretty(world))];
    for name in MEMBER_ORDER {
        if name == "manifest.json" || name == "world.json" {
            continue;
        }
        if let Some((_, m)) = members.iter().find(|(n, _)| *n == name) {
            let data = match m {
                Member::Records(recs) => recs.iter().flat_map(|r| {
                    let mut s = serde_json::to_string(r).expect("a Value always serialises");
                    s.push('\n');
                    s.into_bytes()
                }).collect(),
                Member::Text(t) => format!("{}\n", t.trim_end_matches('\n')).into_bytes(),
            };
            payload.push((name.to_string(), data));
        }
    }
    for (n, _) in &members {
        if !MEMBER_ORDER.contains(n) {
            return Err(Error::new(format!("unknown bundle member: {n}")));
        }
    }
    let mut files = Map::new();
    for (name, data) in &payload {
        let mut info = Map::new();
        info.insert("sha256".into(), json!(hex(data)));
        info.insert("bytes".into(), json!(data.len()));
        if name.ends_with(".jsonl") {
            info.insert("records".into(), json!(data.split(|b| *b == b'\n').filter(|l| !l.is_empty()).count()));
        }
        files.insert(name.clone(), Value::Object(info));
    }
    let pick = |k: &str| world.get(k).cloned().unwrap_or(Value::Null);
    let manifest = json!({"format": FORMAT, "version": SUPPORTED_VERSION, "created": created, "nd_build": nd_build,
        "world": {"id": pick("id"), "name": pick("name"), "slug": pick("slug"), "description": pick("description")}, "files": files});
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(Some(6))
        .last_modified_time(zip::DateTime::from_date_and_time(1980, 1, 1, 0, 0, 0).map_err(|_| Error::new("bad zip date"))?)
        .unix_permissions(0o644);
    let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let zerr = |e: zip::result::ZipError| Error::new(format!("Could not write the bundle: {e}"));
    w.start_file("manifest.json", opts).map_err(zerr)?;
    w.write_all(&pretty(&manifest))?;
    for (name, data) in &payload {
        w.start_file(name.as_str(), opts).map_err(zerr)?;
        w.write_all(data)?;
    }
    let bytes = w.finish().map_err(zerr)?.into_inner();
    std::fs::write(path, bytes)?;
    Ok(())
}
