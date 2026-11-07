//! Compare two bundles of the same world: what was added, changed or removed (shown before "Retrain with the new export").
use std::collections::BTreeMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::bundle::{Bundle, LABELS};
use crate::error::Result;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KindDiff {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

impl KindDiff {
    pub fn total(&self) -> usize {
        self.added.len() + self.changed.len() + self.removed.len()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BundleDiff {
    pub kinds: BTreeMap<String, KindDiff>,
}

impl BundleDiff {
    pub fn total(&self) -> usize {
        self.kinds.values().map(|k| k.total()).sum()
    }

    pub fn summary(&self) -> String {
        if self.total() == 0 {
            return "Nothing changed.".to_string();
        }
        let mut parts = Vec::new();
        for (member, label, _) in LABELS {
            if let Some(d) = self.kinds.get(member).filter(|d| d.total() > 0) {
                parts.push(format!("{label}: {} new, {} changed, {} removed", d.added.len(), d.changed.len(), d.removed.len()));
            }
        }
        parts.join("; ")
    }
}

/// Fields that do not change the meaning of a record.
const IGNORE: [&str; 2] = ["updated_at", "created_at"];

fn digest(rec: &Value) -> String {
    // serde_json maps keep insertion order, so sort the keys first
    fn canon(v: &Value, out: &mut String) {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                out.push('{');
                for k in keys {
                    out.push_str(&serde_json::to_string(k).unwrap_or_default());
                    out.push(':');
                    canon(&m[k], out);
                    out.push(',');
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for x in a {
                    canon(x, out);
                    out.push(',');
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut clean = rec.clone();
    if let Value::Object(m) = &mut clean {
        for k in IGNORE {
            m.remove(k);
        }
    }
    let mut s = String::new();
    canon(&clean, &mut s);
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

fn label(rec: &Value) -> String {
    for k in ["name", "title"] {
        if let Some(s) = rec[k].as_str().filter(|s| !s.is_empty()) {
            return s.to_string();
        }
    }
    if let Some(s) = rec["content"].as_str().filter(|s| !s.is_empty()) {
        return s.chars().take(40).collect();
    }
    rec["id"].to_string()
}

pub fn diff_bundles(old: &Bundle, new: &Bundle) -> Result<BundleDiff> {
    let mut kinds = BTreeMap::new();
    for (member, _, _) in LABELS {
        if !old.counts.contains_key(member) && !new.counts.contains_key(member) {
            continue;
        }
        let key = |r: &Value| r["id"].to_string();
        let a: BTreeMap<String, Value> = old.records(member)?.into_iter().map(|r| (key(&r), r)).collect();
        let b: BTreeMap<String, Value> = new.records(member)?.into_iter().map(|r| (key(&r), r)).collect();
        let mut d = KindDiff::default();
        for (id, rec) in &b {
            match a.get(id) {
                None => d.added.push(label(rec)),
                Some(old_rec) if digest(old_rec) != digest(rec) => d.changed.push(label(rec)),
                _ => {}
            }
        }
        d.removed = a.iter().filter(|(id, _)| !b.contains_key(*id)).map(|(_, r)| label(r)).collect();
        kinds.insert(member.to_string(), d);
    }
    Ok(BundleDiff { kinds })
}

pub fn same_world(a: &Bundle, b: &Bundle) -> bool {
    a.manifest["world"]["slug"] == b.manifest["world"]["slug"]
}
