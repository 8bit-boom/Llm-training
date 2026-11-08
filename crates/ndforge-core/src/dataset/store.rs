//! The draft (generated examples the user can review, drop and edit) and the saved versions (`datasets/vN/`) training uses.
//!
//! ```text
//! datasets/draft/recipe.json      the recipe the draft was generated with
//! datasets/draft/examples.jsonl   one example per line, with the user's decisions (status, edits)
//! datasets/vN/train.jsonl         {"type":"text",...} world book + {"type":"chat",...} practice conversations
//! datasets/vN/test.jsonl          held-out conversations, for step 4
//! datasets/vN/manifest.json       what it was made from and how big it is
//! ```
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use serde_json::{json, Value};

use super::builders::{generate, Example, Recipe, Status, BUILDERS};
use super::chunk::approx_tokens;
use super::docs::load;
use super::split::held_out_sources;
use crate::error::{Error, Result};
use crate::project::Project;

fn dir(p: &Project) -> PathBuf {
    p.folder.join("datasets")
}

fn draft_dir(p: &Project) -> PathBuf {
    dir(p).join("draft")
}

fn write_atomic(path: &std::path::Path, data: &[u8]) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

fn to_jsonl(examples: &[Example]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for e in examples {
        out.extend(serde_json::to_vec(e)?);
        out.push(b'\n');
    }
    Ok(out)
}

pub fn load_recipe(p: &Project) -> Option<Recipe> {
    std::fs::read_to_string(draft_dir(p).join("recipe.json")).ok().and_then(|t| serde_json::from_str(&t).ok())
}

pub fn load_examples(p: &Project) -> Result<Vec<Example>> {
    let text = match std::fs::read_to_string(draft_dir(p).join("examples.jsonl")) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(|e| Error::new(format!("The draft is damaged: {e}")))).collect()
}

fn save_examples(p: &Project, ex: &[Example]) -> Result<()> {
    write_atomic(&draft_dir(p).join("examples.jsonl"), &to_jsonl(ex)?)
}

/// Made by the teacher model (builder ids "t_...").
pub fn is_teacher(e: &Example) -> bool {
    e.builder.starts_with("t_")
}

/// Teacher examples never go to the test file when their source is general (general chat, house style): they test nothing about the world.
fn always_train(e: &Example) -> bool {
    e.builder == "t_general" || e.builder == "t_style"
}

pub fn teacher_cache_dir(p: &Project) -> PathBuf {
    dir(p).join("teacher_cache")
}

/// Adds new teacher examples to the draft; one that already exists (same id) keeps the user's decisions.
pub fn merge_teacher(p: &Project, new: Vec<Example>) -> Result<usize> {
    if new.is_empty() {
        return Ok(0);
    }
    let mut ex = load_examples(p)?;
    let have: std::collections::HashSet<String> = ex.iter().map(|e| e.id.clone()).collect();
    let mut n = 0;
    for e in new {
        if !have.contains(&e.id) {
            ex.push(e);
            n += 1;
        }
    }
    save_examples(p, &ex)?;
    Ok(n)
}

/// Removes all teacher examples (`kind` = one of them, or None for all) and, with `forget_cache`, the stored teacher answers.
pub fn clear_teacher(p: &Project, kind: Option<&str>, forget_cache: bool) -> Result<usize> {
    let ex = load_examples(p)?;
    let before = ex.len();
    let kept: Vec<Example> = ex.into_iter().filter(|e| !(is_teacher(e) && kind.map_or(true, |k| k == e.builder))).collect();
    save_examples(p, &kept)?;
    if forget_cache {
        let _ = std::fs::remove_dir_all(teacher_cache_dir(p));
    }
    Ok(before - kept.len())
}

fn tokens(e: &Example) -> usize {
    e.text.as_deref().map(approx_tokens).unwrap_or(0) + e.messages.iter().map(|m| approx_tokens(&m.content)).sum::<usize>()
}

/// Builds the draft from the project's current world data. Decisions made on an earlier draft (dropped, edited) are kept for every example
/// that still exists, so updating the world data does not lose the review work.
pub fn generate_draft(p: &Project, recipe: &Recipe) -> Result<Value> {
    let bundle = p.current_bundle().ok_or_else(|| Error::new("Import the world data first (Step 1)."))?;
    let loaded = load(&bundle)?;
    let mut fresh = generate(&loaded, recipe);
    let prior_all = load_examples(p)?;
    // teacher examples are not regenerated here: keep those whose source still exists in the world (and the general / style ones)
    let sources: std::collections::HashSet<String> = super::docs::articles(&loaded).into_iter().map(|d| d.source).collect();
    let teacher_kept: Vec<Example> = prior_all.iter().filter(|e| is_teacher(e) && (sources.contains(&e.source) || e.source.starts_with("general:") || e.source.starts_with("instruction:"))).cloned().collect();
    let prior: HashMap<String, Example> = prior_all.into_iter().map(|e| (e.id.clone(), e)).collect();
    // examples nd-world's own AI model wrote and checked (the bundle's teacher_examples.jsonl): they join the draft like any other; a decision made
    // on one earlier (dropped, edited) is applied below, so a newer export never undoes the review work
    let in_fresh: std::collections::HashSet<String> = fresh.iter().map(|e| e.id.clone()).collect();
    let from_nd: Vec<Example> = bundle.records("teacher_examples.jsonl")?.into_iter().filter_map(|v| serde_json::from_value::<Example>(v).ok())
        .filter(|e| is_teacher(&e) && !e.messages.is_empty() && !in_fresh.contains(&e.id)).collect();
    let nd_ids: std::collections::HashSet<String> = from_nd.iter().map(|e| e.id.clone()).collect();
    fresh.extend(from_nd);
    for e in &mut fresh {
        if let Some(old) = prior.get(&e.id) {
            e.status = old.status;
            if old.edited {
                e.edited = true;
                e.messages = old.messages.clone();
                e.text = old.text.clone();
            }
        }
    }
    fresh.extend(teacher_kept.into_iter().filter(|e| !nd_ids.contains(&e.id)));
    write_atomic(&draft_dir(p).join("recipe.json"), &serde_json::to_vec_pretty(recipe)?)?;
    save_examples(p, &fresh)?;
    write_atomic(&draft_dir(p).join("bundle.txt"), bundle.sha256.as_bytes())?;
    Ok(summary(p)?.unwrap_or(Value::Null))
}

/// Counts for the page: per builder, how many are kept / dropped, plus totals and the rough size. None when no draft exists.
pub fn summary(p: &Project) -> Result<Option<Value>> {
    if !draft_dir(p).join("examples.jsonl").exists() {
        return Ok(None);
    }
    let ex = load_examples(p)?;
    let mut by: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for e in &ex {
        let c = by.entry(e.builder.as_str()).or_default();
        match e.status {
            Status::Keep => c.0 += 1,
            Status::Drop => c.1 += 1,
        }
    }
    let name = |id: &str| -> String { if id == "world_book" { "World book (the lore as text)".to_string() } else if let Some(k) = crate::teacher::KINDS.iter().find(|k| k.0 == id) { format!("Teacher: {}", k.1) } else { BUILDERS.iter().find(|(b, _)| *b == id).map(|(_, n)| n.to_string()).unwrap_or_else(|| id.to_string()) } };
    let builders: Vec<Value> = by.iter().map(|(id, (k, d))| json!({"id": id, "name": name(id), "kept": k, "dropped": d})).collect();
    let kept: Vec<&Example> = ex.iter().filter(|e| e.status == Status::Keep).collect();
    let stale = std::fs::read_to_string(draft_dir(p).join("bundle.txt")).ok().zip(p.current_bundle()).map_or(false, |(old, cur)| old != cur.sha256);
    Ok(Some(json!({
        "recipe": load_recipe(p), "builders": builders, "total": ex.len(), "kept": kept.len(),
        "edited": ex.iter().filter(|e| e.edited).count(), "tokens": kept.iter().map(|e| tokens(e)).sum::<usize>(),
        "stale": stale, "nd_world_examples": p.current_bundle().and_then(|b| b.counts.get("teacher_examples.jsonl").copied()).unwrap_or(0),
    })))
}

#[derive(Debug, Default, Clone)]
pub struct Filter {
    pub builder: Option<String>,
    pub search: Option<String>,
    pub status: Option<Status>,
}

fn matches(e: &Example, f: &Filter) -> bool {
    if f.builder.as_deref().is_some_and(|b| !b.is_empty() && b != e.builder) || f.status.is_some_and(|s| s != e.status) {
        return false;
    }
    match f.search.as_deref().map(str::to_lowercase).filter(|s| !s.is_empty()) {
        None => true,
        Some(q) => e.source_title.to_lowercase().contains(&q) || e.text.as_deref().unwrap_or("").to_lowercase().contains(&q) || e.messages.iter().any(|m| m.content.to_lowercase().contains(&q)),
    }
}

pub fn list(p: &Project, f: &Filter, offset: usize, limit: usize) -> Result<(usize, Vec<Example>)> {
    let all: Vec<Example> = load_examples(p)?.into_iter().filter(|e| matches(e, f)).collect();
    let total = all.len();
    Ok((total, all.into_iter().skip(offset).take(limit.min(500)).collect()))
}

pub fn set_status(p: &Project, ids: &[String], status: Status) -> Result<usize> {
    let mut ex = load_examples(p)?;
    let mut n = 0;
    for e in ex.iter_mut().filter(|e| ids.contains(&e.id)) {
        e.status = status;
        n += 1;
    }
    save_examples(p, &ex)?;
    Ok(n)
}

/// Sets every example that matches the filter (for "drop all of this kind").
pub fn set_status_matching(p: &Project, f: &Filter, status: Status) -> Result<usize> {
    let mut ex = load_examples(p)?;
    let mut n = 0;
    for e in ex.iter_mut().filter(|e| matches(e, f)) {
        e.status = status;
        n += 1;
    }
    save_examples(p, &ex)?;
    Ok(n)
}

/// Replaces the answer (the assistant message) or, for the world book, the text.
pub fn edit(p: &Project, id: &str, new_text: &str) -> Result<()> {
    let new_text = new_text.trim();
    if new_text.is_empty() {
        return Err(Error::new("The answer cannot be empty. Use \"Drop\" to leave this example out."));
    }
    let mut ex = load_examples(p)?;
    let e = ex.iter_mut().find(|e| e.id == id).ok_or_else(|| Error::new("That example no longer exists."))?;
    if e.text.is_some() {
        e.text = Some(new_text.to_string());
    } else if let Some(last) = e.messages.iter_mut().rev().find(|m| m.role == "assistant") {
        last.content = new_text.to_string();
    }
    e.edited = true;
    save_examples(p, &ex)
}

fn jsonl_line(e: &Example) -> Result<Vec<u8>> {
    let v = if let Some(t) = &e.text {
        json!({"type": "text", "id": e.id, "source": e.source, "text": t})
    } else {
        json!({"type": "chat", "id": e.id, "builder": e.builder, "source": e.source, "messages": e.messages})
    };
    let mut b = serde_json::to_vec(&v)?;
    b.push(b'\n');
    Ok(b)
}

/// Freezes the kept examples as the next version. Returns its manifest.
pub fn save_version(p: &Project) -> Result<Value> {
    let recipe = load_recipe(p).ok_or_else(|| Error::new("Make the training data first (press \"Make the examples\")."))?;
    let kept: Vec<Example> = load_examples(p)?.into_iter().filter(|e| e.status == Status::Keep).collect();
    if kept.is_empty() {
        return Err(Error::new("Every example is dropped; there is nothing to save."));
    }
    let chat_sources: Vec<String> = kept.iter().filter(|e| e.text.is_none() && !always_train(e)).map(|e| e.source.clone()).collect();
    let held = held_out_sources(&chat_sources, recipe.seed, recipe.test_percent);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    let (mut n_text, mut n_train_chat, mut n_test_chat) = (0, 0, 0);
    for e in &kept {
        if e.text.is_some() {
            train.extend(jsonl_line(e)?);
            n_text += 1;
        } else if held.contains(&e.source) && !always_train(e) {
            test.extend(jsonl_line(e)?);
            n_test_chat += 1;
        } else {
            train.extend(jsonl_line(e)?);
            n_train_chat += 1;
        }
    }
    let mut builders: BTreeMap<String, usize> = BTreeMap::new();
    for e in &kept {
        *builders.entry(e.builder.clone()).or_default() += 1;
    }
    let n = std::fs::read_dir(dir(p)).map(|d| d.flatten().filter_map(|e| e.file_name().to_str()?.strip_prefix('v')?.parse::<u32>().ok()).max().unwrap_or(0)).unwrap_or(0) + 1;
    let vdir = dir(p).join(format!("v{n}"));
    let mut held_sorted: Vec<&String> = held.iter().collect();
    held_sorted.sort();
    let manifest = json!({
        "version": n, "created": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        "world": p.world_name, "bundle_sha256": p.current_bundle().map(|b| b.sha256), "recipe": recipe,
        "counts": {"world_book_texts": n_text, "train_chat": n_train_chat, "test_chat": n_test_chat, "dropped": load_examples(p)?.len() - kept.len(), "edited": kept.iter().filter(|e| e.edited).count()},
        "tokens": {"train": kept.iter().filter(|e| !held.contains(&e.source) || e.text.is_some() || always_train(e)).map(tokens).sum::<usize>(), "test": kept.iter().filter(|e| e.text.is_none() && held.contains(&e.source) && !always_train(e)).map(tokens).sum::<usize>()},
        "builders": builders, "held_out_sources": held_sorted,
    });
    write_atomic(&vdir.join("train.jsonl"), &train)?;
    write_atomic(&vdir.join("test.jsonl"), &test)?;
    write_atomic(&vdir.join("manifest.json"), &serde_json::to_vec_pretty(&manifest)?)?; // last: its presence marks the version as complete
    Ok(manifest)
}

pub fn versions(p: &Project) -> Vec<Value> {
    let mut out: Vec<(u32, Value)> = std::fs::read_dir(dir(p)).map(|d| d.flatten().filter_map(|e| {
        let n: u32 = e.file_name().to_str()?.strip_prefix('v')?.parse().ok()?;
        let m = serde_json::from_slice(&std::fs::read(e.path().join("manifest.json")).ok()?).ok()?;
        Some((n, m))
    }).collect()).unwrap_or_default();
    out.sort_by_key(|(n, _)| *n);
    out.into_iter().map(|(_, m)| m).collect()
}
