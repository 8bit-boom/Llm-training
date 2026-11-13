//! The training rows of a saved data version in the "chatml" shape (one `{"messages": [...]}` per line). Shared by the export for Unsloth Studio's Train page
//! (`export::studio`) and the training run in Unsloth Desktop (`unsloth::run`), so both send exactly the same text.
//!
//! The world book (long texts) has no counterpart in Studio's formats, so by default it becomes answers to "Tell me what the lore says about <title>".
use std::path::Path;

use serde_json::{json, Value};

use crate::dataset::builders::system_prompt;
use crate::project::Project;

fn read_lines(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path).map(|t| t.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()).unwrap_or_default()
}

/// The title of a world-book piece: its first markdown heading, else the source label.
fn title_of(text: &str, source: &str) -> String {
    text.lines().map(str::trim).find_map(|l| l.strip_prefix("# ")).map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
        .unwrap_or_else(|| source.split(['@', ':']).nth(1).unwrap_or(source).to_string())
}

/// Which world a source belongs to (its suffix names it), for the system message.
fn world_of(p: &Project, source: &str) -> String {
    let worlds = p.worlds();
    worlds.iter().filter(|w| !w.suffix.is_empty() && source.ends_with(&w.suffix)).map(|w| w.name.clone()).next()
        .or_else(|| worlds.iter().find(|w| w.suffix.is_empty()).map(|w| w.name.clone()))
        .unwrap_or_else(|| p.display_name())
}

fn chatml(messages: &[Value]) -> Option<String> {
    let msgs: Vec<Value> = messages.iter().filter(|m| m["role"].is_string() && m["content"].is_string()).cloned().collect();
    if msgs.iter().any(|m| m["role"] == "assistant") { serde_json::to_string(&json!({"messages": msgs})).ok() } else { None }
}

/// What `chatml_rows` counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    /// questions and answers
    pub chats: usize,
    /// world-book pieces turned into answers
    pub book: usize,
    /// length in bytes of the longest training row
    pub longest: usize,
}

/// (training rows, held-out rows, counts) of saved data version `version`. A row is one JSON line without the line break. `world_book` false leaves the long texts out.
pub fn chatml_rows(p: &Project, version: u32, world_book: bool) -> (Vec<String>, Vec<String>, Stats) {
    let vdir = p.folder.join("datasets").join(format!("v{version}"));
    let (mut train, mut stats) = (Vec::<String>::new(), Stats::default());
    for v in read_lines(&vdir.join("train.jsonl")) {
        let line = if v["type"] == "text" {
            if !world_book {
                continue;
            }
            let text = v["text"].as_str().unwrap_or("");
            let src = v["source"].as_str().unwrap_or("");
            let title = title_of(text, src);
            stats.book += 1;
            chatml(&[json!({"role": "system", "content": system_prompt(&world_of(p, src))}), json!({"role": "user", "content": format!("Tell me what the lore says about {title}.")}),
                     json!({"role": "assistant", "content": text})])
        } else {
            stats.chats += 1;
            chatml(v["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]))
        };
        if let Some(l) = line {
            stats.longest = stats.longest.max(l.len());
            train.push(l);
        }
    }
    let eval: Vec<String> = read_lines(&vdir.join("test.jsonl")).into_iter().filter_map(|v| chatml(v["messages"].as_array().map(|a| a.as_slice()).unwrap_or(&[]))).collect();
    (train, eval, stats)
}
