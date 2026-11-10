//! The test questions: the held-out conversations of the run's data version (the world book had their facts, training never saw them as
//! questions), plus made-up probes (invented names that appear nowhere in the world) and a few general-chat prompts.
use std::collections::HashSet;

use serde_json::{json, Value};

use crate::dataset::builders::{hash_hex, hash_num, system_prompt};
use crate::dataset::docs::{articles, load};
use crate::error::{Error, Result};
use crate::project::Project;

#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub category: &'static str,
    /// the conversation up to and including the last user turn
    pub messages: Vec<Value>,
    pub question: String,
    pub reference: Option<String>,
}

const GENERAL: [&str; 6] = [
    "Give me three tips for pacing a tabletop session.",
    "How can I make a villain more memorable?",
    "What is a good way to handle a player who dominates the table?",
    "Suggest a name for a tavern run by a retired soldier.",
    "How do I describe a spooky forest in two sentences?",
    "Explain what advantage and disadvantage mean in a d20 game.",
];

const SYL_A: [&str; 10] = ["Zor", "Vel", "Quin", "Mor", "Thal", "Brix", "Xan", "Dro", "Fen", "Kry"];
const SYL_B: [&str; 8] = ["blax", "dara", "mire", "thos", "vane", "kesh", "lume", "gorn"];
const SYL_C: [&str; 6] = ["Prime", "Hollow", "Reach", "Vale", "Spire", "Gate"];

/// Invented names that appear nowhere in the world (checked against its whole text), deterministic for a seed.
pub fn invented_names(corpus_lower: &str, n: usize, seed: u64) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while out.len() < n && i < 500 {
        let h = hash_num(&format!("{seed}|{i}"));
        let name = format!("{}{} {}", SYL_A[h % SYL_A.len()], SYL_B[(h / 7) % SYL_B.len()], SYL_C[(h / 53) % SYL_C.len()]);
        i += 1;
        if !corpus_lower.contains(&name.to_lowercase()) && !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

fn category(builder: &str) -> &'static str {
    match builder {
        "rules" | "t_rules" => "rules",
        "t_missing" => "made_up",
        _ => "lore",
    }
}

/// At most `cap` of the items, chosen by a hash of their id (the same ones every time).
fn pick_stable(mut items: Vec<Question>, cap: usize) -> Vec<Question> {
    items.sort_by_key(|q| hash_hex(&[&q.id]));
    items.truncate(cap);
    items.sort_by(|a, b| a.id.cmp(&b.id));
    items
}

/// The question set of a saved data version. `per_category` caps the held-out questions of each category.
pub fn build(p: &Project, version: u64, per_category: usize) -> Result<Vec<Question>> {
    let test = p.folder.join("datasets").join(format!("v{version}")).join("test.jsonl");
    let text = std::fs::read_to_string(&test).map_err(|_| Error::new(format!("The test questions of data version {version} are missing.")))?;
    let mut held: Vec<Question> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| Error::new(format!("The test file is damaged: {e}")))?;
        if v["type"] != "chat" {
            continue;
        }
        let msgs = v["messages"].as_array().cloned().unwrap_or_default();
        let Some(last_user) = msgs.iter().rposition(|m| m["role"] == "user") else { continue };
        let reference = msgs.last().filter(|m| m["role"] == "assistant").and_then(|m| m["content"].as_str()).map(String::from);
        held.push(Question {
            id: v["id"].as_str().unwrap_or("").to_string(), category: category(v["builder"].as_str().unwrap_or("")),
            question: msgs[last_user]["content"].as_str().unwrap_or("").to_string(), messages: msgs[..=last_user].to_vec(), reference,
        });
    }
    if held.is_empty() {
        return Err(Error::new("No questions were kept for testing in this data version. In Step 2 set \"Kept for testing\" above 0 %, make the examples again and save a new version."));
    }
    let mut out = Vec::new();
    for cat in ["lore", "rules", "made_up"] {
        out.extend(pick_stable(held.iter().filter(|q| q.category == cat).cloned().collect(), per_category));
    }
    // made-up probes and general chat need no data
    // with several worlds the probes are spread over them, each with its own world's system prompt
    let bundles = p.bundles();
    if bundles.is_empty() {
        return Err(Error::new("The world data is missing."));
    }
    let mut corpus = String::new();
    let mut systems = Vec::new();
    for (_, bundle) in &bundles {
        let loaded = load(bundle)?;
        corpus.push_str(&articles(&loaded).iter().map(|d| d.article.to_lowercase()).collect::<Vec<_>>().join("\n"));
        corpus.push('\n');
        systems.push(system_prompt(&loaded.world_name));
    }
    let shapes = ["Tell me about {n}.", "Who is {n}?", "What do the rules say about {n}?", "Where can I find {n}?"];
    for (i, name) in invented_names(&corpus, 8, 7).into_iter().enumerate() {
        let (q, sys) = (shapes[i % shapes.len()].replace("{n}", &name), &systems[i % systems.len()]);
        out.push(Question { id: format!("probe-m{i}"), category: "made_up", messages: vec![json!({"role": "system", "content": sys}), json!({"role": "user", "content": q})], question: q, reference: None });
    }
    for (i, q) in GENERAL.iter().enumerate() {
        let sys = &systems[i % systems.len()];
        out.push(Question { id: format!("probe-g{i}"), category: "general", messages: vec![json!({"role": "system", "content": sys}), json!({"role": "user", "content": q})], question: q.to_string(), reference: None });
    }
    let mut seen = HashSet::new();
    out.retain(|q| seen.insert(q.id.clone()));
    Ok(out)
}

/// A single free question (side-by-side chat).
pub fn custom(world: &str, system: Option<&str>, question: &str) -> Question {
    let sys = system.map(str::to_string).filter(|s| !s.trim().is_empty()).unwrap_or_else(|| system_prompt(world));
    Question { id: "custom".into(), category: "lore", messages: vec![json!({"role": "system", "content": sys}), json!({"role": "user", "content": question})], question: question.to_string(), reference: None }
}
