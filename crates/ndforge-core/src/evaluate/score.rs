//! Scoring answers. No model is needed for the basic scores: word overlap with the reference answer (lore, rules), whether a question about
//! something that does not exist was declined (made-up answers), and whether general answers are sane. A teacher can judge lore and rules
//! instead (see `runs::execute`).
use std::collections::HashSet;

use serde_json::{json, Value};

const STOP: [&str; 40] = [
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "has", "have", "had", "its", "his", "her", "their", "they", "them", "she", "him",
    "but", "not", "you", "your", "can", "will", "who", "what", "which", "into", "than", "then", "also", "been", "being", "does", "did", "about", "there",
];

/// The distinct meaningful words of a text, lower case (accented letters count).
pub fn words(text: &str) -> HashSet<String> {
    text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() >= 3 && !STOP.contains(w)).map(String::from).collect()
}

/// How much of the reference the answer says, 0 to 1: the harmonic mean of the answer's precision (its words that are in the reference) and the
/// reference's recall, where the recall counts at most 30 reference words (answers are cut at a few hundred tokens, so a long reference cannot
/// be fully covered). Identical text scores 1, unrelated text 0.
pub fn overlap(answer: &str, reference: &str) -> f64 {
    let (a, r) = (words(answer), words(reference));
    if a.is_empty() || r.is_empty() {
        return 0.0;
    }
    let hit = a.intersection(&r).count() as f64;
    let (p, rec) = (hit / a.len() as f64, (hit / (r.len().min(30)) as f64).min(1.0));
    if p == 0.0 || rec == 0.0 { 0.0 } else { 2.0 * p * rec / (p + rec) }
}

const DECLINE: [&str; 22] = [
    "not in the lore", "isn't in", "is not in", "don't have", "do not have", "can't find", "cannot find", "no information", "not aware", "no record",
    "doesn't appear", "does not appear", "don't know", "do not know", "not mentioned", "isn't mentioned", "nothing about", "unable to find",
    "not part of", "couldn't find", "could not find", "not something i",
];

/// Does the answer say the thing is not in the lore (instead of inventing)?
pub fn declines(answer: &str) -> bool {
    let a = answer.to_lowercase().replace('’', "'");
    DECLINE.iter().any(|p| a.contains(p))
}

/// A general answer that is a real answer: long enough, not a repetition loop, not a "not in the lore" refusal.
pub fn sane(answer: &str) -> bool {
    let a = answer.trim();
    if a.chars().count() < 20 || declines(a) {
        return false;
    }
    let toks: Vec<&str> = a.split_whitespace().collect();
    if toks.len() >= 12 {
        let distinct: HashSet<&str> = toks.iter().copied().collect();
        if (distinct.len() as f64) / (toks.len() as f64) < 0.3 {
            return false;
        }
    }
    true
}

/// One question's score for one answer, 0 to 1 (for made-up: 1 = invented something, 0 = declined; for general: 1 = sane).
pub fn score_one(category: &str, answer: &str, reference: Option<&str>) -> f64 {
    match category {
        "made_up" => if declines(answer) { 0.0 } else { 1.0 },
        "general" => if sane(answer) { 1.0 } else { 0.0 },
        _ => overlap(answer, reference.unwrap_or("")),
    }
}

pub const CATEGORIES: [(&str, &str); 4] = [("lore", "Lore questions"), ("rules", "Rules questions"), ("made_up", "Made-up answers"), ("general", "General chat")];

/// Per-category means over `scores` ([(category, score)]).
pub fn summarize(scores: &[(String, f64)]) -> Value {
    let mut out = serde_json::Map::new();
    for (cat, _) in CATEGORIES {
        let v: Vec<f64> = scores.iter().filter(|(c, _)| c == cat).map(|(_, s)| *s).collect();
        if !v.is_empty() {
            out.insert(cat.to_string(), json!({"n": v.len(), "mean": (v.iter().sum::<f64>() / v.len() as f64 * 1000.0).round() / 1000.0}));
        }
    }
    Value::Object(out)
}

/// The plain verdict. `rows` = [(category, original score, trained score)].
pub fn verdict(rows: &[(String, f64, f64)]) -> Value {
    let share = |cat: &str, f: &dyn Fn(f64, f64) -> bool| -> Option<(usize, usize)> {
        let r: Vec<_> = rows.iter().filter(|(c, _, _)| c == cat).collect();
        if r.is_empty() { None } else { Some((r.iter().filter(|(_, o, t)| f(*o, *t)).count(), r.len())) }
    };
    let pct = |(k, n): (usize, usize)| (k as f64 / n as f64 * 100.0).round() as i64;
    let mean = |cat: &str, side: usize| -> Option<f64> {
        let v: Vec<f64> = rows.iter().filter(|(c, _, _)| c == cat).map(|(_, o, t)| if side == 0 { *o } else { *t }).collect();
        if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }
    };
    let mut lines = Vec::new();
    let mut headline = "Not enough lore questions to say.".to_string();
    let mut good = None;
    if let Some(b) = share("lore", &|o, t| t > o + 0.02) {
        let worse = share("lore", &|o, t| t < o - 0.02).map(pct).unwrap_or(0);
        headline = format!("Better than the original on {} % of lore questions (worse on {} %).", pct(b), worse);
        good = Some(pct(b) > worse);
    }
    if let Some(b) = share("rules", &|o, t| t > o + 0.02) {
        lines.push(format!("Rules questions: better on {} % of {}.", pct(b), b.1));
    }
    if let (Some(o), Some(t)) = (mean("made_up", 0), mean("made_up", 1)) {
        lines.push(format!("Made-up answers: the original invented an answer for {} % of questions about things that do not exist, the trained model for {} %.", (o * 100.0).round(), (t * 100.0).round()));
        if t > 0.5 && t >= o {
            lines.push("The trained model often invents lore. More \"not in the lore\" examples (Step 2, teacher) help.".to_string());
        }
    }
    if let (Some(o), Some(t)) = (mean("general", 0), mean("general", 1)) {
        lines.push(format!("General chat: {} % sensible answers before, {} % after.", (o * 100.0).round(), (t * 100.0).round()));
        if t + 0.2 < o {
            lines.push("General chat got worse: try fewer steps or a lower learning rate, or add general chat examples.".to_string());
        }
    }
    json!({"headline": headline, "lines": lines, "better": good})
}
