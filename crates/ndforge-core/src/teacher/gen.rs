//! The teacher pipeline: jobs (one per article piece and kind) -> candidate examples -> judge -> draft. Every model call is cached in
//! `datasets/teacher_cache/`, so a stopped or repeated run only pays for what is new.
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::client::{extract_json, Teacher};
use crate::dataset::builders::{hash_hex, shorten, system_prompt, Example, Message, Status};
use crate::dataset::chunk::chunk_article;
use crate::dataset::docs::{articles, load, Doc, Loaded};
use crate::dataset::store;
use crate::error::{Error, Result};
use crate::project::Project;

/// (id, name for the page, what it teaches)
pub const KINDS: [(&str, &str, &str); 8] = [
    ("t_deeper", "Deeper questions", "Why and how questions that need several facts from an article."),
    ("t_rules", "Rules practice", "A concrete game situation and the ruling, with the steps."),
    ("t_history", "Campaign history", "What happened, who took part, what followed."),
    ("t_npc", "NPC role-play", "Talking as an NPC that has a speaking style, without revealing GM secrets."),
    ("t_lore", "Lore search", "Questions answered from a block of lore that is handed over, the way nd-world does it."),
    ("t_missing", "Not in the lore", "Questions about things that do not exist: the answer says so instead of inventing."),
    ("t_general", "General chat", "A little everyday game-master help, so the model does not forget how to talk."),
    ("t_style", "House style", "Answers that follow the world's own standing AI instructions."),
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Options {
    pub kinds: Vec<String>,
    /// examples asked for per piece of text
    pub per_source: u8,
    /// general chat as a share of all teacher examples, in percent (0-30)
    pub general_percent: u8,
    /// check every answer against its source with a second pass
    pub judge: bool,
    /// also drop answers the judge calls "partly" supported (otherwise they are kept and flagged)
    pub drop_partly: bool,
    /// only the first N pieces of text (to try the teacher out cheaply)
    pub limit: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options { kinds: KINDS.iter().map(|k| k.0.to_string()).collect(), per_source: 2, general_percent: 10, judge: true, drop_partly: false, limit: None }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct Report {
    pub jobs: usize,
    pub jobs_done: usize,
    pub generated: usize,
    pub added: usize,
    pub supported: usize,
    pub partly: usize,
    pub unsupported: usize,
    pub unchecked: usize,
    pub rejected: usize,
    pub failed_calls: usize,
    pub first_error: Option<String>,
    pub stopped: bool,
}

#[derive(Clone)]
struct Job {
    kind: &'static str,
    source: String,
    title: String,
    /// what the teacher reads (and the judge checks against)
    text: String,
    /// the world-lore system prompt prefix for "lore search" (nd-world shape), else None
    context: Option<String>,
}

fn cache_path(p: &Project) -> PathBuf {
    store::teacher_cache_dir(p)
}

/// One cached call: the same model, prompts and temperature give the stored answer.
fn ask(t: &Teacher, p: &Project, system: &str, user: &str, temp: f64, max_tokens: u32) -> Result<String> {
    let key = hash_hex(&[&t.cfg.model, system, user, &format!("{temp}")]);
    let file = cache_path(p).join(format!("{key}.txt"));
    if let Ok(text) = std::fs::read_to_string(&file) {
        return Ok(text);
    }
    let text = t.ask(system, user, temp, max_tokens)?;
    if text.trim().is_empty() {
        return Err(Error::new("The teacher returned an empty answer."));
    }
    let _ = std::fs::create_dir_all(cache_path(p));
    let _ = std::fs::write(&file, &text);
    Ok(text)
}

const WRITER: &str = "You write training examples for a small assistant that helps a game master run a tabletop role-playing world. You are careful: you use only what the source text says and never add facts of your own. Answer with JSON only, no commentary.";
const JUDGE: &str = "You are a strict fact-checker. You compare an answer with its source text. Answer with JSON only.";

const SECRET_RULE: &str = "Text written as \"(GM secret: ...)\" is secret from the players: an answer may use it, but must say that it is a GM secret.";

fn user_prompt(job: &Job, n: u8, world: &str) -> String {
    let head = format!("World: {world}\n\nSOURCE TEXT:\n{}\n\n", job.text);
    let shape = "{\"items\":[{\"question\":\"...\",\"answer\":\"...\"}]}";
    match job.kind {
        "t_rules" => format!("{head}Write {n} different practice items. Each question describes a concrete game situation (who does what, with which numbers) that these rules decide. Each answer gives the correct ruling according to the source text only, with the steps, in 2-6 sentences. Do not use rules that are not in the source text.\nOutput: {shape}"),
        "t_history" => format!("{head}Write {n} different question/answer pairs a game master might ask about what happened: who took part, what was decided, what followed. Answers are 2-5 sentences and use only the source text. {SECRET_RULE}\nOutput: {shape}"),
        "t_lore" => format!("{head}Write {n} different question/answer pairs a game master might ask that can be answered from this lore. Answers use only the lore above, in 2-5 sentences. {SECRET_RULE}\nOutput: {shape}"),
        "t_npc" => format!("{head}The NPC in this text is {}. Write {n} different short conversations between a player and {} (1-2 turns each). The NPC answers in character, following the speaking style given in the text, and knows only what the NPC would know. The NPC never reveals GM secrets.\nOutput: {{\"items\":[{{\"turns\":[{{\"player\":\"...\",\"npc\":\"...\"}}]}}]}}", job.title, job.title),
        "t_missing" => format!("{head}Write {n} questions a game master might ask about something that sounds like it belongs next to this text but does NOT exist in it: an invented person, place, item or rule with a made-up name. Give the invented name separately.\nOutput: {{\"items\":[{{\"question\":\"...\",\"invented_name\":\"...\"}}]}}"),
        "t_style" => format!("World: {world}\n\nSTANDING INSTRUCTION FROM THE GAME MASTER ({}):\n{}\n\nWrite {n} different questions a game master might ask the assistant, with answers that follow this instruction exactly. Do not invent facts about the world; keep answers general enough to be true in any world, in 2-5 sentences.\nOutput: {shape}", job.title, job.text),
        _ => format!("{head}Write {n} different question/answer pairs a game master might ask about this text. Each question needs understanding or combining details (why, how, what follows from this, how two things compare), not just copying one sentence. Answers are 2-6 sentences and use only the source text. {SECRET_RULE}\nOutput: {shape}"),
    }
}

fn items(v: &Value) -> Vec<Value> {
    v["items"].as_array().cloned().or_else(|| v.as_array().cloned()).unwrap_or_default()
}

fn text_of(v: &Value, k: &str) -> String {
    v[k].as_str().unwrap_or("").trim().to_string()
}

/// The standing direction of an NPC in nd-world's own words, so the practice matches how the model is used.
fn npc_system(world: &str, l: &Loaded, name: &str) -> Option<String> {
    let e = l.entities.iter().find(|e| e.name == name)?;
    let rp = crate::dataset::markup::clean(&e.roleplay_raw);
    let mut s = format!("You are {} ({}) in the tabletop world \"{world}\". Speak as this character to the players.\n", e.name, e.kind);
    if !e.summary().is_empty() {
        s += &format!("Summary: {}\n", e.summary());
    }
    if !e.tags.is_empty() {
        s += &format!("Tags: {}\n", e.tags.join(", "));
    }
    if !e.custom_fields.is_empty() {
        s += "Custom fields:\n";
        for (k, v) in e.custom_fields.iter().take(12) {
            s += &format!("- {k}: {v}\n");
        }
    }
    if !rp.is_empty() {
        s += &format!("\nHow you speak and behave (standing direction from the GM — follow it over any general instinct):\n{rp}\n");
    }
    Some(s.trim_end().to_string())
}

/// The text of an article without anything secret: paragraphs that mention a GM secret or hidden notes are left out.
fn without_secrets(article: &str) -> String {
    article.split("\n\n").filter(|para| !para.contains("GM secret") && !para.contains("hidden from the players") && !para.contains("hidden from players") && !para.contains("note hidden")).collect::<Vec<_>>().join("\n\n")
}

/// nd-world's "Relevant world lore" block for one entity and the entities it is connected to.
fn lore_context(l: &Loaded, entity: usize) -> String {
    let e = &l.entities[entity];
    let line = |x: &crate::dataset::docs::Entity, with_body: bool| {
        let sub = if x.subtype.is_empty() { String::new() } else { format!(" ({})", x.subtype) };
        let mut s = format!("- [{}] {}{}: {}", x.kind, x.name, sub, x.summary());
        let body = x.body();
        if with_body && !body.is_empty() {
            s += &format!("\n  {}", shorten(&body, 600).replace('\n', "\n  "));
        }
        s
    };
    let mut lines = vec![line(e, true)];
    let mut related: Vec<&str> = Vec::new();
    for r in &l.relations {
        let other = if r.source == e.name { &r.target } else if r.target == e.name { &r.source } else { continue };
        if !related.contains(&other.as_str()) {
            related.push(other);
        }
    }
    for name in related.into_iter().take(3) {
        if let Some(x) = l.entities.iter().find(|x| x.name == *name) {
            lines.push(line(x, false));
        }
    }
    lines.join("\n")
}

fn with_context(ctx: &str, system: &str) -> String {
    format!("Relevant world lore and notes (for accuracy only — don't invent beyond what's here):\n{ctx}\n\n{system}")
}

fn chat(builder: &str, job: &Job, system: String, turns: Vec<(String, String)>) -> Example {
    let first = turns.first().map(|t| t.0.clone()).unwrap_or_default();
    let mut messages = vec![Message { role: "system".into(), content: system }];
    for (u, a) in turns {
        messages.push(Message { role: "user".into(), content: u });
        messages.push(Message { role: "assistant".into(), content: a });
    }
    Example {
        id: hash_hex(&[builder, &job.source, &first])[..12].to_string(), builder: builder.to_string(), source: job.source.clone(), source_title: job.title.clone(),
        text: None, messages, status: Status::Keep, edited: false, check: None, note: None,
    }
}

const NOT_IN_LORE: [&str; 4] = [
    "I can't find \"{n}\" anywhere in the lore I have, so I won't make something up. If you want it to exist, tell me what it should be and we can work it out together.",
    "\"{n}\" isn't in this world's lore or rules as far as I know. I'd rather say so than invent it. Do you want to create it?",
    "There is nothing about \"{n}\" in the lore. I can't tell you about it without inventing details; if it is new, give me the idea and I'll help flesh it out.",
    "I don't have anything on \"{n}\". It isn't in the lore I was trained on, so any answer would be invented.",
];

/// Candidate examples a job produced, each with the text the judge should check it against (None: not checkable).
fn candidates(t: &Teacher, p: &Project, l: &Loaded, corpus: &str, job: &Job, opt: &Options, rep: &mut Report) -> Result<Vec<(Example, Option<String>)>> {
    let world = &l.world_name;
    let n = opt.per_source.clamp(1, 4);
    let raw = ask(t, p, WRITER, &user_prompt(job, n, world), 0.7, 3000)?;
    let Some(v) = extract_json(&raw) else {
        rep.rejected += 1;
        return Ok(vec![]);
    };
    let mut out = Vec::new();
    for it in items(&v).into_iter().take(n as usize + 1) {
        match job.kind {
            "t_npc" => {
                let turns: Vec<(String, String)> = it["turns"].as_array().cloned().unwrap_or_default().iter().map(|x| (text_of(x, "player"), text_of(x, "npc"))).filter(|(a, b)| !a.is_empty() && !b.is_empty()).take(2).collect();
                let Some(sys) = npc_system(world, l, &job.title) else { continue };
                if turns.is_empty() {
                    rep.rejected += 1;
                    continue;
                }
                out.push((chat("t_npc", job, sys, turns), Some(job.text.clone())));
            }
            "t_missing" => {
                let (q, name) = (text_of(&it, "question"), text_of(&it, "invented_name"));
                // the invented thing must really be absent from the whole world
                if q.is_empty() || name.chars().count() < 3 || corpus.contains(&name.to_lowercase()) {
                    rep.rejected += 1;
                    continue;
                }
                let answer = NOT_IN_LORE[crate::dataset::builders::hash_num(&format!("{}{q}", job.source)) % NOT_IN_LORE.len()].replace("{n}", &name);
                let base = system_prompt(world);
                let with_ctx = crate::dataset::builders::hash_num(&q) % 2 == 0 && job.context.is_some();
                let system = if with_ctx { with_context(job.context.as_deref().unwrap_or(""), &base) } else { base };
                out.push((chat("t_missing", job, system, vec![(q, answer)]), None));
            }
            _ => {
                let (q, a) = (text_of(&it, "question"), text_of(&it, "answer"));
                if q.is_empty() || a.is_empty() || q.chars().count() > 500 || a.chars().count() > 3500 {
                    rep.rejected += 1;
                    continue;
                }
                let base = system_prompt(world);
                let system = if job.kind == "t_lore" { with_context(job.context.as_deref().unwrap_or(""), &base) } else { base };
                let check = if job.kind == "t_style" { None } else if job.kind == "t_lore" { job.context.clone() } else { Some(job.text.clone()) };
                out.push((chat(job.kind, job, system, vec![(q, a)]), check));
            }
        }
    }
    Ok(out)
}

/// The judge's verdict on one example: ("supported" | "partly" | "unsupported", reason).
fn judge(t: &Teacher, p: &Project, ex: &Example, source: &str) -> Result<(String, String)> {
    let q = &ex.messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default();
    let a = &ex.messages.last().map(|m| m.content.clone()).unwrap_or_default();
    let npc = if ex.builder == "t_npc" { "The answer is spoken in character; judge only the facts it states, not the style.\n" } else { "" };
    let user = format!("SOURCE:\n{source}\n\nQUESTION:\n{q}\n\nANSWER:\n{a}\n\n{npc}Is every factual claim in the ANSWER supported by the SOURCE (or a plain consequence of it)? \"supported\": yes; \"partly\": mostly, but one detail is not in the source; \"unsupported\": it contradicts the source or invents facts. Do not reward answers that sound good but add facts.\nOutput: {{\"verdict\":\"supported\"|\"partly\"|\"unsupported\",\"reason\":\"one short sentence\"}}");
    let raw = ask(t, p, JUDGE, &user, 0.0, 400)?;
    let v = extract_json(&raw).ok_or_else(|| Error::new("The judge's answer was not readable."))?;
    let verdict = v["verdict"].as_str().unwrap_or("").to_lowercase();
    if !["supported", "partly", "unsupported"].contains(&verdict.as_str()) {
        return Err(Error::new("The judge's verdict was not one of supported / partly / unsupported."));
    }
    Ok((verdict, text_of(&v, "reason")))
}

fn make_jobs(l: &Loaded, docs: &[Doc], opt: &Options) -> Vec<Job> {
    let on = |k: &str| opt.kinds.iter().any(|x| x == k);
    let mut jobs = Vec::new();
    for d in docs {
        let entity = d.source.strip_prefix("entity:").and_then(|id| id.parse::<i64>().ok()).and_then(|id| l.entities.iter().position(|e| e.id == id));
        if d.article.chars().count() < 200 {
            continue;
        }
        let pieces: Vec<String> = chunk_article(&d.title, &d.article, 900).into_iter().take(3).collect();
        let kind: Option<&'static str> = match d.kind.as_str() {
            "rules" => Some("t_rules").filter(|k| on(k)),
            "session" => Some("t_history").filter(|k| on(k)),
            _ => Some("t_deeper").filter(|k| on(k)),
        };
        if let Some(kind) = kind {
            for piece in &pieces {
                jobs.push(Job { kind, source: d.source.clone(), title: d.title.clone(), text: piece.clone(), context: None });
            }
        }
        if let Some(ei) = entity {
            let e = &l.entities[ei];
            if on("t_lore") {
                let ctx = lore_context(l, ei);
                jobs.push(Job { kind: "t_lore", source: d.source.clone(), title: d.title.clone(), text: ctx.clone(), context: Some(ctx) });
            }
            if on("t_missing") && crate::dataset::builders::hash_num(&d.source) % 2 == 0 {
                jobs.push(Job { kind: "t_missing", source: d.source.clone(), title: d.title.clone(), text: shorten(&d.article, 1500), context: Some(lore_context(l, ei)) });
            }
            if on("t_npc") && e.visible && !e.roleplay_raw.trim().is_empty() {
                let card = without_secrets(&d.article);
                jobs.push(Job { kind: "t_npc", source: d.source.clone(), title: d.title.clone(), text: card, context: None });
            }
        } else if on("t_missing") && d.kind == "rules" && crate::dataset::builders::hash_num(&d.source) % 2 == 0 {
            jobs.push(Job { kind: "t_missing", source: d.source.clone(), title: d.title.clone(), text: shorten(&d.article, 1500), context: None });
        }
    }
    if on("t_style") {
        for ins in &l.instructions {
            let (title, content) = (text_of(ins, "title"), crate::dataset::markup::clean(ins["content"].as_str().unwrap_or("")));
            if !content.is_empty() {
                jobs.push(Job { kind: "t_style", source: format!("instruction:{}", ins["id"]), title: if title.is_empty() { "Instruction".into() } else { title }, text: content, context: None });
            }
        }
    }
    if let Some(limit) = opt.limit {
        jobs.truncate(limit);
    }
    jobs
}

const TOPICS: [&str; 12] = ["designing encounters and difficulty", "naming people, places and items", "pacing a session", "handling a player who dominates the table", "improvising when players leave the plot", "describing scenes vividly", "dice odds and probability", "building a mystery with clues", "rewarding players and loot", "running combat smoothly", "making villains memorable", "recapping the last session"];

/// Runs the whole pipeline and merges the new examples into the draft. `emit` gets {"kind":"progress"|"log", ...}. Stops cleanly when `cancel` is set.
pub fn run(p: &Project, t: &Teacher, opt: &Options, cancel: &AtomicBool, emit: &mut dyn FnMut(Value)) -> Result<Value> {
    if !t.cfg.is_set() {
        return Err(Error::new("Choose the teacher first: its address and the model name."));
    }
    let bundles = p.bundles();
    if bundles.is_empty() {
        return Err(Error::new("Import the world data first (Step 1)."));
    }
    // every world of the project gets its own pieces of text; sources carry the world's suffix so that worlds never mix up
    let mut worlds: Vec<Loaded> = Vec::new();
    let mut jobs: Vec<(usize, Job)> = Vec::new();
    let mut corpus = String::new();
    for (wi, (w, bundle)) in bundles.iter().enumerate() {
        let l = load(bundle)?;
        let docs = articles(&l);
        for d in &docs {
            corpus.push_str(&d.article.to_lowercase());
            corpus.push('\n');
        }
        for mut j in make_jobs(&l, &docs, opt) {
            j.source.push_str(&w.suffix);
            jobs.push((wi, j));
        }
        worlds.push(l);
    }
    if let Some(limit) = opt.limit {
        jobs.truncate(limit);
    }
    let l = &worlds[0];
    let mut rep = Report { jobs: jobs.len(), ..Default::default() };
    let existing: HashSet<String> = store::load_examples(p)?.into_iter().map(|e| e.id).collect();
    let mut fresh: Vec<Example> = Vec::new();
    let mut consecutive_failures = 0;
    emit(json!({"kind": "log", "text": format!("{} pieces of text to work on", jobs.len())}));

    let handle = |ex: Example, source: Option<String>, rep: &mut Report, t: &Teacher, p: &Project| -> Result<Option<Example>> {
        rep.generated += 1;
        if existing.contains(&ex.id) {
            return Ok(None);
        }
        let mut ex = ex;
        match (opt.judge, source) {
            (true, Some(src)) => match judge(t, p, &ex, &src) {
                Ok((verdict, reason)) => {
                    match verdict.as_str() {
                        "supported" => rep.supported += 1,
                        "partly" => rep.partly += 1,
                        _ => rep.unsupported += 1,
                    }
                    if verdict == "unsupported" || (verdict == "partly" && opt.drop_partly) {
                        ex.status = Status::Drop;
                    }
                    ex.check = Some(verdict);
                    ex.note = Some(reason).filter(|r| !r.is_empty());
                }
                Err(e) => {
                    rep.failed_calls += 1;
                    rep.first_error.get_or_insert(e.0.clone());
                    rep.unchecked += 1;
                    ex.check = Some("unchecked".into());
                    ex.note = Some("The judge could not check this one.".into());
                }
            },
            _ => {
                rep.unchecked += 1;
                ex.check = Some("unchecked".into());
            }
        }
        Ok(Some(ex))
    };

    for (i, (wi, job)) in jobs.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            rep.stopped = true;
            break;
        }
        emit(json!({"kind": "progress", "done": i, "total": jobs.len(), "text": format!("{}: {}", KINDS.iter().find(|k| k.0 == job.kind).map(|k| k.1).unwrap_or(job.kind), job.title)}));
        match candidates(t, p, &worlds[*wi], &corpus, job, opt, &mut rep) {
            Ok(cands) => {
                consecutive_failures = 0;
                for (ex, src) in cands {
                    if let Some(ex) = handle(ex, src, &mut rep, t, p)? {
                        rep.added += 1;
                        fresh.push(ex);
                    }
                }
            }
            Err(e) => {
                rep.failed_calls += 1;
                consecutive_failures += 1;
                rep.first_error.get_or_insert(e.0.clone());
                if consecutive_failures >= 3 {
                    store::merge_teacher(p, std::mem::take(&mut fresh))?;
                    return Err(Error::new(format!("The teacher keeps failing, so I stopped (what was made so far is saved). {}", e.0)));
                }
            }
        }
        rep.jobs_done = i + 1;
        if i % 5 == 4 {
            store::merge_teacher(p, std::mem::take(&mut fresh))?;
        }
    }

    // a little general chat, as a share of everything the teacher made
    if opt.kinds.iter().any(|k| k == "t_general") && opt.general_percent > 0 && !rep.stopped {
        let have = store::load_examples(p)?.iter().filter(|e| e.builder.starts_with("t_") && e.builder != "t_general").count() + fresh.len();
        let pct = opt.general_percent.min(30) as usize;
        let want = ((have * pct).div_ceil(100 - pct)).min(200);
        let already = store::load_examples(p)?.iter().filter(|e| e.builder == "t_general").count();
        let mut made = already;
        let mut round = 0;
        while made < want && round < 60 && !cancel.load(Ordering::SeqCst) {
            let topic = TOPICS[round % TOPICS.len()];
            round += 1;
            let job = Job { kind: "t_general", source: format!("general:{topic}"), title: topic.to_string(), text: String::new(), context: None };
            let prompt = format!("Topic: {topic} (variation {})\n\nWrite 5 different short question/answer pairs of general help for a tabletop role-playing game master on this topic. They must not depend on any particular world, character or rules system. Answers are 2-5 sentences, practical and friendly.\nOutput: {{\"items\":[{{\"question\":\"...\",\"answer\":\"...\"}}]}}", round / TOPICS.len() + 1);
            emit(json!({"kind": "progress", "done": jobs.len(), "total": jobs.len(), "text": format!("General chat: {topic}")}));
            let Ok(raw) = ask(t, p, WRITER, &prompt, 0.8, 2500) else { rep.failed_calls += 1; continue };
            let Some(v) = extract_json(&raw) else { rep.rejected += 1; continue };
            for it in items(&v) {
                let (q, a) = (text_of(&it, "question"), text_of(&it, "answer"));
                if q.is_empty() || a.is_empty() || made >= want {
                    continue;
                }
                let mut ex = chat("t_general", &job, system_prompt(&l.world_name), vec![(q, a)]);
                rep.generated += 1;
                if existing.contains(&ex.id) || fresh.iter().any(|f| f.id == ex.id) {
                    continue;
                }
                ex.check = Some("unchecked".into());
                rep.unchecked += 1;
                rep.added += 1;
                made += 1;
                fresh.push(ex);
            }
        }
    }

    store::merge_teacher(p, fresh)?;
    if t.cfg.unload_after {
        // best effort: free the graphics memory so training can have it
        let _ = crate::export::ollama::Ollama::new(&t.cfg.ollama_base()).unload(&t.cfg.model);
    }
    emit(json!({"kind": "progress", "done": jobs.len(), "total": jobs.len(), "text": "Finished"}));
    Ok(serde_json::to_value(&rep)?)
}
