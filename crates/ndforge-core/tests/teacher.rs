//! The teacher pipeline against a mock OpenAI-compatible server: prompts, parsing, the judge, caching, secrets, held-out handling.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ndforge_core::bundle::{open_bundle, write_bundle, Member};
use ndforge_core::dataset::builders::{Recipe, Status};
use ndforge_core::dataset::store;
use ndforge_core::project::Project;
use ndforge_core::sample;
use ndforge_core::teacher::{extract_json, run, strip_thinking, Options, Teacher, TeacherConfig};
use serde_json::{json, Value};
use tempfile::tempdir;

#[derive(Default)]
struct Seen {
    calls: Vec<(String, String)>,
    auth: Vec<String>,
}

/// A fake teacher. `fail` answers every call with a 500. Returns (base url, what it saw).
fn mock(fail: bool) -> (String, Arc<Mutex<Seen>>) {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
    let seen = Arc::new(Mutex::new(Seen::default()));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for mut req in server.incoming_requests() {
            let mut body = String::new();
            req.as_reader().read_to_string(&mut body).unwrap();
            let v: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let (sys, user) = (v["messages"][0]["content"].as_str().unwrap_or("").to_string(), v["messages"][1]["content"].as_str().unwrap_or("").to_string());
            {
                let mut g = s2.lock().unwrap();
                g.calls.push((sys.clone(), user.clone()));
                if let Some(h) = req.headers().iter().find(|h| h.field.equiv("Authorization")) {
                    g.auth.push(h.value.to_string());
                }
            }
            if fail {
                let _ = req.respond(tiny_http::Response::from_string("{\"error\":{\"message\":\"boom\"}}").with_status_code(500));
                continue;
            }
            let content = answer(&sys, &user);
            let out = json!({"choices": [{"message": {"role": "assistant", "content": content}}]});
            let _ = req.respond(tiny_http::Response::from_string(out.to_string()).with_header(tiny_http::Header::from_bytes("Content-Type", "application/json").unwrap()));
        }
    });
    (url, seen)
}

fn answer(sys: &str, user: &str) -> String {
    if sys.contains("fact-checker") {
        let a = user.split("ANSWER:\n").nth(1).unwrap_or("");
        let verdict = if a.contains("ZZUNSUPPORTED") { "unsupported" } else if a.contains("ZZPARTLY") { "partly" } else { "supported" };
        return format!("<think>hmm, let me compare</think>\n```json\n{{\"verdict\":\"{verdict}\",\"reason\":\"checked\"}}\n```");
    }
    let title = user.split("SOURCE TEXT:\n# ").nth(1).and_then(|t| t.lines().next()).unwrap_or("").to_string();
    let flavour = |i: usize| -> String {
        if title.contains("Bad") && i == 1 { " ZZUNSUPPORTED invented detail.".into() } else if title.contains("Partly") && i == 0 { " ZZPARTLY extra detail.".into() } else { String::new() }
    };
    let items: Vec<Value> = if user.contains("practice items") {
        (0..2).map(|i| json!({"question": format!("Rules situation {i} in {title}?"), "answer": format!("Under {title}, roll the pool and count sixes.{}", flavour(i))})).collect()
    } else if user.contains("short conversations") {
        (0..2).map(|i| json!({"turns": [{"player": format!("Greetings {i}"), "npc": format!("Hmph. State your business {i}.")}, {"player": "Where is the harbour?", "npc": "Dockside. Go."}]})).collect()
    } else if user.contains("does NOT exist") {
        vec![json!({"question": "What is Zorblax Prime?", "invented_name": "Zorblax Prime"}), json!({"question": "Who runs the Dockside?", "invented_name": "Dockside"})]
    } else if user.starts_with("Topic:") {
        let topic = user.lines().next().unwrap_or("").to_string();
        (0..5).map(|i| json!({"question": format!("{topic} question {i}?"), "answer": format!("A general tip number {i}.")})).collect()
    } else if user.contains("STANDING INSTRUCTION") {
        (0..2).map(|i| json!({"question": format!("Style question {i}?"), "answer": format!("Short and dry answer {i}.")})).collect()
    } else {
        (0..2).map(|i| json!({"question": format!("Why does {title} matter ({i})?"), "answer": format!("{title} matters because of its role.{}", flavour(i))})).collect()
    };
    json!({"items": items}).to_string()
}

fn long(topic: &str) -> String {
    format!("# {topic}\n\n{}", (0..8).map(|i| format!("{topic} has a long history, part {i}, with many details about how it works and who it affects in the town.")).collect::<Vec<_>>().join("\n\n"))
}

fn ent(id: i64, kind: &str, name: &str, rp: &str, visible: bool, body: &str) -> Value {
    json!({"id": id, "kind": kind, "subtype": "", "name": name, "aliases": [], "folder": "", "tags": [], "summary": format!("{name} in short."), "body": body,
           "template": null, "custom_fields": {}, "roleplay_personality": rp, "visible_to_players": visible, "group_ids": [], "rag_priority": false, "updated_at": "2026-11-01T10:00:00"})
}

fn project(d: &std::path::Path) -> Project {
    let mut p = Project::create(&d.join("projects"), "T").unwrap();
    let f = d.join("t.ndtrain");
    let members = vec![
        ("entities.jsonl", Member::Records(vec![
            ent(1, "character", "Gruff Captain", "gruff, terse", true, &format!("{} [gmonly]He fears fire.[/gmonly]", long("Captain"))),
            ent(2, "location", "Dockside", "", true, &long("Dockside")),
            ent(3, "item", "The Bad Rumour", "", true, &long("Bad rumour")),
            ent(4, "item", "The Partly Ring", "", true, &long("Partly ring")),
            ent(5, "character", "Hidden Duke", "oily", false, &long("Duke")),
        ])),
        ("relations.jsonl", Member::Records(vec![json!({"id": 1, "source_id": 1, "source_name": "Gruff Captain", "target_id": 2, "target_name": "Dockside", "relation": "located_in"})])),
        ("rules.md", Member::Text(format!("# Rules\n\n## Dice pool\n\n{}", long("Dice pool").replace("# Dice pool\n\n", "")))),
        ("sessions.jsonl", Member::Records(vec![json!({"id": 1, "session_num": 1, "session_date": "2026-10-01", "title": "Arrival", "summary": long("Arrival").replace("# Arrival\n\n", ""), "npcs": [], "loot": []})])),
        ("ai_instructions.jsonl", Member::Records(vec![json!({"id": 7, "title": "Be brief", "content": "Answer in at most two sentences. Never use bullet lists.", "applies_to_players": false})])),
    ];
    write_bundle(&f, &sample::world(), members, "t", "2026-11-01T12:00:00Z").unwrap();
    p.import_bundle(&open_bundle(f).unwrap()).unwrap();
    p
}

fn teacher(url: &str, key: Option<&str>) -> Teacher {
    let mut cfg = TeacherConfig::new(url, "teach-1", key.map(String::from));
    cfg.retry_ms = 1;
    Teacher::new(cfg)
}

fn go(p: &Project, t: &Teacher, o: &Options) -> Value {
    run(p, t, o, &AtomicBool::new(false), &mut |_| {}).unwrap()
}

#[test]
fn reading_the_teachers_words() {
    assert_eq!(strip_thinking("<think>a\nb</think>\nHello"), "Hello");
    assert_eq!(strip_thinking("no tags"), "no tags");
    assert_eq!(strip_thinking("x<think>never closed"), "x");
    assert_eq!(strip_thinking("reasoning...</think>answer"), "answer");
    assert_eq!(extract_json("{\"a\":1}").unwrap()["a"], 1);
    assert_eq!(extract_json("Sure!\n```json\n{\"a\":2}\n```\nDone").unwrap()["a"], 2);
    assert_eq!(extract_json("Here you go: {\"a\":3} hope that helps").unwrap()["a"], 3);
    assert_eq!(extract_json("[1,2]").unwrap()[1], 2);
    assert!(extract_json("no json here").is_none());
    let c = |u: &str| TeacherConfig::new(u, "m", None).endpoint();
    assert_eq!(c("localhost:11434"), "http://localhost:11434/v1/chat/completions");
    assert_eq!(c("http://x:1/v1/"), "http://x:1/v1/chat/completions");
    assert_eq!(c("https://api.example.com/v1"), "https://api.example.com/v1/chat/completions");
    assert_eq!(TeacherConfig::new("http://x:11434/v1", "m", None).ollama_base(), "http://x:11434");
}

#[test]
fn makes_checks_and_files_every_kind_of_example() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    store::generate_draft(&p, &Recipe::default()).unwrap();
    let (url, seen) = mock(false);
    let t = teacher(&url, Some("sk-test"));
    let rep = go(&p, &t, &Options::default());
    assert!(rep["added"].as_u64().unwrap() > 20, "{rep}");
    assert_eq!(rep["stopped"], false);
    assert!(seen.lock().unwrap().auth.iter().all(|a| a == "Bearer sk-test") && !seen.lock().unwrap().auth.is_empty());

    let ex = store::load_examples(&p).unwrap();
    let by = |b: &str| ex.iter().filter(|e| e.builder == b).collect::<Vec<_>>();
    for b in ["t_deeper", "t_rules", "t_history", "t_npc", "t_lore", "t_missing", "t_general", "t_style"] {
        assert!(!by(b).is_empty(), "no {b} examples: {rep}");
    }
    // the judge: unsupported dropped (with its reason), partly kept but flagged, supported kept
    let bad = ex.iter().find(|e| e.builder == "t_deeper" && e.messages.last().unwrap().content.contains("ZZUNSUPPORTED")).expect("unsupported example exists");
    assert_eq!((bad.status, bad.check.as_deref(), bad.note.as_deref()), (Status::Drop, Some("unsupported"), Some("checked")));
    let partly = ex.iter().find(|e| e.messages.last().is_some_and(|m| m.content.contains("ZZPARTLY"))).expect("partly example exists");
    assert_eq!((partly.status, partly.check.as_deref()), (Status::Keep, Some("partly")));
    assert!(rep["unsupported"].as_u64().unwrap() >= 1 && rep["partly"].as_u64().unwrap() >= 1 && rep["supported"].as_u64().unwrap() >= 5);
    // NPC role-play: nd-world's NPC prompt shape, the speaking style, and nothing secret went to the teacher
    let npc = by("t_npc");
    assert!(npc.iter().all(|e| e.source == "entity:1"), "the hidden duke must not be an NPC example");
    assert!(npc[0].messages[0].content.contains("How you speak and behave (standing direction from the GM") && npc[0].messages[0].content.contains("gruff, terse"));
    assert!(npc.iter().any(|e| e.messages.len() == 5), "two-turn conversations are kept");
    let calls = seen.lock().unwrap().calls.clone();
    let npc_prompt = calls.iter().find(|(_, u)| u.contains("short conversations")).unwrap();
    assert!(!npc_prompt.1.contains("fears fire") && !npc_prompt.1.contains("(GM secret:"), "{}", npc_prompt.1);
    // lore search: nd-world's context block, with a related entry and its [kind] line
    let lore = by("t_lore").into_iter().find(|e| e.source == "entity:1").unwrap();
    assert!(lore.messages[0].content.starts_with("Relevant world lore and notes (for accuracy only") && lore.messages[0].content.contains("- [character] Gruff Captain") && lore.messages[0].content.contains("- [location] Dockside"));
    // not in the lore: invented names only, and an invented name that exists in the world is refused
    let missing = by("t_missing");
    assert!(missing.iter().all(|e| e.messages.last().unwrap().content.contains("Zorblax Prime")));
    assert!(rep["rejected"].as_u64().unwrap() >= 1);
    // general chat is about a tenth, house style uses the world's instruction
    let (g, all) = (by("t_general").len(), ex.iter().filter(|e| e.builder.starts_with("t_")).count());
    assert!(g >= 2 && g * 100 <= all * 15, "general {g} of {all}");
    assert!(calls.iter().any(|(_, u)| u.contains("Never use bullet lists")));
    // every example is a complete conversation ending with the assistant
    for e in ex.iter().filter(|e| e.builder.starts_with("t_")) {
        assert_eq!(e.messages.first().unwrap().role, "system");
        assert_eq!(e.messages.last().unwrap().role, "assistant");
        assert!(e.check.is_some());
    }
}

#[test]
fn a_repeated_run_costs_nothing_and_decisions_survive() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    store::generate_draft(&p, &Recipe::default()).unwrap();
    let (url, seen) = mock(false);
    let t = teacher(&url, None);
    let o = Options::default();
    let first = go(&p, &t, &o);
    let n_calls = seen.lock().unwrap().calls.len();
    assert!(n_calls > 20);
    // the user drops one teacher example and edits another
    let ex = store::load_examples(&p).unwrap();
    let kept: Vec<_> = ex.iter().filter(|e| e.builder == "t_deeper" && e.status == Status::Keep).collect();
    store::set_status(&p, &[kept[0].id.clone()], Status::Drop).unwrap();
    store::edit(&p, &kept[1].id, "My own answer.").unwrap();
    let again = go(&p, &t, &o);
    assert_eq!(seen.lock().unwrap().calls.len(), n_calls, "cached: no new calls");
    assert_eq!(again["added"], 0, "{first} {again}");
    // regenerating the simple examples keeps the teacher ones and the decisions
    store::generate_draft(&p, &Recipe::default()).unwrap();
    let after = store::load_examples(&p).unwrap();
    assert_eq!(after.iter().filter(|e| e.builder.starts_with("t_")).count(), ex.iter().filter(|e| e.builder.starts_with("t_")).count());
    assert_eq!(after.iter().find(|e| e.id == kept[0].id).unwrap().status, Status::Drop);
    assert_eq!(after.iter().find(|e| e.id == kept[1].id).unwrap().messages.last().unwrap().content, "My own answer.");
    // a teacher example whose source vanished from the world is not carried over
    let summary = store::summary(&p).unwrap().unwrap();
    assert!(summary["builders"].as_array().unwrap().iter().any(|b| b["name"].as_str().unwrap().starts_with("Teacher: Deeper questions")));
    // clearing removes only teacher examples
    let n = store::clear_teacher(&p, Some("t_general"), false).unwrap();
    assert!(n >= 1 && store::load_examples(&p).unwrap().iter().all(|e| e.builder != "t_general"));
    store::clear_teacher(&p, None, true).unwrap();
    let rest = store::load_examples(&p).unwrap();
    assert!(!rest.is_empty() && rest.iter().all(|e| !e.builder.starts_with("t_")));
    assert!(!store::teacher_cache_dir(&p).exists());
}

#[test]
fn saved_versions_keep_general_chat_in_training_and_test_with_checked_questions() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    let (url, _) = mock(false);
    go(&p, &teacher(&url, None), &Options::default());
    let m = store::save_version(&p).unwrap();
    let vdir = p.folder.join("datasets").join("v1");
    let read = |f: &str| -> Vec<Value> { std::fs::read_to_string(vdir.join(f)).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect() };
    let (train, test) = (read("train.jsonl"), read("test.jsonl"));
    assert!(!test.is_empty());
    assert!(test.iter().all(|r| !r["source"].as_str().unwrap().starts_with("general:") && !r["source"].as_str().unwrap().starts_with("instruction:")));
    assert!(train.iter().any(|r| r["builder"] == "t_general") && train.iter().any(|r| r["builder"] == "t_style"));
    assert!(test.iter().any(|r| r["builder"].as_str().unwrap().starts_with("t_")), "held-out sources include teacher questions");
    // no dropped (unsupported) example reaches either file
    let text = format!("{train:?}{test:?}");
    assert!(!text.contains("ZZUNSUPPORTED"));
    assert!(m["builders"]["t_deeper"].as_u64().unwrap() >= 1);
}

#[test]
fn stop_trial_limit_and_failures() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let (url, seen) = mock(false);
    let t = teacher(&url, None);
    // stop button pressed before the first piece: nothing is made, nothing is asked
    let stop = AtomicBool::new(true);
    let rep = run(&p, &t, &Options::default(), &stop, &mut |_| {}).unwrap();
    assert_eq!((rep["stopped"].as_bool(), rep["added"].as_u64()), (Some(true), Some(0)));
    assert!(seen.lock().unwrap().calls.is_empty());
    // a trial on 2 pieces; kinds can be chosen; judging can be turned off
    let o = Options { limit: Some(2), kinds: vec!["t_deeper".into()], judge: false, ..Options::default() };
    let events = Mutex::new(Vec::new());
    let rep = run(&p, &t, &o, &AtomicBool::new(false), &mut |v| events.lock().unwrap().push(v)).unwrap();
    assert_eq!(rep["jobs"], 2);
    assert!(rep["added"].as_u64().unwrap() >= 2 && rep["unchecked"] == rep["added"], "{rep}");
    assert!(seen.lock().unwrap().calls.iter().all(|(s, _)| !s.contains("fact-checker")));
    assert!(events.lock().unwrap().iter().any(|e| e["kind"] == "progress" && e["total"] == 2));
    // a dead server stops the run with a clear message after three failures
    let (bad, _) = mock(true);
    let err = run(&p, &teacher(&bad, None), &Options { kinds: vec!["t_deeper".into()], ..Options::default() }, &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(err.0.contains("keeps failing") && err.0.contains("boom"), "{err}");
    // no teacher chosen / nothing imported
    assert!(run(&p, &Teacher::new(TeacherConfig::new("", "", None)), &Options::default(), &AtomicBool::new(false), &mut |_| {}).is_err());
    let empty = Project::create(&d.path().join("e"), "E").unwrap();
    assert!(run(&empty, &t, &Options::default(), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Import the world data"));
    let _ = Ordering::SeqCst;
    // an unreachable address names the problem
    let unreachable = Teacher::new({ let mut c = TeacherConfig::new("http://127.0.0.1:1", "m", None); c.retry_ms = 1; c });
    assert!(unreachable.check().unwrap_err().0.contains("did not answer"));
}

fn nd_example(id: &str, builder: &str, source: &str, answer: &str, status: &str, check: &str) -> Value {
    json!({"id": id, "builder": builder, "source": source, "source_title": "Gruff Captain", "status": status, "edited": false, "check": check, "note": "checked",
           "messages": [{"role": "system", "content": "You are the game master's assistant for the tabletop world \"Ashfall Reach\"."},
                        {"role": "user", "content": format!("Question for {id}?")}, {"role": "assistant", "content": answer}]})
}

fn world_file(dir: &std::path::Path, name: &str, examples: Vec<Value>) -> std::path::PathBuf {
    use ndforge_core::bundle::{write_bundle, Member};
    let f = dir.join(name);
    let members = vec![
        ("entities.jsonl", Member::Records(sample::entities(1))),
        ("teacher_examples.jsonl", Member::Records(examples)),
    ];
    write_bundle(&f, &sample::world(), members, "t", "2026-11-08T12:00:00Z").unwrap();
    f
}

#[test]
fn examples_nd_world_wrote_join_the_draft_with_their_marks_and_survive_a_newer_export() {
    let d = tempdir().unwrap();
    let mut p = Project::create(&d.path().join("projects"), "Ashfall Reach").unwrap();
    let first = vec![
        nd_example("a1", "t_deeper", "entity:1", "Anders matters because of his role.", "keep", "supported"),
        nd_example("b2", "t_deeper", "entity:1", "Invented ZZ detail.", "drop", "unsupported"),
        nd_example("c3", "t_general", "general:pacing", "Alternate tense and quiet scenes.", "keep", "unchecked"),
    ];
    let f1 = world_file(d.path(), "one.ndtrain", first.clone());
    let bundle = open_bundle(&f1).unwrap();
    assert!(bundle.summary_lines().iter().any(|l| l == "3 ready-made examples from nd-world"), "{:?}", bundle.summary_lines());
    p.import_bundle(&bundle).unwrap();
    let sum = store::generate_draft(&p, &Recipe::default()).unwrap();
    assert_eq!(sum["nd_world_examples"], 3);
    let ex = store::load_examples(&p).unwrap();
    let get = |id: &str| ex.iter().find(|e| e.id == id).unwrap();
    assert_eq!((get("a1").status, get("a1").check.as_deref(), get("a1").note.as_deref()), (Status::Keep, Some("supported"), Some("checked")));
    assert_eq!((get("b2").status, get("b2").check.as_deref()), (Status::Drop, Some("unsupported")));
    assert_eq!(get("c3").builder, "t_general");
    assert!(sum["builders"].as_array().unwrap().iter().any(|b| b["name"] == "Teacher: Deeper questions" && b["kept"] == 1 && b["dropped"] == 1));
    // the GM reviews: drops one, brings one back, edits one
    store::set_status(&p, &["a1".into()], Status::Drop).unwrap();
    store::set_status(&p, &["b2".into()], Status::Keep).unwrap();
    store::edit(&p, "c3", "My own answer.").unwrap();
    // a newer export: the same three and a new one; the decisions stay, the new example is added
    let mut second = first;
    second.push(nd_example("d4", "t_rules", "entity:1", "Roll and count sixes.", "keep", "partly"));
    let f2 = world_file(d.path(), "two.ndtrain", second);
    p.import_bundle(&open_bundle(&f2).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    let ex = store::load_examples(&p).unwrap();
    let get = |id: &str| ex.iter().find(|e| e.id == id).unwrap();
    assert_eq!(get("a1").status, Status::Drop);
    assert_eq!(get("b2").status, Status::Keep);
    assert_eq!(get("c3").messages.last().unwrap().content, "My own answer.");
    assert_eq!((get("d4").status, get("d4").check.as_deref()), (Status::Keep, Some("partly")));
    assert_eq!(ex.iter().filter(|e| e.builder.starts_with("t_")).count(), 4, "no duplicates");
    // they reach the saved version like any other (dropped ones do not)
    store::save_version(&p).unwrap();
    let train = std::fs::read_to_string(p.folder.join("datasets/v1/train.jsonl")).unwrap() + &std::fs::read_to_string(p.folder.join("datasets/v1/test.jsonl")).unwrap();
    assert!(train.contains("My own answer.") && train.contains("Roll and count sixes.") && !train.contains("Anders matters because of his role."));
    // a file with a damaged example line is skipped, not fatal
    let f3 = world_file(d.path(), "three.ndtrain", vec![json!({"id": "x", "builder": "t_deeper"}), nd_example("e5", "t_deeper", "entity:1", "Fine.", "keep", "supported")]);
    p.import_bundle(&open_bundle(&f3).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    assert!(store::load_examples(&p).unwrap().iter().any(|e| e.id == "e5"));
}

#[test]
fn every_world_of_a_project_gets_its_teacher_examples_and_they_do_not_mix() {
    let d = tempdir().unwrap();
    let mut p = project(d.path());
    let f = d.path().join("b.ndtrain");
    let mut w = sample::world();
    w["name"] = json!("Brightwater");
    w["slug"] = json!("brightwater");
    let members = vec![("entities.jsonl", Member::Records(vec![
        ent(1, "character", "Gruff Captain", "gruff, terse", true, &long("Captain")),
        ent(2, "location", "Dockside", "", true, &long("Dockside")),
    ]))];
    write_bundle(&f, &w, members, "t", "2026-11-01T12:00:00Z").unwrap();
    p.import_bundle(&open_bundle(f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    let (url, _) = mock(false);
    let rep = go(&p, &teacher(&url, None), &Options::default());
    assert_eq!(rep["stopped"], false, "{rep}");
    let ex = store::load_examples(&p).unwrap();
    let mine = |suffix: bool| ex.iter().filter(|e| e.builder.starts_with("t_") && e.builder != "t_general" && e.source.ends_with("@brightwater") == suffix).count();
    assert!(mine(true) > 0 && mine(false) > 0, "both worlds have teacher examples: {rep}");
    let ids: std::collections::HashSet<&String> = ex.iter().map(|e| &e.id).collect();
    assert_eq!(ids.len(), ex.len(), "ids are unique across worlds, although both worlds have an entity with the same id and name");
    // a second run costs nothing: everything already exists
    let again = go(&p, &teacher(&url, None), &Options::default());
    assert_eq!(again["added"], 0, "{again}");
}
