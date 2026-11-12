use ndforge_core::bundle::{open_bundle, write_bundle, Member};
use ndforge_core::dataset::builders::{generate, Recipe, Status};
use ndforge_core::dataset::chunk::{approx_tokens, chunk_article};
use ndforge_core::dataset::docs::{articles, load, relation_sentence, rule_sections};
use ndforge_core::dataset::split::held_out_sources;
use ndforge_core::dataset::store::{self, Filter};
use ndforge_core::project::Project;
use ndforge_core::sample;
use serde_json::{json, Value};
use tempfile::tempdir;

fn sample_loaded(dir: &std::path::Path) -> ndforge_core::dataset::docs::Loaded {
    let p = dir.join("s.ndtrain");
    sample::build(&p, 1, None).unwrap();
    load(&open_bundle(p).unwrap()).unwrap()
}

fn answer_of<'a>(ex: &'a [ndforge_core::dataset::builders::Example], builder: &str, user_contains: &str) -> Option<&'a str> {
    ex.iter().find(|e| e.builder == builder && e.messages.iter().any(|m| m.role == "user" && m.content.contains(user_contains)))
        .and_then(|e| e.messages.last()).map(|m| m.content.as_str())
}

#[test]
fn sample_world_gives_the_expected_examples() {
    let d = tempdir().unwrap();
    let l = sample_loaded(d.path());
    let ex = generate(&l, &Recipe { phrasings: 3, ..Recipe::default() });
    // the world book: one text per article, secrets kept and labelled
    let book: Vec<&str> = ex.iter().filter_map(|e| e.text.as_deref()).collect();
    assert!(book.iter().any(|t| t.starts_with("# Anders de Vallière") && t.contains("(GM secret: He fears fire.)") && t.contains("Also known as: Anders") && t.contains("Rank: Captain")), "{book:?}");
    assert!(book.iter().any(|t| t.contains("Anders de Vallière is located in Dockside.")));
    assert!(book.iter().any(|t| t.starts_with("# Session 1: Arrival") && t.contains("(hidden from players) The duke is the cult leader.")));
    assert!(book.iter().any(|t| t.starts_with("# Random table: Dock rumours")));
    // simple facts
    assert!(answer_of(&ex, "identity", "Anders de Vallière").unwrap().contains("is a character (player character)"));
    assert!(answer_of(&ex, "aliases", "Anders de Vallière").unwrap().contains("also known as Anders"));
    assert_eq!(answer_of(&ex, "fields", "Anders de Vallière").unwrap(), "Anders de Vallière's rank is Captain.");
    assert_eq!(answer_of(&ex, "relations", "Anders de Vallière").unwrap(), "Anders de Vallière is located in Dockside.");
    assert!(answer_of(&ex, "secrets", "secrets").unwrap().contains("- He fears fire."));
    assert!(answer_of(&ex, "tables", "Dock rumours").unwrap().contains("- A ship vanished"));
    assert!(answer_of(&ex, "history", "session 1").unwrap().contains("The party reached Dockside."));
    assert_eq!(answer_of(&ex, "history", "When was session 1").unwrap(), "Session 1 (\"Arrival\") was played on 2026-10-01.");
    // every chat example is system + user + assistant; all start as kept; ids are unique
    for e in ex.iter().filter(|e| e.text.is_none()) {
        assert_eq!(e.messages.iter().map(|m| m.role.as_str()).collect::<Vec<_>>(), ["system", "user", "assistant"]);
        assert!(e.messages[0].content.contains("Ashfall Reach"));
    }
    assert!(ex.iter().all(|e| e.status == Status::Keep));
    let ids: std::collections::HashSet<&String> = ex.iter().map(|e| &e.id).collect();
    assert_eq!(ids.len(), ex.len());
}

#[test]
fn generation_is_deterministic_and_the_recipe_is_obeyed() {
    let d = tempdir().unwrap();
    let l = sample_loaded(d.path());
    let a: Vec<String> = generate(&l, &Recipe::default()).iter().map(|e| e.id.clone()).collect();
    let b: Vec<String> = generate(&l, &Recipe::default()).iter().map(|e| e.id.clone()).collect();
    assert_eq!(a, b);
    let only_book = generate(&l, &Recipe { builders: vec![], ..Recipe::default() });
    assert!(only_book.iter().all(|e| e.text.is_some()) && !only_book.is_empty());
    let no_book = generate(&l, &Recipe { world_book: false, builders: vec!["fields".into()], ..Recipe::default() });
    assert!(no_book.iter().all(|e| e.builder == "fields") && !no_book.is_empty());
    let more = generate(&l, &Recipe { phrasings: 3, builders: vec!["identity".into()], world_book: false, ..Recipe::default() });
    let less = generate(&l, &Recipe { phrasings: 1, builders: vec!["identity".into()], world_book: false, ..Recipe::default() });
    assert!(more.len() > less.len());
}

#[test]
fn articles_helpers() {
    assert_eq!(relation_sentence("A", "located_in", "B"), "A is located in B.");
    assert_eq!(relation_sentence("A", "owns", "B"), "A owns B.");
    assert_eq!(relation_sentence("A", "related", "B"), "A is related to B.");
    assert_eq!(relation_sentence("A", "member_of", "B"), "A is member of B.");
    let s = rule_sections("Intro text\n\n# Combat\nRoll.\n\n## Initiative\nHighest first.\n\n```\n# not a heading\n```\n#NoSpace\n");
    assert_eq!(s.iter().map(|(h, _)| h.as_str()).collect::<Vec<_>>(), ["Rules", "Combat", "Initiative"]);
    assert!(s[2].1.contains("# not a heading") && s[2].1.contains("#NoSpace"));
}

#[test]
fn long_articles_are_chunked_with_their_title() {
    let para = "Sentence about the world. ".repeat(40);
    let article = format!("# Big Place\n\n{para}\n\n{para}\n\n{para}\n\n{para}");
    let chunks = chunk_article("Big Place", &article, 300);
    assert!(chunks.len() >= 4, "{}", chunks.len());
    assert!(chunks[0].starts_with("# Big Place\n") && chunks[1].starts_with("# Big Place (part 2)\n"));
    assert!(chunks.iter().all(|c| approx_tokens(c) <= 300 + 20));
    assert_eq!(chunk_article("T", "# T\nshort", 300), vec!["# T\nshort".to_string()]);
    // a single huge paragraph (and accented text) is split on character boundaries, not bytes
    let huge = "É".repeat(5000);
    assert!(chunk_article("H", &format!("# H\n\n{huge}"), 200).len() > 5);
}

#[test]
fn held_out_sources_are_stable_and_proportional() {
    let sources: Vec<String> = (0..50).map(|i| format!("entity:{i}")).collect();
    let a = held_out_sources(&sources, 1, 10);
    assert_eq!(a.len(), 5);
    assert_eq!(a, held_out_sources(&sources, 1, 10));
    assert_ne!(a, held_out_sources(&sources, 2, 10));
    assert!(held_out_sources(&sources, 1, 0).is_empty());
    assert_eq!(held_out_sources(&sources[..5].to_vec(), 1, 10).len(), 1);      // at least one once there are 5
    assert!(held_out_sources(&sources[..4].to_vec(), 1, 10).is_empty());       // too few to spare any
    let mut more = sources.clone();
    more.push("entity:new".into());
    assert!(a.iter().filter(|s| held_out_sources(&more, 1, 10).contains(*s)).count() >= 4); // adding one source barely changes the choice
}

fn project_with_world(d: &std::path::Path, variant: u8) -> Project {
    let mut p = Project::create(&d.join("projects"), "W").unwrap();
    let f = d.join(format!("w{variant}.ndtrain"));
    sample::build(&f, variant, None).unwrap();
    p.import_bundle(&open_bundle(f).unwrap()).unwrap();
    p
}

#[test]
fn draft_review_edit_and_save_a_version() {
    let d = tempdir().unwrap();
    let mut p = project_with_world(d.path(), 1);
    assert!(store::summary(&p).unwrap().is_none());
    let sum = store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    assert!(sum["total"].as_u64().unwrap() > 20 && sum["stale"] == false);
    assert!(sum["builders"].as_array().unwrap().iter().any(|b| b["id"] == "world_book"));
    // filter, drop, edit
    let (total, page) = store::list(&p, &Filter { builder: Some("fields".into()), ..Default::default() }, 0, 50).unwrap();
    assert_eq!(total, page.len());
    let target = page[0].id.clone();
    assert_eq!(store::set_status(&p, &[target.clone()], Status::Drop).unwrap(), 1);
    let (_, found) = store::list(&p, &Filter { search: Some("captain".into()), status: Some(Status::Drop), ..Default::default() }, 0, 10).unwrap();
    assert_eq!(found.len(), 1);
    let pick = store::list(&p, &Filter { builder: Some("aliases".into()), ..Default::default() }, 0, 1).unwrap().1.remove(0);
    assert!(store::edit(&p, &pick.id, "   ").is_err());
    store::edit(&p, &pick.id, "Edited answer.").unwrap();
    // regenerating keeps the decisions
    store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    let all = store::load_examples(&p).unwrap();
    assert_eq!(all.iter().find(|e| e.id == target).unwrap().status, Status::Drop);
    let ed = all.iter().find(|e| e.id == pick.id).unwrap();
    assert!(ed.edited && ed.messages.last().unwrap().content == "Edited answer.");
    // save v1: dropped example absent, edit present, held-out sources only in test, world book only in train
    let m = store::save_version(&p).unwrap();
    assert_eq!(m["version"], 1);
    let vdir = p.folder.join("datasets/v1");
    let read = |f: &str| -> Vec<Value> { std::fs::read_to_string(vdir.join(f)).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect() };
    let (train, test) = (read("train.jsonl"), read("test.jsonl"));
    assert!(train.iter().all(|x| x["id"] != json!(target)) && test.iter().all(|x| x["id"] != json!(target)));
    assert!(train.iter().any(|x| x["type"] == "text") && test.iter().all(|x| x["type"] == "chat"));
    let held: Vec<String> = m["held_out_sources"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    assert!(!held.is_empty() && test.iter().all(|x| held.contains(&x["source"].as_str().unwrap().to_string())));
    assert!(train.iter().filter(|x| x["type"] == "chat").all(|x| !held.contains(&x["source"].as_str().unwrap().to_string())));
    assert!(train.iter().chain(test.iter()).any(|x| x["messages"].to_string().contains("Edited answer.")));
    assert_eq!(m["counts"]["train_chat"].as_u64().unwrap() as usize, train.iter().filter(|x| x["type"] == "chat").count());
    assert_eq!(m["counts"]["test_chat"].as_u64().unwrap() as usize, test.len());
    assert_eq!(m["counts"]["dropped"], 1);
    // the project now counts step 2 as done; a second save is v2
    assert!(p.progress()[1].1 && p.unlocked()[2].1);
    assert_eq!(store::save_version(&p).unwrap()["version"], 2);
    assert_eq!(store::versions(&p).len(), 2);
    // new world data makes the draft "stale" until it is made again
    let f2 = d.path().join("w2.ndtrain");
    sample::build(&f2, 2, None).unwrap();
    p.import_bundle(&open_bundle(f2).unwrap()).unwrap();
    assert_eq!(store::summary(&p).unwrap().unwrap()["stale"], true);
    let sum = store::generate_draft(&p, &Recipe::default()).unwrap();
    assert_eq!(sum["stale"], false);
    // dropping everything refuses to save an empty version
    store::set_status_matching(&p, &Filter::default(), Status::Drop).unwrap();
    assert!(store::save_version(&p).unwrap_err().0.contains("nothing to save"));
}

#[test]
fn needs_world_data_first() {
    let d = tempdir().unwrap();
    let p = Project::create(d.path(), "Empty").unwrap();
    assert!(store::generate_draft(&p, &Recipe::default()).unwrap_err().0.contains("Import the world data first"));
    assert!(store::save_version(&p).unwrap_err().0.contains("Make the training data first"));
}

#[test]
fn a_big_world_with_odd_data_is_fast_and_never_fails() {
    let mut ents = Vec::new();
    for i in 0..2000 {
        ents.push(json!({"id": i + 1, "kind": if i % 3 == 0 { "character" } else { "location" }, "subtype": "", "name": format!("Émile Ünïcode {i}"), "aliases": [format!("É{i}")],
            "folder": null, "tags": [], "summary": if i % 7 == 0 { Value::Null } else { json!(format!("Résumé {i}")) },
            "body": if i % 5 == 0 { Value::Null } else { json!(format!("# T\n[gmonly]Secret {i}[/gmonly] 日本語 [[Émile Ünïcode {}]]", i + 1)) },
            "template": null, "custom_fields": if i % 2 == 0 { json!({"a": null, "b": [1, {"c": 2}], "d": {"x": "y"}}) } else { json!([1, 2]) },
            "roleplay_personality": "", "visible_to_players": i % 4 != 0, "group_ids": [], "rag_priority": false, "updated_at": "2026-11-01T10:00:00"}));
    }
    let rels: Vec<Value> = (0..1500).map(|i| json!({"id": i + 1, "source_id": i + 1, "source_name": format!("Émile Ünïcode {i}"), "target_id": i + 2, "target_name": format!("Émile Ünïcode {}", i + 1), "relation": if i % 2 == 0 { Value::Null } else { json!("part_of") }})).collect();
    let d = tempdir().unwrap();
    let f = d.path().join("big.ndtrain");
    write_bundle(&f, &sample::world(), vec![("entities.jsonl", Member::Records(ents)), ("relations.jsonl", Member::Records(rels)), ("rules.md", Member::Text("x".repeat(30_000)))], "t", "2026-11-01T00:00:00Z").unwrap();
    let start = std::time::Instant::now();
    let l = load(&open_bundle(&f).unwrap()).unwrap();
    let ex = generate(&l, &Recipe::default());
    assert!(ex.len() > 8000, "{}", ex.len());
    assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
    assert_eq!(articles(&l).len(), 2000 + 1);
}

fn other_world(dir: &std::path::Path, file: &str, name: &str, slug: &str, created_later: bool) -> std::path::PathBuf {
    let f = dir.join(file);
    let mut w = sample::world();
    w["name"] = json!(name);
    w["slug"] = json!(slug);
    sample::build(&f, if created_later { 2 } else { 1 }, Some(w)).unwrap();
    f
}

#[test]
fn several_worlds_in_one_project_train_together_without_mixing_up() {
    use ndforge_core::project::BundleCheck;
    let d = tempdir().unwrap();
    let mut p = project_with_world(d.path(), 1); // "Ashfall Reach"
    let sole_ids: std::collections::HashSet<String> = {
        store::generate_draft(&p, &Recipe::default()).unwrap();
        store::load_examples(&p).unwrap().into_iter().map(|e| e.id).collect()
    };
    // a different world is offered to be added, not refused; the same file twice is refused
    let b = open_bundle(other_world(d.path(), "b.ndtrain", "Brightwater", "brightwater", false)).unwrap();
    assert!(matches!(p.check_new_bundle(&b).unwrap(), BundleCheck::AddWorld(t) if t.contains("Ashfall Reach") && t.contains("Brightwater")));
    p.import_bundle(&b).unwrap();
    assert!(matches!(p.check_new_bundle(&b).unwrap(), BundleCheck::Refused(_)));
    assert_eq!(p.worlds().len(), 2);
    assert_eq!(p.display_name(), "Ashfall Reach and Brightwater");
    assert_eq!(p.model_label(), p.name);
    // reopening keeps both
    let p2 = Project::open(&p.folder).unwrap();
    assert_eq!(p2.worlds().len(), 2);
    assert_eq!(p2.signature(), p.signature());

    // both worlds' examples are there; ids and sources never collide, each world keeps its own name in the prompts
    let sum = store::generate_draft(&p, &Recipe::default()).unwrap();
    assert_eq!(sum["worlds"], 2);
    let all = store::load_examples(&p).unwrap();
    let ids: std::collections::HashSet<&String> = all.iter().map(|e| &e.id).collect();
    assert_eq!(ids.len(), all.len(), "ids are unique across worlds");
    assert!(sole_ids.iter().all(|i| ids.contains(i)), "the first world's examples keep their ids, so earlier review work stays");
    let brightwater: Vec<_> = all.iter().filter(|e| e.source.ends_with("@brightwater")).collect();
    let ashfall = all.len() - brightwater.len();
    assert!(!brightwater.is_empty() && ashfall == sole_ids.len(), "{} vs {}", ashfall, sole_ids.len());
    let sys = |e: &&ndforge_core::dataset::builders::Example| e.messages.first().map(|m| m.content.clone()).unwrap_or_default();
    assert!(brightwater.iter().filter(|e| !e.messages.is_empty()).all(|e| sys(e).contains("Brightwater")));
    assert!(all.iter().filter(|e| !e.source.ends_with("@brightwater") && !e.messages.is_empty()).all(|e| sys(&e).contains("Ashfall Reach")));
    assert!(!sum["stale"].as_bool().unwrap());

    // a decision made on an example of the second world survives making the examples again
    let victim = brightwater.iter().find(|e| e.builder == "identity").unwrap().id.clone();
    store::set_status(&p, &[victim.clone()], Status::Drop).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    assert_eq!(store::load_examples(&p).unwrap().into_iter().find(|e| e.id == victim).unwrap().status, Status::Drop);

    // updating one world touches only that world; the signature changes
    let sig = p.signature();
    let b2 = open_bundle(other_world(d.path(), "b2.ndtrain", "Brightwater", "brightwater", true)).unwrap();
    assert!(matches!(p.check_new_bundle(&b2).unwrap(), BundleCheck::Update(_)));
    p.import_bundle(&b2).unwrap();
    assert_eq!(p.worlds().len(), 2);
    assert_ne!(p.signature(), sig);
    assert_eq!(store::summary(&p).unwrap().unwrap()["stale"], true);

    // removing the first world makes the other the first one, with its suffix kept (so its ids stay)
    assert_eq!(p.remove_world("ashfall-reach").unwrap(), "Ashfall Reach");
    assert_eq!((p.worlds().len(), p.world_name.as_str(), p.world_suffix.as_str()), (1, "Brightwater", "@brightwater"));
    assert!(p.remove_world("brightwater").is_err(), "the last world stays");
    assert_eq!(p.model_label(), "Brightwater");
}

#[test]
fn a_saved_version_is_exported_as_chatml_files_for_unsloth_studio() {
    use ndforge_core::export::studio;
    let d = tempdir().unwrap();
    let mut p = project_with_world(d.path(), 1);
    // a second world, so the system message of each example can be checked
    let b = open_bundle(other_world(d.path(), "b.ndtrain", "Brightwater", "brightwater", false)).unwrap();
    p.import_bundle(&b).unwrap();
    assert!(studio::export(&p, None, d.path(), true).unwrap_err().0.contains("Save a data version"));
    store::generate_draft(&p, &Recipe { test_percent: 30, ..Recipe::default() }).unwrap();
    store::save_version(&p).unwrap();
    let dest = d.path().join("out");
    let r = studio::export(&p, None, &dest, true).unwrap();
    let dir = std::path::PathBuf::from(r["dir"].as_str().unwrap());
    assert!(dir.ends_with("Ashfall Reach-v1-unsloth-studio") || dir.to_string_lossy().contains("-v1-unsloth-studio"));
    let lines = |f: &str| -> Vec<Value> { std::fs::read_to_string(dir.join(f)).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect() };
    let train = lines("train.jsonl");
    assert_eq!(train.len() as u64, r["rows"].as_u64().unwrap());
    for l in &train {
        assert_eq!(l.as_object().unwrap().keys().collect::<Vec<_>>(), ["messages"], "chatml only: {l}");
        let m = l["messages"].as_array().unwrap();
        assert!(m.last().unwrap()["role"] == "assistant" && m.iter().all(|x| x["role"].is_string() && x["content"].is_string()));
    }
    assert!(r["world_book"].as_u64().unwrap() > 0, "the world book became answers");
    let book = train.iter().find(|l| l["messages"][1]["content"].as_str().is_some_and(|c| c.starts_with("Tell me what the lore says about"))).expect("a world-book row");
    assert!(book["messages"][0]["content"].as_str().unwrap().contains("tabletop world"));
    // each example speaks of its own world
    let sys = |l: &Value| l["messages"][0]["content"].as_str().unwrap_or("").to_string();
    assert!(train.iter().any(|l| sys(l).contains("\"Brightwater\"")) && train.iter().any(|l| sys(l).contains("\"Ashfall Reach\"")));
    assert!(r["eval"].as_u64().unwrap() > 0 && lines("eval.jsonl").iter().all(|l| l["messages"].is_array()));
    let readme = std::fs::read_to_string(dir.join("README.md")).unwrap();
    assert!(readme.contains("chatml") && readme.contains("Train on Completions") && readme.contains("GM secrets") && readme.contains("Context Length"), "{readme}");
    // the same folder twice is refused; without the world book only the questions are exported
    assert!(studio::export(&p, Some(1), &dest, true).unwrap_err().0.contains("already exists"));
    let r2 = studio::export(&p, Some(1), &d.path().join("out2"), false).unwrap();
    assert_eq!((r2["world_book"].as_u64(), r2["rows"].as_u64().unwrap() < r["rows"].as_u64().unwrap()), (Some(0), true));
    assert!(studio::export(&p, Some(9), &d.path().join("out3"), true).is_err());
}
