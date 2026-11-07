//! Step 4: scoring, the question set, a whole test with a fake worker, the free chat, transcripts and the run table.
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::evaluate::questions::{build, invented_names};
use ndforge_core::evaluate::runs;
use ndforge_core::evaluate::score::{declines, overlap, sane, verdict};
use ndforge_core::project::Project;
use ndforge_core::sample;
use ndforge_core::teacher::{Teacher, TeacherConfig};
use ndforge_core::train::plan::{base_hyper, Method, Preset};
use ndforge_core::train::runs as truns;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

#[test]
fn word_overlap_declines_and_sanity() {
    let r = "Anders de Vallière is a Widow-Bound Hunter who hunts colossal beasts near Dockside.";
    assert_eq!(overlap(r, r), 1.0);
    assert_eq!(overlap("Bananas grow in warm places.", r), 0.0);
    assert_eq!(overlap("", r), 0.0);
    let part = overlap("Anders is a hunter of colossal beasts.", r);
    assert!(part > 0.4 && part < 1.0, "{part}");
    assert!(overlap("Anders is a hunter of colossal beasts.", r) > overlap("Anders is tall.", r));
    // accented words count; a very long reference is not fully expected from a short answer
    assert!(overlap("vallière", "Vallière") > 0.9);
    let long = (0..200).map(|i| format!("word{i}x")).collect::<Vec<_>>().join(" ");
    let first30 = (0..30).map(|i| format!("word{i}x")).collect::<Vec<_>>().join(" ");
    assert!(overlap(&first30, &long) > 0.99);
    assert!(declines("I can't find Zorblax in the lore I have.") && declines("That isn’t in the lore.") && declines("There is nothing about it."));
    assert!(!declines("Zorblax Prime is a mighty fortress in the north."));
    assert!(sane("Pacing works best when you alternate tense scenes with quiet ones.") && !sane("ok") && !sane("I don't have anything on that, it is not in the lore.") && !sane(&"loop ".repeat(30)));
}

#[test]
fn the_verdict_is_plain() {
    let rows: Vec<(String, f64, f64)> = vec![("lore", 0.1, 0.8), ("lore", 0.2, 0.9), ("lore", 0.5, 0.5), ("lore", 0.6, 0.1), ("made_up", 1.0, 0.0), ("general", 1.0, 0.0), ("general", 1.0, 1.0)]
        .into_iter().map(|(c, o, t)| (c.to_string(), o, t)).collect();
    let v = verdict(&rows);
    assert_eq!(v["headline"], "Better than the original on 50 % of lore questions (worse on 25 %).");
    assert_eq!(v["better"], true);
    let lines: Vec<&str> = v["lines"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
    assert!(lines.iter().any(|l| l.contains("invented an answer for 100 %") && l.contains("trained model for 0 %")));
    assert!(lines.iter().any(|l| l.contains("General chat got worse")), "{lines:?}");
    assert!(verdict(&[])["headline"].as_str().unwrap().contains("Not enough"));
}

fn project(root: &std::path::Path, test_percent: u8) -> Project {
    let mut p = Project::create(&root.join("projects"), "Ashfall Reach").unwrap();
    let f = root.join("w.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe { test_percent, ..Recipe::default() }).unwrap();
    store::save_version(&p).unwrap();
    p
}

fn done_run(p: &Project, name: &str, method: &str) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("adapter")).unwrap();
    std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    let job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("summary.json"), json!({"steps": 40, "final_loss": 0.8}).to_string()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "method": method, "step": 40, "total_steps": 40}).to_string()).unwrap();
}

/// A fake "python" that answers like a model would: the trained one knows the held-out answers and declines made-up things; the original does not.
#[cfg(unix)]
fn fake_model(root: &std::path::Path, test_jsonl: &std::path::Path, general_broken: bool) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(root.join("fake.py"), format!(r#"
import json, sys
job = json.load(open(sys.argv[1], encoding="utf-8"))
ref = {{}}
for line in open({test:?}, encoding="utf-8"):
    r = json.loads(line)
    if r["type"] == "chat":
        ref[[m for m in r["messages"] if m["role"] == "user"][-1]["content"]] = r["messages"][-1]["content"]
def answer(side, q):
    user = [m for m in q["messages"] if m["role"] == "user"][-1]["content"]
    if user in ref:
        return ref[user] if side == "trained" else "Something about ships and weather."
    if q["id"].startswith("probe-m"):
        return "That isn't in the lore." if side == "trained" else "It is a mighty fortress in the far north."
    if q["id"].startswith("probe-g"):
        return {general} if side == "trained" else "Alternate tense scenes with quiet ones for good pacing."
    return "trained says hi" if side == "trained" else "original says hi"
out = {{s: {{q["id"]: answer(s, q) for q in job["questions"]}} for s in job["which"]}}
print(json.dumps({{"type": "progress", "step": "evaluate", "value": 0.5, "text": "Answering"}}), flush=True)
json.dump(out, open(job["out"], "w", encoding="utf-8"))
print(json.dumps({{"type": "result", "data": {{"out": job["out"], "n": len(job["questions"])}}}}), flush=True)
"#, test = test_jsonl.to_string_lossy(), general = if general_broken { "\"ok\"" } else { "\"Alternate tense scenes with quiet ones for good pacing.\"" })).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\nexec python3 -I {} \"$5\"\n", root.join("fake.py").display())).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[test]
fn the_question_set_has_held_out_questions_made_up_probes_and_general_prompts() {
    let d = tempdir().unwrap();
    let p = project(d.path(), 30);
    let qs = build(&p, 1, 30).unwrap();
    let cats = |c: &str| qs.iter().filter(|q| q.category == c).count();
    assert!(cats("lore") >= 1 && cats("made_up") >= 8 && cats("general") == 6, "{:?}", qs.iter().map(|q| (q.category, &q.question)).collect::<Vec<_>>());
    assert!(qs.iter().filter(|q| q.category == "lore").all(|q| q.reference.is_some()));
    assert!(qs.iter().all(|q| q.messages.last().unwrap()["role"] == "user" && q.messages[0]["role"] == "system"));
    // invented names exist nowhere in the world
    let loaded = ndforge_core::dataset::docs::load(&p.current_bundle().unwrap()).unwrap();
    let corpus: String = ndforge_core::dataset::docs::articles(&loaded).iter().map(|a| a.article.to_lowercase()).collect::<Vec<_>>().join("\n");
    for q in qs.iter().filter(|q| q.id.starts_with("probe-m")) {
        assert!(!corpus.contains(&q.question.to_lowercase()));
    }
    assert!(invented_names("zorblax prime hollow", 5, 7).iter().all(|n| !"zorblax prime hollow".contains(&n.to_lowercase())));
    assert_eq!(invented_names("", 5, 7), invented_names("", 5, 7));
    // a cap per category; no held-out questions at all is explained
    assert!(build(&p, 1, 5).unwrap().iter().filter(|q| q.category == "lore").count() <= 5);
    let none = project(&d.path().join("z"), 0);
    assert!(build(&none, 1, 30).unwrap_err().0.contains("Kept for testing"));
    assert!(build(&p, 9, 30).is_err());
}

#[cfg(unix)]
#[test]
fn a_whole_test_scores_both_models_and_everything_is_saved() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let p = project(d.path(), 30);
    fake_model(d.path(), &p.folder.join("datasets/v1/test.jsonl"), false);
    done_run(&p, "run-a", "lora");
    assert_eq!(runs::testable(&p).len(), 1);
    assert_eq!(runs::testable(&p)[0]["tested"], false);
    assert!(!p.progress()[3].1, "step 4 is not done before a test");
    let mut events = Vec::new();
    let r = runs::execute(&p, "run-a", 30, None, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert_eq!(r["method"], "overlap");
    assert_eq!(r["verdict"]["headline"], "Better than the original on 100 % of lore questions (worse on 0 %).", "{r}");
    assert_eq!(r["scores"]["trained"]["lore"]["mean"], 1.0);
    assert_eq!((r["scores"]["original"]["made_up"]["mean"].as_f64(), r["scores"]["trained"]["made_up"]["mean"].as_f64()), (Some(1.0), Some(0.0)));
    assert_eq!((r["scores"]["original"]["general"]["mean"].as_f64(), r["scores"]["trained"]["general"]["mean"].as_f64()), (Some(1.0), Some(1.0)));
    assert!(events.iter().any(|e| e["kind"] == "progress"));
    assert!(p.progress()[3].1, "step 4 is done once a run has a test.json");
    assert_eq!(runs::read_test(&p, "run-a").unwrap()["n"], r["n"]);
    assert!(runs::read_test(&p, "../x").is_none());
    let t = &runs::testable(&p)[0];
    assert_eq!((t["tested"].clone(), t["headline"].as_str().unwrap().starts_with("Better")), (json!(true), true));
    // the comparison table: a second run with worse general chat; best lore first
    done_run(&p, "run-b", "full");
    fake_model(d.path(), &p.folder.join("datasets/v1/test.jsonl"), true);
    let rb = runs::execute(&p, "run-b", 30, None, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert!(rb["verdict"]["lines"].as_array().unwrap().iter().any(|l| l.as_str().unwrap().contains("General chat got worse")));
    let table = runs::compare(&p);
    assert_eq!(table.len(), 2);
    assert_eq!((table[0]["lore"].as_f64(), table[0]["general"].as_f64().is_some()), (Some(1.0), true));
    assert!(table.iter().any(|r| r["name"] == "run-b" && r["method"] == "lora" && r["general"] == json!(0.0)));
    // the free chat
    let a = runs::ask(&p, "run-a", None, "Who is Anders?", &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!((a["original"].as_str(), a["trained"].as_str()), (Some("original says hi"), Some("trained says hi")));
    assert!(runs::ask(&p, "run-a", None, "  ", &AtomicBool::new(false), &mut |_| {}).is_err());
    // transcripts
    let f = runs::save_transcript(&p, "run-a", &[json!({"question": "Who is Anders?", "original": "A", "trained": "B"})]).unwrap();
    let md = std::fs::read_to_string(&f).unwrap();
    assert!(f.starts_with(p.folder.join("transcripts")) && md.contains("## Who is Anders?") && md.contains("**Trained**\n\nB"));
    assert!(runs::save_transcript(&p, "run-a", &[]).is_err());
}

#[cfg(unix)]
#[test]
fn a_teacher_can_grade_instead_of_word_overlap_and_failures_are_plain() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let p = project(d.path(), 30);
    fake_model(d.path(), &p.folder.join("datasets/v1/test.jsonl"), false);
    done_run(&p, "run-a", "lora");
    // a mock grader: "Something about ships" is wrong, everything else correct
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://127.0.0.1:{}", server.server_addr().to_ip().unwrap().port());
    let hits = std::sync::Arc::new(Mutex::new(0usize));
    let h2 = hits.clone();
    std::thread::spawn(move || {
        for mut req in server.incoming_requests() {
            let mut body = String::new();
            req.as_reader().read_to_string(&mut body).unwrap();
            *h2.lock().unwrap() += 1;
            let v: Value = serde_json::from_str(&body).unwrap();
            let user = v["messages"][1]["content"].as_str().unwrap();
            let cand = user.split("CANDIDATE ANSWER:\n").nth(1).unwrap_or("");
            let verdict = if cand.contains("ships and weather") { "wrong" } else { "correct" };
            let out = json!({"choices": [{"message": {"content": format!("{{\"verdict\":\"{verdict}\"}}")}}]});
            let _ = req.respond(tiny_http::Response::from_string(out.to_string()));
        }
    });
    let t = Teacher::new(TeacherConfig::new(&url, "judge", None));
    let r = runs::execute(&p, "run-a", 30, Some(&t), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["method"], "teacher");
    assert!(r["judged"].as_u64().unwrap() >= 2 && *hits.lock().unwrap() as u64 >= 2 * r["judged"].as_u64().unwrap() / 1, "{r}");
    assert_eq!((r["scores"]["original"]["lore"]["mean"].as_f64(), r["scores"]["trained"]["lore"]["mean"].as_f64()), (Some(0.0), Some(1.0)));
    // a run without a saved model, a missing run, a missing environment
    let rdir = truns::runs_dir(&p).join("run-c");
    std::fs::create_dir_all(&rdir).unwrap();
    std::fs::copy(truns::runs_dir(&p).join("run-a/config.json"), rdir.join("config.json")).unwrap();
    let e = runs::execute(&p, "run-c", 30, None, &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("no saved model"), "{e}");
    assert!(runs::execute(&p, "nope", 30, None, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("does not exist"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("missing"));
    assert!(runs::execute(&p, "run-a", 30, None, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Training setup"));
}
