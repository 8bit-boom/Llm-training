//! The thinking check of a made model: the gates (against the cases the worker is tested with), what is checked, the saved results, and a run with a stand-in worker.
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use ndforge_core::project::Project;
use ndforge_core::reshape::thinkcheck::*;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn cases() -> Value {
    serde_json::from_slice(&std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/tests/data/think_gate_cases.json")).unwrap()).unwrap()
}

#[test]
fn the_gates_follow_the_cases_the_worker_is_tested_with() {
    let c = cases();
    assert_eq!(c["gates"], json!(GATES), "the same gates in the same order");
    let list = c["cases"].as_array().unwrap();
    assert!(list.len() >= 15);
    for case in list {
        let g = gates(&case["original"], &case["candidate"]);
        let name = case["name"].as_str().unwrap();
        assert_eq!(g["verdict"], case["expect"]["verdict"], "{name}");
        assert_eq!(g["failed"], case["expect"]["failed"], "{name}");
        assert_eq!(g["warned"], case["expect"]["warned"], "{name}");
        assert_eq!(g["checks"].as_array().unwrap().iter().map(|x| x["name"].as_str().unwrap()).collect::<Vec<_>>(), GATES, "{name}");
    }
}

#[test]
fn the_limits_come_from_the_original() {
    let o = json!({"closed_rate": 1.0, "accuracy": 0.75, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 800});
    let l = gate_limits(&o);
    let get = |n: &str| l.iter().find(|x| x.0 == n).unwrap().2;
    assert!((get("closed_rate") - 0.9).abs() < 1e-9 && (get("accuracy") - 0.6).abs() < 1e-9 && (get("loop_rate") - 0.1).abs() < 1e-9);
    assert!((get("garbage_rate") - 0.05).abs() < 1e-9 && (get("median_think_tokens") - 2000.0).abs() < 1e-9);
}

fn project(root: &std::path::Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

/// a made model with its original recorded
fn made(p: &Project, name: &str, healed: bool) -> std::path::PathBuf {
    let dir = p.folder.join("models").join(name);
    std::fs::create_dir_all(dir.join("model")).unwrap();
    std::fs::write(dir.join("model/config.json"), "{}").unwrap();
    std::fs::write(dir.join("model/ndforge_reshape.json"), json!({"format": 2, "origin": {"repo": "unsloth/Qwen3.5-27B", "family": "qwen3_5", "params": 27_800_000_000u64}, "params": 21_000_000_000u64}).to_string()).unwrap();
    std::fs::write(dir.join("state.json"), json!({"name": name, "status": "done"}).to_string()).unwrap();
    if healed {
        std::fs::create_dir_all(dir.join("healed")).unwrap();
        std::fs::write(dir.join("healed/config.json"), "{}").unwrap();
    }
    dir
}

fn spec(name: &str, target: &str) -> ThinkSpec {
    ThinkSpec { name: name.into(), target: target.into(), ..Default::default() }
}

#[test]
fn each_target_is_a_different_model_and_every_mistake_is_a_plain_message() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    let dir = made(&p, "slim-a", false);
    assert_eq!(model_of(&p, &spec("slim-a", "original")).unwrap(), ("unsloth/Qwen3.5-27B".to_string(), "hf"));
    assert_eq!(model_of(&p, &spec("slim-a", "cut")).unwrap(), (dir.join("model").to_string_lossy().to_string(), "hf"));
    assert!(model_of(&p, &spec("slim-a", "healed")).unwrap_err().0.contains("not been healed"));
    std::fs::create_dir_all(dir.join("healed")).unwrap();
    std::fs::write(dir.join("healed/config.json"), "{}").unwrap();
    assert_eq!(model_of(&p, &spec("slim-a", "healed")).unwrap().0, dir.join("healed").to_string_lossy());
    let gguf = d.path().join("m-Q4_K_M.gguf");
    std::fs::write(&gguf, "GGUF").unwrap();
    let mut g = spec("slim-a", "gguf");
    assert!(model_of(&p, &g).unwrap_err().0.contains("Choose the GGUF"));
    g.gguf = Some(d.path().join("none.gguf").to_string_lossy().to_string());
    assert!(model_of(&p, &g).unwrap_err().0.contains("was not found"));
    g.gguf = Some(d.path().join("x.bin").to_string_lossy().to_string());
    assert!(model_of(&p, &g).unwrap_err().0.contains("not a GGUF"));
    g.gguf = Some(gguf.to_string_lossy().to_string());
    assert_eq!(model_of(&p, &g).unwrap(), (gguf.to_string_lossy().to_string(), "llama"));
    assert!(model_of(&p, &spec("../x", "cut")).is_err() && model_of(&p, &spec("nope", "cut")).unwrap_err().0.contains("not in this project"));
    assert!(model_of(&p, &spec("slim-a", "other")).unwrap_err().0.contains("Unknown"));
}

#[test]
fn the_job_names_the_files_the_worker_writes_and_reads() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    made(&p, "slim-a", false);
    let mut s = spec("slim-a", "cut");
    s.quick = true;
    s.batch = Some(4);
    s.load = Some("4bit".into());
    let j = build_job(&p, &s).unwrap();
    let t = p.folder.join("models").join("slim-a").join("think_check");     // joined part by part: the expected paths have the same separators as the job's
    assert_eq!((j["job"].clone(), j["backend"].clone(), j["quick"].clone(), j["load"].clone(), j["batch"].clone(), j["control"].clone()), (json!("think_check"), json!("hf"), json!(true), json!("4bit"), json!(4), json!(true)));
    assert_eq!((j["cache_file"].clone(), j["stop_file"].clone()), (json!(t.join("cut.quick.json")), json!(t.join("STOP"))));
    s.quick = false;
    s.load = Some("weird".into());
    s.batch = Some(0);
    let j = build_job(&p, &s).unwrap();
    assert_eq!((j["cache_file"].clone(), j["load"].clone()), (json!(t.join("cut.full.json")), json!("auto")));
    assert!(j.get("batch").is_none(), "an unreasonable batch is not sent");
}

fn write_result(p: &Project, name: &str, target: &str, quick: bool, main: Value, suite: u64) {
    let dir = p.folder.join("models").join(name).join("think_check");
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join(format!("{target}.{}.json", if quick { "quick" } else { "full" }));
    std::fs::write(f, json!({"key": {}, "result": {"suite_version": suite, "main": main, "control": {"n": 8}, "created": "2026-11-17T10:00:00Z", "settings": {"quick": quick}}}).to_string()).unwrap();
}

fn m(closed: f64, acc: f64, loops: f64, median: u64) -> Value {
    json!({"n": 32, "think_rate": 1.0, "closed_rate": closed, "accuracy": acc, "loop_rate": loops, "garbage_rate": 0.0, "truncated_rate": 0.0, "median_think_tokens": median, "p90_think_tokens": median * 2})
}

#[test]
fn the_results_hold_a_table_and_gates_once_the_original_is_checked() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    made(&p, "slim-a", true);
    let r = results(&p, "slim-a").unwrap();
    assert_eq!((r["suite_version"].clone(), r["targets"]["original"].clone(), r["targets"]["cut"].clone(), r["targets"]["healed"].clone()), (json!(1), json!(true), json!(true), json!(true)));
    assert!(r["modes"]["full"]["rows"].as_object().unwrap().is_empty() && r["modes"]["quick"]["gates"].as_object().unwrap().is_empty());
    // a cut checked before the original: its row, no verdict
    write_result(&p, "slim-a", "cut", false, m(0.4, 0.2, 0.5, 4000), 1);
    let r = results(&p, "slim-a").unwrap();
    assert!(r["modes"]["full"]["rows"]["cut"].is_object() && r["modes"]["full"]["gates"].as_object().unwrap().is_empty());
    write_result(&p, "slim-a", "original", false, m(1.0, 0.75, 0.0, 800), 1);
    write_result(&p, "slim-a", "healed", false, m(0.95, 0.7, 0.03, 900), 1);
    let r = results(&p, "slim-a").unwrap();
    let g = &r["modes"]["full"]["gates"];
    assert_eq!(g["cut"]["verdict"], "fail");
    assert_eq!(g["cut"]["failed"], json!(["closed_rate", "accuracy", "loop_rate", "median_think_tokens"]));
    assert_eq!(g["healed"]["verdict"], "pass");
    assert!(g.get("original").is_none(), "the original is not judged against itself");
    // quick and full are kept apart
    assert!(r["modes"]["quick"]["rows"].as_object().unwrap().is_empty());
    write_result(&p, "slim-a", "original", true, m(1.0, 0.75, 0.0, 800), 1);
    assert_eq!(results(&p, "slim-a").unwrap()["modes"]["quick"]["rows"].as_object().unwrap().len(), 1);
    // numbers of another suite are not shown, and a damaged file is ignored
    write_result(&p, "slim-a", "healed", false, m(0.95, 0.7, 0.03, 900), 2);
    std::fs::write(p.folder.join("models/slim-a/think_check/cut.full.json"), "{not json").unwrap();
    let r = results(&p, "slim-a").unwrap();
    assert!(r["modes"]["full"]["rows"].get("healed").is_none() && r["modes"]["full"]["rows"].get("cut").is_none() && r["modes"]["full"]["rows"].get("original").is_some());
    assert!(results(&p, "../x").is_err());
}

#[test]
fn the_gguf_files_of_the_exports_are_offered() {
    let d = tempdir().unwrap();
    let p = project(d.path());
    made(&p, "slim-a", false);
    let e = p.folder.join("exports").join("run1-q4_k_m");
    std::fs::create_dir_all(&e).unwrap();
    std::fs::write(e.join("model-q4_k_m.gguf"), "GGUF").unwrap();
    std::fs::write(e.join("meta.json"), json!({"name": "run1-q4_k_m", "gguf_file": "model-q4_k_m.gguf"}).to_string()).unwrap();
    let r = results(&p, "slim-a").unwrap();
    assert_eq!(r["ggufs"], json!([{"name": "run1-q4_k_m", "path": e.join("model-q4_k_m.gguf")}]));
}

/// A stand-in Python that answers the two think-check jobs the way the worker does and keeps the jobs it was given.
#[cfg(unix)]
fn fake_worker(root: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::write(root.join("fake.py"), format!(r#"
import json, os, sys
root = {root:?}
job = json.load(open(sys.argv[1], encoding="utf-8"))
open(os.path.join(root, "jobs.jsonl"), "a", encoding="utf-8").write(json.dumps(job) + "\n")
def out(**kw): print(json.dumps(kw), flush=True)
if job["job"] == "think_check_plan":
    out(type="result", data={{"suite_version": 1, "items": 16 if job["quick"] else 32, "control_items": 8, "backend": job["backend"], "est_tokens": 50000, "est_minutes": 15.0}}); sys.exit(0)
assert job["job"] == "think_check"
out(type="progress", step="think", value=0.5, text="Question 1 of 2")
out(type="think_item", id="arith1", ok=True, closed=True, think_tokens=120, control=False)
if os.path.exists(os.path.join(root, "stop-me")):
    out(type="result", data={{"stopped": True, "main": {{"n": 1}}}}); sys.exit(0)
res = {{"suite_version": 1, "model": job["model"], "backend": job["backend"], "main": {{"n": 2, "closed_rate": 1.0, "accuracy": 1.0, "loop_rate": 0.0, "garbage_rate": 0.0, "median_think_tokens": 100}}, "control": {{"n": 1}}, "stopped": False, "cached": False}}
os.makedirs(os.path.dirname(job["cache_file"]), exist_ok=True)
json.dump({{"key": {{}}, "result": res}}, open(job["cache_file"], "w"))
out(type="result", data=res)
"#, root = root.to_string_lossy())).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\nexec python3 -I {} \"$5\"\n", root.join("fake.py").display())).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
#[test]
fn a_check_runs_the_job_and_its_numbers_show_up_in_the_results() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path());
    let p = project(d.path());
    made(&p, "slim-a", false);
    let mut events = Vec::new();
    let r = run(&p, &spec("slim-a", "original"), Arc::new(AtomicBool::new(false)), &mut |e| events.push(e)).unwrap();
    assert_eq!((r["name"].clone(), r["target"].clone(), r["model"].clone(), r["backend"].clone()), (json!("slim-a"), json!("original"), json!("unsloth/Qwen3.5-27B"), json!("hf")));
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["step"] == "think") && events.iter().any(|e| e["kind"] == "think_item" && e["id"] == "arith1"), "{events:?}");
    run(&p, &spec("slim-a", "cut"), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap();
    let res = results(&p, "slim-a").unwrap();
    assert_eq!(res["modes"]["full"]["rows"].as_object().unwrap().len(), 2);
    assert_eq!(res["modes"]["full"]["gates"]["cut"]["verdict"], "pass");
    let jobs: Vec<Value> = std::fs::read_to_string(d.path().join("jobs.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(jobs.len(), 2);
    assert!(jobs[1]["model"].as_str().unwrap().ends_with("models/slim-a/model") && jobs[1]["cache_file"].as_str().unwrap().ends_with("think_check/cut.full.json"));
    // a stopped check is not saved by the worker, and the STOP file does not stay behind
    std::fs::write(d.path().join("stop-me"), "").unwrap();
    let r = run(&p, &spec("slim-a", "healed"), Arc::new(AtomicBool::new(false)), &mut |_| {});
    assert!(r.is_err(), "no healed model yet");
    made(&p, "slim-b", true);
    let r = run(&p, &spec("slim-b", "healed"), Arc::new(AtomicBool::new(false)), &mut |_| {}).unwrap();
    assert_eq!(r["stopped"], true);
    assert!(!p.folder.join("models/slim-b/think_check/STOP").exists() && !p.folder.join("models/slim-b/think_check/healed.full.json").exists());
    // the plan loads nothing: it asks the worker for numbers only
    let plan_ = plan(&ThinkSpec { target: "gguf".into(), quick: true, ..Default::default() }, &AtomicBool::new(false)).unwrap();
    assert_eq!((plan_["items"].clone(), plan_["backend"].clone()), (json!(16), json!("llama")));
}

#[test]
fn a_cut_is_chosen_by_how_it_thinks_and_by_loss_only_when_the_thinking_ties() {
    use ndforge_core::reshape::thinkcheck::{lost_much, rank_thinking, think_score, THINK_TIE};
    let main = |closed: f64, lp: f64, garbage: f64| json!({"closed_rate": closed, "loop_rate": lp, "garbage_rate": garbage});
    // the numbers of a real run: cut A (loss 1.267, closed 0.50, loops 0.50) against cut B (1.30, closed 0.44, loops 0.59)
    let a = think_score(&main(0.50, 0.50, 0.0));
    let b = think_score(&main(0.44, 0.59, 0.0));
    assert_eq!((a, b), (Some(0.0), Some(0.44 - 0.59)));
    assert_eq!(rank_thinking(&[(Some(1.267), a), (Some(1.30), b)]), Some(0));
    // the lower loss does not win when the other thinks clearly better
    assert_eq!(rank_thinking(&[(Some(1.20), think_score(&main(0.3, 0.6, 0.0))), (Some(1.40), think_score(&main(0.9, 0.05, 0.0)))]), Some(1));
    // within the tie the lower loss decides, in either order
    let close = [(Some(1.40), Some(0.50)), (Some(1.30), Some(0.50 - THINK_TIE + 0.01))];
    assert_eq!(rank_thinking(&close), Some(1));
    assert_eq!(rank_thinking(&[close[1], close[0]]), Some(0));
    // just outside the tie the better thinker wins
    assert_eq!(rank_thinking(&[(Some(1.40), Some(0.50)), (Some(1.30), Some(0.50 - THINK_TIE - 0.01))]), Some(0));
    // no scores: the lowest loss; some scores: a row without one does not compete; nothing: None
    assert_eq!(rank_thinking(&[(Some(1.4), None), (Some(1.3), None)]), Some(1));
    assert_eq!(rank_thinking(&[(Some(1.0), None), (Some(1.5), Some(0.2))]), Some(1));
    assert_eq!(rank_thinking(&[(None, None)]), None);
    assert_eq!(rank_thinking(&[]), None);
    assert_eq!(think_score(&json!({})), None);
    assert_eq!(think_score(&json!({"closed_rate": 0.9})), Some(0.9));
    // "lost much": finishes its reasoning in under 60 % of the original's rate
    assert!(lost_much(&main(0.97, 0.0, 0.0), &main(0.50, 0.5, 0.0)));
    assert!(!lost_much(&main(0.97, 0.0, 0.0), &main(0.60, 0.1, 0.0)));
    assert!(!lost_much(&json!({}), &main(0.1, 0.0, 0.0)));
}

#[test]
fn the_results_page_names_how_a_cut_goes_beyond_the_guardrails() {
    use ndforge_core::reshape::thinkcheck::cut_guardrails;
    let d = tempdir().unwrap();
    let dir = d.path().join("m");
    std::fs::create_dir_all(dir.join("model")).unwrap();
    assert!(cut_guardrails(&dir).is_empty(), "no record, no guess");
    let meta = |dropped: usize, after: u32| json!({"mode": "prune", "period": 4, "dropped": (0..dropped).collect::<Vec<_>>(), "width": {"intermediate_size": [17408, after]}});
    std::fs::write(dir.join("model").join("ndforge_reshape.json"), meta(12, 10240).to_string()).unwrap();      // cut A: three blocks, 58.8 % of the MLP
    let g = cut_guardrails(&dir);
    assert_eq!(g.len(), 1, "{g:?}");
    assert!(g[0].contains("10240 of 17408") && g[0].contains("58.8 %"), "{g:?}");
    std::fs::write(dir.join("model").join("ndforge_reshape.json"), meta(20, 10240).to_string()).unwrap();
    assert_eq!(cut_guardrails(&dir).len(), 2);
    std::fs::write(dir.join("model").join("ndforge_reshape.json"), meta(8, 12288).to_string()).unwrap();
    assert!(cut_guardrails(&dir).is_empty(), "inside the rules");
    std::fs::write(dir.join("model").join("ndforge_reshape.json"), json!({"mode": "grow", "period": 4, "dropped": []}).to_string()).unwrap();
    assert!(cut_guardrails(&dir).is_empty());
}
