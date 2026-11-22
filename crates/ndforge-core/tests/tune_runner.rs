//! The decisions ("best" beats the default by more than the noise, within the memory and the same results) and the runner with a fake: order, state, resume, stop, waiting for the card, failures, the preset at the end.
use ndforge_core::error::{Error, Result};
use ndforge_core::tune::decide::*;
use ndforge_core::tune::identity::{machine_from, ModelIdentity};
use ndforge_core::tune::runner::*;
use ndforge_core::tune::store::{Scope, Store};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

fn row(b: u32, tps: f64, peak: f64, loss: f64) -> Value {
    json!({"batch_tokens": b, "tokens_per_s": tps, "peak_vram_mb": peak, "first_loss": loss})
}

#[test]
fn the_fastest_feasible_budget_wins_when_it_beats_the_noise() {
    let rows = vec![row(0, 324.0, 21000.0, 2.0), row(2048, 610.0, 22000.0, 2.0), row(4096, 780.0, 24000.0, 2.0001), row(8192, 800.0, 25000.0, 2.0), json!({"batch_tokens": 16384, "tokens_per_s": null, "error": "out of memory", "oom": true})];
    let d = decide_batch(&rows, Some(32607.0), 0.02);
    assert_eq!((d.chosen, d.changed), (8192, true), "{d:?}");
    assert!((d.gain - 2.47).abs() < 0.01);
    assert!(d.guards.iter().all(|g| g.ok) && d.guards.iter().any(|g| g.name == "peak_vram") && d.guards.iter().any(|g| g.name == "first_loss"));
}

#[test]
fn a_tie_within_the_noise_goes_to_the_lower_peak_memory() {
    let rows = vec![row(0, 324.0, 21000.0, 2.0), row(4096, 780.0, 20000.0, 2.0), row(8192, 790.0, 27000.0, 2.0)];
    let d = decide_batch(&rows, Some(32607.0), 0.02);
    assert_eq!(d.chosen, 4096, "790 against 780 is a tie at 2 % noise: the one with more headroom wins");
}

#[test]
fn a_gain_inside_twice_the_noise_keeps_the_default_and_says_so() {
    let rows = vec![row(0, 400.0, 20000.0, 2.0), row(2048, 440.0, 21000.0, 2.0)];
    let d = decide_batch(&rows, Some(32607.0), 0.08);
    assert_eq!((d.chosen, d.changed), (0, false));
    assert!(d.why.contains("Within noise (±8.0 %)"), "{}", d.why);
    assert_eq!(decide_batch(&rows, Some(32607.0), 0.01).chosen, 2048, "10 % beats a 1 % noise");
}

#[test]
fn a_candidate_above_90_percent_of_the_card_or_with_another_first_loss_is_excluded_and_named() {
    let mut worst = row(8192, 900.0, 25000.0, 2.0);
    worst["worst_peak_mb"] = json!(31000.0);
    let rows = vec![row(0, 324.0, 21000.0, 2.0), worst, row(4096, 700.0, 22000.0, 2.003)];
    let d = decide_batch(&rows, Some(32607.0), 0.02);
    assert_eq!(d.chosen, 0);
    assert!(d.why.contains("8192: peak 31000 MB is above 90 % of the card") && d.why.contains("4096: faster but NOT equivalent, the first loss differs by 3.0e-3"), "{}", d.why);
    assert_eq!(decide_batch(&[row(2048, 500.0, 1.0, 2.0)], None, 0.0).chosen, 0, "without the baseline nothing is compared");
}

#[test]
fn the_loss_rows_decide_the_chunk_and_the_logits() {
    let rows = vec![json!({"id": "current", "chunk": 128, "head_checkpoint": true, "ok": true, "tokens_per_s": 300.0}), json!({"id": "chunk_256", "chunk": 256, "head_checkpoint": true, "ok": true, "tokens_per_s": 318.0}),
                    json!({"id": "keep_logits_chunk_256", "chunk": 256, "head_checkpoint": false, "ok": true, "tokens_per_s": 345.0}), json!({"id": "chunk_512", "ok": false, "oom": true})];
    let d = decide_loss(&rows, 0.0);
    assert_eq!((d.chunk, d.head_checkpoint, d.changed), (256, false, true), "{d:?}");
    assert!((d.gain - 1.15).abs() < 0.01);
    let flat = vec![json!({"id": "current", "chunk": 128, "head_checkpoint": true, "ok": true, "tokens_per_s": 300.0}), json!({"id": "chunk_256", "chunk": 256, "head_checkpoint": true, "ok": true, "tokens_per_s": 305.0})];
    let k = decide_loss(&flat, 0.0);
    assert_eq!((k.chunk, k.changed), (128, false), "1.7 % is inside the default 3 % noise");
}

#[test]
fn the_noise_rule_has_the_workers_numbers() {
    let py = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../worker/ndforge_worker/healspeed.py")).unwrap();
    for (name, v) in [("VRAM_SHARE_LIMIT", VRAM_SHARE_LIMIT), ("LOSS_TOLERANCE", LOSS_TOLERANCE), ("MIN_SPEEDUP", MIN_GAIN), ("NOISE_FACTOR", NOISE_FACTOR)] {
        let line = py.lines().find(|l| l.starts_with(&format!("{name} = "))).unwrap_or_else(|| panic!("{name} is not in healspeed.py"));
        let n: f64 = line.split('=').nth(1).unwrap().split('#').next().unwrap().trim().parse().unwrap();
        assert_eq!(n, v, "{name}");
    }
}

// ---- the runner with a fake
struct Fx {
    events: Vec<Value>,
    outs: BTreeMap<String, Result<TestOut>>,
    stop_after_tests: Option<usize>,
    ran: Vec<String>,
    gpu: Option<(u64, u64, Vec<String>)>,
    dirs: Vec<std::path::PathBuf>,
}

impl TuneFx for Fx {
    fn emit(&mut self, ev: Value) { self.events.push(ev); }
    fn stopped(&self) -> bool { self.stop_after_tests.is_some_and(|n| self.ran.len() >= n) }
    fn now(&self) -> String { "2026-11-22T10:00:00Z".into() }
    fn gpu(&self) -> Option<(u64, u64, Vec<String>)> { self.gpu.clone() }
    fn run_test(&mut self, d: &TestDef, ctx: &TestCtx) -> Result<TestOut> {
        self.ran.push(d.id.into());
        std::fs::create_dir_all(&ctx.dir).unwrap();
        std::fs::write(ctx.dir.join("scratch.txt"), "x").unwrap();
        self.dirs.push(ctx.dir.clone());
        self.outs.remove(d.id).unwrap_or_else(|| Err(Error::new("no fake output")))
    }
}

fn fx() -> Fx {
    let mut outs = BTreeMap::new();
    outs.insert("heal.student_batch".to_string(), Ok(TestOut { rows: vec![row(0, 324.0, 21000.0, 2.0), row(4096, 780.0, 24000.0, 2.0)], raw: json!({"spread": 0.02, "rows": {"mean": 173}}), ..Default::default() }));
    outs.insert("heal.loss".to_string(), Ok(TestOut { rows: vec![json!({"id": "current", "chunk": 128, "head_checkpoint": true, "ok": true, "tokens_per_s": 300.0}), json!({"id": "keep_logits_chunk_256", "chunk": 256, "head_checkpoint": false, "ok": true, "tokens_per_s": 345.0})], ..Default::default() }));
    Fx { events: vec![], outs, stop_after_tests: None, ran: vec![], gpu: Some((30000, 32607, vec![])), dirs: vec![] }
}

fn setup(t: &tempfile::TempDir, tests: &[&str], max_hours: Option<f64>) -> (std::path::PathBuf, RunState, Store) {
    let proj = t.path().join("proj");
    std::fs::create_dir_all(&proj).unwrap();
    let mut card = Map::new();
    card.insert("seq".into(), json!(4096));
    card.insert("kl_chunk".into(), json!(128));
    let spec = Spec { model: "slim-w".into(), tests: tests.iter().map(|s| s.to_string()).collect(), card, max_hours, trial_tokens_k: 0 };
    let model = ModelIdentity { name: "slim-w".into(), weights_id: "w-1".into(), shape_id: "s-1".into(), params: 20_100_000_000, ..Default::default() };
    let st = new_run(spec, model, machine_from("RTX 5090", 32607, "621", "windows", BTreeMap::new()), 20.1, "2026-11-22T10:00:00Z");
    (proj.clone(), st, Store { project: Some(proj), library: t.path().join("lib") })
}

#[test]
fn a_run_goes_through_the_tests_in_order_saves_a_preset_and_removes_its_temp_folders() {
    let t = tempfile::tempdir().unwrap();
    let (proj, mut st, store) = setup(&t, &["heal.loss", "heal.student_batch"], None);
    assert_eq!(st.tests.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), vec!["heal.student_batch", "heal.loss"], "ordered by value, not by the order they were ticked");
    let mut f = fx();
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!(st.status, "done");
    assert_eq!(f.ran, vec!["heal.student_batch", "heal.loss"]);
    let preset = store.get(Scope::Project, st.preset_id.as_ref().unwrap()).unwrap();
    let v = preset.values();
    assert_eq!((v["student_batch_tokens"].clone(), v["kl_chunk"].clone(), v["head_checkpoint"].clone(), v["seq"].clone()), (json!(4096), json!(256), json!("off"), json!(4096)), "the winners on top of the whole card");
    assert!(preset.name.starts_with("Tested 2026-11-22 (slim-w)") && preset.functions["heal.student_batch"].gain > 2.0 && preset.functions["heal.student_batch"].class == "tol");
    assert!(f.dirs.iter().all(|d| !d.exists() && d.to_string_lossy().contains("tune-") && d.to_string_lossy().contains("tmp")), "every temp folder is under benchmarks/tune-*/tmp and is gone");
    assert!(proj.join("benchmarks").join(format!("tune-{}", st.id)).join("tune_report.md").is_file() && proj.join("tune").join("runs").join(&st.id).join("state.json").is_file());
    assert!(f.events.iter().any(|e| e["kind"] == "done") && f.events.iter().filter(|e| e["kind"] == "tune_test" && e["status"] == "running").count() == 2);
    let md = std::fs::read_to_string(proj.join("benchmarks").join(format!("tune-{}", st.id)).join("tune_report.md")).unwrap();
    assert!(md.contains("quality knobs are based on a short trial"));
}

#[test]
fn stop_ends_between_tests_and_continue_skips_the_done_ones() {
    let t = tempfile::tempdir().unwrap();
    let (proj, mut st, store) = setup(&t, &["heal.student_batch", "heal.loss"], None);
    let mut f = fx();
    f.stop_after_tests = Some(1);
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!((st.status.as_str(), st.tests[0].status.as_str(), st.tests[1].status.as_str()), ("stopped", "done", "queued"));
    assert!(st.preset_id.is_none());
    let mut again = load_state(&proj, &st.id).unwrap();
    f.stop_after_tests = None;
    run(&mut f, &proj, &mut again, Some(32607.0), &store).unwrap();
    assert_eq!(f.ran, vec!["heal.student_batch", "heal.loss"], "the first test was not run again");
    assert_eq!(again.status, "done");
}

#[test]
fn a_stopped_test_reruns_from_its_start() {
    let t = tempfile::tempdir().unwrap();
    let (proj, mut st, store) = setup(&t, &["heal.student_batch"], None);
    let mut f = fx();
    f.outs.insert("heal.student_batch".into(), Ok(TestOut { stopped: true, ..Default::default() }));
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!((st.status.as_str(), st.tests[0].status.as_str()), ("stopped", "stopped"));
    f.outs.insert("heal.student_batch".into(), Ok(TestOut { rows: vec![row(0, 324.0, 1.0, 2.0), row(2048, 700.0, 1.0, 2.0)], ..Default::default() }));
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!((st.status.as_str(), st.tests[0].status.as_str()), ("done", "done"));
    assert_eq!(f.ran.len(), 2);
}

#[test]
fn the_card_with_another_programs_memory_makes_the_run_wait_and_nothing_is_ended() {
    let t = tempfile::tempdir().unwrap();
    let (proj, mut st, store) = setup(&t, &["heal.student_batch"], None);
    let mut f = fx();
    f.gpu = Some((9000, 32607, vec!["chrome.exe 900 MB".into()]));
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!((st.status.as_str(), st.tests[0].status.as_str()), ("waiting_gpu", "waiting_gpu"));
    assert!(st.note.contains("Free GPU") && st.note.contains("chrome.exe") && f.ran.is_empty());
    assert!(f.events.iter().any(|e| e["kind"] == "tune_wait"));
    f.gpu = Some((30000, 32607, vec![]));
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!(st.status, "done");
}

#[test]
fn a_failed_test_is_recorded_and_the_run_goes_on() {
    let t = tempfile::tempdir().unwrap();
    let (proj, mut st, store) = setup(&t, &["heal.student_batch", "heal.loss"], None);
    let mut f = fx();
    f.outs.insert("heal.student_batch".into(), Err(Error::new("The worker crashed")));
    run(&mut f, &proj, &mut st, Some(32607.0), &store).unwrap();
    assert_eq!((st.tests[0].status.as_str(), st.tests[1].status.as_str(), st.status.as_str()), ("failed", "done", "done"));
    let p = store.get(Scope::Project, st.preset_id.as_ref().unwrap()).unwrap();
    assert!(p.notes.iter().any(|n| n.contains("heal.student_batch: failed")) && p.functions.contains_key("heal.loss") && !p.functions.contains_key("heal.student_batch"));
}

#[test]
fn the_hours_cap_drops_the_tail_and_a_test_this_version_cannot_run_is_not_planned() {
    let t = tempfile::tempdir().unwrap();
    let (_, st, _) = setup(&t, &["heal.student_batch", "heal.loss", "heal.quality", "disk"], Some(0.6));
    assert_eq!(st.tests.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), vec!["heal.student_batch"], "36 minutes cover the first test only; the others are not built or do not fit");
    let spec = Spec { model: "m".into(), tests: vec!["heal.student_batch".into(), "heal.quality".into()], ..Default::default() };
    let v = plan_view(&spec, 20.1, 1_000_000, &BTreeMapNotes::new());
    let q = v["tests"].as_array().unwrap().iter().find(|x| x["id"] == "heal.quality").unwrap();
    assert!(q["unavailable"].as_str().unwrap().contains("not built") && q["tier"] == 2);
    let total = v["total_minutes"].as_array().unwrap();
    assert!(total[0].as_f64().unwrap() >= 15.0 && total[1].as_f64().unwrap() <= 26.0, "only the test that can run counts: {total:?}");
    assert!(v["done_between"][1].as_i64().unwrap() > v["done_between"][0].as_i64().unwrap());
}

#[test]
fn the_state_round_trips_and_a_bad_run_id_is_refused() {
    let t = tempfile::tempdir().unwrap();
    let (proj, st, _) = setup(&t, &["heal.student_batch"], None);
    save_state(&proj, &st).unwrap();
    assert_eq!(load_state(&proj, &st.id).unwrap().spec.model, "slim-w");
    assert!(load_state(&proj, "../x").is_err() && load_state(&proj, "").is_err() && load_state(&proj, "20990101-0000").is_err());
    assert_eq!(list_runs(&proj).len(), 1);
}

#[test]
fn the_cards_values_become_a_heal_spec_and_automatic_values_are_left_out() {
    use ndforge_core::tune::real::heal_spec_of;
    let card: Map<String, Value> = json!({"seq": 4096, "batch": 1, "per_update": 32, "checkpointing": "auto", "rank": 64, "method": "qlora", "optimizer": "adamw_8bit", "kl_chunk": 128, "k": 32, "teacher_load": null,
                                          "warmup": 30, "head_checkpoint": "on", "student_batch_tokens": 0}).as_object().unwrap().clone();
    let s = heal_spec_of(&card).unwrap();
    assert_eq!((s.seq, s.rank, s.kl_chunk, s.per_update), (Some(4096), Some(64), Some(128), Some(32)));
    assert!(s.student_batch_tokens.is_none() && s.head_checkpoint.is_none() && s.checkpointing.is_none() && s.teacher_load.is_none(), "zero, on and auto mean automatic");
    let mut tuned = card.clone();
    tuned.insert("student_batch_tokens".into(), json!(8192));
    tuned.insert("head_checkpoint".into(), json!("off"));
    let t = heal_spec_of(&tuned).unwrap();
    assert_eq!((t.student_batch_tokens, t.head_checkpoint.as_deref()), (Some(8192), Some("off")));
    assert!(heal_spec_of(&json!({"seq": "wide"}).as_object().unwrap().clone()).is_err());
}
