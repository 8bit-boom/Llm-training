//! The job queue state machine (queue.rs): pure, then the file.
use ndforge_core::queue::{label_of, preflight_spec, Queue, KINDS};
use serde_json::json;
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

fn add(q: &mut Queue, kind: &str) -> String {
    q.enqueue("/p", kind, json!({"run": "r1", "quant": "q4_k_m"})).unwrap()
}
fn statuses(q: &Queue) -> Vec<&str> {
    q.items.iter().map(|i| i.status.as_str()).collect()
}

#[test]
fn items_run_one_at_a_time_in_order_and_a_failure_does_not_stop_the_rest() {
    let mut q = Queue::default();
    let (a, b, c) = (add(&mut q, "export"), add(&mut q, "heal"), add(&mut q, "auto"));
    assert_eq!(q.next_runnable(), Some(0));
    q.mark(&a, "running", None, None).unwrap();
    assert_eq!(q.next_runnable(), None, "one at a time");
    assert!(q.items[0].started.is_some());
    q.mark(&a, "failed", Some("Disk full.".into()), Some("logs/jobs/x.log".into())).unwrap();
    assert_eq!((q.items[0].error.as_deref(), q.items[0].log.as_deref(), q.items[0].finished.is_some()), (Some("Disk full."), Some("logs/jobs/x.log"), true));
    assert_eq!(q.next_runnable(), Some(1), "the next one goes on");
    q.mark(&b, "skipped", Some("Not enough free space.".into()), None).unwrap();
    q.mark(&c, "running", None, None).unwrap();
    q.mark(&c, "done", None, None).unwrap();
    assert_eq!(statuses(&q), ["failed", "skipped", "done"]);
    assert_eq!(q.next_runnable(), None);
    assert!(q.mark("nope", "done", None, None).is_err() && q.mark(&a, "weird", None, None).is_err());
}

#[test]
fn a_paused_queue_starts_nothing_and_only_known_kinds_are_taken() {
    let mut q = Queue::default();
    add(&mut q, "export");
    q.paused = true;
    assert_eq!(q.next_runnable(), None);
    q.paused = false;
    assert_eq!(q.next_runnable(), Some(0));
    assert!(q.enqueue("/p", "format_disk", json!({})).is_err());
    for k in KINDS {
        assert!(q.enqueue("/p", k, json!({})).is_ok(), "{k}");
    }
}

#[test]
fn only_waiting_items_move_and_never_in_front_of_what_has_run() {
    let mut q = Queue::default();
    let ids: Vec<String> = (0..4).map(|_| add(&mut q, "export")).collect();
    q.mark(&ids[0], "running", None, None).unwrap();
    q.mark(&ids[0], "done", None, None).unwrap();
    q.reorder(&ids[3], 0).unwrap();
    assert_eq!(q.items.iter().map(|i| i.id.clone()).collect::<Vec<_>>(), [ids[0].clone(), ids[3].clone(), ids[1].clone(), ids[2].clone()], "not in front of the finished one");
    q.reorder(&ids[1], 99).unwrap();
    assert_eq!(q.items.last().unwrap().id, ids[1]);
    assert!(q.reorder(&ids[0], 2).is_err(), "a finished entry stays");
    assert!(q.reorder("nope", 0).is_err());
}

#[test]
fn a_running_item_is_not_removed_and_finished_ones_can_be_cleared() {
    let mut q = Queue::default();
    let (a, b, c) = (add(&mut q, "export"), add(&mut q, "export"), add(&mut q, "export"));
    q.mark(&a, "running", None, None).unwrap();
    assert!(q.remove(&a).is_err());
    q.remove(&b).unwrap();
    q.mark(&a, "stopped", None, None).unwrap();
    q.clear_finished();
    assert_eq!(q.items.len(), 1);
    assert_eq!(q.items[0].id, c);
    assert!(q.remove("gone").is_err());
}

#[test]
fn after_a_restart_what_was_running_is_stopped_and_the_queue_waits() {
    let mut q = Queue::default();
    let (a, _b) = (add(&mut q, "export"), add(&mut q, "heal"));
    q.mark(&a, "running", None, None).unwrap();
    q.recover();
    assert_eq!(statuses(&q), ["stopped", "queued"]);
    assert!(q.paused && q.items[0].error.as_deref().unwrap().contains("closed"));
    assert_eq!(q.next_runnable(), None, "it waits for Run");
    let mut empty = Queue::default();
    empty.recover();
    assert!(!empty.paused, "nothing to wait for");
}

#[test]
fn the_queue_is_kept_in_the_settings_folder_and_a_damaged_file_is_set_aside() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
    let mut q = Queue::load();
    assert!(q.items.is_empty());
    let id = add(&mut q, "export");
    q.paused = true;
    q.save().unwrap();
    let back = Queue::load();
    assert_eq!((back.items.len(), back.items[0].id.clone(), back.paused), (1, id, true));
    assert_eq!(back.items[0].label, "Export r1 as q4_k_m");
    std::fs::write(ndforge_core::queue::file(), "{not json").unwrap();
    assert!(Queue::load().items.is_empty());
    assert!(std::fs::read_dir(d.path()).unwrap().flatten().any(|e| e.file_name().to_string_lossy().starts_with("queue.json.broken")));
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn labels_and_the_preflight_questions_follow_the_start_arguments() {
    assert_eq!(label_of("export", &json!({"run": "r", "quants": ["q4_k_m", "q8_0"]})), "Export r as q4_k_m, q8_0");
    assert_eq!(label_of("heal", &json!({"name": "m"})), "Heal m");
    assert_eq!(label_of("auto", &json!({"resume": "run1"})), "Continue Auto-cut");
    assert_eq!(label_of("auto", &json!({"spec": {"target_b": 21.1}})), "Auto-cut to 21.1 B");
    let p = preflight_spec("export", &json!({"run": "r", "what": "gguf", "quants": ["mixed_4"], "imatrix": {"enabled": true}}));
    assert_eq!((p["imatrix"].clone(), p["quants"][0].clone()), (json!(true), json!("mixed_4")));
    let m = preflight_spec("export", &json!({"run": "r", "what": "merged", "quants": ["q4_k_m"], "imatrix": {"enabled": true}}));
    assert_eq!((m["imatrix"].clone(), m["quants"].clone()), (json!(false), json!([])));
    assert_eq!(preflight_spec("heal", &json!({"name": "m", "spec": {"tokens_m": 20, "mode": "cache"}}))["tokens_m"], 20);
    assert_eq!(preflight_spec("quantcheck", &json!({"spec": {}})), json!({}));
}
