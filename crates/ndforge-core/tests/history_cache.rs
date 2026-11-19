//! `History::recent` keeps its last answer in memory while the saved job logs are the same ones (name, size, modification time): the window asks for a plan after every click.
//! Its own test binary: it sets NDFORGE_CONFIG_DIR, which would race with other tests.
use ndforge_core::timing::History;

/// A log of a merge job: `seconds` for 40 GB written, so the merge rate follows it.
fn merge_log(seconds: f64) -> String {
    format!("# ND Lore Forge job log, started 2026-11-17 10:00:00 UTC\n# job: {{\"job\": \"distill_merge\"}}\n[     4.0s out] {{\"type\": \"log\", \"text\": \"timing: {{\\\"stage\\\": \\\"merge_total\\\", \\\"seconds\\\": {seconds}, \\\"bytes\\\": 40e9}}\"}}\n# ended after 99.0 s: exit code 0\n")
}

#[test]
fn the_history_is_read_once_per_set_of_logs_and_again_when_a_log_changes_or_comes() {
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path());
    let logs = d.path().join("logs").join("jobs");
    std::fs::create_dir_all(&logs).unwrap();
    std::fs::write(logs.join("20261117-100000-distill_merge-1.log"), merge_log(1600.0)).unwrap();
    let rate = |h: &History| h.pick("merge_s_per_gb", 40.0).map(|x| x.0);
    let first = History::recent(60);
    assert!((rate(&first).unwrap() - 40.0).abs() < 1e-9);
    assert_eq!(History::recent(60), first, "the same logs: the same answer (from memory)");
    // a log is rewritten with another size: it is read again
    std::fs::write(logs.join("20261117-100000-distill_merge-1.log"), merge_log(2000.0) + "# a longer log\n").unwrap();
    let changed = History::recent(60);
    assert!((rate(&changed).unwrap() - 50.0).abs() < 1e-9, "{:?}", rate(&changed));
    // a new log comes: it is the newest sample
    std::fs::write(logs.join("20261117-110000-distill_merge-2.log"), merge_log(800.0)).unwrap();
    let more = History::recent(60);
    assert_ne!(more, changed, "the new log is part of the history");
    assert_ne!(rate(&more), rate(&changed));
    // fewer logs asked for is another question
    assert!((rate(&History::recent(1)).unwrap() - 20.0).abs() < 1e-9);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
