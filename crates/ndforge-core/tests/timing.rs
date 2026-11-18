//! Where the time went (timing.rs): the timing lines of synthetic job logs in a temporary folder, the sums, and every advice rule both ways.
use std::path::PathBuf;

use ndforge_core::timing::*;
use serde_json::{json, Value};

/// One log in the format the program saves: the job, the worker's JSON lines with the seconds, the end.
fn log(job: Value, records: &[Value], extra: &[&str]) -> String {
    let mut s = format!("# ND Lore Forge job log, started 2026-11-17 10:00:00 UTC\n# job: {job}\n");
    s.push_str("[     0.1s out] {\"type\": \"log\", \"text\": \"hello\"}\n");
    let mut t = 1.0;
    for r in records {
        t += 3.0;
        s.push_str(&format!("[{t:>8.1}s out] {}\n", json!({"type": "log", "text": format!("timing: {r}")})));
    }
    for e in extra {
        s.push_str(&format!("[{t:>8.1}s out] {}\n", json!({"type": "log", "text": e})));
    }
    s.push_str("# ended after 99.0 s: exit code 0\n");
    s
}

fn rec(stage: &str, seconds: f64, extra: Value) -> Value {
    let mut v = json!({"stage": stage, "seconds": seconds});
    if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
        o.extend(e.clone());
    }
    v
}

#[test]
fn the_timing_lines_of_a_log_are_read_and_the_rest_is_ignored() {
    let text = log(json!({"job": "reshape"}), &[rec("load_model", 12.5, json!({"load": "4bit", "bytes": 5})), rec("write", 30.0, json!({})), rec("job", 50.0, json!({}))], &[]);
    let t = parse_log(&text);
    assert_eq!(t.iter().map(|x| (x.job.as_str(), x.stage.as_str(), x.seconds)).collect::<Vec<_>>(), vec![("reshape", "load_model", 12.5), ("reshape", "write", 30.0), ("reshape", "job", 50.0)]);
    assert_eq!(t[0].fields["load"], "4bit");
    // damaged lines, other lines that mention timing, and a log without a job line
    let messy = "[ 1.0s out] not json timing: {\n[ 1.0s out] {\"type\": \"log\", \"text\": \"timing: {broken\"}\n[ 1.0s err] timing: {\"stage\": \"x\", \"seconds\": 1}\n[ 2.0s out] {\"type\": \"log\", \"text\": \"timing: {\\\"stage\\\": \\\"ok\\\", \\\"seconds\\\": 2.0}\"}\n";
    let t = parse_log(messy);
    assert_eq!(t.len(), 1);
    assert_eq!((t[0].job.as_str(), t[0].stage.as_str()), ("job", "ok"));
    assert!(parse_log("").is_empty());
}

#[test]
fn the_sums_are_per_job_kind_and_stage_with_shares_of_the_job() {
    let a = log(json!({"job": "distill_cache"}), &[rec("load_teacher", 100.0, json!({"load": "4bit"})), rec("teacher_pass", 300.0, json!({})), rec("job", 400.0, json!({}))], &[]);
    let b = log(json!({"job": "distill_cache"}), &[rec("load_teacher", 100.0, json!({"load": "4bit"})), rec("job", 200.0, json!({}))], &[]);
    let s = summarize_texts(&[a, b]);
    assert_eq!(s["logs"], 2);
    assert_eq!(s["job_seconds"], 600.0);
    assert_eq!(s["jobs"]["distill_cache"]["count"], 2);
    assert_eq!(s["jobs"]["distill_cache"]["stages"]["load_teacher"], json!({"seconds": 200.0, "count": 2, "share": 0.33}));
    assert_eq!(s["stages"]["teacher_pass"]["share"], 0.5);
    assert_eq!(s["load_teacher_4bit_seconds"], 200.0);
    // a log without the job's own record counts its stages
    let c = summarize_texts(&[log(json!({"job": "export"}), &[rec("quantize", 10.0, json!({})), rec("convert_f16", 30.0, json!({}))], &[])]);
    assert_eq!(c["job_seconds"], 40.0);
}

fn adv(summary: &Value, id: &str) -> Option<Advice> {
    advice(summary).into_iter().find(|a| a.id == id)
}

#[test]
fn a_saved_4bit_copy_of_the_teacher_is_worth_it_above_ten_percent_of_the_time() {
    let mk = |load: f64| summarize_texts(&[log(json!({"job": "distill_cache"}), &[rec("load_teacher", load, json!({"load": "4bit"})), rec("job", 1000.0, json!({}))], &[])]);
    let yes = adv(&mk(150.0), "teacher_4bit_copy").unwrap();
    assert!(yes.worth && yes.text.contains("15 %") && yes.text.contains("saved 4-bit copy"), "{yes:?}");
    let no = adv(&mk(50.0), "teacher_4bit_copy").unwrap();
    assert!(!no.worth && no.text.contains("would not pay"), "{no:?}");
    // a 16-bit load is no reason
    let sixteen = summarize_texts(&[log(json!({"job": "distill_cache"}), &[rec("load_teacher", 500.0, json!({"load": "16bit"})), rec("job", 1000.0, json!({}))], &[])]);
    assert!(adv(&sixteen, "teacher_4bit_copy").is_none());
}

#[test]
fn reusing_the_16_bit_gguf_needs_a_big_conversion_and_two_exports_of_the_same_model() {
    let export = |model: &str, convert: f64, total: f64| log(json!({"job": "export", "base": {"repo": model}}), &[rec("convert_f16", convert, json!({})), rec("job", total, json!({}))], &[]);
    let yes = summarize_texts(&[export("org/m", 200.0, 1000.0), export("org/m", 200.0, 1000.0)]);
    assert!(adv(&yes, "reuse_f16_gguf").unwrap().worth);
    let other = summarize_texts(&[export("org/a", 200.0, 1000.0), export("org/b", 200.0, 1000.0)]);
    let a = adv(&other, "reuse_f16_gguf").unwrap();
    assert!(!a.worth && a.text.contains("exported once"), "{a:?}");
    let small = summarize_texts(&[export("org/m", 20.0, 1000.0), export("org/m", 20.0, 1000.0)]);
    let a = adv(&small, "reuse_f16_gguf").unwrap();
    assert!(!a.worth && a.text.contains("would not pay"), "{a:?}");
}

#[test]
fn the_merge_is_bound_by_the_processor_or_by_the_disk() {
    let merge = |compute: f64, total: f64| summarize_texts(&[log(json!({"job": "distill_merge"}), &[rec("merge_read", 10.0, json!({})), rec("merge_compute", compute, json!({})), rec("merge_total", total, json!({})), rec("job", total, json!({}))], &[])]);
    let cpu = adv(&merge(900.0, 1000.0), "merge_bound").unwrap();
    assert!(cpu.worth && cpu.text.contains("bound by the processor"), "{cpu:?}");
    let disk = adv(&merge(300.0, 1000.0), "merge_bound").unwrap();
    assert!(disk.text.contains("bound by the disk") && disk.text.contains("fastest drive"), "{disk:?}");
        assert!(adv(&merge(600.0, 900.0), "merge_bound").unwrap().text.contains("disk"), "exactly twice is not more than twice");
    assert!(adv(&summarize_texts(&[]), "merge_bound").is_none());
}

#[test]
fn slow_kernels_are_named_and_a_run_without_them_says_so() {
    let gen = |k: &str| summarize_texts(&[log(json!({"job": "think_check"}), &[rec("think_generate", 100.0, json!({"kernels": k})), rec("job", 120.0, json!({}))], &[])]);
    let slow = adv(&gen("torch"), "fast_kernels").unwrap();
    assert!(slow.worth && slow.text.contains("Install the fast kernels"), "{slow:?}");
    let fast = adv(&gen("fla"), "fast_kernels").unwrap();
    assert!(!fast.worth);
    assert!(adv(&summarize_texts(&[log(json!({"job": "export"}), &[rec("quantize", 5.0, json!({}))], &[])]), "fast_kernels").is_none());
}

#[test]
fn a_teacher_that_slows_down_at_the_end_means_long_rows_dominate() {
    let lines = |speeds: &[u32]| -> Vec<String> { speeds.iter().map(|s| format!("cache_speed: {s} tokens/s ({} tokens so far; this row: 100 tokens)", s * 10)).collect() };
    let mk = |speeds: &[u32]| { let l = lines(speeds); let refs: Vec<&str> = l.iter().map(|x| x.as_str()).collect(); summarize_texts(&[log(json!({"job": "distill_cache"}), &[rec("job", 100.0, json!({}))], &refs)]) };
    assert_eq!(cache_speeds("[ 1.0s out] {\"text\": \"cache_speed: 1,234 tokens/s (5 tokens)\"}\nnothing"), vec![1234.0]);
    let slow = adv(&mk(&[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 400]), "cache_tail").unwrap();
    assert!(slow.worth && slow.text.contains("long rows dominate"), "{slow:?}");
    let steady = adv(&mk(&[1000, 990, 1010, 1000, 995, 1005, 1000, 990, 1000, 980]), "cache_tail").unwrap();
    assert!(!steady.worth);
    assert!(adv(&mk(&[1000, 400]), "cache_tail").is_none(), "too few figures to say");
}

#[test]
fn logs_are_picked_by_the_time_in_their_names_and_the_report_of_the_newest_has_a_summary() {
    let d = tempfile::tempdir().unwrap();
    for (n, kind) in [("20261117-090000", "a"), ("20261117-100000", "b"), ("20261117-110000", "c"), ("20261117-120000", "d")] {
        std::fs::write(d.path().join(format!("{n}-{kind}.log")), "x").unwrap();
    }
    std::fs::write(d.path().join("notes.txt"), "x").unwrap();
    let names = |v: Vec<PathBuf>| v.iter().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect::<Vec<_>>();
    assert_eq!(names(logs_between(d.path(), "2026-11-17T09:30:00Z", "2026-11-17T11:00:00Z")), vec!["20261117-100000-b.log", "20261117-110000-c.log"]);
    assert!(logs_between(d.path(), "bad", "2026-11-17T11:00:00Z").is_empty());
    assert!(logs_between(&d.path().join("nowhere"), "2026-11-17T09:30:00Z", "2026-11-17T11:00:00Z").is_empty());
    let one = d.path().join("20261117-100000-b.log");
    std::fs::write(&one, log(json!({"job": "export"}), &[rec("quantize", 10.0, json!({})), rec("job", 20.0, json!({}))], &[])).unwrap();
    let s = summarize(&[one, d.path().join("missing.log")]);
    assert_eq!(s["logs"], 1);
    assert_eq!(s["job_seconds"], 20.0);
}

#[test]
fn the_history_takes_rates_out_of_the_logs_and_scales_them_by_size() {
    // a teacher of 27 B (54 GB of 16-bit files) answers 1,000 tokens per second; its student of 20 B is merged in 40 s per GB (40 GB written) and trained at 800 tokens per second
    let cache = log(json!({"job": "distill_cache"}), &[rec("load_teacher", 100.0, json!({"load": "4bit", "bytes": 54e9})), rec("teacher_pass", 5000.0, json!({"tokens": 5e6, "tokens_per_s": 1000.0}))], &[]);
    let train = log(json!({"job": "distill"}), &[rec("load_student", 60.0, json!({"method": "qlora"})), rec("train", 6000.0, json!({"tokens": 4.8e6, "tokens_per_s": 800.0}))], &[]);
    let merge = log(json!({"job": "distill_merge"}), &[rec("merge_total", 1600.0, json!({"bytes": 40e9}))], &[]);
    let think = log(json!({"job": "think_check", "quick": true}), &[rec("load_model", 50.0, json!({"load": "16bit", "bytes": 40e9})), rec("think_generate", 800.0, json!({"generated_tokens": 5, "tokens_per_s": 5.0}))], &[]);
    let llama = log(json!({"job": "think_check", "quick": false}), &[rec("think_generate", 800.0, json!({"backend": "llama"}))], &[]);
    let h = History::from_texts(&[cache, train, merge, think, llama]);
    let (t, n) = h.pick("teacher_tok_b_s", 27.0).unwrap();
    assert_eq!(n, 1);
    assert!((t - 27_000.0).abs() < 1.0, "1,000 tokens per second at 27 B: {t}");
    assert!((h.pick("train_tok_b_s", 20.0).unwrap().0 - 16_000.0).abs() < 1.0, "the student's size comes from the merge that follows");
    assert!((h.pick("merge_s_per_gb", 40.0).unwrap().0 - 40.0).abs() < 1e-9);
    assert!((h.pick("think_s_per_b:quick", 20.0).unwrap().0 - 40.0).abs() < 1e-9, "800 s for 20 B");
    assert!(h.pick("think_s_per_b:full", 20.0).is_none(), "a GGUF check has no size and is not used");
    let load = h.pick("load_gb_s:4bit", 54.0).unwrap().0;
    assert!((load - 0.54).abs() < 1e-9, "{load}");
    // the same rate at a model of half the size is twice the tokens per second
    assert!((h.pick("teacher_tok_b_s", 13.5).unwrap().0 / 13.5 - 2000.0).abs() < 1.0);
    assert!(History::from_texts(&[]).is_empty());
    assert!(History::from_texts(&["# job: {\"job\": \"export\"}\n".into()]).is_empty());
}

#[test]
fn the_history_prefers_the_same_size_class_and_the_newest_samples() {
    let mut h = History::default();
    h.push("train_tok_b_s", 8.0, 100.0);
    h.push("train_tok_b_s", 20.0, 300.0);
    h.push("train_tok_b_s", 22.0, 500.0);
    h.push("train_tok_b_s", 21.0, 400.0);
    assert_eq!(h.pick("train_tok_b_s", 21.0), Some((400.0, 3)), "the 8 B sample is another size class; the median of 400, 500, 300");
    assert_eq!(h.pick("train_tok_b_s", 50.0), Some((350.0, 4)), "nothing near: all of them, scaled by the size");
    assert!(h.pick("merge_s_per_gb", 10.0).is_none());
    for i in 0..9 {
        h.push("k", 10.0, i as f64 + 1.0);
    }
    assert_eq!(h.pick("k", 10.0).unwrap().1, 5, "only the newest five");
}
