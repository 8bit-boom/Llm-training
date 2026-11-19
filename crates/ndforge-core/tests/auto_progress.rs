use ndforge_core::reshape::auto_progress::*;
use serde_json::json;

fn st(id: &str, status: &str, est: (f64, f64), frac: f64) -> StepIn {
    StepIn { id: id.into(), status: status.into(), est, frac }
}

fn heal_parts() -> Vec<(String, f64)> {
    vec![("start".into(), 2.0), ("load".into(), 8.0), ("cache".into(), 60.0), ("load".into(), 5.0), ("train".into(), 100.0), ("merge".into(), 25.0)]
}

#[test]
fn weights_make_the_percentage_and_skipped_steps_leave_it() {
    assert_eq!(weight((0.0, 0.5)), 1.0);
    assert_eq!(weight((10.0, 30.0)), 20.0);
    let steps = vec![st("plan", "done", (1.0, 1.0), 1.0), st("try", "skipped", (50.0, 50.0), 0.0), st("s1.heal", "running", (80.0, 120.0), 0.5), st("export", "pending", (20.0, 20.0), 0.0)];
    let o = overall(&steps);
    assert_eq!(o.total_min, 121.0);
    assert_eq!(o.done_min, 51.0);
    assert!((o.pct - 42.1).abs() < 0.05, "{}", o.pct);
}

#[test]
fn a_retry_adds_weight_so_the_bar_can_step_back_once() {
    let before = vec![st("s1.heal", "done", (100.0, 100.0), 1.0), st("s1.check_heal", "done", (20.0, 20.0), 1.0), st("export", "pending", (30.0, 30.0), 0.0)];
    let a = overall(&before);
    let mut after = before.clone();
    after[0].status = "running".into();
    after[0].frac = 0.0;
    after[1].status = "pending".into();
    let b = overall(&after);
    assert!(b.pct < a.pct);
    assert_eq!(b.total_min, a.total_min);
    after.push(st("retry.heal", "pending", (200.0, 200.0), 0.0));
    assert!(overall(&after).total_min > a.total_min);
}

#[test]
fn heal_events_move_through_the_parts() {
    let parts = heal_parts();
    let total: f64 = parts.iter().map(|p| p.1).sum();
    let cache = json!({"kind": "progress", "step": "cache", "value": 0.5, "text": "", "detail": {"unit": "tokens", "done": 12.4e6, "total": 30.0e6, "rate": 3100.0, "rate_unit": "tokens/s", "eta_s": 2460.0}});
    let f = step_fraction("s1.heal", &cache, &parts).unwrap();
    assert!((f.f - (2.0 + 8.0 + 30.0) / total).abs() < 1e-9);
    assert_eq!(f.part.as_deref(), Some("cache"));
    assert_eq!(f.live_eta_s, Some(2460.0));
    assert_eq!(f.after_min, 130.0);
    assert_eq!(f.sub.as_deref(), Some("Teacher cache: 12.4 M of 30.0 M tokens · 3,100 tokens/s · 41 min left"));
    let train = json!({"kind": "metric", "step": 50, "total_steps": 100, "eta_s": 600.0, "tok_s": 1234.0});
    let f = step_fraction("s1.heal", &train, &parts).unwrap();
    assert_eq!(f.part.as_deref(), Some("train"));
    assert!(f.f > 0.6 && f.f < 0.9);
    assert!(f.sub.unwrap().starts_with("Training: step 50 of 100 · 1,234 tokens/s"));
    let merge = json!({"kind": "progress", "step": "merge", "value": 1.0, "text": ""});
    assert!((step_fraction("s1.heal", &merge, &parts).unwrap().f - 1.0).abs() < 1e-9);
    // unknown events say nothing; a step without the part uses the event's own fraction
    assert!(step_fraction("s1.heal", &json!({"kind": "log"}), &parts).is_none());
    let f = step_fraction("measure", &json!({"kind": "progress", "step": "score", "value": 0.25}), &[]).unwrap();
    assert_eq!(f.f, 0.25);
    assert_eq!(f.part, None);
    let f = step_fraction("try", &json!({"kind": "try_pass", "done": 3, "total": 12}), &[]).unwrap();
    assert_eq!(f.f, 0.25);
}

#[test]
fn the_smoother_says_estimating_then_live() {
    let mut s = EtaSmoother::default();
    s.update(0.0, 0.0, None);
    assert_eq!(s.state(30.0, 0.01), EtaState::Estimating);
    // plan's figure while estimating: 1000 s planned, 1 % done
    assert!((s.remaining_s(30.0, 0.01, 1000.0) - 990.0).abs() < 1e-9);
    s.update(61.0, 0.02, None);
    s.update(122.0, 0.04, None);
    assert_eq!(s.state(122.0, 0.04), EtaState::Estimating);
    s.update(183.0, 0.06, None);
    assert_eq!(s.state(183.0, 0.06), EtaState::Live);
    let left = s.remaining_s(183.0, 0.06, 1000.0);
    // 0.02 per 61 s -> 0.94 left -> about 2870 s, but the display may not jump up by more than 20 % a minute
    assert!(left > 900.0, "{left}");
    let f_only = EtaSmoother::default();
    assert_eq!(f_only.state(10.0, 0.05), EtaState::Live);
}

#[test]
fn the_display_never_jumps_up_and_prefers_the_worker_figure() {
    let mut s = EtaSmoother::default();
    s.update(0.0, 0.0, None);
    s.update(61.0, 0.1, Some(500.0));
    let a = s.remaining_s(61.0, 0.1, 5000.0);
    assert!((a - 500.0).abs() < 1e-9);
    // the worker suddenly says 5000 s: one minute later the display may be at most about 20 % above what the countdown gives
    s.update(122.0, 0.11, Some(5000.0));
    let b = s.remaining_s(122.0, 0.11, 5000.0);
    assert!(b <= (500.0 - 61.0) * 1.21 + 2.0, "{b}");
}

#[test]
fn calibration_is_clamped_and_needs_a_finished_step_of_the_kind() {
    let done = vec![("s1.heal".to_string(), 100.0, 150.0), ("plan".to_string(), 1.0, 1.0)];
    assert_eq!(calibration(&done, "s2.heal"), Some(1.5));
    assert_eq!(calibration(&done, "export"), None);
    assert_eq!(calibration(&[("s1.heal".to_string(), 10.0, 100.0)], "s2.heal"), Some(3.0));
    assert_eq!(calibration(&[("s1.heal".to_string(), 100.0, 10.0)], "s2.heal"), Some(0.5));
    let (lo, hi) = remaining(Some(600.0), 10.0, &[("s2.heal".into(), (100.0, 200.0)), ("export".into(), (10.0, 20.0))], &done);
    assert!((lo - (10.0 * 0.9 + 9.0 + 150.0 + 10.0)).abs() < 1e-9);
    assert!((hi - (12.5 + 12.5 + 300.0 + 20.0)).abs() < 1e-9);
}

#[test]
fn texts() {
    assert_eq!(stage_of("s2.heal"), Some(2));
    assert_eq!(stage_of("export"), None);
    assert_eq!(kind_of("s1.check_heal"), "check_heal");
    assert_eq!(short_duration(41.0), "41 min");
    assert_eq!(short_duration(125.0), "2 h 5 min");
    assert_eq!(short_duration(120.0), "2 h");
    assert_eq!(title_text(42.3, Some((1, 2)), "heal", Some(840.0)), "Auto-cut 42 % · stage 1 of 2: heal · about 14 h left");
    assert_eq!(title_text(7.0, Some((1, 1)), "measure", None), "Auto-cut 7 % · measure");
    let ev = json!({"kind": "progress", "step": "merge", "detail": {"unit": "GB", "done": 12.0, "total": 34.0}});
    assert_eq!(sub_status("merge", &ev, None).unwrap(), "Saving the healed model: 12.0 of 34.0 GB");
}
