//! The words of the notifications and the webhook line (notify.rs).
use ndforge_core::notify::*;
use serde_json::json;

#[test]
fn a_toast_names_the_job_and_how_it_ended_and_a_failure_gives_its_first_sentence() {
    assert_eq!(toast("train", &Ok(json!({"loss": 1.0}))), ("Training finished".to_string(), "It is done.".to_string()));
    assert_eq!(toast("heal", &Ok(json!({"stopped": true}))).0, "Healing stopped");
    assert_eq!(toast("export", &Err("Cancelled.".into())).0, "The export stopped");
    let (t, b) = toast("auto", &Err("The disk is full. Free some space and try again. (details: logs/jobs/20261117-x.log) [fix:disk_full]".into()));
    assert_eq!((t.as_str(), b.as_str()), ("Auto-cut failed", "The disk is full."));
    assert_eq!(toast("whatever", &Err(String::new())), ("The job failed".to_string(), "Open the program to see why.".to_string()));
}

#[test]
fn first_sentences_are_cut_at_the_full_stop_and_long_ones_are_shortened() {
    assert_eq!(first_sentence("Version 1.2 is out. Next."), "Version 1.2 is out.");
    assert_eq!(first_sentence("No full stop here"), "No full stop here");
    assert_eq!(first_sentence("llama.cpp was not found [llama_cpp_missing]"), "llama.cpp was not found");
    let long = "word ".repeat(80);
    let s = first_sentence(&long);
    assert!(s.chars().count() <= 160 && s.ends_with('…'));
}

#[test]
fn the_queue_toast_counts_what_happened_and_the_webhook_says_only_kind_and_result() {
    assert_eq!(queue_toast(3, 1, 0, 0), ("The queue is finished".to_string(), "3 done, 1 failed".to_string()));
    assert_eq!(queue_toast(0, 0, 2, 1).1, "1 stopped, 2 skipped");
    let line = webhook_text("heal", "failed");
    assert_eq!(line, "ND Lore Forge: healing failed");
    for kind in ["train", "heal", "cut", "measure", "try", "think_check", "auto", "export", "quantcheck", "imatrix_make", "x"] {
        let l = webhook_text(kind, "finished");
        assert!(!l.contains('/') && !l.contains('\\') && l.len() < 60, "{l}");
    }
}

#[test]
fn only_plain_web_addresses_are_accepted() {
    assert!(valid_webhook("https://ntfy.sh/my-topic") && valid_webhook(" http://192.168.1.5:8080/x "));
    for bad in ["", "ntfy.sh/topic", "ftp://x/y", "https://", "https://a b", "file:///etc/passwd"] {
        assert!(!valid_webhook(bad), "{bad:?}");
    }
    assert!(!send_webhook("not a url", "train", "finished"));
}

#[test]
fn a_webhook_gets_one_plain_text_post() {
    use std::io::{Read, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let t = std::thread::spawn(move || {
        let (mut c, _) = l.accept().unwrap();
        let mut got = Vec::new();
        let mut buf = [0u8; 1024];
        // the head and the body may arrive in separate pieces: read until the announced body is complete
        loop {
            let n = c.read(&mut buf).unwrap();
            got.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&got).to_string();
            if let Some((head, body)) = text.split_once("\r\n\r\n") {
                let want: usize = head.lines().find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                if body.len() >= want || n == 0 {
                    break;
                }
            } else if n == 0 {
                break;
            }
        }
        c.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").unwrap();
        String::from_utf8_lossy(&got).to_string()
    });
    assert!(send_webhook(&format!("http://127.0.0.1:{port}/topic"), "export", "finished"));
    let req = t.join().unwrap();
    assert!(req.starts_with("POST /topic ") && req.ends_with("ND Lore Forge: the export finished"), "{req}");
}

#[test]
fn the_stage_and_end_texts_of_an_auto_cut() {
    use ndforge_core::notify::*;
    use serde_json::json;
    assert_eq!(auto_stage_end("s1.check_heal", true), Some(1));
    assert_eq!(auto_stage_end("s2.heal", true), None);
    assert_eq!(auto_stage_end("s2.heal", false), Some(2));
    assert_eq!(auto_stage_end("export", true), None);
    assert_eq!(stage_toast(1, 2, 47.6, Some(840.0)), ("Auto-cut: stage 1 of 2 healed".to_string(), "48 % done, about 14 h left.".to_string()));
    assert_eq!(stage_toast(1, 2, 47.6, None).1, "48 % done.");
    let done = json!({"progress": {"elapsed_s": 111600.0}, "plan": {"time": {"total_h": [28.0, 40.2]}}});
    assert_eq!(auto_done_body(&done), "It is done. It took 31 h (estimated 28-40 h).");
    assert_eq!(auto_done_body(&json!({})), "It is done.");
    assert_eq!(toast("auto", &Ok(done)).1, "It is done. It took 31 h (estimated 28-40 h).");
    assert_eq!(toast("export", &Ok(json!({}))).1, "It is done.");
    // the webhook still names the kind and the result only
    assert_eq!(webhook_text("auto", "finished"), "ND Lore Forge: auto-cut finished");
}

#[test]
fn the_window_title_comes_from_the_snapshot() {
    use ndforge_core::reshape::auto_progress::title_of;
    use serde_json::json;
    let snap = json!({"pct": 42.3, "state": "live", "left_s": [50400.0, 61200.0], "stage": {"n": 1, "of": 2}, "step": {"label": "Stage 1 of 2: heal"}});
    assert_eq!(title_of(&snap), "Auto-cut 42 % · stage 1 of 2: heal · about 15 h 30 min left");
    let est = json!({"pct": 3.0, "state": "estimating", "left_s": [1.0, 2.0], "stage": null, "step": {"label": "Measure the model"}});
    assert_eq!(title_of(&est), "Auto-cut 3 % · Measure the model");
}

#[test]
fn a_heal_in_rounds_tells_the_person_only_when_it_stops_or_the_merged_model_is_judged() {
    let ev = |d: &str, extra: serde_json::Value| { let mut e = json!({"kind": "heal_round", "round": 3, "tokens_m": 15.0, "cap_m": 40.0, "decision": d}); for (k, v) in extra.as_object().cloned().unwrap_or_default() { e[k] = v; } e };
    assert_eq!(heal_round_toast(&ev("continue", json!({}))), None);
    assert_eq!(heal_round_toast(&ev("baseline", json!({}))), None);
    let (t, b) = heal_round_toast(&ev("stop_pass", json!({}))).unwrap();
    assert_eq!((t.as_str(), b.as_str()), ("Auto-cut: heal round 3 passed the thinking check", "Merging the model and checking the merged model."));
    assert!(heal_round_toast(&ev("stop_plateau", json!({}))).unwrap().1.contains("Neither the thinking nor the loss"));
    assert!(heal_round_toast(&ev("stop_cap", json!({}))).unwrap().0.contains("most tokens (40 M)"));
    assert!(heal_round_toast(&ev("stop_time", json!({}))).unwrap().1.contains("time limit"));
    assert!(heal_round_toast(&ev("merged_fail", json!({}))).unwrap().1.contains("two passed checks in a row"));
    assert_eq!(heal_round_toast(&ev("final", json!({"check": {"verdict": "fail"}, "tokens_m": 40.0}))).unwrap(), ("Auto-cut: the merged model did not pass its thinking check".to_string(), "Not good enough after 40 M tokens.".to_string()));
    let (t, b) = heal_round_toast(&ev("final", json!({"check": {"verdict": "warn"}}))).unwrap();
    assert!(t.contains("passed") && t.contains("warn") && b == "After 15 M tokens of at most 40 M.");
    // the webhook still says the kind and the status only
    assert_eq!(webhook_text("auto", "finished"), "ND Lore Forge: auto-cut finished");
}
