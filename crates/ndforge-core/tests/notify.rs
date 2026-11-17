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
