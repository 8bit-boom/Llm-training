//! What the person is told when a job ends: the words for a desktop notification and the one-line message for an optional webhook (an ntfy-style URL that takes a plain text
//! POST). The notification text names the job and its result in plain words; the webhook gets only the kind of job and how it ended, never a path, a name or an error text.
use serde_json::Value;

/// The job in plain words ("Training", "Healing" ...).
pub fn job_name(kind: &str) -> &'static str {
    match kind {
        "train" => "Training",
        "heal" => "Healing",
        "cut" => "Making the model",
        "measure" => "Measuring the model",
        "try" => "Trying the cuts",
        "think_check" => "The thinking check",
        "auto" => "Auto-cut",
        "export" => "The export",
        "quantcheck" => "Compare quants",
        "imatrix_make" => "The importance matrix",
        _ => "The job",
    }
}

/// How a job ended: "finished" (also what a stopped-by-the-person job that saved its work answers), "stopped" or "failed".
pub fn status_of(r: &Result<Value, String>) -> &'static str {
    match r {
        Ok(d) if d["stopped"] == true => "stopped",
        Ok(_) => "finished",
        Err(m) if m == crate::error::CANCELLED => "stopped",
        Err(_) => "failed",
    }
}

/// The first sentence of an error text, without the marks and the log pointer the window shows separately.
pub fn first_sentence(text: &str) -> String {
    let mut t = text.to_string();
    for cut in [" (details:", " [fix:", " [llama_cpp_missing]"] {
        if let Some(i) = t.find(cut) {
            t.truncate(i);
        }
    }
    let t = t.trim();
    let end = t.char_indices().find(|(i, c)| matches!(c, '.' | '!' | '?') && t[i + c.len_utf8()..].chars().next().is_none_or(|n| n.is_whitespace())).map(|(i, c)| i + c.len_utf8());
    let s = &t[..end.unwrap_or(t.len())];
    if s.chars().count() > 160 { format!("{}…", s.chars().take(157).collect::<String>()) } else { s.to_string() }
}

/// (title, body) of the notification for a job that ended.
pub fn toast(kind: &str, r: &Result<Value, String>) -> (String, String) {
    let name = job_name(kind);
    match status_of(r) {
        "finished" => (format!("{name} finished"), "It is done.".to_string()),
        "stopped" => (format!("{name} stopped"), "It was stopped; it can be continued.".to_string()),
        _ => (format!("{name} failed"), r.as_ref().err().map(|m| first_sentence(m)).filter(|s| !s.is_empty()).unwrap_or_else(|| "Open the program to see why.".into())),
    }
}

/// The notification when the queue has run out: how many ended each way.
pub fn queue_toast(done: usize, failed: usize, skipped: usize, stopped: usize) -> (String, String) {
    let parts: Vec<String> = [(done, "done"), (failed, "failed"), (stopped, "stopped"), (skipped, "skipped")].iter().filter(|(n, _)| *n > 0).map(|(n, w)| format!("{n} {w}")).collect();
    ("The queue is finished".to_string(), parts.join(", "))
}

/// The text a webhook receives: the kind of job and how it ended, nothing else.
pub fn webhook_text(kind: &str, status: &str) -> String {
    format!("ND Lore Forge: {} {}", job_name(kind).to_lowercase(), status)
}

/// A webhook address the program accepts: http or https, no spaces.
pub fn valid_webhook(url: &str) -> bool {
    let u = url.trim();
    (u.starts_with("http://") || u.starts_with("https://")) && !u.contains(char::is_whitespace) && u.len() > 8
}

/// POSTs the text to the webhook (a few seconds at most). The address is never logged or put in an error: it may carry a secret topic.
pub fn send_webhook(url: &str, kind: &str, status: &str) -> bool {
    if !valid_webhook(url) {
        return false;
    }
    ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(5)).build().post(url.trim()).set("Content-Type", "text/plain; charset=utf-8").send_string(&webhook_text(kind, status)).is_ok()
}
