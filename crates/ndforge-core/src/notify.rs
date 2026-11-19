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
        "quantfit" => "The measured fit",
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

/// The stage of an Auto-cut that this step ends, when the person is told about it: the thinking check of the healed model, or the heal itself in a run without thinking.
pub fn auto_stage_end(step_id: &str, keep_thinking: bool) -> Option<u32> {
    let rest = step_id.strip_prefix('s')?;
    let (n, what) = rest.split_once('.')?;
    let n: u32 = n.parse().ok()?;
    (what == if keep_thinking { "check_heal" } else { "heal" }).then_some(n)
}

/// "Auto-cut: stage 1 of 2 healed" and "48 % done, about 14 h left" (the time left is left out while it is still an estimate).
pub fn stage_toast(n: u32, of: u32, pct: f64, left_min: Option<f64>) -> (String, String) {
    let left = left_min.map(|m| format!(", about {} left", crate::reshape::auto_progress::short_duration(m))).unwrap_or_default();
    (format!("Auto-cut: stage {n} of {of} healed"), format!("{} % done{left}.", pct.round() as i64))
}

/// The notification for a `heal_round` event of a heal in rounds, only where the person is told: the round that passed, the end of the rounds (cap, plateau, time) and the result of the merged
/// model. (title, body); None for every other decision. Pure text; webhooks never get it (they say the kind and the status only).
pub fn heal_round_toast(ev: &Value) -> Option<(String, String)> {
    let n = ev["round"].as_u64().unwrap_or(0);
    let tokens = ev["tokens_m"].as_f64().unwrap_or(0.0);
    let cap = ev["cap_m"].as_f64().unwrap_or(0.0);
    let m = |x: f64| if x.fract() == 0.0 { format!("{}", x as i64) } else { format!("{x:.1}") };
    match ev["decision"].as_str()? {
        "stop_pass" => Some((format!("Auto-cut: heal round {n} passed the thinking check"), "Merging the model and checking the merged model.".into())),
        "stop_plateau" => Some((format!("Auto-cut: healing stopped after round {n}"), "Neither the thinking nor the loss improves any more. Merging the model and checking it.".into())),
        "stop_cap" | "stop_never" => Some((format!("Auto-cut: healing reached its most tokens ({} M)", m(cap)), "Merging the model and checking the merged model.".into())),
        "stop_time" => Some((format!("Auto-cut: healing stopped after round {n}"), "The time limit leaves no room for another round. Merging the model and checking it.".into())),
        "merged_fail" => Some((format!("Auto-cut: the merged model did not pass after round {n}"), "Healing goes on; it now needs two passed checks in a row.".into())),
        "final" => {
            let verdict = ev["check"]["verdict"].as_str().unwrap_or("fail");
            if verdict == "fail" {
                Some(("Auto-cut: the merged model did not pass its thinking check".into(), format!("Not good enough after {} M tokens.", m(tokens))))
            } else {
                Some((format!("Auto-cut: the merged model passed its thinking check ({verdict})"), format!("After {} M tokens of at most {} M.", m(tokens), m(cap))))
            }
        }
        _ => None,
    }
}

/// "It is done. It took 31 h (estimated 28-40 h)." for a finished Auto-cut, from its saved state; plain "It is done." without the figures.
pub fn auto_done_body(state: &Value) -> String {
    let took = state["progress"]["elapsed_s"].as_f64().filter(|s| *s > 0.0);
    let est = state["plan"]["time"]["total_h"].as_array().and_then(|a| Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?)));
    match (took, est) {
        (Some(t), Some((a, b))) => format!("It is done. It took {} (estimated {}-{} h).", crate::reshape::auto_progress::short_duration(t / 60.0), hours(a), hours(b)),
        (Some(t), None) => format!("It is done. It took {}.", crate::reshape::auto_progress::short_duration(t / 60.0)),
        _ => "It is done.".to_string(),
    }
}

fn hours(h: f64) -> String {
    if h >= 10.0 { format!("{}", h.round() as i64) } else { format!("{}", (h * 10.0).round() / 10.0) }
}

/// (title, body) of the notification for a job that ended.
pub fn toast(kind: &str, r: &Result<Value, String>) -> (String, String) {
    let name = job_name(kind);
    match status_of(r) {
        "finished" if kind == "auto" => (format!("{name} finished"), r.as_ref().map(auto_done_body).unwrap_or_else(|_| "It is done.".into())),
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
