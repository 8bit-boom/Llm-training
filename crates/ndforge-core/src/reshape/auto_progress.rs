//! Auto-cut's progress arithmetic (docs/AUTO_CUT.md, "How far it is"): how much of the whole run is done, how long it has taken and how long it will probably take.
//! Pure functions of the steps (with the `time_estimate` figures they were planned with) and of the worker's live events; no clock is read, no file touched.
//!
//! The weight of a step is the middle of its time estimate (at least one minute); the parts of a step (`start`, `load`, `cache`, `train`, `merge`, `check` ...) split it.
//! Everything shown is an estimate: before a step has run for a while the remaining time is the plan's, afterwards the live speed's.
use serde_json::Value;

/// A step as the progress sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct StepIn {
    pub id: String,
    /// pending | running | done | skipped | failed | stopped
    pub status: String,
    /// the estimate (low, high) in minutes
    pub est: (f64, f64),
    /// how far a running (or stopped) step is, 0..1
    pub frac: f64,
}

/// What a worker event says about the running step.
#[derive(Debug, Clone, PartialEq)]
pub struct Frac {
    /// how far the whole step is, 0..1
    pub f: f64,
    /// the worker's own figure for the time left of the part that runs, seconds
    pub live_eta_s: Option<f64>,
    /// the line under the bar
    pub sub: Option<String>,
    /// cache | train | merge | check | measure ... (None when the step has no parts)
    pub part: Option<String>,
    /// the estimated minutes of the parts after the running one
    pub after_min: f64,
}

/// The overall state of the run.
#[derive(Debug, Clone, PartialEq)]
pub struct Overall {
    pub pct: f64,
    pub done_min: f64,
    pub total_min: f64,
}

/// The weight of a step: the middle of its estimate, at least a minute.
pub fn weight(est: (f64, f64)) -> f64 {
    ((est.0 + est.1) / 2.0).max(1.0)
}

/// Done steps count at their weight, a running one by its fraction, a skipped one not at all (it leaves the numerator and the denominator).
pub fn overall(steps: &[StepIn]) -> Overall {
    let (mut done, mut total) = (0.0, 0.0);
    for s in steps {
        if s.status == "skipped" {
            continue;
        }
        let w = weight(s.est);
        total += w;
        done += match s.status.as_str() {
            "done" => w,
            "running" | "failed" | "stopped" => w * s.frac.clamp(0.0, 1.0),
            _ => 0.0,
        };
    }
    let pct = if total > 0.0 { (done / total * 1000.0).round() / 10.0 } else { 0.0 };
    Overall { pct, done_min: done, total_min: total }
}

// ------------------------------------------------------------------------------------------------------------------ a heal in rounds

use super::auto_time::RoundsTime;

/// Where a heal in rounds is (from its `heal_round` events and the round tag of the worker's events).
#[derive(Debug, Clone, PartialEq)]
pub struct RoundsLive {
    /// the round in work (1-based)
    pub n: u32,
    /// the most rounds (the cap)
    pub max: u32,
    /// the tokens trained so far, and the cap, in M
    pub tokens_m: f64,
    pub cap_m: f64,
    /// a stop decision was made: only the merge and the check of the merged model are left
    pub stopped: bool,
    /// the check of the last checked round: `{verdict, main}`
    pub last_check: Option<Value>,
}

impl RoundsLive {
    pub fn new(max: u32, cap_m: f64) -> Self {
        RoundsLive { n: 1, max: max.max(1), tokens_m: 0.0, cap_m, stopped: false, last_check: None }
    }

    /// Takes a `heal_round` event: the decision after round `round`.
    pub fn on_round_event(&mut self, ev: &Value) {
        let round = ev["round"].as_u64().unwrap_or(0) as u32;
        self.max = ev["rounds_max"].as_u64().map(|x| x as u32).filter(|x| *x > 0).unwrap_or(self.max);
        self.cap_m = ev["cap_m"].as_f64().filter(|x| *x > 0.0).unwrap_or(self.cap_m);
        if let Some(t) = ev["tokens_m"].as_f64() {
            self.tokens_m = t;
        }
        if ev["check"]["main"].is_object() && ev["decision"] != "final" && ev["decision"] != "merged_fail" {
            self.last_check = Some(ev["check"].clone());
        }
        match ev["decision"].as_str().unwrap_or("") {
            "continue" => {
                self.n = round + 1;
                self.stopped = false;
            }
            "merged_fail" => {
                self.n = round + 1;
                self.stopped = false;
            }
            d if d.starts_with("stop_") || d == "final" => {
                self.n = round.max(1) + 1;
                self.stopped = true;
            }
            _ => {}
        }
    }
}

/// How far a heal in rounds is: the rounds are 95 % of its work (equal shares), the merge and the check of the merged model the rest; inside a round the original's answers are
/// 30 %, the training 50 %, the check 15 % and the loss on the text and the checkpoint 5 %. `part` is cache | train | check | merge, `pf` how far that part is.
pub fn rounds_fraction(live: &RoundsLive, part: &str, pf: f64) -> f64 {
    let pf = pf.clamp(0.0, 1.0);
    if live.stopped {
        return match part {
            "merge" => 0.95 + 0.025 * pf,
            "check" => 0.975 + 0.025 * pf,
            _ => 0.95,
        };
    }
    let inside = match part {
        "cache" => 0.30 * pf,
        "train" => 0.30 + 0.50 * pf,
        "check" => 0.80 + 0.15 * pf,
        _ => 0.95,
    };
    (0.95 * ((live.n.saturating_sub(1)) as f64 + inside) / live.max.max(1) as f64).clamp(0.0, 0.95)
}

/// The time left of a heal in rounds, (low, high) minutes: low = the check of this or the first-check round passes, high = every round to the cap, then the merge and the merged check.
pub fn rounds_left(rt: &RoundsTime, live: &RoundsLive, part: &str, pf: f64) -> (f64, f64) {
    let pf = pf.clamp(0.0, 1.0);
    if live.stopped {
        let done = match part {
            "merge" => 0.4 * pf,
            "check" => 0.4 + 0.6 * pf,
            _ => 0.0,
        };
        return (rt.tail.0 * (1.0 - done), rt.tail.1 * (1.0 - done));
    }
    let inside = match part {
        "cache" => 0.30 * pf,
        "train" => 0.30 + 0.50 * pf,
        "check" => 0.80 + 0.15 * pf,
        _ => 0.0,
    };
    let done_hi = (live.n.saturating_sub(1)) as f64 + inside;
    let hi = ((live.max as f64 - done_hi).max(0.0)) * rt.per_round.1 + rt.tail.1;
    // the check that can end it is that of round `rounds.0` (the first check) or of this round, whichever is later
    let end_at = rt.rounds.0.max(live.n) as f64;
    let lo = ((end_at - done_hi).max(0.0)) * rt.per_round.0 + rt.tail.0;
    (lo.min(hi), hi)
}

/// "Round 3 of up to 8 · 15 M of 40 M tokens · last check: closed 69 %, loops 13 %" (`last check: not yet` before the first one). `base` is the line of the part that runs.
pub fn rounds_sub(live: &RoundsLive, base: Option<&str>) -> String {
    let m = |x: f64| if x.fract() == 0.0 { format!("{}", x as i64) } else { format!("{x:.1}") };
    let chk = match &live.last_check {
        Some(c) if c["main"].is_object() => {
            let p = |k: &str| c["main"][k].as_f64().map(|x| format!("{:.0} %", x * 100.0)).unwrap_or_else(|| "-".into());
            format!("{} (closed {}, loops {})", c["verdict"].as_str().unwrap_or("-"), p("closed_rate"), p("loop_rate"))
        }
        _ => "not yet".to_string(),
    };
    let head = if live.stopped { format!("After round {} · {} M of {} M tokens · last check: {chk}", live.n.saturating_sub(1).max(1), m(live.tokens_m), m(live.cap_m)) }
               else { format!("Round {} of up to {} · {} M of {} M tokens · last check: {chk}", live.n, live.max, m(live.tokens_m), m(live.cap_m)) };
    match base {
        Some(b) => format!("{head} · {b}"),
        None => head,
    }
}

/// "s1.heal" -> Some(1)
pub fn stage_of(id: &str) -> Option<u32> {
    let rest = id.strip_prefix('s')?;
    let (n, _) = rest.split_once('.')?;
    n.parse().ok()
}

/// "s1.heal" -> "heal"; "export" -> "export"
pub fn kind_of(id: &str) -> &str {
    if stage_of(id).is_some() { id.split_once('.').map(|x| x.1).unwrap_or(id) } else { id }
}

fn part_label(part: &str) -> &'static str {
    match part {
        "cache" => "Teacher cache",
        "train" => "Training",
        "merge" => "Saving the healed model",
        "check" => "Thinking check",
        "measure" => "Measuring",
        "load" => "Loading",
        "write" => "Writing the cut",
        "convert" => "Converting",
        "quantize" => "Shrinking",
        "copy" => "Copying",
        _ => "Working",
    }
}

/// The part of a step that a worker event belongs to, and how far that part is.
fn event_part(step_id: &str, ev: &Value) -> Option<(&'static str, f64)> {
    let kind = ev["kind"].as_str().or_else(|| ev["type"].as_str())?;
    let step = ev["step"].as_str().unwrap_or("");
    match kind {
        "metric" => {
            let (s, t) = (ev["step"].as_f64()?, ev["total_steps"].as_f64()?);
            (t > 0.0).then(|| ("train", (s / t).clamp(0.0, 1.0)))
        }
        "try_pass" => {
            let (d, t) = (ev["done"].as_f64()?, ev["total"].as_f64()?);
            (t > 0.0).then(|| ("measure", (d / t).clamp(0.0, 1.0)))
        }
        "progress" => {
            let v = ev["value"].as_f64()?.clamp(0.0, 1.0);
            let heal = kind_of(step_id) == "heal";
            let part = match step {
                "cache" => "cache",
                "train" => "train",
                "merge" if heal => "merge",
                "merge" => "write",
                "think" => "check",
                "score" | "try" => "measure",
                "gguf" => if kind_of(step_id) == "export" { "quantize" } else { "convert" },
                "copy" => "copy",
                _ => return None,
            };
            Some((part, v))
        }
        _ => None,
    }
}

/// How far a step is, from the event the worker just sent. `parts` are the estimated minutes of the step's parts in the order they run. Without the part in `parts`
/// the event's own fraction stands for the whole step.
pub fn step_fraction(step_id: &str, ev: &Value, parts: &[(String, f64)]) -> Option<Frac> {
    let (part, pf) = event_part(step_id, ev)?;
    let total: f64 = parts.iter().map(|p| p.1).sum();
    let at = parts.iter().position(|p| p.0 == part);
    let (f, after) = match at {
        Some(i) if total > 0.0 => {
            let before: f64 = parts[..i].iter().map(|p| p.1).sum();
            let after: f64 = parts[i + 1..].iter().map(|p| p.1).sum();
            ((before + parts[i].1 * pf) / total, after)
        }
        _ => (pf, 0.0),
    };
    let detail = &ev["detail"];
    let live_eta_s = detail["eta_s"].as_f64().or_else(|| ev["eta_s"].as_f64()).filter(|x| *x >= 0.0);
    let sub = sub_status(part, ev, live_eta_s);
    Some(Frac { f: f.clamp(0.0, 1.0), live_eta_s, sub, part: at.map(|_| part.to_string()), after_min: after })
}

fn num(x: f64, digits: usize) -> String {
    let s = format!("{x:.digits$}");
    let (int, frac) = s.split_once('.').map(|(a, b)| (a.to_string(), Some(b.to_string()))).unwrap_or((s.clone(), None));
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 && c != '-' {
            out.push(',');
        }
        out.push(c);
    }
    if let Some(f) = frac {
        out.push('.');
        out.push_str(&f);
    }
    out
}

/// "41 min", "2 h 5 min"
pub fn short_duration(minutes: f64) -> String {
    let m = minutes.max(0.0).round() as u64;
    if m < 60 { format!("{m} min") } else if m % 60 == 0 { format!("{} h", m / 60) } else { format!("{} h {} min", m / 60, m % 60) }
}

/// The line under the bar: "Teacher cache: 12.4 M of 30.0 M tokens · 3,100 tokens/s · 41 min left".
pub fn sub_status(part: &str, ev: &Value, eta_s: Option<f64>) -> Option<String> {
    let d = &ev["detail"];
    let mut bits: Vec<String> = Vec::new();
    if let (Some(done), Some(total), Some(unit)) = (d["done"].as_f64(), d["total"].as_f64(), d["unit"].as_str()) {
        bits.push(if unit == "tokens" {
            format!("{} M of {} M tokens", num(done / 1e6, 1), num(total / 1e6, 1))
        } else if unit == "GB" {
            format!("{} of {} GB", num(done, 1), num(total, 1))
        } else {
            format!("{} of {} {unit}", num(done, 0), num(total, 0))
        });
        if let Some(rate) = d["rate"].as_f64() {
            let ru = d["rate_unit"].as_str().unwrap_or("");
            bits.push(if ru.is_empty() { format!("{}/s", num(rate, 1)) } else if rate >= 100.0 { format!("{} {ru}", num(rate, 0)) } else { format!("{} {ru}", num(rate, 1)) });
        }
    } else if let (Some(s), Some(t)) = (ev["step"].as_f64(), ev["total_steps"].as_f64()) {
        bits.push(format!("step {} of {}", num(s, 0), num(t, 0)));
        if let Some(r) = ev["tok_s"].as_f64() {
            bits.push(format!("{} tokens/s", num(r, 0)));
        }
    } else {
        return None;
    }
    if let Some(e) = eta_s {
        bits.push(format!("{} left", short_duration(e / 60.0)));
    }
    Some(format!("{}: {}", part_label(part), bits.join(" · ")))
}

/// Whether the remaining time is the live speed's or still the plan's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EtaState {
    Estimating,
    Live,
}

const WINDOW_S: f64 = 60.0;
const ALPHA: f64 = 0.3;
const MIN_RUN_S: f64 = 120.0;
const MIN_WINDOWS: u32 = 3;
const LIVE_FRAC: f64 = 0.05;
/// the displayed remaining time may not rise faster than this per minute
const MAX_RISE_PER_MIN: f64 = 0.2;

/// Turns the speed of a running step into a steady remaining time: windows of at least a minute, an exponential average, a state that says "estimating" at first and a
/// display that never jumps up by more than 20 % a minute.
#[derive(Debug, Clone, Default)]
pub struct EtaSmoother {
    start_s: Option<f64>,
    last_t: f64,
    last_f: f64,
    /// fraction per second
    rate: Option<f64>,
    eta: Option<f64>,
    windows: u32,
    shown: Option<(f64, f64)>,
}

impl EtaSmoother {
    /// A new step began (or the run came back after a stop).
    pub fn reset(&mut self) {
        *self = EtaSmoother::default();
    }

    /// One reading: `now_s` seconds on any steady clock, `f` how far the step is, `worker_eta_s` the worker's own figure when it sends one.
    pub fn update(&mut self, now_s: f64, f: f64, worker_eta_s: Option<f64>) {
        if self.start_s.is_none() {
            self.start_s = Some(now_s);
            self.last_t = now_s;
            self.last_f = f;
            return;
        }
        let dt = now_s - self.last_t;
        if dt < WINDOW_S {
            return;
        }
        let df = f - self.last_f;
        if df > 0.0 {
            let r = df / dt;
            self.rate = Some(match self.rate { Some(old) => old + ALPHA * (r - old), None => r });
            if let Some(e) = worker_eta_s {
                self.eta = Some(match self.eta { Some(old) => old + ALPHA * (e - old), None => e });
            }
            self.windows += 1;
        }
        self.last_t = now_s;
        self.last_f = f;
    }

    pub fn state(&self, now_s: f64, f: f64) -> EtaState {
        let ran = self.start_s.map(|s| now_s - s).unwrap_or(0.0);
        if (ran >= MIN_RUN_S && self.windows >= MIN_WINDOWS) || f >= LIVE_FRAC { EtaState::Live } else { EtaState::Estimating }
    }

    /// Whether the worker's own figure for the time left is in use (it counts the running part only, not the parts after it).
    pub fn has_worker_eta(&self) -> bool {
        self.eta.is_some()
    }

    /// The seconds left in the step: the smoothed worker figure when there is one, else from the smoothed rate, else the plan's (`planned_s` times what is left).
    pub fn remaining_s(&mut self, now_s: f64, f: f64, planned_s: f64) -> f64 {
        let raw = if self.state(now_s, f) == EtaState::Live {
            match (self.eta, self.rate) {
                (Some(e), _) => (e - (now_s - self.last_t).max(0.0)).max(0.0),
                (None, Some(r)) if r > 0.0 => (1.0 - f).max(0.0) / r,
                _ => planned_s * (1.0 - f).max(0.0),
            }
        } else {
            planned_s * (1.0 - f).max(0.0)
        };
        let shown = match self.shown {
            Some((t0, s0)) => {
                let counted_down = (s0 - (now_s - t0)).max(0.0);
                let cap = counted_down * (1.0 + MAX_RISE_PER_MIN * ((now_s - t0) / 60.0).max(0.0)) + 1.0;
                raw.min(cap)
            }
            None => raw,
        };
        self.shown = Some((now_s, shown));
        shown
    }
}

/// How much slower (or faster) this run is than the plan, from the steps of the same kind that finished: actual / estimated, between 0.5 and 3; None until one has finished.
/// `done` holds (step id, estimated minutes (middle), actual minutes).
pub fn calibration(done: &[(String, f64, f64)], id: &str) -> Option<f64> {
    let k = kind_of(id);
    let (mut est, mut act) = (0.0, 0.0);
    for (i, e, a) in done {
        if kind_of(i) == k && *e > 0.0 && *a > 0.0 {
            est += e;
            act += a;
        }
    }
    (est > 0.0).then(|| (act / est).clamp(0.5, 3.0))
}

/// The time left, (low, high) in minutes: the running step's live part (x 0.9 to 1.25) plus the steps still to come at their estimates, scaled by the calibration of
/// their kind when this run has one.
pub fn remaining(live_s: Option<f64>, after_min: f64, pending: &[(String, (f64, f64))], done: &[(String, f64, f64)]) -> (f64, f64) {
    let live = live_s.map(|s| s / 60.0).unwrap_or(0.0);
    let (mut lo, mut hi) = (live * 0.9 + after_min * 0.9, live * 1.25 + after_min * 1.25);
    for (id, est) in pending {
        let c = calibration(done, id);
        let (a, b) = match c { Some(c) => (est.0 * c, est.1 * c), None => *est };
        lo += a;
        hi += b;
    }
    (lo, hi)
}

/// The window title and the taskbar text: "Auto-cut 42 % · stage 1 of 2: heal · about 14 h left".
pub fn title_text(pct: f64, stage: Option<(u32, u32)>, step_label: &str, left_min: Option<f64>) -> String {
    let mut t = format!("Auto-cut {} %", pct.round() as i64);
    match stage {
        Some((n, of)) if of > 1 => t.push_str(&format!(" · stage {n} of {of}: {step_label}")),
        _ => t.push_str(&format!(" · {step_label}")),
    }
    if let Some(m) = left_min {
        t.push_str(&format!(" · about {} left", short_duration(m)));
    }
    t
}

/// The window title for an `auto_progress` snapshot: "Auto-cut 42 % · stage 1 of 2: heal · about 14 h left" (none while the time left is still being estimated).
pub fn title_of(snap: &Value) -> String {
    let stage = snap["stage"]["n"].as_u64().zip(snap["stage"]["of"].as_u64()).map(|(n, of)| (n as u32, of as u32));
    let label = snap["step"]["label"].as_str().unwrap_or("working");
    let label = label.split_once(": ").map(|x| x.1).unwrap_or(label);
    let left = (snap["state"] == "live").then(|| snap["left_s"][0].as_f64().zip(snap["left_s"][1].as_f64()).map(|(a, b)| (a + b) / 120.0)).flatten();
    title_text(snap["pct"].as_f64().unwrap_or(0.0), stage, label, left)
}

// ------------------------------------------------------------------------------------------------------------------ the tracker

use serde_json::json;

use super::auto::AutoState;

#[derive(Debug, Clone)]
struct TStep {
    id: String,
    label: String,
    status: String,
    est: Option<(f64, f64)>,
    parts: Vec<(String, f64)>,
    actual_min: Option<f64>,
    /// a heal in rounds: what the plan says of its rounds (its weight is the worst case, the time left a range)
    rounds: Option<RoundsTime>,
}

/// Keeps what the running step reported last and makes the `auto_progress` event of the window from it. `sync` at every change of a step, `on_event` for every live worker event
/// (it answers at most once a second). Clocks are passed in (seconds on any steady scale), so nothing here reads the time.
#[derive(Debug, Clone, Default)]
pub struct ProgressTracker {
    run: String,
    base_elapsed_s: f64,
    start_s: Option<f64>,
    steps: Vec<TStep>,
    of_stages: u32,
    current: Option<usize>,
    frac: Option<Frac>,
    smoother: EtaSmoother,
    last_emit_s: Option<f64>,
    retry_added_min: Option<f64>,
    /// where the heal in rounds that runs is; None for every other step
    live: Option<RoundsLive>,
    /// the time left of that heal, (low, high) minutes
    rounds_left: Option<(f64, f64)>,
}

fn parse_stamp(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|d| d.timestamp())
}

impl ProgressTracker {
    pub fn new() -> Self {
        ProgressTracker::default()
    }

    /// The run's steps changed (a step began, ended, was skipped or reopened): reads them from the state and returns the snapshot.
    pub fn sync(&mut self, st: &AutoState, now_s: f64) -> Value {
        if self.start_s.is_none() {
            self.start_s = Some(now_s);
            self.base_elapsed_s = st.progress.as_ref().and_then(|p| p["elapsed_s"].as_f64()).unwrap_or(0.0);
        }
        self.run = st.id.clone();
        self.of_stages = st.stages.len().max(st.plan.as_ref().map(|p| p.stages.len()).unwrap_or(0)) as u32;
        let old: Vec<(String, String)> = self.steps.iter().map(|s| (s.id.clone(), s.status.clone())).collect();
        self.steps = st.steps.iter().map(|r| TStep {
            id: r.id.clone(), label: r.label.clone(), status: r.status.clone(), est: r.est, parts: r.est_parts.clone(),
            rounds: st.plan.as_ref().and_then(|p| p.time.as_ref()).and_then(|t| t.steps.iter().find(|x| x.id == r.id)).and_then(|x| x.rounds.clone()),
            actual_min: (r.status == "done").then(|| match (r.started.as_deref().and_then(parse_stamp), r.finished.as_deref().and_then(parse_stamp)) {
                (Some(a), Some(b)) if b > a => Some((b - a) as f64 / 60.0),
                _ => None,
            }).flatten(),
        }).collect();
        // a step that was done and is pending again: a retry; its weight comes back into the total
        let reopened: f64 = self.steps.iter().filter(|s| s.status == "pending" && old.iter().any(|o| o.0 == s.id && o.1 == "done")).map(|s| weight(s.est.unwrap_or((1.0, 1.0)))).sum();
        if reopened > 0.0 {
            self.retry_added_min = Some(self.retry_added_min.unwrap_or(0.0) + reopened);
        }
        let cur = self.steps.iter().position(|s| s.status == "running");
        if cur != self.current || cur.is_none() {
            self.frac = None;
            self.smoother.reset();
            self.live = None;
        }
        self.current = cur;
        self.last_emit_s = Some(now_s);
        self.snapshot(now_s)
    }

    /// A live worker event: answers a snapshot at most once a second.
    pub fn on_event(&mut self, ev: &Value, now_s: f64) -> Option<Value> {
        let cur = self.current?;
        if let Some(rt) = self.steps[cur].rounds.clone() {
            return self.on_rounds_event(&rt, ev, now_s);
        }
        let s = &self.steps[cur];
        let fr = step_fraction(&s.id, ev, &s.parts)?;
        self.smoother.update(now_s, fr.f, fr.live_eta_s);
        self.frac = Some(fr);
        if self.last_emit_s.map(|t| now_s - t >= 1.0).unwrap_or(true) {
            self.last_emit_s = Some(now_s);
            Some(self.snapshot(now_s))
        } else {
            None
        }
    }

    /// An event of the heal in rounds that runs: a `heal_round` decision, or a worker event (tagged with its round) of a part of a round.
    fn on_rounds_event(&mut self, rt: &RoundsTime, ev: &Value, now_s: f64) -> Option<Value> {
        let kind = ev["kind"].as_str().or_else(|| ev["type"].as_str()).unwrap_or("");
        let live = self.live.get_or_insert_with(|| RoundsLive::new(rt.rounds.1, rt.cap_m as f64));
        let (part, pf, base) = if kind == "heal_round" {
            live.on_round_event(ev);
            (if live.stopped { "merge" } else { "cache" }, 0.0, None)
        } else {
            let id = self.steps[self.current?].id.clone();
            let (part, pf) = event_part(&id, ev)?;
            if let Some(n) = ev["round"].as_u64() {
                if !live.stopped {
                    live.n = live.n.max(n as u32);
                }
            }
            let live_eta = ev["detail"]["eta_s"].as_f64().or_else(|| ev["eta_s"].as_f64()).filter(|x| *x >= 0.0);
            (part, pf, sub_status(part, ev, live_eta))
        };
        let live = self.live.clone()?;
        let f = rounds_fraction(&live, part, pf);
        let left = rounds_left(rt, &live, part, pf);
        self.frac = Some(Frac { f, live_eta_s: Some(left.1 * 60.0), sub: Some(rounds_sub(&live, base.as_deref())), part: Some(part.to_string()), after_min: 0.0 });
        self.rounds_left = Some(left);
        if self.last_emit_s.map(|t| now_s - t >= 1.0).unwrap_or(true) || kind == "heal_round" {
            self.last_emit_s = Some(now_s);
            Some(self.snapshot(now_s))
        } else {
            None
        }
    }

    fn snapshot(&mut self, now_s: f64) -> Value {
        let f = self.frac.as_ref().map(|x| x.f).unwrap_or(0.0);
        // the weight of a heal in rounds is its worst case: the bar never passes 100 %, and an early stop makes it jump forward
        let est_of = |s: &TStep| match &s.rounds {
            Some(rt) => (rt.worst.1, rt.worst.1),
            None => s.est.unwrap_or((1.0, 1.0)),
        };
        let ins: Vec<StepIn> = self.steps.iter().enumerate().map(|(i, s)| StepIn {
            id: s.id.clone(), status: s.status.clone(), est: est_of(s), frac: if Some(i) == self.current { f } else { 0.0 },
        }).collect();
        let ov = overall(&ins);
        let has_est = self.steps.iter().filter(|s| s.status != "skipped").all(|s| s.est.is_some());
        let elapsed = self.base_elapsed_s + self.start_s.map(|s| (now_s - s).max(0.0)).unwrap_or(0.0);
        // a heal in rounds ends when a check passes: its actual time says nothing about the next heal's estimate
        let done_list: Vec<(String, f64, f64)> = self.steps.iter().filter(|s| s.rounds.is_none()).filter_map(|s| Some((s.id.clone(), weight(s.est?), s.actual_min?))).collect();
        let mut state = "plan";
        let mut left: Option<(f64, f64)> = None;
        if has_est {
            let mut pending: Vec<(String, (f64, f64))> = self.steps.iter().enumerate().filter(|(i, s)| s.status == "pending" && Some(*i) != self.current).map(|(_, s)| (s.id.clone(), s.est.unwrap_or((1.0, 1.0)))).collect();
            let (mut live_s, mut after) = (None, 0.0);
            let mut rounds_part: Option<(f64, f64)> = None;
            if let Some(cur) = self.current {
                let c = self.steps[cur].clone();
                let est = c.est.unwrap_or((1.0, 1.0));
                let live = self.frac.clone().filter(|_| self.smoother.state(now_s, f) == EtaState::Live);
                if c.rounds.is_some() && self.live.is_some() {
                    // a heal in rounds: the time left is a range, from the plan's rounds and where the heal is
                    state = "live";
                    rounds_part = self.rounds_left;
                } else if let Some(fr) = live {
                    state = "live";
                    let planned = weight(est) * 60.0;
                    live_s = Some(self.smoother.remaining_s(now_s, f, planned));
                    if self.smoother.has_worker_eta() {
                        after = fr.after_min;
                    }
                } else {
                    if self.frac.is_some() {
                        state = "estimating";
                    }
                    pending.push((c.id, (est.0 * (1.0 - f), est.1 * (1.0 - f))));
                }
            }
            left = Some(remaining(live_s, after, &pending, &done_list));
            if let (Some(r), Some(l)) = (rounds_part, left) {
                left = Some((l.0 + r.0, l.1 + r.1));
            }
        }
        let cal = self.current.filter(|i| self.steps[*i].rounds.is_none()).and_then(|i| calibration(&done_list, &self.steps[i].id));
        let (stage, step) = match self.current {
            Some(i) => {
                let s = &self.steps[i];
                let counted: Vec<&TStep> = self.steps.iter().filter(|x| x.status != "skipped").collect();
                let index = counted.iter().position(|x| x.id == s.id).map(|x| x + 1).unwrap_or(i + 1);
                (
                    stage_of(&s.id).map(|n| json!({"n": n, "of": self.of_stages.max(n)})),
                    json!({"id": s.id, "label": s.label, "index": index, "count": counted.len(), "pct": (f * 1000.0).round() / 10.0,
                           "part": self.frac.as_ref().and_then(|x| x.part.clone()), "sub": self.frac.as_ref().and_then(|x| x.sub.clone())}),
                )
            }
            None => (None, Value::Null),
        };
        let mut step = step;
        if let (Some(i), Some(l)) = (self.current, &self.live) {
            if self.steps[i].rounds.is_some() {
                step["rounds"] = json!({"n": l.n, "max": l.max, "tokens_m": l.tokens_m, "cap_m": l.cap_m, "stopped": l.stopped, "last_check": l.last_check.as_ref().map(|c| json!({"verdict": c["verdict"], "main": c["main"]}))});
            }
        }
        json!({
            "kind": "auto_progress", "phase": "auto", "run": self.run, "pct": ov.pct, "state": state, "elapsed_s": elapsed.round(),
            "left_s": left.map(|(a, b)| json!([(a * 60.0).round(), (b * 60.0).round()])).unwrap_or(Value::Null),
            "stage": stage.unwrap_or(Value::Null), "step": step,
            "steps": self.steps.iter().map(|s| json!({"id": s.id, "status": s.status, "weight_min": s.est.map(|e| (weight(match &s.rounds { Some(rt) => (rt.worst.1, rt.worst.1), None => e }) * 10.0).round() / 10.0), "actual_min": s.actual_min.map(|a| (a * 10.0).round() / 10.0)})).collect::<Vec<_>>(),
            "retry_added_min": self.retry_added_min.map(|x| (x * 10.0).round() / 10.0), "calibration": cal.map(|c| (c * 100.0).round() / 100.0),
        })
    }
}
