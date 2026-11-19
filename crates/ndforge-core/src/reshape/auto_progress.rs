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
