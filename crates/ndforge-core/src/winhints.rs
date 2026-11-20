//! Windows hints for the speed of a heal (docs/HEAL_SPEED.md): hardware-accelerated GPU scheduling, the power plan, and whether a display hangs on the training card. They are SHOWN, never
//! changed. The readers are `Runner`s (tested with fakes); on another system the list is empty.
use serde_json::Value;

/// A program's answer: `Some(output)` when it ran and succeeded.
pub type Ask<'a> = &'a dyn Fn(&[&str]) -> Option<String>;

/// HAGS from `reg query HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers /v HwSchMode`: 2 = on, 1 = off.
pub fn hags_hint(out: Option<&str>) -> String {
    match out.and_then(|t| t.lines().find(|l| l.contains("HwSchMode"))).and_then(|l| l.split_whitespace().last().map(str::to_lowercase)) {
        Some(v) if v == "0x2" || v == "2" => "Hardware-accelerated GPU scheduling: on (helps launch-bound work: a hypothesis of the research report, not measured here).".into(),
        Some(v) if v == "0x1" || v == "1" => "Hardware-accelerated GPU scheduling: off. Windows Settings, System, Display, Graphics, Change default graphics settings can switch it on (a restart is needed); the research report expects 0-10 % for launch-bound work, unmeasured.".into(),
        _ => "Hardware-accelerated GPU scheduling: could not be read (unknown).".into(),
    }
}

/// The active power plan from `powercfg /getactivescheme`.
pub fn power_hint(out: Option<&str>) -> String {
    match out {
        Some(t) if t.to_lowercase().contains("high performance") || t.to_lowercase().contains("ultimate") || t.to_lowercase().contains("maximum") => format!("Power plan: {} (fine for a long heal).", t.split('(').nth(1).and_then(|x| x.split(')').next()).unwrap_or("high performance")),
        Some(t) => format!("Power plan: {}. A heal runs for hours: \"High performance\" (and \"Prefer maximum performance\" for the graphics card in the NVIDIA control panel) avoids clock drops; this program never changes it.", t.split('(').nth(1).and_then(|x| x.split(')').next()).unwrap_or("unknown")),
        None => "Power plan: could not be read (unknown).".into(),
    }
}

/// Is a display attached to the training card? `nvidia-smi --query-gpu=display_active,display_mode --format=csv,noheader` ("Enabled" / "Disabled").
pub fn display_hint(out: Option<&str>) -> String {
    match out.map(|t| t.to_lowercase()) {
        Some(t) if t.contains("enabled") => "A display is attached to the training card: the desktop uses some of its memory and time. If the computer has another graphics output (the processor's), plugging the monitor there frees the card (not measured).".into(),
        Some(t) if t.contains("disabled") => "No display is attached to the training card (good for a long heal).".into(),
        _ => "Whether a display is attached to the training card could not be read (unknown).".into(),
    }
}

/// The three hints, in plain sentences (empty off Windows; `windows` is a parameter so that the tests run everywhere).
pub fn hints_with(windows: bool, ask: Ask) -> Vec<String> {
    if !windows {
        return Vec::new();
    }
    vec![
        hags_hint(ask(&["reg", "query", r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers", "/v", "HwSchMode"]).as_deref()),
        power_hint(ask(&["powercfg", "/getactivescheme"]).as_deref()),
        display_hint(ask(&["nvidia-smi", "--query-gpu=display_active", "--format=csv,noheader"]).as_deref()),
    ]
}

/// The hints of this computer.
pub fn hints() -> Vec<String> {
    hints_with(cfg!(windows), &|args: &[&str]| crate::vllm::real_runner(args).filter(|r| r.0).map(|r| r.1))
}

/// One line for the Heal card (the first hint that says something is wrong, else None).
pub fn card_line(hints: &[String]) -> Option<String> {
    hints.iter().find(|h| h.contains(": off") || h.contains("A display is attached") || h.contains("never changes")).cloned()
}

pub fn as_value(h: &[String]) -> Value {
    serde_json::json!(h)
}
