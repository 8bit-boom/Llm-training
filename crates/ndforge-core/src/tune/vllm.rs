//! The vLLM teacher's engine settings (`teacher.vllm`, docs/TUNE.md): how many rows the engine scores at once (`max_num_seqs`), its tokens per batch (`vllm_batch_tokens`) and the share of the card it takes
//! (`gpu_util_pct`). The worker scores the SAME rows once per variant against ONE reference and reports the speed and how much the answers differ from the reference; this module lists the variants and decides.
//! A faster setting that changes the answers, or that the program had to lower for the memory, is refused. Each knob is judged alone from the person's settings (they are not tested together).
use serde_json::{json, Map, Value};

use super::decide::{needed_gain, MIN_GAIN_ENGINE};

/// What the engine uses when a setting is 0 ("the program's own value": 32 rows, 32,768 tokens, 88 % of the card).
pub const DEFAULT_SEQS: u64 = 32;
pub const DEFAULT_BATCH_TOKENS: u64 = 32768;
pub const DEFAULT_UTIL_PCT: u64 = 88;
/// A setting may not move the top-1 agreement with the reference down by more than this (absolute) or the mean KL up by more than this.
pub const AGREEMENT_SLACK: f64 = 0.005;
pub const KL_SLACK: f64 = 0.005;

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub id: String,
    pub knob: String,
    pub value: Value,
    pub label: String,
    /// the engine keys of the worker job this variant changes
    pub engine: Value,
}

fn cur(card: &Map<String, Value>, k: &str, default: u64) -> u64 {
    card.get(k).and_then(Value::as_u64).filter(|v| *v > 0).unwrap_or(default)
}

/// The baseline twice (the noise) and, one knob at a time from the person's settings, two other values of each knob.
pub fn variants(card: &Map<String, Value>) -> Vec<Variant> {
    let mut v = vec![Variant { id: "baseline".into(), knob: String::new(), value: Value::Null, label: "Your settings".into(), engine: json!({}) },
                     Variant { id: "baseline2".into(), knob: String::new(), value: Value::Null, label: "Your settings again (the noise)".into(), engine: json!({}) }];
    let seqs = cur(card, "max_num_seqs", DEFAULT_SEQS);
    for n in [16u64, 32, 64].into_iter().filter(|n| *n != seqs) {
        v.push(Variant { id: format!("max_num_seqs={n}"), knob: "max_num_seqs".into(), value: json!(n), label: format!("{n} rows scored at once"), engine: json!({"max_num_seqs": n}) });
    }
    let bt = cur(card, "vllm_batch_tokens", DEFAULT_BATCH_TOKENS);
    for n in [16384u64, 32768, 65536].into_iter().filter(|n| *n != bt) {
        v.push(Variant { id: format!("vllm_batch_tokens={n}"), knob: "vllm_batch_tokens".into(), value: json!(n), label: format!("{n} tokens per batch"), engine: json!({"batch_tokens": n}) });
    }
    let util = cur(card, "gpu_util_pct", DEFAULT_UTIL_PCT);
    for n in [80u64, 88, 92].into_iter().filter(|n| *n != util) {
        v.push(Variant { id: format!("gpu_util_pct={n}"), knob: "gpu_util_pct".into(), value: json!(n), label: format!("{n} % of the card"), engine: json!({"gpu_memory_utilization": n as f64 / 100.0}) });
    }
    v
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnobDecision {
    pub knob: String,
    pub chosen: Value,
    pub changed: bool,
    pub gain: f64,
    pub why: String,
}

fn tps(r: &Value) -> Option<f64> {
    r["tokens_per_s"].as_f64().filter(|t| *t > 0.0)
}

/// Per knob: the fastest setting whose answers agree with the reference as well as the person's settings do (within the slack) and that the engine got as asked, if it beats the baseline (the mean of the two
/// baseline runs) by `max(1.10, 1 + 2 x noise)`. Returns (decisions, noise as a share, the baseline's tokens/s).
pub fn decide(rows: &[Value]) -> (Vec<KnobDecision>, f64, f64) {
    let ok = |id: &str| rows.iter().find(|r| r["id"] == id && r["ok"] == true && tps(r).is_some());
    let Some(a) = ok("baseline") else { return (vec![], 0.0, 0.0) };
    let (ta, tb) = (tps(a).unwrap_or(0.0), ok("baseline2").and_then(tps));
    let base = tb.map(|b| (ta + b) / 2.0).unwrap_or(ta);
    let noise = tb.map(|b| (ta - b).abs() / base).unwrap_or(0.0);
    let need = needed_gain(noise.max(if tb.is_none() { super::decide::DEFAULT_SPREAD } else { 0.0 }), MIN_GAIN_ENGINE);
    let (top1, kl) = (a["top1"].as_f64(), a["kl"].as_f64());
    let mut knobs: Vec<String> = Vec::new();
    for r in rows {
        if let Some(k) = r["knob"].as_str().filter(|k| !k.is_empty()) {
            if !knobs.iter().any(|x| x == k) {
                knobs.push(k.to_string());
            }
        }
    }
    let mut out = Vec::new();
    for k in knobs {
        let mut excluded: Vec<String> = Vec::new();
        let mut best: Option<&Value> = None;
        for r in rows.iter().filter(|r| r["knob"] == k.as_str()) {
            let name = r["label"].as_str().unwrap_or(&k);
            if r["ok"] != true || tps(r).is_none() {
                excluded.push(format!("{name}: {}", if r["oom"] == true { "out of memory".to_string() } else { r["error"].as_str().map(|e| e.chars().take(80).collect::<String>()).unwrap_or_else(|| "did not finish".into()) }));
                continue;
            }
            if r["gpu_lowered"] == true {
                excluded.push(format!("{name}: the engine had to take less of the card than asked (other programs use it)"));
                continue;
            }
            if let (Some(t0), Some(t1)) = (top1, r["top1"].as_f64()) {
                if t1 < t0 - AGREEMENT_SLACK {
                    excluded.push(format!("{name}: faster but its answers agree less with the reference ({:.1} % against {:.1} %)", t1 * 100.0, t0 * 100.0));
                    continue;
                }
            }
            if let (Some(k0), Some(k1)) = (kl, r["kl"].as_f64()) {
                if k1 > k0 + KL_SLACK {
                    excluded.push(format!("{name}: faster but its answers drift from the reference (KL {k1:.4} against {k0:.4})"));
                    continue;
                }
            }
            if best.is_none_or(|b| tps(r) > tps(b)) {
                best = Some(r);
            }
        }
        let tail = if excluded.is_empty() { String::new() } else { format!(" Excluded: {}.", excluded.join("; ")) };
        match best {
            Some(b) if tps(b).unwrap_or(0.0) / base >= need => {
                let gain = (tps(b).unwrap_or(0.0) / base * 100.0).round() / 100.0;
                out.push(KnobDecision { knob: k.clone(), chosen: b["value"].clone(), changed: true, gain, why: format!("{} scores {gain} times as fast as yours (needed {need:.2}: more than twice the noise of {:.1} %), with the same answers.{tail}", b["label"].as_str().unwrap_or(&k), noise * 100.0) });
            }
            other => out.push(KnobDecision { knob: k.clone(), chosen: Value::Null, changed: false, gain: other.and_then(tps).map(|t| (t / base * 100.0).round() / 100.0).unwrap_or(1.0).max(1.0),
                                             why: format!("No value is clearly faster than yours (needed {need:.2}, noise {:.1} %): yours stays.{tail}", noise * 100.0) }),
        }
    }
    (out, noise, base)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, knob: &str, value: Value, t: f64, top1: f64, kl: f64) -> Value {
        json!({"id": id, "knob": knob, "value": value, "label": id, "ok": true, "tokens_per_s": t, "top1": top1, "kl": kl, "gpu_lowered": false})
    }

    #[test]
    fn variants_are_the_baseline_twice_and_two_other_values_per_knob() {
        let card: Map<String, Value> = serde_json::from_value(json!({"max_num_seqs": 0, "vllm_batch_tokens": 65536, "gpu_util_pct": 92})).unwrap();
        let v = variants(&card);
        let ids: Vec<&str> = v.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(ids, vec!["baseline", "baseline2", "max_num_seqs=16", "max_num_seqs=64", "vllm_batch_tokens=16384", "vllm_batch_tokens=32768", "gpu_util_pct=80", "gpu_util_pct=88"], "0 means the program's 32 / 32,768 / 88, and the current value is not tried again");
        assert_eq!(v[7].engine, json!({"gpu_memory_utilization": 0.88}));
        assert_eq!(v[3].engine, json!({"max_num_seqs": 64}));
    }

    #[test]
    fn a_win_must_beat_ten_percent_and_twice_the_noise_with_the_same_answers() {
        let rows = vec![row("baseline", "", Value::Null, 3000.0, 0.99, 0.004), row("baseline2", "", Value::Null, 3060.0, 0.99, 0.004),
                        row("max_num_seqs=16", "max_num_seqs", json!(16), 3500.0, 0.99, 0.004), row("max_num_seqs=64", "max_num_seqs", json!(64), 4200.0, 0.95, 0.004),
                        row("vllm_batch_tokens=16384", "vllm_batch_tokens", json!(16384), 3150.0, 0.99, 0.004),
                        row("gpu_util_pct=92", "gpu_util_pct", json!(92), 4000.0, 0.99, 0.004)];
        let (d, noise, base) = decide(&rows);
        assert!((base - 3030.0).abs() < 1e-9 && (noise - 60.0 / 3030.0).abs() < 1e-9);
        let seqs = d.iter().find(|x| x.knob == "max_num_seqs").unwrap();
        assert!(seqs.changed && seqs.chosen == json!(16), "64 is fastest but agrees less with the reference: {seqs:?}");
        assert!(seqs.why.contains("agree less with the reference"));
        assert!(!d.iter().find(|x| x.knob == "vllm_batch_tokens").unwrap().changed, "4 % is inside the 10 % minimum");
        assert!(d.iter().find(|x| x.knob == "gpu_util_pct").unwrap().changed);
    }

    #[test]
    fn a_lowered_share_a_failed_row_and_a_missing_baseline_decide_nothing() {
        let mut low = row("gpu_util_pct=92", "gpu_util_pct", json!(92), 5000.0, 0.99, 0.004);
        low["gpu_lowered"] = json!(true);
        let failed = json!({"id": "max_num_seqs=64", "knob": "max_num_seqs", "value": 64, "label": "64 rows at once", "ok": false, "oom": true, "error": "out of memory"});
        let rows = vec![row("baseline", "", Value::Null, 3000.0, 0.99, 0.004), low, failed];
        let (d, noise, _) = decide(&rows);
        assert_eq!(noise, 0.0);
        assert!(d.iter().all(|x| !x.changed));
        assert!(d.iter().find(|x| x.knob == "gpu_util_pct").unwrap().why.contains("less of the card than asked"));
        assert!(d.iter().find(|x| x.knob == "max_num_seqs").unwrap().why.contains("out of memory"));
        assert!(decide(&[row("max_num_seqs=16", "max_num_seqs", json!(16), 1.0, 1.0, 0.0)]).0.is_empty());
    }
}
