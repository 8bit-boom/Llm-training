//! Tier 2: the quality settings, each tried alone from the person's settings in a SHORT REAL HEAL on the real rows (the same seeded rows and the same held-out rows for every setting), judged by the held-out
//! loss. This is NOT a full heal: a setting that wins a short trial can lose a long one, and the table says so. The thinking check stays the real gate for quality.
use serde_json::{json, Map, Value};

use super::decide::{needed_gain, DEFAULT_SPREAD, VRAM_SHARE_LIMIT};

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// "baseline", "baseline2" (another seed: the noise), or "<knob>=<value>"
    pub id: String,
    pub knob: String,
    pub value: Value,
    pub label: String,
}

fn num(card: &Map<String, Value>, k: &str, d: f64) -> f64 {
    card.get(k).and_then(Value::as_f64).unwrap_or(d)
}

/// The settings to try, one knob at a time from the card: the baseline twice (two seeds: the run-to-run noise), the LoRA size, the optimizer, the warm-up and the learning rate.
pub fn variants(card: &Map<String, Value>) -> Vec<Variant> {
    let mut v = vec![Variant { id: "baseline".into(), knob: String::new(), value: Value::Null, label: "Your settings".into() }, Variant { id: "baseline2".into(), knob: String::new(), value: Value::Null, label: "Your settings, another seed (the noise)".into() }];
    let rank = num(card, "rank", 64.0) as u64;
    for r in [16u64, 32, 64, 128].into_iter().filter(|r| *r != rank) {
        v.push(Variant { id: format!("rank={r}"), knob: "rank".into(), value: json!(r), label: format!("LoRA size {r}") });
    }
    let opt = card.get("optimizer").and_then(Value::as_str).unwrap_or("adamw_8bit");
    let other = if opt == "adamw" { "adamw_8bit" } else { "adamw" };
    v.push(Variant { id: format!("optimizer={other}"), knob: "optimizer".into(), value: json!(other), label: format!("Optimizer {}", if other == "adamw" { "AdamW (32-bit, more memory)" } else { "8-bit AdamW" }) });
    let warm = num(card, "warmup", 30.0) as u64;
    for w in [0u64, 10, 30, 100].into_iter().filter(|w| *w != warm) {
        v.push(Variant { id: format!("warmup={w}"), knob: "warmup".into(), value: json!(w), label: format!("Warm-up {w} steps") });
    }
    let lr = num(card, "lr", 1e-4);
    for f in [0.5f64, 2.0] {
        let x = (lr * f * 1e9).round() / 1e9;
        v.push(Variant { id: format!("lr={x}"), knob: "lr".into(), value: json!(x), label: format!("Learning rate {x}") });
    }
    v
}

/// The hyper settings a variant changes in the heal job (rank also sets alpha, as the heal does).
pub fn overrides(v: &Variant, seed: u64) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("seed".into(), json!(seed));
    match v.knob.as_str() {
        "combined" => {
            for (k, x) in v.value.as_object().into_iter().flatten() {
                m.insert(k.clone(), x.clone());
                if k == "rank" {
                    m.insert("alpha".into(), x.clone());
                }
            }
        }
        "rank" => {
            m.insert("rank".into(), v.value.clone());
            m.insert("alpha".into(), v.value.clone());
        }
        "" => {}
        k => {
            m.insert(k.into(), v.value.clone());
        }
    }
    m
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnobDecision {
    pub knob: String,
    pub chosen: Value,
    pub changed: bool,
    pub gain_loss: f64,
    pub why: String,
}

fn after(r: &Value) -> Option<f64> {
    r["held_after"].as_f64()
}

/// For every knob: the value with the lowest held-out loss, when it beats the baseline by more than max(2 x the baseline's seed-to-seed spread, 1 % of the loss) and stays below 90 % of the card. `rows` are the
/// results with their variant `id` and `ok`; the baseline value is the mean of the two baseline runs. Returns (decisions, the noise, the baseline's held-out loss).
pub fn decide_quality(rows: &[Value], card_mb: Option<f64>) -> (Vec<KnobDecision>, f64, f64) {
    let ok = |id: &str| rows.iter().find(|r| r["id"] == id && r["ok"] == true).and_then(after);
    let (Some(a), b) = (ok("baseline"), ok("baseline2")) else { return (vec![], 0.0, 0.0) };
    let base = b.map(|b| (a + b) / 2.0).unwrap_or(a);
    let noise = b.map(|b| (a - b).abs()).unwrap_or(0.0);
    let need = (2.0 * noise).max(0.01 * base).max(if b.is_none() { DEFAULT_SPREAD * base } else { 0.0 });
    let mut knobs: Vec<String> = rows.iter().filter_map(|r| r["knob"].as_str().filter(|k| !k.is_empty() && *k != "combined").map(String::from)).collect();
    knobs.dedup();
    let mut seen = std::collections::BTreeSet::new();
    knobs.retain(|k| seen.insert(k.clone()));
    let mut out = Vec::new();
    for k in knobs {
        let mut best: Option<&Value> = None;
        for r in rows.iter().filter(|r| r["knob"] == k.as_str() && r["ok"] == true) {
            if card_mb.is_some_and(|m| r["peak_vram_mb"].as_f64().unwrap_or(0.0) > VRAM_SHARE_LIMIT * m) {
                continue;
            }
            if after(r).is_some_and(|x| best.and_then(after).is_none_or(|y| x < y)) {
                best = Some(r);
            }
        }
        match best.and_then(|r| after(r).map(|x| (r, base - x))) {
            Some((r, gain)) if gain > need => out.push(KnobDecision { knob: k.clone(), chosen: r["value"].clone(), changed: true, gain_loss: gain, why: format!("{} gives a held-out loss {:.4} lower than yours in the short trial (needed more than {:.4}: twice the noise). A short trial, not a full heal.", r["label"].as_str().unwrap_or(&k), gain, need) }),
            other => out.push(KnobDecision { knob: k.clone(), chosen: Value::Null, changed: false, gain_loss: other.map(|x| x.1).unwrap_or(0.0), why: format!("No value is better than yours by more than the noise (±{:.4}): yours stays.", noise) }),
        }
    }
    let _ = needed_gain;
    confirm_together(&mut out, rows, base, need, card_mb);
    (out, noise, base)
}

/// The knobs were each tried alone; the settings that won are now ALSO tried together (the row `combined`, when the run made it). They are kept together only when the combination beats the baseline by the same
/// margin and gets at least 90 % of the best single gain; else only the best single knob stays and the others say they did not add up. A combination that did not finish or was not tried changes nothing here.
fn confirm_together(out: &mut [KnobDecision], rows: &[Value], base: f64, need: f64, card_mb: Option<f64>) {
    let changed = out.iter().filter(|k| k.changed).count();
    if changed < 2 {
        return;
    }
    let Some(c) = rows.iter().find(|r| r["id"] == "combined" && r["ok"] == true).filter(|r| card_mb.is_none_or(|m| r["peak_vram_mb"].as_f64().unwrap_or(0.0) <= VRAM_SHARE_LIMIT * m)) else {
        let why = rows.iter().find(|r| r["id"] == "combined").map(|r| if r["oom"] == true { "out of memory".to_string() } else { "did not finish".to_string() });
        if let Some(why) = why {
            for k in out.iter_mut().filter(|k| k.changed) {
                k.why += &format!(" Tried together with the other winners, the combination {why}: this was not confirmed.");
            }
        }
        return;
    };
    let gain_c = after(c).map(|x| base - x).unwrap_or(f64::MIN);
    let best = out.iter().filter(|k| k.changed).map(|k| k.gain_loss).fold(f64::MIN, f64::max);
    if gain_c > need && gain_c >= 0.9 * best {
        for k in out.iter_mut().filter(|k| k.changed) {
            k.why += &format!(" Tried together with the other winners: {gain_c:.4} lower than yours, so they are kept together.");
        }
        return;
    }
    let keep = out.iter().filter(|k| k.changed).max_by(|a, b| a.gain_loss.partial_cmp(&b.gain_loss).unwrap_or(std::cmp::Ordering::Equal)).map(|k| k.knob.clone());
    for k in out.iter_mut().filter(|k| k.changed) {
        if Some(&k.knob) == keep.as_ref() {
            k.why += &format!(" Tried together with the others the gain was {gain_c:.4}, not more than this one alone: only this setting is kept.");
        } else {
            k.changed = false;
            k.chosen = Value::Null;
            k.why = format!("{} did not add up: all the winners together gained {gain_c:.4} against {best:.4} for the best one alone, so only that one is kept.", k.knob);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, knob: &str, value: Value, after: f64, peak: f64) -> Value {
        json!({"id": id, "knob": knob, "value": value, "label": id, "ok": true, "held_after": after, "peak_vram_mb": peak})
    }

    #[test]
    fn variants_start_with_two_baselines_and_skip_the_current_value() {
        let card: Map<String, Value> = serde_json::from_value(json!({"rank": 64, "optimizer": "adamw_8bit", "warmup": 30, "lr": 1e-4})).unwrap();
        let v = variants(&card);
        assert_eq!(v[0].id, "baseline");
        assert_eq!(v[1].id, "baseline2");
        assert!(!v.iter().any(|x| x.id == "rank=64" || x.id == "warmup=30"));
        assert!(v.iter().any(|x| x.id == "optimizer=adamw"));
        let o = overrides(v.iter().find(|x| x.id == "rank=32").unwrap(), 7);
        assert_eq!(o["rank"], json!(32));
        assert_eq!(o["alpha"], json!(32));
        assert_eq!(o["seed"], json!(7));
    }

    #[test]
    fn a_win_must_beat_the_noise_and_stay_under_the_memory_limit() {
        let rows = vec![
            row("baseline", "", Value::Null, 2.00, 1000.0),
            row("baseline2", "", Value::Null, 2.02, 1000.0),
            row("rank=32", "rank", json!(32), 1.80, 1000.0),
            row("rank=128", "rank", json!(128), 1.70, 9500.0),
            row("warmup=0", "warmup", json!(0), 1.995, 1000.0),
        ];
        let (d, noise, base) = decide_quality(&rows, Some(10000.0));
        assert!((noise - 0.02).abs() < 1e-9 && (base - 2.01).abs() < 1e-9);
        let rank = d.iter().find(|x| x.knob == "rank").unwrap();
        assert!(rank.changed && rank.chosen == json!(32), "128 is over 90 % of the card");
        let w = d.iter().find(|x| x.knob == "warmup").unwrap();
        assert!(!w.changed, "a gain inside the noise changes nothing");
    }

    fn two_winners() -> Vec<Value> {
        vec![row("baseline", "", Value::Null, 2.00, 1000.0), row("baseline2", "", Value::Null, 2.02, 1000.0), row("rank=32", "rank", json!(32), 1.80, 1000.0), row("warmup=100", "warmup", json!(100), 1.85, 1000.0)]
    }

    #[test]
    fn winners_that_also_win_together_are_kept_together() {
        let mut rows = two_winners();
        rows.push(json!({"id": "combined", "knob": "combined", "value": {"rank": 32, "warmup": 100}, "ok": true, "held_after": 1.72, "peak_vram_mb": 1000.0}));
        let (d, _, _) = decide_quality(&rows, Some(10000.0));
        assert!(d.iter().all(|k| k.changed) && d.len() == 2 && d[0].why.contains("kept together"), "{d:?}");
    }

    #[test]
    fn winners_that_do_not_add_up_leave_only_the_best_one() {
        let mut rows = two_winners();
        rows.push(json!({"id": "combined", "knob": "combined", "value": {"rank": 32, "warmup": 100}, "ok": true, "held_after": 1.95, "peak_vram_mb": 1000.0}));
        let (d, _, _) = decide_quality(&rows, Some(10000.0));
        let rank = d.iter().find(|k| k.knob == "rank").unwrap();
        let warm = d.iter().find(|k| k.knob == "warmup").unwrap();
        assert!(rank.changed && !warm.changed && warm.why.contains("did not add up"), "{d:?}");
        let mut failed = two_winners();
        failed.push(json!({"id": "combined", "knob": "combined", "ok": false, "oom": true}));
        let (d2, _, _) = decide_quality(&failed, Some(10000.0));
        assert!(d2.iter().all(|k| k.changed) && d2[0].why.contains("not confirmed"), "an unconfirmed combination does not remove a winner, and says so: {d2:?}");
    }

    #[test]
    fn the_combined_overrides_set_every_knob_and_the_alpha_with_the_rank() {
        let v = Variant { id: "combined".into(), knob: "combined".into(), value: json!({"rank": 32, "warmup": 100}), label: "all".into() };
        let o = overrides(&v, 1);
        assert_eq!((o["rank"].clone(), o["alpha"].clone(), o["warmup"].clone()), (json!(32), json!(32), json!(100)));
    }

    #[test]
    fn without_a_baseline_nothing_is_decided() {
        assert!(decide_quality(&[row("rank=32", "rank", json!(32), 1.0, 1.0)], None).0.is_empty());
    }
}
