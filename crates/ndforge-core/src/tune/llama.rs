//! llama.cpp's offload (`llama.offload`, docs/TUNE.md): how many layers of a GGUF go to the graphics card in the importance matrix and in Compare quants (`-ngl`). Those jobs always asked for every layer (99); a 16-bit
//! GGUF of a big model does not fit a card and llama.cpp then stops with an allocation error. The worker runs a short pass from the most layers to the fewest and stops at the first that runs. This module lists the
//! numbers to try and decides.
use serde_json::{json, Value};

/// "Every layer" (what the jobs ask for when the setting is empty).
pub const ALL: u32 = 99;

/// The numbers of layers to try for a model of `layers` layers: all (99), then 3/4, 1/2, 1/4 and 1/8 of them, then none; most first, without repeats.
pub fn candidates(layers: u32) -> Vec<u32> {
    let mut v = vec![ALL];
    if layers >= 4 {
        for f in [0.75f64, 0.5, 0.25, 0.125] {
            let n = ((layers as f64) * f).round() as u32;
            if n >= 1 && n < layers && !v.contains(&n) {
                v.push(n);
            }
        }
    }
    v.push(0);
    v
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub chosen: Option<u32>,
    pub changed: bool,
    pub tokens_per_s: f64,
    pub why: String,
}

/// The first row that ran is the answer (more layers on the card are never slower). When every layer fits nothing changes; when the default (all) did not run, the number that did becomes the setting. No row
/// that ran: nothing is decided and the reason of the last one is said.
pub fn decide(rows: &[Value]) -> Decision {
    let Some(ok) = rows.iter().find(|r| r["ok"] == true) else {
        let last = rows.last().map(|r| r["error"].as_str().unwrap_or("it did not run").to_string()).unwrap_or_else(|| "nothing was tried".into());
        return Decision { chosen: None, changed: false, tokens_per_s: 0.0, why: format!("llama.cpp did not run at any number of layers on the card ({last}): nothing was changed.") };
    };
    let ngl = ok["ngl"].as_u64().unwrap_or(0) as u32;
    let tps = ok["tokens_per_s"].as_f64().unwrap_or(0.0);
    if ngl >= ALL {
        return Decision { chosen: Some(ngl), changed: false, tokens_per_s: tps, why: format!("Every layer fits on the card ({tps:.0} tokens/s): the default stays.") };
    }
    let tried: Vec<String> = rows.iter().filter(|r| r["ok"] != true).map(|r| format!("{} layers: {}", r["ngl"], if r["oom"] == true { "out of memory".to_string() } else { r["error"].as_str().unwrap_or("did not run").chars().take(60).collect() })).collect();
    Decision { chosen: Some(ngl), changed: true, tokens_per_s: tps, why: format!("With every layer on the card llama.cpp did not run ({}). {ngl} layers run at {tps:.0} tokens/s: that becomes the setting for the importance matrix and Compare quants.", tried.join("; ")) }
}

pub fn decision_json(d: &Decision) -> Value {
    json!({"chosen": d.chosen, "changed": d.changed, "tokens_per_s": d.tokens_per_s, "why": d.why})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_candidates_go_from_all_to_none_without_repeats() {
        assert_eq!(candidates(64), vec![99, 48, 32, 16, 8, 0]);
        assert_eq!(candidates(8), vec![99, 6, 4, 2, 1, 0]);
        assert_eq!(candidates(2), vec![99, 0]);
    }

    #[test]
    fn the_first_number_that_ran_is_the_setting_and_the_default_stays_when_all_fit() {
        let all = vec![json!({"ngl": 99, "ok": true, "tokens_per_s": 900.0})];
        let d = decide(&all);
        assert_eq!((d.chosen, d.changed), (Some(99), false));
        let rows = vec![json!({"ngl": 99, "ok": false, "oom": true, "error": "x"}), json!({"ngl": 48, "ok": false, "oom": true}), json!({"ngl": 32, "ok": true, "tokens_per_s": 210.0})];
        let d = decide(&rows);
        assert_eq!((d.chosen, d.changed), (Some(32), true));
        assert!(d.why.contains("99 layers: out of memory") && d.why.contains("32 layers run at 210 tokens/s"), "{}", d.why);
        let none = decide(&[json!({"ngl": 99, "ok": false, "error": "bad file"}), json!({"ngl": 0, "ok": false, "error": "bad file"})]);
        assert_eq!(none.chosen, None);
        assert!(none.why.contains("bad file"));
    }
}
