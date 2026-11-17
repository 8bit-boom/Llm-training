//! Where the time of the jobs went (docs/AUTO_CUT.md, "Where the time went"): reads the `timing: {...}` lines the worker writes (`protocol.timing`) out of the saved job logs
//! (`logs/jobs/*.log`), adds them up per kind of job and stage, and says in plain words what is worth changing. Nothing here runs a job.
//!
//! A log line looks like `[    12.3s out] {"type": "log", "text": "timing: {\"stage\": \"write\", \"seconds\": 4.2, ...}"}`; the first line of a log names the job
//! (`# job: {...}`). The same records are in the `timings` key of a job's result.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// One timed stage of one job.
#[derive(Debug, Clone, PartialEq)]
pub struct Timing {
    /// the kind of job (`export`, `distill_merge`, ...); "job" when the log does not say
    pub job: String,
    pub stage: String,
    pub seconds: f64,
    /// the whole record (stage, seconds, load, bytes, tokens, kernels ...)
    pub fields: Value,
}

/// What the program suggests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Advice {
    pub id: String,
    /// true: the numbers say this is worth doing; false: the numbers say it is not (shown so the answer is not a silence)
    pub worth: bool,
    pub text: String,
}

/// A job kind from the first lines of a log (`# job: {...}`).
fn kind_of(text: &str) -> String {
    text.lines().take(4).find_map(|l| l.strip_prefix("# job: ")).and_then(|j| serde_json::from_str::<Value>(j).ok()).and_then(|v| v["job"].as_str().map(String::from)).unwrap_or_else(|| "job".into())
}

/// The model a job is about (for "the same model exported twice"): the base repo, the model or the source.
fn model_of(text: &str) -> Option<String> {
    let j = text.lines().take(4).find_map(|l| l.strip_prefix("# job: ")).and_then(|j| serde_json::from_str::<Value>(j).ok())?;
    [&j["base"]["repo"], &j["model"], &j["source"]["repo"], &j["full_model"]].iter().find_map(|v| v.as_str().filter(|s| !s.is_empty()).map(String::from))
}

/// The timing records of one job log. Lines that are not timing lines, or whose JSON cannot be read, are ignored.
pub fn parse_log(text: &str) -> Vec<Timing> {
    let job = kind_of(text);
    let mut out = Vec::new();
    for line in text.lines() {
        if !line.contains("timing: {") {
            continue;
        }
        // `[  12.3s out] {"type": "log", "text": "timing: {...}"}`: the worker's JSON line after the bracket
        let Some(json_start) = line.find("] ").map(|i| i + 2).filter(|i| line[*i..].starts_with('{')) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(&line[json_start..]) else { continue };
        let Some(rec) = v["text"].as_str().and_then(|t| t.strip_prefix("timing: ")).and_then(|t| serde_json::from_str::<Value>(t).ok()) else { continue };
        let (Some(stage), Some(seconds)) = (rec["stage"].as_str(), rec["seconds"].as_f64()) else { continue };
        out.push(Timing { job: job.clone(), stage: stage.to_string(), seconds, fields: rec.clone() });
    }
    out
}

/// The tokens-per-second figures of the `cache_speed:` lines of a log, in order (the teacher's answers are logged every minute).
pub fn cache_speeds(text: &str) -> Vec<f64> {
    text.lines().filter_map(|l| {
        let i = l.find("cache_speed: ")? + "cache_speed: ".len();
        let rest = &l[i..];
        rest[..rest.find(" tokens/s")?].replace(',', "").trim().parse::<f64>().ok()
    }).collect()
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = s.len() / 2;
    if s.len() % 2 == 1 { s[m] } else { (s[m - 1] + s[m]) / 2.0 }
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Adds the job logs up: `{"logs": n, "job_seconds", "jobs": {kind: {"count", "seconds", "stages": {stage: {"seconds", "count", "share"}}}}, "stages": {stage: {"seconds", "count", "share"}},
/// "exports": {model: count}, "load_teacher_4bit_seconds", "convert_f16_seconds", "export_seconds", "merge": {"compute", "total"}, "torch_kernels", "cache_speed": [[...]]}`.
/// `share` is the part of the job's own seconds (the `job` record) a stage took, 0 to 1.
pub fn summarize_texts(logs: &[String]) -> Value {
    let mut jobs: BTreeMap<String, (u64, f64, BTreeMap<String, (f64, u64)>)> = BTreeMap::new();
    let mut stages: BTreeMap<String, (f64, u64)> = BTreeMap::new();
    let mut exports: BTreeMap<String, u64> = BTreeMap::new();
    let (mut job_seconds, mut load4, mut convert, mut export_secs, mut mc, mut mt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut torch_kernels, mut speeds) = (false, Vec::new());
    for text in logs {
        let recs = parse_log(text);
        let kind = kind_of(text);
        let own = recs.iter().find(|t| t.stage == "job").map(|t| t.seconds);
        let e = jobs.entry(kind.clone()).or_default();
        e.0 += 1;
        let total_of_stages: f64 = recs.iter().filter(|t| t.stage != "job").map(|t| t.seconds).sum();
        let this_job = own.unwrap_or(total_of_stages);
        e.1 += this_job;
        job_seconds += this_job;
        if kind == "export" {
            export_secs += this_job;
            *exports.entry(model_of(text).unwrap_or_default()).or_default() += 1;
        }
        for t in recs.iter().filter(|t| t.stage != "job") {
            let s = e.2.entry(t.stage.clone()).or_default();
            s.0 += t.seconds;
            s.1 += 1;
            let g = stages.entry(t.stage.clone()).or_default();
            g.0 += t.seconds;
            g.1 += 1;
            match t.stage.as_str() {
                "load_teacher" if t.fields["load"] == json!("4bit") => load4 += t.seconds,
                "convert_f16" => convert += t.seconds,
                "merge_compute" => mc += t.seconds,
                "merge_total" => mt += t.seconds,
                "think_generate" | "load_model" | "load_student" | "load_teacher" if t.fields["kernels"] == json!("torch") => torch_kernels = true,
                _ => {}
            }
        }
        let sp = cache_speeds(text);
        if !sp.is_empty() {
            speeds.push(sp);
        }
    }
    let share = |secs: f64, of: f64| if of > 0.0 { r2((secs / of).min(1.0)) } else { 0.0 };
    let jobs_json: Value = jobs.iter().map(|(k, (n, secs, st))| (k.clone(), json!({
        "count": n, "seconds": r2(*secs),
        "stages": st.iter().map(|(s, (x, c))| (s.clone(), json!({"seconds": r2(*x), "count": c, "share": share(*x, *secs)}))).collect::<serde_json::Map<_, _>>(),
    }))).collect::<serde_json::Map<_, _>>().into();
    let stages_json: Value = stages.iter().map(|(s, (x, c))| (s.clone(), json!({"seconds": r2(*x), "count": c, "share": share(*x, job_seconds)}))).collect::<serde_json::Map<_, _>>().into();
    json!({"logs": logs.len(), "job_seconds": r2(job_seconds), "jobs": jobs_json, "stages": stages_json, "exports": exports,
           "load_teacher_4bit_seconds": r2(load4), "convert_f16_seconds": r2(convert), "export_seconds": r2(export_secs),
           "merge": {"compute": r2(mc), "total": r2(mt)}, "torch_kernels": torch_kernels, "cache_speed": speeds})
}

/// `summarize_texts` of the log files (a file that cannot be read is left out).
pub fn summarize(files: &[PathBuf]) -> Value {
    let texts: Vec<String> = files.iter().filter_map(|f| std::fs::read_to_string(f).ok()).collect();
    summarize_texts(&texts)
}

/// What is worth changing, from a summary. Every rule answers (`worth` true or false) when its numbers are there; a rule without numbers says nothing.
pub fn advice(summary: &Value) -> Vec<Advice> {
    let mut out = Vec::new();
    let a = |id: &str, worth: bool, text: String| Advice { id: id.into(), worth, text };
    let total = summary["job_seconds"].as_f64().unwrap_or(0.0);
    // the 4-bit teacher is loaded (and quantised) again by every job that needs it
    let load4 = summary["load_teacher_4bit_seconds"].as_f64().unwrap_or(0.0);
    if load4 > 0.0 && total > 0.0 {
        let worth = load4 > 0.10 * total;
        out.push(a("teacher_4bit_copy", worth, if worth {
            format!("Loading the original as 4-bit took {:.0} % of the time of these jobs. A saved 4-bit copy of the original, kept for the run, would save most of it.", load4 / total * 100.0)
        } else {
            format!("Loading the original as 4-bit took {:.0} % of the time of these jobs: a saved copy would not pay.", load4 / total * 100.0)
        }));
    }
    // an export converts to a 16-bit GGUF every time
    let convert = summary["convert_f16_seconds"].as_f64().unwrap_or(0.0);
    let export_secs = summary["export_seconds"].as_f64().unwrap_or(0.0);
    if convert > 0.0 && export_secs > 0.0 {
        let twice = summary["exports"].as_object().is_some_and(|m| m.values().any(|n| n.as_u64().unwrap_or(0) >= 2));
        let worth = convert > 0.10 * export_secs && twice;
        out.push(a("reuse_f16_gguf", worth, if worth {
            format!("The 16-bit conversion took {:.0} % of the export time and the same model was exported more than once: keeping the 16-bit file and the merged folder between exports would save that time.", convert / export_secs * 100.0)
        } else if !twice {
            "The 16-bit conversion is repeated only when the same model is exported again; this model was exported once.".to_string()
        } else {
            format!("The 16-bit conversion took only {:.0} % of the export time: reusing it would not pay.", convert / export_secs * 100.0)
        }));
    }
    // the merge: compute or disk
    let (mc, mt) = (summary["merge"]["compute"].as_f64().unwrap_or(0.0), summary["merge"]["total"].as_f64().unwrap_or(0.0));
    if mt > 0.0 {
        let rest = (mt - mc).max(0.0);
        let cpu = mc > 2.0 * rest;
        out.push(a("merge_bound", true, if cpu {
            format!("Saving the healed model is bound by the processor ({mc:.0} s of {mt:.0} s is arithmetic): a faster processor or fewer background programs help more than a faster drive.")
        } else {
            format!("Saving the healed model is bound by the disk ({rest:.0} s of {mt:.0} s is reading and writing): put the models on the fastest drive (an NVMe drive, not a network or USB one).")
        }));
    }
    // the fast kernels
    if summary["torch_kernels"] == json!(true) {
        out.push(a("fast_kernels", true, "The model's linear-attention layers ran on the slow PyTorch kernels: install the fast kernels (Training setup, \"Install the fast kernels\") to make measuring, healing and the thinking check several times faster.".into()));
    } else if summary["stages"]["think_generate"].is_object() || summary["stages"]["load_model"].is_object() {
        out.push(a("fast_kernels", false, "No slow-kernel warning was recorded in these jobs.".into()));
    }
    // the teacher's answers: slow at the end means long rows dominate
    if let Some(series) = summary["cache_speed"].as_array() {
        let mut worth = false;
        let mut seen = false;
        for s in series.iter().filter_map(|s| s.as_array()) {
            let v: Vec<f64> = s.iter().filter_map(|x| x.as_f64()).collect();
            if v.len() < 5 {
                continue;
            }
            seen = true;
            let tail = &v[v.len() - (v.len() / 5).max(1)..];
            if median(tail) < 0.6 * median(&v) {
                worth = true;
            }
        }
        if seen {
            out.push(a("cache_tail", worth, if worth {
                "The teacher's answers got much slower towards the end (the last fifth ran below 60 % of the usual speed): the long rows dominate. A shorter sequence length, or cutting the longest rows, saves the most.".into()
            } else {
                "The teacher's answers kept their speed to the end.".into()
            }));
        }
    }
    out
}

/// The summary and its advice of the newest `n` job logs: what the diagnostics bundle (`timings.json`) and the page show.
pub fn report_of_latest(n: usize) -> Value {
    let files = crate::envmgr::joblog::latest(n);
    let s = summarize(&files);
    let adv = advice(&s);
    json!({"summary": s, "advice": adv})
}

/// The job logs that started between two times (`%Y-%m-%dT%H:%M:%SZ`, UTC), from the stamp in their names (`YYYYmmdd-HHMMSS-<kind>.log`).
pub fn logs_between(dir: &Path, from: &str, to: &str) -> Vec<PathBuf> {
    let parse = |s: &str| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%SZ").ok();
    let (Some(a), Some(b)) = (parse(from), parse(to)) else { return Vec::new() };
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "log")).filter(|p| {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        name.get(..15).and_then(|s| chrono::NaiveDateTime::parse_from_str(s, "%Y%m%d-%H%M%S").ok()).is_some_and(|t| t >= a && t <= b)
    }).collect()).unwrap_or_default();
    v.sort();
    v
}
