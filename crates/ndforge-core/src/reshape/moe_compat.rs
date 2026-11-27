//! Is the software on this computer new enough for Inkling-2-Mini, and has the model's repository changed since this program was tested against it? Pure functions: the
//! caller reads the installed versions (the worker's `hello` answer, the llama.cpp install record, the vLLM settings); only `fetch_repo_state` talks to the network.
//!
//! The numbers (2026-11-26, docs/MOE_GUIDE.md "What changed upstream"): transformers 5.24.0 is the first release with the `inkling2_loop` model; llama.cpp build b12611 fixes the
//! per-expert NVFP4 scales of loop passes 2 and 3 (older builds give garbage after the first pass); vLLM 0.35 or a nightly image loads the model (0.34 does not). Nothing
//! here changes what Qwen or Gemma need: a finding exists only for the family `inkling2`.
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Error, Result};

pub const MIN_TRANSFORMERS: &str = "5.24.0";
pub const MIN_LLAMA_BUILD: u32 = 12611;
pub const MIN_VLLM: &str = "0.35.0";
/// The Hugging Face revision of thinkingmachines/Inkling-2-Mini this program was written against (2026-11-24, "Add evaluation results").
pub const TESTED_SHA: &str = "c9d028302cd0f035fdf8e7e91f089a4ce94d2195";
pub const REPO: &str = "thinkingmachines/Inkling-2-Mini";
/// the research checkpoint that was trained for 1 or 2 loop passes
pub const VARLOOP_REPO: &str = "thinkingmachines/TML-d24-varloop-research";

/// What is installed here; None = not known (not installed, or not asked yet).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Have {
    #[serde(default)]
    pub transformers: Option<String>,
    /// the llama.cpp release tag, "b12649"
    #[serde(default)]
    pub llama_tag: Option<String>,
    /// the vLLM docker image or version the settings use ("vllm/vllm-openai:v0.34.0", "vllm/vllm-openai:nightly", "0.35.1")
    #[serde(default)]
    pub vllm: Option<String>,
}

/// One line of the check: "ok" / "too_old" / "unknown" (cannot be told: not installed or not asked), with plain words and what to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub tool: String,
    pub need: String,
    pub have: Option<String>,
    pub status: String,
    pub message: String,
    /// the action in plain words ("" when ok)
    pub fix: String,
    /// a command to run by hand where there is one (the program runs nothing for you here), and how to go back
    pub command: Option<String>,
    pub rollback: Option<String>,
    /// what stops working when this is missing: "all" (the model cannot be used at all) or a part ("export", "teacher")
    pub blocks: String,
}

/// "5.17.0" -> [5, 17, 0]; a suffix like "rc1" or ".dev0" stops the numbers.
pub fn parse_version(v: &str) -> Vec<u32> {
    let v = v.trim().trim_start_matches(['v', 'V']);
    let mut out = Vec::new();
    for part in v.split('.') {
        let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
        match digits.parse::<u32>() {
            Ok(n) => out.push(n),
            Err(_) => break,
        }
        if digits.len() != part.len() {
            break;
        }
    }
    out
}

/// have >= need, numbers compared left to right, missing numbers count as 0. A development build of the needed release ("5.24.0.dev0") counts as older than the release itself.
pub fn version_at_least(have: &str, need: &str) -> bool {
    let (h, n) = (parse_version(have), parse_version(need));
    let len = h.len().max(n.len());
    for i in 0..len {
        let (a, b) = (h.get(i).copied().unwrap_or(0), n.get(i).copied().unwrap_or(0));
        if a != b {
            return a > b;
        }
    }
    !(have.to_ascii_lowercase().contains("dev") && h == n)
}

/// "b12649" -> 12649
pub fn llama_build(tag: &str) -> Option<u32> {
    tag.trim().strip_prefix('b').and_then(|n| n.parse().ok())
}

/// Does this vLLM setting load Inkling? Some(true) for a nightly image or version 0.35 and newer, Some(false) for an older pinned one, None when it cannot be read.
pub fn vllm_loads_inkling(setting: &str) -> Option<bool> {
    let s = setting.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    let tag = s.rsplit(':').next().unwrap_or(&s);
    if tag.contains("nightly") || tag.contains("main") {
        return Some(true);
    }
    let v = parse_version(tag);
    if v.len() < 2 {
        return None;
    }
    Some(version_at_least(tag, MIN_VLLM))
}

fn finding(tool: &str, need: &str, have: Option<&str>, ok: Option<bool>, too_old: String, unknown: String, fix: &str, command: Option<&str>, rollback: Option<&str>, blocks: &str) -> Finding {
    let (status, message) = match ok {
        Some(true) => ("ok", format!("{tool} {} is new enough (needs {need}).", have.unwrap_or(""))),
        Some(false) => ("too_old", too_old),
        None => ("unknown", unknown),
    };
    Finding { tool: tool.into(), need: need.into(), have: have.map(String::from), status: status.into(), message, fix: if ok == Some(true) { String::new() } else { fix.into() },
              command: if ok == Some(true) { None } else { command.map(String::from) }, rollback: if ok == Some(true) { None } else { rollback.map(String::from) }, blocks: blocks.into() }
}

/// The checks for a family; an empty list for every family that needs nothing special.
pub fn findings(family: &str, have: &Have) -> Vec<Finding> {
    if family != "inkling2" {
        return Vec::new();
    }
    let mut out = Vec::new();
    let t = have.transformers.as_deref();
    out.push(finding("transformers", &format!("{MIN_TRANSFORMERS} or newer"), t, t.map(|v| version_at_least(v, MIN_TRANSFORMERS)),
        format!("The training software (transformers {}) is too old for Inkling-2-Mini: the model type was added in transformers {MIN_TRANSFORMERS}. Cutting experts, Try first, healing and training need it; cutting loop passes or experts per token and exporting do not.", t.unwrap_or("?")),
        "The training software is not installed or its version could not be read. Run Training setup, then check again.".into(),
        "Upgrade the training software: it only adds the new model type, Qwen and Gemma keep working. Run the command in the program's Python environment.",
        Some(&format!("uv pip install \"transformers>={MIN_TRANSFORMERS},<6\"")), t.map(|v| format!("uv pip install \"transformers=={v}\"")).as_deref(), "measure, try, heal, train"));
    let l = have.llama_tag.as_deref();
    out.push(finding("llama.cpp", &format!("build b{MIN_LLAMA_BUILD} or newer"), l, l.and_then(llama_build).map(|b| b >= MIN_LLAMA_BUILD),
        format!("llama.cpp {} is older than b{MIN_LLAMA_BUILD}. Older builds read the packed experts wrongly after the first loop pass (the model's perplexity is about 130 instead of 7.65), so an export would be garbage.", l.unwrap_or("?")),
        "llama.cpp is not installed here, or its build number could not be read. Install it from the Export step, then check again.".into(),
        "Install the newest llama.cpp (Settings, llama.cpp, Install the newest).", None, None, "export"));
    let v = have.vllm.as_deref();
    out.push(finding("vLLM", &format!("{MIN_VLLM} or a nightly image"), v, v.and_then(vllm_loads_inkling),
        format!("vLLM {} cannot load Inkling-2-Mini: it needs 0.35 or newer, or the nightly image. The program's own pin is 0.34, so the teacher for healing falls back to the slower in-process packed teacher.", v.unwrap_or("?")),
        "vLLM is not set up (it is optional). Healing then uses the in-process packed teacher, which is slower.".into(),
        "Choose the nightly image in Settings, vLLM (this only changes the Inkling teacher; other models keep the pinned image).", Some("vllm/vllm-openai:nightly"), None, "teacher"));
    out
}

/// "ok" when nothing blocks the whole model, else a one-line summary of what is old, for a banner.
pub fn summary(fs: &[Finding]) -> Option<String> {
    let old: Vec<&str> = fs.iter().filter(|f| f.status == "too_old").map(|f| f.tool.as_str()).collect();
    if old.is_empty() {
        None
    } else {
        Some(format!("Upgrade needed for Inkling-2-Mini: {}.", old.join(", ")))
    }
}

/// What a look at the model's repository found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoState {
    pub repo: String,
    pub tested_sha: String,
    pub latest_sha: Option<String>,
    pub changed: bool,
    pub last_modified: Option<String>,
    /// files whose names suggest the model itself changed (weights, config, chat template)
    pub watched_files: Vec<String>,
    pub message: String,
}

/// Compares Hugging Face's answer for the repository with the revision this program was tested on.
pub fn repo_state(info: &Value) -> RepoState {
    let latest = info["sha"].as_str().map(String::from);
    let changed = latest.as_deref().is_some_and(|s| s != TESTED_SHA);
    let files: Vec<String> = info["siblings"].as_array().map(|a| a.iter().filter_map(|f| f["rfilename"].as_str()).filter(|n| n.ends_with(".safetensors") || ["config.json", "chat_template.jinja", "hf_quant_config.json", "tokenizer.json"].contains(n)).map(String::from).collect()).unwrap_or_default();
    let message = match (&latest, changed) {
        (None, _) => "The repository's revision could not be read.".to_string(),
        (Some(_), false) => format!("{REPO} is unchanged since the version this program was tested with ({}).", &TESTED_SHA[..7]),
        (Some(s), true) => format!("{REPO} has new changes ({} instead of {}). The cut code was tested against the older files: look at the changes (config.json, weights, chat template) before cutting, and use Try first to see whether the model still behaves.", &s[..s.len().min(7)], &TESTED_SHA[..7]),
    };
    RepoState { repo: REPO.into(), tested_sha: TESTED_SHA.into(), latest_sha: latest, changed, last_modified: info["lastModified"].as_str().map(String::from), watched_files: files, message }
}

fn base() -> String {
    std::env::var("NDFORGE_HF_URL").unwrap_or_else(|_| "https://huggingface.co".into())
}

/// Asks Hugging Face for the repository's current revision (a few kilobytes; no weights).
pub fn fetch_repo_state() -> Result<RepoState> {
    let url = format!("{}/api/models/{REPO}", base());
    let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_secs(10)).timeout_read(Duration::from_secs(30)).build();
    match agent.get(&url).set("User-Agent", "nd-lore-forge").call() {
        Ok(r) => Ok(repo_state(&r.into_json::<Value>().map_err(|e| Error::new(format!("Hugging Face's answer was not readable: {e}")))?)),
        Err(ureq::Error::Status(code, _)) => Err(Error::new(format!("Hugging Face answered with an error ({code}) for {REPO}. Try again in a moment."))),
        Err(ureq::Error::Transport(t)) => Err(Error::new(format!("Hugging Face could not be reached. Check the internet connection. [{t}]"))),
    }
}

/// The whole answer of the window command `moe_compat`.
pub fn report(family: &str, have: &Have, repo: Option<&RepoState>) -> Value {
    let fs = findings(family, have);
    json!({"family": family, "findings": fs, "summary": summary(&fs), "repo": repo, "tested_sha": TESTED_SHA})
}
