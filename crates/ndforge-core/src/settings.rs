//! Small app-wide settings (recent projects, expert mode, help panel). JSON in the config folder; a problem reading or writing it
//! never stops the app.
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::paths::config_dir;

/// An Ollama server the model can be installed into (this PC, the TrueNAS box ...).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OllamaServer {
    pub name: String,
    pub url: String,
}

/// A computer that can train for us (the TrueNAS server). Its access token is kept in `secrets`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainServer {
    pub name: String,
    pub url: String,
}

/// The Unsloth Desktop (or Studio) that can train for us. Its API key is in `secrets` under `unsloth:<url>`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnslothSettings {
    pub name: String,
    pub url: String,
}

/// The folder Unsloth Studio (in Docker, on a NAS) keeps its files in, as this computer sees it, and the same folder as Studio sees it inside its container. Auto-cut puts
/// the GGUF files into `<share>/exports/<model>` (docs/UNSLOTH_API_CONTRACT.md, "Publish").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct StudioShare {
    /// e.g. `Y:\unsloth\studio` or `\\truenas\apps\studio`
    pub host_path: String,
    /// the same folder inside Studio's container, e.g. `/workspace`; empty: found out from Studio's outputs folder when an API key is set
    pub container_path: String,
    /// tell Studio about the exports folder (`POST /api/models/scan-folders`) when it offers that
    pub register_scan_folder: bool,
    /// also copy the healed model (Hugging Face format) next to the GGUF files and give its container path, so it can be fine-tuned in Studio (off: big)
    pub finetune_handoff: bool,
}

impl Default for StudioShare {
    fn default() -> Self {
        StudioShare { host_path: String::new(), container_path: String::new(), register_scan_folder: true, finetune_handoff: false }
    }
}

/// The model that writes extra training examples (any OpenAI-compatible server; Ollama works). Its access key, if any, is in `secrets` under "teacher".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct TeacherSettings {
    pub url: String,
    pub model: String,
    /// the teacher is an Ollama server: take the model out of memory afterwards
    pub unload_after: bool,
}

impl Default for TeacherSettings {
    fn default() -> Self {
        TeacherSettings { url: "http://localhost:11434".into(), model: String::new(), unload_after: true }
    }
}

/// "Make the original's answers with vLLM" (docs/VLLM.md, "Teacher cache"): OFF by default. The teacher must be the person's ORIGINAL model: a pre-quantized checkpoint is a quantization of
/// ONE repo, so it is used only for the original it was made from (`pairs`: original repo id -> quantized repo id) or one the person confirmed (`confirmed`); every other original goes through
/// the in-flight 4-bit route on the original's own folder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct VllmTeacherSettings {
    /// use vLLM for the teacher's answers (the Heal card still has its own switch)
    pub enabled: bool,
    /// "auto" (a confirmed copy, else the in-flight 4-bit route) | "bnb" (always in-flight) | "prequant" (only a confirmed copy; refuses otherwise)
    pub route: String,
    /// the confirmed pairs {original repo id: quantized repo id}; the default table has the one the research report names and nothing else
    pub pairs: std::collections::BTreeMap<String, String>,
    /// copies the person confirmed ("this repo is a quantized copy of my original model"), stored per original
    pub confirmed: std::collections::BTreeMap<String, String>,
    /// true: a heal stops when vLLM does not work; false: it goes on with the standard teacher and says so
    pub required: bool,
    /// tested engine settings (docs/TUNE.md); 0 = the program's own value (32 rows at once, capped by the cache; 32,768 tokens per batch; 0.88 of the card). They are not part of the cache identity.
    #[serde(default)]
    pub max_num_seqs: u32,
    #[serde(default)]
    pub batch_tokens: u32,
    /// the share of the card in percent (0 = 88)
    #[serde(default)]
    pub gpu_util_pct: u32,
}

impl Default for VllmTeacherSettings {
    fn default() -> Self {
        VllmTeacherSettings { enabled: false, route: "auto".into(), pairs: [("Qwen/Qwen3.8-27B".to_string(), "cyankiwi/Qwen3.8-27B-AWQ-INT4".to_string())].into(), confirmed: Default::default(), required: false, max_num_seqs: 0, batch_tokens: 0, gpu_util_pct: 0 }
    }
}

/// The experimental vLLM engine (docs/VLLM.md): OFF by default, built from a research report and not yet run on this PC. Nothing is installed by the program.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct VllmSettings {
    /// offer "vLLM (experimental)" next to the standard engine and llama.cpp in the thinking check
    pub enabled: bool,
    /// "auto" | "docker" (Windows docker.exe / Docker Desktop) | "docker_wsl" (Docker inside WSL) | "wslc" (WSL 3.x containers) | "wsl" (vLLM installed in WSL) | "url" (a server that runs)
    pub kind: String,
    /// the pinned image of the docker kinds
    pub image: String,
    /// the WSL distribution (empty: the default one)
    pub distro: String,
    /// the vLLM environment inside WSL (kind "wsl")
    pub venv: String,
    /// kind "url": the address of a running server
    pub url: String,
    /// the NAMED Docker volume that keeps downloaded models (must start with `ndforge-vllm-`)
    pub model_volume: String,
    /// optional: a folder with a model the engine reads (a folder under /mnt/ is slow)
    pub model_dir: String,
    /// the pre-quantized checkpoint used for the 27 B original
    pub teacher_repo: String,
    /// check the rounds of a heal (model + LoRA) with vLLM; off until an agreement run showed it matches the standard engine
    pub adapter_rounds: bool,
    /// the KV cache type of the engine: "auto" (default, today's behaviour) | "fp8" (8 bit: about twice the cached tokens, a tiny numeric change; experimental)
    pub kv_cache_dtype: String,
    /// the 8-bit KV cache may ALSO be used for the teacher's answers (the training target): a second, explicit confirmation, off by default
    pub kv_cache_dtype_teacher: bool,
    /// llama.cpp's KV cache type for the GGUF thinking check: "f16" (default) | "q8_0" (needs flash attention)
    pub llama_cache_type: String,
    /// edits of the WSLc command templates (docs/VLLM.md); an object, empty = the defaults
    pub wslc: serde_json::Value,
    /// the teacher's answers through vLLM (off by default)
    pub teacher: VllmTeacherSettings,
}

impl Default for VllmSettings {
    fn default() -> Self {
        VllmSettings { enabled: false, kind: "auto".into(), image: "vllm/vllm-openai:v0.34.0".into(), distro: String::new(), venv: "~/.ndforge/vllm".into(), url: String::new(),
            model_volume: "ndforge-vllm-hf".into(), model_dir: String::new(), teacher_repo: "cyankiwi/Qwen3.8-27B-AWQ-INT4".into(), adapter_rounds: false, kv_cache_dtype: "auto".into(), kv_cache_dtype_teacher: false, llama_cache_type: "f16".into(), wslc: serde_json::json!({}), teacher: VllmTeacherSettings::default() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Settings {
    pub expert_mode: bool,
    pub show_help_panel: bool,
    pub welcome_seen: bool,
    /// "auto" (follow the system), "light" or "dark" (night mode)
    pub theme: String,
    /// keep the computer awake while training (default on: a computer that falls asleep stops the training)
    pub keep_awake: bool,
    /// also keep the screen on while a job runs (default off): on a computer with Modern Standby that may be the only thing that stops sleep
    pub keep_display_on: bool,
    /// when a project is opened, go on by itself with an Auto-cut or a healing that was cut off by closing the program (default off)
    pub auto_continue_interrupted: bool,
    /// a desktop notification when a job or the queue ends (default on)
    pub notifications: bool,
    /// an optional ntfy-style address that gets one line when a job ends: the kind of job and how it ended, never paths (default empty = off)
    pub webhook_url: String,
    pub ollama_servers: Vec<OllamaServer>,
    pub train_servers: Vec<TrainServer>,
    /// Unsloth Desktop as a place to train (one connection)
    pub unsloth: Option<UnslothSettings>,
    /// Unsloth Studio's share, where Auto-cut puts its GGUF files
    pub studio_share: Option<StudioShare>,
    pub teacher: TeacherSettings,
    /// the experimental vLLM engine (off by default)
    pub vllm: VllmSettings,
    /// the original's answers (the teacher cache) take as many rows per forward pass as fit this many tokens (rows sorted by length); 0 = the fixed 4 rows of before. The answers are the same
    /// (up to rounding) and the cache's identity does not change.
    pub teacher_batch_tokens: u32,
    /// how many layers of a GGUF llama.cpp puts on the graphics card in the importance matrix and in Compare quants (`-ngl`); None = all of them (99), what these jobs always did. Set by a tested preset (docs/TUNE.md).
    #[serde(default)]
    pub llama_ngl: Option<u32>,
    /// calibration rows through the original together in Try first and the cut's measuring (0 = 4, what it always was); set by a tested preset
    #[serde(default)]
    pub try_calib_batch: u32,
    /// questions asked together in the thinking check (0 = 8, what it always was); set by a tested preset
    #[serde(default)]
    pub think_batch: u32,
    pub recent_projects: Vec<String>,
    /// Set (for this run of the program only, never saved) when `settings.json` could not be read: where the damaged file was kept. The window shows it once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_note: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { expert_mode: false, show_help_panel: true, welcome_seen: false, theme: "auto".to_string(), keep_awake: true, keep_display_on: false, auto_continue_interrupted: false, notifications: true, webhook_url: String::new(),
            ollama_servers: vec![OllamaServer { name: "This computer".into(), url: "http://localhost:11434".into() }], train_servers: Vec::new(), unsloth: None, studio_share: None, teacher: TeacherSettings::default(), vllm: VllmSettings::default(), teacher_batch_tokens: 16384, llama_ngl: None, try_calib_batch: 0, think_batch: 0, recent_projects: Vec::new(), load_note: None }
    }
}

impl Settings {
    fn file() -> PathBuf {
        config_dir().join("settings.json")
    }

    /// A missing file gives the defaults. A file that cannot be read is kept as `settings.json.broken-<time>` (not overwritten), the defaults are used and
    /// `load_note` says what happened.
    pub fn load() -> Settings {
        let f = Self::file();
        let Ok(text) = std::fs::read_to_string(&f) else {
            return match f.is_file() {
                false => Settings::default(),
                true => Self::broken(&f),
            };
        };
        match serde_json::from_str::<Settings>(&text) {
            Ok(mut s) => {
                s.load_note = None;
                s
            }
            Err(_) => Self::broken(&f),
        }
    }

    fn broken(f: &Path) -> Settings {
        let note = match crate::fsx::set_aside_broken(f) {
            Some(kept) => format!("The settings file could not be read, so the standard settings are used. The old file was kept as {}.", kept.display()),
            None => "The settings file could not be read, so the standard settings are used.".to_string(),
        };
        Settings { load_note: Some(note), ..Settings::default() }
    }

    pub fn save(&self) {
        let mut plain = self.clone();
        plain.load_note = None;
        let _ = crate::fsx::write_json_atomic(&Self::file(), &plain);
    }

    pub fn remember_project(&mut self, folder: &Path) {
        let s = folder.to_string_lossy().to_string();
        self.recent_projects.retain(|p| *p != s);
        self.recent_projects.insert(0, s);
        self.recent_projects.truncate(10);
    }

    /// Recent projects that still exist.
    pub fn existing_recent(&self) -> Vec<String> {
        self.recent_projects.iter().filter(|p| Path::new(p).join("project.toml").exists()).cloned().collect()
    }
}
