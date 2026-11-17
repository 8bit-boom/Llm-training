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
    /// when a project is opened, go on by itself with an Auto-cut or a healing that was cut off by closing the program (default off)
    pub auto_continue_interrupted: bool,
    pub ollama_servers: Vec<OllamaServer>,
    pub train_servers: Vec<TrainServer>,
    /// Unsloth Desktop as a place to train (one connection)
    pub unsloth: Option<UnslothSettings>,
    /// Unsloth Studio's share, where Auto-cut puts its GGUF files
    pub studio_share: Option<StudioShare>,
    pub teacher: TeacherSettings,
    pub recent_projects: Vec<String>,
    /// Set (for this run of the program only, never saved) when `settings.json` could not be read: where the damaged file was kept. The window shows it once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_note: Option<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { expert_mode: false, show_help_panel: true, welcome_seen: false, theme: "auto".to_string(), keep_awake: true, auto_continue_interrupted: false,
            ollama_servers: vec![OllamaServer { name: "This computer".into(), url: "http://localhost:11434".into() }], train_servers: Vec::new(), unsloth: None, studio_share: None, teacher: TeacherSettings::default(), recent_projects: Vec::new(), load_note: None }
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
