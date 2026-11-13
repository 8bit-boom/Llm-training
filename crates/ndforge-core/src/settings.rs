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
    pub ollama_servers: Vec<OllamaServer>,
    pub train_servers: Vec<TrainServer>,
    /// Unsloth Desktop as a place to train (one connection)
    pub unsloth: Option<UnslothSettings>,
    pub teacher: TeacherSettings,
    pub recent_projects: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { expert_mode: false, show_help_panel: true, welcome_seen: false, theme: "auto".to_string(), keep_awake: true,
            ollama_servers: vec![OllamaServer { name: "This computer".into(), url: "http://localhost:11434".into() }], train_servers: Vec::new(), unsloth: None, teacher: TeacherSettings::default(), recent_projects: Vec::new() }
    }
}

impl Settings {
    fn file() -> PathBuf {
        config_dir().join("settings.json")
    }

    pub fn load() -> Settings {
        std::fs::read_to_string(Self::file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self) {
        let f = Self::file();
        if let Some(dir) = f.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(f, text);
        }
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
