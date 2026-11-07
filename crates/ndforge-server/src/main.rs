//! The training worker server. Configuration comes from environment variables:
//!
//! | variable | meaning | default |
//! |---|---|---|
//! | `NDFORGE_TOKEN` | the access token the desktop program must send (at least 16 characters) | required |
//! | `NDFORGE_LISTEN` | address to listen on | `0.0.0.0:8765` |
//! | `NDFORGE_DATA` | folder for jobs, checkpoints and the Hugging Face model cache | `/data` |
//! | `NDFORGE_PYTHON` | the Python of the training environment | `/opt/venv/bin/python` |
//! | `NDFORGE_WORKER_DIR` | the folder with the `ndforge_worker` package | `/app/worker` |
//! | `OLLAMA_URL` | Ollama on this server, unloaded before training to free the card | none |
//! | `NDFORGE_MAX_UPLOAD_MB` | largest accepted upload | `2048` |
//! | `NDFORGE_ALLOW_NO_GPU` | `1` lets it accept jobs without a graphics card (development only) | off |
use std::path::PathBuf;
use std::time::Duration;

use ndforge_core::remote::server::{serve, ServerConfig};

fn env(name: &str, default: &str) -> String {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| default.to_string())
}

fn main() {
    let token = env("NDFORGE_TOKEN", "");
    if token.is_empty() {
        eprintln!("NDFORGE_TOKEN is not set. Choose a long random access token (for example: openssl rand -hex 24) and set it for the container.");
        std::process::exit(2);
    }
    let data_dir = PathBuf::from(env("NDFORGE_DATA", "/data"));
    // models are downloaded into the data folder, so they survive a restart of the container
    if std::env::var_os("HF_HOME").is_none() {
        std::env::set_var("HF_HOME", data_dir.join("hf"));
    }
    let cfg = ServerConfig {
        listen: env("NDFORGE_LISTEN", "0.0.0.0:8765"),
        token,
        data_dir,
        python: PathBuf::from(env("NDFORGE_PYTHON", "/opt/venv/bin/python")),
        ollama_url: Some(env("OLLAMA_URL", "")).filter(|u| !u.is_empty()),
        require_gpu: env("NDFORGE_ALLOW_NO_GPU", "0") != "1",
        max_upload: env("NDFORGE_MAX_UPLOAD_MB", "2048").parse::<u64>().unwrap_or(2048) << 20,
        grace: Duration::from_secs(120),
    };
    if let Err(e) = serve(cfg) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
