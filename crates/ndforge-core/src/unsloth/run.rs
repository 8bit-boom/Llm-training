//! A training run in Unsloth: send the data, follow the job, bring the adapter back (see `docs/UNSLOTH_DESKTOP.md`).

/// The Unsloth the user entered. The key is kept in `secrets` under `unsloth:<url>`.
#[derive(Debug, Clone)]
pub struct UnslothTarget {
    pub name: String,
    pub url: String,
    pub key: String,
}

/// The `server` value the Train page uses for Unsloth (a training server is named by its address).
pub const SERVER_ID: &str = "unsloth";
