//! Training in Unsloth Desktop (or a Studio that is running): Lore Forge sends the saved data version to Unsloth's OpenAI-shaped fine-tuning API and follows
//! the job. See `docs/UNSLOTH_DESKTOP.md` for what Desktop is and what is verified, and `docs/UNSLOTH_API_CONTRACT.md` for every call.
pub mod chatml;
pub mod client;
pub mod run;
