//! Export a saved data version for Unsloth Studio's Train page (Unsloth Desktop): plain JSONL files in the "chatml" format (one `{"messages": [...]}` per line,
//! which Studio reads with Format = chatml or auto), a README with the settings to choose, and nothing else.
//!
//! The rows come from `unsloth::chatml`. The world book (long texts) has no counterpart in Studio's formats, so by default it becomes answers to "Tell me what the lore says about <title>".
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::project::Project;

/// Writes `<dest>/<project>-v<N>-unsloth-studio/` with train.jsonl, eval.jsonl (when the version holds out test questions) and README.md.
/// `version` None = the newest saved one. Returns {"dir", "train", "eval", "world_book", "version"}.
pub fn export(p: &Project, version: Option<u32>, dest: &Path, world_book: bool) -> Result<Value> {
    let versions = crate::dataset::store::versions(p);
    let manifest = match version {
        Some(n) => versions.iter().find(|v| v["version"].as_u64() == Some(n as u64)),
        None => versions.iter().max_by_key(|v| v["version"].as_u64().unwrap_or(0)),
    }.ok_or_else(|| Error::new("Save a data version first (Step 2: \"Save these examples\")."))?.clone();
    let n = manifest["version"].as_u64().unwrap_or(0);
    let safe: String = p.name.chars().map(|c| if c.is_alphanumeric() || "-_".contains(c) { c } else { '_' }).collect();
    let out = dest.join(format!("{safe}-v{n}-unsloth-studio"));
    if out.exists() {
        return Err(Error::new(format!("{} already exists. Choose another folder or remove it.", out.display())));
    }
    std::fs::create_dir_all(&out)?;

    let (train, eval, stats) = crate::unsloth::chatml::chatml_rows(p, n as u32, world_book);
    let (chats, book, longest) = (stats.chats, stats.book, stats.longest);
    if train.is_empty() {
        let _ = std::fs::remove_dir_all(&out);
        return Err(Error::new("This version has no examples Studio can use."));
    }
    std::fs::write(out.join("train.jsonl"), train.join("\n") + "\n")?;
    if !eval.is_empty() {
        std::fs::write(out.join("eval.jsonl"), eval.join("\n") + "\n")?;
    }
    let ctx = (((longest as f64) / 3.5 * 1.15) as usize).next_power_of_two().clamp(2048, 32768);
    let readme = format!("# {world}: training data for Unsloth Studio\n\nMade by ND Lore Forge from saved data version {n}. {rows} rows in `train.jsonl` ({chats} questions and answers{book_txt}){eval_txt}.\n\n\
## In Unsloth Studio (Train)\n\
1. **Model**: choose your model (for example Qwen3.5 9B). Method **LoRA** (or QLoRA when memory is short).\n\
2. **Dataset → Local**: drop `train.jsonl` on the page. **Format**: `chatml` (every line is `{{\"messages\": [...]}}`; `auto` finds it too).\n\
3. **Hyperparameters**: *Context Length* at least **{ctx}** (the longest example is about {tok} tokens). Switch **Train on Completions** ON: the model should learn the answers, not the questions. *Packing* OFF. Rank 16-32, learning rate 2e-4 (lower, 1e-4, for the 9B), 2-3 epochs for a small world.\n\
4. If the model has an image part (Qwen3.5, Gemma 4), train the **language layers only**: leave *Vision Layers* off if Studio shows that checkbox. The vision part then stays exactly as it was and the base model's images keep working.\n\
5. Start training. When it finishes, use **Export** in Studio: *GGUF* for Ollama / llama.cpp, or *Merged model* for 16-bit files.\n\n\
## Notes\n\
- The examples carry the system message of their world, so one model can hold several worlds.\n\
- The text of the world book is {book_note}.\n\
- {eval_note}\n\
- **GM secrets are in this data.** The trained model will repeat them: keep it away from players.\n",
        world = p.display_name(), rows = train.len(), book_txt = if book > 0 { format!(", {book} world-book pieces turned into answers") } else { String::new() },
        eval_txt = if eval.is_empty() { String::new() } else { format!(", {} held-out questions in `eval.jsonl`", eval.len()) },
        tok = (longest as f64 / 3.5) as usize,
        book_note = if book > 0 { "included as answers to \"Tell me what the lore says about <title>\" (Studio has no plain-text mode for this)" } else { "left out (it was switched off for this export)" },
        eval_note = if eval.is_empty() { "This version holds out no test questions, so there is no `eval.jsonl`.".to_string() } else { "`eval.jsonl` holds questions the training never saw. If Studio lets you pick an eval split for a local file, use it; otherwise ignore the file.".to_string() });
    std::fs::write(out.join("README.md"), readme)?;
    Ok(json!({"dir": out, "version": n, "rows": train.len(), "chats": chats, "world_book": book, "eval": eval.len(), "context": ctx}))
}

pub fn default_dest(p: &Project) -> PathBuf {
    p.folder.join("exports")
}
