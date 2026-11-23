//! Putting an exported model into an Ollama server and asking it a question to see that it works.
use std::path::Path;

use serde_json::{json, Value};

use super::modelfile::{create_body, spec_for};
use super::ollama::{file_digest, Ollama};
use crate::error::{Error, Result};

/// Uploads `gguf` (and the mmproj vision file `mmproj`, when the export has one) if the server does not have it yet, creates the model `name`
/// and checks that it is listed. Progress goes to `on_event` as {"kind": "step"|"progress"|"log", ...}.
pub fn install(url: &str, name: &str, gguf: &Path, family: &str, num_ctx: u32, on_event: &mut dyn FnMut(Value), mmproj: Option<&Path>) -> Result<()> {
    let ollama = Ollama::new(url);
    on_event(json!({"kind": "step", "index": 1, "total": 4, "text": format!("Connecting to Ollama at {}", ollama.base())}));
    let version = ollama.version()?;
    on_event(json!({"kind": "log", "text": format!("Ollama {version} is running.")}));
    let file = gguf.file_name().and_then(|n| n.to_str()).ok_or_else(|| Error::new("The GGUF file has no name."))?.to_string();
    on_event(json!({"kind": "step", "index": 2, "total": 4, "text": "Checking the file (a checksum of several GB takes a minute)"}));
    let digest = file_digest(gguf, &mut |v| on_event(json!({"kind": "progress", "value": v})))?;
    let mut projector = None;
    if let Some(m) = mmproj {
        let mfile = m.file_name().and_then(|n| n.to_str()).ok_or_else(|| Error::new("The vision file has no name."))?.to_string();
        let mdigest = file_digest(m, &mut |v| on_event(json!({"kind": "progress", "value": v})))?;
        projector = Some((mfile, mdigest));
        on_event(json!({"kind": "log", "text": "The vision file goes with it, so the model keeps its pictures."}));
    }
    on_event(json!({"kind": "step", "index": 3, "total": 4, "text": "Sending the model to Ollama"}));
    if ollama.blob_exists(&digest)? {
        on_event(json!({"kind": "log", "text": "The server already has this file."}));
    } else {
        ollama.upload_blob(&digest, gguf, &mut |v| on_event(json!({"kind": "progress", "value": v})))?;
    }
    if let Some((_, mdigest)) = &projector {
        if ollama.blob_exists(mdigest)? {
            on_event(json!({"kind": "log", "text": "The server already has the vision file."}));
        } else {
            ollama.upload_blob(mdigest, mmproj.unwrap(), &mut |v| on_event(json!({"kind": "progress", "value": v})))?;
        }
    }
    on_event(json!({"kind": "step", "index": 4, "total": 4, "text": format!("Creating the model \"{name}\"")}));
    ollama.create(&create_body(name, &file, &digest, &spec_for(family, num_ctx), projector.as_ref().map(|(f, d)| (f.as_str(), d.as_str()))))?;
    let have = ollama.models()?;
    if !have.iter().any(|m| m == name || m.strip_suffix(":latest") == Some(name) || format!("{m}") == format!("{name}:latest")) {
        return Err(Error::new(format!("Ollama says it created \"{name}\" but does not list it. Check the Ollama log.")));
    }
    Ok(())
}

/// Asks the installed model one question the way nd-world does (a system message and a user message). Returns the answer.
pub fn probe(url: &str, name: &str, system: &str, question: &str) -> Result<String> {
    Ollama::new(url).chat(name, &[json!({"role": "system", "content": system}), json!({"role": "user", "content": question})], &json!({"temperature": 0, "num_predict": 200}))
}
