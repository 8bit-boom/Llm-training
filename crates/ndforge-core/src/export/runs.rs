//! Export runs: `exports/<run>-<quant>/` holds model-<quant>.gguf, Modelfile, MODEL_CARD.md, meta.json (what it was made from and where it was
//! installed) and export.json (the worker's summary).
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::modelfile::{model_card, modelfile_text, ollama_name, spec_for, CardInfo};
use crate::envmgr::jobs::{run_job, WorkerEvent};
use crate::envmgr::{env_dir, python_path, worker_dir};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::train::runs::runs_dir;

pub const QUANTS: [(&str, &str); 5] = [
    ("q4_k_m", "Q4_K_M: smallest, good quality (recommended)"),
    ("q5_k_m", "Q5_K_M: a little larger, a little better"),
    ("q6_k", "Q6_K: larger again, very close to Q8"),
    ("q8_0", "Q8_0: large, very close to the original"),
    ("f16", "F16: the original precision (very large)"),
];

pub fn exports_dir(p: &Project) -> PathBuf {
    p.folder.join("exports")
}

fn read_json(path: &std::path::Path) -> Option<Value> {
    std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn write_json(path: &std::path::Path, v: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}

/// Runs that can be exported: they have a saved adapter. Newest first.
pub fn exportable_runs(p: &Project) -> Vec<Value> {
    crate::train::runs::list_runs(p).into_iter().filter(|r| r["has_adapter"] == true || r["has_model"] == true).collect()
}

pub fn list_exports(p: &Project) -> Vec<Value> {
    let mut out: Vec<Value> = std::fs::read_dir(exports_dir(p)).map(|d| d.flatten().filter_map(|e| {
        let mut meta = read_json(&e.path().join("meta.json"))?;
        // a "merged model only" export has no GGUF
        let gguf = meta["gguf_file"].as_str().map(|f| e.path().join(f));
        meta["gguf_exists"] = json!(gguf.as_ref().is_some_and(|g| g.is_file()));
        meta["dir"] = json!(e.path());
        meta["size_mb"] = json!(gguf.as_ref().and_then(|g| std::fs::metadata(g).ok()).map(|m| (m.len() as f64 / 1048576.0 * 10.0).round() / 10.0));
        meta["has_vision"] = json!(meta["mmproj_file"].as_str().is_some_and(|f| e.path().join(f).is_file()));
        meta["merged_exists"] = json!(meta["merged_dir"].as_str().is_some_and(|d| std::path::Path::new(d).join("config.json").is_file()));
        Some(meta)
    }).collect()).unwrap_or_default();
    out.sort_by(|a, b| b["name"].as_str().cmp(&a["name"].as_str()));
    out
}

/// Makes the GGUF of a run in the worker and writes the Modelfile and the model card next to it.
pub fn execute(p: &Project, run: &str, quant: &str, num_ctx: u32, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    execute_what(p, run, quant, num_ctx, "gguf", kill, on_event)
}

/// What an export makes: "gguf" (GGUF + Modelfile), "gguf_merged" (also keeps the merged 16-bit model as safetensors, 10 to 20 GB) or "merged" (only that model).
pub const WHAT: [&str; 3] = ["gguf", "gguf_merged", "merged"];

pub fn execute_what(p: &Project, run: &str, quant: &str, num_ctx: u32, what: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let mut v = execute_many(p, run, &[quant.to_string()], num_ctx, what, kill, on_event)?;
    Ok(v.remove(0))
}

/// Several GGUF sizes of one run in one go: the model is merged and converted once, then shrunk to each size. Every size becomes its own export
/// (`exports/<run>-<size>/`, with its own Modelfile and meta). With "gguf_merged" the merged model is kept in the first one. Returns the metas.
pub fn execute_many(p: &Project, run: &str, quants: &[String], num_ctx: u32, what: &str, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Vec<Value>> {
    if !WHAT.contains(&what) {
        return Err(Error::new(format!("Unknown export kind: {what}")));
    }
    let (want_gguf, keep_merged) = (what != "merged", what != "gguf");
    let mut list: Vec<String> = Vec::new();
    for q in quants {
        if !QUANTS.iter().any(|(id, _)| id == q) {
            return Err(Error::new(format!("Unknown file size: {q}")));
        }
        if !list.contains(q) {
            list.push(q.clone());
        }
    }
    if list.is_empty() {
        return Err(Error::new("Choose at least one file size."));
    }
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let rdir = runs_dir(p).join(run);
    let adapter = rdir.join("adapter");
    let full = rdir.join("model");
    let is_full = full.join("config.json").is_file();
    if !is_full && !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new("This run has no saved model to export."));
    }
    let cfg = read_json(&rdir.join("config.json")).ok_or_else(|| Error::new("The settings of this run are missing."))?;
    let summary = read_json(&rdir.join("summary.json")).unwrap_or(Value::Null);
    let (repo, family) = (cfg["model"]["repo"].as_str().unwrap_or("").to_string(), cfg["model"]["family"].as_str().unwrap_or("generic").to_string());
    // one size: the worker writes straight into its export; several: into a work folder, from which each size moves into its own export
    let list = if want_gguf { list } else { list[..1].to_vec() };
    let many = want_gguf && list.len() > 1;
    let name_of = |q: &str| if want_gguf { format!("{run}-{q}") } else { format!("{run}-merged") };
    let out = if many { exports_dir(p).join(format!("{run}-gguf-work")) } else { exports_dir(p).join(name_of(&list[0])) };
    let _ = std::fs::remove_dir_all(if many { out.clone() } else { out.join("__none__") });
    std::fs::create_dir_all(&out)?;
    // a base made by "Slim or grow a model" keeps the original's vision part: its vision file comes from the original model
    let origin = read_json(&std::path::Path::new(&repo).join("ndforge_reshape.json")).and_then(|m| m["origin"]["repo"].as_str().map(String::from));
    let mut job = json!({"job": "export", "base": {"repo": repo, "family": family}, "out": out, "quant": list[0], "quants": list, "backend": "auto", "gguf": want_gguf, "keep_merged": keep_merged,
                     "max_seq": cfg["hyper"]["max_seq"]});
    if let Some(o) = &origin {
        job["base"]["vision_repo"] = json!(o);
    }
    // a full fine-tune saved the whole model (nothing to merge); a LoRA run saved an adapter
    if is_full {
        job["full_model"] = json!(full);
    } else {
        job["adapter"] = json!(adapter);
    }
    let result = run_job(&py, &worker_dir(), &job, kill, &mut |e| on_event(match e {
        WorkerEvent::Log(l) => json!({"kind": "log", "text": l}),
        WorkerEvent::Progress { step, value, text } => json!({"kind": "progress", "step": step, "value": value, "text": text}),
        WorkerEvent::Other(v) => v,
    }))?;
    let oname = ollama_name(&p.model_label(), "");
    let created = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let mut merged_dir = result["merged_dir"].as_str().filter(|_| keep_merged).map(String::from);
    if !want_gguf {
        if merged_dir.is_none() {
            return Err(Error::new("No merged model was made."));
        }
        let meta = json!({"name": name_of(""), "run": run, "quant": null, "family": family, "base_repo": repo, "gguf_file": null, "merged_dir": merged_dir, "ollama_name": oname, "num_ctx": num_ctx,
                          "created": created, "world": p.display_name(), "installed": []});
        write_json(&out.join("meta.json"), &meta)?;
        return Ok(vec![meta]);
    }
    let license = crate::train::catalog::models().into_iter().find(|m| m.repo == repo).map(|m| m.license).unwrap_or_else(|| "see the model page for its licence".to_string());
    let version = cfg["dataset"]["version"].as_u64().unwrap_or(0);
    let mut metas = Vec::new();
    for (i, quant) in list.iter().enumerate() {
        let gguf = result["ggufs"][quant.as_str()].as_str().or_else(|| if list.len() == 1 { result["gguf"].as_str() } else { None }).map(PathBuf::from)
            .ok_or_else(|| Error::new(format!("No GGUF file was made for {quant}.")))?;
        let dir = exports_dir(p).join(name_of(quant));
        if dir != out {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir)?;
        }
        let gguf_file = gguf.file_name().and_then(|n| n.to_str()).unwrap_or("model.gguf").to_string();
        if dir != out {
            std::fs::rename(&gguf, dir.join(&gguf_file))?;
        }
        // the merged model stays once: in the first export
        if let (true, Some(m)) = (i == 0 && dir != out, merged_dir.as_ref()) {
            let to = dir.join("merged");
            std::fs::rename(m, &to)?;
            merged_dir = Some(to.display().to_string());
        }
        let keep_here = if i == 0 { merged_dir.clone() } else { None };
        // the base model's vision file (images; the language layers are all that training changed): once per export, next to its GGUF
        let mmproj_file = match result["mmproj"].as_str().map(PathBuf::from).filter(|m| m.is_file()) {
            Some(m) => {
                let name = m.file_name().and_then(|n| n.to_str()).unwrap_or("mmproj-F16.gguf").to_string();
                if dir != out {
                    if i == list.len() - 1 {
                        std::fs::rename(&m, dir.join(&name))?;
                    } else {
                        std::fs::copy(&m, dir.join(&name))?;
                    }
                }
                std::fs::write(dir.join("VISION.md"), vision_readme(&gguf_file, &name))?;
                Some(name)
            }
            None => None,
        };
        std::fs::write(dir.join("Modelfile"), modelfile_text(&gguf_file, &spec_for(&family, num_ctx)))?;
        std::fs::write(dir.join("MODEL_CARD.md"), model_card(&CardInfo {
            world: &p.display_name(), model_name: &oname, base_repo: &repo, license: &license, method: cfg["method"].as_str().unwrap_or("lora"), quant,
            dataset_version: version, steps: summary["steps"].as_u64(), final_loss: summary["final_loss"].as_f64(), ollama_name: &oname, derived_from: origin.as_deref(),
        }))?;
        let meta = json!({"name": name_of(quant), "run": run, "quant": quant, "family": family, "base_repo": repo, "gguf_file": gguf_file, "mmproj_file": mmproj_file, "modalities": result["modalities"], "vision_note": result["vision_note"], "merged_dir": keep_here, "ollama_name": oname, "num_ctx": num_ctx,
                          "created": created, "world": p.display_name(), "installed": []});
        write_json(&dir.join("meta.json"), &meta)?;
        metas.push(meta);
    }
    if many {
        let _ = std::fs::remove_dir_all(&out);
    }
    Ok(metas)
}

/// Remember that an export was installed into a server under a name.
pub fn record_install(p: &Project, export: &str, url: &str, model: &str) -> Result<()> {
    let f = exports_dir(p).join(export).join("meta.json");
    let mut meta = read_json(&f).ok_or_else(|| Error::new("That export no longer exists."))?;
    let entry = json!({"url": url, "model": model, "when": chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()});
    let mut list: Vec<Value> = meta["installed"].as_array().cloned().unwrap_or_default().into_iter().filter(|e| !(e["url"] == entry["url"] && e["model"] == entry["model"])).collect();
    list.push(entry);
    meta["installed"] = json!(list);
    write_json(&f, &meta)
}

pub fn find_export(p: &Project, export: &str) -> Result<(PathBuf, Value)> {
    if export.contains(['/', '\\']) || export.contains("..") {
        return Err(Error::new("Unknown export."));
    }
    let dir = exports_dir(p).join(export);
    let meta = read_json(&dir.join("meta.json")).ok_or_else(|| Error::new("That export no longer exists."))?;
    let Some(file) = meta["gguf_file"].as_str() else { return Err(Error::new("That export has only the merged model, no GGUF file: Ollama needs a GGUF. Make the files with a GGUF.")) };
    let gguf = dir.join(file);
    if !gguf.is_file() {
        return Err(Error::new("The GGUF file of that export is missing."));
    }
    Ok((gguf, meta))
}

fn copy_dir(from: &std::path::Path, to: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let target = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &target)?;
        } else {
            std::fs::copy(e.path(), target)?;
        }
    }
    Ok(())
}

/// Copies the LoRA adapter of a run (adapter files + tokenizer) to `<dest>/<run>-lora/` with a short README that says which base model it belongs to and how to
/// load it in Unsloth and in plain PEFT. Returns the new folder. A full fine-tune has no adapter.
pub fn copy_adapter(p: &Project, run: &str, dest: &std::path::Path) -> Result<PathBuf> {
    if run.contains(['/', '\\']) || run.contains("..") {
        return Err(Error::new("Unknown run."));
    }
    let rdir = runs_dir(p).join(run);
    let adapter = rdir.join("adapter");
    if !adapter.join("adapter_config.json").is_file() {
        return Err(Error::new(if rdir.join("model").join("config.json").is_file() { "A full fine-tune has no separate LoRA add-on: the whole model is saved in the run's folder." } else { "This run has no saved LoRA add-on." }));
    }
    let to = dest.join(format!("{run}-lora"));
    if to.exists() {
        return Err(Error::new(format!("{} already exists. Choose another folder or remove it.", to.display())));
    }
    copy_dir(&adapter, &to)?;
    let cfg = read_json(&rdir.join("config.json")).unwrap_or(Value::Null);
    let (repo, family) = (cfg["model"]["repo"].as_str().unwrap_or("(see adapter_config.json)"), cfg["model"]["family"].as_str().unwrap_or("generic"));
    let loader = if family == "qwen3_5" || family == "gemma4" { "FastModel" } else { "FastLanguageModel" };
    let readme = format!("# LoRA add-on of {run}\n\nTrained for the world \"{world}\" on **{repo}** (use exactly this base model, not a different one).\n\n\
## Unsloth\n```python\nfrom unsloth import {loader}\nmodel, tokenizer = {loader}.from_pretrained(model_name=\"{path}\", max_seq_length=4096, load_in_4bit=False)\n{loader}.for_inference(model)\n```\n\n\
## Plain PEFT\n```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import PeftModel\nbase = AutoModelForCausalLM.from_pretrained(\"{repo}\", dtype=\"auto\")\nmodel = PeftModel.from_pretrained(base, \"{path}\")\ntokenizer = AutoTokenizer.from_pretrained(\"{path}\")\n```\n\n\
## Bake it into the model (merge)\n```python\nmerged = model.merge_and_unload()   # PEFT\nmerged.save_pretrained(\"merged\"); tokenizer.save_pretrained(\"merged\")\n# Unsloth: model.save_pretrained_merged(\"merged\", tokenizer, save_method=\"merged_16bit\")\n```\n\n\
Ask questions the way the training did: the same kind of system message, and for Qwen3.5 / Gemma 4 with thinking off. This add-on knows everything in the world, including GM secrets.\n",
        world = p.display_name(), path = to.display());
    std::fs::write(to.join("README.md"), readme)?;
    Ok(to)
}

/// What to do with the vision file that sits next to the GGUF.
fn vision_readme(gguf: &str, mmproj: &str) -> String {
    format!("# Images with this model\n\nThis folder holds the fine-tuned text model (`{gguf}`) and the vision file of its base model (`{mmproj}`). Training only changed the language layers, so the base model's vision part still fits.\n\n\
## Unsloth Studio / llama.cpp (images work)\n- **Unsloth Studio**: load this folder (or the GGUF; Studio finds the `mmproj` file next to it) and attach pictures in the chat.\n- **llama.cpp**: `llama-server -m {gguf} --mmproj {mmproj}`, or `llama-mtmd-cli -m {gguf} --mmproj {mmproj} --image picture.png -p \"What is in this picture?\"`.\n\n\
## Ollama (text only)\nOllama's GGUF import takes one file, so the program installs only `{gguf}` and the model answers text. Keep using Unsloth Studio or llama.cpp for pictures, or run a normal vision model next to it in Ollama for the photo tools.\n\n\
Pictures are not part of the training: the model will not know your world's maps or art, only what it could read from a picture before.\n")
}
