//! Mixed-precision sizes ("Mixed precision (our own recipe)", docs/MIXED_QUANTS.md): what the export page needs besides the three recipes and "Fit to a size" in the list of sizes
//! (those are `runs::MIXED_QUANTS`): the GGUF files whose tensor shapes the size estimate can be read from, and the worker's plan (every recipe's rules and estimated size, the fit).
//! The recipes themselves are ONE table in `worker/ndforge_worker/mixedquant.py`; the program only shows what the worker says.
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::imatrix_lib::{for_export, lib_dir};
use super::runs::{list_exports, run_export_job, MIXED_RECIPE};
use crate::envmgr::{env_dir, python_path};
use crate::error::{Error, Result};
use crate::project::Project;

/// What the page shows around the mixed sizes: the name every recipe carries, and the GGUF files of the exports to read tensor shapes from (any size of the model will do:
/// only the names and shapes in its header are read).
pub fn state(p: &Project) -> Value {
    let mut files = vec![];
    for m in list_exports(p) {
        let Some(dir) = m["dir"].as_str().map(PathBuf::from) else { continue };
        let Some(f) = m["gguf_file"].as_str().map(|g| dir.join(g)).filter(|g| g.is_file()) else { continue };
        files.push(json!({"path": f, "run": m["run"], "quant": m["quant"], "name": m["name"], "label": format!("{}: {}", m["name"].as_str().unwrap_or(""), f.file_name().and_then(|n| n.to_str()).unwrap_or(""))}));
    }
    json!({"recipe": MIXED_RECIPE, "files": files})
}

/// The worker's plan before an export: for every recipe its rules, the `llama-quantize` options, the estimated size and bits per weight (from the tensor shapes in `gguf`), and, when
/// `target_gb` and a saved matrix (a library id) are given, the "Fit to a size" proposal. Plain messages for a file that cannot be read and for a matrix that cannot be used.
pub fn plan(p: &Project, gguf: &str, target_gb: Option<f64>, matrix: Option<&str>) -> Result<Value> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    let gguf = gguf.trim();
    if gguf.is_empty() {
        return Err(Error::new("Choose a GGUF file of the model: any size of it will do, only its tensor shapes are read."));
    }
    if !std::path::Path::new(gguf).is_file() {
        return Err(Error::new(format!("The file {gguf} was not found.")));
    }
    let matrix_file = match matrix.filter(|m| !m.is_empty()) {
        Some(id) => Some(for_export(p, id)?.1),
        None => None,
    };
    let target = target_gb.filter(|t| t.is_finite() && *t > 0.0);
    std::fs::create_dir_all(lib_dir(p))?;
    let job = json!({"job": "mixed_plan", "gguf": gguf, "target_gb": target, "matrix": matrix_file});
    run_export_job(&py, &job, &lib_dir(p), None, &AtomicBool::new(false), &mut |_| {})
}
