//! "Mixed: measured fit to a size" as a page of its own (docs/MIXED_QUANTS.md, "Fit to a size (measured)"): from a 16-bit GGUF and an importance matrix the worker measures what every
//! type does to every tensor, plans the types for a size, optionally makes the file and the plain sizes next to it, and the program compares them (Compare quants) and keeps the verdict.
//! Everything that decides is in the worker (`worker/ndforge_worker/quantfit.py`, `quantmeasure.py`); this module builds the jobs, refuses what cannot work in plain words, and keeps what
//! the page shows. Measures are cached in `<project>/quantfit/<key>/`, so a second target, or a run after a stop, does not measure again.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::imatrix_lib::{for_export, free_bytes};
use super::quantcheck::{self, CompareSpec, QuantPick, ReferenceSpec, TextSpec};
use super::runs::{read_json, run_export_job, write_json, MixedMeasured, IMATRIX_STOP_GRACE, MEASURED_ENGINES, MEASURED_LADDER};
use crate::envmgr::{env_dir, python_path};
use crate::error::{Error, Result};
use crate::project::Project;

pub const SAMPLE_ROWS: [u32; 4] = [64, 128, 256, 512];
pub const DEFAULT_SAMPLE_ROWS: u32 = 128;

/// Where the measures, the plans and the files of the page are kept.
pub fn dir(p: &Project) -> PathBuf {
    p.folder.join("quantfit")
}

fn verdict_file(p: &Project) -> PathBuf {
    dir(p).join("verdict.json")
}

/// What the page asks for. Old or partial forms load (every field has a default).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantFitSpec {
    /// the 16-bit GGUF (F16 or BF16: the one an export makes first, `model-f16.gguf`)
    pub gguf: String,
    /// a saved matrix of the library (its id); None = no matrix (a plain weight error, with a warning)
    pub matrix: Option<String>,
    pub target_gb: f64,
    pub floor: Option<String>,
    pub ceiling: Option<String>,
    /// 0 = 128
    pub sample_rows: u32,
    pub engine: Option<String>,
    /// also make the file (and, with `neighbours`, the plain sizes just below and above it)
    pub make: bool,
    pub neighbours: bool,
    /// after making the files: compare them with Compare quants on this text (needs `make`)
    pub compare: Option<TextSpec>,
}

impl QuantFitSpec {
    fn settings(&self) -> MixedMeasured {
        MixedMeasured { target_gb: self.target_gb, matrix: self.matrix.clone(), floor: self.floor.clone(), ceiling: self.ceiling.clone(), sample_rows: self.sample_rows, engine: self.engine.clone() }
    }

    fn check_file(&self) -> Result<()> {
        let f = self.gguf.trim();
        if f.is_empty() {
            return Err(Error::new("Choose the 16-bit GGUF file of the model (model-f16.gguf, made by an export)."));
        }
        if !f.to_ascii_lowercase().ends_with(".gguf") {
            return Err(Error::new(format!("{f} is not a GGUF file (its name should end in .gguf).")));
        }
        if !Path::new(f).is_file() {
            return Err(Error::new(format!("The file {f} was not found.")));
        }
        Ok(())
    }

    /// Plain refusals before anything runs. `need_target`: the size is required (not for the estimate).
    pub fn check(&self, need_target: bool) -> Result<()> {
        self.check_file()?;
        let m = self.settings();
        if need_target || self.target_gb != 0.0 {
            m.check()?;
        } else {
            MixedMeasured { target_gb: 1.0, ..m }.check()?;
        }
        if self.compare.is_some() && !self.make {
            return Err(Error::new("To compare the files they must be made first: tick \"Make the file\"."));
        }
        Ok(())
    }

    fn matrix_file(&self, p: &Project) -> Result<Option<String>> {
        match self.matrix.as_deref().filter(|m| !m.is_empty()) {
            Some(id) => Ok(Some(for_export(p, id)?.1.to_string_lossy().to_string())),
            None => Ok(None),
        }
    }
}

fn python() -> Result<PathBuf> {
    let py = python_path(&env_dir());
    if !py.is_file() {
        return Err(Error::new("The training software is not installed yet. Open \"Training setup\" first."));
    }
    Ok(py)
}

fn rows(spec: &QuantFitSpec) -> u32 {
    if spec.sample_rows == 0 { DEFAULT_SAMPLE_ROWS } else { spec.sample_rows }
}

fn text_or<'a>(v: &'a Option<String>, default: &'a str) -> &'a str {
    v.as_deref().filter(|s| !s.is_empty()).unwrap_or(default)
}

/// About a second: what the measured fit will do, from the header of the file only (minutes, disk, engine, whether llama-quantize has `--tensor-type-file`, the smallest and largest
/// size, the plain sizes, whether the matrix is of this model). No grace time, nothing is written.
pub fn estimate(p: &Project, spec: &QuantFitSpec) -> Result<Value> {
    let py = python()?;
    spec.check(false)?;
    let job = json!({"job": "quantfit_estimate", "gguf": spec.gguf.trim(), "matrix": spec.matrix_file(p)?, "target_gb": (spec.target_gb > 0.0).then_some(spec.target_gb), "sample_rows": rows(spec),
                     "engine": text_or(&spec.engine, "auto"), "floor": text_or(&spec.floor, "Q3_K"), "ceiling": text_or(&spec.ceiling, "Q8_0")});
    std::fs::create_dir_all(dir(p))?;
    run_export_job(&py, &job, &dir(p), None, &AtomicBool::new(false), &mut |_| {})
}

/// The `quantfit` job. Files go to `<project>/quantfit/files/<size>gb/`.
pub fn build_job(p: &Project, spec: &QuantFitSpec, stop_file: &Path) -> Result<Value> {
    spec.check(true)?;
    let out = dir(p).join("files").join(format!("{}gb", spec.target_gb));
    Ok(json!({"job": "quantfit", "gguf": spec.gguf.trim(), "matrix": spec.matrix_file(p)?, "target_gb": spec.target_gb, "floor": text_or(&spec.floor, "Q3_K"), "ceiling": text_or(&spec.ceiling, "Q8_0"),
              "sample_rows": rows(spec), "engine": text_or(&spec.engine, "auto"), "cache_dir": dir(p), "stop_file": stop_file, "make": spec.make, "neighbours": spec.make && spec.neighbours,
              "out_dir": out}))
}

/// Measures, plans and (with `make`) makes the files; with `compare` then runs Compare quants on them with the 16-bit file as the reference and keeps the verdict. Stop writes the
/// STOP file first (the worker ends cleanly, the measures so far are cached; "Measure and plan" again resumes), the worker is killed after the grace time.
pub fn run(p: &Project, spec: &QuantFitSpec, kill: &AtomicBool, on_event: &mut dyn FnMut(Value)) -> Result<Value> {
    let py = python()?;
    std::fs::create_dir_all(dir(p))?;
    let stop = dir(p).join("STOP");
    let job = build_job(p, spec, &stop)?;
    let mut result = run_export_job(&py, &job, &dir(p), Some(IMATRIX_STOP_GRACE), kill, &mut |e| on_event(crate::reshape::runs::event_json(e)))?;
    if let Some(text) = spec.compare.clone().filter(|_| spec.make) {
        let files = result["files"].as_array().cloned().unwrap_or_default();
        if files.is_empty() {
            result["compare_note"] = json!("Nothing was made, so there is nothing to compare.");
        } else {
            let quants: Vec<QuantPick> = files.iter().filter_map(|f| serde_json::from_value(f.clone()).ok()).collect();
            let cs = CompareSpec { reference: ReferenceSpec { kind: "file".into(), file: Some(spec.gguf.trim().to_string()), source: None }, quants, text, ..Default::default() };
            let c = quantcheck::compare(p, &cs, kill, on_event)?;
            let saved = json!({"target_gb": spec.target_gb, "comparison": c["entry"]["id"], "promotion": c["promotion"], "verdict": c["verdict"], "files": result["files"]});
            write_json(&verdict_file(p), &saved)?;
            result["comparison"] = c;
        }
    }
    Ok(result)
}

/// What the page shows: the 16-bit files it can offer (the exports' `model-f16.gguf` and the references kept by comparisons), the choices, the plans made so far and the last verdict.
pub fn state(p: &Project) -> Value {
    let refs: Vec<Value> = quantcheck::candidates(p)["references"].as_array().cloned().unwrap_or_default();
    let mut plans = vec![];
    if let Ok(d) = std::fs::read_dir(dir(p)) {
        for e in d.flatten().filter(|e| e.path().is_dir() && e.file_name() != "files") {
            if let Ok(f) = std::fs::read_dir(e.path()) {
                for f in f.flatten().filter(|f| f.file_name().to_string_lossy().starts_with("plan-")) {
                    if let Some(v) = read_json(&f.path()) {
                        plans.push(json!({"key": e.file_name().to_string_lossy(), "target_gb": v["target_gb"], "engine": v["engine"], "fits": v["fits"], "file": f.path()}));
                    }
                }
            }
        }
    }
    plans.sort_by(|a, b| a["file"].as_str().cmp(&b["file"].as_str()));
    json!({"references": refs, "ladder": MEASURED_LADDER, "engines": MEASURED_ENGINES, "sample_rows": SAMPLE_ROWS, "default_sample_rows": DEFAULT_SAMPLE_ROWS, "default_floor": "Q3_K", "default_ceiling": "Q8_0",
           "plans": plans, "verdict": read_json(&verdict_file(p)).unwrap_or(Value::Null), "free_bytes": free_bytes(&dir(p)), "env_ready": crate::envmgr::health::env_installed(&env_dir()),
           "caveats": "The error is measured with a diagonal approximation of the importance matrix (it ignores how input channels move together). Every tensor and layer counts the same for the same output error; nothing runs the model. Only about 1 to 3 percent of the rows were looked at. Compare quants (KL divergence) is the judge, not this number."})
}
