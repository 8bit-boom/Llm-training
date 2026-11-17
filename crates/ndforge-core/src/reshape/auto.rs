//! Auto-cut, the orchestrator (docs/AUTO_CUT.md): the steps of the Slim page, one after the other, as a state machine that can be stopped at any point and continued.
//!
//! The state is `models/_auto/<run>/auto.json`, written (atomically) after every step; a step is only marked done when its work is on the disk, and every step looks at
//! the state first, so a step that was cut off is simply run again and one that finished is not. What the steps DO is behind the `Effects` trait: the real one
//! (`auto_fx::RealEffects`) calls the existing jobs (`reshape_inspect`, `reshape_measure`, `reshape_try`, `reshape`, the heal jobs, `think_check`, export); the tests use a
//! stand-in that only records, so the state machine is tested on every system without a worker.
//!
//! Steps: plan, measure, try, then per stage cut / check_cut / heal / check_heal, then export, check_gguf, studio_publish (and, in `auto_gates`, the report).
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::auto_plan::{self, AutoPlan, AutoSpec, Candidate, StagePlan};
use super::plan::{Inspect, ReasoningCalib, ReshapeSpec};
use super::runs::{models_dir, now, read_json, write_json, HealSpec, TrySpec};
use super::thinkcheck;
use crate::error::{Error, Result};
use crate::project::Project;

pub const FORMAT: u32 = 1;
/// The samples of the cut's measuring, of the width's measuring and of a quick Try (the Slim page's own numbers).
const CALIB_SAMPLES: u32 = 128;
const WIDTH_SAMPLES: u32 = 256;
const TRY_RECHECK_SAMPLES: u32 = 256;

// ------------------------------------------------------------------------------------------------------------------ what the steps do

/// The answer of a long job: done, or stopped by the person (nothing half-written is kept; the step is run again on Continue).
#[derive(Debug, Clone, PartialEq)]
pub enum Out<T> {
    Done(T),
    Stopped,
}

/// A model of the model list, as far as Auto-cut needs it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub id: String,
    pub repo: String,
    pub name: String,
    pub family: String,
}

/// A model Auto-cut made: the unhealed cut of a stage (`model/` of `models/<name>`) or a healed model (`healed/` of `models/<name>`, the folder `heal_from_source` made).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Made {
    /// "cut" | "healed"
    pub kind: String,
    pub name: String,
    pub model_id: Option<String>,
    pub stage: u32,
    pub removed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CutRef {
    pub name: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealRef {
    /// the folder under `models/` (the cut's own for a first heal)
    pub name: String,
    /// the healed model in the model list
    pub model_id: String,
    /// `models/<name>/healed`
    pub dir: String,
    pub tokens_m: u32,
}

/// What a heal learns from.
#[derive(Debug, Clone, PartialEq)]
pub enum Student {
    /// a model Auto-cut just cut (`models/<name>/model`)
    Made(String),
    /// the healed folder of an earlier try (a second heal on top of the first)
    Folder(String),
}

/// Everything that touches the outside world. All of it is one job at a time on one card.
pub trait Effects {
    /// A progress or state event for the window.
    fn emit(&mut self, event: Value);
    /// The Stop button was pressed.
    fn stopped(&self) -> bool;
    fn model(&self, id: &str) -> Result<ModelRef>;
    fn has_training_data(&self) -> bool;
    fn vram_gb(&self) -> Option<f64>;
    /// free space on the disk the models are written to
    fn free_disk_gb(&self) -> Option<f64>;
    fn inspect(&mut self, model: &ModelRef) -> Result<Out<Inspect>>;
    fn measure(&mut self, model_id: &str, samples: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Result<Out<Value>>;
    /// Runs the cuts of a Try-first request and returns the saved analysis (`reshape_try.json`'s `analysis`).
    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>>;
    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>>;
    /// One thinking check (`target`: original | cut | healed | gguf) of the model folder `name`; returns the worker's result (`main`, ...).
    fn think_check(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>) -> Result<Out<Value>>;
    /// Heals `student` with `teacher_repo` (the original model) as the teacher.
    fn heal(&mut self, student: &Student, teacher_repo: &str, spec: &HealSpec) -> Result<Out<HealRef>>;
    /// Deletes what Auto-cut made (only ever something in the state's list of `Made`).
    fn remove(&mut self, what: &Made) -> Result<()>;
    /// The GGUF files of the healed model: `{"dir", "ggufs": [{"quant", "path"}]}`.
    fn export(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool) -> Result<Out<Value>>;
    /// Puts the GGUF files into Unsloth Studio's share (docs/UNSLOTH_API_CONTRACT.md); `{"published": false, "why": ...}` when Studio is not set up.
    fn publish(&mut self, export: &Value, model_name: &str) -> Result<Out<Value>>;
}

// ------------------------------------------------------------------------------------------------------------------ the state

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepRec {
    pub id: String,
    pub label: String,
    /// pending | running | done | skipped | failed | stopped
    pub status: String,
    pub note: Option<String>,
    pub started: Option<String>,
    pub finished: Option<String>,
}

impl StepRec {
    fn new(id: &str, label: &str) -> Self {
        StepRec { id: id.into(), label: label.into(), status: "pending".into(), note: None, started: None, finished: None }
    }
}

/// One stage as it runs: the plan, what was made, what the checks said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageState {
    pub plan: StagePlan,
    pub cut: Option<CutRef>,
    pub heal: Option<HealRef>,
    /// {"verdict", "gates", "main"} of the thinking check of the cut / the healed model
    pub cut_check: Option<Value>,
    pub heal_check: Option<Value>,
    /// 0: the first heal, 1: healed again on top of the first (`auto_gates`), 2: the cut was backed off
    pub attempt: u32,
    /// more tokens for the next heal (`auto_gates`: 40 million more for the second try)
    pub extra_tokens_m: u32,
    pub passed: bool,
}

impl StageState {
    pub fn new(plan: StagePlan) -> Self {
        StageState { plan, cut: None, heal: None, cut_check: None, heal_check: None, attempt: 0, extra_tokens_m: 0, passed: false }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoState {
    pub format: u32,
    pub id: String,
    pub spec: AutoSpec,
    /// running | stopped | failed | done | gates_failed
    pub status: String,
    pub step: String,
    pub error: Option<String>,
    pub created: String,
    pub updated: String,
    pub original: Option<ModelRef>,
    pub inspect: Option<Inspect>,
    pub plan: Option<AutoPlan>,
    pub steps: Vec<StepRec>,
    pub chosen: Option<Candidate>,
    /// the blocks Try first found cheapest for the chosen number of blocks (cut as the layers of these blocks)
    pub drop_blocks: Option<Vec<u32>>,
    pub try_result: Option<Value>,
    /// the first answer of Try first ({"samples", "analysis"}): kept so that a stop between its two requests does not ask the first again
    pub try_scan: Option<Value>,
    pub measure: Option<Value>,
    pub baseline: Option<Value>,
    pub stages: Vec<StageState>,
    pub made: Vec<Made>,
    pub export: Option<Value>,
    pub gguf_check: Option<Value>,
    pub publish: Option<Value>,
    pub report: Option<Value>,
    /// notes for the report: what was backed off, retried, skipped
    pub notes: Vec<String>,
}

impl AutoState {
    fn new(id: &str, spec: AutoSpec) -> Self {
        AutoState {
            format: FORMAT, id: id.into(), spec, status: "running".into(), step: "plan".into(), error: None, created: now(), updated: now(), original: None, inspect: None, plan: None,
            steps: vec![StepRec::new("plan", "Check the model, the card and the disk")], chosen: None, drop_blocks: None, try_result: None, try_scan: None, measure: None, baseline: None,
            stages: Vec::new(), made: Vec::new(), export: None, gguf_check: None, publish: None, report: None, notes: Vec::new(),
        }
    }

    fn step_mut(&mut self, id: &str) -> Option<&mut StepRec> {
        self.steps.iter_mut().find(|s| s.id == id)
    }

    /// The first step that is not done or skipped.
    pub fn next_step(&self) -> Option<String> {
        self.steps.iter().find(|s| s.status != "done" && s.status != "skipped").map(|s| s.id.clone())
    }

    fn keep_thinking(&self) -> bool {
        self.spec.keep_thinking
    }

    fn reasoning(&self) -> Option<ReasoningCalib> {
        self.keep_thinking().then(ReasoningCalib::default)
    }

    /// The model the cut of stage `idx` (0-based) starts from: the original, or the healed model of the stage before.
    fn input_id(&self, idx: usize) -> Option<String> {
        if idx == 0 {
            self.original.as_ref().map(|o| o.id.clone())
        } else {
            self.stages.get(idx - 1).and_then(|s| s.heal.as_ref()).map(|h| h.model_id.clone())
        }
    }
}

pub fn auto_dir(p: &Project) -> PathBuf {
    models_dir(p).join("_auto")
}

pub fn run_dir(p: &Project, id: &str) -> PathBuf {
    auto_dir(p).join(id)
}

fn state_file(dir: &Path) -> PathBuf {
    dir.join("auto.json")
}

pub fn save(dir: &Path, st: &mut AutoState) -> Result<()> {
    st.updated = now();
    write_json(&state_file(dir), &serde_json::to_value(&*st)?)
}

pub fn load(dir: &Path) -> Option<AutoState> {
    read_json(&state_file(dir)).and_then(|v| serde_json::from_value(v).ok()).filter(|s: &AutoState| s.format == FORMAT)
}

/// Every Auto-cut run of the project, newest first.
pub fn list(p: &Project) -> Vec<AutoState> {
    let mut out: Vec<AutoState> = std::fs::read_dir(auto_dir(p)).map(|d| d.flatten().filter_map(|e| load(&e.path())).collect()).unwrap_or_default();
    out.sort_by(|a, b| b.id.cmp(&a.id));
    out
}

/// The run the window shows: the newest.
pub fn latest(p: &Project) -> Option<AutoState> {
    list(p).into_iter().next()
}

fn run_id(model: &ModelRef, target_b: f64) -> String {
    format!("{}-{}-{}b", chrono::Utc::now().format("%Y%m%d-%H%M%S"), super::runs::slug(&model.name), target_b.round() as i64)
}

// ------------------------------------------------------------------------------------------------------------------ the list of steps

fn stage_steps(n: u32, of: u32, thinking: bool) -> Vec<StepRec> {
    let tag = if of > 1 { format!("Stage {n} of {of}: ") } else { String::new() };
    let mut v = vec![StepRec::new(&format!("s{n}.cut"), &format!("{tag}cut"))];
    if thinking {
        v.push(StepRec::new(&format!("s{n}.check_cut"), &format!("{tag}thinking check of the cut")));
    }
    v.push(StepRec::new(&format!("s{n}.heal"), &format!("{tag}heal")));
    if thinking {
        v.push(StepRec::new(&format!("s{n}.check_heal"), &format!("{tag}thinking check of the healed model")));
    }
    v
}

/// The steps after the plan, for `stages` stages.
fn steps_after_plan(spec: &AutoSpec, stages: u32, with_try: bool) -> Vec<StepRec> {
    let mut v = vec![StepRec::new("measure", "Measure the model")];
    if with_try {
        v.push(StepRec::new("try", "Try the cuts"));
    }
    for n in 1..=stages {
        v.extend(stage_steps(n, stages, spec.keep_thinking));
    }
    v.push(StepRec::new("export", "Make the GGUF files"));
    if spec.keep_thinking {
        v.push(StepRec::new("check_gguf", "Thinking check of the GGUF"));
    }
    if spec.studio_publish {
        v.push(StepRec::new("studio_publish", "Put the files into Unsloth Studio"));
    }
    v
}

/// Replaces the stage steps (after Try first changed the number of stages); everything before them is kept as it is.
fn rebuild_stage_steps(st: &mut AutoState) {
    let keep: Vec<StepRec> = st.steps.iter().filter(|s| s.id == "plan" || s.id == "measure" || s.id == "try").cloned().collect();
    let mut steps = keep;
    let after = steps_after_plan(&st.spec, st.stages.len() as u32, false);
    steps.extend(after.into_iter().filter(|s| s.id != "measure"));
    st.steps = steps;
}

// ------------------------------------------------------------------------------------------------------------------ running

/// What a step ended with.
enum Flow {
    Done(Option<String>),
    Skipped(String),
    Stopped,
    /// the run ends here with this status (`gates_failed`)
    Halt(String),
}

fn mark(st: &mut AutoState, id: &str, status: &str, note: Option<String>) {
    let now = now();
    if let Some(s) = st.step_mut(id) {
        s.status = status.into();
        if status == "running" {
            s.started = Some(now.clone());
            s.finished = None;
        } else if status != "pending" {
            s.finished = Some(now);
        }
        if note.is_some() {
            s.note = note;
        }
    }
}

fn event(st: &AutoState, id: &str, status: &str, note: Option<&str>) -> Value {
    json!({"kind": "auto", "phase": "auto", "auto_step": id, "status": status, "note": note, "run": st.id})
}

/// Starts a new run: the plan first, then the steps. Returns the state when the run ends (done, stopped, failed or gates_failed).
pub fn start(p: &Project, fx: &mut dyn Effects, spec: AutoSpec) -> Result<AutoState> {
    spec.validate()?;
    let model = fx.model(&spec.model_id)?;
    if let Some(live) = list(p).into_iter().find(|s| s.status == "running") {
        return Err(Error::new(format!("Auto-cut is already running on {}. Stop it first.", live.original.map(|o| o.name).unwrap_or_default())));
    }
    let id = run_id(&model, spec.target_b);
    let dir = run_dir(p, &id);
    std::fs::create_dir_all(&dir)?;
    let mut st = AutoState::new(&id, spec);
    st.original = Some(model);
    save(&dir, &mut st)?;
    drive(p, &dir, st, fx)
}

/// Continues a stopped or failed run where it was.
pub fn resume(p: &Project, fx: &mut dyn Effects, id: &str) -> Result<AutoState> {
    let dir = run_dir(p, id);
    let mut st = load(&dir).ok_or_else(|| Error::new("That Auto-cut run was not found."))?;
    match st.status.as_str() {
        "stopped" | "failed" => {}
        "running" => return Err(Error::new("This Auto-cut run is running (or was cut off by closing the program: use Continue after the program has marked it stopped).")),
        "done" => return Ok(st),
        "gates_failed" => return Err(Error::new("This run ended because the checks failed twice. Start a new one with another size.")),
        other => return Err(Error::new(format!("This run cannot be continued ({other})."))),
    }
    st.status = "running".into();
    st.error = None;
    for s in st.steps.iter_mut().filter(|s| s.status == "failed" || s.status == "stopped" || s.status == "running") {
        s.status = "pending".into();
    }
    save(&dir, &mut st)?;
    drive(p, &dir, st, fx)
}

/// A run that was `running` when the program closed is marked stopped, so that it can be continued (the step that was cut off is run again).
pub fn mark_interrupted(p: &Project) -> usize {
    let mut n = 0;
    for st in list(p) {
        if st.status == "running" {
            let dir = run_dir(p, &st.id);
            if let Some(mut s) = load(&dir) {
                s.status = "stopped".into();
                for r in s.steps.iter_mut().filter(|r| r.status == "running") {
                    r.status = "pending".into();
                }
                if save(&dir, &mut s).is_ok() {
                    n += 1;
                }
            }
        }
    }
    n
}

fn drive(p: &Project, dir: &Path, mut st: AutoState, fx: &mut dyn Effects) -> Result<AutoState> {
    loop {
        if fx.stopped() {
            st.status = "stopped".into();
            save(dir, &mut st)?;
            fx.emit(event(&st, &st.step.clone(), "stopped", None));
            return Ok(st);
        }
        let Some(id) = st.next_step() else {
            st.status = "done".into();
            st.step = "done".into();
            save(dir, &mut st)?;
            fx.emit(event(&st, "done", "done", None));
            return Ok(st);
        };
        st.step = id.clone();
        mark(&mut st, &id, "running", None);
        save(dir, &mut st)?;
        fx.emit(event(&st, &id, "running", None));
        match run_step(p, &mut st, &id, fx) {
            Ok(Flow::Done(note)) => {
                fx.emit(event(&st, &id, "done", note.as_deref()));
                mark(&mut st, &id, "done", note);
                save(dir, &mut st)?;
            }
            Ok(Flow::Skipped(why)) => {
                fx.emit(event(&st, &id, "skipped", Some(&why)));
                mark(&mut st, &id, "skipped", Some(why));
                save(dir, &mut st)?;
            }
            Ok(Flow::Stopped) => {
                mark(&mut st, &id, "pending", None);
                st.status = "stopped".into();
                save(dir, &mut st)?;
                fx.emit(event(&st, &id, "stopped", None));
                return Ok(st);
            }
            Ok(Flow::Halt(status)) => {
                let why = st.error.clone();
                mark(&mut st, &id, "failed", why);
                st.status = status;
                save(dir, &mut st)?;
                fx.emit(event(&st, &id, "failed", st.error.clone().as_deref()));
                return Ok(st);
            }
            Err(e) => {
                st.error = Some(e.0.clone());
                st.status = "failed".into();
                mark(&mut st, &id, "failed", Some(e.0.clone()));
                save(dir, &mut st)?;
                fx.emit(event(&st, &id, "failed", Some(&e.0)));
                return Err(e);
            }
        }
    }
}

macro_rules! out {
    ($e:expr) => {
        match $e {
            Out::Done(v) => v,
            Out::Stopped => return Ok(Flow::Stopped),
        }
    };
}

fn run_step(p: &Project, st: &mut AutoState, id: &str, fx: &mut dyn Effects) -> Result<Flow> {
    let _ = p;
    match id {
        "plan" => step_plan(st, fx),
        "measure" => step_measure(st, fx),
        "try" => step_try(st, fx),
        "export" => step_export(st, fx),
        "check_gguf" => step_check_gguf(st, fx),
        "studio_publish" => step_publish(st, fx),
        _ => {
            let (stage, what) = id.split_once('.').ok_or_else(|| Error::new(format!("Unknown step {id}.")))?;
            let idx = stage.trim_start_matches('s').parse::<usize>().ok().and_then(|n| n.checked_sub(1)).filter(|i| *i < st.stages.len()).ok_or_else(|| Error::new(format!("Unknown step {id}.")))?;
            match what {
                "cut" => step_cut(st, idx, fx),
                "check_cut" => step_check_cut(st, idx, fx),
                "heal" => step_heal(st, idx, fx),
                "check_heal" => step_check_heal(st, idx, fx),
                _ => Err(Error::new(format!("Unknown step {id}."))),
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------ the steps

fn step_plan(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let model = st.original.clone().ok_or_else(|| Error::new("The model of this run is missing."))?;
    if !fx.has_training_data() {
        return Err(Error::new("Save a training-data version in Step 2 first: the layers are measured and healed on your world's text."));
    }
    let ins = out!(fx.inspect(&model)?);
    let plan = auto_plan::plan(&ins, &st.spec, fx.vram_gb(), fx.free_disk_gb());
    if !plan.ok {
        st.plan = Some(plan.clone());
        return Err(Error::new(plan.blockers.join(" ")));
    }
    st.stages = plan.stages.iter().cloned().map(StageState::new).collect();
    st.chosen = plan.chosen.clone();
    let try_list = plan.try_list.len();
    st.notes.extend(plan.warnings.iter().cloned());
    st.plan = Some(plan);
    st.inspect = Some(ins);
    let mut steps = vec![st.steps[0].clone()];
    steps.extend(steps_after_plan(&st.spec, st.stages.len() as u32, try_list >= 1));
    st.steps = steps;
    Ok(Flow::Done(st.chosen.as_ref().map(|c| format!("aiming for {:.1} B: {} blocks, MLP {}", c.params_b, c.blocks, c.intermediate.map(|i| i.to_string()).unwrap_or_else(|| "unchanged".into())))))
}

fn step_measure(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let id = st.original.as_ref().map(|o| o.id.clone()).unwrap_or_default();
    let load = st.plan.as_ref().map(|p| p.measure_load.clone()).unwrap_or_else(|| "auto".into());
    let r = out!(fx.measure(&id, WIDTH_SAMPLES, &load, st.reasoning().as_ref())?);
    st.measure = Some(json!({"key": r["key"], "tokens": r["tokens"], "load": r["load"], "loss": r["loss"]}));
    Ok(Flow::Done(None))
}

/// One Try-first row of a candidate.
struct TryRow {
    loss: Option<f64>,
    h1: Option<f64>,
    h2: Option<f64>,
    blocks: Vec<u32>,
}

fn f(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn blocks_of(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect()).unwrap_or_default()
}

/// The row of `analysis` that belongs to candidate `c` (width only: the width curve; blocks only: the best run of that length; both: the combined choice).
fn row_for(analysis: &Value, c: &Candidate, full: Option<u32>, best_blocks: &Value) -> TryRow {
    let thinned = c.intermediate.is_some() && c.intermediate != full;
    let row = |r: &Value, blocks: Vec<u32>| TryRow { loss: f(&r["loss"]), h1: f(&r["loss_h1"]), h2: f(&r["loss_h2"]), blocks };
    let none = TryRow { loss: None, h1: None, h2: None, blocks: Vec::new() };
    if c.blocks == 0 {
        return analysis["widths"].as_array().and_then(|a| a.iter().find(|r| r["intermediate"].as_u64().map(|i| i as u32) == c.intermediate)).map(|r| row(r, Vec::new())).unwrap_or(none);
    }
    let best = &analysis["runs"][c.blocks.to_string()]["best"];
    let blocks = blocks_of(&best["blocks"]);
    if !thinned {
        return if best.is_object() { row(best, blocks) } else { none };
    }
    let want = blocks_of(&best_blocks[c.blocks.to_string()]);
    analysis["choices"].as_array().and_then(|a| a.iter().find(|r| blocks_of(&r["blocks"]) == want && r["intermediate"].as_u64().map(|i| i as u32) == c.intermediate)).map(|r| row(r, want.clone())).unwrap_or(none)
}

fn argmin(v: &[Option<f64>]) -> Option<usize> {
    v.iter().enumerate().filter_map(|(i, x)| x.map(|x| (i, x))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))).map(|(i, _)| i)
}

fn try_spec(st: &AutoState, samples: u32, runs: Vec<u32>, widths: Vec<u32>, choices: Vec<Value>) -> Result<TrySpec> {
    let o = st.original.as_ref().ok_or_else(|| Error::new("The model of this run is missing."))?;
    let load = st.plan.as_ref().map(|p| p.measure_load.clone()).unwrap_or_else(|| "auto".into());
    let v = json!({"model_id": o.id, "samples": samples, "len": 512, "load": load, "measure_samples": WIDTH_SAMPLES, "runs": runs, "widths": widths, "choices": choices,
                   "params_b": st.plan.as_ref().map(|p| p.original_b), "reasoning": st.reasoning()});
    let spec: TrySpec = serde_json::from_value(v).map_err(|e| Error::new(format!("The Try-first request is not valid: {e}")))?;
    spec.validate()?;
    Ok(spec)
}

/// Tries the candidates and picks the cheapest. Two requests: the widths and the best run of every block count, then (for a candidate with blocks AND a thinner MLP) the
/// combined cut with those blocks. When the two halves of the measuring text name different winners, everything is asked again with 256 samples.
fn step_try(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let plan = st.plan.clone().ok_or_else(|| Error::new("The plan is missing."))?;
    let ins = st.inspect.clone().ok_or_else(|| Error::new("The model's description is missing."))?;
    let cands = plan.try_list.clone();
    let full = ins.width_dims.as_ref().map(|d| d.intermediate).filter(|_| ins.width_supported);
    let mut samples = st.spec.try_samples();
    let (rows, agree) = loop {
        let runs: Vec<u32> = { let mut r: Vec<u32> = cands.iter().filter(|c| c.blocks > 0).map(|c| c.blocks).collect(); r.sort_unstable(); r.dedup(); r };
        let widths: Vec<u32> = { let mut w: Vec<u32> = cands.iter().filter(|c| c.blocks == 0).filter_map(|c| c.intermediate).filter(|i| Some(*i) != full).collect(); w.sort_unstable(); w.dedup(); w };
        let first = match st.try_scan.as_ref().filter(|s| s["samples"] == json!(samples)) {
            Some(s) => s["analysis"].clone(),
            None => {
                let a = out!(fx.try_run(&try_spec(st, samples, runs, widths, Vec::new())?)?);
                st.try_scan = Some(json!({"samples": samples, "analysis": a}));
                a
            }
        };
        let mut best_blocks = json!({});
        for c in cands.iter().filter(|c| c.blocks > 0) {
            best_blocks[c.blocks.to_string()] = first["runs"][c.blocks.to_string()]["best"]["blocks"].clone();
        }
        let combos: Vec<Value> = cands.iter().filter(|c| c.blocks > 0 && c.intermediate.is_some() && c.intermediate != full)
            .filter(|c| best_blocks[c.blocks.to_string()].as_array().is_some_and(|a| !a.is_empty()))
            .map(|c| json!({"blocks": best_blocks[c.blocks.to_string()], "width": c.intermediate})).collect();
        // the answers are kept in one file per question, so the second analysis holds the first one's rows as well
        let analysis = if combos.is_empty() { first } else { out!(fx.try_run(&try_spec(st, samples, Vec::new(), Vec::new(), combos)?)?) };
        let rows = cands.iter().map(|c| row_for(&analysis, c, full, &best_blocks)).collect::<Vec<_>>();
        let (h1, h2): (Vec<_>, Vec<_>) = (rows.iter().map(|r| r.h1).collect(), rows.iter().map(|r| r.h2).collect());
        let agree = argmin(&h1).is_none() || argmin(&h1) == argmin(&h2);
        if agree || samples >= TRY_RECHECK_SAMPLES {
            break (rows, agree);
        }
        st.notes.push(format!("The two halves of the measuring text named different winners with {samples} samples; Try first was asked again with {TRY_RECHECK_SAMPLES}."));
        samples = TRY_RECHECK_SAMPLES;
    };
    let losses: Vec<Option<f64>> = rows.iter().map(|r| r.loss).collect();
    let winner = argmin(&losses);
    let table: Vec<Value> = cands.iter().zip(&rows).map(|(c, r)| json!({"blocks": c.blocks, "intermediate": c.intermediate, "params_b": c.params_b, "loss": r.loss, "loss_h1": r.h1, "loss_h2": r.h2, "cut_blocks": r.blocks})).collect();
    st.try_result = Some(json!({"samples": samples, "rows": table, "winner": winner, "halves_agree": agree}));
    let Some(w) = winner else {
        st.notes.push("Try first gave no answer; the planner's first choice is cut.".into());
        return Ok(Flow::Done(Some("no answer: the first choice".into())));
    };
    let chosen = cands[w].clone();
    st.drop_blocks = (chosen.blocks > 0 && !rows[w].blocks.is_empty()).then(|| rows[w].blocks.clone());
    if st.chosen.as_ref() != Some(&chosen) {
        let steps = auto_plan::split_stages(&ins, &chosen, st.spec.stages);
        let stages = auto_plan::stage_plans(&ins, &st.spec, &steps, fx.vram_gb());
        st.stages = stages.iter().cloned().map(StageState::new).collect();
        if let Some(pl) = st.plan.as_mut() {
            pl.chosen = Some(chosen.clone());
            pl.stages = stages;
        }
        st.chosen = Some(chosen.clone());
        rebuild_stage_steps(st);
    }
    Ok(Flow::Done(Some(format!("{} blocks, MLP {}: loss {:.3}", chosen.blocks, chosen.intermediate.map(|i| i.to_string()).unwrap_or_else(|| "unchanged".into()), losses[w].unwrap_or(0.0)))))
}

/// The layers of whole blocks (every block is `period` layers).
fn layers_of(blocks: &[u32], period: u32) -> Vec<u32> {
    blocks.iter().flat_map(|b| (b * period)..((b + 1) * period)).collect()
}

fn made(st: &mut AutoState, kind: &str, name: &str, model_id: Option<String>, stage: u32) {
    if !st.made.iter().any(|m| m.kind == kind && m.name == name) {
        st.made.push(Made { kind: kind.into(), name: name.into(), model_id, stage, removed: false });
    }
}

fn step_cut(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    if st.stages[idx].cut.is_some() {
        return Ok(Flow::Skipped("already cut".into()));
    }
    let input = st.input_id(idx).ok_or_else(|| Error::new("The model to cut is missing (the stage before has no healed model)."))?;
    let plan = st.stages[idx].plan.clone();
    let period = st.inspect.as_ref().map(|i| i.period).unwrap_or(4);
    let mut spec = plan.reshape_spec(&input, st.reasoning(), CALIB_SAMPLES, WIDTH_SAMPLES);
    spec.load = st.plan.as_ref().map(|p| p.measure_load.clone()).unwrap_or_else(|| "auto".into());
    // the blocks Try first found cheapest are cut as they are; blocks of a later stage are measured on the healed model
    if idx == 0 && plan.blocks_cut > 0 && Some(plan.blocks_cut) == st.chosen.as_ref().map(|c| c.blocks) {
        if let Some(b) = st.drop_blocks.as_ref().filter(|b| b.len() as u32 == plan.blocks_cut) {
            spec.drop_layers = Some(layers_of(b, period));
        }
    }
    let r = out!(fx.cut(&spec)?);
    made(st, "cut", &r.name, Some(r.model_id.clone()), idx as u32 + 1);
    st.stages[idx].cut = Some(r);
    Ok(Flow::Done(None))
}

/// The numbers of a check without the long parts.
fn brief(v: &Value) -> Value {
    match v.as_object() {
        Some(o) => Value::Object(o.iter().filter(|(_, x)| x.is_number() || x.is_string() || x.is_boolean() || x.is_null()).map(|(k, x)| (k.clone(), x.clone())).collect()),
        None => Value::Null,
    }
}

fn check_json(result: &Value, baseline: Option<&Value>) -> Value {
    let gates = baseline.map(|b| thinkcheck::gates(&b["main"], &result["main"]));
    json!({"main": brief(&result["main"]), "control": brief(&result["control"]), "verdict": gates.as_ref().map(|g| g["verdict"].clone()), "gates": gates})
}

fn step_check_cut(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    let cut = st.stages[idx].cut.clone().ok_or_else(|| Error::new("There is no cut to check."))?;
    if st.baseline.is_none() {
        // the original is checked once, from the folder of the first cut (that is where its record of the original is)
        let r = out!(fx.think_check(&cut.name, "original", st.spec.quick, None)?);
        st.baseline = Some(r);
    }
    let r = out!(fx.think_check(&cut.name, "cut", st.spec.quick, None)?);
    st.stages[idx].cut_check = Some(check_json(&r, st.baseline.as_ref()));
    Ok(Flow::Done(None))
}

fn step_heal(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    if st.stages[idx].heal.is_some() {
        return Ok(Flow::Skipped("already healed".into()));
    }
    let cut = st.stages[idx].cut.clone().ok_or_else(|| Error::new("There is no cut to heal."))?;
    let teacher = st.original.as_ref().map(|o| o.repo.clone()).ok_or_else(|| Error::new("The original model is missing."))?;
    let mut spec = st.stages[idx].plan.heal_spec(st.keep_thinking());
    spec.tokens_m += st.stages[idx].extra_tokens_m;
    let h = out!(fx.heal(&Student::Made(cut.name.clone()), &teacher, &spec)?);
    made(st, "healed", &h.name, Some(h.model_id.clone()), idx as u32 + 1);
    st.stages[idx].heal = Some(h);
    Ok(Flow::Done(None))
}

fn step_check_heal(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    let h = st.stages[idx].heal.clone().ok_or_else(|| Error::new("There is no healed model to check."))?;
    let r = out!(fx.think_check(&h.name, "healed", st.spec.quick, None)?);
    let chk = check_json(&r, st.baseline.as_ref());
    let verdict = chk["verdict"].as_str().unwrap_or("pass").to_string();
    st.stages[idx].heal_check = Some(chk);
    if verdict == "fail" {
        let failed = st.stages[idx].heal_check.as_ref().map(|c| c["gates"]["failed"].clone()).unwrap_or(Value::Null);
        st.error = Some(format!("The thinking check of the healed model of stage {} failed ({}).", idx + 1, failed.as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default()));
        return Ok(Flow::Halt("gates_failed".into()));
    }
    st.stages[idx].passed = true;
    cleanup_after(st, idx, fx)?;
    Ok(Flow::Done(Some(verdict)))
}

/// Without `keep_intermediate`: once stage `idx` has passed its check, its unhealed cut goes, and the healed model of the stage before (the next stage's input has been made).
/// Only what Auto-cut made is ever removed, and never the healed model of the last stage.
fn cleanup_after(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<()> {
    if st.spec.keep_intermediate {
        return Ok(());
    }
    let stage = idx as u32 + 1;
    let victims: Vec<usize> = st.made.iter().enumerate().filter(|(_, m)| !m.removed && ((m.kind == "cut" && m.stage == stage) || (m.kind == "healed" && m.stage + 1 == stage))).map(|(i, _)| i).collect();
    for i in victims {
        let m = st.made[i].clone();
        fx.remove(&m)?;
        st.made[i].removed = true;
    }
    Ok(())
}

fn final_heal(st: &AutoState) -> Result<HealRef> {
    st.stages.last().and_then(|s| s.heal.clone()).ok_or_else(|| Error::new("There is no healed model to export."))
}

fn step_export(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    if st.export.is_some() {
        return Ok(Flow::Skipped("already exported".into()));
    }
    let h = final_heal(st)?;
    let family = st.original.as_ref().map(|o| o.family.clone()).unwrap_or_else(|| "generic".into());
    let quants = st.spec.export_quants();
    let r = out!(fx.export(&st.id.clone(), &h, &family, &quants, st.keep_thinking())?);
    st.export = Some(r);
    Ok(Flow::Done(None))
}

/// The Q4_K_M file (or the first one) of an export.
fn gguf_of(export: &Value) -> Option<String> {
    let list = export["ggufs"].as_array()?;
    list.iter().find(|g| g["quant"] == "q4_k_m").or_else(|| list.first()).and_then(|g| g["path"].as_str()).map(String::from)
}

fn step_check_gguf(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let h = final_heal(st)?;
    let file = st.export.as_ref().and_then(gguf_of).ok_or_else(|| Error::new("The export made no GGUF file to check."))?;
    let r = out!(fx.think_check(&h.name, "gguf", st.spec.quick, Some(&file))?);
    st.gguf_check = Some(check_json(&r, st.baseline.as_ref()));
    Ok(Flow::Done(None))
}

fn step_publish(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let export = st.export.clone().ok_or_else(|| Error::new("There is nothing to publish: the export is missing."))?;
    let name = st.original.as_ref().map(|o| o.name.clone()).unwrap_or_default();
    let r = out!(fx.publish(&export, &name)?);
    let note = if r["published"] == true { None } else { r["why"].as_str().map(String::from) };
    st.publish = Some(r);
    Ok(Flow::Done(note))
}
