//! Auto-cut, the orchestrator (docs/AUTO_CUT.md): the steps of the Slim page, one after the other, as a state machine that can be stopped at any point and continued.
//!
//! The state is `models/_auto/<run>/auto.json`, written (atomically) after every step; a step is only marked done when its work is on the disk, and every step looks at
//! the state first, so a step that was cut off is simply run again and one that finished is not. What the steps DO is behind the `Effects` trait: the real one
//! (`auto_fx::RealEffects`) calls the existing jobs (`reshape_inspect`, `reshape_measure`, `reshape_try`, `reshape`, the heal jobs, `think_check`, export); the tests use a
//! stand-in that only records, so the state machine is tested on every system without a worker.
//!
//! Steps: plan, measure, try, then per stage cut / check_cut / heal / check_heal, then export, check_gguf, compare_quants (only with a target size), studio_publish (and, when the run ends, the report: `auto_report`).
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::auto_speed;
use super::heal_rounds;
use super::auto_plan::{self, AutoPlan, AutoSpec, Candidate, StagePlan};
use super::plan::{Inspect, ReasoningCalib, ReshapeSpec};
use super::runs::{models_dir, now, read_json, write_json, HealSpec, TrySpec};
use super::auto_progress::ProgressTracker;
use super::auto_report;
use super::search;
use super::thinkcheck;
use crate::error::{Error, Result};
use crate::project::Project;

pub const FORMAT: u32 = 1;
/// The samples of the cut's measuring, of the width's measuring and of a quick Try (the Slim page's own numbers).
/// How many of the newest saved job logs the time estimate learns from.
const HISTORY_LOGS: usize = 60;
const CALIB_SAMPLES: u32 = 128;
const WIDTH_SAMPLES: u32 = 256;
const TRY_RECHECK_SAMPLES: u32 = 256;
/// The tokens the second heal of a stage adds (the retry after a failed check).
pub use super::auto_plan::RETRY_TOKENS_M;

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
    /// the numbers the heal ended with (kd_first, kd_last, calib_loss_before / _after, teacher_calib_loss) for the report
    #[serde(default)]
    pub summary: Option<Value>,
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
    /// The progress tracker of this run (`auto_progress`), if the effects keep one; the default has none (the stand-in of a test opts in).
    fn tracker(&mut self) -> Option<&mut ProgressTracker> {
        None
    }
    /// The Stop button was pressed.
    fn stopped(&self) -> bool;
    fn model(&self, id: &str) -> Result<ModelRef>;
    fn has_training_data(&self) -> bool;
    fn vram_gb(&self) -> Option<f64>;
    /// free space on the disk the models are written to
    fn free_disk_gb(&self) -> Option<f64>;
    /// what this computer's saved job logs say about its speed (the time estimate); nothing by default
    /// The clock of the time limit (seconds since 1970); the stand-in of a test sets it.
    fn now_secs(&self) -> i64 {
        chrono::Utc::now().timestamp()
    }
    fn history(&self) -> crate::timing::History {
        crate::timing::History::default()
    }
    fn inspect(&mut self, model: &ModelRef) -> Result<Out<Inspect>>;
    fn measure(&mut self, model_id: &str, samples: u32, load: &str, reasoning: Option<&ReasoningCalib>) -> Result<Out<Value>>;
    /// Runs the cuts of a Try-first request and returns the saved analysis (`reshape_try.json`'s `analysis`).
    fn try_run(&mut self, spec: &TrySpec) -> Result<Out<Value>>;
    fn cut(&mut self, spec: &ReshapeSpec) -> Result<Out<CutRef>>;
    /// One thinking check (`target`: original | cut | healed | gguf) of the model folder `name`; returns the worker's result (`main`, ...).
    fn think_check(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>) -> Result<Out<Value>>;
    /// The same check through `engine` ("hf": the standard engine, "vllm": the experimental vLLM engine). The result may carry `engine_used` (`{"engine", "fallback", "note", ...}`): vLLM that could not
    /// do it falls back to the standard engine on its own and says so there. The default ignores the engine.
    fn think_check_with(&mut self, name: &str, target: &str, quick: bool, gguf: Option<&str>, engine: &str) -> Result<Out<Value>> {
        let _ = engine;
        self.think_check(name, target, quick, gguf)
    }
    /// Stops any vLLM engine that is still running (a crash, a Stop): called before a step that is not a vLLM step, so that the card is free for training and the other jobs.
    fn engine_cleanup(&mut self) {}
    /// Heals `student` with `teacher_repo` (the original model) as the teacher.
    fn heal(&mut self, student: &Student, teacher_repo: &str, spec: &HealSpec) -> Result<Out<HealRef>>;
    /// Deletes what Auto-cut made (only ever something in the state's list of `Made`).
    fn remove(&mut self, what: &Made) -> Result<()>;
    /// The last stage has passed: the teacher's shared answers that only the models named here use are not needed any more. Nothing to do for an effects object that has none.
    fn drop_teacher_cache(&mut self, _models: &[String]) -> Result<()> {
        Ok(())
    }
    /// The GGUF files of the healed model: `{"dir", "ggufs": [{"quant", "path"}]}`.
    /// How the export is made, before it runs: `imatrix` = measure an importance matrix for the small sizes (the Fast setting leaves it out). The default keeps the old behaviour.
    fn export_options(&mut self, _imatrix: bool) {}
    fn export(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool) -> Result<Out<Value>>;
    /// `export` for a run with a target size: the same, and "mixed_measured" in `quants` is made with the size `target_gb`. The default ignores the target (an effects object
    /// that does not know it still gets "mixed_measured" in `quants`).
    fn export_measured(&mut self, run: &str, healed: &HealRef, family: &str, quants: &[String], keep_thinking: bool, _target_gb: f64) -> Result<Out<Value>> {
        self.export(run, healed, family, quants, keep_thinking)
    }
    /// Compares the GGUF files `picks` (`[{"path", "label", "group", "matrix", "mixed"}]`) with the 16-bit file `reference` (Compare quants) and returns its results (`rows`, `promotion`,
    /// `verdict`). The default does nothing and says so.
    fn compare_quants(&mut self, _reference: &str, _picks: &[Value]) -> Result<Out<Value>> {
        Ok(Out::Done(json!({"skipped": "This run cannot compare files."})))
    }
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
    /// the time estimate of the step in minutes (low, high), filled from the plan's `time_estimate`; None for a run saved before it existed
    #[serde(default)]
    pub est: Option<(f64, f64)>,
    /// the estimated minutes (middle) of the parts of the step in the order they run (`start`, `load`, `cache`, `train`, `merge` ...)
    #[serde(default)]
    pub est_parts: Vec<(String, f64)>,
}

impl StepRec {
    fn new(id: &str, label: &str) -> Self {
        StepRec { id: id.into(), label: label.into(), status: "pending".into(), note: None, started: None, finished: None, est: None, est_parts: Vec::new() }
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
    /// 0: the first heal, 1: healed again on top of the first, 2: the cut was backed off one step and the stage made again
    pub attempt: u32,
    /// more tokens for the next heal (40 million more for the second try)
    pub extra_tokens_m: u32,
    pub passed: bool,
    /// the loss of this stage's cut as Try first measured it (the cut itself is not measured again): kept for the report
    #[serde(default)]
    pub cut_loss: Option<f64>,
    /// one line per healed model that was checked: {attempt, kind: heal | heal_again | back_off, model, tokens_m, verdict, failed, main}
    #[serde(default)]
    pub attempts: Vec<Value>,
    /// the healed model of the first try, healed again on top (attempt 1)
    #[serde(default)]
    pub retry_from: Option<HealRef>,
    /// a heal in rounds (`spec.heal_rounds`): the `rounds` result of each heal of this stage (the latest has every round), for the report
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rounds: Vec<Value>,
}

impl StageState {
    pub fn new(plan: StagePlan) -> Self {
        StageState { plan, cut: None, heal: None, cut_check: None, heal_check: None, attempt: 0, extra_tokens_m: 0, passed: false, cut_loss: None, attempts: Vec::new(), retry_from: None, rounds: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoState {
    pub format: u32,
    pub id: String,
    /// the program was closed while this run was running (set when it is marked stopped); cleared when the run goes on. `auto_continue_interrupted` looks at it.
    #[serde(default)]
    pub interrupted: bool,
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
    /// the thinking check before healing (docs/AUTO_CUT.md): `{"order", "rows", "pending", "winner", "lost_much"}`; None for a run without it (and for one saved before it existed)
    #[serde(default)]
    pub think_gate: Option<Value>,
    /// the comparison of the measured file with the plain sizes (a run with a target size): `{"promoted": bool, "promotion", "verdict", "rows", "skipped"}`; None for a run without it
    #[serde(default)]
    pub compare: Option<Value>,
    /// "I have at most N hours": what the time limit made the run give up (`auto_speed::fit_budget`); None for a run without a limit
    #[serde(default)]
    pub budget: Option<super::auto_speed::BudgetFit>,
    /// the last `auto_progress` snapshot (saved at the boundaries of the steps: a stopped run says "stopped at 42 %"); None for a run saved before it existed
    #[serde(default)]
    pub progress: Option<Value>,
    /// the cuts the search found best (aligned with the rows of `try_result`): the thinking gate and the check of the first stage's cut read them instead of the planner's own list; None for a classic Try first
    #[serde(default)]
    pub try_cands: Option<Vec<Candidate>>,
    /// experimental vLLM: one entry per step that was planned to use it: `{"step", "kind", "engine": "vllm" | "hf", "fallback", "note", "startup_seconds", "tokens_per_s", "teacher_quant", "estimated_min", ...}`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub engines: Vec<Value>,
    /// the engine the baseline (the original's check) was made with: every later check of the run uses the same one ("vllm" | "hf"); None until the first check
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check_engine: Option<String>,
}

impl AutoState {
    fn new(id: &str, spec: AutoSpec) -> Self {
        AutoState {
            format: FORMAT, id: id.into(), interrupted: false, spec, status: "running".into(), step: "plan".into(), error: None, created: now(), updated: now(), original: None, inspect: None, plan: None,
            steps: vec![StepRec::new("plan", "Check the model, the card and the disk")], chosen: None, drop_blocks: None, try_result: None, try_scan: None, measure: None, baseline: None,
            stages: Vec::new(), made: Vec::new(), export: None, gguf_check: None, publish: None, report: None, notes: Vec::new(), think_gate: None, compare: None, budget: None, progress: None, try_cands: None, engines: Vec::new(), check_engine: None,
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

fn stage_steps(n: u32, of: u32, thinking: bool, check_cut: bool) -> Vec<StepRec> {
    let tag = if of > 1 { format!("Stage {n} of {of}: ") } else { String::new() };
    let mut v = vec![StepRec::new(&format!("s{n}.cut"), &format!("{tag}cut"))];
    if thinking && check_cut {
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
        v.push(StepRec::new("try", if spec.search_minutes.is_some() { "Find the best cut" } else { "Try the cuts" }));
        if spec.keep_thinking && spec.think_gate {
            v.push(StepRec::new("think_gate", "Check the thinking of the best cuts before healing"));
        }
    }
    for n in 1..=stages {
        v.extend(stage_steps(n, stages, spec.keep_thinking, spec.check_cut));
    }
    v.push(StepRec::new("export", "Make the GGUF files"));
    if spec.keep_thinking {
        v.push(StepRec::new("check_gguf", "Thinking check of the GGUF"));
    }
    if spec.target_gb.is_some() {
        v.push(StepRec::new("compare_quants", "Compare the measured file with the plain sizes"));
    }
    if spec.studio_publish {
        v.push(StepRec::new("studio_publish", "Put the files into Unsloth Studio"));
    }
    v
}

/// Replaces the stage steps (after Try first changed the number of stages); everything before them is kept as it is.
fn rebuild_stage_steps(st: &mut AutoState) {
    let keep: Vec<StepRec> = st.steps.iter().filter(|s| s.id == "plan" || s.id == "measure" || s.id == "try" || s.id == "think_gate").cloned().collect();
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
    /// the step stays open and the steps the retry needs were set to pending: the loop goes on with the first one that is not done
    Retry(String),
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
        "gates_failed" => return Err(Error::new("This run ended because the thinking checks failed after the retries (healing again, then a smaller cut). Start a new one with another size.")),
        other => return Err(Error::new(format!("This run cannot be continued ({other})."))),
    }
    st.status = "running".into();
    st.interrupted = false;
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
                s.interrupted = true;
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
            tick(fx, &mut st);
            finish(dir, &mut st);
            fx.emit(event(&st, "done", "done", None));
            return Ok(st);
        };
        st.step = id.clone();
        mark(&mut st, &id, "running", None);
        fill_est(&mut st);
        tick(fx, &mut st);
        save(dir, &mut st)?;
        fx.emit(event(&st, &id, "running", None));
        match run_step(p, &mut st, &id, fx) {
            Ok(Flow::Done(note)) => {
                fx.emit(event(&st, &id, "done", note.as_deref()));
                mark(&mut st, &id, "done", note);
                tick(fx, &mut st);
                save(dir, &mut st)?;
            }
            Ok(Flow::Skipped(why)) => {
                fx.emit(event(&st, &id, "skipped", Some(&why)));
                mark(&mut st, &id, "skipped", Some(why));
                tick(fx, &mut st);
                save(dir, &mut st)?;
            }
            Ok(Flow::Stopped) => {
                mark(&mut st, &id, "pending", None);
                st.status = "stopped".into();
                save(dir, &mut st)?;
                fx.emit(event(&st, &id, "stopped", None));
                return Ok(st);
            }
            Ok(Flow::Retry(note)) => {
                fx.emit(event(&st, &id, "retry", Some(&note)));
                mark(&mut st, &id, "pending", Some(note));
                fill_est(&mut st);
                tick(fx, &mut st);
                save(dir, &mut st)?;
            }
            Ok(Flow::Halt(status)) => {
                let why = st.error.clone();
                mark(&mut st, &id, "failed", why);
                st.status = status;
                finish(dir, &mut st);
                fx.emit(event(&st, &id, "failed", st.error.clone().as_deref()));
                return Ok(st);
            }
            Err(e) => {
                st.error = Some(e.0.clone());
                st.status = "failed".into();
                mark(&mut st, &id, "failed", Some(e.0.clone()));
                finish(dir, &mut st);
                fx.emit(event(&st, &id, "failed", Some(&e.0)));
                return Err(e);
            }
        }
    }
}

/// Fills the estimate of every step that has none from the plan's time estimate (a retry or a rebuilt list of steps gets its figures here too).
fn fill_est(st: &mut AutoState) {
    let Some(t) = st.plan.as_ref().and_then(|p| p.time.clone()) else { return };
    for s in st.steps.iter_mut().filter(|s| s.est.is_none()) {
        if let Some(x) = t.steps.iter().find(|x| x.id == s.id) {
            s.est = Some(x.minutes);
            s.est_parts = x.parts.iter().map(|q| (q.id.clone(), ((q.minutes.0 + q.minutes.1) * 5.0).round() / 10.0)).collect();
        }
    }
}

/// Sends the `auto_progress` snapshot of the steps as they are and keeps it in the state.
fn tick(fx: &mut dyn Effects, st: &mut AutoState) {
    fill_est(st);
    let now = fx.now_secs() as f64;
    if let Some(t) = fx.tracker() {
        let snap = t.sync(st, now);
        st.progress = Some(snap.clone());
        fx.emit(snap);
    }
}

/// The run has ended (done, gates_failed or failed): the report is written next to the state, then the state is saved. A report that cannot be written is not a failure of the run.
fn finish(dir: &Path, st: &mut AutoState) {
    st.report = auto_report::write(dir, st).ok();
    let _ = save(dir, st);
}

/// Sets steps back to pending (a retry runs them again).
fn reopen(st: &mut AutoState, ids: &[String]) {
    for id in ids {
        if let Some(s) = st.step_mut(id) {
            s.status = "pending".into();
            s.finished = None;
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
    if st.spec.vllm.as_ref().is_some_and(|v| v.any()) {
        fx.engine_cleanup();          // the card is free before every step: vLLM takes most of it and must never run next to a training or another job
    }
    match id {
        "plan" => step_plan(st, fx),
        "measure" => step_measure(st, fx),
        "try" => step_try(st, fx),
        "think_gate" => step_think_gate(st, fx),
        "export" => step_export(st, fx),
        "check_gguf" => step_check_gguf(st, fx),
        "compare_quants" => step_compare_quants(st, fx),
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

// ------------------------------------------------------------------------------------------------------------------ vLLM (experimental, docs/AUTO_CUT.md)

fn vllm_decision<'a>(st: &'a AutoState, key: &str) -> Option<&'a crate::vllm_plan::Decision> {
    st.plan.as_ref().and_then(|p| p.time.as_ref()).and_then(|t| t.vllm.as_ref()).and_then(|v| v.decision(key))
}

/// What the plan says about a step's use of vLLM, recorded with what really happened (`used`: the worker's or the effects' account of the step) in the state and, for a fallback, in the notes.
fn record_engine(st: &mut AutoState, key: &str, kind: &str, used: Option<&Value>) {
    let d = vllm_decision(st, key).cloned();
    let engine = used.and_then(|u| u["engine"].as_str()).unwrap_or("hf").to_string();
    let fallback = used.map(|u| u["fallback"] == true || (u["engine"] == "hf" && u["requested"] == true)).unwrap_or(false);
    let mut rec = json!({"step": key, "kind": kind, "engine": engine, "fallback": fallback, "planned_vllm": d.as_ref().is_some_and(|d| d.use_vllm)});
    if let Some(u) = used {
        for k in ["note", "startup_seconds", "load_seconds", "tokens_per_s", "teacher_quant", "route", "engine_version", "teacher_check", "problem", "unchecked"] {
            if !u[k].is_null() {
                rec[k] = u[k].clone();
            }
        }
        if rec["engine_version"].is_null() && !u["engine"].is_null() && u["engine"] != "hf" {
            rec["engine_version"] = u["engine"].clone();
        }
    }
    if let Some(d) = &d {
        rec["estimated_min"] = json!({"vllm": d.vllm_min, "standard": d.standard_min, "measured_basis": d.measured});
    }
    if fallback {
        let why = used.and_then(|u| u["note"].as_str().or(u["problem"].as_str())).unwrap_or("it did not work");
        st.notes.push(format!("vLLM could not be used for {key} ({why}); the standard engine did it and the run went on."));
    }
    st.engines.retain(|e| e["step"] != json!(key));
    st.engines.push(rec);
}

/// One thinking check of the run's step `step` (`s1.check_cut`, `think_gate` ...): through vLLM when the plan chose it for the run's checks, else the standard way. ONE engine for the checks of
/// a run: a check is judged against the original's, and a result is only comparable to one made by the same engine. When a vLLM check falls back to the standard engine, the original is checked
/// again that way so the later comparisons stay honest.
fn run_check(st: &mut AutoState, fx: &mut dyn Effects, step: &str, name: &str, target: &str) -> Result<Out<Value>> {
    let quick = st.spec.think_quick();
    let want = match st.check_engine.as_deref() {
        Some(e) => e == "vllm",
        None => vllm_decision(st, "checks").is_some_and(|d| d.use_vllm),
    };
    let key = format!("{step}:{target}");
    if !want {
        let r = fx.think_check(name, target, quick, None)?;
        if let Out::Done(_) = &r {
            st.check_engine.get_or_insert_with(|| "hf".into());
        }
        return Ok(r);
    }
    fx.engine_cleanup();          // a vLLM job starts on a free card
    let r = fx.think_check_with(name, target, quick, None, "vllm")?;
    let Out::Done(v) = r else { return Ok(Out::Stopped) };
    let used = v.get("engine_used").cloned();
    let fell = used.as_ref().is_some_and(|u| u["engine"] != "vllm");
    if !fell {
        st.check_engine.get_or_insert_with(|| "vllm".into());
        record_engine(st, &key, "checks", Some(&used.clone().unwrap_or_else(|| json!({"engine": "vllm", "fallback": false}))));
        return Ok(Out::Done(v));
    }
    // vLLM did not work for this check: the standard engine did it. The baseline made by vLLM cannot be compared with it, so the original is checked the standard way too
    record_engine(st, &key, "checks", Some(&json!({"engine": "hf", "fallback": true, "requested": true, "note": used.as_ref().and_then(|u| u["note"].as_str()).unwrap_or("it did not work")})));
    if st.check_engine.as_deref() == Some("vllm") && st.baseline.is_some() && target != "original" {
        st.notes.push("The original's thinking check was made with vLLM and a later check could not be: the original is checked again with the standard engine so that the two can be compared.".into());
        let b = fx.think_check(name, "original", quick, None)?;
        let Out::Done(b) = b else { return Ok(Out::Stopped) };
        st.baseline = Some(b);
    }
    st.check_engine = Some("hf".into());
    Ok(Out::Done(v))
}

// ------------------------------------------------------------------------------------------------------------------ the steps

fn step_plan(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let model = st.original.clone().ok_or_else(|| Error::new("The model of this run is missing."))?;
    if !fx.has_training_data() {
        return Err(Error::new("Save a training-data version in Step 2 first: the layers are measured and healed on your world's text."));
    }
    let ins = out!(fx.inspect(&model)?);
    // "I have at most N hours": the least harmful settings are given up until the estimate fits, and every one of them is a note of the run and a line of the report
    if let Some(h) = st.spec.max_hours {
        let fit = auto_speed::fit_budget(&ins, &st.spec, fx.vram_gb(), &fx.history(), h);
        st.budget = Some(fit.clone());
        if !fit.fits {
            return Err(Error::new(fit.message.clone().unwrap_or_else(|| "The run does not fit the time you have.".into())));
        }
        if !fit.dropped.is_empty() {
            st.spec = fit.spec.clone();
            st.notes.extend(auto_speed::budget_sentences(&fit));
        }
    }
    let plan = auto_plan::plan(&ins, &st.spec, fx.vram_gb(), fx.free_disk_gb());
    let plan = auto_plan::with_speed(plan, &ins, &st.spec, fx.vram_gb(), &fx.history());
    if !plan.ok {
        st.plan = Some(plan.clone());
        return Err(Error::new(plan.blockers.join(" ")));
    }
    st.stages = plan.stages.iter().cloned().map(StageState::new).collect();
    st.chosen = plan.chosen.clone();
    let try_list = plan.try_list.len();
    if let Some(c) = st.spec.cut.clone() {
        // chosen in Try first's search: the exact blocks are cut, nothing is tried
        st.drop_blocks = (!c.blocks.is_empty()).then(|| c.blocks.clone());
        st.notes.push(format!("The cut was chosen before the run{}: {}{}.", c.search_id.as_ref().map(|i| format!(" (search {i})")).unwrap_or_default(),
                              if c.blocks.is_empty() { "no blocks out".to_string() } else { format!("blocks {} out", c.blocks.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ")) },
                              c.intermediate.map(|i| format!(", MLP {i}")).unwrap_or_default()));
    }
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

/// The cuts Try first measured: what the search found best, else the planner's own list.
fn cands_of(st: &AutoState) -> Vec<Candidate> {
    st.try_cands.clone().or_else(|| st.plan.as_ref().map(|p| p.try_list.clone())).unwrap_or_default()
}

/// "Find the best cut": the search of the Try first card inside the run (docs/TRY_FIRST.md), for `search_minutes` minutes on the one target size. Its best cuts (diverse, at most
/// `TRY_TOP`) become the candidates of the thinking gate; the lowest loss is chosen here, the gate may choose another by thinking. A stop keeps what the worker measured and the
/// answer is asked again on Continue (every measured cut is kept, so nothing is measured twice).
fn step_try_search(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let plan = st.plan.clone().ok_or_else(|| Error::new("The plan is missing."))?;
    let ins = st.inspect.clone().ok_or_else(|| Error::new("The model's description is missing."))?;
    let o = st.original.clone().ok_or_else(|| Error::new("The model of this run is missing."))?;
    let minutes = st.spec.search_minutes.unwrap_or(20);
    let mut sspec = search::SearchSpec::new(&o.id, search::Target::B(st.spec.target_b));
    sspec.budget_min = minutes;
    sspec.keep_thinking = st.spec.keep_thinking;
    sspec.samples = Some(st.spec.try_samples());
    sspec.allow_many_blocks = st.spec.allow_many_blocks;
    sspec.think_check = "off".into();
    sspec.confirm_full = false;
    let sp = search::plan(&ins, &sspec, fx.vram_gb(), &fx.history())?;
    let analysis = match st.try_scan.as_ref().filter(|s| s["search"] == json!(true) && s["samples"] == json!(sp.samples)) {
        Some(s) => s["analysis"].clone(),
        None => {
            let mut t = search::try_spec(&ins, &sspec, &sp, fx.vram_gb())?;
            t.load = plan.measure_load.clone();
            let a = out!(fx.try_run(&t)?);
            st.try_scan = Some(json!({"samples": sp.samples, "search": true, "analysis": a}));
            a
        }
    };
    let keys: Vec<String> = analysis["search"]["finalists"]["0"].as_array().map(|a| a.iter().filter_map(|k| k.as_str().map(String::from)).take(auto_plan::TRY_TOP).collect()).unwrap_or_default();
    let found = search::rows_for_keys(&analysis, &ins, 0, &keys);
    if found.is_empty() {
        st.notes.push("The search measured no cut; the planner's first choice is cut.".into());
        return Ok(Flow::Done(Some("no answer: the first choice".into())));
    }
    let cands: Vec<Candidate> = found.iter().map(|r| auto_plan::candidate_at(&ins, r.k, r.intermediate, st.spec.target_b)).collect();
    let (h1, h2): (Vec<_>, Vec<_>) = (found.iter().map(|r| r.loss_h1).collect(), found.iter().map(|r| r.loss_h2).collect());
    let agree = argmin(&h1).is_none() || argmin(&h1) == argmin(&h2);
    let losses: Vec<Option<f64>> = found.iter().map(|r| Some(r.loss)).collect();
    let winner = argmin(&losses);
    let table: Vec<Value> = cands.iter().zip(&found).map(|(c, r)| json!({"blocks": c.blocks, "intermediate": c.intermediate, "params_b": c.params_b, "loss": r.loss, "loss_h1": r.loss_h1, "loss_h2": r.loss_h2, "cut_blocks": r.blocks})).collect();
    let stop_reason = analysis["search"]["stop_reason"].clone();
    st.try_result = Some(json!({"samples": sp.samples, "rows": table, "winner": winner, "halves_agree": agree,
                                "search": {"minutes": minutes, "stop_reason": stop_reason, "passes": analysis["passes"], "planned": sp.passes, "finalists": keys}}));
    if !agree {
        st.notes.push("The two halves of the measuring text name different winners among the best cuts; the thinking check (when it runs) decides between them.".into());
    }
    if stop_reason == json!("budget") {
        st.notes.push(format!("The search stopped when its {minutes} minutes were used (about {} cuts were planned); what it had measured was ranked.", sp.passes));
    }
    let w = winner.unwrap_or(0);
    let chosen = cands[w].clone();
    st.drop_blocks = (chosen.blocks > 0 && !found[w].blocks.is_empty()).then(|| found[w].blocks.clone());
    st.try_cands = Some(cands);
    choose(st, &ins, &chosen, fx.vram_gb());
    let lower = st.measure.as_ref().and_then(|m| m["loss"].as_f64()).zip(Some(found[w].loss));
    if let Some((orig, cut)) = lower.filter(|(o, c)| *c < *o * 0.98) {
        st.notes.push(format!("The best cut's text-prediction error ({cut:.3}) is lower than the original's ({orig:.3}): check that both were measured the same way. A lower error does not mean the model can still think."));
    }
    Ok(Flow::Done(Some(format!("{} blocks out of {} candidates, MLP {}: loss {:.3}", chosen.blocks, found.len(), chosen.intermediate.map(|i| i.to_string()).unwrap_or_else(|| "unchanged".into()), found[w].loss))))
}

/// Tries the candidates and picks the cheapest. Two requests: the widths and the best run of every block count, then (for a candidate with blocks AND a thinner MLP) the
/// combined cut with those blocks. When the two halves of the measuring text name different winners, everything is asked again with 256 samples.
fn step_try(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    if st.spec.search_minutes.is_some() {
        return step_try_search(st, fx);
    }
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
    choose(st, &ins, &chosen, fx.vram_gb());
    let lower = st.measure.as_ref().and_then(|m| m["loss"].as_f64()).zip(losses[w]);
    if let Some((orig, cut)) = lower.filter(|(o, c)| *c < *o * 0.98) {
        st.notes.push(format!("The best cut's text-prediction error ({cut:.3}) is lower than the original's ({orig:.3}): check that both were measured the same way. A lower error does not mean the model can still think."));
    }
    Ok(Flow::Done(Some(format!("{} blocks, MLP {}: loss {:.3}", chosen.blocks, chosen.intermediate.map(|i| i.to_string()).unwrap_or_else(|| "unchanged".into()), losses[w].unwrap_or(0.0)))))
}

/// Makes `chosen` the cut of the run: when it is not the planner's own choice, the stages are planned again for it and the stage steps rebuilt.
fn choose(st: &mut AutoState, ins: &Inspect, chosen: &Candidate, vram_gb: Option<f64>) {
    if st.chosen.as_ref() == Some(chosen) {
        return;
    }
    let steps = auto_plan::split_stages(ins, chosen, st.spec.stages);
    let stages = auto_plan::stage_plans(ins, &st.spec, &steps, vram_gb);
    st.stages = stages.iter().cloned().map(StageState::new).collect();
    if let Some(pl) = st.plan.as_mut() {
        pl.chosen = Some(chosen.clone());
        pl.stages = stages;
    }
    st.chosen = Some(chosen.clone());
    rebuild_stage_steps(st);
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

/// The thinking check before healing. The cuts Try first liked best (the `think_gate_n` lowest losses) are made one after the other as the FINAL cut, each checked for thinking
/// (the original once, from the first cut's folder) and deleted unless it is the best so far; the cut with the best thinking is chosen (`thinkcheck::rank_thinking`: the
/// thinking decides, the loss breaks a tie). When it is not the cut Try first chose, the stages are planned again for it. With one stage the winner's cut and its check are the
/// stage's own (the steps `s1.cut` and `s1.check_cut` are skipped); with two the winner's cut is deleted too. The gate NEVER halts a run: a cut that lost much of its thinking only
/// adds a warning (the healed model's check decides). Every call is looked at the state first, so Stop and Continue repeat nothing.
fn step_think_gate(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let ins = st.inspect.clone().ok_or_else(|| Error::new("The model's description is missing."))?;
    let cands = cands_of(st);
    let Some(tr) = st.try_result.clone() else { return Ok(Flow::Skipped("Try first gave no answer".into())) };
    let try_rows = tr["rows"].as_array().cloned().unwrap_or_default();
    let period = ins.period.max(1);
    let mut gate = st.think_gate.clone().unwrap_or_else(|| json!({"rows": [], "pending": null}));
    if gate.get("order").is_none() {
        let mut order: Vec<usize> = (0..cands.len().min(try_rows.len())).filter(|i| try_rows[*i]["loss"].as_f64().is_some()
            && (cands[*i].blocks == 0 || blocks_of(&try_rows[*i]["cut_blocks"]).len() as u32 == cands[*i].blocks)).collect();
        order.sort_by(|a, b| try_rows[*a]["loss"].as_f64().partial_cmp(&try_rows[*b]["loss"].as_f64()).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
        order.truncate(st.spec.think_gate_n as usize);
        if order.is_empty() {
            return Ok(Flow::Skipped("Try first measured no cut that can be checked".into()));
        }
        gate["order"] = json!(order);
    }
    let order: Vec<usize> = gate["order"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect()).unwrap_or_default();
    let rank = |gate: &Value| -> Option<usize> {
        let rows = gate["rows"].as_array().cloned().unwrap_or_default();
        let pairs: Vec<(Option<f64>, Option<f64>)> = rows.iter().map(|r| (r["loss"].as_f64(), r["score"].as_f64())).collect();
        thinkcheck::rank_thinking(&pairs)
    };
    for &i in &order {
        if gate["rows"].as_array().is_some_and(|a| a.iter().any(|r| r["row"] == json!(i))) {
            continue;
        }
        let cand = cands[i].clone();
        // 1. the final cut (the gate's own cut is never measured again: Try first measured it)
        let cut: CutRef = match serde_json::from_value::<CutRef>(gate["pending"]["cut"].clone()).ok().filter(|_| gate["pending"]["row"] == json!(i)) {
            Some(c) => c,
            None => {
                let steps = auto_plan::split_stages(&ins, &cand, 1);
                let stage = auto_plan::stage_plans(&ins, &st.spec, &steps, fx.vram_gb()).into_iter().next().ok_or_else(|| Error::new("The cut to check could not be planned."))?;
                let input = st.original.as_ref().map(|o| o.id.clone()).ok_or_else(|| Error::new("The model of this run is missing."))?;
                let mut spec = stage.reshape_spec(&input, st.reasoning(), CALIB_SAMPLES, WIDTH_SAMPLES);
                spec.load = st.plan.as_ref().map(|p| p.measure_load.clone()).unwrap_or_else(|| "auto".into());
                if cand.blocks > 0 {
                    spec.drop_layers = Some(layers_of(&blocks_of(&try_rows[i]["cut_blocks"]), period));
                }
                spec.skip_check = true;
                st.think_gate = Some(gate.clone());
                let r = out!(fx.cut(&spec)?);
                made(st, "cut", &r.name, Some(r.model_id.clone()), 0);
                gate["pending"] = json!({"row": i, "cut": r});
                st.think_gate = Some(gate.clone());
                r
            }
        };
        // 2. the original once, then this cut
        st.think_gate = Some(gate.clone());
        if st.baseline.is_none() {
            let r = out!(run_check(st, fx, "think_gate", &cut.name, "original")?);
            st.baseline = Some(r);
        }
        let r = out!(run_check(st, fx, "think_gate", &cut.name, "cut")?);
        let chk = check_json(&r, st.baseline.as_ref());
        let row = json!({"row": i, "blocks": cand.blocks, "intermediate": cand.intermediate, "params_b": cand.params_b, "loss": try_rows[i]["loss"], "cut": cut.name, "model_id": cut.model_id,
                         "cut_blocks": try_rows[i]["cut_blocks"], "score": thinkcheck::think_score(&chk["main"]), "check": chk, "removed": false});
        gate["rows"].as_array_mut().map(|a| a.push(row));
        gate["pending"] = Value::Null;
        st.think_gate = Some(gate.clone());
        // 3. only the best so far stays on the disk
        if !st.spec.keep_intermediate {
            let best = rank(&gate);
            let rows = gate["rows"].as_array().cloned().unwrap_or_default();
            for (k, r) in rows.iter().enumerate() {
                if Some(k) != best && r["removed"] != json!(true) {
                    remove_made(st, r["cut"].as_str().unwrap_or(""), fx)?;
                    gate["rows"][k]["removed"] = json!(true);
                    st.think_gate = Some(gate.clone());
                }
            }
        }
    }
    // 4. the choice
    let rows = gate["rows"].as_array().cloned().unwrap_or_default();
    let w = rank(&gate).ok_or_else(|| Error::new("The thinking check gave no answer."))?;
    let (win, ci) = (rows[w].clone(), rows[w]["row"].as_u64().unwrap_or(0) as usize);
    let chosen = cands[ci].clone();
    let try_winner = tr["winner"].as_u64().map(|n| n as usize);
    let fmt = |r: &Value| format!("closed {:.0} %, loops {:.0} %", r["check"]["main"]["closed_rate"].as_f64().unwrap_or(0.0) * 100.0, r["check"]["main"]["loop_rate"].as_f64().unwrap_or(0.0) * 100.0);
    let name_of = |c: &Candidate| format!("{} blocks, MLP {}", c.blocks, c.intermediate.map(|x| x.to_string()).unwrap_or_else(|| "unchanged".into()));
    gate["winner"] = json!(ci);
    if try_winner.is_some_and(|tw| tw != ci) {
        let tw = try_winner.unwrap_or(0);
        let tw_row = rows.iter().find(|r| r["row"] == json!(tw));
        st.notes.push(format!("Try first's lowest loss was {} ({}), but it kept less of its thinking ({}); {} was chosen ({}).",
                              try_rows[tw]["loss"].as_f64().map(|l| format!("{l:.3}")).unwrap_or_else(|| "?".into()), name_of(&cands[tw]), tw_row.map(&fmt).unwrap_or_else(|| "not checked".into()), name_of(&chosen), fmt(&win)));
    }
    if let Some(base) = st.baseline.as_ref().filter(|b| thinkcheck::lost_much(&b["main"], &win["check"]["main"])) {
        gate["lost_much"] = json!(true);
        st.notes.push(format!("Even the best cut has lost much of its ability to finish its reasoning before healing (closed {:.0} % against {:.0} %). Healing may bring some back; the healed model's check decides.",
                              win["check"]["main"]["closed_rate"].as_f64().unwrap_or(0.0) * 100.0, base["main"]["closed_rate"].as_f64().unwrap_or(0.0) * 100.0));
    }
    st.drop_blocks = (chosen.blocks > 0).then(|| blocks_of(&win["cut_blocks"])).filter(|b| !b.is_empty());
    choose(st, &ins, &chosen, fx.vram_gb());
    let kept = win["removed"] != json!(true);
    if st.stages.len() == 1 && kept {
        // the stage's own cut and check: its steps are skipped
        let cut = CutRef { name: win["cut"].as_str().unwrap_or("").into(), model_id: win["model_id"].as_str().unwrap_or("").into() };
        if let Some(m) = st.made.iter_mut().find(|m| m.kind == "cut" && m.name == cut.name) {
            m.stage = 1;
        }
        st.stages[0].cut = Some(cut);
        st.stages[0].cut_check = Some(win["check"].clone());
        st.stages[0].cut_loss = win["loss"].as_f64();
        for id in ["s1.cut", "s1.check_cut"] {
            mark(st, id, "skipped", Some("made and checked in the thinking gate".into()));
        }
    } else if kept && !st.spec.keep_intermediate {
        // two stages: the final cut was only checked; stage 1 makes its own (half) cut
        remove_made(st, win["cut"].as_str().unwrap_or(""), fx)?;
        gate["rows"][w]["removed"] = json!(true);
    }
    st.think_gate = Some(gate);
    Ok(Flow::Done(Some(format!("{}: {}", name_of(&chosen), fmt(&win)))))
}

/// Deletes a model Auto-cut made (by its folder name) and marks it removed in the state.
fn remove_made(st: &mut AutoState, name: &str, fx: &mut dyn Effects) -> Result<()> {
    if let Some(i) = st.made.iter().position(|m| m.name == name && !m.removed) {
        let m = st.made[i].clone();
        fx.remove(&m)?;
        st.made[i].removed = true;
    }
    Ok(())
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
    // a stage 1 that makes exactly the cut Try first measured does not measure it again (that would load the new model, and for blocks only the original too)
    if idx == 0 {
        if let Some(loss) = tried_loss(st, &plan) {
            spec.skip_check = true;
            st.stages[idx].cut_loss = Some(loss);
        }
    }
    let r = out!(fx.cut(&spec)?);
    made(st, "cut", &r.name, Some(r.model_id.clone()), idx as u32 + 1);
    st.stages[idx].cut = Some(r);
    Ok(Flow::Done(None))
}

/// The loss Try first measured for the cut this stage makes: only when the stage's cut IS the candidate that won (the same blocks and MLP width) and the run's cut
/// uses the layers Try first found (a plain width cut, or blocks with their saved choice).
fn tried_loss(st: &AutoState, plan: &StagePlan) -> Option<f64> {
    let chosen = st.chosen.as_ref()?;
    if plan.blocks_cut != chosen.blocks || plan.width != chosen.intermediate.filter(|i| Some(*i) != st.inspect.as_ref().and_then(|ins| auto_plan::full_width(ins))) {
        return None;
    }
    if chosen.blocks > 0 && st.drop_blocks.as_ref().is_none_or(|b| b.len() as u32 != chosen.blocks) {
        return None;
    }
    let tr = st.try_result.as_ref()?;
    let w = tr["winner"].as_u64()? as usize;
    tr["rows"].get(w)?["loss"].as_f64()
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
        let r = out!(run_check(st, fx, &format!("s{}.check_cut", idx + 1), &cut.name, "original")?);
        st.baseline = Some(r);
    }
    let r = out!(run_check(st, fx, &format!("s{}.check_cut", idx + 1), &cut.name, "cut")?);
    st.stages[idx].cut_check = Some(check_json(&r, st.baseline.as_ref()));
    Ok(Flow::Done(None))
}

fn step_heal(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    if st.stages[idx].heal.is_some() {
        return Ok(Flow::Skipped("already healed".into()));
    }
    let teacher = st.original.as_ref().map(|o| o.repo.clone()).ok_or_else(|| Error::new("The original model is missing."))?;
    // the second try heals the healed model of the first one again; the teacher stays the original
    let student = match st.stages[idx].retry_from.clone() {
        Some(prev) => Student::Folder(prev.dir),
        None => Student::Made(st.stages[idx].cut.clone().ok_or_else(|| Error::new("There is no cut to heal."))?.name),
    };
    let mut spec = st.stages[idx].plan.heal_spec(st.keep_thinking());
    let rounds = rounds_for(st, idx, fx.now_secs());
    let student = if rounds.is_some() { Student::Made(st.stages[idx].cut.clone().ok_or_else(|| Error::new("There is no cut to heal."))?.name) } else { student };
    match rounds {
        // in rounds the stage goes on from its last round (the same run): the cap is higher, the student is still the cut
        Some(r) => spec.rounds = Some(r),
        None => spec.tokens_m += st.stages[idx].extra_tokens_m,
    }
    let heal_key = format!("s{}.heal:cache", idx + 1);
    let via_vllm = spec.rounds.is_none() && vllm_decision(st, &heal_key).is_some_and(|d| d.use_vllm);
    if via_vllm {
        spec.vllm = Some(super::runs::HealVllm { teacher_cache: true });
    }
    let h = out!(fx.heal(&student, &teacher, &spec)?);
    if via_vllm {
        let used = h.summary.as_ref().and_then(|s| s.get("teacher_engine")).filter(|t| t.is_object()).cloned();
        let rec = match used {
            Some(mut u) => {
                let fell = u["fallback"] == true || u["used"] == false;
                u["engine_version"] = if fell { Value::Null } else { u["engine"].clone() };          // "vllm-0.34.0"
                u["engine"] = json!(if fell { "hf" } else { "vllm" });
                u["fallback"] = json!(fell);
                u
            }
            None => json!({"engine": "hf", "fallback": true, "note": "the heal did not report which engine made the answers"}),
        };
        record_engine(st, &heal_key, "teacher_cache", Some(&rec));
    }
    made(st, "healed", &h.name, Some(h.model_id.clone()), idx as u32 + 1);
    if let Some(r) = h.summary.as_ref().and_then(|s| s.get("rounds")).filter(|r| r.is_object()) {
        st.stages[idx].rounds.push(r.clone());
        if let Some(note) = r["notes"].as_array() {
            for n in note.iter().filter_map(|n| n.as_str()) {
                if !st.notes.iter().any(|x| x == n) {
                    st.notes.push(format!("Stage {}: {n}", idx + 1));
                }
            }
        }
    }
    st.stages[idx].heal = Some(h);
    Ok(Flow::Done(None))
}

/// The settings of a heal in rounds for stage `idx` (None without `heal_rounds`): the cap is the stage's heal budget plus what an extension added, the original's numbers are passed in
/// (it is checked once per run), the merged model is checked in the run's own mode, and with a time limit the heal must end early enough for the steps after it.
fn rounds_for(st: &AutoState, idx: usize, now_secs: i64) -> Option<heal_rounds::RoundSpec> {
    let o = st.spec.heal_rounds.as_ref()?;
    let sg = &st.stages[idx];
    let cap = sg.plan.tokens_m + sg.extra_tokens_m;
    let mode = if st.spec.think_quick() { "quick" } else { "full" };
    let baseline = st.baseline.as_ref().filter(|b| b["main"].is_object()).map(|b| json!({ mode: b["main"].clone() }));
    let deadline = st.spec.max_hours.and_then(|max| {
        let started = chrono::NaiveDateTime::parse_from_str(&st.created, "%Y-%m-%dT%H:%M:%SZ").ok()?.and_utc().timestamp();
        let t = st.plan.as_ref()?.time.as_ref()?;
        let heal = format!("s{}.heal", idx + 1);
        // what has to run after this heal: the high end of the steps that are not done (this heal is the one in work)
        let after: f64 = st.steps.iter().filter(|s| s.status != "done" && s.status != "skipped" && s.id != heal).filter_map(|s| t.steps.iter().find(|x| x.id == s.id)).map(|x| x.minutes.1).sum();
        let _ = now_secs;
        Some((started as f64 + max * 3600.0 - after * 60.0).max(0.0) as u64)
    });
    Some(heal_rounds::RoundSpec {
        round_m: o.round_m, max_tokens_m: Some(cap), first_check_m: Some(o.first_check(sg.plan.tokens_m).min(cap)), check: if o.quick() { "quick".into() } else { "full".into() },
        final_quick: Some(st.spec.think_quick()), stop: "pass".into(), plateau: true, baseline, deadline, extend: sg.extra_tokens_m > 0 && sg.attempt == 1, round_exact: None,
    })
}

fn failed_names(chk: &Value) -> Vec<String> {
    chk["gates"]["failed"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn step_check_heal(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    let h = st.stages[idx].heal.clone().ok_or_else(|| Error::new("There is no healed model to check."))?;
    if st.baseline.is_none() {
        // no check of the cut (and none before healing) came first: the original is checked here, once, so that the gates have something to compare with
        let r = out!(run_check(st, fx, &format!("s{}.check_heal", idx + 1), &h.name, "original")?);
        st.baseline = Some(r);
    }
    let r = out!(run_check(st, fx, &format!("s{}.check_heal", idx + 1), &h.name, "healed")?);
    let chk = check_json(&r, st.baseline.as_ref());
    let verdict = chk["verdict"].as_str().unwrap_or("pass").to_string();
    let attempt = st.stages[idx].attempt;
    let kind = ["heal", "heal_again", "back_off"].get(attempt as usize).copied().unwrap_or("back_off");
    st.stages[idx].attempts.push(json!({"attempt": attempt, "kind": kind, "model": h.name, "tokens_m": h.tokens_m, "verdict": verdict, "failed": failed_names(&chk), "main": chk["main"]}));
    st.stages[idx].heal_check = Some(chk.clone());
    if verdict == "fail" {
        return gate_failed(st, idx, &failed_names(&chk), fx);
    }
    st.stages[idx].passed = true;
    cleanup_after(st, idx, fx)?;
    Ok(Flow::Done(Some(verdict)))
}

/// The healed model of a stage missed a gate. Once: heal it again (the healed model is the student, the original still the teacher) with `RETRY_TOKENS_M` million more tokens.
/// (Without `retry_heal` the first failure goes straight to the back-off; without `back_off` the run ends.) Twice: back the cut off one step (`auto_plan::back_off`) and make the stage again. A third time: the run ends with `gates_failed` and its report. Nothing is deleted on the way:
/// what a failed try made is cleaned up when the stage has passed (and never with `keep_intermediate`).
fn gate_failed(st: &mut AutoState, idx: usize, failed: &[String], fx: &mut dyn Effects) -> Result<Flow> {
    let n = idx + 1;
    let names = failed.join(", ");
    let ids = |what: &[&str]| -> Vec<String> { what.iter().map(|w| format!("s{n}.{w}")).collect() };
    // a stage healed in rounds says why it ended: the cap (one more stretch of rounds), a plateau (more training would not help: back off), the time limit (the run ends)
    let stop_reason = st.stages[idx].rounds.last().and_then(|r| r["stop"]["reason"].as_str().map(String::from));
    let in_rounds = st.spec.heal_rounds.is_some() && stop_reason.is_some();
    // with the retry switched off the first failure goes straight to backing the cut off
    let attempt = match st.stages[idx].attempt {
        0 if !st.spec.retry_heal => 1,
        0 if in_rounds && stop_reason.as_deref() == Some("plateau") => {
            st.stages[idx].attempt = 1;
            1
        }
        a => a,
    };
    if in_rounds && stop_reason.as_deref() == Some("time") {
        let why = "The time limit left no room for more rounds, so the stage ended with what it had.";
        st.notes.push(why.into());
        st.error = Some(format!("The thinking check of the healed model of stage {n} failed ({names}); {why} The run stops here with its report."));
        return Ok(Flow::Halt("gates_failed".into()));
    }
    // "I have at most N hours": a retry or a stage made again that would end after the limit is not started
    if attempt <= 1 && (attempt == 0 || st.spec.back_off) {
        if let Some(why) = too_late(st, idx, attempt == 0, fx.now_secs()) {
            st.notes.push(why.clone());
            st.error = Some(format!("The thinking check of the healed model of stage {n} failed ({names}); {why} The run stops here with its report."));
            return Ok(Flow::Halt("gates_failed".into()));
        }
    }
    match attempt {
        0 if in_rounds => {
            // the stage reached its cap without a pass: ONE more stretch of rounds on the same run (the same student, the same answers of the original), not a second heal from scratch
            let more = auto_plan::RETRY_TOKENS_M;
            let sg = &mut st.stages[idx];
            sg.heal = None;
            sg.extra_tokens_m = more;
            sg.attempt = 1;
            let note = format!("Stage {n} missed its thinking check ({names}) after {} million tokens: healing goes on in rounds for {more} million more (the same run, with the same answers of the original).", sg.plan.tokens_m);
            st.notes.push(note.clone());
            reopen(st, &ids(&["heal", "check_heal"]));
            return Ok(Flow::Retry(note));
        }
        0 => {
            let stage_m = st.stages[idx].plan.tokens_m;
            let total_m = auto_plan::retry_total_m(stage_m);
            let sg = &mut st.stages[idx];
            sg.retry_from = sg.heal.take();
            sg.extra_tokens_m = total_m.saturating_sub(stage_m);
            sg.attempt = 1;
            let note = format!("Stage {n} missed its thinking check ({names}): it is healed again on the healed model, which learns on {total_m} million tokens in this second heal ({} million more of the original's answers).", total_m.saturating_sub(stage_m));
            st.notes.push(note.clone());
            reopen(st, &ids(&["heal", "check_heal"]));
            return Ok(Flow::Retry(note));
        }
        1 if st.spec.back_off => match back_off_stage(st, idx, fx.vram_gb()) {
            Ok(note) => {
                st.notes.push(note.clone());
                let all: Vec<&str> = if st.keep_thinking() { vec!["cut", "check_cut", "heal", "check_heal"] } else { vec!["cut", "heal"] };
                reopen(st, &ids(&all));
                return Ok(Flow::Retry(note));
            }
            Err(why) => st.notes.push(why),
        },
        1 => st.notes.push(format!("Stage {n} missed its thinking check ({names}) and \"cut one step less\" is switched off: the run ends here.")),
        _ => {}
    }
    st.error = Some(format!("The thinking check of the healed model of stage {n} failed ({names}) after {} tries; the run stops here. The report says what was tried.", st.stages[idx].attempts.len()));
    Ok(Flow::Halt("gates_failed".into()))
}

/// With a time limit: would the next try of stage `idx` (the retry heal when `retry`, else the stage made again) end after it? The cost is the high end of the estimate of that try
/// plus the steps that are still to run (the plan's estimate), the time left is the limit minus the time since the run was created. None when there is no limit or it is kept.
fn too_late(st: &AutoState, idx: usize, retry: bool, now_secs: i64) -> Option<String> {
    let max = st.spec.max_hours?;
    let t = st.plan.as_ref()?.time.as_ref()?;
    let started = chrono::NaiveDateTime::parse_from_str(&st.created, "%Y-%m-%dT%H:%M:%SZ").ok()?.and_utc().timestamp();
    let left_h = max - (now_secs - started).max(0) as f64 / 3600.0;
    let n = idx + 1;
    let hi = |id: &str| t.steps.iter().find(|x| x.id == id).map(|x| x.minutes.1).unwrap_or(0.0);
    let mine = |what: &str| format!("s{n}.{what}");
    let cost = if retry {
        // the last stage's retry has its own line in the estimate; another stage's is its heal at the retry's budget and its check
        match t.extra.iter().find(|e| e.id == "retry_heal").filter(|_| idx + 1 == st.stages.len()) {
            Some(e) => e.minutes.1,
            None => hi(&mine("heal")) * auto_plan::retry_total_m(st.stages[idx].plan.tokens_m) as f64 / st.stages[idx].plan.tokens_m.max(1) as f64 + hi(&mine("check_heal")),
        }
    } else {
        ["cut", "check_cut", "heal", "check_heal"].iter().map(|w| hi(&mine(w))).sum()
    };
    // what has not run yet and does not belong to this stage
    let prefix = format!("s{n}.");
    let rest: f64 = st.steps.iter().filter(|s| s.status != "done" && s.status != "skipped" && !s.id.starts_with(&prefix)).map(|s| hi(&s.id)).sum();
    let need_h = (cost + rest) / 60.0;
    (need_h > left_h).then(|| format!("{} would end after your time limit of {max:.0} hours (it needs about {need_h:.0} h more, {:.0} h are left).", if retry { "A retry heal" } else { "Making the stage again with a smaller cut" }, left_h.max(0.0)))
}

/// Backs the cut of stage `idx` off by one step and makes a fresh stage of it (the stages after it are planned again from the new stop). Err(why) when nothing is left to give back,
/// or the card cannot heal the larger model.
fn back_off_stage(st: &mut AutoState, idx: usize, vram_gb: Option<f64>) -> std::result::Result<String, String> {
    let ins = st.inspect.clone().ok_or_else(|| "The model's description is missing, so the cut cannot be backed off.".to_string())?;
    let cur = st.stages[idx].plan.clone();
    let prev = match idx.checked_sub(1) {
        None => auto_plan::Step { blocks: 0, intermediate: auto_plan::full_width(&ins) },
        Some(i) => auto_plan::Step { blocks: st.stages[i].plan.blocks_after, intermediate: st.stages[i].plan.intermediate_after },
    };
    // a stage that cuts blocks gives a block back; a stage that only thins the MLP gives MLP width back (the blocks of the stages before stay cut)
    let (b, i) = if cur.blocks_cut > 0 {
        auto_plan::back_off(&ins, cur.blocks_after, cur.intermediate_after)
    } else {
        auto_plan::back_off(&ins, 0, cur.intermediate_after).map(|(_, i)| (cur.blocks_after, i))
    }.ok_or_else(|| format!("Stage {} cannot be backed off any further.", idx + 1))?;
    let i = match (i, prev.intermediate) {
        (Some(a), Some(p)) if a > p => Some(p),
        _ => i,
    };
    let new = auto_plan::Step { blocks: b, intermediate: i };
    if new == prev {
        return Err(format!("Backing off stage {} would leave nothing to cut in it.", idx + 1));
    }
    let mut steps: Vec<auto_plan::Step> = st.stages.iter().map(|s| auto_plan::Step { blocks: s.plan.blocks_after, intermediate: s.plan.intermediate_after }).collect();
    steps[idx] = new;
    let plans = auto_plan::stage_plans(&ins, &st.spec, &steps, vram_gb);
    if plans[idx].verdict == "over" {
        return Err(format!("Backing off stage {} makes a model ({:.1} B) the card cannot heal (about {:.0} GB needed).", idx + 1, plans[idx].params_b, plans[idx].heal_gb));
    }
    let attempts = std::mem::take(&mut st.stages[idx].attempts);
    for j in idx..st.stages.len() {
        st.stages[j] = StageState::new(plans[j].clone());
        if let Some(pl) = st.plan.as_mut() {
            if let Some(slot) = pl.stages.get_mut(j) {
                *slot = plans[j].clone();
            }
        }
    }
    st.stages[idx].attempts = attempts;
    st.stages[idx].attempt = 2;
    let last = idx + 1 == st.stages.len();
    if last {
        if let Some(c) = st.chosen.as_mut() {
            c.blocks = b;
            c.intermediate = i;
            c.params = plans[idx].params;
            c.params_b = plans[idx].params_b;
            c.layers = plans[idx].layers_after;
        }
    }
    let (from, to) = (cur.params_b, plans[idx].params_b);
    Ok(format!("Stage {} missed its thinking check twice: the cut is backed off one step ({} blocks, MLP {}), so stage {} makes a {:.1} B model instead of {:.1} B.{}",
               idx + 1, b, i.map(|x| x.to_string()).unwrap_or_else(|| "unchanged".into()), idx + 1, to, from, if last { " The final size is larger than asked." } else { "" }))
}

/// Without `keep_intermediate`: once stage `idx` has passed its check, what its tries made goes (the unhealed cut, a healed model that was healed again or whose stage was made
/// again), and the healed model of the stage before (the next stage's input has been made). Only what Auto-cut made is ever removed, and never the healed model of the
/// last stage nor the one that passed.
fn cleanup_after(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<()> {
    if st.spec.keep_intermediate {
        return Ok(());
    }
    let stage = idx as u32 + 1;
    let keep = st.stages[idx].heal.as_ref().map(|h| h.name.clone());
    let victims: Vec<usize> = st.made.iter().enumerate().filter(|(_, m)| !m.removed && (
        (m.kind == "cut" && m.stage == stage) || (m.kind == "healed" && m.stage == stage && Some(&m.name) != keep.as_ref()) || (m.kind == "healed" && m.stage + 1 == stage)
    )).map(|(i, _)| i).collect();
    for i in victims {
        let m = st.made[i].clone();
        fx.remove(&m)?;
        st.made[i].removed = true;
    }
    // the teacher is not needed again once the last stage has passed
    if idx + 1 >= st.stages.len() {
        let names: Vec<String> = st.made.iter().map(|m| m.name.clone()).collect();
        fx.drop_teacher_cache(&names)?;
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
    let final_params = st.stages.last().map(|s| s.plan.params).unwrap_or(0);
    let quants = st.spec.export_quants_for(final_params);
    let run = st.id.clone();
    fx.export_options(st.spec.imatrix);
    let r = match st.spec.target_gb {
        Some(t) => out!(fx.export_measured(&run, &h, &family, &quants, st.keep_thinking(), t)?),
        None => out!(fx.export(&run, &h, &family, &quants, st.keep_thinking())?),
    };
    // the 16-bit GGUF a target size asked for is only the reference of the comparison: it is deleted after it (unless the person chose it or keeps the intermediate files)
    if st.spec.target_gb.is_some() && !st.spec.quants.iter().any(|q| q == "f16") {
        if let Some(name) = r["ggufs"].as_array().and_then(|a| a.iter().find(|g| g["quant"] == "f16")).and_then(|g| g["export"].as_str()) {
            st.made.push(Made { kind: "export".into(), name: name.to_string(), model_id: None, stage: 0, removed: false });
        }
    }
    st.export = Some(r);
    Ok(Flow::Done(None))
}

/// The 16-bit file of an export, the reference of the comparison.
fn f16_of(export: &Value) -> Option<String> {
    export["ggufs"].as_array()?.iter().find(|g| g["quant"] == "f16").and_then(|g| g["path"].as_str()).map(String::from)
}

/// The label, group and matrix words of a file of the export, as Compare quants shows them.
fn pick_of(g: &Value, matrix: bool) -> Value {
    let q = g["quant"].as_str().unwrap_or("");
    if q == "mixed_measured" {
        return json!({"path": g["path"], "label": "Mixed: measured fit to a size", "group": "Mixed: measured fit to a size", "matrix": matrix, "mixed": "mixed_measured"});
    }
    let name = q.to_uppercase();
    json!({"path": g["path"], "label": if matrix { format!("{name}, with matrix") } else { name.clone() }, "group": name, "matrix": if matches!(q, "q8_0" | "f16") { Value::Null } else { json!(matrix) }, "mixed": null})
}

/// Did the measured file beat the plain file of its size beyond the noise? (the promotion of Compare quants, read from its results)
fn promoted_of(result: &Value) -> bool {
    let rows = result["rows"].as_array().cloned().unwrap_or_default();
    result["promotion"].as_array().map(|a| a.iter().any(|p| p["promoted"] == true && rows.iter().any(|r| r["id"] == p["id"] && r["mixed"] == "mixed_measured"))).unwrap_or(false)
}

fn step_compare_quants(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    if st.compare.is_some() {
        return Ok(Flow::Skipped("already compared".into()));
    }
    let export = st.export.clone().ok_or_else(|| Error::new("There is nothing to compare: the export is missing."))?;
    let reference = f16_of(&export).ok_or_else(|| Error::new("The export made no 16-bit file to compare with."))?;
    let picks: Vec<Value> = export["ggufs"].as_array().map(|a| a.iter().filter(|g| g["quant"] != "f16").map(|g| pick_of(g, true)).collect()).unwrap_or_default();
    // a comparison that cannot be done must not lose the files that were made: it is noted, the run goes on, and the measured file is not published
    let res = match fx.compare_quants(&reference, &picks) {
        Ok(Out::Done(v)) => v,
        Ok(Out::Stopped) => return Ok(Flow::Stopped),
        Err(e) if e.is_cancelled() => return Ok(Flow::Stopped),
        Err(e) => json!({"skipped": format!("The comparison could not be made ({}). The measured file was made but is not published; the 16-bit file is kept so that Compare quants can be run by hand.", e.0)}),
    };
    let skipped = res["skipped"].as_str().map(String::from);
    let promoted = skipped.is_none() && promoted_of(&res);
    let verdict = res["verdict"].clone();
    st.compare = Some(json!({"promoted": promoted, "promotion": res["promotion"], "verdict": verdict, "rows": res["rows"], "skipped": skipped, "comparison": res["entry"]["id"]}));
    // the reference is not needed any more
    if skipped.is_none() && !st.spec.keep_intermediate {
        let victims: Vec<String> = st.made.iter().filter(|m| m.kind == "export" && !m.removed).map(|m| m.name.clone()).collect();
        for n in victims {
            remove_made(st, &n, fx)?;
        }
    }
    let note = match (&skipped, promoted) {
        (Some(s), _) => Some(s.clone()),
        (None, true) => Some("the measured file is better than the plain file of its size: it is published".to_string()),
        (None, false) => Some("the measured file is not better than the plain file of its size beyond the noise: it is kept but not published".to_string()),
    };
    Ok(Flow::Done(note))
}

/// The Q4_K_M file (or the first one) of an export.
fn gguf_of(export: &Value) -> Option<String> {
    let list = export["ggufs"].as_array()?;
    list.iter().find(|g| g["quant"] == "q4_k_m").or_else(|| list.first()).and_then(|g| g["path"].as_str()).map(String::from)
}

fn step_check_gguf(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let h = final_heal(st)?;
    let file = st.export.as_ref().and_then(gguf_of).ok_or_else(|| Error::new("The export made no GGUF file to check."))?;
    let r = out!(fx.think_check(&h.name, "gguf", st.spec.think_quick(), Some(&file))?);
    let chk = check_json(&r, st.baseline.as_ref());
    let verdict = chk["verdict"].as_str().unwrap_or("pass").to_string();
    let failed = failed_names(&chk);
    st.gguf_check = Some(chk);
    if verdict == "fail" {
        st.error = Some(format!("The thinking check of the GGUF file failed ({}) although the healed model passed: the export or the quantisation lost it. The files are kept, nothing was published.", failed.join(", ")));
        return Ok(Flow::Halt("gates_failed".into()));
    }
    Ok(Flow::Done(Some(verdict)))
}

fn step_publish(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let export = st.export.clone().ok_or_else(|| Error::new("There is nothing to publish: the export is missing."))?;
    // the folder is named after the model and the size it was cut to, so it is never taken for the original
    let size = st.stages.last().map(|s| s.plan.params_b).unwrap_or(st.spec.target_b);
    let name = format!("{} cut to {size:.1}B", st.original.as_ref().map(|o| o.name.clone()).unwrap_or_default());
    // the files: not the 16-bit reference of a target-size run (unless the person asked for it), and the measured file only when the comparison promoted it
    let export = {
        let mut e = export;
        let (keep_f16, measured_ok) = (st.spec.quants.iter().any(|q| q == "f16"), st.compare.as_ref().is_some_and(|c| c["promoted"] == true));
        if st.spec.target_gb.is_some() {
            if let Some(a) = e["ggufs"].as_array_mut() {
                a.retain(|g| (keep_f16 || g["quant"] != "f16") && (measured_ok || g["quant"] != "mixed_measured"));
            }
        }
        e
    };
    let r = out!(fx.publish(&export, &name)?);
    let note = if r["published"] == true { None } else { r["why"].as_str().map(String::from) };
    st.publish = Some(r);
    Ok(Flow::Done(note))
}

/// The plan the window shows before a run: `spec` and the description of the model the Slim page already has (`Inspect`), the card and the disk. No job runs.
pub fn preview(ins: &Inspect, spec: &AutoSpec, vram_gb: Option<f64>, free_gb: Option<f64>, has_training_data: bool) -> Value {
    let history = crate::vllm::history_with_measured(crate::timing::History::recent(HISTORY_LOGS), None);
    // with a time limit the plan is the one of the settings the run will use (what start would do), with the list of what was given up
    let fit = spec.max_hours.map(|h| auto_speed::fit_budget(ins, spec, vram_gb, &history, h));
    let used = fit.as_ref().filter(|f| f.fits).map(|f| f.spec.clone()).unwrap_or_else(|| spec.clone());
    let mut plan = auto_plan::with_speed(auto_plan::plan(ins, &used, vram_gb, free_gb), ins, &used, vram_gb, &history);
    if let Some(f) = fit {
        if !f.fits {
            plan.ok = false;
            plan.blockers.extend(f.message.clone());
        }
        plan.warnings.splice(0..0, auto_speed::budget_sentences(&f));
        plan.budget = Some(f);
    }
    if !has_training_data {
        plan.ok = false;
        plan.blockers.push("Save a training-data version in Step 2 first: the layers are measured and healed on your world's text.".into());
    }
    json!({"plan": plan})
}

/// A run (the newest when `id` is None) and its report as data and as Markdown, made from the saved state.
pub fn report_of(p: &Project, id: Option<&str>) -> Result<Value> {
    let st = match id {
        Some(i) => load(&run_dir(p, i)),
        None => latest(p),
    }
    .ok_or_else(|| Error::new("There is no Auto-cut run yet."))?;
    let mut r = auto_report::build(&st);
    auto_report::add_timings(&mut r, &st);
    Ok(json!({"run": st.id, "report": r, "markdown": auto_report::markdown(&r)}))
}
