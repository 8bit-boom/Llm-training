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

use super::auto_plan::{self, AutoPlan, AutoSpec, Candidate, StagePlan};
use super::plan::{Inspect, ReasoningCalib, ReshapeSpec};
use super::runs::{models_dir, now, read_json, write_json, HealSpec, TrySpec};
use super::auto_report;
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
pub const RETRY_TOKENS_M: u32 = 40;

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
    /// The Stop button was pressed.
    fn stopped(&self) -> bool;
    fn model(&self, id: &str) -> Result<ModelRef>;
    fn has_training_data(&self) -> bool;
    fn vram_gb(&self) -> Option<f64>;
    /// free space on the disk the models are written to
    fn free_disk_gb(&self) -> Option<f64>;
    /// what this computer's saved job logs say about its speed (the time estimate); nothing by default
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
    /// Heals `student` with `teacher_repo` (the original model) as the teacher.
    fn heal(&mut self, student: &Student, teacher_repo: &str, spec: &HealSpec) -> Result<Out<HealRef>>;
    /// Deletes what Auto-cut made (only ever something in the state's list of `Made`).
    fn remove(&mut self, what: &Made) -> Result<()>;
    /// The last stage has passed: the teacher's shared answers that only the models named here use are not needed any more. Nothing to do for an effects object that has none.
    fn drop_teacher_cache(&mut self, _models: &[String]) -> Result<()> {
        Ok(())
    }
    /// The GGUF files of the healed model: `{"dir", "ggufs": [{"quant", "path"}]}`.
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
}

impl StageState {
    pub fn new(plan: StagePlan) -> Self {
        StageState { plan, cut: None, heal: None, cut_check: None, heal_check: None, attempt: 0, extra_tokens_m: 0, passed: false, cut_loss: None, attempts: Vec::new(), retry_from: None }
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
}

impl AutoState {
    fn new(id: &str, spec: AutoSpec) -> Self {
        AutoState {
            format: FORMAT, id: id.into(), interrupted: false, spec, status: "running".into(), step: "plan".into(), error: None, created: now(), updated: now(), original: None, inspect: None, plan: None,
            steps: vec![StepRec::new("plan", "Check the model, the card and the disk")], chosen: None, drop_blocks: None, try_result: None, try_scan: None, measure: None, baseline: None,
            stages: Vec::new(), made: Vec::new(), export: None, gguf_check: None, publish: None, report: None, notes: Vec::new(), think_gate: None, compare: None,
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
        if spec.keep_thinking && spec.think_gate {
            v.push(StepRec::new("think_gate", "Check the thinking of the best cuts before healing"));
        }
    }
    for n in 1..=stages {
        v.extend(stage_steps(n, stages, spec.keep_thinking));
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
            finish(dir, &mut st);
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
            Ok(Flow::Retry(note)) => {
                fx.emit(event(&st, &id, "retry", Some(&note)));
                mark(&mut st, &id, "pending", Some(note));
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

// ------------------------------------------------------------------------------------------------------------------ the steps

fn step_plan(st: &mut AutoState, fx: &mut dyn Effects) -> Result<Flow> {
    let model = st.original.clone().ok_or_else(|| Error::new("The model of this run is missing."))?;
    if !fx.has_training_data() {
        return Err(Error::new("Save a training-data version in Step 2 first: the layers are measured and healed on your world's text."));
    }
    let ins = out!(fx.inspect(&model)?);
    let plan = auto_plan::plan(&ins, &st.spec, fx.vram_gb(), fx.free_disk_gb());
    let plan = auto_plan::with_history(plan, &ins, &st.spec, &fx.history());
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
    let plan = st.plan.clone().ok_or_else(|| Error::new("The plan is missing."))?;
    let cands = plan.try_list.clone();
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
            let r = out!(fx.think_check(&cut.name, "original", st.spec.quick, None)?);
            st.baseline = Some(r);
        }
        let r = out!(fx.think_check(&cut.name, "cut", st.spec.quick, None)?);
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
    let teacher = st.original.as_ref().map(|o| o.repo.clone()).ok_or_else(|| Error::new("The original model is missing."))?;
    // the second try heals the healed model of the first one again; the teacher stays the original
    let student = match st.stages[idx].retry_from.clone() {
        Some(prev) => Student::Folder(prev.dir),
        None => Student::Made(st.stages[idx].cut.clone().ok_or_else(|| Error::new("There is no cut to heal."))?.name),
    };
    let mut spec = st.stages[idx].plan.heal_spec(st.keep_thinking());
    spec.tokens_m += st.stages[idx].extra_tokens_m;
    let h = out!(fx.heal(&student, &teacher, &spec)?);
    made(st, "healed", &h.name, Some(h.model_id.clone()), idx as u32 + 1);
    st.stages[idx].heal = Some(h);
    Ok(Flow::Done(None))
}

fn failed_names(chk: &Value) -> Vec<String> {
    chk["gates"]["failed"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn step_check_heal(st: &mut AutoState, idx: usize, fx: &mut dyn Effects) -> Result<Flow> {
    let h = st.stages[idx].heal.clone().ok_or_else(|| Error::new("There is no healed model to check."))?;
    let r = out!(fx.think_check(&h.name, "healed", st.spec.quick, None)?);
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
/// Twice: back the cut off one step (`auto_plan::back_off`) and make the stage again. A third time: the run ends with `gates_failed` and its report. Nothing is deleted on the way:
/// what a failed try made is cleaned up when the stage has passed (and never with `keep_intermediate`).
fn gate_failed(st: &mut AutoState, idx: usize, failed: &[String], fx: &mut dyn Effects) -> Result<Flow> {
    let n = idx + 1;
    let names = failed.join(", ");
    let ids = |what: &[&str]| -> Vec<String> { what.iter().map(|w| format!("s{n}.{w}")).collect() };
    match st.stages[idx].attempt {
        0 => {
            let sg = &mut st.stages[idx];
            sg.retry_from = sg.heal.take();
            sg.extra_tokens_m = RETRY_TOKENS_M;
            sg.attempt = 1;
            let note = format!("Stage {n} missed its thinking check ({names}): it is healed again on the healed model with {RETRY_TOKENS_M} million more tokens.");
            st.notes.push(note.clone());
            reopen(st, &ids(&["heal", "check_heal"]));
            return Ok(Flow::Retry(note));
        }
        1 => match back_off_stage(st, idx, fx.vram_gb()) {
            Ok(note) => {
                st.notes.push(note.clone());
                let all: Vec<&str> = if st.keep_thinking() { vec!["cut", "check_cut", "heal", "check_heal"] } else { vec!["cut", "heal"] };
                reopen(st, &ids(&all));
                return Ok(Flow::Retry(note));
            }
            Err(why) => st.notes.push(why),
        },
        _ => {}
    }
    st.error = Some(format!("The thinking check of the healed model of stage {n} failed ({names}) after {} tries; the run stops here. The report says what was tried.", st.stages[idx].attempts.len()));
    Ok(Flow::Halt("gates_failed".into()))
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
    let r = out!(fx.think_check(&h.name, "gguf", st.spec.quick, Some(&file))?);
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
    let mut plan = auto_plan::with_history(auto_plan::plan(ins, spec, vram_gb, free_gb), ins, spec, &crate::timing::History::recent(HISTORY_LOGS));
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
