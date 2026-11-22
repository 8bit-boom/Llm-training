//! The runner: "Auto: test and choose the best settings". It runs the checklist one test after the other, in the order of their value (so a stopped or cut-short run still yields a useful preset),
//! keeps its state in `<project>/tune/runs/<id>/state.json` (resumable), and at the end saves a preset. Everything that touches the world sits behind `TuneFx`, so the whole run is tested with a fake.
//! Nothing is applied here: the window shows one confirmation with every change and its measured gain (docs/TUNE.md).
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use super::decide::{decide_batch, decide_loss, DEFAULT_SPREAD};
use super::identity::{MachineId, ModelIdentity};
use super::preset::{FunctionResult, Guard, Preset};
use crate::error::{Error, Result};

/// One test of the checklist.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TestDef {
    pub id: &'static str,
    pub label: &'static str,
    /// what it tests, in words for the checklist
    pub what: &'static str,
    /// 1 exact speed and memory knobs, 2 quality knobs (short trial heals), 3 other functions
    pub tier: u32,
    /// "exact" | "tol" | "stat" | "quality"
    pub class: &'static str,
    pub default_on: bool,
    /// the knobs of the Heal card (or settings) it decides
    pub knobs: &'static [&'static str],
    /// whether this program version can run it
    pub built: bool,
    /// graphics memory it needs free to start, as a share of the card (a rough guard before it starts)
    pub need_share: f64,
}

pub const CATALOG: [TestDef; 5] = [
    TestDef { id: "heal.student_batch", label: "Tokens per forward pass", what: "Rows of similar length share one forward pass of the student: the same learning, usually much faster. Tests 0, 2048, 4096, 8192 and 16384 tokens on your real rows, with the worst-case update for the memory.",
              tier: 1, class: "tol", default_on: true, knobs: &["student_batch_tokens"], built: true, need_share: 0.55 },
    TestDef { id: "heal.loss", label: "Loss chunk and output-layer logits", what: "The KL chunk (64 to 512) and keeping the output layer's logits: the same maths, different speed and memory.",
              tier: 1, class: "exact", default_on: true, knobs: &["kl_chunk", "head_checkpoint"], built: true, need_share: 0.55 },
    TestDef { id: "heal.quality", label: "Quality settings (short trial heals)", what: "LoRA size, optimizer, warm-up and learning rate, each tried alone from your settings in a short real heal on your rows with the same seed; the held-out loss decides. A short trial, not a full heal.",
              tier: 2, class: "quality", default_on: true, knobs: &["rank", "optimizer", "warmup", "lr"], built: true, need_share: 0.6 },
    TestDef { id: "teacher.standard", label: "The original's answers: tokens per forward pass", what: "How many tokens of rows the original answers in one forward pass (the standard engine).",
              tier: 1, class: "tol", default_on: true, knobs: &["teacher_batch_tokens"], built: false, need_share: 0.7 },
    TestDef { id: "disk", label: "Disk speed", what: "Writes and reads a few GB on the drive that holds the models, to tell how long a cut or a merge will take.",
              tier: 3, class: "exact", default_on: false, knobs: &[], built: false, need_share: 0.0 },
];

pub fn def(id: &str) -> Option<&'static TestDef> {
    CATALOG.iter().find(|d| d.id == id)
}

/// Minutes (low, high) a test takes for a student of `student_b` billion parameters (the plan's numbers for a 20 B student, scaled by size; a fixed part for loading).
pub fn estimate(id: &str, student_b: f64, trial_tokens_k: u32) -> (f64, f64) {
    let sc = (student_b / 20.0).clamp(0.2, 3.0);
    match id {
        "heal.student_batch" => (15.0 * sc, 25.0 * sc),
        "heal.loss" => (20.0 * sc, 35.0 * sc),
        "heal.quality" => ((12.0 + trial_tokens_k as f64 / 40.0) * 5.0 * sc, (20.0 + trial_tokens_k as f64 / 25.0) * 5.0 * sc),
        "teacher.standard" => (10.0, 15.0),
        "disk" => (2.0, 3.0),
        _ => (0.0, 0.0),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Spec {
    pub model: String,
    /// the ids of the tests to run
    #[serde(default)]
    pub tests: Vec<String>,
    /// the Heal card as it is now (every knob): the baseline of every test
    #[serde(default)]
    pub card: Map<String, Value>,
    /// "I have at most this many hours": tests are dropped from the end of the order until the estimate fits (None: no limit)
    #[serde(default)]
    pub max_hours: Option<f64>,
    /// the budget of one trial heal of the quality tests, in thousands of tokens
    #[serde(default)]
    pub trial_tokens_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestState {
    pub id: String,
    /// queued | running | done | failed | skipped | timeout | stopped | waiting_gpu
    pub status: String,
    #[serde(default)]
    pub est_minutes: (f64, f64),
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub rows: Vec<Value>,
    #[serde(default)]
    pub decision: Value,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub result: Option<FunctionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunState {
    pub format: u32,
    pub id: String,
    pub spec: Spec,
    pub created: String,
    /// created | running | done | stopped | waiting_gpu | failed
    pub status: String,
    pub tests: Vec<TestState>,
    #[serde(default)]
    pub preset_id: Option<String>,
    #[serde(default)]
    pub model: ModelIdentity,
    #[serde(default)]
    pub machine: MachineId,
    #[serde(default)]
    pub student_b: f64,
    #[serde(default)]
    pub note: String,
}

pub fn runs_dir(project: &Path) -> PathBuf {
    project.join("tune").join("runs")
}

pub fn state_file(project: &Path, id: &str) -> PathBuf {
    runs_dir(project).join(id).join("state.json")
}

pub fn save_state(project: &Path, st: &RunState) -> Result<()> {
    crate::fsx::write_json_atomic(&state_file(project, &st.id), st)
}

pub fn load_state(project: &Path, id: &str) -> Result<RunState> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(Error::new("That is not a run id."));
    }
    serde_json::from_slice(&std::fs::read(state_file(project, id)).map_err(|_| Error::new("That run is not there any more."))?).map_err(|e| Error::new(format!("That run could not be read: {e}")))
}

/// The runs of a project, newest first (id, status, model, created).
pub fn list_runs(project: &Path) -> Vec<Value> {
    let mut out: Vec<RunState> = std::fs::read_dir(runs_dir(project)).into_iter().flatten().flatten().filter_map(|e| serde_json::from_slice(&std::fs::read(e.path().join("state.json")).ok()?).ok()).collect();
    out.sort_by(|a, b| b.created.cmp(&a.created));
    out.iter().map(|s| json!({"id": s.id, "status": s.status, "model": s.spec.model, "created": s.created, "preset_id": s.preset_id, "tests": s.tests.iter().map(|t| json!({"id": t.id, "status": t.status})).collect::<Vec<_>>()})).collect()
}

/// What the page shows before Start: for each test the state (ready / not built / not relevant), the estimate in minutes and the total with "done around", after the hours cap.
pub fn plan_view(spec: &Spec, student_b: f64, now_secs: i64, notes: &BTreeMapNotes) -> Value {
    let trial = if spec.trial_tokens_k == 0 { 400 } else { spec.trial_tokens_k };
    let mut total = (0.0, 0.0);
    let mut rows = Vec::new();
    for d in CATALOG.iter() {
        let chosen = spec.tests.iter().any(|t| t == d.id);
        let est = estimate(d.id, student_b, trial);
        let reason = if !d.built { Some("not built in this version yet".to_string()) } else { notes.get(d.id).cloned() };
        if chosen && reason.is_none() {
            total = (total.0 + est.0, total.1 + est.1);
        }
        rows.push(json!({"id": d.id, "label": d.label, "what": d.what, "tier": d.tier, "class": d.class, "default_on": d.default_on, "chosen": chosen, "built": d.built, "unavailable": reason, "minutes": [est.0.round(), est.1.round()], "knobs": d.knobs}));
    }
    json!({"tests": rows, "total_minutes": [total.0.round(), total.1.round()], "done_between": [now_secs + (total.0 * 60.0) as i64, now_secs + (total.1 * 60.0) as i64]})
}

pub type BTreeMapNotes = std::collections::BTreeMap<String, String>;

/// The ids to run, in the order of their value (tier 1, then 2, then 3), without the ones this version cannot run, and without the tail that does not fit `max_hours`.
pub fn order(spec: &Spec, student_b: f64) -> Vec<&'static TestDef> {
    let mut v: Vec<&TestDef> = CATALOG.iter().filter(|d| d.built && spec.tests.iter().any(|t| t == d.id)).collect();
    v.sort_by_key(|d| d.tier);
    if let Some(h) = spec.max_hours.filter(|h| *h > 0.0) {
        let trial = if spec.trial_tokens_k == 0 { 400 } else { spec.trial_tokens_k };
        let mut used = 0.0;
        v.retain(|d| {
            used += estimate(d.id, student_b, trial).1;
            used <= h * 60.0
        });
    }
    v
}

// ------------------------------------------------------------------------------------------------------------------ effects

pub struct TestCtx<'a> {
    pub model: &'a str,
    pub card: &'a Map<String, Value>,
    pub student_b: f64,
    pub card_mb: Option<f64>,
    pub trial_tokens_k: u32,
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct TestOut {
    /// every candidate with its numbers (the raw rows of the bench)
    pub rows: Vec<Value>,
    /// extra numbers of the test (the sweep's spread, the rows' length statistics ...)
    pub raw: Value,
    pub stopped: bool,
    pub note: String,
}

pub trait TuneFx {
    fn emit(&mut self, ev: Value);
    fn stopped(&self) -> bool;
    fn now(&self) -> String;
    /// (free MB, total MB, the programs that hold memory) of the card now
    fn gpu(&self) -> Option<(u64, u64, Vec<String>)>;
    fn run_test(&mut self, d: &TestDef, ctx: &TestCtx) -> Result<TestOut>;
}

/// What a finished test becomes: the decision and the entry of the preset.
fn result_of(d: &TestDef, ctx: &TestCtx, out: &TestOut) -> (Value, Option<FunctionResult>) {
    let context: Map<String, Value> = ["seq", "method", "rank", "per_update", "checkpointing", "batch"].iter().filter_map(|k| ctx.card.get(*k).map(|v| (k.to_string(), v.clone()))).collect();
    let deps: Vec<String> = ["gpu", "driver_major", "torch_minor", "transformers_minor", "peft_minor", "bitsandbytes_minor", "triton_minor", "fla_minor"].iter().map(|s| s.to_string()).collect();
    match d.id {
        "heal.student_batch" => {
            let spread = out.raw["spread"].as_f64().unwrap_or(0.0);
            let dec = decide_batch(&out.rows, ctx.card_mb, if spread > 0.0 { spread } else { DEFAULT_SPREAD });
            let base = out.rows.iter().find(|r| r["batch_tokens"] == 0).and_then(|r| r["tokens_per_s"].as_f64()).unwrap_or(0.0);
            let won = out.rows.iter().find(|r| r["batch_tokens"].as_u64() == Some(dec.chosen as u64)).cloned().unwrap_or(Value::Null);
            let fr = FunctionResult {
                chosen: json!({"student_batch_tokens": dec.chosen}).as_object().cloned().unwrap_or_default(), default: json!({"student_batch_tokens": 0}).as_object().cloned().unwrap_or_default(),
                gain: dec.gain, spread: dec.spread, class: d.class.into(), tier: d.tier, guards: dec.guards.clone(),
                measured: json!({"tokens_per_s": won["tokens_per_s"], "default_tokens_per_s": base, "peak_vram_mb": won["peak_vram_mb"], "rows": out.raw["rows"], "student_b": ctx.student_b}),
                tried: out.rows.clone(), depends_on: deps, transferable: vec!["student_batch_tokens".into()], context, note: dec.why.clone(), ..Default::default()
            };
            (json!({"chosen": dec.chosen, "gain": dec.gain, "changed": dec.changed, "why": dec.why}), Some(fr))
        }
        "heal.loss" => {
            let dec = decide_loss(&out.rows, 0.0);
            if dec.chunk == 0 {
                return (json!({"changed": false, "why": dec.why}), None);
            }
            let fr = FunctionResult {
                chosen: json!({"kl_chunk": dec.chunk, "head_checkpoint": if dec.head_checkpoint { "on" } else { "off" }}).as_object().cloned().unwrap_or_default(),
                default: json!({"kl_chunk": ctx.card.get("kl_chunk").cloned().unwrap_or(Value::Null), "head_checkpoint": ctx.card.get("head_checkpoint").cloned().unwrap_or(json!("on"))}).as_object().cloned().unwrap_or_default(),
                gain: dec.gain, spread: DEFAULT_SPREAD, class: d.class.into(), tier: d.tier,
                guards: vec![Guard { name: "same_maths".into(), ok: true, value: None, limit: None }],
                measured: json!({"tokens_per_s": dec.best_tps, "default_tokens_per_s": dec.baseline_tps, "student_b": ctx.student_b}),
                tried: out.rows.clone(), depends_on: deps, transferable: vec![], context, note: dec.why.clone(), ..Default::default()
            };
            (json!({"chunk": dec.chunk, "head_checkpoint": dec.head_checkpoint, "gain": dec.gain, "changed": dec.changed, "why": dec.why}), Some(fr))
        }
        "heal.quality" => {
            let (decisions, noise, base) = super::quality::decide_quality(&out.rows, ctx.card_mb);
            if decisions.is_empty() {
                return (json!({"changed": false, "why": "The trials did not give a baseline, so nothing was decided."}), None);
            }
            let mut chosen = Map::new();
            let mut default = Map::new();
            let mut notes = Vec::new();
            let mut best_gain = 0.0f64;
            for k in decisions.iter().filter(|k| k.changed) {
                chosen.insert(k.knob.clone(), k.chosen.clone());
                default.insert(k.knob.clone(), ctx.card.get(&k.knob).cloned().unwrap_or(Value::Null));
                best_gain += k.gain_loss;
            }
            for k in &decisions {
                notes.push(format!("{}: {}", k.knob, k.why));
            }
            let note = format!("Quality knob: based on a short trial, not a full heal. {}", notes.join(" "));
            let gain = if base > 0.0 { base / (base - best_gain).max(1e-6) } else { 1.0 };
            let fr = FunctionResult {
                chosen: chosen.clone(), default, gain, spread: if base > 0.0 { noise / base } else { 0.0 }, class: d.class.into(), tier: d.tier,
                guards: vec![Guard { name: "peak_vram_below_90_percent".into(), ok: true, value: None, limit: Some(0.9) }],
                measured: json!({"held_out_loss": base, "noise": noise, "trial_tokens_k": ctx.trial_tokens_k, "student_b": ctx.student_b}),
                tried: out.rows.clone(), depends_on: deps, transferable: vec![], context, note: note.clone(), ..Default::default()
            };
            let changed = !chosen.is_empty();
            (json!({"changed": changed, "chosen": chosen, "noise": noise, "baseline_loss": base, "why": note, "knobs": decisions.iter().map(|k| json!({"knob": k.knob, "changed": k.changed, "chosen": k.chosen, "gain_loss": k.gain_loss, "why": k.why})).collect::<Vec<_>>()}), if changed { Some(fr) } else { None })
        }
        _ => (json!({"why": "not built"}), None),
    }
}

/// Runs (or goes on with) a run. The state is saved after every test; Stop (`fx.stopped()`) ends it between tests or inside one (the test then reruns from its start on Continue). A test whose card
/// does not have the memory free waits (`waiting_gpu`): nothing is ended for it. At the end the preset is saved and `done` is emitted.
pub fn run(fx: &mut dyn TuneFx, project: &Path, st: &mut RunState, card_mb: Option<f64>, store: &super::store::Store) -> Result<()> {
    st.status = "running".into();
    save_state(project, st)?;
    for i in 0..st.tests.len() {
        if matches!(st.tests[i].status.as_str(), "done" | "failed" | "skipped") {
            continue;
        }
        let Some(d) = def(&st.tests[i].id) else { continue };
        if fx.stopped() {
            st.status = "stopped".into();
            save_state(project, st)?;
            fx.emit(json!({"kind": "tune_test", "index": i, "total": st.tests.len(), "id": d.id, "status": "stopped"}));
            return Ok(());
        }
        // the card must have room: the memory another program (or an engine) holds is not ours to free
        if let (Some((free, total, others)), true) = (fx.gpu(), d.need_share > 0.0) {
            let need = (total as f64 * d.need_share) as u64;
            if free < need {
                st.tests[i].status = "waiting_gpu".into();
                st.status = "waiting_gpu".into();
                st.note = format!("{} needs about {:.1} GB of the card and only {:.1} GB are free. {}Free the memory (Memory panel, Free GPU), then Continue.", d.label, need as f64 / 1024.0, free as f64 / 1024.0,
                                  if others.is_empty() { String::new() } else { format!("Held by: {}. ", others.join(", ")) });
                save_state(project, st)?;
                fx.emit(json!({"kind": "tune_wait", "id": d.id, "why": st.note, "free_mb": free, "need_mb": need}));
                return Ok(());
            }
        }
        st.tests[i].status = "running".into();
        save_state(project, st)?;
        fx.emit(json!({"kind": "tune_test", "index": i, "total": st.tests.len(), "id": d.id, "label": d.label, "status": "running"}));
        let t0 = std::time::Instant::now();
        let dir = project.join("benchmarks").join(format!("tune-{}", st.id)).join("tmp").join(d.id);
        let ctx = TestCtx { model: &st.spec.model, card: &st.spec.card, student_b: st.student_b, card_mb, trial_tokens_k: st.spec.trial_tokens_k, dir: dir.clone() };
        let out = fx.run_test(d, &ctx);
        let _ = std::fs::remove_dir_all(&dir);
        st.tests[i].seconds = t0.elapsed().as_secs_f64();
        match out {
            Ok(o) if o.stopped => {
                st.tests[i].status = "stopped".into();
                st.status = "stopped".into();
                save_state(project, st)?;
                fx.emit(json!({"kind": "tune_test", "index": i, "total": st.tests.len(), "id": d.id, "status": "stopped"}));
                return Ok(());
            }
            Ok(o) => {
                let (decision, result) = result_of(d, &ctx, &o);
                st.tests[i].rows = o.rows;
                st.tests[i].decision = decision.clone();
                st.tests[i].note = o.note;
                st.tests[i].result = result;
                st.tests[i].status = "done".into();
                fx.emit(json!({"kind": "tune_test", "index": i, "total": st.tests.len(), "id": d.id, "status": "done", "decision": decision, "rows": st.tests[i].rows}));
            }
            Err(e) => {
                st.tests[i].status = "failed".into();
                st.tests[i].note = e.0.chars().take(400).collect();
                fx.emit(json!({"kind": "tune_test", "index": i, "total": st.tests.len(), "id": d.id, "status": "failed", "note": st.tests[i].note}));
            }
        }
        save_state(project, st)?;
    }
    finish(fx, project, st, store)
}

/// The end: the preset, the report, the state.
fn finish(fx: &mut dyn TuneFx, project: &Path, st: &mut RunState, store: &super::store::Store) -> Result<()> {
    let now = fx.now();
    let name = format!("Tested {} ({})", now.get(..10).unwrap_or(&now), st.spec.model);
    let mut preset = Preset::new(&name, &now, &st.id, st.model.clone(), st.machine.clone());
    preset.card = st.spec.card.clone();
    for t in &st.tests {
        if let Some(r) = &t.result {
            preset.functions.insert(t.id.clone(), r.clone());
        }
        if t.status != "done" {
            preset.notes.push(format!("{}: {}{}", t.id, t.status, if t.note.is_empty() { String::new() } else { format!(" ({})", t.note) }));
        }
    }
    preset.report = format!("benchmarks/tune-{}/tune_report.md", st.id);
    write_report(project, st, &preset)?;
    if !preset.functions.is_empty() {
        store.save(super::store::Scope::Project, &preset)?;
        st.preset_id = Some(preset.id.clone());
    }
    st.status = "done".into();
    save_state(project, st)?;
    fx.emit(json!({"kind": "done", "run": st.id, "preset_id": st.preset_id, "name": name}));
    Ok(())
}

fn write_report(project: &Path, st: &RunState, p: &Preset) -> Result<()> {
    let dir = project.join("benchmarks").join(format!("tune-{}", st.id));
    std::fs::create_dir_all(&dir)?;
    crate::fsx::write_json_atomic(&dir.join("tune_report.json"), &json!({"run": st.id, "model": st.spec.model, "created": st.created, "machine": st.machine, "tests": st.tests, "preset": p.id}))?;
    let mut md = format!("# Tune report: {}\n\nRun {} on {} ({} MB).\n\n", st.spec.model, st.id, st.machine.gpu, st.machine.vram_mb);
    md.push_str("| Test | Result | Gain | Class |\n|---|---|---|---|\n");
    for t in &st.tests {
        let (gain, class) = t.result.as_ref().map(|r| (format!("x{:.2}", r.gain), r.class.clone())).unwrap_or_default();
        md.push_str(&format!("| {} | {} {} | {} | {} |\n", t.id, t.status, t.decision["why"].as_str().unwrap_or(""), gain, class));
    }
    md.push_str("\nSpeed knobs are checked for the same results; quality knobs are based on a short trial, not a full heal. The numbers were measured on this PC; a real heal is a little slower.\n");
    crate::fsx::write_atomic(&dir.join("tune_report.md"), md.as_bytes())
}

/// A new run: the state for `spec` with its tests in order, ready to `run`.
pub fn new_run(spec: Spec, model: ModelIdentity, machine: MachineId, student_b: f64, now: &str) -> RunState {
    let id = now.chars().filter(|c| c.is_ascii_digit()).take(12).collect::<String>();
    let id = format!("{}-{}", &id[..8.min(id.len())], &id[8.min(id.len())..]);
    let trial = if spec.trial_tokens_k == 0 { 400 } else { spec.trial_tokens_k };
    let tests = order(&spec, student_b).iter().map(|d| TestState { id: d.id.into(), status: "queued".into(), est_minutes: estimate(d.id, student_b, trial), ..Default::default() }).collect();
    RunState { format: 1, id, spec, created: now.into(), status: "created".into(), tests, model, machine, student_b, ..Default::default() }
}
