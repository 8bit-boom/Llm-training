//! The window shell. All logic is in `ndforge-core` (`AppModel`); these commands forward to it and return its JSON, or the
//! plain-language error text, to the page.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ndforge_core::help::help_html;
use ndforge_core::model::AppModel;
use ndforge_core::train::plan::{Method, Preset};
use serde_json::Value;
use tauri::{Emitter, Manager, State};

struct App(Mutex<AppModel>);

/// One long job at a time (install or self-test); `cancel` stops it.
#[derive(Default)]
struct Tasks {
    busy: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    /// the window was asked to close while a job runs: the job is being stopped (and saved) first
    closing: Arc<AtomicBool>,
    /// "shut the computer down when the training has finished" (the checkbox; not remembered between program starts, on purpose)
    shutdown_after: Arc<AtomicBool>,
    /// the Cancel button of the shutdown countdown
    shutdown_cancel: Arc<AtomicBool>,
}

/// The Storage page's scan: it only reads, so it does not take the one-job flag; its own cancel and "running" flags.
#[derive(Default)]
struct ScanTasks {
    running: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    /// a deletion of the Storage page is the running job (its Cancel must not stop another job)
    deleting: Arc<AtomicBool>,
}

/// The job queue (ndforge_core::queue), shared by the commands and the runner that starts the next job when one ends.
struct QueueState(Mutex<ndforge_core::queue::Queue>);

/// Holds the "a job is running" flag for the life of a job thread: the flag goes back to false when the guard is dropped, also when the thread panics.
struct BusyGuard(Arc<AtomicBool>);
impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Asks the running job to stop and waits until it has ended, at most `max`. A training run saves a checkpoint when it is stopped, so it can be
/// continued later. true = it ended in time.
fn stop_and_wait(busy: &AtomicBool, cancel: &AtomicBool, max: std::time::Duration) -> bool {
    cancel.store(true, Ordering::SeqCst);
    let start = std::time::Instant::now();
    while busy.load(Ordering::SeqCst) {
        if start.elapsed() > max {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    true
}

type Out<T> = Result<T, String>;

/// The lock that keeps a second program out of the settings folder (instance.rs); it lives as long as the program.
#[allow(dead_code)]
struct InstanceHold(Mutex<Option<ndforge_core::instance::Instance>>);

fn lock<'a>(s: &'a State<'a, App>) -> std::sync::MutexGuard<'a, AppModel> {
    s.0.lock().unwrap_or_else(|e| e.into_inner())
}

#[tauri::command]
fn get_state(s: State<App>) -> Value {
    lock(&s).state()
}

#[tauri::command]
fn create_project(s: State<App>, name: String) -> Out<Value> {
    let mut m = lock(&s);
    m.create_project(&name).map_err(|e| e.to_string())?;
    Ok(m.state())
}

#[tauri::command]
fn open_project(app: tauri::AppHandle, s: State<App>, folder: String) -> Out<Value> {
    let state = {
        let mut m = lock(&s);
        m.open_project(&folder).map_err(|e| e.to_string())?;
        m.state()
    };
    auto_continue(&app);
    Ok(state)
}

/// With the setting `auto_continue_interrupted` on: an Auto-cut or a heal that was cut off by closing the program goes into the queue and the queue runs.
fn auto_continue(app: &tauri::AppHandle) {
    let st = app.state::<App>();
    let (enabled, project) = { let m = st.0.lock().unwrap_or_else(|e| e.into_inner()); (m.settings.auto_continue_interrupted, m.project.clone()) };
    let Some(p) = project.filter(|_| enabled) else { return };
    if app.state::<Tasks>().busy.load(Ordering::SeqCst) {
        return;
    }
    ndforge_core::reshape::auto::mark_interrupted(&p);
    let folder = p.folder.to_string_lossy().to_string();
    let added = ndforge_core::queue::interrupted_work(&p).into_iter().filter(|(kind, args)| queue_edit(app, |q| {
        if q.items.iter().any(|i| i.status == "queued" && &i.kind == kind && &i.spec == args) { return false; }
        q.enqueue(&folder, kind, args.clone()).is_ok()
    })).count();
    if added > 0 {
        queue_edit(app, |q| q.paused = false);
        queue_changed(app);
        queue_kick(app);
    }
}

/// Takes a world out of a project that holds several. Returns the new state.
#[tauri::command]
fn remove_world(s: State<App>, slug: String) -> Out<Value> {
    let mut m = lock(&s);
    m.remove_world(&slug).map_err(|e| e.to_string())?;
    Ok(m.state())
}

#[tauri::command(async)]
fn check_bundle(s: State<App>, path: String) -> Out<Value> {
    lock(&s).check_bundle(&path).map_err(|e| e.to_string())
}

/// Returns {"changes": text, "state": ...}
#[tauri::command(async)]
fn apply_bundle(s: State<App>, path: String) -> Out<Value> {
    let mut m = lock(&s);
    let changes = m.apply_bundle(&path).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"changes": changes, "state": m.state()}))
}

#[tauri::command]
fn set_setting(s: State<App>, key: String, value: bool) -> Out<Value> {
    let mut m = lock(&s);
    m.set_setting(&key, value).map_err(|e| e.to_string())?;
    Ok(m.state())
}

#[tauri::command(async)]
fn dataset_state(s: State<App>) -> Out<Value> {
    lock(&s).dataset_state().map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn dataset_generate(s: State<App>, recipe: Value) -> Out<Value> {
    lock(&s).dataset_generate(recipe).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn problem_report(s: State<App>) -> Out<String> {
    // the report asks the machine (nvidia-smi ...): the model is not locked meanwhile
    let (project, settings) = {
        let m = lock(&s);
        (m.project.clone(), m.settings.clone())
    };
    Ok(ndforge_core::diagnose::report(project.as_ref(), &settings))
}

/// Saves the diagnostics bundle (diagnose::bundle: the report, the last job logs, the settings without keys ...) into the folder `dest`; answers with the file.
#[tauri::command(async)]
fn diagnostics_save(s: State<App>, dest: String) -> Out<String> {
    let (project, settings) = {
        let m = lock(&s);
        (m.project.clone(), m.settings.clone())
    };
    ndforge_core::diagnose::bundle(project.as_ref(), &settings, std::path::Path::new(&dest)).map(|p| p.display().to_string()).map_err(|e| e.to_string())
}

/// Where the time of the latest jobs went (the timing lines of the saved job logs, summed per kind of job and stage) and what is worth changing: `{"summary", "advice"}`.
#[tauri::command(async)]
fn timing_report() -> Value {
    ndforge_core::timing::report_of_latest(30)
}

/// Looks a Hugging Face model up (a network call, so off the window's thread).
#[tauri::command]
async fn model_lookup(input: String) -> Out<Value> {
    tauri::async_runtime::spawn_blocking(move || AppModel::custom_model_lookup(&input).map_err(|e| e.to_string())).await.map_err(|e| e.to_string())?
}

/// Adds it to the list; answers with its id.
#[tauri::command]
async fn model_add(input: String) -> Out<String> {
    tauri::async_runtime::spawn_blocking(move || AppModel::custom_model_add(&input).map_err(|e| e.to_string())).await.map_err(|e| e.to_string())?
}

#[tauri::command(async)]
fn model_remove(id: String) -> Out<()> {
    AppModel::custom_model_remove(&id).map_err(|e| e.to_string())
}

/// The general-text chooser: presets, the person's own datasets, the last choice of this project, whether a Hugging Face token is saved.
#[tauri::command]
fn general_state(s: State<App>) -> Value {
    lock(&s).general_state()
}

#[tauri::command]
fn general_choose(s: State<App>, scope: String, source: ndforge_core::general::GeneralSource) -> Out<Value> {
    lock(&s).general_choose(&scope, &source).map_err(|e| e.to_string())
}

#[tauri::command]
fn general_add(s: State<App>, source: ndforge_core::general::GeneralSource) -> Out<Value> {
    lock(&s).general_add(&source).map_err(|e| e.to_string())
}

#[tauri::command]
fn general_remove(s: State<App>, key: String) -> Out<Value> {
    lock(&s).general_remove(&key).map_err(|e| e.to_string())
}

/// The person ticked the licence acknowledgement of a dataset.
#[tauri::command]
fn general_ack(s: State<App>, key: String) -> Out<Value> {
    lock(&s).general_ack(&key).map_err(|e| e.to_string())
}

/// "Check it": reads about 20 rows of the dataset with the worker (a network call, so off the window's thread).
#[tauri::command]
async fn general_check(source: ndforge_core::general::GeneralSource) -> Out<Value> {
    tauri::async_runtime::spawn_blocking(move || ndforge_core::general::check(&source).map_err(|e| e.to_string())).await.map_err(|e| e.to_string())?
}

/// The Hugging Face token for gated datasets: kept with the other secrets, handed to the worker as HF_TOKEN only.
#[tauri::command]
fn hf_token_set(s: State<App>, token: String) -> Out<Value> {
    lock(&s).hf_token_set(&token).map_err(|e| e.to_string())
}

#[tauri::command]
fn hf_token_clear(s: State<App>) -> Out<Value> {
    lock(&s).hf_token_clear().map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn test_state(s: State<App>) -> Out<Value> {
    lock(&s).test_state().map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn test_result(s: State<App>, run: String) -> Out<Value> {
    lock(&s).test_result(&run).map_err(|e| e.to_string())
}

/// Tests a run (original vs trained on the held-out questions). Events on "test-event".
#[tauri::command]
fn test_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, run: String, judge: bool, against: Option<String>) -> Out<()> {
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let teacher = if judge { Some(m.teacher_client()) } else { None };
    drop(m);
    spawn_task(app, &tasks, "test-event", move |cancel, emit| {
        if let Some(t) = &teacher { if !t.cfg.is_set() { return Err("Choose the teacher in Step 2 first, or untick the teacher grading.".to_string()); } }
        // against: "origin" = the model a slimmed / grown base was made from; anything else = the base the run started from
        let r = ndforge_core::evaluate::runs::execute_against(&p, &run, 30, teacher.as_ref(), against.as_deref() == Some("origin"), card_vram_gb(), cancel, emit).map_err(|e| e.to_string())?;
        if let Some(t) = &teacher { if t.cfg.unload_after { let _ = ndforge_core::export::ollama::Ollama::new(&t.cfg.ollama_base()).unload(&t.cfg.model); } }
        Ok(r)
    })
}

/// One free question to both models. Events on "test-event"; the answer is {"chat": {"original", "trained"}}.
#[tauri::command]
fn test_ask(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, run: String, system: Option<String>, question: String, against: Option<String>) -> Out<()> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task(app, &tasks, "test-event", move |cancel, emit| {
        let a = ndforge_core::evaluate::runs::ask_against(&p, &run, system.as_deref(), &question, against.as_deref() == Some("origin"), card_vram_gb(), cancel, emit).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"chat": a, "question": question}))
    })
}

#[tauri::command]
fn test_save_transcript(s: State<App>, run: String, items: Vec<Value>) -> Out<String> {
    lock(&s).test_save_transcript(&run, &items).map_err(|e| e.to_string())
}

#[tauri::command]
fn teacher_set(s: State<App>, url: String, model: String, key: String, unload_after: bool) -> Out<Value> {
    let mut m = lock(&s);
    m.teacher_set(&url, &model, &key, unload_after).map_err(|e| e.to_string())?;
    m.dataset_state().map_err(|e| e.to_string())
}

/// Does the teacher answer? Reported on "teacher-event".
#[tauri::command]
fn teacher_check(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>) -> Out<()> {
    let t = lock(&s).teacher_client();
    spawn_task(app, &tasks, "teacher-event", move |_cancel, _emit| {
        if !t.cfg.is_set() { return Err("Fill in the teacher's address and model name first.".to_string()); }
        t.check().map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"checked": true}))
    })
}

/// Makes the teacher examples. Progress and the final report arrive on "teacher-event".
#[tauri::command]
fn teacher_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, options: Value) -> Out<()> {
    let m = lock(&s);
    let opt = AppModel::teacher_options(options).map_err(|e| e.to_string())?;
    let (t, p) = (m.teacher_client(), m.project_clone().map_err(|e| e.to_string())?);
    drop(m);
    spawn_task(app, &tasks, "teacher-event", move |cancel, emit| ndforge_core::teacher::run(&p, &t, &opt, cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn teacher_clear(s: State<App>, kind: Option<String>, forget_cache: bool) -> Out<Value> {
    lock(&s).teacher_clear(kind.as_deref(), forget_cache).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn dataset_examples(s: State<App>, filter: Value, offset: usize, limit: usize) -> Out<Value> {
    lock(&s).dataset_examples(&filter, offset, limit).map_err(|e| e.to_string())
}

#[tauri::command]
fn dataset_set_status(s: State<App>, ids: Vec<String>, status: String) -> Out<Value> {
    lock(&s).dataset_set_status(ids, &status).map_err(|e| e.to_string())
}

#[tauri::command]
fn dataset_set_status_matching(s: State<App>, filter: Value, status: String) -> Out<Value> {
    lock(&s).dataset_set_status_matching(&filter, &status).map_err(|e| e.to_string())
}

#[tauri::command]
fn dataset_edit(s: State<App>, id: String, text: String) -> Out<()> {
    lock(&s).dataset_edit(&id, &text).map_err(|e| e.to_string())
}

/// Writes a saved data version as JSONL files for Unsloth Studio's Train page into `dest` (default: the project's exports folder). Returns the summary.
#[tauri::command(async)]
fn studio_export(s: State<App>, version: Option<u32>, dest: Option<String>, world_book: bool) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let dest = dest.map(std::path::PathBuf::from).unwrap_or_else(|| ndforge_core::export::studio::default_dest(&project));
    ndforge_core::export::studio::export(&project, version, &dest, world_book).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn dataset_save(s: State<App>) -> Out<Value> {
    lock(&s).dataset_save().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_theme(s: State<App>, theme: String) -> Out<Value> {
    let mut m = lock(&s);
    m.set_theme(&theme).map_err(|e| e.to_string())?;
    Ok(m.state())
}

/// Trains on this PC, on the server the job names, or in Unsloth Desktop.
fn run_training(job: &ndforge_core::model::TrainJob, stop: Arc<AtomicBool>, emit: &mut dyn FnMut(Value)) -> ndforge_core::Result<Value> {
    if let Some(target) = &job.unsloth {
        return ndforge_core::unsloth::run::execute_unsloth(&job.project, &job.model, job.method, &job.hyper, &job.name, job.job.clone(), target, stop,
            job.job["resume"] == true, std::time::Duration::from_secs(3), emit);
    }
    match &job.remote {
        None => ndforge_core::train::runs::execute(&job.project, &job.model, job.method, &job.hyper, &job.name, job.job.clone(), stop, std::time::Duration::from_secs(120), emit),
        Some(target) => ndforge_core::remote::run::execute_remote(&job.project, &job.model, job.method, &job.hyper, &job.name, job.job.clone(), target, stop,
            job.job["resume"] == true, std::time::Duration::from_secs(1), emit),
    }
}

/// The speed benchmark: short real runs with several settings (this computer only). Progress and the result arrive on "bench-event".
#[tauri::command]
fn bench_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, model: String, method: String, preset: String, overrides: Value, deep: bool) -> Out<()> {
    let (method, preset) = parse_choice(&method, &preset)?;
    let job = lock(&s).bench_job(&model, method, preset, &overrides).map_err(|e| e.to_string())?;
    let stop = tasks.cancel.clone();     // the Stop button of the running job
    spawn_task(app, &tasks, "bench-event", move |_cancel, emit| {
        ndforge_core::train::bench::run(&job.project, &job.model, job.method, &job.job, &job.hyper, 16, deep, stop, emit).map_err(|e| e.to_string())
    })
}

#[tauri::command(async)]
fn bench_last(s: State<App>, model: String, method: String) -> Out<Value> {
    let (method, _) = parse_choice(&method, "quick")?;
    lock(&s).bench_last(&model, method).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn train_state(s: State<App>, server: Option<String>) -> Out<Value> {
    lock(&s).train_state(server.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn train_plan(s: State<App>, model: String, method: String, preset: String, server: Option<String>, stages: Option<String>) -> Out<Value> {
    let (method, preset) = parse_choice(&method, &preset)?;
    lock(&s).train_plan(&model, method, preset, server.as_deref(), stages.as_deref()).map_err(|e| e.to_string())
}

fn parse_choice(method: &str, preset: &str) -> Out<(Method, Preset)> {
    let method = match method { "lora" => Method::Lora, "qlora" => Method::Qlora, "full" => Method::Full, other => return Err(format!("Unknown method: {other}")) };
    let preset = match preset { "quick" => Preset::Quick, "normal" => Preset::Normal, "best" => Preset::Best, other => return Err(format!("Unknown preset: {other}")) };
    Ok((method, preset))
}

/// Starts a training run (or continues `resume`); progress goes to the window as "train-event" events.
#[tauri::command]
fn train_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, model: String, method: String, preset: String, overrides: Value, resume: Option<String>, server: Option<String>) -> Out<String> {
    let (method, preset) = parse_choice(&method, &preset)?;
    let job = lock(&s).train_job(&model, method, preset, &overrides, resume.as_deref(), server.as_deref()).map_err(|e| e.to_string())?;
    let (name, keep_awake) = (job.name.clone(), lock(&s).settings.keep_awake);
    let opts = JobOpts { awake: if keep_awake { Awake::Noted } else { Awake::Off }, kind: "train", shutdown_after: true, ..JobOpts::new("train-event") };
    // a run on a server is fetched by this program too: the computer stays awake for all of it
    spawn_job(app, &tasks, opts, move |stop, emit| run_training(&job, stop, emit).map_err(|e| e.to_string()))?;
    Ok(name)
}

/// The warning before the computer is switched off: a countdown the window shows with a Cancel button. Events go to "power-event".
fn shutdown_with_warning(app: &tauri::AppHandle, cancel: &AtomicBool, step: std::time::Duration, seconds: u32) {
    cancel.store(false, Ordering::SeqCst);
    let _ = app.emit("power-event", serde_json::json!({"kind": "armed", "on": false}));
    let done = ndforge_core::power::countdown(seconds, step, cancel, &mut |left| {
        let _ = app.emit("power-event", serde_json::json!({"kind": "countdown", "left": left}));
    });
    if !done {
        let _ = app.emit("power-event", serde_json::json!({"kind": "cancelled"}));
        return;
    }
    let _ = app.emit("power-event", serde_json::json!({"kind": "shutting_down"}));
    if let Err(e) = ndforge_core::power::shutdown_now() {
        let _ = app.emit("power-event", serde_json::json!({"kind": "failed", "text": e.to_string()}));
    }
}

/// The "shut down when training has finished" checkbox. It can be changed while a run is going.
#[tauri::command]
fn set_shutdown_after(tasks: State<Tasks>, on: bool) {
    tasks.shutdown_after.store(on, Ordering::SeqCst);
}

/// The Cancel button of the shutdown countdown.
#[tauri::command]
fn cancel_shutdown(tasks: State<Tasks>) {
    tasks.shutdown_cancel.store(true, Ordering::SeqCst);
}

#[tauri::command]
fn train_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

#[tauri::command(async)]
fn train_metrics(s: State<App>, run: String) -> Out<Value> {
    lock(&s).train_metrics(&run).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn export_state(s: State<App>) -> Out<Value> {
    lock(&s).export_state().map_err(|e| e.to_string())
}

/// Makes the GGUF + Modelfile of a run; progress goes to the window as "export-event" events.
#[tauri::command]
fn export_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, run: String, quant: String, num_ctx: u32, what: Option<String>, quants: Option<Vec<String>>, imatrix: Option<ndforge_core::export::runs::ImatrixOptions>, mixed: Option<ndforge_core::export::runs::MixedOptions>) -> Out<()> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "export-event", "export", move |cancel, emit| {
        let list = quants.filter(|q| !q.is_empty()).unwrap_or_else(|| vec![quant]);
        let mut metas = ndforge_core::export::runs::execute_many_full(&project, &run, &list, num_ctx, what.as_deref().unwrap_or("gguf"), &imatrix.unwrap_or_default(), &mixed.unwrap_or_default(), cancel, emit).map_err(|e| e.to_string())?;
        // one size: its export; several: a summary naming them all
        Ok(if metas.len() == 1 { metas.remove(0) } else { serde_json::json!({"name": metas.iter().filter_map(|m| m["name"].as_str()).collect::<Vec<_>>().join(", "), "run": run, "exports": metas}) })
    })
}

/// Regenerates the vision projector (mmproj) of an export whose model has an intact vision tower but
/// no vision file (e.g. the recorded origin was a deleted intermediate cut). Progress goes to the
/// window as "export-event" events; on success the export's meta/Modelfile/VISION.md are re-wired.
#[tauri::command]
fn export_restore_vision(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, export: String) -> Out<()> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "export-event", "export", move |cancel, emit| {
        let meta = ndforge_core::export::runs::restore_vision(&project, &export, cancel, emit).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"restored": export, "name": meta["name"], "mmproj_file": meta["mmproj_file"]}))
    })
}

/// What the "Save a model as GGUF (no training needed)" card of the Slim or grow page shows (crates/ndforge-core/src/export/runs.rs `exportable_models`).
#[tauri::command(async)]
fn export_model_state(s: State<App>) -> Out<Value> {
    lock(&s).export_model_state().map_err(|e| e.to_string())
}

/// Makes the GGUF + Modelfile of a model without a training run (a model the Slim or grow page made, a folder on this computer, or a model of the list — that one the
/// worker fetches first when it is not on this computer yet); progress goes to the window as "reshape-event" events tagged {"phase": "export"}. `params_b` and
/// `download_gb` say nothing to this command (they are for the queue's checklist); the source is what the card chose.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
fn export_model_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, source: Value, quant: String, num_ctx: u32, quants: Option<Vec<String>>, thinking: Option<bool>, imatrix: Option<ndforge_core::export::runs::ImatrixOptions>, mixed: Option<ndforge_core::export::runs::MixedOptions>, params_b: Option<f64>, download_gb: Option<f64>) -> Out<()> {
    let _ = (params_b, download_gb);
    let src: ndforge_core::export::runs::ModelSource = serde_json::from_value(source).map_err(|e| format!("The choice of model is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_job(app, &tasks, JobOpts { phase: Some("export"), tag_events: true, awake: awake_if_set(), kind: "export", ..JobOpts::new("reshape-event") }, move |cancel, emit| {
        let list = quants.filter(|q| !q.is_empty()).unwrap_or_else(|| vec![quant]);
        let mut metas = ndforge_core::export::runs::execute_no_train(&project, &src, &list, num_ctx, thinking.unwrap_or(false), &imatrix.unwrap_or_default(), &mixed.unwrap_or_default(), &cancel, emit).map_err(|e| e.to_string())?;
        // one size: its export; several: a summary naming them all
        Ok(if metas.len() == 1 { metas.remove(0) } else { serde_json::json!({"name": metas.iter().filter_map(|m| m["name"].as_str()).collect::<Vec<_>>().join(", "), "run": metas.first().and_then(|m| m["run"].as_str()).unwrap_or_default(), "exports": metas}) })
    })
}

/// The mixed-precision plan (docs/MIXED_QUANTS.md): every recipe's rules and estimated size from the tensor shapes of `gguf` (any GGUF of the model), and the "Fit to a size"
/// proposal when `target_gb` and a saved matrix (`matrix`, a library id) are given.
#[tauri::command(async)]
fn mixed_plan(s: State<App>, gguf: String, target_gb: Option<f64>, matrix: Option<String>) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::mixed::plan(&project, &gguf, target_gb, matrix.as_deref()).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------------------------------------ the matrix library (docs/IMATRIX.md)

/// What the "Importance matrices" card shows: the library, the choices for a new matrix, time guesses, the general-text chooser.
#[tauri::command(async)]
fn imatrix_list(s: State<App>) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::export::imatrix_lib::state(&project))
}

/// Makes a matrix on its own (no GGUF export). Events on "imatrix-event"; the end is {"kind": "done", "data": {..., "entry": ...}} or an error. Stop: `imatrix_stop`.
#[tauri::command]
fn imatrix_create(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::export::imatrix_lib::MakeSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "imatrix-event", "imatrix_make", move |cancel, emit| ndforge_core::export::imatrix_lib::create(&project, &spec, cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn imatrix_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

/// Adds a matrix file from this computer (`file`: .gguf or .dat) or the matrix saved next to an earlier export (`export`) to the library. Returns {"added": entry, "state": ...}.
#[tauri::command(async)]
fn imatrix_import(s: State<App>, file: Option<String>, export: Option<String>, label: Option<String>) -> Out<Value> {
    use ndforge_core::export::imatrix_lib as lib;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let label = label.unwrap_or_default();
    let added = match (file, export) {
        (Some(f), _) if !f.trim().is_empty() => lib::import_file(&project, std::path::Path::new(f.trim()), &label, &lib::free_bytes),
        (_, Some(e)) if !e.is_empty() => lib::import_export(&project, &e, &label, &lib::free_bytes),
        _ => return Err("Choose a matrix file (.gguf or .dat) or an earlier export.".into()),
    }.map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"added": added, "state": lib::state(&project)}))
}

/// "Save a copy...": the files of an entry in a new numbered folder inside `dest`. Returns the new folder.
#[tauri::command(async)]
fn imatrix_save_copy(s: State<App>, id: String, dest: String) -> Out<String> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::imatrix_lib::save_copy(&project, &id, std::path::Path::new(dest.trim()), &ndforge_core::export::imatrix_lib::free_bytes).map(|p| p.display().to_string()).map_err(|e| e.to_string())
}

/// Opens the folder of an entry (`id`) or a folder a copy was saved in (`path`) with the system's file manager.
#[tauri::command]
fn imatrix_open_folder(s: State<App>, id: Option<String>, path: Option<String>) -> Out<()> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::imatrix_lib::open_folder(&project, id.as_deref(), path.as_deref()).map_err(|e| e.to_string())
}

/// Makes the legacy .dat of an entry. Events on "imatrix-event" like `imatrix_create`.
#[tauri::command]
fn imatrix_make_dat(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, id: String) -> Out<()> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "imatrix-event", "imatrix_make", move |cancel, emit| ndforge_core::export::imatrix_lib::make_dat(&project, &id, cancel, emit).map_err(|e| e.to_string()))
}

/// Deletes an entry (refused while a job uses it). Returns the new state of the card.
#[tauri::command(async)]
fn imatrix_delete(s: State<App>, id: String) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::imatrix_lib::delete(&project, &id).map_err(|e| e.to_string())?;
    Ok(ndforge_core::export::imatrix_lib::state(&project))
}

/// Merges matrices (library entries, files, Unsloth's published matrix). `spec` is {"inputs": [{"kind": "library"|"file"|"published", "id"|"file"|"base"}], "label"}.
/// Events on "imatrix-event" like `imatrix_create`; the end has {"entry": ...}.
#[tauri::command]
fn imatrix_merge(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::export::imatrix_lib::MergeSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "imatrix-event", "imatrix_make", move |cancel, emit| ndforge_core::export::imatrix_lib::merge(&project, &spec, cancel, emit).map_err(|e| e.to_string()))
}

/// Looks up Unsloth's published matrix for a base model on the Hub (its repo, file and size; nothing is downloaded).
#[tauri::command(async)]
fn imatrix_published(s: State<App>, base: String) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::imatrix_lib::published_info(&project, &base).map_err(|e| e.to_string())
}

/// "Adapt to my cut model": the numbers before anything is made (layers kept, ffn_down before and after, the size), or the plain reason this model / matrix cannot be used.
/// `spec` is {"source": {"kind": "library"|"file"|"published", "id"|"file"|"base"}, "model": a made model's name, "weight", "label", "kept_layers", "source_size"}.
#[tauri::command(async)]
fn imatrix_adapt_preview(s: State<App>, spec: Value) -> Out<Value> {
    let spec: ndforge_core::export::imatrix_adapt::AdaptSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::imatrix_adapt::preview(&project, &spec).map_err(|e| e.to_string())
}

/// Adapts a matrix to a model cut on the Slim page (a new library entry). Events on "imatrix-event" like `imatrix_create`; the end has {"entry": ...}. Stop: `imatrix_stop`.
#[tauri::command]
fn imatrix_adapt(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::export::imatrix_adapt::AdaptSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "imatrix-event", "imatrix_make", move |cancel, emit| ndforge_core::export::imatrix_adapt::adapt(&project, &spec, cancel, emit).map_err(|e| e.to_string()))
}

/// The light check before an export: does this matrix seem to fit the model of `run`? {"warnings": [...], ...}; never a refusal.
#[tauri::command(async)]
fn imatrix_check(s: State<App>, id: String, run: Option<String>) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::imatrix_lib::check(&project, &id, run.as_deref()).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------------------------------------ compare quants (docs/QUANT_COMPARE.md)

/// What the "Compare quants" card shows: the saved comparisons, the files of the exports, the runs and models, the disk.
#[tauri::command(async)]
fn quantcheck_state(s: State<App>) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::export::quantcheck::state(&project))
}

/// The disk need of the 16-bit model's answers for every amount of text, before anything runs (the worker reads the vocabulary from the reference).
#[tauri::command(async)]
fn quantcheck_plan(s: State<App>, reference: Option<String>) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::quantcheck::plan(&project, reference.as_deref()).map_err(|e| e.to_string())
}

/// Runs a comparison. Events on "quantcheck-event"; the end is {"kind": "done", "data": {..., "entry": ...}} or an error. Stop: `quantcheck_stop`.
#[tauri::command]
fn quantcheck_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::export::quantcheck::CompareSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "quantcheck-event", "quantcheck", move |cancel, emit| ndforge_core::export::quantcheck::compare(&project, &spec, cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn quantcheck_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

// ------------------------------------------------------------------------------------------------------------------ Mixed: measured fit to a size (docs/MIXED_QUANTS.md)

/// What the page shows: the 16-bit files it can offer, the choices, the plans made so far and the last verdict.
#[tauri::command(async)]
fn quantfit_state(s: State<App>) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::export::quantfit::state(&project))
}

/// About a second, from the header only: minutes, disk, engine, the smallest and largest size, whether the matrix is of this model.
#[tauri::command(async)]
fn quantfit_estimate(s: State<App>, spec: Value) -> Out<Value> {
    let spec: ndforge_core::export::quantfit::QuantFitSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::quantfit::estimate(&project, &spec).map_err(|e| e.to_string())
}

/// Measures, plans and optionally makes and compares the files. Events on "quantfit-event"; the end is {"kind": "done", "data": ...} or an error. Stop: `quantfit_stop`.
#[tauri::command]
fn quantfit_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::export::quantfit::QuantFitSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task_as(app, &tasks, "quantfit-event", "quantfit", move |cancel, emit| ndforge_core::export::quantfit::run(&project, &spec, cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn quantfit_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

/// One saved comparison (the table, the line, the words).
#[tauri::command(async)]
fn quantcheck_get(s: State<App>, id: String) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::quantcheck::get(&project, &id).map(|(_, r)| r).map_err(|e| e.to_string())
}

/// Deletes a comparison (refused while it runs). Returns the new state of the card.
#[tauri::command(async)]
fn quantcheck_delete(s: State<App>, id: String) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::quantcheck::delete(&project, &id).map_err(|e| e.to_string())?;
    Ok(ndforge_core::export::quantcheck::state(&project))
}

/// Deletes only the big file of the 16-bit model's answers of a comparison. Returns the new state of the card.
#[tauri::command(async)]
fn quantcheck_delete_logits(s: State<App>, id: String) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::quantcheck::delete_logits(&project, &id).map_err(|e| e.to_string())?;
    Ok(ndforge_core::export::quantcheck::state(&project))
}

/// Copies a run's LoRA add-on (with a README on how to load and merge it) into a folder the user chose. Returns the new folder.
#[tauri::command(async)]
fn copy_adapter(s: State<App>, run: String, dest: String) -> Out<String> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::runs::copy_adapter(&project, &run, std::path::Path::new(&dest)).map(|p| p.display().to_string()).map_err(|e| e.to_string())
}

/// Puts an export into an Ollama server and remembers it.
#[tauri::command]
fn ollama_install(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, export: String, url: String, name: String) -> Out<()> {
    let m = lock(&s);
    let (gguf, mmproj, family, num_ctx) = m.export_for_install(&export).map_err(|e| e.to_string())?;
    let project = m.project_clone().map_err(|e| e.to_string())?;
    drop(m);
    spawn_task(app, &tasks, "export-event", move |_cancel, emit| {
        ndforge_core::export::install::install(&url, &name, &gguf, &family, num_ctx, emit, mmproj.as_deref()).map_err(|e| e.to_string())?;
        ndforge_core::export::runs::record_install(&project, &export, &url, &name).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"installed": name, "url": url}))
    })
}

/// Asks the installed model one of the training's own questions.
#[tauri::command]
fn ollama_probe(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, url: String, name: String) -> Out<()> {
    let (system, question) = lock(&s).probe_question().map_err(|e| e.to_string())?;
    spawn_task(app, &tasks, "export-event", move |_cancel, _emit| {
        let answer = ndforge_core::export::install::probe(&url, &name, &system, &question).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"probe": {"question": question, "answer": answer}}))
    })
}

/// Is Ollama there? Answers as an "export-event" done/error.
#[tauri::command]
fn ollama_check(app: tauri::AppHandle, tasks: State<Tasks>, url: String) -> Out<()> {
    spawn_task(app, &tasks, "export-event", move |_cancel, _emit| {
        let o = ndforge_core::export::ollama::Ollama::new(&url);
        let version = o.version().map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"ollama_version": version, "models": o.models().unwrap_or_default()}))
    })
}

#[tauri::command]
fn ollama_server_add(s: State<App>, name: String, url: String) -> Out<Value> {
    let mut m = lock(&s);
    m.add_ollama_server(&name, &url).map_err(|e| e.to_string())?;
    m.export_state().map_err(|e| e.to_string())
}

#[tauri::command]
fn ollama_server_remove(s: State<App>, url: String) -> Out<Value> {
    let mut m = lock(&s);
    m.remove_ollama_server(&url).map_err(|e| e.to_string())?;
    m.export_state().map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn retrain_info(s: State<App>) -> Out<Value> {
    lock(&s).retrain_info().map_err(|e| e.to_string())
}

/// "Retrain with the new export": new data version, then a new training run, then (optionally) the files and the Ollama install.
/// Everything is reported on the "train-event" channel; stages are {"kind": "stage", "text": ...}.
#[tauri::command]
fn retrain_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, from_previous: bool, export_after: bool, install_url: Option<String>, ollama_name: Option<String>) -> Out<String> {
    let job = lock(&s).retrain_job(from_previous).map_err(|e| e.to_string())?;
    let name = job.name.clone();
    spawn_job(app, &tasks, JobOpts { kind: "train", ..JobOpts::new("train-event") }, move |stop, emit| {
        emit(serde_json::json!({"kind": "stage", "text": "Training the new run"}));
        let train = run_training(&job, stop.clone(), emit);
        match train {
            Err(e) => Err(e.to_string()),
            Ok(result) if result["stopped"] == true => Ok(result),
            Ok(result) => (|| {
                if export_after {
                    emit(serde_json::json!({"kind": "stage", "text": "Making the files for nd-world"}));
                    let quant = ndforge_core::export::runs::list_exports(&job.project).first().and_then(|m| m["quant"].as_str().map(String::from)).unwrap_or_else(|| "q4_k_m".into());
                    let meta = ndforge_core::export::runs::execute(&job.project, &job.name, &quant, 8192, &stop, emit).map_err(|e| e.to_string())?;
                    if let Some(url) = install_url.filter(|u| !u.is_empty()) {
                        emit(serde_json::json!({"kind": "stage", "text": "Installing into Ollama"}));
                        let export = meta["name"].as_str().unwrap_or("").to_string();
                        let name = ollama_name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| meta["ollama_name"].as_str().unwrap_or("world-gm").to_string());
                        let gguf = job.project.folder.join("exports").join(&export).join(meta["gguf_file"].as_str().unwrap_or("model.gguf"));
                        let mmproj = meta["mmproj_file"].as_str().map(|f| gguf.parent().unwrap_or(&job.project.folder).join(f)).filter(|p| p.is_file());
                        ndforge_core::export::install::install(&url, &name, &gguf, &job.model.family, 8192, emit, mmproj.as_deref()).map_err(|e| e.to_string())?;
                        ndforge_core::export::runs::record_install(&job.project, &export, &url, &name).map_err(|e| e.to_string())?;
                    }
                }
                Ok(result)
            })(),
        }
    })?;
    Ok(name)
}

#[tauri::command]
fn train_server_add(s: State<App>, name: String, url: String, token: String) -> Out<Value> {
    let mut m = lock(&s);
    m.add_train_server(&name, &url, &token).map_err(|e| e.to_string())?;
    m.train_state(Some(&ndforge_core::export::ollama::normalize_url(&url))).map_err(|e| e.to_string())
}

#[tauri::command]
fn train_server_remove(s: State<App>, url: String) -> Out<Value> {
    let mut m = lock(&s);
    m.remove_train_server(&url).map_err(|e| e.to_string())?;
    m.train_state(None).map_err(|e| e.to_string())
}

/// Asks a training server what it has (card, busy, Ollama); answers as a "server-event" done/error and remembers the card for the plan.
#[tauri::command]
fn train_server_check(app: tauri::AppHandle, s: State<App>, url: String) -> Out<()> {
    let target = lock(&s).target_for(&url).map_err(|e| e.to_string())?;
    let handle = app.clone();
    std::thread::spawn(move || {
        let result = ndforge_core::remote::client::RemoteClient::new(&target.url, &target.token).status();
        let ev = match result {
            Ok(st) => {
                let model: State<App> = handle.state();
                lock(&model).set_remote_status(&target.url, st.clone());
                serde_json::json!({"kind": "done", "url": target.url, "data": st})
            }
            Err(e) => serde_json::json!({"kind": "error", "url": target.url, "text": e.to_string()}),
        };
        let _ = handle.emit("server-event", ev);
    });
    Ok(())
}

/// Remembers the Unsloth to train in. The key goes to the secrets file only.
#[tauri::command]
fn unsloth_set(s: State<App>, name: String, url: String, key: String) -> Out<Value> {
    let mut m = lock(&s);
    m.set_unsloth(&name, &url, &key).map_err(|e| e.to_string())?;
    Ok(m.state())
}

#[tauri::command]
fn unsloth_remove(s: State<App>) -> Out<Value> {
    let mut m = lock(&s);
    m.remove_unsloth().map_err(|e| e.to_string())?;
    Ok(m.state())
}

/// "Test" (and "Find" when `find` is true) in Training setup. `url` and `key` are what is typed in the form; an empty key means the saved one for that address.
/// The answer arrives as an "unsloth-event": {"kind": "done", "data": {probe, hardware, summary, url}} or {"kind": "error", "text"}. The card is remembered for the plan.
#[tauri::command]
fn unsloth_check(app: tauri::AppHandle, s: State<App>, url: String, key: String, find: bool) -> Out<()> {
    let saved = {
        let m = lock(&s);
        m.settings.unsloth.as_ref().and_then(|u| ndforge_core::secrets::get(&format!("unsloth:{}", u.url)).map(|k| (u.url.clone(), k)))
    };
    let handle = app.clone();
    std::thread::spawn(move || {
        let result = (|| -> ndforge_core::Result<serde_json::Value> {
            let url = if find || url.trim().is_empty() {
                let mut candidates = vec![url.trim().to_string()].into_iter().filter(|u| !u.is_empty()).collect::<Vec<_>>();
                candidates.extend(ndforge_core::unsloth::client::candidate_urls());
                ndforge_core::unsloth::client::find(&candidates).map(|(u, _)| u)
                    .ok_or_else(|| ndforge_core::Error::new("Unsloth does not answer on this computer. Open Unsloth Desktop (it serves only while it is open), then press Find again or type the address from Unsloth -> Settings -> API."))?
            } else {
                url
            };
            let norm = ndforge_core::export::ollama::normalize_url(&url);
            let key = if key.trim().is_empty() { saved.filter(|(u, _)| *u == norm).map(|(_, k)| k).unwrap_or_default() } else { key };
            if key.trim().is_empty() {
                // found, but no key yet: say what was found
                let p = ndforge_core::unsloth::client::UnslothClient::new(&norm, "").probe()?;
                return Ok(serde_json::json!({"probe": p, "url": norm, "summary": format!("Unsloth {} found at {norm}. Create an API key in Unsloth (Settings -> API) and paste it here.", if p.desktop { "Desktop" } else { "Studio" })}));
            }
            ndforge_core::unsloth::client::check(&norm, &key)
        })();
        let ev = match result {
            Ok(data) => {
                if let Some(hw) = data.get("hardware") {
                    let model: State<App> = handle.state();
                    lock(&model).set_remote_status(ndforge_core::unsloth::run::SERVER_ID, hw.clone());
                }
                serde_json::json!({"kind": "done", "data": data})
            }
            Err(e) => serde_json::json!({"kind": "error", "text": e.to_string()}),
        };
        let _ = handle.emit("unsloth-event", ev);
    });
    Ok(())
}

#[tauri::command]
fn help(key: String) -> String {
    help_html(&key)
}

#[tauri::command(async)]
fn env_status() -> Value {
    ndforge_core::envmgr::tasks::status()
}

/// The checklist before a job starts (preflight.rs): `{"checks": [{id, label, light, detail, fix}], "can_start": bool}` for a kind of job (auto, heal, export, think_check,
/// quantcheck, imatrix_make, cut, train) and what the page knows about it (`spec`).
#[tauri::command(async)]
fn preflight(s: State<App>, kind: String, spec: Value) -> Out<Value> {
    let project = lock(&s).project.clone();
    let checks = ndforge_core::preflight::check(project.as_ref(), &kind, &spec);
    Ok(serde_json::json!({"checks": checks, "can_start": ndforge_core::preflight::can_start(&checks)}))
}

/// The optional webhook (an ntfy-style address that gets one line when a job ends: its kind and how it ended). An empty text turns it off.
#[tauri::command(async)]
fn webhook_set(s: State<App>, url: String) -> Out<Value> {
    let url = url.trim().to_string();
    if !url.is_empty() && !ndforge_core::notify::valid_webhook(&url) {
        return Err("The address must start with http:// or https:// and have no spaces.".into());
    }
    let mut m = lock(&s);
    m.settings.webhook_url = url;
    m.settings.save();
    Ok(m.state())
}

/// The memory of this computer's first graphics card in GB (None when there is none or it cannot be read).
fn card_vram_gb() -> Option<f64> {
    ndforge_core::envmgr::gpu::detect_gpus_cached().ok().and_then(|g| g.first().map(|x| x.vram_gb()))
}

// ------------------------------------------------------------------------------------------------ slim or grow a model

#[tauri::command(async)]
fn reshape_state(s: State<App>) -> Out<Value> {
    lock(&s).reshape_state().map_err(|e| e.to_string())
}

/// The sizes a model can be slimmed or grown to. The first look at a model may ask the worker (and the internet), so it runs on its own thread.
/// `shape`: "depth" (default), "width" or "both"; `lin_v_heads` / `heads`: the Expert axes of thinner layers.
#[tauri::command]
async fn reshape_options(s: State<'_, App>, model: String, mode: String, shape: Option<String>, lin_v_heads: Option<u32>, heads: Option<u32>) -> Out<Value> {
    lock(&s).reshape_options_ready().map_err(|e| e.to_string())?;
    let shape = shape.unwrap_or_else(|| "depth".into());
    tauri::async_runtime::spawn_blocking(move || AppModel::reshape_options_with(&model, &mode, &shape, lin_v_heads, heads).map_err(|e| e.to_string())).await.map_err(|e| e.to_string())?
}

/// Is the software here new enough for a model that needs more than the usual (Inkling-2-Mini), and, with `check_repo`, has its repository changed since the program was tested?
#[tauri::command]
async fn moe_compat(model: String, check_repo: bool) -> Out<Value> {
    tauri::async_runtime::spawn_blocking(move || AppModel::moe_compat(&model, check_repo).map_err(|e| e.to_string())).await.map_err(|e| e.to_string())?
}

/// The recommendations panel for a mixture of experts: the order of work for a goal ("size", "speed" or "train") on this card and disk.
#[tauri::command]
async fn moe_recommend(model: String, goal: String, target_gb: Option<f64>) -> Out<Value> {
    tauri::async_runtime::spawn_blocking(move || AppModel::moe_recommend(&model, &goal, target_gb).map_err(|e| e.to_string())).await.map_err(|e| e.to_string())?
}

/// What an earlier measurement of a model on the saved data looks like (is there one to reuse?).
#[tauri::command(async)]
fn reshape_measurement(s: State<App>, model: String, samples: u32, len: u32, load: String) -> Out<Value> {
    let m = ndforge_core::train::catalog::find(&model).ok_or("That model is not in the list any more.")?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::reshape::runs::measurement_info(&p, &m, samples, len, &load))
}

/// Measures a model on the project's text now and keeps the result (optional: making a thinner model measures by itself). Events on "reshape-event".
#[tauri::command]
fn reshape_measure(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, model: String, samples: u32, len: u32, load: String, force: Option<bool>) -> Out<()> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_job(app, &tasks, JobOpts { phase: Some("measure"), awake: awake_if_set(), kind: "measure", ..JobOpts::new("reshape-event") }, move |stop, emit| {
        ndforge_core::reshape::runs::measure(&p, &model, samples, len, &load, force.unwrap_or(false), stop, emit).map_err(|e| e.to_string())
    })
}

/// "Try first": how many cuts a request needs, how many are kept already and how long the rest takes. Loads nothing; runs on its own thread (the worker is started).
#[tauri::command]
async fn reshape_try_plan(s: State<'_, App>, spec: Value) -> Out<Value> {
    let spec: ndforge_core::reshape::runs::TrySpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    tauri::async_runtime::spawn_blocking(move || ndforge_core::reshape::runs::try_plan(&p, &spec, &std::sync::atomic::AtomicBool::new(false), &mut |_| {}).map_err(|e| e.to_string()))
        .await.map_err(|e| e.to_string())?
}

/// The answers kept by earlier tries of this model on the saved data with these settings.
#[tauri::command(async)]
fn reshape_try_results(s: State<App>, model: String, samples: u32, len: u32, load: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::runs::try_results(&p, &model, samples, len, &load).map_err(|e| e.to_string())
}

/// Tries the cuts of the spec that are not kept yet. Events on "reshape-event" tagged {"phase": "try"} (`try_pass` after every cut); the end is {"kind": "done"}.
/// Stop (`reshape_stop`) keeps the finished cuts.
#[tauri::command]
fn reshape_try_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::reshape::runs::TrySpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_job(app, &tasks, JobOpts { phase: Some("try"), tag_events: true, awake: awake_if_set(), kind: "try", ..JobOpts::new("reshape-event") }, move |stop, emit| {
        ndforge_core::reshape::runs::try_run(&p, &spec, stop, emit).map_err(|e| e.to_string())
    })
}

// ------------------------------------------------------------------------------------------------ the automatic best-cut search (docs/TRY_FIRST.md)

/// The plan of a search for the model the Slim page shows (`inspect` is its description): the cuts that reach the target, the rounds, the passes and the time, what fits the budget.
/// Runs no job.
#[tauri::command(async)]
fn reshape_search_plan(spec: Value, inspect: Value) -> Out<Value> {
    let spec: ndforge_core::reshape::search::SearchSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let ins: ndforge_core::reshape::plan::Inspect = serde_json::from_value(inspect).map_err(|e| format!("The model's description is not valid: {e}"))?;
    ndforge_core::reshape::search::plan(&ins, &spec, card_vram_gb(), &ndforge_core::timing::History::recent(60)).map(|p| serde_json::json!(p)).map_err(|e| e.to_string())
}

/// Runs a search (`resume`: the id of a stopped one). Events on "reshape-event" tagged {"phase": "search"}: the worker's (`try_pass` with the round, the time left and the best cut so far)
/// and {"kind": "search", "status": "measuring" | "confirming" | "writing_cut"}; the end is {"kind": "done", "data": <search.json>}. Stop is `reshape_stop`; what was measured is kept.
#[tauri::command]
fn reshape_search_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value, resume: Option<String>) -> Out<()> {
    let spec: ndforge_core::reshape::search::SearchSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let opts = JobOpts { phase: Some("search"), tag_events: true, awake: awake_if_set(), kind: "try", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| {
        let mut fx = ndforge_core::reshape::auto_fx::RealEffects::new(&p, stop, card_vram_gb(), emit);
        ndforge_core::reshape::search::run(&p, &mut fx, &spec, resume.as_deref()).map_err(|e| e.to_string())
    })
}

/// The newest search of the project (`search.json`), or, for a search that was stopped before it finished, what it measured so far (`spec` and `inspect` as for the plan).
#[tauri::command(async)]
fn reshape_search_status(s: State<App>, spec: Option<Value>, inspect: Option<Value>) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let saved = ndforge_core::reshape::search::latest(&p);
    let partial = match (spec, inspect) {
        (Some(sp), Some(ins)) => {
            let sp: ndforge_core::reshape::search::SearchSpec = serde_json::from_value(sp).map_err(|e| format!("The choice is not valid: {e}"))?;
            let ins: ndforge_core::reshape::plan::Inspect = serde_json::from_value(ins).map_err(|e| format!("The model's description is not valid: {e}"))?;
            ndforge_core::reshape::search::partial(&p, &sp, &ins, card_vram_gb()).ok()
        }
        _ => None,
    };
    Ok(serde_json::json!({"search": saved, "partial": partial}))
}

/// The thinking check (docs/THINKING_CHECK.md): what the result page shows for a made model (saved numbers of every model checked, the gates, the GGUF files of the exports).
#[tauri::command(async)]
fn think_check_results(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::thinkcheck::results(&p, &name).map_err(|e| e.to_string())
}

/// The experimental vLLM engine (docs/VLLM.md): which way to run it was found (docker | docker_wsl | wslc | wsl | url | missing), the driver and WSL lines and the guided install text.
/// Asks a few small programs, so it runs on its own thread. Nothing is started or installed.
#[tauri::command]
async fn vllm_status(s: State<'_, App>) -> Out<Value> {
    let v = lock(&s).settings.vllm.clone();
    tauri::async_runtime::spawn_blocking(move || ndforge_core::vllm::status(&v, &ndforge_core::vllm::real_runner)).await.map_err(|e| e.to_string())
}

/// Saves the vLLM engine settings (off by default); returns the new state.
#[tauri::command(async)]
fn vllm_set(s: State<App>, vllm: Value) -> Out<Value> {
    let v: ndforge_core::settings::VllmSettings = serde_json::from_value(vllm).map_err(|e| format!("The settings are not valid: {e}"))?;
    let mut m = lock(&s);
    m.vllm_set(v).map_err(|e| e.to_string())?;
    Ok(m.state())
}

/// "Make thinking answers with the original model" (docs/VLLM.md, worker job `trace_gen`): what is on disk for a made model (`{"file", "count", "rejected", "bytes"}`).
#[tauri::command(async)]
fn trace_gen_state(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::tracegen::state(&p, &name).map_err(|e| e.to_string())
}

/// Writes thinking answers for prompts with the original (or a model of your choice). The way (vLLM when it is switched on, else llama-server, else the standard engine) is the worker's choice.
/// Events on "reshape-event" tagged {"phase": "trace"}; the end is {"kind": "done", "data": {...}}; Stop is `reshape_stop`; what was written stays.
#[tauri::command]
fn trace_gen_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let mut spec: ndforge_core::reshape::tracegen::TraceSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let (v, keep_awake) = (m.settings.vllm.clone(), m.settings.keep_awake);
    drop(m);
    spec.engine = (v.enabled && spec.backend.as_deref().is_none_or(|b| b == "auto" || b == "vllm")).then(|| ndforge_core::vllm::engine_job_with_card(&v));
    if spec.backend.as_deref() == Some("vllm") && spec.engine.is_none() {
        return Err("The vLLM engine (experimental) is switched off. Switch it on in Training setup, or choose another way.".into());
    }
    let opts = JobOpts { phase: Some("trace"), tag_events: true, awake: if keep_awake { Awake::Quiet } else { Awake::Off }, kind: "trace_gen", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::reshape::tracegen::run(&p, &spec, stop, emit).map_err(|e| e.to_string()))
}

/// The vLLM self-test (docs/VLLM.md): the research report's experiment, run for the person (prerequisites, GPU in a container, engine start, the thinking suite with 40 requests at once,
/// a plain verdict; `vllm_selftest.json/.md`). Events on "reshape-event" tagged {"phase": "vllm"}; Stop is `reshape_stop`; the engine is stopped in every end.
#[tauri::command]
fn vllm_selftest_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    clean_stale_engines(&app, &tasks);
    let m = lock(&s);
    let (v, project, keep_awake) = (m.settings.vllm.clone(), m.project.clone(), m.settings.keep_awake);
    drop(m);
    let job = ndforge_core::vllm::selftest_job(&v, &spec, project.as_ref()).map_err(|e| e.to_string())?;
    let opts = JobOpts { phase: Some("vllm"), tag_events: true, awake: if keep_awake { Awake::Quiet } else { Awake::Off }, kind: "vllm_selftest", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::vllm::selftest_run(&job, stop, emit).map_err(|e| e.to_string()))
}

/// "Test the container copy" (Heal card): the idle container, a 1 KB file in and out by every way (bind mount, cp, stdin), the container removed; says which way works on this PC. A few seconds to a minute.
/// Events on "reshape-event" tagged {"phase": "vllm_probe"}.
#[tauri::command]
fn vllm_container_probe_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>) -> Out<()> {
    clean_stale_engines(&app, &tasks);
    let m = lock(&s);
    let (v, keep_awake) = (m.settings.vllm.clone(), m.settings.keep_awake);
    drop(m);
    let job = ndforge_core::vllm::container_probe_job(&v);
    let opts = JobOpts { phase: Some("vllm_probe"), tag_events: true, awake: if keep_awake { Awake::Quiet } else { Awake::Off }, kind: "vllm_container_probe", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::vllm::container_probe_run(&job, stop, emit).map_err(|e| e.to_string()))
}

/// "Check the vLLM teacher": compares it with the standard 4-bit teacher on 100 real rows of the heal's data (docs/VLLM.md). Events on "reshape-event" tagged {"phase": "vllm_teacher_check"}.
#[tauri::command]
fn vllm_teacher_check_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, name: String, spec: Value) -> Out<()> {
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let keep_awake = m.settings.keep_awake;
    drop(m);
    let opts = JobOpts { phase: Some("vllm_teacher_check"), awake: if keep_awake { Awake::Quiet } else { Awake::Off }, kind: "vllm_teacher_check", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::reshape::runs::vllm_teacher_check(&p, &name, &spec, card_vram_gb(), stop, std::time::Duration::from_secs(120), emit).map_err(|e| e.to_string()))
}

/// Where the self-test's report is, and its text (for the card: inline, the folder as text, copy and open buttons).
#[tauri::command(async)]
fn vllm_selftest_files() -> Value {
    ndforge_core::vllm::selftest_files()
}

/// "What if I change these settings and continue?" for a stopped or failed Auto-cut run (nothing is changed): the diff, the locks, what needs a confirmation, the time left.
#[tauri::command(async)]
fn auto_change_preview(s: State<App>, id: String, patch: Value) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let m = match patch { Value::Object(m) => m, _ => serde_json::Map::new() };
    ndforge_core::reshape::auto::change_preview(&p, &id, &m, card_vram_gb()).map_err(|e| e.to_string())
}

/// The report of the last vLLM self-test (null when there is none).
#[tauri::command(async)]
fn vllm_selftest_report() -> Value {
    ndforge_core::vllm::selftest_report()
}

/// A thinking-check choice with the vLLM engine settings put in (or a plain message when the engine is switched off in Settings).
fn with_vllm(s: &State<App>, mut spec: ndforge_core::reshape::thinkcheck::ThinkSpec) -> Result<ndforge_core::reshape::thinkcheck::ThinkSpec, String> {
    if ndforge_core::reshape::thinkcheck::uses_vllm(&spec) {
        let v = lock(s).settings.vllm.clone();
        if !v.enabled {
            return Err("The vLLM engine (experimental) is switched off. Switch it on in Settings, vLLM engine, or choose the standard engine.".into());
        }
        let e = ndforge_core::vllm::engine_job_with_card(&v);          // the pick of the weights (16-bit, 8-bit, pre-quantized) depends on the card
        spec.engine = Some(e);
    } else {
        spec.engine = None;
    }
    Ok(spec)
}

/// What a check would do (question counts, settings, a rough size) without loading anything; runs on its own thread (the worker is started).
#[tauri::command]
async fn think_check_plan(spec: Value) -> Out<Value> {
    let spec: ndforge_core::reshape::thinkcheck::ThinkSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    tauri::async_runtime::spawn_blocking(move || ndforge_core::reshape::thinkcheck::plan(&spec, &std::sync::atomic::AtomicBool::new(false)).map_err(|e| e.to_string()))
        .await.map_err(|e| e.to_string())?
}

/// Checks one model. Events on "reshape-event" tagged {"phase": "think"} (`think_item` after every answered question); the end is {"kind": "done", "data": {...}}.
/// Stop is `reshape_stop`.
#[tauri::command]
fn think_check_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::reshape::thinkcheck::ThinkSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let spec = with_vllm(&s, spec)?;
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let keep_awake = m.settings.keep_awake;
    drop(m);
    let opts = JobOpts { phase: Some("think"), tag_events: true, awake: if keep_awake { Awake::Quiet } else { Awake::Off }, kind: "think_check", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::reshape::thinkcheck::run(&p, &spec, stop, emit).map_err(|e| e.to_string()))
}

// ------------------------------------------------------------------------------------------------ Auto-cut (docs/AUTO_CUT.md)

/// The plan of an Auto-cut for the model the Slim page shows (`inspect` is its `reshape_options` answer's `inspect`): sizes, stages, card, disk, blockers. Runs no job.
#[tauri::command(async)]
fn auto_plan(s: State<App>, spec: Value, inspect: Value) -> Out<Value> {
    let spec: ndforge_core::reshape::auto_plan::AutoSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let ins: ndforge_core::reshape::plan::Inspect = serde_json::from_value(inspect).map_err(|e| format!("The model's description is not valid: {e}"))?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let free = ndforge_core::envmgr::health::free_disk_gb(&ndforge_core::reshape::runs::models_dir(&p));
    let has_data = ndforge_core::train::runs::latest_version(&p).is_some();
    Ok(ndforge_core::reshape::auto::preview(&ins, &spec, card_vram_gb(), free, has_data))
}

/// Starts an Auto-cut (`resume` null) or continues a stopped or failed one (`resume` is its run id; `spec` is then ignored). Events on "reshape-event" tagged {"phase": "auto"}:
/// the run's own {"kind": "auto", "auto_step", "status", "note"} and the jobs' events with "auto_step" added; the end is {"kind": "done", "data": <the run's state>} or {"kind": "error"}.
#[tauri::command]
fn auto_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value, resume: Option<String>, changes: Option<Value>, confirmed: Option<bool>) -> Out<()> {
    let changes: Option<(serde_json::Map<String, Value>, bool)> = match (&resume, changes) {
        (Some(_), Some(Value::Object(m))) if !m.is_empty() => Some((m, confirmed.unwrap_or(false))),
        _ => None,
    };
    let spec: Option<ndforge_core::reshape::auto_plan::AutoSpec> = if resume.is_some() { None } else { Some(serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?) };
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let keep_awake = m.settings.keep_awake;
    let toasts = m.settings.notifications;
    drop(m);
    // what the stage notifications need to know: the setting of the run (a continued run keeps its own) and whether it thinks
    let (notify_stages, thinking) = match (&spec, &resume) {
        (Some(sp), _) => (sp.notify_stages, sp.keep_thinking),
        (None, Some(id)) => ndforge_core::reshape::auto::load(&ndforge_core::reshape::auto::run_dir(&p, id)).map(|st| (st.spec.notify_stages, st.spec.keep_thinking)).unwrap_or((true, true)),
        _ => (true, true),
    };
    let opts = JobOpts { phase: Some("auto"), tag_events: true, awake: if keep_awake { Awake::Quiet } else { Awake::Off }, kind: "auto", ..JobOpts::new("reshape-event") };
    let handle = app.clone();
    spawn_job(app, &tasks, opts, move |stop, emit| {
        let mut bar = AutoBar::new(&handle, toasts && notify_stages, thinking);
        let mut wrapped = |v: Value| {
            bar.see(&v);
            emit(v);
        };
        let mut fx = ndforge_core::reshape::auto_fx::RealEffects::new(&p, stop, card_vram_gb(), &mut wrapped);
        let r = match (&resume, spec) {
            (Some(id), _) => ndforge_core::reshape::auto::resume_with(&p, &mut fx, id, changes),
            (None, Some(sp)) => ndforge_core::reshape::auto::start(&p, &mut fx, sp),
            _ => Err(ndforge_core::error::Error::new("Nothing to start.")),
        };
        drop(fx);
        bar.reset();
        r.map(|st| serde_json::json!(st)).map_err(|e| e.to_string())
    })
}

/// What an Auto-cut shows outside the page: the window title and the taskbar progress (Windows; the other systems ignore the bar), refreshed every 5 seconds, and the
/// notification at the end of a stage. Both are put back when the run ends.
struct AutoBar {
    app: tauri::AppHandle,
    title0: Option<String>,
    last: Option<std::time::Instant>,
    snap: Value,
    notify: bool,
    thinking: bool,
    shown: bool,
}

impl AutoBar {
    fn new(app: &tauri::AppHandle, notify: bool, thinking: bool) -> Self {
        let title0 = app.get_webview_window("main").and_then(|w| w.title().ok());
        AutoBar { app: app.clone(), title0, last: None, snap: Value::Null, notify, thinking, shown: false }
    }

    fn see(&mut self, v: &Value) {
        use tauri::window::{ProgressBarState, ProgressBarStatus};
        if v["kind"] == "auto_progress" {
            self.snap = v.clone();
            if self.last.map(|t| t.elapsed().as_secs() >= 5).unwrap_or(true) {
                self.last = Some(std::time::Instant::now());
                if let Some(w) = self.app.get_webview_window("main") {
                    let _ = w.set_title(&ndforge_core::reshape::auto_progress::title_of(v));
                    let _ = w.set_progress_bar(ProgressBarState { status: Some(ProgressBarStatus::Normal), progress: Some(v["pct"].as_f64().unwrap_or(0.0).clamp(0.0, 100.0).round() as u64) });
                    self.shown = true;
                }
            }
        } else if self.notify && v["kind"] == "heal_round" {
            // a heal in rounds: only a stop decision or the verdict on the merged model is told (not every round)
            if let Some((t, b)) = ndforge_core::notify::heal_round_toast(v) {
                Toast(self.app.clone()).show(&t, &b);
            }
        } else if self.notify && v["kind"] == "auto" && v["status"] == "done" {
            let id = v["auto_step"].as_str().unwrap_or("");
            if let (Some(n), Some(of)) = (ndforge_core::notify::auto_stage_end(id, self.thinking), self.snap["stage"]["of"].as_u64()) {
                if (n as u64) < of {
                    let left = (self.snap["state"] == "live").then(|| self.snap["left_s"][0].as_f64().zip(self.snap["left_s"][1].as_f64()).map(|(a, b)| (a + b) / 120.0)).flatten();
                    let (t, b) = ndforge_core::notify::stage_toast(n, of as u32, self.snap["pct"].as_f64().unwrap_or(0.0), left);
                    Toast(self.app.clone()).show(&t, &b);
                }
            }
        }
    }

    fn reset(&mut self) {
        use tauri::window::{ProgressBarState, ProgressBarStatus};
        if !self.shown {
            return;
        }
        if let Some(w) = self.app.get_webview_window("main") {
            let _ = w.set_title(self.title0.as_deref().unwrap_or("ND Lore Forge"));
            let _ = w.set_progress_bar(ProgressBarState { status: Some(ProgressBarStatus::None), progress: None });
        }
    }
}

/// Stop: the running job ends at its next safe point, the run keeps its state and can be continued.
#[tauri::command]
fn auto_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

/// The newest run (its whole state: steps, stages, checks) and whether a job is running. A run that was still "running" when the program closed is marked stopped first.
#[tauri::command(async)]
fn auto_status(s: State<App>, tasks: State<Tasks>) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let running = tasks.busy.load(Ordering::SeqCst);
    if !running {
        ndforge_core::reshape::auto::mark_interrupted(&p);
    }
    let share = lock(&s).settings.studio_share.clone();
    Ok(serde_json::json!({"run": ndforge_core::reshape::auto::latest(&p), "running": running, "studio_share": share}))
}

/// The report of a run (the newest when `run` is null): data and Markdown.
#[tauri::command(async)]
fn auto_report(s: State<App>, run: Option<String>) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::auto::report_of(&p, run.as_deref()).map_err(|e| e.to_string())
}

/// Saves the Studio share (where Auto-cut puts the GGUF files). Nothing here is a secret.
#[tauri::command(async)]
fn studio_share_set(s: State<App>, share: Value) -> Out<Value> {
    let sh: ndforge_core::settings::StudioShare = serde_json::from_value(share).map_err(|e| format!("The choice is not valid: {e}"))?;
    let mut m = lock(&s);
    m.settings.studio_share = Some(sh).filter(|x| !x.host_path.trim().is_empty() || !x.container_path.trim().is_empty());
    m.settings.save();
    Ok(m.state())
}

/// "Check" next to the Studio share: is the folder there, and (with an API key for Studio) does Studio's outputs folder match it? Returns {"folder", "detected"?, "note"}.
#[tauri::command]
async fn studio_share_check(s: State<'_, App>, share: Value) -> Out<Value> {
    let sh: ndforge_core::settings::StudioShare = serde_json::from_value(share).map_err(|e| format!("The choice is not valid: {e}"))?;
    let un = lock(&s).settings.unsloth.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let host = std::path::PathBuf::from(sh.host_path.trim());
        if sh.host_path.trim().is_empty() || !host.is_dir() {
            return Ok(serde_json::json!({"folder": false, "note": format!("The folder {} was not found. Is the drive connected?", sh.host_path.trim())}));
        }
        let key = un.as_ref().and_then(|u| ndforge_core::secrets::get(&format!("unsloth:{}", u.url)).map(|k| (u.url.clone(), k)));
        let Some((url, key)) = key else {
            return Ok(serde_json::json!({"folder": true, "note": "The folder is there. Without an Unsloth address and API key (Training setup) Studio cannot be asked where its folders are: type the path inside the container yourself."}));
        };
        let c = ndforge_core::unsloth::client::UnslothClient::new(&url, &key);
        match ndforge_core::unsloth::publish::detect(&c, &host) {
            Ok(d) => Ok(serde_json::json!({"folder": true, "detected": {"outputs_dir": d.outputs_dir, "container_root": d.container_root, "matched": d.matched, "confirmed": d.confirmed},
                "note": match d.confirmed { Some(true) => "This is Studio's folder.", Some(false) => "Studio lists run folders that are not in this folder: it is probably not the one Studio uses.", None => "Studio lists no run folders yet, so the folder could not be compared." }})),
            Err(e) => Ok(serde_json::json!({"folder": true, "note": e.to_string()})),
        }
    }).await.map_err(|e| e.to_string())?
}

/// What the Heal card shows: how healing would run on this card, the budgets and where it stands.
#[tauri::command(async)]
fn heal_options(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::runs::heal_options(&p, &name, card_vram_gb()).map_err(|e| e.to_string())
}

/// The live memory estimate of the Heal card for the settings chosen so far.
#[tauri::command(async)]
fn heal_estimate(s: State<App>, name: String, spec: Value) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    ndforge_core::reshape::runs::heal_estimate(&p, &name, &spec, card_vram_gb()).map_err(|e| e.to_string())
}

/// Heals a made model (or continues where a stopped heal was). Events on "reshape-event" tagged {"phase": "heal"}; the end is {"kind": "done", "data": {...}}.
#[tauri::command]
fn heal_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, name: String, spec: Value) -> Out<()> {
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let keep_awake = m.settings.keep_awake;
    drop(m);
    let opts = JobOpts { phase: Some("heal"), awake: if keep_awake { Awake::Noted } else { Awake::Off }, kind: "heal", ..JobOpts::new("reshape-event") };
    // healing takes hours: the computer stays awake, like during training
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::reshape::runs::heal(&p, &name, &spec, card_vram_gb(), stop, std::time::Duration::from_secs(120), emit).map_err(|e| e.to_string()))
}

/// "Find the fastest loss settings": short real steps of the healing loss with several settings (this computer only). Events on "reshape-event" tagged {"phase": "healbench"}:
/// {"kind": "heal_bench", "index", "total", "name", "row"?}, the end is {"kind": "done", "data": {...}}. The result is also kept in benchmarks/heal-<model>.json.
#[tauri::command]
fn heal_bench_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, name: String, spec: Value) -> Out<()> {
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let opts = JobOpts { phase: Some("healbench"), ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| {
        ndforge_core::reshape::heal_bench::run(&p, &name, &spec, card_vram_gb(), ndforge_core::reshape::heal_bench::STEPS, stop, emit).map_err(|e| e.to_string())
    })
}

/// "Measure heal speed": profiles a few real steps (where the time goes), times one GatedDeltaNet forward + backward, and optionally the original's answers per rows-per-forward; the worker writes
/// `heal_speed_report.md/.json` (docs/HEAL_SPEED.md). Events on "reshape-event" tagged {"phase": "healspeed"}.
#[tauri::command]
fn heal_speed_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, name: String, spec: Value, with_teacher: bool) -> Out<()> {
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let opts = JobOpts { phase: Some("healspeed"), kind: "heal", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::reshape::heal_speed::run(&p, &name, &spec, card_vram_gb(), with_teacher, stop, emit).map_err(|e| e.to_string()))
}

/// The saved heal speed report of a made model (the text, the folder, the Windows hints).
#[tauri::command(async)]
fn heal_speed_report(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let mut f = ndforge_core::reshape::heal_speed::files(&p, &name);
    f["windows_hints"] = ndforge_core::winhints::as_value(&ndforge_core::winhints::hints());
    Ok(f)
}

/// The last speed test of the loss settings of a made model (null when there is none).
#[tauri::command(async)]
fn heal_bench_last(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::reshape::heal_bench::last(&p, &name).unwrap_or(Value::Null))
}

// ---- Heal a model: any model, not only the ones the Slim page made (docs/HEAL_ANY_MODEL.md). Job types and event names are the same as above.

fn heal_source_of(v: Value) -> Result<ndforge_core::reshape::heal_source::Source, String> {
    serde_json::from_value(v).map_err(|e| format!("The choice of model is not valid: {e}"))
}

fn heal_teacher_of(v: Value) -> Result<Option<ndforge_core::reshape::heal_source::Source>, String> {
    if v.is_null() { Ok(None) } else { heal_source_of(v).map(Some) }
}

/// What the pickers of the Heal section offer: the made models, the model list's folders and Hugging Face ids, and the earlier heals of other models.
#[tauri::command(async)]
fn heal_candidates(s: State<App>) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::reshape::heal_source::heal_candidates(&p))
}

/// The Check of the Heal section: the pair compared in plain words. May ask the worker (and Hugging Face) the first time; `refresh` asks again.
#[tauri::command]
async fn heal_check(s: State<'_, App>, student: Value, teacher: Value, refresh: bool) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let (student, teacher) = (heal_source_of(student)?, heal_teacher_of(teacher)?);
    let vram = card_vram_gb();
    tauri::async_runtime::spawn_blocking(move || {
        let ctx = ndforge_core::reshape::heal_source::Ctx { vram_gb: vram, free_disk_gb: ndforge_core::envmgr::health::free_disk_gb(&ndforge_core::reshape::runs::models_dir(&p)) };
        ndforge_core::reshape::heal_source::heal_check(&p, &student, teacher.as_ref(), ctx, refresh, &AtomicBool::new(false), &mut |_| {}).map_err(|e| e.to_string())
    }).await.map_err(|e| e.to_string())?
}

/// The Check and, when the pair passes it, the Heal card's options for it.
#[tauri::command]
async fn heal_pair(s: State<'_, App>, student: Value, teacher: Value, refresh: bool) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let (student, teacher) = (heal_source_of(student)?, heal_teacher_of(teacher)?);
    let vram = card_vram_gb();
    tauri::async_runtime::spawn_blocking(move || {
        ndforge_core::reshape::heal_source::heal_pair(&p, &student, teacher.as_ref(), vram, refresh, &AtomicBool::new(false), &mut |_| {}).map_err(|e| e.to_string())
    }).await.map_err(|e| e.to_string())?
}

/// The live memory estimate for a pair (the Check's answer is kept, so this is quick).
#[tauri::command]
async fn heal_pair_estimate(s: State<'_, App>, student: Value, teacher: Value, spec: Value) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let (student, teacher) = (heal_source_of(student)?, heal_teacher_of(teacher)?);
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let vram = card_vram_gb();
    tauri::async_runtime::spawn_blocking(move || ndforge_core::reshape::heal_source::heal_pair_estimate(&p, &student, teacher.as_ref(), &spec, vram).map_err(|e| e.to_string()))
        .await.map_err(|e| e.to_string())?
}

/// Heals the chosen pair (checks it again, makes the heal's folder, downloads a Hugging Face model first, then the three jobs of `heal_start`). Same events as
/// `heal_start` on "reshape-event"; the end is {"kind": "done", "data": {...}} with the folder's "name" in it.
#[tauri::command]
fn heal_source_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, student: Value, teacher: Value, spec: Value) -> Out<()> {
    let (student, teacher) = (heal_source_of(student)?, heal_teacher_of(teacher)?);
    let spec: ndforge_core::reshape::runs::HealSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let keep_awake = m.settings.keep_awake;
    drop(m);
    let opts = JobOpts { phase: Some("heal"), awake: if keep_awake { Awake::Noted } else { Awake::Off }, kind: "heal", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| {
        ndforge_core::reshape::heal_source::heal_from_source(&p, &student, teacher.as_ref(), &spec, card_vram_gb(), stop, std::time::Duration::from_secs(120), emit).map_err(|e| e.to_string())
    })
}

/// The Stop button of healing: the worker saves a checkpoint and stops; healing continues from it later.
#[tauri::command]
fn heal_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

/// Makes the model. Events on "reshape-event"; the end is {"kind": "done", "data": {... "name", "model_id"}} (or "stopped": true in data).
#[tauri::command]
fn reshape_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<()> {
    let spec: ndforge_core::reshape::plan::ReshapeSpec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let keep_awake = m.settings.keep_awake;
    drop(m);
    // writing a model of tens of gigabytes takes a while: the computer stays awake, like during training
    let opts = JobOpts { awake: if keep_awake { Awake::Noted } else { Awake::Off }, kind: "cut", ..JobOpts::new("reshape-event") };
    spawn_job(app, &tasks, opts, move |stop, emit| ndforge_core::reshape::runs::execute(&p, &spec, stop, emit).map_err(|e| e.to_string()))
}

/// The Stop button: the worker stops between tensors and removes the half-written model.
#[tauri::command]
fn reshape_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

/// Tests a made model against the model it was made from. Events on "reshape-event".
#[tauri::command]
fn reshape_test(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, name: String, judge: bool, target: Option<String>) -> Out<()> {
    let m = lock(&s);
    let p = m.project_clone().map_err(|e| e.to_string())?;
    let teacher = if judge { Some(m.teacher_client()) } else { None };
    drop(m);
    spawn_task(app, &tasks, "reshape-event", move |cancel, emit| {
        if let Some(t) = &teacher { if !t.cfg.is_set() { return Err("Choose the teacher in Step 2 first, or untick the teacher grading.".to_string()); } }
        let target = target.unwrap_or_else(|| "slim".into());
        let r = ndforge_core::reshape::runs::test_target(&p, &name, &target, teacher.as_ref(), card_vram_gb(), cancel, emit).map_err(|e| e.to_string())?;
        if let Some(t) = &teacher { if t.cfg.unload_after { let _ = ndforge_core::export::ollama::Ollama::new(&t.cfg.ollama_base()).unload(&t.cfg.model); } }
        Ok(serde_json::json!({"test": r, "name": name}))
    })
}

/// Deletes a made model (refused while a run was trained on it). Returns the page state.
#[tauri::command(async)]
fn reshape_remove(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::runs::remove(&p, &name).map_err(|e| e.to_string())?;      // deleting tens of GB: not while the lock is held
    lock(&s).reshape_state().map_err(|e| e.to_string())
}

// ------------------------------------------------------------------------------------------------ the job queue (ndforge_core::queue)

fn queue_changed(app: &tauri::AppHandle) {
    let _ = app.emit("queue-event", serde_json::json!({"kind": "changed"}));
}

fn queue_view(app: &tauri::AppHandle) -> Value {
    app.state::<QueueState>().0.lock().unwrap_or_else(|e| e.into_inner()).view()
}

/// Runs `f` on the queue and saves it afterwards.
fn queue_edit<R>(app: &tauri::AppHandle, f: impl FnOnce(&mut ndforge_core::queue::Queue) -> R) -> R {
    let st = app.state::<QueueState>();
    let mut q = st.0.lock().unwrap_or_else(|e| e.into_inner());
    let r = f(&mut q);
    let _ = q.save();
    r
}

/// Starts the job of a queue item the way its page would (the same commands, the same events).
fn queue_start_item(app: &tauri::AppHandle, it: &ndforge_core::queue::Item) -> Out<()> {
    let a = &it.spec;
    let (s, t) = (app.state::<App>(), app.state::<Tasks>());
    let text = |k: &str| a[k].as_str().unwrap_or("").to_string();
    let bad = |e: serde_json::Error| format!("The saved choice is not valid any more: {e}");
    match it.kind.as_str() {
        "auto" => auto_start(app.clone(), s, t, a["spec"].clone(), a["resume"].as_str().map(String::from), a.get("changes").cloned().filter(|c| !c.is_null()), a["confirmed"].as_bool()),
        "export" => export_start(app.clone(), s, t, text("run"), text("quant"), a["num_ctx"].as_u64().or(a["numCtx"].as_u64()).unwrap_or(8192) as u32, a["what"].as_str().map(String::from),
            serde_json::from_value(a["quants"].clone()).map_err(bad)?, serde_json::from_value(a["imatrix"].clone()).map_err(bad)?, serde_json::from_value(a["mixed"].clone()).map_err(bad)?),
        "export_model" => export_model_start(app.clone(), s, t, a["source"].clone(), text("quant"), a["num_ctx"].as_u64().or(a["numCtx"].as_u64()).unwrap_or(8192) as u32,
            serde_json::from_value(a["quants"].clone()).map_err(bad)?, Some(a["thinking"].as_bool().unwrap_or(false)),
            serde_json::from_value(a["imatrix"].clone()).map_err(bad)?, serde_json::from_value(a["mixed"].clone()).map_err(bad)?,
            a["paramsB"].as_f64(), a["downloadGb"].as_f64()),
        "heal" => heal_start(app.clone(), s, t, text("name"), a["spec"].clone()),
        "think_check" => think_check_start(app.clone(), s, t, a["spec"].clone()),
        "quantcheck" => quantcheck_start(app.clone(), s, t, a["spec"].clone()),
        "imatrix_make" => imatrix_create(app.clone(), s, t, a["spec"].clone()),
        "train" => train_start(app.clone(), s, t, text("model"), text("method"), text("preset"), a["overrides"].clone(), a["resume"].as_str().map(String::from), a["server"].as_str().map(String::from)).map(|_| ()),
        other => Err(format!("{other} cannot be queued.")),
    }
}

/// Starts the next waiting item when the queue may (not paused, nothing running): its preflight list first, and an item that fails it is skipped with the reason, then the next is tried.
fn queue_kick(app: &tauri::AppHandle) {
    for _ in 0..1000 {
        let tasks = app.state::<Tasks>();
        if tasks.busy.load(Ordering::SeqCst) {
            return;
        }
        let Some(it) = queue_edit(app, |q| q.next_runnable().map(|i| q.items[i].clone())) else { return };
        let project = app.state::<App>().0.lock().unwrap_or_else(|e| e.into_inner()).project.clone();
        let skip = |why: String| { let _ = queue_edit(app, |q| q.mark(&it.id, "skipped", Some(why), None)); queue_changed(app); };
        let Some(project) = project.filter(|p| p.folder.to_string_lossy() == it.project) else {
            skip(format!("The project {} is not the one that is open. Open it and put the job in the queue again.", it.project));
            continue;
        };
        let checks = ndforge_core::preflight::check(Some(&project), &it.kind, &ndforge_core::queue::preflight_spec(&it.kind, &it.spec));
        if !ndforge_core::preflight::can_start(&checks) {
            let why: Vec<String> = checks.iter().filter(|c| c.light == ndforge_core::envmgr::health::Light::Fail).map(|c| format!("{}: {} {}", c.label, c.detail, c.fix).trim().to_string()).collect();
            skip(why.join(" "));
            continue;
        }
        let _ = queue_edit(app, |q| q.mark(&it.id, "running", None, None));
        match queue_start_item(app, &it) {
            Ok(()) => { queue_changed(app); return; }
            Err(e) if e.starts_with("Another job is still running") => {
                let _ = queue_edit(app, |q| q.mark(&it.id, "queued", None, None));      // somebody else got in first: it waits
                return;
            }
            Err(e) => { let _ = queue_edit(app, |q| q.mark(&it.id, "failed", Some(e), None)); queue_changed(app); }
        }
    }
}

/// Nothing runs and nothing waits in the queue.
fn queue_idle(app: &tauri::AppHandle) -> bool {
    !app.state::<Tasks>().busy.load(Ordering::SeqCst) && queue_edit(app, |q| q.running().is_none() && !q.items.iter().any(|i| i.status == "queued"))
}

/// Shows a message to the person outside the window. The plugin call lives here only: a build or a platform without it changes this one function.
trait Notifier {
    fn show(&self, title: &str, body: &str);
}
struct Toast(tauri::AppHandle);
impl Notifier for Toast {
    fn show(&self, title: &str, body: &str) {
        use tauri_plugin_notification::NotificationExt;
        let _ = self.0.notification().builder().title(title).body(body).show();
    }
}

/// The notifications and the webhook for a job that ended (settings: `notifications`, `webhook_url`).
fn tell_about(app: &tauri::AppHandle, kind: &str, r: &Result<Value, String>, queue_finished: Option<(usize, usize, usize, usize)>) {
    let (on, hook) = {
        let st = app.state::<App>();
        let m = st.0.lock().unwrap_or_else(|e| e.into_inner());
        (m.settings.notifications, m.settings.webhook_url.clone())
    };
    if kind == "task" {
        return;                         // an install or a check of the setup page: the window shows it
    }
    let status = ndforge_core::notify::status_of(r);
    if on {
        let toast = Toast(app.clone());
        let (t, b) = ndforge_core::notify::toast(kind, r);
        toast.show(&t, &b);
        if let Some((d, f, s, x)) = queue_finished {
            let (t, b) = ndforge_core::notify::queue_toast(d, f, s, x);
            toast.show(&t, &b);
        }
    }
    if ndforge_core::notify::valid_webhook(&hook) {
        let kind = kind.to_string();
        std::thread::spawn(move || { ndforge_core::notify::send_webhook(&hook, &kind, status); });
    }
}

/// A job has ended (its end event is sent and the busy flag is free): the queue item that was running gets its result, and the next one starts.
fn job_ended(app: &tauri::AppHandle, _kind: &str, r: &Result<Value, String>) {
    let stopped = match r {
        Ok(data) => data["stopped"] == true,
        Err(msg) => msg == ndforge_core::error::CANCELLED,
    };
    let (status, error) = match r {
        Ok(_) if stopped => ("stopped", None),
        Ok(_) => ("done", None),
        Err(_) if stopped => ("stopped", None),
        Err(msg) => ("failed", Some(msg.clone())),
    };
    let log = error.as_deref().and_then(|m| m.split("(details: ").nth(1)).and_then(|t| t.split(')').next()).map(String::from);
    let had = queue_edit(app, |q| {
        let id = q.running().map(|i| i.id.clone());
        if let Some(id) = &id {
            let _ = q.mark(id, status, error.clone(), log.clone());
            if status == "stopped" {
                q.paused = true;                    // Stop means stop: the rest waits until the person presses Run
            }
        }
        id.is_some()
    });
    if had {
        queue_changed(app);
    }
    queue_kick(app);
    // the queue has run out: say how it went
    let finished = had.then(|| queue_edit(app, |q| {
        let n = |s: &str| q.items.iter().filter(|i| i.status == s).count();
        (q.running().is_none() && n("queued") == 0 && !q.paused).then(|| (n("done"), n("failed"), n("skipped"), n("stopped")))
    })).flatten();
    tell_about(app, _kind, r, finished);
}

#[tauri::command(async)]
fn queue_list(app: tauri::AppHandle) -> Value {
    queue_view(&app)
}

/// Puts a job in the queue: `kind` and the arguments its start command takes (the same ones the Start button sends). Starts at once when nothing runs and the queue is not paused.
#[tauri::command(async)]
fn queue_add(app: tauri::AppHandle, s: State<App>, kind: String, args: Value) -> Out<Value> {
    let folder = lock(&s).project.as_ref().map(|p| p.folder.to_string_lossy().to_string()).ok_or("Open a project first.")?;
    queue_edit(&app, |q| q.enqueue(&folder, &kind, args)).map_err(|e| e.to_string())?;
    queue_changed(&app);
    queue_kick(&app);
    Ok(queue_view(&app))
}

#[tauri::command(async)]
fn queue_remove(app: tauri::AppHandle, id: String) -> Out<Value> {
    queue_edit(&app, |q| q.remove(&id)).map_err(|e| e.to_string())?;
    queue_changed(&app);
    Ok(queue_view(&app))
}

/// Moves a waiting entry to position `to` (0 = first).
#[tauri::command(async)]
fn queue_move(app: tauri::AppHandle, id: String, to: usize) -> Out<Value> {
    queue_edit(&app, |q| q.reorder(&id, to)).map_err(|e| e.to_string())?;
    queue_changed(&app);
    Ok(queue_view(&app))
}

/// Pause: nothing new starts (what runs goes on). Resume (`on` false) starts the next one.
#[tauri::command(async)]
fn queue_pause(app: tauri::AppHandle, on: bool) -> Value {
    queue_edit(&app, |q| q.paused = on);
    queue_changed(&app);
    if !on {
        queue_kick(&app);
    }
    queue_view(&app)
}

/// Takes the finished entries out of the list.
#[tauri::command(async)]
fn queue_clear_finished(app: tauri::AppHandle) -> Value {
    queue_edit(&app, |q| q.clear_finished());
    queue_changed(&app);
    queue_view(&app)
}

/// Does the computer stay awake while a job runs, and does the window hear why.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Awake {
    Off,
    /// keeps the computer awake
    Quiet,
    /// the same, and a log line when that is not possible here
    Noted,
}

/// `Quiet` when the setting "Keep this computer awake" is on, else `Off`.
fn awake_if_set() -> Awake {
    if ndforge_core::settings::Settings::load().keep_awake { Awake::Quiet } else { Awake::Off }
}

/// How a job is started: where its events go and what its end carries.
#[derive(Clone, Copy)]
struct JobOpts {
    event: &'static str,
    /// the "phase" the end event carries (and every event, with `tag_events`)
    phase: Option<&'static str>,
    tag_events: bool,
    awake: Awake,
    /// what the job is (the queue and the notifications name it)
    kind: &'static str,
    /// a run that finished (not stopped, not failed) may switch the computer off when the person asked for it
    shutdown_after: bool,
}

impl JobOpts {
    fn new(event: &'static str) -> JobOpts {
        JobOpts { event, phase: None, tag_events: false, awake: Awake::Off, kind: "task", shutdown_after: false }
    }
}

/// The one way a job is started: one job at a time (`busy`, released also when the thread panics), the cancel flag reset, the computer kept awake when asked, the events tagged the way
/// the window expects, and the end ({"kind": "done"|"error"}) sent after the flag is released. `body` gets the cancel flag and the event sender.
fn spawn_job(app: tauri::AppHandle, tasks: &Tasks, opts: JobOpts, body: impl FnOnce(Arc<AtomicBool>, &mut dyn FnMut(Value)) -> Result<Value, String> + Send + 'static) -> Out<()> {
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, cancel) = (BusyGuard(tasks.busy.clone()), tasks.cancel.clone());
    let (shutdown_after, shutdown_cancel) = (tasks.shutdown_after.clone(), tasks.shutdown_cancel.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let (channel, phase, tag) = (opts.event, opts.phase, opts.tag_events);
        let mut emit = move |mut v: Value| {
            if let (true, Some(p)) = (tag, phase) {
                v["phase"] = serde_json::json!(p);
            }
            let _ = emitter.emit(channel, v);
        };
        let awake = if opts.awake == Awake::Off { ndforge_core::power::KeepAwake::off() } else { ndforge_core::power::KeepAwake::start_with(ndforge_core::settings::Settings::load().keep_display_on) };
        if opts.awake == Awake::Noted && !awake.note.is_empty() {
            emit(serde_json::json!({"kind": "log", "text": awake.note}));
        }
        let r = body(cancel, &mut emit);
        drop(awake);
        drop(busy);
        let finished = matches!(&r, Ok(data) if data["stopped"] != true);
        let mut end = match &r {
            Ok(data) => serde_json::json!({"kind": "done", "data": data}),
            Err(msg) => serde_json::json!({"kind": "error", "text": msg}),
        };
        if let Some(p) = phase {
            end["phase"] = serde_json::json!(p);
        }
        let _ = app.emit(channel, end);
        job_ended(&app, opts.kind, &r);
        // "shut down when it has finished" covers the whole queue: only when nothing runs and nothing waits any more
        if (opts.shutdown_after || opts.kind != "task") && finished && queue_idle(&app) && shutdown_after.swap(false, Ordering::SeqCst) {
            shutdown_with_warning(&app, &shutdown_cancel, std::time::Duration::from_secs(1), 60);
        }
    });
    Ok(())
}

/// Runs `job` on a thread; its progress goes to the window as events on `channel`, the end as {"kind": "done"|"error"}.
fn spawn_task(app: tauri::AppHandle, tasks: &Tasks, channel: &'static str, job: impl FnOnce(&AtomicBool, &mut dyn FnMut(Value)) -> Result<Value, String> + Send + 'static) -> Out<()> {
    spawn_task_as(app, tasks, channel, "task", job)
}

/// `spawn_task` for a job the person wants to hear about (and the queue can hold): `kind` names it.
fn spawn_task_as(app: tauri::AppHandle, tasks: &Tasks, channel: &'static str, kind: &'static str, job: impl FnOnce(&AtomicBool, &mut dyn FnMut(Value)) -> Result<Value, String> + Send + 'static) -> Out<()> {
    // an export, a matrix or a quant check runs for a long time: the computer is kept awake like for the other long jobs (setting "Keep this computer awake")
    let awake = if kind == "task" { Awake::Off } else { awake_if_set() };
    spawn_job(app, tasks, JobOpts { kind, awake, ..JobOpts::new(channel) }, move |cancel, emit| job(&cancel, emit))
}

#[tauri::command]
fn env_install(app: tauri::AppHandle, tasks: State<Tasks>, unsloth: Option<bool>) -> Out<()> {
    let unsloth = unsloth.unwrap_or(true);
    spawn_task(app, &tasks, "env-event", move |cancel, emit| {
        ndforge_core::envmgr::tasks::install(unsloth, cancel, emit).map(|_| Value::Null).map_err(|e| e.to_string())
    })
}

/// Only the fast kernels (flash-linear-attention, causal-conv1d, and on Windows triton-windows): the button on the setup page when the kernels light is yellow.
#[tauri::command]
fn env_install_kernels(app: tauri::AppHandle, tasks: State<Tasks>) -> Out<()> {
    spawn_task(app, &tasks, "env-event", |cancel, emit| ndforge_core::envmgr::tasks::install_kernels(cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn env_selftest(app: tauri::AppHandle, tasks: State<Tasks>) -> Out<()> {
    spawn_task(app, &tasks, "env-event", |cancel, emit| ndforge_core::envmgr::tasks::selftest(cancel, emit).map_err(|e| e.to_string()))
}


// ------------------------------------------------------------------------------------------------------------------ Tune: tested settings and presets (ndforge_core::tune, docs/TUNE.md)

fn tune_project(s: &State<App>) -> Out<ndforge_core::project::Project> {
    lock(s).project_clone().map_err(|e| e.to_string())
}

fn tune_now() -> String {
    ndforge_core::tune::api::now()
}

fn tune_store(s: &State<App>) -> ndforge_core::tune::store::Store {
    ndforge_core::tune::store::Store::new(lock(s).project_clone().ok().as_ref().map(|p| p.folder.as_path()))
}

/// The presets that fit a made model on this PC (exact, same shape, from a parent, nearest, another PC), with gains and what is stale, and the machine's facts.
#[tauri::command(async)]
fn tune_presets(s: State<App>, model: String) -> Out<Value> {
    let p = tune_project(&s)?;
    ndforge_core::tune::api::presets_view(&p, &model).map_err(|e| e.to_string())
}

/// "Save current settings as preset": the Heal card's values (`card`) become a preset of this model on this machine (`scope` "project" or "library").
#[tauri::command(async)]
fn tune_preset_save(s: State<App>, model: String, name: String, scope: String, card: Value) -> Out<Value> {
    let p = tune_project(&s)?;
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    let card = card.as_object().cloned().ok_or_else(|| "The card's settings are missing.".to_string())?;
    let preset = ndforge_core::tune::api::save_card(&p, &model, &name, scope, card, &tune_now()).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"id": preset.id, "name": preset.name}))
}

#[tauri::command(async)]
fn tune_preset_rename(s: State<App>, id: String, scope: String, name: String) -> Out<Value> {
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    let p = tune_store(&s).rename(scope, &id, &name).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"id": p.id, "name": p.name}))
}

#[tauri::command(async)]
fn tune_preset_duplicate(s: State<App>, id: String, scope: String, name: String) -> Out<Value> {
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    let p = tune_store(&s).duplicate(scope, &id, &name, &tune_now()).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"id": p.id, "name": p.name}))
}

/// Deletes a preset; `confirm` must be its exact name (all three arguments are required: a missing one is an error, never a deletion).
#[tauri::command(async)]
fn tune_preset_delete(s: State<App>, id: String, scope: String, confirm: String) -> Out<()> {
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    tune_store(&s).delete(scope, &id, &confirm).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn tune_preset_export(s: State<App>, id: String, scope: String, path: String) -> Out<()> {
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    tune_store(&s).export(scope, &id, std::path::Path::new(&path)).map_err(|e| e.to_string())
}

/// Reads an exported preset into the app-wide library; `overwrite` is required (an existing id is replaced only when it is true).
#[tauri::command(async)]
fn tune_preset_import(s: State<App>, path: String, overwrite: bool) -> Out<Value> {
    let p = tune_store(&s).import(std::path::Path::new(&path), overwrite).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({"id": p.id, "name": p.name}))
}

/// What using a preset would change on the card (`current` = the card's values now): every change with the measured gain, whether the whole preset or only what carries over is offered, what is stale.
/// Nothing is changed here; the page fills its own settings from the answer and only its own Start saves anything.
#[tauri::command(async)]
fn tune_apply_preview(s: State<App>, preset_id: String, scope: String, model: String, target: String, current: Value) -> Out<Value> {
    let p = tune_project(&s)?;
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    let current = current.as_object().cloned().ok_or_else(|| "The card's settings are missing.".to_string())?;
    ndforge_core::tune::api::apply_preview(&p, scope, &preset_id, &model, &target, &current).map_err(|e| e.to_string())
}

/// Applies a preset to the SAVED settings (`target` "settings_teacher" or "settings_vllm"): the token of a fresh preview must be sent back (a stale or missing one changes nothing). The pages (Heal card, Auto, Try, Think,
/// Imatrix) fill their own fields from the preview and save nothing until their own Start. All arguments are required.
#[tauri::command(async)]
fn tune_apply(s: State<App>, preset_id: String, scope: String, model: String, target: String, token: String) -> Out<Value> {
    let p = tune_project(&s)?;
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    ndforge_core::tune::api::apply_settings(&p, scope, &preset_id, &model, &target, &token).map_err(|e| e.to_string())
}

/// The presets of this PC for the Auto page ("Use tested settings"); another card's are left out.
#[tauri::command(async)]
fn tune_presets_here(s: State<App>) -> Value {
    Value::Array(ndforge_core::tune::api::presets_here(lock(&s).project_clone().ok().as_ref()))
}

/// A preset as the snapshot an Auto-cut run keeps (`AutoSpec.tuned`): the values and the measured speed at this moment.
#[tauri::command(async)]
fn tune_snapshot(s: State<App>, preset_id: String, scope: String) -> Out<Value> {
    let p = tune_project(&s)?;
    let scope = ndforge_core::tune::store::Scope::parse(&scope).map_err(|e| e.to_string())?;
    serde_json::to_value(ndforge_core::tune::api::tuned_snapshot(&p, scope, &preset_id).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// The runner's plan before Start: the model's facts, the checklist (each test ready or why not, its estimate), the total and "done around", and the presets that already fit. No side effects.
#[tauri::command(async)]
fn tune_plan(s: State<App>, spec: Value) -> Out<Value> {
    let p = tune_project(&s)?;
    let mut spec: ndforge_core::tune::runner::Spec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    ndforge_core::tune::api::with_settings(&mut spec.card);
    let id = ndforge_core::tune::api::model_for(&p, &spec.model).map_err(|e| e.to_string())?;
    let mut notes = ndforge_core::tune::runner::BTreeMapNotes::new();
    let vllm = ndforge_core::settings::Settings::load().vllm;
    if !(vllm.enabled && vllm.teacher.enabled) {
        notes.insert("teacher.vllm".into(), "vLLM is not switched on for the original's answers (Settings, vLLM engine): nothing to test.".into());
    }
    if let Some(free) = ndforge_core::tune::disk::fs_free(&ndforge_core::reshape::runs::models_dir(&p)).filter(|f| *f < ndforge_core::tune::disk::need_bytes(ndforge_core::tune::disk::DEFAULT_MB)) {
        notes.insert("disk".into(), format!("The drive of your models has {:.1} GB free; the test needs {:.1} GB.", free as f64 / 1e9, ndforge_core::tune::disk::need_bytes(ndforge_core::tune::disk::DEFAULT_MB) as f64 / 1e9));
    }
    if ndforge_core::export::quantcheck::offload_reference(&p).is_none() {
        notes.insert("llama.offload".into(), "There is no 16-bit GGUF in this project: export a model as F16 once (or run Compare quants, which keeps a reference).".into());
    }
    if ndforge_core::train::runs::latest_version(&p).is_none() {
        for t in ["heal.student_batch", "heal.loss", "heal.quality", "teacher.standard", "teacher.vllm", "try.calib_batch", "think.batch"] {
            notes.insert(t.into(), "Save a training-data version in Step 2 first: the tests use your real rows.".into());
        }
    }
    if !notes.contains_key("heal.quality") {
        let ready = ndforge_core::tune::real::heal_spec_of(&spec.card).ok().and_then(|h| ndforge_core::reshape::heal_bench::trial_ready(&p, &spec.model, &h, card_vram_gb()).ok());
        if ready == Some(false) {
            notes.insert("heal.quality".into(), "Heal this model once first (or run Try first): the quality trials need the original's saved answers for this sequence length.".into());
        }
    }
    let now = chrono_secs();
    let mut v = ndforge_core::tune::runner::plan_view(&spec, id.params as f64 / 1e9, now, &notes);
    v["model"] = serde_json::json!({"name": id.name, "params_b": id.params as f64 / 1e9, "layers": id.layers, "hidden": id.hidden});
    v["card_mb"] = serde_json::json!(card_vram_gb().map(|g| g * 1024.0));
    Ok(v)
}

fn chrono_secs() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Starts a tuning run as a job (one at a time, like every job): the checklist runs test after test and ends with a saved preset. Events on "tune-event" tagged {"phase": "tune"}.
#[tauri::command]
fn tune_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, spec: Value) -> Out<Value> {
    let p = tune_project(&s)?;
    let mut spec: ndforge_core::tune::runner::Spec = serde_json::from_value(spec).map_err(|e| format!("The choice is not valid: {e}"))?;
    ndforge_core::tune::api::with_settings(&mut spec.card);
    if spec.tests.is_empty() {
        return Err("Tick at least one test.".into());
    }
    let identity = ndforge_core::tune::api::model_for(&p, &spec.model).map_err(|e| e.to_string())?;
    let machine = ndforge_core::tune::api::machine_now();
    let st = ndforge_core::tune::runner::new_run(spec, identity.clone(), machine, identity.params as f64 / 1e9, &tune_now());
    if st.tests.is_empty() {
        return Err("None of the ticked tests can run in this version.".into());
    }
    ndforge_core::tune::runner::save_state(&p.folder, &st).map_err(|e| e.to_string())?;
    let id = st.id.clone();
    tune_job(app, &tasks, p, st)?;
    Ok(serde_json::json!({"run": id}))
}

/// Goes on with a stopped, waiting or interrupted run: the tests that are done are not run again, a test that was running starts from its beginning.
#[tauri::command]
fn tune_continue(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, run_id: String) -> Out<()> {
    let p = tune_project(&s)?;
    let mut st = ndforge_core::tune::runner::load_state(&p.folder, &run_id).map_err(|e| e.to_string())?;
    for t in st.tests.iter_mut().filter(|t| matches!(t.status.as_str(), "running" | "stopped" | "waiting_gpu")) {
        t.status = "queued".into();
    }
    st.note.clear();
    tune_job(app, &tasks, p, st)
}

fn tune_job(app: tauri::AppHandle, tasks: &Tasks, p: ndforge_core::project::Project, mut st: ndforge_core::tune::runner::RunState) -> Out<()> {
    let opts = JobOpts { phase: Some("tune"), tag_events: true, awake: Awake::Quiet, kind: "tune", ..JobOpts::new("tune-event") };
    spawn_job(app, tasks, opts, move |stop, emit| {
        let store = ndforge_core::tune::store::Store::new(Some(&p.folder));
        let card_mb = card_vram_gb().map(|g| g * 1024.0);
        let mut fx = ndforge_core::tune::real::Real { project: &p, vram_gb: card_vram_gb(), stop, emit_fn: emit };
        ndforge_core::tune::runner::run(&mut fx, &p.folder, &mut st, card_mb, &store).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"run": st.id, "status": st.status, "preset_id": st.preset_id}))
    })
}

#[tauri::command]
fn tune_stop(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

#[tauri::command(async)]
fn tune_state(s: State<App>, run_id: String) -> Out<Value> {
    let p = tune_project(&s)?;
    let st = ndforge_core::tune::runner::load_state(&p.folder, &run_id).map_err(|e| e.to_string())?;
    serde_json::to_value(st).map_err(|e| e.to_string())
}

#[tauri::command(async)]
fn tune_runs(s: State<App>) -> Out<Value> {
    let p = tune_project(&s)?;
    Ok(Value::Array(ndforge_core::tune::runner::list_runs(&p.folder)))
}

// ------------------------------------------------------------------------------------------------------------------ the memory panel (ndforge_core::memory, docs/MEMORY.md)

fn memory_env() -> ndforge_core::memory::Real {
    ndforge_core::memory::Real { vllm: ndforge_core::settings::Settings::load().vllm }
}

/// The numbers of the footer and the popover: graphics memory, RAM, the app's own share, the engine containers, and the other programs that hold graphics memory (cached for about a second).
#[tauri::command(async)]
fn memory_status(tasks: State<Tasks>) -> Value {
    ndforge_core::memory::status(&memory_env(), &ndforge_core::memory::Ctx { busy: tasks.busy.load(Ordering::SeqCst) })
}

/// What "Free GPU" (`kind` "gpu"), "Free RAM" ("ram") or "Release WSL memory" ("wsl") would end or release: the exact list, the other programs (listed only) and a plan token. Nothing is done.
#[tauri::command(async)]
fn memory_free_plan(tasks: State<Tasks>, kind: String) -> Out<Value> {
    let p = ndforge_core::memory::plan(&memory_env(), &ndforge_core::memory::Ctx { busy: tasks.busy.load(Ordering::SeqCst) }, &kind).map_err(|e| e.to_string())?;
    serde_json::to_value(p).map_err(|e| e.to_string())
}

/// Does what the plan with `plan_token` listed (it is planned again and a token that no longer matches is refused). `stop_running`: the running job is stopped first by the normal Stop (its state is saved,
/// Continue works) and only then is anything freed; without it the items a running job uses are left alone and reported. All three arguments are required.
#[tauri::command(async)]
fn memory_free(tasks: State<Tasks>, kind: String, plan_token: String, stop_running: bool) -> Out<Value> {
    if stop_running && tasks.busy.load(Ordering::SeqCst) && !stop_and_wait(&tasks.busy, &tasks.cancel, std::time::Duration::from_secs(90)) {
        return Err("The running job did not stop in time, so nothing was freed. Try again in a moment.".into());
    }
    ndforge_core::memory::free(&memory_env(), &ndforge_core::memory::Ctx { busy: tasks.busy.load(Ordering::SeqCst) }, &kind, &plan_token).map_err(|e| e.to_string())
}

/// Removes the engine containers an earlier crashed run left behind, before a new vLLM job starts (nothing while a job runs); tells the window's log.
fn clean_stale_engines(app: &tauri::AppHandle, tasks: &State<Tasks>) {
    let gone = ndforge_core::memory::remove_stale(&memory_env(), &ndforge_core::memory::Ctx { busy: tasks.busy.load(Ordering::SeqCst) });
    if !gone.is_empty() {
        let _ = app.emit("reshape-event", serde_json::json!({"kind": "log", "phase": "vllm", "text": format!("Removed the engine container(s) an earlier run left behind: {}.", gone.join(", "))}));
    }
}

// ------------------------------------------------------------------------------------------------------------------ the Storage page (ndforge_core::storage, docs/STORAGE.md)

fn storage_project(s: &State<App>) -> Option<ndforge_core::project::Project> {
    lock(s).project_clone().ok()
}

/// Starts the scan in the background: the items arrive as `storage-event` {"kind": "item"} (no size yet), then {"kind": "size"}, then {"kind": "done", "data": the whole list}.
#[tauri::command]
fn storage_scan(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, scan: State<ScanTasks>) -> Out<()> {
    if scan.running.swap(true, Ordering::SeqCst) {
        return Err("The scan is already running.".into());
    }
    scan.cancel.store(false, Ordering::SeqCst);
    let project = storage_project(&s);
    let (busy, running, cancel) = (tasks.busy.clone(), scan.running.clone(), scan.cancel.clone());
    std::thread::spawn(move || {
        let vllm = ndforge_core::settings::Settings::load().vllm;
        let ctx = ndforge_core::storage::Ctx { project: project.as_ref(), busy: busy.load(Ordering::SeqCst), vllm: Some(&vllm) };
        let emitter = app.clone();
        let data = ndforge_core::storage::scan(&ctx, &cancel, &mut |v| {
            let _ = emitter.emit("storage-event", v);
        });
        running.store(false, Ordering::SeqCst);
        let _ = app.emit("storage-event", serde_json::json!({"kind": "done", "phase": "scan", "data": data}));
    });
    Ok(())
}

/// `dry_run` and `pinned` and `allow_pinned` are REQUIRED (a missing or mis-spelled key is an error, never a real deletion); `dry_run: true`: exactly what would go (nothing is touched, nothing is started) and the `token` of that plan; a real deletion needs that token. Otherwise the deletion runs as a job (one at a time; progress as `storage-event`, the result as its `done`).
#[tauri::command]
fn storage_delete(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, scan: State<ScanTasks>, ids: Vec<String>, dry_run: bool, confirm: Option<String>, pinned: Vec<String>, allow_pinned: bool, token: Option<String>) -> Out<Value> {
    let project = storage_project(&s);
    let opts = ndforge_core::storage::Options { dry_run, confirm: confirm.unwrap_or_default(), pinned, allow_pinned, token };
    let vllm = ndforge_core::settings::Settings::load().vllm;
    if opts.dry_run {
        let ctx = ndforge_core::storage::Ctx { project: project.as_ref(), busy: tasks.busy.load(Ordering::SeqCst), vllm: Some(&vllm) };
        return ndforge_core::storage::delete(&ctx, &ids, &opts, &AtomicBool::new(false), &mut |_| {}).map_err(|e| e.to_string());
    }
    // checked here as well, so that a refusal comes back at once and not as the end of a job
    {
        let ctx = ndforge_core::storage::Ctx { project: project.as_ref(), busy: tasks.busy.load(Ordering::SeqCst), vllm: Some(&vllm) };
        ndforge_core::storage::plan_delete(&ctx, &ids, &opts).map_err(|e| e.to_string())?;
    }
    let deleting = scan.deleting.clone();
    deleting.store(true, Ordering::SeqCst);
    let flag = deleting.clone();
    let started = spawn_task(app, &tasks, "storage-event", move |cancel, emit| {
        let ctx = ndforge_core::storage::Ctx { project: project.as_ref(), busy: false, vllm: Some(&vllm) };
        let r = ndforge_core::storage::delete(&ctx, &ids, &opts, cancel, emit).map_err(|e| e.to_string());
        flag.store(false, Ordering::SeqCst);
        let mut out = r?;
        out["phase"] = Value::from("delete");
        Ok(out)
    });
    if started.is_err() {
        deleting.store(false, Ordering::SeqCst);
    }
    started?;
    Ok(serde_json::json!({"started": true}))
}

/// Ends a scan, or a deletion between two files.
#[tauri::command]
fn storage_cancel(tasks: State<Tasks>, scan: State<ScanTasks>) {
    scan.cancel.store(true, Ordering::SeqCst);
    if scan.deleting.load(Ordering::SeqCst) {
        tasks.cancel.store(true, Ordering::SeqCst);
    }
}

#[tauri::command]
fn env_cancel(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

// ------------------------------------------------------------------------------------------------------------------ llama.cpp (docs/LLAMA_CPP_INSTALL.md)

#[tauri::command(async)]
fn llamacpp_status() -> Value {
    ndforge_core::envmgr::llamacpp::status(&ndforge_core::envmgr::llamacpp::Config::from_env())
}

/// What an install would download (backend, files, size), for the confirmation: nothing is downloaded or changed.
#[tauri::command(async)]
fn llamacpp_plan(backend: Option<String>, tag: Option<String>) -> Out<Value> {
    use ndforge_core::envmgr::llamacpp as lc;
    let opts = lc::Options { want: lc::choose::Want::parse(backend.as_deref().unwrap_or("auto")), tag: tag.filter(|t| !t.trim().is_empty()) };
    lc::plan(&lc::Config::from_env(), &lc::choose::detect_machine(), &opts).map_err(|f| f.message)
}

#[tauri::command]
fn llamacpp_install(app: tauri::AppHandle, tasks: State<Tasks>, backend: Option<String>, tag: Option<String>) -> Out<()> {
    use ndforge_core::envmgr::llamacpp as lc;
    let opts = lc::Options { want: lc::choose::Want::parse(backend.as_deref().unwrap_or("auto")), tag: tag.filter(|t| !t.trim().is_empty()) };
    spawn_task(app, &tasks, "env-event", move |cancel, emit| {
        let mut out = lc::install_task(&lc::Config::from_env(), &opts, cancel, emit).map_err(|e| e.to_string())?;
        out["task"] = Value::from("llamacpp");
        Ok(out)
    })
}

#[tauri::command]
fn llamacpp_cancel(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

#[tauri::command]
fn llamacpp_remove(tasks: State<Tasks>) -> Out<Value> {
    use ndforge_core::envmgr::llamacpp as lc;
    if tasks.busy.load(Ordering::SeqCst) {
        return Err("Something is still running. Wait for it or press Cancel, then remove llama.cpp.".into());
    }
    let cfg = lc::Config::from_env();
    lc::remove(&cfg).map_err(|f| f.message)?;
    Ok(lc::status(&cfg))
}

/// A hint only: is there a newer llama.cpp release than the installed one? Installs nothing.
#[tauri::command(async)]
fn llamacpp_check_update() -> Out<Value> {
    ndforge_core::envmgr::llamacpp::check_update(&ndforge_core::envmgr::llamacpp::Config::from_env()).map_err(|f| f.message)
}

pub fn run() {
    // taken first, before anything reads or writes the settings folder
    let instance = ndforge_core::instance::acquire(&ndforge_core::paths::config_dir());
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .on_window_event(|window, event| {
            // closing the window while something runs (training above all): stop it and let it save first, so training can be continued later
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let tasks = window.state::<Tasks>();
                if tasks.busy.load(Ordering::SeqCst) {
                    api.prevent_close();
                    if !tasks.closing.swap(true, Ordering::SeqCst) {
                        let _ = window.emit("closing", serde_json::json!({"text": "Stopping and saving what is running, then closing\u{2026}"}));
                        let (busy, cancel, handle) = (tasks.busy.clone(), tasks.cancel.clone(), window.app_handle().clone());
                        std::thread::spawn(move || {
                            stop_and_wait(&busy, &cancel, std::time::Duration::from_secs(170));
                            handle.exit(0);
                        });
                    }
                }
            }
        })
        .setup(move |app| {
            // one program per settings folder: two would write the same settings, projects and logs (README, "One program at a time")
            use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
            match instance {
                Ok(hold) => {
                    app.manage(InstanceHold(Mutex::new(Some(hold))));
                }
                Err(ndforge_core::instance::Refused::Held(pid)) => {
                    for (_, w) in app.webview_windows() {
                        let _ = w.hide();
                    }
                    let handle = app.handle().clone();
                    let who = pid.map(|p| format!(" (process {p})")).unwrap_or_default();
                    app.dialog().message(format!("ND Lore Forge is already open{who}. Use that window; two at once would damage each other's settings and projects.")).kind(MessageDialogKind::Error)
                        .title("ND Lore Forge is already open").show(move |_| handle.exit(0));
                }
                Err(ndforge_core::instance::Refused::Unusable(_)) => {}     // the settings folder cannot be used: the program starts anyway and says so where it fails
            }
            Ok(())
        })
        .manage(QueueState(Mutex::new({
            let mut q = ndforge_core::queue::Queue::load();
            q.recover();
            let _ = q.save();
            q
        })))
        .manage(App(Mutex::new(AppModel::load())))
        .manage(Tasks::default())
        .manage(ScanTasks::default())
        .invoke_handler(tauri::generate_handler![get_state, preflight, diagnostics_save, timing_report, webhook_set, queue_list, queue_add, queue_remove, queue_move, queue_pause, queue_clear_finished, create_project, open_project, remove_world, check_bundle, apply_bundle, set_setting, set_theme, help, env_status, env_install, env_install_kernels, env_selftest, env_cancel, llamacpp_status, llamacpp_plan, llamacpp_install, llamacpp_cancel, llamacpp_remove, llamacpp_check_update, dataset_state, dataset_generate, bench_start, bench_last, model_lookup, model_add, model_remove, problem_report, test_state, test_result, test_start, test_ask, test_save_transcript, teacher_set, teacher_check, teacher_start, teacher_clear, dataset_examples, dataset_set_status, dataset_set_status_matching, dataset_edit, dataset_save, studio_export, train_state, train_plan, train_start, set_shutdown_after, cancel_shutdown, train_stop, train_metrics, train_server_add, train_server_remove, train_server_check, unsloth_set, unsloth_remove, unsloth_check, export_state, export_start, export_restore_vision, export_model_state, export_model_start, mixed_plan, imatrix_list, imatrix_create, imatrix_stop, imatrix_import, imatrix_save_copy, imatrix_open_folder, imatrix_make_dat, imatrix_delete, imatrix_check, imatrix_merge, imatrix_published, imatrix_adapt_preview, imatrix_adapt, quantcheck_state, quantcheck_plan, quantcheck_start, quantcheck_stop, quantcheck_get, quantcheck_delete, quantcheck_delete_logits, quantfit_state, quantfit_estimate, quantfit_start, quantfit_stop, copy_adapter, ollama_install, ollama_probe, ollama_check, ollama_server_add, ollama_server_remove, retrain_info, retrain_start, reshape_state, reshape_options, moe_compat, moe_recommend, reshape_measurement, reshape_measure, reshape_start, reshape_stop, reshape_test, reshape_remove, heal_options, heal_estimate, heal_start, heal_bench_start, heal_bench_last, heal_speed_start, heal_speed_report, heal_candidates, heal_check, heal_pair, heal_pair_estimate, heal_source_start, heal_stop, general_state, general_choose, general_add, general_remove, general_ack, general_check, hf_token_set, hf_token_clear, reshape_try_plan, reshape_try_results, reshape_try_start, reshape_search_plan, reshape_search_start, reshape_search_status, think_check_results, think_check_plan, think_check_start, vllm_status, vllm_set, vllm_selftest_start, vllm_container_probe_start, vllm_selftest_report, vllm_selftest_files, vllm_teacher_check_start, trace_gen_state, trace_gen_start, auto_plan, auto_start, auto_change_preview, auto_stop, auto_status, auto_report, studio_share_set, studio_share_check, storage_scan, storage_delete, storage_cancel, memory_status, memory_free_plan, memory_free, tune_presets, tune_preset_save, tune_preset_rename, tune_preset_duplicate, tune_preset_delete, tune_preset_export, tune_preset_import, tune_apply_preview, tune_apply, tune_plan, tune_start, tune_continue, tune_stop, tune_state, tune_runs, tune_presets_here, tune_snapshot])
        .run(tauri::generate_context!())
        .expect("error while running ND Lore Forge");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn the_busy_flag_is_released_when_a_job_thread_panics() {
        let flag = Arc::new(AtomicBool::new(true));
        let guard = BusyGuard(flag.clone());
        let t = std::thread::spawn(move || {
            let _held = guard;
            panic!("job crashed");
        });
        assert!(t.join().is_err());
        assert!(!flag.load(Ordering::SeqCst));
        // dropping it in the normal way does the same
        flag.store(true, Ordering::SeqCst);
        drop(BusyGuard(flag.clone()));
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    fn closing_stops_the_job_and_waits_for_it_to_save() {
        let (busy, cancel) = (Arc::new(AtomicBool::new(true)), Arc::new(AtomicBool::new(false)));
        // a job that notices the stop request after a moment and then ends
        let (b2, c2) = (busy.clone(), cancel.clone());
        let job = std::thread::spawn(move || {
            while !c2.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            std::thread::sleep(std::time::Duration::from_millis(150));    // saving the checkpoint
            b2.store(false, Ordering::SeqCst);
        });
        let t = std::time::Instant::now();
        assert!(stop_and_wait(&busy, &cancel, std::time::Duration::from_secs(5)));
        assert!(cancel.load(Ordering::SeqCst) && !busy.load(Ordering::SeqCst) && t.elapsed() >= std::time::Duration::from_millis(150), "it waited for the save");
        job.join().unwrap();
        // nothing running: nothing to wait for; a job that never ends: gives up after the time allowed
        assert!(stop_and_wait(&AtomicBool::new(false), &AtomicBool::new(false), std::time::Duration::from_millis(50)));
        assert!(!stop_and_wait(&AtomicBool::new(true), &AtomicBool::new(false), std::time::Duration::from_millis(250)));
    }
}
