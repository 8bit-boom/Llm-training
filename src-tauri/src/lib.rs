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
fn open_project(s: State<App>, folder: String) -> Out<Value> {
    let mut m = lock(&s);
    m.open_project(&folder).map_err(|e| e.to_string())?;
    Ok(m.state())
}

/// Takes a world out of a project that holds several. Returns the new state.
#[tauri::command]
fn remove_world(s: State<App>, slug: String) -> Out<Value> {
    let mut m = lock(&s);
    m.remove_world(&slug).map_err(|e| e.to_string())?;
    Ok(m.state())
}

#[tauri::command]
fn check_bundle(s: State<App>, path: String) -> Out<Value> {
    lock(&s).check_bundle(&path).map_err(|e| e.to_string())
}

/// Returns {"changes": text, "state": ...}
#[tauri::command]
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

#[tauri::command]
fn dataset_state(s: State<App>) -> Out<Value> {
    lock(&s).dataset_state().map_err(|e| e.to_string())
}

#[tauri::command]
fn dataset_generate(s: State<App>, recipe: Value) -> Out<Value> {
    lock(&s).dataset_generate(recipe).map_err(|e| e.to_string())
}

#[tauri::command]
fn problem_report(s: State<App>) -> String {
    lock(&s).problem_report()
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

#[tauri::command]
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

#[tauri::command]
fn test_state(s: State<App>) -> Out<Value> {
    lock(&s).test_state().map_err(|e| e.to_string())
}

#[tauri::command]
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

#[tauri::command]
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
#[tauri::command]
fn studio_export(s: State<App>, version: Option<u32>, dest: Option<String>, world_book: bool) -> Out<Value> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    let dest = dest.map(std::path::PathBuf::from).unwrap_or_else(|| ndforge_core::export::studio::default_dest(&project));
    ndforge_core::export::studio::export(&project, version, &dest, world_book).map_err(|e| e.to_string())
}

#[tauri::command]
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

#[tauri::command]
fn bench_last(s: State<App>, model: String, method: String) -> Out<Value> {
    let (method, _) = parse_choice(&method, "quick")?;
    lock(&s).bench_last(&model, method).map_err(|e| e.to_string())
}

#[tauri::command]
fn train_state(s: State<App>, server: Option<String>) -> Out<Value> {
    lock(&s).train_state(server.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
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
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, stop, name) = (tasks.busy.clone(), tasks.cancel.clone(), job.name.clone());
    let (shutdown_after, shutdown_cancel) = (tasks.shutdown_after.clone(), tasks.shutdown_cancel.clone());
    let keep_awake = lock(&s).settings.keep_awake;
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit("train-event", v); };
        // the computer stays awake for as long as the guard lives (this also covers a run on a server: this program fetches the result)
        let awake = if keep_awake { ndforge_core::power::KeepAwake::start() } else { ndforge_core::power::KeepAwake::off() };
        if !awake.note.is_empty() {
            emit(serde_json::json!({"kind": "log", "text": awake.note}));
        }
        let r = run_training(&job, stop, &mut emit);
        drop(awake);
        busy.store(false, Ordering::SeqCst);
        // only a run that finished (not stopped, not failed) switches the computer off
        let finished = matches!(&r, Ok(data) if data["stopped"] != true);
        let end = match r {
            Ok(data) => serde_json::json!({"kind": "done", "data": data}),
            Err(e) => serde_json::json!({"kind": "error", "text": e.to_string()}),
        };
        let _ = app.emit("train-event", end);
        if finished && shutdown_after.swap(false, Ordering::SeqCst) {
            shutdown_with_warning(&app, &shutdown_cancel, std::time::Duration::from_secs(1), 60);
        }
    });
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

#[tauri::command]
fn train_metrics(s: State<App>, run: String) -> Out<Value> {
    lock(&s).train_metrics(&run).map_err(|e| e.to_string())
}

#[tauri::command]
fn export_state(s: State<App>) -> Out<Value> {
    lock(&s).export_state().map_err(|e| e.to_string())
}

/// Makes the GGUF + Modelfile of a run; progress goes to the window as "export-event" events.
#[tauri::command]
fn export_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, run: String, quant: String, num_ctx: u32, what: Option<String>, quants: Option<Vec<String>>, imatrix: Option<ndforge_core::export::runs::ImatrixOptions>) -> Out<()> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    spawn_task(app, &tasks, "export-event", move |cancel, emit| {
        let list = quants.filter(|q| !q.is_empty()).unwrap_or_else(|| vec![quant]);
        let mut metas = ndforge_core::export::runs::execute_many_with(&project, &run, &list, num_ctx, what.as_deref().unwrap_or("gguf"), &imatrix.unwrap_or_default(), cancel, emit).map_err(|e| e.to_string())?;
        // one size: its export; several: a summary naming them all
        Ok(if metas.len() == 1 { metas.remove(0) } else { serde_json::json!({"name": metas.iter().filter_map(|m| m["name"].as_str()).collect::<Vec<_>>().join(", "), "run": run, "exports": metas}) })
    })
}

/// Copies a run's LoRA add-on (with a README on how to load and merge it) into a folder the user chose. Returns the new folder.
#[tauri::command]
fn copy_adapter(s: State<App>, run: String, dest: String) -> Out<String> {
    let project = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::export::runs::copy_adapter(&project, &run, std::path::Path::new(&dest)).map(|p| p.display().to_string()).map_err(|e| e.to_string())
}

/// Puts an export into an Ollama server and remembers it.
#[tauri::command]
fn ollama_install(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, export: String, url: String, name: String) -> Out<()> {
    let m = lock(&s);
    let (gguf, family, num_ctx) = m.export_for_install(&export).map_err(|e| e.to_string())?;
    let project = m.project_clone().map_err(|e| e.to_string())?;
    drop(m);
    spawn_task(app, &tasks, "export-event", move |_cancel, emit| {
        ndforge_core::export::install::install(&url, &name, &gguf, &family, num_ctx, emit).map_err(|e| e.to_string())?;
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

#[tauri::command]
fn retrain_info(s: State<App>) -> Out<Value> {
    lock(&s).retrain_info().map_err(|e| e.to_string())
}

/// "Retrain with the new export": new data version, then a new training run, then (optionally) the files and the Ollama install.
/// Everything is reported on the "train-event" channel; stages are {"kind": "stage", "text": ...}.
#[tauri::command]
fn retrain_start(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, from_previous: bool, export_after: bool, install_url: Option<String>, ollama_name: Option<String>) -> Out<String> {
    let job = lock(&s).retrain_job(from_previous).map_err(|e| e.to_string())?;
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, stop, name) = (tasks.busy.clone(), tasks.cancel.clone(), job.name.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit("train-event", v); };
        emit(serde_json::json!({"kind": "stage", "text": "Training the new run"}));
        let train = run_training(&job, stop.clone(), &mut emit);
        let outcome: Result<Value, String> = match train {
            Err(e) => Err(e.to_string()),
            Ok(result) if result["stopped"] == true => Ok(result),
            Ok(result) => (|| {
                if export_after {
                    emit(serde_json::json!({"kind": "stage", "text": "Making the files for nd-world"}));
                    let quant = ndforge_core::export::runs::list_exports(&job.project).first().and_then(|m| m["quant"].as_str().map(String::from)).unwrap_or_else(|| "q4_k_m".into());
                    let meta = ndforge_core::export::runs::execute(&job.project, &job.name, &quant, 8192, &stop, &mut emit).map_err(|e| e.to_string())?;
                    if let Some(url) = install_url.filter(|u| !u.is_empty()) {
                        emit(serde_json::json!({"kind": "stage", "text": "Installing into Ollama"}));
                        let export = meta["name"].as_str().unwrap_or("").to_string();
                        let name = ollama_name.clone().filter(|n| !n.is_empty()).unwrap_or_else(|| meta["ollama_name"].as_str().unwrap_or("world-gm").to_string());
                        let gguf = job.project.folder.join("exports").join(&export).join(meta["gguf_file"].as_str().unwrap_or("model.gguf"));
                        ndforge_core::export::install::install(&url, &name, &gguf, &job.model.family, 8192, &mut emit).map_err(|e| e.to_string())?;
                        ndforge_core::export::runs::record_install(&job.project, &export, &url, &name).map_err(|e| e.to_string())?;
                    }
                }
                Ok(result)
            })(),
        };
        busy.store(false, Ordering::SeqCst);
        let end = match outcome {
            Ok(data) => serde_json::json!({"kind": "done", "data": data}),
            Err(e) => serde_json::json!({"kind": "error", "text": e}),
        };
        let _ = app.emit("train-event", end);
    });
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

#[tauri::command]
fn env_status() -> Value {
    ndforge_core::envmgr::tasks::status()
}

/// The memory of this computer's first graphics card in GB (None when there is none or it cannot be read).
fn card_vram_gb() -> Option<f64> {
    ndforge_core::envmgr::gpu::detect_gpus().ok().and_then(|g| g.first().map(|x| x.vram_gb()))
}

// ------------------------------------------------------------------------------------------------ slim or grow a model

#[tauri::command]
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

/// What an earlier measurement of a model on the saved data looks like (is there one to reuse?).
#[tauri::command]
fn reshape_measurement(s: State<App>, model: String, samples: u32, len: u32, load: String) -> Out<Value> {
    let m = ndforge_core::train::catalog::find(&model).ok_or("That model is not in the list any more.")?;
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    Ok(ndforge_core::reshape::runs::measurement_info(&p, &m, samples, len, &load))
}

/// Measures a model on the project's text now and keeps the result (optional: making a thinner model measures by itself). Events on "reshape-event".
#[tauri::command]
fn reshape_measure(app: tauri::AppHandle, s: State<App>, tasks: State<Tasks>, model: String, samples: u32, len: u32, load: String, force: Option<bool>) -> Out<()> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, stop) = (tasks.busy.clone(), tasks.cancel.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit("reshape-event", v); };
        let r = ndforge_core::reshape::runs::measure(&p, &model, samples, len, &load, force.unwrap_or(false), stop, &mut emit);
        busy.store(false, Ordering::SeqCst);
        let end = match r {
            Ok(data) => serde_json::json!({"kind": "done", "data": data, "phase": "measure"}),
            Err(e) => serde_json::json!({"kind": "error", "text": e.to_string(), "phase": "measure"}),
        };
        let _ = app.emit("reshape-event", end);
    });
    Ok(())
}

/// What the Heal card shows: how healing would run on this card, the budgets and where it stands.
#[tauri::command]
fn heal_options(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::runs::heal_options(&p, &name, card_vram_gb()).map_err(|e| e.to_string())
}

/// The live memory estimate of the Heal card for the settings chosen so far.
#[tauri::command]
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
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, stop) = (tasks.busy.clone(), tasks.cancel.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit("reshape-event", v); };
        // healing takes hours: the computer stays awake, like during training
        let awake = if keep_awake { ndforge_core::power::KeepAwake::start() } else { ndforge_core::power::KeepAwake::off() };
        if !awake.note.is_empty() {
            emit(serde_json::json!({"kind": "log", "text": awake.note}));
        }
        let r = ndforge_core::reshape::runs::heal(&p, &name, &spec, card_vram_gb(), stop, std::time::Duration::from_secs(120), &mut emit);
        drop(awake);
        busy.store(false, Ordering::SeqCst);
        let end = match r {
            Ok(data) => serde_json::json!({"kind": "done", "data": data, "phase": "heal"}),
            Err(e) => serde_json::json!({"kind": "error", "text": e.to_string(), "phase": "heal"}),
        };
        let _ = app.emit("reshape-event", end);
    });
    Ok(())
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
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, stop) = (tasks.busy.clone(), tasks.cancel.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit("reshape-event", v); };
        // writing a model of tens of gigabytes takes a while: the computer stays awake, like during training
        let awake = if keep_awake { ndforge_core::power::KeepAwake::start() } else { ndforge_core::power::KeepAwake::off() };
        if !awake.note.is_empty() {
            emit(serde_json::json!({"kind": "log", "text": awake.note}));
        }
        let r = ndforge_core::reshape::runs::execute(&p, &spec, stop, &mut emit);
        drop(awake);
        busy.store(false, Ordering::SeqCst);
        let end = match r {
            Ok(data) => serde_json::json!({"kind": "done", "data": data}),
            Err(e) => serde_json::json!({"kind": "error", "text": e.to_string()}),
        };
        let _ = app.emit("reshape-event", end);
    });
    Ok(())
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
#[tauri::command]
fn reshape_remove(s: State<App>, name: String) -> Out<Value> {
    let p = lock(&s).project_clone().map_err(|e| e.to_string())?;
    ndforge_core::reshape::runs::remove(&p, &name).map_err(|e| e.to_string())?;      // deleting tens of GB: not while the lock is held
    lock(&s).reshape_state().map_err(|e| e.to_string())
}

/// Runs `job` on a thread; its progress goes to the window as "env-event" events, the end as {"kind": "done"|"error"}.
fn spawn_task(app: tauri::AppHandle, tasks: &Tasks, channel: &'static str, job: impl FnOnce(&AtomicBool, &mut dyn FnMut(Value)) -> Result<Value, String> + Send + 'static) -> Out<()> {
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, cancel) = (tasks.busy.clone(), tasks.cancel.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit(channel, v); };
        let end = match job(&cancel, &mut emit) {
            Ok(data) => serde_json::json!({"kind": "done", "data": data}),
            Err(msg) => serde_json::json!({"kind": "error", "text": msg}),
        };
        busy.store(false, Ordering::SeqCst);
        let _ = app.emit(channel, end);
    });
    Ok(())
}

#[tauri::command]
fn env_install(app: tauri::AppHandle, tasks: State<Tasks>, unsloth: Option<bool>) -> Out<()> {
    let unsloth = unsloth.unwrap_or(true);
    spawn_task(app, &tasks, "env-event", move |cancel, emit| {
        ndforge_core::envmgr::tasks::install(unsloth, cancel, emit).map(|_| Value::Null).map_err(|e| e.to_string())
    })
}

#[tauri::command]
fn env_selftest(app: tauri::AppHandle, tasks: State<Tasks>) -> Out<()> {
    spawn_task(app, &tasks, "env-event", |cancel, emit| ndforge_core::envmgr::tasks::selftest(cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn env_cancel(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
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
        .manage(App(Mutex::new(AppModel::load())))
        .manage(Tasks::default())
        .invoke_handler(tauri::generate_handler![get_state, create_project, open_project, remove_world, check_bundle, apply_bundle, set_setting, set_theme, help, env_status, env_install, env_selftest, env_cancel, dataset_state, dataset_generate, bench_start, bench_last, model_lookup, model_add, model_remove, problem_report, test_state, test_result, test_start, test_ask, test_save_transcript, teacher_set, teacher_check, teacher_start, teacher_clear, dataset_examples, dataset_set_status, dataset_set_status_matching, dataset_edit, dataset_save, studio_export, train_state, train_plan, train_start, set_shutdown_after, cancel_shutdown, train_stop, train_metrics, train_server_add, train_server_remove, train_server_check, unsloth_set, unsloth_remove, unsloth_check, export_state, export_start, copy_adapter, ollama_install, ollama_probe, ollama_check, ollama_server_add, ollama_server_remove, retrain_info, retrain_start, reshape_state, reshape_options, reshape_measurement, reshape_measure, reshape_start, reshape_stop, reshape_test, reshape_remove, heal_options, heal_estimate, heal_start, heal_stop, general_state, general_choose, general_add, general_remove, general_ack, general_check, hf_token_set, hf_token_clear])
        .run(tauri::generate_context!())
        .expect("error while running ND Lore Forge");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

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
