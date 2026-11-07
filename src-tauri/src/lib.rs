//! The window shell. All logic is in `ndforge-core` (`AppModel`); these commands forward to it and return its JSON, or the
//! plain-language error text, to the page.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ndforge_core::help::help_html;
use ndforge_core::model::AppModel;
use serde_json::Value;
use tauri::{Emitter, State};

struct App(Mutex<AppModel>);

/// One long job at a time (install or self-test); `cancel` stops it.
#[derive(Default)]
struct Tasks {
    busy: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
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

#[tauri::command]
fn help(key: String) -> String {
    help_html(&key)
}

#[tauri::command]
fn env_status() -> Value {
    ndforge_core::envmgr::tasks::status()
}

/// Runs `job` on a thread; its progress goes to the window as "env-event" events, the end as {"kind": "done"|"error"}.
fn spawn_task(app: tauri::AppHandle, tasks: &Tasks, job: impl FnOnce(&AtomicBool, &mut dyn FnMut(Value)) -> Result<Value, String> + Send + 'static) -> Out<()> {
    if tasks.busy.swap(true, Ordering::SeqCst) {
        return Err("Another job is still running. Wait for it or press Cancel.".into());
    }
    tasks.cancel.store(false, Ordering::SeqCst);
    let (busy, cancel) = (tasks.busy.clone(), tasks.cancel.clone());
    std::thread::spawn(move || {
        let emitter = app.clone();
        let mut emit = move |v: Value| { let _ = emitter.emit("env-event", v); };
        let end = match job(&cancel, &mut emit) {
            Ok(data) => serde_json::json!({"kind": "done", "data": data}),
            Err(msg) => serde_json::json!({"kind": "error", "text": msg}),
        };
        busy.store(false, Ordering::SeqCst);
        let _ = app.emit("env-event", end);
    });
    Ok(())
}

#[tauri::command]
fn env_install(app: tauri::AppHandle, tasks: State<Tasks>) -> Out<()> {
    spawn_task(app, &tasks, |cancel, emit| {
        ndforge_core::envmgr::tasks::install(cancel, emit).map(|_| Value::Null).map_err(|e| e.to_string())
    })
}

#[tauri::command]
fn env_selftest(app: tauri::AppHandle, tasks: State<Tasks>) -> Out<()> {
    spawn_task(app, &tasks, |cancel, emit| ndforge_core::envmgr::tasks::selftest(cancel, emit).map_err(|e| e.to_string()))
}

#[tauri::command]
fn env_cancel(tasks: State<Tasks>) {
    tasks.cancel.store(true, Ordering::SeqCst);
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(App(Mutex::new(AppModel::load())))
        .manage(Tasks::default())
        .invoke_handler(tauri::generate_handler![get_state, create_project, open_project, check_bundle, apply_bundle, set_setting, set_theme, help, env_status, env_install, env_selftest, env_cancel, dataset_state, dataset_generate, dataset_examples, dataset_set_status, dataset_set_status_matching, dataset_edit, dataset_save])
        .run(tauri::generate_context!())
        .expect("error while running ND Lore Forge");
}
