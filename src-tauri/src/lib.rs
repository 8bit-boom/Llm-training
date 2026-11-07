//! The window shell. All logic is in `ndforge-core` (`AppModel`); these commands forward to it and return its JSON, or the
//! plain-language error text, to the page.
use std::sync::Mutex;

use ndforge_core::help::help_html;
use ndforge_core::model::AppModel;
use serde_json::Value;
use tauri::State;

struct App(Mutex<AppModel>);

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
fn help(key: String) -> String {
    help_html(&key)
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(App(Mutex::new(AppModel::load())))
        .invoke_handler(tauri::generate_handler![get_state, create_project, open_project, check_bundle, apply_bundle, set_setting, help])
        .run(tauri::generate_context!())
        .expect("error while running ND Lore Forge");
}
