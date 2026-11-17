//! Atomic writes, and a damaged settings / secrets file being kept instead of overwritten.
use ndforge_core::{fsx, secrets, settings::Settings};
use std::sync::Mutex;

static ENV: Mutex<()> = Mutex::new(());

#[test]
fn the_old_content_stays_when_the_new_cannot_be_written() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("state.json");
    fsx::write_atomic(&f, b"old").unwrap();
    assert_eq!(std::fs::read(&f).unwrap(), b"old");
    // something that is not a file sits where the temporary file goes
    std::fs::create_dir(fsx::tmp_path(&f)).unwrap();
    assert!(fsx::write_atomic(&f, b"new").is_err());
    assert_eq!(std::fs::read(&f).unwrap(), b"old");
    std::fs::remove_dir(fsx::tmp_path(&f)).unwrap();
    fsx::write_atomic(&f, b"new").unwrap();
    assert_eq!(std::fs::read(&f).unwrap(), b"new");
    assert!(!fsx::tmp_path(&f).exists());
    // folders are made, JSON is pretty
    let deep = dir.path().join("a").join("b").join("x.json");
    fsx::write_json_atomic(&deep, &serde_json::json!({"k": 1})).unwrap();
    assert_eq!(serde_json::from_slice::<serde_json::Value>(&std::fs::read(&deep).unwrap()).unwrap()["k"], 1);
}

#[test]
fn rename_retry_replaces_files_and_reports_a_missing_source() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (dir.path().join("a.txt"), dir.path().join("b.txt"));
    std::fs::write(&a, "1").unwrap();
    std::fs::write(&b, "2").unwrap();
    fsx::rename_retry(&a, &b).unwrap();
    assert_eq!(std::fs::read_to_string(&b).unwrap(), "1");
    assert!(fsx::rename_retry(&a, &b).is_err());
}

#[test]
fn a_broken_settings_file_is_kept_and_reported_once() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", dir.path());
    // nothing there yet: defaults, no note, nothing created
    let s = Settings::load();
    assert!(s.load_note.is_none() && !dir.path().join("settings.json").exists());
    std::fs::write(dir.path().join("settings.json"), "{ this is not json").unwrap();
    let s = Settings::load();
    assert!(s.load_note.as_deref().unwrap_or("").contains("settings.json.broken-"), "{:?}", s.load_note);
    let kept: Vec<_> = std::fs::read_dir(dir.path()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.starts_with("settings.json.broken-")).collect();
    assert_eq!(kept.len(), 1);
    assert_eq!(std::fs::read_to_string(dir.path().join(&kept[0])).unwrap(), "{ this is not json");
    assert!(!dir.path().join("settings.json").exists());
    // saving writes good settings without the note; the next start is quiet
    let mut s = s;
    s.theme = "dark".into();
    s.save();
    let text = std::fs::read_to_string(dir.path().join("settings.json")).unwrap();
    assert!(!text.contains("load_note"));
    let again = Settings::load();
    assert!(again.load_note.is_none() && again.theme == "dark");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[test]
fn a_broken_secrets_file_is_kept_not_overwritten() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", dir.path());
    std::fs::write(dir.path().join("secrets.json"), "[[ nope").unwrap();
    assert_eq!(secrets::get("teacher"), None);
    secrets::set("teacher", "abc").unwrap();
    assert_eq!(secrets::get("teacher").as_deref(), Some("abc"));
    let kept = std::fs::read_dir(dir.path()).unwrap().flatten().find(|e| e.file_name().to_string_lossy().starts_with("secrets.json.broken-")).expect("the damaged file is kept");
    assert_eq!(std::fs::read_to_string(kept.path()).unwrap(), "[[ nope");
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}

#[cfg(unix)]
#[test]
fn secrets_are_private_from_the_first_byte() {
    use std::os::unix::fs::PermissionsExt;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", dir.path());
    secrets::set("k", "v").unwrap();
    let mode = std::fs::metadata(dir.path().join("secrets.json")).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let f = dir.path().join("p.bin");
    fsx::write_atomic_private(&f, b"x").unwrap();
    assert_eq!(std::fs::metadata(&f).unwrap().permissions().mode() & 0o777, 0o600);
    std::env::remove_var("NDFORGE_CONFIG_DIR");
}
