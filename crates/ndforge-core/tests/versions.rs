//! One version number everywhere: the program (Cargo workspace), the window's installer (tauri.conf.json), the worker (its `__version__` and pyproject).
use std::path::PathBuf;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn quoted_after(text: &str, key: &str) -> Option<String> {
    let line = text.lines().find(|l| l.trim_start().starts_with(key))?;
    Some(line.split('"').nth(1)?.to_string())
}

#[test]
fn the_program_the_installer_and_the_worker_have_the_same_version() {
    let r = repo();
    let read = |p: &[&str]| std::fs::read_to_string(p.iter().fold(r.clone(), |a, b| a.join(b)));
    let (Ok(cargo), Ok(tauri), Ok(init), Ok(pyproject)) = (read(&["Cargo.toml"]), read(&["src-tauri", "tauri.conf.json"]), read(&["worker", "ndforge_worker", "__init__.py"]), read(&["worker", "pyproject.toml"])) else { return };
    let workspace = quoted_after(&cargo, "version").expect("version in Cargo.toml");
    assert_eq!(workspace, env!("CARGO_PKG_VERSION"), "the workspace version is the one this crate has");
    let tauri_v: serde_json::Value = serde_json::from_str(&tauri).unwrap();
    assert_eq!(tauri_v["version"], workspace.as_str(), "src-tauri/tauri.conf.json");
    assert_eq!(quoted_after(&init, "__version__").unwrap(), workspace, "worker/ndforge_worker/__init__.py");
    assert_eq!(quoted_after(&pyproject, "version").unwrap(), workspace, "worker/pyproject.toml");
}

#[test]
fn the_release_workflow_signs_only_with_a_certificate_and_never_publishes_a_dry_run() {
    let Ok(text) = std::fs::read_to_string(repo().join(".github").join("workflows").join("release.yml")) else { return };
    assert!(text.contains("WINDOWS_CERT_PFX") && text.contains("WINDOWS_CERT_PASSWORD") && text.contains("unsigned"));
    assert!(text.contains("if: startsWith(github.ref, 'refs/tags/')"), "only a tag publishes");
    assert!(text.contains("SHA256SUMS.txt"));
    for (name, path) in [("audit", "audit.yml")] {
        let a = std::fs::read_to_string(repo().join(".github").join("workflows").join(path)).unwrap_or_default();
        assert!(a.is_empty() || a.contains("continue-on-error: true"), "{name} never blocks");
    }
}
