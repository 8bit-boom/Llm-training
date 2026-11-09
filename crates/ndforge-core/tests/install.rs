//! Installed layouts, the guide and its pictures, the problem report, and that the packaging files agree with the program.
use std::path::{Path, PathBuf};

use ndforge_core::envmgr::{find_tool_in, resource_dirs_for};
use ndforge_core::help::{help_html, help_markdown};
use ndforge_core::settings::Settings;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn the_worker_and_uv_are_found_in_every_installed_layout() {
    let d = tempfile::tempdir().unwrap();
    // Windows installer / portable: everything next to the program
    let win = d.path().join("win");
    std::fs::create_dir_all(win.join("worker/ndforge_worker")).unwrap();
    std::fs::create_dir_all(win.join("tools")).unwrap();
    std::fs::write(win.join("tools/uv.exe"), "x").unwrap();
    let dirs = resource_dirs_for(&win, None);
    assert!(dirs.iter().map(|x| x.join("worker/ndforge_worker")).any(|x| x.exists()));
    assert_eq!(find_tool_in(&dirs, "uv.exe"), Some(win.join("tools/uv.exe")));
    // AppImage: $APPDIR/usr/lib/ND Lore Forge (Tauri's resource folder), the program in $APPDIR/usr/bin
    let app = d.path().join("app");
    let res = app.join("usr/lib/ND Lore Forge");
    std::fs::create_dir_all(res.join("worker/ndforge_worker")).unwrap();
    std::fs::create_dir_all(res.join("tools")).unwrap();
    std::fs::write(res.join("tools/uv"), "x").unwrap();
    let dirs = resource_dirs_for(&app.join("usr/bin"), Some(&app));
    assert!(dirs.contains(&res));
    assert_eq!(find_tool_in(&dirs, "uv"), Some(res.join("tools/uv")));
    // .deb: /usr/bin/program, /usr/lib/<name>/...
    let deb = d.path().join("deb");
    std::fs::create_dir_all(deb.join("usr/lib/ND Lore Forge/tools")).unwrap();
    std::fs::write(deb.join("usr/lib/ND Lore Forge/tools/uv"), "x").unwrap();
    assert!(find_tool_in(&resource_dirs_for(&deb.join("usr/bin"), None), "uv").is_some());
    assert_eq!(find_tool_in(&resource_dirs_for(&d.path().join("empty"), None), "uv"), None);
}

#[test]
fn every_picture_of_the_guide_exists_and_the_guide_is_in_the_help() {
    let md = help_markdown("guide");
    let pics: Vec<&str> = md.lines().filter_map(|l| l.strip_prefix("![")?.split("](").nth(1)?.strip_suffix(')')).collect();
    assert!(pics.len() >= 7, "{pics:?}");
    for p in &pics {
        let f = repo().join("ui").join(p);
        assert!(f.is_file() && std::fs::metadata(&f).unwrap().len() > 10_000, "missing or empty picture {p}");
    }
    assert!(help_html("guide").contains("<img src=\"guide/welcome.png\""));
    for step in ["Install", "Create a project", "Get the world data", "Training setup", "Make the training data", "Train", "Test", "Use it in nd-world", "Retrain"] {
        assert!(md.contains(step), "guide lacks {step}");
    }
    let t = help_markdown("troubleshooting");
    for problem in ["WebView2", "nvidia-smi", "Out of memory", "Ollama cannot be reached", "Cannot connect", "No questions were kept"] {
        assert!(t.contains(problem), "troubleshooting lacks {problem}");
    }
    assert!(!help_markdown("setup").contains("Later versions"));
}

#[test]
fn the_problem_report_has_the_facts_and_no_secrets() {
    let d = tempfile::tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    ndforge_core::secrets::set("http://truenas:8765", "SECRETTOKEN123").unwrap();
    ndforge_core::secrets::set("teacher", "sk-SECRETKEY456").unwrap();
    let mut s = Settings::default();
    s.train_servers.push(ndforge_core::settings::TrainServer { name: "TrueNAS".into(), url: "http://truenas:8765".into() });
    s.teacher.model = "gemma4:e4b".into();
    let r = ndforge_core::diagnose::report(None, &s);
    for want in ["ND Lore Forge 0.1.0", "== This computer ==", "== Training software ==", "worker:", "uv:", "TrueNAS (http://truenas:8765)", "gemma4:e4b", "no project open"] {
        assert!(r.contains(want), "report lacks {want}:\n{r}");
    }
    assert!(!r.contains("SECRETTOKEN123") && !r.contains("SECRETKEY456"));
    let mut p = ndforge_core::project::Project::create(&d.path().join("p"), "Ashfall Reach").unwrap();
    let f = d.path().join("w.ndtrain");
    ndforge_core::sample::build(&f, 1, None).unwrap();
    p.import_bundle(&ndforge_core::bundle::open_bundle(&f).unwrap()).unwrap();
    let r = ndforge_core::diagnose::report(Some(&p), &s);
    assert!(r.contains("Ashfall Reach") && r.contains("steps done: data=yes"), "{r}");
}

#[test]
fn the_packaging_files_agree_with_the_program() {
    let r = repo();
    let release: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(r.join("src-tauri/tauri.release.conf.json")).unwrap()).unwrap();
    let res = release["bundle"]["resources"].as_object().unwrap();
    // the program looks for "worker" and "tools" in its resource folder
    assert_eq!(res.values().map(|v| v.as_str().unwrap()).collect::<std::collections::BTreeSet<_>>(), ["tools", "worker"].into_iter().collect());
    let stage = std::fs::read_to_string(r.join("packaging/stage.sh")).unwrap();
    assert!(res.keys().all(|k| stage.contains(&format!("dist/stage/{}", k.rsplit('/').next().unwrap()))) || stage.contains("$OUT/worker"));
    let conf: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(r.join("src-tauri/tauri.conf.json")).unwrap()).unwrap();
    assert!(conf["bundle"].get("resources").is_none(), "dev builds must not need the staged files");
    let cargo = std::fs::read_to_string(r.join("src-tauri/Cargo.toml")).unwrap();
    let iss = std::fs::read_to_string(r.join("packaging/windows/ndforge.iss")).unwrap();
    assert!(cargo.contains("name = \"nd-lore-forge\"") && iss.contains("nd-lore-forge.exe"));
    assert!(iss.contains("{app}\\worker") && iss.contains("{app}\\tools") && iss.contains("PrivilegesRequired=lowest") && iss.contains("WebView2"));
    for f in ["src-tauri/icons/icon.ico"] {
        assert!(r.join(f).is_file(), "{f}");
    }
    assert!(std::fs::read_to_string(r.join("packaging/UV_VERSION")).unwrap().trim().split('.').count() == 3);
    let wf = std::fs::read_to_string(r.join(".github/workflows/release.yml")).unwrap();
    assert!(wf.contains("packaging/linux/build.sh") && wf.contains("packaging/windows/build.ps1"));
    let ok = std::process::Command::new("sh").arg("-n").arg(r.join("packaging/stage.sh")).status().unwrap();
    assert!(ok.success());
    assert!(std::process::Command::new("sh").arg("-n").arg(r.join("packaging/linux/build.sh")).status().unwrap().success());
}

#[test]
fn a_git_clone_has_working_install_scripts_and_the_program_finds_the_tools_they_fetch() {
    let r = repo();
    for f in ["install.sh", "run.sh", "packaging/fetch_uv.sh"] {
        assert!(std::process::Command::new("sh").arg("-n").arg(r.join(f)).status().unwrap().success(), "{f} has a syntax error");
    }
    let sh = std::fs::read_to_string(r.join("install.sh")).unwrap();
    let ps = std::fs::read_to_string(r.join("install.ps1")).unwrap();
    // both build with the feature that embeds the window, and both fetch uv into ./tools
    assert!(sh.contains("--features custom-protocol") && ps.contains("--features custom-protocol"));
    assert!(sh.contains("$ROOT/tools") && ps.contains("$root\\tools"));
    assert!(ps.contains("Get-FileHash") && std::fs::read_to_string(r.join("packaging/fetch_uv.sh")).unwrap().contains("sha256sum"));
    assert!(std::fs::read_to_string(r.join("run.cmd")).unwrap().contains("target\\release\\nd-lore-forge.exe"));
    assert!(std::fs::read_to_string(r.join("src-tauri/Cargo.toml")).unwrap().contains("custom-protocol = [\"tauri/custom-protocol\"]"));
    assert!(std::fs::read_to_string(r.join(".gitignore")).unwrap().lines().any(|l| l == "tools/"));
    // the program looks in <repo>/tools, where the scripts put uv
    let tools = r.join("tools");
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    let made = !tools.join(name).exists();
    if made {
        std::fs::create_dir_all(&tools).unwrap();
        std::fs::write(tools.join(name), "x").unwrap();
    }
    std::env::remove_var("NDFORGE_UV");
    let found = ndforge_core::envmgr::find_uv();
    let ok = found.as_ref().is_some_and(|p| p.is_file());    // checked before the stand-in file is removed
    if made {
        std::fs::remove_file(tools.join(name)).unwrap();
        let _ = std::fs::remove_dir(&tools);
    }
    assert!(ok, "find_uv found {found:?}");
}
