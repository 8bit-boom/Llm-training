//! The Storage page's backend (`ndforge_core::storage`): the scan, the guards, the deletion. Nothing here touches a real folder: the config, the home and the Hugging Face cache are temporary.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, MutexGuard};

use ndforge_core::project::Project;
use ndforge_core::storage::*;
use serde_json::{json, Value};
use tempfile::{tempdir, TempDir};

static ENV: Mutex<()> = Mutex::new(());

/// A temporary world: config folder, home, Hugging Face cache and one project. The environment is global, so the tests of this file run one at a time.
struct World {
    _lock: MutexGuard<'static, ()>,
    d: TempDir,
    p: Project,
}

fn world() -> World {
    let lock = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    std::env::set_var("HOME", d.path().join("home"));
    std::env::set_var("USERPROFILE", d.path().join("home"));
    std::env::set_var("HF_HUB_CACHE", d.path().join("hub"));
    std::env::remove_var("HF_HOME");
    std::fs::create_dir_all(d.path().join("cfg")).unwrap();
    std::fs::create_dir_all(d.path().join("hub")).unwrap();
    let p = Project::create(&d.path().join("projects"), "Ashfall Reach").unwrap();
    World { _lock: lock, d, p }
}

fn put(path: impl AsRef<Path>, bytes: usize) {
    let path = path.as_ref();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, vec![7u8; bytes]).unwrap();
}

fn ctx(w: &World) -> Ctx<'_> {
    Ctx { project: Some(&w.p), busy: false, vllm: None }
}

fn items(w: &World) -> Vec<Item> {
    scan_items(&ctx(w))
}

fn find<'a>(items: &'a [Item], kind: &str, label_has: &str) -> &'a Item {
    items.iter().find(|i| i.kind == kind && i.label.contains(label_has)).unwrap_or_else(|| panic!("no {kind} item with {label_has}: {:?}", items.iter().map(|i| (i.kind, i.label.clone())).collect::<Vec<_>>()))
}

fn opts(confirm: &str) -> Options {
    Options { confirm: confirm.into(), ..Options::default() }
}

fn made_model(w: &World, name: &str, healed: bool) -> PathBuf {
    let dir = w.p.folder.join("models").join(name);
    put(dir.join("model").join("model.safetensors"), 4000);
    put(dir.join("model").join("config.json"), 20);
    std::fs::write(dir.join("model").join("ndforge_reshape.json"), json!({"origin": {"repo": "org/Qwen3.8-27B"}}).to_string()).unwrap();
    std::fs::write(dir.join("state.json"), json!({"status": "done"}).to_string()).unwrap();
    if healed {
        put(dir.join("healed").join("model.safetensors"), 3000);
        put(dir.join("healed").join("config.json"), 20);
    }
    dir
}

fn export_dir(w: &World, name: &str, with_gguf: bool) -> PathBuf {
    let dir = w.p.folder.join("exports").join(name);
    put(dir.join("model-f16.gguf"), 9000);
    if with_gguf {
        put(dir.join("model-Q4_K_M.gguf"), 2500);
    }
    put(dir.join("merged").join("model.safetensors"), 5000);
    std::fs::write(dir.join("meta.json"), json!({"name": name, "quant": "Q4_K_M", "gguf_file": "model-Q4_K_M.gguf", "base_repo": "x"}).to_string()).unwrap();
    dir
}

// ------------------------------------------------------------------------------------------------------------------ sizes and the scan

#[test]
fn sizes_add_up_and_a_link_counts_nothing() {
    let w = world();
    let d = w.d.path().join("sz");
    put(d.join("a.bin"), 1000);
    put(d.join("sub").join("b.bin"), 234);
    let outside = w.d.path().join("outside");
    put(outside.join("big.bin"), 50_000);
    #[cfg(unix)]
    std::os::unix::fs::symlink(&outside, d.join("link")).unwrap();
    assert_eq!(dir_size(&d, &AtomicBool::new(false)), 1234, "the linked folder is not counted");
    assert_eq!(dir_size(&d.join("a.bin"), &AtomicBool::new(false)), 1000);
    assert_eq!(dir_size(&d.join("missing"), &AtomicBool::new(false)), 0);
}

#[test]
fn the_scan_lists_models_exports_leftovers_and_caches_with_ids_sizes_and_status() {
    let w = world();
    made_model(&w, "cut-a", true);
    made_model(&w, "cut-b", false);
    export_dir(&w, "run1-Q4_K_M", true);
    put(w.p.folder.join("exports").join("run1-gguf-work").join("tmp.bin"), 100);
    put(w.p.folder.join("exports").join("x.gguf.partial"), 100);
    put(w.p.folder.join("models").join("_teacher_cache").join("abc").join("part-0.npy"), 700);
    put(w.p.folder.join("quantcheck").join("q1").join("base.kld"), 800);
    put(w.d.path().join("hub").join("models--org--Qwen3.8-27B").join("blobs").join("x"), 600);
    let mut events = 0;
    let r = scan(&ctx(&w), &AtomicBool::new(false), &mut |e| {
        events += 1;
        assert!(e["kind"] == "item" || e["kind"] == "size", "{e}");
    });
    assert!(events >= 20, "every item arrives as an event, then every size: {events}");
    let list = r["items"].as_array().unwrap();
    let get = |kind: &str, has: &str| list.iter().find(|i| i["kind"] == kind && i["label"].as_str().unwrap().contains(has)).unwrap_or_else(|| panic!("{kind} {has}: {list:?}")).clone();
    assert!(get("model_entry", "cut-a")["size"].as_u64().unwrap() >= 7020, "model + healed folders add up");
    assert!(get("model_entry", "cut-b")["model"] == true && get("model_entry", "cut-a")["from"].as_str().unwrap().contains("org/Qwen3.8-27B"));
    assert_eq!(get("export_f16", "run1-Q4_K_M")["size"], 9000);
    assert_eq!(get("export_merged", "run1-Q4_K_M")["size"], 5000);
    assert_eq!(get("teacher_cache", "abc")["size"], 700);
    assert_eq!(get("quant_logits", "q1")["size"], 800);
    assert_eq!(get("hf_repo", "org/Qwen3.8-27B")["size"], 600);
    assert!(get("leftover", "gguf-work")["suggest"].is_string() && get("leftover", "x.gguf.partial")["suggest"].is_string());
    assert!(list.iter().all(|i| i.get("path").is_none() && i["id"].as_str().unwrap().starts_with('s')), "the window never gets a full path");
    assert!(list.iter().all(|i| i["path_short"].is_string()));
}

#[test]
fn suggestions_name_the_intermediates_and_never_the_final_files() {
    let w = world();
    made_model(&w, "cut-a", true);
    export_dir(&w, "run1-Q4_K_M", true);
    export_dir(&w, "run2-Q4_K_M", false);
    std::fs::remove_file(w.p.folder.join("exports").join("run2-Q4_K_M").join("model-Q4_K_M.gguf")).ok();
    put(w.p.folder.join("quantcheck").join("q1").join("base.kld"), 800);
    let r = scan(&ctx(&w), &AtomicBool::new(false), &mut |_| {});
    let sug: Vec<&str> = r["suggestions"]["ids"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let list = r["items"].as_array().unwrap();
    let name_of = |id: &str| list.iter().find(|i| i["id"] == id).unwrap()["label"].as_str().unwrap().to_string();
    let labels: Vec<String> = sug.iter().map(|i| name_of(i)).collect();
    assert!(labels.iter().any(|l| l.contains("run1-Q4_K_M: the 16-bit GGUF")), "{labels:?}");
    assert!(labels.iter().any(|l| l.contains("run1-Q4_K_M: the merged")));
    assert!(!labels.iter().any(|l| l.contains("run2-Q4_K_M: the 16-bit GGUF")), "without a quantized file the 16-bit one is the product: {labels:?}");
    assert!(labels.iter().any(|l| l.contains("the cut before healing")) || !r["items"].as_array().unwrap().iter().any(|i| i["kind"] == "cut_only" && i["suggest"].is_string()), "{labels:?}");
    assert!(labels.iter().any(|l| l.contains("base.kld")));
    assert!(!labels.iter().any(|l| l.contains("(cut and healed)") || l.contains("(healed)")), "a whole model is never a suggestion: {labels:?}");
    assert!(r["suggestions"]["total_bytes"].as_u64().unwrap() >= 9000 + 5000 + 800);
    let preset = r["suggestions"]["preset"]["ids"].as_array().unwrap();
    assert!(!preset.is_empty() && preset.iter().all(|id| sug.contains(&id.as_str().unwrap())));
    assert!(!preset.iter().any(|id| name_of(id.as_str().unwrap()).contains("base.kld")), "the preset keeps to the intermediates of a run");
}

// ------------------------------------------------------------------------------------------------------------------ the guards

#[test]
fn the_guard_refuses_what_is_outside_the_folders_this_program_owns() {
    let w = world();
    let mut roots = Roots::new(Some(&w.p));
    let outside = w.d.path().join("owner-original-model");
    put(outside.join("model.safetensors"), 10);
    assert!(roots.guard(&outside).unwrap_err().0.contains("not inside"), "a model on another drive is never touched");
    assert!(roots.guard(&w.p.folder).is_err(), "the project folder");
    assert!(roots.guard(&w.p.folder.join("datasets")).is_err(), "datasets are not an owned folder");
    assert!(roots.guard(&w.p.folder.join("project.toml")).is_err(), "the project file");
    assert!(roots.guard(&w.p.folder.join("models")).is_err(), "a root as a whole");
    assert!(roots.guard(&w.p.folder.join("models").join("..").join("datasets")).unwrap_err().0.contains(".."), "dot-dot tricks");
    put(w.p.folder.join("models").join("m").join("f"), 1);
    assert!(roots.guard(&w.p.folder.join("models").join("m")).is_ok());
    assert!(roots.guard(&w.p.folder.join("models").join("nope")).is_err(), "a path that is not there");
    roots.add(w.d.path().join("extra"));
    std::fs::create_dir_all(w.d.path().join("extra").join("x")).unwrap();
    assert!(roots.guard(&w.d.path().join("extra").join("x")).is_ok(), "a root added before it existed has the same spelling as one that exists");
}

#[test]
fn a_root_spelled_in_another_equivalent_way_still_guards_the_same() {
    let w = world();
    let models = w.p.folder.join("models");
    put(models.join("m").join("f"), 1);
    let base = w.p.folder.to_string_lossy().to_string();
    let m = models.join("m");
    // `..` segments, a trailing separator, a `.` segment, the other kind of separator
    let dotdot = w.p.folder.join("exports").join("..").join("models").join("m");
    let dotted = w.p.folder.join(".").join("models").join("m");
    let trailing = PathBuf::from(format!("{}{}", m.to_string_lossy(), std::path::MAIN_SEPARATOR));
    let mut spellings = vec![m.clone(), dotted, trailing];
    spellings.push(PathBuf::from(m.to_string_lossy().replace('\\', "/")));
    let roots = Roots::new(Some(&w.p));
    for sp in &spellings {
        assert!(roots.guard(sp).is_ok(), "{sp:?} should be inside models/");
    }
    assert!(roots.guard(&dotdot).unwrap_err().0.contains(".."), "a `..` in the path itself is still refused");
    // a root spelled with a trailing separator, `.` and `..`, or a different drive-letter case
    let mut extra = Roots::default();
    let e = w.d.path().join("spelled");
    extra.add(PathBuf::from(format!("{}{}", w.d.path().join("x").join("..").join("spelled").to_string_lossy(), std::path::MAIN_SEPARATOR)));
    put(e.join("a").join("f"), 1);
    assert!(extra.guard(&e.join("a")).is_ok());
    assert!(extra.guard(&e).is_err(), "the root itself is never deleted");
    #[cfg(windows)]
    {
        let flipped: String = {
            let s = e.join("a").to_string_lossy().to_string();
            let mut c = s.chars();
            let first = c.next().unwrap();
            let f = if first.is_ascii_uppercase() { first.to_ascii_lowercase() } else { first.to_ascii_uppercase() };
            format!("{f}{}", c.as_str())
        };
        assert!(extra.guard(Path::new(&flipped)).is_ok(), "the drive letter's case does not matter");
    }
    let _ = base;
}

#[cfg(unix)]
#[test]
fn a_link_is_removed_itself_and_what_it_points_to_is_not_followed() {
    let w = world();
    let outside = w.d.path().join("owner-original-model");
    put(outside.join("model.safetensors"), 4096);
    let link = w.p.folder.join("models").join("linked");
    std::os::unix::fs::symlink(&outside, &link).unwrap();
    let roots = Roots::new(Some(&w.p));
    // a path THROUGH the link leaves the owned folders
    assert!(roots.guard(&link.join("model.safetensors")).is_err());
    // the link itself is an item under models/: only the link goes
    let mut failures = Vec::new();
    let freed = remove_tree(&link, &AtomicBool::new(false), &mut failures);
    assert!(failures.is_empty() && freed == 0);
    assert!(!link.exists() && std::fs::symlink_metadata(&link).is_err(), "the link is gone");
    assert!(outside.join("model.safetensors").is_file(), "the folder it pointed to is untouched");
    // a link inside a folder that is deleted
    let dir = w.p.folder.join("models").join("m2");
    put(dir.join("a"), 10);
    std::os::unix::fs::symlink(&outside, dir.join("inner-link")).unwrap();
    let freed = remove_tree(&dir, &AtomicBool::new(false), &mut failures);
    assert_eq!(freed, 10);
    assert!(!dir.exists() && outside.join("model.safetensors").is_file());
}

#[test]
fn nothing_is_deleted_while_a_job_runs() {
    let w = world();
    let dir = export_dir(&w, "run1-Q4_K_M", true);
    let id = find(&items(&w), "export", "run1").id.clone();
    let busy = Ctx { busy: true, ..ctx(&w) };
    assert!(scan_items(&busy).iter().all(|i| i.status == Status::InUse));
    let e = plan_delete(&busy, &[id], &opts("DELETE")).unwrap_err();
    assert!(e.0.contains("job is running"), "{}", e.0);
    assert!(dir.is_dir());
}

/// Sets the modification time of a folder. Windows opens a folder only with FILE_FLAG_BACKUP_SEMANTICS and write access for the attributes.
fn set_dir_mtime(dir: &Path, t: std::time::SystemTime) {
    #[cfg(windows)]
    let f = {
        use std::os::windows::fs::OpenOptionsExt;
        std::fs::OpenOptions::new().write(true).custom_flags(0x0200_0000).open(dir).unwrap()
    };
    #[cfg(not(windows))]
    let f = std::fs::File::open(dir).unwrap();
    f.set_modified(t).unwrap();
}

#[test]
fn an_id_that_is_not_in_a_fresh_scan_is_refused() {
    let w = world();
    let dir = export_dir(&w, "run1-Q4_K_M", true);
    let it = find(&items(&w), "export", "run1").clone();
    // a made-up id
    assert!(plan_delete(&ctx(&w), &["s0000000000000000".into()], &opts("")).unwrap_err().0.contains("changed since"));
    // the folder changed after the scan (another modification time): the old id is refused
    set_dir_mtime(&dir, std::time::SystemTime::now() + std::time::Duration::from_secs(600));
    let e = plan_delete(&ctx(&w), &[it.id.clone()], &opts("")).unwrap_err();
    assert!(e.0.contains("changed since"), "{}", e.0);
    assert!(dir.is_dir());
    // the folder is gone
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(plan_delete(&ctx(&w), &[it.id], &opts("")).is_err());
    assert!(plan_delete(&ctx(&w), &[], &opts("")).is_err(), "nothing ticked");
}

#[test]
fn a_model_or_more_than_five_gigabytes_needs_the_word_delete_and_a_pinned_model_needs_a_confirmation() {
    let w = world();
    made_model(&w, "cut-b", false);
    let it = find(&items(&w), "model_entry", "cut-b").clone();
    let e = delete(&ctx(&w), &[it.id.clone()], &opts(""), &AtomicBool::new(false), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("DELETE"), "{}", e.0);
    assert!(delete(&ctx(&w), &[it.id.clone()], &opts("delete"), &AtomicBool::new(false), &mut |_| {}).is_err(), "the word is exact");
    let pinned = Options { pinned: vec![it.model_repo.clone().unwrap()], confirm: "DELETE".into(), ..Options::default() };
    assert!(plan_delete(&ctx(&w), &[it.id.clone()], &pinned).unwrap_err().0.contains("chosen on the Slim or Auto page"));
    let ok = Options { allow_pinned: true, ..pinned };
    let r = delete(&ctx(&w), &[it.id], &ok, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["deleted"].as_array().unwrap().len(), 1);
    assert!(!w.p.folder.join("models").join("cut-b").exists());
    // a small plain item needs no word
    export_dir(&w, "run9", true);
    let f16 = find(&items(&w), "export_f16", "run9").id.clone();
    assert!(delete(&ctx(&w), &[f16], &opts(""), &AtomicBool::new(false), &mut |_| {}).is_ok());
}

// ------------------------------------------------------------------------------------------------------------------ deleting

#[test]
fn a_dry_run_says_exactly_what_the_real_deletion_removes() {
    let w = world();
    made_model(&w, "cut-a", true);
    export_dir(&w, "run1-Q4_K_M", true);
    put(w.p.folder.join("exports").join("run1-gguf-work").join("tmp.bin"), 100);
    let all = items(&w);
    let ids: Vec<String> = vec![find(&all, "export_f16", "run1").id.clone(), find(&all, "export_merged", "run1").id.clone(), find(&all, "leftover", "gguf-work").id.clone(),
                                find(&all, "export", "run1-Q4_K_M").id.clone() /* contains the other two: they go with it */];
    let dry = delete(&ctx(&w), &ids, &Options { dry_run: true, ..Options::default() }, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(dry["dry_run"], true);
    let planned: Vec<String> = dry["items"].as_array().unwrap().iter().map(|i| i["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(planned.len(), 2, "an item inside another chosen one is not listed twice: {planned:?}");
    assert!(w.p.folder.join("exports").join("run1-Q4_K_M").join("model-f16.gguf").is_file(), "a dry run removes nothing");
    let real = delete(&ctx(&w), &ids, &opts("DELETE"), &AtomicBool::new(false), &mut |_| {}).unwrap();
    let removed: Vec<String> = real["deleted"].as_array().unwrap().iter().map(|i| i["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(removed, planned);
    assert_eq!(real["freed_bytes"], dry["total_bytes"], "the freed bytes are the planned bytes");
    assert!(!w.p.folder.join("exports").join("run1-Q4_K_M").exists() && !w.p.folder.join("exports").join("run1-gguf-work").exists());
    assert!(w.p.folder.join("models").join("cut-a").is_dir(), "what was not chosen stays");
    let led = deleted_ledger(&w.p);
    assert_eq!(led.len(), 2, "{led:?}");
}

#[test]
fn a_part_of_an_export_can_go_and_the_export_still_lists() {
    let w = world();
    let dir = export_dir(&w, "run1-Q4_K_M", true);
    let all = items(&w);
    let ids = vec![find(&all, "export_f16", "run1").id.clone(), find(&all, "export_merged", "run1").id.clone()];
    delete(&ctx(&w), &ids, &opts("DELETE"), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert!(!dir.join("model-f16.gguf").exists() && !dir.join("merged").exists() && dir.join("model-Q4_K_M.gguf").is_file());
    let list = ndforge_core::export::runs::list_exports(&w.p);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["gguf_exists"], true);
    assert_eq!(list[0]["merged_exists"], false);
}

#[test]
fn one_failure_is_reported_and_the_others_still_go() {
    let w = world();
    export_dir(&w, "run1-Q4_K_M", true);
    let vl = ndforge_core::settings::VllmSettings { kind: "docker".into(), model_volume: "ndforge-vllm-hf-test-none".into(), ..Default::default() };
    let c = Ctx { project: Some(&w.p), busy: false, vllm: Some(&vl) };
    let all = scan_items(&c);
    let vol = find(&all, "vllm_volume", "ndforge-vllm-hf-test-none").id.clone();
    let f16 = find(&all, "export_f16", "run1").id.clone();
    let r = delete(&c, &[vol, f16], &opts("DELETE"), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["deleted"].as_array().unwrap().len(), 1, "{r}");
    assert_eq!(r["failed"].as_array().unwrap().len(), 1, "{r}");
    assert!(r["failed"][0]["errors"][0]["error"].as_str().unwrap().len() > 5);
    assert!(!w.p.folder.join("exports").join("run1-Q4_K_M").join("model-f16.gguf").exists());
}

#[cfg(unix)]
#[test]
fn a_file_that_cannot_be_removed_is_listed_with_its_path() {
    use std::os::unix::fs::PermissionsExt;
    if unsafe { libc_geteuid() } == 0 {
        return;          // root can remove anything: nothing to show
    }
    let w = world();
    let dir = w.p.folder.join("exports").join("locked");
    put(dir.join("sub").join("f.bin"), 10);
    std::fs::set_permissions(dir.join("sub"), std::fs::Permissions::from_mode(0o555)).unwrap();
    let mut failures = Vec::new();
    remove_tree(&dir, &AtomicBool::new(false), &mut failures);
    assert!(!failures.is_empty() && failures[0].path.contains("f.bin"), "{failures:?}");
    std::fs::set_permissions(dir.join("sub"), std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
unsafe fn libc_geteuid() -> u32 {
    extern "C" {
        fn geteuid() -> u32;
    }
    geteuid()
}

#[test]
fn a_cancel_stops_between_files_and_reports_it() {
    let w = world();
    export_dir(&w, "run1-Q4_K_M", true);
    let id = find(&items(&w), "export", "run1-Q4_K_M").id.clone();
    let r = delete(&ctx(&w), &[id], &opts("DELETE"), &AtomicBool::new(true), &mut |_| {}).unwrap();
    assert_eq!(r["cancelled"], true);
    assert!(w.p.folder.join("exports").join("run1-Q4_K_M").join("model-Q4_K_M.gguf").is_file(), "nothing went after the cancel");
}

#[test]
fn old_job_logs_go_and_new_ones_stay() {
    let w = world();
    let jobs = w.d.path().join("cfg").join("logs").join("jobs");
    put(jobs.join("old.log"), 500);
    put(jobs.join("new.log"), 500);
    let old = std::fs::File::options().write(true).open(jobs.join("old.log")).unwrap();
    old.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(40 * 86400)).unwrap();
    let it = find(&items(&w), "old_logs", "older than").clone();
    let r = delete(&ctx(&w), &[it.id], &opts(""), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["freed_bytes"], 500);
    assert!(!jobs.join("old.log").exists() && jobs.join("new.log").is_file());
}

#[test]
fn a_downloaded_model_is_listed_with_the_warning_that_other_programs_share_the_cache() {
    let w = world();
    put(w.d.path().join("hub").join("models--someone--other-model").join("blobs").join("b"), 321);
    let all = items(&w);
    let it = find(&all, "hf_repo", "someone/other-model");
    assert!(it.note.as_ref().unwrap().contains("another program may have downloaded it") && it.note.as_ref().unwrap().contains("share this cache"));
    assert!(it.model, "a model needs the typed word");
    let r = delete(&ctx(&w), &[it.id.clone()], &opts("DELETE"), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["freed_bytes"], 321);
    assert!(!w.d.path().join("hub").join("models--someone--other-model").exists() && w.d.path().join("hub").is_dir(), "the cache root stays");
}

#[test]
fn the_volume_command_follows_the_engine_kind() {
    let mut s = ndforge_core::settings::VllmSettings::default();
    s.kind = "docker".into();
    assert_eq!(volume_command(&s, "ndforge-vllm-hf").unwrap(), vec!["docker", "volume", "rm", "ndforge-vllm-hf"]);
    s.kind = "docker_wsl".into();
    s.distro = "Ubuntu".into();
    assert_eq!(volume_command(&s, "v").unwrap(), vec!["wsl.exe", "-d", "Ubuntu", "--", "docker", "volume", "rm", "v"]);
    s.kind = "wslc".into();
    assert_eq!(volume_command(&s, "v").unwrap(), vec!["wslc", "volume", "remove", "v"]);
    s.wslc = json!({"volume_remove": "wslc volume rm {name}"});
    assert_eq!(volume_command(&s, "v").unwrap(), vec!["wslc", "volume", "rm", "v"]);
    s.kind = "url".into();
    assert!(volume_command(&s, "v").is_none());
}

#[test]
fn the_other_screens_do_not_fail_on_a_deleted_model_or_export() {
    let w = world();
    made_model(&w, "cut-a", true);
    export_dir(&w, "run1-Q4_K_M", true);
    let all = items(&w);
    let ids = vec![find(&all, "model_entry", "cut-a").id.clone(), find(&all, "export", "run1-Q4_K_M").id.clone()];
    delete(&ctx(&w), &ids, &opts("DELETE"), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert!(ndforge_core::reshape::runs::list(&w.p).is_empty());
    assert!(ndforge_core::export::runs::list_exports(&w.p).is_empty());
    let q: Value = ndforge_core::export::quantcheck::state(&w.p);
    assert!(q.is_object());
    assert!(ndforge_core::reshape::auto::latest(&w.p).is_none());
    assert_eq!(deleted_ledger(&w.p).len(), 2);
}

#[test]
fn the_cache_folders_listed_as_one_item_can_be_emptied_and_the_folder_stays() {
    let w = world();
    let cfg = ndforge_core::paths::config_dir();
    put(w.p.folder.join("quantfit").join("a.json"), 100);
    put(w.p.folder.join("benchmarks").join("b.json"), 100);
    put(w.p.folder.join("imatrix_downloads").join("c.dat"), 100);
    put(cfg.join("general_cache").join("g.txt"), 100);
    put(cfg.join("vllm_selftest").join("r.json"), 100);
    let its = items(&w);
    let ids: Vec<String> = its.iter().filter(|i| i.kind == "cache_dir").map(|i| i.id.clone()).collect();
    assert_eq!(ids.len(), 5, "{:?}", its.iter().map(|i| i.label.clone()).collect::<Vec<_>>());
    let r = delete(&ctx(&w), &ids, &opts(""), &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert!(r["failed"].as_array().unwrap().is_empty(), "{r}");
    assert_eq!(r["deleted"].as_array().unwrap().len(), 5, "{r}");
    assert!(w.p.folder.join("quantfit").is_dir() && std::fs::read_dir(w.p.folder.join("quantfit")).unwrap().next().is_none(), "the folder stays, empty");
    assert!(cfg.join("general_cache").is_dir() && std::fs::read_dir(cfg.join("general_cache")).unwrap().next().is_none());
}
