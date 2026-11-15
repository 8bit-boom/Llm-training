//! Merging importance matrices (export/imatrix_lib.rs, "merging matrices"): the choice and its old forms, the job and every refusal, the base models offered for the published
//! matrix, the merge after a creation, and fake-worker runs (unix only). Small files only.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::export::imatrix_lib as lib;
use ndforge_core::project::Project;
use ndforge_core::sample;
use ndforge_core::train::plan::{base_hyper, Method, Preset};
use ndforge_core::train::runs as truns;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn plain_project(root: &Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

fn project_with_data(root: &Path) -> Project {
    let mut p = plain_project(root);
    let f = root.join("w1.ndtrain");
    sample::build(&f, 1, None).unwrap();
    p.import_bundle(&open_bundle(&f).unwrap()).unwrap();
    store::generate_draft(&p, &Recipe::default()).unwrap();
    store::save_version(&p).unwrap();
    p
}

fn fake_done_run(p: &Project, name: &str, repo: Option<&str>) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(rdir.join("adapter")).unwrap();
    std::fs::write(rdir.join("adapter/adapter_config.json"), "{}").unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    let mut job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    if let Some(r) = repo {
        job["model"]["repo"] = json!(r);
    }
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "model_id": m.id, "method": "lora"}).to_string()).unwrap();
}

fn entry_on_disk(p: &Project, id: &str, files: &[(&str, &[u8])], info: Option<Value>) -> PathBuf {
    let d = lib::lib_dir(p).join(id);
    std::fs::create_dir_all(&d).unwrap();
    for (n, b) in files {
        std::fs::write(d.join(n), b).unwrap();
    }
    if let Some(i) = info {
        std::fs::write(d.join("imatrix.json"), serde_json::to_vec(&i).unwrap()).unwrap();
    }
    d
}

fn input(kind: &str, v: &str) -> lib::MergeInput {
    match kind {
        "library" => lib::MergeInput { kind: kind.into(), id: Some(v.into()), ..Default::default() },
        "file" => lib::MergeInput { kind: kind.into(), file: Some(v.into()), ..Default::default() },
        _ => lib::MergeInput { kind: kind.into(), base: Some(v.into()), ..Default::default() },
    }
}

fn spec(inputs: Vec<lib::MergeInput>) -> lib::MergeSpec {
    lib::MergeSpec { inputs, label: "my merge".into() }
}

// ------------------------------------------------------------------------------------------------------------------ the choice

#[test]
fn the_choice_and_old_specs_load() {
    let s: lib::MergeSpec = serde_json::from_value(json!({})).unwrap();
    assert!(s.inputs.is_empty() && s.label.is_empty());
    let s: lib::MergeSpec = serde_json::from_value(json!({"inputs": [{"kind": "published", "base": "Qwen/Qwen3.8-27B"}, {"kind": "library", "id": "a", "extra": 1}], "label": "x"})).unwrap();
    assert_eq!((s.inputs[0].base.as_deref(), s.inputs[1].id.as_deref(), s.inputs[1].file.clone()), (Some("Qwen/Qwen3.8-27B"), Some("a"), None));
    // a creation asked for before this feature existed has no merge
    let m: lib::MakeSpec = serde_json::from_value(json!({"source": {"kind": "run", "run": "r"}, "chunks": 64})).unwrap();
    assert!(!m.merge_published && m.chunks == 64);
    assert!(!lib::MakeSpec::default().merge_published);
}

#[test]
fn hub_ids_and_the_base_of_a_source() {
    for ok in ["Qwen/Qwen3.8-27B", "unsloth/Qwen3.8-27B-GGUF", " a/b "] {
        assert!(lib::is_hub_id(ok), "{ok}");
    }
    for bad in ["", "Qwen3.8-27B", "a/b/c", "/models/thin", "./x/y", "C:\\models\\thin", "C:/models", "a\\b", "owner/"] {
        assert!(!lib::is_hub_id(bad), "{bad}");
    }
    assert_eq!(lib::base_repo_of(&json!({"kind": "run", "base": {"repo": "Qwen/Qwen3.8-27B"}})).as_deref(), Some("Qwen/Qwen3.8-27B"));
    assert_eq!(lib::base_repo_of(&json!({"kind": "model", "model": "Qwen/Qwen3.5-4B"})).as_deref(), Some("Qwen/Qwen3.5-4B"));
    assert_eq!(lib::base_repo_of(&json!({"kind": "model", "model": "/m/thin"})), None);
    assert_eq!(lib::base_repo_of(&json!({"kind": "gguf", "tokenizer": "owner/name"})).as_deref(), Some("owner/name"));
    assert_eq!(lib::base_repo_of(&json!({"kind": "gguf", "tokenizer": null})), None);
    assert_eq!(lib::base_repo_of(&json!({"kind": "run", "base": {"repo": ""}})), None);
}

// ------------------------------------------------------------------------------------------------------------------ the job and every refusal

#[test]
fn the_job_carries_the_inputs_with_their_labels_models_and_chunks() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    entry_on_disk(&p, "20261101-1000-a", &[("imatrix.gguf", b"A")], Some(json!({"label": "mine", "model": "run1", "chunks": 64})));
    entry_on_disk(&p, "20261102-1000-b", &[("imatrix.dat", b"B")], None);
    let f = d.path().join("friend.gguf");
    std::fs::write(&f, "x").unwrap();
    let job = lib::build_merge_job(&p, &spec(vec![input("library", "20261101-1000-a"), input("library", "20261102-1000-b"), input("file", &f.display().to_string()), input("published", " Qwen/Qwen3.8-27B ")]), "20261115-1030-merged", Path::new("S")).unwrap();
    assert_eq!((job["job"].as_str(), job["id"].as_str(), job["label"].as_str(), job["base_repo"].as_str()), (Some("imatrix_merge"), Some("20261115-1030-merged"), Some("my merge"), Some("Qwen/Qwen3.8-27B")));
    assert_eq!((job["root"].clone(), job["cache"].clone()), (json!(lib::lib_dir(&p)), json!(lib::downloads_dir(&p))));
    assert!(lib::downloads_dir(&p).ends_with("imatrix_downloads") && !lib::downloads_dir(&p).starts_with(lib::lib_dir(&p)), "downloads are not library entries");
    let i = job["inputs"].as_array().unwrap();
    assert_eq!((i[0]["label"].as_str(), i[0]["model"].as_str(), i[0]["chunks"].as_u64(), i[0]["kind"].as_str()), (Some("mine"), Some("run1"), Some(64), Some("library")));
    assert!(i[0]["file"].as_str().unwrap().ends_with("imatrix.gguf") && i[1]["file"].as_str().unwrap().ends_with("imatrix.dat"), "a .dat-only entry is merged through its .dat");
    assert_eq!(i[1]["label"], "20261102-1000-b", "no label: the id");
    assert_eq!((i[2]["kind"].as_str(), i[3]["kind"].as_str(), i[3]["base"].as_str()), (Some("file"), Some("published"), Some("Qwen/Qwen3.8-27B")));
    assert!(job["base_repo"].is_string());
    let j2 = lib::build_merge_job(&p, &spec(vec![input("library", "20261101-1000-a"), input("library", "20261102-1000-b")]), "x", Path::new("S")).unwrap();
    assert!(j2["base_repo"].is_null());
}

#[test]
fn everything_a_person_could_get_wrong_is_a_plain_message() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    entry_on_disk(&p, "20261101-1000-a", &[("imatrix.gguf", b"A")], None);
    entry_on_disk(&p, "20261102-1000-b", &[("imatrix.gguf", b"B")], None);
    let (txt, gguf) = (d.path().join("n.txt"), d.path().join("m.gguf"));
    std::fs::write(&txt, "x").unwrap();
    std::fs::write(&gguf, "x").unwrap();
    let err = |inputs: Vec<lib::MergeInput>| lib::build_merge_job(&p, &spec(inputs), "x", Path::new("S")).unwrap_err().0;
    let a = || input("library", "20261101-1000-a");
    assert!(err(vec![]).contains("at least two matrices") && err(vec![a()]).contains("at least two matrices"));
    assert!(err(vec![a(), input("library", "")]).contains("Choose a saved matrix"));
    assert!(err(vec![a(), input("library", "gone")]).contains("no longer exists"));
    assert!(err(vec![a(), input("library", "../x")]).contains("not the name of a saved matrix"));
    assert!(err(vec![a(), input("file", "")]).contains("Choose the matrix file"));
    assert!(err(vec![a(), input("file", &txt.display().to_string())]).contains("Choose an importance matrix file"));
    assert!(err(vec![a(), input("file", "/nowhere/m.gguf")]).contains("was not found"));
    assert!(err(vec![a(), input("published", "  ")]).contains("Choose the base model"));
    assert!(err(vec![a(), input("nonsense", "x")]).contains("neither a saved matrix"));
    assert!(err(vec![a(), a()]).contains("twice"));
    assert!(err(vec![input("file", &gguf.display().to_string()), input("file", &gguf.display().to_string())]).contains("twice"));
    assert!(err(vec![input("published", "o/m"), input("published", "o/m")]).contains("twice"));
    entry_on_disk(&p, "20261103-1000-empty", &[], Some(json!({})));
    assert!(err(vec![a(), input("library", "20261103-1000-empty")]).contains("no longer exists"), "a folder without a matrix file is not an entry");
}

// ------------------------------------------------------------------------------------------------------------------ the page

#[test]
fn the_base_models_offered_are_the_hub_models_of_the_runs_and_the_list() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1", None);
    fake_done_run(&p, "run2", Some("/m/thin-folder"));
    fake_done_run(&p, "run3", Some("owner/custom-base"));
    let b = lib::published_bases(&p);
    let repos: Vec<&str> = b.iter().map(|x| x["repo"].as_str().unwrap()).collect();
    assert!(repos.contains(&"owner/custom-base") && !repos.iter().any(|r| r.contains("thin-folder")), "{repos:?}");
    assert!(b.iter().any(|x| x["label"].as_str().unwrap().contains("the base of the run run3")));
    assert_eq!(repos.len(), repos.iter().collect::<std::collections::BTreeSet<_>>().len(), "no duplicates");
    assert!(repos.iter().all(|r| r.matches('/').count() == 1));
    assert!(lib::state(&p)["published_bases"].as_array().unwrap().len() == repos.len());
}

// ------------------------------------------------------------------------------------------------------------------ fake workers (unix)

#[cfg(unix)]
fn fake_worker(root: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\njob=\"$5\"\nlibroot=$(sed -n 's/.*\"root\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nid=$(sed -n 's/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nstop=$(sed -n 's/.*\"stop_file\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
const MERGE_OK: &str = r#"id=$(grep -o '"id":"[0-9]*-[0-9]*-merged[^"]*"' "$job" | head -1 | cut -d'"' -f4)
cp "$job" "$libroot/last-job.json"
mkdir -p "$libroot/$id"
printf 'GGUF' > "$libroot/$id/imatrix.gguf"
echo '{"format":1,"label":"my merge","source_kind":"merged","source":"a merge of a, b","chunks":128,"created":"2026-11-15T10:30:00Z","via":"llama.cpp"}' > "$libroot/$id/imatrix.json"
echo '{"type":"progress","step":"imatrix","value":0.7,"text":"Combining the matrices"}'
echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"warnings\":[\"w\"],\"notes\":[]}}""#;

#[cfg(unix)]
#[test]
fn a_merge_runs_in_a_fake_worker_holds_its_parents_while_running_and_the_entry_appears() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), MERGE_OK);
    let p = plain_project(d.path());
    let (a, b) = ("20261101-1000-a", "20261102-1000-b");
    entry_on_disk(&p, a, &[("imatrix.gguf", b"A")], Some(json!({"label": "first"})));
    entry_on_disk(&p, b, &[("imatrix.gguf", b"B")], Some(json!({"label": "second"})));
    let mut held = None;
    let mut events = vec![];
    let r = lib::merge(&p, &spec(vec![input("library", a), input("library", b)]), &AtomicBool::new(false), &mut |e| { held = Some((lib::in_use(a), lib::in_use(b), lib::delete(&p, a).unwrap_err().0)); events.push(e); }).unwrap();
    let (ia, ib, why) = held.unwrap();
    assert!(ia && ib && why.contains("being used by a job"), "the parents cannot be deleted while merging");
    assert!(!lib::in_use(a) && !lib::in_use(b), "released afterwards");
    assert!(r["entry"]["id"].as_str().unwrap().ends_with("-merged"), "{r}");
    assert_eq!((r["entry"]["source_kind"].as_str(), r["entry"]["chunks"].as_u64(), r["entry"]["has_gguf"].clone()), (Some("merged"), Some(128), json!(true)));
    assert_eq!(r["warnings"], json!(["w"]));
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["text"] == "Combining the matrices"));
    assert_eq!(lib::list(&p).len(), 3);
    let sent: Value = serde_json::from_slice(&std::fs::read(lib::lib_dir(&p).join("last-job.json")).unwrap()).unwrap();
    assert_eq!(sent["inputs"].as_array().unwrap().len(), 2);
    assert!(!lib::lib_dir(&p).join("STOP").exists(), "the STOP file is cleaned up");
}

#[cfg(unix)]
#[test]
fn stop_and_failures_come_back_in_the_workers_words_and_the_refusals_come_first() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"while [ ! -f "$stop" ]; do sleep 0.05; done
echo '{"type":"error","code":"imatrix_stopped","message":"Cancelled."}'
exit 1"#);
    let p = plain_project(d.path());
    entry_on_disk(&p, "a", &[("imatrix.gguf", b"A")], None);
    entry_on_disk(&p, "b", &[("imatrix.gguf", b"B")], None);
    let s = spec(vec![input("library", "a"), input("library", "b")]);
    let kill = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let r = std::thread::scope(|sc| {
        sc.spawn(|| { std::thread::sleep(std::time::Duration::from_millis(300)); kill.store(true, std::sync::atomic::Ordering::SeqCst); });
        lib::merge(&p, &s, &kill, &mut |_| {})
    });
    assert_eq!(r.unwrap_err().0, "Cancelled.");
    assert!(t0.elapsed() < std::time::Duration::from_secs(20) && lib::list(&p).len() == 2);
    assert!(!lib::in_use("a"));
    fake_worker(d.path(), r#"echo '{"type":"error","code":"imatrix_failed","message":"There is no published matrix for Qwen/Unknown-Model: the Hub has no repository unsloth/Unknown-Model-GGUF."}'; exit 1"#);
    assert!(lib::merge(&p, &spec(vec![input("library", "a"), input("published", "Qwen/Unknown-Model")]), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("no published matrix"));
    assert!(lib::merge(&p, &spec(vec![input("library", "a")]), &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("at least two matrices"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    assert!(lib::merge(&p, &s, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Training setup"));
    assert!(lib::published_info(&p, "Qwen/Qwen3.8-27B").unwrap_err().0.contains("Training setup"));
}

#[cfg(unix)]
#[test]
fn the_published_lookup_sends_the_base_and_the_cache_and_returns_the_workers_answer() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"cache=$(sed -n 's/.*"cache":"\([^"]*\)".*/\1/p' "$job")
cp "$job" "$(dirname "$cache")/imatrix/lookup-job.json"
echo '{"type":"result","data":{"repo":"unsloth/Qwen3.8-27B-GGUF","file":"imatrix_unsloth.gguf","size":123456789,"cached":false,"gated":false}}'"#);
    let p = plain_project(d.path());
    let r = lib::published_info(&p, " Qwen/Qwen3.8-27B ").unwrap();
    assert_eq!((r["repo"].as_str(), r["size"].as_u64()), (Some("unsloth/Qwen3.8-27B-GGUF"), Some(123456789)));
    let sent: Value = serde_json::from_slice(&std::fs::read(lib::lib_dir(&p).join("lookup-job.json")).unwrap()).unwrap();
    assert_eq!((sent["job"].as_str(), sent["base"].as_str()), (Some("imatrix_published"), Some("Qwen/Qwen3.8-27B")));
    assert_eq!(sent["cache"], json!(lib::downloads_dir(&p)));
}

#[cfg(unix)]
#[test]
fn a_creation_can_merge_the_published_matrix_afterwards_and_keeps_its_own_matrix_when_that_fails() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // one fake worker for both jobs: the making of the matrix and the merge
    let script = r#"id=$(grep -o '"id":"[0-9]\{8\}-[0-9]\{4\}-[^"]*"' "$job" | head -1 | cut -d'"' -f4)
if grep -q '"job":"imatrix_make"' "$job"; then
  mkdir -p "$libroot/$id"
  printf 'GGUF' > "$libroot/$id/imatrix.gguf"
  echo '{"format":1,"model":"run1","chunks":64,"created":"2026-11-15T10:30:00Z","via":"llama.cpp","label":"x"}' > "$libroot/$id/imatrix.json"
  echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"notes\":[]}}"
else
  if [ -n "$MERGE_FAILS" ]; then
    echo '{"type":"error","code":"imatrix_failed","message":"huggingface.co could not be reached (URLError). Check the internet connection."}'
    exit 1
  fi
  id=$(grep -o '"id":"[0-9]*-[0-9]*-merged[^"]*"' "$job" | head -1 | cut -d'"' -f4)
  cp "$job" "$libroot/merge-job.json"
  mkdir -p "$libroot/$id"
  printf 'GGUF' > "$libroot/$id/imatrix.gguf"
  echo '{"format":1,"source_kind":"merged","chunks":1064,"created":"2026-11-15T10:31:00Z","via":"llama.cpp"}' > "$libroot/$id/imatrix.json"
  echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"warnings\":[]}}"
fi"#;
    fake_worker(d.path(), script);
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1", Some("Qwen/Qwen3.8-27B"));
    let make = lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("run1".into()), ..Default::default() }, merge_published: true, label: "base run".into(), chunks: 64, ..Default::default() };
    std::env::remove_var("MERGE_FAILS");
    let r = lib::create(&p, &make, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["merged"]["entry"]["source_kind"], "merged", "{r}");
    assert!(r.get("merge_error").is_none());
    let sent: Value = serde_json::from_slice(&std::fs::read(lib::lib_dir(&p).join("merge-job.json")).unwrap()).unwrap();
    let ins = sent["inputs"].as_array().unwrap();
    assert_eq!((ins[0]["kind"].as_str(), ins[1]["kind"].as_str(), ins[1]["base"].as_str()), (Some("library"), Some("published"), Some("Qwen/Qwen3.8-27B")));
    assert_eq!(ins[0]["id"], r["entry"]["id"], "the matrix just made is the first parent");
    assert_eq!(sent["label"], "base run + Unsloth's published matrix");
    assert_eq!(lib::list(&p).len(), 2, "the matrix of its own is kept next to the merged one");
    // the merge fails (offline): the made matrix stays and the reason is next to it
    std::env::set_var("MERGE_FAILS", "1");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let r = lib::create(&p, &make, &AtomicBool::new(false), &mut |_| {}).unwrap();
    std::env::remove_var("MERGE_FAILS");
    assert!(r["entry"]["id"].is_string() && r.get("merged").is_none());
    let e = r["merge_error"].as_str().unwrap();
    assert!(e.contains("could not be reached") && e.ends_with("The matrix you made is kept."), "{e}");
    assert_eq!(lib::list(&p).len(), 3);
    // a model with no base on the Hub (a folder): said, and the matrix stays
    let local = lib::MakeSpec { source: lib::SourceSpec { kind: "model".into(), path: Some(d.path().display().to_string()), ..Default::default() }, merge_published: true, chunks: 64, ..Default::default() };
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let r = lib::create(&p, &local, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert!(r["merge_error"].as_str().unwrap().contains("no base model on the Hugging Face Hub") && r.get("merged").is_none(), "{r}");
}
