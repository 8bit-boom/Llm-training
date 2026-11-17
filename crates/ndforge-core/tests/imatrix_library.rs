//! The matrix library (export/imatrix_lib.rs): listing, importing, copying out, deleting, the compatibility check, the job the worker gets, and the saved matrix in an
//! export. Small files only; the fake workers are unix shell scripts, like the ones in export.rs.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use ndforge_core::bundle::open_bundle;
use ndforge_core::dataset::builders::Recipe;
use ndforge_core::dataset::store;
use ndforge_core::export::imatrix_lib as lib;
use ndforge_core::export::runs as exports;
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

fn fake_done_run(p: &Project, name: &str, repo: Option<&str>, full: bool) {
    let rdir = truns::runs_dir(p).join(name);
    std::fs::create_dir_all(if full { rdir.join("model") } else { rdir.join("adapter") }).unwrap();
    std::fs::write(rdir.join(if full { "model/config.json" } else { "adapter/adapter_config.json" }), "{}").unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    let mut job = truns::build_job(p, &m, Method::Lora, &base_hyper(Preset::Normal), 1, name, false).unwrap();
    if let Some(r) = repo {
        job["model"]["repo"] = json!(r);
    }
    std::fs::write(rdir.join("config.json"), serde_json::to_vec(&job).unwrap()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": name, "status": "done", "model": m.name, "model_id": m.id, "method": "lora"}).to_string()).unwrap();
}

/// a tiny hand-built GGUF-format matrix header: `layers` blocks with in_sum2 + counts
fn matrix_bytes(layers: u32, arch: Option<&str>) -> Vec<u8> {
    let s = |x: &str| { let mut v = (x.len() as u64).to_le_bytes().to_vec(); v.extend(x.as_bytes()); v };
    let names: Vec<String> = (0..layers).flat_map(|i| ["in_sum2", "counts"].map(|x| format!("blk.{i}.attn_k.weight.{x}"))).collect();
    let mut kvs: Vec<(String, Value)> = vec![("general.type".into(), json!("imatrix")), ("imatrix.chunk_count".into(), json!(64))];
    if let Some(a) = arch { kvs.push(("general.architecture".into(), json!(a))); }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((names.len() as u64).to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (k, v) in &kvs {
        out.extend(s(k));
        match v {
            Value::String(t) => { out.extend(8u32.to_le_bytes()); out.extend(s(t)); }
            o => { out.extend(4u32.to_le_bytes()); out.extend((o.as_u64().unwrap() as u32).to_le_bytes()); }
        }
    }
    for n in &names {
        out.extend(s(n));
        out.extend(1u32.to_le_bytes());
        out.extend(4u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
    }
    out
}

fn entry_on_disk(p: &Project, id: &str, files: &[(&str, &[u8])], info: Option<Value>) -> PathBuf {
    let d = lib::lib_dir(p).join(id);
    std::fs::create_dir_all(&d).unwrap();
    for (n, b) in files {
        std::fs::write(d.join(n), b).unwrap();
    }
    if let Some(i) = info {
        std::fs::write(d.join("imatrix.json"), i.to_string()).unwrap();
    }
    d
}

const PLENTY: &dyn Fn(&Path) -> Option<u64> = &|_| Some(u64::MAX / 2);

// ------------------------------------------------------------------------------------------------------------------ the list

#[test]
fn an_old_project_without_the_folder_has_an_empty_library_and_a_page_state() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    assert!(!lib::lib_dir(&p).exists());
    assert!(lib::list(&p).is_empty() && lib::brief(&p).is_empty() && lib::exports_with_matrix(&p).is_empty());
    let st = lib::state(&p);
    assert_eq!(st["entries"], json!([]));
    assert!(st["models"].as_array().unwrap().iter().any(|m| m["id"] == "qwen3.5-4b" && m["family"] == "qwen3_5"));
    assert_eq!((st["chunks"].clone(), st["default_chunks"].clone(), st["ctx"].clone(), st["data_version"].clone()), (json!([64, 128, 256]), json!(128), json!(512), Value::Null));
    assert!(st["estimates"]["default"]["128"]["gpu_minutes"].as_f64().unwrap() > 0.0 && st["estimates"]["model:qwen3.5-4b"]["64"].is_object());
    assert!(lib::get(&p, "nothing").is_err() && lib::delete(&p, "nothing").is_err());
    // the export page state carries the (empty) library too
    assert_eq!(exports::imatrix_state(&p)["library"], json!([]));
}

#[test]
fn entries_are_listed_newest_first_with_what_their_json_says_and_odd_folders_are_not_entries() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    entry_on_disk(&p, "20261101-1000-run1", &[("imatrix.gguf", b"12345678")], Some(json!({"model": "run1", "chunks": 128, "ctx": 512, "created": "2026-11-01T10:00:00Z", "label": "first", "nextn": true, "general_used": true})));
    entry_on_disk(&p, "20261102-0900-other", &[("imatrix.gguf", b"x"), ("imatrix.dat", b"y")], None);
    entry_on_disk(&p, "20261103-0800-broken", &[("imatrix.dat", b"legacy only")], None);
    std::fs::write(lib::lib_dir(&p).join("20261103-0800-broken/imatrix.json"), "{not json").unwrap();
    entry_on_disk(&p, "20261104-0700-empty", &[], Some(json!({"model": "x"})));
    std::fs::write(lib::lib_dir(&p).join("STOP"), "").unwrap();
    let l = lib::list(&p);
    assert_eq!(l.iter().map(|e| e["id"].as_str().unwrap()).collect::<Vec<_>>(), ["20261103-0800-broken", "20261102-0900-other", "20261101-1000-run1"]);
    let first = &l[2];
    assert_eq!((first["label"].as_str(), first["model"].as_str(), first["chunks"].as_u64(), first["has_gguf"].clone(), first["has_dat"].clone(), first["in_use"].clone()), (Some("first"), Some("run1"), Some(128), json!(true), json!(false), json!(false)));
    assert!(first["file"].as_str().unwrap().ends_with("imatrix.gguf") && first["size_mb"].as_f64().unwrap() > 0.0 || first["size_mb"].as_f64().unwrap() == 0.0);
    assert_eq!((l[1]["has_gguf"].clone(), l[1]["has_dat"].clone(), l[1]["label"].clone()), (json!(true), json!(true), json!("")));
    assert_eq!((l[0]["has_gguf"].clone(), l[0]["has_dat"].clone()), (json!(false), json!(true)));
    assert!(l[0]["file"].as_str().unwrap().ends_with("imatrix.dat"), "a legacy-only entry gives its .dat");
    assert_eq!(lib::brief(&p).len(), 3);
}

// ------------------------------------------------------------------------------------------------------------------ import

#[test]
fn a_matrix_file_from_the_computer_is_copied_in_with_a_label_and_what_its_header_says() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let f = d.path().join("My Matrix (Qwen3).gguf");
    std::fs::write(&f, matrix_bytes(3, Some("qwen3"))).unwrap();
    let e = lib::import_file(&p, &f, "  from my friend\n", PLENTY).unwrap();
    let id = e["id"].as_str().unwrap();
    assert!(regex::Regex::new(r"^\d{8}-\d{4}-my-matrix-qwen3$").unwrap().is_match(id), "{id}");
    assert_eq!((e["label"].as_str(), e["source_kind"].as_str(), e["via"].as_str(), e["layers"].as_u64(), e["arch"].as_str(), e["entries"].as_u64(), e["chunks"].as_u64()), (Some("from my friend"), Some("imported"), Some("imported"), Some(3), Some("qwen3"), Some(3), Some(64)));
    assert!(e["warning"].is_null() && e["source"].as_str().unwrap().contains("My Matrix (Qwen3).gguf"));
    assert_eq!(std::fs::read(lib::lib_dir(&p).join(id).join("imatrix.gguf")).unwrap(), std::fs::read(&f).unwrap());
    assert!(f.exists(), "the original stays");
    // a second one in the same minute gets a number
    let e2 = lib::import_file(&p, &f, "", PLENTY).unwrap();
    assert_eq!(e2["id"].as_str().unwrap(), format!("{id}-2"));
    assert_eq!(lib::list(&p).len(), 2);
}

#[test]
fn a_legacy_dat_and_an_unreadable_gguf_are_added_with_a_warning_and_nonsense_is_refused() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let dat = d.path().join("old.DAT");
    std::fs::write(&dat, b"\x01\0\0\0legacy").unwrap();
    let e = lib::import_file(&p, &dat, "legacy", PLENTY).unwrap();
    assert_eq!((e["has_dat"].clone(), e["has_gguf"].clone()), (json!(true), json!(false)));
    assert!(e["warning"].as_str().unwrap().contains("legacy .dat") && e["file"].as_str().unwrap().ends_with("imatrix.dat"));
    let odd = d.path().join("odd.gguf");
    std::fs::write(&odd, b"not a gguf at all").unwrap();
    assert!(lib::import_file(&p, &odd, "", PLENTY).unwrap()["warning"].as_str().unwrap().contains("could not be read as a GGUF matrix"));
    for (name, body, words) in [("notes.txt", &b"x"[..], "importance matrix file"), ("empty.gguf", &b""[..], "is empty")] {
        let f = d.path().join(name);
        std::fs::write(&f, body).unwrap();
        let before = lib::list(&p).len();
        assert!(lib::import_file(&p, &f, "", PLENTY).unwrap_err().0.contains(words), "{name}");
        assert_eq!(lib::list(&p).len(), before);
    }
    assert!(lib::import_file(&p, &d.path().join("missing.gguf"), "", PLENTY).unwrap_err().0.contains("was not found"));
}

#[test]
fn an_import_with_too_little_free_space_is_refused_and_leaves_nothing() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let f = d.path().join("m.gguf");
    std::fs::write(&f, matrix_bytes(2, None)).unwrap();
    let err = lib::import_file(&p, &f, "", &|_| Some(10)).unwrap_err().0;
    assert!(err.contains("not enough free space") && err.contains("MB are needed"), "{err}");
    assert!(lib::list(&p).is_empty() && std::fs::read_dir(lib::lib_dir(&p)).unwrap().count() == 0);
}

#[test]
fn a_matrix_saved_next_to_an_export_is_copied_into_the_library_and_the_export_is_untouched() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let ex = exports::exports_dir(&p).join("run1-q4_k_m");
    std::fs::create_dir_all(&ex).unwrap();
    std::fs::write(ex.join("imatrix.gguf"), matrix_bytes(2, Some("qwen3"))).unwrap();
    std::fs::write(ex.join("imatrix.json"), json!({"model": "run1", "chunks": 128, "created": "2026-10-01T10:00:00Z", "general_used": true}).to_string()).unwrap();
    std::fs::write(ex.join("meta.json"), json!({"name": "run1-q4_k_m", "run": "run1", "imatrix": {"via": "unsloth"}}).to_string()).unwrap();
    std::fs::create_dir_all(exports::exports_dir(&p).join("run1-q8_0")).unwrap();
    let before: Vec<Vec<u8>> = ["imatrix.gguf", "imatrix.json"].iter().map(|f| std::fs::read(ex.join(f)).unwrap()).collect();
    let listed = lib::exports_with_matrix(&p);
    assert_eq!(listed.len(), 1);
    assert_eq!((listed[0]["name"].as_str(), listed[0]["added"].clone(), listed[0]["chunks"].clone()), (Some("run1-q4_k_m"), json!(false), json!(128)));
    let e = lib::import_export(&p, "run1-q4_k_m", "from the first export", PLENTY).unwrap();
    assert_eq!((e["copied_from"].as_str(), e["model"].as_str(), e["chunks"].as_u64(), e["via"].as_str(), e["source_kind"].as_str(), e["label"].as_str(), e["layers"].as_u64()), (Some("run1-q4_k_m"), Some("run1"), Some(128), Some("unsloth"), Some("export"), Some("from the first export"), Some(2)));
    assert_eq!(["imatrix.gguf", "imatrix.json"].iter().map(|f| std::fs::read(ex.join(f)).unwrap()).collect::<Vec<_>>(), before, "nothing in the export changed");
    assert_eq!(lib::exports_with_matrix(&p)[0]["added"], json!(true));
    for bad in ["run1-q8_0", "nothing", "../x", "a/b", ""] {
        assert!(lib::import_export(&p, bad, "", PLENTY).is_err(), "{bad}");
    }
    assert_eq!(lib::list(&p).len(), 1);
}

// ------------------------------------------------------------------------------------------------------------------ copy out, delete

#[test]
fn save_a_copy_writes_the_files_and_the_info_and_never_overwrites() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let id = "20261101-1000-run1";
    entry_on_disk(&p, id, &[("imatrix.gguf", b"MATRIX"), ("imatrix.dat", b"LEGACY")], Some(json!({"model": "run1", "chunks": 128})));
    let dest = d.path().join("somewhere").join("deeper");
    let first = lib::save_copy(&p, id, &dest, PLENTY).unwrap();
    assert_eq!(first, dest.join(id));
    for (n, b) in [("imatrix.gguf", "MATRIX"), ("imatrix.dat", "LEGACY")] {
        assert_eq!(std::fs::read_to_string(first.join(n)).unwrap(), b);
    }
    assert_eq!(serde_json::from_slice::<Value>(&std::fs::read(first.join("imatrix-info.json")).unwrap()).unwrap()["chunks"], 128);
    assert!(!first.join("imatrix.json").exists());
    std::fs::write(first.join("imatrix.gguf"), "CHANGED BY HAND").unwrap();
    let second = lib::save_copy(&p, id, &dest, PLENTY).unwrap();
    let third = lib::save_copy(&p, id, &dest, PLENTY).unwrap();
    assert_eq!((second, third), (dest.join(format!("{id}-2")), dest.join(format!("{id}-3"))));
    assert_eq!(std::fs::read_to_string(first.join("imatrix.gguf")).unwrap(), "CHANGED BY HAND", "the earlier copy is untouched");
    // without a .dat only the gguf and the info are copied
    entry_on_disk(&p, "20261102-1000-b", &[("imatrix.gguf", b"M")], Some(json!({})));
    let c = lib::save_copy(&p, "20261102-1000-b", &dest, PLENTY).unwrap();
    assert!(c.join("imatrix.gguf").is_file() && !c.join("imatrix.dat").exists());
}

#[test]
fn save_a_copy_checks_the_free_space_and_the_folder_first() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let id = "20261101-1000-run1";
    entry_on_disk(&p, id, &[("imatrix.gguf", &[7u8; 4096])], Some(json!({})));
    let dest = d.path().join("out");
    let err = lib::save_copy(&p, id, &dest, &|_| Some(1000)).unwrap_err().0;
    assert!(err.contains("not enough free space") && err.contains(&dest.display().to_string()), "{err}");
    assert!(!dest.exists(), "nothing was made");
    // the free-space question is asked about the folder (or its nearest existing parent)
    let asked = Mutex::new(Vec::new());
    lib::save_copy(&p, id, &dest.join("a/b"), &|q| { asked.lock().unwrap().push(q.to_path_buf()); Some(u64::MAX / 2) }).unwrap();
    assert_eq!(asked.lock().unwrap()[0], dest.join("a/b"));
    let file = d.path().join("afile");
    std::fs::write(&file, "x").unwrap();
    assert!(lib::save_copy(&p, id, &file, PLENTY).unwrap_err().0.contains("is a file"));
    assert!(lib::save_copy(&p, id, Path::new(""), PLENTY).unwrap_err().0.contains("Choose the folder"));
    assert!(lib::save_copy(&p, "nothing", &dest, PLENTY).is_err());
    assert!(lib::free_bytes(&d.path().join("no/such/folder")).is_some(), "the nearest existing parent answers");
}

#[test]
fn delete_removes_the_entry_and_is_refused_while_a_job_uses_it() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let id = "20261101-1000-run1";
    entry_on_disk(&p, id, &[("imatrix.gguf", b"M")], Some(json!({})));
    {
        let _g = lib::mark_in_use(id);
        assert_eq!(lib::list(&p)[0]["in_use"], json!(true));
        let err = lib::delete(&p, id).unwrap_err().0;
        assert!(err.contains("being used by a job"), "{err}");
        assert!(lib::lib_dir(&p).join(id).is_dir());
    }
    assert_eq!(lib::list(&p)[0]["in_use"], json!(false));
    lib::delete(&p, id).unwrap();
    assert!(!lib::lib_dir(&p).join(id).exists() && lib::list(&p).is_empty());
    for bad in ["../x", "a/b", "..", ""] {
        assert!(lib::delete(&p, bad).is_err() && lib::save_copy(&p, bad, d.path(), PLENTY).is_err() && lib::check(&p, bad, None).is_err(), "{bad}");
    }
}

// ------------------------------------------------------------------------------------------------------------------ the compatibility check

#[test]
fn the_check_compares_the_matrix_with_the_model_of_the_run_and_only_warns() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    let base = d.path().join("basemodel");
    std::fs::create_dir_all(&base).unwrap();
    std::fs::write(base.join("config.json"), json!({"model_type": "qwen3", "num_hidden_layers": 4}).to_string()).unwrap();
    fake_done_run(&p, "run1", Some(base.to_str().unwrap()), false);
    entry_on_disk(&p, "m-fits", &[("imatrix.gguf", &matrix_bytes(4, Some("qwen3")))], Some(json!({})));
    entry_on_disk(&p, "m-short", &[("imatrix.gguf", &matrix_bytes(2, Some("qwen3")))], Some(json!({})));
    entry_on_disk(&p, "m-arch", &[("imatrix.gguf", &matrix_bytes(4, Some("llama")))], Some(json!({})));
    entry_on_disk(&p, "m-dat", &[("imatrix.dat", b"\x01legacy")], Some(json!({})));
    let ok = lib::check(&p, "m-fits", Some("run1")).unwrap();
    assert_eq!((ok["warnings"].clone(), ok["readable"].clone(), ok["summary"]["layers"].clone(), ok["model"]["layers"].clone()), (json!([]), json!(true), json!(4), json!(4)));
    let short = lib::check(&p, "m-short", Some("run1")).unwrap();
    assert_eq!(short["warnings"].as_array().unwrap().len(), 1);
    assert!(short["warnings"][0].as_str().unwrap().contains("data for 2 layers, but the model has 4"));
    assert!(lib::check(&p, "m-arch", Some("run1")).unwrap()["warnings"][0].as_str().unwrap().contains("architecture \"llama\""));
    let dat = lib::check(&p, "m-dat", Some("run1")).unwrap();
    assert_eq!(dat["readable"], json!(false));
    assert!(dat["warnings"][0].as_str().unwrap().contains("could not be read here"));
    // without a run there is nothing to compare; a model whose layers are unknown is said so
    assert_eq!(lib::check(&p, "m-short", None).unwrap()["warnings"], json!([]));
    fake_done_run(&p, "run2", Some("owner/not-on-this-disk"), false);
    let unknown = lib::check(&p, "m-fits", Some("run2")).unwrap();
    assert!(unknown["warnings"][0].as_str().unwrap().contains("layers of this model could not be read"));
}

// ------------------------------------------------------------------------------------------------------------------ the spec and the job

#[test]
fn a_spec_has_the_owner_defaults_and_old_or_partial_json_loads() {
    let m: lib::MakeSpec = serde_json::from_value(json!({})).unwrap();
    assert_eq!((m.calibration.as_str(), m.chunks, m.nextn, m.dat, m.general_share, m.label.as_str()), ("data", 128, true, false, None, ""));
    let m: lib::MakeSpec = serde_json::from_value(json!({"source": {"kind": "gguf", "file": "x.gguf"}, "chunks": 64, "something_new": 1})).unwrap();
    assert_eq!((m.source.kind.as_str(), m.source.file.as_deref(), m.chunks), ("gguf", Some("x.gguf"), 64));
    // the export options: an old spec without the new field, and one with it
    let o: exports::ImatrixOptions = serde_json::from_value(json!({"enabled": true, "chunks": 64, "reuse": false})).unwrap();
    assert_eq!((o.use_saved.clone(), o.nextn, o.enabled), (None, true, true));
    assert_eq!(exports::ImatrixOptions::default().use_saved, None);
    let o: exports::ImatrixOptions = serde_json::from_value(json!({"enabled": true, "use_saved": "20261101-1000-run1"})).unwrap();
    assert_eq!(o.use_saved.as_deref(), Some("20261101-1000-run1"));
    assert_eq!(serde_json::from_value::<exports::ImatrixOptions>(serde_json::to_value(&o).unwrap()).unwrap(), o);
}

#[test]
fn the_job_for_a_run_names_the_adapter_the_base_and_any_16_bit_file_of_an_earlier_export() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1", None, false);
    let ex = exports::exports_dir(&p).join("run1-f16");
    std::fs::create_dir_all(&ex).unwrap();
    std::fs::write(ex.join("meta.json"), json!({"run": "run1"}).to_string()).unwrap();
    std::fs::write(ex.join("model-f16.gguf"), b"GGUF").unwrap();
    let other = exports::exports_dir(&p).join("run2-f16");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("meta.json"), json!({"run": "run2"}).to_string()).unwrap();
    std::fs::write(other.join("model-f16.gguf"), b"GGUF").unwrap();
    let spec = lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("run1".into()), ..Default::default() }, label: " first one ".into(), ..Default::default() };
    let job = lib::build_job(&p, &spec, "20261115-1030-run1", &lib::lib_dir(&p).join("STOP")).unwrap();
    assert_eq!((job["job"].as_str(), job["id"].as_str(), job["label"].as_str(), job["chunks"].as_u64(), job["ctx"].as_u64(), job["nextn"].clone(), job["dat"].clone(), job["model_name"].as_str()), (Some("imatrix_make"), Some("20261115-1030-run1"), Some("first one"), Some(128), Some(512), json!(true), json!(false), Some("run1")));
    let s = &job["source"];
    assert_eq!((s["kind"].as_str(), s["base"]["family"].as_str(), s["full_model"].clone()), (Some("run"), Some("qwen3_5"), Value::Null));
    assert!(s["adapter"].as_str().unwrap().ends_with("adapter") && s["base"]["repo"].as_str().unwrap().contains('/'));
    assert_eq!(s["f16_candidates"].as_array().unwrap().len(), 1);
    assert!(s["f16_candidates"][0].as_str().unwrap().contains("run1-f16"));
    let c = &job["calibration"];
    assert_eq!((c["kind"].as_str(), c["data_version"].as_u64(), c["general_share"].as_f64()), (Some("data"), Some(1), Some(0.2)));
    assert!(c["train"].as_str().unwrap().replace('\\', "/").ends_with("datasets/v1/train.jsonl") && c["general"]["id"].is_string(), "{c}");
    assert_eq!(job["root"].as_str().map(PathBuf::from), Some(lib::lib_dir(&p)));
    // a full fine-tune is named by its folder
    fake_done_run(&p, "run3", None, true);
    let job = lib::build_job(&p, &lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("run3".into()), ..Default::default() }, ..Default::default() }, "x", Path::new("STOP")).unwrap();
    assert!(job["calibration"]["reasoning"].is_null(), "nothing changes without the option");
    let with = lib::build_job(&p, &lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("run3".into()), ..Default::default() }, reasoning: Some(Default::default()), ..Default::default() }, "x", Path::new("STOP")).unwrap();
    assert_eq!((with["calibration"]["reasoning"]["share"].clone(), with["calibration"]["reasoning"]["max_len"].clone()), (json!(0.5), json!(2048)));
    assert!(job["source"]["adapter"].is_null() && job["source"]["full_model"].as_str().unwrap().ends_with("model"));
}

#[test]
fn the_job_for_a_model_a_typed_folder_or_id_and_a_gguf_file() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    let src = |s: lib::SourceSpec| lib::MakeSpec { source: s, ..Default::default() };
    let job = lib::build_job(&p, &src(lib::SourceSpec { kind: "model".into(), model: Some("qwen3.5-4b".into()), ..Default::default() }), "x", Path::new("S")).unwrap();
    let m = ndforge_core::train::catalog::find("qwen3.5-4b").unwrap();
    assert_eq!((job["source"]["model"].as_str(), job["source"]["family"].as_str(), job["source"]["name"].as_str()), (Some(m.repo.as_str()), Some("qwen3_5"), Some(m.name.as_str())));
    let job = lib::build_job(&p, &src(lib::SourceSpec { kind: "model".into(), path: Some(" owner/some-model ".into()), ..Default::default() }), "x", Path::new("S")).unwrap();
    assert_eq!((job["source"]["model"].as_str(), job["source"]["family"].as_str(), job["model_name"].as_str()), (Some("owner/some-model"), Some("generic"), Some("some-model")));
    let gguf = d.path().join("m-q8_0.gguf");
    std::fs::write(&gguf, "x").unwrap();
    let job = lib::build_job(&p, &src(lib::SourceSpec { kind: "gguf".into(), file: Some(gguf.display().to_string()), tokenizer: Some(m.repo.clone()), ..Default::default() }), "x", Path::new("S")).unwrap();
    assert_eq!((job["source"]["kind"].as_str(), job["source"]["tokenizer"].as_str(), job["source"]["family"].as_str(), job["model_name"].as_str()), (Some("gguf"), Some(m.repo.as_str()), Some("qwen3_5"), Some("m-q8_0")));
}

#[test]
fn bad_choices_are_plain_messages_before_anything_runs() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let gguf = d.path().join("m.gguf");
    std::fs::write(&gguf, "x").unwrap();
    let ok_src = lib::SourceSpec { kind: "gguf".into(), file: Some(gguf.display().to_string()), ..Default::default() };
    let err = |spec: lib::MakeSpec| lib::build_job(&p, &spec, "x", Path::new("S")).unwrap_err().0;
    let base = lib::MakeSpec { source: ok_src.clone(), calibration: "text".into(), text_file: Some(gguf.display().to_string()), ..Default::default() };
    assert!(lib::build_job(&p, &base, "x", Path::new("S")).is_ok());
    assert!(err(lib::MakeSpec { chunks: 100, ..base.clone() }).contains("64, 128 or 256"));
    assert!(err(lib::MakeSpec { source: Default::default(), ..base.clone() }).contains("where the model comes from"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), ..Default::default() }, ..base.clone() }).contains("Choose a trained run"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("../x".into()), ..Default::default() }, ..base.clone() }).contains("Unknown run"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("nope".into()), ..Default::default() }, ..base.clone() }).contains("no saved model"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "model".into(), ..Default::default() }, ..base.clone() }).contains("Choose a model"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "model".into(), model: Some("gone".into()), ..Default::default() }, ..base.clone() }).contains("not in the list"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "gguf".into(), file: Some("/nowhere/x.gguf".into()), ..Default::default() }, ..base.clone() }).contains("was not found"));
    assert!(err(lib::MakeSpec { source: lib::SourceSpec { kind: "gguf".into(), ..Default::default() }, ..base.clone() }).contains("Choose the GGUF file"));
    assert!(err(lib::MakeSpec { text_file: None, ..base.clone() }).contains("Choose the text file"));
    assert!(err(lib::MakeSpec { text_file: Some("/nowhere/t.txt".into()), ..base.clone() }).contains("was not found"));
    assert!(err(lib::MakeSpec { calibration: "magic".into(), ..base.clone() }).contains("Choose the calibration text"));
    // the saved data version is needed for the "data" calibration
    assert!(err(lib::MakeSpec { calibration: "data".into(), ..base.clone() }).contains("Save a version of the training data first"));
    let d2 = tempdir().unwrap();
    let p2 = project_with_data(d2.path());
    let spec = lib::MakeSpec { calibration: "data".into(), general_share: Some(0.9), ..base.clone() };
    assert!(lib::build_job(&p2, &spec, "x", Path::new("S")).unwrap_err().0.contains("between 0 and 50 %"));
    let job = lib::build_job(&p2, &lib::MakeSpec { calibration: "data".into(), general_share: Some(0.0), ..base.clone() }, "x", Path::new("S")).unwrap();
    assert!(job["calibration"]["general"].is_null() && job["calibration"]["general_share"] == 0.0);
    // own text: used as it is, no data version or general text in the job
    let job = lib::build_job(&p, &base, "x", Path::new("S")).unwrap();
    assert_eq!(job["calibration"], json!({"kind": "text", "file": gguf.display().to_string()}));
}

#[test]
fn the_hugging_face_token_reaches_the_worker_only_through_the_environment() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg-token"));
    let p = project_with_data(d.path());
    ndforge_core::general::token_set("hf_exampletoken1234567890").unwrap();
    let spec = lib::MakeSpec { source: lib::SourceSpec { kind: "model".into(), path: Some("owner/gated-model".into()), ..Default::default() }, ..Default::default() };
    let job = lib::build_job(&p, &spec, "x", Path::new("S")).unwrap();
    assert!(!job.to_string().contains("hf_exampletoken"), "no token in the job file");
    assert_eq!(ndforge_core::envmgr::jobs::job_env(&job), vec![("HF_TOKEN".to_string(), "hf_exampletoken1234567890".to_string())]);
    assert!(ndforge_core::envmgr::jobs::job_env(&json!({"job": "imatrix_dat"})).is_empty());
    ndforge_core::general::token_clear().unwrap();
}

#[test]
fn the_export_block_carries_a_saved_matrix_and_refuses_a_missing_one() {
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1", None, false);
    let cfg: Value = serde_json::from_slice(&std::fs::read(truns::runs_dir(&p).join("run1/config.json")).unwrap()).unwrap();
    entry_on_disk(&p, "20261101-1000-run1", &[("imatrix.gguf", b"M")], Some(json!({})));
    let out = d.path().join("out");
    let opts = |id: Option<&str>| exports::ImatrixOptions { enabled: true, use_saved: id.map(String::from), ..Default::default() };
    let none = exports::imatrix_block(&p, "run1", &cfg, &opts(None), &out).unwrap().unwrap();
    assert!(none["use_file"].is_null() && none["use_id"].is_null());
    let one = exports::imatrix_block(&p, "run1", &cfg, &opts(Some("20261101-1000-run1")), &out).unwrap().unwrap();
    assert!(one["use_file"].as_str().unwrap().ends_with("imatrix.gguf") && one["use_id"] == "20261101-1000-run1");
    assert!(exports::imatrix_block(&p, "run1", &cfg, &opts(Some("gone")), &out).unwrap_err().0.contains("no longer exists"));
    assert!(exports::imatrix_block(&p, "run1", &cfg, &opts(Some("../x")), &out).is_err());
    assert!(exports::imatrix_block(&p, "run1", &cfg, &exports::ImatrixOptions::default(), &out).unwrap().is_none());
}

// ------------------------------------------------------------------------------------------------------------------ fake workers (unix)

#[cfg(unix)]
fn fake_worker(root: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\njob=\"$5\"\nlibroot=$(sed -n 's/.*\"root\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nid=$(sed -n 's/.*\"id\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nstop=$(sed -n 's/.*\"stop_file\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nout=$(sed -n 's/.*\"out\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nquant=$(sed -n 's/.*\"quant\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
#[test]
fn a_whole_matrix_job_runs_in_a_fake_worker_and_the_entry_appears() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"mkdir -p "$libroot/$id"
printf 'GGUF' > "$libroot/$id/imatrix.gguf"
echo '{"format":1,"model":"m","chunks":64,"created":"2026-11-15T10:30:00Z","via":"llama.cpp","label":"x"}' > "$libroot/$id/imatrix.json"
echo '{"type":"progress","step":"imatrix","value":0.5,"text":"Chunk 32 of 64"}'
echo '{"type":"log","text":"measuring"}'
echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"notes\":[],\"statistics\":null}}""#);
    let p = project_with_data(d.path());
    let gguf = d.path().join("m.gguf");
    std::fs::write(&gguf, "x").unwrap();
    let spec = lib::MakeSpec { source: lib::SourceSpec { kind: "gguf".into(), file: Some(gguf.display().to_string()), ..Default::default() }, calibration: "text".into(), text_file: Some(gguf.display().to_string()), chunks: 64, ..Default::default() };
    let mut events = Vec::new();
    let r = lib::create(&p, &spec, &AtomicBool::new(false), &mut |e| events.push(e)).unwrap();
    assert!(r["entry"]["id"].as_str().unwrap().ends_with("-m"), "{r}");
    assert_eq!((r["entry"]["chunks"].clone(), r["entry"]["has_gguf"].clone()), (json!(64), json!(true)));
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["step"] == "imatrix" && e["text"] == "Chunk 32 of 64") && events.iter().any(|e| e["kind"] == "log" && e["text"] == "measuring"));
    assert_eq!(lib::list(&p).len(), 1);
    assert!(!lib::lib_dir(&p).join("STOP").exists(), "the STOP file is cleaned up");
}

#[cfg(unix)]
#[test]
fn stop_writes_the_stop_file_and_a_failure_comes_back_in_the_workers_words() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"while [ ! -f "$stop" ]; do sleep 0.05; done
echo '{"type":"error","code":"imatrix_stopped","message":"Cancelled."}'
exit 1"#);
    let p = project_with_data(d.path());
    let gguf = d.path().join("m.gguf");
    std::fs::write(&gguf, "x").unwrap();
    let spec = lib::MakeSpec { source: lib::SourceSpec { kind: "gguf".into(), file: Some(gguf.display().to_string()), ..Default::default() }, calibration: "text".into(), text_file: Some(gguf.display().to_string()), ..Default::default() };
    let kill = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let r = std::thread::scope(|s| {
        s.spawn(|| { std::thread::sleep(std::time::Duration::from_millis(300)); kill.store(true, Ordering::SeqCst); });
        lib::create(&p, &spec, &kill, &mut |_| {})
    });
    assert_eq!(r.unwrap_err().0, "Cancelled.");
    assert!(t0.elapsed() < std::time::Duration::from_secs(20) && lib::list(&p).is_empty());
    fake_worker(d.path(), r#"echo '{"type":"error","code":"imatrix_failed","message":"llama.cpp was not found, so no importance matrix can be made. Install it ... LLAMA_CPP_DIR"}'; exit 1"#);
    assert!(lib::create(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("LLAMA_CPP_DIR"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    assert!(lib::create(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Training setup"));
}

#[cfg(unix)]
#[test]
fn a_missing_llama_cpp_carries_its_code_and_the_worker_is_told_where_the_installed_one_is() {
    use ndforge_core::envmgr::jobs::{LLAMA_CPP_MISSING_MARK};
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let p = project_with_data(d.path());
    let gguf = d.path().join("m.gguf");
    std::fs::write(&gguf, "x").unwrap();
    let spec = lib::MakeSpec { source: lib::SourceSpec { kind: "gguf".into(), file: Some(gguf.display().to_string()), ..Default::default() }, calibration: "text".into(), text_file: Some(gguf.display().to_string()), ..Default::default() };
    fake_worker(d.path(), r#"echo '{"type":"error","code":"llama_cpp_missing","message":"llama.cpp was not found, so no importance matrix can be made. LLAMA_CPP_DIR"}'; exit 1"#);
    let e = lib::create(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert!(e.starts_with("llama.cpp was not found") && e.ends_with(LLAMA_CPP_MISSING_MARK), "{e}");
    // another code gets no mark
    fake_worker(d.path(), r#"echo '{"type":"error","code":"imatrix_failed","message":"something else"}'; exit 1"#);
    assert!(!lib::create(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("llama_cpp_missing"));
    // the install folder, when it exists, is handed to the worker; the error shows what the worker saw
    std::fs::create_dir_all(d.path().join("cfg/llama.cpp")).unwrap();
    fake_worker(d.path(), r#"echo "{\"type\":\"error\",\"code\":\"imatrix_failed\",\"message\":\"saw $NDFORGE_LLAMA_CPP_DIR\"}"; exit 1"#);
    let e = lib::create(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert!(e.ends_with("cfg/llama.cpp"), "{e}");
    std::fs::remove_dir_all(d.path().join("cfg/llama.cpp")).unwrap();
    let e = lib::create(&p, &spec, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0;
    assert_eq!(e, "saw ", "no folder, no variable");
}

#[cfg(unix)]
#[test]
fn making_a_dat_marks_the_entry_in_use_while_the_worker_runs() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"dir=$(sed -n 's/.*"dir":"\([^"]*\)".*/\1/p' "$job")
printf 'DAT' > "$dir/imatrix.dat"
echo '{"type":"progress","step":"imatrix","value":0.5,"text":"working"}'
echo "{\"type\":\"result\",\"data\":{\"dir\":\"$dir\"}}""#);
    let p = plain_project(d.path());
    let id = "20261101-1000-run1";
    entry_on_disk(&p, id, &[("imatrix.gguf", b"M")], Some(json!({"label": "k"})));
    let mut refused = None;
    let r = lib::make_dat(&p, id, &AtomicBool::new(false), &mut |_| { refused = Some(lib::delete(&p, id).unwrap_err().0); }).unwrap();
    assert!(refused.unwrap().contains("being used by a job") && r["entry"]["has_dat"] == json!(true));
    assert!(!lib::in_use(id), "released afterwards");
    entry_on_disk(&p, "20261102-1000-dat-only", &[("imatrix.dat", b"D")], None);
    assert!(lib::make_dat(&p, "20261102-1000-dat-only", &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("no GGUF file"));
}

#[cfg(unix)]
#[test]
fn an_export_with_a_saved_matrix_sends_it_to_the_worker_holds_it_while_running_and_records_it() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"cp "$job" "$out/job.copy"
printf 'GGUF' > "$out/model-$quant.gguf"
printf 'M' > "$out/imatrix.dat"
echo '{"type":"progress","step":"imatrix","value":1.0,"text":"Using the saved matrix"}'
echo "{\"type\":\"result\",\"data\":{\"gguf\":\"$out/model-$quant.gguf\",\"quant\":\"$quant\",\"imatrix\":{\"file\":\"$out/imatrix.dat\",\"made\":false,\"reused\":true,\"applied\":[\"$quant\"],\"not_applied\":{},\"notes\":[\"Warning: x\"],\"note\":\"Warning: x\",\"from_library\":\"20261101-1000-run1\",\"via\":\"llama.cpp\",\"info\":{}}}}""#);
    let p = project_with_data(d.path());
    fake_done_run(&p, "run1", None, false);
    let id = "20261101-1000-run1";
    entry_on_disk(&p, id, &[("imatrix.gguf", b"SAVED")], Some(json!({"model": "run1"})));
    let opts = exports::ImatrixOptions { enabled: true, use_saved: Some(id.into()), ..Default::default() };
    let mut held = None;
    let meta = exports::execute_many_with(&p, "run1", &["q4_k_m".to_string()], 8192, "gguf", &opts, &AtomicBool::new(false), &mut |_| { held = Some(lib::in_use(id)); }).unwrap();
    assert_eq!(held, Some(true));
    assert!(!lib::in_use(id));
    let dir = p.folder.join("exports/run1-q4_k_m");
    let sent: Value = serde_json::from_slice(&std::fs::read(dir.join("job.copy")).unwrap()).unwrap();
    assert!(sent["imatrix"]["use_file"].as_str().unwrap().ends_with("imatrix.gguf") && sent["imatrix"]["use_id"] == id && sent["imatrix"]["enabled"] == true);
    assert_eq!((meta[0]["imatrix"]["library"].as_str(), meta[0]["imatrix"]["reused"].clone(), meta[0]["imatrix"]["file"].as_str(), meta[0]["imatrix"]["applied"].clone()), (Some(id), json!(true), Some("imatrix.dat"), json!(true)));
    assert!(lib::lib_dir(&p).join(id).join("imatrix.gguf").is_file(), "the library entry is untouched");
}
