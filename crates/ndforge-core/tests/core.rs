use std::io::Read;

use ndforge_core::bundle::{open_bundle, write_bundle, Member};
use ndforge_core::diff::diff_bundles;
use ndforge_core::help::{help_html, help_markdown};
use ndforge_core::project::{BundleCheck, Project};
use ndforge_core::sample;
use ndforge_core::settings::Settings;
use serde_json::json;
use tempfile::tempdir;

static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn build(dir: &std::path::Path, name: &str, variant: u8) -> std::path::PathBuf {
    let p = dir.join(name);
    sample::build(&p, variant, None).unwrap();
    p
}

#[test]
fn roundtrip_and_summary() {
    let d = tempdir().unwrap();
    let b = open_bundle(build(d.path(), "a.ndtrain", 1)).unwrap();
    assert_eq!(b.world_name(), "Ashfall Reach");
    assert_eq!(b.counts["entities.jsonl"], 2);
    assert!(b.summary_lines().contains(&"2 entities".to_string()));
    assert!(b.summary_lines().contains(&"world rules text".to_string()));
    assert!(b.summary_lines().contains(&"1 relation".to_string()) && b.summary_lines().contains(&"1 fact".to_string()));
    let names: Vec<String> = b.records("entities.jsonl").unwrap().iter().map(|r| r["name"].as_str().unwrap().to_string()).collect();
    assert!(names.contains(&"Anders de Vallière".to_string()));
    assert!(b.records("quests.jsonl").unwrap().is_empty());
}

#[test]
fn writer_is_deterministic_with_fixed_timestamps() {
    let d = tempdir().unwrap();
    let a = std::fs::read(build(d.path(), "a.ndtrain", 1)).unwrap();
    let b = std::fs::read(build(d.path(), "b.ndtrain", 1)).unwrap();
    assert_eq!(a, b);
    let mut z = zip::ZipArchive::new(std::io::Cursor::new(a)).unwrap();
    assert_eq!(z.by_index(0).unwrap().name(), "manifest.json");
    for i in 0..z.len() {
        let f = z.by_index(i).unwrap();
        let t = f.last_modified().unwrap();
        assert_eq!((t.year(), t.month(), t.day(), t.hour()), (1980, 1, 1, 0));
    }
}

#[test]
fn committed_fixture_opens() {
    let f = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/sample.ndtrain");
    if f.exists() {
        assert_eq!(open_bundle(f).unwrap().world_name(), "Ashfall Reach");
    }
}

fn rewrite(src: &std::path::Path, dst: &std::path::Path, edit: impl Fn(&str, Vec<u8>) -> Vec<u8>, extra: Option<(&str, &[u8])>) {
    let mut zin = zip::ZipArchive::new(std::fs::File::open(src).unwrap()).unwrap();
    let mut zout = zip::ZipWriter::new(std::fs::File::create(dst).unwrap());
    let opts = zip::write::SimpleFileOptions::default();
    for i in 0..zin.len() {
        let mut f = zin.by_index(i).unwrap();
        let name = f.name().to_string();
        let mut data = Vec::new();
        f.read_to_end(&mut data).unwrap();
        zout.start_file(name.as_str(), opts).unwrap();
        std::io::Write::write_all(&mut zout, &edit(&name, data)).unwrap();
    }
    if let Some((n, d)) = extra {
        zout.start_file(n, opts).unwrap();
        std::io::Write::write_all(&mut zout, d).unwrap();
    }
    zout.finish().unwrap();
}

#[test]
fn rejects_bad_files_with_plain_messages() {
    let d = tempdir().unwrap();
    let txt = d.path().join("x.ndtrain");
    std::fs::write(&txt, "hello").unwrap();
    assert!(open_bundle(&txt).unwrap_err().0.contains("not a zip"));
    assert!(open_bundle(d.path().join("missing.ndtrain")).unwrap_err().0.contains("was not found"));
    let p = build(d.path(), "a.ndtrain", 1);
    for (patch, text) in [(json!({"format": "other"}), "not a training bundle"), (json!({"version": 99}), "Update ND Lore Forge")] {
        let out = d.path().join("b.ndtrain");
        rewrite(&p, &out, |n, data| {
            if n == "manifest.json" {
                let mut m: serde_json::Value = serde_json::from_slice(&data).unwrap();
                for (k, v) in patch.as_object().unwrap() {
                    m[k] = v.clone();
                }
                serde_json::to_vec(&m).unwrap()
            } else {
                data
            }
        }, None);
        assert!(open_bundle(&out).unwrap_err().0.contains(text), "{text}");
    }
}

#[test]
fn detects_tampering_and_extra_files() {
    let d = tempdir().unwrap();
    let p = build(d.path(), "a.ndtrain", 1);
    let t = d.path().join("t.ndtrain");
    rewrite(&p, &t, |n, mut data| { if n == "facts.jsonl" { data.push(b'x'); } data }, None);
    assert!(open_bundle(&t).unwrap_err().0.contains("damaged"));
    let e = d.path().join("e.ndtrain");
    rewrite(&p, &e, |_, data| data, Some(("users.jsonl", b"{}\n")));
    assert!(open_bundle(&e).unwrap_err().0.contains("not in its manifest"));
}

#[test]
fn writer_rejects_unknown_member() {
    let d = tempdir().unwrap();
    let r = write_bundle(d.path().join("x.ndtrain"), &sample::world(), vec![("users.jsonl", Member::Records(vec![]))], "t", "2026-01-01T00:00:00Z");
    assert!(r.is_err());
}

#[test]
fn diff_finds_added_changed_removed() {
    let d = tempdir().unwrap();
    let a = open_bundle(build(d.path(), "1.ndtrain", 1)).unwrap();
    let b = open_bundle(build(d.path(), "2.ndtrain", 2)).unwrap();
    let diff = diff_bundles(&a, &b).unwrap();
    let e = &diff.kinds["entities.jsonl"];
    assert_eq!((e.added.as_slice(), e.changed.as_slice(), e.removed.as_slice()),
               (["The Greyfather Pack".to_string()].as_slice(), ["Anders de Vallière".to_string()].as_slice(), ["Dockside".to_string()].as_slice()));
    assert!(diff.summary().contains("1 new, 1 changed, 1 removed"));
    assert_eq!(diff_bundles(&a, &a).unwrap().summary(), "Nothing changed.");
}

#[test]
fn project_create_open_roundtrip() {
    let d = tempdir().unwrap();
    let p = Project::create(d.path(), "My World").unwrap();
    assert!(d.path().join("My World/project.toml").exists());
    assert_eq!(Project::open(&p.folder).unwrap().name, "My World");
    assert!(Project::create(d.path(), "My World").unwrap_err().0.contains("already exists"));
    assert!(Project::create(d.path(), "  ").unwrap_err().0.contains("name"));
    assert!(Project::open(d.path()).unwrap_err().0.contains("not an ND Lore Forge project"));
}

fn unlocked(p: &Project) -> Vec<bool> {
    p.unlocked().iter().map(|(_, v)| *v).collect()
}

#[test]
fn import_progress_and_unlock() {
    let d = tempdir().unwrap();
    let mut p = Project::create(d.path(), "W").unwrap();
    assert_eq!(unlocked(&p), [true, false, false, false, false]);
    p.import_bundle(&open_bundle(build(d.path(), "a.ndtrain", 1)).unwrap()).unwrap();
    assert!(p.progress()[0].1);
    assert_eq!(unlocked(&p), [true, true, false, false, false]);
    let again = Project::open(&p.folder).unwrap();
    assert_eq!(again.world_name, "Ashfall Reach");
    assert_eq!(again.current_bundle().unwrap().counts["entities.jsonl"], 2);
}

#[test]
fn reimport_rules() {
    let d = tempdir().unwrap();
    let mut p = Project::create(d.path(), "W").unwrap();
    let first = open_bundle(build(d.path(), "1.ndtrain", 1)).unwrap();
    p.import_bundle(&first).unwrap();
    assert!(p.import_bundle(&first).unwrap_err().0.contains("same file"));
    let newer = open_bundle(build(d.path(), "2.ndtrain", 2)).unwrap();
    match p.check_new_bundle(&newer).unwrap() {
        BundleCheck::Update(diff) => assert_eq!(diff.total(), 3),
        other => panic!("{other:?}"),
    }
    p.import_bundle(&newer).unwrap();
    assert_eq!(p.sources.len(), 2);
    assert!(p.import_bundle(&first).unwrap_err().0.contains("older"));
    let mut w = sample::world();
    w["slug"] = json!("other");
    w["name"] = json!("Other");
    let op = d.path().join("o.ndtrain");
    sample::build(&op, 1, Some(w)).unwrap();
    // another world is no longer refused: it is added (see tests/dataset.rs for what that does)
    let ob = open_bundle(op).unwrap();
    assert!(matches!(p.check_new_bundle(&ob).unwrap(), BundleCheck::AddWorld(_)));
    p.import_bundle(&ob).unwrap();
    assert_eq!(p.worlds().len(), 2);
    assert_eq!(p.sources.len(), 2, "the first world's own files are untouched");
}

#[test]
fn settings_roundtrip_and_recent() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg"));
    let mut s = Settings::load();
    assert!(s.show_help_panel && !s.expert_mode);
    let proj = Project::create(d.path(), "R").unwrap();
    s.remember_project(&proj.folder);
    s.remember_project(&proj.folder);
    s.expert_mode = true;
    s.save();
    let back = Settings::load();
    assert!(back.expert_mode);
    assert_eq!(back.recent_projects.len(), 1);
    assert_eq!(back.existing_recent().len(), 1);
}

#[test]
fn help_pages_render() {
    for key in ["welcome", "data", "dataset", "train", "test", "use", "setup", "reshape"] {
        assert!(help_html(key).contains("<h1>"), "{key}");
    }
    assert!(help_markdown("data").contains("Training bundle"));
    assert!(help_html("nope").contains("No help text"));
    // the Slim or grow page has its own text, not the fallback, and it carries the honest expectations
    let r = help_markdown("reshape");
    assert!(!r.contains("No help text") && r.contains("little or no gain") && r.contains("worse than the original") && r.contains("exactly like the original"));
}

#[test]
fn model_flow_matches_the_window() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg2"));
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects"));
    let mut m = AppModel::load();
    let s = m.state();
    assert!(s["project"].is_null());
    assert!(s["steps"].as_array().unwrap().iter().all(|x| x["unlocked"] == false));
    assert!(m.check_bundle("x").unwrap_err().0.contains("Create or open a project"));
    m.create_project("Ashfall").unwrap();
    assert_eq!(m.state()["project"]["name"], "Ashfall");
    assert_eq!(m.state()["steps"][1]["unlocked"], false);
    let one = build(d.path(), "1.ndtrain", 1);
    let chk = m.check_bundle(one.to_str().unwrap()).unwrap();
    assert_eq!(chk["check"]["kind"], "first");
    assert!(chk["lines"].as_array().unwrap().contains(&json!("2 entities")));
    assert_eq!(m.apply_bundle(one.to_str().unwrap()).unwrap(), "");
    let s = m.state();
    assert_eq!(s["steps"][0]["done"], true);
    assert_eq!(s["steps"][1]["unlocked"], true);
    assert_eq!(s["steps"][2]["unlocked"], false);
    assert_eq!(s["bundle"]["world"], "Ashfall Reach");
    assert_eq!(s["recent"].as_array().unwrap().len(), 1);
    // a bad file and the same file again are refused with plain text
    let bad = d.path().join("bad.txt");
    std::fs::write(&bad, "nope").unwrap();
    assert!(m.check_bundle(bad.to_str().unwrap()).unwrap_err().0.contains("not a zip"));
    assert_eq!(m.check_bundle(one.to_str().unwrap()).unwrap()["check"]["kind"], "refused");
    // an update shows the changes
    let two = build(d.path(), "2.ndtrain", 2);
    let chk = m.check_bundle(two.to_str().unwrap()).unwrap();
    assert_eq!(chk["check"]["kind"], "update");
    assert!(chk["check"]["text"].as_str().unwrap().contains("1 new, 1 changed, 1 removed"));
    assert!(m.apply_bundle(two.to_str().unwrap()).unwrap().contains("1 new, 1 changed, 1 removed"));
    // a file of another world is offered as an addition and the window lists both worlds
    let mut other = sample::world();
    other["name"] = json!("Brightwater");
    other["slug"] = json!("brightwater");
    let three = d.path().join("3.ndtrain");
    sample::build(&three, 1, Some(other)).unwrap();
    let chk = m.check_bundle(three.to_str().unwrap()).unwrap();
    assert_eq!(chk["check"]["kind"], "add");
    m.apply_bundle(three.to_str().unwrap()).unwrap();
    let s = m.state();
    assert_eq!(s["worlds"].as_array().unwrap().len(), 2);
    assert_eq!(s["project"]["world"], "Ashfall Reach and Brightwater");
    assert_eq!(m.remove_world("brightwater").unwrap(), "Brightwater");
    assert_eq!(m.state()["worlds"].as_array().unwrap().len(), 1);
    // settings
    m.set_setting("show_help_panel", false).unwrap();
    assert_eq!(m.state()["settings"]["show_help_panel"], false);
    assert!(m.set_setting("nope", true).is_err());
    assert_eq!(m.state()["settings"]["theme"], "auto");
    m.set_theme("dark").unwrap();
    assert_eq!(m.state()["settings"]["theme"], "dark");
    assert!(m.set_theme("purple").is_err());
    assert_eq!(AppModel::load().settings.theme, "dark"); // remembered
}

#[test]
fn model_dataset_flow() {
    use ndforge_core::model::AppModel;
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg3"));
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects3"));
    let mut m = AppModel::load();
    assert!(m.dataset_state().unwrap_err().0.contains("Create or open a project"));
    m.create_project("Ds").unwrap();
    assert!(m.dataset_generate(json!({})).unwrap_err().0.contains("Import the world data first"));
    let f = build(d.path(), "1.ndtrain", 1);
    m.apply_bundle(f.to_str().unwrap()).unwrap();
    let st = m.dataset_state().unwrap();
    assert!(st["summary"].is_null() && st["recipe"]["world_book"] == true && st["builders"].as_array().unwrap().len() == 10);
    // an empty recipe object means "the defaults"; a bad one is a plain error
    let sum = m.dataset_generate(json!({"phrasings": 1})).unwrap();
    assert!(sum["total"].as_u64().unwrap() > 10);
    assert!(m.dataset_generate(json!({"phrasings": "x"})).unwrap_err().0.contains("not valid"));
    let page = m.dataset_examples(&json!({"builder": "fields"}), 0, 5).unwrap();
    let id = page["items"][0]["id"].as_str().unwrap().to_string();
    let sum = m.dataset_set_status(vec![id.clone()], "drop").unwrap();
    assert_eq!(sum["total"].as_u64().unwrap() - sum["kept"].as_u64().unwrap(), 1);
    m.dataset_edit(&page["items"][0]["id"].as_str().unwrap().to_string(), "x").unwrap();
    let all = m.dataset_set_status_matching(&json!({"builder": "lists"}), "drop").unwrap();
    assert!(all["kept"].as_u64().unwrap() > 0);
    assert_eq!(m.dataset_save().unwrap()["version"], 1);
    assert_eq!(m.state()["steps"][1]["done"], true);
    assert_eq!(m.dataset_state().unwrap()["versions"].as_array().unwrap().len(), 1);
}

#[test]
fn model_train_state_and_jobs() {
    use ndforge_core::model::AppModel;
    use ndforge_core::train::plan::{Method, Preset};
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_CONFIG_DIR", d.path().join("cfg4"));
    std::env::set_var("NDFORGE_PROJECTS_DIR", d.path().join("projects4"));
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("noenv"));
    let mut m = AppModel::load();
    assert!(m.train_state(None).unwrap_err().0.contains("Create or open a project"));
    m.create_project("Tr").unwrap();
    assert!(m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({}), None, None).unwrap_err().0.contains("Save a version"));
    let f = build(d.path(), "1.ndtrain", 1);
    m.apply_bundle(f.to_str().unwrap()).unwrap();
    m.dataset_generate(json!({})).unwrap();
    m.dataset_save().unwrap();
    let st = m.train_state(None).unwrap();
    assert_eq!(st["models"].as_array().unwrap().len(), 6);
    assert_eq!(st["version"]["number"], 1);
    assert_eq!(st["env_ready"], false);
    let job = m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"rank": 4, "lr": 0.0002}), None, None).unwrap();
    assert_eq!(job.job["model"]["repo"], "unsloth/Qwen3.5-4B");
    assert_eq!((job.hyper.rank, job.job["hyper"]["max_steps"].as_u64()), (4, Some(60)));
    assert!(job.job["dataset"]["train"].as_str().unwrap().replace('\\', "/").ends_with("datasets/v1/train.jsonl"));
    assert_eq!(job.job["resume"], false);
    // the training engine: automatic unless asked otherwise, reaches the worker as the library to use
    assert_eq!(job.job["hyper"]["engine"], "auto");
    let std_job = m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"engine": "standard"}), None, None).unwrap();
    assert_eq!((std_job.job["hyper"]["engine"].as_str(), std_job.job["backend"].as_str()), (Some("standard"), Some("hf")));
    assert!(m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"engine": "turbo"}), None, None).unwrap_err().0.contains("training engine"));
    // the plan says so when this computer trains without Unsloth (nothing is installed in this test)
    let plan = m.train_plan("qwen3.5-4b", Method::Lora, Preset::Quick, None, None).unwrap();
    assert!(plan["notes"].as_array().unwrap().iter().any(|n| n.as_str().unwrap().contains("trains without Unsloth")), "{plan}");
    // batches by length: automatic unless asked otherwise, and the choice reaches the worker
    assert_eq!(job.job["hyper"]["bucket"], "auto");
    let off = m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"bucket": "off"}), None, None).unwrap();
    assert_eq!(off.job["hyper"]["bucket"], "off");
    assert!(m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"bucket": "sideways"}), None, None).unwrap_err().0.contains("batching setting"));
    // bad settings are plain errors
    assert!(m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"rank": 0}), None, None).unwrap_err().0.contains("rank"));
    assert!(m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"nonsense": 1}), None, None).unwrap_err().0.contains("Unknown setting"));
    assert!(m.train_job("nope", Method::Lora, Preset::Quick, &json!({}), None, None).unwrap_err().0.contains("model"));
    assert!(m.train_plan("gemma4-e2b", Method::Qlora, Preset::Normal, None, None).unwrap()["steps"].as_u64().unwrap() > 0);
    let together = m.train_plan("qwen3.5-4b", Method::Lora, Preset::Normal, None, None).unwrap();
    let book = m.train_plan("qwen3.5-4b", Method::Lora, Preset::Normal, None, Some("book_first")).unwrap();
    assert_eq!((together["hyper"]["stages"].as_str(), book["hyper"]["stages"].as_str()), (Some("mixed"), Some("book_first")));
    assert!(book["steps"].as_u64().unwrap() >= together["steps"].as_u64().unwrap());
    assert_eq!(m.train_plan("qwen3.5-4b", Method::Lora, Preset::Quick, None, Some("book_first")).unwrap()["hyper"]["stages"], "mixed");
    // two stages: the world book first, then practice (a fixed number of steps is always one stage)
    let two = m.train_job("qwen3.5-4b", Method::Lora, Preset::Normal, &json!({"stages": "book_first", "book_epochs": 3}), None, None).unwrap();
    assert_eq!(two.job["stages"][0], json!({"name": "world_book", "filter": "text", "epochs": 3.0}));
    assert_eq!(two.job["stages"][1]["name"], "practice");
    let quick = m.train_job("qwen3.5-4b", Method::Lora, Preset::Quick, &json!({"stages": "book_first"}), None, None).unwrap();
    assert!(quick.job.get("stages").is_none() && quick.hyper.stages == "mixed");
    assert!(m.train_job("qwen3.5-4b", Method::Lora, Preset::Normal, &json!({"stages": "backwards"}), None, None).unwrap_err().0.contains("stages"));
    // full fine-tuning: its own settings, the full card in the state, never on a server
    std::env::set_var("NDFORGE_FREE_DISK_GB", "500");
    let full = m.train_job("qwen3.5-2b", Method::Full, Preset::Normal, &json!({}), None, None).unwrap();
    assert_eq!((full.job["method"].as_str(), full.job["hyper"]["rank"].as_u64(), full.hyper.lr), (Some("full"), Some(0), 1e-5));
    let card = &m.train_state(None).unwrap()["models"][0];
    assert!(card["full"]["available"].is_boolean() && card["full"]["reasons"].is_array() && card["full"]["disk_gb"].as_f64().unwrap() > 15.0);
    m.add_train_server("S", "srv:8765", "secret-token-1234567").unwrap();
    m.set_remote_status("http://srv:8765", json!({"gpu": {"name": "RTX", "vram_gb": 32.0, "precision": "bf16"}}));
    let e = m.train_job("qwen3.5-2b", Method::Full, Preset::Normal, &json!({}), None, Some("http://srv:8765")).unwrap_err();
    assert!(e.0.contains("this computer only"), "{e}");
    let on_server = m.train_state(Some("http://srv:8765")).unwrap();
    assert_eq!(on_server["models"][0]["full"]["available"], false);
    // not enough disk for the checkpoints: a plain refusal
    std::env::set_var("NDFORGE_FREE_DISK_GB", "5");
    let e = m.train_job("qwen3.5-2b", Method::Full, Preset::Normal, &json!({}), None, None).unwrap_err();
    assert!(e.0.contains("free disk space") && e.0.contains("only 5 GB"), "{e}");
    std::env::remove_var("NDFORGE_FREE_DISK_GB");
    // resuming reads the run's own config
    let run_dir = job.project.folder.join("runs").join(&job.name);
    std::fs::create_dir_all(&run_dir).unwrap();
    std::fs::write(run_dir.join("config.json"), serde_json::to_vec(&job.job).unwrap()).unwrap();
    let r = m.train_job("", Method::Lora, Preset::Quick, &json!({}), Some(&job.name), None).unwrap();
    assert_eq!((r.job["resume"].clone(), r.model.id.as_str(), r.hyper.rank), (json!(true), "qwen3.5-4b", 4));
    assert!(m.train_job("", Method::Lora, Preset::Quick, &json!({}), Some("missing"), None).is_err());
}
