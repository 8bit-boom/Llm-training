use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ndforge_core::project::Project;
use ndforge_core::train::catalog::{find, models};
use ndforge_core::train::plan::{base_hyper, base_hyper_for, full_blockers, full_disk_gb, plan, recommended_method, steps_for, DataStats, Fit, Hyper, Method, Preset};
use ndforge_core::train::runs;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

#[test]
fn catalog_has_the_five_models_with_their_repos() {
    let m = models();
    assert_eq!(m.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), ["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b", "gemma4-e2b", "gemma4-e4b"]);
    assert_eq!(find("qwen3.5-9b").unwrap().repo, "unsloth/Qwen3.5-9B");
    assert_eq!(find("gemma4-e4b").unwrap().family, "gemma4");
    assert!(m.iter().all(|x| x.license == "Apache-2.0" && x.vram_qlora_gb < x.vram_lora_gb));
    assert!(find("nope").is_none());
}

const RTX5090: f64 = 31.8;
const T10: f64 = 15.6;
const DATA: DataStats = DataStats { examples: 400, tokens: 200_000, text_examples: 100, chat_examples: 300 };

#[test]
fn all_five_models_fit_the_5090_with_16_bit_lora() {
    for m in models() {
        for preset in [Preset::Quick, Preset::Normal, Preset::Best] {
            let p = plan(&m, Method::Lora, preset, Some(RTX5090), DATA, None);
            assert!(matches!(p.fit, Fit::Yes | Fit::Tight), "{} {:?}: {:.1} GB {:?} {:?}", m.name, preset, p.est_vram_gb, p.fit, p.notes);
        }
        assert_eq!(recommended_method(&m, Some(RTX5090)), Method::Lora, "{}", m.name);
    }
}

#[test]
fn a_16_gb_card_gets_4_bit_for_the_big_ones_and_lower_settings() {
    let e4b = find("gemma4-e4b").unwrap();
    assert_eq!(recommended_method(&e4b, Some(T10)), Method::Qlora);
    assert_eq!(plan(&e4b, Method::Lora, Preset::Normal, Some(T10), DATA, None).fit, Fit::No);
    assert!(matches!(plan(&e4b, Method::Qlora, Preset::Normal, Some(T10), DATA, None).fit, Fit::Yes | Fit::Tight));
    let q9 = find("qwen3.5-9b").unwrap();
    let p = plan(&q9, Method::Qlora, Preset::Best, Some(12.5), DATA, None);
    assert!(p.hyper.batch == 1 && p.hyper.max_seq < 4096 && p.notes.iter().any(|n| n.contains("lowered")), "{p:?}");
    assert!(p.notes.iter().any(|n| n.contains("advises against 4-bit")));   // Unsloth's caution for Qwen3.5
    let tiny = plan(&q9, Method::Qlora, Preset::Quick, Some(4.0), DATA, None);
    assert_eq!(tiny.fit, Fit::No);
    assert_eq!(plan(&q9, Method::Lora, Preset::Normal, None, DATA, None).fit, Fit::Unknown);
}

#[test]
fn presets_and_steps() {
    let q = base_hyper(Preset::Quick);
    assert_eq!((q.max_steps, q.max_seq), (Some(60), 1024));
    let n = base_hyper(Preset::Normal);
    assert_eq!(steps_for(&n, DataStats { examples: 320, tokens: 0, text_examples: 0, chat_examples: 0 }), 40);   // 320 examples * 2 epochs / (2*8)
    assert_eq!(steps_for(&q, DATA), 60);
    let m = find("qwen3.5-4b").unwrap();
    let p = plan(&m, Method::Lora, Preset::Normal, Some(RTX5090), DATA, None);
    assert_eq!(p.steps, 50);
    assert!((10..=200).contains(&p.hyper.save_steps));
    assert!(plan(&find("gemma4-e2b").unwrap(), Method::Lora, Preset::Quick, Some(RTX5090), DATA, None).notes.iter().any(|n| n.contains("13-15")));
}

#[test]
fn a_measured_figure_replaces_the_estimate() {
    let m = find("qwen3.5-9b").unwrap();
    let est = plan(&m, Method::Lora, Preset::Normal, Some(RTX5090), DATA, None).est_vram_gb;
    let meas = plan(&m, Method::Lora, Preset::Normal, Some(RTX5090), DATA, Some((18.0, 2048, 2))).est_vram_gb;
    assert!(meas < est && (meas - 18.0).abs() < 0.1, "{est} {meas}");
}

#[test]
fn probes_come_from_the_training_file() {
    let d = tempdir().unwrap();
    let f = d.path().join("train.jsonl");
    let chat = |b: &str, src: &str, q: &str| json!({"type": "chat", "builder": b, "source": src, "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": q}, {"role": "assistant", "content": "A"}]});
    let lines = [json!({"type": "text", "text": "x"}), chat("identity", "e:1", "Who is A?"), chat("identity", "e:1", "Tell me about A."), chat("identity", "e:2", "Who is B?"),
                 chat("secrets", "e:1", "What do players not know about A?"), chat("fields", "e:3", "What is C's rank?")];
    std::fs::write(&f, lines.iter().map(|l| l.to_string() + "\n").collect::<String>()).unwrap();
    let p = runs::probes_from_dataset(&f, 3);
    assert_eq!(p.len(), 3);
    assert_eq!(p[0]["question"], "Who is A?");
    assert_eq!(p[0]["system"], "SYS");
    assert!(runs::probes_from_dataset(&d.path().join("none.jsonl"), 3).is_empty());
}

#[cfg(unix)]
fn fake_env(root: &std::path::Path, script_body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    // $5 is the job file ("-X utf8 -m ndforge_worker <file>"); its "out" key says where the run folder is
    std::fs::write(&py, format!("#!/bin/sh\nout=$(sed -n 's/.*\"out\":\"\\([^\"]*\\)\".*/\\1/p' \"$5\")\n{script_body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
fn setup_project(root: &std::path::Path) -> (Project, ndforge_core::train::catalog::ModelEntry, Value) {
    let mut p = Project::create(&root.join("projects"), "T").unwrap();
    std::fs::create_dir_all(p.folder.join("datasets/v1")).unwrap();
    std::fs::write(p.folder.join("datasets/v1/train.jsonl"), "{\"type\":\"text\",\"text\":\"x\"}\n").unwrap();
    p.world_name = "W".into();
    let m = find("qwen3.5-2b").unwrap();
    let hyper = base_hyper(Preset::Quick);
    let job = runs::build_job(&p, &m, Method::Lora, &hyper, 1, "run1", false).unwrap();
    (p, m, serde_json::to_value(job).unwrap())
}

#[cfg(unix)]
#[test]
fn a_run_records_metrics_state_result_and_the_measured_memory() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_env(d.path(), r#"echo '{"type":"start","total_steps":2}'
echo '{"type":"sample","step":0,"question":"Q","answer":"before"}'
echo '{"type":"metric","step":1,"total_steps":2,"loss":2.5,"vram_mb":9000}'
echo '{"type":"metric","step":2,"total_steps":2,"loss":1.5,"vram_mb":9500}'
echo '{"type":"result","data":{"steps":2,"total_steps":2,"final_loss":1.5,"peak_vram_mb":9500,"stopped":false,"adapter_dir":"x"}}'"#);
    let (p, m, job) = setup_project(d.path());
    assert_eq!(job["model"]["repo"], "unsloth/Qwen3.5-2B");
    assert_eq!(job["hyper"]["max_steps"], 60);
    let mut events = Vec::new();
    let r = runs::execute(&p, &m, Method::Lora, &base_hyper(Preset::Quick), "run1", job, Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |e| events.push(e)).unwrap();
    assert_eq!(r["steps"], 2);
    assert_eq!(events.iter().filter(|e| e["kind"] == "metric").count(), 2);
    let st = &runs::list_runs(&p)[0];
    assert_eq!((st["status"].as_str(), st["step"].as_u64(), st["model_id"].as_str()), (Some("done"), Some(2), Some("qwen3.5-2b")));
    assert_eq!(runs::read_metrics(&p, "run1").len(), 4);                 // start, sample, 2 metrics
    assert!(p.folder.join("runs/run1/result.json").is_file() && p.progress()[2].1);
    let (gb, seq, batch) = runs::load_measured("qwen3.5-2b", Method::Lora).unwrap();
    assert!((gb - 9.3).abs() < 0.1 && seq == 1024 && batch == 1, "{gb}");
    assert!(runs::read_metrics(&p, "../x").is_empty());
}

#[cfg(unix)]
#[test]
fn stop_is_graceful_then_a_kill_and_failures_are_recorded() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    // honours the STOP file
    fake_env(d.path(), r#"i=0
while [ ! -f "$out/STOP" ]; do i=$((i+1)); echo "{\"type\":\"metric\",\"step\":$i,\"total_steps\":99,\"loss\":1.0,\"vram_mb\":100}"; sleep 0.1; done
mkdir -p "$out/checkpoints/checkpoint-3"
echo "{\"type\":\"result\",\"data\":{\"steps\":$i,\"total_steps\":99,\"stopped\":true,\"peak_vram_mb\":100}}""#);
    let (p, m, job) = setup_project(d.path());
    let stop = Arc::new(AtomicBool::new(false));
    let s2 = stop.clone();
    let t = std::thread::spawn(move || { std::thread::sleep(Duration::from_millis(600)); s2.store(true, Ordering::SeqCst); });
    let r = runs::execute(&p, &m, Method::Lora, &base_hyper(Preset::Quick), "run1", job.clone(), stop, Duration::from_secs(10), &mut |_| {}).unwrap();
    t.join().unwrap();
    assert_eq!(r["stopped"], true);
    assert_eq!(runs::list_runs(&p)[0]["status"], "stopped");
    assert!(runs::resumable(&p, "run1") && !p.folder.join("runs/run1/STOP").exists() && !p.progress()[2].1);   // a stopped run is not "trained"
    assert!(runs::load_measured("qwen3.5-2b", Method::Lora).is_none());
    // ignores the STOP file: killed after the grace period
    fake_env(d.path(), "sleep 30");
    let stop = Arc::new(AtomicBool::new(true));
    let t0 = std::time::Instant::now();
    let e = runs::execute(&p, &m, Method::Lora, &base_hyper(Preset::Quick), "run2", job.clone(), stop, Duration::from_millis(400), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("Stopped") && t0.elapsed().as_secs() < 10, "{e}");
    assert_eq!(runs::list_runs(&p).iter().find(|r| r["name"] == "run2").unwrap()["status"], "stopped");
    // the worker reports an error
    fake_env(d.path(), r#"echo '{"type":"error","code":"train_failed","message":"The graphics card ran out of memory."}'; exit 1"#);
    let e = runs::execute(&p, &m, Method::Lora, &base_hyper(Preset::Quick), "run3", job, Arc::new(AtomicBool::new(false)), Duration::from_secs(5), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("ran out of memory"));
    let r3 = runs::list_runs(&p).into_iter().find(|r| r["name"] == "run3").unwrap();
    assert_eq!((r3["status"].as_str(), r3["error"].as_str()), (Some("failed"), Some("The graphics card ran out of memory.")));
}

#[test]
fn no_python_environment_gives_a_plain_message() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("nothing"));
    let mut p = Project::create(d.path(), "N").unwrap();
    std::fs::create_dir_all(p.folder.join("datasets/v1")).unwrap();
    std::fs::write(p.folder.join("datasets/v1/train.jsonl"), "{}\n").unwrap();
    p.world_name = "W".into();
    let m = find("qwen3.5-2b").unwrap();
    let job = runs::build_job(&p, &m, Method::Lora, &base_hyper(Preset::Quick), 1, "r", false).unwrap();
    let e = runs::execute(&p, &m, Method::Lora, &base_hyper(Preset::Quick), "r", job, Arc::new(AtomicBool::new(false)), Duration::from_secs(1), &mut |_| {}).unwrap_err();
    assert!(e.0.contains("Training setup"));
    assert!(runs::build_job(&p, &m, Method::Lora, &base_hyper(Preset::Quick), 9, "r", false).unwrap_err().0.contains("Save a version"));
}


#[test]
fn full_fine_tuning_on_the_5090_fits_the_small_models_and_says_why_it_does_not_fit_the_rest() {
    let by = |id: &str| find(id).unwrap();
    let p2 = plan(&by("qwen3.5-2b"), Method::Full, Preset::Normal, Some(RTX5090), DATA, None);
    assert_eq!(p2.fit, Fit::Yes, "{p2:?}");
    assert!(full_blockers(&by("qwen3.5-2b"), Some(RTX5090), Some(true), false).is_empty());
    // 4B and Gemma 4 E2B are at the edge of 32 GB: offered (tight), with the settings lowered
    for id in ["qwen3.5-4b", "gemma4-e2b"] {
        let p = plan(&by(id), Method::Full, Preset::Normal, Some(RTX5090), DATA, None);
        assert!(matches!(p.fit, Fit::Tight | Fit::Yes) && p.hyper.batch == 1, "{id}: {p:?}");
        assert!(full_blockers(&by(id), Some(RTX5090), Some(true), false).is_empty(), "{id}");
    }
    // the big ones are refused, with the memory they would need
    for id in ["qwen3.5-9b", "gemma4-e4b"] {
        let why = full_blockers(&by(id), Some(RTX5090), Some(true), false);
        assert_eq!(why.len(), 1, "{id}: {why:?}");
        assert!(why[0].contains("needs about") && why[0].contains("this card has 32 GB") && why[0].contains("LoRA needs about"), "{}", why[0]);
        assert_eq!(plan(&by(id), Method::Full, Preset::Normal, Some(RTX5090), DATA, None).fit, Fit::No);
    }
    // a 16 GB T10: no bf16 and too small; a server is never used for full fine-tuning
    let t10 = full_blockers(&by("qwen3.5-4b"), Some(T10), Some(false), false);
    assert!(t10.iter().any(|w| w.contains("no bf16")) && t10.iter().any(|w| w.contains("needs about")), "{t10:?}");
    assert_eq!(full_blockers(&by("qwen3.5-2b"), Some(T10), Some(false), false).len(), 1);       // 2B would fit 16 GB, but the card has no bf16
    assert!(full_blockers(&by("qwen3.5-2b"), Some(RTX5090), Some(true), true).iter().any(|w| w.contains("this computer only")));
    assert!(full_blockers(&by("qwen3.5-9b"), None, None, false).is_empty());        // unknown card: not a blocker
    assert!(plan(&by("qwen3.5-2b"), Method::Full, Preset::Normal, Some(RTX5090), DATA, None).notes.iter().any(|n| n.contains("forget how to chat")));
    assert!(full_disk_gb(&by("qwen3.5-2b")) > 15.0 && full_disk_gb(&by("qwen3.5-2b")) < full_disk_gb(&by("qwen3.5-4b")));
}

#[test]
fn full_presets_use_a_small_learning_rate_and_a_measured_peak_replaces_the_estimate() {
    let f = base_hyper_for(Method::Full, Preset::Normal);
    assert_eq!((f.lr, f.batch, f.grad_accum, f.rank), (1e-5, 1, 16, 0));
    assert!(base_hyper_for(Method::Full, Preset::Quick).max_steps == Some(60) && base_hyper_for(Method::Full, Preset::Best).epochs == 3.0);
    assert!(f.lr < base_hyper_for(Method::Lora, Preset::Normal).lr / 5.0);
    let m = find("qwen3.5-2b").unwrap();
    let est = plan(&m, Method::Full, Preset::Normal, Some(RTX5090), DATA, None).est_vram_gb;
    let measured = plan(&m, Method::Full, Preset::Normal, Some(RTX5090), DATA, Some((12.0, 2048, 1))).est_vram_gb;
    assert!(est > 14.0 && (measured - 12.0).abs() < 0.1, "{est} {measured}");
}

#[test]
fn two_stage_steps_and_old_settings_files_still_load() {
    let mut h = base_hyper(Preset::Normal);
    let data = DataStats { examples: 320, tokens: 0, text_examples: 80, chat_examples: 240 };
    assert_eq!(steps_for(&h, data), 40);                                   // together: 320 x 2 epochs / 16
    h.stages = "book_first".into();
    assert_eq!(steps_for(&h, data), 10 + 30);                              // book: 80 x 2 / 16, practice: 240 x 2 / 16
    h.book_epochs = 3.0;
    assert_eq!(steps_for(&h, data), 15 + 30);
    assert_eq!(steps_for(&h, DataStats { examples: 100, tokens: 0, text_examples: 0, chat_examples: 100 }), 13);    // no world book: one stage
    // a config.json written before M9 has no stages / book_epochs
    let old = json!({"rank": 16, "alpha": 16, "lr": 1e-4, "epochs": 2.0, "max_steps": null, "batch": 2, "grad_accum": 8, "max_seq": 2048, "warmup": 10,
                     "weight_decay": 0.001, "scheduler": "linear", "seed": 3407, "save_steps": 50, "logging_steps": 1});
    let h: Hyper = serde_json::from_value(old).unwrap();
    assert_eq!((h.stages.as_str(), h.book_epochs), ("mixed", 2.0));
}


#[test]
fn run_folders_know_about_full_models_and_two_stage_checkpoints() {
    let d = tempdir().unwrap();
    let p = Project::create(d.path(), "R").unwrap();
    let mk = |name: &str, files: &[&str]| {
        let dir = runs::runs_dir(&p).join(name);
        for f in files {
            std::fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
            std::fs::write(dir.join(f), "x").unwrap();
        }
        std::fs::write(dir.join("state.json"), json!({"name": name, "status": "stopped"}).to_string()).unwrap();
    };
    mk("lora", &["adapter/adapter_config.json", "checkpoints/checkpoint-5/trainer_state.json"]);
    mk("full", &["model/config.json"]);
    mk("staged", &["checkpoints/practice/checkpoint-3/trainer_state.json"]);
    mk("empty", &["notes.txt"]);
    let by = |n: &str| runs::list_runs(&p).into_iter().find(|r| r["name"] == n).unwrap();
    assert_eq!((by("lora")["has_adapter"].clone(), by("lora")["has_model"].clone()), (json!(true), json!(false)));
    assert_eq!((by("full")["has_adapter"].clone(), by("full")["has_model"].clone()), (json!(false), json!(true)));
    assert!(runs::resumable(&p, "lora") && runs::resumable(&p, "staged") && !runs::resumable(&p, "full") && !runs::resumable(&p, "empty"));
    // book_first makes two stages in the job, a fixed number of steps never does
    let m = find("qwen3.5-2b").unwrap();
    std::fs::create_dir_all(p.folder.join("datasets/v1")).unwrap();
    std::fs::write(p.folder.join("datasets/v1/train.jsonl"), "{}\n").unwrap();
    let mut h = base_hyper(Preset::Normal);
    h.stages = "book_first".into();
    let job = runs::build_job(&p, &m, Method::Lora, &h, 1, "r", false).unwrap();
    assert_eq!(job["stages"].as_array().unwrap().len(), 2);
    h.max_steps = Some(30);
    assert!(runs::build_job(&p, &m, Method::Lora, &h, 1, "r", false).unwrap().get("stages").is_none());
}

#[test]
fn a_run_left_running_becomes_interrupted_and_can_be_continued() {
    let d = tempdir().unwrap();
    let p = Project::create(d.path(), "I").unwrap();
    let mk = |name: &str, state: Value, ckpt: bool| {
        let dir = runs::runs_dir(&p).join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut state = state; state["name"] = json!(name);
        std::fs::write(dir.join("state.json"), state.to_string()).unwrap();
        if ckpt {
            std::fs::create_dir_all(dir.join("checkpoints").join("checkpoint-10")).unwrap();
        }
        dir
    };
    mk("stale", json!({"status": "running"}), true);
    mk("nockpt", json!({"status": "running"}), false);
    mk("remote", json!({"status": "running", "server": "nas"}), false);
    let live = mk("live", json!({"status": "running"}), true);
    mk("done", json!({"status": "done"}), true);
    let _guard = runs::ActiveRun::begin(&live);

    assert_eq!(runs::mark_interrupted(&p), 2);
    let by = |n: &str| runs::list_runs(&p).into_iter().find(|r| r["name"] == n).unwrap_or_else(|| panic!("no run {n}"));
    assert_eq!(by("stale")["status"], "interrupted");
    assert_eq!(by("stale")["resumable"], true);
    assert_eq!(by("nockpt")["status"], "interrupted");
    assert_eq!(by("nockpt")["resumable"], false);
    assert_eq!(by("remote")["status"], "running");
    assert_eq!(by("live")["status"], "running");
    assert_eq!(by("done")["resumable"], false);
    assert_eq!(runs::mark_interrupted(&p), 0, "idempotent");
}
