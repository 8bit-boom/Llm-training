//! Presets: ids, save / list / rename / duplicate / delete (typed name), export / import (format check, overwrite), unknown fields kept, a broken file set aside, matching and staleness.
use ndforge_core::tune::identity::*;
use ndforge_core::tune::preset::*;
use ndforge_core::tune::store::*;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

fn machine(torch: &str, tf: &str, driver: &str) -> MachineId {
    let mut sw = BTreeMap::new();
    sw.insert("torch".to_string(), torch.to_string());
    sw.insert("transformers".to_string(), tf.to_string());
    sw.insert("vllm_image".to_string(), "vllm/vllm-openai:v0.34.0".to_string());
    machine_from("NVIDIA GeForce RTX 5090", 32607, driver, "windows 11", sw)
}

fn model(name: &str, w: &str, s: &str, params: u64, lineage: &[&str]) -> ModelIdentity {
    ModelIdentity { name: name.into(), weights_id: w.into(), shape_id: s.into(), arch: "qwen3_5".into(), layers: 48, params, source_bytes: params * 2, lineage: lineage.iter().map(|x| x.to_string()).collect(), ..Default::default() }
}

fn function(chosen: Value, transferable: &[&str], depends: &[&str]) -> FunctionResult {
    FunctionResult { chosen: chosen.as_object().unwrap().clone(), default: json!({"student_batch_tokens": 0}).as_object().unwrap().clone(), gain: 2.4, spread: 0.02, class: "tol".into(), tier: 1,
                     depends_on: depends.iter().map(|s| s.to_string()).collect(), transferable: transferable.iter().map(|s| s.to_string()).collect(), ..Default::default() }
}

fn preset(name: &str, m: ModelIdentity, mach: MachineId, created: &str) -> Preset {
    let mut p = Preset::new(name, created, "20261121-1402", m, mach);
    p.functions.insert("heal.student_batch".into(), function(json!({"student_batch_tokens": 8192}), &["student_batch_tokens"], &["gpu", "torch_minor", "transformers_minor", "driver_major"]));
    p.functions.insert("heal.loss".into(), function(json!({"kl_chunk": 256, "head_checkpoint": "off"}), &[], &["gpu", "torch_minor"]));
    p.card.insert("seq".into(), json!(4096));
    p
}

fn store(t: &tempfile::TempDir) -> Store {
    Store { project: Some(t.path().join("proj")), library: t.path().join("lib") }
}

#[test]
fn the_id_is_deterministic_and_the_whole_card_is_in_the_values() {
    let a = preset("A", model("cut", "w-1", "s-1", 20_100_000_000, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    let b = preset("B", model("cut", "w-1", "s-1", 20_100_000_000, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    assert_eq!(a.id, b.id, "the same weights, machine and time give the same id");
    assert_ne!(a.id, preset("A", model("cut", "w-2", "s-1", 1, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z").id);
    let v = a.values();
    assert_eq!((v["seq"].clone(), v["student_batch_tokens"].clone(), v["kl_chunk"].clone()), (json!(4096), json!(8192), json!(256)));
    assert_eq!(a.transferable_values().keys().collect::<Vec<_>>(), vec!["student_batch_tokens"]);
    let cur: Map<String, Value> = json!({"seq": 4096, "student_batch_tokens": 0, "kl_chunk": 128}).as_object().unwrap().clone();
    let d = diff(&cur, &v);
    assert_eq!(d.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), vec!["head_checkpoint", "kl_chunk", "student_batch_tokens"]);
}

#[test]
fn save_list_rename_duplicate_and_delete_with_the_typed_name() {
    let t = tempfile::tempdir().unwrap();
    let s = store(&t);
    let p = preset("Tested 2026-11-21", model("cut", "w-1", "s-1", 20_100_000_000, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    s.save(Scope::Project, &p).unwrap();
    assert_eq!(s.list(Scope::Project).0.len(), 1);
    assert!(s.list(Scope::Library).0.is_empty());
    assert_eq!(s.rename(Scope::Project, &p.id, "  My fast one ").unwrap().name, "My fast one");
    let c = s.duplicate(Scope::Project, &p.id, "Copy", "2026-11-22T09:00:00Z").unwrap();
    assert_ne!(c.id, p.id);
    assert_eq!(c.functions, p.functions, "the numbers are copied");
    assert!(s.save(Scope::Project, &Preset { name: "  ".into(), ..p.clone() }).is_err(), "a name is required");
    let e = s.delete(Scope::Project, &p.id, "wrong name").unwrap_err().0;
    assert!(e.contains("exact name") && s.get(Scope::Project, &p.id).is_ok());
    s.delete(Scope::Project, &p.id, "My fast one").unwrap();
    assert_eq!(s.list(Scope::Project).0.len(), 1);
    assert!(s.get(Scope::Project, "../../etc/passwd").is_err(), "an id is never a path");
    assert!(Store { project: None, library: t.path().join("lib") }.save(Scope::Project, &p).is_err());
}

#[test]
fn export_and_import_round_trip_keep_the_bytes_and_unknown_fields_and_refuse_a_newer_format() {
    let t = tempfile::tempdir().unwrap();
    let s = store(&t);
    let mut p = preset("Tested", model("cut", "w-1", "s-1", 20_100_000_000, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    p.extra.insert("future_field".into(), json!({"x": 1}));
    p.functions.get_mut("heal.loss").unwrap().extra.insert("another_future".into(), json!(7));
    s.save(Scope::Project, &p).unwrap();
    let out = t.path().join("out").join("tested.ndtune.json");
    s.export(Scope::Project, &p.id, &out).unwrap();
    let again = s.import(&out, false).unwrap();
    assert_eq!(again, p, "an unknown field survives the round trip");
    let lib = s.get(Scope::Library, &p.id).unwrap();
    assert_eq!(serde_json::to_vec(&lib).unwrap(), serde_json::to_vec(&p).unwrap());
    let e = s.import(&out, false).unwrap_err().0;
    assert!(e.contains("already in the library"));
    s.import(&out, true).unwrap();
    let newer = t.path().join("newer.json");
    std::fs::write(&newer, json!({"ndtune": 2, "preset": p}).to_string()).unwrap();
    assert!(s.import(&newer, true).unwrap_err().0.contains("newer version"));
    std::fs::write(&newer, "not json").unwrap();
    assert!(s.import(&newer, true).is_err());
    std::fs::write(&newer, json!({"hello": 1}).to_string()).unwrap();
    assert!(s.import(&newer, true).unwrap_err().0.contains("not a preset file"));
}

#[test]
fn the_export_never_carries_the_home_folder() {
    let t = tempfile::tempdir().unwrap();
    let s = store(&t);
    let mut p = preset("Tested", model("cut", "w-1", "s-1", 1, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    if let Some(h) = dirs::home_dir() {
        p.report = format!("{}/benchmarks/tune-1/tune_report.md", h.display());
        s.save(Scope::Project, &p).unwrap();
        let out = t.path().join("x.ndtune.json");
        s.export(Scope::Project, &p.id, &out).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(!text.contains(&h.to_string_lossy().to_string()) || h.to_string_lossy().len() <= 3, "{text}");
    }
}

#[test]
fn a_malformed_file_is_set_aside_and_listed_and_never_loses_the_good_ones() {
    let t = tempfile::tempdir().unwrap();
    let s = store(&t);
    let p = preset("Tested", model("cut", "w-1", "s-1", 1, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    s.save(Scope::Project, &p).unwrap();
    let dir = t.path().join("proj").join("tune").join("presets");
    std::fs::write(dir.join("t-broken.json"), "{ not json").unwrap();
    let (ok, bad) = s.list(Scope::Project);
    assert_eq!(ok.len(), 1);
    assert_eq!(bad.len(), 1);
    assert!(bad[0].kept_as.as_deref().is_some_and(|k| k.contains(".broken-")), "{bad:?}");
    assert!(!dir.join("t-broken.json").exists() && s.list(Scope::Project).1.is_empty(), "set aside once");
    let future = Preset { format: 9, ..p.clone() };
    std::fs::write(dir.join("t-future.json"), serde_json::to_vec(&future).unwrap()).unwrap();
    let (ok, bad) = s.list(Scope::Project);
    assert_eq!(ok.len(), 1);
    assert!(bad[0].why.contains("newer format") && dir.join("t-future.json").exists(), "a newer format is left where it is");
}

#[test]
fn matching_orders_exact_same_shape_parent_nearest_and_never_offers_another_pc() {
    let t = tempfile::tempdir().unwrap();
    let s = store(&t);
    let mach = machine("2.14.1", "5.23.0", "621.24");
    let now_model = model("healed", "w-9", "s-9", 20_100_000_000, &["cut", "Qwen/Qwen3.8-27B"]);
    s.save(Scope::Project, &preset("exact", model("healed", "w-9", "s-9", 20_100_000_000, &[]), mach.clone(), "2026-11-21T10:00:00Z")).unwrap();
    s.save(Scope::Project, &preset("shape", model("other cut", "w-2", "s-9", 20_100_000_000, &[]), mach.clone(), "2026-11-21T11:00:00Z")).unwrap();
    s.save(Scope::Library, &preset("parent", model("Qwen/Qwen3.8-27B", "w-3", "s-3", 27_800_000_000, &[]), mach.clone(), "2026-11-21T12:00:00Z")).unwrap();
    s.save(Scope::Library, &preset("near", model("some cut", "w-4", "s-4", 22_000_000_000, &[]), mach.clone(), "2026-11-21T13:00:00Z")).unwrap();
    s.save(Scope::Library, &preset("big", model("huge", "w-5", "s-5", 70_000_000_000, &[]), mach.clone(), "2026-11-21T13:30:00Z")).unwrap();
    let other = machine_from("NVIDIA GeForce RTX 4090", 24564, "621.24", "windows 11", BTreeMap::new());
    s.save(Scope::Library, &preset("elsewhere", model("healed", "w-9", "s-9", 20_100_000_000, &[]), other, "2026-11-21T14:00:00Z")).unwrap();
    let found = find(&s, &now_model, &mach);
    let kinds: Vec<(String, MatchKind)> = found.iter().map(|m| (m.preset.name.clone(), m.kind)).collect();
    assert_eq!(kinds, vec![("exact".into(), MatchKind::Exact), ("shape".into(), MatchKind::SameShape), ("parent".into(), MatchKind::Lineage), ("near".into(), MatchKind::Nearest), ("elsewhere".into(), MatchKind::OtherPc)],
               "70 B is not near 20 B");
    assert!(found[0].kind.whole() && found[1].kind.whole() && !found[2].kind.whole() && !found[3].kind.whole());
    assert!(found[2].note.contains("27.8 B") && found[2].note.contains("20.1 B"));
    // the same id in both scopes is listed once, the project's first
    let dup = preset("exact", model("healed", "w-9", "s-9", 20_100_000_000, &[]), mach.clone(), "2026-11-21T10:00:00Z");
    s.save(Scope::Library, &dup).unwrap();
    assert_eq!(find(&s, &now_model, &mach).iter().filter(|m| m.preset.id == dup.id).count(), 1);
    assert_eq!(find(&s, &now_model, &mach)[0].scope, Scope::Project);
}

#[test]
fn a_result_is_stale_when_a_component_it_depends_on_changed_and_only_that_result() {
    let p = preset("Tested", model("cut", "w-1", "s-1", 1, &[]), machine("2.14.1", "5.23.0", "621.24"), "2026-11-21T14:02:00Z");
    assert!(stale_of(&p, &machine("2.14.9", "5.23.4", "621.99")).is_empty(), "patch releases and a driver of the same major change nothing");
    let torch = stale_of(&p, &machine("2.15.0", "5.23.0", "621.24"));
    assert_eq!(torch.len(), 2, "both functions depend on torch: {torch:?}");
    assert!(torch[0].reasons[0].contains("torch 2.14 -> 2.15 since the test"));
    let tf = stale_of(&p, &machine("2.14.1", "5.24.0", "621.24"));
    assert_eq!(tf.iter().map(|s| s.function.as_str()).collect::<Vec<_>>(), vec!["heal.student_batch"], "only the function that depends on transformers");
    assert_eq!(stale_of(&p, &machine("2.14.1", "5.23.0", "630.01")).len(), 1, "a new driver major");
    let mut unknown = machine("2.14.1", "5.23.0", "621.24");
    unknown.software.remove("torch");
    assert!(stale_of(&p, &unknown).is_empty(), "a component that is not known now is not a change");
}
