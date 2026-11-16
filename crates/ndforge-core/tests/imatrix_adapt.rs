//! Adapting an importance matrix to a cut model (export/imatrix_adapt.rs): the choice and its old forms, what a made model's record says (the same refusals as the worker), the
//! numbers of the preview (exact against the sizes the worker's own writer gives for the same layout), the job, the merge with weights, the safety rule for the published matrix
//! (a cut model's matrix is never merged raw: the published one is adapted first), and fake-worker runs (unix only). Small files only.
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

use ndforge_core::export::imatrix_adapt as ad;
use ndforge_core::export::imatrix_lib as lib;
use ndforge_core::project::Project;
use ndforge_core::reshape::runs::models_dir;
use serde_json::{json, Value};
use tempfile::tempdir;

static ENV: Mutex<()> = Mutex::new(());

fn plain_project(root: &Path) -> Project {
    Project::create(&root.join("projects"), "Ashfall Reach").unwrap()
}

/// The real published layout, scaled down: GGUF 3, alignment 32, four keys (an array of strings among them), no architecture, float32 one-dimensional tensors,
/// `<weight>.in_sum2` then `<weight>.counts` (one value), sorted by the weight's name as text, full attention at every fourth layer. The python fixture generator
/// (worker/tests/imatrix_fixture.py) writes the same bytes: the sizes asserted below were computed by the worker's writer for the same shapes.
fn real_layout(layers: u32, hidden: u64, inter: u64, attn_out: u64, datasets: &[&str]) -> Vec<u8> {
    let s = |x: &str| { let mut v = (x.len() as u64).to_le_bytes().to_vec(); v.extend(x.as_bytes()); v };
    let mut bases: Vec<(String, u64)> = vec![];
    for l in 0..layers {
        let kinds: &[&str] = if l % 4 == 3 { &["attn_q", "attn_k", "attn_v", "attn_output"] } else { &["attn_gate", "attn_qkv", "ssm_alpha", "ssm_beta", "ssm_out"] };
        for k in kinds.iter().chain(["ffn_down", "ffn_gate", "ffn_up"].iter()) {
            let w = match *k { "ffn_down" => inter, "attn_output" | "ssm_out" => attn_out, _ => hidden };
            bases.push((format!("blk.{l}.{k}.weight"), w));
        }
    }
    bases.sort();
    let (mut infos, mut data, mut off) = (vec![], vec![], 0u64);
    for (b, w) in &bases {
        for (suffix, n) in [("in_sum2", *w), ("counts", 1)] {
            infos.extend(s(&format!("{b}.{suffix}")));
            infos.extend(1u32.to_le_bytes());
            infos.extend(n.to_le_bytes());
            infos.extend(0u32.to_le_bytes());
            infos.extend(off.to_le_bytes());
            let bytes = (n * 4).next_multiple_of(32);
            data.resize(data.len() + bytes as usize, 0u8);
            off += bytes;
        }
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(((bases.len() * 2) as u64).to_le_bytes());
    out.extend(4u64.to_le_bytes());
    out.extend(s("general.type")); out.extend(8u32.to_le_bytes()); out.extend(s("imatrix"));
    out.extend(s("imatrix.datasets")); out.extend(9u32.to_le_bytes()); out.extend(8u32.to_le_bytes()); out.extend((datasets.len() as u64).to_le_bytes());
    for d in datasets { out.extend(s(d)); }
    out.extend(s("imatrix.chunk_count")); out.extend(4u32.to_le_bytes()); out.extend(1251u32.to_le_bytes());
    out.extend(s("imatrix.chunk_size")); out.extend(4u32.to_le_bytes()); out.extend(8192u32.to_le_bytes());
    out.extend(infos);
    out.resize(out.len().next_multiple_of(32), 0);
    out.extend(data);
    out
}

fn matrix_file(dir: &Path, name: &str, layers: u32, inter: u64) -> PathBuf {
    let f = dir.join(name);
    std::fs::write(&f, real_layout(layers, 8, inter, 10, &["unsloth_calibration_dataset"])).unwrap();
    f
}

/// models/<name>/model/ndforge_reshape.json with the keys the Slim job writes
fn cut_model(p: &Project, name: &str, src: &[u64], origin_layers: u64, width: Option<(u64, u64)>, edit: impl Fn(&mut Value)) -> PathBuf {
    let dir = models_dir(p).join(name);
    std::fs::create_dir_all(dir.join("model")).unwrap();
    let dropped: Vec<u64> = (0..origin_layers).filter(|i| !src.contains(i)).collect();
    let mut meta = json!({"format": if width.is_some() { 2 } else { 1 }, "tool": "ND Lore Forge", "origin": {"repo": "Qwen/Qwen3.8-27B", "family": "qwen3_5", "layers": origin_layers, "params": 1},
                          "mode": "prune", "layers": src.len(), "src_layers": src, "dropped": dropped, "duplicated": [], "zero_init": []});
    if let Some((a, b)) = width {
        meta["width"] = json!({"intermediate_size": [a, b], "linear_num_value_heads": [4, 4], "num_attention_heads": [6, 6], "kept": "width_kept.npz"});
    }
    edit(&mut meta);
    std::fs::write(dir.join("model").join("ndforge_reshape.json"), serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
    dir
}

const KEEP: [u64; 8] = [0, 1, 2, 3, 12, 13, 14, 15];

fn spec_for(file: &Path, model: &str) -> ad::AdaptSpec {
    ad::AdaptSpec { source: lib::MergeInput { kind: "file".into(), file: Some(file.display().to_string()), ..Default::default() }, model: model.into(), ..Default::default() }
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

// ------------------------------------------------------------------------------------------------------------------ the choice

#[test]
fn the_choice_and_old_specs_load() {
    let s: ad::AdaptSpec = serde_json::from_value(json!({})).unwrap();
    assert!(s.model.is_empty() && s.weight.is_none() && s.kept_layers.is_none() && s.source.kind.is_empty() && s.source_size.is_none());
    let s: ad::AdaptSpec = serde_json::from_value(json!({"source": {"kind": "published", "base": "Qwen/Qwen3.8-27B"}, "model": "slim", "weight": 0.1, "kept_layers": [0, 1], "extra": 1})).unwrap();
    assert_eq!((s.source.base.as_deref(), s.model.as_str(), s.weight, s.kept_layers), (Some("Qwen/Qwen3.8-27B"), "slim", Some(0.1), Some(vec![0, 1])));
    // a merge asked for before this feature existed has no weights and no model to adapt to
    let m: lib::MergeSpec = serde_json::from_value(json!({"inputs": [{"kind": "library", "id": "a"}, {"kind": "published", "base": "o/m"}], "label": "x"})).unwrap();
    assert!(m.inputs.iter().all(|i| i.weight.is_none() && i.adapt_to.is_none()));
    let m: lib::MergeSpec = serde_json::from_value(json!({"inputs": [{"kind": "published", "base": "o/m", "weight": 0.25, "adapt_to": "slim"}]})).unwrap();
    assert_eq!((m.inputs[0].weight, m.inputs[0].adapt_to.as_deref()), (Some(0.25), Some("slim")));
}

#[test]
fn weights_are_numbers_above_zero() {
    assert_eq!(ad::check_weight(None).unwrap(), 1.0);
    assert_eq!(ad::check_weight(Some(0.1)).unwrap(), 0.1);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e7] {
        assert!(ad::check_weight(Some(bad)).unwrap_err().0.contains("must be a number above 0"), "{bad}");
    }
    assert_eq!(ad::thousands(17408), "17,408");
    assert_eq!(ad::thousands(999), "999");
    assert_eq!(ad::thousands(1_149_614), "1,149,614");
}

// ------------------------------------------------------------------------------------------------------------------ what a made model says

#[test]
fn a_record_gives_the_kept_layers_the_widths_and_the_refusals() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let cut = cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    let f = ad::cut_facts(&cut, "slim", None).unwrap();
    assert_eq!((f.origin_layers, f.layers, f.kept.clone(), f.inter, f.refusal.clone(), f.thinner()), (16, 8, Some(KEEP.to_vec()), Some((24, 10)), None, true));
    assert_eq!(f.dropped(), vec![4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(f.origin_repo.as_deref(), Some("Qwen/Qwen3.8-27B"));
    // only the dropped layers are listed
    let c2 = cut_model(&p, "d", &KEEP, 16, None, |m| { m.as_object_mut().unwrap().remove("src_layers"); });
    assert_eq!(ad::cut_facts(&c2, "d", None).unwrap().kept, Some(KEEP.to_vec()));
    // only counts: the person is asked, and the answer is checked
    let c3 = cut_model(&p, "old", &KEEP, 16, None, |m| { let o = m.as_object_mut().unwrap(); o.remove("src_layers"); o.remove("dropped"); });
    let f = ad::cut_facts(&c3, "old", None).unwrap();
    assert_eq!(f.refusal.as_ref().map(|r| r.0.as_str()), Some("needs_layer_map"));
    assert!(f.refusal.unwrap().1.contains("8 layers of the original 16") && ad::cut_models(&p).iter().any(|m| m["name"] == "old" && m["needs_layer_map"] == true));
    let ok = ad::cut_facts(&c3, "old", Some(&[0, 1, 2, 3, 12, 13, 14, 15])).unwrap();
    assert!(ok.refusal.is_none() && ok.layers == 8);
    for bad in [vec![0u32, 1, 2], vec![3, 2, 1, 0, 12, 13, 14, 15], vec![0, 1, 2, 3, 12, 13, 14, 99]] {
        assert_eq!(ad::cut_facts(&c3, "old", Some(&bad)).unwrap().refusal.map(|r| r.0), Some("bad_kept".to_string()), "{bad:?}");
    }
    // the refusals
    let h = cut_model(&p, "heads", &KEEP, 16, Some((24, 10)), |m| m["width"]["num_attention_heads"] = json!([6, 4]));
    let r = ad::cut_facts(&h, "heads", None).unwrap().refusal.unwrap();
    assert!(r.0 == "heads" && r.1.contains("attention heads were cut from 6 to 4") && r.1.contains("experimental"));
    let g = cut_model(&p, "grown", &KEEP, 16, None, |m| { m["duplicated"] = json!([3]); m["mode"] = json!("grow"); });
    assert_eq!(ad::cut_facts(&g, "grown", None).unwrap().refusal.map(|r| r.0), Some("grown".to_string()));
    let n = cut_model(&p, "healed", &KEEP, 16, None, |m| { m["origin"] = json!({}); });
    assert_eq!(ad::cut_facts(&n, "healed", None).unwrap().refusal.map(|r| r.0), Some("no_metadata".to_string()));
    assert!(ad::cut_facts(&d.path().join("nowhere"), "x", None).is_none());
    // the page's list
    let list = ad::cut_models(&p);
    assert_eq!(list.len(), 6);
    let slim = list.iter().find(|m| m["name"] == "slim").unwrap();
    assert_eq!((slim["layers_before"].as_u64(), slim["layers_after"].as_u64(), slim["intermediate_before"].as_u64(), slim["intermediate_after"].as_u64(), slim["thinner"].clone(), slim["refusal"].clone()), (Some(16), Some(8), Some(24), Some(10), json!(true), Value::Null));
    assert!(list.iter().find(|m| m["name"] == "heads").unwrap()["refusal"].as_str().unwrap().contains("head axes"));
    assert!(ad::model_dir(&p, "../x").is_err() && ad::model_dir(&p, "").is_err() && ad::model_dir(&p, "missing").unwrap_err().0.contains("no record of how it was cut"));
    assert!(ad::model_dir(&p, "slim").is_ok());
}

// ------------------------------------------------------------------------------------------------------------------ the preview

#[test]
fn the_preview_gives_the_numbers_and_the_exact_size_of_the_file_the_worker_writes() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    cut_model(&p, "depth", &KEEP, 16, None, |_| {});
    let f = matrix_file(d.path(), "pub.gguf", 16, 24);
    assert_eq!(std::fs::metadata(&f).unwrap().len(), 24736, "the same bytes as the python fixture generator");
    let v = ad::preview(&p, &spec_for(&f, "slim")).unwrap();
    assert_eq!((v["ok"].clone(), v["layers_before"].as_u64(), v["layers_after"].as_u64(), v["ffn_down_before"].as_u64(), v["ffn_down_after"].as_u64(), v["exact"].clone()), (json!(true), Some(16), Some(8), Some(24), Some(10), json!(true)));
    assert_eq!((v["size_after_bytes"].as_u64(), v["tensors_before"].as_u64(), v["tensors_after"].as_u64()), (Some(12192), Some(2 * (12 * 5 + 4 * 4 + 16 * 3)), Some(2 * (6 * 5 + 2 * 4 + 8 * 3))));
    assert_eq!(v["text"], "keeps 8 of 16 layers; ffn_down 24 -> 10 per layer; exactly 0.0 MB");
    let v = ad::preview(&p, &spec_for(&f, "depth")).unwrap();
    assert_eq!((v["size_after_bytes"].as_u64(), v["ffn_down_after"].as_u64(), v["thinner"].clone()), (Some(12448), Some(24), json!(false)));
    assert!(v["text"].as_str().unwrap().starts_with("keeps 8 of 16 layers; ffn_down stays 24 per layer"));
}

#[test]
fn the_preview_refuses_in_the_workers_words() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    cut_model(&p, "heads", &KEEP, 16, Some((24, 10)), |m| m["width"]["num_attention_heads"] = json!([6, 4]));
    cut_model(&p, "old", &KEEP, 16, None, |m| { let o = m.as_object_mut().unwrap(); o.remove("src_layers"); o.remove("dropped"); });
    let code = |s: &ad::AdaptSpec| { let v = ad::preview(&p, s).unwrap(); assert_eq!(v["ok"], false, "{v}"); (v["code"].as_str().unwrap().to_string(), v["text"].as_str().unwrap().to_string()) };
    let good = matrix_file(d.path(), "pub.gguf", 16, 24);
    let few = matrix_file(d.path(), "few.gguf", 12, 24);
    let thin = matrix_file(d.path(), "thin.gguf", 16, 20);
    let done = d.path().join("done.gguf");
    std::fs::write(&done, real_layout(16, 8, 24, 10, &["unsloth_calibration_dataset", "(adapted to other)"])).unwrap();
    let (c, t) = code(&spec_for(&few, "slim"));
    assert!(c == "wrong_original" && t.contains("data for 12 layers") && t.contains("has 16"), "{t}");
    let (c, t) = code(&spec_for(&thin, "slim"));
    assert!(c == "wrong_original" && t.contains("is 20 wide") && t.contains("has 24"), "{t}");
    assert_eq!(code(&spec_for(&done, "slim")).0, "already_adapted");
    assert_eq!(code(&spec_for(&good, "heads")).0, "heads");
    assert_eq!(code(&spec_for(&good, "old")).0, "needs_layer_map");
    let mut s = spec_for(&good, "old");
    s.kept_layers = Some(KEEP.iter().map(|x| *x as u32).collect());
    assert_eq!(ad::preview(&p, &s).unwrap()["ok"], true, "the person's answer is used");
    // errors
    let junk = d.path().join("junk.gguf");
    std::fs::write(&junk, b"nope").unwrap();
    assert_eq!(code(&spec_for(&junk, "slim")).0, "bad_matrix");
    let e = |s: &ad::AdaptSpec| ad::preview(&p, s).unwrap_err().0;
    assert!(e(&spec_for(&d.path().join("x.dat"), "slim")).contains("only an imatrix.gguf can be adapted"));
    assert!(e(&spec_for(&d.path().join("nowhere.gguf"), "slim")).contains("was not found"));
    let mut w = spec_for(&good, "slim");
    w.weight = Some(0.0);
    assert!(e(&w).contains("above 0"));
    assert!(e(&spec_for(&good, "missing")).contains("no record of how it was cut"));
    let mut lib_src = spec_for(&good, "slim");
    lib_src.source = lib::MergeInput { kind: "library".into(), id: Some("nope".into()), ..Default::default() };
    assert!(e(&lib_src).contains("no longer exists"));
    lib_src.source = lib::MergeInput { kind: "nonsense".into(), ..Default::default() };
    assert!(e(&lib_src).contains("Choose the matrix to adapt"));
}

#[test]
fn a_library_entry_and_a_downloaded_published_matrix_are_found_and_an_undownloaded_one_is_estimated() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    let bytes = real_layout(16, 8, 24, 10, &["unsloth_calibration_dataset"]);
    entry_on_disk(&p, "20261101-1000-a", &[("imatrix.gguf", &bytes)], Some(json!({"label": "mine"})));
    entry_on_disk(&p, "20261101-1001-dat", &[("imatrix.dat", b"LEGACY")], None);
    let mut s = spec_for(Path::new("x"), "slim");
    s.source = lib::MergeInput { kind: "library".into(), id: Some("20261101-1000-a".into()), ..Default::default() };
    assert_eq!(ad::preview(&p, &s).unwrap()["size_after_bytes"], 12192);
    s.source.id = Some("20261101-1001-dat".into());
    assert!(ad::preview(&p, &s).unwrap_err().0.contains("only a legacy .dat"));
    // published: not downloaded -> an estimate from the size the Hub states; downloaded -> exact
    s.source = lib::MergeInput { kind: "published".into(), base: Some("Qwen/Qwen3.8-27B".into()), ..Default::default() };
    s.source_size = Some(13_642_656);
    let v = ad::preview(&p, &s).unwrap();
    assert_eq!((v["exact"].clone(), v["layers_after"].as_u64()), (json!(false), Some(8)));
    let est = v["size_after_mb"].as_f64().unwrap();
    assert!((est - 13_642_656.0 / 1048576.0 * 0.5 * (1.0 - 0.3 * (1.0 - 10.0 / 24.0))).abs() < 0.01, "{est}");
    assert!(v["text"].as_str().unwrap().contains("about"));
    s.source_size = None;
    let v = ad::preview(&p, &s).unwrap();
    assert!(v["size_after_mb"].is_null() && v["text"] == "keeps 8 of 16 layers; ffn_down 24 -> 10 per layer");
    let cache = lib::downloads_dir(&p).join("unsloth--Qwen3.8-27B-GGUF");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::write(cache.join("imatrix_unsloth.gguf"), &bytes).unwrap();
    let v = ad::preview(&p, &s).unwrap();
    assert_eq!((v["exact"].clone(), v["size_after_bytes"].as_u64()), (json!(true), Some(12192)));
    assert_eq!(ad::published_repo("unsloth/Qwen3.8-27B-unsloth-bnb-4bit").as_deref(), Some("unsloth/Qwen3.8-27B-GGUF"));
    assert_eq!(ad::published_repo("Qwen3.8-27B").as_deref(), Some("unsloth/Qwen3.8-27B-GGUF"));
    assert_eq!(ad::published_repo("someone/Model-GGUF").as_deref(), Some("someone/Model-GGUF"));
    assert_eq!(ad::published_repo(&d.path().display().to_string()), None);
}

// ------------------------------------------------------------------------------------------------------------------ the job and the merge job

#[test]
fn the_adapt_job_carries_the_source_the_model_and_the_weight() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let cut = cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    let f = matrix_file(d.path(), "pub.gguf", 16, 24);
    let mut s = spec_for(&f, "slim");
    s.weight = Some(0.1);
    s.label = "  adapted  ".into();
    let j = ad::build_job(&p, &s, "20261116-1000-slim-adapted", Path::new("S")).unwrap();
    assert_eq!((j["job"].as_str(), j["id"].as_str(), j["label"].as_str(), j["model_name"].as_str(), j["weight"].as_f64()), (Some("imatrix_adapt"), Some("20261116-1000-slim-adapted"), Some("adapted"), Some("slim"), Some(0.1)));
    assert_eq!((j["model_dir"].clone(), j["root"].clone(), j["cache"].clone(), j["source"]["kind"].clone()), (json!(cut), json!(lib::lib_dir(&p)), json!(lib::downloads_dir(&p)), json!("file")));
    assert!(j["kept_layers"].is_null());
    s.kept_layers = Some(vec![0, 1]);
    assert_eq!(ad::build_job(&p, &s, "x", Path::new("S")).unwrap()["kept_layers"], json!([0, 1]));
    // a published matrix goes by its base; a library entry by its file, label and chunks; an adapted entry is refused
    s.source = lib::MergeInput { kind: "published".into(), base: Some(" Qwen/Qwen3.8-27B ".into()), ..Default::default() };
    let j = ad::build_job(&p, &s, "x", Path::new("S")).unwrap();
    assert_eq!((j["source"]["kind"].as_str(), j["source"]["base"].as_str()), (Some("published"), Some("Qwen/Qwen3.8-27B")));
    entry_on_disk(&p, "e1", &[("imatrix.gguf", b"A")], Some(json!({"label": "mine", "chunks": 64, "source_kind": "merged"})));
    entry_on_disk(&p, "e2", &[("imatrix.gguf", b"A")], Some(json!({"source_kind": "adapted"})));
    s.source = lib::MergeInput { kind: "library".into(), id: Some("e1".into()), ..Default::default() };
    let j = ad::build_job(&p, &s, "x", Path::new("S")).unwrap();
    assert_eq!((j["source"]["label"].as_str(), j["source"]["chunks"].as_u64(), j["source"]["source_kind"].as_str()), (Some("mine"), Some(64), Some("merged")));
    s.source.id = Some("e2".into());
    assert!(ad::build_job(&p, &s, "x", Path::new("S")).unwrap_err().0.contains("already an adapted one"));
}

fn merge_spec(inputs: Vec<lib::MergeInput>) -> lib::MergeSpec {
    lib::MergeSpec { inputs, label: "m".into() }
}

#[test]
fn the_merge_job_carries_weights_and_adapts_the_published_matrix_to_the_cut_model_of_the_other_one() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let cut = cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    entry_on_disk(&p, "own", &[("imatrix.gguf", b"A")], Some(json!({"label": "mine", "chunks": 128, "cut_model": {"dir": cut, "name": "slim"}})));
    entry_on_disk(&p, "plain", &[("imatrix.gguf", b"B")], Some(json!({"label": "plain", "chunks": 128})));
    let lib_in = |id: &str| lib::MergeInput { kind: "library".into(), id: Some(id.into()), ..Default::default() };
    let pubn = |w: Option<f64>, a: Option<&str>| lib::MergeInput { kind: "published".into(), base: Some("Qwen/Qwen3.8-27B".into()), weight: w, adapt_to: a.map(String::from), ..Default::default() };
    // no cut model among the inputs: nothing is adapted, weights default to 1 (the old behaviour)
    let j = lib::build_merge_job(&p, &merge_spec(vec![lib_in("plain"), pubn(None, None)]), "x", Path::new("S")).unwrap();
    let i = j["inputs"].as_array().unwrap();
    assert!(i[1]["adapt_to"].is_null() && i[0]["weight"] == 1.0 && i[1]["weight"] == 1.0);
    // a library entry measured on a cut model: the published matrix is adapted to it, automatically, with the weight asked for
    let j = lib::build_merge_job(&p, &merge_spec(vec![lib_in("own"), pubn(Some(0.1), None)]), "x", Path::new("S")).unwrap();
    let i = j["inputs"].as_array().unwrap();
    assert_eq!((i[1]["adapt_to"]["model_name"].as_str(), i[1]["adapt_to"]["model_dir"].clone(), i[1]["auto_adapted"].clone(), i[1]["weight"].as_f64()), (Some("slim"), json!(cut), json!(true), Some(0.1)));
    assert!(i[0]["adapt_to"].is_null() && i[0]["cut_model"]["name"] == "slim", "the entry measured on the cut model is not adapted (it already fits)");
    // an explicit model works for any input
    let j = lib::build_merge_job(&p, &merge_spec(vec![lib_in("plain"), pubn(None, Some("slim"))]), "x", Path::new("S")).unwrap();
    assert_eq!(j["inputs"][1]["adapt_to"]["model_name"], "slim");
    assert!(j["inputs"][1].get("auto_adapted").is_none());
    // refusals: a model whose cut cannot be adapted, an unknown model, a bad weight, two different cut models
    cut_model(&p, "heads", &KEEP, 16, Some((24, 10)), |m| m["width"]["num_attention_heads"] = json!([6, 4]));
    let e = |inputs: Vec<lib::MergeInput>| lib::build_merge_job(&p, &merge_spec(inputs), "x", Path::new("S")).unwrap_err().0;
    assert!(e(vec![lib_in("plain"), pubn(None, Some("heads"))]).contains("head axes"));
    assert!(e(vec![lib_in("plain"), pubn(None, Some("missing"))]).contains("no record of how it was cut"));
    assert!(e(vec![lib_in("plain"), pubn(Some(-1.0), None)]).contains("above 0"));
    let other = cut_model(&p, "other", &KEEP[..4], 16, None, |_| {});
    entry_on_disk(&p, "own2", &[("imatrix.gguf", b"C")], Some(json!({"label": "mine2", "cut_model": {"dir": other, "name": "other"}})));
    assert!(e(vec![lib_in("own"), lib_in("own2"), pubn(None, None)]).contains("different cut models"));
    // two entries of the same cut model are fine
    entry_on_disk(&p, "own3", &[("imatrix.gguf", b"D")], Some(json!({"label": "mine3", "cut_model": {"dir": cut, "name": "slim"}})));
    assert_eq!(lib::build_merge_job(&p, &merge_spec(vec![lib_in("own"), lib_in("own3"), pubn(None, None)]), "x", Path::new("S")).unwrap()["inputs"][2]["adapt_to"]["model_name"], "slim");
    // an entry whose model folder is gone is not a cut model any more: the worker's rule (refuse a raw merge with other layers) still stands
    entry_on_disk(&p, "gone", &[("imatrix.gguf", b"E")], Some(json!({"cut_model": {"dir": d.path().join("deleted"), "name": "x"}})));
    assert!(lib::build_merge_job(&p, &merge_spec(vec![lib_in("gone"), pubn(None, None)]), "x", Path::new("S")).unwrap()["inputs"][1]["adapt_to"].is_null());
}

#[test]
fn the_state_of_the_page_lists_the_cut_models() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    assert_eq!(lib::state(&p)["cut_models"], json!([]));
    cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    let s = lib::state(&p);
    assert_eq!((s["cut_models"][0]["name"].as_str(), s["cut_models"][0]["layers_after"].as_u64()), (Some("slim"), Some(8)));
}

#[test]
fn the_page_knows_which_runs_and_models_were_cut_and_offers_their_original() {
    let d = tempdir().unwrap();
    let p = plain_project(d.path());
    let cut = cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    let rdir = ndforge_core::train::runs::runs_dir(&p).join("runcut");
    std::fs::create_dir_all(rdir.join("model")).unwrap();
    std::fs::write(rdir.join("model/config.json"), "{}").unwrap();
    std::fs::write(rdir.join("config.json"), json!({"model": {"repo": cut.join("model").display().to_string(), "family": "qwen3_5"}}).to_string()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": "runcut", "status": "done", "model": "slim", "method": "full"}).to_string()).unwrap();
    let rdir2 = ndforge_core::train::runs::runs_dir(&p).join("runhub");
    std::fs::create_dir_all(rdir2.join("model")).unwrap();
    std::fs::write(rdir2.join("model/config.json"), "{}").unwrap();
    std::fs::write(rdir2.join("config.json"), json!({"model": {"repo": "Qwen/Qwen3.5-4B", "family": "qwen3_5"}}).to_string()).unwrap();
    std::fs::write(rdir2.join("state.json"), json!({"name": "runhub", "status": "done", "model": "q", "method": "full"}).to_string()).unwrap();
    let st = lib::state(&p);
    let run = |n: &str| st["runs"].as_array().unwrap().iter().find(|r| r["name"] == n).cloned().unwrap_or_else(|| panic!("{n} not offered: {}", st["runs"]));
    assert_eq!((run("runcut")["base"].as_str(), run("runcut")["cut"].clone()), (Some("Qwen/Qwen3.8-27B"), json!(true)), "a run on a cut model: its original from the record, flagged as cut");
    assert_eq!((run("runhub")["base"].as_str(), run("runhub")["cut"].clone()), (Some("Qwen/Qwen3.5-4B"), json!(false)));
    let bases = st["published_bases"].as_array().unwrap();
    assert!(bases.iter().any(|b| b["repo"] == "Qwen/Qwen3.8-27B" && b["label"].as_str().unwrap().contains("the original of the cut model slim")), "{bases:?}");
}

// ------------------------------------------------------------------------------------------------------------------ fake workers (unix)

#[cfg(unix)]
fn fake_worker(root: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    let bin = root.join("env/bin");
    std::fs::create_dir_all(&bin).unwrap();
    let py = bin.join("python");
    std::fs::write(&py, format!("#!/bin/sh\njob=\"$5\"\nlibroot=$(sed -n 's/.*\"root\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\nid=$(grep -o '\"id\":\"[0-9]\\{{8\\}}-[0-9]\\{{4\\}}-[^\"]*\"' \"$job\" | head -1 | cut -d'\"' -f4)\nstop=$(sed -n 's/.*\"stop_file\":\"\\([^\"]*\\)\".*/\\1/p' \"$job\")\n{body}\n")).unwrap();
    std::fs::set_permissions(&py, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("NDFORGE_ENV_DIR", root.join("env"));
    std::env::set_var("NDFORGE_CONFIG_DIR", root.join("cfg"));
}

#[cfg(unix)]
#[test]
fn an_adaptation_runs_in_a_fake_worker_holds_its_library_source_and_the_entry_appears() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    fake_worker(d.path(), r#"cp "$job" "$libroot/last-job.json"
mkdir -p "$libroot/$id"
printf 'GGUF' > "$libroot/$id/imatrix.gguf"
echo '{"format":1,"label":"adapted","source_kind":"adapted","source":"mine, adapted to the cut model slim","chunks":1251,"created":"2026-11-16T10:30:00Z","via":"adapted","layers":8,"adapt":{"layers_before":16,"layers_after":8}}' > "$libroot/$id/imatrix.json"
echo '{"type":"progress","step":"imatrix","value":0.6,"text":"Writing the adapted matrix"}'
echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"notes\":[\"n\"],\"numbers\":{\"layers_after\":8}}}""#);
    let p = plain_project(d.path());
    cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    entry_on_disk(&p, "20261101-1000-a", &[("imatrix.gguf", b"A")], Some(json!({"label": "mine", "chunks": 64})));
    let mut s = spec_for(Path::new("x"), "slim");
    s.source = lib::MergeInput { kind: "library".into(), id: Some("20261101-1000-a".into()), ..Default::default() };
    s.weight = Some(2.0);
    let (mut held, mut events) = (None, vec![]);
    let r = ad::adapt(&p, &s, &AtomicBool::new(false), &mut |e| { held = Some((lib::in_use("20261101-1000-a"), lib::delete(&p, "20261101-1000-a").unwrap_err().0)); events.push(e); }).unwrap();
    let (inuse, why) = held.unwrap();
    assert!(inuse && why.contains("being used by a job") && !lib::in_use("20261101-1000-a"));
    assert!(r["entry"]["id"].as_str().unwrap().ends_with("-slim-adapted") && r["entry"]["source_kind"] == "adapted" && r["entry"]["has_gguf"] == true, "{r}");
    assert!(events.iter().any(|e| e["kind"] == "progress" && e["text"] == "Writing the adapted matrix"));
    let sent: Value = serde_json::from_slice(&std::fs::read(lib::lib_dir(&p).join("last-job.json")).unwrap()).unwrap();
    assert_eq!((sent["job"].as_str(), sent["model_name"].as_str(), sent["weight"].as_f64(), sent["source"]["id"].as_str()), (Some("imatrix_adapt"), Some("slim"), Some(2.0), Some("20261101-1000-a")));
    assert!(!lib::lib_dir(&p).join("STOP").exists());
    // the worker's refusal comes back in its words; no training software is said in plain words too
    fake_worker(d.path(), r#"echo '{"type":"error","code":"adapt_wrong_original","message":"This matrix has data for 12 layers, but the original of the model slim has 16."}'; exit 1"#);
    assert!(ad::adapt(&p, &s, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("data for 12 layers"));
    assert_eq!(lib::list(&p).len(), 2);
    std::env::set_var("NDFORGE_ENV_DIR", d.path().join("none"));
    assert!(ad::adapt(&p, &s, &AtomicBool::new(false), &mut |_| {}).unwrap_err().0.contains("Training setup"));
}

#[cfg(unix)]
#[test]
fn the_creation_tick_adapts_the_published_matrix_for_a_cut_model_and_merges_it_raw_for_a_matching_one() {
    let _g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let d = tempdir().unwrap();
    let script = r#"if grep -q '"job":"imatrix_make"' "$job"; then
  mkdir -p "$libroot/$id"
  printf 'GGUF' > "$libroot/$id/imatrix.gguf"
  if [ -n "$CUT_DIR" ]; then echo "{\"format\":1,\"model\":\"slim\",\"chunks\":64,\"created\":\"2026-11-16T10:30:00Z\",\"via\":\"llama.cpp\",\"cut_model\":{\"dir\":\"$CUT_DIR\",\"name\":\"slim\"}}" > "$libroot/$id/imatrix.json";
  else echo '{"format":1,"model":"run1","chunks":64,"created":"2026-11-16T10:30:00Z","via":"llama.cpp"}' > "$libroot/$id/imatrix.json"; fi
  echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"notes\":[]}}"
else
  id=$(grep -o '"id":"[0-9]*-[0-9]*-merged[^"]*"' "$job" | head -1 | cut -d'"' -f4)
  cp "$job" "$libroot/merge-job.json"
  mkdir -p "$libroot/$id"
  printf 'GGUF' > "$libroot/$id/imatrix.gguf"
  echo '{"format":1,"source_kind":"merged","chunks":1064,"created":"2026-11-16T10:31:00Z","via":"llama.cpp"}' > "$libroot/$id/imatrix.json"
  echo "{\"type\":\"result\",\"data\":{\"id\":\"$id\",\"warnings\":[]}}"
fi"#;
    fake_worker(d.path(), script);
    let p = plain_project(d.path());
    // a run trained on a model cut on the Slim page: its base is the cut model's folder; the published matrix of the ORIGINAL (from the record) is adapted to it
    let cut = cut_model(&p, "slim", &KEEP, 16, Some((24, 10)), |_| {});
    std::env::set_var("CUT_DIR", &cut);
    let rdir = ndforge_core::train::runs::runs_dir(&p).join("run1");
    std::fs::create_dir_all(rdir.join("model")).unwrap();
    std::fs::write(rdir.join("model/config.json"), "{}").unwrap();
    std::fs::write(rdir.join("config.json"), json!({"model": {"repo": cut.join("model").display().to_string(), "family": "qwen3_5"}}).to_string()).unwrap();
    std::fs::write(rdir.join("state.json"), json!({"name": "run1", "status": "done", "model": "slim", "method": "full"}).to_string()).unwrap();
    let make = lib::MakeSpec { source: lib::SourceSpec { kind: "run".into(), run: Some("run1".into()), ..Default::default() }, calibration: "text".into(), text_file: None, merge_published: true, label: "cut run".into(), chunks: 64, ..Default::default() };
    let txt = d.path().join("t.txt");
    std::fs::write(&txt, "x").unwrap();
    let make = lib::MakeSpec { text_file: Some(txt.display().to_string()), ..make };
    let r = lib::create(&p, &make, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["merged"]["entry"]["source_kind"], "merged", "{r}");
    let sent: Value = serde_json::from_slice(&std::fs::read(lib::lib_dir(&p).join("merge-job.json")).unwrap()).unwrap();
    let ins = sent["inputs"].as_array().unwrap();
    assert_eq!((ins[1]["kind"].as_str(), ins[1]["base"].as_str(), ins[1]["adapt_to"]["model_name"].as_str(), ins[1]["auto_adapted"].clone()), (Some("published"), Some("Qwen/Qwen3.8-27B"), Some("slim"), json!(true)), "{sent}");
    assert_eq!(ins[1]["adapt_to"]["model_dir"], json!(cut));
    // a model that matches the matrix (a base of the Hub): merged as before, nothing adapted
    std::env::remove_var("CUT_DIR");
    std::fs::write(rdir.join("config.json"), json!({"model": {"repo": "Qwen/Qwen3.8-27B", "family": "qwen3_5"}}).to_string()).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let r = lib::create(&p, &make, &AtomicBool::new(false), &mut |_| {}).unwrap();
    assert_eq!(r["merged"]["entry"]["source_kind"], "merged", "{r}");
    let sent: Value = serde_json::from_slice(&std::fs::read(lib::lib_dir(&p).join("merge-job.json")).unwrap()).unwrap();
    assert!(sent["inputs"][1]["adapt_to"].is_null() && sent["inputs"][1]["base"] == "Qwen/Qwen3.8-27B", "{sent}");
}
