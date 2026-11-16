//! A minimal reader for the header of a GGUF file (what an importance matrix says about itself), for the light "does this matrix fit this model?" check of the
//! matrix library. Only the header is read: keys with numbers and strings, and the tensor names. The worker has the same reader (`imatrix.read_gguf_header`) and the
//! same texts (`imatrix.compat_warnings`).
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::{json, Value};

/// What a GGUF-format matrix says about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixSummary {
    pub arch: Option<String>,
    /// the highest `blk.N.` plus one
    pub layers: Option<u64>,
    /// number of tensors measured
    pub entries: u64,
    pub chunks: Option<u64>,
    /// `imatrix.chunk_size`
    pub chunk_size: Option<u64>,
    /// the input width of `ffn_gate` (the model's hidden size); the real published matrix has no `general.architecture`, so shapes tell what it was measured on
    pub hidden: Option<u64>,
    /// the input width of `ffn_down` (the MLP intermediate size)
    pub intermediate: Option<u64>,
}

/// One tensor of the header: its name and dimensions (the real matrix has ONE dimension per tensor).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
}

/// (keys with a number or a string value, tensor names), or None when the file is no readable GGUF (a legacy .dat, a truncated file, an absurd header).
pub fn read_header(path: &Path) -> Option<(BTreeMap<String, Value>, Vec<String>)> {
    read_tensors(path).map(|(m, t)| (m, t.into_iter().map(|t| t.name).collect()))
}

/// Like `read_header`, with the dimensions of every tensor.
pub fn read_tensors(path: &Path) -> Option<(BTreeMap<String, Value>, Vec<TensorInfo>)> {
    read_full(path).map(|h| (h.meta, h.tensors))
}

/// The whole header: keys, tensors with their dimensions, and where the key-values end and the tensor infos end (needed to predict the size of a file with other tensors).
pub struct Header {
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub kv_end: u64,
    pub info_end: u64,
}

pub fn read_full(path: &Path) -> Option<Header> {
    let mut fh = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    fn take<const N: usize>(fh: &mut impl Read) -> Option<[u8; N]> {
        let mut b = [0u8; N];
        fh.read_exact(&mut b).ok()?;
        Some(b)
    }
    fn u32_(fh: &mut impl Read) -> Option<u32> { take::<4>(fh).map(u32::from_le_bytes) }
    fn u64_(fh: &mut impl Read) -> Option<u64> { take::<8>(fh).map(u64::from_le_bytes) }
    fn string(fh: &mut impl Read) -> Option<String> {
        let n = u64_(fh)?;
        if n > 1 << 24 {
            return None;
        }
        let mut b = vec![0u8; n as usize];
        fh.read_exact(&mut b).ok()?;
        Some(String::from_utf8_lossy(&b).into_owned())
    }
    fn value<R: Read + Seek>(fh: &mut R, kind: u32) -> Option<Value> {
        let fixed = |k: u32| -> Option<u64> { Some(match k { 0 | 1 | 7 => 1, 2 | 3 => 2, 4..=6 => 4, 10..=12 => 8, _ => return None }) };
        Some(match kind {
            0 => json!(take::<1>(fh)?[0]),
            1 => json!(take::<1>(fh)?[0] as i8),
            2 => json!(u16::from_le_bytes(take::<2>(fh)?)),
            3 => json!(i16::from_le_bytes(take::<2>(fh)?)),
            4 => json!(u32_(fh)?),
            5 => json!(i32::from_le_bytes(take::<4>(fh)?)),
            6 => json!(f32::from_le_bytes(take::<4>(fh)?)),
            7 => json!(take::<1>(fh)?[0] != 0),
            10 => json!(u64_(fh)?),
            11 => json!(i64::from_le_bytes(take::<8>(fh)?)),
            12 => json!(f64::from_le_bytes(take::<8>(fh)?)),
            8 => json!(string(fh)?),
            9 => {
                let (inner, n) = (u32_(fh)?, u64_(fh)?);
                if let Some(w) = fixed(inner) {
                    fh.seek(SeekFrom::Current((w.checked_mul(n)?) as i64)).ok()?;
                    Value::Null
                } else {
                    if n > 1 << 26 {
                        return None;
                    }
                    let mut items = Vec::new();
                    for _ in 0..n {
                        items.push(value(fh, inner)?);
                    }
                    // a short array of strings (an importance matrix's `imatrix.datasets`) is kept
                    if inner == 8 && n <= 64 { Value::Array(items) } else { Value::Null }
                }
            }
            _ => return None,
        })
    }
    if &take::<4>(&mut fh)? != b"GGUF" || u32_(&mut fh)? < 2 {
        return None;
    }
    let (tensors, kvs) = (u64_(&mut fh)?, u64_(&mut fh)?);
    if tensors > 1 << 22 || kvs > 1 << 22 {
        return None;
    }
    let mut meta = BTreeMap::new();
    for _ in 0..kvs {
        let key = string(&mut fh)?;
        let kind = u32_(&mut fh)?;
        meta.insert(key, value(&mut fh, kind)?);
    }
    let kv_end = fh.stream_position().ok()?;
    let mut names = Vec::new();
    for _ in 0..tensors {
        let name = string(&mut fh)?;
        let nd = u32_(&mut fh)?;
        if nd > 8 {
            return None;
        }
        let mut dims = Vec::new();
        for _ in 0..nd {
            dims.push(u64_(&mut fh)?);
        }
        u32_(&mut fh)?;
        u64_(&mut fh)?;
        names.push(TensorInfo { name, dims });
    }
    let info_end = fh.stream_position().ok()?;
    Some(Header { meta, tensors: names, kv_end, info_end })
}

pub fn matrix_summary(path: &Path) -> Option<MatrixSummary> {
    let (meta, tensors) = read_tensors(path)?;
    let names: Vec<&String> = tensors.iter().map(|t| &t.name).collect();
    let width = |kind: &str| tensors.iter().find(|t| t.name.ends_with(&format!(".{kind}.weight.in_sum2")) && t.dims.len() == 1).map(|t| t.dims[0]);
    let blocks = names.iter().filter_map(|n| n.strip_prefix("blk.")?.split('.').next()?.parse::<u64>().ok());
    let sums = names.iter().filter(|n| n.ends_with(".in_sum2")).count() as u64;
    Some(MatrixSummary {
        arch: meta.get("general.architecture").and_then(|v| v.as_str()).map(String::from),
        layers: blocks.max().map(|m| m + 1),
        entries: if sums > 0 { sums } else { names.len() as u64 / 2 },
        chunks: meta.get("imatrix.chunk_count").and_then(|v| v.as_u64()),
        chunk_size: meta.get("imatrix.chunk_size").and_then(|v| v.as_u64()),
        hidden: width("ffn_gate"),
        intermediate: width("ffn_down"),
    })
}

/// What the model being exported is, as far as known: architecture or model type, layers, MTP layers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelFacts {
    pub arch: Option<String>,
    pub layers: Option<u64>,
    pub mtp_layers: u64,
    /// `hidden_size` and `intermediate_size` of config.json
    pub hidden: Option<u64>,
    pub intermediate: Option<u64>,
}

/// From a model folder's config.json (also inside `text_config`).
pub fn folder_facts(dir: &Path) -> ModelFacts {
    let Some(cfg) = std::fs::read(dir.join("config.json")).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else { return ModelFacts::default() };
    let mut f = ModelFacts::default();
    for d in [&cfg, &cfg["text_config"]] {
        f.arch = f.arch.or_else(|| d["model_type"].as_str().map(String::from));
        f.layers = f.layers.or_else(|| d["num_hidden_layers"].as_u64());
        f.hidden = f.hidden.or_else(|| d["hidden_size"].as_u64());
        f.intermediate = f.intermediate.or_else(|| d["intermediate_size"].as_u64());
        for k in ["mtp_num_hidden_layers", "num_nextn_predict_layers"] {
            f.mtp_layers = f.mtp_layers.max(d[k].as_u64().unwrap_or(0));
        }
    }
    f
}

/// Plain-language warnings about using the matrix for this model; empty when it looks fine or nothing can be compared. Never a refusal. The texts are the worker's
/// (`imatrix.compat_warnings`).
pub fn compat_warnings(matrix: &Path, model: &ModelFacts) -> Vec<String> {
    let name = matrix.file_name().and_then(|n| n.to_str()).unwrap_or("matrix");
    let Some(s) = matrix_summary(matrix) else {
        return vec![format!("The matrix file {name} could not be read here (a legacy .dat file or an unusual file), so it could not be checked against this model. llama-quantize will tell if it does not fit.")];
    };
    let mut out = Vec::new();
    if let (Some(a), Some(b)) = (&s.arch, &model.arch) {
        if a != b {
            out.push(format!("This matrix says it was made for the architecture \"{a}\", but the model is \"{b}\". It will most likely not fit."));
        }
    }
    if let (Some(m), Some(l)) = (s.layers, model.layers) {
        if m != l && m != l + model.mtp_layers {
            out.push(format!("This matrix has data for {m} layers, but the model has {l}. It was probably made from a different model; llama-quantize decides whether it can use it."));
        }
    }
    if let (Some(m), Some(h)) = (s.hidden, model.hidden) {
        if m != h {
            out.push(format!("This matrix was measured on a model with hidden size {m} (the input width of its ffn_gate), but this model's is {h}. It was made from a different model; llama-quantize decides whether it can use it."));
        }
    }
    if let (Some(m), Some(i)) = (s.intermediate, model.intermediate) {
        if m != i {
            out.push(format!("In this matrix the input of ffn_down is {m} wide, in the model {i}. That is what a model made thinner on the Slim or grow page looks like: \"Adapt to my cut model\" (Importance matrices) cuts a matrix to such a model; llama-quantize cannot use this one as it is."));
        }
    }
    out
}

/// A tiny hand-built GGUF header (version 3): string and u32 keys, tensors with one dimension. For tests only.
#[cfg(test)]
pub(crate) fn build_for_tests(kvs: &[(&str, Value)], tensors: &[&str]) -> Vec<u8> {
    let s = |x: &str| { let mut v = (x.len() as u64).to_le_bytes().to_vec(); v.extend(x.as_bytes()); v };
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs {
        out.extend(s(k));
        match v {
            Value::String(t) => { out.extend(8u32.to_le_bytes()); out.extend(s(t)); }
            other => { out.extend(4u32.to_le_bytes()); out.extend((other.as_u64().unwrap() as u32).to_le_bytes()); }
        }
    }
    for t in tensors {
        out.extend(s(t));
        out.extend(1u32.to_le_bytes());
        out.extend(4u64.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn matrix(layers: u32, arch: Option<&str>) -> Vec<u8> {
        let names: Vec<String> = (0..layers).flat_map(|i| ["in_sum2", "counts"].map(|x| format!("blk.{i}.attn_k.weight.{x}"))).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let mut kv = vec![("general.type", json!("imatrix")), ("imatrix.chunk_count", json!(64))];
        if let Some(a) = arch { kv.push(("general.architecture", json!(a))); }
        build_for_tests(&kv, &refs)
    }

    #[test]
    fn a_matrix_header_is_read_and_summarised() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("imatrix.gguf");
        std::fs::write(&f, matrix(3, Some("qwen3"))).unwrap();
        assert_eq!(matrix_summary(&f), Some(MatrixSummary { arch: Some("qwen3".into()), layers: Some(3), entries: 3, chunks: Some(64), chunk_size: None, hidden: None, intermediate: None }));
        let (meta, names) = read_header(&f).unwrap();
        assert_eq!(meta["general.type"], json!("imatrix"));
        assert_eq!(names.len(), 6);
    }


    /// The real published layout, scaled down: GGUF 3, alignment 32, four keys (an array of strings among them), no architecture, every tensor float32 and ONE-dimensional,
    /// `<weight>.in_sum2` then `<weight>.counts` (one value), full-attention layers at 3, 7, ...
    fn real_layout(layers: u32, hidden: u64, inter: u64) -> Vec<u8> {
        let s = |x: &str| { let mut v = (x.len() as u64).to_le_bytes().to_vec(); v.extend(x.as_bytes()); v };
        let mut bases: Vec<(String, u64)> = vec![];
        for l in 0..layers {
            let kinds: &[&str] = if l % 4 == 3 { &["attn_q", "attn_k", "attn_v", "attn_output"] } else { &["attn_gate", "attn_qkv", "ssm_alpha", "ssm_beta", "ssm_out"] };
            for k in kinds.iter().chain(["ffn_down", "ffn_gate", "ffn_up"].iter()) {
                let w = match *k { "ffn_down" => inter, "attn_output" | "ssm_out" => hidden + 4, _ => hidden };
                bases.push((format!("blk.{l}.{k}.weight"), w));
            }
        }
        bases.sort();
        let mut infos = vec![];
        let (mut data, mut off) = (vec![], 0u64);
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
        out.extend(s("imatrix.datasets")); out.extend(9u32.to_le_bytes()); out.extend(8u32.to_le_bytes()); out.extend(1u64.to_le_bytes()); out.extend(s("unsloth_calibration_dataset"));
        out.extend(s("imatrix.chunk_count")); out.extend(4u32.to_le_bytes()); out.extend(1251u32.to_le_bytes());
        out.extend(s("imatrix.chunk_size")); out.extend(4u32.to_le_bytes()); out.extend(8192u32.to_le_bytes());
        out.extend(infos);
        out.resize(out.len().next_multiple_of(32), 0);
        out.extend(data);
        out
    }

    #[test]
    fn the_real_layout_has_no_architecture_and_is_compared_by_its_shapes() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("imatrix.gguf");
        std::fs::write(&f, real_layout(8, 16, 24)).unwrap();
        let s = matrix_summary(&f).unwrap();
        assert_eq!((s.arch.clone(), s.layers, s.entries, s.chunks, s.chunk_size, s.hidden, s.intermediate), (None, Some(8), 62, Some(1251), Some(8192), Some(16), Some(24)));
        let (meta, tensors) = read_tensors(&f).unwrap();
        assert_eq!(meta["imatrix.datasets"], json!(["unsloth_calibration_dataset"]));
        assert!(tensors.iter().all(|t| t.dims.len() == 1) && tensors.len() == 124);
        let facts = |layers: u64, hidden: u64, inter: u64| ModelFacts { arch: Some("qwen3_5_text".into()), layers: Some(layers), mtp_layers: 0, hidden: Some(hidden), intermediate: Some(inter) };
        assert!(compat_warnings(&f, &facts(8, 16, 24)).is_empty());
        let w = compat_warnings(&f, &facts(4, 16, 12));
        assert_eq!(w.len(), 2);
        assert!(w[0].contains("data for 8 layers, but the model has 4") && w[1].contains("ffn_down is 24 wide, in the model 12") && w[1].contains("Adapt to my cut model"));
        assert!(compat_warnings(&f, &facts(8, 32, 24))[0].contains("hidden size 16"));
    }

    /// Run once on the real published file: NDFORGE_REAL_IMATRIX=/path/imatrix_unsloth.gguf cargo test -p ndforge-core real_published
    #[test]
    fn real_published_file_is_read_like_the_worker_reads_it() {
        let Some(path) = std::env::var_os("NDFORGE_REAL_IMATRIX").map(PathBuf::from).filter(|p| p.is_file()) else { return };
        let s = matrix_summary(&path).expect("readable");
        assert_eq!((s.arch, s.layers, s.entries, s.chunks, s.chunk_size, s.hidden, s.intermediate), (None, Some(64), 496, Some(1251), Some(8192), Some(5120), Some(17408)));
        let (_, t) = read_tensors(&path).unwrap();
        assert_eq!(t.len(), 992);
        assert!(t.iter().all(|t| t.dims.len() == 1));
    }

    #[test]
    fn files_that_are_no_gguf_are_none_not_a_crash() {
        let d = tempfile::tempdir().unwrap();
        let good = matrix(2, None);
        for (i, blob) in [vec![], b"GGU".to_vec(), b"NOPE0000000000000000000000".to_vec(), good[..good.len() - 5].to_vec(), {
            let mut b = b"GGUF".to_vec(); b.extend(3u32.to_le_bytes()); b.extend((1u64 << 40).to_le_bytes()); b.extend(0u64.to_le_bytes()); b
        }, {
            let mut b = b"GGUF".to_vec(); b.extend(1u32.to_le_bytes()); b.extend([0u8; 16]); b
        }].into_iter().enumerate() {
            let f = d.path().join(format!("x{i}"));
            std::fs::write(&f, blob).unwrap();
            assert!(read_header(&f).is_none() && matrix_summary(&f).is_none(), "case {i}");
        }
        assert!(read_header(&d.path().join("missing")).is_none());
    }

    #[test]
    fn the_check_warns_about_architecture_and_layers_and_never_refuses() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("imatrix.gguf");
        std::fs::write(&f, matrix(2, Some("llama"))).unwrap();
        let facts = |arch: Option<&str>, layers: Option<u64>, mtp: u64| ModelFacts { arch: arch.map(String::from), layers, mtp_layers: mtp, ..Default::default() };
        assert!(compat_warnings(&f, &facts(Some("llama"), Some(2), 0)).is_empty());
        assert!(compat_warnings(&f, &facts(None, None, 0)).is_empty(), "nothing to compare");
        assert!(compat_warnings(&f, &facts(Some("llama"), Some(1), 1)).is_empty(), "an MTP layer adds one");
        assert_eq!(compat_warnings(&f, &facts(Some("llama"), Some(3), 0)), vec!["This matrix has data for 2 layers, but the model has 3. It was probably made from a different model; llama-quantize decides whether it can use it."]);
        let w = compat_warnings(&f, &facts(Some("qwen3"), Some(2), 0));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("architecture \"llama\"") && w[0].contains("\"qwen3\""));
        let dat = d.path().join("imatrix.dat");
        std::fs::write(&dat, b"\x01\0\0\0legacy").unwrap();
        let w = compat_warnings(&dat, &facts(Some("llama"), Some(2), 0));
        assert!(w[0].contains("imatrix.dat could not be read here") && w[0].contains("llama-quantize will tell"));
    }

    #[test]
    fn a_model_folder_tells_its_layers_and_head() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("config.json"), r#"{"text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 32, "mtp_num_hidden_layers": 1}}"#).unwrap();
        assert_eq!(folder_facts(d.path()), ModelFacts { arch: Some("qwen3_5_text".into()), layers: Some(32), mtp_layers: 1, ..Default::default() });
        assert_eq!(folder_facts(&d.path().join("none")), ModelFacts::default());
    }
}
