"""ggufrows.py: sample rows of a 16-bit GGUF and the small GGUF made of them. The writer in ggufmake.py is independent of the worker's code."""
import struct

import numpy as np
import pytest
from ggufmake import BF16, F16, F32, Q4_K, tiny_model, write

from ndforge_worker import ggufinfo, ggufrows


def full(info, name):
    return next(t for t in info["tensors"] if t["name"] == name)


def tensor_values(path, info, t):
    """all values of a 16-bit tensor, read here with numpy and the header offsets"""
    n = ggufinfo.numel(t)
    ty = {0: "<f4", 1: "<f2"}.get(t["type"])
    raw = np.fromfile(path, dtype="<u2" if t["type"] == 30 else ty, count=n, offset=info["data_start"] + t["offset"])
    a = (raw.astype(np.uint32) << 16).view(np.float32) if t["type"] == 30 else raw.astype(np.float32)
    return a.reshape(tuple(reversed(t["dims"])))


def test_ggufinfo_has_kv_end_and_alignment(tmp_path):
    p = tiny_model(tmp_path / "m.gguf")
    info = ggufinfo.read(p)
    assert info["alignment"] == 32 and 24 < info["kv_end"] < info["data_start"]
    raw = open(p, "rb").read()
    assert raw[:4] == b"GGUF" and struct.unpack("<Q", raw[16:24])[0] == 3      # n_kv


def test_open_source_accepts_16_bit(tmp_path):
    p = tiny_model(tmp_path / "m.gguf")
    info = ggufrows.open_source(p)
    assert info["path"] == str(p) and info["size"] == p.stat().st_size


def test_split_file_refused(tmp_path):
    p = write(tmp_path / "s.gguf", [("blk.0.attn_q.weight", [256, 4], F16, np.zeros((4, 256)))], [("split.count", "u32", 3), ("split.no", "u32", 0)])
    with pytest.raises(ggufrows.SourceError, match="llama-gguf-split --merge"):
        ggufrows.open_source(p)


def test_quantized_source_refused(tmp_path):
    p = write(tmp_path / "q.gguf", [("blk.0.attn_q.weight", [256, 4], Q4_K, bytes(144 * 4))], [("general.architecture", "str", "x")])
    with pytest.raises(ggufrows.SourceError, match="16-bit"):
        ggufrows.open_source(p)


def test_not_a_gguf_and_truncated(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"hello world")
    with pytest.raises(ggufrows.SourceError, match="not a readable GGUF"):
        ggufrows.open_source(tmp_path / "x.gguf")
    p = tiny_model(tmp_path / "m.gguf")
    data = p.read_bytes()
    (tmp_path / "cut.gguf").write_bytes(data[:len(data) - 2000])
    with pytest.raises(ggufrows.SourceError, match="cut short"):
        ggufrows.open_source(tmp_path / "cut.gguf")


def test_sample_plan_shapes_and_determinism(tmp_path):
    info = ggufrows.open_source(tiny_model(tmp_path / "m.gguf"))
    a = ggufrows.sample_plan(info, 32, 0)
    b = ggufrows.sample_plan(info, 32, 0)
    assert a == b and ggufrows.sample_plan(info, 32, 7) != a
    assert "blk.0.attn_norm.weight" not in a and "blk.0.ffn_gate_inp.weight" not in a      # not quantizable: copied whole
    q = a["blk.0.attn_q.weight"]
    assert len(q["rows"]) == 32 and q["rows"] == sorted(q["rows"]) and not q["exact"] and q["experts"] == 1
    # stratified: one row from each of 32 equal strata of the 300 rows
    for i, r in enumerate(q["rows"]):
        assert i * 300 // 32 <= r < (i + 1) * 300 // 32
    k = a["blk.0.attn_k.weight"]                    # 64 rows, n = 32: not exact
    assert len(k["rows"]) == 32
    e = a["blk.0.ffn_up_exps.weight"]               # 4 experts: k = ceil(32 / 4) = 8 rows, same in every expert
    assert len(e["rows"]) == 8 and e["experts"] == 4 and len(ggufrows.flat_rows(e)) == 32 and ggufrows.sample_dims(e) == [512, 8, 4]
    assert ggufrows.flat_rows(e)[8:16] == [96 + r for r in e["rows"]]
    whole = ggufrows.sample_plan(info, 1000)["blk.0.attn_q.weight"]
    assert whole["exact"] and whole["rows"] == list(range(300))
    assert len(ggufrows.sample_plan(info, 1)["blk.0.ffn_up_exps.weight"]["rows"]) == 1


def test_read_rows_equals_source_rows(tmp_path):
    p = tiny_model(tmp_path / "m.gguf")
    info = ggufrows.open_source(p)
    for name in ("blk.0.attn_q.weight", "blk.1.ffn_up_exps.weight", "token_embd.weight"):
        t = full(info, name)
        ref = tensor_values(p, info, t).reshape(-1, t["dims"][0])
        rows = [5, 0, 3, 4, ref.shape[0] - 1, 2]
        got = ggufrows.read_rows(info, t, rows)
        assert got.dtype == np.float32 and np.array_equal(got, ref[rows])      # BF16 exact, any order


def test_read_rows_f16_and_f32(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 64)).astype(np.float32)
    p = write(tmp_path / "f.gguf", [("blk.0.attn_q.weight", [64, 6], F16, x), ("blk.0.attn_k.weight", [64, 6], F32, x)])
    info = ggufrows.open_source(p)
    assert np.array_equal(ggufrows.read_rows(info, full(info, "blk.0.attn_k.weight"), [1, 4]), x[[1, 4]])
    assert np.array_equal(ggufrows.read_rows(info, full(info, "blk.0.attn_q.weight"), [1, 4]), x[[1, 4]].astype(np.float16).astype(np.float32))
    with pytest.raises(ValueError):
        ggufrows.read_rows(info, full(info, "blk.0.attn_q.weight"), [6])


def test_write_sample(tmp_path):
    p = tiny_model(tmp_path / "m.gguf", tied=False)
    info = ggufrows.open_source(p)
    picks = ggufrows.sample_plan(info, 16, 3)
    out = tmp_path / "work" / "sample.gguf"
    res = ggufrows.write_sample(info, out, picks)
    assert out.is_file() and not (tmp_path / "work" / "sample.gguf.partial").exists() and res["tensors"] == len(info["tensors"])
    s = ggufrows.open_source(out)
    assert [t["name"] for t in s["tensors"]] == [t["name"] for t in info["tensors"]]
    # the key-value section is the source's, byte for byte
    a, b = p.read_bytes(), out.read_bytes()
    assert a[24:info["kv_end"]] == b[24:s["kv_end"]] and a[:8] == b[:8] and a[16:24] == b[16:24]
    assert s["meta"]["general.architecture"] == "qwen35moe" and s["meta"]["tokenizer.ggml.tokens#len"] == 3
    for t in s["tensors"]:
        src_t = full(info, t["name"])
        assert t["type"] == src_t["type"] and t["offset"] % 32 == 0
        if t["name"] in picks:
            assert t["dims"] == ggufrows.sample_dims(picks[t["name"]])
            want = ggufrows.read_rows(info, src_t, ggufrows.flat_rows(picks[t["name"]]))
            assert np.array_equal(tensor_values(out, s, t).reshape(-1, t["dims"][0]), want)
        else:
            assert t["dims"] == src_t["dims"]
            assert np.array_equal(tensor_values(out, s, t), tensor_values(p, info, src_t))
    assert res["sampled_weights"] == sum(len(ggufrows.flat_rows(pk)) * pk["cols"] for pk in picks.values())
    assert res["bytes"] < p.stat().st_size


def test_files_can_be_deleted_right_after_use(tmp_path):
    """no open handles are kept (on Windows an open or mapped file cannot be deleted)"""
    p = tiny_model(tmp_path / "m.gguf")
    info = ggufrows.open_source(p)
    ggufrows.read_rows(info, full(info, "blk.0.attn_q.weight"), [0, 1])
    out = tmp_path / "s.gguf"
    ggufrows.write_sample(info, out, ggufrows.sample_plan(info, 8))
    ggufrows.open_source(out)
    out.unlink()
    p.unlink()
    assert not p.exists() and not out.exists()


def test_failed_write_leaves_no_partial(tmp_path, monkeypatch):
    p = tiny_model(tmp_path / "m.gguf")
    info = ggufrows.open_source(p)
    monkeypatch.setattr(ggufrows, "_read_raw", lambda *a, **k: (_ for _ in ()).throw(ggufrows.SourceError("boom")))
    with pytest.raises(ggufrows.SourceError):
        ggufrows.write_sample(info, tmp_path / "s.gguf", ggufrows.sample_plan(info, 8))
    assert not list(tmp_path.glob("s.gguf*"))
