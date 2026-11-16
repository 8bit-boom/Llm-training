"""The importance-matrix readers against the REAL file layout (docs/IMATRIX.md, "Verified on the real published file"): a scaled-down fixture with exactly that layout (committed
generator: imatrix_fixture.py), the committed header dump of the real file, and, when NDFORGE_REAL_IMATRIX points at the real `imatrix_unsloth.gguf`, the real file itself
(skipped otherwise: it is 13.6 MB and not part of the repository)."""
import os
import struct
from pathlib import Path

import pytest
from imatrix_fixture import ALL_KINDS, DUMP, FULL_KINDS, LINEAR_KINDS, REAL, parse_dump, structure_of, value, write_imatrix

from ndforge_worker import ggufinfo, imatrix, mixedquant
from ndforge_worker import imatrixmerge as merge_mod

REAL_FILE = os.environ.get("NDFORGE_REAL_IMATRIX")


# ------------------------------------------------------------------------------------------------------------------ the fixture is the real layout

def test_the_dump_is_the_committed_evidence_of_the_real_file():
    d = parse_dump()
    assert len(DUMP.read_text(encoding="utf-8").splitlines()) == 31
    assert (d["version"], d["tensors"], d["kv_count"], d["layers"]) == (3, 992, 4, 64)
    assert d["kv"]["general.type"] == "imatrix" and d["kv"]["imatrix.chunk_count"] == "1251" and d["kv"]["imatrix.chunk_size"] == "8192"
    assert "general.architecture" not in d["kv"]
    assert sum(c for (kind, suffix), (c, _, _) in d["kinds"].items() if suffix == "in_sum2") == 496 and len(d["kinds"]) == 24
    assert all(typ == 0 for _, _, typ in d["kinds"].values()), "every tensor is float32"


def test_the_generator_reproduces_the_structure_of_the_dump_for_64_layers(tmp_path):
    f = tmp_path / "m.gguf"
    write_imatrix(f, layers=64)
    dump, mine = parse_dump(), structure_of(f)
    assert {k: v[0] for k, v in mine.items()} == {k: v[0] for k, v in dump["kinds"].items()}, "kinds and how many layers have each"
    assert all(v[2] == 0 for v in mine.values())
    info = ggufinfo.read(f)
    assert (info["version"], len(info["tensors"]), len(info["meta"].keys() - {"imatrix.datasets#len", "imatrix.datasets#items"})) == (3, dump["tensors"], dump["kv_count"])
    assert "general.architecture" not in info["meta"] and "general.alignment" not in info["meta"]
    assert imatrix.matrix_summary(f)["layers"] == dump["layers"]
    # the widths scale, the shape of the counts does not: ONE value, ONE dimension
    assert {k[0]: v[1] for k, v in mine.items() if k[1] == "counts"} == {k[0]: 1 for k in mine if k[1] == "counts"}
    assert {k[0]: v[1] for k, v in dump["kinds"].items() if k[1] == "counts"} == {k[0]: 1 for k in dump["kinds"] if k[1] == "counts"}


def test_the_file_layout_is_the_real_one_one_dimension_alignment_32_pairs_sorted_by_name(tmp_path):
    f = tmp_path / "m.gguf"
    out = write_imatrix(f, layers=12)
    info = ggufinfo.read(f)
    assert all(len(t["dims"]) == 1 and t["type"] == 0 for t in info["tensors"])
    assert info["data_start"] % 32 == 0 and all(t["offset"] % 32 == 0 for t in info["tensors"])
    assert f.stat().st_size % 32 == 0
    names = [t["name"] for t in info["tensors"]]
    assert names == out["names"]
    assert all(n.endswith(".in_sum2") for n in names[0::2]) and all(n.endswith(".counts") for n in names[1::2])
    bases = [n.rsplit(".", 1)[0] for n in names[0::2]]
    assert bases == sorted(bases) and bases.index("blk.10.attn_gate.weight") < bases.index("blk.2.attn_gate.weight"), "sorted as text, like the real file"
    assert info["meta"]["imatrix.datasets#items"] == ["unsloth_calibration_dataset"]
    # linear layers and full-attention layers, interval 4 (3, 7, 11)
    full = {int(n.split(".")[1]) for n in names if ".attn_q." in n}
    assert full == {3, 7, 11}
    for L in range(12):
        have = {n.split(".")[2] for n in names if n.startswith(f"blk.{L}.") and n.endswith("in_sum2")}
        assert have == set((FULL_KINDS if L in full else LINEAR_KINDS) + ALL_KINDS)


# ------------------------------------------------------------------------------------------------------------------ the readers

def test_the_header_reader_reads_the_real_layout_without_an_architecture(tmp_path):
    f = tmp_path / "m.gguf"
    write_imatrix(f, layers=8)
    meta, names = imatrix.read_gguf_header(f)
    assert meta == {"general.type": "imatrix", "imatrix.datasets": None, "imatrix.chunk_count": 1251, "imatrix.chunk_size": 8192}
    assert len(names) == 2 * (6 * 5 + 2 * 4 + 8 * 3), "6 linear and 2 full-attention layers, every layer with ffn_down / ffn_gate / ffn_up, in_sum2 and counts of each"
    assert imatrix.gguf_has_mtp(f) is False


def test_the_summary_tells_what_the_real_matrix_says_about_itself(tmp_path):
    f = tmp_path / "m.gguf"
    write_imatrix(f, layers=8, hidden=16, inter=24)
    s = imatrix.matrix_summary(f)
    assert s["arch"] is None and s["layers"] == 8 and s["chunks"] == 1251 and s["chunk_size"] == 8192
    assert s["entries"] == 6 * 5 + 2 * 4 + 8 * 3 == sum(s["kinds"].values())
    assert s["kinds"]["ffn_down"] == 8 and s["kinds"]["attn_qkv"] == 6 and s["kinds"]["attn_q"] == 2 and s["kinds"]["ssm_out"] == 6
    assert (s["hidden"], s["intermediate"], s["tokens"]) == (16, 24, REAL["tokens"])
    assert s["datasets"] == ["unsloth_calibration_dataset"]


def test_without_an_architecture_the_model_is_compared_by_shapes(tmp_path):
    f = tmp_path / "m.gguf"
    write_imatrix(f, layers=8, hidden=16, inter=24)
    assert imatrix.compat_warnings(f, "qwen35", 8, 0, (16, 24)) == [], "no architecture key: nothing to say about it"
    w = imatrix.compat_warnings(f, "qwen35", 4, 0, (16, 24))
    assert len(w) == 1 and "data for 8 layers, but the model has 4" in w[0]
    w = imatrix.compat_warnings(f, None, 8, 0, (32, 24))
    assert len(w) == 1 and "hidden size 16" in w[0] and "32" in w[0]
    w = imatrix.compat_warnings(f, None, 8, 0, (16, 12))
    assert len(w) == 1 and "ffn_down is 24 wide, in the model 12" in w[0] and "Adapt to my cut model" in w[0]
    assert imatrix.compat_warnings(f, None, None, 0, (None, None)) == []


def test_a_model_folder_tells_its_sizes(tmp_path):
    (tmp_path / "config.json").write_text('{"text_config": {"hidden_size": 5120, "intermediate_size": 17408}}', encoding="utf-8")
    assert imatrix.folder_model_sizes(tmp_path) == (5120, 17408)
    assert imatrix.folder_model_sizes(tmp_path / "none") == (None, None)


def test_the_importance_of_the_real_layout_is_in_sum2_over_counts_per_weight(tmp_path):
    f = tmp_path / "m.gguf"
    write_imatrix(f, layers=4, hidden=8, inter=12, attn_out=10)
    got = mixedquant.read_importance(str(f))
    assert len(got) == 3 * 5 + 1 * 4 + 4 * 3 and "blk.0.ffn_down.weight" in got and "blk.3.attn_q.weight" in got
    for L, kind, n in ((0, "ffn_down", 12), (3, "attn_output", 10), (1, "ssm_beta", 8)):
        want = sum(value(L, kind, i) for i in range(n)) / n / REAL["tokens"]
        assert got[f"blk.{L}.{kind}.weight"] == pytest.approx(want, rel=1e-6)


def test_the_merge_checks_read_the_real_layout(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    write_imatrix(a, layers=8)
    write_imatrix(b, layers=4)
    assert merge_mod._tensor_set(a) >= merge_mod._tensor_set(b) and "blk.7.ffn_up.weight" in merge_mod._tensor_set(a)
    w = merge_mod.compat([{"label": "A", "file": a}, {"label": "B", "file": b}])
    assert any("8 layers" in x and "4" in x for x in w)
    rows, note = merge_mod.weights([{"label": "A", "chunks": 1251}, {"label": "B", "chunks": 128}])
    assert rows[0]["chunks"] == 1251 and "carries about 91 %" in note


# ------------------------------------------------------------------------------------------------------------------ the real file (skipped unless NDFORGE_REAL_IMATRIX is set)

@pytest.mark.skipif(not REAL_FILE or not Path(REAL_FILE or "x").is_file(), reason="NDFORGE_REAL_IMATRIX does not point at the real imatrix_unsloth.gguf")
def test_every_reader_agrees_with_the_real_published_file():
    f = Path(REAL_FILE)
    dump = parse_dump()
    info = ggufinfo.read(f)
    assert (info["version"], len(info["tensors"])) == (3, 992) and f.stat().st_size == 13642656
    assert all(len(t["dims"]) == 1 and t["type"] == 0 for t in info["tensors"]) and "general.architecture" not in info["meta"]
    assert info["data_start"] % 32 == 0 and all(t["offset"] % 32 == 0 for t in info["tensors"])
    assert {k: v for k, v in structure_of(f).items()} == {k: v for k, v in dump["kinds"].items()}
    meta, names = imatrix.read_gguf_header(f)
    assert len(names) == 992 and meta["imatrix.chunk_count"] == 1251 and meta["imatrix.chunk_size"] == 8192 and meta["general.type"] == "imatrix"
    s = imatrix.matrix_summary(f)
    assert (s["arch"], s["layers"], s["entries"], s["chunks"], s["chunk_size"]) == (None, 64, 496, 1251, 8192)
    assert (s["hidden"], s["intermediate"], s["tokens"]) == (5120, 17408, 1149614.0) and s["datasets"] == ["unsloth_calibration_dataset"]
    assert s["kinds"] == {"attn_gate": 48, "attn_k": 16, "attn_output": 16, "attn_q": 16, "attn_qkv": 48, "attn_v": 16, "ffn_down": 64, "ffn_gate": 64, "ffn_up": 64,
                          "ssm_alpha": 48, "ssm_beta": 48, "ssm_out": 48}
    assert imatrix.compat_warnings(f, "qwen3_5_text", 64, 0, (5120, 17408)) == []
    assert len(imatrix.compat_warnings(f, "qwen3_5_text", 32, 0, (5120, 10752))) == 2
    imp = mixedquant.read_importance(str(f))
    assert len(imp) == 496 and all(v > 0 for v in imp.values()) and imp["blk.63.ffn_down.weight"] > imp["blk.0.ffn_down.weight"] * 1000
    assert imatrix.gguf_has_mtp(f) is False
    with open(f, "rb") as fh:
        assert fh.read(8) == b"GGUF" + struct.pack("<I", 3)
