"""Adapting an importance matrix to a model cut on the Slim page (imatrixadapt.py, docs/IMATRIX.md "Adapting a matrix to a cut model"): depth (layers dropped and renumbered), width
(ffn_down sliced with the kept neurons), the weight factor, every refusal, the real layout written and read back, the job (library / file / published source, progress, Stop, the
library entry), the merge with weights and the safety rule of the published matrix. Scaled-down fixture of the real layout (imatrix_fixture.py), cut models written by the Slim
job's own writer functions, a stand-in llama-imatrix; no torch, no llama.cpp, no network."""
import json
import os
import struct
import sys
from pathlib import Path

import pytest
from imatrix_fixture import REAL, kinds_of, value, write_imatrix

np = pytest.importorskip("numpy")

from ndforge_worker import imatrix, imatrixadapt as ad  # noqa: E402
from ndforge_worker import imatrixjob, imatrixmerge, protocol, reshapejob  # noqa: E402

LAYERS, HIDDEN, INTER, ATTN = 16, 8, 24, 10
KEEP = [0, 1, 2, 3, 12, 13, 14, 15]          # blocks of 4: layers 4..11 removed (two blocks)


def matrix(path, layers=LAYERS, inter=INTER, **kw):
    out = write_imatrix(path, layers=layers, hidden=HIDDEN, inter=inter, attn_out=ATTN, **kw)
    return str(path), out


def make_cut(root, name="slim", src=KEEP, inter=INTER, new_inter=None, origin_layers=LAYERS, repo="Qwen/Qwen3.8-27B", mutate=None, kept_override=None, npz=True, record=True, heads=False):
    """models/<name>/model with ndforge_reshape.json and width_kept.npz written by the Slim job's own functions (reshapejob._save_kept) and the same keys as its meta."""
    d = Path(root) / "models" / name / "model"
    d.mkdir(parents=True)
    keep = {"mlp": {}, "heads": {}, "gdn": {}}
    kept = {}
    rng = np.random.RandomState(5)
    if new_inter is not None:
        for j, s in enumerate(src):
            idx = sorted(rng.choice(inter, new_inter, replace=False).tolist())
            keep["mlp"][s], kept[j] = idx, idx
        reshapejob._save_kept(d / "width_kept.npz", list(src), keep, None)
    dropped = [i for i in range(origin_layers) if i not in src]
    meta = {"format": 2 if new_inter is not None else 1, "tool": "ND Lore Forge", "created": "2026-11-16T10:00:00Z", "origin": {"repo": repo, "family": "qwen3_5", "layers": origin_layers, "params": 1},
            "mode": "prune", "period": 4, "layers": len(src), "params": 1, "src_layers": list(src), "dropped": dropped, "duplicated": [], "zero_init": [], "strategy": "contiguous",
            "loss_before": 1.0, "loss_after": 1.0, "stable": True}
    if new_inter is not None:
        meta["shape"] = "both" if dropped else "width"
        meta["width"] = {"intermediate_size": [inter, new_inter], "linear_num_value_heads": [4, 4], "num_attention_heads": [6, 6], "kept": "width_kept.npz", "scores_key": "k1", "stable_overlap": 0.9,
                         "mtp": "none"}
    if heads:
        meta["width"]["num_attention_heads"] = [6, 4]
    if mutate:
        mutate(meta)
    if record:
        (d / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d.parent, kept, meta


def names_of(out: dict) -> list:
    return out["names"]


def adapt_to(tmp_path, src_file, cut_dir, weight=1.0, name="slim", kept_layers=None):
    cut = ad.read_cut(cut_dir, name, kept_layers)
    m = ad.read_matrix(src_file)
    return ad.adapt(m, cut, weight) + (cut, m)


def refused(fn, *a, **k):
    with pytest.raises(ad.AdaptRefused) as e:
        fn(*a, **k)
    return e.value


# ------------------------------------------------------------------------------------------------------------------ the writer is the real layout

def test_the_worker_writes_bytes_identical_to_the_independent_fixture_generator(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf")
    m = ad.read_matrix(f)
    ad.write_matrix(tmp_path / "copy.gguf", m)
    assert (tmp_path / "copy.gguf").read_bytes() == Path(f).read_bytes(), "read then write is the identity on the real layout"
    assert ad.layout(m.kvs, [(n, m.lengths[n]) for n in m.names])[2] == Path(f).stat().st_size


def test_an_identity_cut_changes_only_the_dataset_note_and_matches_the_generator_byte_for_byte(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf")
    cut_dir, _, _ = make_cut(tmp_path, src=list(range(LAYERS)))
    out, p = adapt_to(tmp_path, f, cut_dir)[:2]
    ad.write_matrix(tmp_path / "out.gguf", out)
    write_imatrix(tmp_path / "gen.gguf", layers=LAYERS, hidden=HIDDEN, inter=INTER, attn_out=ATTN, datasets=("unsloth_calibration_dataset", "(adapted to slim)"))
    assert (tmp_path / "out.gguf").read_bytes() == (tmp_path / "gen.gguf").read_bytes()


def test_a_scaled_copy_equals_the_generator_with_scaled_sums_and_counts(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf", layers=4)
    ad.scale_file(f, tmp_path / "w.gguf", 4.0)
    write_imatrix(tmp_path / "gen.gguf", layers=4, hidden=HIDDEN, inter=INTER, attn_out=ATTN, tokens=REAL["tokens"] * 4, scale=4.0, datasets=("unsloth_calibration_dataset", "(weighted x4)"))
    assert (tmp_path / "w.gguf").read_bytes() == (tmp_path / "gen.gguf").read_bytes()


def test_the_reader_refuses_what_is_not_the_real_layout(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf", layers=4)
    raw = Path(f).read_bytes()
    for name, blob in (("short", raw[: len(raw) - 40]), ("magic", b"NOPE" + raw[4:]), ("empty", b""), ("old", raw[:4] + struct.pack("<I", 1) + raw[8:])):
        (tmp_path / name).write_bytes(blob)
        with pytest.raises(ValueError):
            ad.read_matrix(tmp_path / name)
    # a tensor with two dimensions (the earlier guess of the layout) and one that is not float32
    two = raw.replace(struct.pack("<I", 1) + struct.pack("<Q", HIDDEN), struct.pack("<I", 2) + struct.pack("<QQ", HIDDEN, 1), 1)
    (tmp_path / "two").write_bytes(two)
    with pytest.raises(ValueError, match="one"):
        ad.read_matrix(tmp_path / "two")
    e = refused(ad.adapt_file, tmp_path / "short", tmp_path / "x.gguf", ad.read_cut(make_cut(tmp_path, src=list(range(LAYERS)))[0]))
    assert e.reason == "bad_matrix" and "could not be read as an importance matrix" in str(e) and e.code == "adapt_bad_matrix"


# ------------------------------------------------------------------------------------------------------------------ depth

def test_depth_keeps_the_kept_layers_renamed_and_drops_the_rest(tmp_path):
    f, src = matrix(tmp_path / "src.gguf")
    cut_dir, _, _ = make_cut(tmp_path)
    out, p, cut, m = adapt_to(tmp_path, f, cut_dir)
    layers = {int(n.split(".")[1]) for n in out.names}
    assert layers == set(range(8)) and p["layers_before"] == 16 and p["layers_after"] == 8 and p["kept"] == KEEP and cut.dropped == [4, 5, 6, 7, 8, 9, 10, 11]
    kinds = {}
    for n in out.names:
        if n.endswith(".in_sum2"):
            kinds[n.split(".")[2]] = kinds.get(n.split(".")[2], 0) + 1
    # old 3 and 15 are full-attention layers: new 3 and 7; the others are linear
    assert kinds == {"attn_q": 2, "attn_k": 2, "attn_v": 2, "attn_output": 2, "attn_gate": 6, "attn_qkv": 6, "ssm_alpha": 6, "ssm_beta": 6, "ssm_out": 6, "ffn_down": 8, "ffn_gate": 8, "ffn_up": 8}
    assert len(out.names) == 2 * (6 * 5 + 2 * 4 + 8 * 3)
    for new_l, old_l in enumerate(KEEP):
        for kind in kinds_of(old_l):
            assert np.array_equal(out.arrays[f"blk.{new_l}.{kind}.weight.in_sum2"], np.array(src["values"][f"blk.{old_l}.{kind}.weight.in_sum2"], dtype=np.float32))
            assert out.arrays[f"blk.{new_l}.{kind}.weight.counts"].tolist() == [REAL["tokens"]]
    assert out.arrays["blk.7.attn_q.weight.in_sum2"][3] == value(15, "attn_q", 3) and "blk.8.attn_q.weight.in_sum2" not in out.arrays
    assert out.kv("imatrix.chunk_count") == 1251 and out.kv("imatrix.chunk_size") == 8192 and out.kv("general.type") == "imatrix"
    assert out.datasets() == ["unsloth_calibration_dataset", "(adapted to slim)"]
    assert [k for k, _, _ in out.kvs] == [k for k, _, _ in m.kvs], "the same keys in the same order"


def test_the_order_of_the_tensors_is_the_sources_order_with_the_kept_ones(tmp_path):
    f, src = matrix(tmp_path / "src.gguf")
    cut_dir, _, _ = make_cut(tmp_path)
    out, p, cut, m = adapt_to(tmp_path, f, cut_dir)
    want = []
    for n in src["names"]:
        old = int(n.split(".")[1])
        if old in KEEP:
            want.append(f"blk.{KEEP.index(old)}." + n.split(".", 2)[2])
    assert out.names == want


# ------------------------------------------------------------------------------------------------------------------ width

def test_width_slices_ffn_down_with_the_kept_neurons_and_nothing_else(tmp_path):
    f, src = matrix(tmp_path / "src.gguf")
    cut_dir, kept, _ = make_cut(tmp_path, src=list(range(LAYERS)), new_inter=10)
    out, p, cut, m = adapt_to(tmp_path, f, cut_dir)
    assert p["ffn_down_before"] == 24 and p["ffn_down_after"] == 10 and cut.thinner
    for j in range(LAYERS):
        a = out.arrays[f"blk.{j}.ffn_down.weight.in_sum2"]
        assert len(a) == 10 and a.tolist() == [value(j, "ffn_down", i) for i in kept[j]], "the values at the kept indices, in the order recorded"
        assert out.arrays[f"blk.{j}.ffn_down.weight.counts"].tolist() == [REAL["tokens"]]
        for kind in ("ffn_gate", "ffn_up"):
            assert len(out.arrays[f"blk.{j}.{kind}.weight.in_sum2"]) == HIDDEN
    for n in out.names:
        if ".ffn_down." not in n:
            assert np.array_equal(out.arrays[n], m.arrays[n]), f"{n} is copied"


def test_depth_and_width_together(tmp_path):
    f, src = matrix(tmp_path / "src.gguf")
    cut_dir, kept, _ = make_cut(tmp_path, new_inter=10)
    out, p, cut, m = adapt_to(tmp_path, f, cut_dir)
    assert p["layers_after"] == 8 and p["ffn_down_after"] == 10
    for new_l, old_l in enumerate(KEEP):
        assert out.arrays[f"blk.{new_l}.ffn_down.weight.in_sum2"].tolist() == [value(old_l, "ffn_down", i) for i in kept[new_l]], "the neurons of the NEW layer, numbered as in the original"
    assert out.arrays["blk.4.attn_gate.weight.in_sum2"].tolist() == [value(12, "attn_gate", i) for i in range(HIDDEN)]
    # the size predicted from the header alone is the size of the file
    ad.write_matrix(tmp_path / "o.gguf", out)
    nums = ad.preview_numbers(p, out.kvs)
    assert nums["size"] == (tmp_path / "o.gguf").stat().st_size and nums["tensors_after"] == len(out.names) and nums["entries_after"] == len(out.names) // 2


def test_a_model_made_by_the_real_slim_job_is_adapted_with_its_own_files(tmp_path):
    """the record and the kept neurons written by reshapejob.run itself (a tiny model), then the matrix adapted to that folder"""
    from test_reshape_width import fake_qwen_width, run_job, write_scores, wjob
    src = fake_qwen_width(tmp_path, layers=16)
    write_scores(tmp_path / "models" / "_scores" / "k1", str(src), layers=16)
    j = wjob(tmp_path, src, {"intermediate_size": 16}, blocks=1)
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    kept = np.load(Path(j["final"]) / "width_kept.npz")
    assert kept["src"].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
    f, _ = matrix(tmp_path / "src.gguf", layers=16, inter=32)
    out, p, cut, m = adapt_to(tmp_path, f, Path(j["final"]).parent, name="m1")
    assert (cut.src, cut.inter_before, cut.inter_after, cut.origin_layers) == (kept["src"].tolist(), 32, 16, 16) and p["layers_after"] == 12
    for new_l, old_l in enumerate(cut.src):
        assert out.arrays[f"blk.{new_l}.ffn_down.weight.in_sum2"].tolist() == [value(old_l, "ffn_down", int(i)) for i in kept[f"mlp_{new_l}"]]
    assert len(out.names) == 2 * sum(len(kinds_of(L)) for L in cut.src)


# ------------------------------------------------------------------------------------------------------------------ the weight

def test_the_weight_multiplies_sums_and_counts_and_keeps_the_mean_square(tmp_path):
    f, src = matrix(tmp_path / "src.gguf")
    cut_dir, kept, _ = make_cut(tmp_path, new_inter=10)
    one, _, _, m = adapt_to(tmp_path, f, cut_dir)
    w, p, _, _ = adapt_to(tmp_path, f, cut_dir, weight=2.5)
    for n in one.names:
        if n.endswith(".in_sum2"):
            c = n[: -len("in_sum2")] + "counts"
            assert np.allclose(w.arrays[n], one.arrays[n].astype(np.float64) * 2.5, rtol=1e-6)
            assert w.arrays[c].tolist() == [REAL["tokens"] * 2.5]
            assert np.allclose(w.arrays[n] / w.arrays[c][0], one.arrays[n] / one.arrays[c][0], rtol=1e-6), "the mean square is unchanged"
    out, _ = ad.adapt(m, ad.read_cut(cut_dir, "slim"), 1)
    assert all(np.array_equal(out.arrays[n], one.arrays[n]) for n in one.names), "weight 1 is bit-identical"
    for bad in (0, -1, float("nan"), float("inf"), 1e9, None, True, "2"):
        e = refused(ad.adapt, m, ad.read_cut(cut_dir, "slim"), bad)
        assert e.reason == "bad_weight" and "above 0" in str(e)


# ------------------------------------------------------------------------------------------------------------------ refusals

def test_a_matrix_of_the_wrong_layer_count_is_refused(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf", layers=12)
    cut_dir, _, _ = make_cut(tmp_path)
    e = refused(adapt_to, tmp_path, f, cut_dir)
    assert e.reason == "wrong_original" and "data for 12 layers" in str(e) and "has 16" in str(e) and "Qwen/Qwen3.8-27B" in str(e)
    f2, _ = matrix(tmp_path / "src2.gguf", layers=17)       # one layer more than the original: the extra one is a speculative-decoding head, left out; ok
    out = adapt_to(tmp_path, f2, cut_dir)[0]
    assert len(out.names) == 2 * (6 * 5 + 2 * 4 + 8 * 3)
    assert adapt_to(tmp_path, f2, cut_dir)[1]["dropped_mtp"] == [16]


def test_a_matrix_of_the_wrong_mlp_width_is_refused(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf", inter=20)
    cut_dir, _, _ = make_cut(tmp_path, new_inter=10)
    e = refused(adapt_to, tmp_path, f, cut_dir)
    assert e.reason == "wrong_original" and "is 20 wide" in str(e) and "has 24" in str(e)


def test_the_head_axes_and_anything_else_that_was_cut_are_refused(tmp_path):
    cut_dir, _, _ = make_cut(tmp_path, new_inter=10, heads=True)
    e = refused(ad.read_cut, cut_dir, "slim")
    assert e.reason == "heads" and "attention heads were cut from 6 to 4" in str(e) and "experimental" in str(e)
    cut2, _, _ = make_cut(tmp_path, name="s2", new_inter=10, mutate=lambda m: m["width"].update(linear_num_value_heads=[4, 2]))
    assert "linear-attention value heads were cut from 4 to 2" in str(refused(ad.read_cut, cut2, "s2"))


def test_a_model_without_a_record_or_not_from_the_slim_page_is_refused(tmp_path):
    plain_dir = tmp_path / "other" / "model"
    plain_dir.mkdir(parents=True)
    e = refused(ad.read_cut, plain_dir.parent)
    assert e.reason == "no_metadata" and "no ndforge_reshape.json" in str(e) and "Create a matrix" in str(e)
    d, _, _ = make_cut(tmp_path, name="broken")
    (d / "model" / "ndforge_reshape.json").write_text("{nope", encoding="utf-8")
    assert refused(ad.read_cut, d).reason == "no_metadata"
    d2, _, _ = make_cut(tmp_path, name="healed", mutate=lambda m: m.update(origin={}, mode="heal"))
    e = refused(ad.read_cut, d2)
    assert e.reason == "no_metadata" and "how many layers its original had" in str(e)


def test_a_grown_model_is_refused(tmp_path):
    d, _, _ = make_cut(tmp_path, mutate=lambda m: m.update(duplicated=[3, 3], mode="grow", zero_init=[4]))
    e = refused(ad.read_cut, d, "grown")
    assert e.reason == "grown" and "layers that were added" in str(e)


def test_a_record_that_has_only_counts_asks_for_the_kept_layers_and_accepts_them(tmp_path):
    def counts_only(m):
        for k in ("src_layers", "dropped"):
            m.pop(k)
    d, _, _ = make_cut(tmp_path, mutate=counts_only)
    e = refused(ad.read_cut, d, "old")
    assert e.reason == "needs_layer_map" and e.code == "adapt_needs_layer_map" and "8 layers of the original 16" in str(e) and "counted from 0" in str(e)
    cut = ad.read_cut(d, "old", KEEP)
    assert cut.src == KEEP and not cut.thinner
    for bad in ([0, 1, 2, 3, 4, 5, 6], [3, 2, 1, 0, 12, 13, 14, 15], [0, 1, 2, 3, 12, 13, 14, 99], [0, 0, 1, 2, 3, 12, 13, 14]):
        assert refused(ad.read_cut, d, "old", bad).reason == "bad_kept"
    f, _ = matrix(tmp_path / "src.gguf")
    out, p = ad.adapt(ad.read_matrix(f), cut)
    assert p["layers_after"] == 8


def test_a_record_that_lists_only_the_dropped_layers_is_enough(tmp_path):
    d, _, _ = make_cut(tmp_path, mutate=lambda m: m.pop("src_layers"))
    assert ad.read_cut(d, "old").src == KEEP


def test_damaged_width_files_are_refused(tmp_path):
    d, _, _ = make_cut(tmp_path, new_inter=10)
    (d / "model" / "width_kept.npz").unlink()
    assert refused(ad.read_cut, d, "x").reason == "bad_kept"
    d2, _, _ = make_cut(tmp_path, name="s2", new_inter=10, mutate=lambda m: m["width"].update(intermediate_size=[24, 11]))
    assert "neurons kept in layer 0" in str(refused(ad.read_cut, d2, "s2"))
    d3, _, _ = make_cut(tmp_path, name="s3", new_inter=10)
    (d3 / "model" / "width_kept.npz").write_bytes(b"not an npz")
    assert "could not be read" in str(refused(ad.read_cut, d3, "s3"))
    d4, _, _ = make_cut(tmp_path, name="s4", new_inter=10, mutate=lambda m: m.update(src_layers=[0, 1, 2, 3, 4, 13, 14, 15]))
    assert "belongs to other layers" in str(refused(ad.read_cut, d4, "s4"))


def test_an_already_adapted_matrix_is_refused(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf", datasets=("unsloth_calibration_dataset", "(adapted to other)"))
    cut_dir, _, _ = make_cut(tmp_path)
    e = refused(adapt_to, tmp_path, f, cut_dir)
    assert e.reason == "already_adapted" and "already adapted" in str(e)
    f2, _ = matrix(tmp_path / "ok.gguf")
    out, p, cut, m = adapt_to(tmp_path, f2, cut_dir)
    ad.write_matrix(tmp_path / "twice.gguf", out)
    assert refused(ad.adapt, ad.read_matrix(tmp_path / "twice.gguf"), cut).reason == "already_adapted", "its own output is recognised"


def test_incomplete_or_foreign_matrices_are_refused(tmp_path):
    cut_dir, _, _ = make_cut(tmp_path)
    p = ad.plan
    ent = [(f"blk.{L}.{k}.weight.{s}", n) for L in range(LAYERS) for k, n in (("ffn_down", INTER), ("ffn_gate", HIDDEN), ("ffn_up", HIDDEN)) for s, n in (("in_sum2", n), ("counts", 1))]
    cut = ad.read_cut(cut_dir, "slim")
    assert len(p(ent, [], cut)["entries"]) == 2 * 3 * 8
    assert refused(p, [e for e in ent if e[0] != "blk.5.ffn_up.weight.counts"], [], cut).reason == "bad_matrix"
    assert "no ffn_up statistics" in str(refused(p, [e for e in ent if not e[0].startswith("blk.3.ffn_up")], [], cut))
    assert refused(p, [("output.weight.in_sum2", 4), ("output.weight.counts", 1)], [], cut).reason == "bad_matrix"
    assert refused(p, [], [], cut).reason == "bad_matrix"


def test_entries_outside_the_blocks_are_copied(tmp_path):
    f, _ = matrix(tmp_path / "src.gguf", layers=4)
    m = ad.read_matrix(f)
    m.names += ["output.weight.in_sum2", "output.weight.counts"]
    m.arrays["output.weight.in_sum2"], m.arrays["output.weight.counts"] = np.arange(6, dtype=np.float32), np.array([5.0], dtype=np.float32)
    m.lengths = {n: len(a) for n, a in m.arrays.items()}
    d, _, _ = make_cut(tmp_path, src=[0, 3], origin_layers=4)
    out, p = ad.adapt(m, ad.read_cut(d, "s"), 2.0)
    assert out.arrays["output.weight.in_sum2"].tolist() == [2.0 * i for i in range(6)] and out.arrays["output.weight.counts"].tolist() == [10.0]


# ------------------------------------------------------------------------------------------------------------------ the job

IMATRIX = '''import json, os, sys
a = sys.argv[1:]
if a == ["--version"]:
    print("version: 9999 (stub1234)")
    sys.exit(0)
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "imatrix", "args": a}) + "\\n")
mode = os.environ.get("ADAPT_MODE", "")
if mode == "refuse":
    print("error: failed to load imatrix")
    sys.exit(1)
if mode == "old" and "--output-format" in a:
    print("error: invalid argument: --output-format")
    sys.exit(1)
ins = [a[i + 1] for i, x in enumerate(a) if x == "--in-file"]
open(a[a.index("-o") + 1], "wb").write(open(ins[0], "rb").read())
'''


@pytest.fixture
def jb(tmp_path, monkeypatch):
    from test_imatrix_merge import Hub
    hub = Hub()
    monkeypatch.setenv("HF_ENDPOINT", hub.url)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    tool = tmp_path / "llama-imatrix.py"
    tool.write_text(IMATRIX, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STUB_LOG", str(log))
    monkeypatch.delenv("ADAPT_MODE", raising=False)
    monkeypatch.setattr(ad, "find_tools", lambda need_convert: (None, tool))
    monkeypatch.setattr(imatrixmerge, "find_tools", lambda need_convert: (None, tool))
    monkeypatch.setattr(imatrixjob, "free_bytes", lambda p: 10 ** 12)
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))

    class W:
        pass

    w = W()
    w.root, w.events, w.tool, w.hub = tmp_path, events, tool, hub
    w.lib, w.cache = tmp_path / "imatrix", tmp_path / "imatrix_downloads"
    w.src, w.vals = matrix(tmp_path / "pub.gguf")
    w.cut, w.kept, _ = make_cut(tmp_path, new_inter=10)
    hub.repos["unsloth/Qwen3.8-27B-GGUF"] = {"files": {"imatrix_unsloth.gguf": Path(w.src).read_bytes()}}
    w.calls = lambda: [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    def job(**over):
        j = {"job": "imatrix_adapt", "root": str(w.lib), "id": "20261116-1000-slim", "label": "adapted", "stop_file": str(tmp_path / "STOP"), "cache": str(w.cache), "model_dir": str(w.cut),
             "model_name": "slim", "weight": 1.0, "notes": [], "source": {"kind": "file", "file": w.src}}
        j.update(over)
        return j

    w.job = job
    w.run = lambda **kw: ad.adapt_job(job(**kw))
    w.folder = w.lib / "20261116-1000-slim"
    yield w
    hub.close()


def test_a_file_is_adapted_into_a_library_entry_that_reads_back_as_predicted(jb):
    r = jb.run()
    out = jb.folder / "imatrix.gguf"
    assert sorted(p.name for p in jb.folder.iterdir()) == ["imatrix.gguf", "imatrix.json"] and Path(r["file"]) == out
    info = json.loads((jb.folder / "imatrix.json").read_text(encoding="utf-8"))
    assert (info["source_kind"], info["via"], info["label"], info["id"], info["model"], info["format"], info["has_dat"], info["weight"]) == ("adapted", "adapted", "adapted", "20261116-1000-slim", "slim", 1, False, 1.0)
    assert info["layers"] == 8 and info["entries"] == 6 * 5 + 2 * 4 + 8 * 3 and info["chunks"] == 1251 and info["tokens"] == REAL["tokens"] and info["size"] == out.stat().st_size
    a = info["adapt"]
    assert (a["layers_before"], a["layers_after"], a["kept_layers"], a["ffn_down_before"], a["ffn_down_after"], a["weight"], a["origin_repo"]) == (16, 8, KEEP, 24, 10, 1.0, "Qwen/Qwen3.8-27B")
    assert a["tensors_before"] == len(jb.vals["names"]) and a["tensors_after"] == len(ad.read_matrix(out).names) and a["dropped_layers"] == [4, 5, 6, 7, 8, 9, 10, 11]
    (p,) = info["parents"]
    assert p["kind"] == "file" and p["path"] == jb.src and p["sha256"] == imatrix.file_sha256(jb.src) and p["chunks"] == 1251
    assert info["cut_model"] == {"dir": str(jb.cut), "name": "slim"}
    assert any("UNMODIFIED model's activations" in n and "not Unsloth's Dynamic recipe" in n for n in info["notes"]) and r["numbers"]["size"] == out.stat().st_size
    # the file reads with the other (header) reader too, and is the real layout
    assert imatrix.matrix_summary(out)["layers"] == 8 and imatrix.matrix_summary(out)["intermediate"] == 10 and imatrix.matrix_summary(out)["datasets"] == ["unsloth_calibration_dataset", "(adapted to slim)"]
    assert imatrix.matrix_summary(out)["tokens"] == REAL["tokens"] and imatrix.read_gguf_header(out)[0]["imatrix.chunk_size"] == 8192
    pr = [x["value"] for x in jb.events if x["type"] == "progress"]
    assert pr == sorted(pr) and pr[-1] == 1.0


def test_the_weight_goes_into_the_entry_and_the_file(jb):
    r = jb.run(weight=3)
    m = ad.read_matrix(jb.folder / "imatrix.gguf")
    assert m.arrays["blk.0.attn_gate.weight.counts"].tolist() == [REAL["tokens"] * 3] and imatrix.matrix_summary(jb.folder / "imatrix.gguf")["tokens"] == REAL["tokens"] * 3
    info = r["info"]
    assert info["weight"] == 3 and info["adapt"]["weight"] == 3 and info["tokens"] == REAL["tokens"] * 3 and any("weight 3" in n and "3 times as strongly" in n for n in info["notes"])


def test_llama_imatrix_reads_the_adapted_file_back_when_it_is_there(jb):
    r = jb.run()
    (c,) = jb.calls()
    a = c["args"]
    assert a[0] == "--in-file" and a[1].endswith("imatrix.gguf") and a[2] == "-o" and a[3].endswith("roundtrip.partial.gguf") and a[4:] == ["--output-format", "gguf"]
    assert r["info"]["llama_check"] == {"ran": True, "ok": True} and r["info"]["llama_cpp"] == "version: 9999 (stub1234)" and not list(jb.folder.glob("roundtrip*"))


def test_a_llama_imatrix_that_refuses_is_a_warning_and_the_entry_is_kept(jb, monkeypatch):
    monkeypatch.setenv("ADAPT_MODE", "refuse")
    r = jb.run()
    assert r["info"]["llama_check"]["ok"] is False and any(n.startswith("Warning: llama-imatrix did not accept the adapted matrix") and "has not been verified" in n for n in r["notes"])
    assert (jb.folder / "imatrix.gguf").is_file()


def test_an_older_llama_imatrix_without_the_output_format_flag_is_retried(jb, monkeypatch):
    monkeypatch.setenv("ADAPT_MODE", "old")
    r = jb.run()
    assert len(jb.calls()) == 2 and "--output-format" not in jb.calls()[1]["args"] and r["info"]["llama_check"]["ok"] is True


def test_without_llama_cpp_the_check_is_skipped_and_the_matrix_is_still_made(jb, monkeypatch):
    def none(need_convert):
        raise imatrixjob.MatrixLlamaMissing("no llama.cpp")
    monkeypatch.setattr(ad, "find_tools", none)
    r = jb.run()
    assert r["info"]["llama_check"]["ran"] is False and r["info"]["llama_cpp"] == "" and not [n for n in r["notes"] if n.startswith("Warning")]


def test_a_library_source_and_an_adapted_one(jb):
    lib = {"kind": "library", "id": "x1", "file": jb.src, "label": "mine", "chunks": 64, "source_kind": "merged"}
    r = jb.run(source=lib)
    p = r["info"]["parents"][0]
    assert (p["kind"], p["id"], p["label"], p["chunks"]) == ("library", "x1", "mine", 64) and r["info"]["source"] == "mine, adapted to the cut model slim"
    e = refused(jb.run, id="20261116-1000-again", source={**lib, "source_kind": "adapted", "label": "already"})
    assert e.reason == "already_adapted" and "already an adapted one" in str(e)
    assert not (jb.lib / "20261116-1000-again").exists()


def test_the_published_matrix_is_looked_up_downloaded_once_and_adapted(jb):
    r = jb.run(source={"kind": "published", "base": "Qwen/Qwen3.8-27B"})
    p = r["info"]["parents"][0]
    assert (p["kind"], p["repo"], p["file"], p["base"]) == ("published", "unsloth/Qwen3.8-27B-GGUF", "imatrix_unsloth.gguf", "Qwen/Qwen3.8-27B") and r["info"]["source"].startswith("Unsloth's published matrix for Qwen/Qwen3.8-27B, adapted")
    assert (jb.cache / "unsloth--Qwen3.8-27B-GGUF" / "imatrix_unsloth.gguf").is_file()
    n = len(jb.hub.seen)
    jb.run(id="20261116-1000-second", source={"kind": "published", "base": "Qwen/Qwen3.8-27B"})
    assert [p for _, p, _ in jb.hub.seen[n:]] == ["/api/models/unsloth/Qwen3.8-27B-GGUF?blobs=true"], "nothing is downloaded twice"


def test_refusals_come_before_anything_is_downloaded_or_written(jb, tmp_path):
    other, _, _ = make_cut(tmp_path, name="headcut", new_inter=10, heads=True)
    n = len(jb.hub.seen)
    e = refused(jb.run, model_dir=str(other), source={"kind": "published", "base": "Qwen/Qwen3.8-27B"})
    assert e.reason == "heads" and len(jb.hub.seen) == n and not jb.lib.exists()
    wrong, _ = matrix(tmp_path / "w.gguf", layers=12)
    e = refused(jb.run, source={"kind": "file", "file": wrong})
    assert e.reason == "wrong_original" and not jb.folder.exists() and not list(jb.lib.glob("*"))


def test_input_mistakes_have_plain_messages(jb, tmp_path):
    with pytest.raises(imatrixjob.MatrixError, match="Choose the matrix to adapt"):
        jb.run(source={"kind": "nothing"})
    with pytest.raises(imatrixjob.MatrixError, match="was not found"):
        jb.run(source={"kind": "file", "file": str(tmp_path / "missing.gguf")})
    with pytest.raises(imatrixjob.MatrixError, match="Choose the cut model"):
        jb.run(model_dir="")
    with pytest.raises(imatrixjob.MatrixError, match="not a valid name"):
        jb.run(id="../x")
    dat = tmp_path / "old.dat"
    dat.write_bytes(b"LEGACY")
    assert "legacy .dat" in str(refused(jb.run, source={"kind": "file", "file": str(dat)}))
    jb.run()
    with pytest.raises(imatrixjob.MatrixError, match="exists already"):
        jb.run()


def test_stop_ends_the_job_and_leaves_no_folder(jb, tmp_path):
    (tmp_path / "STOP").write_text("x")
    with pytest.raises(imatrix.Stopped):
        jb.run()
    assert not jb.folder.exists()


def test_a_wrong_write_is_caught_by_the_verification_and_removed(jb, monkeypatch):
    real = ad.write_matrix

    def broken(path, m):
        n = m.names[0]
        m.arrays[n] = m.arrays[n] + 1
        return real(path, m)
    monkeypatch.setattr(ad, "write_matrix", broken)
    with pytest.raises(imatrixjob.MatrixError, match="is not the source's"):
        jb.run()
    assert not jb.folder.exists()


def test_not_enough_disk_is_said(jb, monkeypatch):
    monkeypatch.setattr(imatrixjob, "free_bytes", lambda p: 100)
    with pytest.raises(imatrixjob.MatrixError, match="not enough free disk space"):
        jb.run()
    assert not jb.folder.exists()


def test_the_preview_gives_the_numbers_of_the_page_and_the_same_refusals(jb, tmp_path):
    nums = ad.preview({"model_dir": str(jb.cut), "model_name": "slim", "weight": 1, "source": {"file": jb.src}})
    assert (nums["layers_before"], nums["layers_after"], nums["ffn_down_before"], nums["ffn_down_after"], nums["kept"], nums["thinner"]) == (16, 8, 24, 10, KEEP, True)
    r = jb.run()
    assert nums["size"] == Path(r["file"]).stat().st_size and nums["size_before"] == Path(jb.src).stat().st_size and nums["origin_repo"] == "Qwen/Qwen3.8-27B"
    other, _, _ = make_cut(tmp_path, name="headcut", new_inter=10, heads=True)
    assert refused(ad.preview, {"model_dir": str(other), "source": {"file": jb.src}}).reason == "heads"
    assert refused(ad.preview, {"model_dir": str(jb.cut), "weight": -1, "source": {"file": jb.src}}).reason == "bad_weight"


# ------------------------------------------------------------------------------------------------------------------ the entry point

def test_the_worker_runs_the_job_and_gives_the_refusal_code(jb, tmp_path):
    import subprocess
    jf = tmp_path / "job.json"
    jf.write_text(json.dumps(jb.job(source={"kind": "file", "file": jb.src}, id="20261116-1000-cli", **{})), encoding="utf-8")
    env = {**os.environ, "STUB_LOG": str(tmp_path / "calls2.jsonl"), "LLAMA_CPP_DIR": str(tmp_path / "nowhere")}
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(jf)], cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=120, env=env)
    ev = [json.loads(line) for line in p.stdout.splitlines()]
    assert p.returncode == 0 and ev[-1]["type"] == "result" and ev[-1]["data"]["info"]["source_kind"] == "adapted", ev[-1]
    jf.write_text(json.dumps(jb.job(id="20261116-1000-bad", model_dir=str(make_cut(tmp_path, name="hc", new_inter=10, heads=True)[0]))), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(jf)], cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=120, env=env)
    ev = [json.loads(line) for line in p.stdout.splitlines()]
    assert p.returncode == 1 and ev[-1]["type"] == "error" and ev[-1]["code"] == "adapt_heads" and "attention heads were cut" in ev[-1]["message"], ev[-1]
    jf.write_text(json.dumps({"job": "imatrix_adapt_preview", "model_dir": str(jb.cut), "model_name": "slim", "source": {"file": jb.src}}), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(jf)], cwd=Path(__file__).parent.parent, capture_output=True, text=True, timeout=120, env=env)
    ev = [json.loads(line) for line in p.stdout.splitlines()]
    assert p.returncode == 0 and ev[-1]["data"]["layers_after"] == 8


# ------------------------------------------------------------------------------------------------------------------ merging with weights, the adapted published matrix and the safety rule

MERGE_TOOL = '''import json, os, sys
a = sys.argv[1:]
if a == ["--version"]:
    print("version: 9999 (stub1234)")
    sys.exit(0)
ins = [a[i + 1] for i, x in enumerate(a) if x == "--in-file"]
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "imatrix", "args": a, "sizes": [os.path.getsize(i) for i in ins], "raw": [open(i, "rb").read().hex() for i in ins]}) + "\\n")
open(a[a.index("-o") + 1], "wb").write(open(ins[0], "rb").read())
'''


@pytest.fixture
def mj(jb, tmp_path):
    jb.tool.write_text(MERGE_TOOL, encoding="utf-8")
    jb.own, _ = matrix(tmp_path / "own.gguf", layers=8, inter=10, tokens=65536.0, chunk_count=128, chunk_size=512)
    jb.lib_job = {"job": "imatrix_merge", "root": str(jb.lib), "id": "20261116-1000-merged", "label": "adapted + mine", "stop_file": str(tmp_path / "STOP"), "cache": str(jb.cache), "notes": [],
                  "inputs": [{"kind": "published", "base": "Qwen/Qwen3.8-27B", "adapt_to": {"model_dir": str(jb.cut), "model_name": "slim"}, "weight": 0.1},
                             {"kind": "library", "id": "own", "file": jb.own, "label": "mine", "model": "slim", "chunks": 128, "cut_model": {"dir": str(jb.cut), "name": "slim"}}]}
    return jb


def test_an_adapted_published_matrix_and_my_own_are_merged_in_one_step_with_their_weights(mj):
    r = imatrixmerge.merge(mj.lib_job)
    (c,) = mj.calls()
    assert c["args"][0] == "--in-file" and "parent0.gguf" in c["args"][1] and c["args"][2] == "--in-file" and c["args"][3] == mj.own
    first = bytes.fromhex(c["raw"][0])
    tmp = mj.root / "sent.gguf"
    tmp.write_bytes(first)
    m = ad.read_matrix(tmp)
    assert len(m.names) == 2 * (6 * 5 + 2 * 4 + 8 * 3) and len(m.arrays["blk.0.ffn_down.weight.in_sum2"]) == 10
    assert m.arrays["blk.0.attn_gate.weight.counts"].tolist() == [pytest.approx(REAL["tokens"] * 0.1, rel=1e-6)] and m.datasets()[-1] == "(adapted to slim)"
    rows = r["weights"]
    assert [x["weight"] for x in rows] == [0.1, 1.0] and rows[0]["tokens"] == pytest.approx(REAL["tokens"] * 0.1) and rows[1]["tokens"] == 65536.0
    assert rows[0]["share"] == pytest.approx(REAL["tokens"] * 0.1 / (REAL["tokens"] * 0.1 + 65536.0), abs=1e-3)
    info = r["info"]
    assert info["parents"][0]["weight"] == 0.1 and info["parents"][0]["adapted"]["layers_after"] == 8 and info["parents"][0]["adapted"]["ffn_down_after"] == 10 and info["parents"][0]["sha256"]
    assert info["cut_model"] == {"dir": str(mj.cut), "name": "slim"} and not (Path(r["dir"]) / "work").exists()
    assert "(adapted to slim)" in info["source"] and any("approximation" in w for w in r["warnings"]) and any("UNMODIFIED base model" in w for w in r["warnings"])


def test_a_weight_alone_writes_a_scaled_copy_and_the_other_parent_is_passed_as_it_is(mj, tmp_path):
    b, _ = matrix(tmp_path / "b.gguf", layers=4, tokens=1000.0)
    a, _ = matrix(tmp_path / "a.gguf", layers=4, tokens=1000.0)
    job = {**mj.lib_job, "inputs": [{"kind": "file", "file": a, "weight": 4}, {"kind": "file", "file": b}]}
    r = imatrixmerge.merge(job)
    (c,) = mj.calls()
    assert c["args"][3] == b and "parent0.gguf" in c["args"][1]
    tmp = mj.root / "s.gguf"
    tmp.write_bytes(bytes.fromhex(c["raw"][0]))
    assert ad.read_matrix(tmp).arrays["blk.0.attn_gate.weight.counts"].tolist() == [4000.0]
    assert [x["share"] for x in r["weights"]] == [0.8, 0.2] and [x["weight"] for x in r["weights"]] == [4.0, 1.0]
    # weight 1 for everything: the files are passed untouched and no copy is made
    r2 = imatrixmerge.merge({**job, "id": "20261116-1000-plain", "inputs": [{"kind": "file", "file": a}, {"kind": "file", "file": b}]})
    assert mj.calls()[1]["args"][1] == a and [x["share"] for x in r2["weights"]] == [0.5, 0.5]


def test_bad_weights_are_refused_before_anything_is_made(mj, tmp_path):
    for bad in (0, -2, "x", True):
        job = {**mj.lib_job, "inputs": [{**mj.lib_job["inputs"][0], "weight": bad}, mj.lib_job["inputs"][1]]}
        with pytest.raises(imatrixjob.MatrixError, match="must be a number above 0"):
            imatrixmerge.merge(job)
    assert not (mj.lib / "20261116-1000-merged").exists()


def test_a_published_matrix_is_not_merged_raw_with_a_matrix_of_a_cut_model(mj):
    job = {**mj.lib_job, "inputs": [{"kind": "published", "base": "Qwen/Qwen3.8-27B"}, mj.lib_job["inputs"][1]]}
    with pytest.raises(ad.AdaptRefused) as e:
        imatrixmerge.merge(job)
    assert e.value.code == "adapt_cut_model" and "describes the original model" in str(e.value) and "8 layers instead of 16" in str(e.value) and "Adapt to my cut model" in str(e.value)
    assert not mj.calls() and not (mj.lib / "20261116-1000-merged").exists(), "llama-imatrix was never started"
    # the same layers, another MLP width (a model made only thinner)
    thin, _ = matrix(mj.root / "thin.gguf", layers=16, inter=10)
    job2 = {**job, "inputs": [{"kind": "published", "base": "Qwen/Qwen3.8-27B"}, {"kind": "file", "file": thin}]}
    with pytest.raises(ad.AdaptRefused, match="an MLP width of 10 instead of 24"):
        imatrixmerge.merge(job2)


def test_a_published_matrix_that_matches_the_other_one_is_still_merged_raw(mj):
    same, _ = matrix(mj.root / "same.gguf", tokens=5000.0)
    job = {**mj.lib_job, "inputs": [{"kind": "published", "base": "Qwen/Qwen3.8-27B"}, {"kind": "file", "file": same}]}
    r = imatrixmerge.merge(job)
    assert [p["kind"] for p in r["info"]["parents"]] == ["published", "file"] and mj.calls()[0]["args"][1].endswith("imatrix_unsloth.gguf")
    assert not any("approximation" in w for w in r["warnings"]) and any("carries about" in w and "weight" in w for w in r["warnings"]), "the published one outweighs: said, with the weight as the way out"


def test_the_adaptation_inside_a_merge_refuses_a_matrix_that_is_not_the_original(mj, tmp_path):
    wrong, _ = matrix(tmp_path / "wrong.gguf", layers=12)
    job = {**mj.lib_job, "inputs": [{"kind": "file", "file": wrong, "adapt_to": {"model_dir": str(mj.cut), "model_name": "slim"}}, mj.lib_job["inputs"][1]]}
    with pytest.raises(ad.AdaptRefused) as e:
        imatrixmerge.merge(job)
    assert e.value.reason == "wrong_original" and not (mj.lib / "20261116-1000-merged").exists()


# ------------------------------------------------------------------------------------------------------------------ the real published file (skipped unless NDFORGE_REAL_IMATRIX is set)

@pytest.mark.skipif(not os.environ.get("NDFORGE_REAL_IMATRIX") or not Path(os.environ.get("NDFORGE_REAL_IMATRIX") or "x").is_file(), reason="NDFORGE_REAL_IMATRIX does not point at the real imatrix_unsloth.gguf")
def test_the_real_published_matrix_is_adapted_to_a_cut_27b_and_reads_back(tmp_path):
    """64 layers cut to 32 (two blocks of 4 kept at the front, the rest of the 16 blocks after block 9), the MLP 17408 -> 10752 per layer, as the owner's model."""
    real = os.environ["NDFORGE_REAL_IMATRIX"]
    src = list(range(0, 8)) + list(range(40, 64))
    cut_dir, kept, _ = make_cut(tmp_path, name="cut16b", src=src, inter=17408, new_inter=10752, origin_layers=64)
    m = ad.read_matrix(real)
    cut = ad.read_cut(cut_dir, "cut16b")
    out, p = ad.adapt(m, cut, 1.0)
    assert (p["layers_before"], p["layers_after"], p["ffn_down_before"], p["ffn_down_after"]) == (64, 32, 17408, 10752)
    ad.write_matrix(tmp_path / "adapted.gguf", out)
    nums = ad.preview_numbers(p, out.kvs)
    assert nums["size"] == (tmp_path / "adapted.gguf").stat().st_size and nums["tensors_after"] == len(out.names)
    back = ad.verify(m, tmp_path / "adapted.gguf", cut, p, 1.0)
    s = imatrix.matrix_summary(tmp_path / "adapted.gguf")
    assert (s["layers"], s["intermediate"], s["hidden"], s["tokens"], s["chunks"], s["chunk_size"]) == (32, 10752, 5120, 1149614.0, 1251, 8192)
    assert s["datasets"] == ["unsloth_calibration_dataset", "(adapted to cut16b)"] and s["kinds"]["ffn_down"] == 32
    full_kept = [L for L in src if L % 4 == 3]
    assert s["kinds"]["attn_q"] == len(full_kept) == 8 and s["kinds"]["attn_qkv"] == 24 and back["tensors"] == len(out.names)
    j = 5
    assert np.array_equal(out.arrays[f"blk.{j}.ffn_down.weight.in_sum2"], m.arrays[f"blk.{src[j]}.ffn_down.weight.in_sum2"][kept[j]])
    assert np.array_equal(out.arrays["blk.30.ssm_out.weight.in_sum2"], m.arrays["blk.62.ssm_out.weight.in_sum2"]) and "blk.31.ssm_out.weight.in_sum2" not in out.arrays, "new 31 is old 63: a full-attention layer"
    print("real adapted:", nums["size"], "bytes,", nums["tensors_after"], "tensors")
