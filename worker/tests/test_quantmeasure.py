"""quantmeasure.py: the importance from a matrix, the weighted error of a tensor, sampling, the cache and stop/resume (numpy engine, tiny synthetic GGUFs)."""
import json
import os

import numpy as np
import pytest
from ggufmake import F32, tiny_model, write

from ndforge_worker import ggufinfo, ggufrows, mixedquant, quantblocks, quantmeasure as qm

KV = [("general.architecture", "str", "qwen35moe")]


def make_matrix(path, info, rng, layout="1d", counts=1000.0, skip=(), expert_counts=None, bad=None):
    """An importance matrix GGUF for the sampled tensors of `info`: skewed channels (some channels 100x the others)."""
    ts = []
    for t in info["tensors"]:
        if not ggufrows.is_sampled(t) or t["name"] in skip:
            continue
        cols, _r, experts = ggufrows.geometry(t)
        stats = (rng.random((experts, cols)) * np.where(rng.random(cols) < 0.1, 100.0, 1.0)).astype(np.float32) * counts / 1000
        if bad == t["name"]:
            stats = stats[:, : cols // 2]
        if experts > 1 or layout == "2d":
            ts.append((t["name"] + ".in_sum2", [stats.shape[1], experts], F32, stats))
            ts.append((t["name"] + ".counts", [experts], F32, np.full(experts, (expert_counts or counts / 10) if experts > 1 else counts, np.float32)))
        else:
            ts.append((t["name"] + ".in_sum2", [stats.shape[1]], F32, stats[0]))
            ts.append((t["name"] + ".counts", [1], F32, np.array([counts], np.float32)))
    return write(path, ts, KV)


@pytest.fixture
def model(tmp_path):
    p = tiny_model(tmp_path / "m.gguf", rows=300, cols=512)
    return ggufrows.open_source(p)


def test_weighted_error_equals_a_hand_computation(model):
    t = next(x for x in model["tensors"] if x["name"] == "blk.0.attn_q.weight")
    pick = ggufrows.sample_plan(model, 40, 1)[t["name"]]
    rows = ggufrows.read_rows(model, t, ggufrows.flat_rows(pick))
    a = (np.random.default_rng(2).random((1, 512)) + 0.5)
    rec = qm.tensor_record(t, pick, rows, a)
    # by hand, row by row and channel by channel
    raw = quantblocks.encode(rows, "Q4_K")
    dec = quantblocks.decode(raw, "Q4_K", *rows.shape)
    tot = 0.0
    for r in range(rows.shape[0]):
        for c in range(512):
            tot += a[0, c] * (float(rows[r, c]) - float(dec[r, c])) ** 2
    assert rec["types"]["Q4_K"]["err"] == pytest.approx(300 / 40 * tot, rel=1e-9)
    assert rec["norm2_w"] == pytest.approx(300 / 40 * float(np.sum(a * rows.astype(np.float64) ** 2)), rel=1e-9)
    assert rec["types"]["Q4_K"]["bytes"] == 512 * 300 * 4.5 / 8
    errs = [rec["types"][r]["err"] for r in qm.RUNGS]
    assert errs == sorted(errs, reverse=True)           # more bits, less error


def test_fallback_types_for_a_row_length_that_does_not_fit(model):
    t = next(x for x in model["tensors"] if x["name"] == "blk.0.ssm_out.weight")        # 320 columns
    pick = ggufrows.sample_plan(model, 100)[t["name"]]
    rec = qm.tensor_record(t, pick, ggufrows.read_rows(model, t, ggufrows.flat_rows(pick)), np.ones((1, 320)))
    assert {r: rec["types"][r]["eff"] for r in qm.RUNGS} == {"Q2_K": "Q4_0", "Q3_K": "Q4_0", "Q4_K": "Q5_0", "Q5_K": "Q5_1", "Q6_K": "Q8_0", "Q8_0": "Q8_0"}
    assert rec["types"]["Q2_K"]["err"] == rec["types"]["Q3_K"]["err"]


def test_moe_importance_divides_by_the_dense_count(tmp_path, model):
    rng = np.random.default_rng(3)
    ts = []
    for t in model["tensors"]:
        if not ggufrows.is_sampled(t):
            continue
        cols, _r, experts = ggufrows.geometry(t)
        if experts > 1:
            ts.append((t["name"] + ".in_sum2", [cols, experts], F32, np.full((experts, cols), 50.0, np.float32)))
            ts.append((t["name"] + ".counts", [experts], F32, np.array([100, 200, 300, 400], np.float32)))
        else:
            ts.append((t["name"] + ".in_sum2", [cols], F32, np.full(cols, 2000.0, np.float32)))
            ts.append((t["name"] + ".counts", [1], F32, np.array([1000.0], np.float32)))
    p = write(tmp_path / "mx.gguf", ts, KV)
    imp, notes = qm.importance(p, model)
    assert notes == []
    e = imp["blk.0.ffn_up_exps.weight"]
    assert e.shape == (4, 512) and np.allclose(e, 50.0 / 1000.0)               # in_sum2 / N, N = 1000 (the dense tensors' count), not the expert's own count
    assert np.allclose(imp["blk.0.attn_q.weight"], 2.0)


def test_one_d_and_two_d_layouts_give_the_same_result(tmp_path, model):
    m1 = make_matrix(tmp_path / "a.gguf", model, np.random.default_rng(4), "1d")
    m2 = make_matrix(tmp_path / "b.gguf", model, np.random.default_rng(4), "2d")
    a, _ = qm.importance(m1, model)
    b, _ = qm.importance(m2, model)
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    assert a["blk.0.attn_q.weight"].shape == (1, 512)


def test_column_mismatch_stops_in_plain_words(tmp_path, model):
    m = make_matrix(tmp_path / "mx.gguf", model, np.random.default_rng(5), bad="blk.1.attn_k.weight")
    with pytest.raises(qm.MeasureError, match="not of this model; adapt it first"):
        qm.importance(m, model)


def test_dat_matrix_refused_and_unreadable_matrix(tmp_path, model):
    d = tmp_path / "imatrix.dat"
    d.write_bytes(b"\x01\x00")
    with pytest.raises(qm.MeasureError, match="legacy .dat"):
        qm.importance(d, model)
    (tmp_path / "x.gguf").write_bytes(b"nope")
    with pytest.raises(qm.MeasureError, match="could not be read"):
        qm.importance(tmp_path / "x.gguf", model)


def test_missing_tensors_get_the_median_and_a_note(tmp_path, model):
    m = make_matrix(tmp_path / "mx.gguf", model, np.random.default_rng(6), skip=("blk.1.attn_q.weight", "token_embd.weight"))
    imp, notes = qm.importance(m, model)
    assert len(notes) == 1 and "2 tensors are not in the matrix" in notes[0]
    got = imp["blk.1.attn_q.weight"]
    assert np.all(got == got.flat[0]) and np.array_equal(got, imp["token_embd.weight"])
    means = [float(v.mean()) for k, v in imp.items() if k not in ("blk.1.attn_q.weight", "token_embd.weight")]
    assert got.flat[0] == pytest.approx(float(np.median(means)))


def test_no_matrix_is_all_ones_with_a_note(model):
    imp, notes = qm.importance(None, model)
    assert notes and "No importance matrix" in notes[0] and np.all(imp["blk.0.attn_q.weight"] == 1.0)
    assert imp["blk.0.ffn_up_exps.weight"].shape == (4, 512)


def full_error(src, t, a, rung):
    """the exact error of ALL rows of a tensor, computed independently of tensor_record"""
    cols, per, experts = ggufrows.geometry(t)
    w = ggufrows.read_rows(src, t, range(per * experts)).astype(np.float64)
    eff = mixedquant.effective_type(rung, cols)
    dec = quantblocks.decode(quantblocks.encode(w.astype(np.float32), eff), eff, w.shape[0], cols).astype(np.float64)
    arow = np.repeat(a, per, axis=0)
    return float(np.sum(arow * (w - dec) ** 2))


def test_full_sampling_equals_the_full_tensor_error_exactly(model):
    t = next(x for x in model["tensors"] if x["name"] == "blk.1.ffn_up_exps.weight")
    a = np.random.default_rng(8).random((4, 512)) + 0.1
    pick = ggufrows.sample_plan(model, 100_000)[t["name"]]
    assert pick["exact"]
    rec = qm.tensor_record(t, pick, ggufrows.read_rows(model, t, ggufrows.flat_rows(pick)), a)
    for rung in ("Q3_K", "Q6_K"):
        assert rec["types"][rung]["err"] == pytest.approx(full_error(model, t, a, rung), rel=1e-9)


def test_sampled_estimate_is_close(tmp_path):
    p = tiny_model(tmp_path / "big.gguf", rows=512, cols=512)
    src = ggufrows.open_source(p)
    t = next(x for x in src["tensors"] if x["name"] == "blk.0.attn_q.weight")
    a = np.random.default_rng(9).random((1, 512)) + 0.1
    pick = ggufrows.sample_plan(src, 64, 5)[t["name"]]
    assert len(pick["rows"]) == 64 and not pick["exact"]
    rec = qm.tensor_record(t, pick, ggufrows.read_rows(src, t, ggufrows.flat_rows(pick)), a)
    for rung in ("Q2_K", "Q4_K", "Q6_K"):
        assert rec["types"][rung]["err"] == pytest.approx(full_error(src, t, a, rung), rel=0.10)


def run(model, tmp_path, name, threads=1, matrix=None, **kw):
    imp, _ = qm.importance(matrix, model)
    opts = {"sample_rows": 16, "seed": 1, "threads": threads, "rungs": ("Q3_K", "Q4_K", "Q6_K"), "key": "k" * 16}
    return qm.measure(model, imp, opts, tmp_path / name, **kw)


def test_deterministic_across_runs_and_thread_counts(tmp_path, model):
    m = make_matrix(tmp_path / "mx.gguf", model, np.random.default_rng(10))
    a = run(model, tmp_path, "c1", 1, m)
    b = run(model, tmp_path, "c2", 4, m)
    c = run(model, tmp_path, "c3", 4, m)
    assert json.dumps(a["tensors"], sort_keys=True) == json.dumps(b["tensors"], sort_keys=True) == json.dumps(c["tensors"], sort_keys=True)
    assert a["done"] and set(a["tensors"]) == {t["name"] for t in model["tensors"] if ggufrows.is_sampled(t)}


def test_stop_then_resume_measures_only_the_rest(tmp_path, model, monkeypatch):
    calls = []
    real = quantblocks.encode
    monkeypatch.setattr(quantblocks, "encode", lambda x, t: (calls.append(1), real(x, t))[1])
    seen = []
    with pytest.raises(qm.Stopped):
        run(model, tmp_path, "c", 1, None, should_stop=lambda: len(seen) >= 3, progress=lambda d, n, name: seen.append(name))
    cache = json.loads((tmp_path / "c" / "measure.json").read_text())
    total = sum(1 for t in model["tensors"] if ggufrows.is_sampled(t))
    got = len(cache["tensors"])
    assert 3 <= got < total and not cache["done"]                            # the tensors already in flight when the stop came are kept too
    assert len(calls) >= 3 * 3                                               # 3 tensors x 3 rungs at least
    calls.clear()
    res = run(model, tmp_path, "c", 1, None)
    assert res["done"] and len(res["tensors"]) == total
    assert 0 < len(calls) <= (total - got) * 3                               # nothing that was measured before the stop was encoded again
    assert not (tmp_path / "c" / "measure.json.tmp").exists()


def test_cache_with_another_key_is_ignored(tmp_path, model):
    run(model, tmp_path, "c", 1, None)
    imp, _ = qm.importance(None, model)
    again = qm.measure(model, imp, {"sample_rows": 16, "seed": 1, "threads": 1, "rungs": ("Q4_K",), "key": "other-key"}, tmp_path / "c")
    assert again["key"] == "other-key" and all(list(r["types"]) == ["Q4_K"] for r in again["tensors"].values())


def test_non_finite_tensor_is_skipped_with_a_note(tmp_path):
    rng = np.random.default_rng(11)
    bad = rng.standard_normal((8, 256)).astype(np.float32)
    bad[2, 3] = np.nan
    p = write(tmp_path / "n.gguf", [("blk.0.attn_q.weight", [256, 8], F32, bad), ("blk.0.attn_k.weight", [256, 8], F32, rng.standard_normal((8, 256)))], KV)
    src = ggufrows.open_source(p)
    imp, _ = qm.importance(None, src)
    res = qm.measure(src, imp, {"sample_rows": 100, "threads": 1, "key": "k" * 16}, tmp_path / "c")
    assert res["skipped"] == ["blk.0.attn_q.weight"] and list(res["tensors"]) == ["blk.0.attn_k.weight"] and "not finite" in res["notes"][0]


def test_cache_key_changes_with_inputs_but_not_with_mtime(tmp_path, model):
    p = tmp_path / "m.gguf"
    fp = qm.fingerprint_model(p)
    os.utime(p, (1_000_000_000, 1_000_000_000))
    assert qm.fingerprint_model(p) == fp
    base = qm.cache_key(fp, "mx", 128, 0, "reference")
    assert len(base) == 16 and base == qm.cache_key(fp, "mx", 128, 0, "reference")
    assert len({base, qm.cache_key(fp, "other", 128, 0, "reference"), qm.cache_key(fp, "mx", 256, 0, "reference"), qm.cache_key(fp, "mx", 128, 1, "reference"),
                qm.cache_key(fp, "mx", 128, 0, "llama"), qm.cache_key(fp, "mx", 128, 0, "llama", "llama.cpp b1234"), qm.cache_key(fp, "none", 128, 0, "reference")}) == 7
    m = make_matrix(tmp_path / "a.gguf", model, np.random.default_rng(1))
    h = qm.fingerprint_file(m)
    os.utime(m, (1_000_000_000, 1_000_000_000))
    assert qm.fingerprint_file(m) == h
    # a different model gives a different fingerprint, a same-size model with other numbers too
    q = tiny_model(tmp_path / "q.gguf", rng=np.random.default_rng(99))
    assert q.stat().st_size == p.stat().st_size and qm.fingerprint_model(q) != fp


def test_speed_probe():
    s = qm.speed_probe(("Q4_K",), 1 << 12)
    assert s["Q4_K"] > 0
