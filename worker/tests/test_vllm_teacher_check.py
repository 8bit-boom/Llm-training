"""The fidelity check "is the vLLM teacher the same teacher?": the metrics on fake arrays with known overlaps and KL, the verdicts, and the job with an existing reference cache and a fake vLLM (no GPU, no torch)."""
import json
import math
import types

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import cachewrite, distilljob as dj, protocol, vllmcache, vllmteachercheck as tc  # noqa: E402
from test_cache_identity import make_model  # noqa: E402

K = 4


def answer(ids, probs, resid=None):
    ids = np.array(ids, dtype=np.int64)
    p = np.array(probs, dtype=np.float32)
    lp = np.log(p)
    r = np.array([1.0 - p[0].sum()] * len(ids), dtype=np.float32) if resid is None else np.array(resid, dtype=np.float32)
    return ids, lp, r


def test_identical_answers_are_perfect():
    a = answer([[0, 1, 2, 3], [4, 5, 6, 7]], [[.4, .3, .2, .05]] * 2)
    s = tc.position_stats(a, a, np)
    assert (s["top1"] == 1).all() and np.allclose(s["jaccard"], 1) and np.allclose(s["kl"], 0, atol=1e-6) and np.allclose(s["tail_diff"], 0)


def test_known_overlap_and_kl():
    ref = answer([[0, 1, 2, 3]], [[.4, .3, .2, .05]])
    # vLLM: top answer differs; shares tokens 1, 2, 3 (and has 9)
    other = answer([[1, 0 + 9, 2, 3]], [[.5, .2, .15, .05]])
    s = tc.position_stats(ref, other, np)
    assert s["top1"][0] == 0
    assert math.isclose(float(s["jaccard"][0]), 3 / 5, rel_tol=1e-6)                      # 3 shared of 4 + 4 - 3 = 5
    p = np.array([.3, .2, .05]); q = np.array([.5, .15, .05])                             # the shared tokens 1, 2, 3 on each side
    p, q = p / p.sum(), q / q.sum()
    assert math.isclose(float(s["kl"][0]), float((p * np.log(p / q)).sum()), rel_tol=1e-4)
    assert math.isclose(float(s["shared"][0]), 0.55, rel_tol=1e-5)
    assert math.isclose(float(s["tail_diff"][0]), (1 - .9) - (1 - .95), abs_tol=1e-6)      # vLLM's leftover 0.10 vs 0.05


def test_no_shared_token_gives_nan_kl_and_counts_as_no_overlap():
    a = answer([[0, 1, 2, 3]], [[.4, .3, .2, .05]])
    b = answer([[4, 5, 6, 7]], [[.4, .3, .2, .05]])
    s = tc.position_stats(a, b, np)
    assert math.isnan(float(s["kl"][0])) and s["jaccard"][0] == 0
    agg = tc.compare([a], [b], np)["overall"]
    assert agg["no_overlap"] == 1.0 and agg["kl"] is None
    assert tc.verdict_of(agg)["verdict"] == "different"


def test_shape_mismatch_and_row_count_are_plain_errors():
    a = answer([[0, 1, 2, 3]], [[.4, .3, .2, .05]])
    with pytest.raises(tc.TeacherCheckError):
        tc.position_stats(a, answer([[0, 1, 2]], [[.5, .3, .1]]), np)
    with pytest.raises(tc.TeacherCheckError):
        tc.compare([a], [], np)


def test_breakdown_by_position():
    n = 600
    ids = np.tile(np.arange(K), (n, 1))
    p = np.tile(np.array([.4, .3, .2, .05], dtype=np.float32), (n, 1))
    ref = (ids, np.log(p), np.full(n, .05, dtype=np.float32))
    ids2 = ids.copy()
    ids2[200:, 0], ids2[200:, 1] = ids[200:, 1], ids[200:, 0]                              # the top answer flips from position 200 on
    out = tc.compare([ref], [(ids2, np.log(p), np.full(n, .05, dtype=np.float32))], np)
    b = {x["from"]: x for x in out["by_position"]}
    assert set(b) == {0, 128, 512} and b[0]["top1"] == 1.0 and b[512]["top1"] == 0.0
    assert out["overall"]["positions"] == 600 and math.isclose(out["overall"]["top1"], 200 / 600, abs_tol=1e-3)
    assert out["rows"] == 1


def test_verdict_thresholds():
    ok = {"top1": 0.99, "kl": 0.005, "no_overlap": 0.0}
    assert tc.verdict_of(ok)["verdict"] == "same"
    assert tc.verdict_of({**ok, "top1": 0.95})["verdict"] == "close"
    assert tc.verdict_of({**ok, "kl": 0.05})["verdict"] == "close"
    assert tc.verdict_of({**ok, "top1": 0.9})["verdict"] == "different"
    assert tc.verdict_of({**ok, "kl": 0.5})["verdict"] == "different"
    assert tc.verdict_of({**ok, "no_overlap": 0.05})["verdict"] == "different"
    v = tc.verdict_of(ok)
    assert "Heuristics" in v["thresholds"]["note"] and v["thresholds"]["same"] == {"top1": 0.97, "kl": 0.02} and "99.0%" in v["text"]


def test_pick_rows_is_even_and_unique():
    assert tc.pick_rows(5, 100, np) == [0, 1, 2, 3, 4]
    p = tc.pick_rows(1000, 100, np)
    assert len(p) == 100 and p[0] == 0 and p[-1] == 999 and p == sorted(set(p))
    assert tc.pick_rows(0, 10, np) == []


# ------------------------------------------------------------------------------------------------------------- the job with a reference cache

def probs(n):
    """the same answers for a row of n tokens, for the reference cache and the fake vLLM"""
    p = np.sort(np.random.default_rng(100 + n).random((n, K)).astype(np.float32), axis=1)[:, ::-1]
    return p / (p.sum(axis=1, keepdims=True) * 1.2)


def make_reference(tmp_path, teacher, n_rows=12, load="4bit", **extra):
    folder = tmp_path / "heal" / "cache"
    folder.mkdir(parents=True)
    rows = [{"input_ids": list(range(1, 6 + i)), "completion_mask": [1] * (5 + i), "kind": "", "source": "project"} for i in range(n_rows)]
    dj.save_rows(folder, rows)
    meta = {"teacher": "Someone/Finetune", "k": K, "rows": n_rows, "data_hash": dj.data_hash(rows), "tokenizer_hash": "t", "max_len": 64, "general_key": {}, "general_rows": 0,
            "teacher_weights": dj.safetensors_io.weights_manifest(teacher), "teacher_load": load, **extra}
    w = cachewrite.CacheWriter(folder, meta, 40)
    for r in rows:
        n = len(r["input_ids"])
        ids = np.tile(np.arange(K), (n, 1))
        p = probs(n)
        w.add_row(ids, np.log(p), (1 - p.sum(axis=1)).astype(np.float32))
    w.close()
    return folder


def job_for(tmp_path, ref, **kw):
    return {"teacher": {"repo": str(tmp_path / "origin")}, "student": str(tmp_path / "student"), "model": {"family": "qwen3_5"}, "data": {}, "k": K, "max_seq": 64, "rows": 5,
            "out": str(tmp_path / "heal" / "vllm_teacher_check.json"), "reference_cache": str(ref) if ref else None,
            "engine": {"kind": "vllm", "teacher": {}, "container": {"kind": "docker"}}, **kw}


def fake_chunk(noise=0.0, flip=False, seen=None):
    def chunk(job, writer, rows_dir, meta, origin, origin_dir, first, last, k, max_seq, stop_file, **kw):
        if seen is not None:
            seen.append((meta, origin))
        z = np.load(rows_dir / "rows.npz")
        off = z["offsets"]
        for i in range(first, last):
            n = int(off[i + 1] - off[i])
            ids = np.tile(np.arange(K), (n, 1))
            p = probs(n) * (1.0 - noise)
            if flip:
                ids = ids[:, ::-1].copy()
            writer.add_row(ids, np.log(p), (1 - p.sum(axis=1)).astype(np.float32))
        writer.flush()
        return {"engine": "vllm-0.34.0", "teacher_quant": "bnb-nf4", "route": "bnb", "reason": "r", "startup_seconds": 510.6, "load_seconds": 480.0, "tokens_per_s": 2900.0, "tokens": 100,
                "load_phases": [{"t": 3.0, "line": "Loading weights took 100 s"}], "timings": {"engine_start_and_model_load": 470.0}, "stopped": False, "seconds": 600.0}
    return chunk


def run_job(tmp_path, monkeypatch, capsys, ref, chunk, **kw):
    make_model(tmp_path / "origin")
    monkeypatch.setattr(vllmcache, "run_chunk", chunk)
    monkeypatch.setattr(protocol, "_out", __import__("sys").stdout)
    rc = tc.run(job_for(tmp_path, ref, **kw))
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return rc, lines


def test_job_with_an_existing_reference_cache_says_same(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    seen: list = []
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, fake_chunk(seen=seen))
    res = [x for x in lines if x["type"] == "result"][0]["data"]
    assert rc == 0 and res["verdict"] == "same" and res["rows"] == 5 and res["reference"]["kind"] == "existing_cache"
    rep = json.loads((tmp_path / "heal" / "vllm_teacher_check.json").read_text())
    assert rep["identity"]["engine"] == "vllm-0.34.0" and rep["identity"]["teacher_quant"] == "bnb-nf4" and rep["vllm"]["startup_seconds"] == 510.6 and rep["vllm"]["tokens_per_s"] == 2900.0
    assert rep["vllm"]["load_phases"] and rep["metrics"]["top1"] == 1.0 and rep["thresholds"]["same"]["top1"] == 0.97
    meta, origin = seen[0]
    assert meta["engine"] == "vllm-0.34.0" and meta["teacher_origin"]["repo"] == str(tmp_path / "origin") and origin == str(tmp_path / "origin")


def test_job_reports_different_when_the_answers_disagree(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)
    rc, lines = run_job(tmp_path, monkeypatch, capsys, ref, fake_chunk(flip=True))
    res = [x for x in lines if x["type"] == "result"][0]["data"]
    assert res["verdict"] == "different" and "NOT used" in res["text"]


def test_a_vllm_or_wrong_weights_cache_is_not_a_reference(tmp_path):
    teacher = make_model(tmp_path / "origin")
    w = dj.safetensors_io.weights_manifest(teacher)
    assert tc._usable_reference(make_reference(tmp_path, teacher), w, K) is not None
    assert tc._usable_reference(make_reference(tmp_path / "b", teacher, engine="vllm-0.34.0", teacher_quant="bnb-nf4"), w, K) is None
    assert tc._usable_reference(make_reference(tmp_path / "c", teacher, load="16bit"), w, K) is None
    assert tc._usable_reference(make_reference(tmp_path / "d", teacher), "dir:other", K) is None
    assert tc._usable_reference(make_reference(tmp_path / "e", teacher), w, 8) is None
    assert tc._usable_reference(tmp_path / "nothing", w, K) is None


def test_the_job_is_in_the_contract_and_dispatched():
    c = json.loads((__import__("pathlib").Path(tc.__file__).parent / "contract.json").read_text())
    assert "vllm_teacher_check" in c["jobs"] and "vllm_teacher_check_failed" in c["error_codes"]


def test_a_failure_of_vllm_is_a_plain_check_error(tmp_path, monkeypatch, capsys):
    teacher = make_model(tmp_path / "origin")
    ref = make_reference(tmp_path, teacher)

    def boom(*a, **k):
        raise vllmcache.VllmCacheError("The container did not answer")
    with pytest.raises(tc.TeacherCheckError, match="did not answer"):
        run_job(tmp_path, monkeypatch, capsys, ref, boom)
