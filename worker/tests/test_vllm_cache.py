"""vllm_cache.py with fake vLLM answers (no vLLM, no torch): the conversion of a row, the leftover mass, batching, resume, stop, the lines on stdout."""
import json
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import cachewrite, distilljob, protocol, vllm_cache as vc  # noqa: E402

V, K = 50, 4


def dists(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, V)).astype(np.float32) * 2
    return z - np.log(np.exp(z).sum(-1, keepdims=True))          # log-softmax, float32


def expected(lp, k=K):
    ids = np.argsort(-lp, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(lp, ids, 1)
    resid = np.maximum(0.0, 1.0 - np.exp(top).sum(1)).astype(np.float32)
    return ids, top, resid


def as_dict(row_lp, extra_token=None, k=K):
    ids = np.argsort(-row_lp, kind="stable")[:k].tolist()
    if extra_token is not None and extra_token not in ids:
        ids.append(extra_token)                                  # the actual token outside the top k: the (k + 1)-th entry
    return {int(t): types.SimpleNamespace(logprob=float(row_lp[t]), rank=i + 1, decoded_token=None) for i, t in enumerate(ids)}


def fake_output(tokens, lp, shape):
    n = len(tokens)
    prompt = [None] + [as_dict(lp[j - 1], tokens[j]) for j in range(1, n)]            # prompt_logprobs[j] predicts token j
    gen = [as_dict(lp[n - 1])]
    if shape == "dict":
        pl, gl = prompt, gen
    elif shape == "flat":
        def flat(entries):
            ids, vals, st, en = [], [], [], []
            for d in entries:
                st.append(len(ids))
                for t, v in (d or {}).items():
                    ids.append(t)
                    vals.append(v.logprob)
                en.append(len(ids))
            return types.SimpleNamespace(token_ids=ids, logprobs=vals, start_indices=st, end_indices=en)
        pl, gl = flat(prompt), flat(gen)
    elif shape == "flat_dictkeys":
        f = fake_output(tokens, lp, "flat")
        pl, gl = f.prompt_logprobs.__dict__, f.outputs[0].logprobs.__dict__
    elif shape == "floats":
        pl = [None] + [{t: v.logprob for t, v in d.items()} for d in prompt[1:]]
        gl = [{t: v.logprob for t, v in gen[0].items()}]
    else:
        raise AssertionError(shape)
    return types.SimpleNamespace(prompt_logprobs=pl, outputs=[types.SimpleNamespace(logprobs=gl, token_ids=[0])])


@pytest.mark.parametrize("shape", ["dict", "flat", "flat_dictkeys", "floats"])
def test_convert_row_matches_the_topk_math(shape):
    toks = [3, 7, 11, 2, 40, 41, 5]
    lp = dists(len(toks), 1)
    o = fake_output(toks, lp, shape)
    ids, logp, resid = vc.convert_row(o.prompt_logprobs, o.outputs[0].logprobs, len(toks), K, np)
    e_ids, e_top, e_res = expected(lp)
    assert ids.shape == (7, K) and logp.dtype == np.float32 and resid.dtype == np.float32
    assert (ids == e_ids).all()
    assert np.allclose(logp, e_top, atol=1e-6)
    assert np.allclose(resid, e_res, atol=1e-6) and (resid >= 0).all()


def test_the_33rd_entry_is_dropped_and_the_leftover_ignores_it():
    toks = [0, 49, 48]
    lp = dists(3, 5)
    worst = int(np.argmin(lp[0]))
    toks[1] = worst                                              # the actual token is the least likely: vLLM adds it as the (k + 1)-th entry
    o = fake_output(toks, lp, "dict")
    assert len(o.prompt_logprobs[1]) == K + 1
    ids, logp, resid = vc.convert_row(o.prompt_logprobs, o.outputs[0].logprobs, 3, K, np)
    assert worst not in ids[0].tolist() and logp.shape == (3, K)
    assert np.allclose(resid[0], max(0.0, 1.0 - float(np.exp(np.sort(lp[0])[::-1][:K]).sum())), atol=1e-6)


def test_leftover_is_clamped_at_zero():
    lp = np.log(np.array([[0.5, 0.3, 0.2000001, 0.0000001]], dtype=np.float32))
    lp = np.repeat(lp, 2, axis=0)
    o = fake_output([1, 2], lp, "dict")
    _, _, resid = vc.convert_row(o.prompt_logprobs, o.outputs[0].logprobs, 2, 4, np)
    assert (resid >= 0).all()


def test_one_token_row_uses_only_the_generated_token():
    lp = dists(1, 2)
    o = fake_output([5], lp, "dict")
    o.prompt_logprobs = [None]
    ids, _, _ = vc.convert_row(o.prompt_logprobs, o.outputs[0].logprobs, 1, K, np)
    assert (ids == expected(lp)[0]).all()


def test_two_dimensional_pair_shape():
    lp = dists(4, 3)
    e_ids, e_top, _ = expected(lp, K + 1)
    pair_prompt = (np.vstack([np.zeros((1, K + 1), dtype=np.int64), e_ids[:3]]), np.vstack([np.zeros((1, K + 1), dtype=np.float32), e_top[:3]]))
    gen = (e_ids[3:4], e_top[3:4])
    # position 0 of a prompt is empty in vLLM; here it carries zeros that convert_row never reads
    ids, logp, _ = vc.convert_row(pair_prompt, gen, 4, K, np)
    assert (ids[0] == e_ids[0][:K]).all() and (ids[3] == e_ids[3][:K]).all()


@pytest.mark.parametrize("bad", ["text", 5, {"x": 1}, [1, 2], [{1: "a"}]])
def test_unknown_shapes_fail_loudly(bad):
    with pytest.raises(vc.CacheScriptError) as e:
        vc.convert_row(bad, [{1: -1.0, 2: -2.0, 3: -3.0, 4: -4.0}], 3, K, np)
    assert str(e.value) and "vLLM" in str(e.value) or "Position" in str(e.value)


def test_too_few_entries_and_missing_pieces_are_plain_errors():
    toks = [1, 2, 3]
    lp = dists(3, 4)
    o = fake_output(toks, lp, "dict")
    with pytest.raises(vc.CacheScriptError, match="max_logprobs"):
        vc.convert_row(o.prompt_logprobs, o.outputs[0].logprobs, 3, K + 2, np)
    with pytest.raises(vc.CacheScriptError, match="generated token"):
        vc.convert_row(o.prompt_logprobs, None, 3, K, np)
    with pytest.raises(vc.CacheScriptError, match="prompt positions"):
        vc.convert_row(o.prompt_logprobs[:2], o.outputs[0].logprobs, 3, K, np)
    o.prompt_logprobs[2] = None
    with pytest.raises(vc.CacheScriptError, match="no log-probabilities"):
        vc.convert_row(o.prompt_logprobs, o.outputs[0].logprobs, 3, K, np)


def test_plan_batches_by_token_budget():
    lengths = [10, 10, 10, 50, 5, 5, 200, 3]
    spans = vc.plan_batches(lengths, 0, 8, 30)
    assert spans == [(0, 3), (3, 4), (4, 6), (6, 7), (7, 8)]
    assert vc.plan_batches(lengths, 3, 8, 1000, max_rows=2) == [(3, 5), (5, 7), (7, 8)]
    assert vc.plan_batches(lengths, 8, 8, 30) == []
    assert all(sum(lengths[a:b]) <= 30 or b - a == 1 for a, b in spans)


# ---- the job with a fake engine

class FakeScorer:
    """Answers like vLLM would for a toy model: the distribution after position i is a function of the token ids so far."""

    def __init__(self, shape="flat"):
        self.calls, self.shape = [], shape

    def score(self, rows):
        self.calls.append([len(r) for r in rows])
        return [fake_output(r, self.lp(r), self.shape) for r in rows]

    @staticmethod
    def lp(r):
        return dists(len(r), sum(r) % 1000)


def make_rows(folder, lens, seed=0):
    rng = np.random.default_rng(seed)
    toks = [rng.integers(0, V, size=n).astype(np.int32) for n in lens]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.savez(Path(folder) / "rows.npz", ids=np.concatenate(toks), mask=np.zeros(off[-1], dtype=np.uint8), offsets=off, general=np.zeros(len(lens), dtype=np.uint8),
             kind=np.zeros(len(lens), dtype=np.int16), kinds=np.array([""]))
    return toks


def args_for(tmp_path, **kw):
    (tmp_path / "meta.json").write_text(json.dumps({"teacher": "t", "k": K, "rows": 0, "data_hash": "h", "tokenizer_hash": "x", "max_len": 64, "general_key": {}, "general_rows": 0}))
    d = dict(model="m", data=str(tmp_path), out=str(tmp_path / "cache"), meta=str(tmp_path / "meta.json"), k=K, max_seq=64, first_row=0, last_row=None, part_tokens=40, batch_tokens=60,
             max_batch_rows=256, gpu_util=0.88, quantization="", max_num_batched_tokens=0, llm_kwargs="", stop_file="")
    d.update(kw)
    return types.SimpleNamespace(**d)


def read_all(folder):
    r = distilljob.CacheReader(folder)
    return [r.get(i) for i in range(r.done_rows)]


@pytest.mark.parametrize("shape", ["flat", "dict"])
def test_run_writes_the_cache_rows_in_order(tmp_path, shape):
    toks = make_rows(tmp_path, [12, 30, 7, 25, 9])
    sc = FakeScorer(shape)
    res = vc.run(args_for(tmp_path), sc, np)
    assert res["done_rows"] == 5 and not res["stopped"] and res["tokens"] == 83
    assert all(sum(c) <= 60 or len(c) == 1 for c in sc.calls) and len(sc.calls) >= 2
    got = read_all(tmp_path / "cache")
    for t, (ids, logp, resid) in zip(toks, got):
        e_ids, e_top, e_res = expected(FakeScorer.lp(t.tolist()))
        assert (ids == e_ids).all() and np.allclose(logp.astype(np.float32), e_top, atol=2e-3) and np.allclose(resid, e_res, atol=1e-6)
    idx = json.loads((tmp_path / "cache" / "index.json").read_text())
    assert idx["complete"] is True and idx["rows"] == 5 and idx["format"] == cachewrite.CACHE_FORMAT


def test_run_resumes_after_a_stop(tmp_path, capsys):
    make_rows(tmp_path, [20, 20, 20, 20, 20, 20])
    stop = tmp_path / "STOP"

    class StoppingScorer(FakeScorer):
        def score(self, rows):
            out = super().score(rows)
            stop.write_text("1")                                  # the Stop button is pressed during the first batch
            return out

    a = args_for(tmp_path, stop_file=str(stop), part_tokens=20, batch_tokens=40)
    r1 = vc.run(a, StoppingScorer(), np)
    assert r1["stopped"] and 0 < r1["done_rows"] < 6
    stop.unlink()
    sc = FakeScorer()
    r2 = vc.run(a, sc, np)
    assert r2["done_rows"] == 6 and not r2["stopped"]
    assert sum(len(c) for c in sc.calls) == 6 - r1["done_rows"], "the rows that were saved are not scored again"
    assert json.loads((tmp_path / "cache" / "index.json").read_text())["complete"] is True
    r3 = vc.run(a, FakeScorer(), np)
    assert r3["resumed"] and r3["tokens_scored"] == 0


def test_run_a_chunk_of_rows(tmp_path):
    toks = make_rows(tmp_path, [10, 11, 12, 13])
    res = vc.run(args_for(tmp_path, first_row=1, last_row=3), FakeScorer(), np)
    assert res["rows"] == 2 and res["done_rows"] == 2
    got = read_all(tmp_path / "cache")
    assert (got[0][0] == expected(FakeScorer.lp(toks[1].tolist()))[0]).all()


def test_stdout_lines_are_protocol_lines(tmp_path, capsys):
    make_rows(tmp_path, [15, 15])
    vc.run(args_for(tmp_path, batch_tokens=15), FakeScorer(), np)
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    prog = [x for x in lines if x["type"] == "progress"]
    assert prog and prog[-1]["step"] == "cache" and prog[-1]["detail"]["unit"] == "tokens" and prog[-1]["detail"]["total"] == 30
    assert any(x["type"] == "log" and x["text"].startswith("timing: ") for x in lines)


def test_make_detail_and_lines_equal_the_protocol(capsys):
    for a in [("tokens", 5, 10), ("tokens", 5, 10, 1.0, 3.0, "tokens/s", 1)]:
        assert vc.make_detail(*a) == protocol.make_detail(*a)
    vc.progress("cache", 0.5, "x", {"unit": "tokens", "done": 1.0, "total": 2.0})
    vc.log("hello")
    mine = capsys.readouterr().out
    import io
    buf = io.StringIO()
    old = protocol._out
    protocol._out = buf
    try:
        protocol.progress("cache", 0.5, "x", {"unit": "tokens", "done": 1.0, "total": 2.0})
        protocol.log("hello")
    finally:
        protocol._out = old
    assert mine == buf.getvalue()


def test_main_reports_a_plain_error(tmp_path, capsys):
    make_rows(tmp_path, [5])

    class Bad:
        def score(self, rows):
            return [types.SimpleNamespace(prompt_logprobs="what", outputs=[types.SimpleNamespace(logprobs=None)])]

    with pytest.raises(vc.CacheScriptError):
        vc.run(args_for(tmp_path), Bad(), np)


def test_script_imports_only_numpy_vllm_and_cachewrite():
    src = Path(vc.__file__).read_text(encoding="utf-8")
    assert "import torch" not in src and "from . import" not in src.replace("from . import cachewrite", "")
    assert "import protocol" not in src
