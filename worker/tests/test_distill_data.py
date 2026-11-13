"""Healing by distillation: the data, the order of the rows and the teacher cache on disk (numpy only, no torch, no network)."""
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import data as data_mod  # noqa: E402
from ndforge_worker import distilljob as dj  # noqa: E402
from ndforge_worker.reshape import ReshapeError  # noqa: E402


class CharTok:
    """a stand-in tokenizer: one token per character (offset by 1)"""
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 250 + 1 for c in text]}


def jsonl(path, items):
    Path(path).write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    return str(path)


def test_hashes_follow_the_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, tk in ((a, '{"v": 1}'), (b, '{"v": 2}')):
        d.mkdir()
        (d / "tokenizer.json").write_text(tk)
    assert dj.tokenizer_hash(a) == dj.tokenizer_hash(a) and dj.tokenizer_hash(a) != dj.tokenizer_hash(b)
    (a / "tokenizer_config.json").write_text("{}")
    assert dj.tokenizer_hash(a) != dj.tokenizer_hash(b)
    rows = [{"input_ids": [1, 2, 3]}, {"input_ids": [4]}]
    assert dj.data_hash(rows) == dj.data_hash([{"input_ids": [1, 2, 3]}, {"input_ids": [4]}])
    assert dj.data_hash(rows) != dj.data_hash([{"input_ids": [1, 2]}, {"input_ids": [3, 4]}])         # the row boundaries count
    assert dj.data_hash(rows) != dj.data_hash(rows[::-1])


def test_general_rows_cut_windows_and_stop_at_the_budget(tmp_path):
    f = jsonl(tmp_path / "g.jsonl", [{"text": "a" * 50}, {"text": "short"}, {"text": [{"role": "user", "content": "b" * 20}, {"role": "assistant", "content": "c" * 20}]},
                                    {"text": "d" * 100}])
    rows, tokens = dj.general_rows({"file": f}, lambda t: CharTok()(t)["input_ids"], max_seq=32, budget=10_000)
    lens = [len(r["input_ids"]) for r in rows]
    # "a" x 50 -> 32 + 18; "short" is dropped; the two chat turns make 41 characters -> 32 (+ 9, dropped); "d" x 100 -> 32 x 3 (+ 4, dropped)
    assert lens == [32, 18, 32, 32, 32, 32] and tokens == sum(lens)
    assert all(r["source"] == "general" and r["kind"] == "general" and r["completion_mask"] == [1] * len(r["input_ids"]) for r in rows)
    assert min(lens) >= dj.MIN_GENERAL_ROW
    capped, t2 = dj.general_rows({"file": f}, lambda t: CharTok()(t)["input_ids"], max_seq=32, budget=40)
    assert t2 >= 40 and len(capped) == 2 and sum(len(r["input_ids"]) for r in capped) == t2
    other, _ = dj.general_rows({"file": jsonl(tmp_path / "h.jsonl", [{"body": "x" * 40}]), "column": "body"}, lambda t: CharTok()(t)["input_ids"], 64, 100)
    assert len(other) == 1


def patch_project(monkeypatch, n=3, length=20):
    rows = [{"input_ids": list(range(1, length + 1)), "completion_mask": [1] * length, "kind": "chat"} for _ in range(n)]
    monkeypatch.setattr(data_mod, "load_training_rows", lambda tok, path, fam, max_len, seed: (rows, {"tokens": n * length}))
    return rows


def test_build_rows_adds_general_text_up_to_its_share(tmp_path, monkeypatch):
    patch_project(monkeypatch)
    g = jsonl(tmp_path / "g.jsonl", [{"text": "g" * 40}] * 10)
    rows, st = dj.build_rows(CharTok(), {"train": "x", "tokens": 200, "general": {"share": 0.5, "file": g}}, "qwen3_5", 64)
    assert st["project_rows"] == 3 and st["project_tokens"] == 60 and st["general_error"] is None and st["general_tokens"] >= 100
    assert [r["source"] for r in rows[:3]] == ["project"] * 3 and all(r["source"] == "general" for r in rows[3:])
    assert st["general_rows"] == len(rows) - 3
    none, st0 = dj.build_rows(CharTok(), {"train": "x", "tokens": 200, "general": None}, "qwen3_5", 64)
    assert len(none) == 3 and st0["general_rows"] == 0 and st0["general_source"] is None
    zero, _ = dj.build_rows(CharTok(), {"train": "x", "tokens": 200, "general": {"share": 0, "file": g}}, "qwen3_5", 64)
    assert len(zero) == 3


def test_offline_general_text_falls_back_to_the_project_text_with_a_plain_message(tmp_path, monkeypatch):
    patch_project(monkeypatch)

    def offline(source):
        raise OSError("Couldn't reach 'huggingface.co' (offline)")
        yield  # pragma: no cover
    monkeypatch.setattr(dj, "_general_texts", offline)
    rows, st = dj.build_rows(CharTok(), {"train": "x", "tokens": 200, "general": {"share": 0.5, "id": "org/data"}}, "qwen3_5", 64)
    assert len(rows) == 3 and st["general_rows"] == 0
    assert "(org/data) could not be downloaded" in st["general_error"] and "your world's text only" in st["general_error"] and "huggingface.co" in st["general_error"]
    rows2, st2 = dj.build_rows(CharTok(), {"train": "x", "tokens": 200, "general": {"share": 0.5, "file": str(tmp_path / "missing.jsonl")}}, "qwen3_5", 64)
    assert len(rows2) == 3 and "could not be downloaded" in st2["general_error"]


def test_training_order_keeps_the_token_shares_and_is_reproducible():
    rows = [{"input_ids": [1] * 10, "source": "project"} for _ in range(5)] + [{"input_ids": [1] * 20, "source": "general"} for _ in range(30)]
    order = dj.training_order(rows, 2000, 0.5, seed=1, block=4)
    toks = {"project": 0, "general": 0}
    for i in order:
        toks[rows[i]["source"]] += len(rows[i]["input_ids"])
    assert 1000 <= toks["project"] < 1010 and 1000 <= toks["general"] < 1020          # the small project set is repeated to fill its half
    assert order == dj.training_order(rows, 2000, 0.5, seed=1, block=4) and order != dj.training_order(rows, 2000, 0.5, seed=2, block=4)
    assert set(order) <= set(range(35)) and len({i for i in order if i < 5}) == 5
    only = dj.training_order(rows[:5], 300, 0.5, seed=1)
    assert len(only) == 30 and set(only) == set(range(5))                                # no general rows: all tokens are the project's
    assert dj.training_order(rows[5:], 100, 0.5, seed=1) and all(i < 30 for i in dj.training_order(rows[5:], 100, 0.5))
    assert dj.training_order([], 100, 0.5) == []


# ------------------------------------------------------------------------------------------------------------ the cache

K = 3


def fake_row(n, seed):
    r = np.random.default_rng(seed)
    return r.integers(0, 100, (n, K)), r.normal(size=(n, K)).astype(np.float16), r.random(n).astype(np.float32)


META = {"teacher": "org/Teacher", "k": K, "rows": 5, "data_hash": "h1", "tokenizer_hash": "t1", "max_len": 64}


def test_cache_round_trip_in_parts(tmp_path):
    rows = [fake_row(n, i) for i, n in enumerate([4, 3, 5, 2, 6])]
    w = dj.CacheWriter(tmp_path / "c", META, part_tokens=7)
    for r in rows:
        w.add_row(*r)
    w.close()
    idx = json.loads((tmp_path / "c" / "index.json").read_text())
    assert idx["format"] == 1 and idx["done_rows"] == 5 and idx["tokens"] == 20 and idx["complete"] is True and idx["teacher"] == "org/Teacher"
    assert [p["rows"] for p in idx["parts"]] == [2, 2, 1] and [p["first_row"] for p in idx["parts"]] == [0, 2, 4]
    assert sorted(f.name for f in (tmp_path / "c").glob("part-*.npz")) == ["part-00001.npz", "part-00002.npz", "part-00003.npz"]
    rd = dj.CacheReader(tmp_path / "c")
    assert rd.done_rows == 5
    for i in (3, 0, 4, 1, 2, 0):                                              # in any order, the parts are loaded as needed
        ids, lp, rs = rd.get(i)
        assert (ids == rows[i][0]).all() and (lp == rows[i][1]).all() and (rs == rows[i][2]).all() and ids.dtype == np.int64 and lp.dtype == np.float16
    with pytest.raises(ReshapeError, match="row 5"):
        rd.get(5)
    with np.load(tmp_path / "c" / "part-00001.npz") as z:
        assert z["ids"].dtype == np.uint32 and z["logp"].dtype == np.float16 and z["resid"].dtype == np.float32 and z["row_offsets"].tolist() == [0, 4, 7]


def test_a_stopped_cache_continues_and_a_cache_of_other_data_starts_again(tmp_path):
    rows = [fake_row(4, i) for i in range(5)]
    w = dj.CacheWriter(tmp_path / "c", META, part_tokens=8)
    for r in rows[:3]:
        w.add_row(*r)
    w.close()                                                                  # stopped after 3 rows: two parts (rows 0-1, row 2)
    idx = json.loads((tmp_path / "c" / "index.json").read_text())
    assert idx["done_rows"] == 3 and idx["complete"] is False
    w2 = dj.CacheWriter(tmp_path / "c", META, part_tokens=8)
    assert w2.done_rows == 3 and w2.tokens == 12
    for r in rows[3:]:
        w2.add_row(*r)
    w2.close()
    rd = dj.CacheReader(tmp_path / "c")
    assert rd.done_rows == 5 and all((rd.get(i)[0] == rows[i][0]).all() for i in range(5))
    assert json.loads((tmp_path / "c" / "index.json").read_text())["complete"] is True
    # the same folder for other data: the old answers are thrown away
    w3 = dj.CacheWriter(tmp_path / "c", {**META, "data_hash": "other"}, part_tokens=8)
    assert w3.done_rows == 0 and not list((tmp_path / "c").glob("part-*.npz"))


def test_cache_matches_every_field_that_matters():
    idx = {**META, "done_rows": 1, "parts": []}
    assert dj.cache_matches(idx, META)
    for key, other in (("teacher", "x"), ("k", 4), ("rows", 6), ("data_hash", "h2"), ("tokenizer_hash", "t2"), ("max_len", 32)):
        assert not dj.cache_matches(idx, {**META, key: other})
    assert dj.read_cache_index(Path("does-not-exist")) is None


def test_an_unreadable_cache_says_it_will_be_made_again(tmp_path):
    (tmp_path / "index.json").write_text("{not json")
    with pytest.raises(ReshapeError, match="made again"):
        dj.CacheReader(tmp_path)
