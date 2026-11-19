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


def test_rows_are_saved_next_to_the_cache_and_come_back_unchanged(tmp_path):
    rows = [{"input_ids": [5, 6, 7], "completion_mask": [0, 1, 1], "kind": "chat", "source": "project"},
            {"input_ids": [8], "completion_mask": [1], "kind": "text", "source": "project"},
            {"input_ids": [9, 10], "completion_mask": [1, 1], "kind": "general", "source": "general"}]
    dj.save_rows(tmp_path, rows)
    assert dj.load_rows(tmp_path) == rows and dj.data_hash(dj.load_rows(tmp_path)) == dj.data_hash(rows)
    with pytest.raises(ReshapeError, match="made again"):
        dj.load_rows(tmp_path / "nothing")


class ChatCharTok(CharTok):
    eos_token = "<e>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        return "".join(f"{m['content']}\n" for m in messages) + ("" if not add_generation_prompt else "")


def chat_row(user_chars, answer_chars):
    return {"messages": [{"role": "user", "content": "q" * user_chars}, {"role": "assistant", "content": "a" * answer_chars}]}


def test_a_chat_conversation_is_one_row_cut_at_the_length_and_a_very_long_one_is_skipped(tmp_path):
    # plain text of the same length is windowed (below); a conversation is "q"*u + "\n" + "a"*a characters = one token each
    f = jsonl(tmp_path / "c.jsonl", [chat_row(10, 39), chat_row(10, 20), chat_row(30, 40), chat_row(5, 6), chat_row(100, 100)])
    stats: dict = {}
    rows, tokens = dj.general_rows({"file": f, "report": {}}, lambda t: CharTok()(t)["input_ids"], max_seq=32, budget=10_000, stats=stats)
    lens = [len(r["input_ids"]) for r in rows]
    # 50 chars -> cut to 32 (no second window); 31 chars -> 31; 71 chars is more than twice 32 -> skipped; 12 chars -> too short; 201 -> skipped
    assert lens == [32, 31] and tokens == 63 and {k: stats[k] for k in ("cut_rows", "skipped_long")} == {"cut_rows": 1, "skipped_long": 2}
    assert sum(stats["length_buckets"].values()) == 5, "the skipped and the too short ones are counted by length too"
    assert all(r["kind"] == "general" and r["completion_mask"] == [1] * len(r["input_ids"]) for r in rows)


def test_plain_text_is_still_windowed_and_a_text_column_of_turns_counts_as_text(tmp_path):
    f = jsonl(tmp_path / "t.jsonl", [{"text": "a" * 100}])
    stats: dict = {}
    rows, _ = dj.general_rows({"file": f, "report": {}}, lambda t: CharTok()(t)["input_ids"], max_seq=32, budget=10_000, stats=stats)
    assert [len(r["input_ids"]) for r in rows] == [32, 32, 32] and {k: stats[k] for k in ("cut_rows", "skipped_long")} == {"cut_rows": 0, "skipped_long": 0}


def test_the_conversation_rule_does_not_need_a_stats_dict_or_a_report(tmp_path):
    f = jsonl(tmp_path / "c.jsonl", [chat_row(10, 39)])
    rows, _ = dj.general_rows({"file": f}, lambda t: CharTok()(t)["input_ids"], max_seq=32, budget=10_000)
    assert [len(r["input_ids"]) for r in rows] == [32]                                        # the report is filled by the reader itself, so the rule holds without one


def test_build_rows_counts_the_long_conversations_and_says_so(tmp_path, monkeypatch):
    patch_project(monkeypatch)
    g = jsonl(tmp_path / "g.jsonl", [chat_row(10, 39), chat_row(100, 100), chat_row(10, 20)] * 4)
    rows, st = dj.build_rows(ChatCharTok(), {"train": "x", "tokens": 400, "general": {"share": 0.5, "file": g}}, "generic", 32)
    assert st["general_cut_rows"] >= 1 and st["general_skipped_long"] >= 1 and st["general_error"] is None
    assert all(len(r["input_ids"]) <= 32 for r in rows)


def test_the_cache_index_keeps_the_teacher_load_and_the_rows_key_but_they_do_not_decide_a_match(tmp_path):
    (tmp_path / "tokenizer.json").write_text('{"v": 1}')
    rows = [{"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]}]
    stats = {"project_rows": 1, "general_rows": 0, "general_source": None}
    cfg = {"teacher": {"repo": "org/T", "load": "4bit"}, "rows_key": "abc123", "data": {"train": "x"}}
    meta = dj.cache_meta(cfg, rows, stats, tmp_path, 8, 128)
    assert meta["teacher_load"] == "4bit" and meta["rows_key"] == "abc123" and meta["k"] == 8 and meta["max_len"] == 128
    old = dj.cache_meta({"teacher": {"repo": "org/T"}, "data": {}}, rows, stats, tmp_path, 8, 128)
    assert old["teacher_load"] == "auto" and old["rows_key"] is None
    assert dj.cache_matches(meta, {**meta, "rows_key": "other", "teacher_load": "auto"}), "the program judges these two before it starts the job"
    w = dj.CacheWriter(tmp_path / "c", meta, part_tokens=8)
    w.close()
    idx = json.loads((tmp_path / "c" / "index.json").read_text())
    assert idx["teacher_load"] == "4bit" and idx["rows_key"] == "abc123"


# ------------------------------------------------------------------------------------------------------------ the shared cache

def rows_of(lengths, start=0):
    return [{"input_ids": [start + i + 1] * n} for i, n in enumerate(lengths)]


def shared_meta(rows, **over):
    return {**META, "rows": len(rows), "data_hash": dj.data_hash(rows), **over}


def fill(store, view, rows, only=None, part_tokens=8):
    """a SharedCacheWriter that answers `rows` (the answers are made from the row's own ids, so a row has the same answer in every cache)"""
    meta = shared_meta(rows)
    hashes = dj.row_hashes(rows)
    w = dj.SharedCacheWriter(store, view, meta, hashes, part_tokens)
    made = []
    for i in range(w.done_rows, len(rows) if only is None else only):
        made.append(i)
        w.add_row(*fake_row(len(rows[i]["input_ids"]), rows[i]["input_ids"][0]))
    w.close()
    return w, made


def test_a_bigger_run_only_adds_the_rows_the_smaller_one_did_not_have(tmp_path):
    base = tmp_path / "shared"
    small, big = rows_of([4, 4, 4, 4]), rows_of([4, 4, 4, 4, 4, 4, 4])
    meta_s, meta_b = shared_meta(small), shared_meta(big)
    a = dj.pick_store(base, "id1", meta_s, dj.row_hashes(small))
    w1, made1 = fill(a, tmp_path / "m1" / "cache", small)
    assert made1 == [0, 1, 2, 3] and w1.done_rows == 4
    b = dj.pick_store(base, "id1", meta_b, dj.row_hashes(big))
    assert b == a, "the same teacher and the first rows are the same: the same shared folder"
    w2, made2 = fill(b, tmp_path / "m2" / "cache", big)
    assert made2 == [4, 5, 6], "only the new rows are answered"
    # both views read their rows back, from the parts in the shared folder
    r1, r2 = dj.CacheReader(tmp_path / "m1" / "cache"), dj.CacheReader(tmp_path / "m2" / "cache")
    for i in range(4):
        assert (r1.get(i)[0] == r2.get(i)[0]).all() and (r1.get(i)[0] == fake_row(4, small[i]["input_ids"][0])[0]).all()
    assert (r2.get(6)[0] == fake_row(4, 7)[0]).all()
    v1, v2 = (json.loads((tmp_path / m / "cache" / "index.json").read_text()) for m in ("m1", "m2"))
    assert v1["rows"] == 4 and v1["complete"] and v1["store"] and v2["rows"] == 7 and v2["complete"] and v1["data_hash"] != v2["data_hash"]
    assert not list((tmp_path / "m1" / "cache").glob("part-*.npz")), "no heavy file in the model's own folder"
    assert sum(1 for _ in base.iterdir()) == 1


def test_a_smaller_run_after_a_bigger_one_needs_nothing_new(tmp_path):
    base = tmp_path / "shared"
    big, small = rows_of([4] * 7), rows_of([4] * 4)
    a = dj.pick_store(base, "id1", shared_meta(big), dj.row_hashes(big))
    fill(a, tmp_path / "m1" / "cache", big)
    assert dj.pick_store(base, "id1", shared_meta(small), dj.row_hashes(small)) == a
    w, made = fill(a, tmp_path / "m2" / "cache", small)
    assert made == [] and w.done_rows >= 4
    v = json.loads((tmp_path / "m2" / "cache" / "index.json").read_text())
    assert v["complete"] and v["rows"] == 4 and all(p["first_row"] < 4 for p in v["parts"])
    assert dj.CacheReader(tmp_path / "m2" / "cache").done_rows >= 4
    # the bigger model's view is untouched
    assert json.loads((tmp_path / "m1" / "cache" / "index.json").read_text())["rows"] == 7


def test_other_rows_or_another_teacher_get_a_new_folder_and_the_old_one_stays_whole(tmp_path):
    base = tmp_path / "shared"
    one = rows_of([4] * 4)
    a = dj.pick_store(base, "id1", shared_meta(one), dj.row_hashes(one))
    fill(a, tmp_path / "m1" / "cache", one)
    before = sorted(p.name for p in a.iterdir())
    other = rows_of([4] * 6, start=100)                                      # not an extension: the first rows differ
    b = dj.pick_store(base, "id1", shared_meta(other), dj.row_hashes(other))
    assert b != a and b.name == "id1-2"
    fill(b, tmp_path / "m2" / "cache", other)
    assert sorted(p.name for p in a.iterdir()) == before, "nothing of the first folder was touched"
    assert (dj.CacheReader(tmp_path / "m1" / "cache").get(0)[0] == fake_row(4, 1)[0]).all()
    assert dj.pick_store(base, "id1", shared_meta(one), dj.row_hashes(one)) == a, "the first rows still find their folder"
    c = dj.pick_store(base, "id1", {**shared_meta(one), "teacher": "org/Other"}, dj.row_hashes(one))
    assert c.name == "id1-3"
    d = dj.pick_store(base, "id1", {**shared_meta(one), "tokenizer_hash": "t2"}, dj.row_hashes(one))
    assert d.name == "id1-3"


def test_a_stopped_shared_cache_goes_on_after_its_last_part(tmp_path):
    base = tmp_path / "shared"
    rows = rows_of([4] * 6)
    store = dj.pick_store(base, "id1", shared_meta(rows), dj.row_hashes(rows))
    _, made = fill(store, tmp_path / "m" / "cache", rows, only=3)
    assert made == [0, 1, 2]
    v = json.loads((tmp_path / "m" / "cache" / "index.json").read_text())
    assert v["complete"] is False and v["done_rows"] == 3
    again = dj.pick_store(base, "id1", shared_meta(rows), dj.row_hashes(rows))
    assert again == store
    _, made = fill(again, tmp_path / "m" / "cache", rows)
    assert made == [3, 4, 5]
    assert json.loads((tmp_path / "m" / "cache" / "index.json").read_text())["complete"] is True


def test_a_damaged_row_hash_file_is_not_trusted(tmp_path):
    base = tmp_path / "shared"
    rows = rows_of([4] * 4)
    store = dj.pick_store(base, "id1", shared_meta(rows), dj.row_hashes(rows))
    fill(store, tmp_path / "m" / "cache", rows)
    (store / "rowhashes.npy").write_bytes(b"junk")
    assert dj.pick_store(base, "id1", shared_meta(rows), dj.row_hashes(rows)).name == "id1-2"


def test_the_store_id_follows_everything_but_the_budget(tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "config.json").write_text('{"a": 1}')
    student = tmp_path / "student"
    student.mkdir()
    (student / "tokenizer.json").write_text('{"v": 1}')
    train = tmp_path / "train.jsonl"
    train.write_text('{"x": 1}\n')

    def sid(**over):
        cfg = {"teacher": {"repo": "t", "load": "auto"}, "model": {"family": "qwen3_5"}, "data": {"train": str(train), "tokens": 1000, "general": {"id": "a/b", "share": 0.5}}, **over}
        return dj.shared_store_id(cfg, teacher, student, 8, 128, 3407)
    base = sid()
    assert sid(data={"train": str(train), "tokens": 9000, "general": {"id": "a/b", "share": 0.2}}) == base, "tokens and share are the budget"
    assert sid(data={"train": str(train), "tokens": 1000, "general": {"id": "a/c", "share": 0.5}}) != base
    assert sid(data={"train": str(train), "tokens": 1000, "general": None}) != base
    assert sid(teacher={"repo": "t", "load": "4bit"}) != base and sid(model={"family": "gemma4"}) != base
    train.write_text('{"x": 2}\n')
    assert sid() != base, "another training file"
    train.write_text('{"x": 1}\n')
    (teacher / "config.json").write_text('{"a": 2}')
    assert sid() != base, "another teacher config"
    assert len(base) == 20


def test_rows_are_answered_longest_first_and_every_row_once():
    window = [{"input_ids": [0] * n} for n in (5, 9, 5, 2, 9, 7)]
    order = dj.length_order(window)
    assert [len(window[j]["input_ids"]) for j in order] == [9, 9, 7, 5, 5, 2]
    assert order == [1, 4, 5, 0, 2, 3], "equal lengths keep their order"
    assert sorted(order) == list(range(6)) and dj.length_order([]) == []


# ------------------------------------------------------------------------------------------------------------ held-out rows

def test_the_held_out_rows_come_after_the_training_rows_and_never_share_a_text(tmp_path):
    f = jsonl(tmp_path / "h.jsonl", [{"text": chr(97 + i % 26) * 40 + str(i)} for i in range(30)])
    kw = {"file": f, "report": {}}
    tok = lambda t: CharTok()(t)["input_ids"]          # noqa: E731
    plain, ptok = dj.general_rows(dict(kw), tok, max_seq=16, budget=100, stats={})
    stats: dict = {}
    rows, tokens = dj.general_rows(dict(kw), tok, max_seq=16, budget=100, stats=stats, holdout=5)
    assert [r["input_ids"] for r in rows][:len(plain) - 0] == [r["input_ids"] for r in plain][:len(rows)] and tokens >= 100
    held = stats["holdout_rows"]
    assert len(held) == 5
    h = lambda rs: {dj.hashlib.sha1(json.dumps(r["input_ids"]).encode()).hexdigest() for r in rs}          # noqa: E731
    assert not h(rows) & h(held), "no held-out row is a training row"
    # a text of several windows gives its training windows only: its later windows are not held out (they sit next to rows that were trained on)
    assert all(r["kind"] == "general" and r["completion_mask"] == [1] * len(r["input_ids"]) for r in held)
    # without the option nothing changes: no key, the same rows
    none: dict = {}
    again, atok = dj.general_rows(dict(kw), tok, max_seq=16, budget=100, stats=none)
    assert "holdout_rows" not in none and [r["input_ids"] for r in again] == [r["input_ids"] for r in plain] and atok == ptok


def test_the_held_out_rows_are_saved_and_read_back_under_their_own_name(tmp_path):
    rows = [{"input_ids": [1, 2, 3], "completion_mask": [1, 1, 1], "kind": "general", "source": "general"}, {"input_ids": [4, 5], "completion_mask": [1, 1], "kind": "general", "source": "general"}]
    dj.save_rows(tmp_path, rows, "holdout")
    assert (tmp_path / "holdout.npz").is_file() and not (tmp_path / "rows.npz").exists()
    back = dj.load_rows(tmp_path, "holdout")
    assert [r["input_ids"] for r in back] == [[1, 2, 3], [4, 5]]
    dj.save_rows(tmp_path, rows)
    assert (tmp_path / "rows.npz").is_file() and dj.load_rows(tmp_path)[0]["input_ids"] == [1, 2, 3]
    assert dj._read_holdout_meta(tmp_path) is None
    (tmp_path / "holdout.json").write_text('{"rows": 2, "teacher_loss": null}')
    assert dj._read_holdout_meta(tmp_path)["rows"] == 2
