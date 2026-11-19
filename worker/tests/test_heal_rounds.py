"""Healing in rounds: the pure helpers (numpy only, no torch, no network)."""
import json

import pytest

np = pytest.importorskip("numpy")

from ndforge_worker import distilljob as dj  # noqa: E402


def make_rows(project=(5, 7), general=(10, 10, 10, 10)):
    rows = [{"input_ids": list(range(n)), "completion_mask": [1] * n, "kind": "p", "source": "project"} for n in project]
    rows += [{"input_ids": list(range(n)), "completion_mask": [1] * n, "kind": "g", "source": "general"} for n in general]
    return rows


# ------------------------------------------------------------------------------------------------------------ W1: rows_upto

def test_rows_upto_always_includes_every_project_row():
    rows = make_rows()
    assert dj.rows_upto(rows, 0) == 2 and dj.rows_upto(rows, -5) == 2 and dj.rows_upto(rows, None) == 2


def test_rows_upto_stops_exactly_at_the_boundary_and_includes_the_row_that_reaches_it():
    rows = make_rows()
    assert dj.rows_upto(rows, 10) == 3          # exactly one general row
    assert dj.rows_upto(rows, 11) == 4          # the second row is the one that reaches 11
    assert dj.rows_upto(rows, 1) == 3
    assert dj.rows_upto(rows, 30) == 5
    assert dj.rows_upto(rows, 40) == 6 and dj.rows_upto(rows, 10_000) == 6


def test_rows_upto_without_general_rows_is_everything():
    rows = make_rows(general=())
    assert dj.rows_upto(rows, 100) == 2 and dj.rows_upto([], 5) == 0


def test_a_budget_gives_a_prefix_of_a_bigger_one():
    rows = make_rows(general=(10,) * 20)
    prev = 0
    for g in range(0, 220, 7):
        n = dj.rows_upto(rows, g)
        assert n >= prev
        prev = n


# ------------------------------------------------------------------------------------------------------------ W1: reusing the saved rows

def tok_dir(tmp_path):
    d = tmp_path / "student"
    d.mkdir()
    (d / "tokenizer.json").write_text('{"v": 1}')
    return d


def cache_with_rows(tmp_path, rows, **over):
    student = tok_dir(tmp_path)
    cfg = {"teacher": {"repo": "T"}, "student": str(student), "model": {"family": "generic"}, "k": 4, "max_seq": 64, "rows_key": "key-1", "seed": 3407,
           "data": {"train": "x", "tokens": 100, "general": {"share": 0.5, "file": "g.jsonl"}}, **over}
    stats = {"project_rows": 2, "general_rows": 4, "general_source": "g.jsonl"}
    out = tmp_path / "cache"
    meta = dj.cache_meta(cfg, rows, stats, student, 4, 64)
    w = dj.CacheWriter(out, meta)
    w.close()
    dj.save_rows(out, rows)
    return cfg, out, student


def test_saved_rows_are_reused_only_when_everything_that_made_them_matches(tmp_path):
    rows = make_rows()
    cfg, out, student = cache_with_rows(tmp_path, rows)
    got = dj._reusable_rows(out, cfg, 4, 64, student)
    assert got is not None
    again, stats = got
    assert dj.data_hash(again) == dj.data_hash(rows) and stats["project_rows"] == 2 and stats["general_rows"] == 4 and stats["general_error"] is None
    assert [r["source"] for r in again] == [r["source"] for r in rows]
    # another rows_key, k, length, teacher, general text or tokenizer: the rows are built again
    assert dj._reusable_rows(out, {**cfg, "rows_key": "key-2"}, 4, 64, student) is None
    assert dj._reusable_rows(out, {**cfg, "rows_key": None}, 4, 64, student) is None
    assert dj._reusable_rows(out, cfg, 8, 64, student) is None
    assert dj._reusable_rows(out, cfg, 4, 32, student) is None
    assert dj._reusable_rows(out, {**cfg, "teacher": {"repo": "U"}}, 4, 64, student) is None
    assert dj._reusable_rows(out, {**cfg, "data": {**cfg["data"], "general": {"share": 0.5, "file": "other.jsonl"}}}, 4, 64, student) is None
    (student / "tokenizer.json").write_text('{"v": 2}')
    assert dj._reusable_rows(out, cfg, 4, 64, student) is None


def test_damaged_saved_rows_are_not_reused(tmp_path):
    rows = make_rows()
    cfg, out, student = cache_with_rows(tmp_path, rows)
    (out / "rows.npz").write_bytes(b"not an archive")
    assert dj._reusable_rows(out, cfg, 4, 64, student) is None
    (out / "rows.npz").unlink()
    assert dj._reusable_rows(out, cfg, 4, 64, student) is None
    assert dj._reusable_rows(tmp_path / "nothing", cfg, 4, 64, student) is None
    idx = json.loads((out / "index.json").read_text())
    assert idx["rows_key"] == "key-1"
