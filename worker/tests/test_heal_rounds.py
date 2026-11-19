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


# ------------------------------------------------------------------------------------------------------------ W2: the schedule

SCH = dj.DEFAULT_ROUND_SCHEDULE


def factors(index, steps, s_start=0, warmup=30):
    return [dj.round_lr_factor(s_start + t, s_start, steps, index, warmup, SCH) for t in range(steps)]


def test_round_one_starts_near_zero_holds_the_peak_and_ends_at_the_floor_of_its_cooldown():
    f = factors(1, 40)
    assert f[0] < 0.2 and f[0] == pytest.approx(1 / 10)                       # warm-up: 25 % of 40 steps = 10, not the job's 30
    assert f[9] == pytest.approx(1.0) and f[20] == 1.0                        # the peak is 1 in round 1
    assert f[-1] == pytest.approx(SCH["min_ratio"])
    assert all(f[i] >= f[i + 1] for i in range(31, 39))                       # the cooldown only falls
    assert max(f) == 1.0


def test_a_later_round_has_a_lower_peak_a_short_rewarm_and_the_same_cooldown():
    f2 = factors(2, 40, s_start=40)
    assert f2[0] == pytest.approx(SCH["min_ratio"] * 0.85 + (0.85 - SCH["min_ratio"] * 0.85) / 4)    # re-warm over 10 % of 40 = 4 steps
    assert f2[3] == pytest.approx(0.85) and f2[20] == pytest.approx(0.85)
    assert f2[-1] == pytest.approx(0.85 * SCH["min_ratio"])
    assert dj.round_lr_factor(0, 0, 40, 3, 30, SCH) == pytest.approx(0.85 ** 2 * SCH["min_ratio"] + (0.85 ** 2 - 0.85 ** 2 * SCH["min_ratio"]) / 4)


def test_the_peak_never_falls_below_the_floor():
    for index in range(1, 30):
        assert max(factors(index, 30)) >= SCH["floor"] - 1e-9
    assert max(factors(30, 30, s_start=0)) == pytest.approx(SCH["floor"])
    assert dj.round_lr_factor(5, 0, 30, 12, 30, SCH) == pytest.approx(SCH["floor"])


def test_the_rewarm_is_at_most_ten_steps_and_the_first_warmup_at_most_a_quarter_of_a_short_round():
    f = factors(2, 200, s_start=0)
    assert f[9] == pytest.approx(0.85) and f[8] < 0.85                         # 10 % of 200 = 20, capped at 10
    short = factors(1, 8)                                                       # warm-up min(30, ceil(0.25 * 8)) = 2
    assert short[1] == pytest.approx(1.0) and short[0] == pytest.approx(0.5)
    assert all(0 < x <= 1 for x in factors(1, 1)) and factors(1, 1)[0] > 0


def test_the_factor_is_stable_outside_the_round_and_uses_global_steps():
    assert dj.round_lr_factor(40, 0, 40, 1, 30, SCH) == dj.round_lr_factor(39, 0, 40, 1, 30, SCH)      # the scheduler's look one step past the end
    assert dj.round_lr_factor(77, 40, 40, 2, 30, SCH) == dj.round_lr_factor(37, 0, 40, 2, 30, SCH)


# ------------------------------------------------------------------------------------------------------------ W2: the order of a round

def run_rows(general=40):
    rows = make_rows(project=(6, 9, 4, 7), general=(8,) * general)
    return rows


def test_a_round_has_every_project_row_cycled_and_its_own_general_rows_once():
    rows = run_rows()
    first_g = 4
    order = dj.round_order(rows, first_g, first_g + 10, 200, 0.5, seed=11)
    kinds = [rows[i]["source"] for i in order]
    gen_in = [i for i in order if rows[i]["source"] == "general"]
    assert sorted(set(gen_in)) == list(range(first_g, first_g + 10)) and len(gen_in) == len(set(gen_in)) == 10     # 80 tokens < 100 wanted: each row once
    assert set(i for i in order if rows[i]["source"] != "general") == {0, 1, 2, 3}
    proj_tokens = sum(len(rows[i]["input_ids"]) for i in order if rows[i]["source"] != "general")
    assert proj_tokens >= 200 - 80 and len(kinds) > 10                          # project rows cycle until the round's tokens are there


def test_the_round_tokens_are_about_the_requested_amount():
    rows = run_rows(200)
    for want in (100, 400, 1000):
        order = dj.round_order(rows, 4, 4 + 200, want, 0.5, seed=3)
        tokens = sum(len(rows[i]["input_ids"]) for i in order)
        assert want <= tokens <= want + 2 * 9 + 2 * 8 + 16


def test_no_general_row_repeats_across_consecutive_rounds():
    rows = run_rows(60)
    seen: set = set()
    start = 4
    for n in range(1, 5):
        end = start + 12
        order = dj.round_order(rows, start, end, 300, 0.5, seed=7 + n)
        gen = {i for i in order if rows[i]["source"] == "general"}
        assert gen == set(range(start, end)) and not (gen & seen)
        seen |= gen
        start = end


def test_rounds_with_different_seeds_order_the_project_rows_differently_and_the_same_seed_repeats():
    rows = run_rows(60)
    a = dj.round_order(rows, 4, 16, 300, 0.5, seed=8)
    assert a == dj.round_order(rows, 4, 16, 300, 0.5, seed=8)
    assert a != dj.round_order(rows, 4, 16, 300, 0.5, seed=9)


def test_a_round_without_general_rows_or_without_a_share_uses_the_project_rows_only():
    rows = run_rows(10)
    order = dj.round_order(rows, 4, 4, 100, 0.5, seed=1)
    assert order and all(rows[i]["source"] != "general" for i in order)
    order = dj.round_order(rows, 4, 14, 100, 0.0, seed=1)
    assert all(rows[i]["source"] != "general" for i in order)


def test_a_run_without_project_rows_takes_general_rows_only():
    rows = make_rows(project=(), general=(8,) * 20)
    order = dj.round_order(rows, 0, 10, 60, 0.5, seed=1)
    assert order and all(rows[i]["source"] == "general" and i < 10 for i in order)


def test_slicing_after_k_steps_drops_exactly_k_times_the_rows_of_a_step():
    order = list(range(100))
    assert dj.round_slice(order, 0, 8) == order
    assert dj.round_slice(order, 3, 8) == order[24:]
    assert dj.round_slice(order, 13, 8) == []
    assert dj.round_slice(order, -2, 8) == order and dj.round_slice(order, 2, 0) == order[2:]


def test_the_normal_order_is_unchanged_by_the_shared_helpers():
    rows = make_rows(project=(6, 9, 4, 7), general=(8,) * 30)
    a = dj.training_order(rows, 300, 0.5, 3407)
    assert a == dj.training_order(rows, 300, 0.5, 3407)
    gen = sum(len(rows[i]["input_ids"]) for i in a if rows[i]["source"] == "general")
    assert 150 <= gen < 160


# ------------------------------------------------------------------------------------------------------------ W3: where a round starts from

def fake_checkpoint(ckdir, step):
    d = ckdir / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"x")
    (d / "optimizer.pt").write_bytes(b"x")
    return d


def test_round_one_starts_from_scratch_or_its_own_checkpoints(tmp_path):
    r = dj.round_start(tmp_path / "ck", 1, 0, tmp_path)
    assert r == {"mode": "continue", "gs_start": 0, "init_adapter": None, "note": None}


def test_a_later_round_continues_from_the_checkpoint_of_the_end_of_the_previous_round(tmp_path):
    ck = tmp_path / "checkpoints"
    fake_checkpoint(ck, 38)
    r = dj.round_start(ck, 2, 38, tmp_path)
    assert r["mode"] == "continue" and r["gs_start"] == 38 and r["init_adapter"] is None
    fake_checkpoint(ck, 50)                                      # one from inside the round
    assert dj.round_start(ck, 2, 38, tmp_path)["mode"] == "continue"
    assert (ck / "checkpoint-38").is_dir() and (ck / "checkpoint-50").is_dir()


def test_without_a_checkpoint_of_this_round_the_previous_adapter_is_the_start_and_the_old_checkpoints_go(tmp_path):
    ck = tmp_path / "checkpoints"
    fake_checkpoint(ck, 20)                                      # from before the round: too old to continue the optimizer of round 1's end
    (tmp_path / "rounds" / "r1" / "adapter").mkdir(parents=True)
    r = dj.round_start(ck, 2, 38, tmp_path)
    assert r["mode"] == "fresh" and r["gs_start"] == 0 and r["init_adapter"] == str(tmp_path / "rounds" / "r1" / "adapter") and "fresh optimizer" in r["note"]
    assert not (ck / "checkpoint-20").exists() and (ck / "FRESH_ROUND").read_text() == "2"
    # a restart inside that round trusts its own (locally counted) checkpoints and does not delete them
    fake_checkpoint(ck, 5)
    again = dj.round_start(ck, 2, 38, tmp_path)
    assert again["mode"] == "fresh" and again["note"] is None and (ck / "checkpoint-5").is_dir()
    # the next round is not "fresh" because of an old marker
    fake_checkpoint(ck, 60)
    assert dj.round_start(ck, 3, 55, tmp_path)["mode"] == "continue"


def test_a_later_round_with_neither_checkpoint_nor_previous_adapter_is_refused_plainly(tmp_path):
    from ndforge_worker.reshape import ReshapeError
    with pytest.raises(ReshapeError, match="Heal again from the start"):
        dj.round_start(tmp_path / "checkpoints", 3, 80, tmp_path)
