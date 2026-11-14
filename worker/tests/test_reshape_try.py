"""The pure parts of "Try before cutting" (no torch): which passes, the cheapest run, colour bands, the cache key, stability, the time estimate."""
import json

import pytest

from ndforge_worker import reshapetry as rt
from ndforge_worker.reshape import ReshapeError

DIMS = {"intermediate": 17408, "heads": 24, "kv_heads": 4, "lin_v_heads": 48, "lin_k_heads": 16}


def test_pass_keys_are_sorted_and_round_trip():
    assert rt.BASE_KEY == "d=|w=" and rt.parse_key(rt.BASE_KEY) == ((), None)
    assert rt.pass_key([5, 3, 3]) == "d=3,5|w=" == rt.pass_key((3, 5))
    assert rt.pass_key((2,), 13312) == "d=2|w=13312" and rt.parse_key("d=2,3|w=13312") == ((2, 3), 13312)
    assert rt.pass_key((), 12288) == "d=|w=12288"


def test_candidates_and_windows_of_the_27b():
    assert rt.candidates(16) == list(range(1, 15)) and len(rt.windows(16, 4)) == 11
    assert rt.windows(16, 4)[0] == (1, 2, 3, 4) and rt.windows(16, 4)[-1] == (11, 12, 13, 14)
    assert rt.windows(16, 14) == [tuple(range(1, 15))] and rt.windows(16, 15) == []
    assert rt.windows(8, 2, 2, 1) == [(2, 3), (3, 4), (4, 5), (5, 6)]
    with pytest.raises(ReshapeError):
        rt.windows(16, 0)


def test_a_chosen_set_follows_the_rules_of_removing_layers():
    assert rt.check_choice([9, 3, 3], 16) == (3, 9) and rt.check_choice([], 16) == ()
    for bad in ([0], [15], [16], [1, 2, 3, 4, 5, 6, 7, 8, 9]):
        with pytest.raises(ReshapeError):
            rt.check_choice(bad, 16)
    with pytest.raises(ReshapeError, match="whole numbers"):
        rt.check_choice(["x"], 16)


def test_expand_want_orders_the_passes_and_asks_for_nothing_twice():
    keys = rt.expand_want({"blocks": True, "runs": [2, 1], "choices": [{"blocks": [3, 4]}]}, 8, 1, 1, dims=None)
    assert keys[0] == rt.BASE_KEY and keys[1] == "d=3,4|w="
    assert keys[2:8] == [f"d={b}|w=" for b in range(1, 7)]
    assert keys.count("d=3,4|w=") == 1 and len(keys) == len(set(keys))          # the choice is also a window of two, tried once
    assert "d=1,2|w=" in keys and "d=5,6|w=" in keys and len(keys) == 1 + 6 + 5      # base, six single blocks (runs of 1 add nothing), five windows of two
    assert rt.expand_want({}, 8, 1, 1) == [rt.BASE_KEY]
    w = rt.expand_want({"widths": [12288, 13312, 12288], "choices": [{"blocks": [2], "width": 12288}]}, 16, 1, 1, dims=DIMS)
    assert w == [rt.BASE_KEY, "d=2|w=12288", "d=|w=12288", "d=|w=13312"]


def test_expand_want_refuses_what_makes_no_sense():
    for want in ({"runs": [0]}, {"runs": [9]}, {"choices": [{"blocks": [0]}]}, {"choices": [{}]}):
        with pytest.raises(ReshapeError):
            rt.expand_want(want, 16, 1, 1, dims=DIMS)
    with pytest.raises(ReshapeError, match="not possible"):
        rt.expand_want({"widths": [12288]}, 16, 1, 1, dims=None)
    with pytest.raises(ReshapeError):
        rt.expand_want({"widths": [12289]}, 16, 1, 1, dims=DIMS)          # not a size the slider offers
    with pytest.raises(ReshapeError):
        rt.expand_want({"widths": [17408]}, 16, 1, 1, dims=DIMS)          # the original's own size


def test_the_separate_strategy_also_asks_for_the_programs_own_choice():
    D = [[abs(a - b) * 0.1 for b in range(9)] for a in range(9)]
    D[2][3] = D[5][6] = 0.001
    keys = rt.expand_want({"runs": [2]}, 8, 1, 1, "separate", D)
    assert "d=2,5|w=" in keys and keys.index("d=2,5|w=") == len(keys) - 1
    assert "d=2,5|w=" not in rt.expand_want({"runs": [2]}, 8, 1, 1, "contiguous", D)


def test_bands_are_cut_at_the_edges_in_one_constant():
    assert rt.BAND_EDGES == (0.03, 0.10) and rt.BAND_NAMES == ("small", "noticeable", "large")
    assert [rt.band_of(x) for x in (None, -0.2, 0.0, 0.0299, 0.03, 0.0999, 0.10, 3.0)] == [None, "small", "small", "small", "noticeable", "noticeable", "large", "large"]
    assert rt.rel_increase(2.2, 2.0) == pytest.approx(0.1) and rt.rel_increase(None, 2.0) is None and rt.rel_increase(2.0, 0) is None
    assert rt.ppl_of(0.0) == 1.0 and rt.ppl_of(None) is None and rt.ppl_of(1.0) == pytest.approx(2.718, abs=1e-3) and rt.ppl_of(1e9) < 1e30


def passes_of(base, losses):
    """{key: pass} with halves that differ a little"""
    out = {rt.BASE_KEY: {"loss": base, "loss_h1": base, "loss_h2": base, "tokens": 1000, "seconds": 2.0}}
    for k, v in losses.items():
        out[k] = {"loss": v, "loss_h1": v + 0.001, "loss_h2": v - 0.001, "tokens": 1000, "seconds": 4.0}
    return out


def test_rows_report_loss_delta_relative_increase_and_perplexity():
    p = passes_of(2.0, {"d=3|w=": 2.1})
    r = rt.row_of(p, "d=3|w=", 2.0)
    assert (r["tried"], r["loss"]) == (True, 2.1) and r["delta"] == pytest.approx(0.1) and r["rel"] == pytest.approx(0.05) and r["band"] == "noticeable"
    assert r["ppl"] == pytest.approx(8.166, abs=1e-3)
    missing = rt.row_of(p, "d=4|w=", 2.0)
    assert (missing["tried"], missing["loss"], missing["rel"], missing["band"]) == (False, None, None, None)
    assert rt.row_of({}, "d=4|w=", None)["delta"] is None


def test_pair_agreement_and_stability():
    assert rt.pair_agreement([1, 2, 3], [10, 20, 30]) == 1.0 and rt.pair_agreement([1, 2, 3], [3, 2, 1]) == 0.0
    assert rt.pair_agreement([1, 1], [1, 2]) is None and rt.pair_agreement([1, 2, 2], [1, 2, 3]) == 1.0          # ties are left out
    rows = [{"loss_h1": a, "loss_h2": b} for a, b in ((1.0, 1.1), (2.0, 2.2), (3.0, 3.3))]
    assert rt.stability(rows) == {"agree": True, "best_agree": True, "pairs": 1.0, "n": 3}
    flipped = [{"loss_h1": a, "loss_h2": b} for a, b in ((1.0, 2.2), (2.0, 1.1), (3.0, 3.3))]
    s = rt.stability(flipped)
    assert s["agree"] is False and s["best_agree"] is False and s["pairs"] == pytest.approx(2 / 3)
    nearly = [{"loss_h1": a, "loss_h2": b} for a, b in ((1.0, 1.0), (2.0, 4.0), (3.0, 3.0), (4.0, 2.0), (5.0, 5.0))]          # the cheapest agrees, the order does not
    assert rt.stability(nearly)["best_agree"] is True and rt.stability(nearly)["agree"] is False
    assert rt.stability([rows[0]]) is None and rt.stability([{"loss_h1": None, "loss_h2": None}] * 3) is None


def test_the_cheapest_run_wins_and_a_tie_goes_to_the_earlier_one():
    rows = [{"loss": 2.3}, {"loss": 2.1}, {"loss": 2.1}, {"loss": None}]
    assert rt.cheapest(rows) is rows[1] and rt.cheapest([{"loss": None}]) is None and rt.cheapest([]) is None


def synthetic(nb=8, matrix=True):
    """a model of 8 blocks: block b alone costs 2.0 + 0.01*b, a run costs more than the sum (the cost of two together is 1.5 times the sum of the increases)"""
    losses = {}
    for b in range(1, nb - 1):
        losses[rt.pass_key((b,))] = 2.0 + 0.01 * (b + 1)
    for k in (2, 3):
        for w in rt.windows(nb, k):
            losses[rt.pass_key(w)] = 2.0 + 1.5 * sum(0.01 * (b + 1) for b in w)
    D = [[abs(a - b) * 0.1 for b in range(nb + 1)] for a in range(nb + 1)]
    D[3][5] = 0.0001                                  # the influence says: blocks 3 and 4
    return passes_of(2.0, losses), (D if matrix else None)


def test_analysis_compares_the_cheapest_run_with_the_programs_default_choice():
    passes, D = synthetic()
    a = rt.analyse(passes, 8, 4, 1, 1, D)
    assert a["base"]["loss"] == 2.0 and a["base"]["ppl"] == pytest.approx(7.389, abs=1e-3) and a["bands"]["edges"] == [0.03, 0.10]
    assert [r["block"] for r in a["blocks"]] == [1, 2, 3, 4, 5, 6] and a["blocks"][0]["layers"] == [4, 5, 6, 7] and a["blocks"][2]["influence"] == D[3][4]
    assert a["blocks"][0]["rel"] == pytest.approx(0.01) and a["blocks"][0]["band"] == "small"
    r2 = a["runs"]["2"]
    assert [w["start"] for w in r2["windows"]] == [1, 2, 3, 4, 5] and r2["complete"] is True
    assert r2["best"]["blocks"] == [1, 2] and r2["best"]["loss"] == pytest.approx(2.0 + 1.5 * 0.05)
    assert r2["default"]["blocks"] == [3, 4] and r2["default"]["layers"] == [12, 13, 14, 15, 16, 17, 18, 19] and r2["default"]["loss"] > r2["best"]["loss"]
    assert "1" in a["runs"] and a["runs"]["1"]["best"]["blocks"] == [1] and a["runs"]["1"]["windows"][0]["distance"] == D[1][2]
    assert a["runs"]["3"]["best"]["blocks"] == [1, 2, 3] and a["widths"] == [] and a["choices"] == []


def test_analysis_without_a_measurement_has_no_influence_and_no_default():
    passes, _ = synthetic(matrix=False)
    a = rt.analyse(passes, 8, 4, 1, 1, None)
    assert a["blocks"][0]["influence"] is None and a["runs"]["2"]["default"] is None and a["runs"]["2"]["best"]["blocks"] == [1, 2]


def test_analysis_of_a_partial_run_marks_what_is_missing():
    passes, D = synthetic()
    del passes[rt.pass_key((2, 3))]
    del passes[rt.pass_key((4,))]
    a = rt.analyse(passes, 8, 4, 1, 1, D)
    assert a["runs"]["2"]["complete"] is False and a["runs"]["2"]["windows"][1]["tried"] is False and a["runs"]["2"]["windows"][1]["loss"] is None
    assert a["blocks"][3]["tried"] is False
    only_base = rt.analyse({rt.BASE_KEY: passes[rt.BASE_KEY]}, 8, 4, 1, 1, D)
    assert only_base["runs"] == {} and all(not r["tried"] for r in only_base["blocks"])
    empty = rt.analyse({}, 8, 4, 1, 1, None)
    assert empty["base"]["loss"] is None and empty["blocks"][0]["rel"] is None


def test_analysis_sorts_the_width_curve_and_keeps_combined_choices_apart():
    passes, D = synthetic()
    passes.update(passes_of(2.0, {"d=|w=12288": 2.4, "d=|w=13312": 2.1, "d=2,5|w=": 2.5, "d=3|w=12288": 2.45, "d=2,3|w=13312": 2.3}))
    a = rt.analyse(passes, 8, 4, 1, 1, D)
    assert [w["intermediate"] for w in a["widths"]] == [12288, 13312] and a["widths"][1]["band"] == "noticeable" and a["widths"][0]["band"] == "large" and a["widths"][0]["rel"] == pytest.approx(0.2)
    got = {(tuple(c["blocks"]), c["intermediate"]) for c in a["choices"]}
    assert got == {((2, 5), None), ((3,), 12288), ((2, 3), 13312)}
    assert a["widths_stability"]["agree"] is True and a["blocks_stability"]["n"] == 6
    json.dumps(a)                           # the whole analysis is plain JSON


def test_the_cache_key_names_what_changed():
    calib = {"version": 3, "samples": 64, "max_len": 512, "seed": 3407}
    key = rt.build_key("org/model", "dir:abc", calib, "h1", "4bit")
    assert rt.key_diff(key, dict(key)) is None and rt.key_diff(None, key) == "the saved answers could not be read"
    for field, value, words in (("source", "other", "another model"), ("weights", "dir:zzz", "other weights"), ("data_version", 4, "another version of the training data"),
                                ("data_hash", "h2", "other training text"), ("samples", 128, "another number of samples"), ("max_len", 256, "another text length"),
                                ("seed", 1, "another seed"), ("load", "16bit", "another way of loading")):
        assert words in rt.key_diff({**key, field: value}, key)


def test_hashes_follow_the_text_and_the_weight_files(tmp_path):
    f = tmp_path / "train.jsonl"
    assert rt.data_hash(f) is None and rt.data_hash(None) is None
    f.write_text("a", encoding="utf-8")
    h = rt.data_hash(f)
    f.write_text("b", encoding="utf-8")
    assert len(h) == 16 and rt.data_hash(f) != h
    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text("{}", encoding="utf-8")
    (m / "a.safetensors").write_bytes(b"12345")
    fp = rt.weights_fingerprint(m)
    assert fp == rt.weights_fingerprint(m) and fp.startswith("dir:")
    (m / "a.safetensors").write_bytes(b"123456")
    assert rt.weights_fingerprint(m) != fp
    snap = tmp_path / "models--x" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    assert rt.weights_fingerprint(snap) == "hub:abc123"


def test_results_are_saved_atomically_and_another_key_is_dropped_with_a_reason(tmp_path):
    key = rt.build_key("m", "w", {"version": 1, "samples": 64, "max_len": 64}, "h", "16bit")
    assert rt.read_results(tmp_path / "none", key) is None
    data = rt.empty_results(key)
    rt.merge_pass(data, rt.BASE_KEY, {"loss": 2.0, "seconds": 3.0})
    rt.merge_pass(data, rt.BASE_KEY, {"loss": 9.0, "seconds": 9.0})              # never overwritten
    rt.merge_pass(data, "d=1|w=", {"loss": 2.1, "seconds": 5.0})
    assert data["passes"][rt.BASE_KEY]["loss"] == 2.0 and data["seconds_per_pass"] == 4.0
    rt.write_results(tmp_path / "f", data)
    assert not (tmp_path / "f" / "reshape_try.json.partial").exists()
    log = []
    assert rt.read_results(tmp_path / "f", key, log.append)["passes"]["d=1|w="]["loss"] == 2.1 and log == []
    assert rt.read_results(tmp_path / "f", {**key, "samples": 128}, log.append) is None and "another number of samples" in log[0]
    (tmp_path / "f" / "reshape_try.json").write_text("{not json", encoding="utf-8")
    assert rt.read_results(tmp_path / "f", key, log.append) is None and "could not be read" in log[1]


def test_time_estimate_uses_a_timed_pass_when_there_is_one():
    e = rt.estimate(20, None, None, None, 27.8, "4bit", 64 * rt.GUESS_TOKENS_PER_SAMPLE)
    assert e["guess"] is True and e["seconds"] == pytest.approx(20 * e["seconds_per_pass"]) and e["seconds_per_pass"] == pytest.approx(2 * 27.8e9 * 16384 / 1.5e13)
    t = rt.estimate(20, 30.0, 32000, 16000, 27.8, "4bit", 0)
    assert t["guess"] is False and t["seconds_per_pass"] == 60.0 and t["seconds"] == 1200.0          # twice the text, twice the time
    assert rt.estimate(3, 30.0, None, None, 27.8, "4bit", 0)["seconds_per_pass"] == 30.0
    assert rt.estimate(0, None, 100, None, 0.001, "16bit", 100)["seconds"] == 0.0 and rt.guess_seconds_per_pass(0.001, 10, "16bit") == 1.0
    assert rt.mean_seconds({}) is None and rt.mean_seconds({"a": {"seconds": 2.0}, "b": {"seconds": None}, "c": {"seconds": 4.0}}) == 3.0
