"""The token-budget part of the heal speed report: the length statistics, the sweep with a failing budget, and the recommendation rules (pure functions, no torch)."""
import pytest

from ndforge_worker import healspeed


def test_length_statistics():
    s = healspeed.length_stats([10, 20, 30, 40, 1000])
    assert s == {"n": 5, "mean": 220.0, "p50": 30, "p90": 1000, "max": 1000}
    assert healspeed.length_stats([]) == {"n": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0}


def res(b, tps, peak=10000, loss=2.0):
    return {"batch_tokens": b, "tokens_per_s": tps, "peak_vram_mb": peak, "first_loss": loss}


def test_the_recommendation_is_the_fastest_within_90_percent_of_the_card_and_with_the_same_loss():
    rs = [res(0, 400), res(2048, 700, 20000), res(4096, 900, 24000), res(8192, 1100, 31000), res(16384, 1200, 29000, loss=2.5)]
    rec = healspeed.recommend_batch_tokens(rs, 32000)
    # 8192 is above 90 % of 32000 MB, 16384 has another first loss: 4096 wins
    assert rec["value"] == 4096 and rec["speedup"] == 2.25 and "same first loss" in rec["why"], rec
    assert [r["speedup"] for r in rs] == [1.0, 1.75, 2.25, 2.75, 3.0]


def test_nothing_is_recommended_when_nothing_is_clearly_faster_or_the_baseline_is_missing():
    assert healspeed.recommend_batch_tokens([res(0, 400), res(2048, 410)], 32000)["value"] == 0
    assert healspeed.recommend_batch_tokens([res(2048, 410)], 32000)["value"] == 0
    only_big = [res(0, 400), res(8192, 900, 31000)]
    assert healspeed.recommend_batch_tokens(only_big, 32000)["value"] == 0


def test_a_budget_that_fails_is_recorded_and_the_others_go_on():
    def run_one(b):
        if b == 8192:
            raise RuntimeError("CUDA out of memory")
        return {"tokens_per_s": 400 + b / 10, "peak_vram_mb": 1000 + b, "first_loss": 1.0}
    out = healspeed.sweep(run_one, (0, 2048, 8192), 32000)
    assert [r["batch_tokens"] for r in out["results"]] == [0, 2048, 8192]
    assert "out of memory" in out["results"][2]["error"] and out["results"][2]["tokens_per_s"] is None
    assert out["recommended"]["value"] == 2048


def test_the_report_shows_the_rows_the_table_and_the_recommendation():
    rep = {"status": "done", "rows": healspeed.length_stats([100, 200, 4000]), "batch_sweep": healspeed.sweep(lambda b: {"tokens_per_s": 400 + b / 5, "peak_vram_mb": 5000, "first_loss": 1.0, "micro_batches": 32 if not b else 3},
                                                                                                           healspeed.BATCH_TOKENS_SWEEP, 32000)}
    rep["verdict"] = healspeed.verdict_lines(rep)
    md = healspeed.markdown(rep)
    assert "## The rows measured" in md and "3 real rows" in md and "| 0 (one row) |" in md and "| 16384 |" in md and "Recommended: 16384 tokens per forward pass" in md
    assert any("Several rows in one forward pass pay off" in v for v in rep["verdict"])


def test_an_out_of_memory_prunes_the_larger_budgets_and_is_recorded():
    def run_one(b):
        if b >= 8192:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return {"tokens_per_s": 400 + b / 10, "peak_vram_mb": 1000, "first_loss": 1.0}
    out = healspeed.sweep(run_one, (0, 2048, 8192, 16384), 32000)
    r = {x["batch_tokens"]: x for x in out["results"]}
    assert r[8192]["oom"] is True and r[16384]["skipped"] is True and "already ran out of memory" in r[16384]["error"]
    assert out["recommended"]["value"] == 2048


def test_the_measured_noise_decides_how_big_a_gain_has_to_be():
    assert healspeed.spread_of([400, 400]) == 0.0 and healspeed.spread_of([400, 440]) == pytest.approx(0.0952, abs=1e-3) and healspeed.spread_of([400]) == 0.0
    base, fast = {"batch_tokens": 0, "tokens_per_s": 400.0, "peak_vram_mb": 1000, "first_loss": 1.0}, {"batch_tokens": 2048, "tokens_per_s": 440.0, "peak_vram_mb": 1000, "first_loss": 1.0}
    assert healspeed.recommend_batch_tokens([dict(base), dict(fast)], 32000, 0.0)["value"] == 2048, "10 % with no noise"
    noisy = healspeed.recommend_batch_tokens([dict(base), dict(fast)], 32000, 0.08)
    assert noisy["value"] == 0 and "noise of 8.0 %" in noisy["why"], "10 % is inside twice an 8 % spread"
    # the worst-case update counts for the memory guard
    big = {"batch_tokens": 4096, "tokens_per_s": 900.0, "peak_vram_mb": 10000, "worst_peak_mb": 30000, "first_loss": 1.0}
    assert healspeed.recommend_batch_tokens([dict(base), big], 32000, 0.0)["value"] == 0


def test_repeats_make_the_mean_the_number_and_the_spread_the_noise():
    calls = {"n": 0}

    def run_one(b):
        calls["n"] += 1
        return {"tokens_per_s": (400.0 if b == 0 else 800.0) * (1.0 + 0.01 * calls["n"]), "peak_vram_mb": 1000, "first_loss": 1.0}
    out = healspeed.sweep(run_one, (0, 2048), 32000, repeat=2)
    r = {x["batch_tokens"]: x for x in out["results"]}
    assert len(r[0]["repeats"]) == 2 and len(r[2048]["repeats"]) == 2 and out["spread"] > 0
    assert r[0]["tokens_per_s"] == pytest.approx(sum(r[0]["repeats"]) / 2, abs=0.01)
    assert out["recommended"]["value"] == 2048 and out["recommended"]["spread"] == out["spread"]
