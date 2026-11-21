"""The token-budget part of the heal speed report: the length statistics, the sweep with a failing budget, and the recommendation rules (pure functions, no torch)."""
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
