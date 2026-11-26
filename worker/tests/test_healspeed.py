"""The heal speed measurement without a GPU: kernel classes, the profile summary, the microbench and teacher bench loops with injected functions, the report (always written, scrubbed)."""
import json

import pytest

from ndforge_worker import healspeed as hs, protocol


def test_kernels_are_classified_by_name_and_the_top_ten_are_listed():
    ks = [("sm120_xmma_gemm_bf16", 6_000_000), ("cutlass_80_tensorop_gemm", 2_000_000), ("chunk_gated_delta_rule_fwd_kernel", 1_500_000), ("fla_recompute_w_u", 500_000), ("elementwise_add", 700_000),
          ("sm120_xmma_gemm_bf16", 1_000_000)]
    c = hs.classify_kernels(ks)
    assert c["gemm_s"] == 9.0 and c["gdn_s"] == 2.0 and c["other_s"] == 0.7
    assert c["top_kernels"][0]["name"] == "sm120_xmma_gemm_bf16" and c["top_kernels"][0]["seconds"] == 7.0 and len(c["top_kernels"]) <= 10
    assert hs.classify_kernels([])["top_kernels"] == []


def test_the_profile_summary_is_per_step_with_shares():
    p = hs.profile_summary({"forward": 6.0, "loss": 1.0, "backward": 11.0, "optimizer": 0.2}, [("gemm", 10_000_000), ("gated_delta", 4_000_000)], 2)
    assert (p["forward_s"], p["loss_s"], p["backward_s"], p["optimizer_s"]) == (3.0, 0.5, 5.5, 0.1) and p["gemm_s"] == 5.0 and p["gdn_s"] == 2.0 and p["step_s"] == 9.1
    assert p["gdn_share_of_step"] == pytest.approx(0.22, abs=0.001) and p["steps"] == 2


def test_sections_time_the_wrapped_calls_with_a_synchronize_at_each_border():
    calls, t = [], [0.0]

    def clock():
        t[0] += 1.0
        return t[0]

    s = hs.Sections(sync=lambda: calls.append("sync"), clock=clock)
    assert s.wrap("forward", lambda x: x * 2)(4) == 8 and s.t["forward"] == 1.0 and calls == ["sync", "sync"]
    with pytest.raises(ValueError):
        s.wrap("loss", lambda: (_ for _ in ()).throw(ValueError("x")))()
    assert "loss" in s.t


def test_the_microbench_reports_each_implementation_and_the_ratio_and_never_raises():
    t = iter(x * 0.1 for x in range(100))
    r = hs.gdn_microbench(steps=2, torch_fn=lambda *a: None, fla_fn=lambda *a: None, make_inputs=lambda: (1,), timer=lambda: next(t))
    assert r["torch"]["steps"] == 2 and r["fla"]["seconds_per_fwd_bwd"] > 0 and "fla_speedup" in r
    bad = hs.gdn_microbench(torch_fn=lambda *a: (_ for _ in ()).throw(RuntimeError("no kernel")), fla_fn=None, make_inputs=lambda: ())
    assert "RuntimeError: no kernel" in bad["torch"]["error"] and "not importable" in bad["fla"]["error"]


def test_the_teacher_bench_sorts_rows_by_length_and_stops_at_an_out_of_memory_size():
    rows = [{"input_ids": list(range(n))} for n in (10, 50, 30, 40, 20, 60, 70, 80)]
    seen = []
    clock = iter(range(0, 1000, 2))

    def answer(batch):
        seen.append([len(r["input_ids"]) for r in batch])
        if len(batch) > 4:
            raise RuntimeError("CUDA out of memory")

    out = hs.teacher_bench(rows, answer, sizes=(1, 2, 4, 8, 16), timer=lambda: next(clock))
    assert [x["rows_per_forward"] for x in out] == [1, 2, 4, 8], "16 is not tried after the out-of-memory size"
    assert out[0]["tokens"] == 360 and out[0]["tokens_per_s"] > 0 and "out of memory" in out[3]["error"]
    assert seen[0] == [80] and seen[8][0] == 80, "longest first"


def test_the_report_is_always_written_and_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("USERNAME", "Zebulon")
    rep = {"created": "t", "status": "failed", "error": "C:\\Users\\Zebulon\\x broke hf_" + "a" * 30, "environment": {"torch": "torch 2.14.1", "gpu": {"name": "RTX 5090"}}, "kernels": {"delta_rule": "torch"},
           "heal_profile": hs.profile_summary({"forward": 1, "loss": 1, "backward": 1, "optimizer": 1}, [("gated_delta", 1_000_000)], 1), "microbench": {"torch": {"error": "x"}, "fla_speedup": 3.1},
           "teacher_bench": [{"rows_per_forward": 4, "tokens_per_s": 1100.0, "seconds": 9}, {"rows_per_forward": 16, "tokens_per_s": 1500.0, "seconds": 7}], "windows_hints": ["HAGS: on"]}
    paths = hs.write_report(tmp_path / "out", rep)
    md, js = (tmp_path / "out" / "heal_speed_report.md").read_text(), (tmp_path / "out" / "heal_speed_report.json").read_text()
    assert paths["report"].endswith("heal_speed_report.json") and "Zebulon" not in md + js and "hf_aaaa" not in md + js
    assert "slow PyTorch kernels" in md and "fla is 3.1 times faster" in md and "best at 16 rows per forward (1500 tokens/s), against 1100 with the fixed 4" in md and "HAGS: on" in md
    assert "hypotheses until this report confirms them" in md
    assert json.loads(js)["status"] == "failed"


def test_the_teacher_sweep_answers_the_same_rows_at_every_budget_and_measures_the_noise_and_the_memory():
    import numpy as np
    rows = [{"input_ids": list(range(n))} for n in (100, 90, 80, 70, 60, 50, 40, 30)]
    seen: dict = {}

    def answer(batch):
        seen.setdefault("batches", []).append(len(batch))
        return [(np.zeros((len(r["input_ids"]), 2)), np.full((len(r["input_ids"]), 2), -1.5), np.zeros(len(r["input_ids"]))) for r in batch]

    clock = iter(x * 0.5 for x in range(10_000))
    group = lambda lengths, budget: max(1, min(16, budget // lengths[0]))        # noqa: E731
    out = hs.teacher_sweep(rows, answer, group, fixed_rows=2, budgets=(0, 200, 400), card_mb=1000, repeat=2, peak=lambda: 123, timer=lambda: next(clock))
    res = out["results"]
    assert [r["batch_tokens"] for r in res] == [0, 200, 400] and all(r["peak_vram_mb"] == 123 and r["first_loss"] == 1.5 for r in res)
    assert out["rows"] == 8 and out["tokens"] == 520 and "spread" in out and out["recommended"]["value"] in (0, 200, 400)
    assert seen["batches"][:4] == [2, 2, 2, 2], "budget 0 is the fixed rows per pass"


def test_the_teacher_sample_is_seeded_keeps_the_longest_rows_and_stops_at_the_token_target():
    rows = [{"input_ids": [0] * n} for n in [10] * 50 + [500]]
    a = hs.teacher_sample(rows, want_tokens=100, seed=1)
    b = hs.teacher_sample(rows, want_tokens=100, seed=1)
    assert [len(r["input_ids"]) for r in a] == [len(r["input_ids"]) for r in b]
    assert any(len(r["input_ids"]) == 500 for r in a) and sum(len(r["input_ids"]) for r in a) < 700


def test_a_teacher_budget_that_runs_out_of_memory_is_a_row_and_the_larger_ones_are_skipped():
    import numpy as np
    rows = [{"input_ids": list(range(50))} for _ in range(8)]

    def answer(batch):
        if len(batch) > 2:
            raise RuntimeError("out of memory")
        return [(np.zeros((50, 2)), np.full((50, 2), -1.0), np.zeros(50)) for _ in batch]

    out = hs.teacher_sweep(rows, answer, lambda lengths, budget: max(1, budget // 50), fixed_rows=1, budgets=(0, 100, 200, 400), repeat=1, timer=iter(range(0, 10_000)).__next__)
    assert out["results"][2].get("oom") and out["results"][3].get("skipped") and out["results"][1]["tokens_per_s"]
