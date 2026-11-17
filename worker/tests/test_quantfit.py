"""quantfit.py: the allocator, pins, rules and report, on synthetic measures (no weights, no encoders) and on the tensor-name lists of llama.cpp's qwen35 architecture."""
import itertools
import json
import re

import numpy as np
import pytest
from test_mixedquant import F16, F32, info_of, qwen35_tensors

from ndforge_worker import ggufinfo, mixedquant, quantfit, quantmeasure

BITS = {"Q2_K": 2.0, "Q3_K": 3.0, "Q4_K": 4.0, "Q5_K": 5.0, "Q6_K": 6.0, "Q8_0": 8.0}


def fake_measures(info, seed=0, skipped=()):
    """For every quantizable tensor a record with the error of every rung: weight x 4^-bits (a 6-bit type has a quarter of the 5-bit error / 4...) times a random per-tensor scale."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in info["tensors"]:
        if not mixedquant.quantizable(t["name"], t["dims"]) or t["name"] in skipped:
            continue
        cols = t["dims"][0]
        n = ggufinfo.numel(t)
        scale = float(rng.uniform(0.2, 5.0)) * n / 1e6
        types = {}
        for r in quantmeasure.RUNGS:
            eff = mixedquant.effective_type(r, cols)
            types[r] = {"eff": eff, "bytes": int(n * ggufinfo.BPW[eff] / 8), "err": scale * 4.0 ** -BITS[r] * (1 + 0.02 * ggufinfo.BPW[eff] * 0), "err_plain": scale}
        out[t["name"]] = {"rows": n // cols, "cols": cols, "experts": 1, "sampled": 8, "norm2_w": scale * 10, "types": types}
    return {"version": 1, "engine": "reference", "tensors": out, "skipped": list(skipped), "notes": [], "done": True}


def small_info(**kw):
    return info_of(qwen35_tensors(**kw), n_layer=kw.get("n_layer", 8))


def test_hull_has_falling_error_and_falling_rate():
    pts = [(10, 100.0, "a"), (20, 60.0, "b"), (30, 55.0, "c"), (40, 20.0, "d"), (50, 19.0, "e"), (25, 70.0, "dominated")]
    h = quantfit._hull(pts)
    assert [p[2] for p in h] == ["a", "b", "d", "e"]
    errs = [p[1] for p in h]
    assert errs == sorted(errs, reverse=True)
    rates = [(h[i][1] - h[i + 1][1]) / (h[i + 1][0] - h[i][0]) for i in range(len(h) - 1)]
    assert rates == sorted(rates, reverse=True)


def test_never_over_budget_and_more_budget_never_more_error():
    info = small_info()
    m = fake_measures(info, 1)
    prev = None
    lo = quantfit.plan_for(info, m, 1, "Q3_K", "Q8_0")          # a budget of nothing: the floor plan
    hi = quantfit.plan_for(info, m, 1e12, "Q3_K", "Q8_0")
    for frac in np.linspace(0, 1.0, 12):
        target = lo.min_bytes + quantfit.margin(info) + frac * (hi.max_bytes - lo.min_bytes)
        p = quantfit.plan_for(info, m, target, "Q3_K", "Q8_0")
        assert p.fits and p.total + quantfit.margin(info) <= target + 1
        if prev is not None:
            assert p.err <= prev + 1e-12
        prev = p.err


def test_below_the_minimum_gives_the_floor_and_fits_false():
    info = small_info()
    m = fake_measures(info, 2)
    p = quantfit.plan_for(info, m, 1000, "Q3_K", "Q8_0")
    assert not p.fits and p.notes and "lowest" in p.notes[0]
    assert set(p.rung.values()) == {"Q3_K"} and p.total == p.min_bytes
    p2 = quantfit.plan_for(info, m, 1000, "Q2_K", "Q8_0")
    assert set(p2.rung.values()) == {"Q2_K"} and p2.min_bytes < p.min_bytes


def test_above_the_maximum_gives_the_ceiling():
    info = small_info()
    m = fake_measures(info, 3)
    p = quantfit.plan_for(info, m, 1e12, "Q3_K", "Q6_K")
    assert p.fits and set(p.rung.values()) == {"Q6_K"} and "more than the highest" in p.notes[0]
    assert p.total == p.max_bytes


def brute_gaps(sizes_of, trials, seed):
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(trials):
        sizes = sizes_of(rng)
        measures = {"tensors": {}}
        for i, n in enumerate(sizes):
            scale = float(rng.uniform(0.5, 4.0)) * float(n)
            measures["tensors"][f"blk.{i}.attn_q.weight"] = {"types": {r: {"eff": r, "bytes": int(n * ggufinfo.BPW[r] / 8), "err": scale * 4.0 ** -BITS[r], "err_plain": 0} for r in quantmeasure.RUNGS}}
        names = sorted(measures["tensors"])
        lo = sum(measures["tensors"][n]["types"]["Q3_K"]["bytes"] for n in names)
        hi = sum(measures["tensors"][n]["types"]["Q8_0"]["bytes"] for n in names)
        for frac in (0.3, 0.5, 0.7):
            budget = int(lo + frac * (hi - lo))
            got = quantfit.allocate(measures, budget, 0, "Q3_K", "Q8_0")
            assert got.total <= budget
            best = min(sum(measures["tensors"][n]["types"][r]["err"] for n, r in zip(names, combo)) for combo in itertools.product(quantfit.allowed_rungs("Q3_K", "Q8_0"), repeat=6)
                       if sum(measures["tensors"][n]["types"][r]["bytes"] for n, r in zip(names, combo)) <= budget)
            gaps.append(got.err / best)
    return gaps


def test_within_two_percent_of_brute_force_for_six_tensors_of_one_size():
    gaps = brute_gaps(lambda rng: [256 * 128] * 6, 10, 4)
    assert max(gaps) <= 1.02, f"the hull allocation was {max(gaps):.4f} times the brute-force optimum"


def test_brute_force_gap_with_uneven_tensors_is_the_knapsack_granularity():
    """Six tensors of 1 to 4 units are too coarse for any greedy method (one upgrade can be a third of the budget); the real model has hundreds of tensors. Median and worst are pinned."""
    gaps = brute_gaps(lambda rng: rng.integers(1, 5, 6) * 256 * 64, 12, 4)
    assert float(np.median(gaps)) <= 1.005 and max(gaps) <= 1.30


def test_fallback_rungs_are_deduplicated():
    """a tensor with 640 columns: Q2_K and Q3_K both fall back to Q4_0, Q4_K to Q5_0, Q5_K to Q5_1"""
    info = info_of([("token_embd.weight", [640, 100], F16), ("output.weight", [640, 100], F16), ("blk.0.ffn_up.weight", [640, 320], F16), ("blk.0.attn_norm.weight", [640], F32)], n_layer=1)
    m = fake_measures(info, 5)
    rec = m["tensors"]["blk.0.ffn_up.weight"]
    assert [rec["types"][r]["eff"] for r in quantmeasure.RUNGS] == ["Q4_0", "Q4_0", "Q5_0", "Q5_1", "Q8_0", "Q8_0"]
    p = quantfit.allocate(m, 10 ** 12, 0, "Q2_K", "Q8_0", exclude=("token_embd.weight", "output.weight"))
    assert p.assign["blk.0.ffn_up.weight"] == "Q8_0"
    mid = quantfit.allocate(m, rec["types"]["Q4_K"]["bytes"] + 10, 0, "Q2_K", "Q8_0", exclude=("token_embd.weight", "output.weight"))
    assert mid.assign["blk.0.ffn_up.weight"] == "Q5_0" and mid.rung["blk.0.ffn_up.weight"] == "Q4_K"


def test_pins_follow_the_size_and_tied_embeddings_take_the_output_type():
    info = small_info()
    q4, q5, q6 = (quantfit.plain_bytes(info, k) for k in ("Q4_K_M", "Q5_K_M", "Q6_K"))
    assert quantfit.pins(q4 * 0.9, info) == {"embd": "Q4_K", "output": "Q6_K", "tied": False}
    assert quantfit.pins(q4 * 1.1, info)["embd"] == "Q5_K" and quantfit.pins(q5 * 1.1, info)["embd"] == "Q6_K"
    assert quantfit.pins(q6 * 1.01, info)["output"] == "Q8_0"
    tied = small_info(tied=True)
    m = fake_measures(tied, 6)
    p = quantfit.plan_for(tied, m, quantfit.plain_bytes(tied, "Q4_K_M"), "Q3_K", "Q8_0")
    assert p.pins["tied"] and p.pinned_assign["token_embd.weight"] == p.pins["output"] == "Q6_K" and "token_embd.weight" not in p.assign
    assert quantfit.rules(p, tied)                                      # verified by the rule semantics (the embedding resolves to the output type)


@pytest.mark.parametrize("kw", [{}, {"fused_linear": True}, {"nextn": False}, {"moe": True, "ff": 1024}, {"tied": True, "moe": True, "ff": 512}])
def test_the_rules_reproduce_the_plan_and_the_estimate_equals_the_total(kw):
    info = small_info(**kw)
    m = fake_measures(info, 7)
    for frac in (0.55, 0.8, 1.0):
        target = quantfit.plain_bytes(info, "Q4_K_M") * frac * 1.3
        plan = quantfit.plan_for(info, m, target, "Q2_K", "Q8_0")
        rs = quantfit.rules(plan, info)                                 # raises when a tensor would not resolve to its type
        est = mixedquant.estimate(info, rs, quantfit.preset_p(plan))
        assert est["unmatched"] == 0 and est["bytes"] == plan.total and not est["dead_rules"]
        for rx, t in rs:
            re.compile(rx)
            assert "=" not in rx and not re.search(r"\\[A-Z]", rx) and rx == rx.lower() and t in quantfit.ORDER
        # ordering inside a group: the higher type first
        assert len(rs) < len(plan.assign)


def test_a_plan_that_does_not_verify_raises():
    info = small_info()
    plan = quantfit.plan_for(info, fake_measures(info, 8), quantfit.plain_bytes(info, "Q4_K_M"))
    rs = quantfit.rules(plan, info)
    broken = [(rx, "Q2_K") for rx, _ in rs]
    with pytest.raises(quantfit.PlanError, match="Internal check failed"):
        quantfit.verify(plan, broken, info)


def test_deterministic():
    info = small_info()
    m = fake_measures(info, 9)
    a = quantfit.plan_for(info, m, quantfit.plain_bytes(info, "Q4_K_M"))
    b = quantfit.plan_for(info, json.loads(json.dumps(m)), quantfit.plain_bytes(info, "Q4_K_M"))
    assert a.to_dict() == b.to_dict() and quantfit.rules(a, info) == quantfit.rules(b, info)


def test_equal_error_tensors_tie_by_name():
    measures = {"tensors": {n: {"types": {r: {"eff": r, "bytes": int(1024 * ggufinfo.BPW[r]), "err": 4.0 ** -BITS[r], "err_plain": 0} for r in quantmeasure.RUNGS}} for n in ("blk.1.attn_q.weight", "blk.0.attn_q.weight")}}
    one_up = int(1024 * (ggufinfo.BPW["Q4_K"] - ggufinfo.BPW["Q3_K"])) + int(2 * 1024 * ggufinfo.BPW["Q3_K"])
    p = quantfit.allocate(measures, one_up, 0, "Q3_K", "Q8_0")
    assert p.rung == {"blk.0.attn_q.weight": "Q4_K", "blk.1.attn_q.weight": "Q3_K"}


def test_skipped_tensors_are_kept_at_q6k():
    info = small_info()
    skip = ("blk.1.ffn_up.weight",)
    m = fake_measures(info, 10, skipped=skip)
    p = quantfit.plan_for(info, m, quantfit.plain_bytes(info, "Q4_K_M") * 1.2)
    assert p.pinned_assign["blk.1.ffn_up.weight"] == "Q6_K" and any("not finite" in n for n in p.notes)
    assert quantfit.rules(p, info)
    with pytest.raises(quantfit.PlanError, match="was not measured"):
        quantfit.plan_for(info, {**m, "tensors": {k: v for k, v in m["tensors"].items() if k != "blk.2.ffn_up.weight"}}, 1e9)


def test_floor_above_ceiling_is_refused():
    with pytest.raises(quantfit.PlanError):
        quantfit.allowed_rungs("Q6_K", "Q3_K")
    with pytest.raises(quantfit.PlanError):
        quantfit.allowed_rungs("IQ2_S", "Q8_0")


def test_flags_with_and_without_a_type_file():
    info = small_info()
    plan = quantfit.plan_for(info, fake_measures(info, 11), quantfit.plain_bytes(info, "Q4_K_M"))
    rs = quantfit.rules(plan, info)
    f = quantfit.flags(plan, rs)
    assert f[:4] == ["--output-tensor-type", plan.pins["output"], "--token-embedding-type", plan.pins["embd"]] and f.count("--tensor-type") == len(rs)
    assert f[5] == f"{rs[0][0]}={rs[0][1]}"
    g = quantfit.flags(plan, rs, "types.txt")
    assert g[4:] == ["--tensor-type-file", "types.txt"] and "--tensor-type" not in g
    assert quantfit.type_file_text(rs).splitlines() == [f"{rx}={t}" for rx, t in rs]


def test_base_ftype_is_the_nearest_plain_size():
    assert quantfit.base_ftype(4.9) == "Q4_K_M" and quantfit.base_ftype(5.9) == "Q5_K_M" and quantfit.base_ftype(8.0) == "Q8_0" and quantfit.base_ftype(3.0) == "Q2_K"


def test_report_has_the_rows_types_and_breakdowns():
    info = small_info()
    m = fake_measures(info, 12)
    plan = quantfit.plan_for(info, m, quantfit.plain_bytes(info, "Q4_K_M"))
    r = quantfit.report(info, m, plan)
    ids = [x["id"] for x in r["rows"]]
    assert ids[0] == "plan" and "uniform_q3_k" in ids and "uniform_q6_k" in ids and ids[-2:] == ["plain_q4_k_m", "plain_q5_k_m"]
    sizes = {x["id"]: x["bytes"] for x in r["rows"]}
    assert sizes["plan"] == plan.total and sizes["uniform_q3_k"] < sizes["uniform_q4_k"] < sizes["uniform_q6_k"]
    errs = {x["id"]: x["error"] for x in r["rows"]}
    assert errs["uniform_q3_k"] > errs["uniform_q4_k"] > errs["uniform_q5_k"] > errs["uniform_q6_k"]
    assert errs["plan"] < errs["uniform_q4_k"] or sizes["plan"] < sizes["uniform_q4_k"]
    assert sum(v["tensors"] for v in r["types"].values()) == len(plan.assign) + len(plan.pinned_assign)
    assert "ffn_up" in r["bpw_by_role"] and "0" in r["bpw_by_layer"] and r["caveats"] == quantmeasure.CAVEATS
    assert "diagonal" in r["caveats"] and "Compare quants" in r["caveats"]


def test_simulated_plain_sizes_follow_the_mixture():
    info = small_info(n_layer=16)
    s4, s5 = quantfit.simulate_plain(info, "Q4_K_M"), quantfit.simulate_plain(info, "Q5_K_M")
    assert s4["output.weight"] == "Q6_K" and s4["blk.0.ffn_down.weight"] == "Q6_K" and s4["blk.1.ffn_down.weight"] == "Q6_K" and s4["blk.3.ffn_down.weight"] == "Q4_K" and s4["blk.4.ffn_down.weight"] == "Q6_K" and s4["blk.3.ffn_up.weight"] == "Q4_K"
    assert s5["blk.3.ffn_down.weight"] == "Q5_K" and s5["blk.3.ffn_up.weight"] == "Q5_K"
    assert s4["blk.1.attn_qkv.weight"] == "Q5_K" and s5["blk.1.attn_qkv.weight"] == "Q6_K"


def test_existing_label_is_not_a_dynamic_quant():
    text = quantfit.ID + quantfit.LABEL + quantfit.DESCRIPTION + open(quantfit.__file__, encoding="utf-8").read()
    assert "UD-" not in text and not re.search(r"unsloth dynamic", text, re.I)
