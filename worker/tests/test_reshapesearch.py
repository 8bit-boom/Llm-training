"""The round planner of the automatic best-cut search (no torch): deterministic, resumable, the rounds in their order, a hidden best set that is not a run is found."""
import itertools
import json

from ndforge_worker import reshapesearch as rsr
from ndforge_worker.reshapetry import BASE_KEY, parse_key, pass_key

NB = 16
# the increase of each block on its own (blocks 3, 7, 11, 12 are cheap, the rest is not); width costs a little
D = {b: 0.05 for b in range(1, 15)}
D.update({3: 0.001, 7: 0.002, 11: 0.003, 12: 0.004, 5: 0.02})
BASE = 2.0
WIDTH_COST = {6144: 0.03, 8192: 0.01}


def loss_of(key):
    blocks, width = parse_key(key)
    extra = 0.002 * len(blocks) ** 2 / 2          # a small interaction: the prediction is not exact
    return BASE + sum(D[b] for b in blocks) + extra + (WIDTH_COST.get(width, 0.0) if width else 0.0)


def result(key):
    v = loss_of(key)
    return {"loss": v, "loss_h1": v + 0.0005, "loss_h2": v - 0.0005, "tokens": 10, "seconds": 1.0}


def ctx(**over):
    c = {"nb": NB, "pf": 1, "pl": 1, "groups": [{"label": "20 B", "target_b": 20.0, "cuts": [
        {"k": 3, "width": 8192, "params": 20e9, "on_target": True}, {"k": 4, "width": 6144, "params": 20e9, "on_target": True}]}]}
    c.update(over)
    return c


def drive(c, passes=None, limit=1000, stop_after=None):
    """What the worker does: ask for a round, measure it, record it. Returns (passes, rounds records, order of keys)."""
    passes = {} if passes is None else passes
    order, rounds = [], list(c.get("rounds") or [])
    while True:
        c = dict(c, rounds=rounds)
        r = rsr.next_round(passes, c)
        if r is None:
            break
        rec = next((x for x in rounds if x["name"] == r["name"]), None)
        if rec is None:
            rec = {"name": r["name"], "keys": r["all_keys"], "done": False}
            rounds.append(rec)
        for k in r["keys"]:
            if stop_after is not None and len(order) >= stop_after:
                return passes, rounds, order
            passes[k] = result(k)
            order.append(k)
        rec["done"] = True
        assert len(order) < limit
    return passes, rounds, order


def test_the_rounds_come_in_their_order_and_nothing_repeats():
    passes, rounds, order = drive(ctx())
    assert [r["name"] for r in rounds] == ["singles", "width_refs", "sets_k", "combined", "refine1", "refine2"]
    assert len(order) == len(set(order))
    assert order[0] == BASE_KEY
    assert order[:15] == [BASE_KEY] + [pass_key((b,)) for b in range(1, 15)]
    # the width references come right after the singles
    assert set(order[15:17]) == {pass_key((), 6144), pass_key((), 8192)}


def test_it_is_deterministic():
    a = drive(ctx())[2]
    b = drive(ctx())[2]
    assert a == b


def test_the_hidden_best_set_that_is_not_a_run_is_found():
    c = ctx()
    passes, _, _ = drive(c)
    best3 = min((loss_of(pass_key(s, 8192)), s) for s in itertools.combinations(range(1, 15), 3))[1]
    assert best3 == (3, 7, 11)
    first = rsr.ranking(passes, c)["0"][0]
    assert parse_key(first)[0] in {(3, 7, 11), (3, 7, 12), (3, 11, 12)} or loss_of(first) <= loss_of(pass_key((3, 7, 11), 8192)) + 0.003
    # the winner of all measured cuts is within a hair of the true optimum over every k
    true = min(loss_of(pass_key(s, w)) for k, w in ((3, 8192), (4, 6144)) for s in itertools.combinations(range(1, 15), k))
    assert loss_of(first) - true < 0.01
    # and it is not a neighbouring run
    assert parse_key(first)[0] != tuple(range(parse_key(first)[0][0], parse_key(first)[0][0] + len(parse_key(first)[0])))


def test_a_stop_and_a_repeat_goes_on_and_repeats_nothing():
    c = ctx()
    full = drive(c)[2]
    passes, rounds, order = drive(c, stop_after=22)
    assert len(order) == 22
    # resumed from what was measured and the round records: the rest, in the same order
    c2 = dict(c, rounds=[dict(r) for r in rounds])
    p2, r2, order2 = drive(c2, passes=dict(passes))
    assert order + order2 == full
    assert set(order).isdisjoint(order2)


def test_a_repeated_complete_search_asks_for_nothing():
    c = ctx()
    passes, rounds, _ = drive(c)
    assert rsr.next_round(passes, dict(c, rounds=rounds)) is None
    # even without the records of the rounds the measured passes are not asked for again (the refinement may look for new swaps: those are new passes, never repeats)
    again = rsr.next_round(passes, c)
    assert again is None or not set(again["keys"]) & set(passes)


def test_the_guardrail_limit_on_k_is_the_contexts():
    c = ctx(groups=[{"label": "g", "target_b": 10.0, "cuts": [{"k": 30, "width": 4096, "params": 1, "on_target": False}]}])
    passes, _, order = drive(c)
    # k = 30 is more than there are blocks that may be cut: no set of that size is proposed
    assert all(len(parse_key(k)[0]) <= 1 for k in order)


def test_trade_off_groups_share_the_singles():
    groups = [{"label": f"{p} %", "target_b": p / 4.0, "cuts": [{"k": k, "width": w, "params": 1, "on_target": True}]} for p, k, w in ((90, 2, 8192), (80, 3, 8192), (70, 4, 6144))]
    c = ctx(groups=groups)
    passes, rounds, order = drive(c)
    singles = [k for k in order if len(parse_key(k)[0]) == 1 and parse_key(k)[1] is None]
    assert len(singles) == len(set(singles)) == 14
    rk = rsr.ranking(passes, c)
    assert set(rk) == {"0", "1", "2"} and all(rk[g] for g in rk)
    fin = rsr.finalists(passes, c)
    assert set(fin) == {"0", "1", "2"}


def test_finalists_are_diverse():
    c = ctx()
    passes, _, _ = drive(c)
    fin = rsr.finalists(passes, c)["0"]
    assert len(fin) == 3
    sets = [parse_key(k)[0] for k in fin]
    for a, b in itertools.combinations(sets, 2):
        assert len(a) != len(b) or len(set(a) & set(b)) < len(a) - 1, (a, b)
    assert sum(1 for s in sets if len(s) == 3) <= 2


def test_the_plan_and_what_fits_the_budget():
    c = ctx()
    plan = rsr.round_plan({}, c)
    assert [p["name"] for p in plan] == ["singles", "width_refs", "sets_k", "combined", "refine1", "refine2"]
    assert plan[0]["count"] == 15
    total = sum(p["count"] for p in plan)
    assert rsr.planned_total({}, c) == total
    fit = rsr.affordable(plan, 16.0, 20 * 60, 180.0)
    assert fit["passes"] == min(total, int((20 * 60 - 180) // 16)) and fit["complete"] == (total <= (20 * 60 - 180) // 16)
    tight = rsr.affordable(plan, 16.0, 10 * 60, 180.0)
    assert tight["passes"] == 26 and tight["rounds"] == 2 and not tight["complete"]
    assert rsr.affordable(plan, 1.0, 10_000.0)["complete"] is True
    assert rsr.affordable(plan, 16.0, 100.0, 200.0)["passes"] == 0


def test_the_matrix_choice_and_thorough_add_sets():
    n_plain = len(drive(ctx())[2])
    n_thorough = len(drive(ctx(thorough=True))[2])
    assert n_thorough > n_plain
    json.dumps(rsr.round_plan({}, ctx(thorough=True)))
