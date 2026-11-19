"""The adaptive search loop of reshape_try (no torch): a fake pass function and a fake clock."""
import pytest

from ndforge_worker import reshapesearch as rsr
from ndforge_worker import reshapetry as rt
from test_reshapesearch import ctx, result

pytestmark = pytest.mark.filterwarnings("ignore")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fresh():
    return {"passes": {}, "seconds_per_pass": None}


def runner(clock, seconds=10.0, log=None):
    def run(key):
        clock.t += seconds
        if log is not None:
            log.append(key)
        r = result(key)
        r["seconds"] = seconds
        return r
    return run


def test_it_runs_to_the_end_when_the_time_is_enough():
    clock, data, seen = Clock(), fresh(), []
    out = rt._search_loop(data, ctx(), runner(clock), lambda: False, clock, 10_000.0, seen.append)
    assert out["stop_reason"] == "done" and out["measured"] == len(data["passes"]) == len(seen)
    assert data["search"]["stop_reason"] == "done" and [r["name"] for r in data["search"]["rounds"]][0] == "singles"
    assert all(r["done"] for r in data["search"]["rounds"])
    info = seen[0]
    assert info["round"] == "singles" and info["round_index"] == 1 and info["planned_total"] >= 1 and info["budget_left_s"] > 0


def test_the_budget_stops_before_a_pass_that_would_not_fit_and_keeps_what_was_measured():
    clock, data = Clock(), fresh()
    out = rt._search_loop(data, ctx(), runner(clock, 10.0), lambda: False, clock, 100.0, lambda i: None)
    assert out["stop_reason"] == "budget"
    assert clock.t <= 100.0 and len(data["passes"]) == 10          # the 11th would end at 110 s
    assert data["search"]["stop_reason"] == "budget" and data["search"]["budget_s"] == 100.0


def test_the_stop_file_stops_the_loop():
    clock, data, n = Clock(), fresh(), [0]

    def stop():
        n[0] += 1
        return n[0] > 5
    out = rt._search_loop(data, ctx(), runner(clock), stop, clock, 10_000.0, lambda i: None)
    assert out["stop_reason"] == "stopped" and len(data["passes"]) == 5


def test_a_stop_inside_a_pass_drops_that_pass():
    clock, data = Clock(), fresh()
    calls = [0]

    def run(key):
        calls[0] += 1
        if calls[0] == 4:
            raise rt._Stopped()
        return runner(clock)(key)
    out = rt._search_loop(data, ctx(), run, lambda: False, clock, 10_000.0, lambda i: None)
    assert out["stop_reason"] == "stopped" and len(data["passes"]) == 3


def test_cached_passes_cost_nothing_and_a_second_search_with_more_time_goes_on():
    clock, data = Clock(), fresh()
    rt._search_loop(data, ctx(), runner(clock), lambda: False, clock, 100.0, lambda i: None)
    first = set(data["passes"])
    assert data["search"]["stop_reason"] == "budget"
    log = []
    clock2 = Clock()
    out = rt._search_loop(data, ctx(), runner(clock2, log=log), lambda: False, clock2, 10_000.0, lambda i: None)
    assert out["stop_reason"] == "done"
    assert not first & set(log) and len(log) == len(set(log))
    # a third search over the same results measures nothing
    log3 = []
    out = rt._search_loop(data, ctx(), runner(Clock(), log=log3), lambda: False, Clock(), 10_000.0, lambda i: None)
    assert out["measured"] == 0 and log3 == [] and out["stop_reason"] == "done"


def test_a_search_with_other_targets_starts_its_rounds_again_but_keeps_the_passes():
    clock, data = Clock(), fresh()
    rt._search_loop(data, ctx(), runner(clock), lambda: False, clock, 10_000.0, lambda i: None)
    n = len(data["passes"])
    other = ctx(groups=[{"label": "g", "target_b": 18.0, "cuts": [{"k": 5, "width": 6144, "params": 1, "on_target": True}]}])
    assert rt.search_signature(other) != rt.search_signature(ctx())
    log = []
    rt._search_loop(data, other, runner(Clock(), log=log), lambda: False, Clock(), 10_000.0, lambda i: None)
    assert len(data["passes"]) == n + len(log) and not set(log) & set(list(data["passes"])[:n])
    assert log, "k = 5 needs sets that were not measured"


def test_the_max_passes_limit():
    clock, data = Clock(), fresh()
    out = rt._search_loop(data, ctx(), runner(clock), lambda: False, clock, 10_000.0, lambda i: None, max_passes=20)
    assert out["stop_reason"] == "max_passes" and len(data["passes"]) == 20


def test_the_summary_has_the_ranking_the_finalists_and_the_rounds():
    clock, data = Clock(), fresh()
    c = ctx()
    rt._search_loop(data, c, runner(clock), lambda: False, clock, 10_000.0, lambda i: None)
    s = rt.search_summary(data, c)
    assert s["stop_reason"] == "done" and s["ranking"]["0"] and len(s["finalists"]["0"]) == 3
    assert s["groups"] == [{"label": "20 B", "target_b": 20.0}] and s["rounds"][0]["name"] == "singles"
    assert rt._best_so_far(data["passes"], c)[0] == s["ranking"]["0"][0]


def test_the_request_is_checked():
    base = {"groups": [{"label": "g", "cuts": [{"k": 3, "width": 8192}]}], "budget_min": 20}
    ok = rt._search_context(base, 16, 1, 1, "contiguous", None)
    assert ok["budget_s"] == 1200.0 and ok["nb"] == 16 and ok["max_passes"] == 400
    for bad in ({**base, "groups": []}, {**base, "groups": [{"cuts": []}]}, {**base, "groups": [{"cuts": [{"k": 40}]}]}, {**base, "budget_min": 1}, {**base, "budget_min": 999},
                {**base, "groups": [{"label": str(i), "cuts": [{"k": 1}]} for i in range(9)]}):
        with pytest.raises(rt.ReshapeError):
            rt._search_context(bad, 16, 1, 1, "contiguous", None)
    assert rt._search_wants_width(base) and not rt._search_wants_width({"groups": [{"cuts": [{"k": 2}]}]}) and not rt._search_wants_width(None)
