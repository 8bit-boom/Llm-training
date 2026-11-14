"""Try before cutting with real (tiny, random) Qwen3.5 models on a CPU: the predicted loss of a choice must equal the calibration loss of the model the `reshape` job
writes for the same choice. Skipped when torch / transformers are not installed (CI has them)."""
import io
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from test_reshape import run_job  # noqa: E402
from test_reshape_width import wjob  # noqa: E402
from test_reshape_width_torch import HEADS, KV, NV, measure_cfg, perturb, run_measure  # noqa: E402
from tinymodel import build_arch_dir  # noqa: E402

from ndforge_worker import reshapejob, reshapetry as rt  # noqa: E402

LAYERS = 24                  # six blocks of four: blocks 1 to 4 may be cut, at most three of them
CAL = {"samples": 8, "max_len": 64, "batch": 2, "version": 1, "seed": 3407}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = build_arch_dir(tmp_path_factory.mktemp("try24") / "base", "qwen3_5", num_layers=LAYERS, heads=HEADS, kv_heads=KV, lin_v_heads=NV)
    perturb(d)
    return d


@pytest.fixture()
def measured(tmp_path, base, monkeypatch):
    """the project's measurement of the model (needed for the neuron scores) and the calibration that goes with it"""
    folder = tmp_path / "models" / "_scores" / "k1"
    code, _ = run_measure(monkeypatch, measure_cfg(tmp_path, base, folder, calibration={"train": str(tmp_path / "train.jsonl"), **CAL}))
    assert code == 0
    return folder, {"train": str(tmp_path / "train.jsonl"), **CAL}


def try_cfg(tmp_path, base, measured, want, out="t1", **over):
    folder, cal = measured
    return {"job": "reshape_try", "source": {"repo": str(base), "family": "qwen3_5"}, "load": "16bit", "out_dir": str(tmp_path / "models" / "_scores" / out), "calibration": cal,
            "measure_dir": str(folder), "measure_samples": cal["samples"], "protect": {"first": 1, "last": 1}, "strategy": "contiguous", "want": want, "ffn_divisor": 4,
            "params_b": 0.001, **over}


def do_try(monkeypatch, cfg, after_pass=None):
    from ndforge_worker import protocol
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    if after_pass:
        real = protocol.emit

        def emit(kind, **kw):
            real(kind, **kw)
            if kind == "try_pass":
                after_pass(kw)
        monkeypatch.setattr(protocol, "emit", emit)
    code = rt.run_try(cfg)
    return code, [json.loads(line) for line in buf.getvalue().splitlines()]


def passes_of(tmp_path, out="t1"):
    return json.loads((tmp_path / "models" / "_scores" / out / "reshape_try.json").read_text(encoding="utf-8"))["passes"]


def written_loss(tmp_path, base, measured, name, **over):
    """the calibration loss of the model the existing job writes: its own check, and the same loss computed on the written folder"""
    folder, cal = measured
    d = tmp_path / "w" / name
    d.mkdir(parents=True)
    j = wjob(tmp_path, base, over.pop("width", None), out=str(d / "model.partial"), final=str(d / "model"), scores_dir=str(folder), calibration=cal, load="16bit", **over)
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    loss, _, n = reshapejob._loss(d / "model", {"calibration": cal, "source": {"family": "qwen3_5"}}, "16bit")
    return loss, ev[-1]["data"], n


def close(a, b):
    assert a is not None and b is not None and a == pytest.approx(b, rel=1e-4), (a, b)


# ------------------------------------------------------------------------------------------------------ the strongest tests

def test_the_baseline_is_the_loss_of_the_original(tmp_path, base, measured, monkeypatch):
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {}))
    assert code == 0 and ev[-1]["type"] == "result"
    folder, cal = measured
    ref, _, n = reshapejob._loss(base, {"calibration": cal, "source": {"family": "qwen3_5"}}, "16bit")
    got = passes_of(tmp_path)[rt.BASE_KEY]
    close(got["loss"], ref)
    assert got["tokens"] == n and math.isfinite(got["loss_h1"]) and math.isfinite(got["loss_h2"]) and got["seconds"] >= 0
    meta = json.loads((folder / "scores.json").read_text())
    close(ev[-1]["data"]["analysis"]["base"]["loss"], meta["loss_before"])          # the measurement's own number
    assert meta["seconds_per_pass"] >= 0          # the measurement records how long its pass took (the first estimate of a pass)


@pytest.mark.parametrize("blocks", [[1], [3], [2, 3], [1, 4], [1, 2, 3]], ids=lambda b: "blocks" + "-".join(map(str, b)))
def test_skipping_blocks_predicts_the_loss_of_the_pruned_model(tmp_path, base, measured, monkeypatch, blocks):
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"choices": [{"blocks": blocks}]}))
    assert code == 0
    predicted = passes_of(tmp_path)[rt.pass_key(blocks)]["loss"]
    actual, r, _ = written_loss(tmp_path, base, measured, "d", drop_layers=reshape_layers(blocks))
    assert r["layers_after"] == LAYERS - 4 * len(blocks)
    close(predicted, actual)
    assert abs(predicted - passes_of(tmp_path)[rt.BASE_KEY]["loss"]) > 1e-6          # the test would be empty if the cut changed nothing


def reshape_layers(blocks):
    return [4 * b + i for b in blocks for i in range(4)]


@pytest.mark.parametrize("width", [96, 80])
def test_masking_neurons_predicts_the_loss_of_the_thinned_model(tmp_path, base, measured, monkeypatch, width):
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"widths": [width]}))
    assert code == 0
    predicted = passes_of(tmp_path)[rt.pass_key((), width)]["loss"]
    actual, r, _ = written_loss(tmp_path, base, measured, "w", width={"intermediate_size": width})
    assert r["shape"] == "width"
    close(predicted, actual)
    assert abs(predicted - passes_of(tmp_path)[rt.BASE_KEY]["loss"]) > 1e-6


@pytest.mark.parametrize("blocks,width", [([2], 96), ([1, 2], 80)])
def test_depth_and_width_together_predict_the_loss_of_the_written_model(tmp_path, base, measured, monkeypatch, blocks, width):
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"choices": [{"blocks": blocks, "width": width}]}))
    assert code == 0
    predicted = passes_of(tmp_path)[rt.pass_key(blocks, width)]["loss"]
    actual, r, _ = written_loss(tmp_path, base, measured, "both", width={"intermediate_size": width}, drop_layers=reshape_layers(blocks))
    assert r["shape"] == "both"
    close(predicted, actual)


def test_the_jobs_own_check_gives_the_same_number(tmp_path, base, measured, monkeypatch):
    """loss_after of the `reshape` job (its built-in before/after check) is the predicted loss; loss_before is the baseline"""
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"choices": [{"blocks": [2]}]}))
    assert code == 0
    _, r, _ = written_loss(tmp_path, base, measured, "chk", drop_layers=reshape_layers([2]), check=True)
    p = passes_of(tmp_path)
    close(p[rt.pass_key((2,))]["loss"], r["loss_after"])
    close(p[rt.BASE_KEY]["loss"], r["loss_before"])


def test_two_blocks_together_cost_more_than_the_sum_of_two_single_costs_is_visible_in_the_numbers(tmp_path, base, measured, monkeypatch):
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"blocks": True, "runs": [2]}))
    assert code == 0
    a = ev[-1]["data"]["analysis"]
    assert a["runs"]["2"]["complete"] and a["runs"]["1"]["complete"]
    for w in a["runs"]["2"]["windows"]:
        s = sum(a["blocks"][b - 1]["delta"] for b in w["blocks"])
        assert w["delta"] != pytest.approx(s, abs=1e-9)           # the honest number is not the sum (the pass is the only way to know)


# ------------------------------------------------------------------------------------------------------ windows, caching, resume

def test_windows_equal_the_single_computations_and_the_cheapest_is_the_lowest(tmp_path, base, measured, monkeypatch):
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"runs": [2]}))
    assert code == 0
    a = ev[-1]["data"]["analysis"]
    r2 = a["runs"]["2"]
    assert [w["blocks"] for w in r2["windows"]] == [[1, 2], [2, 3], [3, 4]] and r2["complete"]
    assert r2["best"]["loss"] == min(w["loss"] for w in r2["windows"]) and r2["default"] is not None and r2["default"]["loss"] >= r2["best"]["loss"]
    # every window asked for on its own, in another folder and in another order, gives the same number
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"choices": [{"blocks": w["blocks"]} for w in reversed(r2["windows"])]}, out="t2"))
    assert code == 0
    p1, p2 = passes_of(tmp_path), passes_of(tmp_path, "t2")
    for w in r2["windows"]:
        assert p1[rt.pass_key(w["blocks"])]["loss"] == p2[rt.pass_key(w["blocks"])]["loss"] == w["loss"]


def test_a_pass_leaves_the_model_as_it_was(tmp_path, base, monkeypatch):
    from test_reshape_torch import load
    model = load(base)
    layers = reshapejob._find_layers(model, LAYERS)
    batches = [{"input_ids": torch.tensor([[5, 17, 42, 99, 3, 250, 8, 61]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}] * 2
    scores = np.random.default_rng(1).random((LAYERS, 128))
    run = lambda k: rt._one_pass(model, layers, batches, 4, k, scores, 128, lambda: False)["loss"]  # noqa: E731
    before = run(rt.BASE_KEY)
    assert run("d=2|w=96") != before and run("d=1,2|w=") != before
    assert run(rt.BASE_KEY) == before and all("forward" not in l.__dict__ for l in layers)
    assert all(not l.mlp.down_proj._forward_pre_hooks for l in layers)


def test_a_repeated_try_runs_nothing_and_another_key_is_computed_again(tmp_path, base, measured, monkeypatch):
    want = {"runs": [2]}
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want))
    assert code == 0 and sum(1 for e in ev if e["type"] == "try_pass") == 4          # the original and three windows
    first = passes_of(tmp_path)

    def boom(*a, **k):
        raise AssertionError("the model must not be loaded again")
    monkeypatch.setattr(reshapejob, "_load", boom)
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want))
    assert code == 0 and not any(e["type"] == "try_pass" for e in ev) and passes_of(tmp_path) == first
    assert ev[-1]["data"]["passes_done"] == ev[-1]["data"]["passes_total"] == 4
    # one more answer asked for: only that one is run
    monkeypatch.undo()
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"runs": [2], "choices": [{"blocks": [1]}]}))
    assert [e["key"] for e in ev if e["type"] == "try_pass"] == ["d=1|w="] and all(passes_of(tmp_path)[k] == v for k, v in first.items())
    # other settings: the old answers are dropped, with the reason in the log
    cfg = try_cfg(tmp_path, base, measured, want)
    cfg["calibration"] = {**cfg["calibration"], "samples": 6}
    code, ev = do_try(monkeypatch, cfg)
    assert code == 0 and any("another number of samples" in e.get("text", "") for e in ev if e["type"] == "log") and sum(1 for e in ev if e["type"] == "try_pass") == 4


def test_resuming_after_a_stop_gives_the_same_results_as_one_run(tmp_path, base, measured, monkeypatch):
    want = {"blocks": True, "runs": [2]}
    stop = tmp_path / "models" / "_scores" / "STOP"
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want), after_pass=lambda kw: stop.write_text("") if kw["done"] == 3 else None)
    assert code == 0 and ev[-1]["data"]["stopped"] is True and ev[-1]["data"]["passes_done"] == 3
    assert len(passes_of(tmp_path)) == 3 and sum(1 for e in ev if e["type"] == "try_pass") == 3          # the three finished passes were saved after each pass
    stop.unlink()
    monkeypatch.undo()
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want))
    total = ev[-1]["data"]["passes_total"]
    assert code == 0 and ev[-1]["data"]["stopped"] is False and total == 1 + 4 + 3 and ev[-1]["data"]["passes_done"] == total
    assert sum(1 for e in ev if e["type"] == "try_pass") == total - 3          # only the rest was run
    monkeypatch.undo()
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want, out="t_once"))
    assert code == 0
    a, b = passes_of(tmp_path), passes_of(tmp_path, "t_once")
    assert set(a) == set(b) and all(a[k]["loss"] == b[k]["loss"] and a[k]["loss_h1"] == b[k]["loss_h1"] for k in a)


def test_a_stop_before_the_first_pass_keeps_nothing_and_loses_nothing(tmp_path, base, measured, monkeypatch):
    (tmp_path / "models" / "_scores" / "STOP").write_text("")
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"blocks": True}))
    assert code == 0 and ev[-1]["data"]["stopped"] is True and ev[-1]["data"]["passes_done"] == 0 and passes_of(tmp_path) == {}


def test_a_width_without_a_measurement_is_refused_and_depth_needs_none(tmp_path, base, measured, monkeypatch):
    cfg = try_cfg(tmp_path, base, measured, {"widths": [96]}, measure_dir=str(tmp_path / "nothing"))
    with pytest.raises(rt.ReshapeError, match="Measure the model first"):
        do_try(monkeypatch, cfg)
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"choices": [{"blocks": [2]}]}, measure_dir=str(tmp_path / "nothing")))
    a = ev[-1]["data"]["analysis"]
    assert code == 0 and a["choices"] == [] and a["runs"]["1"]["default"] is None and a["runs"]["1"]["windows"][1]["distance"] is None and rt.pass_key((2,)) in passes_of(tmp_path)


def test_the_progress_and_the_events_of_a_run(tmp_path, base, measured, monkeypatch):
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, {"blocks": True}))
    steps = [e for e in ev if e["type"] == "progress" and e["step"] == "try"]
    assert code == 0 and any("Cut 2 of 5: block 1 skipped" in e["text"] for e in steps) and steps[-1]["value"] == 1.0
    passes = [e for e in ev if e["type"] == "try_pass"]
    assert [e["done"] for e in passes] == [1, 2, 3, 4, 5] and all(e["total"] == 5 and e["analysis"]["base"]["loss"] is not None for e in passes)
    assert ev[-1]["data"]["seconds_per_pass"] is not None and ev[-1]["data"]["load"] == "16bit" and ev[-1]["data"]["analysis"]["blocks_stability"]["n"] == 4
    assert json.loads((tmp_path / "models" / "_scores" / "t1" / "reshape_try.json").read_text())["seconds_per_pass"] == ev[-1]["data"]["seconds_per_pass"]
