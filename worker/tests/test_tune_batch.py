"""The tuner's batch tests on tiny random Qwen3.5 models on a CPU (plumbing and the equivalence of the numbers; the speed of a CPU means nothing)."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_reshape_torch import load  # noqa: E402,F401
from tinymodel import build_arch_dir  # noqa: E402

from ndforge_worker import protocol, tunebatch as tb  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return build_arch_dir(tmp_path_factory.mktemp("tb") / "m", "qwen3_5", num_layers=8)


def go(monkeypatch, capsys, job):
    monkeypatch.setattr(protocol, "_out", __import__("sys").stdout)
    rc = tb.run(job)
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return rc, [x for x in lines if x["type"] == "result"][0]["data"]


def test_the_default_of_each_test_is_the_programs_own_number():
    assert (tb.effective("calib", 0), tb.effective("think", 0), tb.effective("calib", 8)) == (4, 8, 8)
    reqs = tb.think_requests(40, 6)
    assert len(reqs) == 6 and all(r["thinking"] and r["max_new"] == 40 for r in reqs)


def test_the_calibration_batch_sweep_measures_the_same_loss_at_every_size(tiny, tmp_path, monkeypatch, capsys):
    train = tmp_path / "train.jsonl"
    rows = [{"type": "text", "text": f"The ferryman crossed the river {i} times before the snow came, and he counted every one of them twice."} for i in range(40)]
    train.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rc, res = go(monkeypatch, capsys, {"job": "tune_batch_bench", "what": "calib", "model": str(tiny), "load": "16bit", "family": "qwen3_5", "sizes": [0, 1, 2, 8], "repeat": 2,
                                       "calibration": {"train": str(train), "samples": 16, "max_len": 48, "seed": 3407}})
    sw = res["batch_sweep"]
    assert rc == 0 and res["what"] == "calib" and [r["batch_tokens"] for r in sw["results"]] == [0, 1, 2, 8]
    base = sw["results"][0]["first_loss"]
    assert all(r["tokens_per_s"] and r["first_loss"] is not None for r in sw["results"]), sw["results"]
    assert all(abs(r["first_loss"] - base) <= 1e-3 * max(1.0, abs(base)) for r in sw["results"]), "padding does not change the measured loss"
    assert "spread" in sw and "recommended" in sw


def test_the_thinking_batch_sweep_asks_the_same_questions_at_every_size(tiny, monkeypatch, capsys):
    rc, res = go(monkeypatch, capsys, {"job": "tune_batch_bench", "what": "think", "model": str(tiny), "load": "16bit", "sizes": [0, 4, 16], "repeat": 1, "max_new": 4})
    sw = res["batch_sweep"]
    assert rc == 0 and [r["batch_tokens"] for r in sw["results"]] == [0, 4, 16]
    assert all(r["tokens_per_s"] and r["tokens"] > 0 and r["questions"] == 16 for r in sw["results"]), sw["results"]


def test_an_unknown_test_is_refused_and_the_job_is_in_the_contract():
    with pytest.raises(tb.BenchError):
        tb.run({"what": "nope"})
    c = json.loads((__import__("pathlib").Path(tb.__file__).parent / "contract.json").read_text())
    assert "tune_batch_bench" in c["jobs"] and "tune_batch_bench_failed" in c["error_codes"]
