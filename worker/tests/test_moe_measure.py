"""The measuring job's pieces (statistics over batches, stability of the halves, the cut model's load) on the stand-in model; the job's refusals through the worker."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import inkfake

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import test_moe_stats as T  # noqa: E402
from ndforge_worker import moe_layout as ml  # noqa: E402
from ndforge_worker import moemeasure as mm  # noqa: E402

WORKER = Path(__file__).resolve().parents[1]


class Wrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, T.H)
        self.body = T.Model()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return self.body(self.emb(input_ids))


def batches(n=6):
    g = torch.Generator().manual_seed(2)
    return [{"input_ids": torch.randint(0, 50, (2, 9), generator=g), "attention_mask": torch.ones(2, 9, dtype=torch.long)} for _ in range(n)]


@pytest.fixture()
def model():
    torch.manual_seed(5)
    return Wrap().eval()


def test_measure_counts_every_token_and_stops_when_asked(model):
    layers = list(model.body.layers)
    res = mm.measure(model, layers, batches(), T.E, T.CORE, T.PASSES)
    assert sum(res["layers"]["0"]["freq"]) == 6 * 18 * T.K and sum(res["layers"]["1"]["freq"]) == 6 * 18 * T.K * T.PASSES
    assert mm.measure(model, layers, batches(), T.E, T.CORE, T.PASSES, should_stop=lambda: True) is None
    assert not layers[0].mlp.gate._forward_hooks                             # the hooks are gone after a stop


def test_the_load_of_the_cut_model_is_measured_on_the_same_text(model):
    layers = list(model.body.layers)
    scores = mm.measure(model, layers, batches(), T.E, T.CORE, T.PASSES)
    kept = ml.select_experts(scores, 4, "reap")
    more = mm.measure(model, layers, batches(), T.E, T.CORE, T.PASSES, reap=False, keep=kept)
    load = more["pruned_load"]
    assert sorted(load) == [str(i) for i in range(T.L)] and all(len(v) == 4 and sum(v) == pytest.approx(1.0) for v in load.values())


def test_stability_is_one_for_identical_halves_and_lower_for_different_ones(model):
    scores = mm.measure(model, list(model.body.layers), batches(8), T.E, T.CORE, T.PASSES)
    s = mm.stability(scores, 4)
    assert 0.0 < s <= 1.0
    same = {"layers": {l: {**row, "halves": {"a": row["halves"]["a"], "b": row["halves"]["a"]}} for l, row in scores["layers"].items()}}
    assert mm.stability(same, 4) == 1.0
    sm = mm.summary(scores, T.E, [4, 2])
    assert [r["experts"] for r in sm["counts"]] == [4, 2] and sm["counts"][0]["coverage_mean"] >= sm["counts"][1]["coverage_mean"]


def run_job(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=120)
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


def test_the_job_refuses_a_dense_model_and_an_old_transformers_in_plain_words(tmp_path):
    from fakemodel import fake_model
    dense = fake_model(tmp_path, 8)
    code, ev = run_job(tmp_path, {"job": "moe_measure", "source": {"repo": str(dense)}, "out_dir": str(tmp_path / "s" / "k"), "calibration": {}})
    assert code != 0 and ev[-1]["type"] == "error" and ev[-1]["code"] == "reshape_refused" and "Inkling-2-Mini" in ev[-1]["message"]
    from ndforge_worker import inkling_load
    if inkling_load.transformers_ok()[0]:
        pytest.skip("this computer's transformers is new enough: the job would go on to load the model")
    ink = tmp_path / "ink"
    inkfake.write_folder(ink)
    code, ev = run_job(tmp_path, {"job": "moe_measure", "source": {"repo": str(ink)}, "out_dir": str(tmp_path / "s" / "k"), "calibration": {}})
    assert code != 0 and ev[-1]["code"] == "reshape_refused" and "5.24.0 or newer" in ev[-1]["message"]       # this computer's transformers is older
