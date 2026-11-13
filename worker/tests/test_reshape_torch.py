"""Slim or grow with real (tiny, random) models on a CPU. Skipped when torch / transformers are not installed (CI has them)."""
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_reshape import job, run_job, types  # noqa: E402
from tinymodel import build_arch_dir, sample_rows, write_jsonl  # noqa: E402


def load(path):
    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32)
    return model.eval()


def logits_of(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


IDS = torch.tensor([[5, 17, 42, 99, 3, 250, 8, 61, 120, 7, 33, 4]])


@pytest.fixture(scope="module")
def base12(tmp_path_factory):
    return build_arch_dir(tmp_path_factory.mktemp("arch12") / "base", "qwen3_5", num_layers=12)


@pytest.fixture(scope="module")
def base8(tmp_path_factory):
    return build_arch_dir(tmp_path_factory.mktemp("arch8") / "base", "qwen3_5", num_layers=8)


def test_pruned_model_equals_the_original_with_those_layers_skipped(tmp_path, base12):
    j = job(tmp_path, base12, "prune", drop_layers=[4, 5, 6, 7])
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    pruned = load(j["final"])
    ref = load(base12)
    keep = ev[-1]["data"]["src_layers"]
    assert keep == [0, 1, 2, 3, 8, 9, 10, 11]
    ref.model.layers = torch.nn.ModuleList([ref.model.layers[i] for i in keep])
    ref.config.num_hidden_layers = 8
    ref.config.layer_types = [ref.config.layer_types[i] for i in keep]
    assert pruned.config.num_hidden_layers == 8 and list(pruned.config.layer_types) == types(8)
    assert torch.allclose(logits_of(pruned, IDS), logits_of(ref, IDS), atol=1e-5)


def test_identity_grow_is_exact_and_the_self_check_passes(tmp_path, base8):
    j = job(tmp_path, base8, "grow", check=False)
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    grown, orig = load(j["final"]), load(base8)
    assert grown.config.num_hidden_layers == 12
    assert torch.allclose(logits_of(grown, IDS), logits_of(orig, IDS), atol=1e-5)
    # the same through the job's own check (uses the built-in sentences when there is no training file)
    j2 = job(tmp_path, base8, "grow", check=True, out=str(tmp_path / "m2" / "model.partial"), final=str(tmp_path / "m2" / "model"))
    Path(j2["out"]).parent.mkdir()
    code, ev = run_job(tmp_path, j2)
    assert code == 0, ev[-1]
    r = ev[-1]["data"]
    assert r["loss_before"] is not None and math.isfinite(r["loss_before"]) and abs(r["loss_after"] - r["loss_before"]) <= 1e-3 * r["loss_before"]
    assert r["ppl_before"] == pytest.approx(r["ppl_after"], rel=1e-2)


def test_grow_by_copying_loads_and_generates(tmp_path, base8):
    j = job(tmp_path, base8, "grow", grow={"layout": "solar", "init": "copy"})
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    model = load(j["final"])
    assert len(model.config.layer_types) == 12 and list(model.config.layer_types) == types(12)
    with torch.no_grad():
        out = model.generate(input_ids=IDS, max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert out.shape[1] == IDS.shape[1] + 4


def test_scoring_chooses_the_middle_block_and_reports_numbers(tmp_path, base12):
    train = tmp_path / "train.jsonl"
    write_jsonl(train, sample_rows(12, 4))
    j = job(tmp_path, base12, "prune", blocks=1, check=True, calibration={"train": str(train), "samples": 16, "max_len": 64, "seed": 1, "version": 1})
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    r = ev[-1]["data"]
    assert r["chosen_blocks"] == [1] and len(r["boundary_distance"]) == 3 and r["dropped"] == [4, 5, 6, 7]       # only the middle block may go
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in r["boundary_distance"])
    assert math.isfinite(r["loss_before"]) and math.isfinite(r["loss_after"]) and r["stable"] in (True, False)
    scores = [e for e in ev if e["type"] == "reshape_scores"]
    assert len(scores) == 1 and scores[0]["chosen_blocks"] == [1] and scores[0]["period"] == 4 and len(scores[0]["matrix"]) == 4
    assert {"calibrate", "score", "write", "check"} <= {e["step"] for e in ev if e["type"] == "progress"}
    meta = json.loads((Path(j["final"]) / "ndforge_reshape.json").read_text())
    assert meta["calibration"]["version"] == 1 and meta["calibration"]["tokens"] > 0 and meta["loss_after"] == r["loss_after"]
    assert load(j["final"]).config.num_hidden_layers == 8
