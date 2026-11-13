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


# ------------------------------------------------------------------------- healing: training and testing on a reshaped base

def reshaped_base(tmp_path, mode="prune", **over):
    """a 12-layer original in tmp_path/model (where the training helpers expect it) and the reshaped copy; returns (original, reshaped)"""
    original = build_arch_dir(tmp_path / "model", "qwen3_5", num_layers=12)
    j = job(tmp_path, original, mode, **({"drop_layers": [4, 5, 6, 7]} if mode == "prune" else {"blocks": 1}), **over)
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    return original, Path(j["final"])


def test_a_lora_run_on_the_pruned_base_is_tested_against_the_original(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from test_train_cpu import run_eval
    from test_train_cpu import run_job as run_train
    original, slim = reshaped_base(tmp_path)
    p, ev, out = run_train(tmp_path, "qwen3_5", "real", model={"repo": str(slim)}, hyper={"max_steps": 6})
    assert p.returncode == 0, p.stderr[-3000:]
    p2, ev2 = run_eval(tmp_path, "qwen3_5", adapter=str(out / "adapter"), trained_base=str(slim), load_4bit={"original": False, "trained": False})
    assert p2.returncode == 0, p2.stderr[-3000:]
    ans = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    assert set(ans) == {"original", "trained"} and set(ans["original"]) == {"a", "b"} == set(ans["trained"])
    assert ev2[-1]["data"]["loaded"] == {"original": "16-bit", "trained": "16-bit"}


def test_training_a_grown_identity_model_reaches_the_switched_off_output_projections(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from safetensors.torch import load_file
    from test_train_cpu import run_job as run_train
    original, grown = reshaped_base(tmp_path, "grow")
    p, ev, out = run_train(tmp_path, "qwen3_5", "real", model={"repo": str(grown)}, hyper={"max_steps": 4, "extra_targets": ["out_proj"]})
    assert p.returncode == 0, p.stderr[-3000:]
    keys = list(load_file(str(Path(ev[-1]["data"]["adapter_dir"]) / "adapter_model.safetensors")))
    assert any("linear_attn.out_proj" in k for k in keys), keys[:10]


def test_a_pruned_model_can_be_exported_to_gguf_with_llama_cpp(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from ndforge_worker import exportjob
    if exportjob.find_llama_cpp()[0] is None:
        pytest.skip("llama.cpp is not installed here (set LLAMA_CPP_DIR)")
    from test_train_cpu import run_export
    from test_train_cpu import run_job as run_train
    original, slim = reshaped_base(tmp_path)
    p, ev, out = run_train(tmp_path, "qwen3_5", "real", model={"repo": str(slim)}, hyper={"max_steps": 2})
    assert p.returncode == 0, p.stderr[-3000:]
    p2, ev2 = run_export(tmp_path, "qwen3_5", out, base={"repo": str(slim), "family": "qwen3_5"}, gguf=True, quant="q8_0")
    assert p2.returncode == 0, p2.stderr[-3000:]
    assert Path(ev2[-1]["data"]["gguf"]).stat().st_size > 0
