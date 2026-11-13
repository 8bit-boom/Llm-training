"""The small hooks other jobs got for reshaped models (no torch needed): evaluate's sides, export's vision source, training's extra targets."""
import io
import json
from pathlib import Path

import pytest

from ndforge_worker import backends, evaluatejob, exportjob, protocol


def cfg(**over):
    c = {"model": {"repo": "org/Orig-9B", "family": "qwen3_5"}, "adapter": "/runs/r1/adapter"}
    c.update(over)
    return c


def test_evaluate_sides_for_a_plain_adapter_run_are_unchanged():
    s = evaluatejob.plan_sides(cfg())
    assert s["kind"] == "adapter" and s["original"] == ("org/Orig-9B", False) and s["trained"] == ("org/Orig-9B", False)
    assert s["loaded"] == {"original": "16-bit", "trained": "16-bit"}
    # one model carries both sides of an adapter test, so a 4-bit wish for either side applies to both
    s = evaluatejob.plan_sides(cfg(load_4bit={"trained": True}))
    assert s["original"] == ("org/Orig-9B", True) and s["loaded"] == {"original": "4-bit", "trained": "4-bit"}


def test_evaluate_sides_for_a_run_on_a_slimmed_base():
    s = evaluatejob.plan_sides(cfg(trained_base="/models/m1/model", load_4bit={"original": True, "trained": False}))
    assert s["kind"] == "adapter_on_other_base"
    assert s["original"] == ("org/Orig-9B", True) and s["trained"] == ("/models/m1/model", False)
    assert s["loaded"] == {"original": "4-bit", "trained": "16-bit"}


def test_evaluate_sides_for_a_full_model_and_for_nothing_to_test():
    s = evaluatejob.plan_sides(cfg(adapter=None, full_model="/models/m1/model", trained_base="ignored", load_4bit={"original": True, "trained": True}))
    assert s["kind"] == "full" and s["original"] == ("org/Orig-9B", True) and s["trained"] == ("/models/m1/model", True)
    with pytest.raises(RuntimeError, match="no saved model"):
        evaluatejob.plan_sides(cfg(adapter=None))


def test_export_takes_the_vision_part_from_the_original_of_a_reshaped_base():
    assert exportjob._vision_source({"base": {"repo": "/models/m1/model", "family": "qwen3_5", "vision_repo": "unsloth/Qwen3.5-27B"}}) == "unsloth/Qwen3.5-27B"
    assert exportjob._vision_source({"base": {"repo": "/models/m1/model", "family": "qwen3_5", "vision_repo": None}}) == "/models/m1/model"
    assert exportjob._vision_source({"base": {"repo": "unsloth/Qwen3.5-9B", "family": "qwen3_5"}}) == "unsloth/Qwen3.5-9B"


def run_export_with_stand_ins(tmp_path, monkeypatch, base):
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    asked = {}

    def modalities(repo, hub_download=None):
        asked["modalities"] = repo
        return ["image"]

    def fetch(repo, out, **kw):
        asked["mmproj"] = repo
        return None, "stand-in"

    monkeypatch.setattr(exportjob.vision, "base_modalities", modalities)
    monkeypatch.setattr(exportjob.vision, "fetch_mmproj", fetch)
    monkeypatch.setattr(exportjob, "merge_hf", lambda c, merged: Path(merged).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))

    def fake_gguf(c, merged, out, quants):
        f = Path(out) / "model-q4_k_m.gguf"
        f.write_bytes(b"G")
        return {"q4_k_m": f}

    monkeypatch.setattr(exportjob, "gguf_with_llama_cpp", fake_gguf)
    job = {"job": "export", "adapter": str(tmp_path / "adapter"), "base": base, "out": str(tmp_path / "exp"), "quant": "q4_k_m", "backend": "hf", "gguf": True}
    assert exportjob.run(job) == 0
    return asked, [json.loads(line) for line in buf.getvalue().splitlines()][-1]


def test_export_asks_for_the_original_models_vision_file(tmp_path, monkeypatch):
    asked, last = run_export_with_stand_ins(tmp_path, monkeypatch, {"repo": str(tmp_path / "slim"), "family": "qwen3_5", "vision_repo": "unsloth/Qwen3.5-27B"})
    assert asked == {"modalities": "unsloth/Qwen3.5-27B", "mmproj": "unsloth/Qwen3.5-27B"} and last["type"] == "result"


def test_export_without_vision_repo_asks_about_the_base_itself(tmp_path, monkeypatch):
    asked, last = run_export_with_stand_ins(tmp_path, monkeypatch, {"repo": "unsloth/Qwen3.5-9B", "family": "qwen3_5"})
    assert asked == {"modalities": "unsloth/Qwen3.5-9B", "mmproj": "unsloth/Qwen3.5-9B"}


def test_lora_targets_and_layers():
    assert backends.lora_targets({}) == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    t = backends.lora_targets({"extra_targets": ["out_proj", "o_proj"]})
    assert t[-1] == "out_proj" and t.count("o_proj") == 1 and len(t) == 8
    assert backends.lora_layers({}) is None and backends.lora_layers({"layers_to_transform": []}) is None
    assert backends.lora_layers({"layers_to_transform": [4, "5"]}) == [4, 5]
