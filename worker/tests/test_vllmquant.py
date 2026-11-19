"""Which weights vLLM loads: the pick table as a pure function (the sizes are the research report's estimates; whether it really fits is for the owner's GPU)."""
import json

import pytest

from ndforge_worker import thinkcheck as tc
from ndforge_worker import thinkcheck_vllm as tv
from ndforge_worker import vllmengine as ve
from ndforge_worker import vllmquant as vq


def folder(tmp_path, gb, quantized=False):
    d = tmp_path / "m"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "compressed-tensors"}} if quantized else {}))
    (d / "model.safetensors").write_bytes(b"\0" * 1000)
    return d


E32 = {"vram_gb": 32}


@pytest.mark.parametrize("gb,quant,fits", [(10.0, "none", True), (22.0, "none", True), (23.0, "fp8", True), (34.0, "fp8", True), (39.0, "fp8", True), (40.0, "fp8", False), (55.0, "fp8", False)])
def test_pick_table_for_a_cut_on_a_32_gb_card(gb, quant, fits):
    r = vq.pick("/nowhere", {**E32, "weights_gb": gb})
    assert (r["quant"], r["fits"]) == (quant, fits), r
    assert r["usable_gb"] == 22.2


def test_the_original_uses_the_configured_pre_quantized_checkpoint():
    r = vq.pick("unsloth/Qwen3.5-27B", {**E32, "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "weights_gb": 55})
    assert r["model"] == "cyankiwi/Qwen3.8-27B-AWQ-INT4" and r["quant"] == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4" and r["fits"] is True
    assert vq.pick("unsloth/Qwen3.5-27B", {**E32, "original": True, "teacher_repo": ""})["quant"] in ("none", "fp8")       # nothing configured
    small = vq.pick("org/small", {**E32, "original": True, "teacher_repo": "x/y", "weights_gb": 8})
    assert small["quant"] == "none" and small["model"] == "org/small", "an original that fits is not replaced"


def test_a_given_choice_and_stored_quantization_win(tmp_path):
    assert vq.pick("m", {"quant": "fp8", "weights_gb": 1})["quant"] == "fp8"
    assert vq.pick("m", {"quant": "none", "weights_gb": 99})["quant"] == "none"
    r = vq.pick("m", {"quant": "org/awq"})
    assert (r["quant"], r["model"]) == ("prequant:org/awq", "org/awq")
    s = vq.pick(str(folder(tmp_path, 1, quantized=True)), E32)
    assert s["quant"] == "stored:compressed-tensors"


def test_sizes_from_a_folder_and_a_hub_id_of_unknown_size(tmp_path):
    d = folder(tmp_path, 1)
    assert vq.weights_gb(str(d)) == 0.0 and vq.weights_gb("org/x") is None
    assert vq.pick("org/x", E32)["quant"] == "none" and vq.pick("org/x", E32)["fits"] is None


def test_language_model_only_is_always_passed_and_fp8_only_when_picked():
    assert "--language-model-only" in ve.serve_args("m", quant="none") and "--quantization" not in ve.serve_args("m", quant="prequant:x/y")
    assert ve.serve_args("m", quant="fp8")[ve.serve_args("m", quant="fp8").index("--quantization") + 1] == "fp8"


def test_the_engine_uses_the_pick_and_a_misfit_is_a_plain_error():
    job = {"model": "unsloth/Qwen3.5-27B", "backend": "vllm", "engine": {"kind": "docker", "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "vram_gb": 32}}
    e = tv.make_engine(job)
    assert e.model == "cyankiwi/Qwen3.8-27B-AWQ-INT4" and e.quant == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4"
    assert job["_engine_report"]["pick"]["reason"].startswith("The original does not fit")
    assert tc.cache_key(job, "vllm", tc.settings_of({}))["engine"]["quant"] == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4"
    with pytest.raises(tc.ThinkCheckError, match="do not fit on this card"):
        tv.make_engine({"model": "org/huge", "engine": {"kind": "docker", "weights_gb": 80, "vram_gb": 32}})
    cut = tv.make_engine({"model": "org/cut", "engine": {"kind": "docker", "weights_gb": 34, "vram_gb": 32}})
    assert cut.quant == "fp8" and "--quantization" in cut.build()
