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
    r = vq.pick("Qwen/Qwen3.8-27B", {**E32, "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "weights_gb": 55})
    assert r["model"] == "cyankiwi/Qwen3.8-27B-AWQ-INT4" and r["quant"] == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4" and r["fits"] is True
    assert vq.pick("unsloth/Qwen3.5-27B", {**E32, "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "weights_gb": 55})["quant"] == "bnb-nf4"       # another original: never that checkpoint
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
    job = {"model": "Qwen/Qwen3.8-27B", "backend": "vllm", "engine": {"kind": "docker", "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "vram_gb": 32, "weights_gb": 55}}
    e = tv.make_engine(job)
    assert e.model == "cyankiwi/Qwen3.8-27B-AWQ-INT4" and e.quant == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4"
    assert job["_engine_report"]["pick"]["reason"].startswith("The original does not fit")
    assert tc.cache_key(job, "vllm", tc.settings_of({}))["engine"]["quant"] == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4"
    with pytest.raises(tc.ThinkCheckError, match="do not fit on this card"):
        tv.make_engine({"model": "org/huge", "engine": {"kind": "docker", "weights_gb": 80, "vram_gb": 32}})
    cut = tv.make_engine({"model": "org/cut", "engine": {"kind": "docker", "weights_gb": 34, "vram_gb": 32}})
    assert cut.quant == "fp8" and "--quantization" in cut.build()


# ---- the identity rule for checks, trace generation and the self-test's "original" runs (the same rule as the teacher cache)

E_BIG = {"kind": "docker", "original": True, "vram_gb": 32, "weights_gb": 55}


def test_a_finetune_original_is_never_served_as_the_configured_awq():
    for m in ("Someone/Qwen3.8-27B-Heretic", "C:\\models\\heretic"):
        p = vq.pick(m, {**E_BIG, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4"})
        assert p["quant"] == "bnb-nf4" and p["model"] == m and "cyankiwi" not in json.dumps(p), p
        assert p["served"] == f"original: {m} (4-bit on the fly)" and p["fits"] is True and "UNVERIFIED" in p["reason"]
    assert "--quantization" in ve.serve_args("m", quant="bnb-nf4")


def test_the_base_original_and_a_confirmed_copy_may_use_a_pre_quantized_checkpoint():
    p = vq.pick("Qwen/Qwen3.8-27B", E_BIG)
    assert p["quant"] == "prequant:cyankiwi/Qwen3.8-27B-AWQ-INT4" and p["served"] == "confirmed copy cyankiwi/Qwen3.8-27B-AWQ-INT4"
    c = vq.pick("Someone/Finetune", {**E_BIG, "teacher": {"confirmed": {"Someone/Finetune": "Someone/Finetune-AWQ"}, "quantized_repo": "Someone/Finetune-AWQ"}})
    assert c["quant"] == "prequant:Someone/Finetune-AWQ" and c["model"] == "Someone/Finetune-AWQ"
    other = vq.pick("Other/Model", {**E_BIG, "teacher": {"confirmed": {"Someone/Finetune": "Someone/Finetune-AWQ"}}})
    assert other["quant"] == "bnb-nf4", "a confirmation is per original"


def test_a_refused_route_or_a_model_too_big_even_in_4_bits_is_a_plain_refusal_never_the_base_checkpoint():
    p = vq.pick("Someone/Finetune", {**E_BIG, "teacher": {"route": "prequant"}})
    assert p["fits"] is False and "wrong model" in p["reason"]
    huge = vq.pick("Someone/Huge", {**E_BIG, "weights_gb": 200})
    assert huge["fits"] is False and "NOT used for another model" in huge["reason"] and huge["model"] == "Someone/Huge"
    with pytest.raises(tc.ThinkCheckError, match="wrong model"):
        tv.make_engine({"model": "Someone/Finetune", "engine": {**E_BIG, "kind": "docker", "teacher": {"route": "prequant"}}})


def test_the_check_key_and_report_say_which_model_was_loaded():
    job = {"model": "Someone/Finetune", "backend": "vllm", "engine": {**E_BIG, "kind": "docker"}}
    e = tv.make_engine(job)
    assert e.quant == "bnb-nf4" and e.model == "Someone/Finetune"
    key = tc.cache_key(job, "vllm", tc.settings_of({}))["engine"]
    assert key["quant"] == "bnb-nf4" and key["teacher_origin"] == "Someone/Finetune"
    assert job["_engine_report"]["pick"]["served"].startswith("original: Someone/Finetune")
    plain = {"model": "org/cut", "backend": "vllm", "engine": {"kind": "docker", "vram_gb": 32, "weights_gb": 20}}
    assert "teacher_origin" not in tc.cache_key(plain, "vllm", tc.settings_of({}))["engine"], "other checks keep their keys"
