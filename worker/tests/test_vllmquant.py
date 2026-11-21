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


@pytest.mark.parametrize("gb,quant,fits", [(10.0, "none", True), (22.0, "none", True), (24.0, "none", True), (25.0, "fp8", True), (34.0, "fp8", True), (39.0, "fp8", True), (40.0, "fp8", True),
                                           (45.0, "bnb-nf4", True), (55.0, "bnb-nf4", True), (90.0, "bnb-nf4", False)])
def test_pick_table_for_a_local_model_on_a_32_gb_card(gb, quant, fits):
    # budget = 0.88 x 32 = 28.2 GB; a fit is weights + 1.5 GB overhead + 2.5 GB minimum cache (it was a fixed 6 GB reserve: 22.2 GB usable)
    r = vq.pick("/nowhere", {**E32, "weights_gb": gb})
    assert (r["quant"], r["fits"]) == (quant, fits), r
    assert r["usable_gb"] == 24.2


def test_a_hub_id_that_is_not_the_original_keeps_the_old_8_bit_rule():
    assert vq.pick("org/finetune", {**E32, "weights_gb": 40})["quant"] == "fp8"
    r = vq.pick("org/finetune", {**E32, "weights_gb": 70})
    assert r["quant"] == "fp8" and r["fits"] is False and "even in 8 bits" in r["reason"] and "budget" in r["reason"]


OWNER = {"vram_gb": 31.8, "gpu_memory_utilization": 0.88, "weights_gb": 39.8}


def test_the_owners_healed_model_fits_in_8_bits_with_29_gb_free_and_the_reason_has_the_arithmetic():
    r = vq.pick("/models/healed", {**OWNER, "gpu_free_gb": 29.8})
    assert r["quant"] == "fp8" and r["fits"] is True, r
    assert "weights 22.3 + overhead 1.5 = 23.8 GB of 28.0 GB (0.88 of the 31.8 GB card)" in r["reason"] and "leaving 4.2 GB of cache" in r["reason"] and "answers of 3.3k tokens at once" in r["reason"]
    assert 1 <= r["max_seqs"] <= 40 and r["gpu_util"] == 0.88


def test_with_the_original_served_as_nf4_the_healed_model_goes_the_same_way():
    r = vq.pick("/models/healed", {**OWNER, "gpu_free_gb": 29.8, "route_hint": "bnb-nf4"})
    assert r["quant"] == "bnb-nf4" and r["fits"] is True and r["like_for_like"] is True and "like-for-like" in r["reason"] and "UNVERIFIED for this model" in r["reason"]
    assert "stand-in" in r["reason"]


def test_a_route_that_cannot_be_matched_is_said_to_be_different():
    r = vq.pick("/models/healed", {**OWNER, "gpu_free_gb": 29.8, "route_hint": "prequant:x/y"})
    assert r["quant"] == "fp8" and r["like_for_like"] is False and "different quantization, not like-for-like" in r["reason"]
    small = vq.pick("/models/cut", {**E32, "weights_gb": 10, "route_hint": "bnb-nf4"})
    assert small["quant"] == "none" and small["like_for_like"] is False and "not like-for-like" in small["reason"]


def test_a_card_with_little_free_memory_refuses_with_the_real_arithmetic():
    r = vq.pick("/models/huge", {**OWNER, "weights_gb": 70, "gpu_free_gb": 20})
    assert r["fits"] is False and "even in 4 bits" in r["reason"]
    assert "weights 39.2 + overhead 1.5 + minimum cache 2.5 = 43.2 GB; budget 19.4 GB (free 20 GB minus 0.6)" in r["reason"], r["reason"]
    healed = vq.pick("/models/healed", {**OWNER, "gpu_free_gb": 14})
    assert healed["fits"] is False and "budget 13.4 GB" in healed["reason"]
    assert vq.pick("/models/healed", {**OWNER, "gpu_free_gb": 20})["quant"] == "bnb-nf4", "4 bits still fit in 19.4 GB: 11.1 + 4 = 15.1"


def test_an_unknown_free_memory_falls_back_to_the_configured_utilization():
    r = vq.pick("/models/healed", OWNER)
    assert r["fits"] is True and "the free memory is not known" in r["reason"] and r["gpu_util"] == 0.88


def test_the_share_of_the_card_is_raised_only_when_it_is_needed_and_the_free_memory_allows_it():
    r = vq.pick("/models/healed", {**OWNER, "gpu_memory_utilization": 0.80, "gpu_free_gb": 29.8})
    assert r["fits"] is True and r["gpu_util"] == 0.91 and "raised to 0.91" in r["reason"], r
    never = vq.pick("/models/healed", {**OWNER, "gpu_memory_utilization": 0.80, "gpu_free_gb": 40})
    assert never["gpu_util"] <= 0.92


def test_the_8_bit_size_comes_from_the_safetensors_headers(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    header = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [100000, 1000], "data_offsets": [0, 1]},
              "model.layers.0.mlp.up.weight": {"dtype": "BF16", "shape": [200000, 5000], "data_offsets": [0, 1]},
              "model.layers.0.norm.weight": {"dtype": "BF16", "shape": [100], "data_offsets": [0, 1]},
              "model.visual.blocks.0.attn.weight": {"dtype": "BF16", "shape": [10000, 10000], "data_offsets": [0, 1]},
              "lm_head.weight": {"dtype": "BF16", "shape": [100000, 1000], "data_offsets": [0, 1]}, "__metadata__": {"format": "pt"}}
    raw = json.dumps(header).encode()
    (d / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
    s = vq.route_weights_gb(str(d))
    bf16 = (100000 * 1000 * 2 + 200000 * 5000 * 2 + 100 * 2 + 10000 * 10000 * 2 + 100000 * 1000 * 2) / 1e9
    fp8 = (100000 * 1000 * 2 + 200000 * 5000 * 1 + 100 * 2 + 10000 * 10000 * 2 + 100000 * 1000 * 2) / 1e9
    assert s["bf16"] == pytest.approx(bf16, abs=0.01) and s["fp8"] == pytest.approx(fp8, abs=0.01) and s["fp8"] < s["bf16"] and s["nf4"] < s["fp8"] + 1e-9
    assert vq.route_weights_gb(str(tmp_path / "nothing")) is None
    (d / "model.safetensors").write_bytes(b"garbage")
    assert vq.route_weights_gb(str(d)) is None, "unreadable headers: the share is used"


def test_the_identity_rule_for_originals_is_unchanged():
    r = vq.pick("Qwen/Qwen3.8-27B", {**E32, "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "weights_gb": 55})
    assert r["quant"].startswith("prequant:cyankiwi/") and r["fits"] is True
    other = vq.pick("unsloth/Qwen3.5-27B", {**E32, "original": True, "teacher_repo": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "weights_gb": 55})
    assert other["quant"] == "bnb-nf4" and other["fits"] is True and "UNVERIFIED" in other["reason"] and "prequant" not in other["quant"]


def test_the_cache_key_separates_the_routes(tmp_path):
    from ndforge_worker import vllmengine as vme
    job = {"engine": {"kind": "docker"}, "model": "/m"}
    assert vme.engine_identity(job, "fp8", False)["quant"] != vme.engine_identity(job, "bnb-nf4", False)["quant"]


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
