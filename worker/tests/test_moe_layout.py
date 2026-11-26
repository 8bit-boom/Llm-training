"""Inkling-2-Mini identity: the real config and tensor list, the layer taxonomy, the inspect result (no torch, no download)."""
import json

import pytest

import inkfake
from ndforge_worker import moe_layout as ml
from ndforge_worker import reshape, reshapejob
from ndforge_worker.reshape import ReshapeError

REAL_TENSORS = {k: (v[0], tuple(v[1])) for k, v in json.loads((inkfake.DATA / "inkling2_tensors.json").read_text(encoding="utf-8")).items()}


@pytest.fixture(scope="module")
def real_cfg():
    return inkfake.real_config()


def test_the_shape_rule_reproduces_the_real_index(real_cfg):
    """specs() (which builds the fake folders) must give exactly the real tensors for the real config, apart from the vision tower"""
    got = {n: (d, s) for n, d, s in inkfake.specs(real_cfg)}
    real = {n: v for n, v in REAL_TENSORS.items() if not n.startswith("model.visual.")}
    assert sorted(got) == sorted(real)
    assert {n: v for n, v in got.items() if v != real[n]} == {}


def test_real_numbers(real_cfg):
    groups = ml.group_params_moe(real_cfg, REAL_TENSORS, 24)
    assert groups["total"] == 26_388_641_280                  # 26.39 B with the vision tower; the scales, amax and denominators of the packed experts (5,160 values besides the scales) are not counted
    assert groups["expert_total"] == 24_461_180_928
    assert groups["vision"] == 318_256_128 and groups["mtp"] == 0
    assert groups["packed"] == ["nvfp4"] * 20 + ["mxfp8"] * 4
    assert groups["lm_head"] == 201024 * 2304 and groups["embed"] == 201024 * 2304
    assert sum(groups["layers"]) + groups["embed"] + groups["lm_head"] + groups["vision"] + groups["other"] == groups["total"]
    assert all(e == groups["experts"][0] for e in groups["experts"])           # same experts in every layer: 128 x 3 x 2304 x 1152 = 1.019 B
    assert groups["experts"][0] == 128 * 3 * 2304 * 1152
    assert groups["loop"] > 0


def test_real_describe(real_cfg):
    n = 24
    groups = ml.group_params_moe(real_cfg, REAL_TENSORS, n)
    d = ml.describe(real_cfg, groups, REAL_TENSORS, 19_525_870_752)
    assert (d["arch"], d["experts"], d["top_k"], d["expert_inter"], d["shared"]) == ("inkling2_loop", 128, 8, 1152, 0)
    assert (d["blocks"], d["block_size"], d["core"], d["passes"], d["varloop"], d["visits"]) == (4, 6, [6, 18], 3, False, 48)
    assert d["global_layers"] == [5, 11, 17, 23]
    assert d["packed"] == {"nvfp4": [0, 19], "mxfp8": [20, 23]}
    assert d["router_scoring"] == "sigmoid" and d["router_bias"] is True
    assert d["bf16_bytes"] == 52_777_282_560 and d["packed_bytes"] == 19_525_870_752
    assert d["mtp"] == {"file": None, "targets": [5, 23, 41, 48]}
    assert d["cuts"]["passes"] == "untrained" and d["cuts"]["blocks"] == "expert-only"
    # a token runs through 48 layer slots; the experts used per token are 8 of 128 in each
    assert d["active_params"] == pytest.approx(5.4e9, rel=0.12)
    rows = {r["experts"]: r for r in d["sizes"]}
    assert rows[128]["params"] == 26_388_641_280
    assert rows[96]["params"] == pytest.approx(20.27e9, rel=0.005)             # the plan's size table
    assert rows[64]["params"] == pytest.approx(14.16e9, rel=0.005)
    assert rows[96]["packed_bytes"] == pytest.approx(15.6e9, rel=0.03)
    assert sorted(rows, reverse=True) == [128, 112, 96, 80, 64]


def test_the_offered_counts_follow_the_owners_rules():
    assert ml.expert_counts_allowed(128, 8) == [112, 96, 80, 64]            # multiples of 16, at most 50 % in one stage
    assert ml.expert_counts_allowed(128, 8)[-1] >= 32
    assert ml.expert_counts_allowed(64, 8) == [48, 32]                      # never below 32 (= 4 x top-k)
    assert ml.expert_counts_allowed(32, 8) == []
    assert ml.expert_counts_allowed(128, 16) == [112, 96, 80, 64]           # 4 x 16 = 64 is the floor
    ml.check_experts_kept(128, 8, 96)
    for bad, why in ((96 + 8, "multiple of 16"), (16, "at least 32"), (48, None)):
        if bad == 48:
            ml.check_experts_kept(128, 8, 64)
            with pytest.raises(ml.MoeError, match="at most 50 %"):
                ml.check_experts_kept(128, 8, 48)
        else:
            with pytest.raises(ml.MoeError, match=why):
                ml.check_experts_kept(128, 8, bad)
    with pytest.raises(ml.MoeError, match="smaller"):
        ml.check_experts_kept(128, 8, 128)


def test_classify_roles():
    c = ml.classify
    assert c("model.llm.layers.7.mlp.experts.w13_weight") == (7, "experts", None)
    assert c("model.llm.layers.7.mlp.experts.w13_weight_scale") == (7, "expert_companion", None)
    assert c("model.llm.layers.7.mlp.experts.w2_weight_amax") == (7, "expert_companion", None)
    assert c("model.llm.layers.7.mlp.experts.w2_weight_outer_scale_denominator") == (7, "expert_shared", None)
    assert c("model.llm.layers.7.mlp.gate.weight") == (7, "router", None)
    assert c("model.llm.layers.7.mlp.gate.bias") == (7, "router_bias", None)
    assert c("model.llm.layers.7.controllers.k.down_weight") == (7, "loop_ctrl", None)
    assert c("model.llm.layers.7.attn.wq_du.weight") == (7, "attn", None)
    assert c("model.llm.layers.7.attn.k_sconv.weight") == (7, "conv", None)
    assert c("model.llm.layers.5.k_input_norm.weight") == (5, "norm", None)
    assert c("model.llm.middle_cycle_cross_pass_queries") == (None, "loop_query", None)
    assert c("model.llm.attn_res.attn_queries.4") == (None, "attn_res", None)
    assert c("model.llm.embed.weight") == (None, "embed", None) and c("model.llm.embed_norm.weight") == (None, "norm", None)
    assert c("model.llm.unembed.weight") == (None, "output", None)
    assert c("model.visual.backbone.layers.3.ln_1.bias") == (None, "vision", None)      # a vision layer 3 is not text layer 3
    assert c("fuse_weight") == (None, "mtp", None) and c("layers.0.dflare_norm.weight") == (None, "mtp", None)
    assert ml.classify("model.layers.3.mlp.experts.12.down_proj.weight", "qwen3_moe") == (3, "experts", 12)
    assert ml.classify("model.layers.3.mlp.gate.weight", "qwen3_moe") == (3, "router", None)
    assert ml.classify("model.layers.3.block_sparse_moe.experts.5.w1.weight", "mixtral") == (3, "experts", 5)


def test_layer_kinds_and_loop(real_cfg):
    kinds = ml.layer_kinds(real_cfg)
    assert [i for i, k in enumerate(kinds) if k == "full_attention"] == [5, 11, 17, 23]
    assert reshape.pattern_period(kinds) == 6
    assert ml.loop_of(real_cfg) == {"core": (6, 18), "passes": 3, "varloop": False, "gate_rank": 64}
    assert ml.visits(real_cfg) == 48 and ml.visits(real_cfg, 2) == 36 and ml.visits(real_cfg, 4) == 60
    ml.check_inkling_config(real_cfg)


def test_refusals():
    assert "310 GB" in (ml.refused_reason({"model_type": "inkling2"}) or "")
    with pytest.raises(ReshapeError, match="310 GB"):
        reshape.check_supported({"model_type": "inkling2", "text_config": {"num_hidden_layers": 4}})
    reshape.check_supported(inkfake.real_config())                       # the packed quantization_config does not stop this family
    cfg = inkfake.real_config()
    cfg["auto_map"] = {"AutoModel": "x"}
    with pytest.raises(ReshapeError, match="trust_remote_code"):
        reshape.check_supported(cfg)
    # every other quantized model is still refused
    with pytest.raises(ReshapeError, match="compressed"):
        reshape.check_supported({"model_type": "llama", "quantization_config": {"quant_method": "gptq"}, "num_hidden_layers": 4})


def test_inspect_of_the_fake_folder(tmp_path):
    made = inkfake.write_folder(tmp_path / "ink")
    out = reshapejob.inspect_model(str(tmp_path / "ink"))
    assert out["supported"] is True and out["layers"] == 24 and out["period"] == 6 and out["blocks"] == 4
    assert out["layer_types"].count("full_attention") == 4
    moe = out["moe"]
    assert (moe["experts"], moe["top_k"], moe["passes"], moe["visits"], moe["core"]) == (8, 2, 3, 48, [6, 18])
    assert moe["packed"] == {"nvfp4": [0, 19], "mxfp8": [20, 23]}
    assert moe["mtp"]["file"] is None                      # the index does not list the drafter, as in the real repo
    assert out["width_supported"] is False and "mixture of experts" in out["width_why"]
    assert out["has_vision"] is True and out["vision_params"] > 0 and out["tied"] is False
    assert sum(out["layer_params"]) + out["embed_params"] + out["lm_head_params"] + out["vision_params"] + out["other_params"] == out["total_params"]
    assert out["source_bytes"] == sum(len(v[2]) for v in made["tensors"].values())


def test_inspect_without_an_index_sees_the_drafter(tmp_path):
    inkfake.write_folder(tmp_path / "ink")
    (tmp_path / "ink" / "model.safetensors.index.json").unlink()
    out = reshapejob.inspect_model(str(tmp_path / "ink"))
    assert out["mtp_params"] > 0 and out["moe"]["mtp"]["file"] == "mtp.safetensors"


def test_inspect_dense_models_are_unchanged():
    out = reshapejob._empty("x", "why")
    assert "moe" not in out


def test_shared_cases_with_the_rust_side():
    cases = json.loads((inkfake.DATA / "moe_cases.json").read_text(encoding="utf-8"))
    for c in cases["counts"]:
        assert ml.expert_counts_allowed(c["experts"], c["top_k"]) == c["allowed"], c
    for c in cases["checks"]:
        try:
            ml.check_experts_kept(c["experts"], c["top_k"], c["keep"])
            ok = True
        except ml.MoeError:
            ok = False
        assert ok == c["ok"], c
