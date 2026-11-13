"""Width pruning ("Slim or grow a model" version 2): the pure helpers (no numpy, no torch needed)."""
import copy
import json
from pathlib import Path

import pytest

from ndforge_worker import reshape_width as rw
from ndforge_worker.reshape import ReshapeError

DATA = Path(__file__).parent / "data"
HUB_TOTALS = {"27b": 27_781_427_952, "9b": 9_653_104_368}


def cfg_of(name):
    return json.loads((DATA / f"qwen3_5_{name}_config.json").read_text(encoding="utf-8"))


def vision_params(v: dict) -> int:
    """the vision tower and merger of Qwen3.5, counted from vision_config alone (independent of the Hub totals)"""
    h, i, d, sm = v["hidden_size"], v["intermediate_size"], v["depth"], v["spatial_merge_size"]
    patch = v["in_channels"] * v["temporal_patch_size"] * v["patch_size"] ** 2 * h + h
    block = 2 * h + 2 * h + h * 3 * h + 3 * h + h * h + h + h * i + i + i * h + h
    mh = h * sm * sm
    return patch + v["num_position_embeddings"] * h + d * block + 2 * h + mh * mh + mh + mh * v["out_hidden_size"] + v["out_hidden_size"]


def same(dims):
    return {"intermediate": dims["intermediate"], "lin_v_heads": dims["lin_v_heads"], "heads": dims["heads"]}


# ----------------------------------------------------------------------------------------------------- 1: dims and refusals

def test_width_dims_of_the_real_27b_and_9b_configs():
    d = rw.width_dims(cfg_of("27b"))
    assert d == {"hidden": 5120, "intermediate": 17408, "heads": 24, "kv_heads": 4, "head_dim": 256, "lin_k_heads": 16, "lin_v_heads": 48, "lin_k_dim": 128,
                 "lin_v_dim": 128, "conv_kernel": 4, "mtp_layers": 1, "vocab": 248320, "tied": False}
    d9 = rw.width_dims(cfg_of("9b"))
    assert (d9["hidden"], d9["intermediate"], d9["heads"], d9["lin_v_heads"]) == (4096, 12288, 16, 32)
    assert rw.width_supported(cfg_of("27b")) is None and rw.width_supported(cfg_of("9b")) is None
    four_b = json.loads((DATA / "qwen3_5_config.json").read_text(encoding="utf-8"))
    assert rw.width_dims(four_b)["tied"] is True and rw.width_supported(four_b) is None


def test_head_dim_is_derived_when_missing_and_written_explicitly_by_fix_config():
    cfg = cfg_of("27b")
    del cfg["text_config"]["head_dim"]
    d = rw.width_dims(cfg)
    assert d["head_dim"] == 5120 // 24                      # transformers' fallback; it would change when the heads change
    width = {"intermediate": 17408, "lin_v_heads": 48, "heads": 16}
    new = rw.fix_config_width(cfg, d, width)
    assert new["text_config"]["head_dim"] == 5120 // 24 and new["text_config"]["num_attention_heads"] == 16
    assert "head_dim" not in cfg["text_config"]            # the input is not changed


def test_width_is_refused_for_other_models_with_a_plain_reason():
    moe = cfg_of("27b")
    moe["model_type"] = moe["text_config"]["model_type"] = "qwen3_5_moe"
    assert rw.width_supported(moe) == rw.NOT_DENSE
    moe2 = cfg_of("27b")
    moe2["text_config"]["num_experts"] = 8
    assert rw.width_supported(moe2) == rw.NOT_DENSE
    llama = {"model_type": "llama", "num_hidden_layers": 4, "hidden_size": 8}
    assert rw.width_supported(llama) == rw.NOT_DENSE
    gemma = json.loads((DATA / "gemma4_config.json").read_text(encoding="utf-8"))
    assert "Gemma 4" in rw.width_supported(gemma)
    odd = cfg_of("27b")
    odd["text_config"]["linear_value_head_dim"] = 64
    assert "key and value sizes" in rw.width_supported(odd)
    mixed = cfg_of("27b")
    mixed["text_config"]["mlp_only_layers"] = [0]
    assert "different MLP" in rw.width_supported(mixed)
    quant = cfg_of("27b")
    quant["quantization_config"] = {"bits": 4}
    assert "quantized" in rw.width_supported(quant)
    with pytest.raises(ReshapeError, match="dense Qwen3.5"):
        rw.width_dims(llama)
    with pytest.raises(ReshapeError, match="dense Qwen3.5"):
        rw.fix_config_width(moe, rw.width_dims(cfg_of("27b")), {"intermediate": 8704, "lin_v_heads": 48, "heads": 24})
    broken = cfg_of("27b")
    del broken["text_config"]["linear_num_key_heads"]
    assert "linear_num_key_heads" in rw.width_supported(broken)


# ----------------------------------------------------------------------------------------------------- 2: parameter counts

@pytest.mark.parametrize("name", ["27b", "9b"])
def test_params_for_reproduces_the_hub_totals(name):
    cfg = cfg_of(name)
    d = rw.width_dims(cfg)
    lt = cfg["text_config"]["layer_types"]
    vis = vision_params(cfg["vision_config"])
    assert rw.params_for(d, lt, same(d), vis) == HUB_TOTALS[name]
    text_and_mtp = rw.params_for(d, lt, same(d), 0)
    assert HUB_TOTALS[name] - text_and_mtp == vis           # the part that is not text: the vision tower, counted independently


def table():
    return json.loads((DATA / "width_params.json").read_text(encoding="utf-8"))


def test_the_parity_table_with_the_rust_side():
    cases = table()
    assert len(cases) >= 8 and any(c["params"] == HUB_TOTALS["27b"] for c in cases) and any(c["params"] == HUB_TOTALS["9b"] for c in cases)
    for c in cases:
        n = c["n_lin"] + c["n_full"]
        lt = ["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(n)]
        assert lt.count("full_attention") == c["n_full"] and len(lt) == c["n_lin"] + c["n_full"], c["name"]
        assert rw.params_for(c["dims"], lt, c["width"], c["vision"], c["other"]) == c["params"], c["name"]


def test_the_sizes_named_in_the_owner_decisions():
    by = {c["name"]: c["params"] / 1e9 for c in table()}
    assert round(by["27B mild (I 13312)"], 2) == 23.69 and round(by["27B I 10752"], 2) == 21.14
    assert round(by["27B 60 layers + I 12288"], 2) == 21.46 and round(by["27B 56 layers + I 13312"], 2) == 21.15


# ----------------------------------------------------------------------------------------------------- 3: choices

def test_width_choices():
    d = rw.width_dims(cfg_of("27b"))
    ch = rw.width_choices(d)
    assert ch["intermediate"][0] == 8704 and ch["intermediate"][-1] == 17408 and len(ch["intermediate"]) == 35
    assert all(v % 256 == 0 for v in ch["intermediate"]) and 13312 in ch["intermediate"] and 10752 in ch["intermediate"]
    assert ch["lin_v_heads"] == [48, 32] and ch["heads"] == [24, 20, 16]
    d9 = rw.width_dims(cfg_of("9b"))
    assert rw.width_choices(d9)["lin_v_heads"] == [32, 16] and rw.width_choices(d9)["heads"] == [16, 12]
    assert rw.ffn_choices(32, 4) == [16, 20, 24, 28, 32]


def test_check_width():
    d = rw.width_dims(cfg_of("27b"))
    assert rw.check_width(d, {"intermediate_size": 13312, "linear_num_value_heads": None, "num_attention_heads": None}) == \
        {"intermediate": 13312, "lin_v_heads": 48, "heads": 24}
    assert rw.check_width(d, {"intermediate": 12288, "lin_v_heads": 32, "heads": 16}) == {"intermediate": 12288, "lin_v_heads": 32, "heads": 16}
    assert rw.check_width(d, {"lin_v_heads": 32}) == {"intermediate": 17408, "lin_v_heads": 32, "heads": 24}      # Expert axes alone are allowed
    for bad, text in (({"intermediate": 13313}, "multiple of 256 between half and all of 17,408"), ({"intermediate": 8192}, "multiple of 256"),
                      ({"intermediate": 17664}, "multiple of 256"), ({"lin_v_heads": 40}, "from 48 to 32"), ({"heads": 18}, "multiple of 4 and at least 16"),
                      ({"heads": 12}, "at least 16"), ({"intermediate": True}, "whole number"), ({"colour": 1}, "Unknown width"),
                      ({}, "nothing to thin"), ({"intermediate": 17408}, "nothing to thin")):
        with pytest.raises(ReshapeError, match=text):
            rw.check_width(d, bad)


# ----------------------------------------------------------------------------------------------------- 4: picking

def test_keep_top_is_stable_and_sorted():
    assert rw.keep_top([0.1, 0.9, 0.5, 0.9, 0.2], 2) == [1, 3]
    assert rw.keep_top([1.0, 1.0, 1.0, 1.0], 2) == [0, 1]                      # ties: the lower index
    assert rw.keep_top([0.3, float("nan"), 0.2], 2) == [0, 2]                  # NaN is the smallest
    assert rw.keep_top([5, 4, 3], 3) == [0, 1, 2]
    for k in (0, 4):
        with pytest.raises(ReshapeError):
            rw.keep_top([1, 2, 3], k)


def test_keep_grouped_takes_the_same_number_from_every_group():
    scores = [9, 1, 5,   2, 8, 3]                                              # two groups of three
    assert rw.keep_grouped(scores, 2, 4) == [0, 2, 4, 5]
    assert rw.keep_grouped(scores, 2, 2) == [0, 4]
    assert rw.keep_grouped([1.0] * 6, 2, 4) == [0, 1, 3, 4]                    # ties: the lowest index in each group
    assert rw.keep_grouped(scores, 1, 3) == [0, 2, 4]
    for groups, k in ((2, 3), (4, 4), (2, 0)):
        with pytest.raises(ReshapeError):
            rw.keep_grouped(scores, groups, k)


def test_overlap():
    assert rw.overlap([1, 2, 3], [1, 2, 3]) == 1.0
    assert rw.overlap([1, 2], [3, 4]) == 0.0
    assert rw.overlap([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert rw.overlap([], []) == 1.0


# ----------------------------------------------------------------------------------------------------- 5: slices

TINY = {"hidden": 8, "intermediate": 32, "heads": 6, "kv_heads": 2, "head_dim": 4, "lin_k_heads": 2, "lin_v_heads": 4, "lin_k_dim": 4, "lin_v_dim": 4,
        "conv_kernel": 4, "mtp_layers": 1, "vocab": 16, "tied": False}


def test_slices_for_layer_match_the_hand_written_index_lists():
    full = {"intermediate": 3, "lin_v_heads": 2, "heads": 4}
    keep = {"mlp": [0, 5, 9], "heads": [0, 2, 3, 5], "gdn": [1, 2]}
    s = rw.slices_for_layer("full_attention", TINY, full, keep)
    assert s["mlp.gate_proj.weight"] == (0, [0, 5, 9]) and s["mlp.up_proj.weight"] == (0, [0, 5, 9]) and s["mlp.down_proj.weight"] == (1, [0, 5, 9])
    # head h owns rows [8h, 8h+8) of q_proj (query | gate) and columns [4h, 4h+4) of o_proj
    assert s["self_attn.q_proj.weight"] == (0, list(range(0, 8)) + list(range(16, 24)) + list(range(24, 32)) + list(range(40, 48)))
    assert s["self_attn.o_proj.weight"] == (1, list(range(0, 4)) + list(range(8, 12)) + list(range(12, 16)) + list(range(20, 24)))
    assert set(s) == {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "self_attn.q_proj.weight", "self_attn.o_proj.weight"}
    g = rw.slices_for_layer("linear_attention", TINY, full, keep)
    vrows = list(range(4, 8)) + list(range(8, 12))
    qkv = list(range(16)) + [16 + r for r in vrows]                             # q and k rows (2 * 2 * 4) stay, the v rows of value heads 1 and 2 follow
    assert g["linear_attn.in_proj_qkv.weight"] == (0, qkv) and g["linear_attn.conv1d.weight"] == (0, qkv)
    assert g["linear_attn.in_proj_z.weight"] == (0, vrows) and g["linear_attn.out_proj.weight"] == (1, vrows)
    for name in ("in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias"):
        assert g["linear_attn." + name] == (0, [1, 2])
    assert "linear_attn.norm.weight" not in g and "self_attn.q_norm.weight" not in s


def test_only_the_dimensions_that_change_get_slices():
    only_mlp = {"intermediate": 3, "lin_v_heads": 4, "heads": 6}
    assert set(rw.slices_for_layer("linear_attention", TINY, only_mlp, {"mlp": [0, 1, 2]})) == {"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
    only_heads = {"intermediate": 32, "lin_v_heads": 4, "heads": 4}
    assert set(rw.slices_for_layer("full_attention", TINY, only_heads, {"heads": [0, 1, 3, 4]})) == {"self_attn.q_proj.weight", "self_attn.o_proj.weight"}
    assert rw.slices_for_layer("linear_attention", TINY, only_heads, {"heads": [0, 1, 3, 4]}) == {}


def test_check_keep_catches_wrong_counts_and_unbalanced_groups():
    width = {"intermediate": 3, "lin_v_heads": 2, "heads": 4}
    good = {"mlp": {0: [0, 5, 9]}, "heads": {3: [0, 2, 3, 5]}, "gdn": {0: [1, 2]}}
    rw.check_keep(TINY, width, good)
    for bad in ({"mlp": {0: [0, 5]}}, {"mlp": {0: [5, 0, 9]}}, {"mlp": {0: [0, 5, 32]}}, {"heads": {3: [0, 1, 2, 3]}}, {"gdn": {0: [0, 1]}}):
        with pytest.raises(ReshapeError, match="Internal error"):
            rw.check_keep(TINY, width, bad)


def test_fix_config_width_changes_only_the_text_config_keys():
    cfg = cfg_of("27b")
    d = rw.width_dims(cfg)
    new = rw.fix_config_width(cfg, d, {"intermediate": 10752, "lin_v_heads": 32, "heads": 24})
    tc = new["text_config"]
    assert (tc["intermediate_size"], tc["linear_num_value_heads"], tc["num_attention_heads"], tc["head_dim"]) == (10752, 32, 24, 256)
    cfg2 = copy.deepcopy(cfg)
    for k in ("intermediate_size", "linear_num_value_heads"):
        cfg2["text_config"][k] = tc[k]
    assert new == cfg2                                                       # nothing else was touched (vision_config, layer_types, top level)
    assert cfg["text_config"]["intermediate_size"] == 17408
    broken = cfg_of("27b")
    del broken["text_config"]["intermediate_size"]
    with pytest.raises(ReshapeError):
        rw.fix_config_width(broken, d, {"intermediate": 10752, "lin_v_heads": 48, "heads": 24})
