"""Pure helpers of "Slim or grow a model" (no torch needed)."""
import copy
import json
from pathlib import Path

import pytest

from ndforge_worker import reshape as rs
from ndforge_worker.reshape import ReshapeError

DATA = Path(__file__).parent / "data"


def real_cfg():
    cfg = json.loads((DATA / "qwen3_5_config.json").read_text(encoding="utf-8"))
    cfg["model_type"] = "qwen3_5"
    cfg["vision_config"] = {"depth": 2, "hidden_size": 8, "out_hidden_size": 2560}
    return cfg


def types(n):
    return ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (n // 4)


# ---------------------------------------------------------------------------------------------------- 1, 2: pattern

def test_pattern_period():
    assert rs.pattern_period(real_cfg()["text_config"]["layer_types"]) == 4
    assert rs.pattern_period(["full_attention"] * 12) == 1
    assert rs.pattern_period(["a", "b", "a"]) == 3          # aperiodic -> L
    assert rs.pattern_period(["a", "b", "a", "b"]) == 2


def test_layer_types_without_the_list_follow_the_interval_rule():
    tc = copy.deepcopy(real_cfg()["text_config"])
    real = tc.pop("layer_types")
    assert rs.layer_types_of(tc) == real
    plain = {"num_hidden_layers": 5}
    assert rs.layer_types_of(plain) == ["full_attention"] * 5
    with pytest.raises(ReshapeError, match="layer types"):
        rs.layer_types_of({"num_hidden_layers": 5, "layer_types": ["full_attention"]})


# ---------------------------------------------------------------------------------------------------- 3: keys

def test_split_key():
    assert rs.split_key("model.language_model.layers.12.mlp.up_proj.weight") == ("model.language_model.layers.", 12, "mlp.up_proj.weight")
    assert rs.split_key("model.layers.3.x") == ("model.layers.", 3, "x")
    assert rs.split_key("language_model.model.layers.0.self_attn.q_proj.weight") == ("language_model.model.layers.", 0, "self_attn.q_proj.weight")
    for other in ("mtp.layers.0.x", "model.visual.blocks.3.x", "lm_head.weight", "model.language_model.embed_tokens.weight", "model.norm.weight"):
        assert rs.split_key(other) is None


def test_unknown_layer_prefix_and_out_of_range_layer_raise():
    with pytest.raises(ReshapeError, match="Unknown layer naming"):
        rs.group_params({"foo.layers.1.x": ("BF16", (2,))}, 4)
    with pytest.raises(ReshapeError, match="layer 4"):
        rs.group_params({"model.layers.4.x": ("BF16", (2,))}, 4)


# ---------------------------------------------------------------------------------------------------- 4: params

def tiny_tensors(n_layers=8):
    t = {"model.language_model.embed_tokens.weight": ("BF16", (10, 4)), "model.language_model.norm.weight": ("BF16", (4,)),
         "lm_head.weight": ("BF16", (10, 4)), "model.visual.blocks.0.attn.qkv.weight": ("BF16", (3, 4)),
         "mtp.fc.weight": ("BF16", (4, 8)), "mtp.layers.0.mlp.down_proj.weight": ("BF16", (4, 4))}
    for i in range(n_layers):
        t[f"model.language_model.layers.{i}.mlp.up_proj.weight"] = ("BF16", (4, i + 1))     # layer i has 4*(i+1) parameters
    return t


def test_group_params_and_params_after():
    g = rs.group_params(tiny_tensors(), 8)
    assert g["layers"] == [4 * (i + 1) for i in range(8)]
    assert (g["embed"], g["lm_head"], g["vision"], g["mtp"], g["other"]) == (40, 40, 12, 48, 4)
    assert g["total"] == sum(g["layers"]) + 40 + 40 + 12 + 48 + 4 and g["bytes_per_param"] == 2.0
    assert rs.params_after(g, list(range(8))) == g["total"]
    assert rs.params_after(g, [0, 1, 2, 3]) == g["total"] - (4 * 5 + 4 * 6 + 4 * 7 + 4 * 8)
    assert rs.params_after(g, [0, 0, 7]) == g["total"] - sum(g["layers"]) + 4 + 4 + 32
    assert rs.group_params({"w": ("F32", (3,))}, 1)["bytes_per_param"] == 4.0


def test_block_count_and_limits():
    assert rs.block_count(32, 4) == 8
    with pytest.raises(ReshapeError):
        rs.block_count(30, 4)
    assert rs.limits(16, "prune") == (1, 8) and rs.limits(16, "grow") == (1, 8)
    assert rs.limits(2, "prune") == (1, 0)
    with pytest.raises(ReshapeError):
        rs.limits(4, "nope")


def test_limits_table_shared_with_the_rust_tests():
    cases = json.loads((DATA / "reshape_limits.json").read_text())
    assert len(cases) >= 8
    for c in cases:
        assert rs.limits(c["blocks"], c["mode"], c["pf"], c["pl"]) == (c["min"], c["max"]), c


def test_blocks_for_target():
    g = rs.group_params(tiny_tensors(16), 16)          # 4 blocks of 4 layers
    nb = 4
    mean_block = sum(g["layers"]) / nb
    assert rs.blocks_for_target(g, 4, g["total"] - 1 * mean_block, "prune") == 1
    assert rs.blocks_for_target(g, 4, g["total"] - 2 * mean_block, "prune") == 2
    assert rs.blocks_for_target(g, 4, 1, "prune") == 2                       # clamped to the limit
    grown2 = rs.params_after(g, rs.grow_src(16, 4, 2)[0])
    assert rs.blocks_for_target(g, 4, grown2, "grow") == 2
    assert rs.blocks_for_target(g, 4, 10 ** 9, "grow") == 2
    with pytest.raises(ReshapeError):
        rs.blocks_for_target(rs.group_params(tiny_tensors(4), 4), 4, 1, "prune")


# ---------------------------------------------------------------------------------------------------- 5, 6: choosing

def matrix(pairs, size):
    """symmetric D from {(a, b): value}; unlisted entries are 9"""
    D = [[0.0 if i == j else 9.0 for j in range(size)] for i in range(size)]
    for (a, b), v in pairs.items():
        D[a][b] = D[b][a] = v
    return D


def test_choose_drop_contiguous():
    D = matrix({(2, 4): 0.1, (3, 5): 0.2, (1, 3): 0.3, (0, 2): 0.01, (4, 6): 0.01}, 7)      # 6 blocks, boundaries 0..6
    assert rs.choose_drop(D, 2) == [2, 3]                      # (0,2) and (4,6) are protected (first / last block)
    assert rs.choose_drop(D, 2, protect_first=0, protect_last=0) == [0, 1]       # ties / lowest wins: (0,2)=0.01 vs (4,6)=0.01
    assert rs.choose_drop(D, 2, protect_first=0, protect_last=1) == [0, 1]
    D1 = matrix({(1, 2): 0.5, (2, 3): 0.5, (3, 4): 0.5}, 7)
    assert rs.choose_drop(matrix({(1, 2): 0.5, (2, 3): 0.5}, 7), 1) == [1]      # tie -> lowest index
    assert rs.choose_drop(D1, 1) == [1]
    with pytest.raises(ReshapeError):
        rs.choose_drop(D, 5)                                   # 6 blocks, 2 protected: at most 4 fit
    with pytest.raises(ReshapeError):
        rs.choose_drop(D, 0)
    with pytest.raises(ReshapeError, match="Unknown strategy"):
        rs.choose_drop(D, 1, "nope")


def test_choose_drop_separate_picks_the_least_influential_blocks():
    D = matrix({(0, 1): 0.0, (1, 2): 0.4, (2, 3): 0.1, (3, 4): 0.3, (4, 5): 0.1, (5, 6): 0.0}, 7)
    assert rs.choose_drop(D, 2, "separate") == [2, 4]          # blocks 0 and 5 are protected; 0.1 ties are both taken
    assert rs.choose_drop(D, 3, "separate") == [2, 3, 4]
    assert rs.choose_drop(D, 1, "separate") == [2]             # tie 2 vs 4 -> lowest
    with pytest.raises(ReshapeError):
        rs.choose_drop(D, 5, "separate")


def test_finalize_distance():
    sums = [[0, 6, 10], [0, 0, 4], [0, 0, 0]]
    counts = [[0, 2, 5], [0, 0, 0], [0, 0, 0]]
    D = rs.finalize_distance(sums, counts)
    assert D[0][1] == D[1][0] == 3.0 and D[0][2] == D[2][0] == 2.0 and D[1][2] == 0.0 and D[1][1] == 0.0


def test_halves_agree():
    a = matrix({(2, 4): 0.1}, 7)
    b = matrix({(2, 4): 0.2}, 7)
    c = matrix({(3, 5): 0.1}, 7)
    assert rs.halves_agree(a, b, 2, "contiguous", 1, 1)
    assert not rs.halves_agree(a, c, 2, "contiguous", 1, 1)
    assert not rs.halves_agree(a, b, 99, "contiguous", 1, 1)           # cannot even choose


def test_blocks_and_explicit_layers():
    assert rs.blocks_to_layers([2, 0], 4) == [0, 1, 2, 3, 8, 9, 10, 11]
    assert rs.check_whole_blocks([4, 5, 6, 7], 4, 32) == [1]
    with pytest.raises(ReshapeError, match="do not form whole blocks of 4"):
        rs.check_whole_blocks([3, 4, 5, 6], 4, 32)
    with pytest.raises(ReshapeError, match="protected"):
        rs.check_whole_blocks([0, 1, 2, 3], 4, 32)
    with pytest.raises(ReshapeError, match="protected"):
        rs.check_whole_blocks([28, 29, 30, 31], 4, 32)
    with pytest.raises(ReshapeError, match="at most half"):
        rs.check_whole_blocks(list(range(4, 24)), 4, 32)
    with pytest.raises(ReshapeError):
        rs.check_whole_blocks([], 4, 32)
    with pytest.raises(ReshapeError, match="between"):
        rs.check_whole_blocks([40], 4, 32)


# ---------------------------------------------------------------------------------------------------- 7: src lists

def test_grow_src_expected_lists():
    src, zero = rs.grow_src(32, 4, 4)
    assert src == list(range(24)) + list(range(8, 32)) and len(src) == 48
    assert [i for i, z in enumerate(zero) if z] == list(range(24, 40))             # the second copies of layers 8..23
    assert [s for s, z in zip(src, zero) if not z] == list(range(32))              # first occurrences = the original model
    assert rs.grow_src(64, 4, 2)[0] == list(range(36)) + list(range(28, 64)) and len(rs.grow_src(64, 4, 2)[0]) == 72
    assert rs.grow_src(24, 4, 1)[0] == list(range(12)) + list(range(8, 24)) and len(rs.grow_src(24, 4, 1)[0]) == 28
    for L, p, n in [(32, 4, 4), (64, 4, 2), (24, 4, 1), (12, 1, 6), (8, 4, 1), (16, 2, 3)]:
        s, z = rs.grow_src(L, p, n)
        assert len(set(s)) == L and len(s) == L + n * p and len(z) == len(s)


def test_grow_src_refuses():
    with pytest.raises(ReshapeError, match="at most half"):
        rs.grow_src(32, 4, 5)                  # 20 > 16
    with pytest.raises(ReshapeError):
        rs.grow_src(32, 4, 0)
    with pytest.raises(ReshapeError, match="Unknown layout"):
        rs.grow_src(32, 4, 1, layout="x")


def test_prune_src():
    assert rs.prune_src(8, [4, 5]) == [0, 1, 2, 3, 6, 7]
    for bad in ([], [8], [-1], list(range(8))):
        with pytest.raises(ReshapeError):
            rs.prune_src(8, bad)


# ---------------------------------------------------------------------------------------------------- 8: pattern check

def test_check_pattern():
    t = real_cfg()["text_config"]["layer_types"]
    with pytest.raises(ReshapeError, match="break the attention pattern"):
        rs.check_pattern(t, rs.prune_src(32, [0, 1]), 4)
    rs.check_pattern(t, rs.prune_src(32, [4, 5, 6, 7]), 4)
    rs.check_pattern(t, rs.grow_src(32, 4, 2)[0], 4)
    with pytest.raises(ReshapeError):
        rs.check_pattern(["a", "b", "a"], [0, 1], 3)            # aperiodic stack: nothing can be cut
    rs.check_pattern(["full_attention"] * 6, [0, 1, 2], 1)


# ---------------------------------------------------------------------------------------------------- 9: config

def test_fix_config():
    cfg = real_cfg()
    cfg["text_config"]["mlp_only_layers"] = [4, 30]
    cfg["text_config"]["per_layer_scale"] = list(range(32))
    cfg["text_config"]["max_window_layers"] = 32
    before = copy.deepcopy(cfg)
    src = rs.prune_src(32, [4, 5, 6, 7])
    notes = []
    new = rs.fix_config(cfg, src, notes)
    assert cfg == before                                                    # the input is not modified
    tc = new["text_config"]
    assert tc["num_hidden_layers"] == 28 and len(tc["layer_types"]) == 28 and tc["layer_types"] == types(28)
    assert tc["full_attention_interval"] == 4 and tc["mtp_num_hidden_layers"] == 1
    assert tc["mlp_only_layers"] == [26]                                    # old 4 was dropped, old 30 is new 26
    assert tc["per_layer_scale"] == src and len(notes) == 1 and "per_layer_scale" in notes[0]
    assert tc["max_window_layers"] == 28
    assert new["vision_config"] == before["vision_config"] and new["model_type"] == "qwen3_5"
    g = rs.fix_config(cfg, rs.grow_src(32, 4, 1)[0])
    assert g["text_config"]["num_hidden_layers"] == 36 and g["text_config"]["layer_types"] == types(36)


def test_fix_config_flat_llama_without_layer_types():
    cfg = {"model_type": "llama", "num_hidden_layers": 6, "hidden_size": 8}
    new = rs.fix_config(cfg, [0, 1, 2, 3])
    assert new == {"model_type": "llama", "num_hidden_layers": 4, "hidden_size": 8}      # no layer_types invented


def test_fix_config_window_boundary_inside_the_model_is_refused():
    cfg = {"model_type": "qwen2", "num_hidden_layers": 8, "max_window_layers": 4}
    with pytest.raises(ReshapeError, match="max_window_layers"):
        rs.fix_config(cfg, [0, 1, 2, 3])
    cfg["max_window_layers"] = 0
    assert rs.fix_config(cfg, [0, 1, 2, 3])["max_window_layers"] == 0


def test_check_supported():
    rs.check_supported(real_cfg())
    rs.check_supported({"model_type": "llama", "num_hidden_layers": 4})
    gemma = json.loads((DATA / "gemma4_config.json").read_text(encoding="utf-8"))
    with pytest.raises(ReshapeError) as e:
        rs.check_supported(gemma)
    assert str(e.value) == rs.REFUSED["gemma4"]
    gemma["model_type"] = "gemma4"
    with pytest.raises(ReshapeError) as e:
        rs.check_supported(gemma)
    assert str(e.value) == rs.REFUSED["gemma4"]
    q = real_cfg()
    q["quantization_config"] = {"quant_method": "bitsandbytes"}
    with pytest.raises(ReshapeError, match="quantized"):
        rs.check_supported(q)
    c = {"model_type": "llama", "num_hidden_layers": 4, "auto_map": {"AutoModel": "x.Y"}}
    with pytest.raises(ReshapeError, match="trust_remote_code"):
        rs.check_supported(c)
    with pytest.raises(ReshapeError, match="Only Qwen3.5"):
        rs.check_supported({"model_type": "gpt2", "num_hidden_layers": 4})
    with pytest.raises(ReshapeError, match="num_hidden_layers"):
        rs.check_supported({"model_type": "llama"})
