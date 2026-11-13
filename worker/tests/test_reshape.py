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


# ---------------------------------------------------------------------------------------------------- 11: tensor plan

def refs_for(n_layers=8):
    """TensorRef stand-ins for a Qwen3.5-like model: embed, per-layer tensors by layer type, vision, mtp, lm_head"""
    from ndforge_worker.safetensors_io import TensorRef
    out, pos = {}, 0

    def add(name, shape, dtype="BF16"):
        nonlocal pos
        n = 1
        for d in shape:
            n *= d
        out[name] = TensorRef("/x/model.safetensors", dtype, tuple(shape), pos, pos + n * 2)
        pos += n * 2

    add("model.language_model.embed_tokens.weight", (10, 4))
    for i in range(n_layers):
        p = f"model.language_model.layers.{i}."
        add(p + "input_layernorm.weight", (4,))
        if i % 4 == 3:
            add(p + "self_attn.o_proj.weight", (4, 4))
        else:
            add(p + "linear_attn.out_proj.weight", (4, 4))
        add(p + "mlp.up_proj.weight", (6, 4))
        add(p + "mlp.down_proj.weight", (4, 6))
    add("model.visual.blocks.0.attn.qkv.weight", (3, 4))
    add("mtp.fc.weight", (4, 8))
    add("mtp.layers.0.mlp.down_proj.weight", (4, 4))
    add("lm_head.weight", (10, 4))
    return out


def test_plan_tensors_renames_layers_and_keeps_the_rest_first():
    refs = refs_for(8)
    src = rs.prune_src(8, [4, 5, 6, 7])
    plan = rs.plan_tensors(refs, src, None)
    names = [t.name for t in plan]
    non_layer = [n for n in refs if rs.split_key(n) is None]
    assert names[:len(non_layer)] == non_layer and names[0] == "model.language_model.embed_tokens.weight"
    assert all(t.source is refs[t.name] for t in plan[:len(non_layer)])                 # vision, mtp, embed, lm_head untouched
    assert names[len(non_layer):] == [n for n in refs if rs.split_key(n)][:16]           # layers 0..3 keep their names (16 tensors)
    # grow: new layer 4 is old layer 0, new layer 8 is old layer 4
    plan = rs.plan_tensors(refs, [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7], None)
    by = {t.name: t for t in plan}
    assert by["model.language_model.layers.4.mlp.up_proj.weight"].source is refs["model.language_model.layers.0.mlp.up_proj.weight"]
    assert by["model.language_model.layers.11.self_attn.o_proj.weight"].source is refs["model.language_model.layers.7.self_attn.o_proj.weight"]
    assert len(plan) == len(refs) + 16 and len({t.name for t in plan}) == len(plan)


def test_plan_tensors_zero_only_the_output_projections_of_flagged_layers():
    refs = refs_for(8)
    src, zero = rs.grow_src(8, 4, 1)           # 0..5 + 2..7 -> 12 layers, second copies of 2..5 zeroed
    plan = rs.plan_tensors(refs, src, zero)
    by = {t.name: t for t in plan}
    for j, z in enumerate(zero):
        p = f"model.language_model.layers.{j}."
        out_name = "self_attn.o_proj.weight" if src[j] % 4 == 3 else "linear_attn.out_proj.weight"
        assert (by[p + out_name].source is None) == z
        assert (by[p + "mlp.down_proj.weight"].source is None) == z
        assert by[p + "mlp.up_proj.weight"].source is not None and by[p + "input_layernorm.weight"].source is not None
    assert sum(1 for t in plan if t.source is None) == 2 * 4
    assert by["model.language_model.layers.6.linear_attn.out_proj.weight"].shape == (4, 4)       # zeros keep dtype and shape
    assert by["model.language_model.layers.6.linear_attn.out_proj.weight"].nbytes == 32


def test_plan_tensors_refuses_a_layer_that_cannot_be_zeroed_or_is_missing():
    refs = refs_for(4)
    del refs["model.language_model.layers.1.mlp.down_proj.weight"]
    with pytest.raises(ReshapeError, match="invisible layer"):
        rs.plan_tensors(refs, [0, 1, 1], [False, False, True])
    with pytest.raises(ReshapeError, match="no tensors for layer 9"):
        rs.plan_tensors(refs, [0, 9], None)
    with pytest.raises(ReshapeError, match="zero mask"):
        rs.plan_tensors(refs, [0, 1], [False])


def test_describe():
    g = rs.group_params(tiny_tensors(), 8)
    d = rs.describe(g, 8, [0, 1, 2, 3, 2, 3, 4, 5], ["a", "a", "b", "b"] * 2)
    assert d["layers_before"] == 8 and d["layers_after"] == 8 and d["dropped"] == [6, 7] and d["duplicated"] == [2, 3] and d["period"] == 4
    assert d["params_before"] == g["total"] and d["params_after"] == rs.params_after(g, [0, 1, 2, 3, 2, 3, 4, 5])


# ---------------------------------------------------------------------------------------------------- 12: inspect job

import subprocess  # noqa: E402
import sys  # noqa: E402

from fakemodel import fake_model  # noqa: E402

WORKER = Path(__file__).resolve().parent.parent


def run_job(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=120)
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


INSPECT_KEYS = {"model_type", "supported", "why", "layers", "layer_types", "period", "blocks", "layer_params", "embed_params", "lm_head_params", "vision_params",
                "mtp_params", "other_params", "total_params", "bytes_per_param", "has_vision", "tied", "source_bytes", "local"}


def test_inspect_job_on_a_fake_qwen_folder(tmp_path):
    d = fake_model(tmp_path, 8)
    code, ev = run_job(tmp_path, {"job": "reshape_inspect", "source": {"repo": str(d)}})
    assert code == 0 and ev[-1]["type"] == "result"
    r = ev[-1]["data"]
    assert set(r) == INSPECT_KEYS
    assert r["supported"] and r["why"] is None and r["local"] and r["model_type"] == "qwen3_5"
    assert (r["layers"], r["period"], r["blocks"]) == (8, 4, 2) and r["layer_types"] == types(8)
    per_layer = 8 + 64 + 128 + 128
    assert r["layer_params"] == [per_layer] * 8
    assert (r["embed_params"], r["lm_head_params"], r["vision_params"], r["mtp_params"], r["other_params"]) == (80, 80, 24, 128 + 64, 0)
    assert r["total_params"] == 8 * per_layer + 80 + 80 + 24 + 192 and r["bytes_per_param"] == 2.0
    assert r["has_vision"] and not r["tied"] and r["source_bytes"] == 2 * r["total_params"]


def test_inspect_job_llama_without_layer_types(tmp_path):
    d = fake_model(tmp_path, 6, "llama")
    code, ev = run_job(tmp_path, {"job": "reshape_inspect", "source": {"repo": str(d)}})
    r = ev[-1]["data"]
    assert code == 0 and r["supported"] and r["period"] == 1 and r["blocks"] == 6 and r["layer_types"] == ["full_attention"] * 6
    assert not r["has_vision"] and r["vision_params"] == 0 and r["mtp_params"] == 0


def test_inspect_says_why_a_model_is_not_supported(tmp_path):
    d = tmp_path / "gemma"
    d.mkdir()
    (d / "config.json").write_bytes((DATA / "gemma4_config.json").read_bytes())
    code, ev = run_job(tmp_path, {"job": "reshape_inspect", "source": {"repo": str(d)}})
    r = ev[-1]["data"]
    assert code == 0 and r["supported"] is False and r["why"] == rs.REFUSED["gemma4"] and set(r) == INSPECT_KEYS and r["blocks"] == 0
    d2 = fake_model(tmp_path, 8, name="q")
    cfg = json.loads((d2 / "config.json").read_text())
    cfg["text_config"]["layer_types"][0] = "full_attention"           # breaks the attention interval
    (d2 / "config.json").write_text(json.dumps(cfg))
    r = run_job(tmp_path, {"job": "reshape_inspect", "source": {"repo": str(d2)}})[1][-1]["data"]
    assert r["supported"] is False and "interval" in r["why"]


def test_inspect_errors_are_plain(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, ev = run_job(tmp_path, {"job": "reshape_inspect", "source": {"repo": str(empty)}})
    assert code == 1 and ev[-1]["type"] == "error" and ev[-1]["code"] == "reshape_refused" and "config.json" in ev[-1]["message"]


def test_inspect_of_a_hub_model_reads_headers_only(tmp_path, monkeypatch):
    from ndforge_worker import reshapejob
    d = fake_model(tmp_path, 8)
    from ndforge_worker import safetensors_io as st
    refs = st.list_tensors(d)
    monkeypatch.setattr(reshapejob, "_hub_config", lambda repo: json.loads((d / "config.json").read_text()))
    monkeypatch.setattr(reshapejob, "_hub_tensors", lambda repo: {k: (r.dtype, r.shape) for k, r in refs.items()})
    r = reshapejob.inspect_model("org/Fake-8L")
    local = reshapejob.inspect_model(str(d))
    assert r["local"] is False and local["local"] is True
    assert {k: v for k, v in r.items() if k != "local"} == {k: v for k, v in local.items() if k != "local"}


# ---------------------------------------------------------------------------------------------------- 12: the reshape job

from fakemodel import tensor_numbers  # noqa: E402
from stfile import fill, read_folder  # noqa: E402

RESULT_KEYS = {"model_dir", "mode", "layers_before", "layers_after", "src_layers", "dropped", "duplicated", "zero_init", "params_before", "params_after",
               "bytes_written", "loss_before", "loss_after", "ppl_before", "ppl_after", "stable", "boundary_distance", "chosen_blocks", "load", "seconds",
               "notes", "stopped"}


def job(tmp_path, src, mode, **kw):
    d = tmp_path / "models" / "m1"
    d.mkdir(parents=True, exist_ok=True)
    return {"job": "reshape", "source": {"repo": str(src), "family": "qwen3_5"}, "mode": mode, "blocks": 1, "strategy": "contiguous",
            "protect": {"first": 1, "last": 1}, "drop_layers": None, "grow": {"layout": "solar", "init": "identity"}, "load": "auto", "check": False,
            "shard_gb": 5, "out": str(d / "model.partial"), "final": str(d / "model"), **kw}


def test_grow_with_identity_start(tmp_path):
    src = fake_model(tmp_path, 8)
    j = job(tmp_path, src, "grow")
    code, ev = run_job(tmp_path, j)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    assert set(r) == RESULT_KEYS and r["stopped"] is False
    final = Path(j["final"])
    assert final.is_dir() and not Path(j["out"]).exists()
    cfg = json.loads((final / "config.json").read_text())
    assert cfg["text_config"]["num_hidden_layers"] == 12 and cfg["text_config"]["layer_types"] == types(12) and cfg["vision_config"] == json.loads((src / "config.json").read_text())["vision_config"]
    assert r["layers_before"] == 8 and r["layers_after"] == 12 and r["src_layers"] == list(range(4)) + list(range(8)) and r["zero_init"] == [4, 5, 6, 7]
    assert r["dropped"] == [] and r["duplicated"] == [0, 1, 2, 3] and r["params_after"] > r["params_before"] and r["notes"]
    # side files, the licence, the description
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "preprocessor_config.json"):
        assert (final / name).read_bytes() == (src / name).read_bytes()
    assert (final / "README.md").read_text().startswith("---\nlicense: apache-2.0\n---\n")
    meta = json.loads((final / "ndforge_reshape.json").read_text())
    assert meta["format"] == 1 and meta["origin"]["layers"] == 8 and meta["layers"] == 12 and meta["period"] == 4 and meta["zero_init"] == [4, 5, 6, 7]
    assert meta["src_layers"] == r["src_layers"] and meta["params"] == r["params_after"] and meta["mode"] == "grow" and meta["vision"].startswith("unchanged")
    # the weights: the folder reads back, bytes are those of the source, zeroed outputs are zero
    from ndforge_worker import safetensors_io
    refs = safetensors_io.list_tensors(final)
    assert sum(r_.numel for r_ in refs.values()) == r["params_after"]
    nums = tensor_numbers(8, "qwen3_5")
    data = read_folder(final)
    pre = "model.language_model.layers."
    assert data[pre + "5.mlp.up_proj.weight"][2] == fill(nums[pre + "1.mlp.up_proj.weight"], 16 * 8 * 2)
    assert data[pre + "5.mlp.down_proj.weight"][2] == bytes(8 * 16 * 2) and data[pre + "5.linear_attn.out_proj.weight"][2] == bytes(8 * 8 * 2)
    assert data[pre + "7.self_attn.o_proj.weight"][2] == bytes(8 * 8 * 2)
    assert data[pre + "11.self_attn.o_proj.weight"][2] == fill(nums[pre + "7.self_attn.o_proj.weight"], 8 * 8 * 2)
    assert data["lm_head.weight"][2] == fill(nums["lm_head.weight"], 10 * 8 * 2) and data["mtp.fc.weight"][2] == fill(nums["mtp.fc.weight"], 8 * 16 * 2)
    # progress was reported
    steps = {e["step"] for e in ev if e["type"] == "progress"}
    assert {"inspect", "write", "done"} <= steps


def test_grow_by_copying_keeps_every_weight(tmp_path):
    src = fake_model(tmp_path, 8)
    code, ev = run_job(tmp_path, job(tmp_path, src, "grow", grow={"layout": "solar", "init": "copy"}))
    r = ev[-1]["data"]
    assert code == 0 and r["zero_init"] == [] and r["notes"] == []
    data = read_folder(Path(job(tmp_path, src, "grow")["final"]))
    assert data["model.language_model.layers.5.mlp.down_proj.weight"][2] == data["model.language_model.layers.1.mlp.down_proj.weight"][2] != bytes(8 * 16 * 2)


def test_prune_with_an_explicit_layer_list(tmp_path):
    src = fake_model(tmp_path, 12)
    j = job(tmp_path, src, "prune", drop_layers=[4, 5, 6, 7])
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    r = ev[-1]["data"]
    assert set(r) == RESULT_KEYS and r["layers_after"] == 8 and r["dropped"] == [4, 5, 6, 7] and r["chosen_blocks"] == [1] and r["src_layers"] == [0, 1, 2, 3, 8, 9, 10, 11]
    final = Path(j["final"])
    cfg = json.loads((final / "config.json").read_text())
    assert cfg["text_config"]["num_hidden_layers"] == 8 and cfg["text_config"]["layer_types"] == types(8)
    data = read_folder(final)
    nums = tensor_numbers(12, "qwen3_5")
    pre = "model.language_model.layers."
    assert not any(k.startswith(pre + "8.") for k in data)
    assert data[pre + "4.mlp.up_proj.weight"][2] == fill(nums[pre + "8.mlp.up_proj.weight"], 16 * 8 * 2)
    meta = json.loads((final / "ndforge_reshape.json").read_text())
    assert meta["dropped"] == [4, 5, 6, 7] and meta["strategy"] == "explicit" and meta["layers"] == 8


def test_prune_refuses_layers_that_are_not_whole_blocks(tmp_path):
    src = fake_model(tmp_path, 12)
    code, ev = run_job(tmp_path, job(tmp_path, src, "prune", drop_layers=[3, 4, 5, 6]))
    assert code == 1 and ev[-1]["code"] == "reshape_refused" and "do not form whole blocks of 4" in ev[-1]["message"]
    assert not (tmp_path / "models" / "m1" / "model").exists()


def test_prune_needing_scores_without_torch_says_so(tmp_path):
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed: the scoring path is tested in test_reshape_torch.py")
    except ImportError:
        pass
    src = fake_model(tmp_path, 12)
    code, ev = run_job(tmp_path, job(tmp_path, src, "prune"))
    assert code == 1 and ev[-1]["code"] == "reshape_refused" and "Training setup" in ev[-1]["message"]


def test_out_of_range_blocks_are_refused(tmp_path):
    src = fake_model(tmp_path, 8)
    code, ev = run_job(tmp_path, job(tmp_path, src, "grow", blocks=3))
    assert code == 1 and ev[-1]["code"] == "reshape_refused" and "at most half" in ev[-1]["message"]


def test_not_enough_disk_space_is_reported_before_anything_is_written(tmp_path, monkeypatch):
    import io
    import shutil
    from ndforge_worker import protocol, reshapejob
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: (10, 9, 1))
    src = fake_model(tmp_path, 8)
    j = job(tmp_path, src, "grow")
    assert reshapejob.run(j) == 1
    ev = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert ev[-1]["type"] == "error" and ev[-1]["code"] == "disk_full" and "not enough free space" in ev[-1]["message"]
    assert not Path(j["out"]).exists() and not Path(j["final"]).exists()


def test_a_stop_file_cancels_and_cleans_up(tmp_path):
    src = fake_model(tmp_path, 8)
    j = job(tmp_path, src, "grow")
    (Path(j["out"]).parent / "STOP").write_text("")
    code, ev = run_job(tmp_path, j)
    assert code == 0 and ev[-1]["data"] == {"stopped": True, "model_dir": None}
    assert not Path(j["out"]).exists() and not Path(j["final"]).exists()


def test_a_llama_model_without_layer_types_can_be_cut_and_grown(tmp_path):
    src = fake_model(tmp_path, 6, "llama")
    code, ev = run_job(tmp_path, job(tmp_path, src, "prune", drop_layers=[2, 3]))
    assert code == 0, ev[-1]
    assert ev[-1]["data"]["layers_after"] == 4
    cfg = json.loads((Path(job(tmp_path, src, "prune")["final"]) / "config.json").read_text())
    assert cfg["num_hidden_layers"] == 4 and "layer_types" not in cfg


def test_an_existing_target_is_not_overwritten(tmp_path):
    src = fake_model(tmp_path, 8)
    j = job(tmp_path, src, "grow")
    Path(j["final"]).mkdir(parents=True)
    code, ev = run_job(tmp_path, j)
    assert code == 1 and "already exists" in ev[-1]["message"]
