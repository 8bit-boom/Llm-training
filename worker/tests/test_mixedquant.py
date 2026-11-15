"""Mixed-precision sizes (mixedquant.py): the table of recipes, the rules against real name lists (the tensor names of llama.cpp's qwen35 architecture, read in its source), the
size estimates, "Fit to a size", and the export wiring with stand-in llama.cpp tools. Small hand-built GGUF headers; no torch, no llama.cpp, no big files."""
import json
import re
import struct
from pathlib import Path

import pytest
from test_imatrix import QUANTIZE, flag, world  # noqa: F401  (world is a fixture)
from test_imatrix_unsloth import FakeModel, unsloth_world  # noqa: F401  (a fixture)

from ndforge_worker import __main__ as entry
from ndforge_worker import exportjob, ggufinfo, imatrix, mixedquant, protocol

F32, F16, Q4K = 0, 1, 12


# ------------------------------------------------------------------------------------------------------------------ the tensors of a qwen35 model, as llama.cpp's source names them

def qwen35_tensors(n_layer=8, d=512, ff=2048, vocab=1000, interval=4, fused_linear=False, tied=False, nextn=True, moe=False):
    """(name, dims, type): src/models/qwen35.cpp and src/llama-arch.cpp. A layer is full attention when `il % interval == interval - 1`, else linear attention (gated delta net).
    `fused_linear`: wqkv_z / ssm_beta_alpha (one tensor each) instead of attn_qkv + attn_gate and ssm_beta + ssm_alpha."""
    t = [("token_embd.weight", [d, vocab], F16), ("output_norm.weight", [d], F32)]
    if not tied:
        t.append(("output.weight", [d, vocab], F16))
    for il in range(n_layer):
        p = f"blk.{il}."
        t += [(p + "attn_norm.weight", [d], F32), (p + "post_attention_norm.weight", [d], F32)]
        if il % interval == interval - 1:
            t += [(p + "attn_q.weight", [d, d * 2], F16), (p + "attn_k.weight", [d, 128], F16), (p + "attn_v.weight", [d, 128], F16), (p + "attn_output.weight", [d, d], F16),
                  (p + "attn_q_norm.weight", [64], F32), (p + "attn_k_norm.weight", [64], F32)]
        else:
            t += [(p + "ssm_conv1d.weight", [4, 1024], F32), (p + "ssm_dt.bias", [16], F32), (p + "ssm_a", [16], F32), (p + "ssm_norm.weight", [64], F32), (p + "ssm_out.weight", [512, d], F16)]
            t += [(p + "attn_qkv_gate.weight", [d, 1536], F16)] if fused_linear else [(p + "attn_qkv.weight", [d, 1024], F16), (p + "attn_gate.weight", [d, 512], F16)]
            t += [(p + "ssm_beta_alpha.weight", [d, 32], F16)] if fused_linear else [(p + "ssm_beta.weight", [d, 16], F16), (p + "ssm_alpha.weight", [d, 16], F16)]
        if moe:
            t += [(p + "ffn_gate_inp.weight", [d, 8], F32), (p + "ffn_gate_exps.weight", [d, ff, 8], F16), (p + "ffn_up_exps.weight", [d, ff, 8], F16), (p + "ffn_down_exps.weight", [ff, d, 8], F16)]
        else:
            t += [(p + "ffn_gate.weight", [d, ff], F16), (p + "ffn_down.weight", [ff, d], F16), (p + "ffn_up.weight", [d, ff], F16)]
    if nextn:
        p = f"blk.{n_layer}."
        t += [(p + "nextn.eh_proj.weight", [2 * d, d], F16), (p + "nextn.enorm.weight", [d], F32), (p + "nextn.hnorm.weight", [d], F32), (p + "attn_q.weight", [d, d * 2], F16),
              (p + "ffn_down.weight", [ff, d], F16), (p + "ffn_up.weight", [d, ff], F16)]
    return t


def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_bytes(tensors, arch="qwen35", n_layer=8, align=32, extra_kv=()):
    """a GGUF (version 3) header with the tensor infos and zero data of the right total size"""
    kvs = [s("general.architecture") + struct.pack("<I", 8) + s(arch), s(f"{arch}.block_count") + struct.pack("<II", 4, n_layer), s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3) + b"".join(s(f"t{i}") for i in range(3))]
    kvs += list(extra_kv)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    off, infos, sizes = 0, b"", []
    for name, dims, typ in tensors:
        n = 1
        for d_ in dims:
            n *= d_
        size = int(n * ggufinfo.TYPES[typ][1] / 8)
        infos += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d_) for d_ in dims) + struct.pack("<IQ", typ, off)
        off += (size + align - 1) // align * align
        sizes.append(size)
    body = head + infos
    pad = (align - len(body) % align) % align
    return body + b"\0" * pad + b"\0" * off


def gguf_file(path, tensors, **kw):
    Path(path).write_bytes(gguf_bytes(tensors, **kw))
    return str(path)


def info_of(tensors, **kw):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        return ggufinfo.read(gguf_file(Path(d) / "m.gguf", tensors, **kw))


def first_rule(name, rules):
    for rx, t in rules:
        if re.search(rx, name):
            return t
    return None


# ------------------------------------------------------------------------------------------------------------------ the table of recipes

def test_there_is_one_table_with_three_recipes_labelled_as_our_own():
    assert [p["id"] for p in mixedquant.PRESETS] == ["mixed_4", "mixed_3", "mixed_2"] and mixedquant.PRESET_IDS == ("mixed_4", "mixed_3", "mixed_2")
    assert [p["label"] for p in mixedquant.PRESETS] == ["Mixed 4-bit (balanced)", "Mixed 3-bit (small)", "Mixed 2-bit (smallest: short answers only)"]
    assert mixedquant.LABEL == "Mixed precision (our own recipe)" and mixedquant.FIT_LABEL == "Mixed: fit to a size (experimental)" and mixedquant.MIXED_IDS == ("mixed_4", "mixed_3", "mixed_2", "mixed_fit")
    for p in mixedquant.PRESETS:
        assert p["base"] in mixedquant.PLAIN_BPW and p["embd"] in mixedquant.LEVELS + ("Q8_0",) and p["output"] in mixedquant.LEVELS and p["edge"] >= 1
        assert set(p["types"]) == set(mixedquant.ROLES)
        for role, (hi, mid) in p["types"].items():
            assert hi in mixedquant.LEVELS and mid in mixedquant.LEVELS and mixedquant.LEVELS.index(hi) >= mixedquant.LEVELS.index(mid), f"{p['id']} {role}: the edge layers never get fewer bits"


def test_nothing_in_the_recipes_names_or_implies_unsloths_dynamic_quants():
    import inspect
    text = json.dumps(mixedquant.PRESETS) + inspect.getsource(mixedquant) + mixedquant.LABEL + mixedquant.FIT_LABEL
    assert "UD-" not in text and not re.search(r"unsloth dynamic|Dynamic 2|Dynamic 3|UD_|\bUD\b", text, re.I)
    assert "not Unsloth's" in inspect.getsource(mixedquant), "the module says what it is not"


def test_the_recipes_get_smaller_in_every_role():
    def avg(p):
        return sum(mixedquant.bpw(t) for hi, mid in p["types"].values() for t in (hi, mid))
    sizes = [avg(p) for p in mixedquant.PRESETS]
    assert sizes[0] > sizes[1] > sizes[2]
    m4, m2 = mixedquant.PRESETS[0], mixedquant.PRESETS[2]
    assert m4["types"]["ffn_updown"][1] == "Q4_K" and m2["types"]["ffn_updown"][1] == "Q2_K", "the middle layers' ffn_up / gate get the fewest bits"
    for p in mixedquant.PRESETS:
        assert p["types"]["attn_kvo"][0] == "Q6_K" and p["output"] == "Q6_K", "k / v / output of the edge layers and the output tensor keep the most bits"


# ------------------------------------------------------------------------------------------------------------------ the rules against the real names

def test_rules_for_qwen35_in_exact_order_with_the_edge_layers_first():
    names = [n for n, _, _ in qwen35_tensors()]
    rules, notes = mixedquant.build_rules("mixed_4", "qwen35", 8, names)
    assert notes == []
    e = r"blk\.(?:0|1|2|5|6|7|8)"
    assert [t for _, t in rules] == ["Q6_K", "Q6_K", "Q6_K", "Q5_K", "Q6_K", "Q5_K", "Q5_K", "Q5_K", "Q4_K"]
    assert rules[0] == (e + r"\.(?:attn_q)\.weight$", "Q6_K")
    assert rules[1] == (e + r"\.(?:attn_qkv_gate|attn_qkv|attn_gate|ssm_out|ssm_beta_alpha|ssm_beta|ssm_alpha)\.weight$", "Q6_K")
    assert rules[2] == (e + r"\.(?:ffn_down|ffn_down_exps|ffn_down_shexp)\.weight$", "Q6_K")
    assert rules[3] == (e + r"\.(?:ffn_up|ffn_gate|ffn_up_exps|ffn_gate_exps|ffn_up_shexp|ffn_gate_shexp)\.weight$", "Q5_K")
    assert rules[4] == (r"\.(?:attn_k|attn_v|attn_output|nextn\.eh_proj)\.weight$", "Q6_K") and len(rules) == 9, "k / v / output are Q6_K in every layer: one rule"
    assert [r for r in rules[5:]] == [(r"\.(?:attn_q)\.weight$", "Q5_K"), (r"\.(?:attn_qkv_gate|attn_qkv|attn_gate|ssm_out|ssm_beta_alpha|ssm_beta|ssm_alpha)\.weight$", "Q5_K"),
                                      (r"\.(?:ffn_down|ffn_down_exps|ffn_down_shexp)\.weight$", "Q5_K"), (r"\.(?:ffn_up|ffn_gate|ffn_up_exps|ffn_gate_exps|ffn_up_shexp|ffn_gate_shexp)\.weight$", "Q4_K")]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("pid", ["mixed_4", "mixed_3", "mixed_2"])
def test_every_tensor_of_a_qwen35_model_gets_the_type_the_recipe_means(pid, fused):
    tensors = qwen35_tensors(fused_linear=fused)
    names = [n for n, _, _ in tensors]
    p = mixedquant.preset(pid)
    rules, notes = mixedquant.build_rules(pid, "qwen35", 8, names)
    assert notes == [], notes
    edge = {0, 1, 2, 5, 6, 7, 8} if p["edge"] == 3 else {0, 1, 6, 7, 8}          # 8 is the layer of the speculative-decoding head
    role_of = {"attn_k": "attn_kvo", "attn_v": "attn_kvo", "attn_output": "attn_kvo", "nextn.eh_proj": "attn_kvo", "attn_q": "attn_q", "ffn_down": "ffn_down", "ffn_up": "ffn_updown", "ffn_gate": "ffn_updown",
               "attn_qkv_gate": "lin", "attn_qkv": "lin", "attn_gate": "lin", "ssm_out": "lin", "ssm_beta_alpha": "lin", "ssm_beta": "lin", "ssm_alpha": "lin"}
    for name in names:
        m = re.match(r"blk\.(\d+)\.(.+)\.weight$", name)
        got = first_rule(name, rules)
        if not m or m.group(2) not in role_of:
            assert got is None, f"{name} must follow the base type or its own flag, got {got}"
            continue
        hi, mid = p["types"][role_of[m.group(2)]]
        assert got == (hi if int(m.group(1)) in edge else mid), (name, got)


def test_norms_conv_kernels_state_tensors_the_router_and_the_mtp_head_are_not_touched():
    names = [n for n, _, _ in qwen35_tensors(moe=True)]
    rules, _ = mixedquant.build_rules("mixed_4", "qwen35moe", 8, names)
    for n in names:
        if any(x in n for x in ("_norm.weight", "ssm_conv1d", "ssm_dt", "ffn_gate_inp", "token_embd", "output_norm")) or n in ("output.weight",) or n.endswith(".ssm_a") or ".nextn." in n and "eh_proj" not in n:
            assert first_rule(n, rules) is None, n
    assert first_rule("blk.3.ffn_down_exps.weight", rules) == "Q5_K" and first_rule("blk.0.ffn_gate_exps.weight", rules) == "Q5_K" and first_rule("blk.4.ffn_up_exps.weight", rules) == "Q4_K"
    assert first_rule("blk.8.ffn_down.weight", rules) == "Q6_K", "the head's layer (blk.8, beyond the 8 blocks) is treated like an edge layer: it is one small layer"
    assert first_rule("blk.3.attn_k_norm.weight", rules) is None and first_rule("blk.3.attn_q_norm.weight", rules) is None


def test_a_layer_number_is_matched_whole_not_as_a_prefix():
    rules, _ = mixedquant.build_rules("mixed_3", "qwen35", 64, None)
    assert first_rule("blk.1.ffn_up.weight", rules) == "Q4_K" and first_rule("blk.11.ffn_up.weight", rules) == "Q3_K" and first_rule("blk.62.ffn_up.weight", rules) == "Q4_K" and first_rule("blk.6.ffn_up.weight", rules) == "Q3_K"
    assert first_rule("blk.0.ffn_up.weight", rules) == "Q4_K" and first_rule("blk.63.ffn_up.weight", rules) == "Q4_K" and first_rule("blk.2.ffn_up.weight", rules) == "Q3_K" and first_rule("blk.61.ffn_up.weight", rules) == "Q3_K"


def test_the_regex_is_one_llama_quantize_reads_correctly():
    for pid in mixedquant.PRESET_IDS:
        rules, _ = mixedquant.build_rules(pid, "qwen35", 64, None)
        for rx, t in rules:
            re.compile(rx)
            assert "=" not in rx, "llama-quantize splits NAME=TYPE at the first ="
            assert not re.search(r"\\[A-Z]", rx), "the pattern text is lower-cased by llama-quantize: an upper-case escape would change"
            assert rx == rx.lower() and t in mixedquant.LEVELS and ggufinfo.BPW[t] > 0
            assert t.lower() in ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k"), "a ggml type name, not a file type such as q4_k_m"


def test_flags_are_the_output_and_embedding_types_then_every_rule_in_order():
    rules, _ = mixedquant.build_rules("mixed_3", "qwen35", 8, None)
    f = mixedquant.flags("mixed_3", rules)
    assert f[:4] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K"]
    assert f[4::2] == ["--tensor-type"] * len(rules) and f[5::2] == [f"{rx}={t}" for rx, t in rules]
    assert mixedquant.flags("mixed_fit", [])[:4] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K"]


def test_an_architecture_nobody_taught_this_program_gets_only_the_generic_rules_and_a_note():
    tensors = [("token_embd.weight", [512, 100], F16)] + [t for il in range(6) for t in [(f"blk.{il}.attn_qkv.weight", [512, 1536], F16), (f"blk.{il}.attn_output.weight", [512, 512], F16), (f"blk.{il}.ffn_up.weight", [512, 2048], F16),
                                                                                      (f"blk.{il}.ffn_down.weight", [2048, 512], F16), (f"blk.{il}.ffn_gate.weight", [512, 2048], F16)]]
    names = [n for n, _, _ in tensors]
    rules, notes = mixedquant.build_rules("mixed_4", "llama", None, names)
    assert any("'llama' is not one this program knows the tensor names of" in n and "only the standard blk.N. names" in n for n in notes)
    joined = " ".join(rx for rx, _ in rules)
    assert "ssm_" not in joined and "attn_qkv_gate" not in joined, "no linear-attention rule"
    assert first_rule("blk.3.attn_qkv.weight", rules) == "Q6_K" and first_rule("blk.3.ffn_up.weight", rules) == "Q5_K" and first_rule("blk.0.ffn_up.weight", rules) == "Q5_K", "6 layers with 3 edge layers at each end: all of them are edge layers"
    rules8, _ = mixedquant.build_rules("mixed_4", "llama", 12, names)
    assert first_rule("blk.5.ffn_up.weight", rules8) == "Q4_K" and first_rule("blk.5.attn_qkv.weight", rules8) == "Q6_K"
    rules, notes = mixedquant.build_rules("mixed_4", None, 6, names)
    assert any("architecture of the model could not be read" in n for n in notes)


def test_the_number_of_layers_is_taken_from_the_names_without_the_mtp_head_and_a_note_says_when_it_cannot_be():
    names = [n for n, _, _ in qwen35_tensors(n_layer=8)]
    r1, _ = mixedquant.build_rules("mixed_3", "qwen35", None, names)
    r2, _ = mixedquant.build_rules("mixed_3", "qwen35", 8, names)
    assert r1 == r2, "blk.8. is the MTP head, not a ninth layer"
    rules, notes = mixedquant.build_rules("mixed_3", "qwen35", None, ["token_embd.weight"])
    assert any("number of layers could not be read" in n for n in notes) and all(not rx.startswith("blk") for rx, _ in rules)


def test_a_rule_that_matches_nothing_is_reported():
    names = [n for n, _, _ in qwen35_tensors() if "ssm" not in n and "attn_qkv" not in n and "attn_gate" not in n]
    _, notes = mixedquant.build_rules("mixed_4", "qwen35", 8, names)
    assert any("A rule matched no tensor of this model" in n and "attn_qkv_gate" in n for n in notes)


def test_small_models_have_only_edge_layers():
    assert mixedquant.edge_layers(8, 3) == [0, 1, 2, 5, 6, 7] and mixedquant.edge_layers(4, 3) == [0, 1, 2, 3] and mixedquant.edge_layers(64, 2) == [0, 1, 62, 63] and mixedquant.edge_layers(1, 3) == [0]


# ------------------------------------------------------------------------------------------------------------------ sizes

def test_quantizable_follows_llama_quants_generic_rules():
    q = mixedquant.quantizable
    assert q("blk.0.ffn_up.weight", [512, 2048]) and q("token_embd.weight", [512, 100]) and q("blk.0.ssm_out.weight", [512, 512])
    assert not q("blk.0.attn_norm.weight", [512]) and not q("blk.0.ssm_a", [16]) and not q("blk.0.ssm_dt.bias", [16]) and not q("blk.0.ssm_conv1d.weight", [4, 1024])
    assert not q("blk.0.ffn_gate_inp.weight", [512, 8]) and not q("blk.0.attn_q_norm.weight", [64]) and not q("blk.0.ffn_up.weight", [512])
    assert mixedquant.effective_type("Q4_K", 512) == "Q4_K" and mixedquant.effective_type("Q4_K", 288) == "Q5_0" and mixedquant.effective_type("Q6_K", 288) == "Q8_0" and mixedquant.effective_type("Q3_K", 100) == "F16"


def test_the_estimate_adds_every_tensors_bits_exactly():
    tensors = [("token_embd.weight", [256, 1000], F16), ("blk.0.ffn_up.weight", [256, 512], F16), ("blk.0.attn_norm.weight", [256], F32)]
    info = info_of(tensors, n_layer=1)
    rules, _ = mixedquant.build_rules("mixed_4", "qwen35", 1, [n for n, _, _ in tensors])
    est = mixedquant.estimate(info, rules, mixedquant.preset("mixed_4"))
    # no output.weight: the embedding follows the output type (Q6_K); the only layer is an edge layer: ffn_up Q5_K; the norm stays F32
    want = info["data_start"] + 256 * 1000 * 6.5625 / 8 + 256 * 512 * 5.5 / 8 + 256 * 32 / 8
    assert est["bytes"] == int(want) and est["gb"] == round(want / 1e9, 2) and est["params"] == 256 * 1000 + 256 * 512 + 256
    assert est["bpw"] == round(want * 8 / est["params"], 3) and est["unmatched"] == 0 and est["by_type"] == {"Q6_K": 256000, "Q5_K": 131072, "F32": 256}
    # with its own output tensor the embedding gets its own type
    tensors.append(("output.weight", [256, 1000], F16))
    est2 = mixedquant.estimate(info_of(tensors, n_layer=1), rules, mixedquant.preset("mixed_4"))
    assert est2["by_type"]["Q5_K"] == 131072 + 256000 and est2["by_type"]["Q6_K"] == 256000


def test_the_presets_get_smaller_in_that_order_on_a_qwen35_model_and_the_estimate_matches_a_hand_sum():
    tensors = qwen35_tensors()
    info = info_of(tensors)
    names = [n for n, _, _ in tensors]
    ests = []
    for p in mixedquant.PRESETS:
        rules, _ = mixedquant.build_rules(p["id"], "qwen35", 8, names)
        ests.append(mixedquant.estimate(info, rules, p))
    assert ests[0]["bytes"] > ests[1]["bytes"] > ests[2]["bytes"] and all(e["dead_rules"] == [] for e in ests)
    f16 = info["data_start"] + sum(ggufinfo.numel({"dims": d}) * ggufinfo.TYPES[t][1] / 8 for _, d, t in tensors)
    assert ests[0]["bytes"] < f16 / 2 and ests[2]["bytes"] < ests[0]["bytes"]
    # recompute mixed_2 by hand from the first-match rule of every tensor
    rules, _ = mixedquant.build_rules("mixed_2", "qwen35", 8, names)
    p = mixedquant.preset("mixed_2")
    total = info["data_start"]
    for name, dims, typ in tensors:
        n = ggufinfo.numel({"dims": dims})
        if not mixedquant.quantizable(name, dims):
            total += n * ggufinfo.TYPES[typ][1] / 8
        elif name == "token_embd.weight":
            total += n * 6.5625 / 8 if "output.weight" not in names else n * ggufinfo.BPW[p["embd"]] / 8
        elif name == "output.weight":
            total += n * ggufinfo.BPW[p["output"]] / 8
        else:
            t = first_rule(name, rules) or "Q2_K"
            total += n * ggufinfo.BPW[mixedquant.effective_type(t, dims[0])] / 8
    assert ests[2]["bytes"] == int(total)


def test_the_bits_per_weight_table_matches_ggml_commons_block_structs():
    # bytes per block x 8 / values per block, from the static_asserts of ggml/src/ggml-common.h (QK_K = 256)
    blocks = {"Q2_K": (2 * 2 + 16 + 64, 256), "Q3_K": (2 + 64 + 32 + 12, 256), "Q4_K": (2 * 2 + 12 + 128, 256), "Q5_K": (2 * 2 + 12 + 128 + 32, 256), "Q6_K": (2 + 16 + 192, 256), "Q8_0": (2 + 32, 32),
              "IQ2_XXS": (2 + 64, 256), "IQ2_XS": (2 + 64 + 8, 256), "IQ2_S": (2 + 64 + 16, 256), "IQ3_XXS": (2 + 96, 256), "IQ3_S": (2 + 104 + 4, 256), "IQ1_S": (2 + 32 + 16, 256), "IQ1_M": (32 + 16 + 8, 256),
              "IQ4_NL": (2 + 16, 32), "IQ4_XS": (2 + 2 + 4 + 128, 256), "Q4_0": (2 + 16, 32), "Q5_0": (2 + 4 + 16, 32), "Q4_1": (2 * 2 + 16, 32), "Q5_1": (2 * 2 + 4 + 16, 32)}
    for name, (size, n) in blocks.items():
        assert ggufinfo.BPW[name] == size * 8 / n, name
    assert ggufinfo.BPW["Q8_0"] == 8.5 and ggufinfo.BPW["Q6_K"] == 6.5625 and ggufinfo.BPW["Q5_K"] == 5.5 and ggufinfo.BPW["Q4_K"] == 4.5 and ggufinfo.BPW["Q3_K"] == 3.4375 and ggufinfo.BPW["Q2_K"] == 2.625


# ------------------------------------------------------------------------------------------------------------------ fit to a size

def fit_world(n_layer=8):
    tensors = qwen35_tensors(n_layer=n_layer)
    info = info_of(tensors, n_layer=n_layer)
    imp = {}
    for name, dims, _ in tensors:
        if mixedquant.quantizable(name, dims) and name.startswith("blk.") and "nextn" not in name:
            il = int(name.split(".")[1])
            imp[name] = 1.0 + il * 0.5 + (3.0 if "attn_k" in name or "attn_v" in name else 0.0) + (0.7 if "ffn_down" in name else 0.0)
    return tensors, info, imp


def levels_of(res):
    return {n: mixedquant.LEVELS.index(t) for n, t in res["assignment"].items()}


def test_the_fit_is_deterministic_and_never_exceeds_the_budget_and_its_estimate_agrees():
    _, info, imp = fit_world()
    fx = {"base": mixedquant.FIT_BASE, "embd": mixedquant.FIT_EMBD, "output": mixedquant.FIT_OUTPUT}
    lo, hi = mixedquant.fit(info, 0, imp)["min_bytes"], mixedquant.fit(info, 0, imp)["max_bytes"]
    for frac in (0.0, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0):
        target = int(lo + (hi - lo) * frac)
        a, b = mixedquant.fit(info, target, imp), mixedquant.fit(info, target, imp)
        assert a == b and a["rules"] == b["rules"], "the same input gives the same rules"
        assert a["fits"] and a["bytes"] <= target
        est = mixedquant.estimate(info, a["rules"], fx)
        assert est["bytes"] == a["bytes"], "the rule list, read the way llama-quantize reads it, makes the size the fit counted"
        assert est["unmatched"] == 0 and est["dead_rules"] == []
        for name, t in a["assignment"].items():
            assert first_rule(name, a["rules"]) == t, name


def test_a_bigger_budget_never_gives_any_tensor_fewer_bits_and_never_a_smaller_file():
    _, info, imp = fit_world()
    lo = mixedquant.fit(info, 0, imp)["min_bytes"]
    hi = mixedquant.fit(info, 0, imp)["max_bytes"]
    prev = None
    for i in range(0, 41):
        target = int(lo + (hi - lo) * i / 40)
        r = mixedquant.fit(info, target, imp)
        lv = levels_of(r)
        if prev is not None:
            assert r["bytes"] >= prev[0] and all(lv[n] >= prev[1][n] for n in lv), f"budget step {i}"
        prev = (r["bytes"], lv)
    assert prev[0] == hi, "the highest budget gives the highest types everywhere"


def test_the_important_tensors_get_the_bits_first():
    _, info, imp = fit_world()
    lo, hi = mixedquant.fit(info, 0, imp)["min_bytes"], mixedquant.fit(info, 0, imp)["max_bytes"]
    r = mixedquant.fit(info, int(lo + (hi - lo) * 0.4), imp)
    lv = levels_of(r)
    assert lv["blk.7.attn_v.weight"] >= lv["blk.3.attn_v.weight"] >= lv["blk.7.ffn_up.weight"] >= 0, "importance falls with the layer number and with the role"
    assert lv["blk.7.ffn_down.weight"] >= lv["blk.7.ffn_up.weight"]
    assert max(lv.values()) > min(lv.values()), "a middle budget gives a mixture"
    # the same tensors in two layers with the same figure tie: the lower name wins, always the same way
    flat = {n: 2.0 for n in imp}
    t1 = mixedquant.fit(info, int(lo + (hi - lo) * 0.3), flat)
    assert t1 == mixedquant.fit(info, int(lo + (hi - lo) * 0.3), dict(reversed(list(flat.items()))))


def test_a_budget_below_the_lowest_types_and_one_above_the_highest_are_said_plainly():
    _, info, imp = fit_world()
    r = mixedquant.fit(info, 1000, imp)
    assert r["fits"] is False and set(r["assignment"].values()) == {"Q2_K"} and r["bytes"] == r["min_bytes"] and any("Even the lowest types (Q2_K) need" in n for n in r["notes"])
    r = mixedquant.fit(info, 10 ** 12, imp)
    assert r["fits"] and set(r["assignment"].values()) == {"Q6_K"} and any("more than the highest types (Q6_K) can use" in n for n in r["notes"])


def test_the_embedding_and_output_are_fixed_and_the_untouched_tensors_stay():
    tensors, info, imp = fit_world()
    r = mixedquant.fit(info, 10 ** 12, imp)
    assert not any(n in r["assignment"] for n in ("token_embd.weight", "output.weight", "blk.0.attn_norm.weight", "blk.0.ssm_conv1d.weight"))
    est = mixedquant.estimate(info, r["rules"], {"base": "Q2_K", "embd": mixedquant.FIT_EMBD, "output": mixedquant.FIT_OUTPUT})
    assert est["by_type"]["Q4_K"] >= 512 * 1000 and est["by_type"]["Q6_K"] >= 512 * 1000


def test_tensors_the_matrix_has_no_figure_for_get_the_median_and_a_note():
    _, info, imp = fit_world()
    some = {k: v for i, (k, v) in enumerate(sorted(imp.items())) if i % 2 == 0}
    r = mixedquant.fit(info, int(mixedquant.fit(info, 0, imp)["max_bytes"] * 0.7), some)
    assert r["missing"] == len(imp) - len(some) + len([n for n in r["assignment"] if n not in imp]) and any("had no figure in the matrix" in n for n in r["notes"])
    with pytest.raises(mixedquant.FitError, match="no statistics for any tensor"):
        mixedquant.fit(info, 10 ** 10, {"unrelated.weight": 1.0})
    with pytest.raises(mixedquant.FitError, match="no tensors to fit"):
        mixedquant.fit(info_of([("n.bias", [4], F32)], n_layer=1), 10 ** 6, {"x": 1.0})


def test_the_rule_list_of_a_fit_is_short_and_names_layers():
    _, info, imp = fit_world(n_layer=16)
    r = mixedquant.fit(info, int(mixedquant.fit(info, 0, imp)["max_bytes"] * 0.6), imp)
    assert 2 <= len(r["rules"]) <= 8 * 5 and all(rx.startswith("blk\\.(?:") and rx.endswith("$") for rx, _ in r["rules"])
    assert len({rx for rx, _ in r["rules"]}) == len(r["rules"]), "one rule for each role and type"


# ------------------------------------------------------------------------------------------------------------------ reading the matrix

def matrix_with_data(path, stats, align=32):
    """a GGUF importance matrix with float32 data: stats = {weight name: (n_in, n_experts, sums list, counts list)}"""
    kvs = [s("general.type") + struct.pack("<I", 8) + s("imatrix"), s("imatrix.chunk_count") + struct.pack("<II", 4, 64)]
    tensors, blob, off = [], b"", 0
    for name, (n_in, nexp, sums, counts) in stats.items():
        for suffix, dims, vals in ((".in_sum2", [n_in, nexp], sums), (".counts", [nexp], counts)):
            data = struct.pack(f"<{len(vals)}f", *vals)
            tensors.append(s(name + suffix) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, off))
            pad = (align - len(data) % align) % align
            blob += data + b"\0" * pad
            off += len(data) + pad
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)
    Path(path).write_bytes(head + b"\0" * ((align - len(head) % align) % align) + blob)
    return str(path)


def test_the_importance_of_a_tensor_is_its_mean_squared_input_from_in_sum2_and_counts(tmp_path):
    m = matrix_with_data(tmp_path / "m.gguf", {"blk.0.attn_k.weight": (4, 1, [2, 4, 6, 8], [2]), "blk.1.ffn_up_exps.weight": (2, 2, [1, 3, 10, 30], [1, 10]), "blk.2.empty.weight": (2, 1, [5, 5], [0])})
    got = mixedquant.read_importance(m)
    assert got["blk.0.attn_k.weight"] == pytest.approx(2.5) and got["blk.1.ffn_up_exps.weight"] == pytest.approx(2.0)
    assert got["blk.2.empty.weight"] == 0.0


def test_a_matrix_that_cannot_be_read_gives_a_plain_reason(tmp_path):
    (tmp_path / "old.dat").write_bytes(b"LEGACY-DAT-FILE")
    with pytest.raises(mixedquant.FitError, match="could not be read as a GGUF matrix"):
        mixedquant.read_importance(str(tmp_path / "old.dat"))
    with pytest.raises(mixedquant.FitError, match="no per-tensor statistics"):
        mixedquant.read_importance(gguf_file(tmp_path / "model.gguf", qwen35_tensors(n_layer=1, nextn=False)))
    good = Path(matrix_with_data(tmp_path / "g.gguf", {"blk.0.attn_k.weight": (4, 1, [2, 4, 6, 8], [2])}))
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(good.read_bytes()[:-40])
    with pytest.raises(mixedquant.FitError, match="cut short"):
        mixedquant.read_importance(str(cut))
    with pytest.raises(mixedquant.FitError, match="could not be read"):
        mixedquant.read_importance(str(tmp_path / "gone.gguf"))


# ------------------------------------------------------------------------------------------------------------------ what the export and the page use

def real_matrix(tmp_path, tensors, name="m.gguf"):
    stats = {n: (d[0], 1, [1.0 + i % 5] * d[0], [10.0]) for i, (n, d, _) in enumerate(tensors) if mixedquant.quantizable(n, d) and n.startswith("blk.") and "nextn" not in n}
    return matrix_with_data(tmp_path / name, stats)


def test_for_export_gives_a_preset_with_its_flags_estimate_and_label(tmp_path):
    f16 = gguf_file(tmp_path / "f16.gguf", qwen35_tensors())
    r = mixedquant.for_export("mixed_3", f16)
    assert (r["id"], r["label"], r["recipe"], r["base"], r["fit"]) == ("mixed_3", "Mixed 3-bit (small)", "Mixed precision (our own recipe)", "Q3_K_M", None)
    assert r["flags"][:4] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K"] and r["flags"][4] == "--tensor-type" and len(r["rules"]) == (len(r["flags"]) - 4) // 2
    assert r["estimate"]["bytes"] > 0 and r["estimate"]["dead_rules"] == [] and r["notes"] == []
    with pytest.raises(KeyError):
        mixedquant.for_export("q4_k_m", f16)
    (tmp_path / "bad.gguf").write_bytes(b"nope")
    with pytest.raises(ValueError):
        mixedquant.for_export("mixed_4", str(tmp_path / "bad.gguf"))


def test_the_fit_for_an_export_uses_the_matrix_and_the_target_and_falls_back_to_a_preset_with_a_note(tmp_path):
    tensors = qwen35_tensors()
    f16 = gguf_file(tmp_path / "f16.gguf", tensors)
    m = real_matrix(tmp_path, tensors)
    sizes = {p["id"]: mixedquant.for_export(p["id"], f16)["estimate"]["bytes"] for p in mixedquant.PRESETS}
    target_gb = (sizes["mixed_3"] + 100) / 1e9
    r = mixedquant.for_export("mixed_fit", f16, {"target_gb": target_gb}, m)
    assert r["id"] == "mixed_fit" and r["label"] == "Mixed: fit to a size (experimental)" and r["fit"]["fits"] and r["estimate"]["bytes"] <= target_gb * 1e9 and r["recipe"] == "Mixed precision (our own recipe)" and "fell_back_to" not in r
    assert r["base"] == "Q2_K" and r["flags"][:4] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K"]
    # no matrix: the largest preset that fits the target, and the reason
    r = mixedquant.for_export("mixed_fit", f16, {"target_gb": (sizes["mixed_3"] + 100) / 1e9}, None)
    assert r["fell_back_to"] == "mixed_3" and r["id"] == "mixed_3" and r["fit"] is None and any("Fit to a size could not be done (No importance matrix is available" in n and 'The preset "Mixed 3-bit (small)" was used instead' in n for n in r["notes"])
    # a target the lowest types cannot meet: the fit says so and gives the lowest types (it does not pretend)
    low = mixedquant.for_export("mixed_fit", f16, {"target_gb": 0.0001}, m)
    assert low["fit"]["fits"] is False and low["id"] == "mixed_fit" and any("Even the lowest types" in n for n in low["notes"])
    # no matrix and a target nothing fits: the smallest preset
    r = mixedquant.for_export("mixed_fit", f16, {"target_gb": 0.00001}, None)
    assert r["fell_back_to"] == "mixed_2"
    # an unreadable matrix and no target
    (tmp_path / "old.dat").write_bytes(b"LEGACY")
    assert mixedquant.for_export("mixed_fit", f16, {"target_gb": 1.0}, str(tmp_path / "old.dat"))["fell_back_to"] in mixedquant.PRESET_IDS
    assert "No target size was given" in " ".join(mixedquant.for_export("mixed_fit", f16, {}, m)["notes"])


def test_the_plan_job_lists_every_preset_with_rules_and_size_and_the_fit(tmp_path):
    tensors = qwen35_tensors()
    f16 = gguf_file(tmp_path / "f16.gguf", tensors)
    m = real_matrix(tmp_path, tensors)
    r = mixedquant.plan_job({"job": "mixed_plan", "gguf": f16})
    assert [p["id"] for p in r["presets"]] == ["mixed_4", "mixed_3", "mixed_2"] and r["arch"] == "qwen35" and r["n_layer"] == 8 and r["recipe"] == "Mixed precision (our own recipe)" and r["fit"] is None
    assert [p["estimate"]["gb"] for p in r["presets"]] == sorted((p["estimate"]["gb"] for p in r["presets"]), reverse=True)
    assert all(p["rules"] and p["flags"][0] == "--output-tensor-type" and p["estimate"]["bpw"] for p in r["presets"]) and r["plain_bpw"]["Q4_K_M"] == 4.8944
    r = mixedquant.plan_job({"job": "mixed_plan", "gguf": f16, "target_gb": 0.05, "matrix": m})
    assert r["fit"]["ok"] and r["fit"]["label"] == "Mixed: fit to a size (experimental)" and r["fit"]["rules"] and r["fit"]["estimate"]["bytes"] <= 0.05e9 and r["fit"]["flags"][0] == "--output-tensor-type"
    r = mixedquant.plan_job({"job": "mixed_plan", "gguf": f16, "target_gb": 0.05})
    assert r["fit"]["ok"] is False and "Choose a saved importance matrix" in r["fit"]["error"]
    r = mixedquant.plan_job({"job": "mixed_plan", "gguf": f16, "target_gb": 0.05, "matrix": str(tmp_path / "nope.gguf")})
    assert r["fit"]["ok"] is False and "could not be read" in r["fit"]["error"]
    with pytest.raises(ValueError, match="Choose a complete GGUF file of this model"):
        mixedquant.plan_job({"job": "mixed_plan", "gguf": str(tmp_path / "gone.gguf")})
    with pytest.raises(ValueError):
        mixedquant.plan_job({"job": "mixed_plan"})


def test_the_worker_entry_runs_the_plan_and_reports_a_bad_file_in_plain_words(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))
    f16 = gguf_file(tmp_path / "f16.gguf", qwen35_tensors())
    jobfile = tmp_path / "job.json"
    jobfile.write_text(json.dumps({"job": "mixed_plan", "gguf": f16}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0 and [e for e in events if e["type"] == "result"][0]["data"]["presets"][0]["id"] == "mixed_4"
    jobfile.write_text(json.dumps({"job": "mixed_plan", "gguf": str(tmp_path / "gone.gguf")}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 1
    e = [x for x in events if x["type"] == "error"][0]
    assert e["code"] == "mixed_failed" and "Choose a complete GGUF file of this model" in e["message"]


def test_the_rust_table_of_sizes_and_this_one_say_the_same_things():
    src = Path(__file__).resolve().parents[2] / "crates" / "ndforge-core" / "src" / "export" / "runs.rs"
    if not src.is_file():
        pytest.skip("the Rust program is not next to the worker here (the worker is embedded elsewhere)")
    text = src.read_text(encoding="utf-8")
    for p in mixedquant.PRESETS:
        assert f'("{p["id"]}", "{p["label"]}' in text, p["id"]
    assert f'("{mixedquant.FIT_ID}", "{mixedquant.FIT_LABEL}' in text and mixedquant.LABEL in text


# ------------------------------------------------------------------------------------------------------------------ the export

QUANTIZE_FLAGS = '''import json, os, sys
a = sys.argv[1:]
open(os.environ["STUB_LOG"], "a").write(json.dumps({"tool": "quantize", "args": a}) + "\\n")
mat, rest, i = None, [], 0
while i < len(a):
    if a[i] in ("--imatrix", "--output-tensor-type", "--token-embedding-type", "--tensor-type"):
        if a[i] == "--imatrix":
            mat = a[i + 1]
        i += 2
    else:
        rest.append(a[i])
        i += 1
if mat and os.environ.get("QUANTIZE_REFUSES_IMATRIX"):
    print("error: invalid argument: --imatrix")
    sys.exit(1)
src, dst, quant = rest[0], rest[1], rest[2]
open(dst, "wb").write(("GGUF-" + quant + ("+imatrix" if mat else "")).encode())
'''

CONVERT_GGUF = '''import os, shutil, sys
a = sys.argv
shutil.copy(os.environ["PREBUILT_GGUF"], a[a.index("--outfile") + 1])
print("converted")
'''


@pytest.fixture
def mw(world, monkeypatch, tmp_path):
    """world + a converter that writes a real qwen35 GGUF header, so that the recipes can be planned"""
    (world.lcpp / "convert_hf_to_gguf.py").write_text(CONVERT_GGUF, encoding="utf-8")
    (world.lcpp / "llama-quantize.py").write_text(QUANTIZE_FLAGS, encoding="utf-8")
    tensors = qwen35_tensors()
    monkeypatch.setenv("PREBUILT_GGUF", gguf_file(tmp_path / "prebuilt.gguf", tensors))
    world.tensors = tensors
    return world


def quantize_calls(w):
    return [c["args"] for c in w.calls("quantize")]


def test_a_mixed_size_is_one_llama_quantize_call_with_the_recipes_flags_before_the_files(mw):
    files, rep, out = mw.export(["mixed_4", "q4_k_m"])
    mixed, plain = quantize_calls(mw)
    assert mixed[:2] == ["--imatrix", str(out / "imatrix.gguf")] and mixed[2:6] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q5_K"]
    assert mixed[-3:] == [str(out / "model-f16.gguf"), str(out / "model-mixed_4.gguf"), "Q4_K_M"], "the files last, then the base type of the 4-bit recipe"
    rules = [mixed[i + 1] for i, x in enumerate(mixed) if x == "--tensor-type"]
    assert len(rules) == 9 and rules[0].endswith("=Q6_K") and all(r.count("=") == 1 for r in rules)
    assert mixed[6:-3] == [x for r in rules for x in ("--tensor-type", r)], "options first, then the files (the position llama-quantize's README shows)"
    assert plain == ["--imatrix", str(out / "imatrix.gguf"), str(out / "model-f16.gguf"), str(out / "model-q4_k_m.gguf"), "Q4_K_M"], "the plain size is made exactly as before"
    assert files["mixed_4"] == out / "model-mixed_4.gguf" and files["mixed_4"].read_bytes() == b"GGUF-Q4_K_M+imatrix" and files["q4_k_m"].is_file()
    assert rep["applied"] == ["mixed_4", "q4_k_m"] and rep["notes"] == [], "the matrix is given to the mixed size like to the plain ones"


def test_without_a_matrix_the_mixed_size_still_gets_its_flags(mw):
    out = mw.root / "out2"
    out.mkdir()
    cfg = {k: v for k, v in mw.cfg().items() if k != "imatrix"}
    mixed_info = {}
    files = exportjob.gguf_with_llama_cpp(cfg, mw.merged, out, ["mixed_2"], None, mixed_info)
    (call,) = quantize_calls(mw)
    assert "--imatrix" not in call and call[:4] == ["--output-tensor-type", "Q6_K", "--token-embedding-type", "Q4_K"] and call[-1] == "Q2_K"
    info = mixed_info["mixed_2"]
    assert (info["id"], info["label"], info["recipe"], info["base"], info["matrix_used"]) == ("mixed_2", "Mixed 2-bit (smallest: short answers only)", "Mixed precision (our own recipe)", "Q2_K", False)
    assert info["rules"] == [call[i + 1] for i, x in enumerate(call) if x == "--tensor-type"] and info["estimate"]["bytes"] > 0 and info["actual_bytes"] == files["mixed_2"].stat().st_size


def test_the_report_and_the_files_never_use_the_words_dynamic_or_ud(mw):
    _, rep, out = mw.export(["mixed_4", "mixed_3", "mixed_2"])
    assert not any(re.search(r"dynamic|UD-", p.name, re.I) for p in out.iterdir())
    assert sorted(p.name for p in out.glob("model-*.gguf")) == ["model-mixed_2.gguf", "model-mixed_3.gguf", "model-mixed_4.gguf"]
    assert not re.search(r"dynamic|UD-", json.dumps(rep), re.I)


def test_a_quantize_that_refuses_the_matrix_makes_the_mixed_size_without_it_and_says_so(mw, monkeypatch):
    monkeypatch.setenv("QUANTIZE_REFUSES_IMATRIX", "1")
    files, rep, out = mw.export(["mixed_3"])
    assert files["mixed_3"].is_file() and rep["applied"] == [] and rep["not_applied"]["mixed_3"] == "llama-quantize refused the matrix"
    assert any("did not accept the importance matrix for Mixed 3-bit (small)" in n for n in rep["notes"])
    calls = quantize_calls(mw)
    assert "--imatrix" in calls[0] and "--imatrix" not in calls[1] and calls[1][:2] == ["--output-tensor-type", "Q6_K"]


def test_a_mixed_size_with_a_library_matrix_fits_to_the_target_with_the_matrix_statistics(mw, tmp_path):
    m = real_matrix(tmp_path, mw.tensors, "lib.gguf")
    target = (mixedquant.for_export("mixed_3", str(tmp_path / "prebuilt.gguf"))["estimate"]["bytes"] + 100) / 1e9
    cfg = mw.cfg(use_file=m, use_id="lib1")
    cfg["mixed_fit"] = {"target_gb": target, "matrix": None}
    out = mw.root / "out3"
    out.mkdir()
    info = {}
    rep = exportjob.new_report()
    exportjob.gguf_with_llama_cpp(cfg, mw.merged, out, ["mixed_fit"], rep, info)
    r = info["mixed_fit"]
    assert r["id"] == "mixed_fit" and r["fit"]["fits"] and r["estimate"]["bytes"] <= target * 1e9 and r["label"] == "Mixed: fit to a size (experimental)"
    (call,) = quantize_calls(mw)
    assert call[0] == "--imatrix" and call[-1] == "Q2_K" and rep["applied"] == ["mixed_fit"]
    assert [call[i + 1] for i, x in enumerate(call) if x == "--tensor-type"] == r["rules"]


def test_with_unsloth_chosen_the_mixed_sizes_go_the_llama_cpp_way_for_the_whole_batch_and_it_is_said(unsloth_world, monkeypatch, tmp_path):
    w = unsloth_world.world
    (w.lcpp / "convert_hf_to_gguf.py").write_text(CONVERT_GGUF, encoding="utf-8")
    (w.lcpp / "llama-quantize.py").write_text(QUANTIZE_FLAGS, encoding="utf-8")
    monkeypatch.setenv("PREBUILT_GGUF", gguf_file(tmp_path / "prebuilt.gguf", qwen35_tensors()))
    out, result = unsloth_world.job(quants=("mixed_4", "q4_k_m"))
    assert FakeModel.calls == [] and unsloth_world.loaded == [], "Unsloth is not loaded and not asked to save"
    assert result["via"] == "llama.cpp" and result["imatrix"]["via"] == "llama.cpp"
    assert any("mixed-precision sizes need llama-quantize's per-tensor types, which Unsloth's export cannot do" in n and n.endswith("for every size.") for n in result["mixed_notes"])
    assert set(result["ggufs"]) == {"mixed_4", "q4_k_m"} and result["mixed"]["mixed_4"]["label"] == "Mixed 4-bit (balanced)" and result["mixed"]["mixed_4"]["recipe"] == "Mixed precision (our own recipe)"
    assert (out / "model-mixed_4.gguf").is_file() and (out / "model-q4_k_m.gguf").is_file()
    out2, result2 = unsloth_world.job(out=w.root / "job2", quants=("mixed_3",))
    assert result2["mixed_notes"][0].endswith("llama-quantize (convert, then llama-quantize)") is False and result2["mixed_notes"][0].endswith(".") and "for every size" not in result2["mixed_notes"][0]
    assert result2["via"] == "llama.cpp"


def test_unsloth_with_only_plain_sizes_is_exactly_as_before(unsloth_world):
    out, result = unsloth_world.job(quants=("q4_k_m", "q8_0"))
    assert result["via"] == "unsloth" and result["mixed"] is None and result["mixed_notes"] == []
    assert len(FakeModel.calls) == 1


def test_mixed_sizes_without_llama_cpp_say_so_before_anything_is_made(unsloth_world, monkeypatch):
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    with pytest.raises(RuntimeError, match="The mixed-precision sizes need llama.cpp .*Unsloth cannot give single tensors their own types"):
        unsloth_world.job(quants=("mixed_4",))
    assert unsloth_world.loaded == [] and unsloth_world.merges == []


def test_an_unknown_size_is_still_refused_and_the_mixed_ones_are_known():
    assert "mixed_4" in exportjob.QUANTS and "mixed_fit" in exportjob.QUANTS and exportjob.QUANTS[:5] == ("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16")
    with pytest.raises(ValueError, match="Unknown quantisation"):
        exportjob.run({"out": "x", "quants": ["mixed_9"], "base": {"repo": "a", "family": "b"}})


def test_an_export_that_cannot_read_the_16_bit_file_for_the_recipe_fails_plainly(world):
    (world.lcpp / "convert_hf_to_gguf.py").write_text("import sys\na = sys.argv\nopen(a[a.index('--outfile') + 1], 'wb').write(b'GGUF-f16')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="The 16-bit GGUF could not be read to plan the mixed-precision size"):
        world.export(["mixed_4"])
