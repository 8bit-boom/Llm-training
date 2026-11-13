"""A small fake model folder (real key layout, tiny shapes, deterministic bytes) for the "Slim or grow" tests. No torch needed."""
import json
from pathlib import Path

from stfile import BYTES, fill, numel, write_file

DATA = Path(__file__).parent / "data"
HIDDEN, FFN, VOCAB = 8, 16, 10


def layer_shapes(i, family):
    """(rest, shape) of one decoder layer; Qwen3.5 layers 3, 7, ... are full attention, the others linear attention"""
    mixer = "self_attn.o_proj.weight" if (family != "qwen3_5" or i % 4 == 3) else "linear_attn.out_proj.weight"
    return [("input_layernorm.weight", (HIDDEN,)), (mixer, (HIDDEN, HIDDEN)), ("mlp.up_proj.weight", (FFN, HIDDEN)), ("mlp.down_proj.weight", (HIDDEN, FFN))]


def tensor_list(layers, family):
    pre = "model.language_model.layers." if family == "qwen3_5" else "model.layers."
    emb = "model.language_model.embed_tokens.weight" if family == "qwen3_5" else "model.embed_tokens.weight"
    t = [(emb, (VOCAB, HIDDEN))]
    for i in range(layers):
        t += [(f"{pre}{i}.{rest}", shape) for rest, shape in layer_shapes(i, family)]
    if family == "qwen3_5":
        t += [("model.visual.blocks.0.attn.qkv.weight", (3, HIDDEN)), ("mtp.fc.weight", (HIDDEN, 2 * HIDDEN)), ("mtp.layers.0.mlp.down_proj.weight", (HIDDEN, HIDDEN))]
    t.append(("lm_head.weight", (VOCAB, HIDDEN)))
    return t


def config(layers, family):
    if family == "qwen3_5":
        cfg = json.loads((DATA / "qwen3_5_config.json").read_text(encoding="utf-8"))
        cfg["model_type"] = "qwen3_5"
        cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
        cfg["vision_config"] = {"depth": 1, "hidden_size": HIDDEN, "out_hidden_size": HIDDEN}
        tc = cfg["text_config"]
        tc.update(hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=layers, vocab_size=VOCAB, mlp_only_layers=[],
                  layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (layers // 4))
        cfg.pop("note", None)
        return cfg
    return {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": HIDDEN, "intermediate_size": FFN, "num_hidden_layers": layers,
            "vocab_size": VOCAB, "num_attention_heads": 2}


def _side_files(d, vision):
    (d / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    (d / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (d / "generation_config.json").write_text('{"do_sample": true}', encoding="utf-8")
    (d / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Fake model\n", encoding="utf-8")
    if vision:
        (d / "preprocessor_config.json").write_text('{"image_mean": [0.5]}', encoding="utf-8")


def fake_model(tmp, layers=8, family="qwen3_5", name="src"):
    """Writes the folder (two shards + index, config, tokenizer and processor files, a README with a licence) and returns its path."""
    d = Path(tmp) / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config(layers, family), indent=2), encoding="utf-8")
    items = [(n, "BF16", s, fill(i, numel(s) * 2)) for i, (n, s) in enumerate(tensor_list(layers, family))]
    half = len(items) // 2
    write_file(d / "model-00001-of-00002.safetensors", items[:half])
    write_file(d / "model-00002-of-00002.safetensors", items[half:])
    wm = {n: "model-00001-of-00002.safetensors" for n, *_ in items[:half]} | {n: "model-00002-of-00002.safetensors" for n, *_ in items[half:]}
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}), encoding="utf-8")
    _side_files(d, family == "qwen3_5")
    return d


def tensor_numbers(layers, family):
    """{name: tensor number i} so a test can recompute a tensor's bytes with fill()"""
    return {n: i for i, (n, _) in enumerate(tensor_list(layers, family))}


# ----------------------------------------------------------------------------------------------- Qwen3.5 with the real layer shapes (width pruning)

W = {"hidden": 8, "inter": 32, "heads": 6, "kv": 2, "hd": 4, "nk": 2, "nv": 4, "dk": 4, "dv": 4, "K": 4, "vocab": 16}
VISION_PARAMS = 5


def width_tensor_list(layers=8, w=None):
    """[(name, dtype, shape)] of a Qwen3.5 text model with the real tensor names and layout of every layer type, the multi-token head, a vision tensor
    and A_log / the gated norm in F32 (as on the Hub); sizes from W (tiny)."""
    w = w or W
    H, I, nh, kv, hd, nk, nv, dk, dv, K, V = (w[k] for k in ("hidden", "inter", "heads", "kv", "hd", "nk", "nv", "dk", "dv", "K", "vocab"))
    pre = "model.language_model.layers."
    conv = 2 * nk * dk + nv * dv

    def layer(base, linear):
        t = [("input_layernorm.weight", "BF16", (H,)), ("post_attention_layernorm.weight", "BF16", (H,))]
        if linear:
            t += [("linear_attn.in_proj_qkv.weight", "BF16", (conv, H)), ("linear_attn.conv1d.weight", "BF16", (conv, 1, K)),
                  ("linear_attn.in_proj_z.weight", "BF16", (nv * dv, H)), ("linear_attn.in_proj_b.weight", "BF16", (nv, H)),
                  ("linear_attn.in_proj_a.weight", "BF16", (nv, H)), ("linear_attn.A_log", "F32", (nv,)), ("linear_attn.dt_bias", "BF16", (nv,)),
                  ("linear_attn.norm.weight", "F32", (dv,)), ("linear_attn.out_proj.weight", "BF16", (H, nv * dv))]
        else:
            t += [("self_attn.q_proj.weight", "BF16", (nh * hd * 2, H)), ("self_attn.k_proj.weight", "BF16", (kv * hd, H)),
                  ("self_attn.v_proj.weight", "BF16", (kv * hd, H)), ("self_attn.o_proj.weight", "BF16", (H, nh * hd)),
                  ("self_attn.q_norm.weight", "BF16", (hd,)), ("self_attn.k_norm.weight", "BF16", (hd,))]
        t += [("mlp.gate_proj.weight", "BF16", (I, H)), ("mlp.up_proj.weight", "BF16", (I, H)), ("mlp.down_proj.weight", "BF16", (H, I))]
        return [(base + n, d, s) for n, d, s in t]

    out = [("model.language_model.embed_tokens.weight", "BF16", (V, H))]
    for i in range(layers):
        out += layer(f"{pre}{i}.", i % 4 != 3)
    out += [("model.language_model.norm.weight", "BF16", (H,)), ("model.visual.blocks.0.attn.qkv.weight", "BF16", (VISION_PARAMS,)),
            ("mtp.fc.weight", "BF16", (H, 2 * H)), ("mtp.pre_fc_norm_hidden.weight", "BF16", (H,)), ("mtp.pre_fc_norm_embedding.weight", "BF16", (H,)),
            ("mtp.norm.weight", "BF16", (H,))]
    out += layer("mtp.layers.0.", False)
    out.append(("lm_head.weight", "BF16", (V, H)))
    return out


def width_config(layers=8, w=None):
    w = w or W
    cfg = config(layers, "qwen3_5")
    cfg["vision_config"] = {"depth": 1, "hidden_size": w["hidden"], "out_hidden_size": w["hidden"]}
    cfg["tie_word_embeddings"] = False
    tc = cfg["text_config"]
    tc.pop("tie_word_embeddings", None)
    tc.update(hidden_size=w["hidden"], intermediate_size=w["inter"], num_attention_heads=w["heads"], num_key_value_heads=w["kv"], head_dim=w["hd"],
              linear_num_key_heads=w["nk"], linear_num_value_heads=w["nv"], linear_key_head_dim=w["dk"], linear_value_head_dim=w["dv"],
              linear_conv_kernel_dim=w["K"], vocab_size=w["vocab"], mtp_num_hidden_layers=1)
    return cfg


def fake_qwen_width(tmp, layers=8, name="srcw", w=None):
    """A Qwen3.5 text model folder with the real tensor names and layer shapes at tiny sizes (see W), two shards + index; a few kB. Returns the path."""
    d = Path(tmp) / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(width_config(layers, w), indent=2), encoding="utf-8")
    items = [(n, dt, s, fill(i, numel(s) * BYTES[dt])) for i, (n, dt, s) in enumerate(width_tensor_list(layers, w))]
    half = len(items) // 2
    f1, f2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    write_file(d / f1, items[:half])
    write_file(d / f2, items[half:])
    wm = {n: f1 for n, *_ in items[:half]} | {n: f2 for n, *_ in items[half:]}
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": wm}), encoding="utf-8")
    _side_files(d, True)
    return d


def width_tensor_numbers(layers=8, w=None):
    return {n: i for i, (n, _, _) in enumerate(width_tensor_list(layers, w))}
