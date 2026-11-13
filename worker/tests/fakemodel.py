"""A small fake model folder (real key layout, tiny shapes, deterministic bytes) for the "Slim or grow" tests. No torch needed."""
import json
from pathlib import Path

from stfile import fill, numel, write_file

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
    (d / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    (d / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (d / "generation_config.json").write_text('{"do_sample": true}', encoding="utf-8")
    (d / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Fake model\n", encoding="utf-8")
    if family == "qwen3_5":
        (d / "preprocessor_config.json").write_text('{"image_mean": [0.5]}', encoding="utf-8")
    return d


def tensor_numbers(layers, family):
    """{name: tensor number i} so a test can recompute a tensor's bytes with fill()"""
    return {n: i for i, (n, _) in enumerate(tensor_list(layers, family))}
