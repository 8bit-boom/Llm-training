"""A fake Inkling-2-Mini folder for the tests: the REAL tensor names, dtypes and structure (24 layers = 4 attention-residual blocks of 6, loop core 6-17 run 3 times, global
attention in layers 5/11/17/23, NVFP4 experts in layers 0-19 and MXFP8 in layers 20-23, a drafter file that the index does not list) with tiny widths. No torch.

`specs(cfg)` is the shape rule that reproduces the real index (tests/data/inkling2_tensors.json) for the real config, which `test_moe_layout.py` checks; the same rule makes the tiny
folder, so the folder and the real model cannot drift apart. Bytes are deterministic and differ per expert, so a cut can be compared with a plain Python reference.
"""
import copy
import json
from pathlib import Path

import stfile

DATA = Path(__file__).resolve().parent / "data"
MXFP8_FROM = 20


def real_config() -> dict:
    return json.loads((DATA / "inkling2_config.json").read_text(encoding="utf-8"))


def tiny_config(experts: int = 8, top_k: int = 2) -> dict:
    cfg = real_config()
    tc = cfg["text_config"]
    tc.update({"hidden_size": 64, "vocab_size": 48, "unpadded_vocab_size": 40, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16, "swa_head_dim": 16,
               "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "d_rel": 4, "rel_extent": 8, "intermediate_size": 32, "n_routed_experts": experts,
               "num_experts_per_tok": top_k, "sliding_window_size": 8})
    tc["loop_config"]["gate_rank"] = 4
    cfg["vision_config"].update({"hidden_size": 32, "num_hidden_layers": 1, "text_hidden_size": 64, "projection_hidden_dim": 64})
    cfg["mtp_config"].update({"hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 64})
    return cfg


def specs(cfg: dict, vision_layers: int | None = None) -> list:
    """[(name, dtype, shape)] of the checkpoint for a config. Vision tensors are only emitted when vision_layers is given (the real file has 23 layers)."""
    tc = cfg["text_config"]
    H, V, n = tc["hidden_size"], tc["vocab_size"], tc["num_hidden_layers"]
    heads, kv, hd, rel, ext = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], tc["d_rel"], tc["rel_extent"]
    E, I, k = tc["n_routed_experts"], tc["intermediate_size"], tc["sconv_kernel_size"]
    lc = tc["loop_config"]
    local = set(tc["local_layer_ids"])
    out = []

    def add(name, dtype, shape):
        out.append((name, dtype, tuple(shape)))
    for i in range(n - 1):
        add(f"model.llm.attn_res.attn_queries.{i}", "F32", [3, H] if (i + 1) not in local else [H])
    for i in range(n):
        add(f"model.llm.attn_res.mlp_queries.{i}", "F32", [H])
    add("model.llm.attn_res.output_query", "F32", [H])
    add("model.llm.embed.weight", "BF16", [V, H])
    add("model.llm.embed_norm.weight", "BF16", [H])
    core = lc["core_end"] - lc["core_start"]
    add("model.llm.middle_cycle_cross_pass_kv_queries", "F32", [2 * core, H])
    add("model.llm.middle_cycle_cross_pass_queries", "F32", [2 * core, H])
    add("model.llm.norm.weight", "BF16", [H])
    add("model.llm.unembed.weight", "BF16", [V, H])
    for L in range(n):
        p = f"model.llm.layers.{L}."
        add(p + "attn.attn_sink_proj.weight", "BF16", [heads, H])
        add(p + "attn.gate_attn.weight", "BF16", [heads * hd, H])
        add(p + "attn.q_l2_clip.scale", "BF16", [heads, hd])
        add(p + "attn.rel_l2_clip.scale", "BF16", [heads, rel])
        add(p + "attn.rel_logits_proj.proj", "BF16", [rel, ext])
        add(p + "attn.wk_dv.weight", "BF16", [kv * hd, H])
        add(p + "attn.wo_ud.weight", "BF16", [H, heads * hd])
        add(p + "attn.wq_du.weight", "BF16", [heads * hd, H])
        add(p + "attn.wr_du.weight", "BF16", [heads * rel, H])
        add(p + "attn.wv_dv.weight", "BF16", [kv * hd, H])
        add(p + "attn_sconv.weight", "BF16", [H, 1, k])
        add(p + "mlp_norm.weight", "BF16", [H])
        add(p + "mlp_sconv.weight", "BF16", [H, 1, k])
        add(p + "mlp.gate.weight", "BF16", [E, H])
        add(p + "mlp.gate.bias", "F32", [E])
        if L in local:
            add(p + "attn_norm.weight", "BF16", [H])
            add(p + "attn.k_sconv.weight", "BF16", [kv * hd, 1, k])
            add(p + "attn.v_sconv.weight", "BF16", [kv * hd, 1, k])
        else:
            for x in "kv":
                add(p + f"attn.{x}_sconv.static_w", "BF16", [k, kv * hd])
                add(p + f"attn.{x}_sconv.weight_proj", "BF16", [k * kv, H])
            for x in "kqv":
                add(p + f"{x}_input_norm.weight", "BF16", [H])
        if lc["core_start"] <= L < lc["core_end"]:
            for c in ("attn", "k", "mlp", "v"):
                add(p + f"controllers.{c}.down_weight", "BF16", [lc["gate_rank"], 2 * H])
                add(p + f"controllers.{c}.forget_weight", "BF16", [H, lc["gate_rank"]])
                add(p + f"controllers.{c}.gain_weight", "BF16", [H, lc["gate_rank"]])
        e = p + "mlp.experts."
        if L < MXFP8_FROM:
            add(e + "w13_weight", "U8", [E, 2 * I, H // 2])
            add(e + "w13_weight_scale", "F8_E4M3", [E, 2 * I, H // 16])
            add(e + "w13_weight_amax", "F32", [E])
            add(e + "w13_weight_outer_scale_denominator", "F32", [1])
            add(e + "w2_weight", "U8", [E, H, I // 2])
            add(e + "w2_weight_scale", "F8_E4M3", [E, H, I // 16])
            add(e + "w2_weight_amax", "F32", [E])
            add(e + "w2_weight_outer_scale_denominator", "F32", [1])
        else:
            add(e + "w13_weight", "F8_E4M3", [E, 2 * I, H])
            add(e + "w13_weight_scale", "U8", [E, 2 * I, H // 32])
            add(e + "w2_weight", "F8_E4M3", [E, H, I])
            add(e + "w2_weight_scale", "U8", [E, H, I // 32])
    if vision_layers is not None:
        vh = cfg["vision_config"]["hidden_size"]
        add("model.visual.backbone.embeddings.class_embedding", "BF16", [vh])
        add("model.visual.proj_in.weight", "BF16", [vh, vh])
        add("model.visual.proj_out.weight", "BF16", [H, vh])
        for j in range(vision_layers):
            add(f"model.visual.backbone.layers.{j}.ln_1.weight", "BF16", [vh])
            add(f"model.visual.backbone.layers.{j}.mlp.c_fc.weight", "BF16", [vh * 2, vh])
    return out


def quant_config(sp: list) -> dict:
    """hf_quant_config.json of the fake folder: the real structure (parameter groups by compute type), tensor names from the specs"""
    groups = {"bf16": [], "f32": [], "nvfp4": [], "mxfp8": []}
    for name, dtype, _ in sp:
        if name.endswith(("w13_weight", "w2_weight")):
            groups["nvfp4" if dtype == "U8" else "mxfp8"].append(name)
        elif ".mlp.experts." in name:
            continue
        elif dtype == "F32":
            groups["f32"].append(name)
        else:
            groups["bf16"].append(name)
    pg = [{"compute_dtype": "float32", "tensors": sorted(groups["f32"]), "weight_transform": "cast"},
          {"compute_dtype": "bfloat16", "tensors": sorted(groups["bf16"]), "weight_transform": "cast"},
          {"compute_dtype": "nvfp4", "tensors": sorted(groups["nvfp4"]), "weight_block_shape": [1, 16], "weight_transform": "nvfp4_nearest"},
          {"compute_dtype": "mxfp8", "tensors": sorted(groups["mxfp8"]), "weight_block_shape": [32, 32], "weight_transform": "mxfp8_nearest"}]
    return {"model": {"parameter_groups": pg}, "mtp": {"parameter_groups": [{"compute_dtype": "bfloat16", "tensors": ["fuse_weight", "layers.0.dflare_norm.weight"]}]},
            "quant_method": "inkling2"}


def tensor_bytes(i: int, name: str, dtype: str, shape) -> bytes:
    n = stfile.numel(shape) * stfile.BYTES[dtype]
    if name.endswith("mlp.gate.bias") or name.endswith("_amax"):          # float32 values that are finite and distinct per entry
        import struct
        count = stfile.numel(shape)
        return b"".join(struct.pack("<f", 0.25 * (j + 1) + i) for j in range(count))
    return stfile.fill(i, n)


def write_folder(folder, cfg: dict | None = None, shards: int = 2, mtp: bool = True, vision_layers: int = 1) -> dict:
    """Writes config.json, hf_quant_config.json, a sharded model + index, side files and (unlisted in the index, as in the real repo) mtp.safetensors.
    Returns {"cfg", "specs", "tensors": {name: (dtype, shape, bytes)}}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(cfg if cfg is not None else tiny_config())
    sp = specs(cfg, vision_layers)
    data = {name: (dtype, shape, tensor_bytes(i, name, dtype, shape)) for i, (name, dtype, shape) in enumerate(sp)}
    names = list(data)
    per = -(-len(names) // shards)
    weight_map = {}
    for s in range(shards):
        part = names[s * per:(s + 1) * per]
        if not part:
            continue
        fname = f"model-{s + 1:05d}-of-{shards:05d}.safetensors"
        stfile.write_file(folder / fname, [(n, data[n][0], data[n][1], data[n][2]) for n in part])
        weight_map.update({n: fname for n in part})
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": sum(len(v[2]) for v in data.values())}, "weight_map": weight_map}), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (folder / "hf_quant_config.json").write_text(json.dumps(quant_config(sp), indent=2), encoding="utf-8")
    (folder / "generation_config.json").write_text(json.dumps({"do_sample": True, "temperature": 1.0}), encoding="utf-8")
    (folder / "processor_config.json").write_text(json.dumps({"processor_class": "Inkling2Processor"}), encoding="utf-8")
    (folder / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|content_model_end_sampling|>"}), encoding="utf-8")
    if mtp:
        H = cfg["text_config"]["hidden_size"]
        m = [("fuse_weight", "BF16", (H, H), stfile.fill(901, H * H * 2)), ("layers.0.dflare_norm.weight", "BF16", (H,), stfile.fill(902, H * 2))]
        stfile.write_file(folder / "mtp.safetensors", m)
    return {"cfg": cfg, "specs": sp, "tensors": data}
