"""Pure helpers for width pruning in "Slim or grow a model" (version 2): make every layer of a Qwen3.5-style model thinner. Standard library only.

The MLP of every layer loses neurons (`intermediate_size`); in Expert mode the linear-attention (GatedDeltaNet) layers can lose value heads and the
full-attention layers query heads. Which structures stay is decided from activation scores measured on the project's own text (reshapejob.py); this module
holds everything that needs no weights: the dimensions, the allowed sizes, the exact parameter count, picking the structures to keep, which rows and
columns of which tensor belong to a structure (docs/RESHAPE.md, "Width pruning"), and the config rewrite.

Method: Minitron (Muralidharan et al. 2024, arXiv 2407.14679; Sreenivas et al. 2024, arXiv 2408.11796; Minitron-SSM, arXiv 2504.11409) and NVIDIA's
Model-Optimizer reference implementation: importance of a structure = L2 over the samples of the mean absolute activation that feeds the output
projection of its block. The count of kept structures is the same in every layer (transformers and llama.cpp build all layers from one config value).
"""
from __future__ import annotations

import copy
import math
import re

from .reshape import ReshapeError, split_key, text_config

FFN_DIVISOR = 256            # k-quants in GGUF work on rows of 256 values; NVIDIA's tool uses 512 by default
MAX_SHARE_REMOVED = 0.5      # at most half of the MLP neurons
WIDTH_KEYS = ("intermediate_size", "linear_num_value_heads", "num_attention_heads")
SHORT = {"intermediate_size": "intermediate", "linear_num_value_heads": "lin_v_heads", "num_attention_heads": "heads"}
MOE_KEYS = ("num_experts", "moe_intermediate_size", "num_experts_per_tok", "shared_expert_intermediate_size", "decoder_sparse_step")
NOT_DENSE = ("Thinning the layers is only available for dense Qwen3.5 / Qwen3.8 models so far. You can still remove whole layers.")

# tensors of a decoder layer (suffix after "layers.N.") that width pruning never changes
UNCHANGED_SUFFIXES = {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                      "self_attn.q_norm.weight", "self_attn.k_norm.weight", "linear_attn.norm.weight"}
MLP_SUFFIXES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
ATTN_SUFFIXES = ("self_attn.q_proj.weight", "self_attn.o_proj.weight")
GDN_SUFFIXES = ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.out_proj.weight")
MTP_KEY = re.compile(r"^mtp\.layers\.(\d+)\.(.+)$")


# ----------------------------------------------------------------------------------------------------------------- dims

def _int(tc: dict, key: str, default=None) -> int:
    v = tc.get(key, default)
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise ReshapeError(f"config.json has no usable {key}.")
    return v


def width_dims(cfg: dict) -> dict:
    """The sizes width pruning works on. ReshapeError unless the config is a Qwen3.5 text model (or a Qwen3.5 model with a vision part)."""
    tc = text_config(cfg)
    mt = str(tc.get("model_type") or cfg.get("model_type") or "")
    if mt not in ("qwen3_5", "qwen3_5_text"):
        raise ReshapeError(NOT_DENSE)
    hidden, heads = _int(tc, "hidden_size"), _int(tc, "num_attention_heads")
    if heads < 1:
        raise ReshapeError("config.json has no usable num_attention_heads.")
    head_dim = tc.get("head_dim") or hidden // heads
    tied = tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    return {"hidden": hidden, "intermediate": _int(tc, "intermediate_size"), "heads": heads, "kv_heads": _int(tc, "num_key_value_heads"),
            "head_dim": int(head_dim), "lin_k_heads": _int(tc, "linear_num_key_heads"), "lin_v_heads": _int(tc, "linear_num_value_heads"),
            "lin_k_dim": _int(tc, "linear_key_head_dim"), "lin_v_dim": _int(tc, "linear_value_head_dim"), "conv_kernel": _int(tc, "linear_conv_kernel_dim"),
            "mtp_layers": _int(tc, "mtp_num_hidden_layers", 0), "vocab": _int(tc, "vocab_size"), "tied": bool(tied)}


def width_supported(cfg: dict) -> str | None:
    """None when width pruning is possible, else the plain reason."""
    from .reshape import check_supported
    try:
        check_supported(cfg)
        tc = text_config(cfg)
        mt = str(tc.get("model_type") or cfg.get("model_type") or "")
        if mt not in ("qwen3_5", "qwen3_5_text") or any(k in tc for k in MOE_KEYS):
            return NOT_DENSE
        if tc.get("mlp_only_layers"):
            return "Some layers of this model have a different MLP (mlp_only_layers); thinning the MLP of every layer is not possible."
        d = width_dims(cfg)
    except ReshapeError as exc:
        return str(exc)
    if d["lin_k_dim"] != d["lin_v_dim"]:
        return "The linear-attention key and value sizes of this model differ; the GGUF export needs them equal, so the layers cannot be thinned."
    if d["kv_heads"] < 1 or d["heads"] % d["kv_heads"] or d["lin_k_heads"] < 1 or d["lin_v_heads"] % d["lin_k_heads"]:
        return "The attention head counts of this model do not divide evenly into groups; the layers cannot be thinned."
    return None


def ffn_choices(intermediate: int, divisor: int = FFN_DIVISOR) -> list[int]:
    """Multiples of `divisor` from half of the MLP to the whole of it, ascending."""
    lo = -(-intermediate // 2)
    first = -(-lo // divisor) * divisor
    return list(range(first, intermediate + 1, divisor))


def width_choices(dims: dict, divisor: int = FFN_DIVISOR) -> dict:
    """The sizes the page may offer. "intermediate": ascending; "lin_v_heads" and "heads": descending, the original first (one step only for value heads)."""
    nv, nk, nh, kv = dims["lin_v_heads"], dims["lin_k_heads"], dims["heads"], dims["kv_heads"]
    v = [nv, nv - nk] if nv - nk >= nk and (nv - nk) % nk == 0 else [nv]
    h = [x for x in range(nh, math.ceil(nh * 2 / 3) - 1, -kv) if x % kv == 0]
    return {"intermediate": ffn_choices(dims["intermediate"], divisor), "lin_v_heads": v, "heads": h}


def check_width(dims: dict, width: dict, divisor: int = FFN_DIVISOR) -> dict:
    """{"intermediate", "lin_v_heads", "heads"} with None filled from the model; accepts the short names and the config.json names.
    ReshapeError with a plain text when a value is not one of width_choices, or when nothing would change."""
    given = {SHORT.get(k, k): v for k, v in (width or {}).items()}
    for k in given:
        if k not in ("intermediate", "lin_v_heads", "heads"):
            raise ReshapeError(f"Unknown width setting {k!r}.")
    ch = width_choices(dims, divisor)
    out = {}
    for k in ("intermediate", "lin_v_heads", "heads"):
        v = given.get(k)
        out[k] = dims[k] if v is None else v
        if not isinstance(out[k], int) or isinstance(out[k], bool):
            raise ReshapeError(f"The width setting {k!r} must be a whole number.")
    I = dims["intermediate"]
    if out["intermediate"] != I and out["intermediate"] not in ch["intermediate"]:
        raise ReshapeError(f"The MLP size must be a multiple of {divisor} between half and all of {I:,}.")
    if out["lin_v_heads"] not in ch["lin_v_heads"]:
        raise ReshapeError(f"Linear-attention value heads can only go from {dims['lin_v_heads']} to {ch['lin_v_heads'][-1]}."
                           if len(ch["lin_v_heads"]) > 1 else "This model's linear-attention value heads cannot be reduced.")
    if out["heads"] not in ch["heads"]:
        raise ReshapeError(f"Attention heads must stay a multiple of {dims['kv_heads']} and at least {ch['heads'][-1]}.")
    if all(out[k] == dims[k] for k in out):
        raise ReshapeError("The chosen width is the same as the original model's; there is nothing to thin.")
    return out


def params_for(dims: dict, layer_types: list[str], width: dict, vision: int, other: int = 0) -> int:
    """Exact parameter count of a Qwen3.5 model with the given layer types and widths (layers + embeddings + output layer + final norm + multi-token head
    + `vision` + `other`). Reproduces the Hugging Face totals of Qwen3.5-27B and -9B."""
    H, hd, kv = dims["hidden"], dims["head_dim"], dims["kv_heads"]
    I, nh = width["intermediate"], width["heads"]
    nk, nv, dk, dv = dims["lin_k_heads"], width["lin_v_heads"], dims["lin_k_dim"], dims["lin_v_dim"]
    mlp = 3 * H * I
    norms = 2 * H
    conv = 2 * nk * dk + nv * dv
    gdn = H * conv + H * nv * dv + 2 * H * nv + conv * dims["conv_kernel"] + 2 * nv + dv + nv * dv * H
    att = H * nh * hd * 2 + 2 * H * kv * hd + nh * hd * H + 2 * hd
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_lin = len(layer_types) - n_full
    layers = n_lin * (gdn + mlp + norms) + n_full * (att + mlp + norms)
    mtp = dims["mtp_layers"] * (2 * H * H + att + mlp + norms + 3 * H)
    head = 0 if dims["tied"] else dims["vocab"] * H
    return layers + dims["vocab"] * H + head + H + mtp + vision + other


# --------------------------------------------------------------------------------------------------------- choosing what stays

def head_rows(h: int, size: int) -> list[int]:
    return list(range(h * size, (h + 1) * size))


def keep_top(scores: list[float], k: int) -> list[int]:
    """Sorted indices of the k largest scores (a tie is won by the lower index; NaN counts as the smallest)."""
    n = len(scores)
    if not 0 < k <= n:
        raise ReshapeError(f"Cannot keep {k} of {n}.")
    key = lambda i: (0.0 if scores[i] == scores[i] else 1.0, -scores[i] if scores[i] == scores[i] else 0.0, i)  # noqa: E731
    return sorted(sorted(range(n), key=key)[:k])


def keep_grouped(scores: list[float], groups: int, k: int) -> list[int]:
    """Group-aware: the same number k/groups of the highest scores inside each of the `groups` equal consecutive groups, sorted."""
    n = len(scores)
    if groups < 1 or n % groups or k % groups or not 0 < k <= n:
        raise ReshapeError(f"Cannot keep {k} of {n} in {groups} equal groups.")
    per, take = n // groups, k // groups
    out: list[int] = []
    for g in range(groups):
        out += [g * per + i for i in keep_top(scores[g * per:(g + 1) * per], take)]
    return out


def overlap(a: list[int], b: list[int]) -> float:
    """Jaccard overlap of two index lists (1.0 for two empty ones)."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


# ------------------------------------------------------------------------------------------------------- tensors and slices

def slices_for_layer(layer_type: str, dims: dict, width: dict, keep: dict) -> dict:
    """{suffix: (axis, kept indices)} for the tensors of one decoder layer, only for the dimensions that change (everything else is a byte copy).
    keep: {"mlp": kept neuron indices, "heads": kept query-head indices (full attention), "gdn": kept value-head indices (linear attention)}.
    Rows of the Qwen3.5 tensors (docs/RESHAPE.md): MLP neuron i = row i of gate/up and column i of down; q_proj holds [query | gate] (2*head_dim rows) per head
    and o_proj head_dim columns per head; in_proj_qkv / conv1d rows are [q (nk*dk) | k (nk*dk) | v (nv*dv)]; z has dv rows per value head; b and a, A_log,
    dt_bias one entry per value head; out_proj has dv columns per value head."""
    out: dict = {}
    if width["intermediate"] != dims["intermediate"]:
        idx = list(keep["mlp"])
        out.update({"mlp.gate_proj.weight": (0, idx), "mlp.up_proj.weight": (0, idx), "mlp.down_proj.weight": (1, idx)})
    if layer_type == "full_attention" and width["heads"] != dims["heads"]:
        h, hd = list(keep["heads"]), dims["head_dim"]
        out["self_attn.q_proj.weight"] = (0, [r for x in h for r in head_rows(x, 2 * hd)])
        out["self_attn.o_proj.weight"] = (1, [r for x in h for r in head_rows(x, hd)])
    if layer_type == "linear_attention" and width["lin_v_heads"] != dims["lin_v_heads"]:
        v, dv = list(keep["gdn"]), dims["lin_v_dim"]
        vrows = [r for x in v for r in head_rows(x, dv)]
        qkv = list(range(2 * dims["lin_k_heads"] * dims["lin_k_dim"])) + [2 * dims["lin_k_heads"] * dims["lin_k_dim"] + r for r in vrows]
        out.update({"linear_attn.in_proj_qkv.weight": (0, qkv), "linear_attn.conv1d.weight": (0, qkv), "linear_attn.in_proj_z.weight": (0, vrows),
                    "linear_attn.in_proj_b.weight": (0, v), "linear_attn.in_proj_a.weight": (0, v), "linear_attn.A_log": (0, v), "linear_attn.dt_bias": (0, v),
                    "linear_attn.out_proj.weight": (1, vrows)})
    return out


def check_keep(dims: dict, width: dict, keep: dict) -> None:
    """The kept index lists must have the promised sizes, be sorted and in range, and (heads) equal per group."""
    def ok(v, n, count, groups=1):
        v = list(v)
        return len(v) == count and v == sorted(set(v)) and all(0 <= x < n for x in v) and all(
            sum(1 for x in v if g * (n // groups) <= x < (g + 1) * (n // groups)) == count // groups for g in range(groups))
    for part, n, count, groups in (("mlp", dims["intermediate"], width["intermediate"], 1), ("heads", dims["heads"], width["heads"], dims["kv_heads"]),
                                   ("gdn", dims["lin_v_heads"], width["lin_v_heads"], dims["lin_k_heads"])):
        for layer, v in (keep.get(part) or {}).items():
            if not ok(v, n, count, groups):
                raise ReshapeError(f"Internal error: the kept {part} of layer {layer} do not fit the chosen size.")


def _changed_dims(dims: dict, width: dict) -> set:
    bad: set = set()
    if width["intermediate"] != dims["intermediate"]:
        bad.add(dims["intermediate"])
    if width["heads"] != dims["heads"]:
        bad |= {dims["heads"] * dims["head_dim"], 2 * dims["heads"] * dims["head_dim"]}
    if width["lin_v_heads"] != dims["lin_v_heads"]:
        bad |= {dims["lin_v_heads"], dims["lin_v_heads"] * dims["lin_v_dim"],
                2 * dims["lin_k_heads"] * dims["lin_k_dim"] + dims["lin_v_heads"] * dims["lin_v_dim"]}
    return bad


def _sliced(t, sl: dict, suffix: str, changed: set, what: str):
    from .safetensors_io import OutTensor
    if suffix in sl:
        axis, idx = sl[suffix]
        if t.source is None:
            raise ReshapeError(f"{what}: a layer that is added as an invisible copy cannot be thinned.")
        if len(t.shape) <= axis:
            raise ReshapeError(f"Tensor {t.name!r} has the shape {tuple(t.shape)}, which does not fit the layout of a Qwen3.5 layer.")
        shape = list(t.shape)
        if idx and max(idx) >= shape[axis]:
            raise ReshapeError(f"Tensor {t.name!r} is smaller than the model's config.json says; its weights do not match.")
        shape[axis] = len(idx)
        return OutTensor(t.name, t.dtype, tuple(shape), t.source, (axis, tuple(idx)))
    if suffix not in UNCHANGED_SUFFIXES and suffix not in MLP_SUFFIXES + ATTN_SUFFIXES + GDN_SUFFIXES and changed & set(t.shape):
        raise ReshapeError(f"Tensor {t.name!r} has a size that width pruning changes but no rule for it; nothing was written.")
    return t


def plan_width(out: list, src: list[int], types: list[str], dims: dict, width: dict, keep: dict, mtp_keep: dict | None) -> list:
    """Takes the output of reshape.plan_tensors (renamed layer tensors) and returns the same list with every tensor that width pruning changes replaced
    by one that carries `slice` and the new shape. keep: {"mlp": {old layer: indices}, "heads": {...}, "gdn": {...}} (see slices_for_layer); the layer j of the
    new model is old layer src[j]. mtp_keep: the same for the multi-token head ({"mlp": {0: ...}, "heads": {0: ...}}), None to leave it unchanged.
    Vision, embeddings, norms and everything else stay byte copies."""
    check_keep(dims, width, keep)
    if mtp_keep:
        check_keep(dims, width, mtp_keep)
    changed = _changed_dims(dims, width)
    result = []
    for t in out:
        sp = split_key(t.name)
        if sp is not None:
            j, suffix = sp[1], sp[2]
            if j >= len(src):
                raise ReshapeError(f"Internal error: layer {j} is not in the layer list.")
            s = src[j]
            lt = types[s]
            sl = slices_for_layer(lt, dims, width, {k: (keep.get(k) or {}).get(s) for k in ("mlp", "heads", "gdn")}) if _have(keep, s, lt, dims, width) else {}
            result.append(_sliced(t, sl, suffix, changed, f"layer {s}"))
            continue
        m = MTP_KEY.match(t.name)
        if m and mtp_keep is not None:
            n = int(m.group(1))
            sl = slices_for_layer("full_attention", dims, width, {k: (mtp_keep.get(k) or {}).get(n) for k in ("mlp", "heads")})
            result.append(_sliced(t, sl, m.group(2), changed, "the multi-token head"))
            continue
        result.append(t)
    return result


def _have(keep: dict, s: int, lt: str, dims: dict, width: dict) -> bool:
    """True when layer s needs slices; ReshapeError when a needed kept-list is missing."""
    need = []
    if width["intermediate"] != dims["intermediate"]:
        need.append("mlp")
    if lt == "full_attention" and width["heads"] != dims["heads"]:
        need.append("heads")
    if lt == "linear_attention" and width["lin_v_heads"] != dims["lin_v_heads"]:
        need.append("gdn")
    for part in need:
        if (keep.get(part) or {}).get(s) is None:
            raise ReshapeError(f"Internal error: no kept {part} are known for layer {s}.")
    return bool(need)


def fix_config_width(cfg: dict, dims: dict, width: dict) -> dict:
    """Deep copy of config.json with the new widths in text_config. head_dim is written explicitly (transformers would otherwise derive it from
    hidden_size // num_attention_heads, which changes with the heads). Refuses MoE and mixed-MLP configs."""
    new = copy.deepcopy(cfg)
    tc = text_config(new)
    why = width_supported(cfg)
    if why:
        raise ReshapeError(why)
    for key, short in SHORT.items():
        if key not in tc:
            raise ReshapeError(f"config.json has no {key}.")
        if width[short] != dims[short]:
            tc[key] = int(width[short])
    tc["head_dim"] = int(dims["head_dim"])
    return new
