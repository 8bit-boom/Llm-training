"""Mixture-of-experts layouts: which tensor is an expert, a router, a loop gate, and what a config says about them. Standard library only, nothing is loaded.

Two kinds of layout are known:

* `inkling2_loop` (thinkingmachines/Inkling-2-Mini): experts of a layer are FOUR fused tensors in the authors' packed format (NVFP4 in layers 0-19, MXFP8 in the
  rest, see docs/MOE_GUIDE.md), the first axis of every one of them is the expert. The model has 4 "attention-residual blocks" of 6 layers and runs the middle two
  blocks (layers 6-17) `num_passes` times (a loop), so a token visits 48 layer-slots although there are 24 layers.
* per-expert layouts (`qwen3_moe`, `qwen3_5_moe`, `mixtral`): one weight per expert and projection; the experts of a layer are renumbered when some are removed.

Every function here is pure. The cut itself (writing files) lives in `reshapejob.py`; the golden tests are `tests/test_moe_layout.py` and `tests/test_moe_config.py`.
"""
from __future__ import annotations

import copy
import math
import re

INKLING = "inkling2_loop"
INKLING_SMALL = "inkling2"          # the 310 GB model of the same family: not supported
MXFP8, NVFP4 = "mxfp8", "nvfp4"
COUNT_STEP = 16                      # expert counts are multiples of this (owner's decision D4)
MIN_EXPERTS = 32                     # ... and never below this
MAX_REMOVED = 0.5                    # ... and at most this share of the experts per stage
MIN_PER_TOPK = 4                     # never fewer experts than this many times the per-token count

INKLING_LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
_EXPERT_PER_KEY = re.compile(r"^(?P<pre>.*?\.layers\.)(?P<layer>\d+)\.(?P<mid>.*?experts\.)(?P<e>\d+)\.(?P<rest>.+)$")
_PER_EXPERT_LAYERS = re.compile(r"^(model\.language_model\.layers\.|model\.layers\.|language_model\.model\.layers\.)(\d+)\.(.+)$")


class MoeError(ValueError):
    """plain-language message for the user (the job turns it into a ReshapeError)"""


# ------------------------------------------------------------------------------------------------------------------ layouts

LAYOUTS = {
    INKLING: {"arch": INKLING, "fused": True, "packed": True, "count_key": "n_routed_experts", "topk_key": "num_experts_per_tok", "inter_key": "intermediate_size",
              "scoring": "sigmoid", "status": "supported"},
    "qwen3_moe": {"arch": "qwen3_moe", "fused": False, "packed": False, "count_key": "num_experts", "topk_key": "num_experts_per_tok", "inter_key": "moe_intermediate_size",
                  "scoring": "softmax", "status": "supported"},
    "qwen3_5_moe": {"arch": "qwen3_5_moe", "fused": False, "packed": False, "count_key": "num_experts", "topk_key": "num_experts_per_tok", "inter_key": "moe_intermediate_size",
                    "scoring": "softmax", "status": "supported"},
    "mixtral": {"arch": "mixtral", "fused": False, "packed": False, "count_key": "num_local_experts", "topk_key": "num_experts_per_tok", "inter_key": "intermediate_size",
                "scoring": "softmax", "status": "experimental"},
}
REFUSED_LAYOUTS = {
    "deepseek_v3": "DeepSeek-style routing picks experts in groups (n_group, topk_group); removing experts would break that grouping, so MoE cuts are not offered for it yet.",
    "gpt_oss": "gpt-oss stores its experts in a block-quantized form with biases; MoE cuts are not offered for it yet.",
    INKLING_SMALL: "Inkling 2 Small (310 GB) is not supported; only Inkling-2-Mini can be cut.",
}


def _mt(cfg: dict) -> str:
    mt = cfg.get("model_type")
    if isinstance(mt, str) and mt:
        return mt
    tc = cfg.get("text_config")
    mt = tc.get("model_type") if isinstance(tc, dict) else ""
    return mt if isinstance(mt, str) else ""


def text_cfg(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def arch_of(cfg: dict) -> str:
    """the layout key for this config ("" when the model is not a mixture of experts we know)"""
    mt = _mt(cfg)
    tmt = str(text_cfg(cfg).get("model_type") or "")
    if mt == INKLING or tmt.startswith(INKLING):
        return INKLING
    for key in LAYOUTS:
        if key != INKLING and (mt.startswith(key) or tmt.startswith(key)):
            return key
    return ""


def is_inkling(cfg: dict) -> bool:
    return arch_of(cfg) == INKLING


def refused_reason(cfg: dict) -> str | None:
    mt = _mt(cfg)
    for key, why in REFUSED_LAYOUTS.items():
        if mt == key or mt.startswith(key + "_"):
            return why
    return None


def has_moe_config(cfg: dict) -> bool:
    """a config that names experts at all (any known key), whether or not we know the layout"""
    tc = text_cfg(cfg)
    return any(tc.get(k) for k in ("n_routed_experts", "num_experts", "num_local_experts"))


# ------------------------------------------------------------------------------------------------------------------ the loop

def loop_of(cfg: dict) -> dict:
    """{"core": (start, end), "passes": n, "varloop": bool, "gate_rank": r} of a looped model; passes 1 and core (0, 0) when there is no loop"""
    lc = text_cfg(cfg).get("loop_config")
    if not isinstance(lc, dict):
        return {"core": (0, 0), "passes": 1, "varloop": False, "gate_rank": 0}
    return {"core": (int(lc.get("core_start", 0)), int(lc.get("core_end", 0))), "passes": int(lc.get("num_passes", 1)), "varloop": bool(lc.get("is_varloop", False)),
            "gate_rank": int(lc.get("gate_rank", 0))}


def visits(cfg: dict, passes: int | None = None) -> int:
    """layer slots a token runs through: the layers, plus the core once more for every pass after the first"""
    n = int(text_cfg(cfg).get("num_hidden_layers", 0))
    lp = loop_of(cfg)
    p = lp["passes"] if passes is None else passes
    a, b = lp["core"]
    return n + max(0, p - 1) * max(0, b - a)


def block_size_of(cfg: dict) -> int:
    return int(text_cfg(cfg).get("n_layers_per_attn_res_block") or 0)


def layer_kinds(cfg: dict) -> list[str]:
    """"sliding_attention" / "full_attention" per layer from layer_types, else from local_layer_ids (the layers NOT listed are the global ones)"""
    tc = text_cfg(cfg)
    n = int(tc["num_hidden_layers"])
    lt = tc.get("layer_types")
    if isinstance(lt, list) and len(lt) == n:
        return [str(x) for x in lt]
    local = tc.get("local_layer_ids")
    if isinstance(local, list):
        ls = {int(i) for i in local}
        return ["sliding_attention" if i in ls else "full_attention" for i in range(n)]
    return ["full_attention"] * n


def check_inkling_config(cfg: dict) -> None:
    """MoeError unless the config has the structure the cut code relies on (blocks of n_layers_per_attn_res_block, a core made of whole blocks)"""
    tc = text_cfg(cfg)
    n, bs = tc.get("num_hidden_layers"), block_size_of(cfg)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1 or bs < 1 or n % bs:
        raise MoeError("The model's config.json does not describe whole attention-residual blocks (num_hidden_layers / n_layers_per_attn_res_block).")
    lp = loop_of(cfg)
    a, b = lp["core"]
    if lp["passes"] < 1 or (b > a and (a % bs or b % bs or not 0 < a < b <= n)):
        raise MoeError("The loop of this model does not start and end on a block boundary; it cannot be cut safely.")
    for key in ("n_routed_experts", "num_experts_per_tok"):
        v = tc.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise MoeError(f"config.json has no usable {key}.")


# ------------------------------------------------------------------------------------------------------------------ names

def inkling_layer_of(name: str):
    """(layer index, rest) of a text decoder tensor "model.llm.layers.N.rest", else None"""
    m = INKLING_LAYER.match(name)
    return (int(m.group(1)), m.group(2)) if m else None


def classify(name: str, arch: str = INKLING) -> tuple:
    """(layer or None, role, expert index or None). Roles: embed, output, attn, router, router_bias, experts, expert_companion, expert_shared, loop_ctrl, loop_query,
    attn_res, conv, norm, vision, mtp, other. `expert_companion` has the expert on its first axis (scales, amax); `expert_shared` is one value for the whole layer."""
    if arch == INKLING:
        return _classify_inkling(name)
    return _classify_per_expert(name)


def _classify_inkling(name: str) -> tuple:
    if name.startswith("model.visual.") or ".visual." in name:
        return None, "vision", None
    if not name.startswith("model."):          # mtp.safetensors (the speculative-decoding drafter) has unprefixed names: "fuse_weight", "layers.0...."
        return None, "mtp", None
    lr = inkling_layer_of(name)
    if lr is None:
        if name.startswith("model.llm.embed"):
            return None, "embed" if name.endswith("embed.weight") else "norm", None
        if name.startswith("model.llm.unembed"):
            return None, "output", None
        if name.startswith("model.llm.attn_res."):
            return None, "attn_res", None
        if name.startswith("model.llm.middle_cycle_cross_pass"):
            return None, "loop_query", None
        if name == "model.llm.norm.weight":
            return None, "norm", None
        return None, "other", None
    layer, rest = lr
    if rest.startswith("mlp.experts."):
        leaf = rest[len("mlp.experts."):]
        if leaf in ("w13_weight", "w2_weight"):
            return layer, "experts", None
        if leaf.endswith("_outer_scale_denominator"):
            return layer, "expert_shared", None
        return layer, "expert_companion", None
    if rest == "mlp.gate.weight":
        return layer, "router", None
    if rest == "mlp.gate.bias":
        return layer, "router_bias", None
    if rest.startswith("controllers."):
        return layer, "loop_ctrl", None
    if rest.startswith("attn."):
        return layer, "conv" if "sconv" in rest else "attn", None
    if "sconv" in rest:
        return layer, "conv", None
    if rest.endswith("norm.weight"):
        return layer, "norm", None
    return layer, "other", None


def _classify_per_expert(name: str) -> tuple:
    if "visual." in name or "vision" in name:
        return None, "vision", None
    if name.startswith("mtp."):
        return None, "mtp", None
    m = _PER_EXPERT_LAYERS.match(name)
    if m is None:
        if "embed_tokens" in name:
            return None, "embed", None
        if name.startswith("lm_head"):
            return None, "output", None
        return None, "other", None
    layer, rest = int(m.group(2)), m.group(3)
    e = re.match(r"^(?:mlp\.experts|block_sparse_moe\.experts)\.(\d+)\.", rest)
    if e:
        return layer, "experts", int(e.group(1))
    if rest in ("mlp.gate.weight", "block_sparse_moe.gate.weight"):
        return layer, "router", None
    if rest.startswith("mlp.experts.") or rest.startswith("mlp.shared_expert"):
        return layer, "expert_shared", None            # fused v5-style experts or shared experts: untouched by an expert cut
    if rest.startswith(("self_attn.", "linear_attn.")):
        return layer, "attn", None
    if rest.endswith("norm.weight"):
        return layer, "norm", None
    return layer, "other", None


# ------------------------------------------------------------------------------------------------------------------ sizes

def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def logical_params(name: str, dtype: str, shape) -> int:
    """parameters the tensor stands for: a packed NVFP4 byte holds two weights; scales, amax and denominators are not parameters"""
    layer_role = _classify_inkling(name)[1]
    n = numel(shape)
    if layer_role == "experts":
        return n * 2 if dtype == "U8" else n
    if layer_role in ("expert_companion", "expert_shared"):
        return 0
    return n


def packed_scheme(dtype: str) -> str | None:
    """"nvfp4" for a U8 expert weight, "mxfp8" for an F8_E4M3 one"""
    return {"U8": NVFP4, "F8_E4M3": MXFP8}.get(dtype)


def group_params_moe(cfg: dict, tensors: dict, n_layers: int) -> dict:
    """tensors: {name: (dtype, shape)} of an Inkling folder. Same keys as `reshape.group_params` (so the page and the planner read it unchanged) plus
    "experts" and "dense" per layer, "expert_total", "packed" (scheme per layer) and "loop" (parameters of the loop controllers and queries).
    layers[i] counts the layer's dense tensors and its experts; "embed"/"lm_head" are the token table and the (padded) output table; attention-residual queries
    and loop queries go to "other"."""
    layers, experts, dense = [0] * n_layers, [0] * n_layers, [0] * n_layers
    g = {"embed": 0, "lm_head": 0, "vision": 0, "mtp": 0, "other": 0}
    packed: dict[int, str] = {}
    loop = 0
    for name, (dtype, shape) in tensors.items():
        n = logical_params(name, dtype, shape)
        layer, role, _ = classify(name, INKLING)
        if role == "mtp":
            g["mtp"] += numel(shape)
        elif role == "vision":
            g["vision"] += numel(shape)
        elif layer is not None:
            if layer >= n_layers:
                raise MoeError(f"The weights contain layer {layer} but config.json says the model has {n_layers} layers.")
            layers[layer] += n
            if role == "experts":
                experts[layer] += n
                packed[layer] = packed_scheme(dtype) or "other"
            else:
                dense[layer] += n
                if role == "loop_ctrl":
                    loop += n
        elif role == "embed":
            g["embed"] += n
        elif role == "output":
            g["lm_head"] += n
        else:
            g["other"] += n
            if role == "loop_query":
                loop += n
    total = sum(layers) + sum(g.values())
    return {"layers": layers, **g, "total": total, "bytes_per_param": 2.0, "experts": experts, "dense": dense, "expert_total": sum(experts),
            "packed": [packed.get(i) for i in range(n_layers)], "loop": loop}


def experts_of(cfg: dict) -> int:
    return int(text_cfg(cfg).get(LAYOUTS[arch_of(cfg)]["count_key"], 0))


def top_k_of(cfg: dict) -> int:
    return int(text_cfg(cfg).get(LAYOUTS[arch_of(cfg)]["topk_key"], 0))


def active_params(groups: dict, cfg: dict, experts: int | None = None, top_k: int | None = None, passes: int | None = None) -> int:
    """parameters used for one token: every dense tensor and the chosen share of the experts of each layer SLOT, summed over the visits, plus the output table
    (the token table is a lookup and is not counted). A core layer counts once per pass."""
    e = experts_of(cfg) if experts is None else experts
    e0 = experts_of(cfg)
    k = top_k_of(cfg) if top_k is None else top_k
    lp = loop_of(cfg)
    p = lp["passes"] if passes is None else passes
    a, b = lp["core"]
    total = groups["lm_head"]
    for i, (d, x) in enumerate(zip(groups["dense"], groups["experts"])):
        times = p if a <= i < b else 1
        total += times * (d + x * k // max(1, e0))
    return total


def blocks_removed_ok(cfg: dict, drop_blocks: list[int]) -> str | None:
    """None when the whole blocks may go, else the plain reason. Only the middle (loop) blocks and the last block can be dropped; never the first"""
    bs, n = block_size_of(cfg), int(text_cfg(cfg)["num_hidden_layers"])
    nb = n // bs
    if not drop_blocks:
        return "No block was chosen."
    if any(not 0 <= b < nb for b in drop_blocks):
        return f"Blocks are numbered 0 to {nb - 1}."
    if 0 in drop_blocks:
        return "The first block cannot be removed: the loop has to start after it."
    if len(set(drop_blocks)) > nb // 2:
        return "Remove at most half of the blocks."
    return None


# ------------------------------------------------------------------------------------------------------------------ the config surgery

def expert_counts_allowed(experts: int, top_k: int, stage_floor: float = 1 - MAX_REMOVED) -> list[int]:
    """kept-expert counts the planner may offer, largest first: multiples of 16, not below 32 and not below 4 x top-k, at most half removed in one stage"""
    floor = max(MIN_EXPERTS, MIN_PER_TOPK * top_k, math.ceil(experts * stage_floor))
    floor = -(-floor // COUNT_STEP) * COUNT_STEP
    return [n for n in range(experts - COUNT_STEP, floor - 1, -COUNT_STEP) if n >= 1]


def check_experts_kept(experts: int, top_k: int, keep: int, stage_floor: float = 1 - MAX_REMOVED) -> None:
    """MoeError for a kept-expert count the rules forbid (owner's decision D4)"""
    if keep >= experts:
        raise MoeError("The kept number of experts must be smaller than the number the model has.")
    if keep % COUNT_STEP:
        raise MoeError(f"The number of experts to keep must be a multiple of {COUNT_STEP}.")
    if keep < MIN_EXPERTS:
        raise MoeError(f"Keep at least {MIN_EXPERTS} experts per layer.")
    if keep < MIN_PER_TOPK * top_k:
        raise MoeError(f"Keep at least {MIN_PER_TOPK} times the experts used per token ({MIN_PER_TOPK * top_k}).")
    if keep < experts * stage_floor:
        raise MoeError(f"Remove at most {round(MAX_REMOVED * 100)} % of the experts in one stage; cut again from the result for more.")


def check_passes(cfg: dict, passes: int) -> None:
    lp = loop_of(cfg)
    if lp["core"][1] <= lp["core"][0]:
        raise MoeError("This model has no loop; there are no passes to change.")
    if passes < 1:
        raise MoeError("At least one pass is needed.")
    if passes < 2 and not lp["varloop"]:
        raise MoeError("This model was built for a fixed loop and the GGUF converter refuses fewer than 2 passes; "
                       "use the variable-loop research checkpoint (thinkingmachines/TML-d24-varloop-research) for 1 pass.")
    if passes > 8:
        raise MoeError("More than 8 passes is not offered.")


def check_top_k(cfg: dict, top_k: int) -> None:
    e = experts_of(cfg)
    if top_k < 1 or top_k > e:
        raise MoeError(f"Experts per token must be between 1 and {e}.")
    if top_k > 4 * top_k_of(cfg):
        raise MoeError(f"Experts per token above {4 * top_k_of(cfg)} is not offered.")


def fix_config_inkling(cfg: dict, passes: int | None = None, top_k: int | None = None, experts: int | None = None, drop_blocks: list[int] | None = None,
                       notes: list | None = None) -> dict:
    """Deep copy of config.json with the changed fields (see docs/MOE_GUIDE.md "what must stay consistent"). Passes/top-k/experts touch one field each; dropping
    blocks renumbers num_hidden_layers, local_layer_ids and the loop core, and removes mtp_config (its target layers are visit numbers that no longer exist).
    Changing the passes also removes mtp_config. Everything else, vision_config, quantization_config and the top-level keys, is left alone."""
    check_inkling_config(cfg)
    new = copy.deepcopy(cfg)
    tc = text_cfg(new)
    drop_mtp = False
    if passes is not None and passes != loop_of(cfg)["passes"]:
        check_passes(cfg, passes)
        tc["loop_config"]["num_passes"] = int(passes)
        drop_mtp = True
    if top_k is not None and top_k != top_k_of(cfg):
        check_top_k(cfg, top_k)
        tc["num_experts_per_tok"] = int(top_k)
    if experts is not None and experts != experts_of(cfg):
        check_experts_kept(experts_of(cfg), top_k_of(new), experts)
        tc["n_routed_experts"] = int(experts)
    if drop_blocks:
        why = blocks_removed_ok(cfg, drop_blocks)
        if why:
            raise MoeError(why)
        bs, n = block_size_of(cfg), int(tc["num_hidden_layers"])
        gone = {b * bs + i for b in drop_blocks for i in range(bs)}
        keep = [i for i in range(n) if i not in gone]
        a, b = loop_of(cfg)["core"]
        if b > a:
            kept_core = [i for i in keep if a <= i < b]
            if not kept_core:
                raise MoeError("That would remove the whole loop.")
            new_a = keep.index(kept_core[0])
            new_b = new_a + len(kept_core)
            if any(i not in kept_core and a <= i < b for i in keep) or kept_core != list(range(kept_core[0], kept_core[-1] + 1)):
                raise MoeError("The loop must stay one piece.")
            tc["loop_config"]["core_start"], tc["loop_config"]["core_end"] = new_a, new_b
        tc["num_hidden_layers"] = len(keep)
        if isinstance(tc.get("local_layer_ids"), list):
            old = set(tc["local_layer_ids"])
            tc["local_layer_ids"] = [j for j, s in enumerate(keep) if s in old]
        if isinstance(tc.get("layer_types"), list):
            tc["layer_types"] = [tc["layer_types"][s] for s in keep]
        drop_mtp = True
        if notes is not None:
            notes.append("config.json: num_hidden_layers, local_layer_ids and the loop core were renumbered for the removed blocks.")
    if drop_mtp and "mtp_config" in new:
        del new["mtp_config"]
        if notes is not None:
            notes.append("The speculative-decoding (multi-token) head was removed: its target layers no longer exist after this change.")
    return new


# ------------------------------------------------------------------------------------------------------------------ describe

def size_table(groups: dict, cfg: dict, counts: list[int] | None = None) -> list[dict]:
    """logical parameters and packed bytes for each kept-expert count (every layer keeps the same number); packed bytes assume the authors' format
    (4 bits + a block scale per 16 for NVFP4 layers, 8 bits + a block scale per 32 for MXFP8 layers) and are an estimate to a few percent."""
    e0 = experts_of(cfg)
    rows = []
    for n in counts if counts is not None else [e0] + expert_counts_allowed(e0, top_k_of(cfg)):
        ex = groups["expert_total"] * n // e0
        packed = 0
        for i, scheme in enumerate(groups["packed"]):
            per = groups["experts"][i] * n // e0
            packed += per * (0.5 + 1 / 16 if scheme == NVFP4 else 1 + 1 / 32 if scheme == MXFP8 else 2)
        dense = groups["total"] - groups["expert_total"] - groups["vision"] - groups["mtp"]
        rows.append({"experts": n, "params": dense + ex + groups["vision"], "packed_bytes": int(packed + (dense + groups["vision"]) * 2),
                     "bf16_bytes": (dense + ex + groups["vision"]) * 2})
    return rows


def describe(cfg: dict, groups: dict, tensors: dict, source_bytes: int = 0) -> dict:
    """The "moe" block of the inspect result (see docs/MOE_GUIDE.md). `groups` comes from `group_params_moe`."""
    tc = text_cfg(cfg)
    arch = arch_of(cfg)
    e, k = experts_of(cfg), top_k_of(cfg)
    lp = loop_of(cfg)
    types = layer_kinds(cfg)
    bs = block_size_of(cfg)
    n = int(tc["num_hidden_layers"])
    packed = groups.get("packed") or []
    ranges = {}
    for scheme in (NVFP4, MXFP8):
        idx = [i for i, s in enumerate(packed) if s == scheme]
        if idx:
            ranges[scheme] = [idx[0], idx[-1]]
    has_mtp_file = any(classify(name, INKLING)[1] == "mtp" for name in tensors) if arch == INKLING else False
    out = {"arch": arch, "experts": e, "top_k": k, "expert_inter": int(tc.get(LAYOUTS[arch]["inter_key"], 0)), "shared": int(tc.get("n_shared_experts") or 0),
           "router_scoring": LAYOUTS[arch]["scoring"], "router_bias": bool(tc.get("use_gate_bias", False)),
           "blocks": n // bs if bs else 0, "block_size": bs, "core": list(lp["core"]), "passes": lp["passes"], "varloop": lp["varloop"], "visits": visits(cfg),
           "global_layers": [i for i, t in enumerate(types) if t == "full_attention"],
           "expert_params": groups.get("expert_total", 0), "dense_params": groups["total"] - groups.get("expert_total", 0) - groups["vision"] - groups["mtp"],
           "active_params": active_params(groups, cfg) if "experts" in groups else 0, "packed": ranges, "packed_bytes": int(source_bytes),
           "bf16_bytes": groups["total"] * 2, "mtp": {"file": "mtp.safetensors" if has_mtp_file else None, "targets": list((cfg.get("mtp_config") or {}).get("target_layer_ids") or [])},
           "cuts": {"experts": True, "top_k": True, "passes": "untrained" if lp["core"][1] > lp["core"][0] else False, "blocks": "expert-only" if bs else False,
                    "expert_width": "experimental"},
           "status": LAYOUTS[arch]["status"], "sizes": size_table(groups, cfg) if "experts" in groups else []}
    return out


def describe_per_expert(cfg: dict, tensors: dict, groups: dict) -> dict:
    """The "moe" block for a per-expert layout (qwen3_moe, qwen3_5_moe, mixtral). `groups` comes from `reshape.group_params`. Experts per layer are counted from the tensor
    names; a checkpoint that stores them fused (transformers v5 style) has none, and then only the config's numbers are reported."""
    arch = arch_of(cfg)
    n = int(text_cfg(cfg)["num_hidden_layers"])
    per_layer = [set() for _ in range(n)]
    expert_params = 0
    for name, (dtype, shape) in tensors.items():
        layer, role, e = classify(name, arch)
        if role == "experts" and layer is not None and layer < n and e is not None:
            per_layer[layer].add(e)
            expert_params += numel(shape)
    counts = sorted({len(s) for s in per_layer if s})
    tc = text_cfg(cfg)
    return {"arch": arch, "experts": experts_of(cfg), "top_k": top_k_of(cfg), "expert_inter": int(tc.get(LAYOUTS[arch]["inter_key"], 0)), "shared": int(tc.get("n_shared_experts") or (1 if tc.get("shared_expert_intermediate_size") else 0)),
            "router_scoring": LAYOUTS[arch]["scoring"], "router_bias": False, "blocks": 0, "block_size": 0, "core": [0, 0], "passes": 1, "varloop": False, "visits": n,
            "global_layers": [], "expert_params": expert_params, "dense_params": max(0, groups["total"] - expert_params - groups["vision"] - groups["mtp"]),
            "active_params": 0, "packed": {}, "packed_bytes": 0, "bf16_bytes": groups["total"] * 2, "mtp": {"file": None, "targets": []},
            "per_layer_experts": counts, "stored_per_expert": bool(counts), "status": LAYOUTS[arch]["status"], "sizes": [],
            "cuts": {"experts": bool(counts), "top_k": True, "passes": False, "blocks": False, "expert_width": False}}
