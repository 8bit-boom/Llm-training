"""Pure helpers for "Slim or grow a model": layer math, choosing blocks, and rewriting config.json. Standard library only.

A decoder-only model is a stack of layers 0..L-1. Qwen3.5 repeats a pattern of 4 layers (3 linear-attention + 1 full attention), so layers are cut and
copied only in whole "blocks" of one pattern period; the rewritten model then still follows the same pattern rule (old and new transformers / llama.cpp agree).

Methods: block-boundary angular distance (Gromov et al. 2024, "The Unreasonable Ineffectiveness of the Deeper Layers"), Block Influence (ShortGPT),
depth up-scaling (SOLAR 10.7B / mergekit "passthrough"), identity initialisation of new layers (LLaMA Pro). See docs/RESHAPE.md.
"""
from __future__ import annotations

import copy
import re

from .safetensors_io import DTYPE_BYTES

SUPPORTED_TYPES = {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "llama", "mistral", "qwen2", "qwen3", "qwen3_moe"}
REFUSED = {"gemma4": "Gemma 4 stores per-layer inputs for all layers in one table and shares attention memory between its last layers; "
                     "cutting or copying layers is not supported for it yet."}
LAYER_PREFIXES = ("model.language_model.layers.", "model.layers.", "language_model.model.layers.")
LAYER_KEY = re.compile(r"^(model\.language_model\.layers\.|model\.layers\.|language_model\.model\.layers\.)(\d+)\.(.+)$")
ZERO_SUFFIXES = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "self_attn.o_proj.bias", "mlp.down_proj.weight", "mlp.down_proj.bias")
INDEX_LIST_KEYS = ("mlp_only_layers",)          # lists of layer indices (remapped)
INT_INDEX_KEYS = ("max_window_layers",)         # an index boundary (clamped; refused if it would change meaning, see fix_config)

_ANY_LAYERS = re.compile(r"(^|\.)layers\.(\d+)\.")
UNSUPPORTED_TEXT = "Only Qwen3.5 and plain Llama/Qwen/Mistral-style models can be slimmed or grown so far."


class ReshapeError(ValueError):
    """plain-language message for the user"""


# ---------------------------------------------------------------------------------------------------------------- config

def text_config(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def model_type(cfg: dict) -> str:
    mt = cfg.get("model_type")
    if isinstance(mt, str) and mt:
        return mt
    mt = text_config(cfg).get("model_type")
    return mt if isinstance(mt, str) else ""


def check_supported(cfg: dict) -> None:
    """Raises ReshapeError for models this tool cannot reshape (refused families, unknown types, quantized, custom code, no layer count)."""
    tc = text_config(cfg)
    for mt in (model_type(cfg), str(tc.get("model_type") or "")):
        for key, why in REFUSED.items():
            if mt == key or mt.startswith(key + "_"):
                raise ReshapeError(why)
    mt = model_type(cfg)
    if mt not in SUPPORTED_TYPES:
        raise ReshapeError(UNSUPPORTED_TEXT + (f" (This one is of type {mt!r}.)" if mt else ""))
    if cfg.get("quantization_config") or tc.get("quantization_config"):
        raise ReshapeError("This model is stored in a compressed (quantized) form. Slim or grow needs the full 16-bit original.")
    if cfg.get("auto_map") or tc.get("auto_map"):
        raise ReshapeError("This model needs its own program code to load (trust_remote_code); that is not supported here.")
    n = tc.get("num_hidden_layers")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ReshapeError("The model's config.json has no usable num_hidden_layers.")


def layer_types_of(tc: dict) -> list[str]:
    """tc["layer_types"], else the transformers rule when full_attention_interval is given, else all full attention."""
    n = int(tc["num_hidden_layers"])
    lt = tc.get("layer_types")
    if isinstance(lt, list):
        if len(lt) != n:
            raise ReshapeError(f"config.json lists {len(lt)} layer types for {n} layers.")
        return [str(x) for x in lt]
    interval = tc.get("full_attention_interval")
    if isinstance(interval, int) and not isinstance(interval, bool) and interval > 0:
        return ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(n)]
    return ["full_attention"] * n


def pattern_period(types: list[str]) -> int:
    """smallest p dividing L with types[i] == types[i % p]; 1 for all-equal; L if aperiodic"""
    n = len(types)
    for p in range(1, n + 1):
        if n % p == 0 and all(types[i] == types[i % p] for i in range(n)):
            return p
    return n


# -------------------------------------------------------------------------------------------------------------- tensors

def split_key(name: str):
    """(prefix, index, rest) for decoder-layer tensors; None for mtp.*, model.visual.*, embed, norm, lm_head"""
    m = LAYER_KEY.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def group_params(tensors: dict, n_layers: int) -> dict:
    """tensors: {name: (dtype, shape)}. Counts parameters per group; ReshapeError for a layer index >= n_layers or an unknown "*.layers.N." prefix."""
    layers = [0] * n_layers
    g = {"embed": 0, "lm_head": 0, "vision": 0, "mtp": 0, "other": 0}
    by_dtype: dict[str, int] = {}
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= int(d)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + n
        if name.startswith("mtp."):
            g["mtp"] += n
            continue
        if "visual." in name or "vision" in name:
            g["vision"] += n
            continue
        sp = split_key(name)
        if sp is not None:
            if sp[1] >= n_layers:
                raise ReshapeError(f"The weights contain layer {sp[1]} but config.json says the model has {n_layers} layers.")
            layers[sp[1]] += n
        elif _ANY_LAYERS.search(name):
            raise ReshapeError(f"Unknown layer naming in the weights: {name!r}. This model's layout is not supported.")
        elif "embed_tokens" in name:
            g["embed"] += n
        elif name.startswith("lm_head"):
            g["lm_head"] += n
        else:
            g["other"] += n
    total = sum(layers) + sum(g.values())
    dominant = max(by_dtype, key=lambda d: by_dtype[d]) if by_dtype else "BF16"
    return {"layers": layers, **g, "total": total, "bytes_per_param": float(DTYPE_BYTES.get(dominant, 2))}


def params_after(groups: dict, src: list[int]) -> int:
    layers = groups["layers"]
    return groups["total"] - sum(layers) + sum(layers[s] for s in src)


def block_count(n_layers: int, period: int) -> int:
    if period < 1 or n_layers % period:
        raise ReshapeError(f"{n_layers} layers do not divide into blocks of {period}.")
    return n_layers // period


def limits(n_blocks: int, mode: str, protect_first: int = 1, protect_last: int = 1) -> tuple[int, int]:
    """(min, max) blocks to drop/add; (1, 0) when nothing is feasible."""
    if mode == "prune":
        mx = min(n_blocks // 2, n_blocks - protect_first - protect_last)
    elif mode == "grow":
        mx = n_blocks // 2
    else:
        raise ReshapeError(f"Unknown mode {mode!r}.")
    return 1, max(0, mx)


def blocks_for_target(groups: dict, period: int, target_params: float, mode: str, protect_first: int = 1, protect_last: int = 1) -> int:
    """The feasible block count whose resulting size is nearest to target_params (ties: the smaller change)."""
    n_layers = len(groups["layers"])
    nb = block_count(n_layers, period)
    lo, hi = limits(nb, mode, protect_first, protect_last)
    if hi < lo:
        raise ReshapeError("This model is too small to change in whole blocks.")
    mean_block = sum(groups["layers"]) / nb
    best, best_gap = lo, None
    for n in range(lo, hi + 1):
        if mode == "prune":
            size = groups["total"] - n * mean_block
        else:
            size = params_after(groups, grow_src(n_layers, period, n)[0])
        gap = abs(size - target_params)
        if best_gap is None or gap < best_gap:
            best, best_gap = n, gap
    return best


# ------------------------------------------------------------------------------------------------------------- choosing

def finalize_distance(sums: list, counts: list) -> list:
    """D[a][b] = sums/counts (a<b), 0 on the diagonal, mirrored. Entries without counts are 0."""
    size = len(sums)
    D = [[0.0] * size for _ in range(size)]
    for a in range(size):
        for b in range(a + 1, size):
            v = sums[a][b] / counts[a][b] if counts[a][b] else 0.0
            D[a][b] = D[b][a] = v
    return D


def choose_drop(D: list, n: int, strategy: str = "contiguous", protect_first: int = 1, protect_last: int = 1) -> list[int]:
    """D over block boundaries 0..B (B = number of blocks). "contiguous": a in [protect_first, B - protect_last - n] minimising D[a][a+n] (Gromov et al.);
    "separate": the n blocks k in [protect_first, B - protect_last) with the smallest D[k][k+1] (ShortGPT Block Influence on blocks). Ties -> lowest index.
    Returns sorted block indices; ReshapeError when n does not fit."""
    B = len(D) - 1
    if n < 1:
        raise ReshapeError("Choose at least one block.")
    if strategy == "contiguous":
        first, last = protect_first, B - protect_last - n
        if last < first:
            raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
        a = min(range(first, last + 1), key=lambda i: (D[i][i + n], i))
        return list(range(a, a + n))
    if strategy == "separate":
        cand = list(range(protect_first, B - protect_last))
        if n > len(cand):
            raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
        picked = sorted(sorted(cand, key=lambda k: (D[k][k + 1], k))[:n])
        return picked
    raise ReshapeError(f"Unknown strategy {strategy!r}.")


def halves_agree(D1: list, D2: list, n: int, strategy: str, protect_first: int, protect_last: int) -> bool:
    """same choice from both halves of the calibration set"""
    try:
        return choose_drop(D1, n, strategy, protect_first, protect_last) == choose_drop(D2, n, strategy, protect_first, protect_last)
    except ReshapeError:
        return False


def blocks_to_layers(blocks: list[int], period: int) -> list[int]:
    return [b * period + i for b in sorted(blocks) for i in range(period)]


def check_whole_blocks(drop: list[int], period: int, n_layers: int, protect_first: int = 1, protect_last: int = 1) -> list[int]:
    """Explicit layer list -> block indices. ReshapeError unless the layers are whole blocks that are not protected."""
    drop = sorted(set(drop))
    if not drop:
        raise ReshapeError("No layers to remove were given.")
    if drop[0] < 0 or drop[-1] >= n_layers:
        raise ReshapeError(f"Layers must be between 0 and {n_layers - 1}.")
    blocks = sorted({d // period for d in drop})
    if drop != blocks_to_layers(blocks, period):
        raise ReshapeError(f"Layers {drop[0]} to {drop[-1]} do not form whole blocks of {period}"
                           + (" (3 fast + 1 full attention)" if period == 4 else "") + ". Remove whole blocks so the model keeps its pattern.")
    nb = n_layers // period
    if blocks[0] < protect_first or blocks[-1] >= nb - protect_last:
        raise ReshapeError("The first and the last block are protected and cannot be removed.")
    if len(blocks) > nb // 2:
        raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
    return blocks


def prune_src(n_layers: int, drop: list[int]) -> list[int]:
    """kept old indices in order; ReshapeError if drop empty/out of range/would remove everything"""
    gone = set(drop)
    if not gone:
        raise ReshapeError("No layers to remove were given.")
    if min(gone) < 0 or max(gone) >= n_layers:
        raise ReshapeError(f"Layers must be between 0 and {n_layers - 1}.")
    if len(gone) >= n_layers:
        raise ReshapeError("That would remove every layer.")
    return [i for i in range(n_layers) if i not in gone]


def grow_src(L: int, period: int, n_blocks: int, layout: str = "solar") -> tuple[list[int], list[bool]]:
    """SOLAR depth up-scaling / mergekit passthrough (SOLAR 10.7B = layers 0-23 + 8-31 of a 32-layer model), in whole periods.
    Returns (source layer of every new layer, True for the SECOND occurrence of a duplicated source layer)."""
    if layout != "solar":
        raise ReshapeError("Unknown layout")   # the only layout in v1 (a "repeat the middle span" layout gives the same list whenever spans are period-aligned)
    m = n_blocks * period                      # layers added
    if m <= 0 or m > L // 2:
        raise ReshapeError("Grow by at most half of the layers.")
    half = (L + m) // 2
    half -= half % period                      # whole periods
    if half - m < 0:
        raise ReshapeError("Grow by at most half of the layers.")
    src = list(range(0, half)) + list(range(half - m, L))   # the m layers half-m .. half-1 appear twice; len == L + m
    seen: set[int] = set()
    zero = []
    for s in src:
        zero.append(s in seen)
        seen.add(s)
    return src, zero


def check_pattern(types_before: list[str], src: list[int], period: int) -> None:
    """The new layer types must still be the period pattern repeated (so the interval rule and layer_types agree)."""
    new = [types_before[s] for s in src]
    pattern = types_before[:period]
    ok = period >= 1 and len(new) % period == 0 and all(t == pattern[i % period] for i, t in enumerate(new))
    if not ok:
        raise ReshapeError("This change would break the attention pattern of the model (blocks of "
                           f"{period} layers must stay whole). Choose whole blocks.")


def fix_config(cfg: dict, src: list[int], notes: list | None = None) -> dict:
    """Deep copy of config.json for the new layer list. In text_config: num_hidden_layers = len(src); layer_types remapped (only if present before);
    full_attention_interval kept (check_pattern guarantees consistency); INDEX_LIST_KEYS remapped; INT_INDEX_KEYS clamped or refused;
    any OTHER list in text_config whose length equals the old layer count is remapped like layer_types (and a line is appended to `notes`).
    vision_config, mtp_num_hidden_layers and top-level keys stay untouched."""
    new = copy.deepcopy(cfg)
    tc = text_config(new)
    old_n = int(tc["num_hidden_layers"])
    for s in src:
        if not 0 <= s < old_n:
            raise ReshapeError(f"Layer {s} does not exist.")
    handled = {"num_hidden_layers", *INDEX_LIST_KEYS, *INT_INDEX_KEYS, "layer_types"}
    if isinstance(tc.get("layer_types"), list):
        tc["layer_types"] = [tc["layer_types"][s] for s in src]
    for key in INDEX_LIST_KEYS:
        if isinstance(tc.get(key), list):
            keep = set(tc[key])
            tc[key] = [j for j, s in enumerate(src) if s in keep]
    for key in INT_INDEX_KEYS:
        v = tc.get(key)
        if isinstance(v, int) and not isinstance(v, bool):
            if v >= old_n:
                tc[key] = len(src)
            elif v != 0:
                raise ReshapeError(f"The model switches layer behaviour inside the stack ({key} = {v}); cutting or copying layers would change that.")
    for key, v in list(tc.items()):
        if key in handled or not isinstance(v, list) or len(v) != old_n or any(isinstance(x, (dict, list)) for x in v):
            continue
        tc[key] = [v[s] for s in src]
        if notes is not None:
            notes.append(f"config.json list {key!r} has one entry per layer and was remapped like layer_types.")
    tc["num_hidden_layers"] = len(src)
    return new
