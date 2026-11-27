"""The importance matrix of a mixture of experts, cut along the EXPERT axis (docs/MOE_GUIDE.md section 6).

llama.cpp measures a matrix per expert tensor of a layer: `blk.N.ffn_down_exps.weight.in_sum2` holds n_expert x n_in values (expert-major) and `.counts` one number per expert. A cut that keeps
experts `kept[N]` of layer N needs the rows of those experts, in the order they now have; every other tensor (attention, routers, shared parts) is the same as in the original.
Pure numpy; the caller reads and writes the file (`imatrixadapt`'s reader and writer).
"""
from __future__ import annotations

import re

import numpy as np

EXPS_RE = _EXPS = re.compile(r"^blk\.(\d+)\.(ffn_(?:down|up|gate|gate_up)_exps)\.weight\.(in_sum2|counts)$")


class MoeMatrixError(ValueError):
    pass


def slice_experts(arrays: dict, kept: dict, n_experts: int) -> dict:
    """{name: array} with every `*_exps` statistic reduced to the kept experts of its layer (`kept[layer]` = original expert numbers, in the new order). Other entries are returned unchanged."""
    out = {}
    for name, a in arrays.items():
        m = _EXPS.match(name)
        if not m:
            out[name] = a
            continue
        layer, kind = int(m.group(1)), m.group(3)
        if layer not in kept and str(layer) not in kept:
            raise MoeMatrixError(f"The cut says nothing about the experts of layer {layer}.")
        idx = np.asarray(kept.get(layer, kept.get(str(layer))), dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n_experts):
            raise MoeMatrixError(f"Layer {layer}: an expert number is outside 0 to {n_experts - 1}.")
        flat = np.asarray(a)
        if kind == "counts":
            if flat.size != n_experts:
                raise MoeMatrixError(f"{name} has {flat.size} counts, not one per expert ({n_experts}).")
            out[name] = flat.reshape(n_experts)[idx].copy()
        else:
            if flat.size % n_experts:
                raise MoeMatrixError(f"{name} has {flat.size} values, which cannot be split over {n_experts} experts.")
            out[name] = flat.reshape(n_experts, -1)[idx].reshape(-1).copy()
    return out
