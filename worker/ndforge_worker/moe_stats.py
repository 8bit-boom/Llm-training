"""Counting what the experts of a mixture-of-experts model do, with hooks on a model that follows the transformers v5 convention (checked in Inkling2Loop):

    layer.mlp.gate(hidden [T, H])                       -> (logits [T, E], weights [T, k] (already x routed_scaling_factor), indices [T, k])
    layer.mlp.experts(hidden [T, H], indices, weights)  -> [T, H]       (experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I], experts._apply_gate)

Per layer and expert it adds up, over every token and EVERY loop pass (the passes of a loop share the weights): how often the router picked it (`freq`), the routing weight it got
(`gate`, before the routed scaling factor) and weight x the size of its output (`sum_wn`; REAP's score is `sum_wn / freq`, Lasby et al. 2025). Statistics are also kept per pass, and per
half of the calibration batches (batch parity), so a measurement can say whether both halves choose the same experts. It also builds the masked router the cut is imitated with
(removed experts can never be picked), and the load the kept experts get under that mask, which the router-bias rebalance needs.

torch is imported here only; nothing in this module needs a GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class LayerStats:
    def __init__(self, experts: int, passes: int):
        self.experts, self.passes = experts, passes
        z = lambda: torch.zeros(passes, 2, experts, dtype=torch.float64)          # [pass, half, expert]
        self.freq, self.gate, self.sum_wn = z(), z(), z()
        self.tokens = torch.zeros(passes, 2, dtype=torch.float64)

    def add(self, pass_no: int, half: int, indices, weights, norms, scale: float) -> None:
        """indices/weights/norms: [T, k] (norms: the size of each routed expert's output for that token, or None when REAP is not wanted)"""
        e = self.experts
        idx = indices.reshape(-1)
        ok = idx < e                                           # an index equal to the expert count is the "no expert" mark of the reference implementation
        idx = idx[ok]
        w = (weights.float() / scale).reshape(-1)[ok]
        self.freq[pass_no, half] += torch.bincount(idx, minlength=e).double().cpu()
        self.gate[pass_no, half] += torch.bincount(idx, weights=w.double(), minlength=e).cpu()
        if norms is not None:
            self.sum_wn[pass_no, half] += torch.bincount(idx, weights=(w * norms.float().reshape(-1)[ok]).double(), minlength=e).cpu()
        self.tokens[pass_no, half] += indices.shape[0]

    def merged(self, halves=(0, 1)):
        h = list(halves)
        freq, gate, wn = (x[:, h].sum(dim=(0, 1)) for x in (self.freq, self.gate, self.sum_wn))
        reap = torch.where(freq > 0, wn / freq.clamp(min=1), torch.zeros_like(wn))
        return {"freq": freq.tolist(), "gate": gate.tolist(), "reap": reap.tolist(), "sum_wn": wn.tolist()}


def expert_outputs(experts, hidden, idx):
    """‖expert_e(x)‖ for every (token, slot) in idx [T, k]: the size of the output of the expert the router picked, before it is weighted. One expert at a time (the tokens it got)."""
    out = torch.zeros(idx.shape, dtype=torch.float32, device=hidden.device)
    n = getattr(experts, "num_experts", None)
    for e in torch.unique(idx).tolist():
        if n is not None and e >= n:
            continue
        slot_tok, slot_k = torch.where(idx == e)
        x = hidden[slot_tok]
        if hasattr(experts, "expert_forward"):                   # the packed experts decode one expert at a time
            y = experts.expert_forward(e, x)
        else:
            y = F.linear(experts._apply_gate(F.linear(x, experts.gate_up_proj[e])), experts.down_proj[e])
        out[slot_tok, slot_k] = y.float().norm(dim=-1)
    return out


class Recorder:
    """Hooks on the MoE blocks of `layers` (a list of decoder layers with `.mlp.gate` and `.mlp.experts`). `core` = (start, end) layers that run `passes` times."""

    def __init__(self, layers, experts: int, core=(0, 0), passes: int = 1, reap: bool = True):
        self.layers, self.experts, self.core, self.passes, self.reap = list(layers), experts, tuple(core), max(1, passes), reap
        self.stats = [LayerStats(experts, self.passes if self.core[0] <= i < self.core[1] else 1) for i in range(len(self.layers))]
        self.calls = [0] * len(self.layers)
        self.half = 0
        self._handles = []
        self._pending: dict = {}

    def attach(self):
        for i, layer in enumerate(self.layers):
            moe = layer.mlp
            self._handles.append(moe.gate.register_forward_hook(self._gate_hook(i)))
            if self.reap:
                self._handles.append(moe.experts.register_forward_pre_hook(self._experts_hook(i)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def new_batch(self, half: int):
        self.half = half
        self.calls = [0] * len(self.layers)

    def _gate_hook(self, i):
        def hook(module, args, output):
            _, weights, indices = output
            n_pass = self.calls[i] % self.stats[i].passes
            self.calls[i] += 1
            self._pending[i] = (n_pass, indices, weights, getattr(module, "routed_scaling_factor", 1.0) or 1.0)
            if not self.reap:
                self.stats[i].add(n_pass, self.half, indices, weights, None, self._pending[i][3])
        return hook

    def _experts_hook(self, i):
        def hook(module, args):
            hidden, idx, _ = args[0], args[1], args[2]
            n_pass, indices, weights, scale = self._pending.pop(i)
            with torch.no_grad():
                norms = expert_outputs(module, hidden, idx)
            self.stats[i].add(n_pass, self.half, indices, weights, norms, scale)
        return hook

    def result(self) -> dict:
        """{"layers": {"<l>": {freq, gate, reap, sum_wn, per_pass_freq, halves: {"a": {...}, "b": {...}}}}, "tokens"}; every number summed over the passes"""
        out = {}
        for i, s in enumerate(self.stats):
            row = s.merged()
            row["per_pass_freq"] = [s.freq[p].sum(dim=0).tolist() for p in range(s.passes)]
            row["halves"] = {"a": s.merged((0,)), "b": s.merged((1,))}
            out[str(i)] = row
        return {"layers": out, "tokens": [float(s.tokens.sum()) for s in self.stats]}


# ------------------------------------------------------------------------------------------------------ imitating a cut

def masked_router_output(router, hidden_flat, keep_mask, top_k: int | None = None):
    """What the router would answer if only the experts with keep_mask[e] True existed, with the same arithmetic as Inkling2LoopTopKRouter (scores = score_fn(logits); the
    selection adds the bias; the weights are the chosen scores, normalised, x the routed scaling factor). Returns (logits, weights, indices) with indices in the ORIGINAL numbering."""
    k = top_k or router.top_k
    h = hidden_flat.float().reshape(-1, router.hidden_dim)
    logits = F.linear(h, router.weight)
    scores = router.score_fn(logits)
    sel = scores + router.e_score_correction_bias
    sel = sel.masked_fill(~keep_mask.to(sel.device), float("-inf"))
    indices = torch.topk(sel, k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, weights * router.routed_scaling_factor, indices


class Imitation:
    """Replaces the output of every layer's router (a forward hook that returns a new output) by the masked one, and counts the load of each kept expert (share of the routed
    slots), per layer: the load the pruned model would have on this text. `keep`: {layer: sorted kept old ids}; `top_k`: experts per token or None."""

    def __init__(self, layers, keep: dict, experts: int, top_k: int | None = None):
        self.layers, self.keep, self.experts, self.top_k = list(layers), keep, experts, top_k
        self.load = {l: torch.zeros(experts, dtype=torch.float64) for l in keep}
        self._handles = []

    def attach(self):
        for i, layer in enumerate(self.layers):
            if i not in self.keep:
                continue
            mask = torch.zeros(self.experts, dtype=torch.bool)
            mask[list(self.keep[i])] = True
            router = layer.mlp.gate

            def hook(module, args, output, i=i, mask=mask):
                new = masked_router_output(module, args[0], mask, self.top_k)
                self.load[i] += torch.bincount(new[2].reshape(-1), minlength=self.experts).double().cpu()
                return new
            self._handles.append(router.register_forward_hook(hook))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def loads(self) -> dict:
        """{"<layer>": load of each KEPT expert in the kept order, summing to 1}"""
        out = {}
        for l, ids in self.keep.items():
            v = self.load[l][list(ids)]
            total = float(v.sum())
            out[str(l)] = (v / total).tolist() if total else [0.0] * len(ids)
        return out
