"""Refitting the kept neurons after a width cut ("local reconstruction", experimental, off by default).

When the MLP of a layer loses neurons, the output of the layer changes by what the removed neurons used to add. A least-squares refit of the kept neurons' output
weights can make up for part of it without any training: with A = the activation entering `down_proj` on real tokens, K = the kept neurons, R = the removed ones and
W0 = the original `down_proj` ([H, I]), find D so that  A_K (W0_K + D)^T  is close to  A W0^T, with a ridge penalty:

    E   = A_R W0_R^T                      what the cut takes away            [N, H]
    G   = A_K^T A_K                       the kept neurons' second moments   [k, k]
    C   = A_K^T E                                                           [k, H]
    D^T = (G + lambda I)^-1 C,  lambda = alpha * trace(G) / k

Only G, C and e2 = ||E||^2 are ever accumulated (never the activations themselves), so the memory is k x k + k x H per layer. This module is the part that needs no torch:
the accumulation, the solve, the held-out error that follows exactly from the statistics, the choice of the rows and the name of a result. The pass that streams the model
through the layers is the second half of this file (added with the Try-first pass).

Honest limits: this is standard linear algebra; whether it keeps more quality is NOT known for any real model and is measured by Try first before anything is written. A
gain before healing may not survive healing. numpy only.
"""
from __future__ import annotations

import hashlib
import json

from . import slice_io

DEFAULT_ALPHA = 0.01                # ridge strength relative to the mean eigenvalue of G (trace / k)
DEFAULT_SAMPLES = 128               # project rows used (besides the reasoning rows the measurement also uses)
DEFAULT_LEN = 512
VARIANTS = ("pruned", "original")   # which stream feeds the layer: the already-cut prefix (sequential, exact) or the uncut model's own activations
KEEP_TAGS = 2                       # recon sidecars kept per model folder


# ------------------------------------------------------------------------------------------------------------ the statistics

def new_stats(k: int, hidden: int) -> dict:
    """Empty statistics of one layer for `k` kept neurons and `hidden` output channels (float64)."""
    n = slice_io.np()
    return {"G": n.zeros((k, k), dtype=n.float64), "C": n.zeros((k, hidden), dtype=n.float64), "e2": 0.0, "n": 0}


def add_stats(stats: dict, a_kept, a_removed, w0_removed) -> dict:
    """Adds one batch to `stats` (in place; returned). `a_kept` [N, k] and `a_removed` [N, r]: the activations entering `down_proj` of the real tokens; `w0_removed` [H, r]: the
    original down_proj columns of the removed neurons."""
    n = slice_io.np()
    ak = n.asarray(a_kept, dtype=n.float64)
    err = n.asarray(a_removed, dtype=n.float64) @ n.asarray(w0_removed, dtype=n.float64).T           # E [N, H]
    stats["G"] += ak.T @ ak
    stats["C"] += ak.T @ err
    stats["e2"] += float((err * err).sum())
    stats["n"] += int(ak.shape[0])
    return stats


def merge_stats(a: dict, b: dict) -> dict:
    """The statistics of both halves together (a new dict)."""
    return {"G": a["G"] + b["G"], "C": a["C"] + b["C"], "e2": a["e2"] + b["e2"], "n": a["n"] + b["n"]}


def ridge_lambda(g, alpha: float = DEFAULT_ALPHA) -> float:
    """lambda = alpha x the mean eigenvalue of G (trace / k); 0 for an empty problem."""
    n = slice_io.np()
    k = g.shape[0]
    return float(alpha) * float(n.trace(g)) / k if k else 0.0


# ------------------------------------------------------------------------------------------------------------ solving

def solve_ridge(g, c, lam: float, tol: float = 1e-6):
    """X = (G + lam I)^-1 C in float64 by a Cholesky solve, checked against its residual; a failed factorisation or a residual above `tol` (relative to ||C||) falls back to
    least squares. Returns X [k, H]."""
    n = slice_io.np()
    a = n.asarray(g, dtype=n.float64) + lam * n.eye(g.shape[0])
    c = n.asarray(c, dtype=n.float64)
    try:
        chol = n.linalg.cholesky(a)
        x = n.linalg.solve(chol.T, n.linalg.solve(chol, c))
        ok = float(n.linalg.norm(a @ x - c)) <= tol * max(float(n.linalg.norm(c)), 1e-300)
    except n.linalg.LinAlgError:
        ok = False
    if not ok:
        x = n.linalg.lstsq(a, c, rcond=None)[0]
    return x


def solve_delta(stats: dict, alpha: float = DEFAULT_ALPHA):
    """D [H, k]: the change of the kept columns of down_proj that the statistics ask for (D^T = (G + lambda I)^-1 C)."""
    return solve_ridge(stats["G"], stats["C"], ridge_lambda(stats["G"], alpha)).T


# ------------------------------------------------------------------------------------------------------------ the held-out measure

def heldout_error(delta, stats: dict) -> float:
    """||A_K D^T - E||^2 on the rows of `stats`, exactly from the statistics: tr(D^T G D) - 2 tr(D^T C) + ||E||^2 (D = delta [H, k]). Fit on one half, call it with the other
    half's statistics for an honest number."""
    n = slice_io.np()
    dt = n.asarray(delta, dtype=n.float64).T
    return float((dt * (stats["G"] @ dt)).sum() - 2.0 * (dt * stats["C"]).sum() + stats["e2"])


def removed_share(delta, stats: dict):
    """The share of what the cut took away (||E||^2) that the refit gives back on these rows: 1 - error / ||E||^2. 0 = nothing (D = 0 gives 0), 1 = all of it; negative = the
    refit made it worse on rows it was not fitted on. None when nothing was taken away."""
    if not stats["e2"]:
        return None
    return 1.0 - heldout_error(delta, stats) / stats["e2"]


def layer_summary(shares: list) -> dict:
    """Median and worst of the per-layer held-out shares (None entries left out): {"median", "worst", "layers"}."""
    vals = sorted(s for s in shares if s is not None)
    if not vals:
        return {"median": None, "worst": None, "layers": 0}
    mid = len(vals) // 2
    return {"median": vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0, "worst": vals[0], "layers": len(vals)}


def fit_two_halves(stats1: dict, stats2: dict, alpha: float = DEFAULT_ALPHA) -> dict:
    """Fit on half 1, score on half 2 (and the other way round), then the final fit on both. {"delta": D of both halves, "share_h1_to_h2", "share_h2_to_h1", "share": the mean
    of the two held-out shares}. A refit that only fits its own half shows up as a low or negative held-out share."""
    d1, d2 = solve_delta(stats1, alpha), solve_delta(stats2, alpha)
    a, b = removed_share(d1, stats2), removed_share(d2, stats1)
    held = [x for x in (a, b) if x is not None]
    return {"delta": solve_delta(merge_stats(stats1, stats2), alpha), "share_h1_to_h2": a, "share_h2_to_h1": b, "share": sum(held) / len(held) if held else None}


# ------------------------------------------------------------------------------------------------------------ the rows

def tail_rows(n: int, head: int, want: int) -> list:
    """The indices of the rows the refit uses out of `n` rows in their seeded shuffled order: the LAST `want` of them, never one of the first `head` (those are the rows Try
    first, the measurement and the cut check take). Fewer than `want` when the data is too small; the caller says so."""
    start = max(int(head), int(n) - int(want))
    return list(range(start, int(n))) if start < n else []


def row_hashes(rows: list) -> list:
    """sha1 of the token ids of every row (rows are lists of ids)."""
    return [hashlib.sha1(json.dumps([int(x) for x in r], separators=(",", ":")).encode()).hexdigest() for r in rows]


def overlap(rows_a: list, rows_b: list) -> int:
    """How many rows of `rows_a` also appear (by the hash of their token ids) in `rows_b`: 0 for a refit that is judged on rows it never saw."""
    seen = set(row_hashes(rows_b))
    return sum(1 for h in row_hashes(rows_a) if h in seen)


# ------------------------------------------------------------------------------------------------------------ naming and writing the result

def recon_tag(weights: str, width: int, scoring: str, skipped_blocks, variant: str, alpha: float, rows_key: str) -> str:
    """The name of one refit: everything its numbers depend on (the identity of the weight files, the width, how the neurons were ranked, the blocks skipped, which stream feeds
    the layers, the ridge strength and the rows). Another input is another tag, so a sidecar is never used for the wrong cut."""
    ident = {"v": 1, "weights": weights, "width": int(width), "scoring": scoring or "activation", "skipped": sorted({int(b) for b in skipped_blocks or ()}), "variant": variant,
             "alpha": float(alpha), "rows": rows_key}
    return hashlib.sha1(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def combine_bf16(w0_kept, delta):
    """The bf16 bits (uint16 [H, k]) of W0_K + D, added in float32: `w0_kept` is the original columns as float32 (`slice_io.bf16_to_f32` of the file's values) and `delta` the
    refit. The same rounding as every other bf16 value the program writes."""
    n = slice_io.np()
    return slice_io.f32_to_bf16(n.asarray(w0_kept, dtype=n.float32) + n.asarray(delta, dtype=n.float32))


# ============================================================================================================ the pass (needs torch)
# The model is streamed through its decoder layers one at a time (layer-wise replay): one partial forward captures the input of layer 0 and the keyword arguments that the
# layers of each type are called with (linear-attention and full-attention layers have different masks and position embeddings); the hidden states then live in CPU memory,
# one list per stream, and every layer is run on them from there. "pruned" (default) fits each layer on what the already-cut model feeds it, which is exactly what the written
# model will see; "original" fits it on the uncut model's activations. The statistics are accumulated on the card in float32 (TF32 off), solved there with a Cholesky
# factorisation, and checked; a failed check is solved again on the CPU in float64.

from . import protocol  # noqa: E402
from .reshape import ReshapeError  # noqa: E402


class Stopped(Exception):
    """The stop file appeared while the layers were being refitted."""


class _Captured(Exception):
    """Raised inside the capture forward once every layer type has been seen: the rest of the forward is not needed."""


def _to_cpu(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    if isinstance(x, tuple):
        return tuple(_to_cpu(v) for v in x)
    if isinstance(x, list):
        return [_to_cpu(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_cpu(v) for k, v in x.items()}
    return x


def _to_dev(x, dev):
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, tuple):
        return tuple(_to_dev(v, dev) for v in x)
    if isinstance(x, list):
        return [_to_dev(v, dev) for v in x]
    if isinstance(x, dict):
        return {k: _to_dev(v, dev) for k, v in x.items()}
    return x


def type_of(t: str) -> str:
    return "full_attention" if t == "full_attention" else "linear_attention"


def capture_inputs(model, layers, batches: list, types: list) -> dict:
    """{"h0": [hidden state entering layer 0 of every batch, CPU], "kwargs": [{layer type: the keyword arguments that layer type is called with, CPU} per batch]}. One partial
    forward per batch; it stops as soon as the first layer of every type has been seen."""
    import torch
    dev = next(model.parameters()).device
    first = {}
    for i, t in enumerate(types):
        first.setdefault(type_of(t), i)
    out = {"h0": [], "kwargs": []}
    for b in batches:
        cur: dict = {"kw": {}}

        def hook_for(i, kind):
            def hook(module, args, kwargs):
                kw = {k: v for k, v in kwargs.items() if k != "hidden_states"}
                cur["kw"][kind] = _to_cpu(kw)
                if i == 0:
                    cur["h0"] = _to_cpu(args[0] if args else kwargs["hidden_states"])
                if len(cur["kw"]) == len(first) and "h0" in cur:
                    raise _Captured()
            return hook

        handles = [layers[i].register_forward_pre_hook(hook_for(i, kind), with_kwargs=True) for kind, i in first.items()]
        if 0 not in first.values():
            handles.append(layers[0].register_forward_pre_hook(hook_for(0, type_of(types[0])), with_kwargs=True))
        try:
            with torch.no_grad():
                model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), use_cache=False)
        except _Captured:
            pass
        finally:
            for h in handles:
                h.remove()
        if "h0" not in cur or len(cur["kw"]) < len(first):
            raise ReshapeError("The inputs of the layers could not be captured, so the neurons cannot be refitted for this model.")
        out["h0"].append(cur["h0"])
        out["kwargs"].append(cur["kw"])
    return out


def run_layer(layer, h, kw, dev):
    """The output hidden state of one decoder layer for the (CPU) input `h` and the captured arguments `kw`; the result is back on the CPU in the input's dtype."""
    import torch
    with torch.no_grad():
        o = layer(h.to(dev), **_to_dev(kw, dev))
    o = o[0] if isinstance(o, tuple) else o
    return o.detach().to("cpu")


class _Replace:
    """While the `with` block runs: the MLP neurons that a width cut removes are zeroed in front of `down_proj` and, for a refitted layer, its output is the kept neurons
    times the refitted weights W_K ([H, k], the same bf16 numbers the written model has): exactly what the cut model computes. `raw` (optional) is called with the unmasked
    input of down_proj first (the statistics hook)."""

    def __init__(self, layer, mask=None, kept=None, w_k=None, raw=None):
        self.layer, self.mask, self.kept, self.w_k, self.raw, self.handles = layer, mask, kept, w_k, raw, []

    def __enter__(self):
        down = self.layer.mlp.down_proj
        if self.raw is not None:
            self.handles.append(down.register_forward_pre_hook(lambda m, args: self.raw(args[0])))            # registered first: it sees the raw activations
        if self.mask is not None:
            def pre(module, args):
                x = args[0]
                return (x * self.mask.to(device=x.device, dtype=x.dtype),) + tuple(args[1:])
            self.handles.append(down.register_forward_pre_hook(pre))
        if self.w_k is not None:
            def post(module, args, output):
                x = args[0]
                return x.index_select(-1, self.kept.to(x.device)) @ self.w_k.to(device=x.device, dtype=x.dtype).t()
            self.handles.append(down.register_forward_hook(post))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


def _solve_t(g, c, alpha: float):
    """(G + lambda I)^-1 C on the card in float32 (Cholesky, residual checked); a failure is solved again in float64 on the CPU. Returns a float32 tensor on the card."""
    import torch
    k = g.shape[0]
    lam = float(alpha) * float(torch.trace(g)) / k if k else 0.0
    a = g + lam * torch.eye(k, device=g.device, dtype=g.dtype)
    chol, info = torch.linalg.cholesky_ex(a)
    if int(info) == 0:
        x = torch.cholesky_solve(c, chol)
        if float(torch.linalg.norm(a @ x - c)) <= 1e-3 * max(float(torch.linalg.norm(c)), 1e-30):
            return x
    protocol.log("A refit was solved again on the CPU in double precision (the card's solve was not accurate enough).")
    x = solve_ridge(g.double().cpu().numpy(), c.double().cpu().numpy(), lam)
    return torch.as_tensor(x, dtype=g.dtype, device=g.device)


def _heldout_t(dt, st) -> float:
    return float((dt * (st["G"] @ dt)).sum() - 2.0 * (dt * st["C"]).sum() + st["e2"])


def _share_t(dt, st):
    return None if not st["e2"] else 1.0 - _heldout_t(dt, st) / st["e2"]


def _new_t(k: int, hidden: int, dev):
    import torch
    return {"G": torch.zeros((k, k), device=dev), "C": torch.zeros((k, hidden), device=dev), "e2": 0.0, "n": 0}


def _add_t(st, a, kept, removed, w0_removed):
    """Adds the real-token activations `a` [N, I] (float32 on the card) in pieces of 8,192 rows."""
    for lo in range(0, a.shape[0], 8192):
        piece = a[lo:lo + 8192]
        ak = piece.index_select(1, kept)
        err = piece.index_select(1, removed) @ w0_removed.t()
        st["G"] += ak.t() @ ak
        st["C"] += ak.t() @ err
        st["e2"] += float((err * err).sum())
        st["n"] += int(piece.shape[0])


def streams_bytes(batches: list, hidden: int, streams: int = 2) -> int:
    """The CPU memory the hidden-state streams take: real tokens x hidden x 2 bytes, per stream."""
    return sum(int(b["attention_mask"].sum()) for b in batches) * hidden * 2 * streams


def fit_layers(model, layers, batches: list, types: list, skip_layers, keep: dict, w0_of, hidden: int, alpha: float = DEFAULT_ALPHA, variant: str = "pruned",
               stop=lambda: False, ram_bytes: int = 0) -> dict:
    """Refits the kept neurons of every layer that has a width cut. `keep` {layer: sorted kept neuron indices} (the layers that are not skipped); `w0_of(layer)` -> the
    original down_proj of that layer as a float32 torch tensor [H, I] (from the weight file: right even when the model is loaded in 4-bit); `skip_layers` the layers of the
    blocks a depth cut removes. Returns {"w_k": {layer: bf16 bits (uint16 numpy [H, k])}, "shares": {layer: {"share", "h1_to_h2", "h2_to_h1", "tokens"}}, "summary":
    layer_summary of the held-out shares, "tokens": n}. Raises ReshapeError (plain words) when the streams would not fit in memory and `Stopped` when `stop()` says so."""
    import torch
    if variant not in VARIANTS:
        raise ReshapeError("The refit takes its input from the cut model (\"pruned\") or from the original (\"original\").")
    dev = next(model.parameters()).device
    need = streams_bytes(batches, hidden, 2 if variant == "original" else 1)
    if ram_bytes and need > 0.5 * ram_bytes:
        raise ReshapeError(f"Refitting the neurons keeps the hidden states of the measuring text in memory: about {need / 1e9:.1f} GB, more than half of this computer's {ram_bytes / 1e9:.1f} GB. "
                           "Use fewer samples.")
    skip = set(skip_layers or ())
    tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        cap = capture_inputs(model, layers, batches, types)
        h_p = list(cap["h0"])                                              # the cut model's stream
        h_o = list(cap["h0"]) if variant == "original" else None           # the uncut model's stream
        masks_in = [b["attention_mask"] for b in batches]
        w_k_bits: dict = {}
        shares: dict = {}
        total = sum(1 for i in range(len(layers)) if i not in skip and i in keep)
        done, t0, tokens = 0, __import__("time").monotonic(), sum(int(m.sum()) for m in masks_in)
        for l, layer in enumerate(layers):
            if stop():
                raise Stopped()
            kw_type = type_of(types[l])
            if l in skip:
                if h_o is not None:
                    h_o = [run_layer(layer, h, cap["kwargs"][b][kw_type], dev) for b, h in enumerate(h_o)]
                continue
            if l not in keep:                                              # a layer without a width cut: just advance the streams
                h_p = [run_layer(layer, h, cap["kwargs"][b][kw_type], dev) for b, h in enumerate(h_p)]
                if h_o is not None:
                    h_o = [run_layer(layer, h, cap["kwargs"][b][kw_type], dev) for b, h in enumerate(h_o)]
                continue
            kept_idx = torch.as_tensor(keep[l], dtype=torch.long, device=dev)
            size = layer.mlp.down_proj.in_features if hasattr(layer.mlp.down_proj, "in_features") else int(w0_of(l).shape[1])
            removed_idx = torch.as_tensor([i for i in range(size) if i not in set(keep[l])], dtype=torch.long, device=dev)
            w0 = w0_of(l).to(dev)
            w0_r, w0_k = w0.index_select(1, removed_idx), w0.index_select(1, kept_idx)
            halves = [_new_t(len(keep[l]), w0.shape[0], dev), _new_t(len(keep[l]), w0.shape[0], dev)]
            for bi in range(len(batches)):
                m = masks_in[bi].to(dev).bool()
                part = halves[bi % 2]
                src = h_o if variant == "original" else h_p
                grab = lambda x, m=m, part=part, w0_r=w0_r: _add_t(part, x.float()[m] if x.dim() == 3 else x.float(), kept_idx, removed_idx, w0_r)      # noqa: E731
                with _Replace(layer, raw=grab):
                    out = run_layer(layer, src[bi], cap["kwargs"][bi][kw_type], dev)
                if variant == "original":
                    h_o[bi] = out
            if halves[1]["n"] == 0:                                        # a single batch: both halves are the same measurement
                halves[1] = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in halves[0].items()}
            x1, x2 = _solve_t(halves[0]["G"], halves[0]["C"], alpha), _solve_t(halves[1]["G"], halves[1]["C"], alpha)
            both = {"G": halves[0]["G"] + halves[1]["G"], "C": halves[0]["C"] + halves[1]["C"], "e2": halves[0]["e2"] + halves[1]["e2"], "n": halves[0]["n"] + halves[1]["n"]}
            dt = _solve_t(both["G"], both["C"], alpha)                      # D^T [k, H]
            s12, s21 = _share_t(x1, halves[1]), _share_t(x2, halves[0])
            held = [s for s in (s12, s21) if s is not None]
            shares[l] = {"share": sum(held) / len(held) if held else None, "h1_to_h2": s12, "h2_to_h1": s21, "tokens": both["n"]}
            bits = combine_bf16(w0_k.float().cpu().numpy(), dt.t().float().cpu().numpy())               # the numbers the written model has
            w_k_bits[l] = bits
            w_k = torch.as_tensor(slice_io.bf16_to_f32(bits), device=dev)
            mask = torch.zeros(size, device=dev)
            mask[kept_idx] = 1.0
            h_p = [run_layer_with(layer, h, cap["kwargs"][b][kw_type], dev, mask, kept_idx, w_k) for b, h in enumerate(h_p)]
            del halves, both, x1, x2, dt, w0, w0_r, w0_k, w_k
            done += 1
            protocol.progress("recon", done / max(total, 1), f"Refitting the kept neurons: layer {done} of {total}",
                              protocol.make_detail("layers", done, total, t0, rate_unit="layers/s"))
        return {"w_k": w_k_bits, "shares": shares, "summary": layer_summary([s["share"] for s in shares.values()]), "tokens": tokens}
    finally:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = tf32


def run_layer_with(layer, h, kw, dev, mask, kept, w_k):
    """`run_layer` with the width cut and the refitted down_proj of this layer in place."""
    with _Replace(layer, mask=mask, kept=kept, w_k=w_k):
        return run_layer(layer, h, kw, dev)


class Corrected:
    """Context for the layers of a loaded model: the refitted down_proj of every layer in `fit` ({layer: (kept index tensor, W_K tensor [H, k])}), on top of the cut that `reshapetry._Cut` already makes (zeroed neurons), for
    the Try-first pass of a refitted cut. Entering it twice is a mistake; it removes its hooks on exit."""

    def __init__(self, layers, fit: dict):
        self.parts = [_Replace(layers[l], mask=None, kept=k, w_k=w) for l, (k, w) in fit.items()]

    def __enter__(self):
        for p in self.parts:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in self.parts:
            p.__exit__(*exc)
        return False


# ------------------------------------------------------------------------------------------------------------ the sidecar

def sidecar_dir(scores_dir, tag: str):
    from pathlib import Path
    return Path(scores_dir) / "recon" / tag


def write_sidecar(scores_dir, tag: str, w_k_bits: dict, kept: dict, meta: dict) -> None:
    """`<scores_dir>/recon/<tag>/model.safetensors` (tensors `layers.<old layer>.down_proj.weight`, BF16 [H, k]) and `recon.json` (the numbers `meta` and the kept indices of
    every layer), written under a temporary name and moved into place; at most KEEP_TAGS tags are kept in the folder (the newest)."""
    import os
    import shutil
    from pathlib import Path
    from . import safetensors_io
    base = Path(scores_dir) / "recon"
    base.mkdir(parents=True, exist_ok=True)
    tmp, final = base / f"{tag}.partial", base / tag
    shutil.rmtree(tmp, ignore_errors=True)
    tensors = [safetensors_io.OutTensor(f"layers.{l}.down_proj.weight", "BF16", tuple(bits.shape), None, data=bits) for l, bits in sorted(w_k_bits.items())]
    safetensors_io.write_shards(tensors, tmp, shard_bytes=1 << 62)
    (tmp / "recon.json").write_text(json.dumps({**meta, "kept": {str(l): [int(i) for i in kept[l]] for l in w_k_bits}}), encoding="utf-8")
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    others = sorted((p for p in base.iterdir() if p.is_dir() and not p.name.endswith(".partial")), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in others[KEEP_TAGS:]:
        shutil.rmtree(old, ignore_errors=True)


def read_sidecar(scores_dir, tag: str):
    """({layer: TensorRef of its W_K}, recon.json) of a sidecar, or None when there is none (or it is incomplete)."""
    from . import safetensors_io
    d = sidecar_dir(scores_dir, tag)
    try:
        meta = json.loads((d / "recon.json").read_text(encoding="utf-8"))
        refs = safetensors_io.list_tensors(d)
    except (OSError, ValueError):
        return None
    out = {}
    for name, ref in refs.items():
        parts = name.split(".")
        if len(parts) == 4 and parts[0] == "layers" and parts[1].isdigit():
            out[int(parts[1])] = ref
    return (out, meta) if out and set(str(l) for l in out) == set(meta.get("kept", {})) else None


def down_refs(tensors: dict, n_layers: int) -> dict:
    """{layer: TensorRef of that decoder layer's mlp.down_proj.weight} from the weight files' tensors."""
    from .reshape import split_key
    found = {}
    for name, ref in tensors.items():
        sp = split_key(name)
        if sp is not None and sp[2] == "mlp.down_proj.weight":
            found[sp[1]] = ref
    if any(i not in found for i in range(n_layers)):
        raise ReshapeError("A layer of this model has no plain MLP output projection, so its neurons cannot be refitted.")
    return found


def w0_reader(refs: dict):
    """w0_of(layer) for `fit_layers`: the original down_proj as float32 torch [H, I], read from the weight file (bf16 / f16 / f32), whatever way the model is loaded."""
    import torch

    def w0_of(layer):
        ref = refs[layer]
        a = slice_io.read_array(ref)
        if ref.dtype == "BF16":
            a = slice_io.bf16_to_f32(a)
        return torch.as_tensor(a.astype("float32", copy=False))
    return w0_of


def sidecar_tensor(ref, dev, keep_on_card: bool):
    """The refitted W_K of one layer as a bf16 torch tensor, on the card or (when memory is short) on the CPU."""
    import torch
    bits = slice_io.read_array(ref)
    t = torch.from_numpy(bits.view("int16").copy()).view(torch.bfloat16)
    return t.to(dev) if keep_on_card else t


def rows_key(reasoning, samples: int, length: int, head: int, data: str | None, seed: int) -> str:
    """The name of the rows of a refit: the reasoning part of the calibration, how many samples of which length, how many head samples it keeps away from, the training file and the seed."""
    return hashlib.sha1(json.dumps({"r": reasoning, "n": int(samples), "len": int(length), "head": int(head), "d": data, "seed": seed}, sort_keys=True).encode()).hexdigest()[:12]


def find_sidecar(scores_dir, weights: str, width: int, scoring: str, skipped_blocks, variant: str, alpha: float):
    """The newest sidecar in `<scores_dir>/recon/` that was made for exactly this cut (weights, width, ranking, skipped blocks, variant, alpha): ({layer: TensorRef}, recon.json), or
    None. The rows it was fitted on are not compared: a cut that finds a sidecar Try first made uses those exact numbers, which is what Try measured."""
    from pathlib import Path
    base = Path(scores_dir) / "recon"
    if not base.is_dir():
        return None
    want = (weights, int(width), scoring or "activation", sorted({int(b) for b in skipped_blocks or ()}), variant, float(alpha))
    for d in sorted((p for p in base.iterdir() if p.is_dir() and not p.name.endswith(".partial")), key=lambda p: p.stat().st_mtime, reverse=True):
        found = read_sidecar(scores_dir, d.name)
        if found is None:
            continue
        m = found[1]
        if (m.get("weights"), m.get("width"), m.get("scoring") or "activation", sorted(m.get("skipped") or ()), m.get("variant"), float(m.get("alpha", -1))) == want:
            return found
    return None
