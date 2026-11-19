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
