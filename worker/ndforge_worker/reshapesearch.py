"""The round planner of "Find the best cut for me" (docs/TRY_FIRST.md, "The automatic search"). Pure Python, no torch, no clock, no random numbers: from the passes that
are already measured (the saved `reshape_try.json`) and a context it says which passes to measure next, so a stop and a repeat (or a search with more time) goes on where
it was and repeats nothing.

The question the search answers: for a model of N blocks and a target size, WHICH blocks to take out. The target size fixes how many blocks (k) go together with which MLP
width (`ctx["groups"]`, made by the program from its own planner); the unknown is which k and which set of k blocks. The rounds, in the order that keeps the most useful
answers when the time runs out:

    singles     the original and every block that may be cut, on its own (the same passes as "Every block on its own")
    width_refs  the widths of the cuts with no block removed (one pass each)
    sets_k      for every number of blocks k >= 2: the best sets by an additive prediction (the original plus the sum of the single increases), the best neighbouring runs,
                the program's own choice from the measurement; only depth (no width change yet)
    combined    the best sets of each k at the width that k's target size needs: the cuts that could really be made
    refine1/2   around the best two combined cuts: one block swapped for one of the best unused singles
    (finalists  not a round: the best few cuts, kept diverse)

`ctx` (all keys optional except nb):
    nb, pf, pl            the number of blocks and how many at each end are protected
    groups                [{"label", "target_b", "cuts": [{"k", "width" (None = unchanged), "params", "on_target"}]}]
    rounds                [{"name", "keys", "done"}]: what was planned and finished so far (the loop writes it; keeps a round the same when it is resumed)
    matrix, strategy      the measurement's block distances and "contiguous" / "separate" (the program's own choice per k)
    thorough              also every neighbouring run of every k (the older sweep)
    sets_per_k, runs_per_k, combined_per_k, swaps, refine_rounds, finalists, extra_singles   the sizes of the rounds (defaults below)"""
from __future__ import annotations

import itertools

from . import reshape
from .reshapetry import BASE_KEY, candidates, parse_key, pass_key, windows

ROUND_ORDER = ("singles", "width_refs", "sets_k", "combined", "refine1", "refine2")
ROUND_LABELS = {"singles": "each block on its own", "width_refs": "the widths", "sets_k": "sets of blocks", "combined": "the real cuts (blocks and width)",
                "refine1": "swapping one block", "refine2": "swapping one block again"}
SETS_PER_K = 6
RUNS_PER_K = 2
COMBINED_PER_K = 3
SWAPS_KEPT = 6
EXTRA_SINGLES = 3          # a set of k blocks is chosen among the best k + 3 singles
SWAP_POOL = 2              # a swap takes one of the best k + 2 unused singles
FINALISTS = 3
MAX_PER_K = 2              # at most this many finalists with the same number of blocks
REFINE_AROUND = 2
DEFAULT_REFINE_ROUNDS = 2


def _n(ctx: dict, name: str, default: int) -> int:
    return int(ctx.get(name, default))


# ------------------------------------------------------------------------------------------------------------ what the passes say

def base_loss(passes: dict):
    p = passes.get(BASE_KEY)
    return p["loss"] if p else None


def single_deltas(passes: dict, ctx: dict) -> dict:
    """{block: loss of the block on its own - the original's loss} of the blocks that were measured."""
    base = base_loss(passes)
    if base is None:
        return {}
    out = {}
    for b in candidates(int(ctx["nb"]), _pf(ctx), _pl(ctx)):
        p = passes.get(pass_key((b,)))
        if p and p.get("loss") is not None:
            out[b] = p["loss"] - base
    return out


def _pf(ctx):
    return int(ctx.get("pf", 1))


def _pl(ctx):
    return int(ctx.get("pl", 1))


def predict(blocks, deltas: dict, base: float):
    """The additive prediction: the original's loss plus the increases of the blocks on their own (None when one of them was not measured)."""
    try:
        return base + sum(deltas[b] for b in blocks)
    except KeyError:
        return None


def best_singles(deltas: dict, n: int) -> list:
    """The n blocks with the smallest increase (a tie: the lower block)."""
    return [b for b, _ in sorted(deltas.items(), key=lambda x: (x[1], x[0]))[:n]]


def _ks(ctx: dict, low: int = 1) -> list:
    movable = len(candidates(int(ctx["nb"]), _pf(ctx), _pl(ctx)))
    return sorted({int(c["k"]) for g in ctx.get("groups") or () for c in g.get("cuts") or () if low <= int(c["k"]) <= movable})


def _cuts(ctx: dict):
    for gi, g in enumerate(ctx.get("groups") or ()):
        for c in g.get("cuts") or ():
            yield gi, c


def _matches(key: str, cut: dict) -> bool:
    blocks, width = parse_key(key)
    return len(blocks) == int(cut["k"]) and width == (int(cut["width"]) if cut.get("width") else None)


def measured_for(passes: dict, cut: dict) -> list:
    """[(loss, key)] of the measured passes that are this cut (k blocks at this width), best first."""
    out = [(p["loss"], k) for k, p in passes.items() if p.get("loss") is not None and k != BASE_KEY and _matches(k, cut)]
    return sorted(out, key=lambda x: (x[0], x[1]))


# ------------------------------------------------------------------------------------------------------------ the rounds

def _singles(passes, ctx) -> list:
    return [BASE_KEY] + [pass_key((b,)) for b in candidates(int(ctx["nb"]), _pf(ctx), _pl(ctx))]


def _width_refs(passes, ctx) -> list:
    widths = sorted({int(c["width"]) for _, c in _cuts(ctx) if c.get("width") and int(c["k"]) >= 0})
    return [pass_key((), w) for w in widths]


def sets_for_k(passes: dict, ctx: dict, k: int) -> list:
    """The sets of k blocks that the sets_k round measures (depth only), best prediction first."""
    nb, pf, pl = int(ctx["nb"]), _pf(ctx), _pl(ctx)
    base = base_loss(passes)
    deltas = single_deltas(passes, ctx)
    movable = candidates(nb, pf, pl)
    if k < 2 or k > len(movable) or base is None or not deltas:
        return []
    out: dict = {}
    pool = best_singles(deltas, k + _n(ctx, "extra_singles", EXTRA_SINGLES))
    combos = [tuple(sorted(c)) for c in itertools.combinations(pool, k)]
    combos.sort(key=lambda c: (predict(c, deltas, base), c))
    for c in combos[:_n(ctx, "sets_per_k", SETS_PER_K)]:
        out[c] = True
    runs = [w for w in windows(nb, k, pf, pl) if predict(w, deltas, base) is not None]
    runs.sort(key=lambda c: (predict(c, deltas, base), c))
    for c in runs[:_n(ctx, "runs_per_k", RUNS_PER_K)]:
        out[tuple(c)] = True
    matrix = ctx.get("matrix")
    if matrix:
        try:
            out[tuple(sorted(reshape.choose_drop(matrix, k, ctx.get("strategy", "contiguous"), pf, pl)))] = True
        except reshape.ReshapeError:
            pass
    if ctx.get("thorough"):
        for w in windows(nb, k, pf, pl):
            out[tuple(w)] = True
    return list(out)


def _sets_k(passes, ctx) -> list:
    keys = []
    for k in _ks(ctx, 2):
        keys += [pass_key(s) for s in sets_for_k(passes, ctx, k)]
    return keys


def _combined(passes, ctx) -> list:
    """The cuts that could be made: for every cut of every group, the best measured depth sets of its k at its width."""
    keys = []
    for _, cut in _cuts(ctx):
        k, width = int(cut["k"]), (int(cut["width"]) if cut.get("width") else None)
        if k < 1:
            continue
        depth = [(p["loss"], key) for key, p in passes.items() if p.get("loss") is not None and key != BASE_KEY and parse_key(key)[1] is None and len(parse_key(key)[0]) == k]
        depth.sort(key=lambda x: (x[0], x[1]))
        for _, key in depth[:_n(ctx, "combined_per_k", COMBINED_PER_K)]:
            if width:
                keys.append(pass_key(parse_key(key)[0], width))
    return keys


def _refine(passes, ctx) -> list:
    """Around the best two combined cuts of the whole search: swap one block for one of the best unused singles; the best predictions are measured."""
    deltas = single_deltas(passes, ctx)
    movable = candidates(int(ctx["nb"]), _pf(ctx), _pl(ctx))
    rows = []
    for _, cut in _cuts(ctx):
        if int(cut["k"]) < 1:
            continue
        width = int(cut["width"]) if cut.get("width") else None
        for loss, key in measured_for(passes, cut):
            rows.append((loss, key, width))
    rows.sort(key=lambda x: (x[0], x[1]))
    seen, best = set(), []
    for loss, key, width in rows:
        if key not in seen:
            seen.add(key)
            best.append((loss, key, width))
    cand: dict = {}
    for loss, key, width in best[:_n(ctx, "refine_around", REFINE_AROUND)]:
        blocks = parse_key(key)[0]
        k = len(blocks)
        unused = [b for b in best_singles(deltas, len(deltas)) if b not in blocks][:k + SWAP_POOL]
        for out_b in blocks:
            for in_b in unused:
                new = tuple(sorted((set(blocks) - {out_b}) | {in_b}))
                nk = pass_key(new, width)
                if nk in passes or not all(b in movable for b in new):
                    continue
                pred = loss + deltas.get(in_b, 0.0) - deltas.get(out_b, 0.0)
                if nk not in cand or pred < cand[nk]:
                    cand[nk] = pred
    kept = sorted(cand.items(), key=lambda x: (x[1], x[0]))[:_n(ctx, "swaps", SWAPS_KEPT)]
    return [k for k, _ in kept]


_BUILD = {"singles": _singles, "width_refs": _width_refs, "sets_k": _sets_k, "combined": _combined, "refine1": _refine, "refine2": _refine}


def _round_names(ctx: dict) -> list:
    n = max(0, min(2, _n(ctx, "refine_rounds", DEFAULT_REFINE_ROUNDS)))
    return [r for r in ROUND_ORDER if not (r.startswith("refine") and int(r[-1]) > n)]


def next_round(passes: dict, ctx: dict):
    """The next round to measure: {"name", "label", "keys" (only the ones not measured yet, each once), "planned_total"} or None when the search is complete. A round that was
    planned (`ctx["rounds"]`) but not finished is returned with the same keys, a finished one is not asked again, whatever the answers say now."""
    records = {r["name"]: r for r in ctx.get("rounds") or ()}
    for name in _round_names(ctx):
        rec = records.get(name)
        if rec and rec.get("done"):
            continue
        keys = list(rec["keys"]) if rec else list(dict.fromkeys(_BUILD[name](passes, ctx)))
        todo = [k for k in keys if k not in passes]
        if rec is None and not todo:
            # nothing to measure in this round (all known already): it is complete, and the next one builds on that
            records[name] = {"name": name, "keys": keys, "done": True}
            continue
        if not todo:
            continue
        return {"name": name, "label": ROUND_LABELS[name], "keys": todo, "all_keys": keys, "planned_total": planned_total(passes, ctx, name, len(todo))}
    return None


def planned_total(passes: dict, ctx: dict, current: str | None = None, current_todo: int | None = None) -> int:
    """About how many passes the whole search measures from the start (an estimate: the later rounds depend on the answers of the earlier ones)."""
    movable = len(candidates(int(ctx["nb"]), _pf(ctx), _pl(ctx)))
    ks2 = _ks(ctx, 2)
    widths = len({int(c["width"]) for _, c in _cuts(ctx) if c.get("width")})
    per_k = _n(ctx, "sets_per_k", SETS_PER_K) + _n(ctx, "runs_per_k", RUNS_PER_K) + (1 if ctx.get("matrix") else 0)
    cuts = sum(1 for _, c in _cuts(ctx) if int(c["k"]) >= 1 and c.get("width"))
    total = 1 + movable + widths + per_k * len(ks2) + cuts * _n(ctx, "combined_per_k", COMBINED_PER_K) + _round_refine_count(ctx) * _n(ctx, "swaps", SWAPS_KEPT)
    if ctx.get("thorough"):
        total += sum(max(0, movable - k + 1) for k in ks2)
    return max(total, len(passes))


def _round_refine_count(ctx: dict) -> int:
    return len([r for r in _round_names(ctx) if r.startswith("refine")])


def round_plan(passes: dict, ctx: dict) -> list:
    """[{"name", "label", "count"}]: the rounds and how many passes each needs now (the later ones as estimates), for the plan line before the search starts."""
    movable = len(candidates(int(ctx["nb"]), _pf(ctx), _pl(ctx)))
    per_k = _n(ctx, "sets_per_k", SETS_PER_K) + _n(ctx, "runs_per_k", RUNS_PER_K) + (1 if ctx.get("matrix") else 0)
    cuts = sum(1 for _, c in _cuts(ctx) if int(c["k"]) >= 1 and c.get("width"))
    est = {"singles": 1 + movable, "width_refs": len({int(c["width"]) for _, c in _cuts(ctx) if c.get("width")}), "sets_k": per_k * len(_ks(ctx, 2)),
           "combined": cuts * _n(ctx, "combined_per_k", COMBINED_PER_K), "refine1": _n(ctx, "swaps", SWAPS_KEPT), "refine2": _n(ctx, "swaps", SWAPS_KEPT)}
    if ctx.get("thorough"):
        est["sets_k"] += sum(max(0, movable - k + 1) for k in _ks(ctx, 2))
    return [{"name": n, "label": ROUND_LABELS[n], "count": est[n]} for n in _round_names(ctx)]


def affordable(rounds: list, seconds_per_pass: float, budget_s: float, load_s: float = 0.0) -> dict:
    """How much of the plan fits the time: {"rounds": whole rounds that fit, "passes": passes that fit, "seconds": their time with the loading}. Pure arithmetic."""
    left = max(0.0, budget_s - load_s)
    sp = max(float(seconds_per_pass), 1e-9)
    n_passes = int(left // sp)
    whole, used = 0, 0
    for r in rounds:
        if used + r["count"] <= n_passes:
            used += r["count"]
            whole += 1
        else:
            break
    total = sum(r["count"] for r in rounds)
    fit = min(n_passes, total)
    return {"rounds": whole, "passes": fit, "seconds": load_s + fit * sp, "complete": fit >= total}


# ------------------------------------------------------------------------------------------------------------ the answer

def ranking(passes: dict, ctx: dict) -> dict:
    """{group index: [keys of the measured cuts of that group, lowest loss first]}"""
    out: dict = {}
    for gi, g in enumerate(ctx.get("groups") or ()):
        rows = {}
        for cut in g.get("cuts") or ():
            for loss, key in measured_for(passes, cut):
                rows[key] = loss
        out[str(gi)] = [k for k, _ in sorted(rows.items(), key=lambda x: (x[1], x[0]))]
    return out


def _differs_by_one(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and len(set(a) & set(b)) >= len(a) - 1


def finalists(passes: dict, ctx: dict, n: int | None = None) -> dict:
    """{group index: [the best n keys, diverse]}: at most two with the same number of blocks, and a set that differs from a better one by a single block gives way to the next."""
    n = n or _n(ctx, "finalists", FINALISTS)
    out: dict = {}
    for gi, keys in ranking(passes, ctx).items():
        chosen: list = []
        per_k: dict = {}
        for key in keys:
            blocks, _ = parse_key(key)
            if not blocks:
                continue
            if per_k.get(len(blocks), 0) >= MAX_PER_K or any(_differs_by_one(blocks, parse_key(c)[0]) for c in chosen):
                continue
            chosen.append(key)
            per_k[len(blocks)] = per_k.get(len(blocks), 0) + 1
            if len(chosen) >= n:
                break
        if len(chosen) < n:                # too few different ones: fill up with the best of the rest
            for key in keys:
                if key not in chosen and parse_key(key)[0]:
                    chosen.append(key)
                    if len(chosen) >= n:
                        break
        out[gi] = chosen
    return out
