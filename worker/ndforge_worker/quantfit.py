"""Choosing a GGUF quantization for a target size from MEASURED per-tensor error ("Mixed: measured fit to a size", docs/MIXED_QUANTS.md "Fit to a size (measured)").

quantmeasure.py measures, for every tensor and every type of a ladder (Q2_K .. Q6_K, Q8_0), how much importance-weighted error that type leaves in YOUR weights. This module turns the
measures into a plan: the type of every tensor that makes the whole file fit a size with the least total error, written as llama-quantize rules.

    allocate     the budgeted choice (pure, deterministic): lower convex hull of (bytes, error) of every tensor, then the steepest error-per-byte upgrades first (a port of the idea of
                 quant-convert-gui's kquant_plan.py: same author, MIT)
    plan_for     the whole plan for a model header: pins, fixed bytes, budget, rules, verified against the rule semantics of llama-quantize
    rules        the plan as ordered `(regex, type)` rules, checked by running every tensor through mixedquant.tensor_type (first match wins)
    report       the numbers the page shows: the plan, uniform types, the plain Q4_K_M / Q5_K_M as llama.cpp's mixture is approximated here, types chosen, bits by role and layer

What is fixed and not planned: tensors llama-quantize never quantizes (norms, 1-D, `ssm_conv1d`, the router, `ssm_a` / `dt`) keep their type; `output.weight` is pinned (Q6_K, Q8_0 when
the target is at least the size of Q6_K); `token_embd.weight` follows the nearest plain size (Q4_K up to the size of Q4_K_M, Q5_K up to Q5_K_M, Q6_K above); a model with tied embeddings
gives the embedding the output type. The vision tower lives in a separate mmproj file and is not counted. Sizes are exact (`numel x bits per weight / 8`), 1 GB is 10^9 bytes.

The caveats of the metric (quantmeasure.CAVEATS) belong to every plan: a diagonal approximation, equal harm across layers assumed, about 1 to 3 percent of the rows looked at, and with the
reference engine a rounding that is not the importance-matrix rounding. The judge is Compare quants (KL divergence), never the predicted error."""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field

from . import ggufinfo, mixedquant, quantmeasure

ID = "mixed_measured"
LABEL = "Mixed: measured fit to a size"
DESCRIPTION = "Mixed precision (our own recipe): each tensor's error measured on your weights, weighted by the importance matrix"
LADDER = quantmeasure.RUNGS
DEFAULT_FLOOR, DEFAULT_CEILING = "Q3_K", "Q8_0"
# the types a rule may name, lowest bits first (bits per weight: 2.625, 3.4375, 4.5, 4.5, 5.5, 5.5, 6.0, 6.5625, 8.5, 16); ties by this order
ORDER = ("Q2_K", "Q3_K", "Q4_0", "Q4_K", "Q5_0", "Q5_K", "Q5_1", "Q6_K", "Q8_0", "F16")
PLAIN_SIZES = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
MARGIN_BYTES = 64 << 10              # the budget keeps this and 32 bytes per tensor free for the header and the alignment of the output file


class PlanError(ValueError):
    """The plan cannot be made or does not verify; the text is plain language."""


@dataclass
class Plan:
    assign: dict                                    # planned tensor -> the type it really gets (effective)
    rung: dict                                      # planned tensor -> the rung of the ladder that was chosen
    planned_bytes: int
    err: float                                      # weighted squared error of the planned tensors
    fits: bool
    min_bytes: int                                  # the smallest the whole file can be (floor everywhere)
    max_bytes: int                                  # the largest (ceiling everywhere)
    budget: int
    fixed: int                                      # everything that is not planned: header, kept tensors, pinned tensors
    pins: dict = field(default_factory=dict)        # {"embd", "output", "tied"}
    pinned_assign: dict = field(default_factory=dict)       # pinned tensors -> type (in the whole-model total, not in `assign`)
    notes: list = field(default_factory=list)

    @property
    def total(self) -> int:
        """The size of the file as mixedquant.estimate counts it (header + tensors, no margin)."""
        return int(self.fixed + self.planned_bytes)

    def to_dict(self) -> dict:
        return {"assign": self.assign, "rung": self.rung, "planned_bytes": self.planned_bytes, "err": self.err, "fits": self.fits, "min_bytes": self.min_bytes, "max_bytes": self.max_bytes,
                "budget": self.budget, "fixed": self.fixed, "total": self.total, "pins": self.pins, "pinned_assign": self.pinned_assign, "notes": self.notes}


# ------------------------------------------------------------------------------------------------------------------ the allocator

def _hull(points):
    """points: [(bytes, err, type)] -> the lower convex hull with strictly falling error, bytes ascending (kquant_plan._hull)."""
    pts, best = [], float("inf")
    for b, e, t in sorted(points):
        if e < best:                                   # more bytes must buy less error
            pts.append((b, e, t))
            best = e
    hull = []
    for p in pts:
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (a[1] - b[1]) * (p[0] - b[0]) < (b[1] - p[1]) * (b[0] - a[0]):      # b lies above the segment a..p: the rate must fall along the hull
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def allowed_rungs(floor: str, ceiling: str) -> tuple:
    if floor not in LADDER or ceiling not in LADDER:
        raise PlanError(f"The lowest and highest types must be among {', '.join(LADDER)}.")
    lo, hi = LADDER.index(floor), LADDER.index(ceiling)
    if lo > hi:
        raise PlanError(f"The lowest type ({floor}) is above the highest ({ceiling}).")
    return LADDER[lo:hi + 1]


def allocate(measures: dict, budget: int, fixed: int = 0, floor: str = DEFAULT_FLOOR, ceiling: str = DEFAULT_CEILING, exclude=()) -> Plan:
    """Chooses a type for every measured tensor (but those in `exclude`: pinned ones) minimising the total weighted error with `fixed + sum(bytes) <= budget`. The rungs between `floor`
    and `ceiling` are mapped to the types the tensors really get (a row length that does not fit a block falls back, two rungs that land on one type count once). Deterministic: the
    steepest error-per-byte upgrade first, ties by the tensor name; an upgrade that does not fit is skipped and a smaller one may still be taken. Below the minimum `fits` is False and
    the floor is returned; above the maximum every tensor gets the ceiling."""
    allowed = allowed_rungs(floor, ceiling)
    skip = set(exclude)
    hulls = {}
    for name, rec in measures["tensors"].items():
        if name in skip:
            continue
        pts = [(rec["types"][r]["bytes"], rec["types"][r]["err"], r) for r in allowed if r in rec["types"]]
        if not pts:
            raise PlanError(f"{name} was not measured at any type between {floor} and {ceiling}.")
        hulls[name] = _hull(pts)
    level = {n: 0 for n in hulls}
    planned = sum(h[0][0] for h in hulls.values())
    min_bytes = fixed + planned
    max_bytes = fixed + sum(h[-1][0] for h in hulls.values())
    fits = min_bytes <= budget
    notes = []
    if fits:
        heap = []
        for n, h in hulls.items():
            if len(h) > 1:
                heap.append((-(h[0][1] - h[1][1]) / (h[1][0] - h[0][0]), n, 0))
        heapq.heapify(heap)
        while heap:
            _, n, lv = heapq.heappop(heap)
            h = hulls[n]
            extra = h[lv + 1][0] - h[lv][0]
            if fixed + planned + extra > budget:
                continue                                # too big for what is left; a smaller upgrade of another tensor may still fit
            planned += extra
            level[n] = lv + 1
            if lv + 2 < len(h):
                heapq.heappush(heap, (-(h[lv + 1][1] - h[lv + 2][1]) / (h[lv + 2][0] - h[lv + 1][0]), n, lv + 1))
        if fixed + planned >= max_bytes - 1:
            notes.append(f"The size asked for is more than the highest type ({ceiling}) can use: every tensor got it.")
    else:
        notes.append(f"Even the lowest type ({floor}) needs {min_bytes / 1e9:.2f} GB, more than the {budget / 1e9:.2f} GB asked for; the lowest types are given.")
    rung = {n: hulls[n][level[n]][2] for n in hulls}
    assign = {n: measures["tensors"][n]["types"][rung[n]]["eff"] for n in hulls}
    err = sum(hulls[n][level[n]][1] for n in hulls)
    return Plan(assign=assign, rung=rung, planned_bytes=int(planned), err=float(err), fits=fits, min_bytes=int(min_bytes), max_bytes=int(max_bytes), budget=int(budget), fixed=int(fixed), notes=notes)


# ------------------------------------------------------------------------------------------------------------------ pins, fixed bytes, budget

def _params(info: dict) -> int:
    return sum(ggufinfo.numel(t) for t in info["tensors"])


def plain_bytes(info: dict, ftype: str) -> float:
    """About how big the plain file of that size is for this model: parameters x the table of llama.cpp's own bits per weight."""
    return _params(info) * mixedquant.PLAIN_BPW[ftype] / 8


def is_tied(info: dict) -> bool:
    return "output.weight" not in {t["name"] for t in info["tensors"]}


def pins(target_bytes: float, info: dict) -> dict:
    """{"embd", "output", "tied"}: the types of the two tensors the matrix does not measure well (the output tensor is only in a matrix made with --process-output; the embedding is a lookup)."""
    output = "Q8_0" if target_bytes >= plain_bytes(info, "Q6_K") else "Q6_K"
    if target_bytes <= plain_bytes(info, "Q4_K_M"):
        embd = "Q4_K"
    elif target_bytes <= plain_bytes(info, "Q5_K_M"):
        embd = "Q5_K"
    else:
        embd = "Q6_K"
    return {"embd": embd, "output": output, "tied": is_tied(info)}


def _src_bytes(t: dict) -> float:
    name = ggufinfo.type_name(t["type"])
    return ggufinfo.numel(t) * (ggufinfo.BPW[name] if name in ggufinfo.BPW else 16) / 8


def pinned_types(info: dict, pin: dict) -> dict:
    """{tensor name: type} of the pinned tensors present in the model (with the row length fallback)."""
    out = {}
    for t in info["tensors"]:
        if not mixedquant.quantizable(t["name"], t["dims"]):
            continue
        if t["name"] == "output.weight":
            out[t["name"]] = mixedquant.effective_type(pin["output"], t["dims"][0])
        elif t["name"] == "token_embd.weight":
            out[t["name"]] = mixedquant.effective_type(pin["output"] if pin["tied"] else pin["embd"], t["dims"][0])
    return out


def fixed_bytes(info: dict, pin: dict, kept: dict | None = None) -> int:
    """The header plus everything that is not planned: the tensors that stay as they are, the pinned ones, and `kept` ({name: type}: tensors the measurement had to leave out)."""
    kept = kept or {}
    pinned = pinned_types(info, pin)
    total = info["data_start"]
    for t in info["tensors"]:
        n = ggufinfo.numel(t)
        if not mixedquant.quantizable(t["name"], t["dims"]):
            total += _src_bytes(t)
        elif t["name"] in pinned:
            total += n * ggufinfo.BPW[pinned[t["name"]]] / 8
        elif t["name"] in kept:
            total += n * ggufinfo.BPW[kept[t["name"]]] / 8
    return int(total)


def margin(info: dict) -> int:
    return MARGIN_BYTES + len(info["tensors"]) * 32


def base_ftype(avg_bpw: float) -> str:
    """The plain size nearest in bits per weight; it only labels `general.file_type` of the file (every planned tensor has a rule)."""
    return min(mixedquant.PLAIN_BPW, key=lambda k: (abs(mixedquant.PLAIN_BPW[k] - avg_bpw), k))


def plan_for(info: dict, measures: dict, target_bytes: float, floor: str = DEFAULT_FLOOR, ceiling: str = DEFAULT_CEILING) -> Plan:
    """The plan for the header `info` of the 16-bit GGUF: pins by the target, tensors the measurement skipped (not finite numbers) are kept at Q6_K, the budget is the target minus
    the margin of the header and alignment. Raises PlanError when a quantizable tensor was not measured (the cache does not belong to this file)."""
    pin = pins(target_bytes, info)
    pinned = pinned_types(info, pin)
    skipped = set(measures.get("skipped") or [])
    kept = {}
    by = {t["name"]: t for t in info["tensors"]}
    for name in sorted(skipped):
        if name in by and name not in pinned:
            kept[name] = mixedquant.effective_type("Q6_K", by[name]["dims"][0])
    for t in info["tensors"]:
        if mixedquant.quantizable(t["name"], t["dims"]) and t["name"] not in pinned and t["name"] not in kept and t["name"] not in measures["tensors"]:
            raise PlanError(f"{t['name']} was not measured: the measurements do not belong to this model file. Measure again.")
    fixed = fixed_bytes(info, pin, kept)
    plan = allocate(measures, int(target_bytes) - margin(info), fixed, floor, ceiling, exclude=set(pinned) | set(kept))
    plan.pins = pin
    plan.pinned_assign = {**pinned, **kept}
    if kept:
        plan.notes.append(f"{len(kept)} tensor(s) had numbers that are not finite and were not measured; they are kept at Q6_K.")
    return plan


# ------------------------------------------------------------------------------------------------------------------ rules

def preset_p(plan: Plan, avg_bpw: float | None = None) -> dict:
    """The preset-like dict mixedquant.estimate and tensor_type want: {"base", "embd", "output"}."""
    avg = avg_bpw if avg_bpw is not None else 4.9
    return {"base": base_ftype(avg), "embd": plan.pins["embd"], "output": plan.pins["output"]}


def rules(plan: Plan, info: dict | None = None) -> list:
    """The plan as ordered `(regex, type)` rules: one per (tensor role, type) naming its layers, tensors outside the blocks one each. With `info`, every tensor of the model is run through
    mixedquant.tensor_type (the first matching rule wins, as in llama-quantize) and must come out as the plan says; a mismatch raises PlanError. The pinned tensors are not in the
    rules (they have their own flags)."""
    rs = mixedquant._compress({**plan.assign, **{k: v for k, v in plan.pinned_assign.items() if k not in ("output.weight", "token_embd.weight")}}, order=ORDER)
    if info is not None:
        verify(plan, rs, info)
    return rs


def verify(plan: Plan, rs: list, info: dict) -> None:
    p = preset_p(plan, None)
    tied = is_tied(info)
    want = {**plan.assign, **plan.pinned_assign}
    for t in info["tensors"]:
        got, why = mixedquant.tensor_type(t["name"], t["dims"], ggufinfo.type_name(t["type"]), rs, p, tied)
        if t["name"] in want:
            if got != want[t["name"]]:
                raise PlanError(f"Internal check failed: {t['name']} would be written as {got}, not {want[t['name']]} (rule {why}).")
        elif why not in ("kept", "embd", "output"):
            raise PlanError(f"Internal check failed: {t['name']} is named by no part of the plan (it would follow the base type).")
    est = mixedquant.estimate(info, rs, p)
    if est["unmatched"] or est["bytes"] != plan.total:
        raise PlanError(f"Internal check failed: the rules make {est['bytes']} bytes with {est['unmatched']} unmatched tensors, the plan says {plan.total}.")


def type_file_text(rs: list) -> str:
    """The `--tensor-type-file` text: one `regex=type` per line."""
    return "".join(f"{rx}={t}\n" for rx, t in rs)


def flags(plan: Plan, rs: list, type_file=None) -> list:
    """The options of llama-quantize before the file names: output and embedding types, then the rules as `--tensor-type-file PATH` (when `type_file` is given) or repeated `--tensor-type`."""
    out = ["--output-tensor-type", plan.pins["output"], "--token-embedding-type", plan.pins["embd"]]
    if type_file is not None:
        return out + ["--tensor-type-file", str(type_file)]
    for rx, t in rs:
        out += ["--tensor-type", f"{rx}={t}"]
    return out


# ------------------------------------------------------------------------------------------------------------------ simulated plain sizes

def _more_bits(i: int, n: int) -> bool:
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def simulate_plain(info: dict, ftype: str) -> dict:
    """{tensor: rung}: an APPROXIMATION of the types llama-quantize gives to a plain Q4_K_M or Q5_K_M file (the mixture of llama-quant.cpp: the base type everywhere, Q6_K for attn_v and ffn_down
    in the layers `use_more_bits` names (first eighth, last eighth, every third in between), Q6_K for the output tensor and for the fused attention projection of Q5_K_M, Q5_K for it in Q4_K_M).
    Only tensors the ladder measures are listed. The engine `llama` reads the real assignment from the file instead."""
    base = {"Q4_K_M": "Q4_K", "Q5_K_M": "Q5_K"}[ftype]
    layers = sorted({int(m.group(1)) for t in info["tensors"] for m in [re.match(r"blk\.(\d+)\.", t["name"])] if m and not re.search(r"nextn", t["name"])})
    n = len(layers)
    out = {}
    for t in info["tensors"]:
        name = t["name"]
        if not mixedquant.quantizable(name, t["dims"]):
            continue
        ty = base
        m = re.match(r"blk\.(\d+)\.", name)
        i = layers.index(int(m.group(1))) if m and int(m.group(1)) in layers else None
        if name in ("output.weight",):
            ty = "Q6_K"
        elif name == "token_embd.weight":
            ty = base if not is_tied(info) else "Q6_K"
        elif i is not None and re.search(r"\.(attn_v|ffn_down)\.weight$", name) and _more_bits(i, n):
            ty = "Q6_K"
        elif re.search(r"\.attn_qkv\.weight$", name):
            ty = "Q5_K" if ftype == "Q4_K_M" else "Q6_K"
        out[name] = ty
    return out


# ------------------------------------------------------------------------------------------------------------------ the report

def _role(name: str) -> str:
    m = re.match(r"blk\.\d+\.(.+?)(?:\.weight)?$", name)
    return m.group(1) if m else name


def _layer(name: str):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _rel(err: float, norm2: float) -> float:
    return math.sqrt(err / norm2) if norm2 > 0 else 0.0


def uniform(measures: dict, info: dict, rung: str, base_fixed: int) -> tuple:
    """(file bytes, weighted error) if every measured tensor used `rung` (a row length that does not fit falls back); `base_fixed` is the header plus the tensors that are never quantized."""
    b = base_fixed + sum(r["types"][rung]["bytes"] for r in measures["tensors"].values() if rung in r["types"])
    e = sum(r["types"][rung]["err"] for r in measures["tensors"].values() if rung in r["types"])
    return int(b), float(e)


def report(info: dict, measures: dict, plan: Plan, simulated: dict | None = None) -> dict:
    """What the page shows. {"rows": [{"id", "label", "bytes", "gb", "bpw", "error"}] (the plan, the uniform rungs, the simulated plain sizes: `error` is sqrt(sum D / sum N) over all measured
    tensors), "types": {type: {"tensors", "params"}}, "bpw_by_role", "bpw_by_layer", "notes", "caveats", "norm2"}. `simulated` is {"Q4_K_M": {tensor: rung}, ...} (default: simulate_plain)."""
    tensors = measures["tensors"]
    params = _params(info)
    nonq = info["data_start"] + sum(_src_bytes(t) for t in info["tensors"] if not mixedquant.quantizable(t["name"], t["dims"]))
    norm2 = sum(r["norm2_w"] for r in tensors.values())
    by = {t["name"]: t for t in info["tensors"]}
    rows = []

    def row(rid, label, total, err):
        rows.append({"id": rid, "label": label, "bytes": int(total), "gb": round(total / 1e9, 3), "bpw": round(total * 8 / params, 3) if params else None, "error": round(_rel(err, norm2), 5)})

    pe = 0.0
    for name, ty in plan.pinned_assign.items():            # pinned and kept tensors: their measured error at the type they get
        rec = tensors.get(name)
        rung = next((r for r in LADDER if rec and r in rec["types"] and rec["types"][r]["eff"] == ty), None)
        if rung:
            pe += rec["types"][rung]["err"]
    row("plan", "Measured plan", plan.total, plan.err + pe)
    for rung in LADDER[1:5]:
        b, e = uniform(measures, info, rung, int(nonq))
        row("uniform_" + rung.lower(), f"Every tensor {rung}", b, e)
    sims = simulated if simulated is not None else {ft: simulate_plain(info, ft) for ft in ("Q4_K_M", "Q5_K_M")}
    for ft, assign in sims.items():
        b, e = int(nonq), 0.0
        for name, rec in tensors.items():
            rung = assign.get(name, "Q4_K")
            ty = rec["types"].get(rung) or rec["types"]["Q4_K"]
            b += ty["bytes"]
            e += ty["err"]
        row("plain_" + ft.lower(), f"Plain {ft} (as llama.cpp's mixture is approximated here)", b, e)
    types, role_bits, layer_bits = {}, {}, {}
    allx = {**plan.assign, **plan.pinned_assign}
    for name, ty in allx.items():
        n = ggufinfo.numel(by[name]) if name in by else 0
        e = types.setdefault(ty, {"tensors": 0, "params": 0})
        e["tensors"] += 1
        e["params"] += n
        for key, table in ((_role(name), role_bits), (_layer(name), layer_bits)):
            if key is None:
                continue
            a = table.setdefault(key, [0.0, 0])
            a[0] += n * ggufinfo.BPW[ty]
            a[1] += n
    return {"rows": rows, "types": types, "bpw_by_role": {k: round(v[0] / v[1], 3) for k, v in sorted(role_bits.items()) if v[1]},
            "bpw_by_layer": {str(k): round(v[0] / v[1], 3) for k, v in sorted(layer_bits.items()) if v[1]}, "notes": list(plan.notes) + list(measures.get("notes") or []),
            "caveats": quantmeasure.CAVEATS, "norm2": norm2}
