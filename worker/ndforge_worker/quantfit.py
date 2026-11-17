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
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import ggufinfo, mixedquant, protocol, quantmeasure

ID = "mixed_measured"
LABEL = "Mixed: measured fit to a size"
DESCRIPTION = "Mixed precision (our own recipe): each tensor's error measured on your weights, weighted by the importance matrix"
LADDER = quantmeasure.RUNGS
DEFAULT_FLOOR, DEFAULT_CEILING = "Q3_K", "Q8_0"
# the types a rule may name, lowest bits first (bits per weight: 2.625, 3.4375, 4.5, 4.5, 5.5, 5.5, 6.0, 6.5625, 8.5, 16); ties by this order
ORDER = quantmeasure.RULE_ORDER
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
    tensors), "types": {type: {"tensors", "params"}}, "bpw_by_role", "bpw_by_layer", "notes", "caveats", "norm2"}. `simulated` is {"Q4_K_M": {tensor: rung or {"eff", "err", "bytes"}}, ...}: the llama engine gives the exact assignment of llama-quantize's own plain file, else simulate_plain approximates it."""
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
            pick = assign.get(name, "Q4_K")
            ty = pick if isinstance(pick, dict) else (rec["types"].get(pick) or rec["types"]["Q4_K"])        # exact (read from llama-quantize's own file) or approximated (a rung)
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


# ------------------------------------------------------------------------------------------------------------------ the jobs

class QuantFitError(Exception):
    """A failure with a plain-language text for the person (the job's error code is `quantfit_failed`)."""


# what is assumed, not measured, in the estimate of the minutes (said so wherever it is shown)
ASSUMED_LLAMA_SPEEDUP = 5.0
ASSUMED_READ_MB_PER_S = 200.0
ASSUMED_Q8_VS_Q6 = 8.0
ASSUMED_THREAD_EFFICIENCY = 0.8


def size_range(info: dict, floor: str, ceiling: str, target_bytes: float | None = None) -> tuple:
    """(smallest, largest) size in bytes of the whole file with every planned tensor at `floor` / at `ceiling` (header, kept tensors, pinned tensors and the margin included)."""
    allowed_rungs(floor, ceiling)
    pin = pins(target_bytes if target_bytes else plain_bytes(info, "Q4_K_M"), info)
    pinned = pinned_types(info, pin)
    fixed = fixed_bytes(info, pin)
    lo = hi = 0.0
    for t in info["tensors"]:
        if mixedquant.quantizable(t["name"], t["dims"]) and t["name"] not in pinned:
            n = ggufinfo.numel(t)
            lo += n * ggufinfo.BPW[mixedquant.effective_type(floor, t["dims"][0])] / 8
            hi += n * ggufinfo.BPW[mixedquant.effective_type(ceiling, t["dims"][0])] / 8
    m = margin(info)
    return int(fixed + lo + m), int(fixed + hi + m)


def _llama_quantize():
    from . import llamacpp
    _, quantize = llamacpp.find_llama_cpp()
    return quantize


def _engine_for(requested: str, quantize) -> str:
    return "llama" if requested in ("auto", "llama", None, "") and quantize is not None else "reference"


def estimate_job(job: dict) -> dict:
    """Job `quantfit_estimate` {"gguf", "matrix": file | None, "target_gb": optional, "sample_rows", "engine", "floor", "ceiling"}: what the measured fit will do, from the header only (about a second).
    {"params", "sampled_weights", "read_bytes", "disk_need", "min_gb", "max_gb", "plain_estimates": {ftype: gb}, "engine", "llama_quantize", "llama_version", "tensor_type_file", "minutes_estimate",
    "minutes_note", "speed", "matrix_ok", "notes"}."""
    from . import ggufrows, llamacpp, quantmeasure
    try:
        src = ggufrows.open_source(job.get("gguf") or "")
    except ggufrows.SourceError as exc:
        raise QuantFitError(str(exc))
    n_rows = int(job.get("sample_rows") or quantmeasure.DEFAULT_SAMPLE_ROWS)
    floor, ceiling = job.get("floor") or DEFAULT_FLOOR, job.get("ceiling") or DEFAULT_CEILING
    try:
        allowed_rungs(floor, ceiling)
    except PlanError as exc:
        raise QuantFitError(str(exc))
    picks = ggufrows.sample_plan(src, n_rows, int(job.get("seed") or 0))
    by = {t["name"]: t for t in src["tensors"]}
    weights = sum(len(ggufrows.flat_rows(p)) * p["cols"] for p in picks.values())
    read = sum(len(ggufrows.flat_rows(p)) * p["cols"] * ggufrows.value_bytes(by[n]) for n, p in picks.items())
    quantize = _llama_quantize()
    engine = _engine_for(job.get("engine"), quantize)
    notes = []
    if job.get("engine") == "llama" and quantize is None:
        notes.append("llama-quantize was not found, so the numpy engine will measure (its rounding is not the importance-matrix rounding of an export).")
    matrix_ok = None
    if job.get("matrix"):
        try:
            quantmeasure.importance(job["matrix"], src)
            matrix_ok = True
        except quantmeasure.MeasureError as exc:
            matrix_ok = False
            notes.append(str(exc))
    else:
        notes.append("No importance matrix was chosen: the measurement would count every input channel the same. Choose a matrix (an adapted one for a cut model).")
    target = float(job["target_gb"]) * 1e9 if job.get("target_gb") else None
    lo, hi = size_range(src, floor, ceiling, target)
    speed = quantmeasure.speed_probe()
    threads = quantmeasure.default_threads() * ASSUMED_THREAD_EFFICIENCY
    ref_s = 0.0
    for rung in quantmeasure.RUNGS:
        v = speed.get(rung) or speed["Q6_K"] * ASSUMED_Q8_VS_Q6
        ref_s += weights / (v * max(threads, 1.0))
    read_s = read / (ASSUMED_READ_MB_PER_S * 1e6)
    llama_s = ref_s / len(quantmeasure.RUNGS) / ASSUMED_LLAMA_SPEEDUP * (len(quantmeasure.RUNGS) + 3)
    chosen = (llama_s if engine == "llama" else ref_s) + read_s
    version = ""
    if engine == "llama":
        from . import imatrix
        version = imatrix.tool_version(quantize)
    return {"params": ggufinfo.n_parameters(src), "sampled_weights": weights, "read_bytes": read, "disk_need": int(read * 2.2 * 2.2) if engine == "llama" else 0,
            "min_gb": round(lo / 1e9, 3), "max_gb": round(hi / 1e9, 3), "plain_estimates": {ft: round(plain_bytes(src, ft) / 1e9, 2) for ft in PLAIN_SIZES},
            "engine": engine, "llama_quantize": str(quantize) if quantize else None, "llama_version": version,
            "tensor_type_file": bool(quantize and llamacpp.quantize_supports(quantize, "--tensor-type-file")),
            "minutes_estimate": {"reference": round((ref_s + read_s) / 60, 1), "llama": round((llama_s + read_s) / 60, 1), "chosen": round(chosen / 60, 1)},
            "minutes_note": f"About {max(1, round(chosen / 60))} minutes (an estimate: the numpy speed was measured just now; the {ASSUMED_LLAMA_SPEEDUP:.0f} times faster llama engine and a disk "
                            f"read speed of {ASSUMED_READ_MB_PER_S:.0f} MB/s are assumptions).",
            "speed": {k: round(v) for k, v in speed.items()}, "matrix_ok": matrix_ok, "notes": notes}


def run_estimate(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(estimate_job(job))
    return 0


def _matrix_path(job_matrix, export_matrix):
    return str(job_matrix) if job_matrix else (str(export_matrix) if export_matrix else None)


def measure_and_plan(gguf, matrix, cfg: dict, quantize=None, im=None, notes=None) -> dict:
    """Measure (or load from the cache) and plan. `cfg`: {"target_gb", "floor", "ceiling", "sample_rows", "engine", "cache_dir", "seed"}. Returns {"src", "measures", "plan", "rules", "report",
    "key", "engine", "notes"}. Raises QuantFitError (plain text), quantmeasure.Stopped, quantmeasure.MeasureError, ggufrows.SourceError, PlanError."""
    from . import ggufrows, imatrix, quantmeasure
    notes = notes if notes is not None else []
    target = float(cfg.get("target_gb") or 0) * 1e9
    if target <= 0:
        raise QuantFitError("Give the size you want, in GB (for example 10.6).")
    floor, ceiling = cfg.get("floor") or DEFAULT_FLOOR, cfg.get("ceiling") or DEFAULT_CEILING
    src = ggufrows.open_source(gguf)
    n_rows = int(cfg.get("sample_rows") or quantmeasure.DEFAULT_SAMPLE_ROWS)
    seed = int(cfg.get("seed") or 0)
    engine = _engine_for(cfg.get("engine"), quantize)
    imp, imp_notes = quantmeasure.importance(matrix, src)
    notes += imp_notes
    version = imatrix.tool_version(quantize) if engine == "llama" else ""
    key = quantmeasure.cache_key(quantmeasure.fingerprint_model(src["path"], src), quantmeasure.fingerprint_file(matrix) if matrix else "none", n_rows, seed, engine, version)
    cache = Path(cfg.get("cache_dir") or Path(gguf).parent / "quantfit") / key
    opts = {"sample_rows": n_rows, "seed": seed, "key": key, "engine": engine, "rungs": quantmeasure.RUNGS, "threads": cfg.get("threads")}
    total = len([t for t in src["tensors"] if ggufrows.is_sampled(t)])

    def progress(done, n, name):
        protocol.progress("quantfit", 0.7 * min(done / max(n or total, 1), 1.0), f"Measuring ({done} of {n or total}): {name}")
    protocol.progress("quantfit", 0.0, "Reading the weights")
    with protocol.timed("measure", engine=engine, rows=n_rows):
        measures = quantmeasure.measure_any(src, imp, opts, cache, quantize=quantize if engine == "llama" else None, matrix=matrix, im=im, progress=progress)
    with protocol.timed("allocate"):
        plan = plan_for(src, measures, target, floor, ceiling)
        rs = rules(plan, src)
        sims = measures.get("simulated") or None
        rep = report(src, measures, plan, sims)
    notes += [n for n in measures.get("notes", []) if n not in notes]
    return {"src": src, "measures": measures, "plan": plan, "rules": rs, "report": rep, "key": key, "engine": measures.get("engine", engine), "notes": notes}


def _plain_neighbours(src: dict, target: float) -> list:
    """The plain sizes just below and just above the target: [("Q4_K_M", ...), ...] by the size table."""
    sizes = sorted((plain_bytes(src, ft), ft) for ft in PLAIN_SIZES)
    below = [ft for b, ft in sizes if b <= target]
    above = [ft for b, ft in sizes if b > target]
    return ([below[-1]] if below else []) + ([above[0]] if above else [])


def _save_plan(cache_dir: Path, target_gb, result: dict) -> None:
    from . import fsutil
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / "plan.tmp"
    tmp.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    fsutil.replace_retry(tmp, cache_dir / f"plan-{float(target_gb):g}.json")


def run_job(job: dict) -> dict:
    """Job `quantfit`: {"gguf" (16-bit), "matrix" (file | None), "target_gb", "floor", "ceiling", "sample_rows", "engine", "cache_dir", "stop_file", "make": bool, "neighbours": bool, "out_dir"}.
    Measures (resuming from the cache), plans, saves `plan-<gb>.json` next to the measures and, with `make`, writes `model-mixed_measured.gguf` (and with `neighbours` the plain sizes just
    below and above the target) into `out_dir`. Result: {"plan", "report", "rules", "flags", "files": [{"path", "label", "group", "matrix", "mixed"}], "measure_key", "engine", "notes", "fits"}."""
    from . import exportjob, ggufrows, imatrix, llamacpp, quantmeasure
    quantize = _llama_quantize()
    gguf = job.get("gguf") or ""
    matrix = job.get("matrix") or None
    im = {"stop_file": job.get("stop_file")} if job.get("stop_file") else {}
    if job.get("make") and quantize is None:
        raise llamacpp.LlamaCppMissing(llamacpp.not_found_text("the file cannot be made (llama-quantize makes it)"))
    notes: list = []
    try:
        r = measure_and_plan(gguf, matrix, job, quantize, im, notes)
    except ggufrows.SourceError as exc:
        raise QuantFitError(str(exc))
    except quantmeasure.MeasureError as exc:
        raise QuantFitError(str(exc))
    except PlanError as exc:
        raise QuantFitError(str(exc))
    plan, rs, src = r["plan"], r["rules"], r["src"]
    target = float(job["target_gb"]) * 1e9
    out_dir = Path(job.get("out_dir") or Path(gguf).parent)
    use_file = bool(quantize and llamacpp.quantize_supports(quantize, "--tensor-type-file"))
    type_file = out_dir / f"{ID}.tensor-types.txt" if use_file else None
    fl = flags(plan, rs, type_file)
    result = {"plan": plan.to_dict(), "report": r["report"], "rules": [f"{rx}={t}" for rx, t in rs], "flags": fl, "files": [], "measure_key": r["key"], "engine": r["engine"], "notes": r["notes"],
              "fits": plan.fits, "min_gb": round(plan.min_bytes / 1e9, 3), "target_gb": float(job["target_gb"]), "base": base_ftype(plan.total * 8 / max(_params(src), 1))}
    result["plan"]["assign"] = plan.assign
    _save_plan(Path(job.get("cache_dir") or Path(gguf).parent / "quantfit") / r["key"], job["target_gb"], {k: result[k] for k in ("plan", "report", "rules", "engine", "measure_key", "notes", "fits", "target_gb")})
    if job.get("make"):
        if not plan.fits:
            result["notes"].append(f"The file was not made: even the lowest types need {plan.min_bytes / 1e9:.2f} GB, more than the {job['target_gb']} GB asked for.")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            base = imatrix.launcher(quantize)
            final = out_dir / f"model-{ID}.gguf"
            protocol.progress("quantfit", 0.72, "Making the measured file")
            made = make_file(base, Path(gguf), final, plan, rs, matrix, im, type_file, result["notes"])
            result["files"].append({"path": str(final), "label": LABEL, "group": LABEL, "matrix": bool(matrix) and made["matrix_used"], "mixed": ID})
            if job.get("neighbours"):
                for i, ft in enumerate(_plain_neighbours(src, target)):
                    protocol.progress("quantfit", 0.8 + 0.18 * i / 2, f"Making the plain {ft} to compare")
                    path = out_dir / f"model-{ft.lower()}.gguf"
                    used = False
                    if matrix:
                        rc, ok = exportjob.quantize_to(base, Path(gguf), path, ft, ["--imatrix", matrix], im)
                        used = rc == 0 and ok
                        if not used:
                            result["notes"].append(f"llama-quantize did not accept the importance matrix for {ft}; that size was made without it.")
                    if not used:
                        run_quantize(base, Path(gguf), path, ft, [], im, f"The quantisation ({ft})")
                    result["files"].append({"path": str(path), "label": f"{ft}, with matrix" if used else ft, "group": ft, "matrix": used if matrix else None, "mixed": None})
            result["actual_bytes"] = final.stat().st_size if final.is_file() else None
    protocol.progress("quantfit", 1.0, "Done")
    return result


def make_file(base: list, f16: Path, final: Path, plan: Plan, rs: list, matrix, im: dict, type_file, notes: list) -> dict:
    """llama-quantize for the plan: `[--imatrix M] flags f16 final BASE`, once more without the matrix when llama-quantize refuses it. The rules go in a type file when `type_file` is given
    (written here), else as repeated `--tensor-type` arguments (QuantFitError above the command-length limit). Returns {"matrix_used"}."""
    from . import exportjob, quantmeasure

    ftype = base_ftype(_avg_bpw(plan, f16))
    if type_file is not None:
        Path(type_file).parent.mkdir(parents=True, exist_ok=True)
        Path(type_file).write_text(type_file_text(rs), encoding="utf-8")
    fl = flags(plan, rs, type_file)
    cmd_len = len(" ".join(str(x) for x in list(base) + fl + [str(f16), str(final), ftype])) + (len(str(matrix)) + 12 if matrix else 0)
    if type_file is None and cmd_len >= quantmeasure.COMMAND_LIMIT:
        raise QuantFitError(f"The plan has {len(rs)} rules, too many for the command line of this system, and this llama-quantize has no --tensor-type-file: update llama.cpp (this plan needs --tensor-type-file).")
    used = False
    if matrix:
        rc, ok = exportjob.quantize_to(base, f16, final, ftype, ["--imatrix", matrix] + fl, im)
        if rc == 0 and ok:
            used = True
        else:
            msg = f"llama-quantize did not accept the importance matrix for {LABEL} (exit code {rc}); that size was made without it."
            notes.append(msg)
            protocol.log(msg)
    if not used:
        run_quantize(base, f16, final, ftype, fl, im, f"The quantisation ({LABEL})")
    return {"matrix_used": used}


def run_quantize(base, f16, final, kind, flags, im, label) -> None:
    """llama-quantize without a matrix, stoppable (the STOP file kills it); QuantFitError when it fails."""
    from . import exportjob
    rc, ok = exportjob.quantize_to(base, f16, final, kind, flags, im if im is not None else {}, label)
    if not ok:
        raise QuantFitError(f"{label} failed (llama-quantize exit code {rc}). See the log.")


def _avg_bpw(plan: Plan, f16) -> float:
    from . import ggufinfo as gi
    n = gi.n_parameters(gi.read(f16))
    return plan.total * 8 / n if n else 4.9


def run(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(run_job(job))
    return 0


def for_export(f16, matrix_file, cfg: dict, quantize, im=None, out_dir=None, quant: str = ID) -> dict:
    """What llama-quantize needs for the size `mixed_measured` of an export made from the 16-bit file `f16`, like mixedquant.for_export: {"id", "label", "recipe", "base", "embd", "output", "flags",
    "rules": ["regex=TYPE"], "estimate", "notes", "fit", "matrix" (the file the plan was weighted by), "measured"}. `cfg` = {"target_gb", "floor", "ceiling", "sample_rows", "engine", "cache_dir",
    "matrix"}; `matrix_file` is the export's own matrix, used when `cfg` names none. The rules go to `<out_dir>/<quant>.tensor-types.txt` when this llama-quantize has `--tensor-type-file`.
    A target below the smallest size is NOT refused here (an export never stops after the merge): the smallest plan is made and `fit["fits"]` is False with a loud note.
    Raises QuantFitError and the measurement's own errors (the caller falls back to the guessed fit); imatrix.Stopped on Stop."""
    from . import ggufrows, imatrix, llamacpp
    matrix = (cfg or {}).get("matrix") or matrix_file
    notes: list = []
    try:
        r = measure_and_plan(f16, matrix, cfg or {}, quantize, im, notes)
    except quantmeasure.Stopped:
        raise imatrix.Stopped()
    plan, rs, src = r["plan"], r["rules"], r["src"]
    use_file = bool(quantize and llamacpp.quantize_supports(quantize, "--tensor-type-file"))
    type_file = Path(out_dir or Path(f16).parent) / f"{quant}.tensor-types.txt" if use_file else None
    if type_file is not None:
        type_file.parent.mkdir(parents=True, exist_ok=True)
        type_file.write_text(type_file_text(rs), encoding="utf-8")
    fl = flags(plan, rs, type_file)
    if type_file is None and len(" ".join(fl)) >= quantmeasure.COMMAND_LIMIT:
        raise QuantFitError(f"The plan has {len(rs)} rules, too many for the command line of this system, and this llama-quantize has no --tensor-type-file: update llama.cpp (this plan needs --tensor-type-file).")
    avg = plan.total * 8 / max(_params(src), 1)
    est = mixedquant.estimate(src, rs, preset_p(plan, avg))
    target = float((cfg or {}).get("target_gb") or 0) * 1e9
    if not plan.fits:
        notes.append(f"The size asked for ({target / 1e9:.2f} GB) is smaller than the smallest this model can be with {(cfg or {}).get('floor') or DEFAULT_FLOOR} as the lowest type ({plan.min_bytes / 1e9:.2f} GB). "
                     "The smallest plan was made instead; lower the lowest type (Q2_K) or ask for a bigger size.")
    notes += [n for n in plan.notes if n not in notes]
    rep = r["report"]
    simulated = {row["id"]: {"gb": row["gb"], "error": row["error"]} for row in rep["rows"] if row["id"].startswith("plain_")}
    plan_row = next(row for row in rep["rows"] if row["id"] == "plan")
    return {"id": ID, "label": LABEL, "recipe": mixedquant.LABEL, "base": base_ftype(avg), "embd": plan.pins["embd"], "output": plan.pins["output"], "summary": DESCRIPTION,
            "rules": [f"{rx}={t}" for rx, t in rs], "flags": fl, "estimate": est, "notes": notes, "matrix": matrix,
            "fit": {"fits": plan.fits, "target_bytes": int(target), "min_bytes": plan.min_bytes, "max_bytes": plan.max_bytes, "bytes": plan.total, "gb": round(plan.total / 1e9, 2),
                    "bpw": round(avg, 3), "types": rep["types"]},
            "measured": {"engine": r["engine"], "sample_rows": int((cfg or {}).get("sample_rows") or quantmeasure.DEFAULT_SAMPLE_ROWS), "predicted": {"error": plan_row["error"], "gb": plan_row["gb"]},
                         "simulated": simulated, "fits": plan.fits, "measure_key": r["key"], "tensor_type_file": str(type_file) if type_file else None}}
