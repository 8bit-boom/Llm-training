"""Where does a heal's time go on THIS computer? (the "Measure heal speed" button; everything here is OFF unless that button starts it; nothing is saved but the report)

    BUILT FROM A RESEARCH REPORT (docs/HEAL_SPEED.md), NOT RUN ON A 5090: every statement about speed is a hypothesis that this measurement is meant to confirm or refute.

`distill.bench` keeps its plain speed number; with `bench: {"profile": true}` it ALSO runs two more timed steps in sections (forward, the KL loss, backward, the optimizer: wall seconds with a
synchronize at each border) under `torch.profiler`, and adds `heal_profile {forward_s, loss_s, backward_s, optimizer_s, gdn_s, gemm_s, top_kernels}`; `"microbench": true` times one GatedDeltaNet
forward + backward with the PyTorch reference and, when importable, with fla; `"teacher_bench": true` times the original's answers per rows-per-forward. The kernel-class split uses the NAMES of
the GPU kernels (GDN_NAMES / GEMM_NAMES): a rough classification, not an exact one. The report `heal_speed_report.md/.json` is written on EVERY end (also on a failure), with user names and home
folders taken out. The pure functions (classification, summary, report, the loops with injected functions) are tested without a GPU.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import protocol

GDN_NAMES = re.compile(r"(gated_delta|chunk_gated|delta_rule|gdn|chunk_scaled_dot_kkt|recompute_w_u|fwd_prepare|l2norm|solve_tril|fla[_.])", re.I)
GEMM_NAMES = re.compile(r"(gemm|cutlass|cublas|nvjet|xmma|sm\d+_xmma|gemv|ampere_|hopper_|wgmma|matmul|dequant|nf4|bnb)", re.I)
ROWS_PER_FORWARD = (1, 2, 4, 8, 16)
TEACHER_ROWS = 200
DISCLAIMER = ("Measured on this computer with a rough classification of GPU kernel names; the research report's speed claims are hypotheses until this report confirms them. "
              "A real heal is a little slower (checkpoints, checks).")


# ------------------------------------------------------------------------------------------------------------------------------ kernels and sections

def classify_kernels(kernels: list) -> dict:
    """`kernels`: [(name, gpu_microseconds)]. -> {"gdn_s", "gemm_s", "other_s", "top_kernels": [{"name", "seconds", "share"}] (10 largest)}."""
    total = sum(float(us) for _, us in kernels) or 0.0
    gdn = sum(float(us) for n, us in kernels if GDN_NAMES.search(n))
    gemm = sum(float(us) for n, us in kernels if not GDN_NAMES.search(n) and GEMM_NAMES.search(n))
    merged: dict = {}
    for n, us in kernels:
        merged[n] = merged.get(n, 0.0) + float(us)
    top = sorted(merged.items(), key=lambda x: -x[1])[:10]
    return {"gdn_s": round(gdn / 1e6, 4), "gemm_s": round(gemm / 1e6, 4), "other_s": round((total - gdn - gemm) / 1e6, 4),
            "top_kernels": [{"name": n[:160], "seconds": round(us / 1e6, 4), "share": round(us / total, 4) if total else 0.0} for n, us in top]}


def profile_summary(sections: dict, kernels: list, steps: int) -> dict:
    """The `heal_profile` of `steps` profiled steps: the section seconds PER STEP and the kernel classes' GPU seconds per step."""
    n = max(1, int(steps))
    c = classify_kernels(kernels)
    per = lambda x: round(float(x) / n, 4)      # noqa: E731
    out = {"steps": n, "forward_s": per(sections.get("forward", 0.0)), "loss_s": per(sections.get("loss", 0.0)), "backward_s": per(sections.get("backward", 0.0)),
           "optimizer_s": per(sections.get("optimizer", 0.0)), "gdn_s": per(c["gdn_s"]), "gemm_s": per(c["gemm_s"]), "other_gpu_s": per(c["other_s"]), "top_kernels": c["top_kernels"]}
    wall = out["forward_s"] + out["loss_s"] + out["backward_s"] + out["optimizer_s"]
    out["step_s"] = round(wall, 4)
    out["gdn_share_of_step"] = round(out["gdn_s"] / wall, 3) if wall else None
    out["gemm_share_of_step"] = round(out["gemm_s"] / wall, 3) if wall else None
    return out


class Sections:
    """Wall seconds per section, with a synchronize at each border (`sync` is torch.cuda.synchronize or a no-op)."""

    def __init__(self, sync=lambda: None, clock=time.monotonic):
        self.t: dict = {}
        self.sync, self.clock = sync, clock

    def wrap(self, label: str, fn):
        def inner(*a, **k):
            self.sync()
            t0 = self.clock()
            try:
                return fn(*a, **k)
            finally:
                self.sync()
                self.t[label] = self.t.get(label, 0.0) + (self.clock() - t0)
        return inner


# ------------------------------------------------------------------------------------------------------------------------------------ microbench

def gdn_microbench(steps: int = 3, torch_fn=None, fla_fn=None, make_inputs=None, timer=time.monotonic, sync=lambda: None) -> dict:
    """One GatedDeltaNet forward + backward with the PyTorch reference and with fla (when importable). The functions and the inputs are injected in the tests; by default they are imported from
    transformers / fla (the call signature is the reference implementation's: UNVERIFIED on every version, so a failure is reported as text, never raised). -> {"torch": {...} | {"error"}, "fla": ...}."""
    res: dict = {}

    def one(label, fn):
        if fn is None:
            res[label] = {"error": f"{label} implementation not importable here"}
            return
        try:
            args = make_inputs() if make_inputs else None
            fn(*args)                                                      # warm-up (Triton compiles on the first call)
            sync()
            t0 = timer()
            for _ in range(steps):
                fn(*args)
            sync()
            res[label] = {"seconds_per_fwd_bwd": round((timer() - t0) / steps, 5), "steps": steps}
        except Exception as exc:                          # noqa: BLE001 - a microbench that fails is a line of the report
            res[label] = {"error": f"{type(exc).__name__}: {protocol.scrub(str(exc))[:200]}"}

    if torch_fn is None and fla_fn is None and make_inputs is None:
        torch_fn, fla_fn, make_inputs, sync = _default_gdn()
    one("torch", torch_fn)
    one("fla", fla_fn)
    t, f = res.get("torch", {}).get("seconds_per_fwd_bwd"), res.get("fla", {}).get("seconds_per_fwd_bwd")
    if t and f:
        res["fla_speedup"] = round(t / f, 2)
    return res


def _default_gdn():
    """(torch reference, fla function or None, input maker, synchronize) for the real thing; any import problem gives None functions (the report then says so)."""
    try:
        import torch
    except ImportError:
        return None, None, None, (lambda: None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    ref = fla = None
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
        ref = getattr(mq, "torch_chunk_gated_delta_rule", None)
    except Exception:       # noqa: BLE001
        ref = None
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla
    except Exception:       # noqa: BLE001
        fla = None

    def inputs():
        b, t, h, d = 1, 2048, 16, 128
        g = torch.Generator(device="cpu").manual_seed(0)
        mk = lambda *s: torch.randn(*s, generator=g).to(dev, torch.bfloat16).requires_grad_(True)      # noqa: E731
        q, k, v = mk(b, t, h, d), mk(b, t, h, d), mk(b, t, h, d)
        gate = (-torch.rand(b, t, h, generator=g).to(dev, torch.float32)).requires_grad_(True)
        beta = torch.rand(b, t, h, generator=g).to(dev, torch.bfloat16).requires_grad_(True)
        return q, k, v, gate, beta

    def wrap(fn):
        if fn is None:
            return None

        def run(q, k, v, g, beta):
            out = fn(q, k, v, g, beta)
            out = out[0] if isinstance(out, tuple) else out
            out.float().sum().backward()
        return run

    return wrap(ref), wrap(fla), inputs, sync


# --------------------------------------------------------------------------------------------------------------------------------- teacher bench

def teacher_bench(rows: list, answer, sizes=ROWS_PER_FORWARD, limit: int = TEACHER_ROWS, timer=time.monotonic, sync=lambda: None) -> list:
    """The original's answers per rows-per-forward: `answer(batch_rows)` writes the answers of a batch (the real `_teacher_answers` with the model loaded, a fake in the tests). Rows are sorted by
    length so that a batch is of similar length, as the cache job does. -> [{"rows_per_forward", "tokens_per_s", "seconds", "tokens", "error"?}]."""
    use = sorted(rows[:limit], key=lambda r: -len(r["input_ids"]))
    out = []
    for n in sizes:
        try:
            sync()
            t0, tokens = timer(), 0
            for i in range(0, len(use), n):
                b = use[i:i + n]
                answer(b)
                tokens += sum(len(r["input_ids"]) for r in b)
            sync()
            secs = max(timer() - t0, 1e-9)
            out.append({"rows_per_forward": n, "tokens_per_s": round(tokens / secs, 1), "seconds": round(secs, 2), "tokens": tokens, "rows": len(use)})
        except Exception as exc:                          # noqa: BLE001 - an out-of-memory size is a line of the report, the next one is tried
            out.append({"rows_per_forward": n, "error": f"{type(exc).__name__}: {protocol.scrub(str(exc))[:160]}"})
            if "out of memory" in str(exc).lower():
                break
    return out


# ----------------------------------------------------------------------------------------------------------------------------------- the report

BATCH_TOKENS_SWEEP = (0, 2048, 4096, 8192, 16384)
VRAM_SHARE_LIMIT = 0.90          # a recommended budget keeps the peak below this share of the card
LOSS_TOLERANCE = 1e-3            # relative: the first update's loss must be the same as with one row per step
MIN_SPEEDUP = 1.05


def length_stats(lengths) -> dict:
    """The length distribution of the rows a measurement used: {"n", "mean", "p50", "p90", "max"} in tokens."""
    t = sorted(int(x) for x in lengths)
    if not t:
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0}
    q = lambda f: t[min(len(t) - 1, int(f * (len(t) - 1) + 0.5))]      # noqa: E731
    return {"n": len(t), "mean": round(sum(t) / len(t), 1), "p50": q(0.5), "p90": q(0.9), "max": t[-1]}


NOISE_FACTOR = 2.0               # a gain must exceed this many times the measured run-to-run spread to count


def spread_of(values) -> float:
    """|a - b| / mean of the repeated measurements of one candidate (the largest relative distance from the mean for more than two); 0 for fewer than two."""
    v = [float(x) for x in values if x]
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return round(max(abs(x - m) for x in v) / m * (2.0 if len(v) == 2 else 1.0), 4) if m else 0.0


def recommend_batch_tokens(results: list, card_mb, spread: float = 0.0) -> dict:
    """`results`: [{"batch_tokens", "tokens_per_s", "peak_vram_mb", "first_loss", "error"?}] with 0 (one row per step) as the baseline. The recommendation is the value with the largest speedup whose peak
    stays below 90 % of the card and whose first-update loss equals the baseline's within a relative 1e-3; None (value 0) when nothing is faster by 5 %. Adds `speedup` to every result.
    -> {"value", "speedup", "why"}."""
    base = next((r for r in results if r.get("batch_tokens") == 0 and r.get("tokens_per_s")), None)
    if base is None:
        return {"value": 0, "speedup": 1.0, "why": "The measurement with one row per step is missing, so nothing can be compared."}
    for r in results:
        r["speedup"] = round(r["tokens_per_s"] / base["tokens_per_s"], 2) if r.get("tokens_per_s") else None
    best = None
    for r in results:
        if r.get("batch_tokens") == 0 or not r.get("tokens_per_s") or r.get("error"):
            continue
        peak = max(r.get("peak_vram_mb") or 0, r.get("worst_peak_mb") or 0)
        if card_mb and peak and peak > VRAM_SHARE_LIMIT * card_mb:
            continue
        if base.get("first_loss") is not None and r.get("first_loss") is not None and abs(r["first_loss"] - base["first_loss"]) > LOSS_TOLERANCE * max(1.0, abs(base["first_loss"])):
            continue
        if best is None or r["speedup"] > best["speedup"]:
            best = r
    need = max(MIN_SPEEDUP, 1.0 + NOISE_FACTOR * float(spread or 0.0))
    if best is None or best["speedup"] < need:
        return {"value": 0, "speedup": 1.0, "spread": spread, "why": f"No token budget is clearly faster (at least {round((need - 1) * 100)} %, above twice the measured noise of {round(float(spread or 0.0) * 100, 1)} %) within 90 % of the card's memory and with the same first loss: one row per step stays."}
    return {"value": best["batch_tokens"], "speedup": best["speedup"], "spread": spread, "why": f"{best['batch_tokens']} tokens per forward pass: {best['speedup']} times the speed of one row per step, peak {best.get('peak_vram_mb')} MB"
                                                                              + (f" of {round(card_mb)} MB" if card_mb else "") + ", the same first loss."}


def sweep(run_one, budgets=BATCH_TOKENS_SWEEP, card_mb=None, repeat=None) -> dict:
    """Runs `run_one(budget) -> {"tokens_per_s", "peak_vram_mb", "first_loss", ...}` for every budget (an exception, an out-of-memory error included, is recorded for that budget and the others go on)
    and adds the recommendation. -> {"results": [...], "recommended": {...}}."""
    results = []
    oom_at = None
    for b in budgets:
        if oom_at is not None and int(b) > oom_at:
            results.append({"batch_tokens": int(b), "tokens_per_s": None, "error": f"skipped: {oom_at} already ran out of memory", "skipped": True})
            continue
        try:
            r = {"batch_tokens": int(b), **run_one(int(b))}
        except Exception as exc:        # noqa: BLE001 - one budget failing (memory) must not lose the others
            r = {"batch_tokens": int(b), "tokens_per_s": None, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        if r.get("error") and "out of memory" in str(r["error"]).lower():
            oom_at = int(b) if oom_at is None else min(oom_at, int(b))
            r["oom"] = True
        results.append(r)
    spread = 0.0
    if repeat and int(repeat) > 1:
        # the baseline and the fastest candidate are measured again (the same process, the same rows): the run-to-run noise decides how big a gain has to be
        ok = [r for r in results if r.get("tokens_per_s") and not r.get("error")]
        again = [r for r in ok if r["batch_tokens"] == 0] + sorted([r for r in ok if r["batch_tokens"] != 0], key=lambda r: -r["tokens_per_s"])[:1]
        for r in again:
            vals = [r["tokens_per_s"]]
            for _ in range(int(repeat) - 1):
                try:
                    vals.append(run_one(r["batch_tokens"]).get("tokens_per_s"))
                except Exception:        # noqa: BLE001
                    break
            r["repeats"] = [v for v in vals if v]
            spread = max(spread, spread_of(r["repeats"]))
            if len(r["repeats"]) > 1:
                r["tokens_per_s"] = round(sum(r["repeats"]) / len(r["repeats"]), 2)       # the mean of the repeats is the number the comparison uses
    return {"results": results, "spread": spread, "recommended": recommend_batch_tokens(results, card_mb, spread)}


TEACHER_SWEEP = (0, 4096, 8192, 16384, 32768, 65536)
TEACHER_SWEEP_TOKENS = 300_000          # about this many tokens of real rows are answered for every budget
TEACHER_WINDOW = 64                      # the cache job answers a window of this many rows in length order


def teacher_sample(rows: list, want_tokens: int = TEACHER_SWEEP_TOKENS, seed: int = 3407, longest: int = 4) -> list:
    """The rows the teacher sweep answers: a seeded sample of the real rows up to `want_tokens`, and always the `longest` longest rows (the worst case of a forward pass for the memory)."""
    import random
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    keep, total = [], 0
    for i in order:
        if total >= want_tokens:
            break
        keep.append(i)
        total += len(rows[i]["input_ids"])
    top = sorted(range(len(rows)), key=lambda i: -len(rows[i]["input_ids"]))[:longest]
    keep += [i for i in top if i not in set(keep)]
    return [rows[i] for i in keep]


def teacher_pass(sample: list, answer, budget: int, fixed_rows: int, group, window: int = TEACHER_WINDOW, timer=time.monotonic, sync=lambda: None) -> dict:
    """One pass of the original's answers over `sample` the way the cache job does it: windows of rows, longest first, batches by the token budget (`group(lengths_longest_first, budget)` = how many rows
    go together; budget 0 = `fixed_rows` rows). `answer(batch_rows)` returns [(ids, logp, resid)] per row. -> {"tokens_per_s", "tokens", "seconds", "first_loss"}: first_loss is the mean top-1 negative
    log-probability of the rows of the FIRST window (the same rows for every budget: if batching changed the answers, it moves)."""
    sync()
    t0, tokens = timer(), 0
    probe_sum, probe_n = 0.0, 0
    for w in range(0, len(sample), window):
        win = sample[w:w + window]
        order = sorted(range(len(win)), key=lambda j: -len(win[j]["input_ids"]))
        pos = 0
        while pos < len(order):
            n = fixed_rows if budget <= 0 else group([len(win[j]["input_ids"]) for j in order[pos:pos + 16]], budget)
            idx = order[pos:pos + max(1, n)]
            got = answer([win[j] for j in idx])
            for j, a in zip(idx, got):
                tokens += len(win[j]["input_ids"])
                if w == 0:
                    probe_sum += float(-a[1][:, 0].mean())
                    probe_n += 1
            pos += len(idx)
    sync()
    secs = max(timer() - t0, 1e-9)
    return {"tokens_per_s": round(tokens / secs, 2), "tokens": tokens, "seconds": round(secs, 2), "first_loss": round(probe_sum / probe_n, 6) if probe_n else None}


def teacher_sweep(sample: list, answer, group, fixed_rows: int = 4, budgets=TEACHER_SWEEP, card_mb=None, repeat: int = 2, peak=lambda: None, reset_peak=lambda: None, **kw) -> dict:
    """The original's tokens per forward pass (job key `teacher_batch_tokens`), measured on the real rows with the same machinery as the student's sweep: `sweep` gives the rows, the noise and the
    recommendation (budget 0 = the fixed rows per pass is the baseline). The peak memory of every budget is read after its pass. -> {"results", "spread", "recommended", "rows", "tokens"}."""
    def run_one(budget: int) -> dict:
        reset_peak()
        r = teacher_pass(sample, answer, budget, fixed_rows, group, **kw)
        r["peak_vram_mb"] = peak()
        return r
    out = sweep(run_one, budgets, card_mb, repeat)
    out["rows"], out["tokens"] = len(sample), sum(len(r["input_ids"]) for r in sample)
    return out


def environment_facts() -> dict:
    """Versions and the card (never raises): torch, triton(-windows), flash-linear-attention, transformers, the GPU's name and memory."""
    import importlib.metadata as md
    facts: dict = {}
    for key, names in (("torch", ("torch",)), ("triton", ("triton-windows", "triton")), ("flash_linear_attention", ("flash-linear-attention", "fla-core")), ("transformers", ("transformers",)),
                       ("bitsandbytes", ("bitsandbytes",))):
        for n in names:
            try:
                facts[key] = f"{n} {md.version(n)}"
                break
            except Exception:       # noqa: BLE001
                continue
        else:
            facts[key] = None
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            facts["gpu"] = {"name": p.name, "memory_gb": round(p.total_memory / 1e9, 1), "capability": list(torch.cuda.get_device_capability(0))}
    except Exception:       # noqa: BLE001
        pass
    try:
        from . import tritonpin
        facts["triton_match"] = tritonpin.installed_status()        # torch, triton(-windows) and whether they belong together
    except Exception:       # noqa: BLE001
        pass
    return facts


def verdict_lines(rep: dict) -> list:
    """Plain sentences read off the numbers (hypotheses stay hypotheses)."""
    out = []
    tm = (rep.get("environment") or {}).get("triton_match") or {}
    if tm.get("matches") is False:
        out.append(tm["text"])
    kern = rep.get("kernels") or {}
    if kern.get("delta_rule") == "torch":
        out.append("GatedDeltaNet runs on the slow PyTorch kernels: the fast kernels (flash-linear-attention) are the first thing to try.")
    elif kern.get("delta_rule") == "unknown":
        out.append("It is not known which GatedDeltaNet kernels run; the profile below tells (a large gdn share means the slow ones).")
    bs = rep.get("batch_sweep") or {}
    rec = bs.get("recommended") or {}
    if rec.get("value"):
        out.append(f"Several rows in one forward pass pay off here: {rec['why']} (Heal card: Use the recommended batch size; nothing is changed by itself.)")
    elif bs.get("results"):
        out.append(rec.get("why") or "Several rows per forward pass were measured; none was clearly better.")
    p = rep.get("heal_profile")
    if p and p.get("gdn_share_of_step") is not None:
        out.append(f"GatedDeltaNet kernels take about {p['gdn_share_of_step'] * 100:.0f} % of a step, matrix multiplications about {(p.get('gemm_share_of_step') or 0) * 100:.0f} % (by kernel names: rough).")
    mb = rep.get("microbench") or {}
    if mb.get("fla_speedup"):
        out.append(f"One GatedDeltaNet forward + backward: fla is {mb['fla_speedup']} times faster than the PyTorch reference on this card (a hypothesis of the report, now measured).")
    tb = [x for x in rep.get("teacher_bench") or [] if x.get("tokens_per_s")]
    if tb:
        best = max(tb, key=lambda x: x["tokens_per_s"])
        four = next((x for x in tb if x["rows_per_forward"] == 4), None)
        out.append(f"The original's answers: best at {best['rows_per_forward']} rows per forward ({best['tokens_per_s']:.0f} tokens/s)" + (f", against {four['tokens_per_s']:.0f} with the fixed 4." if four else "."))
    return out


def markdown(rep: dict) -> str:
    L = ["# Heal speed report", "", f"> {DISCLAIMER}", "", f"Created {rep.get('created')}. Status: **{rep.get('status')}**" + (f": {rep['error']}" if rep.get("error") else ""), ""]
    for v in rep.get("verdict", []):
        L.append(f"- {v}")
    L += ["", "## This computer", ""] + [f"- {k}: {v['text'] if k == 'triton_match' and isinstance(v, dict) else v}" for k, v in (rep.get("environment") or {}).items()] + [""]
    if rep.get("windows_hints"):
        L += ["## Windows settings (shown, never changed)", ""] + [f"- {h}" for h in rep["windows_hints"]] + [""]
    if rep.get("kernels"):
        L += ["## Kernels", "", f"`{json.dumps(rep['kernels'], sort_keys=True)}`", ""]
    if rep.get("bench"):
        b = rep["bench"]
        L += ["## Training step speed", "", f"{b.get('tokens_per_s')} tokens/s, peak {b.get('peak_vram_mb')} MB, sequence length {b.get('max_seq')}, batch {b.get('batch')}, chunk {b.get('chunk')}.", ""]
    if rep.get("rows"):
        r = rep["rows"]
        L += ["## The rows measured", "", f"{r['n']} real rows: {r['mean']} tokens on average, median {r['p50']}, 90 % are at most {r['p90']}, longest {r['max']}.", ""]
    if (rep.get("batch_sweep") or {}).get("results"):
        L += ["## Tokens per forward pass (student micro-batches of rows of similar length)", "", "| tokens per forward | tokens/s | speed-up | peak VRAM (MB) | first loss | forward passes per update |", "|---|---|---|---|---|---|"]
        for x in rep["batch_sweep"]["results"]:
            L.append(f"| {x['batch_tokens'] or '0 (one row)'} | {x.get('tokens_per_s') or x.get('error')} | {x.get('speedup') or ''} | {x.get('peak_vram_mb') or ''} | {x.get('first_loss') if x.get('first_loss') is not None else ''} | {x.get('micro_batches') or ''} |")
        L += ["", "Recommended: " + (rep["batch_sweep"].get("recommended") or {}).get("why", ""), ""]
    p = rep.get("heal_profile")
    if p:
        L += ["## Where a step goes (per step, wall seconds with synchronized borders)", "", "| forward | KL loss | backward | optimizer | GDN kernels (GPU) | matmul kernels (GPU) |", "|---|---|---|---|---|---|",
              f"| {p['forward_s']} | {p['loss_s']} | {p['backward_s']} | {p['optimizer_s']} | {p['gdn_s']} | {p['gemm_s']} |", "", "Largest GPU kernels:", ""]
        L += [f"- {k['seconds']} s ({k['share'] * 100:.1f} %) {k['name']}" for k in p.get("top_kernels", [])] + [""]
    if rep.get("microbench"):
        L += ["## One GatedDeltaNet forward + backward", "", "```", json.dumps(rep["microbench"], indent=1), "```", ""]
    if rep.get("teacher_bench"):
        L += ["## The original's answers per rows-per-forward", "", "| rows per forward | tokens/s | seconds |", "|---|---|---|"]
        L += [f"| {x['rows_per_forward']} | {x.get('tokens_per_s', x.get('error'))} | {x.get('seconds', '')} |" for x in rep["teacher_bench"]] + [""]
    if rep.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in rep["notes"]] + [""]
    return "\n".join(L) + "\n"


def write_report(out_dir, rep: dict) -> dict:
    """Always: `heal_speed_report.json` and `.md` in `out_dir` (made if needed), secrets and the user's name / home folder taken out. Returns the paths."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    rep = {**rep, "verdict": verdict_lines(rep)}
    clean = lambda t: protocol.redact_user(protocol.scrub(t))      # noqa: E731
    (d / "heal_speed_report.json").write_text(clean(json.dumps(rep, ensure_ascii=False, indent=1, default=str)), encoding="utf-8")
    (d / "heal_speed_report.md").write_text(clean(markdown(rep)), encoding="utf-8")
    return {"report": str(d / "heal_speed_report.json"), "report_md": str(d / "heal_speed_report.md")}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------------------------------ the profiler (needs torch)

def profile_steps(model, rows, collate, targets, chunk, head_ckpt, opt, batch, dev, n: int = 2, offset: int = 0) -> dict:
    """`n` steps under torch.profiler with the sections timed; returns `heal_profile`. Needs torch and the student's functions (distilljob)."""
    import torch
    from . import distilljob as dj
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    sec = Sections(sync)
    orig_fwd, orig_kd = dj.forward_hidden, dj.chunked_kd_loss
    dj.forward_hidden, dj.chunked_kd_loss = sec.wrap("forward", orig_fwd), sec.wrap("loss", orig_kd)
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
    try:
        with torch.profiler.profile(activities=acts) as prof:
            for i in range(n):
                sel = [{**rows[(offset + i * batch + j) % len(rows)], "row": (offset + i * batch + j) % len(rows)} for j in range(batch)]
                inputs = {kk: v.to(dev) for kk, v in collate(sel).items()}
                loss, _, _ = dj.student_loss(model, inputs, targets, 1.0, chunk, head_ckpt)
                sync()
                t0 = time.monotonic()
                loss.backward()
                sync()
                sec.t["backward"] = sec.t.get("backward", 0.0) + (time.monotonic() - t0)
                t0 = time.monotonic()
                opt.step()
                opt.zero_grad(set_to_none=True)
                sync()
                sec.t["optimizer"] = sec.t.get("optimizer", 0.0) + (time.monotonic() - t0)
    finally:
        dj.forward_hidden, dj.chunked_kd_loss = orig_fwd, orig_kd
    kernels = []
    for ev in prof.key_averages():
        us = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0) or 0
        if us and getattr(ev, "device_type", None) is not None and "cuda" in str(ev.device_type).lower():
            kernels.append((ev.key, us))
    return profile_summary(sec.t, kernels, n)
