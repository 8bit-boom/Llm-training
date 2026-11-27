"""job moe_measure: which experts matter. Runs a sample of the project's own text through a mixture-of-experts model (Inkling-2-Mini) and writes `moe_scores.json` into
cfg["out_dir"] (see moe_stats.py for what is counted: frequency, routing weight and REAP's weight x output size per expert, summed over all loop passes, also per half of the
batches). With cfg["rebalance_for"] = {"keep": N, "scoring": ...} it then runs the text once more through the routers as the pruned model would have them (removed experts masked) and
stores the load of every kept expert (`pruned_load`), which `moejob` uses to nudge the router bias. A measurement that is there for the same model, text and sizes is reused unless
cfg["force"]. Events: progress, `moe_scores` (a summary the page shows: the share of the routing mass the best experts hold, per layer), result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import inkling_load, moe_layout, moejob, protocol
from .reshape import ReshapeError

PROTOCOL_KEYS = ("experts", "top_k", "passes", "samples", "max_len", "version", "load", "source")


def _load(src_dir: Path, mode: str):
    """(model, tokenizer, load_used): the packed loader by default, transformers' own with mode "hf" """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src_dir))
    if mode == "hf":
        return inkling_load.load_hf(src_dir), tok, "hf"
    return inkling_load.load_packed(src_dir), tok, "packed"


def measure(model, layers, batches: list, experts: int, core, passes: int, reap: bool = True, on_batch=None, should_stop=None, keep: dict | None = None, top_k=None):
    """The statistics of `batches` ({"input_ids", "attention_mask"} tensors): the dict `Recorder.result` gives, plus "pruned_load" when `keep` is given. Returns None when stopped."""
    import torch

    from . import moe_stats
    rec = moe_stats.Recorder(layers, experts, core, passes, reap).attach()
    dev = next(model.parameters()).device
    try:
        with torch.no_grad():
            for i, b in enumerate(batches):
                if should_stop and should_stop():
                    return None
                rec.new_batch(i % 2)
                model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), use_cache=False)
                if on_batch:
                    on_batch(i + 1, len(batches))
    finally:
        rec.detach()
    out = rec.result()
    if keep:
        im = moe_stats.Imitation(layers, keep, experts, top_k).attach()
        try:
            with torch.no_grad():
                for i, b in enumerate(batches):
                    if should_stop and should_stop():
                        return None
                    model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), use_cache=False)
        finally:
            im.detach()
        out["pruned_load"] = im.loads()
    return out


def stability(scores: dict, keep: int, scoring: str = "reap") -> float:
    """how much the two halves of the batches agree about the experts to keep (mean Jaccard overlap of the kept sets over the layers, 1.0 = the same experts)"""
    halves = {h: {"layers": {l: row["halves"][h] for l, row in scores["layers"].items()}} for h in ("a", "b")}
    ka, kb = (moe_layout.select_experts(halves[h], keep, scoring) for h in ("a", "b"))
    jac = [len(set(ka[l]) & set(kb[l])) / len(set(ka[l]) | set(kb[l])) for l in ka]
    return sum(jac) / len(jac) if jac else 0.0


def summary(scores: dict, experts: int, counts: list[int]) -> dict:
    """for the page: for each allowed count, the share of the routing mass the kept experts hold (min and mean over the layers)"""
    rows = []
    for n in counts:
        kept = moe_layout.select_experts(scores, n, "reap")
        cov = moe_layout.coverage(scores, kept)
        rows.append({"experts": n, "coverage_min": round(min(cov.values()), 4), "coverage_mean": round(sum(cov.values()) / len(cov), 4)})
    return {"counts": rows}


def run(cfg: dict) -> int:
    from . import reshapejob
    protocol.reserve_stdout()
    started = time.monotonic()
    src_dir = reshapejob._resolve_source(cfg["source"]["repo"])
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    if not moe_layout.is_inkling(config):
        raise ReshapeError("Measuring experts is only available for Inkling-2-Mini so far.")
    moe_layout.check_inkling_config(config)
    try:
        inkling_load.require_transformers()
    except inkling_load.InklingLoadError as exc:
        raise ReshapeError(str(exc)) from None
    folder = Path(cfg["out_dir"])
    stop_file = folder.parent / "STOP"
    lp = moe_layout.loop_of(config)
    experts, top_k = moe_layout.experts_of(config), moe_layout.top_k_of(config)
    calib = cfg.get("calibration") or {}
    key = {"experts": experts, "top_k": top_k, "passes": lp["passes"], "samples": calib.get("samples"), "max_len": calib.get("max_len"), "version": calib.get("version"),
           "load": cfg.get("load", "auto"), "source": cfg["source"]["repo"]}
    f = folder / moejob.SCORES_FILE
    rebalance = cfg.get("rebalance_for")
    if cfg.get("force") and folder.is_dir():
        import shutil
        shutil.rmtree(folder)
    if f.is_file():
        old = json.loads(f.read_text(encoding="utf-8"))
        if old.get("meta", {}).get("key") == key and (not rebalance or "pruned_load" in old):
            protocol.result({"key": folder.name, "reused": True, "dir": str(folder), **{k: old["meta"].get(k) for k in ("tokens", "load", "stable_overlap")}})
            return 0
    mode = cfg.get("load", "auto")
    protocol.progress("load_model", 0.0, "Loading the model (the experts stay packed on the card)")
    with protocol.timed("load_model", load=mode, bytes=sum(p.stat().st_size for p in src_dir.glob("*.safetensors"))):
        model, tokenizer, used = _load(src_dir, "hf" if mode == "hf" else "packed")
    try:
        batches = reshapejob._calibration(tokenizer, calib, "inkling2")
        layers = inkling_load.text_layers(model)
        keep = None
        if rebalance:
            try:
                moe_layout.check_experts_kept(experts, top_k, int(rebalance["keep"]))
            except moe_layout.MoeError as exc:
                raise ReshapeError(str(exc)) from None
        t0 = time.monotonic()
        protocol.progress("score", 0.0, "Measuring the experts")
        with protocol.timed("score", load=used) as t:
            scores = measure(model, layers, batches, experts, lp["core"], lp["passes"], reap=True,
                             on_batch=lambda d, n: protocol.progress("score", d / n, f"Measuring the experts: {d} of {n} batches"), should_stop=lambda: stop_file.exists())
            if scores is not None:
                t["tokens"] = sum(float(x) for x in scores["tokens"])
        if scores is None:
            protocol.result({"stopped": True, "dir": None})
            return 0
        if rebalance:
            kept = moe_layout.select_experts(scores, int(rebalance["keep"]), rebalance.get("scoring", "reap"))
            protocol.progress("score", 0.0, "Measuring the load of the cut model")
            more = measure(model, layers, batches, experts, lp["core"], lp["passes"], reap=False, keep=kept, should_stop=lambda: stop_file.exists())
            if more is None:
                protocol.result({"stopped": True, "dir": None})
                return 0
            scores["pruned_load"] = more["pruned_load"]
        tokens = sum(float(x) for x in scores["tokens"])
        counts = moe_layout.expert_counts_allowed(experts, top_k)
        overlap = {str(n): round(stability(scores, n), 4) for n in counts}
        scores["meta"] = {"key": key, "tokens": tokens, "load": used, "stable_overlap": overlap, "seconds": round(time.monotonic() - t0, 1), "core": list(lp["core"]), "format": 1}
        folder.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".partial")
        tmp.write_text(json.dumps(scores), encoding="utf-8")
        tmp.replace(f)
        protocol.emit("moe_scores", **summary(scores, experts, counts), stable_overlap=overlap)
    finally:
        del model
        reshapejob._free()
    protocol.progress("done", 1.0, "")
    protocol.result({"key": folder.name, "reused": False, "dir": str(folder), "tokens": tokens, "load": used, "stable_overlap": overlap, "seconds": round(time.monotonic() - started, 1)})
    return 0
