"""Try first for a mixture of experts (Inkling-2-Mini): the loss of the calibration text under each cut, without making the cut.

A question is a key `e=<kept>|k=<experts per token>|p=<loop passes>|s=<reap|freq|gate>` (every part optional; `e=` alone means all experts). Each one is answered by imitating the cut exactly
(docs/MOE_GUIDE.md): the routers of every layer are masked to the kept experts (`moe_stats.Imitation`), the experts per token and the loop passes are changed in the loaded model, and the
loss of the same text is measured. A kept-expert answer needs `moe_scores.json` (the measuring job). Results are kept in `out_dir/moe_try.json`; a key that is there is not asked again.
The job `reshape_try` hands over here when the source is an Inkling folder. Result: {"analysis": {"baseline", "rows": [...]}, "passes_total", "passes_done", "stopped", "load", "tokens", "dir"}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import inkling_load, moe_layout, moejob, protocol
from .reshape import ReshapeError

RESULTS = "moe_try.json"


def make_key(e=None, k=None, p=None, s=None) -> str:
    parts = []
    for name, v in (("e", e), ("k", k), ("p", p), ("s", s)):
        if v is not None:
            parts.append(f"{name}={v}")
    return "|".join(parts) or "base"


def parse_key(key: str) -> dict:
    out = {"e": None, "k": None, "p": None, "s": None}
    if key in ("", "base"):
        return out
    for part in key.split("|"):
        name, _, v = part.partition("=")
        if name not in out or not v:
            raise ReshapeError(f"Not a question Try first understands: {key!r}.")
        out[name] = v if name == "s" else int(v)
    return out


def default_questions(experts: int, top_k: int, passes: int, varloop: bool = False, goal: str = "size") -> list[str]:
    """The Quick list for a goal (docs/MOE_GUIDE.md section 8): the baseline; fewer experts per token; the counts the rules allow; fewer passes only as a question, and only for
    the goal that asks for speed (Auto-cut never chooses it). `goal`: "size" (counts first), "speed" (top-k and passes first)."""
    counts = moe_layout.expert_counts_allowed(experts, top_k)
    qs = ["base"]
    ks = [k for k in (6, 4) if k < top_k]
    ps = [p for p in ((1, 2) if varloop else (2,)) if p < passes]
    if goal == "grow":
        # the two free grows: one more pass, more experts per token (nobody knows whether either helps: Try first measures it)
        qs += [make_key(p=passes + 1), make_key(k=top_k + 2), make_key(p=passes + 1, k=top_k + 2)]
    elif goal == "speed":
        qs += [make_key(k=k) for k in ks] + [make_key(p=p) for p in ps] + [make_key(e=n) for n in counts[:2]]
    else:
        qs += [make_key(e=n) for n in counts] + [make_key(k=k) for k in ks] + [make_key(p=p) for p in ps]
    if len(counts) >= 2:
        qs.append(make_key(e=counts[1], s="freq"))
        qs.append(make_key(e=counts[1], s="gate"))
    if counts and ks:
        qs.append(make_key(e=counts[1] if len(counts) > 1 else counts[0], k=ks[0]))
    return list(dict.fromkeys(qs))


def band(rise: float | None) -> str:
    """how bad a rise of the loss over the baseline is, in plain words (the same bands as Try first for layers)"""
    if rise is None:
        return "?"
    return "small" if rise < 0.03 else "moderate" if rise < 0.10 else "large"


def _set_passes(model, p: int) -> None:
    seen = set()
    for obj in (model, getattr(model, "model", None)):
        c = getattr(obj, "config", None)
        for cfg in (c, getattr(c, "text_config", None)):
            if cfg is not None and id(cfg) not in seen and hasattr(cfg, "loop_num_passes"):
                cfg.loop_num_passes = p
                seen.add(id(cfg))


def _set_top_k(layers, k: int) -> list:
    """changes the experts per token of every router; returns the old values"""
    old = []
    for layer in layers:
        r = layer.mlp.gate
        old.append(r.top_k)
        r.top_k = k
    return old


def loss_of(model, batches: list, should_stop=None):
    """(mean loss, mean loss of the even batches, of the odd ones, tokens) or None when stopped"""
    import torch

    from . import reshapejob
    dev = next(model.parameters()).device
    tot = [0.0, 0.0]
    cnt = [0, 0]
    with torch.no_grad():
        for i, b in enumerate(batches):
            if should_stop and should_stop():
                return None
            logits = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev), use_cache=False).logits
            s, n = reshapejob._token_losses(logits, b["input_ids"], b["attention_mask"])
            tot[i % 2] += s
            cnt[i % 2] += n
            del logits
    n = sum(cnt)
    if not n:
        raise ReshapeError("Nothing could be measured: the text gave no tokens.")
    return sum(tot) / n, (tot[0] / cnt[0] if cnt[0] else None), (tot[1] / cnt[1] if cnt[1] else None), n


def ask(model, layers, batches, key: str, cfg: dict, scores: dict | None, scorings_default: str = "reap", should_stop=None):
    """the answer to one question: {"loss", "loss_h1", "loss_h2", "tokens", "seconds", "coverage": mean routing mass kept | None, "spread": max/mean load of the kept experts | None}"""
    q = parse_key(key)
    experts, top_k, passes = moe_layout.experts_of(cfg), moe_layout.top_k_of(cfg), moe_layout.loop_of(cfg)["passes"]
    t0 = time.monotonic()
    from . import moe_stats
    im = None
    cov = spread = None
    if q["e"] is not None:
        if scores is None:
            raise ReshapeError("Removing experts is tried on the measurement of the experts: measure them first.")
        try:
            moe_layout.check_experts_kept(experts, q["k"] or top_k, q["e"])
        except moe_layout.MoeError as exc:
            raise ReshapeError(str(exc)) from None
        keep = moe_layout.select_experts(scores, q["e"], q["s"] or scorings_default, layers=list(range(len(layers))))
        c = moe_layout.coverage(scores, keep)
        cov = sum(c.values()) / len(c)
        im = moe_stats.Imitation(layers, keep, experts, q["k"]).attach()
    old_k = _set_top_k(layers, q["k"]) if (q["k"] is not None and q["e"] is None) else None
    if q["p"] is not None:
        _set_passes(model, q["p"])
    try:
        r = loss_of(model, batches, should_stop)
    finally:
        if im:
            im.detach()
        if old_k is not None:
            for layer, k in zip(layers, old_k):
                layer.mlp.gate.top_k = k
        if q["p"] is not None:
            _set_passes(model, passes)
    if r is None:
        return None
    loss, h1, h2, tokens = r
    if im:
        loads = im.loads()
        spreads = [max(v) / (sum(v) / len(v)) for v in loads.values() if v and sum(v) > 0]
        spread = max(spreads) if spreads else None
    return {"loss": loss, "loss_h1": h1, "loss_h2": h2, "tokens": tokens, "seconds": round(time.monotonic() - t0, 2), "coverage": cov, "spread": spread}


def rows_of(data: dict, cfg: dict) -> dict:
    """the analysis: every answered question with its rise over the baseline, the kept size and the speed change"""
    base = (data["passes"].get("base") or {}).get("loss")
    experts, top_k, passes = moe_layout.experts_of(cfg), moe_layout.top_k_of(cfg), moe_layout.loop_of(cfg)["passes"]
    rows = []
    for key, res in data["passes"].items():
        q = parse_key(key)
        e, k, p = q["e"] or experts, q["k"] or top_k, q["p"] or passes
        rise = (res["loss"] - base) / base if base else None
        halves_agree = None
        if base and res.get("loss_h1") is not None and res.get("loss_h2") is not None and data["passes"]["base"].get("loss_h1") is not None:
            halves_agree = (res["loss_h1"] > data["passes"]["base"]["loss_h1"]) == (res["loss_h2"] > data["passes"]["base"]["loss_h2"])
        rows.append({"key": key, "experts": e, "top_k": k, "passes": p, "scoring": q["s"], "loss": res["loss"], "rise": rise, "band": band(rise), "coverage": res.get("coverage"), "load_spread": res.get("spread"),
                     "visits_ratio": moe_layout.visits(cfg, p) / moe_layout.visits(cfg), "active_ratio": k / top_k, "halves_agree": halves_agree})
    return {"baseline": base, "rows": rows}


def run(cfg: dict, src_dir: Path, config: dict) -> int:
    from . import reshapejob
    started = time.monotonic()
    folder = Path(cfg["out_dir"]) if cfg.get("out_dir") else Path(cfg["scores_dir"]).parent / "try"
    stop_file = folder.parent / "STOP"
    want = (cfg.get("want") or {}).get("moe") or {}
    questions = list(want.get("keys") or default_questions(moe_layout.experts_of(config), moe_layout.top_k_of(config), moe_layout.loop_of(config)["passes"],
                                                            moe_layout.loop_of(config)["varloop"], want.get("goal", "size")))
    questions = list(dict.fromkeys(["base"] + questions))
    for k in questions:
        parse_key(k)
    store = folder / RESULTS
    data = json.loads(store.read_text(encoding="utf-8")) if store.is_file() else {"passes": {}, "meta": {}}
    todo = [k for k in questions if k not in data["passes"]]
    scores = None
    if cfg.get("scores_dir") and (Path(cfg["scores_dir"]) / moejob.SCORES_FILE).is_file():
        scores = moejob.read_scores(cfg["scores_dir"])
    if any(parse_key(k)["e"] is not None for k in todo) and scores is None:
        raise ReshapeError("Removing experts is tried on the measurement of the experts: measure them first.")
    if cfg.get("plan_only"):
        protocol.result({"passes_total": len(questions), "passes_done": len(questions) - len(todo), "plan_only": True, "keys": questions})
        return 0
    try:
        inkling_load.require_transformers()
    except inkling_load.InklingLoadError as exc:
        raise ReshapeError(str(exc)) from None
    calib = cfg.get("calibration") or {}
    protocol.progress("load_model", 0.0, "Loading the model")
    from . import moemeasure
    with protocol.timed("load_model", load=cfg.get("load", "auto"), bytes=sum(p.stat().st_size for p in src_dir.glob("*.safetensors"))):
        model, tokenizer, used = moemeasure._load(src_dir, "hf" if cfg.get("load") == "hf" else "packed")
    stopped = False
    try:
        batches = reshapejob._calibration(tokenizer, calib, "inkling2")
        layers = inkling_load.text_layers(model)
        folder.mkdir(parents=True, exist_ok=True)
        # the baseline first: every other answer is read against it
        order = (["base"] if "base" in todo else []) + [k for k in todo if k != "base"]
        for i, key in enumerate(order):
            with protocol.timed("try_pass", load=used) as t:
                res = ask(model, layers, batches, key, config, scores, should_stop=lambda: stop_file.exists())
                if res is not None:
                    t["tokens"] = res["tokens"]
            if res is None:
                stopped = True
                break
            data["passes"][key] = res
            data["meta"] = {"load": used, "samples": calib.get("samples"), "max_len": calib.get("max_len"), "version": calib.get("version")}
            data["analysis"] = rows_of(data, config)
            tmp = store.with_suffix(".partial")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(store)
            protocol.progress("try", (i + 1) / len(order), f"Question {i + 1} of {len(order)}: {key}")
            protocol.emit("try_pass", key=key, loss=res["loss"], seconds=res["seconds"], done=len(questions) - len(order) + i + 1, total=len(questions), analysis=data["analysis"])
    finally:
        del model
        reshapejob._free()
    data["analysis"] = rows_of(data, config) if "base" in data["passes"] else {"baseline": None, "rows": []}
    protocol.progress("done", 1.0, "")
    protocol.result({"analysis": data["analysis"], "passes_total": len(questions), "passes_done": sum(1 for k in questions if k in data["passes"]), "stopped": stopped, "load": used,
                     "tokens": sum(r.get("tokens", 0) for r in data["passes"].values()), "dir": str(folder), "seconds": round(time.monotonic() - started, 1)})
    return 0
