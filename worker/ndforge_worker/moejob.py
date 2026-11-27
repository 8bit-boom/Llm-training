"""The "Slim or grow a model" job for Inkling-2-Mini (and later other mixtures of experts): cuts that change the config of a packed model, written as a new folder.

`reshapejob.run` hands the job here when the source is an Inkling folder. Kinds of cut (job keys; any mix may be given):

    passes   loop passes (3 -> 2 or 4): config only, the weights are copied (owner's decision D8: copy, never hard links)
    top_k    experts used per token: config only
    experts  {"keep": N, "scoring": "reap"|"freq"|"gate", "scores_dir": folder with moe_scores.json, "rebalance": bool}: N experts stay in EVERY layer; the expert tensors,
             their scales and the router rows are sliced along the expert axis, byte for byte, in the authors' packed format (nothing is decoded)

Whole blocks follow in `moejob` too (see docs/MOE_GUIDE.md).
The result has the same keys as a dense cut (so the page and Auto-cut read it unchanged) plus "moe".
"""
from __future__ import annotations

import datetime
import json
import shutil
import struct
import time
from pathlib import Path

from . import fsutil, moe_layout, protocol, reshape, safetensors_io
from .reshape import ReshapeError

COPY_CHUNK = 64 * 2**20
SCORES_FILE = "moe_scores.json"
DRAFTER = "mtp.safetensors"
EXTRA_DIRS = ("tiktoken",)               # the tokenizer's other form: copied as it is
EXTRA_FILES = ("hf_quant_config.json",)


def _copy_file(src: Path, dst: Path, stop_file: Path, on_bytes) -> None:
    with open(src, "rb") as a, open(dst, "wb") as b:
        while True:
            chunk = a.read(COPY_CHUNK)
            if not chunk:
                break
            b.write(chunk)
            on_bytes(len(chunk))
            if stop_file.exists():
                raise safetensors_io.Cancelled()


def kinds_of(cfg: dict) -> dict:
    """the requested cuts: {"passes": int|None, "top_k": int|None}; ReshapeError when none or an unknown one is asked for"""
    out = {"passes": cfg.get("passes"), "top_k": cfg.get("top_k"), "experts": cfg.get("experts")}
    for k in ("passes", "top_k"):
        v = out[k]
        if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
            raise ReshapeError(f"{k} must be a whole number.")
    e = out["experts"]
    if e is not None and (not isinstance(e, dict) or not isinstance(e.get("keep"), int) or isinstance(e.get("keep"), bool)):
        raise ReshapeError("experts must say how many to keep: {\"keep\": 96}.")
    if all(v is None for v in out.values()):
        raise ReshapeError("Nothing to change: give the number of loop passes, of experts per token or of experts to keep.")
    return out


def read_scores(scores_dir) -> dict:
    f = Path(scores_dir or "") / SCORES_FILE
    if not f.is_file():
        raise ReshapeError("The experts have not been measured yet. Measure them first (Try first or Measure experts); removing experts without it would be a guess.")
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError:
        raise ReshapeError("The saved measurement of the experts could not be read; measure again.") from None


def _expert_plan(tensors: dict, kept: dict, keep: int, bias_new: dict) -> list:
    """OutTensor list of the pruned model: sliced expert and router tensors, everything else as it is"""
    from .safetensors_io import OutTensor
    layout = moe_layout.prune_layout({k: (r.dtype, r.shape) for k, r in tensors.items()}, kept, keep)
    out = []
    for name, ref in tensors.items():
        ids = layout.get(name)
        if ids is None:
            out.append(OutTensor(name, ref.dtype, ref.shape, ref))
        elif name in bias_new:
            vals = bias_new[name]
            out.append(OutTensor(name, ref.dtype, (keep,), None, data=lambda v=vals: struct.pack(f"<{len(v)}f", *v)))
        else:
            out.append(OutTensor(name, ref.dtype, (keep,) + tuple(ref.shape[1:]), ref, slice=(0, ids)))
    return out


def _rebalanced_bias(tensors: dict, kept: dict, scores: dict, notes: list) -> dict:
    """new router bias per layer (float32 list) when the measurement holds the loads of the pruned model; {} (and a note) otherwise"""
    loads = scores.get("pruned_load")
    if not loads:
        notes.append("The router bias was not rebalanced: the measurement holds no load of the pruned model. Run the measurement with this cut to get it.")
        return {}
    from .slice_io import read_array
    out = {}
    for name, ref in tensors.items():
        layer, role, _ = moe_layout.classify(name, moe_layout.INKLING)
        if role != "router_bias" or str(layer) not in loads:
            continue
        old = [float(x) for x in read_array(ref).tolist()]
        sliced = [old[e] for e in kept[layer]]
        load = loads[str(layer)]
        row = scores["layers"][str(layer)]["freq"]
        total = sum(row[e] for e in kept[layer]) or 1.0
        target = [row[e] / total for e in kept[layer]]
        out[name] = moe_layout.rebalance_bias(sliced, load, target)
    notes.append("The router bias of every layer was nudged toward the original load of the kept experts (no training).")
    return out


def _new_quant_config(src_dir: Path, drop_mtp: bool):
    f = src_dir / "hf_quant_config.json"
    if not f.is_file():
        return None
    q = json.loads(f.read_text(encoding="utf-8"))
    if drop_mtp:
        q.pop("mtp", None)
    return q


def run(cfg: dict, src_dir: Path, config: dict) -> int:
    started = time.monotonic()
    out = Path(cfg["out"])
    final = Path(cfg["final"]) if cfg.get("final") else out.with_name("model")
    stop_file = out.parent / "STOP"
    if final.exists():
        raise ReshapeError(f"{final} already exists.")
    if cfg.get("mode") != "prune":
        raise ReshapeError("Only cutting is possible for Inkling-2-Mini; growing is a loop-pass or experts-per-token change (Try first).")
    if cfg.get("width"):
        raise ReshapeError("Thinning the layers' width does not apply to a mixture of experts; remove experts instead.")
    want = kinds_of(cfg)
    exp = want["experts"]
    notes: list = []
    protocol.progress("inspect", 0.0, "Reading the model's layout")
    try:
        new_cfg = moe_layout.fix_config_inkling(config, passes=want["passes"], top_k=want["top_k"], experts=exp["keep"] if exp else None, notes=notes)
    except moe_layout.MoeError as exc:
        raise ReshapeError(str(exc)) from None
    n = int(moe_layout.text_cfg(config)["num_hidden_layers"])
    tensors = safetensors_io.list_tensors(src_dir)
    groups = moe_layout.group_params_moe(config, {k: (r.dtype, r.shape) for k, r in tensors.items()}, n)
    protocol.progress("inspect", 1.0, f"{n} layers, {moe_layout.experts_of(config)} experts")
    kept = scores = None
    bias_new: dict = {}
    scoring = (exp or {}).get("scoring", "reap")
    if exp:
        scores = read_scores(exp.get("scores_dir"))
        try:
            kept = moe_layout.select_experts(scores, exp["keep"], scoring, layers=list(range(n)))
        except moe_layout.MoeError as exc:
            raise ReshapeError(str(exc)) from None
        if exp.get("rebalance", True):
            bias_new = _rebalanced_bias(tensors, kept, scores, notes)
    drop_mtp = "mtp_config" in config and "mtp_config" not in new_cfg
    drafter = src_dir / DRAFTER
    keep_drafter = drafter.is_file() and not drop_mtp
    if keep_drafter:
        notes.append("The speculative-decoding (multi-token) head file was copied unchanged; it was not healed, so speculative decoding accepts fewer tokens.")
    own = {k: r for k, r in tensors.items() if Path(r.file).name != DRAFTER}        # without an index the drafter's tensors are found too; it is a file of its own
    plan = None
    if kept:
        plan = _expert_plan(own, kept, exp["keep"], bias_new)
        need = sum(t.nbytes for t in plan)
        files = []
    else:
        files = sorted({Path(r.file) for r in own.values()})
        need = sum(f.stat().st_size for f in files)
    if keep_drafter:
        need += drafter.stat().st_size
    free = fsutil.free_bytes(out.parent)
    if free < need * 1.05:
        protocol.error("disk_full", f"There is not enough free space: {need * 1.05 / 1e9:.1f} GB needed in {out.parent}, {free / 1e9:.1f} GB free.")
        return 1
    if out.exists():
        shutil.rmtree(out)
    fsutil.ensure_parent(out)
    fsutil.ensure_parent(final)
    out.mkdir(parents=True)
    done = [0]
    last = [-1.0]

    def on_bytes(k):
        done[0] += k
        if need and (done[0] / need - last[0] >= 0.01 or done[0] == need):
            last[0] = done[0] / need
            protocol.progress("write", done[0] / need, f"Writing the new model: {done[0] / 1e9:.1f} of {need / 1e9:.1f} GB")
    try:
        with protocol.timed("write", bytes=need):
            if plan is not None:
                safetensors_io.write_shards(plan, out, shard_bytes=int(float(cfg.get("shard_gb", 5)) * 2**30), on_progress=lambda d, t: on_bytes(d - done[0]), should_stop=lambda: stop_file.exists())
            else:
                for f in files:
                    _copy_file(f, out / f.name, stop_file, on_bytes)
                index_src = src_dir / safetensors_io.INDEX_NAME
                if index_src.is_file():
                    shutil.copy2(index_src, out / safetensors_io.INDEX_NAME)
            if keep_drafter:
                _copy_file(drafter, out / DRAFTER, stop_file, on_bytes)
        (out / "config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        q = _new_quant_config(src_dir, drop_mtp)
        if q is not None:
            (out / "hf_quant_config.json").write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
        from . import reshapejob
        reshapejob._copy_side_files(src_dir, out)
        for d in EXTRA_DIRS:
            if (src_dir / d).is_dir():
                shutil.copytree(src_dir / d, out / d)
        lp_old, lp_new = moe_layout.loop_of(config), moe_layout.loop_of(new_cfg)
        moe_meta = {"passes": [lp_old["passes"], lp_new["passes"]], "top_k": [moe_layout.top_k_of(config), moe_layout.top_k_of(new_cfg)],
                    "experts": [moe_layout.experts_of(config), moe_layout.experts_of(new_cfg)],
                    "visits": [moe_layout.visits(config), moe_layout.visits(new_cfg)], "weights": "sliced along the expert axis, packed bytes unchanged" if kept else "copied unchanged",
                    "mtp": "removed" if drop_mtp else "kept, not healed" if drafter.is_file() else "none"}
        if kept:
            cov = moe_layout.coverage(scores, kept)
            moe_meta.update({"scoring": scoring, "scores_key": Path(exp.get("scores_dir") or "").name or None, "kept": "moe_kept.json", "rebalanced": bool(bias_new),
                             "coverage_min": round(min(cov.values()), 4), "coverage_mean": round(sum(cov.values()) / len(cov), 4)})
            (out / "moe_kept.json").write_text(json.dumps({"scoring": scoring, "experts_before": moe_layout.experts_of(config), "kept": {str(k): v for k, v in kept.items()}}), encoding="utf-8")
        if kept:
            lay = moe_layout.prune_layout({k: (r.dtype, r.shape) for k, r in own.items()}, kept, exp["keep"])
            after = moe_layout.group_params_moe(new_cfg, {k: (r.dtype, (len(lay[k]),) + tuple(r.shape[1:]) if k in lay else tuple(r.shape)) for k, r in own.items()}, n)
        else:
            after = groups
        meta = {"format": 3, "tool": "ND Lore Forge", "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "origin": {"repo": cfg["source"]["repo"], "family": cfg["source"].get("family"), "layers": n, "params": groups["total"]},
                "mode": "prune", "period": moe_layout.block_size_of(config), "layers": n, "params": after["total"], "src_layers": list(range(n)), "dropped": [], "duplicated": [],
                "zero_init": [], "strategy": "config" if not kept else f"experts-{scoring}", "calibration": None, "loss_before": None, "loss_after": None, "stable": None,
                "vision": "unchanged (copied from the original)" if groups["vision"] else "none", "mtp": moe_meta["mtp"], "shape": "moe", "moe": moe_meta}
        (out / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        what = ", ".join(x for x in (f"loop passes {lp_old['passes']} to {lp_new['passes']}" if want["passes"] is not None else "",
                                     f"experts per token {moe_meta['top_k'][0]} to {moe_meta['top_k'][1]}" if want["top_k"] is not None else "",
                                     f"experts per layer {moe_meta['experts'][0]} to {moe_meta['experts'][1]}" if exp else "") if x)
        (out / "README.md").write_text(f"---\n{reshapejob._licence_front_matter(src_dir / 'README.md')}---\n# {out.parent.name}\n\nCut from {cfg['source']['repo']}: {what}.\n\n"
                                       "Made with ND Lore Forge (Slim or grow a model). The expert weights are the original's, byte for byte; see ndforge_reshape.json.\n", encoding="utf-8")
        fsutil.replace_retry(out, final)
    except safetensors_io.Cancelled:
        shutil.rmtree(out, ignore_errors=True)
        protocol.result({"stopped": True, "model_dir": None})
        return 0
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    protocol.progress("done", 1.0, "")
    protocol.result({"model_dir": str(final), "mode": "prune", "layers_before": n, "layers_after": n, "src_layers": list(range(n)), "dropped": [], "duplicated": [], "zero_init": [],
                     "params_before": groups["total"], "params_after": after["total"], "params_breakdown": reshape.breakdown(after, list(range(n))), "bytes_written": need,
                     "loss_before": None, "loss_after": None, "ppl_before": None, "ppl_after": None, "stable": None, "boundary_distance": None, "chosen_blocks": None, "load": None,
                     "seconds": round(time.monotonic() - started, 1), "notes": notes, "stopped": False, "shape": "moe", "width": None, "moe": moe_meta})
    return 0
