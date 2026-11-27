"""The "Slim or grow a model" job for Inkling-2-Mini (and later other mixtures of experts): cuts that change the config of a packed model, written as a new folder.

`reshapejob.run` hands the job here when the source is an Inkling folder. Kinds of cut (job keys; any mix may be given):

    passes   loop passes (3 -> 2 or 4): config only, the weights are copied (owner's decision D8: copy, never hard links)
    top_k    experts used per token: config only

Experts and whole blocks follow in `moejob` too (see docs/MOE_GUIDE.md). The expert tensors stay in the authors' packed format and are never decoded.
The result has the same keys as a dense cut (so the page and Auto-cut read it unchanged) plus "moe".
"""
from __future__ import annotations

import datetime
import json
import shutil
import time
from pathlib import Path

from . import fsutil, moe_layout, protocol, reshape, safetensors_io
from .reshape import ReshapeError

COPY_CHUNK = 64 * 2**20
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
    out = {"passes": cfg.get("passes"), "top_k": cfg.get("top_k")}
    for k, v in out.items():
        if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
            raise ReshapeError(f"{k} must be a whole number.")
    if out["passes"] is None and out["top_k"] is None:
        raise ReshapeError("Nothing to change: give the number of loop passes or of experts per token.")
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
    notes: list = []
    protocol.progress("inspect", 0.0, "Reading the model's layout")
    try:
        new_cfg = moe_layout.fix_config_inkling(config, passes=want["passes"], top_k=want["top_k"], notes=notes)
    except moe_layout.MoeError as exc:
        raise ReshapeError(str(exc)) from None
    n = int(moe_layout.text_cfg(config)["num_hidden_layers"])
    tensors = safetensors_io.list_tensors(src_dir)
    groups = moe_layout.group_params_moe(config, {k: (r.dtype, r.shape) for k, r in tensors.items()}, n)
    protocol.progress("inspect", 1.0, f"{n} layers, {moe_layout.experts_of(config)} experts")
    drop_mtp = "mtp_config" in config and "mtp_config" not in new_cfg
    drafter = src_dir / DRAFTER
    if drafter.is_file() and not drop_mtp:
        notes.append("The speculative-decoding (multi-token) head file was copied unchanged; it was not healed, so speculative decoding accepts fewer tokens.")
    files = sorted({Path(r.file) for r in tensors.values()})
    if drafter.is_file() and not drop_mtp and drafter not in files:
        files.append(drafter)
    index_src = src_dir / safetensors_io.INDEX_NAME
    need = sum(f.stat().st_size for f in files)
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
            protocol.progress("write", done[0] / need, f"Copying the weights: {done[0] / 1e9:.1f} of {need / 1e9:.1f} GB")
    try:
        with protocol.timed("write", bytes=need):
            for f in files:
                _copy_file(f, out / f.name, stop_file, on_bytes)
        if index_src.is_file():
            shutil.copy2(index_src, out / safetensors_io.INDEX_NAME)
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
                    "visits": [moe_layout.visits(config), moe_layout.visits(new_cfg)], "weights": "copied unchanged", "mtp": "removed" if drop_mtp else "kept, not healed" if drafter.is_file() else "none"}
        meta = {"format": 3, "tool": "ND Lore Forge", "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "origin": {"repo": cfg["source"]["repo"], "family": cfg["source"].get("family"), "layers": n, "params": groups["total"]},
                "mode": "prune", "period": moe_layout.block_size_of(config), "layers": n, "params": groups["total"], "src_layers": list(range(n)), "dropped": [], "duplicated": [],
                "zero_init": [], "strategy": "config", "calibration": None, "loss_before": None, "loss_after": None, "stable": None,
                "vision": "unchanged (copied from the original)" if groups["vision"] else "none", "mtp": moe_meta["mtp"], "shape": "moe", "moe": moe_meta}
        (out / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        what = ", ".join(x for x in (f"loop passes {lp_old['passes']} to {lp_new['passes']}" if want["passes"] is not None else "",
                                     f"experts per token {moe_meta['top_k'][0]} to {moe_meta['top_k'][1]}" if want["top_k"] is not None else "") if x)
        from . import reshapejob
        (out / "README.md").write_text(f"---\n{reshapejob._licence_front_matter(src_dir / 'README.md')}---\n# {out.parent.name}\n\nCut from {cfg['source']['repo']}: {what}.\n\n"
                                       "Made with ND Lore Forge (Slim or grow a model). The weights are the original's, byte for byte; see ndforge_reshape.json.\n", encoding="utf-8")
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
                     "params_before": groups["total"], "params_after": groups["total"], "params_breakdown": reshape.breakdown(groups, list(range(n))), "bytes_written": need,
                     "loss_before": None, "loss_after": None, "ppl_before": None, "ppl_after": None, "stable": None, "boundary_distance": None, "chosen_blocks": None, "load": None,
                     "seconds": round(time.monotonic() - started, 1), "notes": notes, "stopped": False, "shape": "moe", "width": None, "moe": moe_meta})
    return 0
