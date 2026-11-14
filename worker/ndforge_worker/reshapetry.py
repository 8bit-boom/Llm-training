"""The "Try before cutting" job of "Slim or grow a model": what a cut will cost, found out BEFORE a new model folder is written.

    reshape_try   {"job": "reshape_try", "source": {"repo", "family"}, "load", "out_dir": "<project>/models/_scores/<key>", "calibration": {"train", "samples", "max_len", "seed",
                   "version"}, "measure_dir": "<folder of the measurement>", "measure_samples": 256, "protect": {"first", "last"}, "strategy": "contiguous",
                   "want": {"blocks": true, "runs": [4], "widths": [13312], "choices": [{"blocks": [9, 10], "width": 12288}]}, "plan_only": false, "force": false}

The original model is loaded once, in memory, and every question is one pass over the project's own calibration text (the same batches, the same loss as the
`loss_before` / `loss_after` numbers of the `reshape` job). A cut is imitated with hooks: the layers of a removed block pass their input on unchanged, the MLP neurons
that a width cut removes are zeroed in front of `down_proj` (the torch tests prove that this equals the written model). Nothing is written except `reshape_try.json`,
saved after every pass, so a stop or a crash keeps what was done and a repeated "Try" skips the passes it already has.

This file has the parts that need no torch (which passes, the choice of the cheapest run, the colour bands, the cache key, the stability check, the time estimate);
the part that runs the model is below them."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from . import protocol, reshape, reshape_width
from .reshape import ReshapeError

TRY_FORMAT = 1
TRY_FILE = "reshape_try.json"

# Colour bands of the relative increase of the text-prediction error (new loss / original loss - 1). They are RULES OF THUMB, not measurements of anything: below the
# first edge a cut is usually mild, above the second it usually needs a lot of healing. The page and docs/TRY_FIRST.md say so. The one place to change them.
BAND_EDGES = (0.03, 0.10)
BAND_NAMES = ("small", "noticeable", "large")
# the two halves of the text disagree about the ranking when fewer than this share of the pairs is ordered the same way (or the cheapest differs)
AGREE_PAIRS = 0.8
# how long one pass takes when nothing was measured yet: effective operations per second of the whole pass, by the way the model is loaded (a rough guess)
GUESS_FLOPS = {"16bit": 4.0e13, "4bit": 1.5e13, "offload": 2.0e12}
GUESS_TOKENS_PER_SAMPLE = 256


# ------------------------------------------------------------------------------------------------------------ pass keys

def pass_key(blocks, width=None) -> str:
    """The name of one question: which blocks are skipped and which MLP width is kept ("d=" and "w=" empty = the original). Sorted, so the same set is the same key."""
    b = ",".join(str(int(x)) for x in sorted(set(blocks or ())))
    return f"d={b}|w={int(width) if width else ''}"


def parse_key(key: str):
    """(blocks tuple, width or None) of a pass key."""
    d, w = key.split("|")
    blocks = tuple(int(x) for x in d[2:].split(",") if x)
    return blocks, (int(w[2:]) if w[2:] else None)


BASE_KEY = pass_key(())


def candidates(nb: int, pf: int = 1, pl: int = 1) -> list[int]:
    """The blocks that may be cut: never the first pf or the last pl."""
    return list(range(pf, nb - pl))


def windows(nb: int, k: int, pf: int = 1, pl: int = 1) -> list[tuple]:
    """Every run of k neighbouring blocks that may be cut, in order of the first block."""
    if k < 1:
        raise ReshapeError("Choose at least one block.")
    return [tuple(range(a, a + k)) for a in range(pf, nb - pl - k + 1)]


def check_choice(blocks, nb: int, pf: int = 1, pl: int = 1) -> tuple:
    """A set of blocks a person typed or clicked -> the sorted tuple; ReshapeError unless it is allowed (the same rules as removing layers)."""
    try:
        out = tuple(sorted({int(x) for x in blocks or ()}))
    except (TypeError, ValueError):
        raise ReshapeError("The blocks must be whole numbers.") from None
    if not out:
        return out
    if out[0] < pf or out[-1] >= nb - pl:
        raise ReshapeError("The first and the last block are protected and cannot be removed.")
    if len(out) > reshape.limits(nb, "prune", pf, pl)[1]:
        raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
    return out


def check_widths(dims: dict, widths, divisor: int = reshape_width.FFN_DIVISOR) -> list[int]:
    """The MLP widths asked for -> ascending, without duplicates; ReshapeError unless each is a size the page offers."""
    out = []
    for w in widths or ():
        out.append(reshape_width.check_width(dims, {"intermediate": int(w)}, divisor)["intermediate"])
    return sorted(set(out))


def expand_want(want: dict, nb: int, pf: int, pl: int, strategy: str = "contiguous", matrix=None, dims: dict | None = None,
                divisor: int = reshape_width.FFN_DIVISOR) -> list[str]:
    """The pass keys a request needs, in the order they are run, each once: the original first, then the person's own choices (the most wanted answers), the single
    blocks, the runs and the widths. With strategy "separate" the program's own default choice for a run length is not a run of neighbours, so it is added too."""
    want = want or {}
    keys = [BASE_KEY]
    if (want.get("widths") or any(c.get("width") for c in want.get("choices") or ())) and dims is None:
        raise ReshapeError("Thinning the layers is not possible for this model.")
    for ch in want.get("choices") or ():
        blocks = check_choice(ch.get("blocks") or (), nb, pf, pl)
        width = check_widths(dims, [ch["width"]], divisor)[0] if ch.get("width") else None
        if not blocks and width is None:
            raise ReshapeError("Choose at least one block or a width to try.")
        keys.append(pass_key(blocks, width))
    if want.get("blocks"):
        keys += [pass_key((b,)) for b in candidates(nb, pf, pl)]
    lo, hi = reshape.limits(nb, "prune", pf, pl)
    for k in sorted({int(x) for x in want.get("runs") or ()}):
        if not lo <= k <= hi:
            raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
        keys += [pass_key(w) for w in windows(nb, k, pf, pl)]
        if strategy == "separate" and matrix:
            keys.append(pass_key(reshape.choose_drop(matrix, k, "separate", pf, pl)))
    for w in check_widths(dims, want.get("widths"), divisor) if want.get("widths") else ():
        keys.append(pass_key((), w))
    return list(dict.fromkeys(keys))


# ----------------------------------------------------------------------------------------------------------- numbers

def rel_increase(loss, base):
    """loss / base - 1 (None when either is unknown)."""
    if loss is None or not base:
        return None
    return loss / base - 1.0


def band_of(rel):
    """"small" / "noticeable" / "large" by BAND_EDGES (None for an unknown value); a drop in the error counts as small."""
    if rel is None:
        return None
    return BAND_NAMES[sum(1 for e in BAND_EDGES if rel >= e)]


def ppl_of(loss):
    return None if loss is None else round(math.exp(min(loss, 50.0)), 3)


def row_of(passes: dict, key: str, base_loss) -> dict:
    """One answer as the page shows it: loss, change, relative increase, perplexity, band (all None while the pass has not been run)."""
    p = passes.get(key)
    loss = p["loss"] if p else None
    rel = rel_increase(loss, base_loss)
    return {"key": key, "tried": p is not None, "loss": loss, "delta": None if loss is None or base_loss is None else loss - base_loss, "rel": rel,
            "ppl": ppl_of(loss), "band": band_of(rel), "loss_h1": p.get("loss_h1") if p else None, "loss_h2": p.get("loss_h2") if p else None}


def pair_agreement(a: list, b: list):
    """The share of the pairs (i, j) that the two lists put in the same order, ties in either list left out (None when no pair is left)."""
    same = total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            total += 1
            same += (da > 0) == (db > 0)
    return same / total if total else None


def stability(rows: list) -> dict | None:
    """Do the two halves of the measuring text agree about the ranking of these answers? {"agree", "best_agree", "pairs"} (None with fewer than two answers).
    agree = the cheapest is the same in both halves and at least AGREE_PAIRS of the pairs are ordered the same way."""
    rows = [r for r in rows if r["loss_h1"] is not None and r["loss_h2"] is not None]
    if len(rows) < 2:
        return None
    h1, h2 = [r["loss_h1"] for r in rows], [r["loss_h2"] for r in rows]
    best = lambda v: min(range(len(v)), key=lambda i: (v[i], i))  # noqa: E731
    best_agree = best(h1) == best(h2)
    pairs = pair_agreement(h1, h2)
    return {"agree": bool(best_agree and (pairs is None or pairs >= AGREE_PAIRS)), "best_agree": bool(best_agree), "pairs": pairs, "n": len(rows)}


def cheapest(rows: list):
    """The run with the lowest loss among the tried ones (a tie: the earlier one); None when none was tried."""
    done = [r for r in rows if r["loss"] is not None]
    return min(done, key=lambda r: r["loss"]) if done else None


def mean_seconds(passes: dict):
    """The average time of a pass in seconds (None when none was timed)."""
    t = [p["seconds"] for p in passes.values() if p.get("seconds") is not None]
    return sum(t) / len(t) if t else None


def guess_seconds_per_pass(params_b: float, tokens: float, load: str) -> float:
    """Before anything was timed: 2 operations per parameter and token, at a guessed speed of the way the model is loaded. A guess, and the page says so."""
    return max(1.0, 2.0 * params_b * 1e9 * tokens / GUESS_FLOPS.get(load, GUESS_FLOPS["16bit"]))


def estimate(todo: int, known, tokens, known_tokens, params_b, load: str, guess_tokens: float) -> dict:
    """The time of `todo` passes: {"seconds_per_pass", "seconds", "guess"}. A timed pass (`known` seconds over `known_tokens`) is scaled to `tokens`."""
    if known:
        s = known * (tokens / known_tokens if tokens and known_tokens else 1.0)
        guess = False
    else:
        s, guess = guess_seconds_per_pass(params_b, tokens or guess_tokens, load), True
    return {"seconds_per_pass": s, "seconds": s * todo, "guess": guess}


# ----------------------------------------------------------------------------------------------------- the cache key

def data_hash(path) -> str | None:
    """A short hash of the calibration file (None when there is none): another text is another measurement."""
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def weights_fingerprint(folder) -> str:
    """What identifies the weights of a model folder: its config and the names and sizes of the weight files (the commit of a Hub snapshot when it is one)."""
    folder = Path(folder)
    if folder.parent.name == "snapshots":
        return "hub:" + folder.name
    h = hashlib.sha1()
    cfg = folder / "config.json"
    h.update(cfg.read_bytes() if cfg.is_file() else b"")
    for f in sorted(folder.glob("*.safetensors")):
        h.update(f"{f.name}:{f.stat().st_size};".encode())
    return "dir:" + h.hexdigest()[:16]


def build_key(repo: str, weights: str | None, calib: dict, train_hash: str | None, load: str) -> dict:
    """Everything the answers depend on. The same key is the same question; another key is another set of answers."""
    return {"format": TRY_FORMAT, "source": repo, "weights": weights, "data_version": calib.get("version"), "data_hash": train_hash, "samples": calib.get("samples"),
            "max_len": calib.get("max_len"), "seed": calib.get("seed", 3407), "load": load}


KEY_WORDS = {"format": "an older file format", "source": "another model", "weights": "other weights of the model", "data_version": "another version of the training data",
             "data_hash": "other training text", "samples": "another number of samples", "max_len": "another text length", "seed": "another seed",
             "load": "another way of loading the model"}


def key_diff(old, new: dict):
    """None when the saved answers belong to this question; else the plain reason they do not."""
    if not isinstance(old, dict):
        return "the saved answers could not be read"
    for k in new:
        if old.get(k) != new[k]:
            return KEY_WORDS.get(k, k)
    return None


# ------------------------------------------------------------------------------------------------------- the results

def empty_results(key: dict) -> dict:
    return {"format": TRY_FORMAT, "key": key, "passes": {}, "tokens": None, "load_used": None, "seconds_per_pass": None, "created": None, "analysis": None}


def read_results(folder, key: dict | None = None, log=None):
    """The saved answers of `folder` (None when there are none). With `key`, answers of another question are dropped and `log` is told why."""
    f = Path(folder) / TRY_FILE
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("passes"), dict):
            raise ValueError("not a results file")
    except (OSError, ValueError) as exc:
        if log:
            log(f"The earlier answers could not be read ({exc}); trying again from the start.")
        return None
    if key is not None:
        why = key_diff(data.get("key"), key)
        if why:
            if log:
                log(f"The earlier answers belong to {why}; they are computed again.")
            return None
    return data


def write_results(folder, data: dict) -> None:
    """Atomically (a crash never leaves half a file)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / (TRY_FILE + ".partial")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, folder / TRY_FILE)


def merge_pass(data: dict, key: str, result: dict) -> dict:
    """Adds one finished pass (the first answer for a key is kept: a pass is never overwritten) and updates the timing."""
    data["passes"].setdefault(key, result)
    data["seconds_per_pass"] = mean_seconds(data["passes"])
    return data


# ---------------------------------------------------------------------------------------------------------- the analysis

def analyse(passes: dict, nb: int, period: int, pf: int, pl: int, matrix=None, strategy: str = "contiguous") -> dict:
    """What the page shows, from the answers so far: the original's loss, one row per block, per run length every run with the cheapest next to the program's own
    default choice, the width curve, and every other choice. `matrix`: the block-boundary distances of the measurement (the "influence"), when there is one."""
    base = passes.get(BASE_KEY)
    base_loss = base["loss"] if base else None
    blocks_rows = []
    for b in candidates(nb, pf, pl):
        r = row_of(passes, pass_key((b,)), base_loss)
        r.update({"block": b, "layers": reshape.blocks_to_layers([b], period), "influence": matrix[b][b + 1] if matrix else None})
        blocks_rows.append(r)
    runs: dict = {}
    singles, widths, others = set(pass_key((b,)) for b in candidates(nb, pf, pl)), [], []
    by_k: dict = {}
    for key in passes:
        blocks, width = parse_key(key)
        if key == BASE_KEY:
            continue
        if not blocks and width:
            widths.append(row_of(passes, key, base_loss) | {"intermediate": width})
        elif width is None and len(blocks) >= 2 and blocks == tuple(range(blocks[0], blocks[0] + len(blocks))):
            by_k.setdefault(len(blocks), []).append(blocks)
        elif key not in singles:
            others.append(key)
    for k, ws in sorted(by_k.items()):
        rows = []
        for a in range(pf, nb - pl - k + 1):
            blocks = tuple(range(a, a + k))
            r = row_of(passes, pass_key(blocks), base_loss)
            r.update({"start": a, "blocks": list(blocks), "layers": reshape.blocks_to_layers(list(blocks), period), "distance": matrix[a][a + k] if matrix else None})
            rows.append(r)
        default = None
        if matrix:
            try:
                chosen = reshape.choose_drop(matrix, k, strategy, pf, pl)
                default = row_of(passes, pass_key(chosen), base_loss) | {"blocks": chosen, "layers": reshape.blocks_to_layers(chosen, period)}
            except ReshapeError:
                default = None
        best = cheapest(rows)
        runs[str(k)] = {"k": k, "windows": rows, "best": best, "default": default, "complete": all(r["tried"] for r in rows), "stability": stability(rows)}
    # the single blocks are runs of length one
    if any(r["tried"] for r in blocks_rows):
        rows = [dict(r, start=r["block"], blocks=[r["block"]], distance=r["influence"]) for r in blocks_rows]
        default = None
        if matrix:
            try:
                chosen = reshape.choose_drop(matrix, 1, strategy, pf, pl)
                default = next((r for r in rows if r["blocks"] == chosen), None)
            except ReshapeError:
                default = None
        runs["1"] = {"k": 1, "windows": rows, "best": cheapest(rows), "default": default, "complete": all(r["tried"] for r in rows), "stability": stability(rows)}
    widths.sort(key=lambda r: r["intermediate"])
    choices = []
    for key in others:
        blocks, width = parse_key(key)
        choices.append(row_of(passes, key, base_loss) | {"blocks": list(blocks), "layers": reshape.blocks_to_layers(list(blocks), period), "intermediate": width})
    return {"base": {"loss": base_loss, "ppl": ppl_of(base_loss), "tokens": base.get("tokens") if base else None},
            "bands": {"edges": list(BAND_EDGES), "names": list(BAND_NAMES), "agree_pairs": AGREE_PAIRS},
            "blocks": blocks_rows, "blocks_stability": stability(blocks_rows), "runs": runs, "widths": widths, "widths_stability": stability(widths), "choices": choices,
            "passes": len(passes)}


# ------------------------------------------------------------------------------------------------ running the model (needs torch)

class _Stopped(Exception):
    """The stop file appeared: the pass in progress is dropped, the finished ones are kept."""


def describe_key(key: str) -> str:
    """"block 5 skipped", "blocks 9-12 skipped, MLP 13,312": what a pass does, for the progress line."""
    blocks, width = parse_key(key)
    parts = []
    if blocks:
        parts.append((f"block {blocks[0]}" if len(blocks) == 1 else "blocks " + (f"{blocks[0]}-{blocks[-1]}" if blocks == tuple(range(blocks[0], blocks[0] + len(blocks)))
                                                                                  else ", ".join(map(str, blocks)))) + " skipped")
    if width:
        parts.append(f"MLP {width:,}")
    return ", ".join(parts) or "the original"


def _passthrough(*args, **kwargs):
    """What a skipped layer does: its input is its output."""
    h = args[0] if args else kwargs["hidden_states"]
    return h


class _Cut:
    """Imitates a cut inside the loaded model while the `with` block runs: the layers of the skipped blocks pass their input on, and the MLP neurons that a width cut
    removes are zeroed in front of `down_proj` (a zero column of down_proj is a removed neuron: the torch tests prove that this equals the written model)."""

    def __init__(self, layers, skip_layers, masks):
        self.layers, self.skip, self.masks, self.handles = layers, list(skip_layers), masks or {}, []

    def __enter__(self):
        for i in self.skip:
            self.layers[i].forward = _passthrough
        for s, m in self.masks.items():
            def hook(module, args, m=m):
                x = args[0]
                return (x * m.to(device=x.device, dtype=x.dtype),) + tuple(args[1:])
            self.handles.append(self.layers[s].mlp.down_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        for i in self.skip:
            self.layers[i].__dict__.pop("forward", None)
        return False


def _masks_for(mlp_scores, layer_ids, width: int, size: int) -> dict:
    """{layer: float mask of `size` neurons, 1 for the `width` kept ones}: the neurons the written model keeps (reshape_width.keep_top on the same scores)."""
    import torch
    out = {}
    for s in layer_ids:
        m = torch.zeros(size, dtype=torch.float32)
        m[torch.as_tensor(reshape_width.keep_top([float(x) for x in mlp_scores[s]], width), dtype=torch.long)] = 1.0
        out[s] = m
    return out


def _one_pass(model, layers, batches, period, key: str, mlp_scores, size: int, stop) -> dict:
    """One pass over the batches with the cut of `key` imitated. The two halves are the batches of even and odd position (as in the measurement)."""
    import torch
    from . import reshapejob
    blocks, width = parse_key(key)
    skip = reshape.blocks_to_layers(list(blocks), period)
    masks = _masks_for(mlp_scores, [i for i in range(len(layers)) if i not in set(skip)], width, size) if width else {}
    dev = next(model.parameters()).device
    sums, counts = [0.0, 0.0], [0, 0]
    started = time.monotonic()
    with torch.no_grad(), _Cut(layers, skip, masks):
        for bi, b in enumerate(batches):
            if stop():
                raise _Stopped()
            ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            ls, ln = reshapejob._token_losses(model(input_ids=ids, attention_mask=mask, use_cache=False).logits, ids, mask)
            sums[bi % 2] += ls
            counts[bi % 2] += ln
    n = counts[0] + counts[1]
    h1 = sums[0] / counts[0] if counts[0] else None
    return {"loss": (sums[0] + sums[1]) / n if n else None, "loss_h1": h1, "loss_h2": sums[1] / counts[1] if counts[1] else h1, "tokens": n,
            "seconds": round(time.monotonic() - started, 3)}


def _config_and_dir(repo: str, download: bool):
    """(config.json, folder) of the model. Without `download` a Hub model that is not on this computer gives (config, None) and nothing is downloaded."""
    from . import reshapejob
    if Path(repo).is_dir():
        d = Path(repo)
    elif download:
        d = reshapejob._resolve_source(repo)
    else:
        d = None
        try:
            from huggingface_hub import try_to_load_from_cache
            f = try_to_load_from_cache(repo, "config.json")
            d = Path(f).parent if isinstance(f, (str, os.PathLike)) else None
        except Exception:
            d = None
    if d is None:
        return reshapejob._hub_config(repo), None
    return json.loads((d / "config.json").read_text(encoding="utf-8")), d


def run_try(cfg: dict) -> int:
    """job reshape_try (see the top of this file). The result: {"analysis", "passes_total", "passes_done", "stopped", "seconds_per_pass", "load", "tokens", "dir"}; with
    cfg["plan_only"] only the number of passes and the time they will take (no model is loaded, no torch is needed)."""
    from . import reshapejob
    protocol.reserve_stdout()
    repo = cfg["source"]["repo"]
    plan_only = bool(cfg.get("plan_only"))
    config, src_dir = _config_and_dir(repo, not plan_only)
    reshape.check_supported(config)
    tc = reshape.text_config(config)
    types = reshape.layer_types_of(tc)
    period = reshapejob.check_period(config, types)
    n_layers = int(tc["num_hidden_layers"])
    nb = reshape.block_count(n_layers, period)
    protect = cfg.get("protect") or {}
    pf, pl = int(protect.get("first", 1)), int(protect.get("last", 1))
    strategy = cfg.get("strategy", "contiguous")
    why = reshape_width.width_supported(config)
    dims = None if why else reshape_width.width_dims(config)
    divisor = int(cfg.get("ffn_divisor", reshape_width.FFN_DIVISOR))
    want = cfg.get("want") or {}
    wants_width = bool(want.get("widths")) or any(c.get("width") for c in want.get("choices") or ())
    if wants_width and why:
        raise ReshapeError(why)
    calib = cfg.get("calibration") or {}
    if not (calib.get("train") and Path(calib["train"]).is_file()):
        raise ReshapeError("Trying a cut is measured on your own text. Save a training-data version in Step 2 first.")
    meas = None
    if dims and cfg.get("measure_dir"):
        meas = reshapejob._read_scores(cfg["measure_dir"], {"source": cfg["source"], "calibration": {"version": calib.get("version"), "samples": cfg.get("measure_samples"),
                                                                                                    "max_len": calib.get("max_len")}}, n_layers, dims, types)
    needs_measurement = wants_width and meas is None
    if needs_measurement and not plan_only:
        raise ReshapeError("Trying a width needs the measurement of the neurons. Measure the model first.")
    matrix = (meas["meta"].get("matrix") or None) if meas else None
    keys = expand_want(want, nb, pf, pl, strategy, matrix, dims, divisor)
    folder = Path(cfg["out_dir"])
    if cfg.get("force") and (folder / TRY_FILE).is_file():
        (folder / TRY_FILE).unlink()
    key = build_key(repo, weights_fingerprint(src_dir) if src_dir else None, calib, data_hash(calib["train"]), cfg.get("load", "auto"))
    data = read_results(folder, key if src_dir else None, protocol.log) if src_dir else None
    if data is None:
        data = empty_results(key)
    todo = [k for k in keys if k not in data["passes"]]
    mm = meas["meta"] if meas else {}
    known, known_tokens = data["seconds_per_pass"], data["tokens"]
    if known is None and mm.get("seconds_per_pass"):
        known, known_tokens = mm["seconds_per_pass"], mm.get("tokens")
    tokens = data["tokens"] or (mm["tokens"] * calib["samples"] / cfg["measure_samples"] if mm.get("tokens") and cfg.get("measure_samples") and calib.get("samples") else None)
    est = estimate(len(todo), known, tokens, known_tokens, float(cfg.get("params_b") or 10.0), cfg.get("load", "auto"),
                   int(calib.get("samples", 64)) * min(int(calib.get("max_len", 512)), GUESS_TOKENS_PER_SAMPLE))
    plan = {"passes_total": len(keys), "passes_cached": len(keys) - len(todo), "passes_todo": len(todo), "needs_measurement": needs_measurement, **est}
    if plan_only:
        protocol.result(plan)
        return 0

    def finish(stopped: bool) -> int:
        data["analysis"] = analyse(data["passes"], nb, period, pf, pl, matrix, strategy)
        write_results(folder, data)
        protocol.progress("try", 1.0, "")
        protocol.result({"analysis": data["analysis"], "passes_total": len(keys), "passes_done": len(keys) - len([k for k in keys if k not in data["passes"]]),
                         "stopped": stopped, "seconds_per_pass": data["seconds_per_pass"], "load": data["load_used"], "tokens": data["tokens"], "dir": str(folder)})
        return 0

    if not todo:
        protocol.log("Every answer is already saved; nothing is run.")
        return finish(False)
    reshapejob._require_torch()
    import transformers
    if int(transformers.__version__.split(".")[0]) < 5:          # before 5 a decoder layer returns a tuple: a skipped layer would need to as well (not tested)
        raise ReshapeError("Trying a cut needs a newer version of the training software (Training setup, update).")
    est_bytes = sum(r.nbytes for r in reshapejob.safetensors_io.list_tensors(src_dir).values())
    mode = reshapejob._choose_load(cfg.get("load", "auto"), est_bytes)
    if data["load_used"] not in (None, mode):
        protocol.log(f"The earlier answers were made with the model loaded as {data['load_used']}; they are computed again as {mode}.")
        data = empty_results(key)
        todo = list(keys)
    stop_file = folder.parent / "STOP"
    stop = stop_file.exists
    from transformers import AutoTokenizer
    protocol.progress("calibrate", 0.0, "Preparing the text")
    batches = reshapejob._calibration(AutoTokenizer.from_pretrained(str(src_dir)), calib, cfg["source"].get("family", "generic"))
    data["tokens"] = sum(int(b["attention_mask"].sum()) for b in batches)
    data["load_used"] = mode
    protocol.log(f"Trying {len(todo)} cuts on {len(batches)} batches ({data['tokens']:,} tokens), model loaded as {mode}.")
    protocol.progress("calibrate", 1.0, "")
    protocol.progress("try", 0.0, f"Loading the model ({mode})")
    model = reshapejob._load(src_dir, mode)
    layers = reshapejob._find_layers(model, n_layers)
    mlp = meas["mlp"] if meas else None
    size = dims["intermediate"] if dims else 0
    stopped = False
    try:
        for i, k in enumerate(todo):
            if stop():
                stopped = True
                break
            protocol.progress("try", i / len(todo), f"Cut {i + 1} of {len(todo)}: {describe_key(k)}")
            try:
                try:
                    res = _one_pass(model, layers, batches, period, k, mlp, size, stop)
                except RuntimeError as exc:
                    if not reshapejob._is_oom(exc):
                        raise
                    protocol.log("Out of memory; trying again with one text at a time.")
                    reshapejob._free()
                    batches = reshapejob._split_single(batches)
                    res = _one_pass(model, layers, batches, period, k, mlp, size, stop)
            except _Stopped:
                stopped = True
                break
            except RuntimeError as exc:
                if reshapejob._is_oom(exc):
                    raise ReshapeError("The graphics card ran out of memory while trying the cuts. Choose 4-bit loading (Expert mode) or fewer samples.") from None
                raise
            merge_pass(data, k, res)
            data["analysis"] = analyse(data["passes"], nb, period, pf, pl, matrix, strategy)
            write_results(folder, data)
            protocol.emit("try_pass", key=k, loss=res["loss"], seconds=res["seconds"], done=i + 1, total=len(todo), seconds_per_pass=data["seconds_per_pass"],
                          analysis=data["analysis"])
    finally:
        del layers
        del model
        reshapejob._free()
    return finish(stopped)
