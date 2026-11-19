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

from . import protocol, reshape, reshape_width, safetensors_io, speed
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

def pass_key(blocks, width=None, scoring=None) -> str:
    """The name of one question: which blocks are skipped and which MLP width is kept ("d=" and "w=" empty = the original). Sorted, so the same set is the same key.
    A cut that ranks the neurons by their output (`scoring` "output") has the suffix "|s=output"; with the default ranking (activation) the key is exactly the old one."""
    b = ",".join(str(int(x)) for x in sorted(set(blocks or ())))
    key = f"d={b}|w={int(width) if width else ''}"
    if scoring and scoring != "activation":
        key += f"|s={scoring}"
    return key


def parse_key(key: str):
    """(blocks tuple, width or None) of a pass key (optional suffix segments such as "s=output" are ignored: see `key_options`)."""
    parts = key.split("|")
    d, w = parts[0], parts[1]
    blocks = tuple(int(x) for x in d[2:].split(",") if x)
    return blocks, (int(w[2:]) if w[2:] else None)


def key_options(key: str) -> dict:
    """The optional segments of a pass key as {name: value}, e.g. {"s": "output"}; empty for a key without them."""
    out = {}
    for seg in key.split("|")[2:]:
        name, _, value = seg.partition("=")
        out[name] = value
    return out


def plain_key(key: str) -> str:
    """The key of the same cut without the optional segments (what a variant is compared with)."""
    return "|".join(key.split("|")[:2])


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
        scoring = reshape_width.check_scoring(ch.get("scoring"))
        if scoring != "activation":
            if width is None:
                raise ReshapeError("Ranking the neurons by their output needs a width to try.")
            keys.append(pass_key(blocks, width))               # the same cut with the default ranking: what the variant is compared with
        keys.append(pass_key(blocks, width, scoring))
    if want.get("blocks"):
        keys += [pass_key((b,)) for b in candidates(nb, pf, pl)]
    lo, hi = reshape.limits(nb, "prune", pf, pl)
    for k in sorted({int(x) for x in want.get("runs") or ()}):
        if not lo <= k <= hi:
            raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
        keys += [pass_key(w) for w in windows(nb, k, pf, pl)]
        if strategy == "separate" and matrix:
            keys.append(pass_key(reshape.choose_drop(matrix, k, "separate", pf, pl)))
    compare = reshape_width.check_scoring(want.get("scoring")) if want.get("widths") else "activation"
    for w in check_widths(dims, want.get("widths"), divisor) if want.get("widths") else ():
        keys.append(pass_key((), w))
        if compare != "activation":
            keys.append(pass_key((), w, compare))
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


def build_key(repo: str, weights: str | None, calib: dict, train_hash: str | None, load: str, weights_sample: str | None = None) -> dict:
    """Everything the answers depend on. The same key is the same question; another key is another set of answers.
    `weights_sample` (safetensors_io.weights_manifest) is part of it only when given; saved answers without it are accepted (see key_diff)."""
    from . import reshapejob
    key = {"format": TRY_FORMAT, "source": repo, "weights": weights, "data_version": calib.get("version"), "data_hash": train_hash, "samples": calib.get("samples"),
           "max_len": calib.get("max_len"), "seed": calib.get("seed", 3407), "load": load, "reasoning": reshapejob.reasoning_key(calib)}
    if weights_sample:
        key["weights_sample"] = weights_sample
    return key


KEY_WORDS = {"format": "an older file format", "source": "another model", "weights": "other weights of the model", "data_version": "another version of the training data",
             "data_hash": "other training text", "samples": "another number of samples", "max_len": "another text length", "seed": "another seed",
             "load": "another way of loading the model", "reasoning": "another reasoning text", "weights_sample": "other weights of the model"}


def key_diff(old, new: dict):
    """None when the saved answers belong to this question; else the plain reason they do not."""
    if not isinstance(old, dict):
        return "the saved answers could not be read"
    for k in new:
        if k == "weights_sample" and k not in old:
            continue                                    # answers saved before the weights were sampled count as matching
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
    variant_keys = []
    for key in passes:
        blocks, width = parse_key(key)
        if key == BASE_KEY:
            continue
        if key_options(key):
            variant_keys.append(key)                        # a cut with another ranking of the neurons: shown on its own, next to the plain cut
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
    out = {"base": {"loss": base_loss, "ppl": ppl_of(base_loss), "tokens": base.get("tokens") if base else None},
           "bands": {"edges": list(BAND_EDGES), "names": list(BAND_NAMES), "agree_pairs": AGREE_PAIRS},
           "blocks": blocks_rows, "blocks_stability": stability(blocks_rows), "runs": runs, "widths": widths, "widths_stability": stability(widths), "choices": choices,
           "passes": len(passes)}
    if variant_keys:
        out["variants"] = variant_rows(passes, variant_keys, base_loss, period)
    return out


OPTION_WORDS = {"s": {"output": "neurons ranked by their output"}}


def key_label(key: str) -> str:
    """"MLP 13,312, neurons ranked by their output": what a pass with options does, in words."""
    return describe_key(key)


def variant_rows(passes: dict, keys: list, base_loss, period: int) -> list:
    """One row per pass with options: its loss, the change against the original and against the same cut without the options (when that was tried), and whether it is better
    in BOTH halves of the text (the only comparison the program calls a gain: one half alone can be luck)."""
    rows = []
    for key in sorted(keys):
        blocks, width = parse_key(key)
        r = row_of(passes, key, base_loss)
        plain = passes.get(plain_key(key))
        p_loss = plain["loss"] if plain else None
        both = None
        if plain and None not in (r["loss_h1"], r["loss_h2"], plain.get("loss_h1"), plain.get("loss_h2")):
            both = bool(r["loss_h1"] < plain["loss_h1"] and r["loss_h2"] < plain["loss_h2"])
        rows.append({**r, "label": key_label(key), "options": key_options(key), "plain_key": plain_key(key), "plain_loss": p_loss, "rel_vs_plain": rel_increase(r["loss"], p_loss),
                     "better_both_halves": both, "blocks": list(blocks), "layers": reshape.blocks_to_layers(list(blocks), period), "intermediate": width})
    return rows


# ------------------------------------------------------------------------------------------------ running the model (needs torch)

class _Stopped(Exception):
    """The stop file appeared: the pass in progress is dropped, the finished ones are kept."""


def describe_key(key: str) -> str:
    """"block 5 skipped", "blocks 9-12 skipped, MLP 13,312, neurons ranked by their output": what a pass does, for the progress line."""
    blocks, width = parse_key(key)
    parts = []
    if blocks:
        parts.append((f"block {blocks[0]}" if len(blocks) == 1 else "blocks " + (f"{blocks[0]}-{blocks[-1]}" if blocks == tuple(range(blocks[0], blocks[0] + len(blocks)))
                                                                                  else ", ".join(map(str, blocks)))) + " skipped")
    if width:
        parts.append(f"MLP {width:,}")
    parts += [OPTION_WORDS.get(n, {}).get(v, f"{n}={v}") for n, v in key_options(key).items()]
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


SEARCH_MAX_PASSES = 400
SEARCH_MAX_GROUPS = 8
SEARCH_ROWS_KEPT = 40
SEARCH_BUDGET_MIN = (5, 240)


def _search_wants_width(search) -> bool:
    return isinstance(search, dict) and any(c.get("width") for g in search.get("groups") or () if isinstance(g, dict) for c in g.get("cuts") or () if isinstance(c, dict))


def _search_context(search: dict, nb: int, pf: int, pl: int, strategy: str, matrix) -> dict:
    """The request's `want.search` -> the context of the round planner (checked): the groups of target cuts, the budget in seconds and the block limits of the model."""
    groups = search.get("groups")
    if not isinstance(groups, list) or not groups or len(groups) > SEARCH_MAX_GROUPS:
        raise ReshapeError(f"The search needs between 1 and {SEARCH_MAX_GROUPS} target sizes.")
    movable = len(candidates(nb, pf, pl))
    for g in groups:
        cuts = g.get("cuts") if isinstance(g, dict) else None
        if not isinstance(cuts, list) or not cuts:
            raise ReshapeError("Every target size of the search needs at least one cut.")
        for c in cuts:
            if not isinstance(c, dict) or int(c.get("k", -1)) < 0 or int(c["k"]) > movable:
                raise ReshapeError("A cut of the search takes out more blocks than may be cut.")
    minutes = float(search.get("budget_min", 20))
    if not SEARCH_BUDGET_MIN[0] <= minutes <= SEARCH_BUDGET_MIN[1]:
        raise ReshapeError(f"The time for the search is {SEARCH_BUDGET_MIN[0]} to {SEARCH_BUDGET_MIN[1]} minutes.")
    ctx = {k: v for k, v in search.items() if k not in ("budget_min",)}
    ctx.update(nb=nb, pf=pf, pl=pl, strategy=strategy, matrix=matrix, budget_s=minutes * 60.0, groups=groups)
    ctx["max_passes"] = min(int(search.get("max_passes", SEARCH_MAX_PASSES)), SEARCH_MAX_PASSES)
    return ctx
# what a pass costs before one was timed in this job: the measured seconds of the saved answers, else this guess (the same as the plan's)
SEARCH_FALLBACK_SECONDS = 20.0


def search_signature(ctx: dict) -> str:
    """What the search's answers depend on besides the passes: the target groups and the sizes of the rounds. Another signature starts the rounds from the first."""
    keep = {k: v for k, v in ctx.items() if k not in ("rounds", "budget_s", "budget_min", "matrix")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _search_loop(data: dict, ctx: dict, run_pass, stop, clock, budget_s: float, on_pass, t0: float | None = None, max_passes: int = SEARCH_MAX_PASSES) -> dict:
    """The adaptive search: ask the round planner for the next passes, measure them one by one, until it is done, the time is used, the person stops it or there are too many
    passes. Pure apart from the callbacks: `run_pass(key)` returns the pass's result (raising `_Stopped` when the stop file appears), `clock()` is a steady clock in seconds,
    `on_pass(info)` is told about every pass (the caller merges, saves and reports it). A pass that is cached costs nothing and is never asked for.
    Before every pass: elapsed + the time of a pass must fit `budget_s`, else the search stops with "budget" (what was measured is kept). Returns
    {"stop_reason": "done" | "stopped" | "budget" | "max_passes", "rounds": [...], "measured": n}."""
    from . import reshapesearch
    t0 = clock() if t0 is None else t0
    sig = search_signature(ctx)
    saved = data.get("search") or {}
    rounds = [dict(r) for r in saved.get("rounds") or ()] if saved.get("sig") == sig else []
    measured = 0
    reason = "done"
    while True:
        r = reshapesearch.next_round(data["passes"], dict(ctx, rounds=rounds))
        if r is None:
            break
        rec = next((x for x in rounds if x["name"] == r["name"]), None)
        if rec is None:
            rec = {"name": r["name"], "keys": r["all_keys"], "done": False}
            rounds.append(rec)
        interrupted = False
        for key in r["keys"]:
            if key in data["passes"]:
                continue
            if stop():
                reason, interrupted = "stopped", True
                break
            if len(data["passes"]) >= max_passes:
                reason, interrupted = "max_passes", True
                break
            spp = data.get("seconds_per_pass") or SEARCH_FALLBACK_SECONDS
            elapsed = clock() - t0
            if elapsed + spp > budget_s:
                reason, interrupted = "budget", True
                break
            try:
                res = run_pass(key)
            except _Stopped:
                reason, interrupted = "stopped", True
                break
            measured += 1
            merge_pass(data, key, res)
            data["search"] = {"sig": sig, "rounds": rounds, "stop_reason": None, "budget_s": budget_s}
            on_pass({"key": key, "result": res, "round": r["name"], "round_label": r["label"], "planned_total": max(r["planned_total"], len(data["passes"])),
                     "budget_left_s": max(0.0, budget_s - (clock() - t0)), "measured": measured, "round_index": [x["name"] for x in rounds].index(r["name"]) + 1})
        if interrupted:
            break
        rec["done"] = True
    data["search"] = {"sig": sig, "rounds": rounds, "stop_reason": reason, "budget_s": budget_s, "elapsed_s": round(clock() - t0, 1)}
    return {"stop_reason": reason, "rounds": rounds, "measured": measured}


def search_summary(data: dict, ctx: dict) -> dict:
    """What `data["search"]` and the analysis say about the search: its rounds, why it ended, the ranking and the finalists of every group."""
    from . import reshapesearch
    s = dict(data.get("search") or {})
    s["ranking"] = reshapesearch.ranking(data["passes"], ctx)
    s["finalists"] = reshapesearch.finalists(data["passes"], ctx)
    s["groups"] = [{"label": g.get("label"), "target_b": g.get("target_b")} for g in ctx.get("groups") or ()]
    # the numbers of the best cuts of every group (the program ranks and explains them): loss and the two halves
    keep = {k for keys in s["ranking"].values() for k in keys[:SEARCH_ROWS_KEPT]} | {k for keys in s["finalists"].values() for k in keys}
    s["rows"] = {k: {"loss": data["passes"][k]["loss"], "loss_h1": data["passes"][k].get("loss_h1"), "loss_h2": data["passes"][k].get("loss_h2")} for k in sorted(keep) if k in data["passes"]}
    s["rounds"] = [{"name": r["name"], "keys": r["keys"], "done": r.get("done", False)} for r in s.get("rounds") or ()]
    return s


def _best_so_far(passes: dict, ctx: dict):
    """(key, loss) of the best measured cut of the first group, or None."""
    from . import reshapesearch
    rk = reshapesearch.ranking(passes, ctx).get("0") or []
    return (rk[0], passes[rk[0]]["loss"]) if rk else None


def run_try(cfg: dict) -> int:
    """job reshape_try (see the top of this file). The result: {"analysis", "passes_total", "passes_done", "stopped", "seconds_per_pass", "load", "tokens", "dir"}; with
    cfg["plan_only"] only the number of passes and the time they will take (no model is loaded, no torch is needed)."""
    from . import reshapejob
    protocol.reserve_stdout()
    t_job0 = time.monotonic()
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
    wants_width = bool(want.get("widths")) or any(c.get("width") for c in want.get("choices") or ()) or _search_wants_width(want.get("search"))
    if wants_width and why:
        raise ReshapeError(why)
    calib = cfg.get("calibration") or {}
    if not (calib.get("train") and Path(calib["train"]).is_file()):
        raise ReshapeError("Trying a cut is measured on your own text. Save a training-data version in Step 2 first.")
    meas = None
    if dims and cfg.get("measure_dir"):
        meas = reshapejob._read_scores(cfg["measure_dir"], {"source": cfg["source"], "calibration": {"version": calib.get("version"), "samples": cfg.get("measure_samples"),
                                                                                                    "max_len": calib.get("max_len"), "reasoning": calib.get("reasoning")}}, n_layers, dims, types,
                                      weights=safetensors_io.weights_manifest(src_dir) if src_dir else None)
    keys = expand_want(want, nb, pf, pl, strategy, (meas["meta"].get("matrix") or None) if meas else None, dims, divisor)
    wants_ms = any(key_options(k).get("s") == "output" for k in keys)
    needs_measurement = wants_width and (meas is None or (wants_ms and "mlp_ms" not in meas))
    if needs_measurement and not plan_only:
        if wants_ms and meas is not None:
            raise ReshapeError("Ranking the neurons by their output needs a measurement that recorded the mean square of the neurons, and the earlier one did not. Measure the model again.")
        raise ReshapeError("Trying a width needs the measurement of the neurons. Measure the model first.")
    matrix = (meas["meta"].get("matrix") or None) if meas else None
    search = want.get("search") if isinstance(want.get("search"), dict) else None
    sctx = None
    if search is not None:
        sctx = _search_context(search, nb, pf, pl, strategy, matrix)
    folder = Path(cfg["out_dir"])
    if cfg.get("force") and (folder / TRY_FILE).is_file():
        (folder / TRY_FILE).unlink()
    key = build_key(repo, weights_fingerprint(src_dir) if src_dir else None, calib, data_hash(calib["train"]), cfg.get("load", "auto"),
                    safetensors_io.weights_manifest(src_dir) if src_dir else None)
    data = read_results(folder, key if src_dir else None, protocol.log) if src_dir else None
    if data is None:
        data = empty_results(key)
    # the search measures the original in its own first round: the plain list leaves it out, so every pass of a search carries its round
    own = (lambda ks: [k for k in ks if k != BASE_KEY]) if sctx is not None and not (want.get("blocks") or want.get("runs") or want.get("widths") or want.get("choices")) else (lambda ks: list(ks))
    todo = own([k for k in keys if k not in data["passes"]])
    mm = meas["meta"] if meas else {}
    known, known_tokens = data["seconds_per_pass"], data["tokens"]
    if known is None and mm.get("seconds_per_pass"):
        known, known_tokens = mm["seconds_per_pass"], mm.get("tokens")
    tokens = data["tokens"] or (mm["tokens"] * calib["samples"] / cfg["measure_samples"] if mm.get("tokens") and cfg.get("measure_samples") and calib.get("samples") else None)
    est = estimate(len(todo), known, tokens, known_tokens, float(cfg.get("params_b") or 10.0), cfg.get("load", "auto"),
                   int(calib.get("samples", 64)) * min(int(calib.get("max_len", 512)), GUESS_TOKENS_PER_SAMPLE))
    plan = {"passes_total": len(keys), "passes_cached": len(keys) - len(todo), "passes_todo": len(todo), "needs_measurement": needs_measurement, **est}
    if sctx is not None:
        from . import reshapesearch
        rounds_plan = reshapesearch.round_plan(data["passes"], sctx)
        load_s = float(sctx.get("load_s") or 0.0)
        plan["search"] = {"rounds": rounds_plan, "planned_total": reshapesearch.planned_total(data["passes"], sctx), "passes_cached": len(data["passes"]), "budget_s": sctx["budget_s"],
                          "load_s": load_s, "affordable": reshapesearch.affordable(rounds_plan, est["seconds_per_pass"], sctx["budget_s"], load_s)}
    if plan_only:
        protocol.result(plan)
        return 0

    def finish(stopped: bool) -> int:
        data["analysis"] = analyse(data["passes"], nb, period, pf, pl, matrix, strategy)
        if sctx is not None:
            data["analysis"]["search"] = search_summary(data, sctx)
        write_results(folder, data)
        protocol.progress("try", 1.0, "")
        out = {"analysis": data["analysis"], "passes_total": len(keys), "passes_done": len(keys) - len([k for k in keys if k not in data["passes"]]),
               "stopped": stopped, "seconds_per_pass": data["seconds_per_pass"], "load": data["load_used"], "tokens": data["tokens"], "dir": str(folder)}
        if sctx is not None:
            out["stop_reason"] = (data.get("search") or {}).get("stop_reason") or ("stopped" if stopped else "done")
        protocol.result(out)
        return 0

    def search_pending() -> bool:
        from . import reshapesearch
        saved = (data.get("search") or {}) if data.get("search", {}).get("sig") == search_signature(sctx) else {}
        return reshapesearch.next_round(data["passes"], dict(sctx, rounds=saved.get("rounds") or [])) is not None

    if not todo and not (sctx is not None and search_pending()):
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
        todo = own(keys)
    stop_file = folder.parent / "STOP"
    stop = stop_file.exists
    from transformers import AutoTokenizer
    protocol.progress("calibrate", 0.0, "Preparing the text")
    with protocol.timed("calibrate"):
        batches = reshapejob._calibration(AutoTokenizer.from_pretrained(str(src_dir)), calib, cfg["source"].get("family", "generic"))
    data["tokens"] = sum(int(b["attention_mask"].sum()) for b in batches)
    data["load_used"] = mode
    protocol.log(f"Trying {len(todo)} cuts on {len(batches)} batches ({data['tokens']:,} tokens), model loaded as {mode}.")
    protocol.progress("calibrate", 1.0, "")
    protocol.progress("try", 0.0, f"Loading the model ({mode})")
    model = reshapejob._load(src_dir, mode)
    speed.report_kernels(model)
    layers = reshapejob._find_layers(model, n_layers)
    mlp = meas["mlp"] if meas else None
    mlp_out = None
    if wants_ms:                    # the scores of the neurons by their output: mean square of the activation x the squared size of the neuron's down_proj column
        mlp_out = reshape_width.output_scores(meas["mlp_ms"], reshapejob._down_norms(cfg.get("measure_dir"), src_dir, n_layers))
    size = dims["intermediate"] if dims else 0
    stopped = False
    t_pass0 = time.monotonic()
    state = {"batches": batches, "model": model, "layers": layers}

    def scores_of(k):
        """the neuron scores the cut of pass key `k` keeps by: the activation scores, or the output scores for a key with "|s=output" """
        return mlp_out if key_options(k).get("s") == "output" else mlp

    def run_pass(k):
        """One pass; a graphics card that runs out of memory is asked again with one text at a time."""
        try:
            try:
                return _one_pass(state["model"], state["layers"], state["batches"], period, k, scores_of(k), size, stop)
            except RuntimeError as exc:
                if not reshapejob._is_oom(exc):
                    raise
                protocol.log("Out of memory; trying again with one text at a time.")
                reshapejob._free()
                state["batches"] = reshapejob._split_single(state["batches"])
                return _one_pass(state["model"], state["layers"], state["batches"], period, k, scores_of(k), size, stop)
        except RuntimeError as exc:
            if reshapejob._is_oom(exc):
                raise ReshapeError("The graphics card ran out of memory while trying the cuts. Choose 4-bit loading (Expert mode) or fewer samples.") from None
            raise

    def on_search_pass(info: dict) -> None:
        res = info["result"]
        protocol.timing("try_pass", res["seconds"], key=info["key"], tokens=data["tokens"])
        data["analysis"] = analyse(data["passes"], nb, period, pf, pl, matrix, strategy)
        data["analysis"]["search"] = search_summary(data, sctx)
        write_results(folder, data)
        best = _best_so_far(data["passes"], sctx)
        done, total = len(data["passes"]), max(info["planned_total"], len(data["passes"]))
        protocol.progress("try", min(done / total, 0.999), f"Round {info['round_index']} ({info['round_label']}): pass {done} of about {total}: {describe_key(info['key'])}",
                          protocol.make_detail("passes", done, total, t_pass0, rate_unit="passes/s", done_at_start=done - info["measured"]))
        protocol.emit("try_pass", key=info["key"], loss=res["loss"], seconds=res["seconds"], done=done, total=total, seconds_per_pass=data["seconds_per_pass"], analysis=data["analysis"],
                      round=info["round"], round_label=info["round_label"], round_index=info["round_index"], planned_total=total, budget_left_s=round(info["budget_left_s"], 1),
                      best={"key": best[0], "loss": best[1]} if best else None)

    try:
        for i, k in enumerate(todo):
            if stop():
                stopped = True
                break
            protocol.progress("try", i / len(todo), f"Cut {i + 1} of {len(todo)}: {describe_key(k)}", protocol.make_detail("passes", i, len(todo), t_pass0, rate_unit="passes/s"))
            try:
                res = run_pass(k)
            except _Stopped:
                stopped = True
                break
            protocol.timing("try_pass", res["seconds"], key=k, tokens=data["tokens"])
            merge_pass(data, k, res)
            data["analysis"] = analyse(data["passes"], nb, period, pf, pl, matrix, strategy)
            write_results(folder, data)
            protocol.emit("try_pass", key=k, loss=res["loss"], seconds=res["seconds"], done=i + 1, total=len(todo), seconds_per_pass=data["seconds_per_pass"],
                          analysis=data["analysis"])
        if sctx is not None and not stopped:
            outcome = _search_loop(data, sctx, run_pass, stop, time.monotonic, sctx["budget_s"], on_search_pass, t0=t_job0, max_passes=sctx["max_passes"])
            stopped = outcome["stop_reason"] == "stopped"
            protocol.log(f"The search ended: {outcome['stop_reason']} after {outcome['measured']} new cuts.")
    finally:
        state.clear()
        del layers
        del model
        reshapejob._free()
    return finish(stopped)
