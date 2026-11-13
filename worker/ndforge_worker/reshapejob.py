"""The "Slim or grow a model" jobs.

    reshape_inspect   {"job": "reshape_inspect", "source": {"repo": "<hub id or folder>"}}  -> what the model is made of (no weights are read or downloaded)
    reshape           cut whole blocks of layers out of a model (prune) or copy a span of layers in (grow), writing a new model folder

Layers are moved as raw bytes between safetensors files (safetensors_io), so neither job needs a GPU or even torch, except for measuring which
layers to cut (prune without an explicit layer list)."""
import datetime
import json
import re
import shutil
import time
from pathlib import Path

from . import protocol, reshape, safetensors_io, vision
from .reshape import ReshapeError

# copied byte for byte from the source folder when present (the weights and the index are written, never copied)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json", "tokenizer.model",
                   "chat_template.jinja", "chat_template.json", "generation_config.json")
SIDE_FILES = tuple(dict.fromkeys(TOKENIZER_FILES + vision.PROCESSOR_FILES))
DOWNLOAD_PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.txt", "*.model", "*.tiktoken", "README*", "LICENSE*"]


# ---------------------------------------------------------------------------------------------------------------- sources

def _hub_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))


def _hub_tensors(repo: str) -> dict:
    """{name: (dtype, shape)} from the headers on the Hub (nothing is downloaded)."""
    from huggingface_hub import HfApi
    return safetensors_io.tensors_from_hub_metadata(HfApi().get_safetensors_metadata(repo))


def check_period(cfg: dict, types: list) -> int:
    """The pattern period, and for Qwen3.5 a check that it is the model's own full_attention_interval (so the interval rule stays true after a cut)."""
    period = reshape.pattern_period(types)
    interval = reshape.text_config(cfg).get("full_attention_interval")
    if reshape.model_type(cfg).startswith("qwen3_5") and isinstance(interval, int) and interval > 0 and period != interval:
        raise ReshapeError(f"The layer types of this model do not follow its attention interval ({interval}); it cannot be reshaped safely.")
    return period


# ------------------------------------------------------------------------------------------------------------------ inspect

def _empty(mt: str, why: str, layers: int = 0, local: bool = False) -> dict:
    return {"model_type": mt, "supported": False, "why": why, "layers": layers, "layer_types": [], "period": 1, "blocks": 0, "layer_params": [],
            "embed_params": 0, "lm_head_params": 0, "vision_params": 0, "mtp_params": 0, "other_params": 0, "total_params": 0, "bytes_per_param": 2.0,
            "has_vision": False, "tied": False, "source_bytes": 0, "local": local}


def inspect_model(repo: str) -> dict:
    local = Path(repo).is_dir()
    if local:
        f = Path(repo) / "config.json"
        if not f.is_file():
            raise ReshapeError(f"There is no config.json in {repo}.")
        cfg = json.loads(f.read_text(encoding="utf-8"))
    else:
        cfg = _hub_config(repo)
    mt = reshape.model_type(cfg)
    tc = reshape.text_config(cfg)
    n = tc.get("num_hidden_layers")
    n = n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0
    try:
        reshape.check_supported(cfg)
        if local:
            refs = safetensors_io.list_tensors(repo)
            tensors = {k: (r.dtype, r.shape) for k, r in refs.items()}
            source_bytes = sum(r.nbytes for r in refs.values())
        else:
            tensors = _hub_tensors(repo)
            source_bytes = sum(reshape_numel(s) * safetensors_io.DTYPE_BYTES.get(d, 2) for d, s in tensors.values())
        types = reshape.layer_types_of(tc)
        period = check_period(cfg, types)
        groups = reshape.group_params(tensors, n)
    except ReshapeError as exc:
        return _empty(mt, str(exc), n, local)
    return {"model_type": mt, "supported": True, "why": None, "layers": n, "layer_types": types, "period": period, "blocks": n // period,
            "layer_params": groups["layers"], "embed_params": groups["embed"], "lm_head_params": groups["lm_head"], "vision_params": groups["vision"],
            "mtp_params": groups["mtp"], "other_params": groups["other"], "total_params": groups["total"], "bytes_per_param": groups["bytes_per_param"],
            "has_vision": groups["vision"] > 0 or cfg.get("vision_config") is not None, "tied": groups["lm_head"] == 0,
            "source_bytes": source_bytes, "local": local}


def reshape_numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def run_inspect(cfg: dict) -> int:
    protocol.reserve_stdout()
    protocol.progress("inspect", 0.0, "Reading the model's layout")
    out = inspect_model(cfg["source"]["repo"])
    protocol.progress("inspect", 1.0, "")
    protocol.result(out)
    return 0


# ------------------------------------------------------------------------------------------------------------------- the job

def _resolve_source(repo: str) -> Path:
    """A local folder as it is, a Hub id downloaded (weights, configs, tokenizer; no GGUF or pictures) into the Hugging Face cache."""
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download
    protocol.progress("download", 0.0, f"Downloading {repo} (this can take a while)")
    path = Path(snapshot_download(repo, allow_patterns=DOWNLOAD_PATTERNS))
    protocol.progress("download", 1.0, "")
    return path


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ReshapeError("Measuring the layers needs the training software (Training setup).") from None


def _score(src_dir: Path, cfg: dict, period: int):
    raise ReshapeError("Measuring the layers is not available in this build.")


def _licence_front_matter(readme: Path) -> str:
    """The licence lines of a model card's front matter (license, license_name, license_link), or ''."""
    if not readme.is_file():
        return ""
    m = re.match(r"^---\r?\n(.*?)\r?\n---", readme.read_text(encoding="utf-8", errors="replace"), re.S)
    if not m:
        return ""
    return "".join(line + "\n" for line in m.group(1).splitlines() if re.match(r"^license(_name|_link)?\s*:", line))


def _copy_side_files(src_dir: Path, out: Path) -> list:
    copied = []
    names = [f for f in SIDE_FILES if (src_dir / f).is_file()] + sorted(p.name for p in src_dir.glob("LICENSE*") if p.is_file())
    for name in dict.fromkeys(names):
        shutil.copy2(src_dir / name, out / name)
        copied.append(name)
    return copied


def run(cfg: dict) -> int:
    protocol.reserve_stdout()
    started = time.monotonic()
    mode = cfg["mode"]
    if mode not in ("prune", "grow"):
        raise ReshapeError(f"Unknown mode {mode!r}.")
    out = Path(cfg["out"])
    final = Path(cfg["final"]) if cfg.get("final") else out.with_name("model")
    stop_file = out.parent / "STOP"
    if final.exists():
        raise ReshapeError(f"{final} already exists.")
    notes: list = []

    src_dir = _resolve_source(cfg["source"]["repo"])
    protocol.progress("inspect", 0.0, "Reading the model's layout")
    config = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    reshape.check_supported(config)
    tc = reshape.text_config(config)
    n_layers = int(tc["num_hidden_layers"])
    tensors = safetensors_io.list_tensors(src_dir)
    groups = reshape.group_params({k: (r.dtype, r.shape) for k, r in tensors.items()}, n_layers)
    types = reshape.layer_types_of(tc)
    period = check_period(config, types)
    protect = cfg.get("protect") or {}
    pf, pl = int(protect.get("first", 1)), int(protect.get("last", 1))
    nb = reshape.block_count(n_layers, period)
    protocol.progress("inspect", 1.0, f"{n_layers} layers in blocks of {period}")

    scores, loss_before, loss_after, stable, strategy, chosen = None, None, None, None, cfg.get("strategy", "contiguous"), []
    calibration = cfg.get("calibration") or {}
    zero = None
    if mode == "prune":
        if cfg.get("drop_layers"):
            chosen = reshape.check_whole_blocks([int(x) for x in cfg["drop_layers"]], period, n_layers, pf, pl)
            strategy = "explicit"
        else:
            lo, hi = reshape.limits(nb, "prune", pf, pl)
            blocks = int(cfg["blocks"])
            if not lo <= blocks <= hi:
                raise ReshapeError("Slim at most half of the layers, and never the first or last block.")
            _require_torch()
            D, D1, D2, loss_before = _score(src_dir, cfg, period)
            chosen = reshape.choose_drop(D, blocks, strategy, pf, pl)
            stable = reshape.halves_agree(D1, D2, blocks, strategy, pf, pl)
            scores = {"boundary_distance": [D[k][k + 1] for k in range(nb)], "matrix": D, "chosen_blocks": chosen, "period": period, "stable": stable}
            protocol.emit("reshape_scores", **scores)
        drop = reshape.blocks_to_layers(chosen, period)
        src = reshape.prune_src(n_layers, drop)
        protocol.log(f"Removing layers {drop[0]} to {drop[-1]} ({len(chosen)} blocks of {period})." if len(chosen) == 1 or drop == list(range(drop[0], drop[-1] + 1))
                     else f"Removing blocks {chosen} (layers {drop}).")
    else:
        grow = cfg.get("grow") or {}
        src, zero_all = reshape.grow_src(n_layers, period, int(cfg["blocks"]), grow.get("layout", "solar"))
        init = grow.get("init", "identity")
        if init not in ("identity", "copy"):
            raise ReshapeError(f"Unknown start {init!r}.")
        zero = zero_all if init == "identity" else None
        if zero is not None:
            notes.append("The new layers start switched off (their output weights are zero), so the grown model first answers exactly like the original.")
    reshape.check_pattern(types, src, period)
    notes_cfg: list = []
    new_cfg = reshape.fix_config(config, src, notes_cfg)
    notes += notes_cfg
    plan = reshape.plan_tensors(tensors, src, zero)
    need = sum(t.nbytes for t in plan)
    free = shutil.disk_usage(out.parent)[2]
    if free < need * 1.1:
        protocol.error("disk_full", f"There is not enough free space: {need * 1.1 / 1e9:.1f} GB needed in {out.parent}, {free / 1e9:.1f} GB free.")
        return 1

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    last = [-1.0]

    def on_progress(done, total):
        if total and (done / total - last[0] >= 0.01 or done == total):
            last[0] = done / total
            protocol.progress("write", done / total, f"Writing the new model: {done / 1e9:.1f} of {total / 1e9:.1f} GB")

    try:
        safetensors_io.write_shards(plan, out, shard_bytes=int(float(cfg.get("shard_gb", 5)) * 2**30), on_progress=on_progress,
                                    should_stop=lambda: stop_file.exists())
        (out / "config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        _copy_side_files(src_dir, out)
        zero_init = [j for j, z in enumerate(zero) if z] if zero else []
        info = reshape.describe(groups, n_layers, src, types)
        meta = {"format": 1, "tool": "ND Lore Forge", "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "origin": {"repo": cfg["source"]["repo"], "family": cfg["source"].get("family"), "layers": n_layers, "params": groups["total"]},
                "mode": mode, "period": period, "layers": len(src), "params": info["params_after"], "src_layers": src, "dropped": info["dropped"],
                "duplicated": info["duplicated"], "zero_init": zero_init, "strategy": strategy if mode == "prune" else cfg.get("grow", {}).get("layout", "solar"),
                "calibration": ({"version": calibration.get("version"), "samples": calibration.get("samples"), "max_len": calibration.get("max_len"),
                                 "tokens": (scores or {}).get("tokens")} if scores else None),
                "loss_before": loss_before, "loss_after": loss_after, "stable": stable,
                "vision": "unchanged (copied from the original)" if groups["vision"] else "none", "mtp": "copied unchanged" if groups["mtp"] else "none"}
        (out / "ndforge_reshape.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        label = f"{'Slimmed' if mode == 'prune' else 'Grown'} from {cfg['source']['repo']}: {n_layers} to {len(src)} layers."
        (out / "README.md").write_text(f"---\n{_licence_front_matter(src_dir / 'README.md')}---\n# {out.parent.name}\n\n{label}\n\n"
                                        "Made with ND Lore Forge (Slim or grow a model). See ndforge_reshape.json for exactly which layers were used.\n", encoding="utf-8")
        out.rename(final)
    except safetensors_io.Cancelled:
        shutil.rmtree(out, ignore_errors=True)
        protocol.result({"stopped": True, "model_dir": None})
        return 0
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    protocol.progress("done", 1.0, "")
    protocol.result({"model_dir": str(final), "mode": mode, "layers_before": n_layers, "layers_after": len(src), "src_layers": src, "dropped": info["dropped"],
                     "duplicated": info["duplicated"], "zero_init": zero_init, "params_before": groups["total"], "params_after": info["params_after"],
                     "bytes_written": need, "loss_before": loss_before, "loss_after": loss_after, "ppl_before": None, "ppl_after": None, "stable": stable,
                     "boundary_distance": (scores or {}).get("boundary_distance"), "chosen_blocks": chosen if mode == "prune" else None, "load": cfg.get("load"),
                     "seconds": round(time.monotonic() - started, 1), "notes": notes, "stopped": False})
    return 0
