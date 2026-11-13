"""The "Slim or grow a model" jobs.

    reshape_inspect   {"job": "reshape_inspect", "source": {"repo": "<hub id or folder>"}}  -> what the model is made of (no weights are read or downloaded)
    reshape           cut whole blocks of layers out of a model (prune) or copy a span of layers in (grow), writing a new model folder

Layers are moved as raw bytes between safetensors files (safetensors_io), so neither job needs a GPU or even torch, except for measuring which
layers to cut (prune without an explicit layer list)."""
import json
from pathlib import Path

from . import protocol, reshape, safetensors_io
from .reshape import ReshapeError


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
