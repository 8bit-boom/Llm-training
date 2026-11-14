"""job heal_check: what the Heal section needs to know about a model it is asked to heal (or to use as the teacher), from the files alone.

Nothing is loaded: config.json, the tokenizer files, the .safetensors headers (and, for a Hugging Face id, the same files over the network; the weights
are not downloaded) and the model card are read. Standard library only (huggingface_hub when an id is given).

    {"job": "heal_check", "sources": {"student": "<folder or hub id>", "teacher": "<folder or hub id>"}, "fetch": false}
    -> {"student": facts, "teacher": facts}

`fetch: true` also downloads the weights of a hub id into the Hugging Face cache (the same call healing uses) and adds "folder" to the facts.
Facts never raise: a model that cannot be read has "error" with a plain sentence. The comparison of the two is done by the program (Rust).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import protocol, reshape, safetensors_io
from .distilljob import tokenizer_hash
from .reshape import ReshapeError

TOKENIZER_NAMES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json")
HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def is_folder(source: str) -> bool:
    return Path(source).is_dir()


def vocabulary_size(folder: Path):
    """How many tokens the tokenizer has (its own vocabulary plus the added tokens), or None when the files do not say."""
    f = folder / "tokenizer.json"
    if f.is_file():
        try:
            tj = json.loads(f.read_text(encoding="utf-8"))
            vocab = (tj.get("model") or {}).get("vocab")
            n = len(vocab) if isinstance(vocab, (dict, list)) else None
            if n is not None:
                base = set(vocab) if isinstance(vocab, dict) else set()
                extra = [t for t in tj.get("added_tokens") or [] if isinstance(t, dict) and t.get("content") not in base]
                return n + len(extra)
        except (OSError, ValueError):
            pass
    f = folder / "vocab.json"
    if f.is_file():
        try:
            n = len(json.loads(f.read_text(encoding="utf-8")))
            extra = folder / "added_tokens.json"
            if extra.is_file():
                n += len(json.loads(extra.read_text(encoding="utf-8")))
            return n
        except (OSError, ValueError):
            pass
    return None


def card_fields(text: str) -> dict:
    """license and base_model from a model card's front matter (a list of lines is enough: no YAML library)."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---", text or "", re.S)
    out: dict = {"license": None, "base_model": []}
    if not m:
        return out
    lines = m.group(1).splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^license\s*:", line):
            out["license"] = line.split(":", 1)[1].strip().strip("'\"") or None
        elif re.match(r"^base_model\s*:", line):
            rest = line.split(":", 1)[1].strip()
            if rest and not rest.startswith("["):
                out["base_model"].append(rest.strip("'\""))
            elif rest.startswith("["):
                out["base_model"] += [x.strip().strip("'\"") for x in rest.strip("[]").split(",") if x.strip()]
            else:
                for nxt in lines[i + 1:]:
                    mm = re.match(r"^\s*-\s*(.+)$", nxt)
                    if not mm:
                        break
                    out["base_model"].append(mm.group(1).strip().strip("'\""))
    out["base_model"] = [b for b in out["base_model"] if HUB_ID.match(b)]
    return out


def guess_base(cfg: dict, card: dict, own: str):
    """A guess at the model this one was made from: the card's `base_model`, else config.json's `_name_or_path` (when it is a hub id, not this folder)."""
    for b in card.get("base_model") or []:
        if b.lower() != own.lower():
            return {"repo": b, "from": "base_model in the model card"}
    n = cfg.get("_name_or_path")
    if isinstance(n, str) and HUB_ID.match(n) and n.lower() != own.lower() and not Path(n).exists():
        return {"repo": n, "from": "_name_or_path in config.json"}
    return None


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _facts_from_config(f: dict, cfg: dict) -> None:
    tc = reshape.text_config(cfg)
    f["model_type"] = reshape.model_type(cfg)
    f["layers"] = tc.get("num_hidden_layers") if isinstance(tc.get("num_hidden_layers"), int) else None
    f["hidden"] = tc.get("hidden_size") if isinstance(tc.get("hidden_size"), int) else None
    f["intermediate"] = tc.get("intermediate_size") if isinstance(tc.get("intermediate_size"), int) else None
    f["vocab_size"] = tc.get("vocab_size") if isinstance(tc.get("vocab_size"), int) else None
    f["has_vision"] = cfg.get("vision_config") is not None
    f["quantized"] = bool(cfg.get("quantization_config") or tc.get("quantization_config"))
    f["family"] = "qwen3_5" if f["model_type"].startswith("qwen3_5") else (f["model_type"] or "generic")
    try:
        reshape.check_supported(cfg)
    except ReshapeError as exc:
        f["supported"], f["why"] = False, str(exc)


def facts_local(folder: Path, source: str) -> dict:
    f: dict = {"source": source, "kind": "folder", "on_disk": True, "folder": str(folder), "supported": True, "why": None, "error": None, "quantized": False, "gguf": False,
               "params": None, "bytes": None, "tokenizer_hash": None, "vocab_tokens": None, "license": None, "base_guess": None, "download_bytes": 0}
    if not (folder / "config.json").is_file():
        f["error"] = f"There is no config.json in {source}. Choose the model's own folder (the one that holds config.json and the .safetensors files)."
        return f
    try:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        f["error"] = f"The config.json in {source} could not be read."
        return f
    _facts_from_config(f, cfg)
    f["gguf"] = any(folder.glob("*.gguf")) and not any(folder.glob("*.safetensors"))
    if f["gguf"]:
        f["supported"], f["why"] = False, "This is a GGUF file. Healing needs the full-precision .safetensors model."
    try:
        refs = safetensors_io.list_tensors(folder)
        f["params"] = sum(_numel(r.shape) for r in refs.values())
        f["bytes"] = sum(r.nbytes for r in refs.values())
    except Exception:                                  # noqa: BLE001 - reported below
        if not f["gguf"]:
            f["error"] = f"There are no readable .safetensors files in {source}."
    f["tokenizer_hash"] = tokenizer_hash(folder) if any((folder / n).is_file() for n in TOKENIZER_NAMES) else None
    f["vocab_tokens"] = vocabulary_size(folder)
    readme = folder / "README.md"
    card = card_fields(readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else "")
    f["license"] = card["license"]
    f["base_guess"] = guess_base(cfg, card, source)
    return f


def facts_hub(repo: str) -> dict:
    f: dict = {"source": repo, "kind": "hub", "on_disk": False, "folder": None, "supported": True, "why": None, "error": None, "quantized": False, "gguf": False,
               "params": None, "bytes": None, "tokenizer_hash": None, "vocab_tokens": None, "license": None, "base_guess": None, "download_bytes": 0}
    if not HUB_ID.match(repo):
        f["error"] = f"{repo} is not a folder on this computer and not a Hugging Face name (owner/model)."
        return f
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        f["error"] = "The Hugging Face library is not installed in the training software."
        return f
    try:
        cfg_path = Path(hf_hub_download(repo, "config.json"))
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:                           # noqa: BLE001
        f["error"] = f"{repo} could not be read on Hugging Face ({type(exc).__name__}). Check the spelling and the internet connection; a private or gated model needs a login."
        return f
    _facts_from_config(f, cfg)
    folder = cfg_path.parent
    for name in TOKENIZER_NAMES + ("README.md",):
        try:
            hf_hub_download(repo, name)
        except EntryNotFoundError:
            pass
        except Exception:                              # noqa: BLE001 - a missing optional file is not an error
            pass
    f["tokenizer_hash"] = tokenizer_hash(folder) if any((folder / n).is_file() for n in TOKENIZER_NAMES) else None
    f["vocab_tokens"] = vocabulary_size(folder)
    readme = folder / "README.md"
    card = card_fields(readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else "")
    f["license"] = card["license"]
    f["base_guess"] = guess_base(cfg, card, repo)
    try:
        tensors = safetensors_io.tensors_from_hub_metadata(HfApi().get_safetensors_metadata(repo))
        f["params"] = sum(_numel(s) for _, s in tensors.values())
        f["bytes"] = sum(_numel(s) * safetensors_io.DTYPE_BYTES.get(d, 2) for d, s in tensors.values())
    except Exception:                                  # noqa: BLE001 - no safetensors on the Hub (a GGUF-only repo) or no network
        try:
            names = [s.rfilename for s in HfApi().model_info(repo).siblings or []]
        except Exception:                              # noqa: BLE001
            names = []
        if any(n.endswith(".gguf") for n in names) and not any(n.endswith(".safetensors") for n in names):
            f["gguf"] = True
            f["supported"], f["why"] = False, "This is a GGUF model. Healing needs the full-precision .safetensors model."
        else:
            f["error"] = f.get("error") or f"The size of {repo} could not be read from Hugging Face."
    try:                                               # already downloaded? (the weights, not only the small files)
        snapshot_download(repo, allow_patterns=["*.safetensors"], local_files_only=True)
        f["on_disk"] = True
    except Exception:                                  # noqa: BLE001
        f["on_disk"] = False
    f["download_bytes"] = 0 if f["on_disk"] else int(f["bytes"] or 0)
    try:
        info = HfApi().model_info(repo)
        lic = (getattr(info, "card_data", None) or {}).get("license") if getattr(info, "card_data", None) else None
        if lic:
            f["license"] = str(lic)
    except Exception:                                  # noqa: BLE001 - the licence is a courtesy; never a failure
        pass
    return f


def facts_of(source: str, fetch: bool = False) -> dict:
    source = str(source).strip().strip("\"'")
    f = facts_local(Path(source), source) if is_folder(source) else facts_hub(source)
    if fetch and f["kind"] == "hub" and not f["error"]:
        from . import reshapejob
        f["folder"] = str(reshapejob._resolve_source(source))
        f["on_disk"] = True
    return f


def run_check(job: dict) -> int:
    protocol.reserve_stdout()
    out: dict = {}
    srcs = job.get("sources") or {}
    names = [k for k in ("student", "teacher") if srcs.get(k)]
    for i, k in enumerate(names):
        protocol.progress("check", i / max(1, len(names)), f"Reading the {k} model")
        out[k] = facts_of(srcs[k], bool(job.get("fetch")) and k == "student")
    protocol.progress("check", 1.0, "")
    protocol.result(out)
    return 0
