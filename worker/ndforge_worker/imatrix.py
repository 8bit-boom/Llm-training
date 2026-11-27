"""The importance matrix ("imatrix") step of the GGUF export: a few minutes of the model reading typical text show llama.cpp which weights matter most,
and the small sizes (Q6_K and below) are then rounded with more care. llama.cpp's `llama-imatrix` measures it on the 16-bit GGUF; `llama-quantize --imatrix`
uses it. It helps the sizes at about 4 bits and below; Q8_0 and F16/BF16 are left alone.

    calibration text   the project's saved training data as the model sees it (practice conversations through the model's own chat template, world-book
                       texts as they are), about 80 %, plus about 20 % general text (distilljob's general text machinery; offline -> the project's text only)
    imatrix.gguf       the matrix, saved next to the GGUFs with imatrix.json (what it was made from); a later export of the same model, data and settings
                       reuses it
    never fatal        every problem becomes a note and the sizes are made without the matrix; only the Stop button ends the export (Stopped)

Config (job["imatrix"]): {"enabled": true, "model": run name, "data_version": n, "train": train.jsonl path, "chunks": 128, "ctx": 512, "general_share": 0.2,
"reuse": true, "reuse_dirs": [export folders that may hold a saved matrix], "general": {"id" | "file", "column", ...} | None, "stop_file": path,
"nextn": true (also measure the model's speculative-decoding / MTP head; only when the model has one and this llama-imatrix has --nextn)}."""
import hashlib
import os
import queue
import random
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from . import data, ggufinfo, protocol

FORMAT = 1
NO_IMATRIX = ("q8_0", "f16", "bf16")        # high precision: the matrix does (almost) nothing there
CHUNKS = (64, 128, 256)
DEFAULT_CTX = 512
DEFAULT_SHARE = 0.2
SLACK = 1.1                                 # llama.cpp may count a few more tokens than the tokenizer here: a little extra text keeps the chunks full
MAX_GENERAL_DOC = 2                         # one general text counts for at most this many chunks of text: many different texts beat one long one
HEARTBEAT = 5.0


class Stopped(Exception):
    """The Stop button (the STOP file) ended the export."""


def launcher(path) -> list:
    """The command that starts a tool. A real llama.cpp tool is a program; a `.py` file (the tests' stand-ins, which must also run on Windows) runs through this Python."""
    return [sys.executable, str(path)] if str(path).lower().endswith(".py") else [str(path)]


def find_llama_imatrix(quantize):
    """llama-imatrix of the llama.cpp that `quantize` (llama-quantize) belongs to, in the other places llama.cpp is looked for, or on PATH (llamacpp.find_tool).
    None when this llama.cpp has none (an older install)."""
    from . import llamacpp
    return llamacpp.find_tool("llama-imatrix", quantize)


def tool_version(tool) -> str:
    """The version line of a llama.cpp tool ("version: 5000 (abcdef)"), or "" when it cannot be had."""
    try:
        p = subprocess.run(launcher(tool) + ["--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=30)
        for line in p.stdout.splitlines():
            if "version" in line.lower():
                return line.strip()[:200]
    except Exception:       # noqa: BLE001 - the version string is a nicety
        pass
    return ""


# ------------------------------------------------------------------------------------------------------------------ the speculative-decoding (MTP / NextN) head

MTP_CONFIG_KEYS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")
NEXTN_REFUSAL = ("nextn", "kv cache", "mtp")      # words of the text llama-imatrix prints when it cannot measure the head


def mtp_in_config(cfg) -> bool:
    """Does a model's config.json (a dict; the keys may sit in `text_config`) announce MTP / NextN layers?"""
    if not isinstance(cfg, dict):
        return False
    for d in (cfg, cfg.get("text_config")):
        if isinstance(d, dict):
            for k in MTP_CONFIG_KEYS:
                v = d.get(k)
                if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                    return True
    return False


def mtp_in_tensor_names(names) -> bool:
    """Is any tensor of the MTP head among these names (`mtp.layers.0...`, `model.mtp...`)?"""
    return any(n.startswith("mtp.") or n.startswith("model.mtp.") for n in names)


def folder_has_mtp(folder):
    """True / False / None (cannot tell) for a model folder: the config keys, else the tensor names of its safetensors files (the index when there is one)."""
    import json
    from . import safetensors_io
    folder = Path(folder)
    if not folder.is_dir():
        return None
    try:
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cfg = None
    if mtp_in_config(cfg):
        return True
    try:
        index = folder / safetensors_io.INDEX_NAME
        if index.is_file():
            names = list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
        else:
            files = sorted(folder.glob("*.safetensors"))
            if not files:
                return False if cfg is not None else None
            names = [n for f in files for n in safetensors_io.read_header(f)[0] if n != "__metadata__"]
    except (OSError, ValueError, KeyError, TypeError):
        return None if cfg is None else False
    return mtp_in_tensor_names(names)


_NO_MTP_SUPPORT: dict = {}
NO_MTP_ASSERT = "opt_num_mtp_layers"


def converter_has_no_mtp(conv) -> bool:
    """Does this convert_hf_to_gguf.py know `--no-mtp` (alias `--no-nextn`)? Read from the script text once per script and remembered."""
    key = str(conv)
    if key not in _NO_MTP_SUPPORT:
        try:
            text = Path(conv).read_text(encoding="utf-8", errors="replace")
            _NO_MTP_SUPPORT[key] = "--no-mtp" in text or "--no-nextn" in text
        except OSError:
            _NO_MTP_SUPPORT[key] = False
    return _NO_MTP_SUPPORT[key]


def is_inkling_folder(folder) -> bool:
    import json
    try:
        mt = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError, AttributeError):
        return False
    return mt == "inkling2_loop"


def no_mtp_extra(conv, folder) -> list:
    """["--no-mtp"] when the model in FOLDER has no MTP head (config keys 0 / absent and no `mtp.` tensors) and the converter has the flag, else []. A newer converter asserts
    when a model has no head and the flag is missing; a model that has one is converted exactly as before. Inkling-2-Mini never gets it: its drafter is a file of its own that the
    converter skips by itself, and `mtp_config` in its config.json is not a Qwen-style head."""
    if is_inkling_folder(folder):
        return []
    if folder_has_mtp(folder) is False and converter_has_no_mtp(conv):
        return ["--no-mtp"]
    return []


def retry_wants_no_mtp(conv, cmd, tail) -> bool:
    """After a failed conversion: was it the converter's MTP assertion, and may it be run again with `--no-mtp`?"""
    return (NO_MTP_ASSERT in "\n".join(tail or []) and "--no-mtp" not in cmd and "--no-nextn" not in cmd and converter_has_no_mtp(conv))


def tail_note(tail, n: int = 25) -> str:
    """The last N lines of a tool's output, scrubbed and with the user name redacted, as one text for an error message."""
    lines = [protocol.redact_user(protocol.scrub(str(l))) for l in (tail or []) if str(l).strip()]
    return "\n".join(lines[-n:])


_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_header(path):
    """(metadata, tensor names) of a GGUF file: `metadata` maps each key to its value (numbers, booleans and strings; None for an array), the names are the tensors'.
    Only the header is read. Raises ValueError when the file is not a readable GGUF (not one, version below 2, truncated, absurd sizes), OSError when it cannot be opened."""
    try:
        with open(path, "rb") as fh:
            def take(fmt):
                size = struct.calcsize(fmt)
                raw = fh.read(size)
                if len(raw) != size:
                    raise ValueError("short")
                return struct.unpack("<" + fmt, raw)[0]

            def string(keep=False):
                n = take("Q")
                if n > 1 << 24:
                    raise ValueError("long string")
                if keep:
                    return fh.read(n).decode("utf-8", "replace")
                fh.seek(n, 1)

            def value(kind, keep=False):
                if kind in _GGUF_FIXED:
                    return take({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])
                if kind == 8:
                    return string(keep)
                if kind == 9:
                    inner, n = take("I"), take("Q")
                    if inner in _GGUF_FIXED:
                        fh.seek(_GGUF_FIXED[inner] * n, 1)
                    else:
                        for _ in range(n):
                            value(inner)
                    return None
                raise ValueError("unknown value type")

            if fh.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            if take("I") < 2:
                raise ValueError("an old GGUF version")
            tensors, kvs = take("Q"), take("Q")
            if tensors > 1 << 22 or kvs > 1 << 22:
                raise ValueError("absurd counts")
            meta = {}
            for _ in range(kvs):
                key = string(keep=True)
                meta[key] = value(take("I"), keep=True)
            names = []
            for _ in range(tensors):
                names.append(string(keep=True))
                for _ in range(take("I")):
                    take("Q")
                take("I")
                take("Q")
            return meta, names
    except struct.error as exc:
        raise ValueError(str(exc))


def gguf_has_mtp(path):
    """True / False / None for a GGUF file: a key ending `nextn_predict_layers` with a value above 0, or a tensor named `...nextn...` / `mtp...`. None when the file is not
    a readable GGUF (the tests' stand-in files, a truncated file). Only the header is read."""
    try:
        meta, names = read_gguf_header(path)
    except (OSError, ValueError):
        return None
    if any(k.endswith("nextn_predict_layers") and isinstance(v, int) and not isinstance(v, bool) and v > 0 for k, v in meta.items()):
        return True
    return any("nextn" in n or n.startswith("mtp.") or ".mtp." in n for n in names)


# ------------------------------------------------------------------------------------------------------------------ does a matrix fit a model? (a light check, never a refusal)

def _counts_tokens(path, info) -> float | None:
    """The number of rows (tokens) a matrix was measured on, from its `.counts` tensors: the largest value among the first few (float32 only). llama.cpp's imatrix writes
    one `.counts` value per weight tensor (per expert for a mixture-of-experts model); the published matrix of a 27B model has the same 1149614.0 everywhere. None when
    it cannot be read. This, not `imatrix.chunk_count`, is what a merge adds up (docs/IMATRIX.md, "Merging matrices")."""
    best = None
    try:
        with open(path, "rb") as fh:
            for t in [t for t in info["tensors"] if t["name"].endswith(".counts") and t["type"] == 0][:8]:
                n = ggufinfo.numel(t)
                if not 0 < n <= 4096:
                    continue
                fh.seek(info["data_start"] + t["offset"])
                raw = fh.read(4 * n)
                if len(raw) != 4 * n:
                    return None
                top = max(struct.unpack("<%df" % n, raw))
                best = top if best is None else max(best, top)
    except (OSError, struct.error):
        return None
    return best


def _kind_of(name: str):
    """(layer number, kind, suffix) of `blk.12.ffn_down.weight.in_sum2` -> (12, "ffn_down", "in_sum2"); None for a name that is not a per-layer statistic."""
    m = re.match(r"blk\.(\d+)\.(.+)\.weight\.(in_sum2|counts)$", name)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def matrix_summary(path):
    """What a GGUF-format matrix says about itself, from its header (the real file layout is in docs/IMATRIX.md, "Verified on the real published file"):
    {"arch": str | None (the real file has no `general.architecture`), "layers": int | None (the highest `blk.N.` plus one), "entries": number of tensors measured,
    "chunks": int | None, "chunk_size": int | None, "datasets": [str], "kinds": {tensor kind: how many layers have it} (`ffn_down`, `attn_qkv`, `ssm_out`, ...),
    "hidden": int | None (the input width of `ffn_gate`: the hidden size of the model), "intermediate": int | None (the input width of `ffn_down`), "tokens": float | None
    (the rows the statistics were summed over, the `.counts` value)}. None when the file cannot be read as a GGUF (a legacy .dat, a stand-in, a truncated file)."""
    try:
        info = ggufinfo.read(path)
    except (OSError, ValueError):
        return None
    meta, names = info["meta"], [t["name"] for t in info["tensors"]]
    blocks = [int(m.group(1)) for n in names for m in [re.match(r"blk\.(\d+)\.", n)] if m]
    sums = [n for n in names if n.endswith(".in_sum2")]
    arch = meta.get("general.architecture")
    chunks, size = meta.get("imatrix.chunk_count"), meta.get("imatrix.chunk_size")
    kinds: dict = {}
    widths: dict = {}
    for t in info["tensors"]:
        k = _kind_of(t["name"])
        if k and k[2] == "in_sum2":
            kinds[k[1]] = kinds.get(k[1], 0) + 1
            if len(t["dims"]) == 1:
                widths.setdefault(k[1], t["dims"][0])
    datasets = meta.get("imatrix.datasets#items")
    return {"arch": arch if isinstance(arch, str) else None, "layers": max(blocks) + 1 if blocks else None, "entries": len(sums) or len(names) // 2,
            "chunks": chunks if isinstance(chunks, int) else None, "chunk_size": size if isinstance(size, int) else None,
            "datasets": [d for d in datasets if isinstance(d, str)] if isinstance(datasets, list) else [], "kinds": dict(sorted(kinds.items())),
            "hidden": widths.get("ffn_gate"), "intermediate": widths.get("ffn_down"), "tokens": _counts_tokens(path, info)}


def gguf_model_facts(path):
    """(architecture, block count) of a model GGUF, each None when not readable."""
    try:
        meta, _ = read_gguf_header(path)
    except (OSError, ValueError):
        return None, None
    arch = meta.get("general.architecture")
    arch = arch if isinstance(arch, str) else None
    n = meta.get(f"{arch}.block_count") if arch else None
    return arch, n if isinstance(n, int) and not isinstance(n, bool) else None


def folder_model_facts(folder):
    """(model type, layer count, MTP layer count) from a model folder's config.json (also inside `text_config`); None where it is not readable (the MTP count is 0 then)."""
    import json
    try:
        cfg = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None, 0
    typ, layers, mtp = None, None, 0
    for d in (cfg, cfg.get("text_config") if isinstance(cfg, dict) else None):
        if isinstance(d, dict):
            typ = typ or (d.get("model_type") if isinstance(d.get("model_type"), str) else None)
            n = d.get("num_hidden_layers")
            layers = layers if layers is not None else (n if isinstance(n, int) and not isinstance(n, bool) else None)
            for k in MTP_CONFIG_KEYS:
                v = d.get(k)
                if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                    mtp = max(mtp, v)
    return typ, layers, mtp


def folder_model_sizes(folder):
    """(hidden size, MLP intermediate size) from a model folder's config.json (also inside `text_config`); None where it is not readable."""
    import json
    try:
        cfg = json.loads((Path(folder) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    hid = inter = None
    for d in (cfg, cfg.get("text_config") if isinstance(cfg, dict) else None):
        if isinstance(d, dict):
            h, i = d.get("hidden_size"), d.get("intermediate_size")
            hid = hid if hid is not None else (h if isinstance(h, int) and not isinstance(h, bool) else None)
            inter = inter if inter is not None else (i if isinstance(i, int) and not isinstance(i, bool) else None)
    return hid, inter


def compat_warnings(matrix, model_arch=None, model_layers=None, mtp_layers: int = 0, model_sizes=(None, None)) -> list:
    """Plain-language warnings about using `matrix` (a file) for a model with this architecture and layer count (None = not known); `model_sizes` is (hidden size, MLP
    intermediate size). [] when it looks fine or nothing can be compared. Never a refusal: llama-quantize itself decides. The real published matrix has no
    `general.architecture`, so the model is also compared by the shapes of the statistics (the input widths of ffn_gate and ffn_down)."""
    info = matrix_summary(matrix)
    if info is None:
        return [f"The matrix file {Path(matrix).name} could not be read here (a legacy .dat file or an unusual file), so it could not be checked against this model. llama-quantize will tell if it does not fit."]
    out = []
    if info["arch"] and model_arch and info["arch"] != model_arch:
        out.append(f"This matrix says it was made for the architecture \"{info['arch']}\", but the model is \"{model_arch}\". It will most likely not fit.")
    if info["layers"] is not None and model_layers is not None and info["layers"] not in (model_layers, model_layers + max(mtp_layers, 0)):
        out.append(f"This matrix has data for {info['layers']} layers, but the model has {model_layers}. It was probably made from a different model; llama-quantize decides whether it can use it.")
    hidden, inter = model_sizes
    if info["hidden"] and hidden and info["hidden"] != hidden:
        out.append(f"This matrix was measured on a model with hidden size {info['hidden']} (the input width of its ffn_gate), but this model's is {hidden}. It was made from a different model; llama-quantize decides whether it can use it.")
    if info["intermediate"] and inter and info["intermediate"] != inter:
        out.append(f"In this matrix the input of ffn_down is {info['intermediate']} wide, in the model {inter}. That is what a model made thinner on the Slim or grow page looks like: "
                   "\"Adapt to my cut model\" (Importance matrices) cuts a matrix to such a model; llama-quantize cannot use this one as it is.")
    return out


def hf_cache_dirs(repo: str) -> list:
    """Snapshot folders of a Hugging Face model id in the local cache (newest first); [] when it is not cached or `repo` is not an id."""
    if not repo or "/" not in repo or Path(repo).exists():
        return []
    roots = [os.environ.get("HF_HUB_CACHE"), os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
             str(Path.home() / ".cache" / "huggingface" / "hub")]
    found = []
    for root in filter(None, roots):
        snaps = Path(root) / ("models--" + repo.replace("/", "--")) / "snapshots"
        if snaps.is_dir():
            found += sorted((d for d in snaps.iterdir() if (d / "config.json").is_file()), key=lambda d: d.stat().st_mtime, reverse=True)
    return found


def model_has_mtp(cfg: dict, merged, f16=None):
    """Does the model being exported have an MTP head? True / False / None (cannot tell). The converted 16-bit GGUF is the best witness (it is what llama-imatrix
    reads); else the merged folder, the full model, the base model (a folder or the Hugging Face cache)."""
    if f16 is not None and Path(f16).is_file():
        seen = gguf_has_mtp(f16)
        if seen is not None:
            return seen
    folders = [merged, cfg.get("full_model"), cfg["base"].get("repo")]
    folders += hf_cache_dirs(cfg["base"].get("repo") or "")
    for f in folders:
        if f and Path(f).is_dir():
            seen = folder_has_mtp(f)
            if seen is not None:
                return seen
    return None


def nextn_supported(tool) -> bool:
    """Does this llama-imatrix know --nextn? Asked once with `--help`."""
    try:
        p = subprocess.run(launcher(tool) + ["--help"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=60)
        return "--nextn" in p.stdout
    except Exception:       # noqa: BLE001 - cannot ask: leave the head out
        return False


def nextn_refused(tail) -> str:
    """The line of llama-imatrix's output that tells why it cannot measure the MTP head (mentions nextn or the KV cache), or "" when the failure is something else."""
    for line in tail:
        if any(w in line.lower() for w in NEXTN_REFUSAL):
            return line.strip()[:200]
    return ""


# ------------------------------------------------------------------------------------------------------------------ the calibration text

def count_tokens(tokenizer, text: str) -> int:
    return len(data.text_tokenizer(tokenizer)(text, add_special_tokens=False)["input_ids"])


def render_rows(tokenizer, rows: list, fam_name: str):
    """([text of every usable row], skipped): a practice conversation through the model's chat template (the same text training saw, prompt + answer + end
    marker, special tokens included), a world-book text as it is."""
    docs, skipped = [], 0
    for r in rows:
        try:
            if r.get("type") == "text":
                text = (r.get("text") or "").strip()
            else:
                text = data.render_chat(tokenizer, r["messages"], fam_name)[1]
        except Exception:       # noqa: BLE001 - one odd row must not stop the export
            skipped += 1
            continue
        if text.strip():
            docs.append(text)
        else:
            skipped += 1
    return docs, skipped


def fetch_general(source: dict, tokenizer, budget: int, ctx: int, should_stop=lambda: False, fam_name: str = None, seed: int = 3407, report: dict = None):
    """(texts, note): general text of about `budget` tokens through the healing code's source reader (a local file, or a Hugging Face dataset streamed; text,
    chat and prompt+completion rows; chat rows go through the model's chat template when fam_name is given). A text is cut at MAX_GENERAL_DOC chunks. Any
    problem (offline, gated, not found, no usable column, nothing read) gives ([], plain-language note) and the matrix is made from the project's text only."""
    from . import distilljob, generaltext
    source = {**source}
    if not (source.get("file") or source.get("id")):
        source["id"] = distilljob.DEFAULT_GENERAL
    name = source.get("file") or source.get("id")
    rep = report if report is not None else {}
    source.update(seed=seed, report=rep)
    if fam_name:
        source["render"] = lambda msgs: data.render_chat(tokenizer, msgs, fam_name)[1]
    try:
        texts, got, cap = [], 0, ctx * MAX_GENERAL_DOC
        for text in distilljob._general_texts(source):
            if should_stop():
                raise Stopped()
            n = count_tokens(tokenizer, text)
            if n < distilljob.MIN_GENERAL_ROW:
                continue
            if n > cap:
                text, n = text[: int(len(text) * cap / n)], cap
            texts.append(text)
            got += n
            if got >= budget:
                break
        if not texts:
            return [], f"The general text ({name}) had nothing to read. The matrix is made from your project's text only."
        return texts, ""
    except Stopped:
        raise
    except Exception as exc:    # noqa: BLE001 - offline, gated, wrong id or column, library missing: all end the same plain way
        note, code = generaltext.failure_note(exc, source, "read", "The matrix is made from your project's text only.", distilljob.DEFAULT_GENERAL)
        rep["error_code"] = code
        return [], note


def reasoning_key(im: dict):
    """What makes the reasoning part of the calibration text another one (None when there is none): `im["reasoning"]` is {"source": {"id" | "file", ...}, "share": 0.5,
    "max_len": 2048}, the same request as the reshape measuring's (reshapejob.reasoning_key; one definition of the key). Part of the key of a saved matrix."""
    from . import reshapejob
    return reshapejob.reasoning_key(im or {})


def fetch_reasoning(key: dict, tokenizer, budget: int, should_stop=lambda: False, fam_name: str = None, seed: int = 3407, report: dict = None):
    """(texts, note): conversations WITH thinking of about `budget` tokens, written the way the model thinks (data.render_chat_full: the template over the whole
    conversation, thinking on, `...<think>\n{trace}\n</think>\n\n{answer}`), each cut to key["max_len"] tokens (the first part only: a later part would have neither the
    question nor the chat frame). Conversations without a trace are passed over for a family that thinks. Any problem gives ([], plain note) and the matrix is made
    without the reasoning text."""
    from . import distilljob, generaltext, reshapejob
    from .families import family
    src = {k: key[k] for k in ("id", "file", "config", "split", "column") if key.get(k)}
    name = src.get("file") or src.get("id") or reshapejob.REASONING_ID
    rep = report if report is not None else {}
    src.update(seed=seed, report=rep)
    thinks = bool(fam_name) and bool(family(fam_name).get("thinks"))
    if fam_name:
        src["render"] = lambda msgs: data.render_chat_full(tokenizer, msgs, fam_name)[1]
    cap = int(key["max_len"])
    try:
        texts, got, scanned = [], 0, 0
        for text in distilljob._general_texts(src):
            if should_stop():
                raise Stopped()
            scanned += 1
            if thinks and "</think>" not in text.replace("<think>\n\n</think>", ""):
                if scanned >= reshapejob.REASONING_SCAN * 64:
                    break
                continue
            n = count_tokens(tokenizer, text)
            if n < distilljob.MIN_GENERAL_ROW:
                continue
            if n > cap:
                text, n = text[: int(len(text) * cap / n)], cap
            texts.append(text)
            got += n
            if got >= budget:
                break
        if not texts:
            return [], f"The reasoning text ({name}) had no conversation with a thinking trace. The matrix is made without it."
        return texts, ""
    except Stopped:
        raise
    except Exception as exc:    # noqa: BLE001 - offline, gated, wrong id: the same plain way as the general text
        note, code = generaltext.failure_note(exc, src, "read", "The matrix is made without the reasoning text.", reshapejob.REASONING_ID)
        rep["error_code"] = code
        return [], note


def _interleave(a: list, b: list):
    """The texts of `a` and `b` (lists of (text, tokens)) mixed evenly by tokens, each in its own order."""
    out, ai, bi, at, bt = [], 0, 0, 0, 0
    ta, tb = sum(n for _, n in a), sum(n for _, n in b)
    while ai < len(a) or bi < len(b):
        if bi >= len(b) or (ai < len(a) and at * max(tb, 1) <= bt * max(ta, 1)):
            out.append(a[ai]); at += a[ai][1]; ai += 1
        else:
            out.append(b[bi]); bt += b[bi][1]; bi += 1
    return out


def build_calibration(tokenizer, rows: list, fam_name: str, ctx: int, chunks: int, general: list, share: float = DEFAULT_SHARE, seed: int = 3407, reasoning: list = None, reasoning_share: float = 0.0):
    """(text, stats). About ctx x chunks tokens: `share` of them general text (when `general` has any), the rest the project's own rows (shuffled with `seed`,
    repeated when the data is smaller than that). Both kinds are mixed evenly through the file, so that --chunks cuts nothing out of one of them.
    `reasoning` (conversations with thinking, `reasoning_share` of the tokens) is mixed in with the general text. The same rows, settings and texts always give the same file."""
    budget = int(ctx * chunks * SLACK) + 1
    docs, skipped = render_rows(tokenizer, rows, fam_name)
    if not docs:
        raise ValueError("The training data has no text to measure with.")
    g_budget = int(budget * share) if general else 0
    gen, g_tok = [], 0
    for text in general:
        if g_tok >= g_budget:
            break
        n = count_tokens(tokenizer, text)
        gen.append((text, n))
        g_tok += n
    rsn, r_tok = [], 0
    r_budget = int(budget * reasoning_share) if reasoning else 0
    for text in (reasoning or []):
        if r_tok >= r_budget:
            break
        n = count_tokens(tokenizer, text)
        rsn.append((text, n))
        r_tok += n
    if rsn:
        gen = _interleave(rsn, gen)             # the reasoning text and the general text are one stream next to the project's own
    g_tok_general, g_tok = g_tok, g_tok + r_tok
    p_budget = max(budget - g_tok, 1)
    random.Random(seed).shuffle(docs)
    sized = [(d, count_tokens(tokenizer, d)) for d in docs]
    unique_tokens = sum(n for _, n in sized)
    proj, p_tok, i = [], 0, 0
    while p_tok < p_budget:
        d, n = sized[i % len(sized)]
        proj.append((d, n))
        p_tok += max(n, 1)
        i += 1
    # a Bresenham-style merge: always take from the kind that is further behind its share
    out, pi, gi, pt, gt = [], 0, 0, 0, 0
    while pi < len(proj) or gi < len(gen):
        if gi >= len(gen) or (pi < len(proj) and pt * max(g_tok, 1) <= gt * max(p_tok, 1)):
            out.append(proj[pi][0])
            pt += proj[pi][1]
            pi += 1
        else:
            out.append(gen[gi][0])
            gt += gen[gi][1]
            gi += 1
    stats = {"project_rows": len(docs), "project_rows_used": len(proj), "project_tokens": p_tok, "unique_project_tokens": unique_tokens, "repeated": i > len(sized),
             "general_rows": len(gen) - len(rsn), "general_tokens": g_tok_general, "reasoning_rows": len(rsn), "reasoning_tokens": r_tok, "skipped_rows": skipped, "tokens": p_tok + g_tok, "wanted_tokens": ctx * chunks}
    return "\n".join(t.strip("\n") for t in out) + "\n", stats


# ------------------------------------------------------------------------------------------------------------------ what a saved matrix was made from

def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_fingerprint(cfg: dict) -> str:
    """Names, sizes and times of the safetensors files of the exported model (the adapter, or the full model): the same run trained further gives another one."""
    folder = cfg.get("full_model") or cfg.get("adapter")
    h = hashlib.sha256()
    if folder and Path(folder).is_dir():
        for f in sorted(Path(folder).glob("*.safetensors")):
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def settings_key(cfg: dict, im: dict, data_hash: str, nextn: bool = False) -> dict:
    """What a saved matrix was made from. "general" is the dataset (id or file, config, split, column, share, seed): another dataset never reuses this matrix
    (a matrix saved before this key existed has no "general" and counts as made from other data). "nextn" is whether the MTP head is part of the measuring:
    exact match, so a matrix made without the head is not used when the head is wanted, nor the reverse (one saved before this key existed counts as without)."""
    from . import generaltext
    share = float(im.get("general_share", DEFAULT_SHARE))
    return {"model": im.get("model") or "", "weights": weights_fingerprint(cfg), "data_hash": data_hash, "chunks": int(im.get("chunks", 128)),
            "ctx": int(im.get("ctx", DEFAULT_CTX)), "general_share": share, "general": generaltext.general_key(im.get("general") or {"id": generaltext.DEFAULT_ID}, share, int(im.get("seed", 3407)), render=None), "nextn": bool(nextn),
            "reasoning": reasoning_key(im)}


def _saved_value(info: dict, k: str):
    """What a saved imatrix.json says for key `k`. "nextn" is compared with what was asked and possible then ("nextn_asked"; a matrix whose head was refused still
    answers the same question next time), and a file saved before the key existed counts as made without the head."""
    if k == "nextn":
        return bool(info.get("nextn_asked", info.get("nextn", False)))
    return info.get(k)


def find_saved(dirs, key: dict, general_wanted: bool, reasoning_wanted: bool = False):
    """The newest folder whose imatrix.gguf + imatrix.json were made from exactly `key`. A matrix made without the general text (offline then) is not reused
    when the general text is wanted now."""
    import json
    best = None
    for d in dirs or []:
        d = Path(d)
        try:
            info = json.loads((d / "imatrix.json").read_text(encoding="utf-8"))
            f = d / "imatrix.gguf"
            if not f.is_file() or info.get("format") != FORMAT or any(_saved_value(info, k) != v for k, v in key.items()):
                continue
            if general_wanted and not info.get("general_used"):
                continue
            if reasoning_wanted and not info.get("reasoning_used"):
                continue
            if best is None or f.stat().st_mtime > best[0]:
                best = (f.stat().st_mtime, d, info)
        except (OSError, ValueError):
            continue
    return (best[1], best[2]) if best else (None, None)


def nextn_plan(tool, head):
    """(use --nextn, why not): `head` is whether the model has an MTP head (True / False / None = cannot tell). Asks `tool` only when there is a head."""
    if head is True:
        if nextn_supported(tool):
            return True, ""
        return False, "this llama-imatrix has no --nextn option (an older build)"
    if head is False:
        return False, "the model has no speculative-decoding head"
    return False, "it could not be told whether the model has one"


def calibration_text(tokenizer, rows: list, fam_name: str, im: dict, ctx: int, chunks: int, share: float, note, at: float = 0.0):
    """(text, stats, general texts, general note): the calibration file's text, with the general text fetched first when `share` > 0 (im["general"], im["seed"], the STOP
    file of `im`). Notes about the general text go through `note` and the log. Used by the export and by the matrix library's own job."""
    from . import generaltext
    general, gnote = [], ""
    seed = int(im.get("seed", 3407))
    if share > 0:
        protocol.progress("imatrix", at, "Getting the general text")
        greport: dict = {}
        general, gnote = fetch_general(im.get("general") or {}, tokenizer, int(ctx * chunks * SLACK * share), ctx, lambda: stop_requested(im), fam_name, seed, greport)
        if general:
            protocol.log(f"General text for the matrix: {(im.get('general') or {}).get('file') or (im.get('general') or {}).get('id') or 'the default'} "
                         f"(read as {generaltext.describe(greport['detected']) if greport.get('detected') else 'text'}).")
        if gnote:
            note(gnote)
            protocol.log(gnote)
    reasoning, rkey, rshare = [], reasoning_key(im), 0.0
    if rkey:
        rshare = float(rkey["share"])
        protocol.progress("imatrix", at, "Getting the reasoning text")
        reasoning, rnote = fetch_reasoning(rkey, tokenizer, int(ctx * chunks * SLACK * rshare), lambda: stop_requested(im), fam_name, seed)
        if reasoning:
            protocol.log(f"Reasoning text for the matrix: {len(reasoning)} conversations with thinking from {rkey.get('file') or rkey.get('id')} (cut to {rkey['max_len']} tokens).")
        if rnote:
            note(rnote)
            protocol.log(rnote)
            gnote = (gnote + " " + rnote).strip() if gnote else rnote
    text, stats = build_calibration(tokenizer, rows, fam_name, ctx, chunks, general, share, seed, reasoning, rshare)
    protocol.log(f"Calibration text: {stats['project_rows_used']} project texts ({stats['project_tokens']:,} tokens"
                 + (f", your data repeated to fill it" if stats["repeated"] else "") + f"), {stats['general_rows']} general texts ({stats['general_tokens']:,} tokens)"
                 + (f" and {stats['reasoning_rows']} conversations with thinking ({stats['reasoning_tokens']:,} tokens)." if stats["reasoning_rows"] else "."))
    return text, stats, general, gnote


def matrix_info(key: dict, im: dict, base_repo, stats: dict, general: list, gnote: str, used_nextn: bool, want_nextn: bool, left_out: str, tool) -> dict:
    """The contents of imatrix.json (what the matrix was made from)."""
    g = im.get("general") or {}
    return {"format": FORMAT, **key, "nextn": used_nextn, "nextn_asked": want_nextn, "data_version": im.get("data_version"), "base_repo": base_repo, "general_used": bool(general), "reasoning_used": bool(stats.get("reasoning_rows")), "reasoning_tokens": stats.get("reasoning_tokens", 0),
            "general_source": g.get("file") or g.get("id") or ("default" if general else None), "general_error": gnote or None, "project_tokens": stats["project_tokens"],
            "general_tokens": stats["general_tokens"], "nextn_note": left_out or None, "llama_cpp": tool_version(tool), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def make_matrix(tool, f16, calib, partial, im: dict, chunks: int, ctx: int, want_nextn: bool = False, left_out: str = "", ngl: int = 99, parse_special: bool = True, span=(0.0, 1.0)):
    """Measures the matrix: llama-imatrix reads the calibration file `calib` through the 16-bit (or any) GGUF `f16` and writes `partial` (GGUF format). Returns
    (exit code, whether the MTP head is in it, why the head was left out or ""). Two retries, as the export always had: without --output-format when this llama-imatrix does
    not know the flag, and without --nextn when llama-imatrix says it cannot measure the head. Raises Stopped on the STOP file (im["stop_file"]); the caller removes the files."""
    base = launcher(tool) + ["-m", f16, "-f", calib, "-o", partial]
    flags = ["--no-ppl", "--chunks", chunks] + (["--parse-special"] if parse_special else []) + ["-ngl", ngl, "-c", ctx]

    def measure(with_head):
        extra = ["-b", ctx, "--nextn"] if with_head else []         # --nextn needs one sequence per batch: the batch size must not exceed the context
        rc, tail = run_streaming(base + ["--output-format", "gguf"] + flags + extra, im, chunks, span=span)
        if rc != 0 and any("output-format" in line for line in tail):      # an older llama-imatrix that does not know the flag (its file format is the default there)
            protocol.log("This llama-imatrix has no --output-format; trying without it.")
            rc, tail = run_streaming(base + flags + extra, im, chunks, span=span)
        return rc, tail

    used_nextn = want_nextn
    rc, tail = measure(want_nextn)
    if rc != 0 and want_nextn:
        why = nextn_refused(tail)
        if why:                                                  # the head cannot be measured here: once more without it
            left_out, used_nextn = why, False
            protocol.log(f"llama-imatrix would not measure the MTP head ({why}); trying once without it.")
            Path(partial).unlink(missing_ok=True)
            rc, tail = measure(False)
    return rc, used_nextn, left_out


# ------------------------------------------------------------------------------------------------------------------ running llama-imatrix

def stop_requested(im: dict) -> bool:
    return bool(im.get("stop_file")) and os.path.exists(im["stop_file"])


def run_streaming(cmd: list, im: dict, chunks: int, label="imatrix", span=(0.0, 1.0), tail_len: int = 40, log_tail: bool = True, what: str = "Measuring which weights matter most",
                  marker: str = r"\[(\d+)\]"):
    """Runs a tool and returns (exit code, last output lines). Output is read on a thread so that the STOP file is seen at once: it kills the tool (Stopped).
    Progress (unless label is None): the chunk markers "[12]" llama-imatrix prints, else a heartbeat with the elapsed time; `span` is the part of the bar this call fills
    (the matrix library's job shares the bar with the model conversion). The last `tail_len` lines of the output are returned and, with `log_tail`, go to the log.
    `what` is the progress text and `marker` the regular expression whose first group is the number of the chunk a line reports (other tools, such as the quant comparison, use their own)."""
    protocol.log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    tail: deque = deque(maxlen=tail_len)
    t0, beat, seen = time.time(), 0.0, 0
    if label:
        protocol.progress(label, span[0], what)
    try:
        while True:
            if stop_requested(im):
                raise Stopped()
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                if label and time.time() - beat >= HEARTBEAT and not seen:
                    beat = time.time()
                    protocol.progress(label, span[0], f"{what} ({int(beat - t0)} s)")
                continue
            if line is None:
                break
            tail.append(line)
            marks = [int(m) for m in re.findall(marker, line)]
            if label and marks and max(marks) > seen:
                seen = max(marks)
                protocol.progress(label, span[0] + (span[1] - span[0]) * min(seen / max(chunks, 1), 0.99), f"Chunk {seen} of {chunks}")
        return proc.wait(), list(tail)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if log_tail:
            for line in tail:
                protocol.log(line)


# ------------------------------------------------------------------------------------------------------------------ the step

def load_tokenizer(sources):
    """The model's own tokenizer with its chat template: the merged model folder first, then the base model."""
    from transformers import AutoTokenizer
    last = None
    for s in sources:
        try:
            return AutoTokenizer.from_pretrained(str(s))
        except Exception as exc:    # noqa: BLE001
            last = exc
    raise last or ValueError("no tokenizer")


def model_facts(cfg: dict, merged, f16=None):
    """(architecture or model type, layer count, MTP layers) of the model being exported, from its 16-bit GGUF when there is one, else from a model folder's config.json
    (the merged one, the full model, the base model as a folder or in the Hugging Face cache). None for what cannot be read."""
    if f16 is not None and not callable(f16) and Path(f16).is_file():
        arch, layers = gguf_model_facts(f16)
        if layers is not None:
            return arch, layers, 0
    for f in [merged, cfg.get("full_model"), cfg["base"].get("repo")] + hf_cache_dirs(cfg["base"].get("repo") or ""):
        if f and Path(f).is_dir():
            typ, layers, mtp = folder_model_facts(f)
            if layers is not None:
                return typ, layers, mtp
    return None, None, 0


def model_sizes_of(cfg: dict, merged, f16=None):
    """(hidden size, MLP intermediate size) of the model being exported: from its 16-bit GGUF (`<arch>.embedding_length`, `<arch>.feed_forward_length`) when there is one, else from a
    model folder's config.json, searched as in model_facts. (None, None) when it cannot be read."""
    if f16 is not None and not callable(f16) and Path(f16).is_file():
        try:
            meta, _ = read_gguf_header(f16)
            arch = meta.get("general.architecture")
            h, i = meta.get(f"{arch}.embedding_length"), meta.get(f"{arch}.feed_forward_length")
            if isinstance(h, int) and not isinstance(h, bool):
                return h, i if isinstance(i, int) and not isinstance(i, bool) else None
        except (OSError, ValueError):
            pass
    for f in [merged, cfg.get("full_model"), cfg["base"].get("repo")] + hf_cache_dirs(cfg["base"].get("repo") or ""):
        if f and Path(f).is_dir():
            h, i = folder_model_sizes(f)
            if h is not None or i is not None:
                return h, i
    return None, None


def use_saved_file(cfg: dict, merged, out: Path, f16, im: dict, report: dict):
    """The export uses a matrix from the library (im["use_file"], a .gguf or a legacy .dat; im["use_id"] is its library id): it is copied into the export folder as
    imatrix.gguf / imatrix.dat (with its imatrix.json when it has one) and given to llama-quantize / Unsloth exactly like a freshly measured one. A light check against
    the model puts warnings among the notes; a matrix that does not seem to fit is still used (llama-quantize decides). A missing file is a note and no matrix."""
    import json
    src = Path(im["use_file"])
    note = report["notes"].append
    if not src.is_file():
        msg = f"The saved matrix file is missing ({src}), so the sizes were made without the importance matrix."
        note(msg)
        protocol.log(msg)
        return None
    target = out / ("imatrix.dat" if src.suffix.lower() == ".dat" else "imatrix.gguf")
    if src.resolve() != target.resolve():
        shutil.copy2(src, target)
    info = None
    try:
        info = json.loads(src.with_name("imatrix.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    info = dict(info) if isinstance(info, dict) else {"format": FORMAT, "source": "a matrix file"}
    info["from_library"] = im.get("use_id")
    (out / "imatrix.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    arch, layers, mtp = model_facts(cfg, merged, f16)
    for w in compat_warnings(target, arch, layers, mtp, model_sizes_of(cfg, merged, f16)):
        note("Warning: " + w)
        protocol.log("Warning: " + w)
    report.update(made=False, reused=True, file=str(target), info=info, nextn=bool(info.get("nextn")), from_library=im.get("use_id"))
    protocol.log(f"The saved matrix {im.get('use_id') or src.name} from the library is used.")
    protocol.progress("imatrix", 1.0, "Using the saved matrix")
    return target


def prepare(cfg: dict, merged: Path, out: Path, f16, quantize, quants: list, report: dict):
    """Makes (or reuses) out/imatrix.gguf for the quantisations below Q8_0. Returns the matrix file, or None when there is none: then report["notes"] says why
    in plain words and the sizes are made without it. Only the STOP file raises (Stopped).
    `f16` is the 16-bit GGUF to measure on: a path, or a function that makes it (called only when the matrix is really measured, not when a saved one is reused;
    the file need not exist yet then). With im["nextn"] the speculative-decoding head is measured too when the model has one and this llama-imatrix knows --nextn."""
    import json
    im = cfg["imatrix"]
    note = report["notes"].append

    def give_up(text):
        note(text)
        protocol.log(text)
        return None

    targets = [q for q in quants if q not in NO_IMATRIX]
    if not targets:
        return give_up("The importance matrix helps the sizes below Q8_0; none of them was chosen, so it was not made.")
    if im.get("use_file"):
        return use_saved_file(cfg, merged, out, f16, im, report)
    tool = find_llama_imatrix(quantize)
    if tool is None:
        return give_up("This llama.cpp has no llama-imatrix (an older install), so the sizes were made without the importance matrix. "
                       "Update llama.cpp (llama-imatrix sits next to llama-quantize) to use it.")
    try:
        train = im.get("train")
        if not train or not Path(train).is_file():
            return give_up("The training data of this run is missing, so the importance matrix could not be made. The sizes were made without it.")
        want_nextn, head, nnote = False, None, ""
        if im.get("nextn"):
            head = model_has_mtp(cfg, merged, None if callable(f16) else f16)
            if head is None and callable(f16):          # nothing on disk tells yet (the model is merged only when the 16-bit file is made): make it now
                f16 = f16()
                head = model_has_mtp(cfg, merged, f16)
            want_nextn, nnote = nextn_plan(tool, head)
        key = settings_key(cfg, im, file_sha256(train), want_nextn)
        share = key["general_share"]
        target = out / "imatrix.gguf"
        if im.get("reuse", True):
            folder, info = find_saved(im.get("reuse_dirs"), key, general_wanted=share > 0, reasoning_wanted=bool(key.get("reasoning")))
            if folder is not None:
                if (folder / "imatrix.gguf").resolve() != target.resolve():
                    shutil.copy2(folder / "imatrix.gguf", target)
                    shutil.copy2(folder / "imatrix.json", out / "imatrix.json")
                report.update(made=False, reused=True, file=str(target), info=info, nextn=bool(info.get("nextn")))
                if nnote:
                    note(f"The MTP head was left out: {nnote}.")
                elif want_nextn and not info.get("nextn"):
                    note(f"The MTP head was left out: {info.get('nextn_note') or 'llama-imatrix could not measure it'}.")
                protocol.log(f"The saved importance matrix of this model and data is used again ({key['chunks']} chunks).")
                return target
        tokenizer = load_tokenizer([merged, cfg["base"]["repo"]])
        rows = data.read_jsonl(train)
        ctx, chunks = key["ctx"], key["chunks"]
        text, stats, general, gnote = calibration_text(tokenizer, rows, cfg["base"]["family"], im, ctx, chunks, share, note)
        calib = out / "imatrix-calibration.txt"
        calib.write_text(text, encoding="utf-8")
        partial = out / "imatrix.partial.gguf"
        partial.unlink(missing_ok=True)
        if callable(f16):
            f16 = f16()
        try:
            rc, used_nextn, left_out = make_matrix(tool, f16, calib, partial, im, chunks, ctx, want_nextn, nnote)
        finally:
            calib.unlink(missing_ok=True)
        if left_out:
            note(f"The MTP head was left out: {left_out}.")
        if rc != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            return give_up(f"llama-imatrix failed (exit code {rc}; see the log), so the sizes were made without the importance matrix.")
        os.replace(partial, target)
        info = matrix_info(key, im, cfg["base"]["repo"], stats, general, gnote, used_nextn, want_nextn, left_out, tool)
        (out / "imatrix.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        report.update(made=True, reused=False, file=str(target), info=info, nextn=used_nextn)
        protocol.progress("imatrix", 1.0, "Importance matrix ready")
        return target
    except Stopped:
        for f in (out / "imatrix.partial.gguf", out / "imatrix-calibration.txt"):
            f.unlink(missing_ok=True)
        raise
    except Exception as exc:    # noqa: BLE001 - the matrix is a bonus: whatever went wrong, the export goes on without it
        return give_up(f"The importance matrix could not be made ({type(exc).__name__}: {str(exc)[:200]}), so the sizes were made without it.")
