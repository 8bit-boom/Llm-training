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

from . import data, protocol

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
    """llama-imatrix of the llama.cpp that `quantize` (llama-quantize) belongs to, or on PATH. None when this llama.cpp has none (an older install)."""
    exe = "llama-imatrix.exe" if os.name == "nt" else "llama-imatrix"
    if quantize:
        q = Path(quantize)
        for d in (q.parent, q.parent.parent):
            if (d / exe).is_file():
                return d / exe
    found = shutil.which("llama-imatrix")
    return Path(found) if found else None


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


_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_has_mtp(path):
    """True / False / None for a GGUF file: a key ending `nextn_predict_layers` with a value above 0, or a tensor named `...nextn...` / `mtp...`. None when the file is not
    a readable GGUF (the tests' stand-in files, a truncated file). Only the header is read."""
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

            def value(kind):
                if kind in _GGUF_FIXED:
                    return take({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])
                if kind == 8:
                    return string()
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
                return None
            if take("I") < 2:
                return None
            tensors, kvs = take("Q"), take("Q")
            if tensors > 1 << 22 or kvs > 1 << 22:
                return None
            found = False
            for _ in range(kvs):
                key = string(keep=True)
                v = value(take("I"))
                if key.endswith("nextn_predict_layers") and isinstance(v, int) and v > 0:
                    found = True
            for _ in range(tensors):
                name = string(keep=True)
                for _ in range(take("I")):
                    take("Q")
                take("I")
                take("Q")
                if "nextn" in name or name.startswith("mtp.") or ".mtp." in name:
                    found = True
            return found
    except (OSError, ValueError, struct.error):
        return None


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
    return len(data._inner(tokenizer)(text, add_special_tokens=False)["input_ids"])


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


def build_calibration(tokenizer, rows: list, fam_name: str, ctx: int, chunks: int, general: list, share: float = DEFAULT_SHARE, seed: int = 3407):
    """(text, stats). About ctx x chunks tokens: `share` of them general text (when `general` has any), the rest the project's own rows (shuffled with `seed`,
    repeated when the data is smaller than that). Both kinds are mixed evenly through the file, so that --chunks cuts nothing out of one of them.
    The same rows, settings and general text always give the same file."""
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
             "general_rows": len(gen), "general_tokens": g_tok, "skipped_rows": skipped, "tokens": p_tok + g_tok, "wanted_tokens": ctx * chunks}
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
            "ctx": int(im.get("ctx", DEFAULT_CTX)), "general_share": share, "general": generaltext.general_key(im.get("general") or {"id": generaltext.DEFAULT_ID}, share, int(im.get("seed", 3407))), "nextn": bool(nextn)}


def _saved_value(info: dict, k: str):
    """What a saved imatrix.json says for key `k`. "nextn" is compared with what was asked and possible then ("nextn_asked"; a matrix whose head was refused still
    answers the same question next time), and a file saved before the key existed counts as made without the head."""
    if k == "nextn":
        return bool(info.get("nextn_asked", info.get("nextn", False)))
    return info.get(k)


def find_saved(dirs, key: dict, general_wanted: bool):
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
            if best is None or f.stat().st_mtime > best[0]:
                best = (f.stat().st_mtime, d, info)
        except (OSError, ValueError):
            continue
    return (best[1], best[2]) if best else (None, None)


# ------------------------------------------------------------------------------------------------------------------ running llama-imatrix

def stop_requested(im: dict) -> bool:
    return bool(im.get("stop_file")) and os.path.exists(im["stop_file"])


def run_streaming(cmd: list, im: dict, chunks: int, label="imatrix"):
    """Runs a tool and returns (exit code, last output lines). Output is read on a thread so that the STOP file is seen at once: it kills the tool (Stopped).
    Progress (unless label is None): the chunk markers "[12]" llama-imatrix prints, else a heartbeat with the elapsed time. The tail of the output goes to the log."""
    protocol.log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    lines: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    tail: deque = deque(maxlen=40)
    t0, beat, seen = time.time(), 0.0, 0
    if label:
        protocol.progress(label, 0.0, "Measuring which weights matter most")
    try:
        while True:
            if stop_requested(im):
                raise Stopped()
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                if label and time.time() - beat >= HEARTBEAT and not seen:
                    beat = time.time()
                    protocol.progress(label, 0.0, f"Measuring which weights matter most ({int(beat - t0)} s)")
                continue
            if line is None:
                break
            tail.append(line)
            marks = [int(m) for m in re.findall(r"\[(\d+)\]", line)]
            if label and marks and max(marks) > seen:
                seen = max(marks)
                protocol.progress(label, min(seen / max(chunks, 1), 0.99), f"Chunk {seen} of {chunks}")
        return proc.wait(), list(tail)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
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
            if head is True:
                if nextn_supported(tool):
                    want_nextn = True
                else:
                    nnote = "this llama-imatrix has no --nextn option (an older build)"
            elif head is False:
                nnote = "the model has no speculative-decoding head"
            else:
                nnote = "it could not be told whether the model has one"
        key = settings_key(cfg, im, file_sha256(train), want_nextn)
        share = key["general_share"]
        target = out / "imatrix.gguf"
        if im.get("reuse", True):
            folder, info = find_saved(im.get("reuse_dirs"), key, general_wanted=share > 0)
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
        general, gnote = [], ""
        from . import generaltext
        if share > 0:
            protocol.progress("imatrix", 0.0, "Getting the general text")
            greport: dict = {}
            general, gnote = fetch_general(im.get("general") or {}, tokenizer, int(ctx * chunks * SLACK * share), ctx, lambda: stop_requested(im), cfg["base"].get("family"),
                                           int(im.get("seed", 3407)), greport)
            if general:
                protocol.log(f"General text for the matrix: {(im.get('general') or {}).get('file') or (im.get('general') or {}).get('id') or 'the default'} "
                             f"(read as {generaltext.describe(greport['detected']) if greport.get('detected') else 'text'}).")
            if gnote:
                note(gnote)
                protocol.log(gnote)
        text, stats = build_calibration(tokenizer, rows, cfg["base"]["family"], ctx, chunks, general, share)
        calib = out / "imatrix-calibration.txt"
        calib.write_text(text, encoding="utf-8")
        protocol.log(f"Calibration text: {stats['project_rows_used']} project texts ({stats['project_tokens']:,} tokens"
                     + (f", your data repeated to fill it" if stats["repeated"] else "") + f") and {stats['general_rows']} general texts ({stats['general_tokens']:,} tokens).")
        partial = out / "imatrix.partial.gguf"
        partial.unlink(missing_ok=True)
        if callable(f16):
            f16 = f16()
        base = launcher(tool) + ["-m", f16, "-f", calib, "-o", partial]
        flags = ["--no-ppl", "--chunks", chunks, "--parse-special", "-ngl", 99, "-c", ctx]

        def measure(with_head):
            extra = ["-b", ctx, "--nextn"] if with_head else []         # --nextn needs one sequence per batch: the batch size must not exceed the context
            rc, tail = run_streaming(base + ["--output-format", "gguf"] + flags + extra, im, chunks)
            if rc != 0 and any("output-format" in line for line in tail):      # an older llama-imatrix that does not know the flag (its file format is the default there)
                protocol.log("This llama-imatrix has no --output-format; trying without it.")
                rc, tail = run_streaming(base + flags + extra, im, chunks)
            return rc, tail

        used_nextn, left_out = want_nextn, nnote
        try:
            rc, tail = measure(want_nextn)
            if rc != 0 and want_nextn:
                why = nextn_refused(tail)
                if why:                                                  # the head cannot be measured here: once more without it
                    left_out, used_nextn = why, False
                    protocol.log(f"llama-imatrix would not measure the MTP head ({why}); trying once without it.")
                    partial.unlink(missing_ok=True)
                    rc, tail = measure(False)
        finally:
            calib.unlink(missing_ok=True)
        if left_out:
            note(f"The MTP head was left out: {left_out}.")
        if rc != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            return give_up(f"llama-imatrix failed (exit code {rc}; see the log), so the sizes were made without the importance matrix.")
        os.replace(partial, target)
        info = {"format": FORMAT, **key, "nextn": used_nextn, "nextn_asked": want_nextn, "data_version": im.get("data_version"), "base_repo": cfg["base"]["repo"], "general_used": bool(general), "general_source": (im.get("general") or {}).get("file")
                or (im.get("general") or {}).get("id") or ("default" if general else None), "general_error": gnote or None, "project_tokens": stats["project_tokens"], "general_tokens": stats["general_tokens"],
                "nextn_note": left_out or None, "llama_cpp": tool_version(tool), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
