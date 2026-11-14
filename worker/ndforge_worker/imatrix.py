"""The importance matrix ("imatrix") step of the GGUF export: a few minutes of the model reading typical text show llama.cpp which weights matter most,
and the small sizes (Q6_K and below) are then rounded with more care. llama.cpp's `llama-imatrix` measures it on the 16-bit GGUF; `llama-quantize --imatrix`
uses it. It helps the sizes at about 4 bits and below; Q8_0 and F16/BF16 are left alone.

    calibration text   the project's saved training data as the model sees it (practice conversations through the model's own chat template, world-book
                       texts as they are), about 80 %, plus about 20 % general text (distilljob's general text machinery; offline -> the project's text only)
    imatrix.gguf       the matrix, saved next to the GGUFs with imatrix.json (what it was made from); a later export of the same model, data and settings
                       reuses it
    never fatal        every problem becomes a note and the sizes are made without the matrix; only the Stop button ends the export (Stopped)

Config (job["imatrix"]): {"enabled": true, "model": run name, "data_version": n, "train": train.jsonl path, "chunks": 128, "ctx": 512, "general_share": 0.2,
"reuse": true, "reuse_dirs": [export folders that may hold a saved matrix], "general": {"id" | "file", "column", ...} | None, "stop_file": path}."""
import hashlib
import os
import queue
import random
import re
import shutil
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


def fetch_general(source: dict, tokenizer, budget: int, ctx: int, should_stop=lambda: False):
    """(texts, note): general text of about `budget` tokens through the healing code's source reader (a local JSONL file, or a Hugging Face dataset streamed).
    A text is cut at MAX_GENERAL_DOC chunks. Any problem (offline, no `datasets` library, nothing read) gives ([], plain-language note)."""
    name = source.get("file") or source.get("id") or "the default general text"
    try:
        from . import distilljob
        if not (source.get("file") or source.get("id")):
            source = {**source, "id": distilljob.DEFAULT_GENERAL}
            name = distilljob.DEFAULT_GENERAL
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
        return [], f"The general text ({name}) could not be read ({str(exc)[:160]}). The matrix is made from your project's text only."


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


def settings_key(cfg: dict, im: dict, data_hash: str) -> dict:
    return {"model": im.get("model") or "", "weights": weights_fingerprint(cfg), "data_hash": data_hash, "chunks": int(im.get("chunks", 128)),
            "ctx": int(im.get("ctx", DEFAULT_CTX)), "general_share": float(im.get("general_share", DEFAULT_SHARE))}


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
            if not f.is_file() or info.get("format") != FORMAT or any(info.get(k) != v for k, v in key.items()):
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


def prepare(cfg: dict, merged: Path, out: Path, f16: Path, quantize, quants: list, report: dict):
    """Makes (or reuses) out/imatrix.gguf for the quantisations below Q8_0. Returns the matrix file, or None when there is none: then report["notes"] says why
    in plain words and the sizes are made without it. Only the STOP file raises (Stopped)."""
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
        key = settings_key(cfg, im, file_sha256(train))
        share = key["general_share"]
        target = out / "imatrix.gguf"
        if im.get("reuse", True):
            folder, info = find_saved(im.get("reuse_dirs"), key, general_wanted=share > 0)
            if folder is not None:
                if (folder / "imatrix.gguf").resolve() != target.resolve():
                    shutil.copy2(folder / "imatrix.gguf", target)
                    shutil.copy2(folder / "imatrix.json", out / "imatrix.json")
                report.update(made=False, reused=True, file=str(target), info=info)
                protocol.log(f"The saved importance matrix of this model and data is used again ({key['chunks']} chunks).")
                return target
        tokenizer = load_tokenizer([merged, cfg["base"]["repo"]])
        rows = data.read_jsonl(train)
        ctx, chunks = key["ctx"], key["chunks"]
        general, gnote = [], ""
        if share > 0:
            protocol.progress("imatrix", 0.0, "Getting the general text")
            general, gnote = fetch_general(im.get("general") or {}, tokenizer, int(ctx * chunks * SLACK * share), ctx, lambda: stop_requested(im))
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
        base = launcher(tool) + ["-m", f16, "-f", calib, "-o", partial]
        flags = ["--no-ppl", "--chunks", chunks, "--parse-special", "-ngl", 99, "-c", ctx]
        try:
            rc, tail = run_streaming(base + ["--output-format", "gguf"] + flags, im, chunks)
            if rc != 0 and any("output-format" in line for line in tail):      # an older llama-imatrix that does not know the flag (its file format is the default there)
                protocol.log("This llama-imatrix has no --output-format; trying without it.")
                rc, tail = run_streaming(base + flags, im, chunks)
        finally:
            calib.unlink(missing_ok=True)
        if rc != 0 or not partial.is_file():
            partial.unlink(missing_ok=True)
            return give_up(f"llama-imatrix failed (exit code {rc}; see the log), so the sizes were made without the importance matrix.")
        os.replace(partial, target)
        info = {"format": FORMAT, **key, "data_version": im.get("data_version"), "base_repo": cfg["base"]["repo"], "general_used": bool(general), "general_source": (im.get("general") or {}).get("file")
                or (im.get("general") or {}).get("id") or ("default" if general else None), "project_tokens": stats["project_tokens"], "general_tokens": stats["general_tokens"],
                "llama_cpp": tool_version(tool), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out / "imatrix.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        report.update(made=True, reused=False, file=str(target), info=info)
        protocol.progress("imatrix", 1.0, "Importance matrix ready")
        return target
    except Stopped:
        for f in (out / "imatrix.partial.gguf", out / "imatrix-calibration.txt"):
            f.unlink(missing_ok=True)
        raise
    except Exception as exc:    # noqa: BLE001 - the matrix is a bonus: whatever went wrong, the export goes on without it
        return give_up(f"The importance matrix could not be made ({type(exc).__name__}: {str(exc)[:200]}), so the sizes were made without it.")
