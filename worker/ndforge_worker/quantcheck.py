"""Compare quants (docs/QUANT_COMPARE.md): how far is each quantised GGUF from the 16-bit model, measured with llama.cpp's `llama-perplexity` as KL divergence.

    one base pass       llama-perplexity -m reference.gguf -f test.txt --kl-divergence-base base.kld -c 512 --chunks N     writes the 16-bit model's log-probabilities
    one pass per quant  llama-perplexity -m quant.gguf --kl-divergence-base base.kld --kl-divergence -c 512 --chunks N     prints mean / percentile / maximum KLD, same-top-p, PPL
    quantcheck/<id>/    results.json (the table, the key it was measured with), test.txt (the text), base.kld (deleted at the end unless kept)

The size of base.kld is known from llama.cpp's own source (tools/perplexity/perplexity.cpp): 20 bytes of header, the tokens (4 bytes each) of every chunk, and for the
second half of every chunk (n_ctx - 1 - n_ctx/2 tokens) one record of 2 x (2 x ceil(n_vocab / 2) + 4) bytes, so it is computed and shown BEFORE anything runs (`plan`).
Nothing here is fatal for the program: every problem is a QuantCheckError with a plain-language text. Stop (the STOP file) keeps the passes that are finished.

Job {"job": "quantcheck", "root": folder of all comparisons, "id": folder name, "resume": bool, "stop_file": path, "chunks": 16, "ctx": 512, "ngl": 99, "keep_logits": false,
"reference": {"kind": "file", "file": path} | {"kind": "make", "source": <a matrix-library source, see imatrixjob>},
"quants": [{"path", "label", "group" (the size name, "Q4_K_M"), "matrix": true | false | null, "mixed": preset id | null}],
"text": {"kind": "data", "train": path, "test": path | None, "family": ..., "tokenizer": [folder or Hugging Face id], "seed": 1234} | {"kind": "file", "file": path}}
and {"job": "quantcheck_plan", "reference": path | None, "ctx": 512, "root": folder}."""
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

from . import data, fsutil, ggufinfo, imatrix, llamacpp, protocol

FORMAT = 1
CTX = 512
CHUNKS = (4, 8, 16, 32, 64, 128)
DEFAULT_CHUNKS = 16
MIN_CHUNKS = 2                  # llama-perplexity needs at least two contexts of text
HEADROOM = 0.9                  # never plan to fill the disk completely
ASSUMED_VOCAB = 250_000         # when the reference does not say how many words its vocabulary has
SEED = 1234                     # not the calibration's seed (3407): another sample of the saved data
SLACK = 1.1
STOP_MARKER = r"^\s*(\d+)\s+[-+]?\d+\.\d+\s*±"          # a row of the per-chunk table of the comparison pass
INSTALL_HELP = ("llama-perplexity is part of llama.cpp. Install a current llama.cpp (https://github.com/ggml-org/llama.cpp; a release or a build that has llama-quantize also has "
                "llama-perplexity), then tell this program where it is: set the environment variable LLAMA_CPP_DIR to its folder. " + llamacpp.SEARCH_ORDER + " " + llamacpp.CAN_INSTALL)


class QuantCheckError(Exception):
    """A failure with a plain-language text for the person."""


class QuantLlamaMissing(QuantCheckError, llamacpp.LlamaCppMissing):
    """llama-perplexity (llama.cpp) is not there: error code llama_cpp_missing."""


def plain_gb(n) -> str:
    return f"{n / 1e9:.2f} GB"


# ------------------------------------------------------------------------------------------------------------------ the tool, the disk

def find_perplexity():
    """llama-perplexity: next to llama-quantize and in the other places llama.cpp is looked for, then on the PATH (llamacpp.find_tool). None when not found."""
    from . import exportjob
    _, quantize = exportjob.find_llama_cpp()
    return llamacpp.find_tool("llama-perplexity", quantize)


def free_bytes(path) -> int:
    """Free space on the disk that holds `path` (the nearest folder that exists). Tests replace this function."""
    return fsutil.free_bytes(path)


def kld_file_bytes(ctx: int, chunks: int, vocab: int) -> int:
    """The size of the base-logits file (`--kl-divergence-base`), as llama.cpp's perplexity.cpp writes it: 8 bytes `_logits_`, n_ctx, n_vocab, n_chunk (4 bytes each), the
    tokens of all chunks (4 bytes each), then for each chunk the second half of its tokens (n_ctx - 1 - n_ctx/2 of them), each as 2 x (2 x ceil(n_vocab / 2) + 4) bytes (the
    log-probabilities squeezed into 16 bits, with a scale and an offset in front)."""
    per_token = 2 * (2 * ((vocab + 1) // 2) + 4)
    return 20 + 4 * ctx * chunks + chunks * max(ctx - 1 - ctx // 2, 0) * per_token


def plan(vocab, ctx: int = CTX, free=None, chunk_options=CHUNKS) -> dict:
    """What the base file would need, for every amount of text, against the free space (`free`, bytes, None = unknown). {"vocab", "vocab_assumed", "ctx", "scored_per_chunk",
    "options": [{chunks, tokens, scored, bytes, gb, fits}], "free", "usable", "default_chunks", "max_chunks", "refusal": plain text | None}. Pure: the disk is asked by the caller."""
    assumed = not vocab
    v = int(vocab) if vocab else ASSUMED_VOCAB
    usable = None if free is None else int(free * HEADROOM)
    opts = []
    for c in chunk_options:
        b = kld_file_bytes(ctx, c, v)
        opts.append({"chunks": c, "tokens": c * ctx, "scored": c * max(ctx - 1 - ctx // 2, 0), "bytes": b, "gb": round(b / 1e9, 2), "fits": usable is None or b <= usable})
    per_chunk = kld_file_bytes(ctx, 1, v) - 20
    mx = None if usable is None else max(int((usable - 20) // per_chunk), 0)
    fitting = [o["chunks"] for o in opts if o["fits"]]
    default = DEFAULT_CHUNKS if DEFAULT_CHUNKS in fitting else (max(fitting) if fitting else None)
    if default is None and mx is not None and mx >= MIN_CHUNKS:
        default = mx
    refusal = None
    if default is None:
        need = kld_file_bytes(ctx, MIN_CHUNKS, v)
        refusal = (f"There is not enough free disk space for this comparison: even the smallest test ({MIN_CHUNKS} chunks of {ctx} tokens) needs a file of about {plain_gb(need)} "
                   f"for the 16-bit model's answers (its vocabulary has {v:,} words), and only {plain_gb(free)} are free (a tenth is kept in reserve). Free some space, or choose a "
                   "folder on a bigger disk for the project.")
    return {"vocab": v, "vocab_assumed": assumed, "ctx": ctx, "scored_per_chunk": max(ctx - 1 - ctx // 2, 0), "options": opts, "free": free, "usable": usable,
            "default_chunks": default, "max_chunks": mx, "refusal": refusal}


# ------------------------------------------------------------------------------------------------------------------ the text

SPECIAL = [re.compile(p) for p in (r"<\|[^|\n]{1,40}\|>", r"<(?:start|end)_of_turn>", r"\[/?INST\]", r"<(?:bos|eos|s|/s)>")]


def strip_specials(text: str, tokenizer=None) -> str:
    """llama-perplexity cannot parse special tokens (it has no --parse-special; perplexity.cpp tokenises with special parsing off), so the markers of the chat template
    (`<|im_start|>`, `<end_of_turn>`, ...) would be read as ordinary characters. They are taken out here and the role names and line breaks stay: a plain conversation."""
    inner = data.text_tokenizer(tokenizer) if tokenizer is not None else None
    for tok in sorted(set(getattr(inner, "all_special_tokens", None) or []), key=len, reverse=True):
        if isinstance(tok, str) and tok:
            text = text.replace(tok, "\n")
    for rx in SPECIAL:
        text = rx.sub("\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"


def read_rows(path):
    p = Path(path or "")
    if not p.is_file():
        return []
    try:
        return data.read_jsonl(p)
    except Exception:       # noqa: BLE001 - an unreadable file is "no rows"
        return []


def build_text(spec: dict, ctx: int, chunks: int, notes: list):
    """(text, source description, rows used, chunks that the text can fill (at most `chunks`)). The held-out rows of the saved data version (`test.jsonl`) when there are
    any, else a sample of the training rows with another seed than the calibration's; rendered through the model's chat template, the special markers taken out. A text file
    of the person's own is used as it is."""
    if spec.get("kind") == "file":
        f = Path(spec.get("file") or "")
        if not f.is_file():
            raise QuantCheckError(f"The text file {f} was not found. Choose it again.")
        try:
            text = f.read_bytes()[: 1 << 26].decode("utf-8", "replace")
        except OSError as exc:
            raise QuantCheckError(f"The text file {f.name} could not be read ({exc.strerror or type(exc).__name__}).")
        if len(text.strip()) < 2 * ctx:
            raise QuantCheckError(f"The text file {f.name} is too short: the comparison needs at least {2 * ctx} tokens (about {4 * 2 * ctx} letters) and a few thousand words are better.")
        return text, f"the text file {f.name} (used as it is)", None, chunks
    if spec.get("kind") == "general":
        return _general_text(spec, ctx, chunks, notes)
    if spec.get("kind") != "data":
        raise QuantCheckError("Choose the test text: the held-out part of your saved data, or a text file of your own.")
    rows = read_rows(spec.get("test"))
    source = "held-out"
    if not rows:
        rows = read_rows(spec.get("train"))
        source = "sample"
        if not rows:
            raise QuantCheckError("The saved training data of this project (train.jsonl) could not be read, so there is no test text. Choose a text file of your own, or make the training data again (Step 2).")
        notes.append("This version of your data holds out no test questions, so the test text is a sample of the training rows (another sample than the matrix calibration's). "
                     "Quants measured on text the model was trained on look a little better than on new text; the ranking is still fair.")
    try:
        tokenizer = imatrix.load_tokenizer([s for s in spec.get("tokenizer") or [] if s])
    except Exception as exc:    # noqa: BLE001
        raise QuantCheckError("The chat template (the tokenizer) of the model could not be read, so the practice conversations cannot be written the way the model sees them. "
                              f"({type(exc).__name__}: {str(exc)[:160]}) Tell the page which model the files were made from (a folder or a Hugging Face id), or choose a text file of your own.")
    docs, _ = imatrix.render_rows(tokenizer, rows, spec.get("family") or "generic")
    docs = [strip_specials(d, tokenizer) for d in docs]
    if not docs:
        raise QuantCheckError("The saved data has no text to measure with.")
    random.Random(int(spec.get("seed", SEED))).shuffle(docs)
    want = int(ctx * chunks * SLACK) + 1
    out, got = [], 0
    for d in docs:
        out.append(d)
        got += imatrix.count_tokens(tokenizer, d)
        if got >= want:
            break
    possible = got // ctx
    if possible < MIN_CHUNKS:
        raise QuantCheckError(f"There is too little test text: {got:,} tokens, the comparison needs at least {MIN_CHUNKS * ctx:,}. Use a text file of your own (a few thousand words), or save more data.")
    if possible < chunks:
        notes.append(f"The test text is short ({got:,} tokens): {possible} chunks of {ctx} tokens were used instead of {chunks}.")
    return "\n".join(out), f"{'the held-out questions' if source == 'held-out' else 'a sample of the training data'} of the saved data version, as the model sees them ({len(out)} texts)", len(out), min(chunks, possible)


def _general_text(spec: dict, ctx: int, chunks: int, notes: list):
    """The test text of kind "general": general text (the source of the calibration's general share: a Hugging Face dataset or a file) with seed 1234, for a project that has no data of its own
    (Auto-cut of a model folder). It may overlap the matrix calibration's general sample a little, which makes every file look slightly better, about equally."""
    try:
        tokenizer = imatrix.load_tokenizer([s for s in spec.get("tokenizer") or [] if s])
    except Exception as exc:    # noqa: BLE001
        raise QuantCheckError("The chat template (the tokenizer) of the model could not be read, so the general text cannot be prepared. "
                              f"({type(exc).__name__}: {str(exc)[:160]}) Tell the page which model the files were made from, or choose a text file of your own.")
    want = int(ctx * chunks * SLACK) + 1
    texts, note = imatrix.fetch_general(spec.get("source") or {}, tokenizer, want, ctx, seed=int(spec.get("seed", SEED)), fam_name=spec.get("family") or None)
    if not texts:
        raise QuantCheckError((note or "The general text had nothing to read.") + " Choose a text file of your own instead.")
    got = sum(imatrix.count_tokens(tokenizer, t) for t in texts)
    possible = got // ctx
    if possible < MIN_CHUNKS:
        raise QuantCheckError(f"There is too little general text: {got:,} tokens, the comparison needs at least {MIN_CHUNKS * ctx:,}. Use a text file of your own (a few thousand words).")
    if possible < chunks:
        notes.append(f"The general text is short ({got:,} tokens): {possible} chunks of {ctx} tokens were used instead of {chunks}.")
    notes.append("The test text is general text, not your world's: it may overlap the general text the importance matrix was measured on a little, which makes every file look slightly better, about equally.")
    return "\n".join(texts), f"general text (seed {int(spec.get('seed', SEED))}; {len(texts)} texts)", len(texts), min(chunks, possible)


# ------------------------------------------------------------------------------------------------------------------ reading llama-perplexity's output

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_LINE = re.compile(rf"^[\s|]*(?P<label>[^:|=]*?[A-Za-z%\)][^:|=]*?)\s*[:|=]\s*(?P<val>{_NUM})\s*(?:%|)\s*(?:(?:±|\+/-|\+-)\s*(?P<unc>{_NUM}))?")


def _key(label: str):
    """Which figure a line is about, from its label ("Mean    KLD", "99.9%   KLD", "Same top p", "Mean PPL(Q)" ...), None for lines that are none of them."""
    low = re.sub(r"\s+", " ", label.lower()).strip()
    if "kld" not in low and "kl div" not in low and "ppl" not in low and "same top" not in low and "top-1" not in low and "top 1" not in low and "rms" not in low:
        return None
    if "ppl" in low:
        if "(base)" in low and low.startswith("mean ppl") and "/" not in low and "-" not in low.replace("(base)", ""):
            return "ppl_base"
        if low.startswith("mean ppl(q)") and "/" not in low and "-" not in low.replace("ppl(q)", ""):
            return "ppl"
        return None
    if "same top" in low or "top-1" in low or "top 1" in low:
        return "same_top"
    if "rms" in low:
        return "rms_dp"
    if "kld" in low or "kl div" in low:
        m = re.match(r"^(\d+(?:\.\d+)?)\s*%", low)
        if m:
            return {"99.9": "p99_9", "99": "p99", "99.0": "p99", "95": "p95", "95.0": "p95", "90": "p90", "90.0": "p90"}.get(m.group(1))
        if low.startswith("mean") and "ln" not in low:
            return "mean_kld"
        if low.startswith(("max", "maximum")):
            return "max_kld"
        if low.startswith("median"):
            return "median_kld"
    return None


def parse_kld(lines) -> dict:
    """The figures of a `llama-perplexity --kl-divergence` run, read leniently (the exact text may change between builds): {"mean_kld", "kld_unc", "median_kld", "p99_9", "p99", "p95",
    "p90", "max_kld", "same_top" (percent), "rms_dp" (percent), "ppl", "ppl_base", "partial"}; a key is missing when its line was not found. Both the plain lines
    (`Mean    KLD:   0.012345 ± 0.000100`) and a table (`| Mean KLD | 0.0123 ± 0.0001 |`) are understood; if only the running table of the chunks was printed, its last row gives
    the mean KLD, same-top-p and RMS and `partial` is true. Never raises."""
    out: dict = {}
    try:
        last_row = None
        for raw in lines:
            line = str(raw).replace("Δ", "d").replace("Δ", "d")
            m = _LINE.match(line)
            if m:
                k = _key(m.group("label"))
                if k and k not in out:
                    out[k] = float(m.group("val"))
                    if k == "mean_kld" and m.group("unc"):
                        out["kld_unc"] = float(m.group("unc"))
                continue
            row = re.match(rf"^\s*(\d+)\s+({_NUM})\s*±\s*{_NUM}\s+({_NUM})\s*±\s*{_NUM}\s+({_NUM})\s*±\s*({_NUM})\s+({_NUM})\s*±\s*{_NUM}\s*%\s+({_NUM})\s*±\s*{_NUM}\s*%", line)
            if row:
                last_row = row
        if "mean_kld" not in out and last_row is not None:
            out.update(mean_kld=float(last_row.group(4)), kld_unc=float(last_row.group(5)), rms_dp=float(last_row.group(6)), same_top=float(last_row.group(7)), ppl=float(last_row.group(2)), partial=True)
    except Exception:       # noqa: BLE001 - reading the output is a nicety, never a crash
        pass
    return out


# ------------------------------------------------------------------------------------------------------------------ the table, the line, the words

def _gb(row) -> float:
    return row["size"] / 1e9


def pareto(rows: list) -> list:
    """The ids of the rows on the front of "KLD against size" (lower left is better): going from the smallest file up, a row is on it when it is closer to the 16-bit
    model than every smaller (or equally large) file. Rows without a figure are left out."""
    good = sorted((r for r in rows if r.get("status") == "done" and r.get("mean_kld") is not None), key=lambda r: (r["size"], r["mean_kld"], r["id"]))
    best, front = None, []
    for r in good:
        if best is None or r["mean_kld"] < best:
            front.append(r["id"])
            best = r["mean_kld"]
    return front


def _better(a, b) -> int:
    """How many percent lower the KLD of `a` is than the KLD of `b`."""
    return round((b - a) / b * 100) if b else 0


def promotion(rows: list) -> list:
    """For every finished mixed file M: is it worth keeping over the plain file of about its size? [{"id", "against", "promoted", "reason", "delta_pct", "noise"}]. P is the largest plain file
    with size <= 1.01 x M's (else the smallest plain one). M is promoted iff `M.mean_kld + sqrt(uM^2 + uP^2) < P.mean_kld`, with u the +- figures llama-perplexity printed (0 when absent,
    said in `noise`); a smaller difference is "a tie within measurement noise" and is not promoted. The reason also says how M compares with the next bigger plain file. Pure."""
    done = [r for r in rows if r.get("status") == "done" and r.get("mean_kld") is not None]
    plain = sorted((r for r in done if not r.get("mixed")), key=lambda r: (r["size"], r["id"]))
    out = []
    for m in (r for r in done if r.get("mixed")):
        if not plain:
            continue
        le = [r for r in plain if r["size"] <= 1.01 * m["size"]]
        p = le[-1] if le else plain[0]
        um, up = m.get("kld_unc"), p.get("kld_unc")
        noise = math.sqrt((um or 0.0) ** 2 + (up or 0.0) ** 2)
        missing = um is None or up is None
        promoted = m["mean_kld"] + noise < p["mean_kld"]
        pct = round((p["mean_kld"] - m["mean_kld"]) / p["mean_kld"] * 100, 1) if p["mean_kld"] else 0.0
        d = (m["size"] - p["size"]) / 1e9
        size = f"{abs(d):.2f} GB {'larger' if d > 0 else 'smaller'}" if abs(d) >= 0.005 else "the same size"
        if promoted:
            reason = f"{m['label']} ({size} than {p['label']}) is {abs(pct)} % closer to the 16-bit model, more than the measurement noise: it is worth keeping."
        elif m["mean_kld"] <= p["mean_kld"] + noise and abs(m["mean_kld"] - p["mean_kld"]) <= noise:
            reason = f"{m['label']} and {p['label']} are a tie within measurement noise (mean KLD {m['mean_kld']:.4f} against {p['mean_kld']:.4f}): the plain file is as good, so the mixed one is not promoted."
        else:
            reason = f"{m['label']} is {abs(pct)} % further from the 16-bit model than {p['label']} (mean KLD {m['mean_kld']:.4f} against {p['mean_kld']:.4f}): the plain file is better, so the mixed one is not promoted."
        if missing:
            reason += " (llama-perplexity gave no plus-minus figure for one of them, so the noise was taken as 0.)"
        bigger = [r for r in plain if r["size"] > 1.01 * m["size"] and r is not p]
        if bigger:
            n = bigger[0]
            dn = (n["size"] - m["size"]) / 1e9
            pn = _better(n["mean_kld"], m["mean_kld"])
            reason += f" {n['label']} is {dn:.2f} GB larger and {abs(pn)} % {'closer to' if pn >= 0 else 'further from'} the 16-bit model than {m['label']}."
        out.append({"id": m["id"], "against": p["id"], "promoted": bool(promoted), "reason": reason, "delta_pct": pct, "noise": round(noise, 6) if not missing else None})
    return out


def verdict(rows: list) -> list:
    """Plain sentences about the table (a list of strings). KLD is only a proxy, and the last sentence says so."""
    done = [r for r in rows if r.get("status") == "done" and r.get("mean_kld") is not None]
    out = []
    if not done:
        return ["No quant was measured, so there is nothing to compare yet."]
    front = set(pareto(done))
    by_size = sorted(done, key=lambda r: r["size"])
    if len(done) == 1:
        r = done[0]
        out.append(f"{r['label']} ({plain_gb(r['size'])}) has a mean KL divergence of {r['mean_kld']:.4f} from the 16-bit model" + (f" and gives the same top answer in {r['same_top']:.1f} % of the places" if r.get("same_top") is not None else "") + ".")
    else:
        small, close = by_size[0], min(done, key=lambda r: (r["mean_kld"], r["size"]))
        out.append(f"The smallest file is {small['label']} ({plain_gb(small['size'])}, mean KLD {small['mean_kld']:.4f}); the one closest to the 16-bit model is {close['label']} "
                   f"({plain_gb(close['size'])}, mean KLD {close['mean_kld']:.4f}).")
        names = [r["label"] for r in by_size if r["id"] in front]
        out.append("On the line (nothing smaller is closer to the 16-bit model): " + ", ".join(names) + ".")
        off = [r["label"] for r in by_size if r["id"] not in front]
        if off:
            out.append("Not worth it (a smaller file is at least as close to the 16-bit model): " + ", ".join(off) + ".")
        for r in by_size:
            if r["id"] in front and r is not small:
                out.append(f"At about {plain_gb(r['size'])}, {r['label']} keeps the most of the 16-bit model among the files that size or smaller (mean KLD {r['mean_kld']:.4f}).")
                break
    # with and without a matrix, same size name
    groups: dict = {}
    for r in done:
        if r.get("group") and r.get("matrix") is not None and not r.get("mixed"):
            groups.setdefault(r["group"], {})[bool(r["matrix"])] = r
    for g, v in groups.items():
        if True in v and False in v:
            w, wo = v[True], v[False]
            pct = _better(w["mean_kld"], wo["mean_kld"])
            out.append(f"{g} with an importance matrix is {abs(pct)} % {'closer to' if pct >= 0 else 'further from'} the 16-bit model than without one (mean KLD {w['mean_kld']:.4f} against {wo['mean_kld']:.4f}).")
    # mixed precision against the plain size nearest in file size
    plain = [r for r in done if not r.get("mixed")]
    for m in (r for r in done if r.get("mixed")):
        if not plain:
            break
        near = min(plain, key=lambda r: (abs(r["size"] - m["size"]), r["mean_kld"]))
        d = (m["size"] - near["size"]) / 1e9
        pct = _better(m["mean_kld"], near["mean_kld"])
        out.append(f"{m['label']} is {abs(d):.2f} GB {'larger' if d > 0 else 'smaller'} than {near['label']} and {abs(pct)} % {'closer to' if pct >= 0 else 'further from'} the 16-bit model (mean KLD {m['mean_kld']:.4f} against {near['mean_kld']:.4f}).")
    out += [p["reason"] for p in promotion(rows)]
    out.append("KL divergence is a proxy: it says how much the answers' probabilities moved on this text, which is also what Unsloth reports for its quants. It does not replace trying the "
               "model on your own questions (Step 4 · Test).")
    return out


# ------------------------------------------------------------------------------------------------------------------ the passes

def _fingerprint(path) -> str:
    st = Path(path).stat()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
    return f"{st.st_size}:{st.st_mtime_ns}:{h.hexdigest()[:16]}"


def _write(path: Path, obj: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _tail_text(tail) -> str:
    return "\n".join(tail).lower()


def _pass_error(what: str, rc: int, tail) -> str:
    low = _tail_text(tail)
    if "no space left" in low or "failed to write" in low:
        return f"{what} stopped because the disk is full. Free some space (the comparison folder is inside the project) and try again."
    if "out of memory" in low or "failed to allocate" in low or "cudamalloc failed" in low:
        return f"{what} ran out of memory. Close other programs, or use fewer layers on the graphics card, or compare a smaller model."
    if "unable to load model" in low or "failed to load model" in low or "error loading model" in low:
        return f"{what} could not load the file. Is it a complete GGUF file that this llama.cpp can read? (exit code {rc}; the log has its last lines)"
    if "inconsistent vocabulary" in low:
        return f"{what} says the vocabulary differs from the 16-bit reference: this file is not made from the same model."
    if "you need at least" in low:
        return f"{what} says the test text is too short for the context. Use more text or fewer chunks."
    return f"{what} failed (exit code {rc}; the log has its last lines). Common causes: not enough memory, a file llama.cpp cannot read, or an old llama.cpp."


NOT_WRITTEN = "llama-perplexity did not write the file of the 16-bit model's answers (--kl-divergence-base). Is this llama.cpp too old for it? Update llama.cpp and try again."


def _check_base(path: Path, vocab):
    """(chunks in the base file, error text or ""): the header written by llama-perplexity (`_logits_`, n_ctx, n_vocab, n_chunk)."""
    import struct
    try:
        with open(path, "rb") as fh:
            head = fh.read(20)
    except OSError as exc:
        return 0, f"The file of the 16-bit model's answers could not be read ({exc.strerror or type(exc).__name__})."
    if len(head) < 20 or head[:8] != b"_logits_":
        return 0, NOT_WRITTEN
    n_ctx, n_vocab, n_chunk = struct.unpack("<iii", head[8:20])
    if n_chunk < 1:
        return 0, "The 16-bit model's answers file is empty: the test text was too short for even one chunk."
    return n_chunk, ""


def start_rows(job_quants: list, old: list) -> list:
    """The rows of the table for the quants asked for: an old row that was measured (same file, same size and time) is kept as it is, so it is skipped."""
    rows = []
    for i, q in enumerate(job_quants, 1):
        path = str(q.get("path") or "")
        row = {"id": str(i), "path": path, "name": Path(path).name, "label": str(q.get("label") or Path(path).name), "group": q.get("group"), "matrix": q.get("matrix"), "mixed": q.get("mixed"),
               "status": "pending", "error": None}
        try:
            row["fp"] = _fingerprint(path)
        except OSError:
            row["fp"] = None
        for o in old:
            if o.get("path") == path and o.get("fp") and o.get("fp") == row["fp"] and o.get("status") == "done":
                row = {**o, "id": row["id"], "label": row["label"], "group": row["group"], "matrix": row["matrix"], "mixed": row["mixed"]}
                break
        rows.append(row)
    return rows


def _summary(res: dict) -> dict:
    res["pareto"] = pareto(res["rows"])
    res["verdict"] = verdict(res["rows"])
    res["promotion"] = promotion(res["rows"])
    res["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return res


def make(job: dict) -> dict:
    """Runs a comparison into <root>/<id>/. Returns the result (the contents of results.json). Raises QuantCheckError (plain text) or imatrix.Stopped; what was finished is kept."""
    root, cid = Path(job["root"]), str(job["id"])
    if not cid or cid.startswith(("_", ".")) or "/" in cid or "\\" in cid or ".." in cid:
        raise QuantCheckError("That is not a valid name for a comparison.")
    ctx, chunks = int(job.get("ctx", CTX)), int(job.get("chunks", DEFAULT_CHUNKS))
    if chunks < MIN_CHUNKS or chunks > 1024 or ctx < 64:
        raise QuantCheckError(f"Choose between {MIN_CHUNKS} and 1024 chunks of at least 64 tokens.")
    quants = [q for q in job.get("quants") or [] if q.get("path")]
    if not quants:
        raise QuantCheckError("Choose at least one quantised GGUF file to compare.")
    tool = find_perplexity()
    if tool is None:
        raise QuantLlamaMissing("llama-perplexity was not found, so the quants cannot be compared. " + INSTALL_HELP)
    im = {"stop_file": job.get("stop_file")}
    out = root / cid
    resume = bool(job.get("resume")) and (out / "results.json").is_file()
    notes: list = []
    old = {}
    if resume:
        try:
            old = json.loads((out / "results.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise QuantCheckError("The earlier comparison could not be read any more (its results.json is damaged). Start a new comparison.")
    elif out.exists():
        raise QuantCheckError(f"The folder {out} exists already. Try again in a minute.")
    ref_spec = job.get("reference") or {}
    # --- the reference file
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    base_file, test_file = out / "base.kld", out / "test.txt"
    try:
        reference = resolve_reference(job, ref_spec, out, im, notes, old if resume else {})
        try:
            rdesc = ggufinfo.describe(reference)
        except (OSError, ValueError) as exc:
            raise QuantCheckError(f"The reference file {Path(reference).name} could not be read as a GGUF file ({exc}). Choose a 16-bit GGUF (F16 or BF16).")
        if rdesc["full_precision"] is False:
            notes.append(f"The reference file {Path(reference).name} is not a 16-bit file: it is already quantised, so the figures say how far each file is from it, not from the original model. "
                         "Use an F16 or BF16 file when you have one.")
        # --- the test text (kept in the folder, so that a resumed comparison measures the same text)
        text_info = (old.get("text") or {}) if resume else {}
        if resume and test_file.is_file() and text_info:
            text_chunks, src_desc, rows_used = int(old["settings"]["chunks"]), text_info.get("source"), text_info.get("rows")
        else:
            text, src_desc, rows_used, text_chunks = build_text(job.get("text") or {}, ctx, chunks, notes)
            test_file.write_text(text, encoding="utf-8")
        chunks = min(chunks, text_chunks)
        text_hash = imatrix.file_sha256(test_file)
        # --- the disk, before anything is written
        pl = plan(rdesc["vocab"], ctx, free_bytes(out))
        need = kld_file_bytes(ctx, chunks, pl["vocab"])
        if not (resume and base_file.is_file()) and pl["usable"] is not None and need > pl["usable"]:
            if pl["max_chunks"] is not None and pl["max_chunks"] >= MIN_CHUNKS:
                raise QuantCheckError(f"There is not enough free disk space for {chunks} chunks: the 16-bit model's answers would need a file of about {plain_gb(need)} and only "
                                      f"{plain_gb(pl['free'])} are free (a tenth is kept in reserve). {pl['max_chunks']} chunks would fit; choose that, or free some space.")
            raise QuantCheckError(pl["refusal"] or "There is not enough free disk space for this comparison.")
        version = imatrix.tool_version(tool)
        key = {"reference": _fingerprint(reference), "text": text_hash, "chunks": chunks, "ctx": ctx, "llama_cpp": version}
        same_key = bool(resume and old.get("key") == key)
        if resume and not same_key:
            if base_file.is_file():
                raise QuantCheckError("The earlier comparison was made with other settings (another reference file, text, amount of text or llama.cpp version), so its saved answers of the 16-bit "
                                      "model do not fit. Start a new comparison.")
            old = {}                    # nothing to continue from: its measured rows were made against other answers
        res = {"format": FORMAT, "id": cid, "created": old.get("created") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": "running", "key": key,
               "reference": {"path": str(reference), "name": Path(reference).name, "size": rdesc["size"], "params": rdesc["params"], "bpw": rdesc["bpw"], "vocab": rdesc["vocab"], "arch": rdesc["arch"]},
               "text": {"source": src_desc, "rows": rows_used, "file": "test.txt"}, "settings": {"chunks": chunks, "ctx": ctx, "ngl": int(job.get("ngl", 99))},
               "base": {"done": bool(same_key and base_file.is_file() and old.get("base", {}).get("done")), "bytes": 0, "expected_bytes": need, "kept": False},
               "rows": start_rows(quants, old.get("rows") or []), "notes": notes, "disk": {"free": pl["free"], "need": need}}
        return _run_passes(job, tool, reference, rdesc, res, out, base_file, test_file, im)
    except (QuantCheckError, imatrix.Stopped):
        _cleanup_empty(out, resume)
        raise
    except OSError as exc:
        _cleanup_empty(out, resume)
        raise QuantCheckError("The disk is full. Free some space and try again." if getattr(exc, "errno", None) == 28 else f"A file could not be written ({exc.strerror or type(exc).__name__}).")


def _cleanup_empty(out: Path, resume: bool):
    """A comparison that failed before anything was measured leaves no folder behind (a resumed one keeps what it had)."""
    if not resume and out.is_dir() and not (out / "results.json").is_file():
        shutil.rmtree(out, ignore_errors=True)


def resolve_reference(job: dict, spec: dict, out: Path, im: dict, notes: list, old: dict) -> Path:
    if spec.get("kind") == "file":
        f = Path(spec.get("file") or "")
        if not f.is_file():
            raise QuantCheckError(f"The reference file {f} was not found. Choose it again.")
        return f
    if spec.get("kind") != "make":
        raise QuantCheckError("Choose the 16-bit reference file (F16 or BF16), or let the program make one with llama.cpp.")
    target = out / "reference-f16.gguf"
    if target.is_file():
        protocol.log("The 16-bit reference made earlier in this folder is used again.")
        return target
    from . import imatrixjob
    try:
        conv, _ = imatrixjob.find_tools(need_convert=True)
        f16, _, _, described, _ = imatrixjob.source_f16(job, spec.get("source") or {}, out, conv, im, notes)
    except imatrixjob.MatrixError as exc:
        raise (QuantLlamaMissing if isinstance(exc, llamacpp.LlamaCppMissing) else QuantCheckError)(str(exc))
    finally:
        shutil.rmtree(out / "merged", ignore_errors=True)
    f16 = Path(f16)
    if f16.resolve() != target.resolve() and f16.parent.resolve() == out.resolve():
        os.replace(f16, target)
        f16 = target
    notes.append(f"The 16-bit reference was made from {described} and kept in the comparison folder ({plain_gb(f16.stat().st_size)}); deleting the comparison deletes it.")
    return f16


def _run_passes(job, tool, reference, rdesc, res, out, base_file, test_file, im) -> dict:
    ctx, chunks, ngl = res["settings"]["ctx"], res["settings"]["chunks"], res["settings"]["ngl"]
    todo = [r for r in res["rows"] if r["status"] != "done"]
    base_needed = bool(todo) and not res["base"]["done"]
    total = (1 if base_needed else 0) + len(todo)
    done = 0
    results_file = out / "results.json"

    def save(state):
        res["state"] = state
        _write(results_file, _summary(res))

    save("running")
    try:
        if base_needed:
            protocol.progress("quantcheck", 0.0, "The 16-bit model reads the test text")
            base_file.unlink(missing_ok=True)
            cmd = imatrix.launcher(tool) + ["-m", reference, "-f", test_file, "--kl-divergence-base", base_file, "-c", ctx, "--chunks", chunks, "-ngl", ngl]
            rc, tail = imatrix.run_streaming(cmd, im, chunks, "quantcheck", (0.0, 1 / total), tail_len=300, log_tail=False, what="The 16-bit model reads the test text")
            _log_tail(tail)
            n_chunk, err = _check_base(base_file, rdesc["vocab"]) if base_file.is_file() else (0, NOT_WRITTEN if rc == 0 else "")
            if rc != 0 or err or not base_file.is_file():
                base_file.unlink(missing_ok=True)
                raise QuantCheckError(err or _pass_error("llama-perplexity (the 16-bit model's pass)", rc, tail))
            res["base"].update(done=True, bytes=base_file.stat().st_size, chunks_used=n_chunk)
            save("running")
            done = 1
        for row in todo:
            row_label = row["label"]
            protocol.progress("quantcheck", done / max(total, 1), f"Comparing {row_label}")
            try:
                d = ggufinfo.describe(row["path"])
            except (OSError, ValueError) as exc:
                row.update(status="failed", error=f"The file could not be read as a GGUF file ({exc}).")
                done += 1
                save("running")
                continue
            if rdesc["vocab"] and d["vocab"] and d["vocab"] != rdesc["vocab"]:
                row.update(status="failed", error=f"Its vocabulary has {d['vocab']:,} words, the reference's has {rdesc['vocab']:,}: it is not made from the same model, so the answers cannot be compared.")
                done += 1
                save("running")
                continue
            if rdesc["arch"] and d["arch"] and d["arch"] != rdesc["arch"]:
                row.update(status="failed", error=f"Its architecture is {d['arch']}, the reference's is {rdesc['arch']}: not the same model.")
                done += 1
                save("running")
                continue
            cmd = imatrix.launcher(tool) + ["-m", row["path"], "--kl-divergence-base", base_file, "--kl-divergence", "-c", ctx, "--chunks", chunks, "-ngl", ngl]
            rc, tail = imatrix.run_streaming(cmd, im, chunks, "quantcheck", (done / total, (done + 1) / total), tail_len=600, log_tail=False, what=f"Comparing {row_label}", marker=STOP_MARKER)
            _log_tail(tail)
            figures = parse_kld(tail)
            row.update(size=d["size"], params=d["params"], bpw=d["bpw"])
            if "mean_kld" in figures:
                row.update(status="done", error=None, **figures)
                if rc != 0:
                    row["error"] = f"llama-perplexity ended with exit code {rc} after printing its figures."
            else:
                row.update(status="failed", error=_pass_error("llama-perplexity", rc, tail) if rc != 0 else
                           "llama-perplexity finished but printed no KL divergence figures (see the log). This llama.cpp may print them in a form this program does not know, or the text was too short.")
            done += 1
            save("running")
    except imatrix.Stopped:
        if not res["base"]["done"]:
            base_file.unlink(missing_ok=True)
        save("stopped")
        raise
    except QuantCheckError:
        save("failed")
        raise
    ok = all(r["status"] == "done" for r in res["rows"])
    if ok and not job.get("keep_logits") and base_file.is_file():
        base_file.unlink()
        res["base"].update(kept=False, deleted=True)
    elif base_file.is_file():
        res["base"].update(kept=True, bytes=base_file.stat().st_size)
        if not ok:
            res["notes"].append(f"Some files could not be measured. The 16-bit model's answers ({plain_gb(base_file.stat().st_size)}) were kept so that a later run can continue without reading the text again; delete them when you are done.")
    save("done")
    protocol.progress("quantcheck", 1.0, "Comparison ready")
    return res


def _log_tail(tail, n=80):
    for line in list(tail)[-n:]:
        protocol.log(line)


# ------------------------------------------------------------------------------------------------------------------ the jobs

def plan_job(job: dict) -> dict:
    """The disk need before a run. `reference` is a GGUF file (its vocabulary is read from the header) or None (the vocabulary is then assumed; `vocab` may give it)."""
    vocab, note = job.get("vocab"), None
    ref = job.get("reference")
    if ref:
        try:
            vocab = ggufinfo.describe(ref)["vocab"] or vocab
        except (OSError, ValueError) as exc:
            note = f"The reference file could not be read ({exc}); the vocabulary is assumed."
    root = Path(job.get("root") or ".")
    out = plan(vocab, int(job.get("ctx", CTX)), free_bytes(root))
    out["note"] = note
    return out


def run(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(make(job))
    return 0


def run_plan(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(plan_job(job))
    return 0
