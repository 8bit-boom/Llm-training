"""General text for healing (distillation) and for the importance-matrix calibration: where it comes from and what shape it has.

A source is a dict: {"id": Hugging Face dataset id | "file": local path, "config", "split", "column" (None = find it), "seed", "share"}. Rows are read
one at a time (a Hub dataset is streamed, never downloaded as a whole; a local file is read line by line where its format allows it) and turned into texts:

    text column           "text": "..."                                            used as it is
    chat, messages        "messages": [{"role": "user", "content": "..."}, ...]    rendered with the MODEL's own chat template
    chat, conversations   "conversations": [{"from": "human", "value": "..."}]     (ShareGPT style: human/user, gpt/assistant/bot/model, system)
    prompt + completion   "prompt" + "completion" / "response" / "output" / "answer" the same as a two-turn chat

The Hugging Face token (for gated datasets) comes from the environment (HF_TOKEN) only: it is never part of a job file. Everything that reaches the
log or an error message goes through `scrub` (protocol.py) first.

Nothing here needs torch, transformers or the `datasets` library except `hub_loader`, which is the one place that talks to the Hub; tests (and
`source["loader"]`) replace it with any function (source, token) -> iterable of rows.
"""
from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

from . import protocol

DEFAULT_ID = "mlabonne/open-perfectblend"       # open (Apache-2.0), chat; the preset list in the app (crates/ndforge-core/src/general.rs) names the same
SHUFFLE_BUFFER = 2000                           # local files: rows are mixed within a window of this many (the same seed gives the same order)
CHECK_ROWS = 20

CHAT_COLUMNS = ("messages", "conversations", "conversation", "chat")
TEXT_COLUMNS = ("text", "content", "body", "document", "raw_content", "article", "passage")
PROMPT_COLUMNS = ("prompt", "instruction", "question", "input")
ANSWER_COLUMNS = ("completion", "response", "output", "answer")
USER_ROLES = {"user", "human", "prompter", "customer"}
ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "chatgpt", "claude"}
SYSTEM_ROLES = {"system"}


class GeneralTextError(Exception):
    """A problem with the general text that has a plain-language explanation. `code`: gated | not_found | offline | no_column | empty | bad_config | other."""

    def __init__(self, code: str, plain: str):
        super().__init__(plain)
        self.code, self.plain = code, plain


def hub_token() -> str | None:
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return t or None


def dataset_url(source: dict) -> str:
    return f"https://huggingface.co/datasets/{source.get('id') or DEFAULT_ID}"


def source_name(source: dict) -> str:
    return source.get("file") or source.get("id") or DEFAULT_ID


def general_key(source, share, seed) -> dict:
    """What makes one general text a different one: part of the keys of the teacher cache and of the saved importance matrix. The app's Rust code builds
    the same dict (crates/ndforge-core/src/general.rs `key`); keep them identical."""
    if not source or not float(share or 0) > 0:
        return {"used": False}
    sid = source.get("id") or (None if source.get("file") else DEFAULT_ID)
    return {"used": True, "id": sid, "file": source.get("file") or None, "config": source.get("config") or None, "split": source.get("split") or None,
            "column": source.get("column") or None, "share": float(share), "seed": int(seed)}


def same_general(index_key, source, share) -> bool:
    """True when a cache made with `index_key` (the `general_key` stored in its index) is for this general text: the same dataset, config, split, column and share.
    The seed is NOT compared with the caller's: the cache's rows come from the cache itself, so the training run's seed (changed in Expert mode) cannot make them
    other rows; the cache is judged against the seed it was made with."""
    made_with = (index_key or {}).get("seed", 0)
    return (index_key or {}) == general_key(source, share, int(made_with))


# ------------------------------------------------------------------------------------------------------------------ reading rows

def buffer_shuffle(items, seed: int, size: int = SHUFFLE_BUFFER):
    """The items in a mixed order that only depends on `seed` (a window of `size` rows: constant memory for any length)."""
    rng = random.Random(seed)
    buf: list = []
    for it in items:
        if len(buf) < size:
            buf.append(it)
            continue
        j = rng.randrange(size)
        yield buf[j]
        buf[j] = it
    rng.shuffle(buf)
    yield from buf


def _as_row(value):
    return value if isinstance(value, dict) else {"text": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}


def _json_values(fh):
    for n, line in enumerate(fh, 1):
        if line.strip():
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise GeneralTextError("no_column", f"Line {n} of the file is not valid JSON ({str(exc)[:80]}).") from None


def _read_local(path: str):
    p = Path(path)
    if not p.is_file():
        raise GeneralTextError("not_found", f"The file {path} was not found.")
    ext = p.suffix.lower()
    if ext in (".txt", ".md", ".text"):
        yield from _text_blocks(p)
    elif ext in (".csv", ".tsv"):
        csv.field_size_limit(min(2**31 - 1, 1 << 30))
        with open(p, encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ",")
    elif ext == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise GeneralTextError("other", "Reading a .parquet file needs the pyarrow library, which the training software does not have. Use .jsonl, .json, .csv or .txt.") from None
        for batch in pq.ParquetFile(str(p)).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif ext == ".json":
        text = p.read_text(encoding="utf-8-sig")
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if data is None:                                           # not one JSON document: maybe lines of them
            with open(p, encoding="utf-8-sig") as fh:
                for v in _json_values(fh):
                    yield _as_row(v)
            return
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
            data = lists[0] if lists else [data]
        for v in data if isinstance(data, list) else [data]:
            yield _as_row(v)
    else:                                                          # .jsonl, .ndjson and anything else: lines of JSON when the file looks like it, else text
        with open(p, encoding="utf-8-sig") as fh:
            first = ""
            for line in fh:
                if line.strip():
                    first = line.lstrip()[:1]
                    break
        if first in ("{", "[", '"') or ext in (".jsonl", ".ndjson"):
            with open(p, encoding="utf-8-sig") as fh:
                for v in _json_values(fh):
                    yield _as_row(v)
        else:
            yield from _text_blocks(p)


def _text_blocks(p: Path):
    """A text file: blocks separated by blank lines are the documents (a file without blank lines is one document)."""
    block: list = []
    with open(p, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                block.append(line.rstrip("\r\n"))
            elif block:
                yield {"text": "\n".join(block)}
                block = []
    if block:
        yield {"text": "\n".join(block)}


def hub_loader(source: dict, token: str | None):
    """Rows of a Hugging Face dataset, streamed (needs the `datasets` library). The one function in this module that touches the network."""
    from datasets import load_dataset
    kw = {"path": source.get("id") or DEFAULT_ID, "split": source.get("split") or "train", "streaming": True}
    if source.get("config"):
        kw["name"] = source["config"]
    if token:
        kw["token"] = token
    ds = load_dataset(**kw)
    if source.get("seed") is not None and hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=int(source["seed"]), buffer_size=SHUFFLE_BUFFER)
    return iter(ds)


def open_rows(source: dict):
    """An iterator over the rows (dicts). A local file is read the same way whatever its format; with a seed its rows are mixed (buffer_shuffle)."""
    if source.get("file"):
        rows = _read_local(str(source["file"]))
        return buffer_shuffle(rows, int(source["seed"])) if source.get("seed") is not None else rows
    loader = source.get("loader") or hub_loader
    return iter(loader({**source, "id": source.get("id") or DEFAULT_ID}, hub_token()))


# ------------------------------------------------------------------------------------------------------------------ formats

def text_of(value) -> str:
    """a cell as text: a string, or a list of messages / strings joined as lines"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for m in value:
            if isinstance(m, dict):
                out.append(text_of(m.get("content", m.get("value", m.get("text", "")))))
            else:
                out.append(str(m))
        return "\n".join(out)
    return "" if value is None else str(value)


def _chat_shape(value):
    """"messages" | "conversations" | None for a cell that holds a chat"""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        keys = set(value[0])
        if keys & {"role", "content"}:
            return "messages"
        if keys & {"from", "value"}:
            return "conversations"
    return None


def detect(row: dict, column: str | None = None) -> dict:
    """How to read a row: {"kind": "text" | "messages" | "conversations" | "pair", "column": name, "answer": name (pairs)}. With `column` the cell's own shape
    decides; without it the first of: a chat column, a prompt + answer pair, a text column, the longest text of the row. Raises GeneralTextError("no_column")."""
    cols = list(row)
    listing = ", ".join(str(c) for c in cols[:12]) or "none"
    if column:
        if column not in row:
            raise GeneralTextError("no_column", f"The dataset has no column called '{column}'. Its columns: {listing}.")
        v = row[column]
        shape = _chat_shape(v)
        if shape:
            return {"kind": shape, "column": column}
        if isinstance(v, str) and column in PROMPT_COLUMNS:
            ans = next((a for a in ANSWER_COLUMNS if isinstance(row.get(a), str)), None)
            if ans:
                return {"kind": "pair", "column": column, "answer": ans}
        if isinstance(v, (str, list)):
            return {"kind": "text", "column": column}
        raise GeneralTextError("no_column", f"The column '{column}' holds no text (it holds {type(v).__name__}). Its columns: {listing}.")
    for c in CHAT_COLUMNS:
        shape = _chat_shape(row.get(c))
        if shape:
            return {"kind": shape, "column": c}
    for p in PROMPT_COLUMNS:
        if isinstance(row.get(p), str):
            ans = next((a for a in ANSWER_COLUMNS if isinstance(row.get(a), str)), None)
            if ans:
                return {"kind": "pair", "column": p, "answer": ans}
    for c in TEXT_COLUMNS:
        if isinstance(row.get(c), str):
            return {"kind": "text", "column": c}
    strs = [(len(v), c) for c, v in row.items() if isinstance(v, str) and len(v) >= 20]
    if strs:
        return {"kind": "text", "column": max(strs)[1]}
    raise GeneralTextError("no_column", f"No column with usable text was found. Its columns: {listing}. Name the column in \"My own...\" (a text column, or a chat column such as messages or conversations).")


def describe(spec: dict) -> str:
    k, c = spec["kind"], spec["column"]
    if k == "text":
        return f"column '{c}' as plain text"
    if k == "pair":
        return f"columns '{c}' + '{spec['answer']}' as a question and its answer (chat)"
    return f"column '{c}' as chat ({'role/content' if k == 'messages' else 'from/value'} messages)"


def normalize_messages(value):
    """[{"role", "content"}] from either message style (role/content or from/value), or None when the cell is not a usable conversation"""
    if not isinstance(value, list):
        return None
    out = []
    for m in value:
        if not isinstance(m, dict):
            return None
        role = str(m.get("role", m.get("from", m.get("speaker", "")))).strip().lower()
        content = text_of(m.get("content", m.get("value", m.get("text", "")))).strip()
        if role in USER_ROLES:
            role = "user"
        elif role in ASSISTANT_ROLES:
            role = "assistant"
        elif role in SYSTEM_ROLES:
            role = "system"
        else:
            return None                       # tool calls, function results ...: the model's template may not know them
        if m.get("tool_calls") or m.get("tool_call_id"):
            return None                       # a turn that needs tools: the whole row is skipped (the simpler, robust choice)
        reasoning = m.get("reasoning_content")
        if role == "assistant" and isinstance(reasoning, str) and reasoning.strip() and "<think>" not in content:
            content = f"<think>\n{reasoning.strip()}\n</think>\n\n{content}"       # separate thinking field: put it inline once, never double-wrap
        if content:
            out.append({"role": role, "content": content})
    return out or None


def chat_ready(msgs):
    """the conversation cut so that it ends with the assistant's answer and has a user turn before it; None if nothing like that is left"""
    msgs = list(msgs or [])
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    if not msgs or not any(m["role"] == "user" for m in msgs[:-1]):
        return None
    return msgs


def row_item(row: dict, spec: dict):
    """("text", str) | ("chat", messages) | None (a row to skip)"""
    k = spec["kind"]
    if k == "text":
        t = text_of(row.get(spec["column"]))
        return ("text", t) if t.strip() else None
    if k == "pair":
        q, a = row.get(spec["column"]), row.get(spec["answer"])
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            return ("chat", [{"role": "user", "content": q.strip()}, {"role": "assistant", "content": a.strip()}])
        return None
    msgs = chat_ready(normalize_messages(row.get(spec["column"])))
    return ("chat", msgs) if msgs else None


def plain_chat(msgs) -> str:
    return "\n".join(m["content"] for m in msgs)


def texts(source: dict):
    """The general texts, one string per row. Chat rows are rendered with source["render"](messages) -> text (the model's chat template; without it the
    turns are joined as plain lines). source["report"] (a dict, optional) receives {"detected", "rows_read", "texts", "skipped"}. Raises GeneralTextError
    when nothing usable is there; anything else raised by the source (offline, gated ...) passes through for `explain`."""
    report = source.get("report") if isinstance(source.get("report"), dict) else {}
    render = source.get("render")
    report.update(rows_read=0, texts=0, skipped=0, detected=None)
    spec, first_error = None, None
    for row in open_rows(source):
        if not isinstance(row, dict):
            row = _as_row(row)
        report["rows_read"] += 1
        if spec is None:
            if not row:
                continue
            try:
                spec = detect(row, source.get("column") or None)
            except GeneralTextError as exc:
                if source.get("column") or report["rows_read"] >= 5:
                    raise
                first_error = exc
                continue
            report["detected"] = dict(spec)
            protocol.log(f"General text ({source_name(source)}): reading {describe(spec)}" + (" with the model's own chat template." if spec["kind"] != "text" and render else "."))
        item = row_item(row, spec)
        if item is None:
            report["skipped"] += 1
            continue
        kind, value = item
        if kind == "chat":
            try:
                text = render(value) if render else plain_chat(value)
            except Exception:       # noqa: BLE001 - a conversation the template refuses (strict alternation ...) is skipped, not fatal
                report["skipped"] += 1
                continue
        else:
            text = value
        if text and text.strip():
            report["texts"] += 1
            yield text
        else:
            report["skipped"] += 1
    if report["texts"] == 0:
        if spec is None and first_error is not None:
            raise first_error
        if report["rows_read"] == 0:
            raise GeneralTextError("empty", "The dataset gave no rows. Check the config and the split (for example 'train').")
        raise GeneralTextError("no_column", f"None of the {report['rows_read']} rows read had usable text in {describe(spec) if spec else 'any column'}"
                                            + (" (the model's chat template refused the conversations)." if spec and spec["kind"] != "text" and render else "."))


# ------------------------------------------------------------------------------------------------------------------ plain explanations

def explain(exc: BaseException, source: dict) -> GeneralTextError:
    """Any problem reading the general text as a GeneralTextError with a plain-language `plain`: gated, not found, offline, no usable column, ..."""
    if isinstance(exc, GeneralTextError):
        return exc
    name, text = type(exc).__name__, str(exc)
    low = text.lower()
    if source.get("file"):
        if isinstance(exc, FileNotFoundError):
            return GeneralTextError("not_found", f"The file {source['file']} was not found.")
        if isinstance(exc, (UnicodeDecodeError, ValueError)):
            return GeneralTextError("no_column", f"The file could not be read as text ({name}: {text[:100]}).")
        return GeneralTextError("other", f"The file could not be read ({name}: {text[:120]}).")
    url = dataset_url(source)
    sent = "Your Hugging Face token was sent, but this dataset did not accept it: ask for access with the same account. " if hub_token() else ""
    if name == "GatedRepoError" or "gated dataset" in low or "ask for access" in low or "access to dataset" in low or "you must be authenticated" in low:
        return GeneralTextError("gated", f"This dataset needs access approval on Hugging Face ({url}). {sent}Ask for access and add your Hugging Face token, or choose another dataset.")
    if name in ("RepositoryNotFoundError", "DatasetNotFoundError", "RevisionNotFoundError", "FileNotFoundError") or "doesn't exist" in low or "not found" in low or "404" in low or "couldn't find" in low:
        return GeneralTextError("not_found", f"Hugging Face has no dataset called '{source.get('id') or DEFAULT_ID}' that this program can read. Check the spelling (owner/name); "
                                             "a private dataset needs your Hugging Face token.")
    if "401" in low or "403" in low or "unauthorized" in low or "forbidden" in low:
        return GeneralTextError("gated", f"This dataset needs access approval on Hugging Face ({url}). {sent}Ask for access and add your Hugging Face token, or choose another dataset.")
    if "config name" in low or "builderconfig" in low or "unknown split" in low or "bad split" in low or "available splits" in low or "available configs" in low:
        return GeneralTextError("bad_config", f"Hugging Face does not know that config or split of this dataset ({text[:140]}). Check the config and split names on the dataset's page ({url}).")
    if isinstance(exc, (ConnectionError, TimeoutError)) or any(w in low for w in ("offline", "unreachable", "couldn't reach", "could not reach", "connection", "timed out", "name or service", "max retries", "ssl", "network")):
        return GeneralTextError("offline", f"Hugging Face could not be reached; check the internet connection and try again ({text[:110]}).")
    if isinstance(exc, ModuleNotFoundError) and "datasets" in low:
        return GeneralTextError("other", "The training software has no 'datasets' library, so Hugging Face data cannot be read. Run the training setup again, or use a local file.")
    return GeneralTextError("other", f"{name}: {text[:160]}")


def failure_note(exc: BaseException, source: dict, what: str, tail: str, default: str = DEFAULT_ID) -> tuple[str, str]:
    """(plain sentence for the page and the log, code): "The general text (name) could not be <what>: <why> <tail>", with the hint to use the default instead"""
    err = explain(exc, source)
    name = source_name(source)
    hint = f" Or use {default} instead (choose it in the list)." if source.get("id") != default and not source.get("file") or source.get("file") else ""
    if source.get("id") == default and not source.get("file"):
        hint = ""
    return protocol.scrub(f"The general text ({name}) could not be {what}: {err.plain} {tail}{hint}"), err.code


# ------------------------------------------------------------------------------------------------------------------ "Check it"

VERDICTS = {"gated": "needs_access", "not_found": "not_found", "bad_config": "not_found", "offline": "offline", "no_column": "no_column", "empty": "empty", "other": "error"}


def check(source: dict, limit: int = CHECK_ROWS) -> dict:
    """Reads about `limit` rows (streaming) and says what they look like: {"verdict": works | needs_access | not_found | no_column | offline | empty | error,
    "message": plain words, "detected": {...} | None, "rows_read", "texts", "skipped", "sample": [short texts]}. Never raises."""
    source = {k: v for k, v in source.items() if k not in ("seed", "render", "report")}
    report: dict = {}
    sample: list = []
    try:
        for t in texts({**source, "report": report}):
            if len(sample) < 3:
                sample.append(t.strip()[:240])
            if report["texts"] >= limit:
                break
    except Exception as exc:        # noqa: BLE001 - every problem is an answer
        err = explain(exc, source)
        return {"verdict": VERDICTS.get(err.code, "error"), "message": protocol.scrub(err.plain), "detected": report.get("detected"), "rows_read": report.get("rows_read", 0),
                "texts": report.get("texts", 0), "skipped": report.get("skipped", 0), "sample": sample}
    d = report["detected"]
    return {"verdict": "works", "message": f"This works: {describe(d)}; {report['texts']} of {report['rows_read']} rows read had usable text."
            + (" Chat rows are turned into text with your model's own chat template when it is used." if d["kind"] != "text" else ""),
            "detected": d, "rows_read": report["rows_read"], "texts": report["texts"], "skipped": report["skipped"], "sample": sample}


def run_check(cfg: dict) -> int:
    """job general_check: {"source": {...}, "rows": 20}. Result: see `check`."""
    protocol.reserve_stdout()
    protocol.result(check(cfg.get("source") or {}, int(cfg.get("rows") or CHECK_ROWS)))
    return 0
