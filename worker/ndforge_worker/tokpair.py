"""Do a student and a teacher produce the same token ids? (docs/HEAL_ANY_MODEL.md, "What is compared")

Distillation needs one thing of the two tokenizers: the same text gives the same token ids, because the teacher's answers are per-position probabilities over
the shared vocabulary. `distilljob.tokenizer_hash` (every tokenizer file, chat template included) stays what it is: the key of the teacher cache. This module
decides the PAIR:

1. `ids_hash`: sha256 over the files that decide the ids (a canonical form of tokenizer.json, vocab.json, merges.txt, tokenizer.model, added_tokens.json).
   Equal => the same tokenizer.
2. The vocabulary sizes must agree where the files say them.
3. When the hashes differ, both tokenizers are loaded (when `transformers` is installed) and encode the fixed `PROBES`; any difference or load error refuses.

Standard library only, except the probe (transformers, imported late).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

IDS_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json")
ADDED_KEYS = ("content", "id", "special", "normalized", "lstrip", "rstrip", "single_word")

# One constant, tested (worker/tests/test_tokpair.py). Every kind of text where two tokenizers that look alike can still differ.
PROBES = (
    "hello",
    "The quick brown fox jumps over the lazy dog.",
    "Anders de Vallière is a Captain in Dockside.",
    "Hello, world! How are you? (fine) [ok] {yes} <no> 'a' \"b\" - _ / \\ | ~ ` @ # $ % ^ & * + = ; :",
    "0123456789",
    "3.14159 and 1,000,000 and -42 and 1e-5 and 0xFF",
    "2026-11-14 12:30:45",
    "def add(a, b):\n    return a + b\n",
    "for (int i = 0; i < n; ++i) { sum += v[i]; }",
    "SELECT name FROM users WHERE id = 42;",
    "<div class=\"x\">text</div>",
    "{\"key\": [1, 2, 3], \"ok\": true, \"none\": null}",
    "https://example.com/path?q=1&r=2#frag",
    "name@example.com",
    "CamelCaseIdentifier and snake_case_identifier",
    "UPPERCASE lowercase MiXeD",
    "naïve café résumé façade Zoë Ångström",
    "Müller Straße größer",
    "Привет, мир! Как дела?",
    "Γειά σου κόσμε",
    "你好，世界！这是一个测试。",
    "日本語のテキストです。ひらがなとカタカナ",
    "안녕하세요 세계",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "😀 🎉 👍🏽 🇫🇷 ❤️",
    "a  b   c       d",
    " leading space",
    "trailing space ",
    "\t tab\tseparated\t",
    "line one\nline two\r\nline three\n\n\nline six",
    "\n",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|>",
    "<s>[INST] hi [/INST] hello</s>",
    "<think>reasoning</think>",
    "<bos><start_of_turn>user\nhi<end_of_turn>\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>",
    " ",
    "a",
)

NO_TRANSFORMERS = "the test texts were not run because the transformers library is not installed in this software."


def _digest_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def _canon_post(pp):
    """The part of a post-processor that decides which special tokens are inserted (not offsets or trimming)."""
    if not isinstance(pp, dict):
        return pp
    kind = pp.get("type")
    if kind == "ByteLevel":
        return {"type": "ByteLevel"}
    if kind == "Sequence":
        return {"type": "Sequence", "processors": [_canon_post(x) for x in pp.get("processors") or []]}
    return pp                                           # TemplateProcessing / Bert / Roberta / unknown: all of it (conservative)


def canonical_tokenizer_json(tj: dict) -> dict:
    """What tokenizer.json says that decides ids. Left out on purpose: padding, truncation, decoder, version (and any other key)."""
    added = [{k: t.get(k) for k in ADDED_KEYS} for t in tj.get("added_tokens") or [] if isinstance(t, dict)]
    added.sort(key=lambda t: (t["id"] if isinstance(t["id"], int) else -1, str(t["content"])))
    return {"model": tj.get("model"), "added_tokens": added, "normalizer": tj.get("normalizer"), "pre_tokenizer": tj.get("pre_tokenizer"),
            "post_processor": _canon_post(tj.get("post_processor"))}


def ids_hash(folder):
    """sha256 over the files that decide token ids, or None when the folder has none of them."""
    h = hashlib.sha256()
    found = False
    for name in IDS_FILES:
        f = Path(folder) / name
        if not f.is_file():
            continue
        found = True
        data = f.read_bytes()
        if name == "tokenizer.json":
            try:
                data = _digest_json(canonical_tokenizer_json(json.loads(data.decode("utf-8"))))
            except (ValueError, AttributeError):
                pass                                    # not readable as JSON: the raw bytes count (conservative)
        h.update(name.encode() + b"\0" + data + b"\0")
    return h.hexdigest() if found else None


def chat_template(folder):
    """The chat template of a model folder as text (chat_template.jinja, else chat_template.json, else tokenizer_config.json), or None."""
    folder = Path(folder)
    try:
        j = folder / "chat_template.jinja"
        if j.is_file():
            return j.read_text(encoding="utf-8")
        for name in ("chat_template.json", "tokenizer_config.json"):
            f = folder / name
            if f.is_file():
                t = json.loads(f.read_text(encoding="utf-8")).get("chat_template")
                if t:
                    return t if isinstance(t, str) else json.dumps(t, sort_keys=True)
    except (OSError, ValueError, AttributeError):
        return None
    return None


def template_hash(folder):
    t = chat_template(folder)
    return hashlib.sha256(t.encode("utf-8")).hexdigest() if t else None


def vocabulary_size(folder):
    """How many tokens the tokenizer has (its own vocabulary plus the added tokens), or None when the files do not say."""
    folder = Path(folder)
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


class TransformersMissing(Exception):
    pass


def default_loader(folder):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TransformersMissing(str(exc)) from exc
    return AutoTokenizer.from_pretrained(str(folder))


def _show(text: str) -> str:
    s = json.dumps(text, ensure_ascii=False)
    return s if len(s) <= 70 else s[:67] + "..."


def _ids(ids) -> str:
    ids = list(ids)
    return "[" + ", ".join(str(i) for i in ids[:16]) + (", ..." if len(ids) > 16 else "") + "]"


def probe_pair(student_dir, teacher_dir, loader=None, probes=PROBES) -> dict:
    """Encode the probe texts with both tokenizers (add_special_tokens=False). Returns {"status": "passed" | "different" | "error" | "skipped", "n", "text"?, ...}."""
    loader = loader or default_loader
    try:
        a, b = loader(student_dir), loader(teacher_dir)
    except TransformersMissing:
        return {"status": "skipped", "n": 0, "reason": NO_TRANSFORMERS}
    except Exception as exc:                           # noqa: BLE001 - any failure to load is a refusal
        return {"status": "error", "n": 0, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
    for i, text in enumerate(probes):
        try:
            x, y = list(a.encode(text, add_special_tokens=False)), list(b.encode(text, add_special_tokens=False))
        except Exception as exc:                       # noqa: BLE001
            return {"status": "error", "n": i, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
        if x != y:
            return {"status": "different", "n": i, "index": i, "probe": text, "student_ids": x, "teacher_ids": y}
    return {"status": "passed", "n": len(probes)}


def _result(level: str, text: str, **extra) -> dict:
    return {"ok": level != "refuse", "level": level, "text": text, **extra}


def decide_pair(student_dir, teacher_dir, loader=None, probes=PROBES) -> dict:
    """The one decision on a pair: {"ok", "level": "ok" | "warn" | "refuse", "text", "probe"?, "sizes"?}. Used by the Check and by the distill jobs."""
    sh, th = ids_hash(student_dir), ids_hash(teacher_dir)
    if sh is None or th is None:
        return _result("refuse", "A tokenizer's files are missing in one of the models, so the program cannot tell whether they use the same tokenizer. "
                                 "Healing needs them (tokenizer.json or vocab.json).")
    sv, tv = vocabulary_size(student_dir), vocabulary_size(teacher_dir)
    if sv is not None and tv is not None and sv != tv:
        return _result("refuse", f"The two models do not use the same tokenizer, so the teacher's answers cannot teach this model. The vocabularies have {sv:,} and {tv:,} tokens.",
                       sizes=[sv, tv])
    if sh == th:
        return _result("ok", "The tokenizer is identical" + (f" ({sv:,} tokens)" if sv is not None else "") + ".", sizes=[sv, tv])
    p = probe_pair(student_dir, teacher_dir, loader, probes)
    if p["status"] == "passed":
        return _result("warn", f"The tokenizer files differ but both produce the same token ids on {p['n']} test texts; healing can go on.", probe=p, sizes=[sv, tv])
    if p["status"] == "different":
        return _result("refuse", "The two models do not use the same tokenizer: the same text gives different token ids, so the teacher's answers cannot teach this model. "
                                 f"First difference, on the text {_show(p['probe'])}: the model to heal gives {_ids(p['student_ids'])}, the original gives {_ids(p['teacher_ids'])}.",
                       probe=p, sizes=[sv, tv])
    why = p["reason"] if p["status"] == "skipped" else f"the tokenizers could not be compared ({p['reason']})."
    return _result("refuse", f"The tokenizer files of the two models differ, and {why} So the program cannot tell whether they use the same tokenizer, "
                             "and the teacher's answers cannot be used.", probe=p, sizes=[sv, tv])
