"""A model that is loaded as a processor (a multimodal model, or Unsloth's wrapper) hands back an object whose own `__call__` and `apply_chat_template` look for images.
Plain text must always go through its inner text tokenizer (`data.text_tokenizer`)."""
import ast
import json
from pathlib import Path

import pytest

from ndforge_worker import data, imatrix, thinkcheck_hf
from ndforge_worker.thinkcheck import ThinkCheckError

PKG = Path(__file__).resolve().parents[1] / "ndforge_worker"


class FakeTok:
    """A text tokenizer: one id per character."""
    bos_token = None
    eos_token = "<|im_end|>"
    eos_token_id = 7
    pad_token = None
    pad_token_id = None
    unk_token_id = 0
    all_special_tokens = ["<|im_end|>"]

    def __call__(self, text, add_special_tokens=True, **kw):
        return {"input_ids": [ord(c) % 250 + 10 for c in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        assert not tokenize
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def convert_tokens_to_ids(self, name):
        return {"<|im_end|>": 7}.get(name)

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


class ProcessorStub:
    """What a vision-language loader returns: asking it to tokenize or template text is an error ("parsed for images")."""

    def __init__(self):
        self.tokenizer = FakeTok()

    def __call__(self, *a, **kw):
        raise TypeError("the processor was parsed for images")

    def apply_chat_template(self, *a, **kw):
        raise TypeError("the processor was parsed for images")


CHAT = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_text_tokenizer_unwraps_a_processor_and_leaves_a_plain_tokenizer(monkeypatch):
    proc, plain = ProcessorStub(), FakeTok()
    assert data.text_tokenizer(proc) is proc.tokenizer and data.text_tokenizer(plain) is plain
    assert data._inner is data.text_tokenizer                         # the old name still works


def test_rendering_and_encoding_go_through_the_inner_tokenizer():
    proc = ProcessorStub()
    prompt, full = data.render_chat(proc, CHAT, "generic")
    assert prompt.endswith("<|im_start|>assistant\n") and full.startswith(prompt)
    p2, f2 = data.render_chat_full(proc, CHAT, "generic")
    assert f2.startswith(p2)
    enc, stats = data.encode_rows(proc, [{"type": "text", "text": "some world-book text"}, {"messages": CHAT}], "generic", 200)
    assert stats["text"] == 1 and stats["chat"] == 1 and len(enc) == 2
    assert imatrix.count_tokens(proc, "abc") == 3


def test_the_thinking_check_prompt_and_stops_use_the_inner_tokenizer():
    proc = ProcessorStub()
    inner = data.text_tokenizer(proc)
    request = {"messages": [{"role": "user", "content": "2+2?"}], "thinking": True}
    assert thinkcheck_hf.prompt_of(inner, request).endswith("assistant\n")
    assert thinkcheck_hf.stop_ids(inner, None) == [7]
    with pytest.raises(ThinkCheckError):                               # the processor itself cannot: the failure is the plain sentence, not a crash
        thinkcheck_hf.prompt_of(proc, request)


def _templated_receivers(tree):
    """The names `.apply_chat_template(...)` is called on, with whether each is known to be a text tokenizer in this file."""
    known = {"inner"}                                                   # `inner` parameters are handed the text tokenizer by their callers
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if fname in ("text_tokenizer", "_inner"):
                known |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "apply_chat_template":
            recv = node.func.value
            yield node.lineno, (recv.id if isinstance(recv, ast.Name) else None), isinstance(recv, ast.Name) and recv.id in known


ALLOWED_FILES = {"exportjob.py"}              # Unsloth's own GGUF save needs the processor itself


def test_no_module_applies_a_chat_template_on_something_that_may_be_a_processor():
    bad = []
    for path in sorted(PKG.glob("*.py")):
        if path.name in ALLOWED_FILES:
            continue
        for line, name, ok in _templated_receivers(ast.parse(path.read_text(encoding="utf-8"))):
            if not ok:
                bad.append(f"{path.name}:{line} {name}")
    assert not bad, bad


def test_the_guard_notices_a_bad_call():
    tree = ast.parse("def f(tokenizer):\n    return tokenizer.apply_chat_template([], tokenize=False)\n")
    assert [ok for _, _, ok in _templated_receivers(tree)] == [False]
    tree = ast.parse("def f(tokenizer):\n    tok = text_tokenizer(tokenizer)\n    return tok.apply_chat_template([])\n")
    assert [ok for _, _, ok in _templated_receivers(tree)] == [True]
