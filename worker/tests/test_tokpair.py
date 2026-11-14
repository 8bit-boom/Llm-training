"""The pair decision of the Heal check: the ids hash, the probe texts, the chat template (docs/HEAL_ANY_MODEL.md)."""
import copy
import json
from pathlib import Path

import pytest

from ndforge_worker import tokpair as tp
from ndforge_worker.distilljob import tokenizer_hash


def tj(words=("a", "b", "c"), **over):
    d = {"version": "1.0", "truncation": None, "padding": None,
         "added_tokens": [{"id": 90, "content": "<|im_end|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
         "normalizer": None, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False}, "post_processor": {"type": "ByteLevel", "trim_offsets": True},
         "decoder": {"type": "ByteLevel"}, "model": {"type": "BPE", "vocab": {w: i for i, w in enumerate(words)}, "merges": ["a b"]}}
    d.update(over)
    return d


def folder(path: Path, tokenizer=None, config=None, special=None, template=None, **files):
    path.mkdir(parents=True, exist_ok=True)
    if tokenizer is not False:
        (path / "tokenizer.json").write_text(json.dumps(tokenizer or tj()), encoding="utf-8")
    cfg = dict(config or {})
    if template is not None:
        cfg["chat_template"] = template
    (path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if special is not None:
        (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    for name, text in files.items():
        (path / name.replace("__", ".")).write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------------------------------ the probe set

def test_the_probe_set_is_one_constant_with_every_kind_of_text():
    P = tp.PROBES
    assert 36 <= len(P) <= 48 and len(set(P)) == len(P) and all(isinstance(x, str) and x for x in P)
    joined = "".join(P)
    for needle in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "é", "你", "😀", "\n", "\t", "  ", "def ", "0123456789", "Привет"):
        assert needle in joined, needle
    assert " " in P and any(not x.strip() for x in P)                         # an empty-ish text


# ------------------------------------------------------------------------------------------------------------------------ the ids hash

def test_the_ids_hash_ignores_what_does_not_decide_ids(tmp_path):
    base = tp.ids_hash(folder(tmp_path / "a"))
    other = tj(padding={"strategy": "Longest"}, truncation={"max_length": 5}, decoder={"type": "WordPiece"}, version="2.0")
    other["post_processor"] = {"type": "ByteLevel", "trim_offsets": False}
    b = folder(tmp_path / "b", tokenizer=other, config={"padding_side": "left", "model_max_length": 5}, special={"eos_token": "<|im_end|>"}, template="{{ x }}")
    assert tp.ids_hash(b) == base
    assert tokenizer_hash(tmp_path / "a") != tokenizer_hash(b), "the cache's own hash is unchanged and still strict"


def test_the_ids_hash_changes_with_what_decides_ids(tmp_path):
    base = tp.ids_hash(folder(tmp_path / "a"))
    variants = {
        "vocab": tj(words=("a", "c", "b")),
        "merge": tj(model={"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": ["b c"]}),
        "added id": tj(added_tokens=[{"id": 91, "content": "<|im_end|>", "special": True, "normalized": False}]),
        "added flag": tj(added_tokens=[{"id": 90, "content": "<|im_end|>", "special": False, "normalized": False}]),
        "added content": tj(added_tokens=[{"id": 90, "content": "<|im_start|>", "special": True, "normalized": False}]),
        "normalizer": tj(normalizer={"type": "NFC"}),
        "pre_tokenizer": tj(pre_tokenizer={"type": "ByteLevel", "add_prefix_space": True}),
        "post_processor": tj(post_processor={"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}], "pair": None, "special_tokens": {}}),
    }
    seen = {base}
    for name, t in variants.items():
        h = tp.ids_hash(folder(tmp_path / name.replace(" ", "_"), tokenizer=t))
        assert h not in seen, name
        seen.add(h)


def test_the_order_of_added_tokens_and_keys_does_not_matter(tmp_path):
    a = tj(added_tokens=[{"id": 1, "content": "x", "special": True}, {"id": 2, "content": "y", "special": False}])
    b = tj(added_tokens=[{"special": False, "content": "y", "id": 2}, {"special": True, "content": "x", "id": 1}])
    assert tp.ids_hash(folder(tmp_path / "a", tokenizer=a)) == tp.ids_hash(folder(tmp_path / "b", tokenizer=b))


def test_the_other_id_files_count_and_a_bad_json_falls_back_to_the_bytes(tmp_path):
    base = tp.ids_hash(folder(tmp_path / "a", tokenizer=False, vocab__json='{"a": 0}', merges__txt="a b\n"))
    assert base == tp.ids_hash(folder(tmp_path / "b", tokenizer=False, vocab__json='{"a": 0}', merges__txt="a b\n", config={"x": 1}))
    assert base != tp.ids_hash(folder(tmp_path / "c", tokenizer=False, vocab__json='{"a": 0}', merges__txt="b a\n"))
    assert base != tp.ids_hash(folder(tmp_path / "d", tokenizer=False, vocab__json='{"a": 0}', merges__txt="a b\n", added_tokens__json='{"<x>": 1}'))
    bad1, bad2 = folder(tmp_path / "e", tokenizer=False), folder(tmp_path / "f", tokenizer=False)
    (bad1 / "tokenizer.json").write_text("not json", encoding="utf-8")
    (bad2 / "tokenizer.json").write_text("not json!", encoding="utf-8")
    assert tp.ids_hash(bad1) != tp.ids_hash(bad2)
    assert tp.ids_hash(folder(tmp_path / "g", tokenizer=False)) is None


def test_the_chat_template_is_found_wherever_it_is_stored(tmp_path):
    a = folder(tmp_path / "a", template="T1")
    b = folder(tmp_path / "b")
    (b / "chat_template.jinja").write_text("T1", encoding="utf-8")
    c = folder(tmp_path / "c", template="T2")
    assert tp.template_hash(a) == tp.template_hash(b) != tp.template_hash(c)
    assert tp.template_hash(folder(tmp_path / "d")) is None


# ------------------------------------------------------------------------------------------------------------------------ the pair decision

def test_identical_folders_and_folders_that_differ_only_in_settings_are_accepted_without_a_probe(tmp_path):
    a = folder(tmp_path / "a", template="T1", special={"eos_token": "x"})
    b = folder(tmp_path / "b", config={"padding_side": "left"}, special={"eos_token": "y"}, template="T2")

    def boom(_):
        raise AssertionError("no probe needed")
    for other in (folder(tmp_path / "same"), b):
        d = tp.decide_pair(a, other, loader=boom)
        assert d["ok"] and d["level"] == "ok" and d["text"].startswith("The tokenizer is identical (4 tokens)")


def test_a_vocabulary_size_mismatch_is_refused_with_both_sizes_and_never_probed(tmp_path):
    a, b = folder(tmp_path / "a"), folder(tmp_path / "b", tokenizer=tj(("a", "b", "c", "d")))

    def boom(_):
        raise AssertionError("not reached")
    d = tp.decide_pair(a, b, loader=boom)
    assert not d["ok"] and d["level"] == "refuse" and "same tokenizer" in d["text"] and "4 and 5 tokens" in d["text"]


def test_a_missing_tokenizer_is_refused(tmp_path):
    d = tp.decide_pair(folder(tmp_path / "a"), folder(tmp_path / "b", tokenizer=False))
    assert not d["ok"] and "missing" in d["text"] and "same tokenizer" in d["text"]


class Tok:
    """a stand-in tokenizer: every character is its code point (so two stand-ins agree unless told otherwise)"""
    def __init__(self, tweak=None, fail_on=None):
        self.tweak, self.fail_on = tweak or {}, fail_on

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        if self.fail_on and self.fail_on in text:
            raise ValueError("cannot encode")
        return self.tweak.get(text) or [ord(c) for c in text]


def loader_for(a, b):
    return lambda folder: a if Path(folder).name == "a" else b


def differing(tmp_path):
    return folder(tmp_path / "a"), folder(tmp_path / "b", tokenizer=tj(("a", "b", "z")))             # same size, other ids


def test_the_probe_accepts_equal_ids_with_a_yellow_warning(tmp_path):
    a, b = differing(tmp_path)
    d = tp.decide_pair(a, b, loader=loader_for(Tok(), Tok()))
    assert d["ok"] and d["level"] == "warn"
    assert d["text"] == f"The tokenizer files differ but both produce the same token ids on {len(tp.PROBES)} test texts; healing can go on."


def test_the_probe_refuses_one_different_text_and_shows_it(tmp_path):
    a, b = differing(tmp_path)
    target = tp.PROBES[20]
    d = tp.decide_pair(a, b, loader=loader_for(Tok(), Tok({target: [1, 2, 3]})))
    assert not d["ok"] and d["level"] == "refuse" and d["probe"]["index"] == 20
    assert "same tokenizer" in d["text"] and json.dumps(target, ensure_ascii=False)[:20] in d["text"] and "[1, 2, 3]" in d["text"] and "First difference" in d["text"]
    # the FIRST one is shown, not the last
    d2 = tp.decide_pair(a, b, loader=loader_for(Tok(), Tok({tp.PROBES[3]: [9], tp.PROBES[30]: [8]})))
    assert d2["probe"]["index"] == 3


def test_a_load_or_encode_error_is_refused(tmp_path):
    a, b = differing(tmp_path)

    def broken(folder):
        raise OSError("no such tokenizer")
    d = tp.decide_pair(a, b, loader=broken)
    assert not d["ok"] and "could not be compared" in d["text"] and "no such tokenizer" in d["text"] and "same tokenizer" in d["text"]
    d = tp.decide_pair(a, b, loader=loader_for(Tok(), Tok(fail_on="hello")))
    assert not d["ok"] and "cannot encode" in d["text"]


def test_without_transformers_the_probe_is_skipped_and_the_pair_refused_with_a_note(tmp_path, monkeypatch):
    a, b = differing(tmp_path)

    def missing(folder):
        raise tp.TransformersMissing("no transformers")
    d = tp.decide_pair(a, b, loader=missing)
    assert not d["ok"] and d["probe"]["status"] == "skipped" and "transformers library is not installed" in d["text"] and "same tokenizer" in d["text"]
    import builtins
    real = builtins.__import__

    def no_tf(name, *args, **kw):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("blocked")
        return real(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", no_tf)
    assert tp.decide_pair(a, b)["probe"]["status"] == "skipped"


# ------------------------------------------------------------------------------------------------------------------------ the tiny real tokenizers

def tiny_pair(tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tinymodel import build_tokenizer
    tok = build_tokenizer("qwen3_5")
    a = tmp_path / "a"
    tok.save_pretrained(a)
    return tok, a


def copy_folder(src: Path, dst: Path):
    dst.mkdir()
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


def test_real_tiny_tokenizers_a_changed_chat_template_is_accepted_with_a_template_note(tmp_path):
    tok, a = tiny_pair(tmp_path)
    b = copy_folder(a, tmp_path / "b")
    tok.chat_template = "{% for m in messages %}[{{ m['content'] }}]{% endfor %}"
    tok.save_pretrained(tmp_path / "t")
    for f in (tmp_path / "t").iterdir():                                     # the template wherever this transformers version keeps it, plus tokenizer_config.json
        if f.name in ("chat_template.jinja", "chat_template.json"):
            (b / f.name).write_bytes(f.read_bytes())
    cfg = json.loads((b / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg["chat_template"] = tok.chat_template
    cfg["model_max_length"] = 123
    (b / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (b / "special_tokens_map.json").write_text(json.dumps({"eos_token": "<|im_end|>"}), encoding="utf-8")
    assert tokenizer_hash(a) != tokenizer_hash(b)
    d = tp.decide_pair(a, b)
    assert d["ok"] and d["level"] == "ok" and tp.template_hash(a) != tp.template_hash(b)


def test_real_tiny_tokenizers_cosmetic_tokenizer_json_changes_are_accepted(tmp_path):
    _, a = tiny_pair(tmp_path)
    b = copy_folder(a, tmp_path / "b")
    j = json.loads((b / "tokenizer.json").read_text(encoding="utf-8"))
    j["padding"], j["truncation"], j["version"] = {"strategy": "BatchLongest", "direction": "Left", "pad_to_multiple_of": None, "pad_id": 0, "pad_type_id": 0, "pad_token": "<|endoftext|>"}, None, "1.1"
    (b / "tokenizer.json").write_text(json.dumps(j), encoding="utf-8")
    assert tp.decide_pair(a, b)["level"] == "ok"


def test_real_tiny_tokenizers_a_different_vocabulary_is_refused_by_the_probe(tmp_path):
    _, a = tiny_pair(tmp_path)
    b = copy_folder(a, tmp_path / "b")
    j = json.loads((b / "tokenizer.json").read_text(encoding="utf-8"))
    v = j["model"]["vocab"]
    ka, kb = "a", "e"
    assert ka in v and kb in v
    v[ka], v[kb] = v[kb], v[ka]                                              # same size, two ids swapped
    (b / "tokenizer.json").write_text(json.dumps(j), encoding="utf-8")
    d = tp.decide_pair(a, b)
    assert not d["ok"] and d["probe"]["status"] == "different" and "First difference" in d["text"]
    assert tp.decide_pair(b, a)["ok"] is False
    assert copy.deepcopy(d)["probe"]["student_ids"] != d["probe"]["teacher_ids"]
