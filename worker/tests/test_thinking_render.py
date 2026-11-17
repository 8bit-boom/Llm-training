"""Thinking-aware rendering of chat rows (data.render_chat_full) with the REAL Qwen3.5 chat template (jinja only, no torch), and the general-text key."""
import json
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment

from ndforge_worker import data, distilljob
from ndforge_worker import generaltext as gt

DATA = Path(__file__).parent / "data"


class JinjaTok:
    """A tokenizer that only has the chat template: the real Qwen3.5 one, rendered the way transformers renders it."""
    eos_token = "<|im_end|>"

    def __init__(self):
        def raise_exception(msg):
            raise ValueError(msg)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        env.globals["raise_exception"] = raise_exception
        self.tpl = env.from_string((DATA / "qwen3_5_chat_template.jinja").read_text(encoding="utf-8"))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        assert not tokenize
        return self.tpl.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)


TOK = JinjaTok()
Q = {"role": "user", "content": "What is 6 times 7?"}
THINK = {"role": "assistant", "content": "<think>\nSix sevens: 42.\n</think>\n\nIt is 42."}
PLAIN = {"role": "assistant", "content": "It is 42."}


def test_a_thinking_row_has_one_block_and_the_real_prefix():
    prompt, full = data.render_chat_full(TOK, [Q, THINK], "qwen3_5")
    assert "</think>\n\n<think>" not in full and "<think>\n\n</think>" not in full and full.count("<think>") == 1
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert full.endswith("<|im_start|>assistant\n<think>\nSix sevens: 42.\n</think>\n\nIt is 42.<|im_end|>\n")
    assert full.startswith(prompt)
    # what render_chat did (the bug): the empty block of thinking-off in front of the trace
    _, old = data.render_chat(TOK, [Q, THINK], "qwen3_5")
    assert "<think>\n\n</think>\n\n<think>" in old


def test_reasoning_content_gives_the_same_row_as_the_inline_trace():
    sep = {"role": "assistant", "content": "It is 42.", "reasoning_content": "Six sevens: 42."}
    assert data.has_thinking([Q, sep]) and data.has_thinking([Q, THINK]) and not data.has_thinking([Q, PLAIN])
    assert data.render_chat_full(TOK, [Q, sep], "qwen3_5") == data.render_chat_full(TOK, [Q, THINK], "qwen3_5")


def test_a_row_without_thinking_is_rendered_as_before():
    assert data.render_chat_full(TOK, [Q, PLAIN], "qwen3_5") == data.render_chat(TOK, [Q, PLAIN], "qwen3_5")
    prompt, full = data.render_chat_full(TOK, [Q, PLAIN], "qwen3_5")
    assert prompt.endswith("assistant\n<think>\n\n</think>\n\n") and full.endswith("It is 42.<|im_end|>\n")
    assert not data.has_thinking([Q]) and not data.has_thinking([])
    assert not data.has_thinking([Q, {"role": "assistant", "content": "<think>\n\n</think>\n\nIt is 42."}])        # an empty block is no thinking


def test_an_empty_block_in_the_answer_is_not_written_twice():
    empty = {"role": "assistant", "content": "<think>\n\n</think>\n\nIt is 42."}
    _, full = data.render_chat_full(TOK, [Q, empty], "qwen3_5")
    assert full.count("<think>") == 1 and full.endswith("assistant\n<think>\n\n</think>\n\nIt is 42.<|im_end|>\n")


def test_earlier_turns_and_a_system_turn_keep_the_frame():
    msgs = [{"role": "system", "content": "Be brief."}, Q, THINK, {"role": "user", "content": "And 6 times 8?"},
            {"role": "assistant", "content": "<think>\nSix eights: 48.\n</think>\n\n48."}]
    prompt, full = data.render_chat_full(TOK, msgs, "qwen3_5")
    assert full.startswith("<|im_start|>system\nBe brief.<|im_end|>\n") and full.startswith(prompt)
    assert full.endswith("<|im_start|>assistant\n<think>\nSix eights: 48.\n</think>\n\n48.<|im_end|>\n")
    assert "Six sevens" not in full                      # the template drops the trace of a turn before the last question


def test_other_families_are_not_touched():
    class T(JinjaTok):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
            return "".join(f"[{m['role']}]{m['content']}\n" for m in messages) + ("[assistant]" if add_generation_prompt else "")
    assert data.render_chat_full(T(), [Q, THINK], "generic") == data.render_chat(T(), [Q, THINK], "generic")


def test_the_general_text_hook_renders_thinking_rows_right(monkeypatch, tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text(json.dumps({"messages": [Q, THINK]}) + "\n" + json.dumps({"messages": [Q, PLAIN]}) + "\n", encoding="utf-8")
    texts = list(gt.texts({"file": str(f), "render": lambda m: data.render_chat_full(TOK, m, "qwen3_5")[1]}))
    assert len(texts) == 2 and "</think>\n\n<think>" not in texts[0] and texts[0].count("<think>") == 1
    assert texts[1].endswith("assistant\n<think>\n\n</think>\n\nIt is 42.<|im_end|>\n")
    # the reasoning field of a dataset row ends up in the same shape
    f.write_text(json.dumps({"messages": [Q, {"role": "assistant", "content": "It is 42.", "reasoning_content": "Six sevens: 42."}]}) + "\n", encoding="utf-8")
    t2 = list(gt.texts({"file": str(f), "render": lambda m: data.render_chat_full(TOK, m, "qwen3_5")[1]}))
    assert t2 == [texts[0]]


def test_general_key_matches_the_shared_cases_and_an_old_key_is_other_data():
    t = json.loads((DATA / "general_key_cases.json").read_text())
    assert t["render_version"] == gt.RENDER_VERSION
    for c in t["cases"]:
        g = c["general"]
        assert gt.general_key(g, (g or {}).get("share", 0), c["seed"]) == c["expect"], c["name"]
    src = {"id": "a/b"}
    old = {k: v for k, v in gt.general_key(src, 0.5, 3407).items() if k != "render"}
    assert not gt.same_general(old, src, 0.5) and gt.same_general(gt.general_key(src, 0.5, 3407), src, 0.5)
    meta = {"teacher": "t", "k": 8, "rows": 1, "data_hash": "h", "tokenizer_hash": "x", "max_len": 64, "general_rows": 1, "general_key": old}
    assert not distilljob.cache_matches(meta, {**meta, "general_key": gt.general_key(src, 0.5, 3407)})


def test_the_importance_matrix_key_is_not_changed_by_the_rendering():
    from ndforge_worker import imatrix
    k = imatrix.settings_key({}, {"general_share": 0.2, "general": {"id": "a/b"}}, "dh")
    assert "render" not in k["general"] and k["general"]["id"] == "a/b"
