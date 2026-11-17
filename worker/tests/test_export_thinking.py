"""The exported chat template: answers without thinking by default (thinking_default "off", as before) or stays as the model ships it ("keep"). No torch."""
import json
from pathlib import Path

import pytest
from jinja2 import Environment

from ndforge_worker import exportjob

TPL = (Path(__file__).parent / "data" / "qwen3_5_chat_template.jinja").read_text(encoding="utf-8")


def make(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(TPL, encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": TPL, "other": 1}), encoding="utf-8")
    return tmp_path


def texts(folder):
    f, c = folder / "chat_template.jinja", folder / "tokenizer_config.json"
    return (f.read_text(encoding="utf-8") if f.exists() else None), (json.loads(c.read_text())["chat_template"] if c.exists() else None)


def ends(t, **kw):
    return Environment().from_string(t).render(messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True, **kw)


def test_off_is_the_default_and_changes_the_template(tmp_path):
    exportjob.patch_template_dir(make(tmp_path), "qwen3_5")
    for t in texts(tmp_path):
        assert t != TPL and ends(t).endswith("<think>\n\n</think>\n\n") and ends(t, enable_thinking=True).endswith("<think>\n")


def test_keep_leaves_the_template_untouched(tmp_path):
    exportjob.patch_template_dir(make(tmp_path), "qwen3_5", "keep")
    assert texts(tmp_path) == (TPL, TPL)
    assert json.loads((tmp_path / "tokenizer_config.json").read_text())["other"] == 1
    assert ends(TPL).endswith("<|im_start|>assistant\n<think>\n")             # the model thinks unless the client switches it off


def test_other_families_are_never_touched(tmp_path):
    exportjob.patch_template_dir(make(tmp_path), "gemma4")
    assert texts(tmp_path) == (TPL, TPL)


@pytest.mark.parametrize("given,expect", [({}, "off"), ({"thinking_default": "keep"}, "keep"), ({"thinking_default": "off"}, "off"),
                                          ({"thinking_default": "maybe"}, "off"), ({"thinking_default": None}, "off")])
def test_the_job_key(given, expect):
    assert exportjob.thinking_default_of(given) == expect
