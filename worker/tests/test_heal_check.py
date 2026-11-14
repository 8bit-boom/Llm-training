"""The heal_check job: what the Heal section learns about a model from its files (no torch, no network)."""
import json
from pathlib import Path

import pytest

from stfile import fill, write_file
from test_worker import run

from ndforge_worker import healcheck as hc
from ndforge_worker.distilljob import tokenizer_hash

CFG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 4, "hidden_size": 8, "intermediate_size": 16, "vocab_size": 10,
       "num_attention_heads": 2}


def tok_json(words=("a", "b", "c")):
    return {"model": {"type": "BPE", "vocab": {w: i for i, w in enumerate(words)}, "merges": []}, "added_tokens": [{"id": 90, "content": "<|im_end|>"}]}


def model(path: Path, cfg=None, tokenizer=None, readme=None, shape=(10, 8), weights=True, gguf=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(CFG if cfg is None else cfg), encoding="utf-8")
    if tokenizer is not False:
        (path / "tokenizer.json").write_text(json.dumps(tokenizer or tok_json()), encoding="utf-8")
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if weights:
        n = shape[0] * shape[1]
        write_file(path / "model.safetensors", [("model.embed_tokens.weight", "BF16", shape, fill(1, n * 2))])
    if gguf:
        (path / "m.gguf").write_bytes(b"GGUF")
    if readme is not None:
        (path / "README.md").write_text(readme, encoding="utf-8")
    return path


def facts(path):
    return hc.facts_of(str(path))


def test_a_plain_folder_is_described_from_its_files(tmp_path):
    f = facts(model(tmp_path / "m", readme="---\nlicense: apache-2.0\n---\n# hi\n"))
    assert f["error"] is None and f["supported"] is True and f["kind"] == "folder" and f["on_disk"] is True
    assert (f["model_type"], f["layers"], f["hidden"], f["intermediate"], f["vocab_size"]) == ("qwen3", 4, 8, 16, 10)
    assert (f["params"], f["bytes"], f["quantized"], f["gguf"], f["license"]) == (80, 160, False, False, "apache-2.0")
    assert f["tokenizer_hash"] == tokenizer_hash(tmp_path / "m") and f["vocab_tokens"] == 4          # 3 words + 1 added token


def test_the_tokenizer_hash_tells_equal_from_different(tmp_path):
    a, b = facts(model(tmp_path / "a")), facts(model(tmp_path / "b"))
    c = facts(model(tmp_path / "c", tokenizer=tok_json(("a", "b", "c", "d"))))
    assert a["tokenizer_hash"] == b["tokenizer_hash"] != c["tokenizer_hash"] and (a["vocab_tokens"], c["vocab_tokens"]) == (4, 5)


def test_what_cannot_be_healed_is_said(tmp_path):
    q = facts(model(tmp_path / "q", cfg={**CFG, "quantization_config": {"quant_method": "bnb"}}))
    assert q["quantized"] is True and q["supported"] is False and "quantized" in q["why"]
    g = facts(model(tmp_path / "g", weights=False, gguf=True))
    assert g["gguf"] is True and g["supported"] is False and "GGUF" in g["why"]
    u = facts(model(tmp_path / "u", cfg={**CFG, "model_type": "gpt2"}))
    assert u["supported"] is False and "gpt2" in u["why"]
    gem = facts(model(tmp_path / "gem", cfg={**CFG, "model_type": "gemma4"}))
    assert gem["supported"] is False
    nw = facts(model(tmp_path / "nw", weights=False))
    assert nw["error"] and ".safetensors" in nw["error"]
    (tmp_path / "empty").mkdir()
    assert "config.json" in facts(tmp_path / "empty")["error"]


def test_a_multimodal_config_reads_its_text_part(tmp_path):
    f = facts(model(tmp_path / "v", cfg={"model_type": "qwen3_5", "vision_config": {"depth": 1}, "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 8, "hidden_size": 64, "vocab_size": 10}}))
    assert (f["layers"], f["hidden"], f["has_vision"], f["family"]) == (8, 64, True, "qwen3_5")


def test_the_guess_of_the_base_model(tmp_path):
    card = "---\nlicense: mit\nbase_model:\n  - acme/Big-Model\n---\n"
    f = facts(model(tmp_path / "a", readme=card))
    assert f["base_guess"] == {"repo": "acme/Big-Model", "from": "base_model in the model card"}
    one = facts(model(tmp_path / "b", readme="---\nbase_model: acme/One\n---\n"))
    assert one["base_guess"]["repo"] == "acme/One"
    inline = facts(model(tmp_path / "i", readme="---\nbase_model: [acme/Two, acme/Three]\n---\n"))
    assert inline["base_guess"]["repo"] == "acme/Two"
    by_cfg = facts(model(tmp_path / "c", cfg={**CFG, "_name_or_path": "acme/FromConfig"}))
    assert by_cfg["base_guess"] == {"repo": "acme/FromConfig", "from": "_name_or_path in config.json"}
    local_path = facts(model(tmp_path / "d", cfg={**CFG, "_name_or_path": str(tmp_path / "d")}))        # its own folder is no guess
    assert local_path["base_guess"] is None and facts(model(tmp_path / "e"))["base_guess"] is None
    not_an_id = facts(model(tmp_path / "f", readme="---\nbase_model: a local thing\n---\n"))
    assert not_an_id["base_guess"] is None


def test_a_name_that_is_neither_a_folder_nor_a_hub_id_is_an_error_not_a_crash():
    f = hc.facts_of("not a model at all")
    assert f["error"] and "Hugging Face name" in f["error"]


def test_the_job_answers_both_sides(tmp_path):
    s, t = model(tmp_path / "s"), model(tmp_path / "t")
    code, ev = run(tmp_path, {"job": "heal_check", "sources": {"student": str(s), "teacher": str(t)}})
    assert code == 0 and ev[-1]["type"] == "result"
    data = ev[-1]["data"]
    assert set(data) == {"student", "teacher"} and data["student"]["tokenizer_hash"] == data["teacher"]["tokenizer_hash"]
    code, ev = run(tmp_path, {"job": "heal_check", "sources": {"student": str(s)}})
    assert code == 0 and set(ev[-1]["data"]) == {"student"}
