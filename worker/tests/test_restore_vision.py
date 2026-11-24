"""Restore vision: regenerate the mmproj of an export whose model has an intact vision tower but no
projector (the recorded origin can be a deleted intermediate cut). Converts from the model's own
folder; a text-only model is refused in plain words."""
import json
from pathlib import Path

from ndforge_worker import exportjob


def test_restore_vision_converts_from_the_models_own_folder(tmp_path, monkeypatch):
    base = tmp_path / "model"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"], "vision_config": {}}), encoding="utf-8")
    out = tmp_path / "export"
    seen = {}

    def fake_convert(conv, source, final, label="x", extra=()):
        seen["source"] = source
        seen["extra"] = list(extra)
        Path(final).write_bytes(b"MMPROJ")

    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: ("/fake/convert.py", None))
    monkeypatch.setattr(exportjob, "convert_to", fake_convert)
    rc = exportjob.restore_vision({"job": "restore_vision", "base": {"repo": str(base), "family": "qwen3_5", "vision_repo": str(base)}, "out": str(out)})
    assert rc == 0
    assert seen["source"] == str(base) and "--mmproj" in seen["extra"]
    assert (out / "mmproj-F16.gguf").is_file()


def test_restore_vision_refuses_a_text_only_model(tmp_path, monkeypatch):
    base = tmp_path / "text"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}), encoding="utf-8")
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: ("/fake/convert.py", None))
    rc = exportjob.restore_vision({"job": "restore_vision", "base": {"repo": str(base)}, "out": str(tmp_path / "o")})
    assert rc == 1  # no_vision error, nothing converted
