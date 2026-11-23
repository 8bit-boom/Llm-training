"""The export without a training run: when a job names neither `full_model` nor `adapter`, the model of `base.repo` is used — a folder on this computer as it is,
the copy in the Hugging Face cache when there is one, otherwise it is downloaded there first. A repo without 16-bit safetensors weights is refused in plain words."""
import json
from pathlib import Path

import pytest

from ndforge_worker import exportjob


def make_cfg(repo):
    return {"job": "export", "base": {"repo": repo, "family": "qwen3_5"}, "out": "/tmp/unused", "quant": "q4_k_m"}


def fake_hub(monkeypatch, calls, cached=None, downloaded=None):
    """A stand-in for huggingface_hub.snapshot_download: `cached` answers the cache-only ask, `downloaded` the real download (None: it raises)."""
    import huggingface_hub

    def snapshot(repo, **kw):
        calls.append(kw)
        if kw.get("local_files_only"):
            if cached is None:
                raise RuntimeError("not cached")
            return cached
        if downloaded is None:
            raise RuntimeError("offline")
        return downloaded

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)


def test_a_local_folder_is_used_as_it_is(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert exportjob.resolve_full_model(make_cfg(str(tmp_path))) == str(tmp_path)


def test_a_cached_copy_is_used_without_downloading(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}", encoding="utf-8")
    calls = []
    fake_hub(monkeypatch, calls, cached=str(snap))
    assert exportjob.resolve_full_model(make_cfg("org/model")) == str(snap)
    assert calls == [{"local_files_only": True}], "the cache is asked once, nothing is downloaded"


def test_a_model_that_is_not_here_is_downloaded_into_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    (cache / "config.json").write_text("{}", encoding="utf-8")
    (cache / "model-00001-of-00002.safetensors").write_bytes(b"x")
    calls = []
    fake_hub(monkeypatch, calls, downloaded=str(cache))
    assert exportjob.resolve_full_model(make_cfg("org/model")) == str(cache)
    assert [c.get("local_files_only") for c in calls] == [True, None], "the cache is asked first, the download comes after it"


def test_a_gguf_only_repo_is_refused_in_plain_words(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    (cache / "config.json").write_text("{}", encoding="utf-8")
    (cache / "model-Q4_K_M.gguf").write_bytes(b"x")
    calls = []
    fake_hub(monkeypatch, calls, downloaded=str(cache))
    with pytest.raises(ValueError, match="safetensors"):
        exportjob.resolve_full_model(make_cfg("org/gguf-only"))


def test_no_config_json_after_download_is_refused_too(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    (cache / "weights.safetensors").write_bytes(b"x")
    calls = []
    fake_hub(monkeypatch, calls, downloaded=str(cache))
    with pytest.raises(ValueError, match="config.json"):
        exportjob.resolve_full_model(make_cfg("org/odd"))


def test_a_cached_copy_without_config_json_is_downloaded_again(tmp_path, monkeypatch):
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "README.md").write_text("nothing useful", encoding="utf-8")
    cache = tmp_path / "fresh"
    cache.mkdir()
    (cache / "config.json").write_text("{}", encoding="utf-8")
    (cache / "model.safetensors").write_bytes(b"x")
    calls = []
    fake_hub(monkeypatch, calls, cached=str(stale), downloaded=str(cache))
    assert exportjob.resolve_full_model(make_cfg("org/model")) == str(cache)


def test_a_made_model_says_its_image_ability_may_be_weaker():
    """A base made by "Slim or grow a model" carries the original as `vision_repo`; the vision note then says what healing did (and did not) change."""
    from ndforge_worker import vision

    made = {"base": {"repo": "/home/me/models/mycut", "vision_repo": "unsloth/Qwen3.8-27B"}}
    note = vision.made_model_note(made)
    assert note and "slimmed" in note and "unchanged" in note
    # a normal base model (or one whose vision file is its own) gets no note
    assert vision.made_model_note({"base": {"repo": "unsloth/Qwen3.5-4B"}}) is None
    assert vision.made_model_note({"base": {"repo": "unsloth/Qwen3.8-27B", "vision_repo": "unsloth/Qwen3.8-27B"}}) is None
    assert vision.made_model_note({}) is None
