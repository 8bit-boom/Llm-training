import os

import pytest

from ndforge_worker import fsutil


def test_replace_retry_replaces_and_tries_again_on_permission_errors(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("1")
    b.write_text("2")
    fsutil.replace_retry(a, b)
    assert b.read_text() == "1" and not a.exists()
    real, calls = os.replace, []

    def flaky(s, d):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "in use")
        return real(s, d)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    a.write_text("3")
    fsutil.replace_retry(a, b, tries=5, pause=0.0)
    assert len(calls) == 3 and b.read_text() == "3"
    monkeypatch.setattr(fsutil.os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError(13, "locked")))
    with pytest.raises(PermissionError):
        fsutil.replace_retry(a, b, tries=2, pause=0.0)
    monkeypatch.setattr(fsutil.os, "replace", lambda s, d: (_ for _ in ()).throw(FileNotFoundError(2, "gone")))
    with pytest.raises(FileNotFoundError):
        fsutil.replace_retry(a, b, tries=5, pause=0.0)


def test_partial_names_and_clean_up(tmp_path):
    assert fsutil.partial_of(tmp_path / "model-q4_k_m.gguf").name == "model-q4_k_m.gguf.partial"
    (tmp_path / "x.gguf.partial").write_bytes(b"1")
    (tmp_path / "keep.gguf").write_bytes(b"1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.gguf.partial").write_bytes(b"1")
    assert fsutil.remove_partials(tmp_path) == ["x.gguf.partial"]
    assert (tmp_path / "keep.gguf").exists() and (tmp_path / "sub" / "y.gguf.partial").exists()
    assert fsutil.remove_partials(tmp_path / "nowhere") == []
