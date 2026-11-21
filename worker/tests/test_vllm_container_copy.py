"""How files get into and out of the vLLM container (bind mount, cp in its variants, stdin), the self-explaining failures, the early probe, and that the container is always removed.
The real `wslc cp` is only proven by the owner's run of "Test the container copy"; these tests use a fake that implements the cp rules that were suspected (directory-only destination, drive-letter confusion)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from ndforge_worker import distilljob as dj, vllmcache  # noqa: E402
from test_vllm_cache_job import FakeWorld, cache_job, make_rows, new_writer, run_it  # noqa: E402


def job(kind="docker"):
    return {"engine": {"kind": "vllm", "teacher": {}, "container": {"kind": kind}}}


def test_the_bind_mount_is_used_first_and_no_cp_runs(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="ok")
    r = vllmcache.probe(job())
    assert r["ok"] and r["route_in"] == "bind" and r["route_out"] == "bind" and r["bind"]
    assert not [c for c in world.calls if c[:2] == ["docker", "cp"]], "no copy command at all"
    assert not world.running, "the container is removed"


def test_a_refused_bind_falls_back_to_cp_and_says_so(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse")
    r = vllmcache.probe(job())
    assert r["ok"] and r["route_in"] == "cp:file" and r["route_out"] == "cp:file" and r["bind"] is None
    assert any(not t["ok"] and t["route"].startswith("bind") for t in r["tried"]) and "mounted folder was refused" in r["text"]
    assert not world.running


def test_a_bind_that_the_container_does_not_see_is_not_trusted(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="invisible")
    r = vllmcache.probe(job())
    assert r["ok"] and r["bind"] is None and r["route_in"].startswith("cp"), r
    assert any(t["route"].startswith("bind") and not t["ok"] for t in r["tried"])
    assert not world.running


def test_a_directory_only_cp_destination_uses_the_directory_form(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="dir_only")
    r = vllmcache.probe(job())
    assert r["ok"], r["text"]
    assert r["route_in"] in ("cp:dir", "cp:touch"), r["route_in"]
    cps = [c for c in world.calls if c[:2] == ["docker", "cp"]]
    assert any(c[-1].endswith(":/work/") for c in cps) or r["route_in"] == "cp:touch"
    assert not world.running


def test_a_drive_letter_source_is_tried_with_forward_slashes(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="drive_letter")
    monkeypatch.setattr(vllmcache.Container, "host", lambda self, p: "C:\\fakehost\\" + str(p).lstrip("/").replace("/", "\\"))
    r = vllmcache.probe(job())
    assert r["route_in"] == "cp:fwd" and r["route_out"] == "cp:fwd", r
    assert any("no such container" in t["detail"] for t in r["tried"] if not t["ok"])


def test_when_every_cp_fails_the_file_is_streamed_through_exec(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="none")
    r = vllmcache.probe(job())
    assert r["ok"] and r["route_in"] == "stdin" and r["route_out"] == "stdin", r
    assert any(c[:3] == ["docker", "exec", "-i"] for c in world.calls)
    assert not world.running


def test_nothing_works_is_a_diagnosis_with_the_remedy_and_the_container_is_removed(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="none", stdin_ok=False)
    r = vllmcache.probe(job())
    assert not r["ok"]
    assert "Tried:" in r["text"] and "cp:file" in r["text"] and "stdin" in r["text"] and "switch the vLLM teacher cache off" in r["text"] and "Test the container copy" in r["text"]
    assert not world.running
    assert [c[1] for c in world.calls if c[1] in ("stop", "rm")][-2:] == ["stop", "rm"]


def test_a_copy_failure_names_the_command_the_error_and_looks_inside(tmp_path, monkeypatch):
    logs: list = []
    monkeypatch.setattr(vllmcache.protocol, "log", logs.append)
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10, 10])
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="docker")
    orig = world.run

    def run(cmd, **kw):
        if cmd[1] == "cp" and cmd[-1].endswith(":/work/vllm_cache.py"):
            world.calls.append(list(cmd))
            import types
            return types.SimpleNamespace(returncode=1, stdout=b"", stderr=b"Could not find the file /work/vllm_cache.py in container. Error code: ERROR_PATH_NOT_FOUND")
        return orig(cmd, **kw)

    monkeypatch.setattr(vllmcache.vllmengine, "_run", run)
    with pytest.raises(vllmcache.VllmCacheError) as e:
        run_it(tmp_path, world, new_writer(tmp_path, 2), rows, 0, 2)
    msg = str(e.value)
    assert "Copying vllm_cache.py into the container: `docker cp " in msg and ":/work/vllm_cache.py` failed (exit code 1)" in msg and "ERROR_PATH_NOT_FOUND" in msg
    assert "Checked right then: ls -la /work" in msg
    text = "\n".join(logs)
    assert "Diagnostics, ls -la /work" in text and "Diagnostics, mounts" in text
    assert not world.running, "the container is removed after the failure"


def test_a_required_vllm_fails_early_with_the_diagnosis_before_the_rows_exist(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="none", stdin_ok=False)
    cfg = {"engine": {"kind": "vllm", "required": True, "teacher": {}, "container": {"kind": "docker"}}, "teacher": {"repo": "Someone/Finetune"}}
    with pytest.raises(dj.ReshapeError) as e:
        dj._early_vllm_probe(cfg)                       # run_cache calls this before it reads the model and makes the rows
    assert "container cannot be used" in str(e.value) and "switch the vLLM teacher cache off" in str(e.value)
    assert not world.running


def test_the_fp8_teacher_cache_logs_a_plain_warning(tmp_path, monkeypatch):
    logs: list = []
    monkeypatch.setattr(vllmcache.protocol, "log", logs.append)
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10])
    world = FakeWorld(tmp_path, monkeypatch, bind="ok")
    j = cache_job(tmp_path, kv_cache_dtype="fp8", kv_cache_dtype_teacher=True)
    run_it(tmp_path, world, new_writer(tmp_path, 1), rows, 0, 1, job=j)
    assert any("no speed gain" in x for x in logs)


def test_a_full_chunk_works_through_the_bind_mount_and_through_stdin(tmp_path, monkeypatch):
    for mode, kw in (("bind", dict(bind="ok")), ("stdin", dict(bind="refuse", cp="none"))):
        sub = tmp_path / mode
        sub.mkdir()
        rows = sub / "rows"
        rows.mkdir()
        make_rows(rows, [12, 30, 7])
        world = FakeWorld(sub, monkeypatch, **kw)
        w = new_writer(sub, 3)
        r = run_it(sub, world, w, rows, 0, 3)
        w.close()
        assert r["rows"] == 3, mode
        assert not world.running


def test_wslc_with_a_directory_only_cp_and_a_refused_bind_is_found_by_the_probe(tmp_path, monkeypatch):
    world = FakeWorld(tmp_path, monkeypatch, bind="refuse", cp="dir_only", exe="wslc")
    r = vllmcache.probe(job("wslc"))
    assert r["ok"] and r["route_in"] in ("cp:dir", "cp:touch") and r["bind"] is None, r
    assert any(c[:2] == ["wslc", "cp"] for c in world.calls)
    assert not world.running and any(c[:2] == ["wslc", "remove"] for c in world.calls)


def test_a_verbatim_windows_path_is_given_to_the_engine_in_the_plain_form():
    from ndforge_worker import vllmengine as ve
    assert ve.plain_path("\\\\?\\C:\\Users\\me\\m") == "C:\\Users\\me\\m" and ve.plain_path("\\\\?\\UNC\\srv\\sh\\m") == "\\\\srv\\sh\\m" and ve.plain_path("/tmp/x") == "/tmp/x"
    assert ve.wsl_path("\\\\?\\C:\\Users\\me\\m") == "/mnt/c/Users/me/m"
