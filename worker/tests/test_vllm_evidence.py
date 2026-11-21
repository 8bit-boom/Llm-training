"""The root cause of a vLLM failure is kept (the container is removed afterwards), and the card's free memory is checked and logged before an engine starts."""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from ndforge_worker import vllmcache, vllmengine as ve  # noqa: E402
from test_vllm_cache_job import FakeWorld, make_rows, new_writer, run_it  # noqa: E402

LOG = """INFO 11-21 12:00:01 [__init__.py:216] Automatically detected platform cuda.
(EngineCore_DP0 pid=212) INFO 11-21 12:00:09 [core.py:77] Initializing a V1 LLM engine
Loading safetensors checkpoint shards:  50% 3/6 [00:20<00:20,  6.7s/it]
(EngineCore_DP0 pid=212) ERROR 11-21 12:02:51 [core.py:700] Traceback (most recent call last):
(EngineCore_DP0 pid=212) ERROR 11-21 12:02:51 [core.py:700] ValueError: Free memory on device (15.1/31.8 GiB) on startup is less than desired GPU memory utilization (0.88, 28.0 GiB).
(EngineCore_DP0 pid=212) ERROR 11-21 12:02:51 [core.py:700] ValueError: Free memory on device (15.1/31.8 GiB) on startup is less than desired GPU memory utilization (0.88, 28.0 GiB).
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""


def test_the_root_cause_lines_are_picked_out_of_a_noisy_log():
    rc = ve.root_cause(LOG)
    assert any("Free memory on device (15.1/31.8 GiB)" in x for x in rc) and any("Engine core initialization failed" in x for x in rc) and any("Traceback" in x for x in rc)
    assert not any("it/s" in x or "s/it" in x or "Automatically detected" in x for x in rc), "no progress bars or normal INFO lines"
    assert len([x for x in rc if "Free memory" in x]) == 1, "a repeated line once"
    assert len(ve.root_cause("\n".join(f"ValueError: {i}" for i in range(100)))) == 25


def test_a_failing_script_puts_its_root_cause_into_the_error_and_the_job_log_and_the_container_is_removed(tmp_path, monkeypatch):
    logs: list = []
    monkeypatch.setattr(vllmcache.protocol, "log", logs.append)
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10, 10])
    world = FakeWorld(tmp_path, monkeypatch, bind="ok")

    def popen(cmd, **kw):
        world.calls.append(list(cmd))
        lines = [x + "\n" for x in LOG.splitlines()] + [json.dumps({"type": "error", "message": "RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}"}) + "\n"]
        return types.SimpleNamespace(stdout=iter(lines), returncode=1, wait=lambda t=None: 1, poll=lambda: 1, terminate=lambda: None)

    monkeypatch.setattr(ve, "_popen", popen)
    with pytest.raises(vllmcache.VllmCacheError) as e:
        run_it(tmp_path, world, new_writer(tmp_path, 2), rows, 0, 2)
    msg = str(e.value)
    assert "Engine core initialization failed" in msg and "Root cause:" in msg and "Free memory on device (15.1/31.8 GiB)" in msg
    assert any("Root cause lines" in x and "Free memory on device" in x for x in logs), "the job log has them too"
    assert not world.running


def test_a_script_that_printed_nothing_says_so(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10])
    world = FakeWorld(tmp_path, monkeypatch, bind="ok")
    monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: types.SimpleNamespace(stdout=iter([]), returncode=137, wait=lambda t=None: 137, poll=lambda: 137, terminate=lambda: None))
    with pytest.raises(vllmcache.VllmCacheError, match="exit code 137.*No line of the engine's output names a cause.*The script printed nothing"):
        run_it(tmp_path, world, new_writer(tmp_path, 1), rows, 0, 1)


def smi(total_mb, used_mb):
    return lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout=f"{total_mb}, {used_mb}\n".encode(), stderr=b"")


def test_the_free_memory_before_the_engine_start_is_logged_with_the_arithmetic(monkeypatch):
    logs: list = []
    monkeypatch.setattr(vllmcache.protocol, "log", logs.append)
    monkeypatch.setattr(ve, "_smi_run", smi(32563, 2000))
    total, used = ve.gpu_mb()
    g = ve.plan_gpu_util(total, used, 0.88)
    vllmcache._log_gpu_before_start(g, total, used)
    assert any(x.startswith("GPU free before the engine start: 29.8 GB of 31.8 GB; the engine takes 28.0 GB (0.88 of the card)") for x in logs), logs


def test_a_card_that_is_mostly_taken_refuses_before_any_container_starts(tmp_path, monkeypatch):
    rows = tmp_path / "rows"
    rows.mkdir()
    make_rows(rows, [10])
    world = FakeWorld(tmp_path, monkeypatch, bind="ok")
    monkeypatch.setattr(ve, "_smi_run", smi(32563, 17000))           # 15 GB free: the HF model of an earlier step is still there
    with pytest.raises(vllmcache.VllmCacheError, match="in use by other programs"):
        run_it(tmp_path, world, new_writer(tmp_path, 1), rows, 0, 1)
    assert not [c for c in world.calls if c[:2] == ["docker", "run"]], "nothing was started"


def test_the_4_bit_route_is_refused_early_when_the_original_cannot_fit(tmp_path, monkeypatch):
    d = tmp_path / "orig"
    d.mkdir()
    header = {"model.layers.0.mlp.weight": {"dtype": "BF16", "shape": [400000, 200000], "data_offsets": [0, 1]}}
    raw = json.dumps(header).encode()
    (d / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
    g = {"util": 0.88, "text": "fine"}
    with pytest.raises(vllmcache.VllmCacheError, match=r"does not fit in 4 bits.*weights 42\.4 \+ overhead 1\.5 \+ minimum cache 2\.5 = 46\.4 GB; budget 29\.2 GB"):
        vllmcache._check_bnb_fit(d, g, 32563, 2000)
    small = tmp_path / "small"
    small.mkdir()
    header = {"model.layers.0.mlp.weight": {"dtype": "BF16", "shape": [40000, 200000], "data_offsets": [0, 1]}}
    raw = json.dumps(header).encode()
    (small / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
    vllmcache._check_bnb_fit(small, g, 32563, 2000)                    # fits: no error
