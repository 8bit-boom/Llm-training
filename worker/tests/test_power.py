"""Keeping the computer awake while a job runs: a no-op off Windows, the right flags on Windows (a fake kernel32), switches, and no failure ever reaches the job."""
import sys
import types

import pytest

from ndforge_worker import power, protocol


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("NDFORGE_KEEP_AWAKE", raising=False)
    monkeypatch.delenv("NDFORGE_KEEP_DISPLAY", raising=False)
    monkeypatch.setattr(power, "_warned", [False])
    logs = []
    monkeypatch.setattr(protocol, "log", lambda m, *a, **k: logs.append(m))
    return logs


def fake_windows(monkeypatch, result=1, boom=None):
    calls = []

    def state(flags):
        calls.append(flags)
        if boom:
            raise boom
        return result

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(power, "_set", state)
    return calls


def test_it_does_nothing_off_windows(clean):
    with power.keep_awake("train", {}) as active:
        assert active is False
    assert clean == []


def test_windows_sets_the_flags_and_resets_on_exit(monkeypatch, clean):
    calls = fake_windows(monkeypatch)
    with power.keep_awake("train", {}) as active:
        assert active is True and calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED]
    assert calls[-1] == power.ES_CONTINUOUS and len(calls) == 2
    assert clean == ["Keeping this computer awake while the job runs."]


def test_it_resets_when_the_body_raises(monkeypatch):
    calls = fake_windows(monkeypatch)
    with pytest.raises(RuntimeError):
        with power.keep_awake("distill", {}):
            raise RuntimeError("boom")
    assert calls[-1] == power.ES_CONTINUOUS


def test_the_display_stays_on_only_when_asked(monkeypatch):
    calls = fake_windows(monkeypatch)
    monkeypatch.setenv("NDFORGE_KEEP_DISPLAY", "1")
    with power.keep_awake("train", {}):
        pass
    assert calls[0] == power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED | power.ES_DISPLAY_REQUIRED


def test_the_environment_and_the_job_can_switch_it_off(monkeypatch):
    calls = fake_windows(monkeypatch)
    with power.keep_awake("train", {"keep_awake": False}) as a:
        assert a is False
    monkeypatch.setenv("NDFORGE_KEEP_AWAKE", "0")
    with power.keep_awake("train", {}) as a:
        assert a is False
    assert calls == []


def test_short_jobs_are_not_wrapped(monkeypatch):
    calls = fake_windows(monkeypatch)
    for kind in ("hello", "reshape_inspect", "think_check_plan", "reshape_plan", "general_check", "imatrix_published", "imatrix_adapt_preview", "heal_check"):
        with power.keep_awake(kind, {}):
            pass
    assert calls == []
    for kind in ("auto", "heal", "export", "train", "think_check", "reshape", "reshape_try", "distill", "imatrix_make", "quantcheck"):
        assert not power.is_short(kind)


def test_a_failing_api_is_swallowed_and_logged_once(monkeypatch, clean):
    fake_windows(monkeypatch, boom=OSError("no kernel32"))
    for _ in range(2):
        with power.keep_awake("train", {}) as active:
            assert active is False
    assert len(clean) == 1 and "Could not ask Windows" in clean[0]


def test_a_refused_request_is_logged_and_the_job_goes_on(monkeypatch, clean):
    fake_windows(monkeypatch, result=0)
    with power.keep_awake("train", {}) as active:
        assert active is False
    assert len(clean) == 1


def test_the_real_call_goes_through_ctypes(monkeypatch):
    seen = []
    k32 = types.SimpleNamespace(SetThreadExecutionState=lambda f: seen.append(f) or 1)
    fake_ctypes = types.SimpleNamespace(windll=types.SimpleNamespace(kernel32=k32))
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    assert power._set(power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED) == 1 and seen == [0x80000001]
