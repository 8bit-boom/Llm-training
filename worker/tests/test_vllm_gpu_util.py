"""How much of the card the engine may take when the desktop already uses some: no change / lowered / refused / unknown."""
import types

import pytest

from ndforge_worker import vllmengine as ve, vllmselftest as st


def test_plenty_of_free_memory_changes_nothing():
    g = ve.plan_gpu_util(32607, 1931, 0.88)
    assert g["ok"] is True and g["lowered"] is False and g["util"] == 0.88


def test_2713_mb_in_use_on_a_32_gb_card_passes_with_the_utilization_lowered():
    g = ve.plan_gpu_util(32607, 2713, 0.88)          # free 29.9 GB, needed 28.0 GB + 0.5: exactly the owner's card
    assert g["ok"] is True
    assert 0.70 <= g["util"] <= 0.88 and g["text"]
    t = ve.plan_gpu_util(32607, 4500, 0.88)
    assert t["ok"] and t["lowered"] and t["util"] < 0.88 and "instead of 0.88" in t["text"] and "vLLM will take" in t["text"]
    assert t["util"] * 32607 <= 32607 - 4500 - 500, "what the engine reserves fits next to the other programs"


def test_8_gb_in_use_is_lowered_to_about_the_floor_and_12_gb_is_refused():
    a = ve.plan_gpu_util(32607, 8192, 0.88)
    assert a["ok"] and a["util"] >= 0.70 and a["lowered"]
    b = ve.plan_gpu_util(32607, 12288, 0.88)
    assert b["ok"] is False and "12.0 GB of the card" in b["text"] and "Close those programs" in b["text"]


def test_an_unreadable_card_continues_with_a_note():
    for args in [(None, None), (32607, None), (0, 100)]:
        g = ve.plan_gpu_util(*args, 0.88)
        assert g["ok"] is None and g["util"] == 0.88 and "could not be read" in g["text"]


def test_gpu_mb_reads_nvidia_smi_and_survives_a_missing_one(monkeypatch):
    monkeypatch.setattr(ve, "_smi_run", lambda *a, **k: types.SimpleNamespace(stdout=b"32607, 2713\n", stderr=b""))
    assert ve.gpu_mb() == (32607, 2713)
    monkeypatch.setattr(ve, "_smi_run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert ve.gpu_mb() == (None, None)


def test_the_selftest_prerequisite_uses_the_rule(monkeypatch):
    def run(cmd, timeout=600):
        out = "581.42, 2713, 32607" if "driver_version" in " ".join(cmd) else ""
        return {"cmd": cmd, "returncode": 0, "output": out}
    monkeypatch.setattr(st, "_run", run)
    pre = st.prerequisites(ve.engine_config({"engine": {"kind": "docker"}}), "org/m", gpu_test=False)
    g = next(p for p in pre if p["id"] == "gpu_free")
    assert g["ok"] is True and g["gpu_util"] <= 0.88


def test_wslc_running_tolerates_unknown_shapes():
    n = "ndforge-vllm-5"
    for text in ['[{"Name": "ndforge-vllm-5", "State": "running"}]', '{"containers": [{"id": "ndforge-vllm-5"}]}', "NAMES  STATUS\nndforge-vllm-5  Up 3 seconds", '[{"names": ["ndforge-vllm-5"], "x": 1}]']:
        assert ve.wslc_running(text, n) is True, text
    for text in ['[{"Name": "ndforge-vllm-5", "State": "exited"}]', "ndforge-vllm-5  Exited (1)", "[]", '[{"Name": "other"}]']:
        assert ve.wslc_running(text, n) is False, text
    assert ve.wslc_running("", n) is None and ve.wslc_running("garbage", n) is None


def test_captured_help_is_tidied_and_wslc_templates_are_the_owners():
    assert st.tidy("a\r\r\n\r\n\r\n\r\nb\rc") == "a\n\nb\nc"
    t = ve.WSLC_TEMPLATES
    assert t["cp_in"] == "wslc cp {src} {name}:{dst}" and t["cp_out"] == "wslc cp {name}:{src} {dst}" and t["exec"] == "wslc exec {name} {args}" and t["exec_detached"] == "wslc exec -d {name} {args}"


def test_failed_report_lists_the_criteria_as_not_reached(tmp_path):
    rep = {"version": 1, "created": "t", "model": "m", "engine": {"kind": "wslc", "wslc_version": "wslc 3.0.5.0"}, "note": "n", "error": "x", "prerequisites": [], "comparisons": [], "hints": [], "commands": []}
    st._finish(rep, tmp_path, 0.0)
    md = (tmp_path / "vllm_selftest.md").read_text()
    assert "not reached" in md and "engine ready in under" in md
