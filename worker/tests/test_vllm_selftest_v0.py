"""The self-test: the model facts, the refusal when the weights cannot fit, the report that is always written (also after a crash at start)."""
import json
import types

import pytest

from ndforge_worker import protocol, vllmengine as ve, vllmselftest as st


def test_parameter_count_from_the_name_and_the_card_refusal():
    assert st.param_count_b("Qwen/Qwen3.8-27B")[0] == 27.0
    assert st.param_count_b("cyankiwi/Qwen3.5-9B-AWQ-INT4")[0] == 9.0
    assert st.param_count_b("org/model", 8_950_000_000)[0] == 8.95 and st.param_count_b("org/model")[0] is None
    facts = {"model": "Qwen/Qwen3.8-27B", "weights_gb": 55.4, "quant": "16-bit"}
    msg = st.fit_problem(facts, 32.0)
    assert "about 55 GB in 16-bit" in msg and "your card has 32 GB" in msg and "leave the model field empty" in msg and st.DEFAULT_MODEL in msg
    assert st.fit_problem({**facts, "weights_gb": 18}, 32.0) is None and st.fit_problem({**facts, "quant": "quantized (awq)"}, 32.0) is None and st.fit_problem({**facts, "weights_gb": None}, 32.0) is None


def test_the_refusal_still_writes_the_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(protocol, "_out", __import__("sys").stdout)
    monkeypatch.setattr(st, "model_facts", lambda m, q="none": {"model": m, "params_b": 27.0, "params_source": "name", "weights_gb": 55.4, "weights_source": "hub", "quant": "16-bit", "note": ""})

    def fake_run(cmd, timeout=600):
        return {"cmd": cmd, "returncode": 0, "output": "NVIDIA GeForce RTX 5090, 32607, 621.24" if "name,memory.total" in " ".join(cmd) else ""}

    monkeypatch.setattr(st, "_run", fake_run)
    rc = st.run({"model": "Qwen/Qwen3.8-27B", "confirm_download": True, "out_dir": str(tmp_path), "engine": {"kind": "docker"}})
    rep = json.loads((tmp_path / "vllm_selftest.json").read_text())
    assert rc == 0 and rep["verdict"] == "fail" and "your card has 32 GB" in rep["error"] and rep["gpu"]["name"] == "NVIDIA GeForce RTX 5090"
    assert rep["model_params_b"] == 27.0 and (tmp_path / "vllm_selftest.md").read_text().count("What was tested") == 1


def test_a_container_that_crashes_at_start_gives_a_report_with_its_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_out", __import__("sys").stdout)
    monkeypatch.setattr(st, "model_facts", lambda m, q="none": {"model": m, "params_b": 9.0, "params_source": "name", "weights_gb": 6.0, "weights_source": "hub", "quant": "quantized (awq)", "note": ""})
    monkeypatch.setattr(st, "prerequisites", lambda *a, **k: [])
    monkeypatch.setattr(st, "_run", lambda cmd, timeout=600: {"cmd": cmd, "returncode": 0, "output": ""})
    calls = []

    def run(cmd, **kw):
        calls.append(list(cmd))
        text = {"inspect": b"false", "logs": b"INFO x\nCUDA out of memory\n"}.get(cmd[1], b"")
        return types.SimpleNamespace(returncode=0, stdout=text, stderr=b"")

    monkeypatch.setattr(ve, "_run", run)
    monkeypatch.setattr(ve, "_popen", lambda cmd, **kw: types.SimpleNamespace(poll=lambda: None, returncode=None, terminate=lambda: None, wait=lambda t=None: 0))
    monkeypatch.setattr(ve, "STATE_EVERY", 0.0)
    monkeypatch.setattr(ve, "free_port", lambda: 4242)
    st.run({"model": "org/Model-9B-AWQ", "confirm_download": True, "out_dir": str(tmp_path), "engine": {"kind": "docker"}, "start_timeout": 1500})
    rep = json.loads((tmp_path / "vllm_selftest.json").read_text())
    assert rep["verdict"] == "fail" and "CUDA out of memory" in rep["exit_logs"] and rep["start_command"][0] == "docker" and rep["engine_stopped"] is True
    assert "docker" in " ".join(rep["start_command"]) and "--rm" not in rep["start_command"]
    assert [c[1] for c in calls if c[1] in ("logs", "stop", "rm")][:3] == ["logs", "stop", "rm"]
    assert "CUDA out of memory" in (tmp_path / "vllm_selftest.md").read_text()


def test_the_report_has_no_user_name_or_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("USERNAME", "Zebulon")
    text = protocol.redact_user("Settings file: C:\\Users\\Zebulon\\AppData\\Local\\wslc\\settings.yaml and /home/zebulon/x and /Users/bob/y, user Zebulon")
    assert "Zebulon" not in text and "zebulon" not in text.lower() and "bob" not in text and "C:\\Users\\<user>\\AppData" in text
    st._write(tmp_path, {"version": 1, "verdict": "fail", "verdict_text": "x", "created": "t", "model": "m", "engine": {}, "note": "n", "error": "C:\\Users\\Zebulon\\x", "criteria": [], "prerequisites": [], "comparisons": [], "hints": [], "commands": []})
    assert "Zebulon" not in (tmp_path / "vllm_selftest.json").read_text() + (tmp_path / "vllm_selftest.md").read_text()


def test_wslc_list_json_is_read_defensively():
    assert ve.wslc_running('[{"Name": "ndforge-vllm-5", "State": "Running"}]', "ndforge-vllm-5") is True
    assert ve.wslc_running('[{"names": ["ndforge-vllm-5"], "status": "exited (1)"}]', "ndforge-vllm-5") is False
    assert ve.wslc_running("[]", "ndforge-vllm-5") is False
    assert ve.wslc_running('[{"Name": "other", "State": "running"}]', "ndforge-vllm-5") is False
    assert ve.wslc_running("not json", "ndforge-vllm-5") is None
    assert ve.wslc_running('[{"id": "ndforge-vllm-5", "weird": 1}]', "ndforge-vllm-5") is True
    t = ve.WSLC_TEMPLATES
    assert (t["stop"], t["remove"], t["logs"], t["pull"]) == ("wslc stop -t 30 {name}", "wslc remove -f {name}", "wslc logs -n 60 {name}", "wslc pull -q {image}")


def test_answer_statistics_and_the_short_answer_warning():
    s = st.answer_stats([60, 80, 90, 100, 120, 3000])
    assert s["n"] == 6 and s["median"] == 100 and s["total"] == 3450 and s["p90"] == 3000 and st.answer_stats([])["n"] == 0
    rep = {"startup_seconds": 263.5, "tokens_per_s": 447.6, "long_tokens_per_s": 280.0}
    crit = {c["id"]: c for c in st.criteria_of(rep)}
    assert crit["speed"]["value"] == 280.0 and crit["speed"]["ok"] is False and "long answers" in crit["speed"]["label"], "the long-answer speed decides, not the short one"
    assert st.criteria_of({"startup_seconds": 263.5, "tokens_per_s": 447.6})[1]["label"].endswith("not measured on long ones)")


def test_the_report_says_it_is_a_generic_engine_test_not_a_test_of_your_original(tmp_path):
    rep = {"version": 1, "created": "t", "model": "cyankiwi/Qwen3.8-27B-AWQ-INT4", "engine": {"kind": "docker"}, "note": "n", "error": "x", "prerequisites": [], "comparisons": [], "hints": [], "commands": [], "generic_test": st.generic_note("cyankiwi/Qwen3.8-27B-AWQ-INT4")}
    st._finish(rep, tmp_path, 0.0)
    md = (tmp_path / "vllm_selftest.md").read_text()
    assert "generic engine test" in md and "does NOT test your own original" in md


def stats(median):
    return {"n": 16, "total": 16 * median, "mean": median, "median": median, "p90": median * 2}


REF = {"main": {"median_think_tokens": 121}}


def test_answers_similar_to_the_saved_check_are_fine():
    a = st.assess_answers(stats(219), {"think_rate": 1.0}, 16, 16, REF)
    assert a["level"] == "ok" and a["reference_median"] == 121 and "as long as" in a["text"]


def test_answers_much_shorter_than_the_saved_check_warn():
    a = st.assess_answers(stats(40), {"think_rate": 1.0}, 16, 16, REF)
    assert a["level"] == "warn" and "much shorter" in a["text"] and "did not think" not in a["text"]


def test_no_think_block_or_a_low_think_rate_is_the_only_reason_to_say_did_not_think():
    assert st.assess_answers(stats(300), {"think_rate": 1.0}, 0, 16, REF)["level"] == "warn"
    w = st.assess_answers(stats(300), {"think_rate": 0.0}, 16, 16, None)
    assert w["level"] == "warn" and "did not think" in w["text"]


def test_without_a_reference_it_is_only_a_note_with_the_typical_range():
    a = st.assess_answers(stats(219), {"think_rate": 1.0}, 16, 16, None)
    assert a["level"] == "note" and "median around 100-250 tokens" in a["text"] and "did not think" not in a["text"] and "compare with your standard check" in a["text"]
    assert st.assess_answers({"n": 0}, {}, 0, 0, None)["level"] == "note"


def test_the_report_prints_the_rates_the_think_blocks_and_two_samples(tmp_path):
    rep = {"version": 1, "created": "t", "model": "m", "engine": {"kind": "docker"}, "note": "n", "prerequisites": [], "comparisons": [], "hints": [], "commands": [], "completion_tokens": 3400, "wall_seconds": 7.3, "tokens_per_s": 472.3,
           "suite": {"think_rate": 1.0, "closed_rate": 0.97}, "answer_tokens": stats(219), "think_blocks": 15, "answer_check": st.assess_answers(stats(219), {"think_rate": 1.0}, 15, 16, None),
           "samples": [{"id": "a", "text": "<think>\nhmm " + "x" * 400}, {"id": "b", "text": "<think>\nsecond"}, {"id": "c", "text": "third"}]}
    st._finish(rep, tmp_path, 0.0)
    md = (tmp_path / "vllm_selftest.md").read_text()
    assert "think_rate 1.0, closed_rate 0.97, 15 of 16 with a `<think>` block" in md and "First 300 characters of sample answer `a`" in md
    assert "x" * 400 not in md.split("First 300 characters of sample answer `a`:")[1].split("First 300")[0]
    assert "SHORT" not in md and "did not think" not in md
