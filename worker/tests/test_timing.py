"""Where the time went: `protocol.timing` lines and the `timings` key of a job's result (no new event type)."""
import json

from ndforge_worker import protocol


def collect(monkeypatch):
    lines = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: lines.append({"type": kind, **f}))
    protocol.start_job()
    return lines


def test_a_timing_is_one_log_line_with_json_and_is_kept(monkeypatch):
    lines = collect(monkeypatch)
    protocol.timing("load_model", 12.345, load="4bit", bytes=10, nothing=None)
    assert lines == [{"type": "log", "text": 'timing: {"bytes": 10, "load": "4bit", "seconds": 12.35, "stage": "load_model"}'}]
    assert protocol.timings() == [{"stage": "load_model", "seconds": 12.35, "load": "4bit", "bytes": 10}]
    assert json.loads(lines[0]["text"].split("timing: ", 1)[1])["stage"] == "load_model"


def test_timed_records_even_when_the_body_fails_and_takes_extra_fields(monkeypatch):
    collect(monkeypatch)
    try:
        with protocol.timed("write", bytes=5) as t:
            t["files"] = 2
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    rec = protocol.timings()[0]
    assert rec["stage"] == "write" and rec["bytes"] == 5 and rec["files"] == 2 and rec["seconds"] >= 0


def test_the_result_carries_the_timings_and_a_job_total_only_when_something_was_timed(monkeypatch):
    lines = collect(monkeypatch)
    protocol.result({"a": 1})
    assert lines[-1]["data"] == {"a": 1}                   # nothing timed: the answer is as it was
    protocol.timing("score", 3.0, tokens=100)
    protocol.result({"a": 1})
    data = lines[-1]["data"]
    assert data["a"] == 1 and [t["stage"] for t in data["timings"]] == ["score", "job"] and data["timings"][-1]["seconds"] >= 0
    protocol.result({"timings": "mine"})
    assert lines[-1]["data"]["timings"] == "mine"          # a key the job set itself is never replaced
    protocol.start_job()
    assert protocol.timings() == []                         # a new job starts clean
