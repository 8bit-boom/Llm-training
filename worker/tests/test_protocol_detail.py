import json

from ndforge_worker import protocol


def _lines(monkeypatch):
    out = []

    class W:
        def write(self, s):
            out.append(s)

        def flush(self):
            pass

    monkeypatch.setattr(protocol, "_out", W())
    return out


def test_progress_without_detail_is_byte_identical(monkeypatch):
    out = _lines(monkeypatch)
    protocol.progress("cache", 0.5, "half")
    assert out[0] == '{"type": "progress", "step": "cache", "value": 0.5, "text": "half"}\n'


def test_progress_with_detail_adds_one_key(monkeypatch):
    out = _lines(monkeypatch)
    d = protocol.make_detail("tokens", 50, 100, 0.0, now=10.0, rate_unit="tokens/s")
    protocol.progress("cache", 0.5, "half", d)
    got = json.loads(out[0])
    assert got["detail"] == {"unit": "tokens", "done": 50.0, "total": 100.0, "rate": 5.0, "rate_unit": "tokens/s", "eta_s": 10.0}
    assert got["step"] == "cache" and got["value"] == 0.5


def test_detail_has_no_rate_before_work_was_done():
    d = protocol.make_detail("tokens", 20, 100, 0.0, now=10.0, done_at_start=20)
    assert "rate" not in d and "eta_s" not in d
    d = protocol.make_detail("files", 1, 3)
    assert d == {"unit": "files", "done": 1.0, "total": 3.0}
