"""The `hello` job: what the program learns about the worker and the installed libraries without starting any of them."""
import json
import sys

from ndforge_worker import __main__ as entry
from ndforge_worker import __version__, protocol


def run_hello(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(protocol, "emit", lambda kind, **f: events.append({"type": kind, **f}))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"job": "hello"}), encoding="utf-8")
    assert entry.main(["x", str(job)]) == 0
    return [e for e in events if e["type"] == "result"][0]["data"]


def test_hello_names_the_protocol_and_the_versions_of_the_libraries(tmp_path, monkeypatch):
    data = run_hello(tmp_path, monkeypatch)
    assert data["worker"] == __version__ and data["protocol"] == protocol.PROTOCOL_VERSION == 1
    assert list(data["packages"]) == ["torch", "transformers", "peft", "trl", "accelerate", "bitsandbytes", "unsloth", "datasets"]
    assert all(v is None or isinstance(v, str) for v in data["packages"].values())


def test_hello_never_imports_the_heavy_libraries(tmp_path, monkeypatch):
    for name in [m for m in sys.modules if m.split(".")[0] in ("torch", "transformers", "peft", "trl", "unsloth", "bitsandbytes")]:
        monkeypatch.delitem(sys.modules, name)
    run_hello(tmp_path, monkeypatch)
    assert not [m for m in sys.modules if m.split(".")[0] in ("torch", "transformers", "peft", "trl", "unsloth", "bitsandbytes")], "reading versions must not load the libraries"


def test_a_missing_library_is_none_and_an_installed_one_has_its_version(tmp_path, monkeypatch):
    from importlib import metadata
    real = metadata.version
    monkeypatch.setattr(metadata, "version", lambda n: "9.9.9" if n == "peft" else real(n) if n == "pytest" else (_ for _ in ()).throw(metadata.PackageNotFoundError(n)))
    data = run_hello(tmp_path, monkeypatch)
    assert data["packages"]["peft"] == "9.9.9" and data["packages"]["torch"] is None
