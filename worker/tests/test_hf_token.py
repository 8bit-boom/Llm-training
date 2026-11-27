"""The app-wide Hugging Face token (docs/HF_TOKEN.md): refusals (401 / 403 / gated) get one plain message that points to the Settings card, and the token (only a made-up one here)
is never in a log, an error, a job file or a command line. Nothing here reaches the network."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ndforge_worker import __main__ as entry
from ndforge_worker import hf_access, protocol
from ndforge_worker import vllmengine as ve

FAKE = "hf_FAKETOKEN123"
WORKER = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, code):
        self.status_code = code


class HfHubHTTPError(Exception):
    """what huggingface_hub raises (the real class has the same name and a `response`)"""

    def __init__(self, msg, code):
        super().__init__(msg)
        self.response = FakeResponse(code)


class GatedRepoError(HfHubHTTPError):
    pass


@pytest.fixture
def out(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    return buf


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


@pytest.mark.parametrize("exc", [
    GatedRepoError("403 Client Error. Cannot access gated repo for url https://huggingface.co/acme/m/resolve/main/config.json.", 403),
    HfHubHTTPError("anything", 401),
    OSError("You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/acme/m."),
    OSError("401 Client Error: Unauthorized for url: https://huggingface.co/api/models/acme/m (Request ID: Root=1)"),
    RuntimeError("Access to model acme/m is restricted and you are not in the authorized list. Visit https://huggingface.co/acme/m to ask for access."),
    OSError("Invalid credentials in Authorization header"),
])
def test_a_refusal_of_the_hub_is_recognised(exc):
    assert hf_access.is_access_error(exc)
    assert hf_access.is_access_error(str(exc)) or isinstance(exc, HfHubHTTPError) and exc.response.status_code in (401, 403)


@pytest.mark.parametrize("text", [
    "404 Client Error: Not Found for url: https://huggingface.co/api/models/acme/typo",
    "step 1401 of 5000, loss 1.2",
    "CUDA out of memory",
    "The disk is full",
    "HTTP Error 403: Forbidden from a server that is not the Hub",
    "",
])
def test_other_failures_are_not_taken_for_a_refusal(text):
    assert not hf_access.is_access_error(text)
    assert not hf_access.is_access_error(OSError(text))


def test_the_message_names_the_card_and_never_the_token(monkeypatch):
    m = hf_access.message("acme/m")
    assert "Training setup" in m and "Hugging Face token" in m and "huggingface.co/settings/tokens" in m and "acme/m" in m and "none is saved" in m
    monkeypatch.setenv("HF_TOKEN", FAKE)
    m2 = hf_access.message()
    assert "was not accepted" in m2 and "gated" in m2 and "Training setup" in m2 and FAKE not in m2 and m2 != m


def test_explain_and_error_lines_use_the_plain_message(out):
    exc = GatedRepoError("403 Client Error. Cannot access gated repo for url https://huggingface.co/acme/m/resolve/main/config.json.", 403)
    assert entry._explain(exc) == hf_access.message()
    assert "Training setup" in entry._explain_heal("distill", exc)
    # a raw Hub error that reaches protocol.error as text is answered in the same words
    protocol.error("export_failed", "OSError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/models/acme/m")
    line = json.loads(out.getvalue().strip().splitlines()[-1])
    assert line["code"] == "export_failed" and line["message"] == hf_access.message()
    # a message that already says it is left alone
    protocol.error("export_failed", hf_access.message("acme/m"))
    assert json.loads(out.getvalue().strip().splitlines()[-1])["message"] == hf_access.message("acme/m")


def test_the_old_branch_still_answers_what_is_not_a_refusal():
    assert "could not be downloaded" in entry._explain(OSError("repository not found"))


def test_the_hub_reading_modules_give_the_plain_message(monkeypatch):
    from ndforge_worker import imatrixjob
    assert imatrixjob.plain(GatedRepoError("403 Client Error gated repo huggingface.co", 403)) == hf_access.message()


def test_the_container_gets_the_name_of_the_variable_and_never_the_value(monkeypatch):
    cfg = ve.engine_config({})
    plain = ve.docker_command(cfg, 8000, ["--model", "x"])
    assert "-e" not in plain and "HF_TOKEN" not in plain
    monkeypatch.setenv("HF_TOKEN", FAKE)
    cmd = ve.docker_command(cfg, 8000, ["--model", "x"])
    assert cmd[cmd.index("-e") + 1] == "HF_TOKEN" and not any(FAKE in w for w in cmd)
    assert cmd.index("-e") < cmd.index(cfg["image"])
    wsl = ve.docker_command(ve.engine_config({"engine": {"kind": "docker_wsl"}}), 8000, [])
    assert "HF_TOKEN" in wsl and not any(FAKE in w for w in wsl)
    import os
    assert "HF_TOKEN" in os.environ["WSLENV"].split(":")


def test_the_scrub_takes_the_token_out_of_every_line(monkeypatch, out):
    monkeypatch.setenv("HF_TOKEN", FAKE)
    protocol.log("using " + FAKE + " now")
    protocol.error("export_failed", "failed with " + FAKE)
    text = out.getvalue()
    assert FAKE not in text and text.count("[token]") == 2


def _run(job: dict, tmp_path: Path, token: str | None):
    jf = tmp_path / "job.json"
    jf.write_text(json.dumps(job), encoding="utf-8")
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["PYTHONPATH"] = str(WORKER)
    if token:
        env["HF_TOKEN"] = token
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(jf)], capture_output=True, text=True, env=env, cwd=WORKER, timeout=120)
    return p, jf


def test_a_real_worker_process_never_writes_the_token_to_stdout_stderr_or_its_job_file(tmp_path):
    # an unknown job whose name carries the token: the error text, the only thing the worker echoes of the job, comes back scrubbed
    p, jf = _run({"job": "no_such_job_" + FAKE}, tmp_path, FAKE)
    assert p.returncode == 1
    assert FAKE not in p.stdout and FAKE not in p.stderr, (p.stdout, p.stderr)
    last = json.loads(p.stdout.strip().splitlines()[-1])
    assert last["type"] == "error" and last["code"] == "unknown_job" and "[token]" in last["message"]
    # hello (which lists packages) with the variable set
    p, _ = _run({"job": "hello"}, tmp_path, FAKE)
    assert p.returncode == 0 and FAKE not in p.stdout + p.stderr


def test_no_worker_module_writes_the_token_to_a_file_or_a_result():
    """Only these modules read the variable at all (to send it as a header or to scrub it); none of them stores it."""
    readers = set()
    for f in sorted((WORKER / "ndforge_worker").glob("*.py")):
        t = f.read_text(encoding="utf-8")
        if "HF_TOKEN" in t or "HUGGING_FACE_HUB_TOKEN" in t:
            readers.add(f.name)
    assert readers <= {"protocol.py", "generaltext.py", "imatrixmerge.py", "imatrixjob.py", "imatrixadapt.py", "vllmengine.py", "hf_access.py"}, readers
    for name in readers:
        t = (WORKER / "ndforge_worker" / name).read_text(encoding="utf-8")
        assert "write_text(" not in t.split("HF_TOKEN")[1][:200] and "json.dump" not in t.split("HF_TOKEN")[1][:120], name
