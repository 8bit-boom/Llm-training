import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parent.parent


def run(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=60)
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


def test_hello(tmp_path):
    code, ev = run(tmp_path, {"job": "hello"})
    assert code == 0 and ev == [{"type": "result", "data": {"worker": "0.1.0"}}]


def test_unknown_job_and_bad_file(tmp_path):
    code, ev = run(tmp_path, {"job": "nope"})
    assert code == 1 and ev[-1]["type"] == "error" and ev[-1]["code"] == "unknown_job"
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(tmp_path / "missing.json")], cwd=WORKER, capture_output=True, text=True)
    assert p.returncode == 1 and json.loads(p.stdout)["code"] == "bad_job"


def test_selftest_without_torch_says_so_plainly(tmp_path):
    try:
        import torch  # noqa: F401
        return  # torch is installed here: the real self-test is for a GPU machine
    except ImportError:
        pass
    code, ev = run(tmp_path, {"job": "selftest"})
    assert code == 1 and ev[-1]["code"] == "torch_missing" and "setup" in ev[-1]["message"]


def test_selftest_tells_a_cpu_only_pytorch_from_a_driver_problem(tmp_path):
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return  # a real GPU machine: the real self-test runs there
    code, ev = run(tmp_path, {"job": "selftest"})
    last = ev[-1]
    assert code == 1 and last["type"] == "error"
    if torch.version.cuda is None:     # the PyTorch build has no CUDA at all: the driver is not the problem, and the message says what to press
        assert last["code"] == "cpu_torch" and "CPU-only" in last["message"] and "Install the training software" in last["message"] and "driver is fine" in last["message"]
    else:                              # a CUDA build that sees no card: the driver, a restart, or another program holding the card
        assert last["code"] == "no_cuda" and "restart" in last["message"] and torch.__version__ in last["message"]
