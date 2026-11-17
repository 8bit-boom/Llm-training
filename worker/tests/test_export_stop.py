"""Stop ends the llama.cpp tools of an export (converter, quantize) at once, not after they are done."""
import sys
import threading
import time

import pytest

from ndforge_worker import exportjob, imatrix


@pytest.fixture(autouse=True)
def _forget_job():
    yield
    exportjob._JOB["cfg"] = None


def test_the_stop_file_ends_a_running_tool_quickly(tmp_path):
    stub = tmp_path / "convert_stub.py"
    stub.write_text("import time\nprint('converting', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    stop = tmp_path / "STOP"
    exportjob._JOB["cfg"] = {"stop_file": str(stop)}
    threading.Timer(0.5, lambda: stop.write_text("")).start()
    t0 = time.time()
    with pytest.raises(imatrix.Stopped):
        exportjob._run([sys.executable, stub], "The GGUF conversion")
    assert time.time() - t0 < 2.0 + 0.5


def test_the_old_place_of_the_stop_file_still_works(tmp_path):
    stub = tmp_path / "quantize_stub.py"
    stub.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    stop = tmp_path / "STOP"
    stop.write_text("")
    exportjob._JOB["cfg"] = {"imatrix": {"stop_file": str(stop)}}
    with pytest.raises(imatrix.Stopped):
        exportjob._run([sys.executable, stub], "The quantisation")


def test_a_tool_without_a_stop_request_runs_to_its_end_and_a_failure_is_reported(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("print('fine')\n", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nprint('broken')\nsys.exit(3)\n", encoding="utf-8")
    exportjob._JOB["cfg"] = {"stop_file": str(tmp_path / "STOP")}
    exportjob._run([sys.executable, ok], "The quantisation")
    with pytest.raises(RuntimeError, match=r"The quantisation failed \(exit code 3\)"):
        exportjob._run([sys.executable, bad], "The quantisation")
