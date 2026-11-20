"""The GGUF conversion of a model without an MTP head passes --no-mtp (a newer llama.cpp asserts otherwise); failures show the converter's last lines."""
import json
import sys

import pytest

from ndforge_worker import exportjob, imatrix, imatrixjob

NEW = '''import argparse, sys
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--outfile")
ap.add_argument("--outtype")
ap.add_argument("--no-mtp", "--no-nextn", action="store_true")
ap.add_argument("--mmproj", action="store_true")
a = ap.parse_args()
print("INFO:hf-to-gguf:Model architecture: Qwen3_5ForConditionalGeneration", flush=True)
if not a.no_mtp and not a.mmproj:
    print('  File "C:\\\\Users\\\\Someone\\\\llama.cpp\\\\conversion\\\\qwen.py", line 323, in __init__', flush=True)
    print("    assert self.opt_num_mtp_layers != 0", flush=True)
    print("AssertionError", flush=True)
    sys.exit(1)
open(a.outfile, "wb").write(b"GGUF")
'''
ALWAYS = NEW.replace('if not a.no_mtp and not a.mmproj:', 'if True:')
OLD = NEW.replace('ap.add_argument("--no-mtp", "--no-nextn", action="store_true")\n', 'a_no_mtp = False\n').replace("a = ap.parse_args()", "a = ap.parse_args()\na.no_mtp = False")


def folder(tmp_path, mtp_tensors=False, mtp_layers=0):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"text_config": {"mtp_num_hidden_layers": mtp_layers}}), encoding="utf-8")
    names = ["model.layers.0.w"] + (["mtp.layers.0.w"] if mtp_tensors else [])
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "m.safetensors" for n in names}}), encoding="utf-8")
    return d


def conv(tmp_path, text, name="convert_hf_to_gguf.py"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _job():
    imatrix._NO_MTP_SUPPORT.clear()
    exportjob._JOB["cfg"] = None
    yield
    exportjob._JOB["cfg"] = None


def test_a_model_without_an_mtp_head_is_converted_with_the_flag(tmp_path):
    c, m = conv(tmp_path, NEW), folder(tmp_path)
    assert imatrix.no_mtp_extra(c, m) == ["--no-mtp"]
    exportjob.convert_to(c, m, tmp_path / "out.gguf")
    assert (tmp_path / "out.gguf").is_file()


def test_a_model_with_an_mtp_head_gets_no_flag(tmp_path):
    c = conv(tmp_path, NEW)
    for m in (folder(tmp_path, mtp_tensors=True),):
        assert imatrix.no_mtp_extra(c, m) == []
    d = tmp_path / "second"
    d.mkdir()
    m2 = folder(d, mtp_layers=1)
    assert imatrix.no_mtp_extra(c, m2) == []


def test_the_vision_conversion_never_gets_the_flag(tmp_path):
    c, m = conv(tmp_path, NEW), folder(tmp_path)
    exportjob.convert_to(c, m, tmp_path / "mm.gguf", "The vision conversion", ["--mmproj"])
    assert (tmp_path / "mm.gguf").is_file()


def test_an_old_converter_without_the_flag_is_run_as_before(tmp_path):
    c, m = conv(tmp_path, OLD), folder(tmp_path)
    assert not imatrix.converter_has_no_mtp(c)
    assert imatrix.no_mtp_extra(c, m) == []


def test_the_support_check_reads_the_script_once(tmp_path):
    c = conv(tmp_path, NEW)
    assert imatrix.converter_has_no_mtp(c)
    c.write_text("nothing", encoding="utf-8")
    assert imatrix.converter_has_no_mtp(c)


def test_a_model_of_unknown_kind_is_retried_with_the_flag_after_the_assertion(tmp_path, monkeypatch):
    c, m = conv(tmp_path, NEW), tmp_path / "unknown"
    m.mkdir()
    logs = []
    monkeypatch.setattr(exportjob.protocol, "log", lambda t: logs.append(t))
    exportjob.convert_to(c, m, tmp_path / "out.gguf")
    assert (tmp_path / "out.gguf").is_file()
    assert any("once more with --no-mtp" in t for t in logs)


def test_a_failing_converter_reports_the_cause(tmp_path):
    c = conv(tmp_path, ALWAYS)
    m = folder(tmp_path, mtp_tensors=True)
    with pytest.raises(RuntimeError) as e:
        exportjob.convert_to(c, m, tmp_path / "out.gguf")
    msg = str(e.value)
    assert "failed (exit code 1)" in msg and "Last lines" in msg and "assert self.opt_num_mtp_layers != 0" in msg and "AssertionError" in msg
    assert "Someone" not in msg


def test_an_old_converter_that_asserts_is_not_retried(tmp_path):
    c = conv(tmp_path, OLD.replace("a.no_mtp = False", "a.no_mtp = False\nopen(a.outfile + '.n', 'w').write('x')"))
    m = folder(tmp_path)
    with pytest.raises(RuntimeError, match="opt_num_mtp_layers"):
        exportjob.convert_to(c, m, tmp_path / "out.gguf")


def test_the_matrix_conversion_uses_the_flag_and_reports_the_tail(tmp_path):
    c, m = conv(tmp_path, NEW), folder(tmp_path)
    imatrixjob.convert_to_f16(c, m, tmp_path / "f.gguf", {}, (0.0, 0.1))
    assert (tmp_path / "f.gguf").is_file()
    d = tmp_path / "y"
    d.mkdir()
    c2 = conv(tmp_path, ALWAYS, "other.py")
    with pytest.raises(imatrixjob.MatrixError, match="opt_num_mtp_layers"):
        imatrixjob.convert_to_f16(c2, folder(d, mtp_tensors=True), tmp_path / "g.gguf", {}, (0.0, 0.1))
