"""One finder for llama.cpp (ndforge_worker/llamacpp.py): the order of the places, the folder the program installs into (NDFORGE_LLAMA_CPP_DIR), the tools that go through it
(export, importance matrices, matrix merge, Compare quants), and the error code `llama_cpp_missing` the page uses to offer the installer."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ndforge_worker import exportjob, imatrix, imatrixjob, imatrixmerge, llamacpp, quantcheck

EXE = lambda n: n + (".exe" if os.name == "nt" else "")      # noqa: E731


def make_llama(folder: Path, tools=("llama-quantize", "llama-imatrix", "llama-perplexity"), convert=True, where="build/bin") -> Path:
    """A stand-in llama.cpp folder: the files only exist (the finder looks for files, it does not run them)."""
    bin_dir = folder / where
    bin_dir.mkdir(parents=True, exist_ok=True)
    for t in tools:
        (bin_dir / EXE(t)).write_text("x")
    if convert:
        (folder / "convert_hf_to_gguf.py").write_text("x")
    return folder


@pytest.fixture
def places(tmp_path, monkeypatch):
    """No llama.cpp anywhere: a fresh home folder, no environment variables, an empty PATH."""
    home = tmp_path / "home"
    home.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    for var in ("LLAMA_CPP_DIR", "NDFORGE_LLAMA_CPP_DIR"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "emptypath"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return type("Places", (), {"home": home, "tmp": tmp_path, "path": empty})


def test_the_places_come_in_the_documented_order(places, monkeypatch):
    assert [h for h, _ in llamacpp.candidates()] == ["home_folder", "next_to_program"]
    monkeypatch.setenv("NDFORGE_LLAMA_CPP_DIR", str(places.tmp / "installed"))
    monkeypatch.setenv("LLAMA_CPP_DIR", str(places.tmp / "mine"))
    c = llamacpp.candidates()
    assert [h for h, _ in c] == ["env_var", "program_install", "home_folder", "next_to_program"]
    assert c[0][1] == places.tmp / "mine" and c[1][1] == places.tmp / "installed" and c[2][1] == places.home / "llama.cpp"
    assert llamacpp.find_llama_cpp() == (None, None) and llamacpp.locate() == {}


def test_an_explicit_llama_cpp_dir_wins_then_the_installed_folder_then_the_home_folder(places, monkeypatch):
    mine, installed, home = (make_llama(places.tmp / n) for n in ("mine", "installed", places.home / "llama.cpp"))
    for d in (mine, installed, home):
        assert d.exists()
    monkeypatch.setenv("NDFORGE_LLAMA_CPP_DIR", str(installed))
    monkeypatch.setenv("LLAMA_CPP_DIR", str(mine))
    assert llamacpp.locate()["how"] == "env_var" and llamacpp.locate()["folder"] == str(mine)
    conv, quant = llamacpp.find_llama_cpp()
    assert conv == mine / "convert_hf_to_gguf.py" and quant == mine / "build" / "bin" / EXE("llama-quantize")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(places.tmp / "nowhere"))      # an explicit folder that is no llama.cpp does not hide the installed one
    assert llamacpp.locate()["how"] == "program_install" and llamacpp.find_llama_cpp()[0] == installed / "convert_hf_to_gguf.py"
    monkeypatch.delenv("LLAMA_CPP_DIR")
    assert llamacpp.locate()["how"] == "program_install"
    monkeypatch.delenv("NDFORGE_LLAMA_CPP_DIR")
    assert llamacpp.locate()["how"] == "home_folder" and llamacpp.find_llama_cpp()[0] == home / "convert_hf_to_gguf.py"


@pytest.mark.parametrize("where", ["", "build/bin", "build/bin/Release"])
def test_the_three_layouts_of_a_folder_are_accepted(places, where):
    d = make_llama(places.tmp / "x", where=where or ".")
    conv, quant = llamacpp.layout(d)
    assert conv == d / "convert_hf_to_gguf.py" and quant.name == EXE("llama-quantize")
    assert llamacpp.layout(make_llama(places.tmp / "noconv", convert=False, where=where or ".")) == (None, None)
    assert llamacpp.layout(make_llama(places.tmp / "noq", tools=("llama-imatrix",), where=where or ".")) == (None, None)


def test_every_user_finds_the_installed_folder_through_the_one_finder(places, monkeypatch):
    installed = make_llama(places.tmp / "installed")
    monkeypatch.setenv("NDFORGE_LLAMA_CPP_DIR", str(installed))
    conv, quant = exportjob.find_llama_cpp()                                         # export
    assert conv == installed / "convert_hf_to_gguf.py"
    assert imatrix.find_llama_imatrix(quant) == installed / "build" / "bin" / EXE("llama-imatrix")      # importance matrix
    assert quantcheck.find_perplexity() == installed / "build" / "bin" / EXE("llama-perplexity")        # Compare quants
    c, tool = imatrixjob.find_tools(need_convert=True)                               # matrices from a model
    assert c == conv and tool.name == EXE("llama-imatrix")
    assert imatrixmerge.find_tools is imatrixjob.find_tools                          # matrix merge: the same function
    assert imatrixjob.find_tools(need_convert=False)[1] == tool


def test_a_tool_is_found_in_a_folder_without_a_converter_and_on_the_path(places, monkeypatch):
    d = make_llama(places.tmp / "lonely", tools=("llama-perplexity",), convert=False)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(d))
    assert exportjob.find_llama_cpp() == (None, None)
    assert quantcheck.find_perplexity() == d / "build" / "bin" / EXE("llama-perplexity")
    monkeypatch.delenv("LLAMA_CPP_DIR")
    assert quantcheck.find_perplexity() is None
    (places.path / EXE("llama-perplexity")).write_text("x")
    (places.path / EXE("llama-imatrix")).write_text("x")
    if os.name != "nt":
        for n in ("llama-perplexity", "llama-imatrix"):
            (places.path / n).chmod(0o755)
    assert quantcheck.find_perplexity().name == EXE("llama-perplexity")
    assert imatrix.find_llama_imatrix(None).name == EXE("llama-imatrix")
    # the converter and llama-quantize are NOT taken from the PATH
    (places.path / EXE("llama-quantize")).write_text("x")
    (places.path / "convert_hf_to_gguf.py").write_text("x")
    assert exportjob.find_llama_cpp() == (None, None)


def test_a_missing_llama_cpp_has_the_code_the_real_search_order_and_the_install_hint(places):
    with pytest.raises(imatrixjob.MatrixError) as e:
        imatrixjob.find_tools(need_convert=True)
    exc = e.value
    assert isinstance(exc, llamacpp.LlamaCppMissing) and exc.code == "llama_cpp_missing"
    text = str(exc)
    assert text.startswith("llama.cpp was not found, so no importance matrix can be made.")
    assert "https://github.com/ggml-org/llama.cpp" in text and "convert_hf_to_gguf.py" in text and "LLAMA_CPP_DIR" in text      # the old text stays for logs
    assert "where it installed llama.cpp itself" in text and "home folder" in text and "next to the program" in text
    assert "llama-imatrix and llama-perplexity are also looked for on the PATH (the converter script is not)" in text
    assert "Install llama.cpp now" in text
    assert "then on the PATH" not in text, "the old, inaccurate sentence is gone"
    # llama-imatrix found, converter missing: also the code (the installer fixes it)
    make_llama(places.tmp / "half", tools=("llama-imatrix",), convert=False)
    os.environ["LLAMA_CPP_DIR"] = str(places.tmp / "half")
    try:
        with pytest.raises(imatrixjob.MatrixError) as e2:
            imatrixjob.find_tools(need_convert=True)
        assert e2.value.code == "llama_cpp_missing" and "convert_hf_to_gguf.py" in str(e2.value) and "Install llama.cpp now" in str(e2.value)
        assert imatrixjob.find_tools(need_convert=False)[0] is None
    finally:
        os.environ.pop("LLAMA_CPP_DIR")


def test_compare_quants_and_the_export_raise_the_same_code(places, tmp_path):
    job = {"root": str(tmp_path / "q"), "id": "c1", "quants": [{"path": str(tmp_path / "a.gguf"), "label": "a"}]}
    with pytest.raises(quantcheck.QuantCheckError) as e:
        quantcheck.run(job)
    assert isinstance(e.value, llamacpp.LlamaCppMissing) and e.value.code == "llama_cpp_missing"
    assert "llama-perplexity was not found" in str(e.value) and "LLAMA_CPP_DIR" in str(e.value) and "https://github.com/ggml-org/llama.cpp" in str(e.value) and "Install llama.cpp now" in str(e.value)
    with pytest.raises(exportjob.ExportLlamaMissing) as e2:
        exportjob.gguf_with_llama_cpp({"imatrix": None}, tmp_path / "merged", tmp_path / "out", "q4_k_m")
    assert e2.value.code == "llama_cpp_missing" and "LLAMA_CPP_DIR" in str(e2.value) and "Install llama.cpp now" in str(e2.value)


def run_worker(places, job: dict, extra_env=None) -> list:
    """Runs the real entry point on a job file and returns the JSON lines it printed."""
    f = places.tmp / "job.json"
    f.write_text(json.dumps(job))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(Path(llamacpp.__file__).resolve().parents[1])}
    env.update(extra_env or {})
    p = subprocess.run([sys.executable, "-X", "utf8", "-m", "ndforge_worker", str(f)], capture_output=True, text=True, env=env, timeout=120)
    return [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]


def test_the_worker_reports_the_missing_llama_cpp_with_its_code(places):
    gguf = places.tmp / "m.gguf"
    gguf.write_bytes(b"x")
    job = {"job": "imatrix_make", "root": str(places.tmp / "lib"), "id": "m1", "chunks": 64, "source": {"kind": "gguf", "file": str(gguf)}, "calibration": {"kind": "text", "file": str(gguf)}}
    (err,) = [e for e in run_worker(places, job) if e["type"] == "error"]
    assert err["code"] == "llama_cpp_missing" and err["message"].startswith("llama.cpp was not found")
    # the same job with the installed folder: the finder passes (the job then goes on to fail for other reasons, not with this code)
    make_llama(places.tmp / "installed")
    events = run_worker(places, job, {"NDFORGE_LLAMA_CPP_DIR": str(places.tmp / "installed")})
    assert not any(e.get("code") == "llama_cpp_missing" for e in events)
    # Compare quants
    qjob = {"job": "quantcheck", "root": str(places.tmp / "q"), "id": "c1", "quants": [{"path": str(gguf), "label": "a"}]}
    (err,) = [e for e in run_worker(places, qjob) if e["type"] == "error"]
    assert err["code"] == "llama_cpp_missing" and "llama-perplexity was not found" in err["message"]
    # any other failure keeps its own code
    (err,) = [e for e in run_worker(places, {**qjob, "quants": []}) if e["type"] == "error"]
    assert err["code"] == "quantcheck_failed"
