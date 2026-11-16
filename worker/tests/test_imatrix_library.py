"""The matrix library's jobs (imatrixjob.py): a matrix made without any export, the legacy .dat, the statistics, the light compatibility check and "use a saved matrix" in
an export. Same stand-in llama.cpp tools as test_imatrix.py (small Python scripts, the same on Windows and Linux); no torch, no llama.cpp, no network."""
import errno
import json
import struct
import sys
import threading
import time
from pathlib import Path

import pytest
from test_imatrix import CONVERT, IMATRIX, StubTok, _stop_when_started, flag, general_file, project_rows, write_rows, world  # noqa: F401  (world is a fixture)

from ndforge_worker import __main__ as entry
from ndforge_worker import exportjob, imatrix, imatrixjob, protocol

# the stand-in llama-imatrix of test_imatrix.py, taught the two other things the library asks of it: --in-file (the legacy conversion, the statistics)
IN_FILE = '''if "--in-file" in a:
    open(log, "a").write(json.dumps({"tool": "imatrix-infile", "args": a}) + "\\n")
    if "--show-statistics" in a:
        if os.environ.get("STATS_MODE") == "unsupported":
            print("error: invalid argument: --show-statistics")
            sys.exit(1)
        print("Computing statistics for imatrix.gguf (3 tensors)")
        print("Layer\\tTensor\\tMin\\tMax\\tμ\\tσ\\t% Active")
        for i, (mx, mu) in enumerate([(2.0, 0.5), (8.0, 1.5), (4.0, 1.0)]):
            print("%d\\tblk.%d.attn_k\\t0.0\\t%s\\t%s\\t0.1\\t90.0" % (i, i, mx, mu))
        sys.exit(0)
    dat_mode = os.environ.get("DAT_MODE", "")
    if dat_mode == "never" or (dat_mode == "noflag" and "--output-format" in a):
        print("error: invalid argument: --output-format" if dat_mode == "noflag" else "error: cannot convert")
        sys.exit(1)
    open(a[a.index("-o") + 1], "wb").write(b"LEGACY-DAT")
    sys.exit(0)
'''
IMATRIX_LIB = IMATRIX.replace('calib = a[a.index("-f") + 1]', IN_FILE + 'calib = a[a.index("-f") + 1]')
CONVERT_LIB = CONVERT.replace('open(out, "wb")', 'import os\nif os.environ.get("CONVERT_MODE") == "fail":\n    print("error: unknown architecture")\n    sys.exit(1)\nif os.environ.get("CONVERT_MODE") == "full":\n    raise OSError(28, "No space left on device")\nopen(os.environ["STUB_LOG"], "a").write(__import__("json").dumps({"tool": "convert", "args": a[1:]}) + "\\n")\nopen(out, "wb")')


def gguf(kvs: dict, tensors: list) -> bytes:
    """a tiny hand-built GGUF header (version 3)"""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, v in kvs.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    for t in tensors:
        out += s(t) + struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    return out


def matrix_gguf(layers=2, arch=None):
    names = [f"blk.{i}.attn_k.weight.{x}" for i in range(layers) for x in ("in_sum2", "counts")]
    kvs = {"general.type": "imatrix", "imatrix.chunk_count": 64}
    if arch:
        kvs["general.architecture"] = arch
    return gguf(kvs, names)


@pytest.fixture
def lib(world, monkeypatch):
    (world.lcpp / "llama-imatrix.py").write_text(IMATRIX_LIB, encoding="utf-8")
    (world.lcpp / "convert_hf_to_gguf.py").write_text(CONVERT_LIB, encoding="utf-8")
    for var in ("CONVERT_MODE", "DAT_MODE", "STATS_MODE"):
        monkeypatch.delenv(var, raising=False)
    world.root_lib = world.root / "imatrix"
    adapter = world.root / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"W")
    merges = []

    def fake_merge(cfg, merged):
        merges.append(cfg)
        merged.mkdir(parents=True)
        (merged / "config.json").write_text("{}")

    monkeypatch.setattr(exportjob, "merge_hf", fake_merge)
    world.merges = merges
    own = world.root / "mytext.txt"
    own.write_text("The old kingdom of Ard stood between two rivers.\n" * 400, encoding="utf-8")
    world.own = own

    def job(source=None, calibration=None, **over):
        j = {"job": "imatrix_make", "root": str(world.root_lib), "id": "20261115-1030-run1", "label": "my first", "model_name": "run1", "stop_file": str(world.root / "STOP"),
             "chunks": 64, "ctx": 128, "nextn": False, "dat": False, "statistics": True,
             "source": source or {"kind": "run", "run": "run1", "adapter": str(adapter), "full_model": None, "base": {"repo": "org/base", "family": "qwen3_5"}, "f16_candidates": []},
             "calibration": calibration or {"kind": "data", "train": world.train, "data_version": 3, "general_share": 0.2, "general": {"file": world.general}}}
        j.update(over)
        return j

    world.job = job
    world.make = lambda **kw: imatrixjob.make(job(**kw))
    return world


def result_error(lib, exc_type=imatrixjob.MatrixError, **kw):
    with pytest.raises(exc_type) as e:
        lib.make(**kw)
    return str(e.value)


# ------------------------------------------------------------------------------------------------------------------ a trained run

def test_a_run_is_merged_converted_measured_and_saved_as_a_library_entry(lib):
    r = lib.make()
    folder = lib.root_lib / "20261115-1030-run1"
    assert r["id"] == "20261115-1030-run1" and Path(r["dir"]) == folder and Path(r["file"]) == folder / "imatrix.gguf"
    assert (folder / "imatrix.gguf").read_bytes() == b"IMATRIX-FILE" and not (folder / "work").exists() and not (folder / "imatrix.partial.gguf").exists()
    assert sorted(p.name for p in folder.iterdir()) == ["imatrix.gguf", "imatrix.json"]
    (m,) = lib.merges
    assert m["adapter"].endswith("adapter") and m["base"] == {"repo": "org/base", "family": "qwen3_5"}
    (conv,) = lib.calls("convert")
    assert conv["args"][0].endswith("merged") and flag(conv["args"], "--outtype") == "f16" and "work" in flag(conv["args"], "--outfile")
    (im,) = lib.calls("imatrix")
    a = im["args"]
    assert Path(flag(a, "-m")).name == "model-f16.gguf" and Path(flag(a, "-f")).name == "calibration.txt" and Path(flag(a, "-o")).name == "imatrix.partial.gguf"
    assert flag(a, "--output-format") == "gguf" and flag(a, "--chunks") == "64" and flag(a, "-c") == "128" and flag(a, "-ngl") == "99" and "--parse-special" in a and "--no-ppl" in a
    assert "--nextn" not in a
    # the calibration is the project's data through the chat template plus general text
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\nfact0_0" in im["calibration"] and "gen0_0" in im["calibration"]
    info = json.loads((folder / "imatrix.json").read_text(encoding="utf-8"))
    for k in ("model", "weights", "data_hash", "chunks", "ctx", "general_share", "general_used", "general_source", "nextn", "nextn_asked", "llama_cpp", "via", "created", "size", "source", "label", "id"):
        assert k in info, k
    assert (info["model"], info["chunks"], info["ctx"], info["general_share"], info["general_used"], info["via"], info["label"], info["size"]) == ("run1", 64, 128, 0.2, True, "llama.cpp", "my first", 12)
    assert info["data_hash"] == imatrix.file_sha256(lib.train) and info["llama_cpp"] == "version: 9999 (stub1234)" and "run1" in info["source"] and info["data_version"] == 3
    assert info["has_dat"] is False and info["source_kind"] == "run" and info["calibration_kind"] == "data"


def test_progress_is_the_imatrix_phase_and_never_goes_back(lib):
    lib.make()
    prog = [e for e in lib.events if e["type"] == "progress"]
    assert {e["step"] for e in prog} == {"imatrix"} and prog[-1]["value"] == 1.0
    assert all(a["value"] <= b["value"] for a, b in zip(prog, prog[1:])), [e["value"] for e in prog]
    assert any(e["text"] == "Chunk 64 of 64" for e in prog)
    logs = [e["text"] for e in lib.events if e["type"] == "log"]
    assert any(t.startswith("$ ") and "--parse-special" in t for t in logs) and "[64]1.0," in logs


def test_a_16_bit_file_of_an_earlier_export_is_used_instead_of_merging_again(lib):
    old = lib.root / "exports" / "run1-f16" / "model-f16.gguf"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"GGUF-old")
    src = lib.job()["source"]
    src["f16_candidates"] = [str(lib.root / "exports" / "missing.gguf"), str(old)]
    r = lib.make(source=src)
    assert lib.merges == [] and lib.calls("convert") == []
    assert flag(lib.calls("imatrix")[0]["args"], "-m") == str(old) and old.exists(), "the earlier export's file is left alone"
    assert "earlier export" in " ".join(r["notes"])


def test_a_16_bit_file_older_than_the_weights_is_not_used(lib):
    old = lib.root / "exports" / "run1-f16" / "model-f16.gguf"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"GGUF-old")
    import os
    os.utime(old, (1, 1))
    src = lib.job()["source"]
    src["f16_candidates"] = [str(old)]
    lib.make(source=src)
    assert len(lib.merges) == 1 and len(lib.calls("convert")) == 1


def test_a_full_fine_tune_is_converted_as_it_is(lib):
    full = lib.root / "run1" / "model"
    full.mkdir(parents=True)
    (full / "config.json").write_text("{}")
    src = {"kind": "run", "run": "run1", "adapter": None, "full_model": str(full), "base": {"repo": "org/base", "family": "qwen3_5"}, "f16_candidates": []}
    lib.make(source=src)
    assert lib.merges == [] and lib.calls("convert")[0]["args"][0] == str(full)


# ------------------------------------------------------------------------------------------------------------------ a model, a GGUF file

def test_a_model_folder_is_converted_like_the_export_does(lib):
    folder = lib.root / "somemodel"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    lib.make(source={"kind": "model", "model": str(folder), "family": "qwen3_5", "name": "Some model"})
    (conv,) = lib.calls("convert")
    assert conv["args"][0] == str(folder) and flag(conv["args"], "--outtype") == "f16"
    info = json.loads((lib.root_lib / "20261115-1030-run1" / "imatrix.json").read_text(encoding="utf-8"))
    assert info["source_kind"] == "model" and "Some model" in info["source"]


def test_a_hugging_face_id_is_fetched_then_converted(lib, monkeypatch):
    folder = lib.root / "hubsnap"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    asked = []
    monkeypatch.setattr(imatrixjob, "fetch_model_folder", lambda repo, im: (asked.append(repo), folder)[1])
    lib.make(source={"kind": "model", "model": "owner/name", "family": "generic", "name": "owner/name"})
    assert asked == ["owner/name"] and lib.calls("convert")[0]["args"][0] == str(folder)


def test_a_hub_id_in_the_local_cache_is_not_downloaded(lib, monkeypatch, tmp_path):
    snap = tmp_path / "hub" / "models--owner--name" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert imatrixjob.fetch_model_folder("owner/name", {}) == snap


@pytest.mark.parametrize("error, words", [("401 Client Error: gated repo", "needs access"), ("404 Repository Not Found", "was not found"), ("Connection refused", "internet connection")])
def test_a_hub_download_that_fails_is_explained(monkeypatch, tmp_path, error, words):
    import types
    mod = types.ModuleType("huggingface_hub")
    mod.snapshot_download = lambda **kw: (_ for _ in ()).throw(RuntimeError(error))
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    with pytest.raises(imatrixjob.MatrixError) as e:
        imatrixjob.fetch_model_folder("owner/name", {})
    assert "owner/name" in str(e.value) and words in str(e.value) and "RuntimeError" in str(e.value)


def test_an_existing_gguf_is_measured_directly_without_converting(lib):
    f = lib.root / "model-q8.gguf"
    f.write_bytes(b"GGUF-q8")
    lib.make(source={"kind": "gguf", "file": str(f), "tokenizer": "org/base", "family": "qwen3_5"})
    assert lib.calls("convert") == [] and lib.merges == []
    assert flag(lib.calls("imatrix")[0]["args"], "-m") == str(f) and f.exists()
    info = json.loads((lib.root_lib / "20261115-1030-run1" / "imatrix.json").read_text(encoding="utf-8"))
    assert info["source_kind"] == "gguf" and "model-q8.gguf" in info["source"]


def test_a_gguf_needs_a_tokenizer_for_the_saved_data_but_not_for_a_text_file(lib):
    f = lib.root / "m.gguf"
    f.write_bytes(b"GGUF-q8")
    msg = result_error(lib, source={"kind": "gguf", "file": str(f)})
    assert "chat template" in msg and "text file of your own" in msg and not (lib.root_lib / "20261115-1030-run1").exists()
    lib.make(source={"kind": "gguf", "file": str(f)}, calibration={"kind": "text", "file": str(lib.own)})
    assert (lib.root_lib / "20261115-1030-run1" / "imatrix.gguf").is_file()


# ------------------------------------------------------------------------------------------------------------------ calibration of the person's own

def test_an_own_text_file_is_used_as_it_is(lib):
    f = lib.root / "m.gguf"
    f.write_bytes(b"GGUF-q8")
    r = lib.make(source={"kind": "gguf", "file": str(f)}, calibration={"kind": "text", "file": str(lib.own)})
    (im,) = lib.calls("imatrix")
    assert flag(im["args"], "-f") == str(lib.own) and im["calibration"] == lib.own.read_text(encoding="utf-8")
    assert "--parse-special" not in im["args"], "plain text has no template tokens"
    assert lib.own.exists() and r["info"]["calibration_kind"] == "text" and r["info"]["general_used"] is False and r["info"]["general_share"] == 0.0
    assert "mytext.txt" in r["info"]["calibration"]


def test_an_own_text_with_template_tokens_is_parsed_special(lib):
    f = lib.root / "m.gguf"
    f.write_bytes(b"GGUF-q8")
    t = lib.root / "chat.txt"
    t.write_text("<|im_start|>user\nhello<|im_end|>\n" * 50, encoding="utf-8")
    lib.make(source={"kind": "gguf", "file": str(f)}, calibration={"kind": "text", "file": str(t)})
    assert "--parse-special" in lib.calls("imatrix")[0]["args"]


# ------------------------------------------------------------------------------------------------------------------ the options

def test_chunks_ctx_and_ngl_reach_llama_imatrix(lib):
    lib.make(chunks=256, ctx=512, ngl=0)
    a = lib.calls("imatrix")[0]["args"]
    assert flag(a, "--chunks") == "256" and flag(a, "-c") == "512" and flag(a, "-ngl") == "0"


def test_the_head_option_adds_nextn_and_a_batch_and_retries_without_when_refused(lib, monkeypatch):
    f = lib.root / "m.gguf"
    f.write_bytes(gguf({"qwen3.nextn_predict_layers": 1}, ["blk.0.attn_k.weight"]))
    src = {"kind": "gguf", "file": str(f)}
    cal = {"kind": "text", "file": str(lib.own)}
    r = lib.make(source=src, calibration=cal, nextn=True)
    a = lib.calls("imatrix")[0]["args"]
    assert "--nextn" in a and flag(a, "-b") == "128" and flag(a, "-c") == "128" and r["info"]["nextn"] is True
    assert len(lib.calls("imatrix-help")) == 1
    monkeypatch.setenv("NEXTN_MODE", "refuse")
    r = imatrixjob.make(lib.job(source=src, calibration=cal, nextn=True, id="20261115-1031-run1"))
    calls = lib.calls("imatrix")
    assert "--nextn" in calls[1]["args"] and "--nextn" not in calls[2]["args"] and r["info"]["nextn"] is False and r["info"]["nextn_asked"] is True
    assert any(n.startswith("The MTP head was left out") for n in r["notes"])


def test_a_model_without_a_head_is_measured_without_nextn_and_says_so(lib):
    f = lib.root / "m.gguf"
    f.write_bytes(gguf({"general.architecture": "qwen3"}, ["blk.0.attn_k.weight"]))
    r = lib.make(source={"kind": "gguf", "file": str(f)}, calibration={"kind": "text", "file": str(lib.own)}, nextn=True)
    assert "--nextn" not in lib.calls("imatrix")[0]["args"] and "no speculative-decoding head" in " ".join(r["notes"])


def test_the_legacy_dat_is_made_from_the_gguf_with_the_documented_command(lib):
    r = lib.make(dat=True)
    folder = lib.root_lib / "20261115-1030-run1"
    (c,) = lib.calls("imatrix-infile")[-1:]
    a = c["args"]
    assert flag(a, "--in-file") == str(folder / "imatrix.gguf") and flag(a, "--output-format") == "dat" and Path(flag(a, "-o")).name == "imatrix.partial.dat"
    assert (folder / "imatrix.dat").read_bytes() == b"LEGACY-DAT" and not (folder / "imatrix.partial.dat").exists()
    assert r["dat"] == str(folder / "imatrix.dat") and r["info"]["has_dat"] is True
    assert json.loads((folder / "imatrix.json").read_text(encoding="utf-8"))["has_dat"] is True


def test_the_dat_falls_back_to_the_extension_and_then_to_a_note(lib, monkeypatch):
    monkeypatch.setenv("DAT_MODE", "noflag")
    r = lib.make(dat=True)
    conv = [c["args"] for c in lib.calls("imatrix-infile") if "--show-statistics" not in c["args"]]
    assert "--output-format" in conv[0] and "--output-format" not in conv[1] and r["info"]["has_dat"] is True
    monkeypatch.setenv("DAT_MODE", "never")
    r = imatrixjob.make(lib.job(dat=True, id="20261115-1031-run1"))
    assert r["dat"] is None and r["info"]["has_dat"] is False and any("legacy .dat" in n for n in r["notes"]) and (lib.root_lib / "20261115-1031-run1" / "imatrix.gguf").is_file()


def test_a_dat_can_be_made_later_for_a_saved_entry(lib, monkeypatch):
    lib.make()
    folder = lib.root_lib / "20261115-1030-run1"
    r = imatrixjob.make_dat_job({"dir": str(folder), "stop_file": str(lib.root / "STOP")})
    assert (folder / "imatrix.dat").is_file() and r["dat"] == str(folder / "imatrix.dat")
    assert json.loads((folder / "imatrix.json").read_text(encoding="utf-8"))["has_dat"] is True
    monkeypatch.setenv("DAT_MODE", "never")
    (folder / "imatrix.dat").unlink()
    with pytest.raises(imatrixjob.MatrixError, match="legacy .dat"):
        imatrixjob.make_dat_job({"dir": str(folder), "stop_file": str(lib.root / "STOP")})
    assert not (folder / "imatrix.partial.dat").exists() and (folder / "imatrix.gguf").is_file(), "the matrix itself is untouched"
    with pytest.raises(imatrixjob.MatrixError, match="imatrix.gguf of this entry"):
        imatrixjob.make_dat_job({"dir": str(lib.root / "nothing")})


# ------------------------------------------------------------------------------------------------------------------ statistics

def test_the_statistics_are_read_when_supported(lib):
    r = lib.make()
    st = r["statistics"]
    assert st["entries"] == 3 and "Tensors measured: 3" in st["lines"] and "Largest importance value: 8" in st["lines"]
    assert any(l.startswith("Mean importance") and "1" in l for l in st["lines"]) and any("% on average" in l for l in st["lines"])
    a = [c["args"] for c in lib.calls("imatrix-infile")][0]
    assert "--show-statistics" in a and flag(a, "--in-file").endswith("imatrix.gguf") and flag(a, "-m").endswith("model-f16.gguf")
    assert json.loads((lib.root_lib / "20261115-1030-run1" / "imatrix.json").read_text(encoding="utf-8"))["statistics"]["entries"] == 3


def test_without_the_statistics_option_of_this_build_nothing_is_missing(lib, monkeypatch):
    monkeypatch.setenv("STATS_MODE", "unsupported")
    r = lib.make()
    assert r["statistics"] is None and (lib.root_lib / "20261115-1030-run1" / "imatrix.gguf").is_file()
    r = imatrixjob.make(lib.job(statistics=False, id="20261115-1031-run1"))
    assert r["statistics"] is None


@pytest.mark.parametrize("lines", [[], ["nonsense"], ["Layer\tTensor\tMin\tMax"], ["", "\t\t"], ["Computing statistics (x tensors)"], [None]])
def test_odd_statistics_text_never_raises(lines):
    assert imatrixjob.parse_statistics(lines) in (None, {"lines": [], "entries": None}) or isinstance(imatrixjob.parse_statistics(lines), dict)


def test_statistics_with_a_count_only(lines=("Computing statistics for x.gguf (320 tensors)",)):
    assert imatrixjob.parse_statistics(lines) == {"lines": ["Tensors measured: 320"], "entries": 320}


# ------------------------------------------------------------------------------------------------------------------ Stop and failures

def test_stop_kills_llama_imatrix_and_leaves_no_folder(lib, monkeypatch):
    monkeypatch.setenv("IMATRIX_MODE", "hang")
    t = _stop_when_started(lib)
    t0 = time.time()
    with pytest.raises(imatrix.Stopped):
        lib.make()
    t.join()
    assert time.time() - t0 < 15 and not (lib.root / "started.finished").exists()
    assert not (lib.root_lib / "20261115-1030-run1").exists()


def test_stop_before_the_measuring_ends_cleanly(lib):
    (lib.root / "STOP").write_text("")
    with pytest.raises(imatrix.Stopped):
        lib.make()
    assert lib.calls("imatrix") == [] and not (lib.root_lib / "20261115-1030-run1").exists()


def test_the_worker_entry_reports_stop_failure_and_the_unexpected_in_plain_words(lib, monkeypatch):
    jobfile = lib.root / "job.json"
    jobfile.write_text(json.dumps(lib.job()), encoding="utf-8")
    monkeypatch.setattr(imatrixjob, "make", lambda job: (_ for _ in ()).throw(imatrix.Stopped()))
    assert entry.main(["x", str(jobfile)]) == 1
    monkeypatch.setattr(imatrixjob, "make", lambda job: (_ for _ in ()).throw(imatrixjob.MatrixError("plain words")))
    assert entry.main(["x", str(jobfile)]) == 1
    monkeypatch.setattr(imatrixjob, "make", lambda job: (_ for _ in ()).throw(KeyError("source")))
    assert entry.main(["x", str(jobfile)]) == 1
    errs = [e for e in lib.events if e["type"] == "error"]
    assert errs[0] == {"type": "error", "code": "imatrix_stopped", "message": "Cancelled."} and errs[1] == {"type": "error", "code": "imatrix_failed", "message": "plain words"}
    assert errs[2]["code"] == "imatrix_failed" and errs[2]["message"].startswith("The matrix could not be made (KeyError")


def test_the_worker_entry_runs_a_whole_job_and_a_dat_job(lib):
    jobfile = lib.root / "job.json"
    jobfile.write_text(json.dumps(lib.job()), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0
    res = [e for e in lib.events if e["type"] == "result"]
    assert res[0]["data"]["id"] == "20261115-1030-run1"
    jobfile.write_text(json.dumps({"job": "imatrix_dat", "dir": str(lib.root_lib / "20261115-1030-run1"), "stop_file": str(lib.root / "STOP")}), encoding="utf-8")
    assert entry.main(["x", str(jobfile)]) == 0 and (lib.root_lib / "20261115-1030-run1" / "imatrix.dat").is_file()


def test_no_llama_cpp_says_how_to_install_it_and_where_the_folder_is_read(lib, monkeypatch):
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(lib.root / "nowhere"))
    monkeypatch.setenv("PATH", str(lib.root / "nowhere"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: lib.root / "nohome"))
    msg = result_error(lib)
    assert "https://github.com/ggml-org/llama.cpp" in msg and "LLAMA_CPP_DIR" in msg and "convert_hf_to_gguf.py" in msg and not lib.root_lib.exists()
    assert "convert_hf_to_gguf.py" not in result_error(lib, source={"kind": "gguf", "file": "x.gguf"})


def test_llama_imatrix_alone_is_found_through_llama_cpp_dir_for_a_gguf(lib, monkeypatch):
    import os
    exe = "llama-imatrix.exe" if os.name == "nt" else "llama-imatrix"
    d = lib.root / "lcpp2" / "build" / "bin"
    d.mkdir(parents=True)
    (d / exe).write_text("x")
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    monkeypatch.setattr(imatrix, "find_llama_imatrix", lambda q: None)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(lib.root / "lcpp2"))
    assert imatrixjob.find_tools(need_convert=False) == (None, d / exe)
    assert "convert_hf_to_gguf.py" in str(pytest.raises(imatrixjob.MatrixError, imatrixjob.find_tools, True).value)


def test_a_model_that_cannot_be_converted_is_explained(lib, monkeypatch):
    monkeypatch.setenv("CONVERT_MODE", "fail")
    msg = result_error(lib)
    assert "could not turn this model into a GGUF" in msg and "safetensors" in msg and not (lib.root_lib / "20261115-1030-run1").exists()


def test_a_full_disk_while_converting_is_explained(lib, monkeypatch):
    monkeypatch.setattr(imatrix, "run_streaming", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.ENOSPC, "No space left on device")))
    assert "The disk is full" in result_error(lib)


def test_not_enough_free_space_is_seen_before_converting(lib, monkeypatch):
    folder = lib.root / "somemodel"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"x" * 2000)
    monkeypatch.setattr(imatrixjob, "free_bytes", lambda p: 100)
    msg = result_error(lib, source={"kind": "model", "model": str(folder), "name": "m"})
    assert "not enough free disk space" in msg and lib.calls("convert") == []


def test_llama_imatrix_failing_is_explained_and_leaves_nothing(lib, monkeypatch):
    monkeypatch.setenv("IMATRIX_MODE", "fail")
    msg = result_error(lib)
    assert "llama-imatrix failed (exit code 3" in msg and "Update llama.cpp" in msg and not (lib.root_lib / "20261115-1030-run1").exists()


@pytest.mark.parametrize("what, words", [
    (dict(calibration={"kind": "text", "file": "/nowhere/x.txt"}), "was not found"),
    (dict(calibration={"kind": "data", "train": "/nowhere/train.jsonl"}), "training data"),
    (dict(calibration={"kind": "nonsense"}), "Choose the calibration text"),
    (dict(source={"kind": "nonsense"}), "Choose where the model comes from"),
    (dict(source={"kind": "model", "model": ""}), "Choose a model"),
    (dict(source={"kind": "model", "model": "nowhere"}), "was not found"),
    (dict(source={"kind": "gguf", "file": "/nowhere/m.gguf"}), "was not found"),
    (dict(chunks=99), "64, 128 or 256"),
    (dict(id="../evil"), "not a valid name"),
])
def test_bad_input_gives_a_plain_message_and_makes_nothing(lib, what, words):
    assert words in result_error(lib, **what)
    assert not list(lib.root_lib.glob("*")) and lib.calls("imatrix") == []


def test_an_empty_text_file_and_a_folder_without_config_are_explained(lib, tmp_path):
    empty = lib.root / "empty.txt"
    empty.write_text("  \n")
    f = lib.root / "m.gguf"
    f.write_bytes(b"GGUF-q8")
    assert "is empty" in result_error(lib, source={"kind": "gguf", "file": str(f)}, calibration={"kind": "text", "file": str(empty)})
    bare = lib.root / "bare"
    bare.mkdir()
    assert "no config.json" in result_error(lib, source={"kind": "model", "model": str(bare), "name": "b"})


def test_a_tokenizer_that_cannot_be_read_is_explained(lib, monkeypatch):
    monkeypatch.setattr(imatrix, "load_tokenizer", lambda sources: (_ for _ in ()).throw(OSError("no tokenizer files")))
    msg = result_error(lib)
    assert "chat template" in msg and "text file of your own" in msg and not (lib.root_lib / "20261115-1030-run1").exists()


def test_data_without_any_text_is_explained(lib):
    bad = write_rows(lib.root / "empty.jsonl", [{"type": "text", "text": "  "}])
    msg = result_error(lib, calibration={"kind": "data", "train": bad, "general_share": 0.0})
    assert "no text to measure with" in msg


def test_a_second_matrix_in_the_same_minute_does_not_overwrite_the_first(lib):
    lib.make()
    assert "exists already" in result_error(lib) and (lib.root_lib / "20261115-1030-run1" / "imatrix.gguf").is_file()


# ------------------------------------------------------------------------------------------------------------------ one function for both paths

def test_the_export_and_the_new_job_call_the_same_make_matrix(lib, monkeypatch):
    seen = []
    real = imatrix.make_matrix
    monkeypatch.setattr(imatrix, "make_matrix", lambda *a, **k: (seen.append("x"), real(*a, **k))[1])
    lib.export(["q4_k_m"])
    lib.make()
    assert seen == ["x", "x"]


def test_the_general_text_failing_is_a_note_and_the_matrix_is_still_made(lib):
    r = lib.make(calibration={"kind": "data", "train": lib.train, "data_version": 3, "general_share": 0.2, "general": {"file": str(lib.root / "missing.jsonl")}})
    assert r["info"]["general_used"] is False and any("general text" in n.lower() for n in r["notes"]) and (lib.root_lib / "20261115-1030-run1" / "imatrix.gguf").is_file()


# ------------------------------------------------------------------------------------------------------------------ the GGUF reader and the compatibility check

def test_the_header_reader_reads_keys_and_tensor_names(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(gguf({"general.architecture": "qwen3", "qwen3.block_count": 36}, ["blk.0.a", "blk.35.b"]))
    meta, names = imatrix.read_gguf_header(f)
    assert meta == {"general.architecture": "qwen3", "qwen3.block_count": 36} and names == ["blk.0.a", "blk.35.b"]
    assert imatrix.gguf_model_facts(f) == ("qwen3", 36)


@pytest.mark.parametrize("blob", [b"", b"GGU", b"NOPE" + b"\0" * 40, b"GGUF" + struct.pack("<I", 1) + b"\0" * 16, b"GGUF" + struct.pack("<IQQ", 3, 1 << 40, 0), gguf({"a": 1}, ["t"])[:-5]])
def test_a_file_that_is_no_readable_gguf_raises_value_error_and_gives_none(tmp_path, blob):
    f = tmp_path / "x"
    f.write_bytes(blob)
    with pytest.raises(ValueError):
        imatrix.read_gguf_header(f)
    assert imatrix.matrix_summary(f) is None and imatrix.gguf_model_facts(f) == (None, None) and imatrix.gguf_has_mtp(f) is None
    assert imatrix.matrix_summary(tmp_path / "missing") is None


def test_the_matrix_summary_counts_layers_and_entries(tmp_path):
    f = tmp_path / "imatrix.gguf"
    f.write_bytes(matrix_gguf(layers=3, arch="qwen3"))
    assert imatrix.matrix_summary(f) == {"arch": "qwen3", "layers": 3, "entries": 3, "chunks": 64, "chunk_size": None, "datasets": [], "kinds": {"attn_k": 3}, "hidden": None,
                                         "intermediate": None, "tokens": None}      # a header with no data behind it: no counts to read


def test_the_compatibility_check_warns_but_never_refuses(tmp_path):
    f = tmp_path / "imatrix.gguf"
    f.write_bytes(matrix_gguf(layers=2, arch="llama"))
    assert imatrix.compat_warnings(f, "llama", 2) == []
    assert imatrix.compat_warnings(f, "llama", 3, mtp_layers=0)[0].startswith("This matrix has data for 2 layers, but the model has 3")
    assert imatrix.compat_warnings(f, "llama", 1, mtp_layers=1) == [], "a head adds a layer"
    (w,) = imatrix.compat_warnings(f, "qwen3", 2)
    assert 'architecture "llama"' in w and '"qwen3"' in w
    assert imatrix.compat_warnings(f, None, None) == [], "nothing to compare"
    legacy = tmp_path / "imatrix.dat"
    legacy.write_bytes(b"\x01\x00\x00\x00 legacy")
    (w,) = imatrix.compat_warnings(legacy, "llama", 2)
    assert "could not be read here" in w and "llama-quantize will tell" in w


def test_model_facts_from_a_folder(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 32, "mtp_num_hidden_layers": 1}}), encoding="utf-8")
    assert imatrix.folder_model_facts(tmp_path) == ("qwen3_5_text", 32, 1)
    assert imatrix.folder_model_facts(tmp_path / "none") == (None, None, 0)


# ------------------------------------------------------------------------------------------------------------------ "use a saved matrix" in an export

def library_entry(lib, name="20261101-1000-run1", data=None, info=None, ext="gguf"):
    d = lib.root_lib / name
    d.mkdir(parents=True)
    f = d / f"imatrix.{ext}"
    f.write_bytes(data if data is not None else b"SAVED-MATRIX")
    if info is not None:
        (d / "imatrix.json").write_text(json.dumps(info), encoding="utf-8")
    return f


def test_an_export_given_a_saved_matrix_measures_nothing_and_passes_that_file(lib):
    f = library_entry(lib, info={"format": 1, "model": "run1", "nextn": False, "chunks": 128})
    files, rep, out = lib.export(["q4_k_m", "q8_0"], use_file=str(f), use_id="20261101-1000-run1")
    assert lib.calls("imatrix") == [] and (out / "imatrix.gguf").read_bytes() == b"SAVED-MATRIX"
    q = {c["args"][-1]: c["args"] for c in lib.calls("quantize")}
    assert q["Q4_K_M"][:2] == ["--imatrix", str(out / "imatrix.gguf")] and "--imatrix" not in q["Q8_0"]
    assert rep["reused"] and not rep["made"] and rep["applied"] == ["q4_k_m"] and rep["from_library"] == "20261101-1000-run1" and rep["file"] == str(out / "imatrix.gguf")
    assert json.loads((out / "imatrix.json").read_text(encoding="utf-8"))["from_library"] == "20261101-1000-run1"
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M+imatrix"


def test_a_legacy_dat_is_given_to_llama_quantize_as_it_is(lib):
    f = library_entry(lib, ext="dat", data=b"\x01legacy")
    files, rep, out = lib.export(["q4_k_m"], use_file=str(f))
    q = lib.calls("quantize")[0]["args"]
    assert q[:2] == ["--imatrix", str(out / "imatrix.dat")] and (out / "imatrix.dat").read_bytes() == b"\x01legacy" and rep["applied"] == ["q4_k_m"]
    assert any(n.startswith("Warning: The matrix file imatrix.dat could not be read here") for n in rep["notes"])


def test_a_matrix_that_does_not_seem_to_fit_is_a_warning_and_is_still_used(lib):
    (lib.merged / "config.json").write_text(json.dumps({"num_hidden_layers": 4, "model_type": "qwen3"}), encoding="utf-8")
    f = library_entry(lib, data=matrix_gguf(layers=2, arch="qwen3"))
    files, rep, out = lib.export(["q4_k_m"], use_file=str(f))
    (w,) = [n for n in rep["notes"] if n.startswith("Warning:")]
    assert "2 layers, but the model has 4" in w and rep["applied"] == ["q4_k_m"] and files["q4_k_m"].read_bytes().endswith(b"+imatrix")


def test_a_matrix_that_fits_gives_no_warning(lib):
    (lib.merged / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "model_type": "qwen3"}), encoding="utf-8")
    f = library_entry(lib, data=matrix_gguf(layers=2, arch="qwen3"))
    _, rep, _ = lib.export(["q4_k_m"], use_file=str(f))
    assert rep["notes"] == [] and rep["applied"] == ["q4_k_m"]


def test_a_saved_matrix_that_vanished_is_a_note_and_the_sizes_are_made_without_it(lib):
    files, rep, out = lib.export(["q4_k_m"], use_file=str(lib.root_lib / "gone" / "imatrix.gguf"))
    assert "saved matrix file is missing" in " ".join(rep["notes"]) and rep["applied"] == [] and files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M"


def test_a_saved_matrix_for_only_q8_is_not_used(lib):
    f = library_entry(lib)
    _, rep, out = lib.export(["q8_0"], use_file=str(f))
    assert "below Q8_0" in " ".join(rep["notes"]) and not (out / "imatrix.gguf").exists()


def test_through_unsloth_the_saved_matrix_needs_no_llama_cpp_at_all(lib, monkeypatch):
    monkeypatch.setattr(exportjob, "find_llama_cpp", lambda: (None, None))
    cfg = lib.cfg(use_file=str(library_entry(lib)))
    rep = exportjob.new_report()
    assert exportjob.plan_gguf("unsloth", cfg, ["q4_k_m"], rep, lambda: True) == (True, True) and rep["notes"] == []
    rep = exportjob.new_report()
    assert exportjob.plan_gguf("unsloth", cfg, ["q4_k_m"], rep, lambda: False) == (True, False) and "was not found" in " ".join(rep["notes"])
    out = lib.root / "u"
    out.mkdir()
    rep = exportjob.new_report()
    m = exportjob.matrix_for_unsloth(cfg, lib.merged, out, ["q4_k_m"], rep)
    assert m == out / "imatrix.gguf" and m.read_bytes() == b"SAVED-MATRIX" and rep["reused"]
