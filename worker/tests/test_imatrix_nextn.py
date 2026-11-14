"""--nextn: the importance matrix also covers the model's speculative-decoding (MTP / NextN) head. Same stand-in llama.cpp tools as test_imatrix.py (small Python
scripts, the same on Windows and Linux); no torch, no llama.cpp, no network."""
import json
import struct
from pathlib import Path

import pytest
from stfile import write_file
from test_imatrix import StubTok, world  # noqa: F401  (world is a fixture)

from ndforge_worker import imatrix


def put_head(world, how="config"):
    """gives the stand-in merged model an MTP head"""
    if how == "config":
        (world.merged / "config.json").write_text(json.dumps({"text_config": {"mtp_num_hidden_layers": 1}}), encoding="utf-8")
    else:
        write_file(world.merged / "model.safetensors", [("mtp.layers.0.mlp.up_proj.weight", "F16", [2], b"\0\0\0\0")])


def imatrix_calls(world):
    return [c["args"] for c in world.calls("imatrix")]


# ------------------------------------------------------------------------------------------------------------------ is there a head?

def test_the_config_keys_of_a_head():
    assert imatrix.mtp_in_config({"mtp_num_hidden_layers": 1}) and imatrix.mtp_in_config({"text_config": {"num_nextn_predict_layers": 2}})
    assert not imatrix.mtp_in_config({"mtp_num_hidden_layers": 0}) and not imatrix.mtp_in_config({}) and not imatrix.mtp_in_config(None)
    assert not imatrix.mtp_in_config({"mtp_num_hidden_layers": True}) and not imatrix.mtp_in_config({"text_config": "x"})


def test_the_tensor_names_of_a_head():
    assert imatrix.mtp_in_tensor_names(["model.layers.0.w", "mtp.layers.0.w"]) and imatrix.mtp_in_tensor_names(["model.mtp.fc.weight"])
    assert not imatrix.mtp_in_tensor_names(["model.layers.0.w", "lm_head.weight", "model.language_model.mtp_like.w"])


def test_a_model_folder_with_and_without_a_head(tmp_path):
    none, cfg, tens, idx = (tmp_path / n for n in ("none", "cfg", "tens", "idx"))
    for d in (none, cfg, tens, idx):
        d.mkdir()
    assert imatrix.folder_has_mtp(tmp_path / "nowhere") is None and imatrix.folder_has_mtp(none) is None          # nothing to read: cannot tell
    (none / "config.json").write_text("{}")
    assert imatrix.folder_has_mtp(none) is False
    write_file(none / "model.safetensors", [("model.layers.0.w", "F16", [2], b"\0\0\0\0")])
    assert imatrix.folder_has_mtp(none) is False
    (cfg / "config.json").write_text(json.dumps({"mtp_num_hidden_layers": 1}))
    assert imatrix.folder_has_mtp(cfg) is True
    (tens / "config.json").write_text("{}")
    write_file(tens / "model-00001.safetensors", [("mtp.layers.0.w", "F16", [2], b"\0\0\0\0")])
    assert imatrix.folder_has_mtp(tens) is True
    (idx / "config.json").write_text("{}")
    (idx / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.layers.0.w": "a.safetensors", "mtp.fc.weight": "b.safetensors"}}))
    assert imatrix.folder_has_mtp(idx) is True                                  # the index is enough: no shard is opened


def _gguf(path, keys=(), tensors=(), version=3):
    """a tiny GGUF header: keys are (name, u32 value) pairs, tensors are names (one dimension of 2 elements, type 0)"""
    def s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw
    blob = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(keys) + 2)
    blob += s("general.architecture") + struct.pack("<I", 8) + s("qwen35")
    blob += s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + s("a") + s("bb")        # an array of strings is skipped
    for name, v in keys:
        blob += s(name) + struct.pack("<II", 4, v)
    for t in tensors:
        blob += s(t) + struct.pack("<IQIQ", 1, 2, 0, 0)
    Path(path).write_bytes(blob)


def test_a_gguf_with_and_without_a_head(tmp_path):
    f = tmp_path / "m.gguf"
    _gguf(f, keys=[("qwen35.block_count", 24)], tensors=["blk.0.attn_q.weight"])
    assert imatrix.gguf_has_mtp(f) is False
    _gguf(f, keys=[("qwen35.nextn_predict_layers", 1)], tensors=["blk.0.attn_q.weight"])
    assert imatrix.gguf_has_mtp(f) is True
    _gguf(f, keys=[("qwen35.nextn_predict_layers", 0)], tensors=["blk.24.nextn.eh_proj.weight"])
    assert imatrix.gguf_has_mtp(f) is True                                      # a tensor of the head
    f.write_bytes(b"GGUF-f16")                                                  # the stand-in tools' files, a truncated file, not a GGUF at all
    assert imatrix.gguf_has_mtp(f) is None
    f.write_bytes(b"hello")
    assert imatrix.gguf_has_mtp(f) is None and imatrix.gguf_has_mtp(tmp_path / "missing.gguf") is None


def test_the_converted_file_is_the_best_witness_then_the_folders(world, tmp_path):
    cfg = world.cfg()
    assert imatrix.model_has_mtp(cfg, world.merged) is False                    # config.json is "{}"
    put_head(world)
    assert imatrix.model_has_mtp(cfg, world.merged) is True
    f16 = tmp_path / "f.gguf"
    _gguf(f16, tensors=["blk.0.attn_q.weight"])                                 # the converter dropped the head: what llama-imatrix reads decides
    assert imatrix.model_has_mtp(cfg, world.merged, f16) is False
    f16.write_bytes(b"GGUF-f16")                                                # unreadable: fall back to the folder
    assert imatrix.model_has_mtp(cfg, world.merged, f16) is True
    assert imatrix.model_has_mtp(cfg, tmp_path / "nowhere") is None


def test_a_base_model_in_the_huggingface_cache(world, tmp_path, monkeypatch):
    snap = tmp_path / "hub" / "models--org--base" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"mtp_num_hidden_layers": 1}))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert imatrix.hf_cache_dirs("org/base") == [snap] and imatrix.hf_cache_dirs("org/other") == [] and imatrix.hf_cache_dirs("nothing") == []
    assert imatrix.model_has_mtp(world.cfg(), tmp_path / "no-merged-yet") is True


# ------------------------------------------------------------------------------------------------------------------ the command line

def test_nextn_adds_the_flag_and_a_batch_not_larger_than_the_context(world):
    put_head(world)
    files, rep, out = world.export(["q4_k_m"], nextn=True)
    (args,) = imatrix_calls(world)
    assert args.count("--nextn") == 1 and args[args.index("-b") + 1] == args[args.index("-c") + 1] == "128"
    assert [c["tool"] for c in world.calls()].count("imatrix-help") == 1                         # asked once
    assert rep["made"] and rep["nextn"] is True and rep["notes"] == []
    info = json.loads((out / "imatrix.json").read_text(encoding="utf-8"))
    assert info["nextn"] is True and info["nextn_asked"] is True and info["nextn_note"] is None


def test_without_the_option_nothing_is_probed_or_added(world):
    put_head(world)
    files, rep, out = world.export(["q4_k_m"])
    (args,) = imatrix_calls(world)
    assert "--nextn" not in args and "-b" not in args and [c["tool"] for c in world.calls()].count("imatrix-help") == 0
    info = json.loads((out / "imatrix.json").read_text(encoding="utf-8"))
    assert info["nextn"] is False and rep["nextn"] is False and rep["notes"] == []


def test_a_model_without_a_head_is_measured_as_before_and_it_is_said(world):
    files, rep, out = world.export(["q4_k_m"], nextn=True)
    (args,) = imatrix_calls(world)
    assert "--nextn" not in args and rep["made"] and rep["nextn"] is False
    assert rep["notes"] == ["The MTP head was left out: the model has no speculative-decoding head."]
    assert world.calls("imatrix-help") == []                                    # nothing to ask when there is no head


def test_an_llama_imatrix_without_the_option_leaves_the_head_out(world, monkeypatch):
    put_head(world)
    monkeypatch.setenv("NEXTN_MODE", "unsupported")
    files, rep, out = world.export(["q4_k_m"], nextn=True)
    (args,) = imatrix_calls(world)
    assert "--nextn" not in args and "-b" not in args and rep["made"] and rep["nextn"] is False
    assert rep["notes"] == ["The MTP head was left out: this llama-imatrix has no --nextn option (an older build)."]
    assert files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M+imatrix"                    # the sizes still get the matrix
    info = json.loads((out / "imatrix.json").read_text(encoding="utf-8"))
    assert info["nextn"] is False and info["nextn_asked"] is False


def test_a_refusal_is_tried_once_more_without_the_head(world, monkeypatch):
    put_head(world)
    monkeypatch.setenv("NEXTN_MODE", "refuse")
    files, rep, out = world.export(["q4_k_m"], nextn=True)
    first, second = imatrix_calls(world)
    assert "--nextn" in first and "--nextn" not in second and "-b" not in second
    expected = [x for x in first if x != "--nextn"]
    del expected[expected.index("-b"):expected.index("-b") + 2]
    assert second == expected                                                    # the plain command line of a model without a head
    assert rep["made"] and rep["nextn"] is False and files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M+imatrix"
    (note,) = rep["notes"]
    assert note.startswith("The MTP head was left out: ") and "KV cache" in note
    info = json.loads((out / "imatrix.json").read_text(encoding="utf-8"))
    assert info["nextn"] is False and info["nextn_asked"] is True and "KV cache" in info["nextn_note"]


def test_another_failure_with_the_option_is_not_retried(world, monkeypatch):
    put_head(world)
    monkeypatch.setenv("NEXTN_MODE", "other")
    files, rep, out = world.export(["q4_k_m"], nextn=True)
    assert len(imatrix_calls(world)) == 1 and not rep["made"]
    assert "llama-imatrix failed" in rep["notes"][0] and files["q4_k_m"].read_bytes() == b"GGUF-Q4_K_M"       # never fatal: made without the matrix


def test_the_old_output_format_retry_keeps_the_head_flags(world, monkeypatch):
    put_head(world)
    monkeypatch.setenv("IMATRIX_MODE", "old")
    _, rep, _ = world.export(["q4_k_m"], nextn=True)
    first, second = imatrix_calls(world)
    assert "--output-format" in first and "--output-format" not in second and "--nextn" in second and "-b" in second and rep["nextn"] is True


# ------------------------------------------------------------------------------------------------------------------ the reuse key

def test_a_matrix_is_reused_only_for_the_same_head_choice(world):
    put_head(world)
    _, rep1, out1 = world.export(["q4_k_m"], nextn=True)
    assert rep1["made"] and rep1["nextn"]
    _, rep2, out2 = world.export(["q4_k_m"], out=world.root / "o2", reuse_dirs=[str(out1)], nextn=True)
    assert rep2["reused"] and rep2["nextn"] is True and len(imatrix_calls(world)) == 1
    _, rep3, out3 = world.export(["q4_k_m"], out=world.root / "o3", reuse_dirs=[str(out1)], nextn=False)       # one with the head is not one without it
    assert rep3["made"] and not rep3["reused"] and rep3["nextn"] is False
    _, rep4, _ = world.export(["q4_k_m"], out=world.root / "o4", reuse_dirs=[str(out3)], nextn=True)             # nor the reverse
    assert rep4["made"] and not rep4["reused"] and rep4["nextn"] is True
    assert len(imatrix_calls(world)) == 3


def test_a_matrix_saved_before_the_key_existed_counts_as_without_the_head(world):
    _, rep1, out1 = world.export(["q4_k_m"])
    info = json.loads((out1 / "imatrix.json").read_text(encoding="utf-8"))
    del info["nextn"], info["nextn_asked"], info["nextn_note"]
    (out1 / "imatrix.json").write_text(json.dumps(info), encoding="utf-8")
    _, rep2, _ = world.export(["q4_k_m"], out=world.root / "o2", reuse_dirs=[str(out1)])
    assert rep2["reused"] and rep2["nextn"] is False
    put_head(world)
    _, rep3, _ = world.export(["q4_k_m"], out=world.root / "o3", reuse_dirs=[str(out1)], nextn=True)
    assert rep3["made"] and not rep3["reused"]


def test_a_refused_head_does_not_make_the_next_export_measure_again(world, monkeypatch):
    put_head(world)
    monkeypatch.setenv("NEXTN_MODE", "refuse")
    _, rep1, out1 = world.export(["q4_k_m"], nextn=True)
    assert rep1["nextn"] is False and len(imatrix_calls(world)) == 2
    _, rep2, _ = world.export(["q4_k_m"], out=world.root / "o2", reuse_dirs=[str(out1)], nextn=True)
    assert rep2["reused"] and rep2["nextn"] is False and len(imatrix_calls(world)) == 2
    assert rep2["notes"] and rep2["notes"][0].startswith("The MTP head was left out: ")


def test_a_stop_during_the_head_measuring_still_ends_the_export(world, monkeypatch):
    import threading
    import time
    put_head(world)
    monkeypatch.setenv("IMATRIX_MODE", "hang")
    (world.root / "started").unlink(missing_ok=True)

    def waiter():
        for _ in range(400):
            if (world.root / "started").exists():
                break
            time.sleep(0.05)
        (world.root / "STOP").write_text("")

    threading.Thread(target=waiter, daemon=True).start()
    with pytest.raises(imatrix.Stopped):
        world.export(["q4_k_m"], nextn=True)
    assert not (world.root / "out" / "imatrix.partial.gguf").exists()
