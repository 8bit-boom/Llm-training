"""Merging a healed LoRA into a base model folder with numpy only (no torch, no PEFT)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from fakemodel import fake_qwen_width  # noqa: E402
from stfile import read_folder, write_file  # noqa: E402

from ndforge_worker import safetensors_io as st  # noqa: E402
from ndforge_worker import slice_io  # noqa: E402
from ndforge_worker.reshape import ReshapeError  # noqa: E402

WORKER = Path(__file__).resolve().parents[1]
R, ALPHA = 4, 8


def write_adapter(folder, modules, seed=0, r=R, alpha=ALPHA, **conf):
    """a PEFT-style adapter folder for the given {module path: (out, in)}; returns {module: (A, B)}"""
    rng = np.random.default_rng(seed)
    Path(folder).mkdir(parents=True, exist_ok=True)
    items, mats = [], {}
    for module, (o, i) in modules.items():
        a, b = rng.normal(size=(r, i)).astype("<f4"), rng.normal(size=(o, r)).astype("<f4")
        mats[module] = (a, b)
        items += [(f"base_model.model.{module}.lora_A.weight", "F32", (r, i), a.tobytes()), (f"base_model.model.{module}.lora_B.weight", "F32", (o, r), b.tobytes())]
    write_file(Path(folder) / "adapter_model.safetensors", items)
    (Path(folder) / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": r, "lora_alpha": alpha, "target_modules": "all-linear", **conf}), encoding="utf-8")
    return mats


# the language model's layers are `model.layers.N` in the adapter (a text-only model was trained) but `model.language_model.layers.N` in the folder
MODULES = {"model.layers.0.mlp.gate_proj": (32, 8), "model.layers.3.self_attn.o_proj": (8, 24), "model.layers.1.linear_attn.in_proj_b": (4, 8)}


def bf16_value(raw):
    return slice_io.bf16_to_f32(np.frombuffer(raw, dtype="<u2")).astype(np.float64)


def test_merge_equals_w_plus_scaled_ba(tmp_path):
    src = fake_qwen_width(tmp_path)
    mats = write_adapter(tmp_path / "adapter", MODULES)
    out = tmp_path / "merged"
    res = slice_io.merge_lora(st.list_tensors(src), tmp_path / "adapter", out, shard_bytes=2000)
    assert res == {"merged": 3, "scaling": 2.0, "rank": 4}
    before, after = read_folder(src), read_folder(out)
    assert set(before) == set(after) and (out / "model.safetensors.index.json").is_file()
    for module, (a, b) in mats.items():
        name = "model.language_model." + module[len("model."):] + ".weight"
        dtype, shape, raw = before[name]
        assert after[name][:2] == (dtype, shape)
        want = bf16_value(raw).reshape(shape) + 2.0 * (b.astype(np.float64) @ a.astype(np.float64))
        want_bits = slice_io.f32_to_bf16(want.astype(np.float32)).astype(np.int64)
        got_bits = np.frombuffer(after[name][2], dtype="<u2").astype(np.int64).reshape(shape)
        assert np.abs(want_bits - got_bits).max() <= 1                                          # float32 against float64: at most one bf16 step apart
        assert (got_bits != np.frombuffer(raw, dtype="<u2").reshape(shape)).any()
    changed = {"model.language_model." + m[len("model."):] + ".weight" for m in MODULES}
    for name in before:
        if name not in changed:
            assert after[name] == before[name], name                                            # everything else is byte-identical


def test_merge_float32_and_a_flat_text_model_layout(tmp_path):
    w = np.arange(12, dtype="<f4").reshape(3, 4)
    d = tmp_path / "base"
    d.mkdir()
    write_file(d / "model.safetensors", [("model.layers.0.mlp.up_proj.weight", "F32", (3, 4), w.tobytes()), ("model.norm.weight", "F32", (3,), np.ones(3, "<f4").tobytes())])
    mats = write_adapter(tmp_path / "ad", {"model.layers.0.mlp.up_proj": (3, 4)}, r=2, alpha=1)
    slice_io.merge_lora(st.list_tensors(d), tmp_path / "ad", tmp_path / "out")
    got = np.frombuffer(read_folder(tmp_path / "out")["model.layers.0.mlp.up_proj.weight"][2], dtype="<f4").reshape(3, 4)
    a, b = mats["model.layers.0.mlp.up_proj"]
    assert np.allclose(got, w + 0.5 * (b @ a), atol=1e-5)


def test_merge_refuses_what_does_not_fit_or_is_not_plain_lora(tmp_path):
    src = fake_qwen_width(tmp_path)
    base = st.list_tensors(src)
    cases = [({"model.layers.0.mlp.no_such_proj": (32, 8)}, {}, "does not fit the slim model"),                    # a module the folder does not have
             ({"model.layers.0.mlp.gate_proj": (20, 8)}, {}, "does not fit the slim model"),                       # a wrong shape (an adapter of the unthinned model)
             ({"model.layers.0.mlp.gate_proj": (32, 8)}, {"use_dora": True}, "plain LoRA"),
             ({"model.layers.0.mlp.gate_proj": (32, 8)}, {"rank_pattern": {"gate_proj": 8}}, "plain LoRA"),
             ({"model.layers.0.mlp.gate_proj": (32, 8)}, {"modules_to_save": ["lm_head"]}, "plain LoRA")]
    for n, (modules, conf, text) in enumerate(cases):
        write_adapter(tmp_path / f"ad{n}", modules, **conf)
        with pytest.raises(ReshapeError, match=text):
            slice_io.merge_lora(base, tmp_path / f"ad{n}", tmp_path / f"o{n}")
    with pytest.raises(ReshapeError, match="adapter_config"):
        (tmp_path / "empty").mkdir()
        write_file(tmp_path / "empty" / "adapter_model.safetensors", [("x", "F32", (1,), b"\0\0\0\0")])
        slice_io.merge_lora(base, tmp_path / "empty", tmp_path / "oe")


def test_rslora_uses_the_square_root_scaling(tmp_path):
    src = fake_qwen_width(tmp_path)
    write_adapter(tmp_path / "ad", {"model.layers.0.mlp.gate_proj": (32, 8)}, r=4, alpha=8, use_rslora=True)
    res = slice_io.merge_lora(st.list_tensors(src), tmp_path / "ad", tmp_path / "o")
    assert res["scaling"] == pytest.approx(4.0)


def test_a_stop_during_the_merge_cancels(tmp_path):
    src = fake_qwen_width(tmp_path)
    write_adapter(tmp_path / "ad", MODULES)
    with pytest.raises(st.Cancelled):
        slice_io.merge_lora(st.list_tensors(src), tmp_path / "ad", tmp_path / "o", should_stop=lambda: True)


def run_job(tmp_path, job):
    f = tmp_path / "job.json"
    f.write_text(json.dumps(job), encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "ndforge_worker", str(f)], cwd=WORKER, capture_output=True, text=True, timeout=120)
    return p.returncode, [json.loads(line) for line in p.stdout.splitlines()]


def job_for(tmp_path, src):
    d = tmp_path / "models" / "m1" / "healed"
    d.parent.mkdir(parents=True)
    (src / "ndforge_reshape.json").write_text(json.dumps({"format": 2, "origin": {"repo": "unsloth/Qwen3.5-4B", "family": "qwen3_5"}, "params": 123}), encoding="utf-8")
    (src / "width_kept.npz").write_bytes(b"npz")
    return {"job": "distill_merge", "base": str(src), "adapter": str(tmp_path / "adapter"), "out": str(d.parent / "healed.partial"), "final": str(d),
            "summary": {"mode": "cache", "steps": 20, "tokens_seen": 999, "kd_first": 0.9, "kd_last": 0.3, "calib_loss_before": 3.0, "calib_loss_after": 2.4, "teacher_calib_loss": 2.0,
                        "unrelated": "left out"}}


def test_the_merge_job_writes_a_self_contained_folder(tmp_path):
    src = fake_qwen_width(tmp_path)
    write_adapter(tmp_path / "adapter", MODULES)
    j = job_for(tmp_path, src)
    code, ev = run_job(tmp_path, j)
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    r = ev[-1]["data"]
    final = Path(j["final"])
    assert r["model_dir"] == str(final) and r["merged"] == 3 and not Path(j["out"]).exists() and r["stopped"] is False
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "README.md", "ndforge_reshape.json", "width_kept.npz"):
        assert (final / f).is_file(), f
    assert (final / "config.json").read_bytes() == (src / "config.json").read_bytes() and (final / "README.md").read_text().startswith("---\nlicense: apache-2.0\n---")
    meta = json.loads((final / "ndforge_reshape.json").read_text())
    assert meta["origin"]["repo"] == "unsloth/Qwen3.5-4B" and meta["params"] == 123                       # the origin survives (the vision file and the teacher are found through it)
    h = meta["healed"]
    assert h["merged_weights"] == 3 and h["lora_rank"] == 4 and h["steps"] == 20 and h["calib_loss_after"] == 2.4 and "unrelated" not in h
    assert set(st.list_tensors(final)) == set(st.list_tensors(src))


def test_the_merge_job_says_plainly_when_the_adapter_does_not_fit(tmp_path):
    src = fake_qwen_width(tmp_path)
    write_adapter(tmp_path / "adapter", {"model.layers.0.mlp.gate_proj": (20, 8)})
    j = job_for(tmp_path, src)
    code, ev = run_job(tmp_path, j)
    assert code == 1 and ev[-1]["code"] == "reshape_refused" and "problem report" in ev[-1]["message"]
    assert not Path(j["out"]).exists() and not Path(j["final"]).exists()


def test_a_stop_file_cancels_the_merge_job(tmp_path):
    src = fake_qwen_width(tmp_path)
    write_adapter(tmp_path / "adapter", MODULES)
    j = job_for(tmp_path, src)
    (Path(j["out"]).parent / "STOP").write_text("")
    code, ev = run_job(tmp_path, j)
    assert code == 0 and ev[-1]["data"] == {"stopped": True, "model_dir": None} and not Path(j["final"]).exists()
