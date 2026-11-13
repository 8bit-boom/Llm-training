"""Width pruning with real (tiny, random) Qwen3.5 models on a CPU. Skipped when torch / transformers are not installed (CI has them)."""
import io
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from test_reshape_torch import IDS, load  # noqa: E402
from test_reshape_width import write_scores, wjob  # noqa: E402
from tinymodel import build_arch_dir, sample_rows, write_jsonl  # noqa: E402

from ndforge_worker import reshape_width as rw  # noqa: E402
from ndforge_worker import reshapejob  # noqa: E402
from test_reshape import run_job  # noqa: E402

HEADS, KV, NV = 6, 2, 6


def logits_of(model):
    with torch.no_grad():
        return model(input_ids=IDS, use_cache=False).logits.float()


def perturb(path, seed=1):
    """non-trivial weights everywhere (the norm weights of a fresh model are zero-centred), saved back in place"""
    m = load(path)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * (0.3 if p.dim() == 1 else 0.05))
    m.save_pretrained(path)


@pytest.fixture(scope="module")
def base8(tmp_path_factory):
    d = build_arch_dir(tmp_path_factory.mktemp("w8") / "base", "qwen3_5", num_layers=8, heads=HEADS, kv_heads=KV, lin_v_heads=NV)
    perturb(d)
    return d


def tiny_dims(src):
    return rw.width_dims(json.loads((Path(src) / "config.json").read_text(encoding="utf-8")))


def zero_columns(w, keep_columns):
    gone = torch.ones(w.shape[1], dtype=torch.bool)
    gone[torch.as_tensor(list(keep_columns), dtype=torch.long)] = False
    w[:, gone] = 0


def masked(model, kept, dims):
    """the model with the input columns of the output projections of every removed structure set to zero: what slicing must be equal to"""
    for j, layer in enumerate(model.model.layers):
        with torch.no_grad():
            if f"mlp_{j}" in kept.files:
                zero_columns(layer.mlp.down_proj.weight, kept[f"mlp_{j}"])
            if f"heads_{j}" in kept.files:
                hd = dims["head_dim"]
                zero_columns(layer.self_attn.o_proj.weight, [r for h in kept[f"heads_{j}"] for r in range(int(h) * hd, (int(h) + 1) * hd)])
            if f"gdn_{j}" in kept.files:
                dv = dims["lin_v_dim"]
                zero_columns(layer.linear_attn.out_proj.weight, [r for h in kept[f"gdn_{j}"] for r in range(int(h) * dv, (int(h) + 1) * dv)])
    return model


def run_width(tmp_path, src, width, layers=8, **kw):
    write_scores(tmp_path / "models" / "_scores" / "k1", str(src), layers=layers, dims=tiny_dims(src))
    j = wjob(tmp_path, src, width, **kw)
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    return j, ev[-1]["data"]


# ------------------------------------------------------------------------------------------------ 1: sliced == masked

@pytest.mark.parametrize("width", [{"intermediate_size": 80}, {"intermediate_size": 80, "linear_num_value_heads": 4, "num_attention_heads": 4},
                                   {"linear_num_value_heads": 4}, {"num_attention_heads": 4}], ids=["mlp", "mlp+heads+gdn", "gdn only", "heads only"])
def test_the_thinned_model_equals_the_original_with_those_structures_switched_off(tmp_path, base8, width):
    j, r = run_width(tmp_path, base8, width)
    dims = tiny_dims(base8)
    thin = load(j["final"])
    kept = np.load(Path(j["final"]) / "width_kept.npz")
    ref = masked(load(base8), kept, dims)
    cfg = thin.config
    assert cfg.intermediate_size == width.get("intermediate_size", dims["intermediate"]) and cfg.head_dim == dims["head_dim"]
    assert cfg.num_attention_heads == width.get("num_attention_heads", HEADS) and cfg.linear_num_value_heads == width.get("linear_num_value_heads", NV)
    a, b = logits_of(thin), logits_of(ref)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
    assert float((logits_of(load(base8)) - b).abs().max()) > 1e-3          # the test would be empty if switching the structures off changed nothing
    assert r["params_after"] < r["params_before"]


def test_depth_and_width_together_equal_the_layer_cut_model_with_structures_switched_off(tmp_path, base8):
    wide = build_arch_dir(tmp_path / "b16", "qwen3_5", num_layers=16, heads=HEADS, kv_heads=KV, lin_v_heads=NV)
    perturb(wide)
    j, r = run_width(tmp_path, wide, {"intermediate_size": 80, "num_attention_heads": 4}, layers=16, drop_layers=[4, 5, 6, 7])
    assert r["shape"] == "both" and r["layers_after"] == 12
    depth_only = {"job": "reshape", "source": {"repo": str(wide), "family": "qwen3_5"}, "mode": "prune", "drop_layers": [4, 5, 6, 7], "blocks": 1, "strategy": "contiguous",
                  "protect": {"first": 1, "last": 1}, "load": "auto", "check": False, "shard_gb": 5, "out": str(tmp_path / "d" / "model.partial"), "final": str(tmp_path / "d" / "model")}
    (tmp_path / "d").mkdir()
    code, ev = run_job(tmp_path, depth_only)
    assert code == 0, ev[-1]
    kept = np.load(Path(j["final"]) / "width_kept.npz")
    ref = masked(load(tmp_path / "d" / "model"), kept, tiny_dims(wide))
    thin = load(j["final"])
    assert thin.config.num_hidden_layers == 12
    assert torch.allclose(logits_of(thin), logits_of(ref), atol=1e-4)


# ------------------------------------------------------------------------------------------------ 2: measuring

def run_measure(monkeypatch, cfg):
    from ndforge_worker import protocol
    buf = io.StringIO()
    monkeypatch.setattr(protocol, "_out", buf)
    monkeypatch.setattr(protocol, "reserve_stdout", lambda: None)
    code = reshapejob.run_measure(cfg)
    return code, [json.loads(line) for line in buf.getvalue().splitlines()]


def measure_cfg(tmp_path, src, out_dir, **over):
    train = tmp_path / "train.jsonl"
    write_jsonl(train, sample_rows())
    return {"job": "reshape_measure", "source": {"repo": str(src), "family": "qwen3_5"}, "load": "16bit", "out_dir": str(out_dir),
            "calibration": {"train": str(train), "samples": 8, "max_len": 64, "batch": 2, "version": 1}, **over}


def test_measuring_gives_finite_deterministic_scores_and_is_cached(tmp_path, base8, monkeypatch):
    d = tiny_dims(base8)
    code, ev = run_measure(monkeypatch, measure_cfg(tmp_path, base8, tmp_path / "s1"))
    assert code == 0 and ev[-1]["type"] == "result"
    r = ev[-1]["data"]
    assert r["key"] == "s1" and r["tokens"] > 0 and r["load"] == "16bit" and math.isfinite(r["loss"]) and r["stable_depth"] in (True, False)
    z = np.load(tmp_path / "s1" / "scores.npz")
    assert z["mlp"].shape == (8, d["intermediate"]) and z["attn"].shape == (2, HEADS) and z["gdn"].shape == (6, NV)
    for k in z.files:
        assert np.isfinite(z[k]).all() and (z[k] >= 0).all() and z[k].dtype == np.float32
    assert {"mlp_h1", "mlp_h2", "attn_h1", "gdn_h2"} <= set(z.files) and float(z["mlp"].max()) > 0
    meta = json.loads((tmp_path / "s1" / "scores.json").read_text())
    assert meta["format"] == 1 and meta["data_version"] == 1 and meta["samples"] == 8 and meta["max_len"] == 64 and len(meta["matrix"]) == 3
    # the same measurement again, to another folder: identical
    code, _ = run_measure(monkeypatch, measure_cfg(tmp_path, base8, tmp_path / "s2"))
    z2 = np.load(tmp_path / "s2" / "scores.npz")
    assert code == 0 and all(np.array_equal(z[k], z2[k]) for k in z.files)
    # an existing measurement is reused without loading the model
    def boom(*a, **k):
        raise AssertionError("the model must not be loaded again")
    monkeypatch.setattr(reshapejob, "_load", boom)
    code, ev = run_measure(monkeypatch, measure_cfg(tmp_path, base8, tmp_path / "s1"))
    assert code == 0 and ev[-1]["data"]["key"] == "s1" and any("earlier measurement" in e.get("text", "") for e in ev if e["type"] == "log")


def test_the_scores_agree_with_the_hand_computed_aggregation(tmp_path, base8, monkeypatch):
    """the first layer's MLP score of channel c is sqrt(sum over texts of (mean over real tokens of |input of down_proj|)^2)"""
    from transformers import AutoTokenizer
    cfg = measure_cfg(tmp_path, base8, tmp_path / "s1", calibration={"train": str(tmp_path / "train.jsonl"), "samples": 4, "max_len": 64, "batch": 4, "version": 1})
    code, _ = run_measure(monkeypatch, cfg)
    assert code == 0
    got = np.load(tmp_path / "s1" / "scores.npz")["mlp"][0]
    tok = AutoTokenizer.from_pretrained(str(base8))
    batches = reshapejob._calibration(tok, cfg["calibration"], "qwen3_5")
    model = load(base8)
    seen = []
    model.model.layers[0].mlp.down_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().float()))
    total = torch.zeros(got.shape[0])
    with torch.no_grad():
        for b in batches:
            seen.clear()
            model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False)
            m = b["attention_mask"].float()
            per = (seen[0].abs() * m[..., None]).sum(1) / m.sum(1)[:, None]
            total += (per ** 2).sum(0)
    assert np.allclose(got, total.sqrt().numpy(), rtol=1e-4, atol=1e-6)


def test_the_job_measures_when_nothing_is_cached_and_reuses_it_next_time(tmp_path, monkeypatch):
    src = build_arch_dir(tmp_path / "b12", "qwen3_5", num_layers=12, heads=HEADS, kv_heads=KV, lin_v_heads=NV)
    train = tmp_path / "train.jsonl"
    write_jsonl(train, sample_rows())
    j = wjob(tmp_path, src, {"intermediate_size": 80}, blocks=1, calibration={"train": str(train), "samples": 8, "max_len": 64, "batch": 2, "version": 1})
    code, ev = run_job(tmp_path, j)
    assert code == 0, ev[-1]
    r = ev[-1]["data"]
    assert r["shape"] == "both" and r["layers_after"] == 8 and r["load"] == "16bit" and r["loss_before"] is not None
    assert (tmp_path / "models" / "_scores" / "k1" / "scores.json").is_file() and {"reshape_scores", "width_scores"} <= {e["type"] for e in ev}
    assert load(j["final"]).config.intermediate_size == 80
    j2 = {**j, "out": str(tmp_path / "m2" / "model.partial"), "final": str(tmp_path / "m2" / "model"), "width": {"intermediate_size": 96}}
    (tmp_path / "m2").mkdir()
    code, ev2 = run_job(tmp_path, j2)
    assert code == 0 and any("earlier measurement" in e.get("text", "") for e in ev2 if e["type"] == "log")
    assert ev2[-1]["data"]["chosen_blocks"] == r["chosen_blocks"]


# ------------------------------------------------------------------------------------------------ 6: GGUF with a real llama.cpp

def test_a_width_pruned_model_can_be_exported_to_gguf_with_llama_cpp(tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from ndforge_worker import exportjob
    if exportjob.find_llama_cpp()[0] is None:
        pytest.skip("llama.cpp is not installed here (set LLAMA_CPP_DIR)")
    from test_train_cpu import run_export
    from test_train_cpu import run_job as run_train
    original = build_arch_dir(tmp_path / "model", "qwen3_5", num_layers=8, heads=HEADS, kv_heads=KV, lin_v_heads=NV)
    j, r = run_width(tmp_path, original, {"intermediate_size": 80, "linear_num_value_heads": 4, "num_attention_heads": 4})
    slim = Path(j["final"])
    p, ev, out = run_train(tmp_path, "qwen3_5", "real", model={"repo": str(slim)}, hyper={"max_steps": 2})
    assert p.returncode == 0, p.stderr[-3000:]
    p2, ev2 = run_export(tmp_path, "qwen3_5", out, base={"repo": str(slim), "family": "qwen3_5"}, gguf=True, quant="q8_0")
    assert p2.returncode == 0, p2.stderr[-3000:]
    assert Path(ev2[-1]["data"]["gguf"]).stat().st_size > 0
