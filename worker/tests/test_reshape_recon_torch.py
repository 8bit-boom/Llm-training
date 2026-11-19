"""Refitting the kept neurons with real (tiny, random) Qwen3.5 models on a CPU: the layer-wise replay equals the full forward, the refit reports a held-out share, the Try-first pass is
repeatable and the sidecar holds what the written model will. Skipped when torch / transformers are not installed (CI has them)."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from test_reshape_torch import load  # noqa: E402
from test_reshape_try_torch import CAL, base, do_try, measured, passes_of, try_cfg  # noqa: E402,F401
from test_reshape_width_torch import HEADS, KV, NV, tiny_dims  # noqa: E402,F401

from ndforge_worker import reshape, reshape_recon as rr, reshape_width as rw, reshapejob, reshapetry as rt, safetensors_io  # noqa: E402


def setup_model(base_dir, measured_):
    from transformers import AutoTokenizer
    cfg = json.loads((Path(base_dir) / "config.json").read_text())
    tc = reshape.text_config(cfg)
    types = reshape.layer_types_of(tc)
    _, cal = measured_
    batches = reshapejob._calibration(AutoTokenizer.from_pretrained(str(base_dir)), cal, "qwen3_5")
    model = load(base_dir)
    return model, reshapejob._find_layers(model, len(types)), types, batches


def test_layer_wise_replay_of_the_uncut_model_equals_the_full_forward(base, measured):
    model, layers, types, batches = setup_model(base, measured)
    assert "full_attention" in types and "linear_attention" in types, "mixed layer types are the point of this test"
    cap = rr.capture_inputs(model, layers, batches, types)
    assert len(cap["h0"]) == len(batches) and all(set(kw) == {"full_attention", "linear_attention"} for kw in cap["kwargs"])
    dev = next(model.parameters()).device
    for bi, b in enumerate(batches[:3]):
        h = cap["h0"][bi]
        for l, layer in enumerate(layers):
            h = rr.run_layer(layer, h, cap["kwargs"][bi][rr.type_of(types[l])], dev)
        with torch.no_grad():
            got = model.model.norm(h)
            want = model.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).last_hidden_state
        m = b["attention_mask"].bool()
        assert torch.allclose(got[m], want[m], atol=1e-5), float((got[m] - want[m]).abs().max())


def keep_sets(model_dir, layers, width, skip=()):
    r = np.random.default_rng(5)
    dims = tiny_dims(model_dir)
    return {l: rw.keep_top([float(x) for x in r.random(dims["intermediate"])], width) for l in range(layers) if l not in set(skip)}


def test_the_refit_reports_a_held_out_share_for_every_layer_and_the_weights_have_the_shape_of_the_cut(base, measured, tmp_path):
    model, layers, types, batches = setup_model(base, measured)
    dims = tiny_dims(base)
    keep = keep_sets(base, len(layers), 80)
    refs = rr.down_refs(safetensors_io.list_tensors(base), len(layers))
    out = rr.fit_layers(model, layers, batches, types, [], keep, rr.w0_reader(refs), dims["hidden"], 0.01, "pruned")
    assert set(out["w_k"]) == set(range(len(layers))) and all(b.dtype == np.uint16 and b.shape == (dims["hidden"], 80) for b in out["w_k"].values())
    assert all(s["h1_to_h2"] is not None and np.isfinite(s["share"]) for s in out["shares"].values())
    assert out["summary"]["layers"] == len(layers) and out["summary"]["median"] is not None
    assert any(s["share"] > 0 for s in out["shares"].values()), "the refit gives back something on rows it was not fitted on, in at least one layer"
    # the model itself is untouched by the refit
    ref = load(base)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), ref.state_dict().values()))
    # a sidecar round trip: bf16 [H, k] under the old layer numbers, the kept indices, at most two tags kept
    rr.write_sidecar(tmp_path, "tag1", out["w_k"], keep, {"tag": "tag1", "summary": out["summary"]})
    refs_back, meta = rr.read_sidecar(tmp_path, "tag1")
    assert set(refs_back) == set(out["w_k"]) and refs_back[0].dtype == "BF16" and refs_back[0].shape == (dims["hidden"], 80) and meta["kept"]["0"] == keep[0]
    assert np.array_equal(slice_bits(refs_back[3]), out["w_k"][3])
    for t in ("tag2", "tag3"):
        rr.write_sidecar(tmp_path, t, out["w_k"], keep, {"tag": t})
    assert sorted(p.name for p in (tmp_path / "recon").iterdir()) in (["tag2", "tag3"], ["tag1", "tag3"], ["tag1", "tag2"]) and len(list((tmp_path / "recon").iterdir())) == rr.KEEP_TAGS


def slice_bits(ref):
    from ndforge_worker import slice_io
    return slice_io.read_array(ref)


def test_a_depth_skipped_block_passes_through_and_the_original_variant_runs(base, measured):
    model, layers, types, batches = setup_model(base, measured)
    dims = tiny_dims(base)
    skip = reshape.blocks_to_layers([1], 4)
    keep = keep_sets(base, len(layers), 80, skip)
    refs = rr.down_refs(safetensors_io.list_tensors(base), len(layers))
    for variant in rr.VARIANTS:
        out = rr.fit_layers(model, layers, batches, types, skip, keep, rr.w0_reader(refs), dims["hidden"], 0.01, variant)
        assert set(out["w_k"]) == set(keep) and not set(out["w_k"]) & set(skip)


def test_a_try_pass_with_a_refit_is_repeatable_reuses_the_sidecar_and_leaves_the_plain_cut_alone(tmp_path, base, measured, monkeypatch):
    want = {"choices": [{"width": 80, "reconstruct": "pruned"}]}
    code, ev = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want))
    assert code == 0 and ev[-1]["type"] == "result", ev[-1]
    plain, refit = rt.pass_key((), 80), rt.pass_key((), 80, None, "pruned")
    first = passes_of(tmp_path)
    assert {plain, refit} <= set(first) and refit == "d=|w=80|r=pruned"
    folder, _ = measured
    tags = [p.name for p in (folder / "recon").iterdir()]
    assert len(tags) == 1
    saved = json.loads((tmp_path / "models" / "_scores" / "t1" / "reshape_try.json").read_text())
    info = saved["recon"][refit]
    assert info["layers"] == 24 and info["median_removed_share"] is not None and info["overlap"] in (0, None)
    row = [v for v in saved["analysis"]["variants"] if v["key"] == refit][0]
    assert row["recon"]["tag"] == tags[0] and row["plain_key"] == plain
    # again from scratch (the answers are dropped, the sidecar is kept): the same loss, and no new refit
    mtime = (folder / "recon" / tags[0] / "model.safetensors").stat().st_mtime
    code, _ = do_try(monkeypatch, try_cfg(tmp_path, base, measured, want, force=True))
    assert code == 0
    again = passes_of(tmp_path)
    assert again[refit]["loss"] == pytest.approx(first[refit]["loss"], rel=1e-6) and again[plain]["loss"] == pytest.approx(first[plain]["loss"], rel=1e-6)
    assert (folder / "recon" / tags[0] / "model.safetensors").stat().st_mtime == mtime
    assert abs(first[refit]["loss"] - first[plain]["loss"]) > 1e-7, "the refit changes the cut model's loss"
