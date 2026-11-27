"""Decoding the packed experts, the packed experts module and the rename table of the packed loader (torch on the CPU, no real model)."""
import json

import pytest

import inkfake

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402

from ndforge_worker import inkling_load as il  # noqa: E402
from ndforge_worker import moe_layout as ml  # noqa: E402

E, ROWS, COLS = 3, 4, 32


def numpy_nvfp4(weight, scale, amax, den):
    """an independent decode: nibble by nibble, in plain numpy"""
    table = np.array(il.E2M1, dtype=np.float64)
    e, rows, nb = weight.shape
    out = np.zeros((e, rows, nb * 2))
    for i in range(e):
        for r in range(rows):
            for b in range(nb):
                lo, hi = weight[i, r, b] & 0xF, weight[i, r, b] >> 4
                for k, code in enumerate((lo, hi)):
                    col = 2 * b + k
                    out[i, r, col] = table[code] * float(scale[i, r, col // 16]) * (float(amax[i]) / float(den[0]))
    return out


def nvfp4_data(seed=0, e=E, rows=ROWS, cols=COLS):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (e, rows, cols // 2), generator=g, dtype=torch.uint8)
    s = (torch.rand(e, rows, cols // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
    amax = torch.rand(e, generator=g) * 3 + 1
    den = torch.tensor([6.0 * 448.0])
    return w, s, amax, den


def test_nvfp4_decode_matches_an_independent_numpy_decode():
    w, s, amax, den = nvfp4_data()
    got = il.dequant_nvfp4(w, s, amax, den, torch.float32).numpy()
    want = numpy_nvfp4(w.numpy(), s.float().numpy(), amax.numpy(), den.numpy())
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    assert got.shape == (E, ROWS, COLS)
    # the even element is the LOW nibble: byte 0x21 is codes (1, 2) -> (0.5, 1.0)
    one = il.dequant_nvfp4(torch.tensor([[[0x21] * 8]], dtype=torch.uint8), torch.ones(1, 1, 1).to(torch.float8_e4m3fn), torch.ones(1), torch.ones(1), torch.float32)
    assert one[0, 0, :2].tolist() == [0.5, 1.0]


def test_mxfp8_decode_uses_power_of_two_block_scales():
    vals = (torch.arange(E * ROWS * 64).reshape(E, ROWS, 64) % 13 - 6).float().to(torch.float8_e4m3fn)
    scale = torch.randint(120, 134, (E, ROWS, 2), dtype=torch.uint8)
    got = il.dequant_mxfp8(vals, scale, torch.float32)
    want = vals.float().reshape(E, ROWS, 2, 32) * (2.0 ** (scale.float() - 127))[..., None]
    assert torch.equal(got, want.reshape(E, ROWS, 64))


def reference_forward(w13, w2, gate_cap, up_cap, x, idx, w):
    """the arithmetic of Inkling2LoopExperts on decoded, DE-interleaved weights (gate rows then up rows, as transformers loads them)"""
    out = torch.zeros_like(x, dtype=torch.float32)
    n = w13.shape[0]
    half = w13.shape[1] // 2
    for e in range(n):
        tok, slot = torch.where(idx == e)
        if not tok.numel():
            continue
        gu = F.linear(x[tok], w13[e])
        g, u = gu[..., :half].float(), gu[..., half:].float()
        act = (torch.tanh(g / gate_cap) * gate_cap * torch.sigmoid(g) * torch.tanh(u / up_cap) * up_cap)
        out.index_add_(0, tok, F.linear(act, w2[e].float()) * w[tok, slot, None])
    return out


def test_packed_experts_equal_the_reference_experts_on_decoded_weights():
    H, I = 32, 32
    g = torch.Generator().manual_seed(3)
    # layer of NVFP4: w13 rows 2I (interleaved), cols H; w2 rows H, cols I
    w13 = nvfp4_data(1, 4, 2 * I, H)
    w2 = nvfp4_data(2, 4, H, I)
    t = {"w13_weight": w13[0], "w13_weight_scale": w13[1], "w13_weight_amax": w13[2], "w13_weight_outer_scale_denominator": w13[3],
         "w2_weight": w2[0], "w2_weight_scale": w2[1], "w2_weight_amax": w2[2], "w2_weight_outer_scale_denominator": w2[3]}
    mod = il.make_packed_experts_class()(t, 4.0, 25.0, torch.float32, chunk=2)
    d13 = il.dequant_nvfp4(*w13, torch.float32)
    d2 = il.dequant_nvfp4(*w2, torch.float32)
    deint = torch.cat([d13[:, 0::2], d13[:, 1::2]], dim=1)                   # transformers' Interleave(dim=1): gate rows first
    x = torch.randn(9, H, generator=g)
    idx = torch.randint(0, 4, (9, 2), generator=g)
    idx[0, 0] = 4                                                            # the "no expert" mark of the reference is skipped
    w = torch.rand(9, 2, generator=g)
    got = mod(x, idx, w)
    want = reference_forward(deint, d2, 4.0, 25.0, x, idx, w)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    # one expert alone
    y = mod.expert_forward(2, x[:3])
    gu = F.linear(x[:3], deint[2])
    h = gu.shape[-1] // 2
    ref = F.linear(torch.tanh(gu[..., :h] / 4.0) * 4.0 * torch.sigmoid(gu[..., :h]) * torch.tanh(gu[..., h:] / 25.0) * 25.0, d2[2])
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


def test_mxfp8_layers_work_too():
    H, I = 32, 32
    w13 = (torch.randn(2, 2 * I, H).to(torch.float8_e4m3fn), torch.randint(120, 130, (2, 2 * I, H // 32), dtype=torch.uint8))
    w2 = (torch.randn(2, H, I).to(torch.float8_e4m3fn), torch.randint(120, 130, (2, H, I // 32), dtype=torch.uint8))
    mod = il.make_packed_experts_class()({"w13_weight": w13[0], "w13_weight_scale": w13[1], "w2_weight": w2[0], "w2_weight_scale": w2[1]}, 4.0, 25.0, torch.float32)
    assert mod.packed is False and mod.num_experts == 2
    x = torch.randn(5, H)
    idx = torch.tensor([[0, 1]] * 5)
    out = mod(x, idx, torch.full((5, 2), 0.5))
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_every_real_tensor_has_a_home():
    tensors = json.loads((inkfake.DATA / "inkling2_tensors.json").read_text(encoding="utf-8"))
    targets = {}
    for name in tensors:
        layer, role, _ = ml.classify(name)
        t = il.hf_name(name)
        if role in ("vision", "experts", "expert_companion", "expert_shared", "loop_query"):
            assert t is None, name
        else:
            assert t and t not in targets, (name, t)
            targets[t] = name
    assert targets["lm_head.weight"] == "model.llm.unembed.weight"
    assert targets["model.layers.5.attention_input.query"] == "model.llm.attn_res.attn_queries.4"          # the table starts at layer 1
    assert targets["model.layers.0.mlp_input.query"] == "model.llm.attn_res.mlp_queries.0"
    assert "model.final_input.query" in targets
    assert "model.layers.17.mlp_message_gate.forget_gate_proj.weight" in targets and "model.layers.5.q_message_gate.down_proj.weight" not in targets      # only the loop layers have gates
    assert sum(1 for t in targets if t.endswith("e_score_correction_bias")) == 24
    assert not any(".attn." in t or "controllers" in t or t.endswith("sconv.weight") for t in targets), "an old name survived"


def test_the_two_loop_query_tables_become_one_in_the_order_query_key_value_mlp():
    q = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)          # (query, mlp) of loop layer i at rows 2i, 2i+1
    kv = 1000 + torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)   # (key, value)
    merged = il.merge_message_queries(q, kv)
    assert merged.shape == (12, 4, 3)
    assert torch.equal(merged[0, 0], q[0]) and torch.equal(merged[0, 1], kv[0]) and torch.equal(merged[0, 2], kv[1]) and torch.equal(merged[0, 3], q[1])
    assert torch.equal(merged[5, 0], q[10]) and torch.equal(merged[5, 3], q[11])


def test_an_old_transformers_is_refused_in_plain_words(monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "5.17.0")
    ok, v = il.transformers_ok()
    assert (ok, v) == (False, "5.17.0")
    with pytest.raises(il.InklingLoadError, match="5.24.0 or newer.*5.17.0"):
        il.require_transformers()
    monkeypatch.setattr(md, "version", lambda name: "5.24.1")
    assert il.transformers_ok() == (True, "5.24.1")
