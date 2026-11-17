"""quantblocks.py: the encoders and decoders of the ggml block formats. The decoders and the legacy encoders are ports of gguf-py's, so gguf-py itself is the oracle
(importorskip: CI installs it); the K-quant encoders are pinned by the golden fixture the real ggml encoder made."""
from pathlib import Path

import numpy as np
import pytest

from ndforge_worker import quantblocks as qb

GOLDEN = np.load(Path(__file__).parent / "fixtures" / "kquants_golden.npz")
QUANTIZED = list(qb.QUANTIZED)
LADDER = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0")


def rel_err(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))


def gaussian(rows=48, cols=512, seed=5):
    return (np.random.default_rng(seed).standard_normal((rows, cols)) * 0.02).astype(np.float32)


def laplace(rows=48, cols=512, seed=6):
    return (np.random.default_rng(seed).laplace(size=(rows, cols)) * 0.02).astype(np.float32)


def test_golden_blocks_decode_with_the_error_ladder():
    x = GOLDEN["x"]
    err = {}
    for t in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
        raw = GOLDEN[t]
        assert raw.shape == (len(x), qb.BLOCK[t][1])
        err[t] = rel_err(qb.decode(raw, t, len(x), 256), x)
    assert err["Q2_K"] > err["Q3_K"] > err["Q4_K"] > err["Q5_K"] > err["Q6_K"]
    # the golden blocks are deliberately awkward (huge outliers, constants), so only the order and the coarse size are pinned
    assert err["Q6_K"] < 0.1


@pytest.mark.parametrize("make", [gaussian, laplace])
def test_round_trip_ladder_for_all_quantized_types(make):
    x = make()
    err = {t: rel_err(qb.decode(qb.encode(x, t), t, *x.shape), x) for t in QUANTIZED}
    # error falls as bits rise, across the K-quants and the legacy 8-bit
    assert err["Q2_K"] > err["Q3_K"] > err["Q4_K"] > err["Q5_K"] > err["Q6_K"] > err["Q8_0"]
    assert err["Q4_0"] > err["Q5_0"] > err["Q8_0"]
    assert err["Q5_1"] < err["Q5_0"] * 1.1
    assert err["Q8_0"] < 0.01 and err["Q2_K"] < 0.5


def test_float_types_round_trip():
    x = gaussian(4, 64)
    assert np.array_equal(qb.decode(qb.encode(x, "F32"), "F32", 4, 64), x)
    assert rel_err(qb.decode(qb.encode(x, "F16"), "F16", 4, 64), x) < 1e-3
    assert rel_err(qb.decode(qb.encode(x, "BF16"), "BF16", 4, 64), x) < 1e-2


@pytest.mark.parametrize("t", list(qb.BLOCK))
def test_zeros_give_zeros(t):
    x = np.zeros((3, 256), np.float32)
    raw = qb.encode(x, t)
    assert raw.shape == (3, qb.row_bytes(t, 256))
    assert not qb.decode(raw, t, 3, 256).any()


@pytest.mark.parametrize("t", QUANTIZED)
def test_bad_lengths_raise_value_error(t):
    n = qb.BLOCK[t][0]
    with pytest.raises(ValueError):
        qb.encode(np.zeros((2, n + 1), np.float32), t)
    with pytest.raises(ValueError):
        qb.decode(b"\x00" * 7, t, 2, n)
    with pytest.raises(ValueError):
        qb.row_bytes(t, n + 1)
    with pytest.raises(ValueError):
        qb.encode(np.zeros(n, np.float32), t)


def test_unknown_type_is_refused():
    with pytest.raises(ValueError, match="unknown type"):
        qb.row_bytes("IQ4_XS", 256)


@pytest.mark.parametrize("t", QUANTIZED + ["F16", "BF16", "F32"])
def test_decoders_equal_gguf_py(t):
    gguf = pytest.importorskip("gguf")
    quants = gguf.quants
    x = laplace(16, 512)
    raw = qb.encode(x, t)
    mine = qb.decode(raw, t, *x.shape)
    if t in ("F16", "BF16", "F32"):
        ref = {"F16": lambda r: r.view(np.float16).astype(np.float32), "F32": lambda r: r.view(np.float32),
               "BF16": lambda r: (r.view(np.int16).astype(np.int32) << 16).view(np.float32)}[t](np.ascontiguousarray(raw).reshape(-1)).reshape(x.shape)
    else:
        ref = quants.dequantize(np.ascontiguousarray(raw), gguf.GGMLQuantizationType[t])
    assert np.array_equal(mine, ref)


@pytest.mark.parametrize("t", ["Q8_0", "Q4_0", "Q5_0", "Q5_1", "BF16"])
def test_legacy_encoders_equal_gguf_py(t):
    gguf = pytest.importorskip("gguf")
    x = laplace(16, 512)
    ref = gguf.quants.quantize(x, gguf.GGMLQuantizationType[t])
    assert np.array_equal(qb.encode(x, t), ref.reshape(x.shape[0], -1))


@pytest.mark.parametrize("t", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_k_quant_encoders_match_the_golden_blocks(t):
    x = GOLDEN["x"]
    assert np.array_equal(qb.encode(x, t).reshape(-1, qb.BLOCK[t][1]), GOLDEN[t])


def test_decode_accepts_bytes_and_arrays():
    x = gaussian(2, 256)
    raw = qb.encode(x, "Q4_K")
    assert np.array_equal(qb.decode(raw.tobytes(), "Q4_K", 2, 256), qb.decode(raw, "Q4_K", 2, 256))
