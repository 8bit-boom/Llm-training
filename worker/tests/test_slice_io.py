"""slice_io: slicing raw tensor bytes and bf16 conversion (needs numpy, not torch)."""
import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from stfile import fill, numel, slice_reference, write_file  # noqa: E402

from ndforge_worker import safetensors_io as st  # noqa: E402
from ndforge_worker import slice_io  # noqa: E402
from ndforge_worker.reshape import ReshapeError  # noqa: E402


def ref_of(tmp_path, dtype, shape, number=1):
    width = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}[dtype]
    data = fill(number, numel(shape) * width)
    write_file(Path(tmp_path) / "one.safetensors", [("pad", "BF16", (3,), fill(9, 6)), ("w", dtype, shape, data)])
    return st.list_tensors(tmp_path)["w"], data, width


@pytest.mark.parametrize("dtype,shape,axis,idx", [
    ("BF16", (6, 5), 0, [4, 0, 5]), ("BF16", (6, 5), 1, [1, 2, 4]), ("F32", (4, 7), 1, [6, 0]), ("F32", (4, 7), 0, [3]),
    ("BF16", (5, 1, 4), 0, [0, 2, 3]), ("F32", (9,), 0, [8, 7, 1]), ("I64", (3, 2), 0, [2, 0]), ("F16", (3, 4), 1, [3, 3, 0]),
])
def test_sliced_bytes_equal_a_pure_python_reference(tmp_path, dtype, shape, axis, idx):
    ref, data, width = ref_of(tmp_path, dtype, shape)
    assert ref.shape == shape and ref.dtype == dtype
    got = slice_io.sliced_bytes(ref, axis, idx)
    assert got == slice_reference(data, shape, width, axis, idx)
    assert len(got) == len(data) // shape[axis] * len(idx)


def test_sliced_bytes_refuses_bad_requests(tmp_path):
    ref, _, _ = ref_of(tmp_path, "BF16", (4, 3))
    with pytest.raises(ReshapeError, match="no axis 2"):
        slice_io.sliced_bytes(ref, 2, [0])
    with pytest.raises(ReshapeError, match="outside"):
        slice_io.sliced_bytes(ref, 0, [4])
    with pytest.raises(ReshapeError, match="outside"):
        slice_io.sliced_bytes(ref, 1, [-1])
    bad = st.TensorRef(ref.file, "F8_E4M3", ref.shape, ref.start, ref.end)
    with pytest.raises(ReshapeError, match="cannot be thinned"):
        slice_io.read_array(bad)


def u32(bits):
    return np.frombuffer(struct.pack("<I", bits), dtype="<f4")


@pytest.mark.parametrize("bits,expect", [
    (0x3F800000, 0x3F80), (0x00000000, 0x0000), (0x80000000, 0x8000),               # 1.0, +0, -0
    (0x7F800000, 0x7F80), (0xFF800000, 0xFF80),                                     # +-inf
    (0x3F808000, 0x3F80), (0x3F818000, 0x3F82), (0x3F808001, 0x3F81),               # tie to even (down, up), just above a tie
    (0x3F807FFF, 0x3F80), (0xBF818000, 0xBF82),
    (0x00000001, 0x0000), (0x00008000, 0x0000), (0x00018000, 0x0002), (0x80010000, 0x8001),   # subnormals
    (0x7F7FFFFF, 0x7F80),                                                          # the largest float32 rounds up to infinity
])
def test_f32_to_bf16_rounds_to_nearest_even(bits, expect):
    assert int(slice_io.f32_to_bf16(u32(bits))[0]) == expect


def test_f32_to_bf16_keeps_nan_and_round_trips_every_bf16():
    nan = slice_io.f32_to_bf16(np.array([np.nan, -np.nan], dtype="<f4"))
    assert all((int(x) & 0x7F80) == 0x7F80 and (int(x) & 0x7F) != 0 for x in nan)
    every = np.arange(0, 65536, dtype="<u2")
    f = slice_io.bf16_to_f32(every)
    back = slice_io.f32_to_bf16(f)
    finite = ~np.isnan(f)
    assert (back[finite] == every[finite]).all()
    assert slice_io.bf16_to_f32(np.array([0x3F80, 0xC000], dtype="<u2")).tolist() == [1.0, -2.0]


def test_weight_norm_scores_of_the_multi_token_head(tmp_path):
    dims = {"head_dim": 2}
    down = np.array([[3.0, 0.0, 1.0], [4.0, 0.0, 1.0]], dtype="<f4")             # [H=2, I=3]: column norms 5, 0, sqrt2
    up = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]], dtype="<f4")             # [I=3, H=2]: row norms 1, sqrt2, 2
    o = np.array([[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 4.0]], dtype="<f4")      # [H=2, nh*hd=4]: head 0 -> sqrt5, head 1 -> 5
    items = [("mtp.layers.0.mlp.down_proj.weight", "F32", (2, 3), down.tobytes()), ("mtp.layers.0.mlp.up_proj.weight", "F32", (3, 2), up.tobytes()),
             ("mtp.layers.0.self_attn.o_proj.weight", "F32", (2, 4), o.tobytes()), ("model.layers.0.x", "F32", (1,), fill(1, 4))]
    write_file(Path(tmp_path) / "m.safetensors", items)
    s = slice_io.weight_norm_scores(st.list_tensors(tmp_path), dims)
    assert list(s) == [0]
    assert s[0]["mlp"] == pytest.approx([5.0, 0.0, math.sqrt(2) * 2])
    assert s[0]["heads"] == pytest.approx([math.sqrt(5), 5.0])


def test_numpy_is_not_imported_by_the_depth_modules():
    code = "import sys; import ndforge_worker.reshape, ndforge_worker.reshapejob, ndforge_worker.safetensors_io, ndforge_worker.reshape_width; print('numpy' in sys.modules)"
    worker = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, "-c", code], cwd=worker, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False"
