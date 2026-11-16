"""Adapting an importance matrix to a model that was CUT on the "Slim or grow" page (docs/IMATRIX.md, "Adapting a matrix to a cut model").

A matrix is measured per layer and per tensor on the ORIGINAL model. A cut model keeps some of those layers (renumbered) and, when it was made thinner, some of the neurons of
every MLP. So a matrix can be adapted without measuring anything:

    depth   keep only the entries `blk.<old>.` of the kept layers and rename them `blk.<new>.`
    width   `blk.<L>.ffn_down.weight.in_sum2` has one value per MLP neuron (its input is the MLP's intermediate size): slice it with the neurons the cut model kept in that layer
            (`width_kept.npz`, `mlp_<new layer>`, indices in the ORIGINAL numbering). The other entries (ffn_gate, ffn_up, the attention and linear-attention ones) take the hidden
            state as input, which Slim does not change: they are copied; `counts` are copied
    weight  an optional factor w > 0 multiplies BOTH `in_sum2` and `counts`: the mean square (in_sum2 / counts) stays, the matrix weighs w times as much when it is merged
            (llama-imatrix ADDS the sums of the matrices it merges)

The matrix still describes the UNMODIFIED model's activations: for the cut (and healed) model it is an approximation; it is not Unsloth's Dynamic recipe. Anything else that
Slim changed (the experimental head axes), a model that was not made on the Slim page, and a matrix that does not belong to the model's original are refused with a plain
text (AdaptRefused); nothing is guessed.

Pure functions (numpy only, no torch) on the real layout (GGUF 3, one-dimensional float32 tensors, alignment 32; verified on the published file): `read_matrix`, `write_matrix`,
`read_cut`, `plan`, `adapt_arrays`, `scale_arrays`, `verify`; the job `imatrix_adapt` (`adapt`) and `imatrix_adapt_preview` (`preview`) follow imatrixmerge.py.

Job: {"job": "imatrix_adapt", "root": library folder, "id", "label", "stop_file", "cache": folder of the downloaded published matrices, "model_dir": models/<name>, "model_name",
"weight": 1.0, "kept_layers": [original layer numbers] | None (only for a model made by an older version that did not record them),
"source": {"kind": "library", "id", "file", "label", "chunks", "source_kind"} | {"kind": "file", "file"} | {"kind": "published", "base": "owner/name"}, "notes": [text]}.
The Hugging Face token reaches the worker only as the environment variable HF_TOKEN."""
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

from . import ggufinfo, imatrix, protocol
from .imatrixjob import MatrixError, check_space, find_tools, plain  # noqa: F401

ALIGN = 32
ADAPTED_MARK = "(adapted to "
WEIGHT_MAX = 1e6
_FIXED = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_LAYER_RE = re.compile(r"blk\.(\d+)\.(.+)$")
FFN_DOWN = "ffn_down.weight.in_sum2"


class AdaptRefused(MatrixError):
    """The matrix cannot be adapted to this model: a plain text; `reason` names the reason, `code` is "adapt_" + it (no_metadata, needs_layer_map, grown, heads, wrong_original, already_adapted, bad_kept, bad_matrix, bad_weight)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.reason = code
        self.code = "adapt_" + code            # the error code of the job: the page can offer the right way out (adapt_needs_layer_map: ask which layers)


def np():
    try:
        import numpy
    except ImportError:         # pragma: no cover - numpy comes with the training software
        raise MatrixError("numpy is missing in the training software (it is needed to adapt a matrix). Run the training setup again.")
    return numpy


# ------------------------------------------------------------------------------------------------------------------ the matrix file: read and write exactly the real layout

class Matrix:
    """kvs: [(key, type id, value)] in file order (an array is ("array", inner type id, [items])); names: tensor names in file order; arrays: {name: float32 numpy array}."""

    def __init__(self, kvs, names, arrays, alignment=ALIGN, lengths=None):
        self.kvs, self.names, self.arrays, self.alignment = kvs, names, arrays, alignment
        self.lengths = lengths if lengths is not None else {n: len(a) for n, a in arrays.items()}

    def kv(self, key, default=None):
        for k, _, v in self.kvs:
            if k == key:
                return v
        return default

    def datasets(self) -> list:
        v = self.kv("imatrix.datasets")
        return [x for x in v[2] if isinstance(x, str)] if isinstance(v, tuple) and v[0] == "array" else []


def read_matrix(path, data: bool = True) -> Matrix:
    """The whole matrix: every key-value (kept with its type, so that it can be written back as it was) and every tensor as a float32 array (with data=False only the lengths:
    a preview). Raises ValueError (plain words) when the file is not a GGUF importance matrix of the real layout: not a GGUF, truncated, a tensor that is not float32 or not
    one-dimensional."""
    numpy = np()
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"the file could not be read ({exc.strerror or type(exc).__name__})")
    pos = [0]

    def take(fmt):
        n = struct.calcsize("<" + fmt)
        if pos[0] + n > len(raw):
            raise ValueError("the file is cut short")
        v = struct.unpack_from("<" + fmt, raw, pos[0])[0]
        pos[0] += n
        return v

    def string():
        n = take("Q")
        if n > 1 << 24 or pos[0] + n > len(raw):
            raise ValueError("the file is cut short")
        s = raw[pos[0]:pos[0] + n].decode("utf-8", "replace")
        pos[0] += n
        return s

    def value(kind):
        if kind in _FIXED:
            return take(_FIXED[kind])
        if kind == 8:
            return string()
        if kind == 9:
            inner, n = take("I"), take("Q")
            if inner == 9 or n > 1 << 24:
                raise ValueError("an unusual array")
            return ("array", inner, [value(inner) for _ in range(n)])
        raise ValueError("an unknown value type")

    if raw[:4] != b"GGUF":
        raise ValueError("it is not a GGUF file")
    pos[0] = 4
    version = take("I")
    if version < 2 or version > 3:
        raise ValueError(f"GGUF version {version} is not one this program reads")
    n_tensors, n_kv = take("Q"), take("Q")
    if n_tensors > 1 << 22 or n_kv > 1 << 16:
        raise ValueError("absurd counts")
    kvs = []
    for _ in range(n_kv):
        key = string()
        kind = take("I")
        kvs.append((key, kind, value(kind)))
    infos = []
    for _ in range(n_tensors):
        name = string()
        nd = take("I")
        if nd != 1:
            raise ValueError(f"the tensor {name} has {nd} dimensions; a matrix of the real layout has one")
        n, typ, off = take("Q"), take("I"), take("Q")
        if typ != 0:
            raise ValueError(f"the tensor {name} is not float32")
        infos.append((name, n, off))
    align = next((v for k, _, v in kvs if k == "general.alignment"), ALIGN)
    align = align if isinstance(align, int) and align > 0 else ALIGN
    start = (pos[0] + align - 1) // align * align
    names, arrays, lengths = [], {}, {}
    for name, n, off in infos:
        a, b = start + off, start + off + 4 * n
        if b > len(raw) or n > 1 << 31:
            raise ValueError(f"the data of {name} is cut short")
        if name in lengths:
            raise ValueError(f"the tensor {name} is in the file twice")
        lengths[name] = n
        if data:
            arrays[name] = numpy.frombuffer(raw, dtype="<f4", count=n, offset=a).astype(numpy.float32)
        names.append(name)
    return Matrix(kvs, names, arrays, align, lengths)


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc(kind, v) -> bytes:
    if kind in _FIXED:
        return struct.pack("<" + _FIXED[kind], v)
    if kind == 8:
        return _s(v)
    return struct.pack("<IQ", v[1], len(v[2])) + b"".join(_enc(v[1], x) for x in v[2])


def layout(kvs, entries, alignment=ALIGN):
    """(header bytes, [offset of every tensor], total size of the file) for kvs and entries [(name, length)] (one-dimensional float32). The header is padded to the alignment and
    every tensor to the alignment, as in the real file. The size predicted for a preview and the size written are this one function's."""
    info, offs, off = b"", [], 0
    for name, n in entries:
        info += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<IQ", 0, off)
        offs.append(off)
        off += (4 * n + alignment - 1) // alignment * alignment
    head = b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(kvs))
    for k, t, v in kvs:
        head += _s(k) + struct.pack("<I", t) + _enc(t, v)
    head += info
    head += b"\0" * (-len(head) % alignment)
    return head, offs, len(head) + off


def write_matrix(path, m: Matrix) -> int:
    """Writes `m` (its kvs, its tensors in its order, float32, one-dimensional, alignment kept) and returns the size."""
    numpy = np()
    entries = [(n, m.lengths[n]) for n in m.names]
    head, offs, total = layout(m.kvs, entries, m.alignment)
    with open(path, "wb") as fh:
        fh.write(head)
        for n, off in zip(m.names, offs):
            a = numpy.ascontiguousarray(m.arrays[n], dtype="<f4")
            fh.write(a.tobytes())
            fh.write(b"\0" * (-a.nbytes % m.alignment))
    return total


# ------------------------------------------------------------------------------------------------------------------ the cut model

class Cut:
    """What a model of the Slim page says about how it was made. src[j] = the ORIGINAL layer that became layer j; kept_mlp[j] = the original neuron numbers kept in layer j (None
    when the MLP was not made thinner)."""

    def __init__(self, name, origin_repo, origin_layers, src, inter_before, inter_after, kept_mlp, folder):
        self.name, self.origin_repo, self.origin_layers, self.src = name, origin_repo, origin_layers, src
        self.inter_before, self.inter_after, self.kept_mlp, self.folder = inter_before, inter_after, kept_mlp, folder

    @property
    def dropped(self) -> list:
        return [i for i in range(self.origin_layers) if i not in set(self.src)]

    @property
    def new_of(self) -> dict:
        return {old: j for j, old in enumerate(self.src)}

    @property
    def thinner(self) -> bool:
        return self.kept_mlp is not None


NOT_SLIM = ("This model has no record of how it was cut (no ndforge_reshape.json), so the layers and neurons it kept are unknown. Only a model made on the Slim or grow page can have a matrix "
            "adapted to it. For any other model, measure a matrix on the model itself (Create a matrix).")


def model_folder(model_dir) -> Path:
    """`models/<name>/model` for a models/<name> folder, the folder itself when it holds the record."""
    d = Path(model_dir)
    for c in (d, d / "model"):
        if (c / "ndforge_reshape.json").is_file():
            return c
    raise AdaptRefused("no_metadata", NOT_SLIM)


def cut_model_of(*candidates):
    """{"dir": models/<name>, "name"} of the first candidate (a folder path) that is a model made on the Slim page (it holds ndforge_reshape.json, itself or in `model/`), else None.
    Used to remember, in a matrix's record, that it was measured on a cut model."""
    for c in candidates:
        if not c or not isinstance(c, (str, os.PathLike)):
            continue
        try:
            d = Path(c)
            for f in (d, d / "model"):
                if (f / "ndforge_reshape.json").is_file():
                    top = f.parent if f.name == "model" else f
                    return {"dir": str(top), "name": top.name}
        except (OSError, ValueError):
            continue
    return None


def _ints(v):
    return isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)


def read_cut(model_dir, name: str = "", kept_layers=None) -> Cut:
    """The record of a model made on the Slim page: ndforge_reshape.json (`origin.layers`, `src_layers`, `dropped`, `duplicated`, `width`) and, when the MLP was made thinner,
    width_kept.npz. Raises AdaptRefused with a plain text for everything this adaptation cannot honestly do. `kept_layers` is the person's answer for a model made by an
    older version that recorded only counts."""
    folder = model_folder(model_dir)
    try:
        meta = json.loads((folder / "ndforge_reshape.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise AdaptRefused("no_metadata", NOT_SLIM)
    if not isinstance(meta, dict):
        raise AdaptRefused("no_metadata", NOT_SLIM)
    origin = meta.get("origin") if isinstance(meta.get("origin"), dict) else {}
    n_orig = origin.get("layers")
    if not isinstance(n_orig, int) or isinstance(n_orig, bool) or n_orig < 1:
        raise AdaptRefused("no_metadata", "This model's record does not say how many layers its original had (it was not made on the Slim or grow page, or by a much older version), so a matrix cannot be "
                                          "adapted to it. Measure a matrix on the model itself (Create a matrix).")
    name = name or folder.parent.name
    if meta.get("duplicated") or (meta.get("zero_init") or []) or meta.get("mode") == "grow":
        raise AdaptRefused("grown", f"The model {name} has layers that were added (a grown model). A matrix measures the layers of the original; there is nothing measured for a copy, so it is not adapted. "
                                    "Measure a matrix on the model itself (Create a matrix).")
    width = meta.get("width") if isinstance(meta.get("width"), dict) else None
    if width:
        for key, what in (("num_attention_heads", "attention heads"), ("linear_num_value_heads", "linear-attention value heads")):
            pair = width.get(key)
            if isinstance(pair, list) and len(pair) == 2 and pair[0] != pair[1]:
                raise AdaptRefused("heads", f"The model {name} was made thinner in more than its MLP: its {what} were cut from {pair[0]} to {pair[1]}. That changes the input of the attention output "
                                            "of every layer, which a matrix of the original cannot be cut to without measuring (the head axes are experimental). Only a cut of whole layers and of the MLP can be adapted; "
                                            "measure a matrix on the model itself (Create a matrix).")
    src = meta.get("src_layers")
    if not _ints(src) or not src:
        src = None
    if src is None and _ints(meta.get("dropped")):          # an older record: the dropped layers are listed, the kept ones follow
        src = [i for i in range(n_orig) if i not in set(meta["dropped"])]
        if meta.get("layers") is not None and len(src) != meta.get("layers"):
            src = None
    if src is None and kept_layers is not None:
        src = kept_layers
    if src is None:
        raise AdaptRefused("needs_layer_map", f"The record of the model {name} says that it has {meta.get('layers', '?')} layers of the original {n_orig}, but not which ones (it was made by an older version that kept only the counts). "
                                              "Tell which layers of the original it kept (their numbers, counted from 0, for example 0, 1, 2, 3, 8, 9, 10, 11) and try again.")
    if not _ints(src) or len(set(src)) != len(src) or any(i < 0 or i >= n_orig for i in src) or src != sorted(src):
        raise AdaptRefused("bad_kept", f"The kept layers of {name} are not a list of different layers of the original in order ({src[:12]}...), so they cannot be used." if isinstance(src, list) else "The kept layers are not a list.")
    if isinstance(meta.get("layers"), int) and meta["layers"] != len(src):
        raise AdaptRefused("bad_kept", f"The record of {name} says {meta['layers']} layers but lists {len(src)} kept layers. Its record is inconsistent, so it is not used.")
    before = after = None
    kept_mlp = None
    pair = width.get("intermediate_size") if width else None
    if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, int) for x in pair):
        before, after = pair
    if width and before is not None and after != before:
        numpy = np()
        kept_file = folder / str(width.get("kept") or "width_kept.npz")
        if not kept_file.is_file():
            raise AdaptRefused("bad_kept", f"The model {name} was made thinner, but its list of kept neurons ({kept_file.name}) is missing, so the matrix cannot be cut to it.")
        try:
            with numpy.load(kept_file) as z:
                recorded = z["src"].tolist() if "src" in z.files else None
                kept_mlp = {j: z[f"mlp_{j}"].astype("int64") for j in range(len(src)) if f"mlp_{j}" in z.files}
        except Exception as exc:    # noqa: BLE001 - a damaged file of any kind
            raise AdaptRefused("bad_kept", f"The list of kept neurons of {name} could not be read ({type(exc).__name__}).")
        if recorded is not None and recorded != src:
            raise AdaptRefused("bad_kept", f"The list of kept neurons of {name} belongs to other layers than the record says. Its files do not fit together, so they are not used.")
        for j in range(len(src)):
            v = kept_mlp.get(j)
            if v is None or len(v) != after or len(set(v.tolist())) != after or (len(v) and (int(v.min()) < 0 or int(v.max()) >= before)):
                raise AdaptRefused("bad_kept", f"The neurons kept in layer {j} of {name} are missing or do not fit ({after} of {before} expected). Its width record is damaged, so it is not used.")
    else:
        after = before
    return Cut(name, origin.get("repo"), n_orig, src, before, after, kept_mlp, folder)


# ------------------------------------------------------------------------------------------------------------------ the plan: what comes out

def _entries(m: Matrix) -> list:
    return [(n, m.lengths[n]) for n in m.names]


def _layer_info(entries):
    """({layer: {tail of the name: length}}, [non-layer names]) of [(name, length)]."""
    layers, other = {}, []
    for name, n in entries:
        mm = _LAYER_RE.match(name)
        if mm:
            layers.setdefault(int(mm.group(1)), {})[mm.group(2)] = n
        else:
            other.append(name)
    return layers, other


def plan(entries, datasets, cut: Cut, weight: float = 1.0) -> dict:
    """What adapting these entries [(name, length)] to `cut` gives, from the header alone, or raises AdaptRefused: {"entries": [(new name, new length, source name)], "layers_before", "layers_after",
    "kept", "ffn_down_before", "ffn_down_after", "dropped_mtp": [layers], "size": None (set by the caller from the layout)}."""
    if not (isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight) and 0 < weight <= WEIGHT_MAX):
        raise AdaptRefused("bad_weight", f"The weight must be a number above 0 (and at most {WEIGHT_MAX:g}), not {weight!r}.")
    if any(d.startswith(ADAPTED_MARK) for d in datasets):
        raise AdaptRefused("already_adapted", "This matrix was already adapted to a cut model (its record says so). Adapt the original matrix instead; adapting twice would renumber layers that are already renumbered.")
    layers, other = _layer_info(entries)
    if not layers:
        raise AdaptRefused("bad_matrix", "This file has no per-layer statistics (no tensors named like blk.0.ffn_down.weight.in_sum2), so it is not an importance matrix that can be adapted.")
    names = {n for n, _ in entries}
    for n in names:
        if n.endswith(".in_sum2") and n[: -len(".in_sum2")] + ".counts" not in names or n.endswith(".counts") and n[: -len(".counts")] + ".in_sum2" not in names:
            raise AdaptRefused("bad_matrix", f"The statistics of {n.rsplit('.', 1)[0]} are incomplete (its in_sum2 and counts do not both exist), so the file is not a whole importance matrix.")
    trunk = sorted(l for l in layers if l < cut.origin_layers)
    extra = sorted(l for l in layers if l >= cut.origin_layers)
    if trunk != list(range(cut.origin_layers)):
        have = max(layers) + 1
        raise AdaptRefused("wrong_original", f"This matrix has data for {have} layers, but the original of the model {cut.name} ({cut.origin_repo or 'its original'}) has {cut.origin_layers}. "
                                              "It was not measured on that model (it may belong to another model, or to a model that was cut already). Choose the matrix of the original model.")
    for l in trunk:
        missing = [k for k in ("ffn_down.weight.in_sum2", "ffn_gate.weight.in_sum2", "ffn_up.weight.in_sum2") if k not in layers[l]]
        if missing:
            raise AdaptRefused("wrong_original", f"Layer {l} of this matrix has no {missing[0].split('.')[0]} statistics; every layer of the original model has ffn_down, ffn_gate and ffn_up, so this is not a whole matrix of it.")
    down = {l: layers[l][FFN_DOWN] for l in trunk}
    if cut.inter_before is not None and any(n != cut.inter_before for n in down.values()):
        bad = next(l for l, n in down.items() if n != cut.inter_before)
        raise AdaptRefused("wrong_original", f"In this matrix the MLP (ffn_down) input of layer {bad} is {down[bad]} wide, but the original of the model {cut.name} has {cut.inter_before}. "
                                              "It was not measured on that model. Choose the matrix of the original model.")
    if cut.inter_before is None and len(set(down.values())) != 1:
        raise AdaptRefused("wrong_original", "The MLP widths of this matrix differ from layer to layer, so it is not the matrix of a model this program cuts.")
    before = next(iter(down.values()))
    after = cut.inter_after if cut.thinner else before
    new_of = cut.new_of
    out = []
    for name, n in entries:
        mm = _LAYER_RE.match(name)
        if mm is None:
            out.append((name, n, name))
            continue
        old = int(mm.group(1))
        if old in new_of:
            tail = mm.group(2)
            out.append((f"blk.{new_of[old]}.{tail}", after if tail == FFN_DOWN else n, name))
    return {"entries": out, "layers_before": cut.origin_layers, "layers_after": len(cut.src), "kept": list(cut.src), "ffn_down_before": before, "ffn_down_after": after,
            "dropped_mtp": extra, "other": other}


def _is_down(name: str) -> bool:
    m = _LAYER_RE.match(name)
    return bool(m) and m.group(2) == FFN_DOWN


def _weighted(a, weight):
    numpy = np()
    if weight == 1.0:
        return numpy.array(a, dtype=numpy.float32, copy=True)
    return (a.astype(numpy.float64) * weight).astype(numpy.float32)


def adapt_arrays(m: Matrix, cut: Cut, weight: float, p: dict) -> dict:
    """{new tensor name: float32 array} for the plan `p` of matrix `m`: depth renamed, width sliced (the kept neurons of the new layer, original numbering), weight applied."""
    out = {}
    for new, _, old in p["entries"]:
        a = m.arrays[old]
        if cut.thinner and _is_down(new):
            a = a[cut.kept_mlp[int(_LAYER_RE.match(new).group(1))]]
        out[new] = _weighted(a, weight)
    return out


def adapted_kvs(m: Matrix, name: str):
    """The key-values of the adapted matrix: the source's, in order, with an entry "(adapted to <name>)" appended to `imatrix.datasets` (added when there is none)."""
    note = f"{ADAPTED_MARK}{name})"
    out, done = [], False
    for k, t, v in m.kvs:
        if k == "imatrix.datasets" and isinstance(v, tuple) and v[1] == 8:
            v, done = ("array", 8, list(v[2]) + [note]), True
        out.append((k, t, v))
    if not done:
        out.append(("imatrix.datasets", 9, ("array", 8, [note])))
    return out


def scaled_kvs(m: Matrix, weight: float):
    note = f"(weighted x{weight:g})"
    out, done = [], False
    for k, t, v in m.kvs:
        if k == "imatrix.datasets" and isinstance(v, tuple) and v[1] == 8:
            v, done = ("array", 8, list(v[2]) + [note]), True
        out.append((k, t, v))
    if not done:
        out.append(("imatrix.datasets", 9, ("array", 8, [note])))
    return out


def adapt(m: Matrix, cut: Cut, weight: float = 1.0):
    """(adapted Matrix, plan). Pure: no files."""
    p = plan(_entries(m), m.datasets(), cut, weight)
    arrays = adapt_arrays(m, cut, weight, p)
    names = [e[0] for e in p["entries"]]
    return Matrix(adapted_kvs(m, cut.name), names, arrays, m.alignment), p


def scale(m: Matrix, weight: float) -> Matrix:
    """The same matrix with in_sum2 and counts multiplied by `weight` (the mean square stays); weight 1 is a plain copy."""
    if not (isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight) and 0 < weight <= WEIGHT_MAX):
        raise AdaptRefused("bad_weight", f"The weight must be a number above 0 (and at most {WEIGHT_MAX:g}), not {weight!r}.")
    return Matrix(scaled_kvs(m, weight) if weight != 1.0 else list(m.kvs), list(m.names), {n: _weighted(m.arrays[n], weight) for n in m.names}, m.alignment)


def preview_numbers(p: dict, kvs, alignment=ALIGN) -> dict:
    """The numbers the page shows before anything is made."""
    entries = [(n, ln) for n, ln, _ in p["entries"]]
    _, _, size = layout(kvs, entries, alignment)
    return {"layers_before": p["layers_before"], "layers_after": p["layers_after"], "kept": p["kept"], "ffn_down_before": p["ffn_down_before"], "ffn_down_after": p["ffn_down_after"],
            "entries_after": sum(1 for n, _, _ in p["entries"] if n.endswith(".in_sum2")), "tensors_after": len(entries), "size": size, "dropped_mtp": p["dropped_mtp"]}


# ------------------------------------------------------------------------------------------------------------------ verifying what was written

def verify(src: Matrix, out_path, cut: Cut, p: dict, weight: float) -> dict:
    """Reads the file that was written back with the real-layout reader and checks it against what was predicted: tensor count, names, shapes, the header of the other reader,
    and that the sums of the kept entries are the source's (times the weight). Raises MatrixError when anything differs. Returns {"tensors", "entries"}."""
    numpy = np()
    got = read_matrix(out_path)
    want = [(n, ln) for n, ln, _ in p["entries"]]
    if [(n, got.lengths[n]) for n in got.names] != want:
        raise MatrixError("The adapted matrix does not read back as predicted (its tensors or their lengths differ); it was removed. Please make a problem report.")
    info = ggufinfo.read(out_path)
    if len(info["tensors"]) != len(want) or any(len(t["dims"]) != 1 or t["type"] != 0 for t in info["tensors"]) or info["data_start"] % got.alignment:
        raise MatrixError("The adapted matrix has an unexpected layout; it was removed. Please make a problem report.")
    for new, _, old in p["entries"]:
        a = src.arrays[old]
        if cut.thinner and _is_down(new):
            a = a[cut.kept_mlp[int(_LAYER_RE.match(new).group(1))]]
        b = got.arrays[new]
        if weight == 1.0:
            same = numpy.array_equal(a, b) and float(a.astype(numpy.float64).sum()) == float(b.astype(numpy.float64).sum())
        else:
            same = numpy.allclose(a.astype(numpy.float64) * weight, b.astype(numpy.float64), rtol=1e-6, atol=0.0)
        if not same:
            raise MatrixError(f"The adapted entry {new} is not the source's {old} at the kept places; it was removed. Please make a problem report.")
    return {"tensors": len(want), "entries": sum(1 for n, _ in want if n.endswith(".in_sum2"))}


# ------------------------------------------------------------------------------------------------------------------ the jobs

def _check_id(mid):
    if not mid or mid.startswith("_") or "/" in mid or "\\" in mid or ".." in mid:
        raise MatrixError("That is not a valid name for a matrix.")


def _read_source(path) -> Matrix:
    try:
        return read_matrix(path)
    except ValueError as exc:
        raise AdaptRefused("bad_matrix", f"The matrix {Path(path).name} could not be read as an importance matrix of the real layout ({exc}). A legacy .dat file cannot be adapted; use a .gguf matrix.")


def _read_source_header(path) -> Matrix:
    try:
        return read_matrix(path, data=False)
    except ValueError as exc:
        raise AdaptRefused("bad_matrix", f"The matrix {Path(path).name} could not be read as an importance matrix of the real layout ({exc}). A legacy .dat file cannot be adapted; use a .gguf matrix.")


def resolve_source(src: dict, job: dict, im: dict) -> dict:
    """{"kind", "file", "label", "id", "chunks", "hub"?}: the matrix file to adapt (the published matrix is looked up and downloaded first, exactly as in a merge)."""
    from . import imatrixmerge
    kind = src.get("kind")
    if kind == "library":
        f = Path(src.get("file") or "")
        if not f.is_file():
            raise MatrixError(f"The saved matrix {src.get('label') or src.get('id')} has no file any more.")
        if src.get("source_kind") == "adapted":
            raise AdaptRefused("already_adapted", f"The saved matrix {src.get('label') or src.get('id')} is already an adapted one. Adapt the original matrix instead; adapting twice would renumber layers that are already renumbered.")
        return {"kind": "library", "file": f, "label": src.get("label") or src.get("id") or f.name, "id": src.get("id"), "chunks": src.get("chunks")}
    if kind == "file":
        f = Path(src.get("file") or "")
        if not f.is_file():
            raise MatrixError(f"The matrix file {f} was not found. Choose it again.")
        if f.suffix.lower() != ".gguf":
            raise AdaptRefused("bad_matrix", f"{f.name} is not a GGUF matrix: only an imatrix.gguf can be adapted (a legacy .dat file cannot be looked into).")
        return {"kind": "file", "file": f, "label": f.name, "id": None, "chunks": None}
    if kind == "published":
        protocol.progress("imatrix", 0.02, "Looking up Unsloth's published matrix")
        info = imatrixmerge.lookup(src.get("base") or "", job.get("cache"))
        f = imatrixmerge.download(info, im, span=(0.05, 0.4))
        return {"kind": "published", "file": f, "label": f"Unsloth's published matrix for {info['base']}", "id": None, "chunks": None, "hub": info}
    raise MatrixError("Choose the matrix to adapt: a saved matrix, a matrix file, or Unsloth's published matrix.")


def adapt_file(src_file, dst_file, cut: Cut, weight: float = 1.0) -> dict:
    """Reads `src_file`, adapts it to `cut`, writes `dst_file` and verifies it. Returns {"plan", "numbers"}. Used by the job and by the merge (which adapts a published matrix in its work folder)."""
    m = _read_source(src_file)
    out, p = adapt(m, cut, weight)
    numbers = preview_numbers(p, out.kvs, out.alignment)
    write_matrix(dst_file, out)
    try:
        verify(m, dst_file, cut, p, weight)
    except BaseException:
        Path(dst_file).unlink(missing_ok=True)
        raise
    return {"plan": p, "numbers": numbers, "matrix": out}


def scale_file(src_file, dst_file, weight: float) -> None:
    """A copy of a matrix with in_sum2 and counts multiplied by `weight` (the merge's way of giving a parent a different weight)."""
    m = _read_source(src_file)
    write_matrix(dst_file, scale(m, weight))


def _cut_for(job: dict) -> Cut:
    if not job.get("model_dir"):
        raise MatrixError("Choose the cut model to adapt the matrix to.")
    kept = job.get("kept_layers")
    if kept is not None and not _ints(kept):
        raise AdaptRefused("bad_kept", "The kept layers must be a list of layer numbers.")
    return read_cut(job["model_dir"], job.get("model_name") or "", kept)


def preview(job: dict) -> dict:
    """What adapting a matrix FILE to the cut model would give, from the file's header and the model's record alone (nothing is written, nothing downloaded): the numbers of the page,
    and the same refusals the real job would give."""
    cut = _cut_for(job)
    f = Path((job.get("source") or {}).get("file") or "")
    if not f.is_file():
        raise MatrixError("The matrix file was not found.")
    m = _read_source_header(f)
    p = plan(_entries(m), m.datasets(), cut, job.get("weight") or 1.0)
    out = preview_numbers(p, adapted_kvs(m, cut.name), m.alignment)
    out.update(size_before=f.stat().st_size, model=cut.name, origin_repo=cut.origin_repo, thinner=cut.thinner)
    return out


def adapt_job(job: dict) -> dict:
    """Adapts a matrix to a cut model into <root>/<id>/. Returns the result. Raises MatrixError (plain text) or imatrix.Stopped; the folder is removed again then."""
    root, mid = Path(job["root"]), str(job["id"])
    _check_id(mid)
    weight = job.get("weight", 1.0)
    weight = 1.0 if weight is None else weight
    if not (isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight) and 0 < weight <= WEIGHT_MAX):
        raise AdaptRefused("bad_weight", f"The weight must be a number above 0 (and at most {WEIGHT_MAX:g}), not {weight!r}.")
    out = root / mid
    if out.exists():
        raise MatrixError(f"The folder {out} exists already. Try again in a minute.")
    im = {"stop_file": job.get("stop_file")}
    cut = _cut_for(job)                                                  # refused before anything is downloaded
    notes = list(job.get("notes") or [])
    src = resolve_source(job.get("source") or {}, job, im)
    if imatrix.stop_requested(im):
        raise imatrix.Stopped()
    protocol.progress("imatrix", 0.45, "Reading the matrix")
    m = _read_source(src["file"])
    out_m, p = adapt(m, cut, weight)
    numbers = preview_numbers(p, out_m.kvs, out_m.alignment)
    root.mkdir(parents=True, exist_ok=True)
    check_space(root, int(numbers["size"] * 1.1), "the adapted matrix")
    out.mkdir()
    try:
        protocol.progress("imatrix", 0.6, "Writing the adapted matrix")
        partial = out / "imatrix.partial.gguf"
        write_matrix(partial, out_m)
        if imatrix.stop_requested(im):
            raise imatrix.Stopped()
        protocol.progress("imatrix", 0.8, "Checking what was written")
        checked = verify(m, partial, cut, p, weight)
        target = out / "imatrix.gguf"
        os.replace(partial, target)
        llama = llama_check(target, out, im)
        if llama.get("warning"):
            notes.append("Warning: " + llama["warning"])
        if p["dropped_mtp"]:
            notes.append(f"The statistics of the speculative-decoding head (layers {p['dropped_mtp'][0]} and above of the matrix) were left out: Slim cuts that head without measuring it.")
        if weight != 1.0:
            notes.append(f"The matrix was given the weight {weight:g}: its sums and counts are multiplied by it, so that a merge counts it {weight:g} times as strongly. Its mean square per channel is unchanged.")
        notes.append("This matrix still describes the UNMODIFIED model's activations: for the cut model it is an approximation. It is not Unsloth's Dynamic recipe. "
                     "Merge it with a matrix measured on the cut (and healed) model, and compare the sizes (Compare quants).")
        summary = imatrix.matrix_summary(target) or {}
        sha = imatrix.file_sha256
        parent = {"kind": src["kind"], "label": src["label"], "id": src.get("id"), "path": str(src["file"]), "sha256": sha(src["file"]), "size": Path(src["file"]).stat().st_size,
                  "chunks": src.get("chunks") if isinstance(src.get("chunks"), int) else m.kv("imatrix.chunk_count")}
        if src["kind"] == "published":
            parent.update(repo=src["hub"]["repo"], file=src["hub"]["file"], base=src["hub"]["base"], url=src["hub"]["url"])
        adapted = {"model": cut.name, "model_dir": str(cut.folder.parent if cut.folder.name == "model" else cut.folder), "origin_repo": cut.origin_repo, "layers_before": p["layers_before"],
                   "layers_after": p["layers_after"], "kept_layers": p["kept"], "dropped_layers": cut.dropped, "ffn_down_before": p["ffn_down_before"], "ffn_down_after": p["ffn_down_after"],
                   "sliced_ffn_down": cut.thinner, "weight": weight, "tensors_before": len(m.names), "tensors_after": checked["tensors"], "entries_after": checked["entries"],
                   "dropped_mtp_layers": p["dropped_mtp"]}
        tokens = summary.get("tokens")
        info = {"format": imatrix.FORMAT, "id": mid, "label": str(job.get("label") or "").strip()[:80], "model": cut.name,
                "source": f"{src['label']}, adapted to the cut model {cut.name}", "source_kind": "adapted", "via": "adapted", "parents": [parent], "adapt": adapted,
                "chunks": m.kv("imatrix.chunk_count"), "tokens": tokens, "weight": weight, "ctx": None, "general_used": None, "nextn": None, "nextn_asked": False, "arch": summary.get("arch"),
                "layers": summary.get("layers"), "entries": summary.get("entries"), "size": target.stat().st_size, "has_dat": False, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "llama_cpp": llama.get("version") or "", "llama_check": {k: v for k, v in llama.items() if k != "version"}, "base_repo": cut.origin_repo,
                "cut_model": {"dir": adapted["model_dir"], "name": cut.name}, "calibration": "the text the source matrix was measured on", "calibration_kind": "adapted", "notes": notes,
                "statistics": None}
        (out / "imatrix.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        protocol.progress("imatrix", 1.0, "Adapted matrix ready")
        return {"id": mid, "dir": str(out), "file": str(target), "dat": None, "size_mb": round(target.stat().st_size / 1048576, 2), "info": info, "notes": notes, "adapted": adapted,
                "numbers": numbers, "statistics": None}
    except BaseException as exc:
        shutil.rmtree(out, ignore_errors=True)
        if isinstance(exc, (MatrixError, imatrix.Stopped, KeyboardInterrupt)):
            raise
        raise MatrixError(plain(exc))


def llama_check(target: Path, out: Path, im: dict) -> dict:
    """Does llama.cpp's llama-imatrix read the adapted file? `llama-imatrix --in-file adapted -o check.gguf`, when this computer has llama.cpp: it must exit 0 and write a matrix with the same
    tensors. {"ran": bool, "ok": bool | None, "warning"?, "version"?}. A refusal is a warning, not a failure (the real tool's behaviour is NOT verified, docs/IMATRIX.md); Stop still works."""
    try:
        _, tool = find_tools(need_convert=False)
    except MatrixError:
        return {"ran": False, "ok": None, "why": "llama.cpp was not found here"}
    check = out / "roundtrip.partial.gguf"
    base = imatrix.launcher(tool) + ["--in-file", target, "-o", check]
    rc, ok = -1, False
    try:
        rc, tail = imatrix.run_streaming(base + ["--output-format", "gguf"], im, 1, None)
        if rc != 0 and any("output-format" in line for line in tail):
            check.unlink(missing_ok=True)
            rc, tail = imatrix.run_streaming(base, im, 1, None)
        ok = rc == 0 and check.is_file()
        if ok:
            a, b = ggufinfo.read(target), ggufinfo.read(check)
            ok = sorted(t["name"] for t in a["tensors"]) == sorted(t["name"] for t in b["tensors"])
    except imatrix.Stopped:
        raise
    except (OSError, ValueError):
        ok = False
    finally:
        check.unlink(missing_ok=True)
    res = {"ran": True, "ok": bool(ok), "version": imatrix.tool_version(tool)}
    if not ok:
        res["warning"] = (f"llama-imatrix did not accept the adapted matrix as a file to read and write again (exit code {rc}; see the log). It is kept, but llama-quantize may not take it. "
                          "Whether llama.cpp accepts an adapted matrix has not been verified.")
    return res


def run(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(adapt_job(job))
    return 0


def run_preview(job: dict) -> int:
    protocol.reserve_stdout()
    protocol.result(preview(job))
    return 0
