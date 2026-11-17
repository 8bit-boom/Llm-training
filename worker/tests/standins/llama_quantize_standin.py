"""A stand-in for llama-quantize that really quantizes a small 16-bit GGUF (numpy encoders from the worker): enough to test the pipeline end to end without llama.cpp.

It understands `--pure`, `--imatrix FILE` (checked against the tensor shapes, then ignored: its rounding is the no-matrix reference rounding), `--tensor-type REGEX=TYPE` (first match wins),
`--tensor-type-file FILE`, `--output-tensor-type`, `--token-embedding-type`, `--help` and `--version`, then `INPUT OUTPUT FTYPE`. Types come from the same rule semantics as mixedquant.tensor_type;
a plain (not `--pure`) Q4_K_M / Q5_K_M gets the approximate mixture of quantfit.simulate_plain. Environment switches for tests: STANDIN_NO_TYPEFILE (the help does not list
--tensor-type-file), STANDIN_FLIP (flip a byte in every K-quant block: the self-check must fail), STANDIN_FAIL (exit 1), STANDIN_SLOW (seconds to sleep before working; so a stop can arrive),
STANDIN_LOG (append the argument list as a JSON line to this file).

This proves the pipeline is consistent. It proves nothing about llama.cpp itself."""
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from ndforge_worker import ggufinfo, ggufrows, mixedquant, quantblocks, quantfit  # noqa: E402

HELP = """usage: llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--output-tensor-type] [--token-embedding-type] [--tensor-type]%s model-f32.gguf [model-quant.gguf] type [nthreads]
  --pure: Disable k-quant mixtures and quantize all tensors to the same type
  --imatrix file_name: use data in file_name as importance matrix for quant optimizations
  --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor
  --token-embedding-type ggml_type: use this ggml_type for the token embeddings tensor
  --tensor-type TENSOR=TYPE: quantize this tensor to this ggml_type. example: --tensor-type attn_q=q8_0
%s"""
BASE = {"Q2_K": "Q2_K", "Q3_K_M": "Q3_K", "Q3_K_S": "Q3_K", "Q4_K_M": "Q4_K", "Q4_K_S": "Q4_K", "Q5_K_M": "Q5_K", "Q5_K_S": "Q5_K", "Q6_K": "Q6_K", "Q8_0": "Q8_0"}


def main(argv):
    if os.environ.get("STANDIN_LOG"):
        with open(os.environ["STANDIN_LOG"], "a", encoding="utf-8") as fh:
            fh.write(json.dumps(argv) + "\n")
    if "--help" in argv or "-h" in argv:
        typefile = not os.environ.get("STANDIN_NO_TYPEFILE")
        print(HELP % (" [--tensor-type-file]" if typefile else "", "  --tensor-type-file FILE: read per-tensor type rules from FILE, one TENSOR=TYPE per line" if typefile else ""), file=sys.stderr)
        return 1
    if "--version" in argv:
        print("version: 4321 (standin)")
        return 0
    if os.environ.get("STANDIN_FAIL"):
        print("error: standin told to fail", file=sys.stderr)
        return 1
    if os.environ.get("STANDIN_SLOW"):
        time.sleep(float(os.environ["STANDIN_SLOW"]))
    pure, matrix, out_t, emb_t, rules, rest, i = False, None, None, None, [], [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--pure":
            pure = True
            i += 1
        elif a in ("--imatrix", "--output-tensor-type", "--token-embedding-type", "--tensor-type", "--tensor-type-file"):
            v = argv[i + 1]
            if a == "--imatrix":
                matrix = v
            elif a == "--output-tensor-type":
                out_t = v.upper()
            elif a == "--token-embedding-type":
                emb_t = v.upper()
            elif a == "--tensor-type":
                rx, t = v.rsplit("=", 1)
                rules.append((rx, t.upper()))
            else:
                for line in Path(v).read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rx, t = line.rsplit("=", 1)
                        rules.append((rx, t.upper()))
            i += 2
        else:
            rest.append(a)
            i += 1
    src_path, dst_path, ftype = rest[0], rest[1], rest[2]
    src = ggufrows.open_source(src_path)
    if matrix:
        check_matrix(matrix, src)
    base = BASE[ftype]
    mixture = {} if pure or ftype not in ("Q4_K_M", "Q5_K_M") else quantfit.simulate_plain(src, ftype)
    tied = "output.weight" not in {t["name"] for t in src["tensors"]}
    new_types, data = {}, {}
    for t in src["tensors"]:
        name, cols = t["name"], t["dims"][0]
        if not mixedquant.quantizable(name, t["dims"]):
            new_types[name] = t["type"]
            continue
        ty = mixture.get(name, base)
        if name == "token_embd.weight" and emb_t:
            ty = emb_t
        if name == "output.weight" and out_t:
            ty = out_t
        for rx, rt in rules:
            if re.search(rx.lower(), name):
                ty = rt
                break
        ty = mixedquant.effective_type(ty, cols)
        new_types[name] = next(k for k, v in ggufinfo.TYPES.items() if v[0] == ty)
    write(src, dst_path, new_types)
    return 0


def check_matrix(path, src):
    """llama-quantize complains when the matrix does not fit a tensor: the statistics must be cols x experts long"""
    info = ggufinfo.read(path)
    by = {t["name"]: t for t in info["tensors"]}
    for t in src["tensors"]:
        s = by.get(t["name"] + ".in_sum2")
        if s is not None and mixedquant.quantizable(t["name"], t["dims"]):
            cols, _r, experts = ggufrows.geometry(t)
            if ggufinfo.numel(s) != cols * experts:
                print(f"error: imatrix size {ggufinfo.numel(s)} is different from tensor size {cols * experts} for {t['name']}", file=sys.stderr)
                sys.exit(1)


def write(src, dst_path, new_types):
    align = src["alignment"]
    blobs, layout, off = [], [], 0
    flip = bool(os.environ.get("STANDIN_FLIP"))
    for t in src["tensors"]:
        ty = new_types[t["name"]]
        cols, per, experts = ggufrows.geometry(t)
        if ty != t["type"] or t["type"] not in ggufrows.RAW:
            tname = ggufinfo.type_name(ty)
            w = ggufrows.read_rows(src, t, range(per * experts))
            raw = quantblocks.encode(w, tname)
            if flip and tname in quantblocks.QUANTIZED and tname.endswith("K"):
                raw = raw.copy()
                raw[:, 0] ^= 0xFF
            blob = raw.tobytes()
        else:
            n = ggufinfo.numel(t) * ggufrows.RAW[t["type"]][1]
            with open(src["path"], "rb") as fh:
                fh.seek(src["data_start"] + t["offset"])
                blob = fh.read(n)
        blobs.append(blob)
        layout.append((t, ty, off))
        off = (off + len(blob) + align - 1) // align * align
    with open(src["path"], "rb") as fh:
        head = fh.read(src["kv_end"])
    infos = b"".join(ggufrows._string(t["name"]) + struct.pack("<I", len(t["dims"])) + b"".join(struct.pack("<Q", d) for d in t["dims"]) + struct.pack("<IQ", ty, o) for t, ty, o in layout)
    header = b"GGUF" + head[4:8] + struct.pack("<Q", len(layout)) + head[16:24] + head[24:]
    start = (len(header) + len(infos) + align - 1) // align * align
    with open(dst_path, "wb") as fo:
        fo.write(header + infos + b"\0" * (start - len(header) - len(infos)))
        for blob, (_t, _ty, o) in zip(blobs, layout):
            fo.seek(start + o)
            fo.write(blob)
        fo.truncate(start + off)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
