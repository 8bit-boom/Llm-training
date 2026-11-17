# Third-party notices

ND Lore Forge includes or is derived from the following MIT-licensed work. The full notice texts sit in the header of
the files named below.

| Work | Used for | Where | Licence |
|---|---|---|---|
| llama.cpp / ggml (c) 2023-2026 The ggml authors | the K-quant reference quantizers (`ggml/src/ggml-quants.c`), ported to numpy; the tools we run (`llama-quantize`, `llama-imatrix`, `llama-perplexity`) are used as separate programs, not included | `worker/ndforge_worker/kquants.py`, `worker/ndforge_worker/quantblocks.py` | MIT |
| gguf-py (part of llama.cpp) | the block layouts and the dequantizers and legacy quantizers (Q4_0, Q5_0, Q5_1, Q8_0), ported to numpy; the `gguf` package is also used by tests, not shipped | `worker/ndforge_worker/quantblocks.py` | MIT |
| quant-convert-gui (c) 2026 8bit-boom (same owner) | `kquants.py` (the numpy port of the K-quant encoders, with a golden fixture made by the real ggml encoder) | `worker/ndforge_worker/kquants.py`, `worker/tests/fixtures/kquants_golden.npz` | MIT |

`worker/ndforge_worker/mixedquant.py` is in turn the origin of `quant_gui/mixed_quant.py` in quant-convert-gui (a port of this repository's file; see `docs/MIXED_QUANTS.md`).

## MIT License (quant-convert-gui)

Copyright (c) 2026 8bit-boom

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## MIT License (llama.cpp, ggml, gguf-py)

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
