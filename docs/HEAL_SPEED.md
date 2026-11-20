# Heal speed on one RTX 5090 (Windows): measure first (experimental)

> **Built from a research report (docs: the lead's HEAL_SPEEDUP_5090 report); nothing here was run on a 5090 or on Windows.** Every statement about speed below is a hypothesis that the measurement on your PC must confirm or refute. Nothing runs unless you press a button.

## 0b. Which GatedDeltaNet kernels really run

`speed.gdn_kernels` looked for a `chunk_gated_delta_rule` attribute on the layers; transformers 5.23 has none (the choice is made once, at import time, in the modeling module). It now reads: the old attribute (older versions), the module's `chunk_gated_delta_rule`, its `is_fast_path_available` flag, an fla callable next to the torch reference (`torch_chunk_gated_delta_rule`), and the warning transformers logs when it falls back (a logging handler installed when `speed` is imported, before the model loads). The answer is `{"delta_rule": "fla" | "torch" | "unknown", "causal_conv1d": ..., "how": what it is based on}`; "unknown" is said plainly and never guessed. It is logged (`kernels: {...}`) and added to the result of the heal (student) and the teacher's cache job, and printed in the heal speed report.

## 0c. "Measure heal speed" (Heal card)

One button; off unless pressed. It runs the existing speed test (`distill` with `bench`) with extra keys: `profile` (two more steps under `torch.profiler`; wall seconds of forward, the KL loss, backward and the optimizer with a synchronize at each border, and the GPU seconds of kernels classified by NAME as GatedDeltaNet or matrix multiplication: rough, `heal_profile`, with the 10 largest kernels), `microbench` (one GatedDeltaNet forward + backward with the PyTorch reference and, when importable, fla: the call signature is the reference implementation's and unverified, so a failure is a line in the report), and optionally `teacher_bench` (the original's answers per rows-per-forward 1, 2, 4, 8, 16 on about 200 rows: loads the original, slow). `heal_speed_report.md/.json` is written to `benchmarks/heal_speed-<model>/` on EVERY end (also on a failure), with secrets, user names and home folders taken out; the card shows the path with Copy path / Open the folder and the text. The report also lists the Windows hints below.

## 1. Pinned fast kernels

`triton-windows` is no longer installed unpinned: it is pinned from the installed torch (2.14.x -> `>=3.8,<3.9`, 2.15 -> `>=3.9,<3.10`; an unknown torch leaves Triton as it is and says so), `flash-linear-attention>=0.6` is installed, and afterwards a check imports `fla.ops.gated_delta_rule` and runs a tiny forward + backward on the graphics card. The check says plainly ok / does not import / kernel failed on this card; **nothing is rolled back silently** (the environment is as the install left it, and a failed install leaves it as it was). The existing "Install the fast kernels" button does exactly this; the standard setup already tried these packages before (now with the pin).

## Windows hints (shown, never changed)

Hardware-accelerated GPU scheduling (HAGS), the power plan, and whether a display is attached to the training card (`nvidia-smi`): one line on the Heal card when something is probably wrong, all of them in the report. The program never changes a Windows setting. The research report expects 0-10 % from them: unmeasured.

## The original's answers: rows per forward by a token budget

The cache job takes as many rows per forward as fit `teacher_batch_tokens` (setting, default 16,384; rows sorted by length, padded to the longest of the batch, at most 16 rows; 0 = the fixed 4 rows of before; the job key is only sent when it is above 0). The answers are the same up to rounding and the cache's identity, `rows_key` and store id do not depend on it (the test compares a cache made with 400 tokens against one made with 4 rows: same rows, same indices and probabilities within tolerance, same index keys; that test needs torch and runs in CI). An out-of-memory error halves the budget (and ends at one row). Whether it is faster is the teacher bench's question.

## Not built yet (they come after your profiler numbers)

Student micro-batching / packing (a 3-token conv1d boundary leak in transformers 5.23 must be fixed or tested away first), partial checkpointing, torch.compile, Liger, the Unsloth heal path.
