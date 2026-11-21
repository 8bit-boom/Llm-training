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

## Triton and torch must belong together

`triton-windows` has to follow the minor of the installed PyTorch. The pin is read from the torch wheel's own requirement (`importlib.metadata.requires("torch")`: `triton==3.x.y`) and becomes `triton-windows>=3.x,<3.(x+1)`; the table in `envmgr/setup.rs` (torch 2.14, 2.15) is only the fallback, and a torch that is in neither leaves Triton alone ("unknown").
The post-install check, the heal-speed report (`environment.triton_match`, additive) and a light on the Settings page and the Heal card say: `Triton 3.9.0 does not match torch 2.11.0 (wants 3.6.x): press Install fast kernels to fix.` Nothing is installed automatically; only the existing "Install the fast kernels" button runs pip.

## The first measurement on the owner's PC (RTX 5090, healed 19.9 B student, QLoRA NF4)
Settings: batch 1, max_seq 4096, grad_accum 32, gradient checkpointing on, head checkpoint on. Training data: 96,945 rows (9,061 texts and 87,884 conversations) = 16.8 M tokens, about **173 tokens per row**; every micro-step is ONE row.

| | measured |
|---|---|
| bench (synthetic targets, 3 timed steps of about 470 tokens) | 324 tokens/s, 1,414 tokens in 4.37 s, peak 21.3 GB |
| a real heal round | 5 M tokens in 221 minutes = 377 tokens/s (so the bench is representative) |
| one step (1.146 s) | forward 0.502, loss 0.006, backward 0.612, optimizer 0.025 |
| GatedDeltaNet kernels (fla is active) | 0.0054 s = 0.5 % of the step |
| matrix multiplications | 0.348 s = 30 % of the step; cutlass bf16 GEMMs together about 37 % of GPU time |
| top kernel | bitsandbytes `kDequantizeBlockwise` 0.168 s = 17.7 % of GPU time (more than any GEMM) |
| GPU time against wall time | about 0.95 s of 1.146 s: about 17 % launch / CPU gaps |

What the numbers rule out: **FlashAttention, Triton / GatedDeltaNet kernel work and Liger** (the GDN share is 0.5 %; they cannot help). What remains: the cost is per forward pass (NF4 dequantization of every linear layer, launch gaps) and each forward pass carries ONE row of about 173 tokens. **Putting several rows in one forward pass is the main lever** (GEMM time grows with tokens; dequantization and gaps are per forward). The gain is an estimate until measured: roughly 2 to 3 times at 4 to 8 thousand tokens per forward pass, a ceiling near 1,300 tokens/s. Optional dequantization tricks only after batching has been measured.

## Student token-budget micro-batching (`hyper.student_batch_tokens`, off by default)
* **What it does.** The rows of ONE optimizer update (batch x gradient accumulation, 32 by default, in the order the heal always used) arrive together and are split by `speed.split_by_token_budget`: from the longest row to the shortest, a micro-batch is closed when one more row would pass the budget once padded; a row longer than the budget is a micro-batch of its own. The micro-batches run in a seeded shuffled order (`speed.micro_batches`, seed and update number), are right-padded with an attention mask, and each is back-propagated at once (only one is in memory). The sequences per update, the order of the rows and the number of updates (steps, checkpoints) are exactly those of the heal without a budget.
* **Loss.** Every row keeps the weight it has as a batch of its own: its own mean over its tokens, then the mean over the update's rows (`distilljob.student_loss_rows`: the per-token weight is 1 / the row's length; NOT a token-weighted mean). Only with the KL term (alpha 1); with a cross-entropy term the budget is ignored with a log line.
* **Why right padding is exact.** A causal model never lets a real position see a later (padded) one: attention is causal, and the GatedDeltaNet short convolution and recurrence are causal too. transformers 5.23 also zeroes padded positions before the GatedDeltaNet layers (`apply_mask_to_padding_states`) and builds a recurrent mask. Tests on a tiny Qwen3.5-style model (the PyTorch reference path of the layers) show a row's hidden states are the same alone and inside a longer padded batch for three different pad tokens, and that the loss and every gradient of a padded batch equal the average of the single-row steps within 1e-5, for rows of 3 to 40 tokens. **The fla (Triton) path with padding is covered only by a run on the owner's GPU.**
* **Memory and retries.** The activations scale with tokens per forward pass. The automatic out-of-memory ladder halves the budget FIRST (down to 512, then one row per forward pass again), before checkpointing, the batch or the KL chunk. The Rust "Try again with safer settings" step does not know the key yet (the worker's retries do).
* **Measuring.** "Measure heal speed" runs the update loop for 0, 2048, 4096, 8192 and 16384 tokens per forward pass on a seeded random sample of REAL rows (their true length distribution is in the report: mean, median, 90 %, longest) with the cache's real answers when it is complete, else seeded synthetic answers that are a function of the row (so every budget sees the same targets), the weights frozen (learning rate 0) so every budget starts from the same model. The table has tokens/s, speed-up, peak VRAM, the first update's loss and the forward passes per update. **Recommended:** the largest speed-up with a peak below 90 % of the card and the same first loss (within 1e-3, relative), only when it is at least 5 % faster. The Heal card shows it with the button "Use the recommended batch size" (never automatic); the Expert field "Tokens per forward pass" and the Auto page's field set it by hand.
* **Detector.** `kernels.conv` now says "torch" when transformers logged that `causal_conv1d_fn` falls back to its reference PyTorch implementation (or the function is a PyTorch reference); `delta_rule: fla` is kept. Nothing is installed for it (the GatedDeltaNet share is 0.5 %).
