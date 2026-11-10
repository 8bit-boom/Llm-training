# Training speed: what was looked at, what is built, what is next

Researched 2026-11-10 for an RTX 5090 (Blackwell, sm_120, 32 GB, Windows) and a Tesla T10 (Turing, 16 GB) with Qwen3.5 and Gemma 4, Unsloth first and Transformers + PEFT + TRL as the fallback.
**Nothing here was benchmarked on a 5090**; the sources are vendor claims, papers and forum reports (listed at the end). The Unsloth pages could not be fetched from the sandbox, so Unsloth's numbers come from search summaries.

## Verdict by idea
| Idea | Verdict | Why |
|---|---|---|
| Cut padding: batches of similar length now, padding-free later | **Built (bucketing)** | Our examples range from one line to long world-book pieces; every batch is padded to its longest example. Padding-free batching measured 1.4-2x elsewhere (IBM / Hugging Face), depending on the length spread; it needs a FlashAttention-2 style kernel. |
| Qwen3.5 fast kernels (flash-linear-attention, causal-conv1d) | **Built (detected and warned)** | Qwen3.5 has linear-attention layers; without these kernels Transformers uses a pure-PyTorch fallback reported as ~10x slower (one secondhand source). Unsloth brings its own. causal-conv1d is hard to build on Windows. |
| Gradient checkpointing only when memory needs it | **Built for the plain-libraries path** | Checkpointing costs about a third more time. The path tests the longest batch without it and keeps it off if it fits with room to spare. Unsloth manages its own checkpointing and is left alone. Not benchmarked. |
| FlashAttention | Not worth chasing on the 5090 | PyTorch's built-in attention is already close to the card's limit; FlashAttention-4 targets datacenter Blackwell (sm_100/103), not sm_120; FlashAttention-2 wheels for Windows are community builds (CUDA 13 / nightly). Its real value for us is padding-free batching. |
| SageAttention | Skip | Mainly an inference tool; its 8-bit training variant (SageBwd) is research code, not in Hugging Face or Unsloth. |
| `torch.compile` | Low priority | Unsloth already compiles its hot paths; Windows needs community Triton builds; varying shapes recompile unless lengths are bucketed. |
| QLoRA / 4-bit | Not for speed | Slower than 16-bit LoRA (one anecdote: ~23 %) and Unsloth advises against 4-bit for Qwen3.5. All five models fit the 5090 in 16-bit. |
| FP8 / NVFP4 | Not now | Unsloth reports its first FP8 LoRA was ~4x slower than BF16 on an H100, ~80 % of BF16 after rework: a memory saving. FP4 training on consumer cards is poorly documented. |
| Fused / chunked cross-entropy (Liger, cut cross-entropy) | Later, plain-libraries path only | ~250k-token vocabularies make the logits large; Unsloth already handles it. Liger support for Qwen3.5 / Gemma 4 not confirmed. |
| Tesla T10 | Few options | No bf16, no FlashAttention (needs sm_80+): fp16 and PyTorch's memory-efficient attention. Gemma in fp16 stability: unverified. |

## What is built (worker `speed.py`, `trainjob.py`, `callbacks.py`)
- **Bucketing** (`bucket` = `auto` | `on` | `off` in the job's hyper settings; default `auto`): before each stage the padding of random batches is estimated; when it is 8 % or more, batches are made of examples of similar length (shuffled in pieces of 8 batches, sorted inside each piece, batch order shuffled, a new grouping every epoch). Measured on a tiny CPU run with a deliberately mixed-length set: padding 49.7 % -> 8.3 %, real tokens per second +47 %, training time -25 %. This shows the mechanism only.
- **A side effect to watch**: with bucketing, a batch holds examples of one kind more often (length goes with kind: long world-book text, short answers), so the loss moves more from one step to the next; each optimizer step still averages several shuffled batches when gradient accumulation is above 1. In Expert mode, *Batches by length* switches it `off` (or `on`). Whether it changes what the model learns has not been measured: compare two runs on your data before you rely on it.
- **Numbers in the window**: every metric carries `tok_s` and `pad_pct` (running averages since the stage began); a `speed` event per stage gives the estimated padding with and without bucketing, mean and longest length, the attention backend, the checkpointing state and any warnings (Qwen3.5 fast kernels missing on the plain-libraries path, a card without bf16).
- **Checkpointing probe** (plain-libraries path, LoRA / QLoRA only): switched off when the longest batch needs at most 85 % of the card's memory without it. The decision logic and the switching are tested; the probe itself needs a CUDA card and has not run on one.

## Also built (benchmark and advanced options)
- **Benchmark** (Step 3, "Find the fastest settings"; `train/bench.rs`): 16 real training steps for each of several settings, one worker process per setting (a setting that runs out of memory only fails its own row), no checkpoints and no sample answers. Compared: your current settings, batches grouped by length on / off, a batch of 2x and 4x with the accumulation steps reduced to match, and with "also try the advanced options" torch.compile, Liger and padding-free (plain-libraries path only; with Unsloth they would be ignored). The table shows tokens per second, the factor against your current settings and the memory peak; "Use the fastest settings" applies them to the next run. Results are kept in `benchmarks/<model>-<method>.json`. This computer only. Start-up time is part of every number alike, so only clear differences mean something.
- **Advanced options** (`compile`, `liger`, `padding_free` = `off` | `on`, default off; Expert mode): applied only on the plain-libraries path, and each switches itself off, with the reason shown in the window, when it cannot work: torch.compile needs a graphics card and Triton; Liger needs the liger-kernel package and a model type it supports; padding-free batching (examples end to end in one row with restarting position ids, FlashAttention-2 finds the boundaries) needs the flash-attn package and an RTX 30 series or newer, and stays off for Qwen3.5 and Gemma 4 until their attention layers are checked for it. The gating, the padding-free batch layout (positions, labels across a boundary) and "everything asked for on a machine without a card" are tested; **none of the three options has run on a graphics card**.

## Next
1. Run a Quick test on the 5090 and read the `speed` line: how much padding, which attention backend, are the Qwen3.5 kernels active.
2. Run the benchmark on the 5090 (also with the advanced options if the plain-libraries path is in use) and make the winning settings the defaults.
3. Check Qwen3.5 and Gemma 4 for padding-free batching (their linear-attention and sliding-window layers) before it can be enabled for them.

## Sources
Unsloth docs (Blackwell / RTX 50, Qwen3.5, Gemma 4, FP8 RL); NVIDIA developer blog on Unsloth for Blackwell; VeOmni and ms-swift Qwen3.5 guides; IBM / Hugging Face on padding-free packing; TRL "reducing memory usage"; the SageAttention3 / SageBwd paper (arXiv 2505.11594); the Chronicals preprint (arXiv 2601.02609, self-reported); community notes on FlashAttention-4 for sm_120 and Windows wheels; Turing / T4 notes on FlashAttention and bf16.
