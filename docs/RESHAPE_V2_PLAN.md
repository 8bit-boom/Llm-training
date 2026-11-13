> This is the plan of version 2 as it was written before the code, kept as it was (like `docs/RESHAPE_PLAN.md`). What the code actually does, the owner's answers to section 7, the deviations
> from this plan and what has not been verified is in `docs/RESHAPE.md` ("Version 2"). The numerical check it mentions (`.../scratchpad/wp/verify_width.py`) was a scratch script outside the
> repository; it became the CI test `worker/tests/test_reshape_width_torch.py`.

# ND Lore Forge: "Slim or grow a model" v2 — WIDTH pruning (Minitron-style) + distillation healing — implementation plan

Status: plan only. Nothing in `/home/user/Llm-training` or `/home/user/nd-world` was changed. Written 2026-11-13 against `/home/user/Llm-training` main
(`e702176`, v1 merged: `docs/RESHAPE.md`, `docs/RESHAPE_PLAN.md`), transformers **5.22.2** (`modeling_qwen3_5.py`, read from the wheel in the scratch
`trainenv`), llama.cpp `master` copies from the v1 session (`conversion/qwen.py`, `src/models/qwen35.cpp`, `src/llama-model.cpp`, `src/llama-hparams.cpp`),
NVIDIA Model-Optimizer `main` (`modelopt/torch/prune/plugins/mcore_minitron.py`, `examples/pruning/README.md`, `examples/megatron_bridge/README.md`,
`examples/megatron_bridge/distill.py`, fetched from raw.githubusercontent.com), the real `config.json` + `model.safetensors.index.json` of
`Qwen/Qwen3.5-27B`, `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.8-27B` (Hugging Face), Minitron model cards/configs on Hugging Face, the HF paper API abstracts of
arXiv 2407.14679, 2408.11796, 2504.11409, 2305.11627, and the LLM-Pruner README.

Legend (as v1): **[V]** verified in source/data I read or in a run I made (cited) · **[A]** assumption, must be checked · **[GPU]** only a real GPU run can verify.

Network note: arxiv.org, developer.nvidia.com and github.com were blocked in this session (HF and raw.githubusercontent.com were reachable). Paper
*bodies* were therefore not read; claims about them come from the abstracts, NVIDIA's model cards, and NVIDIA's own implementation + guidelines in
Model-Optimizer, which is the maintained code of the Minitron method. Where I rely on memory of a paper body it is marked [A].

**A numerical check was run** (scratch script `.../scratchpad/wp/verify_width.py`, CPU torch 2.15 + transformers 5.22.2, tiny random Qwen3.5 text model
with 8 layers = 2 blocks, *all* weights perturbed): for every width surgery below, the sliced model's logits equal the original model with the removed
structures masked (their input columns of the output projection zeroed), max abs difference ≤ 1.9e-6 in fp32 (masking itself changes the logits by 0.9–3.2):

| Surgery | max \|masked − sliced\| |
|---|---|
| MLP intermediate 128 → 80 (different kept set per layer, same count) | 0.0 |
| full-attention query heads 6 → 4 (3 → 2 per KV group, gated `q_proj`) | 0.0 |
| a whole KV group removed (kv 2 → 1, heads 6 → 3) | 0.0 |
| GatedDeltaNet value heads 6 → 4 (3 → 2 per key head) | 0.0 |
| GatedDeltaNet key head removed with its value heads (nk 2 → 1, nv 6 → 3) | 1.9e-6 |
| MLP + attention heads + GDN value heads together | 5.0e-7 |
| bf16 tensor sliced as raw `uint16` with numpy, read back as bf16 | bit-exact |

So the index bookkeeping in §2 is **[V]** (CPU, tiny model). That script is the template for the CI test in §6.2.

---

## 0. Verdict and recommended scope

1. **v2 prunes MLP width (`intermediate_size`) by default, optionally combined with v1's depth cut. GatedDeltaNet value heads and full-attention
   query heads are Expert-mode options. `hidden_size` is NOT pruned in v2.** Head dims (`head_dim`, `linear_*_head_dim`) are never pruned.
   - MLP is 17.11 B of the 27.78 B parameters (62 %) [V: formula §1.1 reproduces the Hub totals exactly]; it is the axis NVIDIA calls the most robust
     ("MLP dimensions (`ffn_hidden_size`) can typically be pruned more aggressively than embedding dimensions (`hidden_size`) and attention/Mamba
     dimensions" — Model-Optimizer `examples/pruning/README.md`, Width Pruning) [V].
   - **Hidden size is excluded** because (a) the vision merger writes into it (`model.visual.merger.linear_fc2`: out = `vision_config.out_hidden_size`
     = 5120 = text `hidden_size` [V config + `Qwen3_5VisionPatchMerger`]) so slicing hidden forces slicing the merger output, the official mmproj
     GGUF no longer fits (re-convert needed), and text-only distillation never trains the image path; (b) NVIDIA's own tool refuses it for exactly this
     model family: *"The language model of vision-language models (e.g. Qwen3.5-VL, Gemma3-VL) can be pruned as well; the vision tower is left intact.
     `hidden_size` is not pruned for VLMs as it is shared with the vision projector"* [V `examples/pruning/README.md` note 2; `mcore_minitron.py` ~l.369:
     "unsafe for `hidden_size` on VLMs (shared with the frozen vision projector…)"]; (c) it touches every tensor (embeddings 1.27 B, lm_head 1.27 B,
     every norm, `mtp.fc` 2H×H), the largest surgery for the most fragile axis. Honest cost: Minitron's best width models DO prune hidden
     (Llama-3.1-Minitron-4B-Width: hidden 4096→3072, MLP 14336→9216, heads unchanged [V HF config + card]), so v2 gives up some quality per parameter.
     A text-only variant (drop vision) could prune hidden later (v3).
   - **Attention heads** are only 1.68 B in 16 layers; 24 → 16 heads saves 0.52 B (+ MTP). Model-Optimizer does not support head pruning for gated
     attention or GatedDeltaNet (README note 3) — a tool limitation, the slicing itself is exact (table above) — and its own example skips heads
     (`--hparams_to_skip num_attention_heads` for Qwen3-8B [V megatron_bridge README]). Expert option only.
   - **GatedDeltaNet value heads** (48 → 32 per GDN layer) save 1.52 B on the 27B. Analogue: Nemotron-H 8B→4B pruned Mamba heads 128 → 112 (12.5 %)
     with group-aware pruning [V README examples; 2504.11409 abstract "group-aware pruning strategy that preserves the structural integrity of SSM
     blocks"]. 48→32 is a 33 % cut of the recurrent layers' capacity: Expert option, not default.
2. **Width vs depth.** NVIDIA: width = "better accuracy than depth-pruned models at a fixed number of parameters", depth = "faster inference … at a
   fixed number of parameters" [V README]. 2408.11796 compares both on Llama-3.1-8B→4B with 94 B distillation tokens each [V model cards: both
   "continued training with distillation using 94 billion tokens"]. So: width for quality, depth for speed — the page must say both.
3. **Healing is not optional for width pruning.** Every NVIDIA recommendation assumes distillation with "~80-100B tokens" and says ">50% pruning is
   hard to recover" [V README]. ND Lore Forge on one RTX 5090 can afford ~10–50 M tokens (§3.6). Therefore v2's **default target is mild**: ~15–25 %
   of the parameters (27B → ~21–23 B; 9B → ~7.5–8 B), and the page says clearly that the result is a weaker general model that is then trained on the
   world. Distillation is the new "Heal" step; the normal Step 3 training (unchanged) teaches the world afterwards.
4. **LLM-Pruner is not used.** It supports Llama/Vicuna/Baichuan/ChatGLM/BLOOM, no Qwen [V README "Supported LLMs"]; Taylor importance needs
   backward passes (memory); its pruned models are saved without `.from_pretrained()` support ("[ ] Support `.from_pretrained()`" open TODO [V]) and
   it prunes only some layers, i.e. per-layer dimensions — which transformers' `qwen3_5` and llama.cpp's `qwen35` cannot load (§2.3). Borrowed idea:
   coupled-structure groups (our §2 table) and LoRA recovery.
5. **Uniform width across layers is mandatory** for this architecture: transformers builds every MLP from `config.intermediate_size`
   (`Qwen3_5DecoderLayer`: `Qwen3_5MLP(config, config.intermediate_size)`), every attention/GDN from global config keys [V modeling_qwen3_5.py
   l.848–860, 490–533, 736–761]; llama.cpp's qwen35 loader uses `n_ff`, `n_head`, `ssm_*` without a layer index (`create_tensor(... {n_embd, n_ff})`,
   `n_embd_head_k * n_head * 2`) [V qwen35.cpp l.59–101]. Which neurons are kept differs per layer; how many does not. Per-layer budgets are refused.

### 0.1 Size arithmetic (exact formula, §1.1; vision 0.461 B and MTP included)

Qwen3.5-27B / Qwen3.8-27B (same dims [V configs]): H 5120, I 17408, L 64 (48 GDN + 16 full), heads 24, kv 4, head_dim 256, GDN nk 16, nv 48, dk = dv = 128,
total 27.781 B.

| Variant | Layers | I | GDN nv | heads | Params |
|---|---|---|---|---|---|
| original | 64 | 17408 | 48 | 24 | 27.78 B |
| **width, MLP only (default for ~21 B)** | 64 | **10752** (−38 %) | 48 | 24 | **21.14 B** |
| width, MLP only, milder (recommended first try) | 64 | 13312 (−24 %) | 48 | 24 | 23.69 B |
| width, MLP + GDN heads | 64 | 12288 | 32 | 24 | 21.15 B |
| width, MLP + GDN + attention heads | 64 | 13312 | 32 | 16 | 21.64 B |
| **depth + width** (1 block = 4 layers, v1) | 60 | 12288 (−29 %) | 48 | 24 | 21.46 B |
| depth + width (2 blocks) | 56 | 13312 (−24 %) | 48 | 24 | 21.15 B |
| v1 depth only (4 blocks) | 48 | 17408 | 48 | 24 | ≈21.6 B (v1 doc) |

Qwen3.5-9B: H 4096, I 12288, L 32, heads 16, kv 4, nk 16, nv 32, total 9.653 B → I 9216: 8.41 B; I 8192: 7.99 B; I 9216 + nv 24 (Expert) 8.10 B;
depth −1 block + I 10240: 8.06 B.

Divisor: I is a multiple of **256** (k-quants in GGUF work on rows of 256 values; `down_proj` has I as its inner dimension [A: llama-quantize falls
back to a worse type for rows not divisible by 256 — check on the first GGUF]); Model-Optimizer uses an `ffn_hidden_size_divisor` of 512 by default
[V `mcore_minitron.py` l.835–856]. GDN nv ∈ multiples of nk (16): 48, 32 (one step only in the UI). Heads ∈ multiples of kv (4): 24, 20, 16.
Limits: at most 50 % of I, at most one GDN step, heads ≥ 16 (27B) / ≥ 12 (9B) [design choice, mirrors NVIDIA's "max_width_pruning 0.4"/">50% hard"].

### 0.2 Honest expectations

- **Before healing** a 38 % MLP cut is a damaged model (much higher loss on the calibration text; answers degrade visibly) [GPU: measured by the
  existing loss check and the Test page]. 24 % is clearly better. Show "before healing" numbers on the page, never hide them.
- **After healing with our budget** (20–40 M tokens): expect recovery of fluency and most behaviour on the kind of text used for healing, a
  remaining gap in general knowledge/reasoning versus the original. NVIDIA needed 94 B tokens to *beat* models of the same size — we will not match
  that; we aim at "a usable 21 B that runs on 24 GB cards after Step 3", not at "as good as 27 B". [GPU]
- **Speed**: fewer FLOPs per token in the MLP (≈ −24…38 % of MLP time); depth pruning is faster per parameter removed [V README].
- **Memory for running it**: Q4_K_M ≈ 0.6 bytes/param → ~13 GB for 21 B vs ~17 GB for 27 B [A].

---

## 1. Facts the design depends on

### 1.1 Parameter formula (per layer, Qwen3.5) [V: reproduces the Hub totals exactly]
`params(L, I, nh, nkv, hd, nk, nv, dk, dv, K, H, V)` with P = 4, `n_full = L/4`, `n_lin = L − n_full`:
- MLP: `3·H·I`; norms: `2·H` per layer
- GDN: `H·(2·nk·dk + nv·dv)` (in_proj_qkv) `+ H·nv·dv` (z) `+ 2·H·nv` (a, b) `+ (2·nk·dk + nv·dv)·K` (conv1d) `+ 2·nv` (A_log, dt_bias) `+ dv` (norm) `+ nv·dv·H` (out_proj)
- full attention: `H·nh·hd·2` (q + gate) `+ 2·H·nkv·hd` (k, v) `+ nh·hd·H` (o) `+ 2·hd` (q_norm, k_norm)
- embed `V·H`, lm_head `V·H` (27B and 9B are untied: `tie_word_embeddings: false` [V]), final norm `H`
- MTP (`mtp_num_hidden_layers` = 1): `2·H·H` (fc) + one full-attention decoder layer with MLP + `3·H` norms
- vision: unchanged (0.4607 B for 27B, 0.4560 B for 9B; tower 0.4159 B + merger)
Check: 27B text 26.896 + MTP 0.425 + vision 0.461 = 27.781 B (Hub 27,781,427,952); 9B 8.954 + 0.243 + 0.456 = 9.653 B (Hub 9,653,104,368) [V].
dtypes: all BF16 except `linear_attn.A_log` and `linear_attn.norm.weight` in **F32** (27B F32 count 8448 = 48·48 + 48·128; 9B 3840 = 24·32 + 24·128) [V Hub
dtype counts]. The slicer keeps each tensor's dtype.

### 1.2 What the modelling code fixes [V modeling_qwen3_5.py, transformers 5.22.2]
| # | Fact | Where |
|---|---|---|
| W1 | MLP: `down(act(gate(x)) * up(x))`, no bias; neuron i = row i of gate/up, column i of down | l.811–825 |
| W2 | Full attention: `q_proj` out = `nh·hd·2`, viewed `(…, nh, 2·hd)` and `chunk(2)` → per head the rows are `[q_h (hd) | gate_h (hd)]`, head-interleaved; output `* sigmoid(gate)` then `o_proj` (in = nh·hd) | l.748–808 |
| W3 | GQA: `num_key_value_groups = nh // nkv`; `repeat_kv` maps kv head j to q heads `j·g … j·g+g−1` → kept q heads must be equal per group, original order | l.699–744 |
| W4 | `q_norm`/`k_norm` are over `head_dim` (shared by all heads) → never sliced by head pruning | l.760 |
| W5 | GDN: `in_proj_qkv` rows = `[q: nk·dk | k: nk·dk | v: nv·dv]`; `conv1d` (depthwise, weight `[conv_dim, 1, K]`, no bias) has the same channel order; `z` rows nv·dv; `b`, `a` rows nv; `A_log`, `dt_bias` [nv]; `norm` = RMSNormGated over `dv` (shared, never sliced); `out_proj` in = nv·dv | l.490–533 |
| W6 | GDN grouping: `q,k.repeat_interleave(nv // nk, dim=2)` → value head j uses key head `j // (nv/nk)`; kept value heads must be equal per key head, order kept | l.608–610 |
| W7 | Recurrence and gated norm are per value head (state `[b, nv, dk, dv]`; norm on `reshape(-1, dv)`) → removing a value head removes exactly its contribution | l.317–320, 644–647 |
| W8 | Layer = `h + mixer(norm(h))`, `h + mlp(norm(h))` (v1 F5); RMSNorm is `x·rsqrt(mean x²)·(1+w)` over **hidden** → width pruning of I/heads never changes a norm | l.828–902 |
| W9 | `Qwen3_5ForCausalLM` (what v1 loads) ignores `mtp.*` and `model.visual.*` (v1 F6) → MTP is never run by transformers; MTP shapes must still be consistent for llama.cpp | v1 F6/F8 |

### 1.3 llama.cpp [V, files read]
| # | Fact | Where |
|---|---|---|
| G1 | Converter writes `ssm_conv_kernel = linear_conv_kernel_dim`, `ssm_state_size = linear_key_head_dim`, `ssm_group_count = linear_num_key_heads`, `ssm_time_step_rank = linear_num_value_heads`, `ssm_inner_size = dv·nv` from config scalars; feed-forward length and head counts from the generic config keys | `conversion/qwen.py` `Qwen3NextModel.set_gguf_parameters` |
| G2 | Runtime: `head_k_dim = head_v_dim = ssm_d_state` → **llama.cpp requires `linear_key_head_dim == linear_value_head_dim`** (never prune head dims) | `qwen35.cpp` l.59–60 |
| G3 | All per-layer tensors use the scalar `n_ff`, `n_head`, `n_head_kv` (no `il`) → uniform width only | `qwen35.cpp` l.72–101 |
| G4 | The MTP (nextn) layer is built with the same `n_ff`/`n_head`/`n_embd` → **MTP tensors must be sliced to the same new I/heads** | `qwen35.cpp` l.111–126 |
| G5 | `_reorder_v_heads` uses `num_v_per_k = nv // nk` (grouped → tiled) → nv must stay a multiple of nk | `conversion/qwen.py` `_LinearAttentionVReorderBase` |
| G6 | Without MTP tensors and with `mtp_num_hidden_layers == 0` the converter asserts unless `--no-mtp` (v1 F8) → keep a (sliced) MTP | `_QwenMtpMixin.__init__` |
| G7 | Model type label from `n_layer`/`n_embd` only (`LLM_TYPE_UNKNOWN` otherwise) — a label | `qwen35.cpp` l.25–30 |
Unverified: Unsloth's bundled llama.cpp and Ollama's Qwen3.5 engine accept non-standard I / nv / heads [A/GPU]; k-quant row rule for I (§0.1) [A].

### 1.4 Minitron method as implemented by NVIDIA [V `mcore_minitron.py`]
- **MLP neuron importance**: forward hook on the MLP output projection (`linear_fc2`), its *input* (= `act(gate)·up`), `abs().mean(dim=seq)` per
  sample, then `pow(2).sum(dim=batch)` accumulated over the calibration loop; score = `sqrt(·)` (L2 over samples of the per-sample mean |a|).
- **Attention heads**: same aggregation on the *input of `linear_proj`* (o_proj) → per-channel score, view `[nh, hd]`, L2 over hd per head,
  ranked **within each query group** ("group-aware"); KV groups themselves are not pruned (choices are multiples of `num_query_groups`).
- **Mamba heads**: aggregation on `in_proj` output (the `x` part), per-head L2, ranked within groups; `ngroups` kept.
- **Hidden channels**: post-LayerNorm activations, same aggregation, summed over layers.
- **Depth**: `1 − cos(h_in, h_out)` per layer (v1 already uses the block-boundary angular distance).
- Calibration: 1024 samples default (`num_samples=1024` in the README example; "512-1024 samples … ~5 minutes for an 8B model") [V README].
- Distillation (`distill.py`): `--kd_loss_alpha` default 1.0 (pure KD; "1.0 skips the LM loss entirely"), optional `--logit_kl_top_k` ("teacher's
  top-k vocabulary entries plus a residual bucket for the remaining probability mass"), LR 1e-4 → 1e-5, warmup 50 [V]; README: seq 8192 (or 4096),
  LR "1e-4 → 1e-5 (linear decay) for 30-50% pruning … as model gets larger → reduce LR", warmup 100, 80–100 B tokens [V]. VLM: "distills only the
  language model (on text data) and leaves the vision tower and projector untouched" [V]. MTP: "not pruned yet — they are dropped … for speculative
  decoding, run a short MTP SFT on the pruned model" [V]. [A, paper body not re-read: 2407.14679 reports logit-only KD as best unless depth is cut a lot.]

v2 applies the same aggregation, at the input of each output projection (`mlp.down_proj`, `self_attn.o_proj`, `linear_attn.out_proj`): it is the
quantity that the removed columns would have multiplied, so the score measures each structure's contribution to the residual stream, and it maps 1:1
to the masking used in the correctness test.

---

## 2. Dimension table (what can be pruned, and how)

| Dimension | Tensors touched (per layer; suffix after `…layers.N.`) | Constraints | Config keys rewritten (`text_config`) | Risk |
|---|---|---|---|---|
| **MLP `intermediate_size`** (default) | `mlp.gate_proj.weight` rows, `mlp.up_proj.weight` rows, `mlp.down_proj.weight` columns — all 64 layers **and** `mtp.layers.0.mlp.*` | same count in every layer (W1, G3); multiple of 256; ≥ 50 % kept | `intermediate_size` | low (most robust axis, exact surgery) |
| **GDN value heads** `linear_num_value_heads` (Expert) | `linear_attn.in_proj_qkv.weight` rows of the v part (offset `2·nk·dk`, `dv` rows per head), `linear_attn.conv1d.weight` same channels (dim 0), `in_proj_z.weight` rows, `in_proj_b.weight` / `in_proj_a.weight` rows (1 per head), `A_log` / `dt_bias` (1 per head, A_log F32), `out_proj.weight` columns (`dv` per head); `linear_attn.norm.weight` NOT touched | equal count per key head (W6, G5): nv ∈ {48, 32} for 27B, {32, 24, 16} for 9B; nk unchanged | `linear_num_value_heads` | medium (recurrent memory capacity; no NVIDIA precedent on GDN; exact surgery) |
| GDN key heads `linear_num_key_heads` | as above + q and k rows `[h·dk, (h+1)·dk)` in both the q and k parts of `in_proj_qkv`/`conv1d`; removes the `nv/nk` value heads of that key head | keep nv/nk ratio; | `linear_num_key_heads`, `linear_num_value_heads` | high — **not offered** (verified exact, but halves recurrent groups) |
| **Full-attention query heads** `num_attention_heads` (Expert) | `self_attn.q_proj.weight` rows `[h·2hd, (h+1)·2hd)` (query + gate of head h, W2), `self_attn.o_proj.weight` columns `[h·hd, (h+1)·hd)`; `q_norm`/`k_norm` NOT touched; also `mtp.layers.0.self_attn.*` | equal count per KV group (W3); multiple of nkv | `num_attention_heads` | medium (attention is only 16 layers; NVIDIA calls heads sensitive) |
| KV heads `num_key_value_heads` | k/v rows of the group + its q heads | — | `num_key_value_heads`, `num_attention_heads` | high — **not offered** (long-context quality; KV cache is already small: 4 heads) |
| `head_dim`, `linear_key/value_head_dim` | all q/k/v/norms, RoPE (`partial_rotary_factor` 0.25, `mrope_section` [11,11,10] sum = rotary/2) | llama.cpp needs dk = dv (G2); RoPE sections fixed | — | **refused** |
| `hidden_size` | embed, lm_head, every norm, every in/out projection, `mtp.fc` (2H→H), `mtp.*norm*`, **`model.visual.merger.linear_fc2` weight rows + bias** | vision path untrained by text distillation; official mmproj unusable | `hidden_size`, `vision_config.out_hidden_size` | high — **not in v2** (§0) |
| depth (v1) | whole blocks of 4 layers | v1 rules | v1 `fix_config` | as v1 |

Untouched always: `model.visual.*` (byte copy), embeddings, lm_head, all norms, `mtp.fc`, `mtp.*norm*`, `linear_attn.norm.weight`, `q_norm`, `k_norm`.

**MTP slicing.** The MTP layer is never run by transformers (W9), so it cannot be measured with the hooks. It is sliced to the new I / heads with a
weight-magnitude score (deterministic, no torch): neuron i → `‖down[:, i]‖₂ · ‖up[i, :]‖₂`, head h → `‖o_proj[:, cols(h)]‖₂` (group-aware). The
note in the result says: "The multi-token head was cut to fit but was not measured or healed; do not use it for speculative decoding." [A/GPU: llama.cpp
loads it; speculative acceptance not measured].

---

## 3. Design

### 3.1 Overview
```
UI "🧬 Slim or grow a model" (existing page)
  Step A  "How": Depth (v1) | Width (new) | Both          size slider: exact parameter count (width is exact before measuring)
  Step B  Measure (one pass, cached)  -> reshape_measure job: v1 block distances + width scores, models/_scores/<key>/
  Step C  Make the model              -> reshape job (mode prune, + width{...}): pure-numpy slicing, no torch
  Step D  Heal (optional, recommended for width) -> distill_cache job (teacher) + distill job (student)  [or one online job for small models]
  Step E  Test: original vs slim (v1) vs healed; existing Test page / compare()
worker: reshape_measure (torch) · reshape (numpy only) · distill_cache (torch) · distill (torch, trainjob-based)
```
Width surgery needs numpy (in the worker env: numpy comes with torch/transformers; this sandbox has numpy 2.5.3 [V `pip list`]). v1 stays stdlib-only
for depth; width imports numpy lazily and refuses with a plain message if it is missing.

### 3.2 Importance estimation (`reshape_measure`)
- **Reuse** v1 `_choose_load` / `_load` (4-bit NF4 for 27B on 32 GB, 16-bit for 9B; `offload` fallback), `_find_layers`, `_calibration` (project data:
  half world-book text, half chats; built-in sentences refused for width: "Save a training-data version in Step 2 first"), `_measure`'s hook pattern.
- **One forward pass collects both** v1 boundary distances (depth) and width scores, so "Both" costs one measurement.
- Hooks (`register_forward_pre_hook` on the projection module; input `args[0]` `[b, t, C]`; with 4-bit bnb layers the module is `Linear4bit`, the
  input is still the bf16 activation [A/GPU]):
  - `layers[i].mlp.down_proj` → C = I; `layers[i].self_attn.o_proj` → C = nh·hd; `layers[i].linear_attn.out_proj` → C = nv·dv.
  - per batch: `a = x.float().abs()`; masked mean over valid tokens per sample: `m = (a * mask[...,None]).sum(1) / mask.sum(1)[:,None]` → `[b, C]`;
    `acc[i] += (m**2).sum(0)` (float32 on the GPU, `[C]`); also into half-1/half-2 accumulators by batch parity (stability, as v1).
  - final per-channel score `s = sqrt(acc)` (NVIDIA's aggregation, §1.4). Head score = `‖s[h·hd:(h+1)·hd]‖₂` (heads), `‖s[j·dv:(j+1)·dv]‖₂` (GDN v heads).
- Memory: accumulators 64·17408 + 16·6144 + 48·6144 floats ≈ 6 MB; per-batch extra activation copy ≤ `4·512·17408·4 B` = 143 MB at the moment of the
  hook (freed immediately). Same 4-bit footprint as v1 (~17 GB for 27B) [A/GPU].
- Defaults: 256 samples × 512 tokens (≈ 130 k tokens; v1 used 128 for depth; NVIDIA uses 512–1024 at much longer length — 256 is a compromise for
  4-bit speed on one card) [GPU: time, check stability]. Expert: 128/256/512/1024.
- Deterministic: v1 `_calibration` is seeded (3407) and ordered; scores are summed in a fixed order; ties broken by lower index. Results identical
  for the same data version, sample count, load mode [A: bnb kernels can be non-deterministic in the last bits → ranking ties only].
- Ranking/picking (pure, `reshape_width.py`): per layer, `keep = sorted(argsort(-score, stable)[:k])` — uniform k across layers (the only loadable
  shape, §0.5). Heads: per group keep the top `k/groups` within that group (group-aware, as NVIDIA). **Stability**: report the mean Jaccard overlap of
  the kept-neuron sets from half 1 vs half 2 at the chosen size; `stable = overlap ≥ 0.9` [design choice; GPU check].
- **Cache**: `models/_scores/<key>/` with `scores.npz` (float32 arrays: `mlp [L, I]`, `attn [n_full, nh]`, `gdn [n_lin, nv]`, and halves `*_h1`, `*_h2`),
  `scores.json` (`{format:1, source, repo_key, data_version, samples, max_len, seed, load, tokens, period, boundary_distance, matrix, matrix_h1, matrix_h2,
  loss_before, created}`), key = `slug(source)-v{data_version}-{samples}x{max_len}-{load}`. A later `reshape` job with the same key skips the
  measurement entirely. The **slider needs no scores**: width parameter counts are exact from the dims (§1.1). Stability (half-1 vs half-2 overlap)
  is computed by the worker when the model is made and shown on the result card; Rust never reads score arrays.

### 3.3 The surgery (`reshape` job, mode `prune` with `width`)
- New module `worker/ndforge_worker/slice_io.py` (numpy, lazy import): read one tensor from a `TensorRef` with `numpy.fromfile(offset=start,
  count=numel, dtype=RAW[dtype]).reshape(shape)` where `RAW = {"BF16": "<u2", "F16": "<u2", "F32": "<f4", "F64": "<f8", "I64": "<i8", ...}` — **bf16 is
  sliced as raw `uint16`** (slicing never interprets values; bit-exact [V check above]); `np.take(arr, idx, axis)` → `np.ascontiguousarray(...).tobytes()`.
  Peak memory = one tensor in + one out: largest sliced tensor in 27B is `gate_proj` 17408×5120×2 B = 178 MB [V shapes from formula]; fine.
- `safetensors_io.OutTensor` gains an optional `slice: tuple[int, tuple[int, ...]] | None` (axis, kept indices); `write_shards` writes
  `slice_io.sliced_bytes(source, slice)` for such tensors (in `chunk` pieces of the produced bytes), byte copy otherwise (v1 path unchanged).
  `OutTensor.shape` is the new shape; the size check compares against `numel(new shape) × dtype bytes`.
- `reshape_width.plan_width(tensors, layer_types_new, keep)` → the `OutTensor` list: for every layer tensor in the §2 table, the slice; vision/embeds/
  norms untouched; MTP per §2. It is applied **after** v1's `plan_tensors` renaming (depth first: `src` list; width indices are looked up by the
  *old* layer index `src[j]`, because the scores were measured on the unpruned model).
- Config: `reshape_width.fix_config_width(cfg, width)` sets `intermediate_size`, `linear_num_value_heads`, `num_attention_heads` in `text_config`
  only; refuses if a key is missing, a value is not a valid multiple, or `mlp_only_layers`/MoE keys are present (`qwen3_5_moe*` refused for width:
  experts are a different structure). `head_dim` is written explicitly if it was absent (transformers' fallback `hidden_size // num_attention_heads`
  would change when heads change! [V `Qwen3_5Attention`: `getattr(config, "head_dim", hidden // heads)`]) — the real configs have `head_dim: 256`
  [V], the tiny test config must cover the absent case.
- Exact parameter count: `group_params` on the planned output (shapes known) — same function as v1, so `ndforge_reshape.json.params` is exact.
- `ndforge_reshape.json` format 2 adds `"width": {"intermediate_size": [before, after], "linear_num_value_heads": [...], "num_attention_heads": [...],
  "kept": "width_kept.npz", "scores_key": "...", "stable_overlap": 0.93, "mtp": "sliced by weight size, not measured"}` and writes
  `width_kept.npz` (kept indices per layer, int32, ≈ 2.8 MB for 27B) next to it — fully reproducible surgery.
- Disk: output bytes known exactly → v1 disk check unchanged.
- **Loss check** (v1 `_check`): before/after loss on the calibration batches; for width the "after" model is loaded in the same mode.

### 3.4 Loading in other programs
- transformers: uses only config scalars (W1–W6) → loads [V CPU, tiny]; real 21B [GPU].
- Unsloth `FastLanguageModel`: builds from the transformers config [A/GPU].
- llama.cpp GGUF: G1–G6 say yes for uniform dims with dk = dv and MTP sliced [V reading]; quant row-size note §0.1 [A]; run in CI with the tiny model
  (v1 test 9.2-7 pattern, skips without llama.cpp).
- Ollama: [GPU] (its Qwen3.5 engine may hard-code dims; check `ollama create` + Test it).
- Vision: hidden unchanged → v1 `vision_repo` path (official mmproj) still valid [V reasoning: merger output = hidden size].

### 3.5 Healing by distillation
**Goal**: give the pruned student back the original's behaviour (general and on the project's kind of text). World knowledge is still taught in Step 3
(unchanged). Teacher = the original model (`origin.repo` from `ndforge_reshape.json`); student = the pruned folder.

**Loss** (per token t, non-padding, all positions of the row — prompt and answer; teacher distribution over the next token):
`L = α · KL_topk(p_T ‖ p_S) + (1 − α) · CE(answer tokens)`, temperature T = 1, default **α = 1.0** (NVIDIA's default [V distill.py]; the CE term is for
Expert use). `KL_topk`: teacher top-k = 32 ids + log-probs, plus a residual bucket `r_T = 1 − Σ top-k p_T`; student `log p_S` from a full-vocab
log-softmax; `KL = Σ_k p_T,k (log p_T,k − log p_S,k) + r_T (log r_T − log(1 − Σ_k p_S,k))` (same idea as Model-Optimizer's `--logit_kl_top_k`
"plus a residual bucket" [V]). No hidden-state loss in v2 (widths differ only inside layers, hidden size is equal, but logit KD is the reported
best default [A]; an intermediate loss is an open item for "Both" with heavy depth cuts).

**Never materialise full logits**: vocab 248,320 × 2048 tokens × 4 B = 2.0 GB per sequence. `chunked_kd_loss(hidden, lm_head_weight, cache_slice, chunk=256)`
computes `logits = hidden[c] @ W.T` per 256-token chunk under `torch.utils.checkpoint` (recomputed in backward), so peak ≈ 256 × 248k × 4 B = 254 MB.
`hidden` = last hidden state after the final norm, taken with a forward hook on `lm_head` (input) while calling the model with `logits_to_keep=1`
(transformers' standard argument; avoids the full logits) [A: confirm `Qwen3_5ForCausalLM.forward` accepts `logits_to_keep` in 5.22.2 — CPU test].

**Teacher memory — two modes (both implemented, auto-chosen):**
| Mode | How | 27B → 21B on 32 GB | 9B → 7–8B on 32 GB |
|---|---|---|---|
| `online` | teacher (4-bit NF4, eval, no grad) and student (LoRA) in the same process; teacher forward per batch | teacher ~17 GB + student QLoRA ~17 GB + activations → **does not fit** | teacher 4-bit ~6 GB + student 16-bit LoRA ≈ `2.2·8 + 1` ≈ 19 GB (`hub::estimate_vram`) → fits [A/GPU] |
| `cache` (default for big) | job `distill_cache`: teacher alone writes top-k per token to disk; then job `distill`: student alone reads it | teacher 4-bit ~17 GB, then student QLoRA ~17 GB → fits | fits (slower overall than online) |
Rule: `online` if `teacher_4bit_gb + student_train_gb + 4 ≤ vram`, else `cache`. Cache cost: 32 × (4 B id + 2 B fp16 log-prob) + 4 B residual
= 196 B/token → 20 M tokens ≈ 3.9 GB, 40 M ≈ 7.8 GB [V arithmetic]. Format: `models/<name>/distill/cache/part-00001.npz` with `ids uint32 [n_tok, k]`,
`logp float16 [n_tok, k]`, `resid float32 [n_tok]`, `row_offsets int64`, plus `index.json` (`{format:1, teacher, k, rows, tokens, data_hash, tokenizer_hash,
max_len}`); the student job refuses a cache whose `data_hash`/`tokenizer_hash`/`max_len` differ ("The saved teacher answers belong to other data;
they will be made again"). The teacher must see exactly the student's token ids: both use the same tokenizer files (copied unchanged by v1) [V v1 §4.4].
Teacher quality in 4-bit: the teacher's top-k is computed from a 4-bit teacher → a slightly noisier target [A/GPU: compare 9B 16-bit vs 4-bit teacher
KL once].

**Data**: the project's saved data version (`datasets/vN/train.jsonl`, encoded with `data.load_training_rows` like training) **plus general text**
(default 50 % of the tokens) so general ability is healed, not only the world's style. General source: a Hugging Face dataset streamed with
`datasets.load_dataset(..., streaming=True)` and cut to the token budget; default `nvidia/Nemotron-Pretraining-SFT-v1` (NVIDIA's recommendation for
distillation [V README "Dataset quality"]) [A: config/split name and licence to be checked on the owner's PC]; Expert: any dataset id + text column, or
"project data only". Rows are packed? No: v2 keeps the existing row-per-example collator (`trainjob._collator`) for simplicity; `max_seq` 2048.

**Student training**: LoRA on all language linear layers (never vision): hf path `target_modules="all-linear"` + `exclude_modules` (as `backends.load_hf`
[V]); Unsloth path `QWEN_TARGETS + extra_targets ["in_proj_qkv", "in_proj_z", "out_proj"]` (v1 `lora_targets` mechanism [V backends.py]) so the GDN
layers can adapt too. 27B-derived (≥ 15 B): QLoRA (4-bit student); ≤ 15 B: 16-bit LoRA. Full fine-tuning of a 21 B student does not fit 32 GB (weights
42 GB) — not offered.

**Backends**: the KD loss needs logits/hidden states inside `compute_loss`. Implemented as a subclass of TRL's `SFTTrainer` overriding `compute_loss`
(works with any model object). Unsloth: its models may skip returning logits during training (fused CE) [A: Unsloth env `UNSLOTH_RETURN_LOGITS=1`];
our loss uses the hidden state from the `lm_head` hook, not `outputs.logits`, which sidesteps that, but Unsloth's patched forward may bypass the
`lm_head` module call [A]. **Decision: `distill` runs on the plain-libraries path (`backend: hf`) by default; Unsloth is tried only if a CPU/GPU self-test
(`distill_selftest`: one step, loss finite and gradients non-zero on LoRA weights) passes, else falls back with a log line.** [GPU]

**After healing**: merge the LoRA into the pruned folder → a new self-contained base (`models/<name>/healed/` with its own `ndforge_reshape.json`
`healed: {...}`), registered as `custom-local-<name>-healed`. Merge **without torch and without loading 42 GB**: per target tensor
`W' = W + (alpha/r) · B @ A` in numpy float32 (bf16 → f32 by `uint16 << 16` viewed as f32; back with round-to-nearest-even `((u + 0x7FFF + ((u >> 16) & 1)) >> 16)`),
one tensor at a time (peak ≈ 3 × 178 MB); adapter read with `safetensors_io` (PEFT key names `base_model.model.<module>.lora_A.weight`). Only plain
LoRA is merged (no DoRA, no rsLoRA: the distill job sets them off) [V PEFT formula for plain LoRA: scaling = alpha / r; A: key-name mapping checked in
CPU test against PEFT's own `merge_and_unload`]. QLoRA adapters merged into bf16 weights differ slightly from the 4-bit model they were trained on
(known effect) [A/GPU: measure loss healed-merged vs adapter-on-4-bit].

**Checkpoints / resume / stop**: exactly trainjob's: `save_steps`, `checkpoints/`, `resume`, the STOP file via `LiveCallback`; the cache job resumes by
skipping finished parts (`index.json.done_rows`). Stop → state `stopped`, partial cache kept (it is valid up to `done_rows`).

**Metrics** (events, same `metric` format as training so the existing loss chart works): `kd` (KL), `ce` (if α<1), `lr`, `tokens`, and every
`eval_every` steps the **calibration loss** of the student on the held-out half of the measurement batches (the same number the slim result card
shows as "before healing"), plus the teacher's loss on them once (the target).

**Hyperparameter defaults** (all Expert-overridable; reasoning):
| Key | Default | Why |
|---|---|---|
| tokens | 20 M (≥ 15 B student), 40 M (smaller) | ~5 h / ~4 h on a 5090 (estimate below); 0.02–0.05 % of NVIDIA's 80–100 B, so recovery is partial — the honest budget for one card |
| general share | 0.5 | heal general behaviour, not only the world's style |
| seq length | 2048 | project rows are short; NVIDIA's 8192 does not fit QLoRA 21 B + chunked KD on 32 GB [A] |
| batch × accum | 4 × 8 = 32 rows (~ 40–65 k tokens/step) | NVIDIA uses GBS 768 × 8192; we scale down; ~300–500 steps total |
| LR | 1e-4 peak, cosine to 1e-5, warmup 30 steps | NVIDIA's 1e-4 → 1e-5 band for 30–50 % pruning [V]; LoRA usually tolerates 1e-4–2e-4 |
| LoRA r / alpha | 64 / 64, dropout 0 | more capacity than world training (16) because the whole model must readjust; alpha = r keeps scale 1 |
| top-k / T | 32 / 1.0 | residual bucket keeps the distribution normalised [V MO]; T = 1 matches NVIDIA's KD |
| α | 1.0 | NVIDIA default [V]; world learning stays in Step 3 |
| optimizer | adamw_8bit (as trainjob) | memory |
| save_steps / eval_every | 50 / 50 | resume granularity ≈ 15–30 min |

Time/memory estimates [A/GPU, to be replaced by measured numbers]: teacher 27B 4-bit prefill ≈ 2–4 k tokens/s → 20 M tokens ≈ 1.5–3 h; student 21 B QLoRA
with gradient checkpointing ≈ 0.8–1.5 k tokens/s → 4–7 h; peak ≈ 17 GB weights + 6–10 GB activations/KD chunks. 9B→8B online: ≈ 2–3 k tokens/s → 40 M
tokens ≈ 4–6 h. The page shows the estimate from the benchmark's measured tokens/s when available (`train/bench.rs`), else these ranges marked "estimate".

### 3.6 Comparison on the Test page
v1's `reshape::runs::test` (original vs made model, `test.json`) is reused for: original vs slim (before healing), original vs healed (new
`reshape_test` target `"healed"`). `compare()` rows: `model:<name>` (slim) and `model:<name>-healed` with `method` `slim-width` / `slim-depth` /
`slim-both` / `healed`. The result card shows three columns side by side (original · slim · healed): lore / rules / general scores (existing CATS)
and the calibration loss. A depth-pruned and a width-pruned model of the same size made from the same origin appear in the same compare table, which is the
"original vs depth vs width" comparison the owner asked for.

---

## 4. Architecture (files, signatures, formats)

### 4.1 Worker
**New `worker/ndforge_worker/reshape_width.py`** (pure; stdlib only, numpy NOT needed):
```python
WIDTH_KEYS = ("intermediate_size", "linear_num_value_heads", "num_attention_heads")
FFN_DIVISOR = 256
def width_dims(cfg: dict) -> dict        # {"hidden", "intermediate", "heads", "kv_heads", "head_dim", "lin_k_heads", "lin_v_heads", "lin_k_dim", "lin_v_dim",
                                         #  "conv_kernel", "mtp_layers", "vocab", "tied"}; head_dim = tc.get("head_dim") or hidden // heads; ReshapeError if not qwen3_5/qwen3_5_text
def width_supported(cfg: dict) -> str | None       # None, or the plain reason (MoE type, quantized, lin_k_dim != lin_v_dim, missing keys)
def width_choices(dims: dict) -> dict              # {"intermediate": [multiples of 256 from ceil(I/2) to I], "lin_v_heads": [nv, nv - nk] if nv - nk >= nk and (nv - nk) % nk == 0 else [nv],
                                                   #  "heads": [h for h in range(nh, ceil(nh*2/3)-1, -kv) if h % kv == 0]}
def check_width(dims: dict, width: dict) -> dict   # normalised {"intermediate": int, "lin_v_heads": int, "heads": int}; ReshapeError with plain text when not in width_choices
def params_for(dims: dict, layer_types: list[str], width: dict, vision: int, other: int = 0) -> int   # exact formula §1.1 (layers + embed + head + final norm + MTP + vision + other)
def head_rows(h: int, size: int) -> list[int]      # [h*size, (h+1)*size)
def keep_top(scores: list[float], k: int) -> list[int]                       # sorted indices of the k largest (stable: lower index wins ties)
def keep_grouped(scores: list[float], groups: int, k: int) -> list[int]      # k % groups == 0; top k/groups inside each group, sorted
def overlap(a: list[int], b: list[int]) -> float                             # Jaccard
def mtp_scores_from_weights(...)   # not here: needs numpy -> slice_io.weight_norm_scores
def slices_for_layer(layer_type: str, dims: dict, width: dict, keep: dict) -> dict[str, tuple[int, list[int]]]
     # suffix -> (axis, kept indices), exactly the §2 table (verified mapping of verify_width.py): mlp.gate_proj.weight (0, n), mlp.up_proj.weight (0, n), mlp.down_proj.weight (1, n);
     # self_attn.q_proj.weight (0, rows(h, 2hd)), self_attn.o_proj.weight (1, rows(h, hd));
     # linear_attn.in_proj_qkv.weight / conv1d.weight (0, [0..2*nk*dk) + 2*nk*dk + rows(v, dv)), in_proj_z.weight (0, rows(v, dv)), in_proj_b/a.weight (0, v), A_log (0, v), dt_bias (0, v), out_proj.weight (1, rows(v, dv))
     # only for dimensions that change (unchanged dims -> no entry -> byte copy)
def fix_config_width(cfg: dict, dims: dict, width: dict) -> dict   # deep copy; text_config keys only; writes head_dim explicitly
def plan_width(out: list, tensors: dict, src: list[int], dims: dict, width: dict, keep: dict, mtp_keep: dict | None) -> list
     # takes v1 plan_tensors output (renamed), attaches OutTensor.slice + new shape for layer j using keep[src[j]]; mtp.layers.0.* with mtp_keep; ReshapeError for an unknown suffix
     # whose shape contains a changed dimension (safety net: never write a tensor with a stale shape)
```
**New `worker/ndforge_worker/slice_io.py`** (numpy, lazy):
```python
RAW = {"BF16": "<u2", "F16": "<u2", "F32": "<f4", "F64": "<f8", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1", "U8": "u1", "BOOL": "u1"}
def read_array(ref: TensorRef)                       # numpy.fromfile(ref.file, dtype=RAW[ref.dtype], count=ref.numel, offset=ref.start).reshape(ref.shape)
def sliced_bytes(ref: TensorRef, axis: int, idx) -> bytes
def bf16_to_f32(u16) / f32_to_bf16(f32)             # bit tricks (round to nearest even); used by weight norms and the LoRA merge
def weight_norm_scores(tensors: dict, dims: dict) -> dict     # MTP: {"mlp": [I] = ||down[:, i]|| * ||up[i, :]||, "attn": [nh] = ||o_proj cols(h)||}
def merge_lora(base: dict[str, TensorRef], adapter_dir, out_dir, on_progress=None, should_stop=None) -> dict   # §3.5 merge; returns {"merged": n, "skipped": [...]}
```
**`safetensors_io.py`**: `OutTensor.slice: tuple | None = None` (default keeps v1 behaviour); `write_shards` handles it (lazy `slice_io` import). Also
`OutTensor.data: bytes | None` for the merge (small: written from memory).

**`reshapejob.py`**:
```python
def run_measure(cfg) -> int          # job "reshape_measure": {"source", "calibration", "load", "out_dir": "<project>/models/_scores/<key>"} -> scores.npz/json; result {"key", "tokens", "load", "loss", "stable_depth"}
def _measure_all(model, layers, period, batches, mode, width: bool)   # v1 _measure + width hooks (returns v1 tuple + width accumulators incl. halves)
def _scores_for(cfg, src_dir, period) -> dict    # cache hit (same key, same files) or _score/_measure_all and save
def run(cfg)                         # existing; mode "prune" + cfg["width"] (optional): depth choice as v1 (or none: blocks 0 allowed only with width),
                                     # then keep sets from cached scores, plan_width, fix_config_width; result/json extended (§4.3)
```
v1 rule "prune needs blocks ≥ 1" becomes "blocks ≥ 1 or width given". `run_inspect` result gains `width_dims`, `width_supported`, `width_why`,
`width_choices`, `vision_params` (exists).

**New `worker/ndforge_worker/distilljob.py`**:
```python
def run_cache(cfg) -> int     # job "distill_cache": {"teacher": {"repo", "load": "auto|4bit|16bit"}, "data": {...}, "k": 32, "max_seq": 2048, "out": ".../distill/cache"}
def run(cfg) -> int           # job "distill": trainjob-like keys (§4.3) + {"distill": {"mode": "cache|online", "cache": dir, "teacher": {...}, "alpha", "k", "temperature", "chunk"}}
def build_rows(tokenizer, data_cfg, family, max_seq, seed) -> (rows, stats)   # project rows (data.load_training_rows) + general rows (streamed, cut to token share), deterministic order
def data_hash(rows) -> str    # sha256 over the token ids
def topk_from_logits(logits, k) -> (ids, logp, resid)                            # torch; also used in tests
def kd_topk_loss(student_logp_chunk, ids, logp, resid, mask) -> (sum, n)         # torch, §3.5 formula
def chunked_kd_loss(hidden, weight, targets, mask, chunk=256) -> torch.Tensor    # checkpointed chunks
class DistillTrainer(SFTTrainer): compute_loss(...)                              # hook on lm_head input, logits_to_keep=1, chunked KD (+ CE when alpha < 1)
def merge(cfg) -> int         # job "distill_merge": {"base": slim dir, "adapter": dir, "out": ".../healed.partial", "final": ".../healed"} -> slice_io.merge_lora + side files + ndforge_reshape.json
```
`__main__.py`: dispatch `reshape_measure` (`measure_failed`), `distill_cache` / `distill` (`distill_failed`), `distill_merge` (`merge_failed`);
`ReshapeError` → `reshape_refused` as v1. `_explain`: "numpy" missing → "The training software is incomplete (numpy). Run Training setup again."

### 4.2 Rust core
- `reshape/plan.rs`:
  ```rust
  #[serde(rename_all = "lowercase")] pub enum Shape { Depth, Width, Both }               // UI choice; Mode stays Prune|Grow
  #[derive(Default)] pub struct WidthDims { hidden, intermediate, heads, kv_heads, head_dim, lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_kernel, mtp_layers, vocab: u32, tied: bool }
  pub struct WidthChoices { intermediate: Vec<u32>, lin_v_heads: Vec<u32>, heads: Vec<u32> }
  pub struct Width { pub intermediate: u32, #[serde(default)] pub lin_v_heads: Option<u32>, #[serde(default)] pub heads: Option<u32> }
  // Inspect gains: #[serde(default)] width_dims: Option<WidthDims>, width_supported: bool, width_why: Option<String>, width_choices: Option<WidthChoices>
  // ReshapeSpec gains: #[serde(default)] width: Option<Width>, #[serde(default = "s256")] width_samples (calibration samples when width is on)
  pub fn width_params(ins: &Inspect, layers_kept: &[usize], w: &Width) -> u64      // §1.1, mirror of reshape_width.params_for (parity JSON)
  pub fn width_options(ins: &Inspect, depth_blocks: u32, vram_gb: Option<f64>) -> Vec<SizeOption>   // one stop per intermediate choice (lin_v_heads/heads at their current Expert values)
  pub fn validate(...)        // extended: width only for qwen3_5*; values in choices; Depth+Width => blocks >= 1 and width; Width-only => blocks == 0
  pub fn warnings(...)        // + "Removing N % of the MLP: answers get clearly worse until healed (Heal step)." (> 25 %), "Healing is strongly recommended for width slimming."
  pub fn heal_plan(student_b: f64, teacher_b: f64, vram_gb: Option<f64>) -> HealPlan   // {mode: online|cache, student_method: lora|qlora, teacher_load, est_gb, fits: bool, cache_gb(tokens)}
  ```
  `SizeOption` gets `intermediate: Option<u32>` and `exact: bool` (width is exact; depth prune stays "about").
  Depth+width params: layer params after width are computed per layer type, so `params_after` for Both = embed + head + vision + other + MTP(width) +
  Σ over kept layers (depth: assume the mean block, exact once made).
- `reshape/runs.rs`: `build_job` adds `"width"` (+ calibration samples = `width_samples`) and `"scores_dir": models/_scores/<key>`; `name_for` →
  `…-slim-48L` (depth), `…-slim-w10752` (width), `…-slim-60L-w12288` (both); `list()` skips `_scores`; new
  ```rust
  pub fn heal(p: &Project, name: &str, spec: &HealSpec, stop: Arc<AtomicBool>, on_event: &mut dyn FnMut(Value)) -> Result<Value>
      // cache (if mode cache and no valid cache) -> distill -> merge -> register(name + "-healed"); state in models/<name>/heal.json {status, spec, step: cache|train|merge, error}
  pub struct HealSpec { tokens_m: u32, general_share: f32, general_dataset: Option<String>, alpha: f32, rank: u32, lr: f64, mode: Option<String>, backend: Option<String> }
  pub fn mark_interrupted(...)  // also heal.json "running" -> "stopped" (resume possible: checkpoints and cache parts are kept)
  pub fn remove_scores(p: &Project, key: &str) -> Result<()>
  ```
- `evaluate/runs.rs`: `test()` target `"healed"` (Side::Full(models/<name>/healed)); `compare()` rows per §3.6.
- `train/runs.rs`: a run on a healed/width base needs nothing new (LoRA targets: hf "all-linear"); for Unsloth add `extra_targets` GDN names when
  the base's `ndforge_reshape.json` has `width` (so world training can adapt GDN too) — optional, logged.
- `export/runs.rs`: unchanged (`vision_repo` from `origin.repo` works because hidden is unchanged).
- `project.rs` doc: `models/<name>/healed/`, `models/<name>/distill/`, `models/_scores/`.

### 4.3 JSON formats
Job `reshape` (v1 keys unchanged) + `"width": {"intermediate_size": 10752, "linear_num_value_heads": null, "num_attention_heads": null}`,
`"scores_dir": "<project>/models/_scores/<key>"`; `"blocks": 0` allowed with width.
Result (v1 §5.3) + `"width": {"before": {...}, "after": {...}, "stable_overlap": 0.93, "scores_key": "...", "mtp": "sliced by weight size"}`, `"shape": "width|depth|both"`.
Events: `reshape_scores` (v1) + `{"type": "width_scores", "layers": L, "mlp_mean": [per-layer mean score], "kept_share": 0.62, "stable_overlap": 0.93}` (small, for a chart).
Job `distill`: `{"job": "distill", "model": {"repo": slim dir, "family"}, "method": "qlora|lora", "backend": "hf|auto", "dataset": {"train": "...", "general": {"id", "column", "share"}},
"hyper": {...trainjob keys..., "rank": 64, "alpha": 64, "lr": 1e-4, "max_seq": 2048, "tokens": 20000000}, "distill": {"mode": "cache", "cache": ".../distill/cache",
"teacher": {"repo": origin, "load": "4bit"}, "alpha": 1.0, "k": 32, "temperature": 1.0, "chunk": 256}, "eval": {"batches_from": scores_dir}, "out": ".../distill/run", "resume": bool}`.
Result: trainjob summary + `{"kd_first", "kd_last", "calib_loss_before", "calib_loss_after", "teacher_calib_loss", "tokens_seen", "mode"}`.
`models/<name>/heal.json`: `{status, step, spec, created, updated, error, result}`.

### 4.4 Tauri commands (`src-tauri/src/lib.rs`)
`reshape_options(model, mode, shape)` (adds `shape`; returns width choices/options), `reshape_measure(model)` (optional explicit "Measure now";
`reshape_start` measures if needed), `heal_options(name) -> Value` (`heal_plan` + cache size + time estimate), `heal_start(name, spec)`, `heal_stop()`
(events on `"reshape-event"` with `"phase": "heal"`), `reshape_test(name, judge, target: Option<String>)` ("slim" | "healed").

### 4.5 UI (`ui/app.js`, existing page)
1. **Slim / Grow** radio (v1). Under Slim: **How** — "Remove whole layers (faster)" (Depth) · "Make every layer thinner (keeps more quality)"
   (Width) · "Both" — Width/Both only shown when `width_supported`, else the `width_why` line.
2. **Size slider**: Depth → v1 stops; Width → one stop per `intermediate` choice: "MLP 10,752 of 17,408 (−38 %) · **21.14 B parameters (exact)** · 42 GB on
   disk · training: QLoRA ≈ 17 GB ✓". Both → a second slider for blocks (v1) and the width slider; the label shows the combined exact/approx total.
3. Expert: "Also thin the linear-attention layers (48 → 32 value heads)", "Also remove attention heads (24 → 20 / 16)", measuring samples
   (128/256/512/1024), load mode (v1).
4. "Measure on: data version vN, 256 samples" + "Measured earlier — reused" when the cache key exists (with a "Measure again" link).
5. **Make the model** (v1 button/progress/log/Stop). Result card: v1 depth chart (if any) + a small per-layer bar chart of mean MLP score with the kept
   share; "Thinned the MLP 17,408 → 10,752 in all 64 layers; 27.78 → 21.14 B"; "Text-prediction error on your data: 1.92 → 2.9 before healing";
   stability ("the choice of neurons was stable (93 % the same in both halves)" or the v1-style warning).
6. **Heal it (recommended)** card (shown for width/both, optional for depth): plain explanation ("The original model teaches the slim one to answer
   like it did, on your world's text and on general text"), tokens slider (10/20/40/80 M) with time estimate and disk for the teacher answers,
   memory line from `heal_plan` ("teacher and student take turns: fits on 32 GB" / "both at once: fits"), Expert: general dataset id/share, α, rank, LR,
   engine. Progress: phase "teacher answers" (cache) → "learning" (loss chart reuse of `lossChart` with KD + calibration loss) → "saving"; Stop/Continue.
7. **Side-by-side**: Original · Slim · Healed table (lore / rules / general from Test, calibration loss), buttons Test (slim / healed), Train it on your
   world (uses the healed model id when present), Delete (removes healed too).
Smoke test `ui/smoke/reshape.test.mjs` extended (stand-in invoke answers with width options, heal events).

### 4.6 Error messages (plain language)
| Condition | Message |
|---|---|
| width on a non-Qwen3.5 / MoE model | "Thinning the layers is only available for dense Qwen3.5 / Qwen3.8 models so far. You can still remove whole layers." |
| width value not allowed | "The MLP size must be a multiple of 256 between half and all of 17,408." / "Linear-attention value heads can only go from 48 to 32." / "Attention heads must stay a multiple of 4 and at least 16." |
| no data version | v1 text |
| numpy missing | "The training software is incomplete (numpy). Run Training setup again." |
| scores of other data | log only: "The earlier measurement used data version v2; measuring again on v3." |
| OOM measuring | v1 text |
| heal does not fit | "Healing a model of 21 B parameters needs about 23 GB with the teacher's answers saved first; this card has 16 GB. Choose a smaller model or heal on a bigger card." |
| cache disk | "The teacher's answers need 3.9 GB in <folder>; 1.2 GB free." |
| general data download fails | "The general text (<id>) could not be downloaded: <reason>. Heal on your world's text only, or try again later." (Expert switch "project data only") |
| distill loss not finite | "Healing stopped: the training became unstable (the error went to infinity). Lower the learning rate (Expert) and continue." |
| merge key mismatch | "The healing result does not fit the slim model (different layers). Nothing was changed; please make a problem report." |

---

## 5. Test plan

### 5.1 No torch (here and in CI; numpy available here: 2.5.3) — `worker/tests/test_reshape_width.py`
Fixture: extend `fakemodel.py` with `fake_qwen_width(tmp, layers=8)`: real-key-name tensors with the §1.1 shapes at tiny dims (H 8, I 32, nh 6, kv 2,
hd 4, nk 2, nv 4, dk = dv = 4, K 4, V 16, MTP 1, a vision tensor, `A_log`/`norm` in F32, others BF16), written with the independent `stfile.py` writer and
deterministic bytes (v1 pattern) — total < 100 kB (Windows CI: no big files, nothing sparse).
1. `width_dims` / `width_supported` on the real 27B / 9B / 4B configs (add `qwen3_5_27b_config.json` = text_config of the real file, small) and refusals
   (MoE, missing head_dim + changed heads → explicit head_dim written, dk ≠ dv).
2. `params_for` reproduces the Hub totals for 27B and 9B (27,781,427,952 / 9,653,104,368): the test computes text + MTP with `params_for(vision=0)` and
   asserts that `total − that` gives the same vision-tower part for both (merger difference 4096 vs 5120 output columns accounted), and
   `params_for == group_params` of the fake model before and after a plan.
3. `width_choices` (multiples of 256, nv 48→[48,32], heads 24→[24,20,16]); `check_width` errors.
4. `keep_top`, `keep_grouped` (group-aware, ties lowest index), `overlap`.
5. `slices_for_layer`: index lists for each layer type equal hand-written expectations (copy the mapping of `verify_width.py`).
6. `slice_io.sliced_bytes` on BF16/F32 vs a pure-Python reference; `bf16_to_f32`/`f32_to_bf16` round trip on edge values (±0, inf, nan, subnormals,
   round-to-even ties).
7. `plan_width` + `write_shards` on the fake model: every output tensor's bytes == the reference slice of the source bytes; unchanged tensors byte-identical;
   MTP sliced by `weight_norm_scores`; vision untouched; `fix_config_width` keys; safety net raises for an unknown tensor with a changed dim.
8. Job end-to-end (subprocess, v1 pattern): `reshape` with `width` and `drop_layers` + a pre-written fake scores cache (`scores.npz`/`scores.json`
   written by the test) → folder, `config.json`, `ndforge_reshape.json` format 2, `width_kept.npz`, result keys; Width-only with `blocks: 0`; refused without
   cache and without torch ("needs the training software").
9. `merge_lora` with numpy on a fake adapter (A, B random) == `W + (alpha/r)·B@A` computed in float64 then rounded to bf16 (≤ 1 ulp).
10. Parity table `worker/tests/data/width_params.json` (≥ 8 cases: dims, layer_types, width → params) read by both suites.
11. `distilljob.kd_topk_loss` math has a numpy twin `kd_topk_reference` tested here (KL = 0 when student = teacher; ≥ 0; residual bucket).

### 5.2 torch on CPU (CI job `worker`) — `worker/tests/test_reshape_width_torch.py` (`importorskip` torch/transformers)
1. **Sliced == masked** (the strong test; port of `verify_width.py` onto the real job): `tinymodel.build_arch_dir(…, "qwen3_5", num_layers=8)` with
   `num_attention_heads 6, num_key_value_heads 2, linear_num_value_heads 6` (add these as parameters to `build_arch_dir`), perturb weights, run the
   `reshape` job with width and a scores cache whose kept sets are known; load output with `AutoModelForCausalLM`; compare logits with the original
   whose removed columns of `down_proj`/`o_proj`/`out_proj` are zeroed — atol 1e-4 (fp32). Cases: MLP; MLP + heads + GDN; Both (drop block 1 + MLP).
2. Measuring path: `reshape_measure` on the tiny model with `sample_rows` → `scores.npz` shapes `[8, I]`, `[2, nh]`, `[6, nv]`, finite, deterministic
   (two runs equal), cache reused (second job does not load the model: monkeypatch `_load` to raise).
3. `chunked_kd_loss` == full-logits KD (atol 1e-5) and gradients equal; `logits_to_keep=1` + lm_head hook gives the normed last hidden state.
4. Tiny distill (hf backend, LoRA, 4 steps, `mode: online` with the original as teacher in fp32 on CPU): KD decreases; `cache` mode: `distill_cache` +
   `distill` give the same first-step loss as online (same rows, k = vocab so top-k is exact).
5. `distill_merge` result == PEFT `merge_and_unload()` of the same adapter (atol 1e-2 in bf16) — validates key mapping and scaling.
6. GGUF of the width-pruned tiny model with a real llama.cpp (skip unless found, v1 test 9.2-7 pattern).

### 5.3 Rust (`cargo test`) — `crates/ndforge-core/tests/reshape.rs`
`width_params` == the parity table; `width_options` for a 27B Inspect (10752 → 21.14 B, exact flag); `validate` (width on gemma, bad values, Width with
blocks > 0); `heal_plan` (27B→21B on 32 GB → cache + qlora; 9B→8B → online + lora; 16 GB → does not fit); `build_job` contains `width` / `scores_dir`;
fake-worker `heal()` (`#[cfg(unix)]`): cache → distill → merge → registered `-healed`, stop path, interrupted path; `compare()` rows.

### 5.4 GPU-only checklist (RTX 5090; record in `docs/RESHAPE.md`)
1. 9B: measure 16-bit vs 4-bit → Jaccard of kept MLP sets at I 9216 (≥ 0.9 expected) and time/peak VRAM.
2. 27B: measure in 4-bit (256 × 512): time, peak VRAM; make I 13312 and I 10752: time, disk; load in transformers and Unsloth; loss before/after.
3. Test page original vs slim (both 4-bit), then heal (20 M tokens, cache mode): cache time, cache GB, student tokens/s, peak VRAM, KD curve,
   calibration loss vs teacher; Test original vs healed; then Step 3 on the healed base and Test.
4. Same size depth-only (v1, 48 L) vs width-only (I 10752) vs both: Test-page scores and tokens/s side by side (the owner's comparison).
5. Export: GGUF (Unsloth + llama.cpp) of the width model; Q4_K_M quantises all MLP tensors as K-quants (I multiple of 256); `llama-server` with the
   official mmproj answers an image question; Ollama install + Test it.
6. Unsloth distill self-test result (does the `lm_head` hook see the hidden state under Unsloth?).
7. Merge: loss of the merged healed folder (16-bit) vs adapter on 4-bit student.

---

## 6. Ordered steps (each one commit; after each: `cargo test --workspace` and `cd worker && python -m pytest -q tests`)

**Phase F — pure width helpers (no torch)**
1. **Test data.** Add `worker/tests/data/qwen3_5_27b_config.json` (text_config + `vision_config` of `Qwen/Qwen3.5-27B`, plus a `"note"`), and
   `width_params.json` (cases incl. the two Hub totals). Accept: files < 10 kB each.
2. **`reshape_width.py`** with every function of §4.1 except `plan_width`. Tests §5.1-1…5, 10. Accept: pass without numpy.
3. **`slice_io.py`** (`RAW`, `read_array`, `sliced_bytes`, `bf16_to_f32`, `f32_to_bf16`, `weight_norm_scores`). Tests §5.1-6. Accept: pass with numpy;
   module import does not import numpy at top level of `reshape.py`/`reshapejob.py`.
4. **Writer support.** `safetensors_io.OutTensor.slice` (+ `data`), `write_shards` handles both; v1 tests unchanged. Accept: old tests pass; new
   round-trip test for a sliced tensor.
5. **`plan_width` + `fix_config_width`** + `fakemodel.fake_qwen_width`. Tests §5.1-7. Accept: byte-exact slices; MTP sliced; vision untouched.
6. **Inspect.** `run_inspect` adds `width_dims`, `width_supported`, `width_why`, `width_choices`. Test: subprocess inspect on the fake width model.
7. **Job without torch.** `reshapejob.run` accepts `width` + `scores_dir` with a cache present (`blocks: 0` allowed with width; v1 checks kept),
   writes `ndforge_reshape.json` format 2 and `width_kept.npz`; refuses without cache and without torch. Tests §5.1-8. Accept: `list_tensors` reads the
   output; `group_params(output) == params_for(...)`.

**Phase G — torch parts (CPU in CI)**
8. **Tiny model knobs.** `tinymodel.build_arch_dir(..., heads=4, kv_heads=2, lin_v_heads=4)` keyword parameters (defaults unchanged).
9. **Measuring.** `_measure_all` (v1 `_measure` + width hooks; v1 `_measure` becomes a call with `width=False` so v1 results are identical),
   `run_measure`, `_scores_for` cache. Dispatch `reshape_measure`. Tests §5.2-1, 2. Accept: CI green; v1 torch tests unchanged.
10. **Distill math.** `distilljob.py`: `topk_from_logits`, `kd_topk_loss`, `chunked_kd_loss`, numpy twin. Tests §5.1-11, §5.2-3.
11. **Distill cache job.** `run_cache`, `build_rows`, `data_hash`, general-data streaming (stubbed in tests by a local JSONL via `"general": {"file": ...}`).
    Test: tiny teacher writes a cache; index fields; resume skips done rows.
12. **Distill training job.** `DistillTrainer`, `run` (online + cache modes, hf backend; Unsloth guarded by `distill_selftest`), events, checkpoints,
    STOP. Test §5.2-4.
13. **Merge job.** `slice_io.merge_lora`, `distilljob.merge` (+ side files, `ndforge_reshape.json` `healed`). Tests §5.1-9, §5.2-5.
14. **GGUF test** §5.2-6 (skips without llama.cpp).

**Phase H — Rust**
15. **plan.rs**: `Shape`, `WidthDims`, `WidthChoices`, `Width`, Inspect/Spec fields (serde defaults: old JSON loads), `width_params`, `width_options`,
    `validate`/`warnings` extensions, `heal_plan`. Tests §5.3 (parity file read). Accept: v1 tests unchanged.
16. **runs.rs**: `build_job` width keys, `name_for` variants, `_scores` folder handling (`list` skips it), `heal`, `HealSpec`, `heal.json`,
    `mark_interrupted`; `evaluate/runs.rs` healed target + compare rows; `project.rs` doc. Fake-worker tests (unix).
17. **Tauri commands** §4.4 + `generate_handler!`. Accept: `cargo build -p nd-lore-forge`.

**Phase I — UI, help, docs**
18. **UI** §4.5 (+ smoke test). Accept: smoke test passes; manual fake-worker run shows width options, a made model, a heal run, the 3-column table.
19. **Help + docs**: `crates/ndforge-core/help/reshape.md` (sections "Thinner layers (width)", "Healing", honest expectations, memory/disk/time),
    `help/troubleshooting.md` (heal does not fit, unstable training, general data download), `docs/RESHAPE.md` (v2 section: method with citations — Minitron
    2407.14679 / 2408.11796, Minitron-SSM 2504.11409, NVIDIA Model-Optimizer as the reference implementation; the §2 table; formats; deviations;
    "not verified" list; GPU checklist table §5.4), `README.md` status row, `PLAN.md` milestone M14 "Thinner layers + healing", `docs/TRAINING_SPEED.md`
    paragraph (width vs depth speed). Keep `docs/RESHAPE_PLAN.md` unchanged; add this plan as `docs/RESHAPE_V2_PLAN.md`.

**Phase J — GPU (owner's PC)**
20. Run §5.4; record numbers; fix what breaks (likely: Unsloth path of distill, Ollama dims, bnb hook dtype).
21. *(optional)* hidden-size pruning for a text-only export (drops `model.visual.*`, slices merger-free; requires `--no-mtp` or MTP fc slicing) — design separately.
22. *(optional)* intermediate-state (hidden) KD term for "Both" with ≥ 3 blocks removed, if §5.4-4 shows depth hurts more after healing.

CI notes: no test file > 1 MB, no sparse/placeholder big files (Windows `set_len` is not sparse — v1 commit d17594e); numpy-only tests must not
import torch; torch tests use `importorskip`; general-data tests never touch the network (local JSONL stand-in); subprocess tests use `sys.executable`.

---

## 7. Open questions for the owner (with my recommended default)

1. **Default target for width on the 27B**: the mild 23.7 B (I 13312, −24 % MLP) as the slider's first position, or the 21.1 B (I 10752, −38 %)?
   *Recommended: start at 23.7 B; 21 B reachable on the slider with a "needs healing" warning; or 21.5 B by "Both" (−1 block + I 12288).*
2. **General text for healing**: download a general dataset (default `nvidia/Nemotron-Pretraining-SFT-v1`, ~50 % of the heal tokens), or heal on the
   project's own data only (no download, but general ability recovers less)? *Recommended: general data on by default, switch in Expert mode.*
3. **Heal budget**: 20 M tokens (~5–8 h on the 5090 for 27B→21B, estimate) as the default? *Recommended: 20 M for ≥ 15 B students, 40 M below;
   the page shows the estimate and allows 10/20/40/80 M.*
4. **Merge the heal into a new base folder** (one more ~42 GB on disk, clean for Step 3/export) or keep it as an adapter? *Recommended: merge
   (numpy, no extra RAM), delete the distill checkpoints afterwards but keep the teacher cache until the model is deleted.*
5. **Expert axes**: offer GDN value heads (48 → 32) and attention heads (24 → 16/20) at all in v2, given no published precedent for GatedDeltaNet
   head pruning? *Recommended: yes, Expert mode only, labelled "experimental", after the GPU checklist passes for MLP-only.*
