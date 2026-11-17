# Mixed-precision sizes ("Mixed precision (our own recipe)")

Four extra file sizes in Step 5 next to Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16: **Mixed 4-bit (balanced)**, **Mixed 3-bit (small)**, **Mixed 2-bit (smallest: short answers only)** and **Mixed: fit to a size (experimental)**. Instead of one type for the whole model, `llama-quantize` is given a base type plus an ordered list of `--tensor-type REGEX=TYPE` rules, an `--output-tensor-type` and a `--token-embedding-type`. Built and tested with stand-in tools and hand-built GGUF headers; **never run with a real llama.cpp or model** (see "Not verified yet"). Nobody has measured whether the recipes beat the plain sizes: that is what the Compare quants page ([QUANT_COMPARE.md](QUANT_COMPARE.md)) is for, and the comparison table puts them next to plain sizes (with and without a matrix).

Code: `worker/ndforge_worker/mixedquant.py` (ONE table of recipes, the rules, the size estimate, the fit, the plan job `mixed_plan`), `worker/ndforge_worker/exportjob.py` (`_make_mixed`, the llama.cpp-only rule), `crates/ndforge-core/src/export/runs.rs` (`MIXED_QUANTS`, `MixedOptions`) and `export/mixed.rs`, the page in `ui/app.js` (`mixedBox`), help `use.md`.

## About Unsloth Dynamic quants: why they are not reproduced here, and what this offers instead

The owner asked whether Unsloth's "Dynamic" GGUFs (2.0 / 3.0: names such as UD-Q4_K_XL, UD-Q2_K_XL, UD-IQ1_S) could be added. They cannot, and nothing in this program is named after them or implies it:

- **They come from Unsloth's own, unpublished pipeline.** Its public description (`unsloth.ai/docs/basics/dynamic-3.0-ggufs`) is general: a better, more diverse calibration set for the importance matrix, improved layer selection, "many more quantization techniques", post-training quantization only, the multi-token-prediction module dropped from the small quants, and a published matrix "available for the community". No per-layer recipe is published.
- **The released code does not contain them.** The `unsloth` and `unsloth_zoo` 2026.11.4 wheels (read as plain files, never run) know only the standard llama.cpp types (`ALLOWED_QUANTS`, `IMATRIX_QUANTS` in `unsloth/save.py`), no `UD-` type. The only preset with per-tensor flags in them is `q2_k_l` (a Q2_K base with the output and token-embedding tensors kept at a higher type: `unsloth/save.py` says q8_0, the zoo's `quantize_gguf` passes `--tensor-type "\.ffn_down_exps=Q3_K" --tensor-type "\.ffn_down=Q3_K" --output-tensor-type Q6_K --token-embedding-type Q4_K`). So "Unsloth's export cannot do per-tensor types" means: it offers fixed names, not arbitrary rules.
- **What this program offers instead** (all labelled "Mixed precision (our own recipe)", never with their names): (1) [Compare quants](QUANT_COMPARE.md): KL divergence of any GGUF from the 16-bit model, which is the figure Unsloth reports for its own quants, so a recipe can be judged on facts; (2) [Merging matrices](IMATRIX.md): Unsloth's **published importance matrix** for a base model can be downloaded (ungated) and merged with yours; (3) these mixed-precision recipes, built on the same `llama-quantize` options. Whether they come close to Unsloth's quants is unknown; the comparison tool is how to find out on your own model.

## The recipes (ONE table: `mixedquant.PRESETS`)

`types[role] = (type in the first and last N layers, type in the layers between)`; N = `edge`. Types are ggml names (`q4_k`, not `q4_k_m`).

| | Mixed 4-bit (balanced) `mixed_4` | Mixed 3-bit (small) `mixed_3` | Mixed 2-bit `mixed_2` |
|---|---|---|---|
| base type given to llama-quantize (for tensors no rule names) | Q4_K_M | Q3_K_M | Q2_K |
| `--token-embedding-type` / `--output-tensor-type` | Q5_K / Q6_K | Q4_K / Q6_K | Q4_K / Q6_K |
| edge layers (first and last N) | N = 3 | N = 2 | N = 2 |
| attention k, v, output (`attn_kvo`) | Q6_K / Q6_K | Q6_K / Q5_K | Q6_K / Q4_K |
| attention q (`attn_q`, includes its gate) | Q6_K / Q5_K | Q5_K / Q4_K | Q5_K / Q3_K |
| linear-attention projections (`lin`) | Q6_K / Q5_K | Q5_K / Q4_K | Q5_K / Q4_K |
| ffn_down | Q6_K / Q5_K | Q5_K / Q4_K | Q5_K / Q3_K |
| ffn_up and ffn_gate (`ffn_updown`) | Q5_K / Q4_K | Q4_K / Q3_K | Q3_K / Q2_K |

(edge / middle). The shape follows common practice for hybrid models: the embedding and output tensors, the attention k / v / output, the linear-attention (state-space style) projections, ffn_down and the first and last layers get more bits; ffn_up / ffn_gate of the middle layers, the biggest tensors, get the fewest. **The numbers themselves are our choice, not a measurement.** Only K types and no IQ types are used, so no recipe needs an importance matrix to run (a matrix still helps them: it is given to these sizes like to the plain ones).

## The real tensor names (read, not remembered)

llama.cpp `src/models/qwen35.cpp` and `src/llama-arch.cpp` (master, 2026-11), architecture `qwen35` (and `qwen35moe`):

- global: `token_embd.weight`, `output_norm.weight`, `output.weight` (absent when the embedding is tied);
- every layer: `blk.N.attn_norm.weight`, `blk.N.post_attention_norm.weight`; feed-forward `blk.N.ffn_gate.weight`, `ffn_down.weight`, `ffn_up.weight` (for the MoE variant `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` plus the router `ffn_gate_inp`, which is never quantised);
- **full-attention layers**: `attn_q.weight` (it holds the query and its gate), `attn_k.weight`, `attn_v.weight` (or one fused `attn_qkv.weight`), `attn_output.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`;
- **linear-attention (gated delta net) layers**: `attn_qkv_gate.weight` **or** `attn_qkv.weight` + `attn_gate.weight`, `ssm_conv1d.weight` (never quantised), `ssm_dt.bias`, `ssm_a`, `ssm_norm.weight`, `ssm_out.weight`, `ssm_beta_alpha.weight` **or** `ssm_beta.weight` + `ssm_alpha.weight`;
- the speculative-decoding head (layer index = the number of layers): `blk.N.nextn.eh_proj.weight`, `nextn.enorm`, `nextn.hnorm`, optional `nextn.embed_tokens`, `nextn.shared_head_*`, and the ordinary attention and feed-forward tensors of that layer.

Roles (`mixedquant.role_names`): `attn_kvo` = `attn_k`, `attn_v`, `attn_output`, `nextn.eh_proj` (plus a fused `attn_qkv` for an unknown architecture); `attn_q`; `lin` = `attn_qkv_gate`, `attn_qkv`, `attn_gate`, `ssm_out`, `ssm_beta_alpha`, `ssm_beta`, `ssm_alpha` (hybrid architectures only: `qwen35`, `qwen35moe`); `ffn_down` (also `_exps`, `_shexp`); `ffn_updown` = ffn_up and ffn_gate (also `_exps`, `_shexp`).

## How the rules are built (`build_rules`, pure)

First the edge layers (`blk\.(?:0|1|2|61|62|63)\.(?:...)\.weight$=TYPE`, one rule per role whose edge type differs), then one rule per role for every layer (`\.(?:...)\.weight$=TYPE`); the layer of the speculative-decoding head is treated like an edge layer (it is one small layer). The number of layers comes from `<arch>.block_count` in the header, else from the tensor names (without the head). For an architecture this program does not know only the standard `blk.N.` names and the generic roles are used, with a note ("The architecture 'x' is not one this program knows the tensor names of; only the standard blk.N. names and the generic rules ... Read the list of rules before you rely on it."). A rule that matches no tensor of the model is reported in the notes. Tested against name lists of the `qwen35` layout (separate and fused projections, tied and untied output, with and without the head, MoE) and an unknown architecture.

What `llama-quantize` does with them (read in `tools/quantize/quantize.cpp` and `src/llama-quant.cpp`): the part before the first `=` is a regular expression, **searched** in the tensor name (`std::regex_search`, ECMAScript), the **first** matching rule wins, and the pattern text is **lower-cased** before use (so the rules avoid upper-case escapes such as `\D`); the type is a ggml type name, case-insensitive; a tensor no rule names follows the base type's own mixture; 1-D tensors, norms, `ssm_conv1d`, the MoE router and a few other small tensors are never quantised; when a row length does not fit the type's block size llama-quantize falls back to a type with smaller blocks (Q4_K to Q5_0 and so on, F16 as the last resort); `--output-tensor-type` / `--token-embedding-type` set those two tensors (a tied embedding gets the output type).

## The size estimate, before exporting

`mixedquant.estimate` adds `number of weights x bits per weight / 8` for every tensor of a GGUF's header (shapes and types only, no weights are read) after applying the rules the way llama-quantize does (first match, the fallback table, tensors that are not quantised keep their type), plus the size of the header and token list of that file. Any GGUF of the model will do (a Q4_K_M file has the same names and shapes as the 16-bit one); the page offers the GGUF files of the project's exports or a typed path; **without any GGUF no estimate is shown** (the page says so) because the shapes cannot be read from anything else here. The result is "estimated GB and bits per weight" per recipe, the exact rules, and a reminder of the plain sizes' bits per weight from llama.cpp's own table (Llama-3.1-8B: Q2_K 3.16, Q3_K_M 4.00, Q4_K_M 4.89, Q5_K_M 5.70, Q6_K 6.56, Q8_0 8.50; other models differ). Tensors named by no rule follow the base type's own mixture, which adds some higher types, so that small part of the estimate is approximate (counted and shown). The real size is shown after the export (`actual_bytes`).

Bits per weight of each type (`ggufinfo.BPW`) = bytes per block x 8 / values per block, from the `static_assert(sizeof(block_*))` lines of `ggml/src/ggml-common.h` (QK_K = 256, K_SCALE_SIZE = 12): Q8_0 8.5 (34 bytes / 32), Q6_K 6.5625 (210 / 256), Q5_K 5.5 (176), Q4_K 4.5 (144), Q3_K 3.4375 (110), Q2_K 2.625 (84), Q4_0 4.5, Q5_0 5.5, Q4_1 5.0, Q5_1 6.0, IQ4_NL 4.5, IQ4_XS 4.25, IQ3_S 3.4375, IQ3_XXS 3.0625, IQ2_S 2.5625, IQ2_XS 2.3125, IQ2_XXS 2.0625, IQ1_M 1.75, IQ1_S 1.5625; F32 32, F16 and BF16 16 (a test recomputes each from the block sizes). The type numbers are those of `ggml/include/ggml.h`.

## Fit to a size (experimental)

Given a target size in GB and a saved matrix: every large tensor starts at the lowest allowed type (Q2_K), then the tensor with the most **importance per extra byte** is upgraded one step (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) until the next upgrade would not fit the budget (a pure prefix: a bigger budget never gives any tensor fewer bits, and never a smaller file). The embedding and output tensors are fixed (Q4_K and Q6_K) because the matrix does not measure them; tensors that are not quantised stay as they are. The output is the same `--tensor-type` rule list (one rule per role and type, naming the layers: `blk\.(?:3|7|11)\.attn_k\.weight$=Q6_K`), shown on the page before exporting ("Work out the rules").

**The importance.** The matrix file stores, for every weight tensor, `<name>.in_sum2` (the sum of the squared inputs the tensor saw, shape inputs x experts) and `<name>.counts` (the rows summed per expert) as float32 (read in `imatrix.cpp` and `common/imatrix-loader.cpp`; the README of `tools/imatrix` documents the format only in words). The mean of `in_sum2 / counts` over inputs and experts, E, is the tensor's mean squared input. Rounding to b bits adds noise that falls by a factor of 4 per bit (4^-b), and it reaches the output weighted by E and by the number of weights; upgrading from b0 to b1 bits therefore removes `numel x E x (4^-b0 - 4^-b1)` of error for `numel x (b1 - b0) / 8` bytes. The number of weights cancels; the order is by `E x (4^-b0 - 4^-b1) / (b1 - b0)`, ties by tensor name, so the result is deterministic. **This ignores how large the weights of each tensor are (the matrix does not know) and the interaction between layers: it is a guess, not a measurement.** A tensor the matrix has no figure for gets the median one (counted in a note).

**Honesty and fall-backs.** Shipped as an option clearly marked experimental. If the matrix cannot be read as a GGUF-format matrix with those tensors (a legacy `.dat`, none given), the budget is below what the lowest types need, or no target is given, the fit says so in plain words and, in an export, the largest recipe that fits the target (else the smallest) is used instead, with the reason in the notes and `fell_back_to` in the export's record. A target above what the highest type can use gives the highest everywhere and says so. Whether a fitted recipe is any better than a preset of the same size is **unknown**: compare them.

## In the export

- Sizes `mixed_4`, `mixed_3`, `mixed_2`, `mixed_fit` (`exportjob.QUANTS`, `runs::MIXED_QUANTS`; a worker test reads the Rust table and checks that ids and labels agree). Files `model-mixed_4.gguf` and so on. Always labelled "Mixed precision (our own recipe)" in the page, the export record (`meta.json` `mixed`: id, label, recipe, base, embd, output, rules, estimate, notes, fit, matrix_used, note), the model card ("Mixed 4-bit (balanced) (Mixed precision (our own recipe))") and the result. No name, file or text says Unsloth Dynamic or UD-.
- **The llama.cpp path.** Made with `llama-quantize` after the 16-bit conversion, with the importance matrix when there is one (`--imatrix M` first, then the recipe's options, then the files and the base type), and once more without it when llama-quantize refuses it (as for the plain sizes). With the **Unsloth back end** selected, Unsloth cannot give single tensors their own types, so llama.cpp makes the **whole batch** (a second pass through Unsloth for the plain sizes would merge the model again) and the result says so (`mixed_notes`, in `meta.json` `mixed.note`, in the page's message); `via` is then `llama.cpp`. Without llama.cpp the export stops at once with a plain message (the mixed sizes need it); choose only plain sizes then.
- "Fit to a size" in an export: the job carries `mixed_fit: {target_gb, matrix: <library file> | null}`; without a library matrix the export's own matrix (measured or reused) is used. Page: the target, a matrix chosen from the library, "Work out the rules".
- **Not added (follow-up):** dropping the MTP head from the small sizes. The head is kept and, when the model has one, its layer gets the edge layers' types.

## Tests

- Worker `worker/tests/test_mixedquant.py`: the table (three recipes, labels, never fewer bits at the edges, nothing that names or implies another project's quants), the rules in exact order against the `qwen35` name lists (separate and fused projections, MoE, tied output, the head), every tensor's type under each recipe, norms / conv kernels / router / state tensors untouched, whole layer numbers (`blk.1` is not `blk.11`), valid regular expressions without `=` or upper-case escapes, the flags, an unknown architecture, layers from the names, dead rules, the estimate against a hand sum and the bits-per-weight table against the block structs, the fit (deterministic, budget respected, the estimate of its rules equals its own count, monotonic over 41 budgets, importance order, impossible and generous budgets, missing figures, no statistics), reading a matrix with real float data and every unreadable case, `for_export` and its fall-backs, the plan job and its worker entry, the Rust/Python table check, and the export wiring (exact `llama-quantize` command lines with and without the matrix, a refusal of the matrix, a library matrix for the fit, the Unsloth back end making the batch with llama.cpp and saying so, no llama.cpp, an unreadable 16-bit file, nothing named after another project).
- Rust `crates/ndforge-core/tests/mixed.rs`: the sizes and labels, the options and their old forms, the page state, a mixed export's record and card, the fit block and every refusal, the plan, fake-worker runs (unix only).
- Window `ui/smoke/mixed.test.mjs`: the sizes under the recipe's name, the explanation, the estimated sizes (table, rules, notes, errors), the typed file and the missing file, Fit to a size (target, matrix, rules, failures, unreachable target), what Make sends, and the list of files.

## Deviations

- **The whole batch goes through llama.cpp** when a mixed size is chosen with the Unsloth back end (the brief said "for those sizes"): doing the plain sizes through Unsloth as well would merge the model a second time.
- **The size estimate needs a GGUF of the model** (any size); with none, only a message is shown. The brief asked for the estimate from the header of the 16-bit GGUF; no reliable alternative exists before the first export.
- The layer of the **speculative-decoding head is treated like an edge layer** (not asked for; it is one small layer and the head is only useful at good precision).
- The four sizes are kept in their own table (`MIXED_QUANTS`) next to `QUANTS` so that existing code and tests that count the plain sizes are unchanged.

## Not verified yet (needs llama.cpp and a real model)

- **That the rules match the real names** of a real converted GGUF (the names were read from llama.cpp's source for `qwen35`, not from a file), and the real layer count and edge layers of the 27 B model; the notes report dead rules, but only after the export.
- **That `llama-quantize` accepts** these options in this order on the user's build (`--output-tensor-type`, `--token-embedding-type`, many `--tensor-type`), with and without `--imatrix`, that a long regular expression with an alternation of layer numbers is accepted, and the exact output (the warning "applying manual override" per tensor).
- **That the size estimate matches the file** (the fallback table, the header size, the tied embedding, the base mixture for tensors no rule names).
- **Whether the recipes beat the plain sizes**, and by how much, on the user's own model: build them, then Compare quants (with and without a matrix).
- **Fit to a size**: that the matrix of a real model carries `in_sum2` and `counts` as the source says (float32, the layouts above), that the importance formula orders tensors usefully, and the time of `read_importance` on a 27 B matrix (the data is read in Python).
- That the 2-bit recipe works at all without an importance matrix on the user's build (it uses only K types), and what its answers look like.
- Windows: very long command lines of a fitted rule list (a fit with 60 layers makes up to a few dozen rules of a few hundred characters).

## Follow-ups

- Dropping the speculative-decoding (MTP) head from the small sizes (not in this version, by decision).
- IQ types for the 1-3 bit recipes (they need a matrix and a real test first).
- Tuning the recipes after the first real comparison, and a "learn from the comparison" step.

## Where llama.cpp comes from
The mixed sizes need llama.cpp (`convert_hf_to_gguf.py` and `llama-quantize`). The program can install it for the person (**Install llama.cpp now** under the message, or Training setup): docs/LLAMA_CPP_INSTALL.md. A missing llama.cpp ends the export with the error code `llama_cpp_missing`; a copy named in `LLAMA_CPP_DIR` always wins.

## Disk space

The export checks the disk before it starts (see README, "Stopping" section): a mixed size is counted at 5.5 bits per weight (the real size comes from the plan, which is not used for this check). A GGUF made by `llama-quantize` is written as `model-<size>.gguf.partial` and renamed when complete.
