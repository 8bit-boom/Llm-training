# Cutting, healing and growing a mixture-of-experts model (Inkling-2-Mini and others)

Written 2026-11-27. It says what the program does, what it does **not** do, and, at the end, what nobody has seen working yet. Nothing here was run on a graphics card or
on the real 20 GB checkpoint: the code is tested on small stand-in folders that have the real tensor names, the real packed layout and tiny widths, and on the
real `config.json` and tensor list of the model (`worker/tests/data/inkling2_*`). The plan behind it is in the session notes; the numbers below are marked
**[checked]** (read in a file or a page), **[computed]** (from those files) or **[not verified]**.

## 1. What a mixture of experts is, in one paragraph

A normal ("dense") layer has one big feed-forward block that every token passes through. A mixture-of-experts layer has many small ones, the **experts**, and a small
**router** that picks, for every token, the few experts that work on it (**top-k**). So there are two sizes: the **total** parameters (all experts, what has to fit in
memory) and the **active** parameters (the experts of one token plus the rest, what costs time). Inkling-2-Mini has 128 experts in each of 24 layers and uses 8 per token:
26.4 B parameters in all, about 2.7 B active in one pass through the layers [computed], and, because its middle layers run three times, about 5.4 B per token.

## 2. Inkling-2-Mini's special parts [checked: config.json, tensor list]

- **The loop.** 24 layers in four blocks of 6 (the authors call them attention-residual blocks): the first block is the prelude, blocks 2 and 3 (layers 6 to 17) are the
  **core** and run **three times** (a token passes through 48 layer slots), the last block is the coda. Passes 2 and 3 read messages from the previous pass through small
  rank-64 gates (`controllers.*`). In every block of 6, five layers use a sliding window and one (layers 5, 11, 17, 23) looks at everything.
- **Packed storage.** The checkpoint is 19.5 GB because the experts are stored in 4-bit NVFP4 (layers 0 to 19) and 8-bit MXFP8 (layers 20 to 23) with block scales, in the
  authors' own tensor layout. The program **never decodes them to cut**: removing an expert is slicing the first axis of its tensors, byte for byte.
- **A vision tower** (23 layers) and a **speculative-decoding drafter** (`mtp.safetensors`, a separate file that the index does not list).
- It needs software newer than the program's usual pins: transformers 5.24 or newer, llama.cpp build b12611 or newer, vLLM 0.35 or the nightly image. The Slim page checks
  this and says what to do (section 7). It also asks Hugging Face whether the model's repository changed since the program was tested (revision `c9d0283`, 2026-11-24).

## 3. The cuts

| Cut | Saves | Risk | How it is made | In the program |
|---|---|---|---|---|
| **Fewer experts** (128 to 112, 96, 80, 64) | file size and memory, nothing in speed | low to medium at 25 %, medium to high at 50 % | slice the first axis of the expert tensors and of the router; same number in every layer | needs the experts measured first |
| **Fewer experts per token** (8 to 6 or 4) | a little speed (the experts are about half of the work) | unknown: the authors did not test it | `num_experts_per_tok` in config.json | Slim page, no measuring |
| **Fewer loop passes** (3 to 2) | a quarter of the compute and a third of the loop's memory | high and unknown: the model was never trained with 2 passes. Only the variable-loop research checkpoint was | `loop_config.num_passes`; the drafter is removed | Slim page; Try first measures it; Auto-cut **never chooses it** |
| **Whole blocks** of 6 layers | a quarter of the weights per block | very high | renumber layers, the loop and the attention-residual queries | Expert mode only |
| More passes, more experts per token | nothing: costs compute | unknown | config only | Slim page, as a question to try |
| Thinner experts, merging experts, attention cuts | – | – | – | **not offered** (section 9) |

**Rules for the expert count.** Multiples of 16, never below 32 (and never below 4 times the experts used per token), and at most half of the experts removed in one stage.
To go further, cut again from the result (with its own healing). A cut keeps the **same number of experts in every layer** because the model files and llama.cpp know only one
number; *which* experts stay differs per layer.

**How the experts are chosen.** The program runs a sample of your text through the model and counts, for every expert in every layer and over **all three loop passes**
(they share the weights), how often the router picked it, the routing weight it got, and the size of its output times that weight (REAP: router-weighted expert
activation, Lasby et al. 2025). The default ranking is REAP; frequency and gate weight are alternatives in Expert mode. When the routers pick by a sigmoid score and
normalise over the chosen experts, as Inkling's do, the pruned model routes exactly like the original with the removed experts switched off.
After pruning, the router's selection bias can be nudged toward the load the kept experts had (no training, a few hundred bytes per layer).

## 4. Healing

**Built, never run on a real card.** The heal job loads the student packed (`backends.load_inkling`: experts frozen, a LoRA on the attention projections only, `moeheal.lora_targets`), computes the KL loss on the logits the model itself
produces (final softcap 30, `distilljob.head_softcap`; the output layer already has the 200,058 real rows), and uses the packed model in this program's own process as the teacher (`reshapejob._load_model`). vLLM for the original's answers is refused for
Inkling with a note (the program's vLLM route quantizes a 16-bit original while loading, which this packed model is not): the in-process teacher answers instead, slower. Auto-cut heals only a cut that fails its thinking check, **in rounds** of 5 million tokens
with a quick check after each. The memory and time figures are estimates. The design below is what the code follows. The teacher is the original packed model, the student the cut one. By default the student keeps its packed experts **frozen** and trains a LoRA on the dense parts
(attention projections, the loop gates), with the router frozen; the experts stay byte-identical, so the result stays in the authors' packed format. Distillation uses the model's final
logit softcap (30) and its real vocabulary (200,058, not the padded 201,024). Budget: 20 M tokens for up to 25 % of the experts, 40 M for more, 80 M for fewer passes.
Healing a 32 GB card needs the in-process packed teacher (about 20 GB) or vLLM nightly. Times are estimates: 20 M tokens is 4 to 7 hours **[not verified]**.

## 5. Growing

**Built as questions and config-only copies.** Two free grows exist as questions to try: loop passes 3 to 4 (+33 % compute per token) and experts per token 8 to 10. Nobody knows whether they help (the authors' discussion #51
about changing top-k is unanswered). Adding experts (copying the most used ones) is a start for training, not an improvement by itself. Splitting experts and adding blocks are not offered.

## 6. Export

llama.cpp converts the packed folder directly (it reads NVFP4 and MXFP8 itself). It needs **build b12611 or newer**: older builds use wrong per-expert scales on passes 2 and 3 and
give a perplexity of about 130 instead of 7.65 **[checked: the GGUF repository's notes]**. The 16-bit GGUF is about 52 GB for the original and 40 GB at 96 experts, so the
disk check uses that size. A cut that was not healed can be saved with "Save a model as GGUF" without any training. Sizes of the original: Q4 about 16.9 GB, NVFP4 17.6 GB [checked]; at 96 experts
about 13 GB for Q4 [computed]. The drafter is not exported (llama.cpp support for it is still a proposal). Pictures work through the projector file (`--mmproj`), which a cut never changes.
Whether Ollama runs this architecture is not known; use llama.cpp or Unsloth Studio if it refuses.

## 7. Software and repository check

The Slim page shows, for Inkling, a list with each tool, what it needs and what is installed: transformers >= 5.24.0 (the program's environment may have an older one: the check
says so and gives the command and the command to go back; Qwen and Gemma keep working with either), llama.cpp >= b12611 (Settings, llama.cpp), vLLM >= 0.35 or the nightly image
(Settings, vLLM; the program's pin stays 0.34 for every other model). A button asks Hugging Face for the model's current revision and says whether it differs from the one the code was written for.

**Changes upstream, 2026-11-26** [checked]: the model's repository got one config change since the upload (a top-level `dtype` field, harmless), the evaluation tables, and a text fix. transformers 5.24.0
(2026-11-21) is the first release with the model. llama.cpp's mainline support came in two steps (the converter, then the per-expert scale fix in b12611). vLLM supports it only on main (the 0.34.0 release does not).
Hugging Face discussions mention repetitive reasoning loops (#45, #48), which the thinking check's loop rate also measures.

## 8. The order to work in on a 32 GB card

0. **Decide the goal.** A smaller file (smaller cards, more context), faster answers, or a model that knows your world. The original already runs on a 5090; a cut only helps the first two.
   To teach it your world, train the uncut model (the Train page, QLoRA from 22 GB) and do not cut at all.
1. **Measure** the experts once (minutes).
2. **Try first** (about 20 passes): the baseline, fewer passes, fewer experts per token, 112/96/80/64 experts.
3. For a smaller file: **128 to 96 experts**, make it, run the thinking check; if it passes, export without healing and compare the quants; if not, heal 20 M tokens and check again. Go to 64 only as a second stage with its own heal.
   For a target of about 18 to 19 B parameters, 80 experts is the nearest standard count (about 17.2 B); 96 experts is a gentler cut at about 20.3 B.
4. For speed: fewer experts per token first; fewer passes only if its error is small **and** the thinking check passes (expect to heal 40 to 80 M tokens), or start from the variable-loop research checkpoint.
5. Export with llama.cpp b12611 or newer, an importance matrix adapted from the original's, Q4_K_M.

Warnings: vLLM 0.34 cannot load the model; llama.cpp older than b12611 gives garbage; 16-bit LoRA needs about 59 GB; a 16-bit copy needs 53 GB of disk; the original itself may loop in long reasoning;
a cut that removes the drafter's targets (passes, blocks) loses speculative decoding.

## 9. What is not offered, and why

- **Merging experts** (MC-SMoE style): REAP's authors found it adds an error that pruning does not, because the router loses the fine control it had **[checked: abstract]**.
- **Thinner experts and attention cuts**: the experts' second matrix is packed in blocks along the dimension that would be cut, so it would have to be decoded and stored in 16-bit; the attention part is 7 % of the parameters.
- **Different expert counts per layer**: no file format and no runtime used here supports it.
- **Inkling 2 Small** (310 GB): out of scope.

## 10. Other mixtures of experts

The same taxonomy (`worker/ndforge_worker/moe_layout.py`) describes the per-expert layouts of Qwen3-MoE, Qwen3.5-MoE and Mixtral (one weight per expert), so the Slim page shows
their expert counts. Removing experts from them is not built yet; they keep the whole-block depth cut they always had. DeepSeek-style group routing and gpt-oss are refused.

## Not verified yet

- Everything on a graphics card and the real checkpoint: the memory and speed of the measuring pass, the real scores, whether REAP beats frequency on this model, the error of each cut, the thinking check of a cut model.
- That llama.cpp b12611 and vLLM nightly load a **cut** model (they read the expert count from the config; nobody has tried).
- That the exact tensor shapes I derived from the safetensors header (read from Hugging Face on 2026-11-26) hold for every future revision.
- The size of the rebalance's effect, the best budget of tokens, the heal times, vLLM nightly on an RTX 5090 (the vLLM page does not list that card; SGLang's does).
- Whether Ollama runs the exported file.


## Auto-cut in one click

The Auto page handles this model end to end (docs/AUTO_CUT.md, "A mixture of experts"): choose **a smaller file** (the default), **faster answers** or **a better model** (which points to the Train page), give the size of the finished Q4_K_M file in GB, and the plan
turns it into a number of experts per layer. It measures, tries, cuts, checks the thinking, heals only when the check fails, exports and compares. Loop passes are never taken out by Auto-cut. Everything in that flow except healing and the real jobs was tested with stand-ins.


## Grow, export and quant: what exists

- **Grow**: more loop passes (up to 8) and more experts per token (up to 4 times) are config-only copies and Try first questions (`goal` grow); nothing trains after them, and no one knows whether either helps.
- **Mixed sizes**: the fused `ffn_gate_up_exps` tensor is in the "up/gate" role for Inkling only (every other family's rules are unchanged).
- **Importance matrix along the expert axis**: the Adapt step (`imatrixadapt`) now cuts a matrix of a cut Inkling to its kept experts (`moe_kept.json`: rows and counts of every `*_exps` statistic, layers unchanged), verifies the written file against the source, and refuses a matrix of another model (wrong number of experts) or a damaged record. Tested on synthetic matrices that follow my understanding of llama.cpp's layout for experts (`n_expert x n_in` values and one count per expert): **not checked against a real file**.
- **NVFP4 size**: shown as an estimate (about the packed model's size) in the recommendations table next to the Q4 size. No NVFP4 file is made: the program does not know that the installed llama.cpp can write one, and it is offered only after Compare quants has measured it.
- **Compare quants**: nothing in it is specific to the architecture, so a cut Inkling can be compared like any model; its 16-bit reference needs `f16_gb` of disk (shown in the table). Not run on an Inkling.
- **Ollama**: the export writes a Modelfile that uses the template stored in the GGUF with the authors' sampling, and the install step is the generic one. Whether the installed Ollama can load this architecture is unknown.
- **vLLM as the original's teacher**: refused for Inkling with a note, and the packed in-process teacher is used. vLLM can serve the packed checkpoint on a nightly image, but making the program's teacher route serve it (instead of quantizing a 16-bit original while loading) cannot be built safely without a real checkpoint and engine to test against.
