# Pruning Qwen3.8 27B with ND Lore Forge: recommendations and a step-by-step guide

This is the practical guide for the **Slim** half of *Slim or grow a model*: **version 1** removes whole blocks of layers, **version 2** makes every layer **thinner** (width pruning) and **heals** the
result with the original as teacher. How the page works is in `docs/RESHAPE.md`; what has and has not been tried on a real GPU is listed there too. **Nothing in this guide has been run on a real
Qwen3.8 27B yet**: the figures marked "estimate" are calculated, not measured, and the quality you get after slimming is something your own test has to show. The parameter counts of the thinner
variants are exact (they come from the model's own dimensions), the time and memory figures are estimates.

## 1. Should you prune it at all?

Read this first, because it may save you a day.

- **The 27B already fits ordinary cards without pruning.** Unsloth's page for Qwen3.8 says the model runs locally on about 17 GB (4-bit GGUF) and that **QLoRA training works with 24 GB**. Lore Forge
  estimates QLoRA at about 20 GB, 16-bit LoRA at about 60 GB.
- **Pruning is worth it when you need the smaller size**: a card that cannot take the 27B at all (under 24 GB), faster answers, or room for a long context next to the model. It is **not** a way to get a
  *better* model: a slimmed model starts out worse than the original and healing only wins part of that back.
- **If your card has less than 24 GB**, a 27B pruned to 21 B will still not train comfortably. Use a smaller model that already exists: Qwen3.5 9B (QLoRA about 12 GB) or 4B. They are trained as complete models,
  which beats a cut-down bigger one at the same size.
- **Two ways to slim, with different strengths.** NVIDIA's Minitron work found that **width pruning** (thinner layers) keeps more quality than removing layers at the same size once the model is healed,
  while **depth pruning** (fewer layers) makes the model run faster at the same size. Version 2 offers both, and *Both* together. Whichever you choose, keep the cut small.
- **Thinner layers need healing.** Right after thinning, the model is clearly weaker; the original model then teaches it to answer like it did (20 million words by default, hours on one card). NVIDIA healed
  its models with about 94 *billion* words, so expect a usable model, not one as good as the original.

## 2. What you get: the sizes

Qwen3.8 27B has **64 layers = 16 blocks of 4** (3 linear-attention layers and 1 full-attention layer each), hidden size 5120, a 262 K context and a vision part (checked in its `config.json` on Hugging Face).
The two big embedding tables (input and output, not shared) and the vision part stay as they are; each block you remove takes out about **1.54 B parameters**.

| Blocks removed | Layers left | Parameters (estimate) | Disk, 16-bit | QLoRA training (estimate) | 16-bit LoRA (estimate) |
|---|---|---|---|---|---|
| 0 (original) | 64 | 27.8 B | 56 GB | about 20 GB | about 60 GB |
| 1 | 60 | 26.2 B | 52 GB | about 19 GB | about 59 GB |
| 2 | 56 | 24.7 B | 49 GB | about 18 GB | about 55 GB |
| 3 | 52 | 23.2 B | 46 GB | about 17 GB | about 52 GB |
| **4** | **48** | **21.6 B** | **43 GB** | **about 16 GB** | **about 48 GB** |
| 5 | 44 | 20.1 B | 40 GB | about 15 GB | about 45 GB |
| 6 | 40 | 18.5 B | 37 GB | about 14 GB | about 42 GB |
| 8 (the most the page allows: half) | 32 | 15.5 B | 31 GB | about 12 GB | about 35 GB |

The page shows the exact figures for your model and your card before you press the button; use those.

### 2b. Thinner layers (version 2): the sizes

Every layer keeps its shape except the MLP, which loses neurons (multiples of 256, at most half). The sizes are exact:

| MLP width | MLP removed | Parameters | Disk, 16-bit | QLoRA training (estimate) |
|---|---|---|---|---|
| 17,408 (original) | 0 % | 27.78 B | 56 GB | about 20 GB |
| 15,360 | 12 % | 25.74 B | 51 GB | about 20 GB |
| 14,336 | 18 % | 24.71 B | 49 GB | about 19 GB |
| **13,312 (the page starts here)** | **24 %** | **23.69 B** | **47 GB** | **about 19 GB** |
| 12,288 | 29 % | 22.67 B | 45 GB | about 18 GB |
| 11,264 | 35 % | 21.65 B | 43 GB | about 17 GB |
| 10,752 | 38 % | 21.14 B | 42 GB | about 17 GB |
| 8,704 (the most the page allows: half of the MLP) | 50 % | 19.09 B | 38 GB | about 15 GB |

**Both** (remove blocks and thin the rest): 60 layers with MLP 12,288 is 21.46 B; 56 layers with MLP 13,312 is 21.15 B.

### 2c. Where the parameters are (text only, with the vision tower, with the multi-token head)

A width cut that thins the MLP removes the multi-token head (it no longer fits), so the table above, which still counts it, is higher than the file the program writes. The page and the Auto-cut report now show the parts of every size (`ParamsBreakdown`: text layers, embeddings, output layer, vision tower, multi-token head, rest, file total; the parts add up to the size shown). For the 27 B at its own head count (computed from the same formula, exact):

| MLP width | text model only | with the vision tower (the file) | with the multi-token head kept and sliced |
|---|---|---|---|
| 17,408 (original) | 27.32 B | 27.78 B | 27.78 B |
| 13,312 | 22.87 B | 23.33 B | 23.69 B |
| 12,288 | 21.86 B | 22.32 B | 22.67 B |
| 10,752 | 20.35 B | 20.81 B | 21.14 B |
| 9,728 | 19.35 B | 19.81 B | 20.11 B |
| 8,704 | 18.34 B | 18.80 B | 19.09 B |

So a figure such as "20.23 B at 9,728" in an outside note is another way of counting (it keeps a head); the program's number is the second column, the size of the files it writes. A model made by the program reports the same parts in its result (`params_breakdown`) and the page shows them under the size.

## 3. Recommendations

0. **Prefer thinner layers (MLP 13,312, 23.7 B) for the first try, and heal it.** It is the page's first position. Go to 10,752 (21.1 B), or to *Both* (one block less and MLP 12,288, 21.5 B), only after the
   mild cut has been healed and tested. Choose *Remove whole layers* when the speed of the finished model matters more than its quality.
1. **Take out as little as you can live with.** (Layers) Start with **2 blocks (about 25 B)**, test, and only then try 4. The page warns above 25 % removed; treat that as the edge of the sensible range, not as a target.
2. **Never the first or the last block** (the program already refuses). The blocks in the middle to late middle usually do the least work; let the measurement choose, do not guess.
3. **One run of neighbouring blocks** (the default) heals more easily than scattered blocks. Use Expert mode's separate blocks only to compare.
4. **Measure on your own text, with enough of it**: 128 samples is the default; use 256 if the page says the two halves of your text chose different layers.
5. **Measure in 4-bit** on a 32 GB card (the page does this by itself for a model that does not fit in 16-bit). Expect about 17 GB while measuring. Thinner layers are measured on 256 samples by default (one
   measurement serves every size on the same data); on 24 GB lower the samples.
6. **Heal with QLoRA** on a 32 GB card (16-bit LoRA of a 21.6 B model needs about 48 GB). 4-bit training is fine for a dense Qwen model; it loses a little more quality than 16-bit. For thinner layers the
   **Heal card** does this: the original answers first and saves its answers to disk (about 4 GB for 20 million tokens), then the thinner model learns; the result is a new base model, *healed*.
7. **Heal with more than your world.** One world teaches the world, not the general ability the cut took away. Keep the *general chat* share of the training data higher than usual (Step 2) when training a slimmed model, and
   if the model should keep thinking before it answers, keep **at least 75 % reasoning-style examples** (Unsloth's advice for Qwen3.8; in nd-world's export this is the *Reasoning share* option).
   Reasoning rows of the general text are now written the way the model thinks (`<think>` + trace + `</think>` + answer, once; see GENERAL_TEXT.md "Thinking"), and a model healed
   with the thinking recipe is **exported with the model's own chat template** (it thinks unless the client switches thinking off); any other model still answers without thinking by default.
8. **Do not stack changes.** Slim first, test, heal, test again. Do not grow the same model, change the training data and change the settings at once: you will not know what helped.
9. **Keep the original.** The program never changes it; do not delete it until the slim model has passed your own test.

## 4. Before you start: what you need

- **Disk space**: the original in the Hugging Face cache (about 56 GB) **plus** the new model (about 43 GB for 48 layers) and room for the healing run's checkpoints. About **120 GB free** is a safe figure. The page checks the
  space first and says how much is missing.
- **Memory**: a 32 GB card for 4-bit measuring and QLoRA healing; 24 GB is tight but possible with 64 samples and context 2048 (estimates).
- **A saved data version** (Step 2): the measuring uses it. The more of your world it covers, the better the choice of blocks.
- **The latest Transformers (v5)**: Unsloth says Qwen3.8 needs it, and the first run is slower while the Gated DeltaNet kernels compile.

## 5. Recommended workflow (thinner layers, version 2)

1. **Baseline.** Step 4: test the **original** on your world (4-bit) and write down lore / rules / general.
2. **🧬 Slim or grow a model**: choose *Qwen3.8 27B*, **Slim it**, **Make every layer thinner**. The slider starts on **MLP 13,312 (23.7 B)**. Check the disk line (about 47 GB) and the training line.
3. **Make the model.** It measures your own text once (the page says when an earlier measurement is reused), cuts, and shows the per-layer chart and the text-prediction error *before healing*. Read that number:
   it is the honest starting point.
4. **Heal it.** On the result card: leave **Keep thinking** ticked if the model should still think before it answers (it uses the reasoning traces of Qwen3.8-27B as the general text, 40 or 80 million tokens, longer texts, more memory; docs/RESHAPE.md), general text on (it is downloaded from Hugging Face; offline it falls back to your text only), **20 million tokens** (40 or 80 with Keep thinking), press *Start healing*. The original answers first
   (a few hours on a 5090, about 4 GB of answers saved to disk), then the thinner model learns (several hours). You can **Stop** and **Continue** at any time. Watch the two curves: the KL and the error on your data
   should fall towards the original's.
5. **Test the healed model** (button on the card) and run the **Thinking check** (original, cut, healed; docs/THINKING_CHECK.md). The table shows original · thinner · healed. A good outcome is a healed model whose general numbers are close to the original's.
6. **Train it on your world (Step 3)** with the healed model (QLoRA, the *Normal* preset, a higher general share, 75 % reasoning-style examples if it should keep thinking), then **test again** and export (Step 5).
7. **Not good enough?** Go back to step 2 with a milder cut (MLP 14,336), or heal for longer (40 or 80 million tokens). **Too slow when it runs?** Try layers instead, or *Both*.
8. **Delete the extras when you are done**: deleting the thinned model also deletes its healed version and the teacher's saved answers.

The rest of this guide describes removing whole layers (version 1), which stays available.

## 6. Step by step (removing whole layers, version 1)

1. **Baseline first.** In Step 4 test the **original** Qwen3.8 27B on your world (loaded in 4-bit). Write down the three numbers (lore, rules, general). Without this you cannot judge anything later.
2. **Open 🧬 Slim or grow a model** and pick *Qwen3.8 27B* as the model. The page reads its size and tells you whether it can be slimmed.
3. **Choose Slim** and set the size with the slider: start with **56 layers (24.7 B)**. Look at the disk space and the training line (*QLoRA about 18 GB ✓ on this card*).
4. **Press Make.** It measures (a few minutes to a quarter of an hour depending on the card), shows the bar chart (removed blocks in red) and writes the new model into the project's `models/` folder. You can stop it; nothing is half-written.
5. **Test it against the original** (the button on the result card). Both are loaded in 4-bit. Read the three numbers next to your baseline.
   - A **small drop** means a short healing run will do.
   - A **big drop on the general numbers** means slim less (fewer blocks) or heal with more general data.
   - **Broken answers** (repeating text, wrong language) mean the chosen blocks were important: measure again with 256 samples, or choose other layers in Expert mode.
6. **Heal it (Step 3).** The slim model is now in the model list. Choose it, **QLoRA**, the *Normal* preset, context 2048 to start, your saved data version with a higher general share (see recommendation 7). Unsloth's own recipe for this
   family is rank 16 / alpha 16, learning rate 2e-4, vision layers off for text-only data; Lore Forge's Step 3 fills these in. Watch the loss go down steadily.
7. **Test again** against the original (Step 4, with *compare with the original* chosen). A slim model is good enough when its numbers are close to the original's **on your world and on the general questions**.
   The program cannot tell you "close enough": it is your call. A sensible habit is to stop at the smallest model whose general score you still accept, and to go back one block count if it is below that.
8. **If it is not good enough**: go back to step 3 with one block fewer removed. Do not heal for longer on the same data hoping it will recover: if the general numbers do not move after a full run, more of the same data will not help.
9. **Export (Step 5)** like any model: *Merged model* or *GGUF* (Q4_K_M is the usual size; several at once are possible). The picture (vision) file is taken from the original, because the vision part is unchanged; it sits next to the GGUF (`mmproj-F16.gguf`), the Modelfile lists it, and **Install into Ollama** sends it along (see the "lost vision" row of the table below).
   **Then press Test it**: a model with 48 layers instead of 64 is something llama.cpp and Ollama may or may not load. This is not verified yet.

## 7. Time and cost (rough, estimates)

- Measuring: minutes on a 5090. Making the new folder: copy time for 43 GB.
- Healing: QLoRA on a 21 B model is slower than on the 9 B (the same data takes about two to three times longer). Run the Step 3 benchmark once to see your card's real speed before a long run.
- Healing a thinner 27B (20 million tokens): the original answers in roughly 1.5 to 3 hours, the thinner model learns in roughly 4 to 7 hours (estimates; the card shows its own estimate before you start).

## 8. What can go wrong

| Message / symptom | What to do |
|---|---|
| Out of memory while measuring | Expert mode: load for measuring = 4-bit, samples 64. |
| Not enough disk space | The message names the folder and the missing space; free it or change the project's folder. |
| "The two halves of your text chose different layers" | 256 samples, or choose layers yourself. |
| Slim model repeats itself / answers in another language | Too many blocks removed or the wrong ones. Remove fewer and re-measure. |
| Healing loss does not go down | Check the data version (is it the one you measured on?), lower the learning rate to 1e-4, and make sure *QLoRA* and the right base were chosen. |
| llama.cpp / Ollama refuses the GGUF | Export the merged 16-bit model instead and report the message; this is one of the things not yet verified. |
| The GGUF answers (thinking works) but it ignores pictures, or Ollama says it cannot read images | The vision file was not part of what Ollama got. The export folder has it (`mmproj-F16.gguf`) and the Modelfile lists it as a second `FROM`: make the model in Ollama again from that folder (`ollama create <name> -f Modelfile`), update Ollama if it is old, and install again. If pictures still do not work, your Ollama build cannot load this family's projector: use llama.cpp for pictures — `llama-server -m model-q4_k_m.gguf --mmproj mmproj-F16.gguf` (or `llama-mtmd-cli --image photo.png …`, see `VISION.md` in the export folder). Remember also that healing teaches text only: the vision part itself is the original's and unchanged, but image understanding can be weaker than the original's — try one image question with `llama-mtmd-cli` before judging the model. |
| Healing says it does not fit this card | Take the milder cut or a smaller model; healing needs about 19 GB in the one-after-the-other mode (a 21 B student in QLoRA). |
| The general text could not be downloaded | Healing carries on with your world's text only; try *Continue healing* with a connection later for better general ability. |
| Healing stopped: the training became unstable | Expert mode, lower the learning rate (0.00005), *Continue healing*. |
| The page says the model cannot be slimmed | Compressed (quantized) checkpoints and Gemma 4 are not supported; use the full-precision Qwen3.8 27B (`unsloth/Qwen3.8-27B` or `Qwen/Qwen3.8-27B`). |

## 9. Not verified yet (needs a GPU run)

Everything about a real 27B: that 4-bit measuring picks the same blocks (and the same MLP neurons) as 16-bit, the time and memory figures above, the quality after healing (layers and, above all, thinner
layers: 20 million words is a small part of what NVIDIA used), whether Unsloth loads a 48-layer or a thinned folder, and whether llama.cpp and Ollama run its GGUF. Two more about the vision file that
travels with the export: whether your Ollama build loads the projector (the Modelfile's second `FROM`; other families have had bugs there — if it refuses, llama.cpp with `--mmproj` is the fallback), and
how much image understanding survives a cut and a text-only heal (one `llama-mtmd-cli --image` question answers it). The full checklist is in `docs/RESHAPE.md`. After your first real run, please write the
actual numbers into `docs/RESHAPE.md` (section "Not verified yet") so the next person starts from facts.

## 10. What comes next

Version 2 (thinner layers and healing) is built. Still open: the first real run (checklist in `docs/RESHAPE.md`), the experimental thinning of the attention layers (offered in Expert mode once the checklist
passes), hidden-size pruning for a text-only export, and healing with Unsloth for speed. These are listed as follow-ups in `docs/RESHAPE.md`.

## Research notes: an outside document's claims against this code

An outside research note on pruning and healing this model was read against the code. Its citations were **not** verified by us; only what the code does was. What follows says what is true, what was already done, what was added and what stays unmeasured.

| Claim of the note | What the code does | What was done about it |
|---|---|---|
| Neuron scoring uses activations only, with no down-projection factor | True. The score is, per text, the mean of \|activation\| over its tokens, squared, summed over the texts, square-rooted (`reshapejob._measure_all`). That is not the mean square per token, so an output score cannot be derived from it | The measurement now also records the mean square (`mlp_ms`); `width_scoring: "output"` ranks by mean square x squared `down_proj` column norm, an option next to the default (docs/TRY_FIRST.md) |
| The shared teacher cache id hashes the teacher's path and config, not its weights | True for a local folder; mostly harmless for a Hub id (a snapshot is a commit). The cache meta, `rows_key`, the Try key and the measurement had no weights identity either | `safetensors_io.weights_manifest` (commit or a sampled hash of every weight file, no file times) is part of the store id (version 2), the index, the Try key and the measurement; old stores are adopted once with a note; the program's `rows_key` counts a local teacher's files (docs/HEAL_SETTINGS.md) |
| The loss checkpoints every output chunk, and there is no chunk knob | The first is true (`chunked_kd_loss`). The second is false: the chunk size is a setting (`distill.chunk`, "KL chunk" 64 to 512, halved by the out-of-memory ladder) | `distill.head_checkpoint` "off" keeps the logits instead (same loss, tokens x vocabulary x 10 bytes more memory); "Find the fastest loss settings" measures the options on the owner's card |
| Keeping only the top-k does not make the student's output layer cheap | True: the student still makes the full log-softmax over about 248,000 words (`_chunk_losses`); the cache costs 196 bytes per token | Documented; a fused cross-entropy is not a drop-in (docs/TRAINING_SPEED.md), not built |
| The planner's totals do not match the document's | Explained, not wrong: at an MLP width of 9,728 the text model is 19.35 B and the file with the vision tower 19.81 B (the multi-token head is removed by a width cut); the document's 20.23 B keeps a head | The parts of every size are shown (text layers, embeddings, output, vision, multi-token head removed, file total) |
| Long chat rows keep the first tokens, rows over twice the length are skipped, and 1,024 loses reasoning answers | True for the mechanism (`general_rows`); outdated for the recipe, which heals at 8,192 (4,096 when 8,192 does not fit) on 80 % reasoning traces | A coverage report says how many rows are whole, cut or left out, how many reasoning rows lost their answer, and the lengths (docs/HEAL_SETTINGS.md) |
| Calibrate on reasoning traces | Already done (`reasoning_key`, 50 % thinking traces of 2,048 tokens in the measurement) | none needed |
| It has never been run on a real GPU | Partly outdated: `docs/REAL_RUN_17B.md` is a real run (Qwen3.5-family; nothing on Qwen3.8 yet). The kernel diagnostics exist and their speed effect is unmeasured | unchanged |
| A 32-question check should not be the main gate | Partly outdated: answers are graded, the closed-trace rate and truncation are counted | A broader suite (structured output, multi-turn correction, tasks from the project's own data, code with tests) is not built |
| (found while reading) the "calibration loss" of a heal | It is measured on `train.jsonl`, the file the student learns from: **not held out**, so it reads better than it is | A held-out loss on general rows the heal never learned from (docs/HEAL_SETTINGS.md) |

### Added for the note's ideas, all experimental and off by default

* **Refitting the kept neurons** (a least-squares correction of the kept neurons' output weights): Try first compares it with the plain cut; the cut can write it. The maths is standard algebra and is tested against `numpy.linalg.lstsq`; whether it keeps more quality is **not known** for any real model.
* **Ranking the neurons by their expected output**: an option next to the default; Try first compares the two and calls a gain only when both halves of the text agree.
* **Keeping the output layer's logits** and **the speed test** of the loss settings: the loss is the same; only speed and memory change, and the numbers are the owner's to produce.

### What is measured here and what is not

Measured (by tests, on tiny models and by arithmetic): that the replay of the layers equals the full forward, that the written bytes equal the refit, that the Try pass loss equals the written model's loss for the same cut, that every breakdown adds up, that the head switch does not change the loss, and that the held-out rows are not training rows. **Not measured, because nothing here runs on the owner's GPU:** whether the refit or the output ranking beats the default at any real size; how long a refit takes on a 27 B (the time shown is a formula); whether a gain before healing survives healing and quantisation to a GGUF; how much faster the loss is with the logits kept; how well the coverage numbers predict how the healed model thinks; and every outside citation.

### Not doing (from the note)

* **Reinforcement learning for recovery**: a generation loop with verifiers and rewards, many times the compute, a new training engine.
* **Different widths for different layers**: the model has one `intermediate_size`; Transformers and llama.cpp build every layer from it.
* **Packing for the hybrid recurrent layers**: the linear-attention state and the convolution cache would leak across packed examples unless the sequence boundaries are proven; length bucketing stays.
* **A long bucket of rows over 8,192 tokens within one heal**: two sequence lengths mean two caches; the coverage report only says when it would matter.
* **Changing the KL objective** (top-k-only logits for the student).
