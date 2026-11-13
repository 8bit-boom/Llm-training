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
4. **Heal it.** On the result card: general text on (it is downloaded from Hugging Face; offline it falls back to your text only), **20 million tokens**, press *Start healing*. The original answers first
   (a few hours on a 5090, about 4 GB of answers saved to disk), then the thinner model learns (several hours). You can **Stop** and **Continue** at any time. Watch the two curves: the KL and the error on your data
   should fall towards the original's.
5. **Test the healed model** (button on the card). The table shows original · thinner · healed. A good outcome is a healed model whose general numbers are close to the original's.
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
9. **Export (Step 5)** like any model: *Merged model* or *GGUF* (Q4_K_M is the usual size; several at once are possible). The picture (vision) file is taken from the original, because the vision part is unchanged.
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
| Healing says it does not fit this card | Take the milder cut or a smaller model; healing needs about 19 GB in the one-after-the-other mode (a 21 B student in QLoRA). |
| The general text could not be downloaded | Healing carries on with your world's text only; try *Continue healing* with a connection later for better general ability. |
| Healing stopped: the training became unstable | Expert mode, lower the learning rate (0.00005), *Continue healing*. |
| The page says the model cannot be slimmed | Compressed (quantized) checkpoints and Gemma 4 are not supported; use the full-precision Qwen3.8 27B (`unsloth/Qwen3.8-27B` or `Qwen/Qwen3.8-27B`). |

## 9. Not verified yet (needs a GPU run)

Everything about a real 27B: that 4-bit measuring picks the same blocks (and the same MLP neurons) as 16-bit, the time and memory figures above, the quality after healing (layers and, above all, thinner
layers: 20 million words is a small part of what NVIDIA used), whether Unsloth loads a 48-layer or a thinned folder, and whether llama.cpp and Ollama run its GGUF. The full checklist is in `docs/RESHAPE.md`. After your first real run, please write the actual numbers into `docs/RESHAPE.md` (section "Not verified yet") so the next person starts from facts.

## 10. What comes next

Version 2 (thinner layers and healing) is built. Still open: the first real run (checklist in `docs/RESHAPE.md`), the experimental thinning of the attention layers (offered in Expert mode once the checklist
passes), hidden-size pruning for a text-only export, and healing with Unsloth for speed. These are listed as follow-ups in `docs/RESHAPE.md`.
