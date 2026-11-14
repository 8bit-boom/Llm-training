# Slim or grow a model

## What this page does
It makes a **new base model** from one you already have, by changing its size:
- **Slim it** takes whole blocks of layers out (*Remove whole layers*), or keeps every layer and makes each one **thinner** (*Make every layer thinner*, Qwen3.5 and 3.8 only), or both. The model gets smaller and needs less memory (a 27 B model that does not fit your graphics card can become one that does). Removing layers also makes it faster.
- **Grow it** copies a stretch of layers in. The model gets bigger and slower.

The new model is saved inside your project (folder `models/`) and appears in the model list of Step 3 like any other model. Nothing is changed in the original.

## Be realistic
- A **slimmed** model answers **worse than the original** until you train it again on your world (Step 3, the same way you train any model). Training it again is called *healing*. Always test it first.
- A **grown** model starts out answering **exactly like the original** (the new layers are switched off) and costs about 40 % more memory, disk space and time. The papers that made big models better this way trained on billions of words; one world is a few million at most. Expect **little or no gain** from your world's text alone.
- Nothing here is free. The page shows the size, the disk space and what training the result would need on your card before you press the button.

## Slim: how the layers are chosen
The program runs a sample of **your own world's text** (the saved data version of Step 2: half world book, half practice answers) through the model and measures, block by block, how much the text changes inside it. A block that changes the text very little does the least work, so it is the best one to take out. By default it removes **one run of neighbouring blocks** where the change is smallest. The bar chart in the result shows the measurement; the removed blocks are red.

- **Why whole blocks?** Qwen3.5 repeats a pattern of 4 layers: 3 fast linear-attention layers and 1 full-attention layer. Taking out whole blocks keeps that pattern, so every program that reads the model (Transformers, Unsloth, llama.cpp, Ollama) agrees about it. The first and the last block are never removed, and at most half of all blocks.
- **Measuring takes memory.** Models that do not fit the card in 16-bit are measured in **4-bit** (the ranking of the layers hardly changes). The page says how the model will be loaded. Choose fewer samples if the card runs out of memory.
- **Expert mode** adds: separate blocks instead of one run, protecting more blocks, the way the model is loaded, and typing the exact layers to remove (for example `36-51`, which skips the measuring).
- The **vision part** of a model that can read pictures is copied unchanged.

## Grow: the two starts
- **As an exact copy of the original** (default, recommended): the layers that are copied in have their output weights set to zero, so they do nothing yet. The program checks this on a sample of your text (the error must be the same before and after) and refuses to keep a model that fails the check. Training then switches them on. Because they are zero, the program also trains their output weights (Step 3 does this by itself); Expert mode has *Train only the new layers*.
- **Copy the layers** (SOLAR style): the copies work at once. The model is disturbed until it has been trained.

## After it is made
1. **Test against the original.** The same questions as Step 4, asked to the original and to the new model. Big models are loaded in 4-bit so both fit the card; the result says so.
2. **Train it on your world** (Step 3). Big slimmed models usually need QLoRA on a 32 GB card. 4-bit training is fine for dense Qwen3.5 models (Unsloth's caution is about the MoE ones), but it loses a little more quality than 16-bit, so check the result with the test.
3. In Step 4 a run on a made model can be compared with the made model or with the **original** it was made from.
4. **Step 5** exports it like any model. The picture (vision) file of the original is reused, because the vision part is unchanged. A model with an unusual number of layers (48 instead of 64) is something llama.cpp and Ollama may or may not accept: check with *Test it* on Step 5.

## Thinner layers (width)
Instead of removing layers, **every layer is made narrower**. About 60 % of a Qwen3.5 model is the MLP of its layers; the program measures on **your own text** which of its neurons matter least and takes them out of every layer (the same number everywhere, as the model format needs). This is the Minitron method of NVIDIA, and for the same size it usually keeps **more quality than removing layers**, but the result runs less fast than a model with fewer layers.

- **The size is exact before anything is measured.** The slider shows, for example, *MLP 13,312 of 17,408 (−24 %) · 64 layers · 23.69 B parameters (exact)*. For the 27 B the slider **starts on the mild cut (23.7 B)**; 21.1 B (−38 % of the MLP) is on the same slider and the page warns that it needs healing. **Both** removes a block *and* thins the rest (for example 60 layers and MLP 12,288, about 21.5 B).
- **Measuring** runs your text (256 samples by default) through the model once and keeps the result for this model and this data: *Measured earlier: reused* appears for every other size. *Measure now* does it ahead of time, *Measure again* redoes it (for example after a new data version).
- **Honest expectations.** Right after thinning, a 38 % cut is a damaged model and a 24 % cut is a clearly weaker one; the page shows the text-prediction error on your data before healing and never hides it. After healing with the budgets one card can afford (20 to 40 million words), expect that fluency and most behaviour come back, with some general knowledge and reasoning still missing. NVIDIA used about 94 *billion* words for its models. The goal is a usable model that trains on 24 GB after Step 3, not one as good as the original.
- **Expert mode** adds the number of measuring samples (128 to 1024) and, labelled *experimental* and **switched off for now**, thinning the linear-attention layers (48 → 32 value heads) and removing attention heads. Nobody has published how well that works for this kind of model, and it has not been tried on a real card yet.
- The **vision part**, the embeddings and the hidden size stay as they are, so the original's picture file still fits. The multi-token head is cut to fit but not measured or healed: do not use it for speculative decoding.

## Healing
The thinner model is **healed** by distillation: the **original model is the teacher** and answers your world's text and general text; the thinner one learns to answer like it did. This is what brings back most of what thinning took away. Your world's facts are still taught afterwards in Step 3.

- **General text is downloaded.** By default about half of the healing text is general text from Hugging Face (`mlabonne/open-perfectblend`, open, Apache-2.0), streamed while the original answers (only the rows that are needed are read), so you need an internet connection. **Offline, or with the box unticked, healing uses your world's text only** and says so; general ability then comes back less.
- **Choose the general text.** The *General text* list on the Heal card shows each dataset with its licence and kind (chat or plain text); a few need your OK on the licence, and one (NVIDIA's Nemotron) needs access approval and your Hugging Face token. *My own…* takes any Hugging Face dataset (owner/name, config, split, column) or a file on this computer (.jsonl, .json, .txt, .csv, .parquet); **Check it** reads about 20 rows and says whether it works, what it found, and why not if it does not. The licence of a dataset you add is yours to check. Chat rows are turned into text with your model's own chat template. Reasoning traces are long: use a sequence length of 4096 or more.
- **Budget:** 10, 20, 40 or 80 million tokens. The default is 20 million for models of 15 B or more and 40 million below. The card shows an estimate of the time (hours; a 27 B on one card needs roughly 5 to 10 hours for 20 million) until the benchmark has measured your card.
- **Memory and disk.** When the original and the thinner model do not fit the card together, they take turns: the original answers first and **saves its answers to disk** (about 4 GB for 20 million tokens, kept until you delete the model), then the thinner one learns (QLoRA from 15 B, else LoRA). The card tells you whether it fits and what it needs.
- **Stop and continue.** Stop saves a checkpoint; *Continue healing* (also after closing the program or an error) goes on where it was. The teacher's answers are never made twice.
- **The result** is a new, self-contained model (`models/<name>/healed`, as large as the thinner one) in the model list of Step 3, marked *healed*. The chart shows how far the thinner model's answers are from the original's and the text-prediction error on your data. The distillation checkpoints are deleted afterwards.
- **Test it.** *Test the healed model* asks the same questions as Step 4; the table shows **original · thinner · healed** side by side. Train the healed model in Step 3 as usual.

## Files and disk space
The new model is a full copy: about 2 bytes per parameter (a 21 B model is about 43 GB); healing adds another copy of that size (the healed model) and the teacher's saved answers. The page shows the space needed and checks the free space first. A model that is not on this computer is downloaded first (that needs its full size in the Hugging Face cache as well). **Delete** removes a made model; it is refused while a training run used it as its base.

## Problems
- **Out of memory while measuring.** Expert mode: load for measuring = 4-bit, or use fewer samples (64).
- **Not enough disk space.** The message says how much is needed and where.
- **"This model cannot be slimmed or grown".** The page says why. Today: Qwen3.5 and plain Llama / Qwen / Mistral-style models in 16-bit can lose layers; only dense Qwen3.5 / 3.8 can be made thinner. Gemma 4 and compressed (quantized) models are not supported.
- **Healing does not fit this card.** The card says how much memory it needs; choose a smaller model, fewer layers, or heal on a bigger card.
- **The measurement was not stable.** The two halves of your text chose different layers. Use 256 samples, or choose the layers yourself in Expert mode.

## Recommended: Qwen3.8 27B
A step-by-step guide is in the program's folder under `docs/PRUNE_QWEN38_27B.md`. The short version:
- **The 27B already fits most cards** (QLoRA from 24 GB, a 4-bit GGUF runs on about 17 GB). Slim it only if you need the smaller size; the result is never better than the original.
- **Start with the mild thinner-layers cut (MLP 13,312, 23.7 B)**, heal it, test, and only then try a thinner one (10,752, 21.1 B) or *Both*. Removing whole layers (2 blocks, 56 layers, 24.7 B) is the faster-running alternative.
- **Measure in 4-bit with 256 samples** of your own text; **heal with the default 20 million tokens** (general text on), then train in Step 3 with QLoRA. To keep a model thinking, keep at least 75 % reasoning-style examples.
- **Test the original first**, write down its three numbers, and compare every step against them.
- You need about 200 GB of free disk for thinning and healing (the original, the thinned model, the healed model and the teacher's answers).
