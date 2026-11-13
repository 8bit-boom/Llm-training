# Slim or grow a model

## What this page does
It makes a **new base model** from one you already have, by changing the number of its layers:
- **Slim it** takes whole blocks of layers out. The model gets smaller, faster and needs less memory (a 27 B model that does not fit your graphics card can become one that does).
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
2. **Train it on your world** (Step 3). Big slimmed models usually need QLoRA on a 32 GB card. Unsloth advises against 4-bit for Qwen3.5, so quality after training is something to check with the test, not to assume.
3. In Step 4 a run on a made model can be compared with the made model or with the **original** it was made from.
4. **Step 5** exports it like any model. The picture (vision) file of the original is reused, because the vision part is unchanged. A model with an unusual number of layers (48 instead of 64) is something llama.cpp and Ollama may or may not accept: check with *Test it* on Step 5.

## Files and disk space
The new model is a full copy: about 2 bytes per parameter (a 21 B model is about 43 GB). The page shows the space needed and checks the free space first. A model that is not on this computer is downloaded first (that needs its full size in the Hugging Face cache as well). **Delete** removes a made model; it is refused while a training run used it as its base.

## Problems
- **Out of memory while measuring.** Expert mode: load for measuring = 4-bit, or use fewer samples (64).
- **Not enough disk space.** The message says how much is needed and where.
- **"This model cannot be slimmed or grown".** The page says why. Today: Qwen3.5 and plain Llama / Qwen / Mistral-style models in 16-bit. Gemma 4 and compressed (quantized) models are not supported.
- **The measurement was not stable.** The two halves of your text chose different layers. Use 256 samples, or choose the layers yourself in Expert mode.
