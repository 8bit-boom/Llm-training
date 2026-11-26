# Tune this PC for a model

## What it does
**Auto: test and choose the best settings** tests, one after the other, the settings that decide how fast and how well a heal runs for the chosen model on THIS computer. At the end it shows every change with what it is worth, and you confirm once. The result is also kept as a **preset** you can select again later (Heal card, Auto page, this page). Nothing is changed before you confirm, and a stop at any time still gives a preset from what was finished.

## What is tested, in which order
1. **Speed and memory (same results).** Tokens per forward pass of the student; the loss chunk and keeping the output layer's logits; the original's tokens per forward pass; Try first's rows measured together; the thinking check's questions asked together (same on average, not one by one); the vLLM engine settings (only when vLLM is switched on).
2. **Quality settings (a short trial, not a full heal).** LoRA size, optimizer, warm-up and learning rate, each tried alone from your settings in a short real heal on your own rows with the same seed. The held-out loss decides, and only when it beats twice the run-to-run noise. When several settings win, they are tried together once; settings that do not add up are not applied together. A short trial can disagree with a long heal: use the thinking check on the real result.
3. **Other functions.** llama.cpp's layers on the card (needs a 16-bit GGUF in the project), and the disk speed (it only measures).

Not tested: sequence length, batch per step, sequences per update, gradient checkpointing, method (LoRA or QLoRA), answers kept per token (top-k) and how the original is loaded. They change what the model learns or how it is trained, so a speed test cannot choose them; the program's own memory plan does.

## Before you press it
The page lists every test with its time and a total ("done around ..."). Tick fewer tests or set "I have at most N hours": the most valuable tests run first. A test says why it cannot run (for example, no 16-bit GGUF, vLLM off, not enough disk). The big tests need the graphics card: if another program holds it, the run waits and names the program; nothing is ended for you.

## After it
The confirmation lists the changes for the Heal card and the program settings that were tested (the original's batch, the vLLM engine, llama.cpp's layers, Try first and the thinking check). **Apply all** fills the Heal card and saves those settings. Select the preset later from the dropdown, or on the Auto page under **Use tested settings for this PC**: Auto-cut then uses the speed settings of the preset in every heal and its time estimate uses the measured speed. The quality settings of a short trial are not taken by Auto-cut; use them from the Heal card if you want them.

A preset belongs to one model and one graphics card. A preset made on another card is listed and never used here. A result is marked stale when the driver, torch or another part it depends on changed.
