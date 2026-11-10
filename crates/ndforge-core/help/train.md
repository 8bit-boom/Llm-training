# Step 3 · Train

## What this step does
Teaches the chosen model your world. It uses the **newest saved version** from Step 2: the world book (the lore as text) and the practice questions with answers. The model itself is not changed: a small add-on (**LoRA**) is trained and saved, which keeps the original model safe and needs far less memory.

## What you need before you start
- Step 2 done (a saved version) and **⚙ Training setup** all green, including the graphics-card test.
- The first run of a model **downloads it** (2 to 20 GB) from Hugging Face, so you need an internet connection once.

## What to do
0. **Where to train.** On this computer, or on your TrueNAS server's graphics card (add it once with its address and access token; see docs/TRUENAS_WORKER.md). With a server chosen, the cards show what fits *its* card. While it trains there, AI chat, transcription and image generation on that server wait. The trained model is copied back to this computer.
1. **Choose the model.** Each card shows whether it fits your graphics card (✓ fits, ⚠ tight, ✗ too big) and the memory it needs. Qwen3.5 4B is a good start; on a 32 GB card all five fit.
2. **Choose how to train.** Two questions:
   - **How much of the model should change?**
     - **LoRA, 16-bit** trains a small add-on and leaves the original model safe. It is the best choice whenever it fits.
     - **QLoRA, 4-bit** is the same with the model shrunk to 4-bit: it needs less memory and loses a little quality. For **Qwen3.5** the makers of the training software advise against it.
     - **Full fine-tune** changes every weight of the model. It learns your world most strongly, but needs much more memory and disk (the card tells you how much), can make the model forget how to chat if pushed too far (so it uses a very small learning rate), and the result is a full-size copy of the model, not a small add-on. It needs a graphics card with bf16 (RTX 30 series or newer) and runs on **this computer only**. On a 32 GB card it fits Qwen3.5 2B, and 4B and Gemma 4 E2B only just; a card that is greyed out says why.
   - **What should it learn from?**
     - **The world book and the practice answers together** (default): one pass over everything.
     - **The world book first, then the practice answers**: two stages. First the model reads all your lore as text, then it practises answering questions. This tends to remember facts better and takes longer. The charts and the step counter carry on across the two stages. If you stop during the second stage, **Continue** skips the finished first stage.
   - **Quick test** trains 60 steps in a few minutes: use it first to prove everything works (the model will not have learned much; it is always one stage). **Normal** (2 passes over the data) is for real use; **Best quality** is slower with a longer memory span.
3. Read **The plan**: the number of steps, the memory estimate and any notes. If your card is short of memory the plan lowers the batch size or the sequence length for you and says so.
4. Press **Start training**. Watch the **loss** chart (it should go down) and the **sample answers**: the same few questions are answered before training and then again every few minutes, so you can see the model learning your world.
5. **Stop and save** saves the current point and ends cleanly after the current step. Every stopped, failed or interrupted run in the list has its own **Continue** button that picks up from the saved point with the same settings. If you close the program while training, it first stops and saves (up to about two minutes); if the computer shuts down or the program crashes, the run shows as *interrupted* next time and can still be continued from the last automatic checkpoint (only the steps since then are repeated). The graphics card is free again as soon as training ends.

## Good to know
- **Loss numbers:** a loss of 13–15 for Gemma 4 E2B/E4B is normal (a quirk of these multimodal models). What matters is that it goes **down**.
- **Memory:** the figures are estimates until a run has finished on your computer; then the real measured peak is shown instead.
- **Time:** the estimate appears after the first steps. Qwen3.5 starts slowly the first time (special kernels are prepared).
- **Expert mode** shows rank, learning rate, epochs, batch size and sequence length. Changing them is rarely needed.
- A trained model is saved in the run's folder: a small add-on (LoRA / QLoRA) in `adapter/`, or the whole model for a full fine-tune in `model/`. Step 5 turns either into a file for nd-world.

## If something goes wrong
- *"ran out of memory"*: pick a smaller model, QLoRA, or lower the sequence length in Expert mode.
- *"could not be downloaded"*: check the internet connection and try again.
- *The loss stays flat or becomes `nan`*: lower the learning rate (Expert mode) and try again.

## Adding a model of your own from Hugging Face
Under the model cards press **＋ Add a model from Hugging Face…**, type the model's name as it appears on <https://huggingface.co> (like `unsloth/Qwen3-4B`) or paste its address, and press **Look it up**. The program reads the model's public description and tells you what it found: size, licence, chat format and a memory estimate. Then **Add it to the list**; it appears as a card marked *added* and works like the five built-in ones. The **✕** button next to the list removes it again.

What can be added: an **open** model (not gated, no login) with **.safetensors** weights that is a **chat / instruct** model (it has a chat template). What is refused, with the reason: gated or private models, GGUF-only repositories (those are for running, not training), base models without a chat template, models that need to run their own code to load, models already shrunk by someone else (GPTQ / AWQ / bitsandbytes), and models that are not language models.

Good to know:
- Only Qwen3.5 and Gemma 4 have been checked in detail. For any other model the program uses the model's **own chat template**: do a **Quick test** first, then look at the sample answers. The memory figure is an estimate until a run has measured it.
- Installing such a model into Ollama uses the template stored in the GGUF file instead of a hand-written one.
- A model's licence is its own: read the model page before you share the trained model.

## Speed numbers while training
The training card shows **Speed** (tokens per second, counting only real text) and **Padding** (the share of the computed tokens that is wasted filling to make every example in a batch equally long). Below it one line says how batches were made: when many examples differ a lot in length, the program puts examples of similar length together, which cuts the padding and the time. Warnings appear there too, for example when Qwen3.5's fast kernels are missing (training then runs much slower) or when the graphics card has no bf16. Details and what was looked at: `docs/TRAINING_SPEED.md`.

## Find the fastest settings
Under the plan, **Find the fastest settings** runs a few short real training runs (16 steps each) with different settings and shows tokens per second and memory for each: your current settings, batches grouped by length or not, and larger batches. Every run loads the model again, so it takes several minutes; **Stop** ends it and keeps the rows made so far. A row that runs out of memory says so and does not stop the others. **Use the fastest settings** applies the winner to the next run. Tick *Also try torch.compile, Liger kernels and padding-free batching* to compare those too; they exist only on the plain-libraries path (not with Unsloth), and each one switches itself off with a reason when it cannot work. In Expert mode the same options are dropdowns. Only clear differences mean something: start-up time is part of every number.
