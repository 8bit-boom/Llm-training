# Step 3 · Train

## What this step does
Teaches the chosen model your world. It uses the **newest saved version** from Step 2: the world book (the lore as text) and the practice questions with answers. The model itself is not changed: a small add-on (**LoRA**) is trained and saved, which keeps the original model safe and needs far less memory.

## What you need before you start
- Step 2 done (a saved version) and **⚙ Training setup** all green, including the graphics-card test.
- The first run of a model **downloads it** (2 to 20 GB) from Hugging Face, so you need an internet connection once.

## What to do
0. **Where to train.** On this computer, or on your TrueNAS server's graphics card (add it once with its address and access token; see docs/TRUENAS_WORKER.md). With a server chosen, the cards show what fits *its* card. While it trains there, AI chat, transcription and image generation on that server wait. The trained model is copied back to this computer.
   A third choice, **Unsloth Desktop**, appears once you have saved it in *Training setup* (see below): the data goes to Unsloth Desktop and the training runs there.
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
5. **Two boxes under the buttons** make long runs easier to leave alone:
   - **Keep this computer awake during training** (on by default, remembered): the computer does not go to sleep while a training runs, because a sleeping computer stops the training. The screen may still turn off. It applies from the next start; the computer sleeps normally again afterwards. If the system does not allow it, the log says so and you should switch off sleep in the system settings.
   - **Shut down this computer when training has finished** (off at every program start, on purpose): when a run finishes normally, a bar appears with a **60-second countdown and a Cancel button**, then the computer is switched off. It does *not* switch off after a stop or an error, so you can look at what happened. The box clears itself after it has been used. It can be ticked while a run is going. For a run on a server, it switches off *this* computer once the result has been fetched. Linux may refuse to power off without permission; then the program says so and the computer stays on.
6. **Stop and save** saves the current point and ends cleanly after the current step. Every stopped, failed or interrupted run in the list has its own **Continue** button that picks up from the saved point with the same settings. If you close the program while training, it first stops and saves (up to about two minutes); if the computer shuts down or the program crashes, the run shows as *interrupted* next time and can still be continued from the last automatic checkpoint (only the steps since then are repeated). The graphics card is free again as soon as training ends.

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

## Slimmed and grown models
A model made on **🧬 Slim or grow a model** is in the list too (marked *added*, with a line such as *slimmed: 64 → 48 layers*). Train it like any other. A slimmed model answers worse than the original until it has been trained; a big one usually needs QLoRA on a 32 GB card. A grown model that started as an exact copy also trains its switched-off layers' output weights automatically; Expert mode has *Train only the new layers*.

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
Under the plan, **Find the fastest settings** runs a few short real training runs (16 steps each) with different settings and shows tokens per second and memory for each: your current settings, batches grouped by length or not, and larger batches. Every run loads the model again, so it takes several minutes; **Stop** ends it and keeps the rows made so far. A row that runs out of memory says so and does not stop the others. **Use the fastest settings** applies the winner to the next run. Tick *Also try torch.compile, Liger kernels and padding-free batching* to compare those too; they exist only on the plain-libraries path (not with Unsloth), and each one switches itself off with a reason when it cannot work. In Expert mode the same options are dropdowns. Speeds are measured after the start-up (the first quarter of the steps is left out), so the first run is not held back by compiling. Tick the extra options and a **second round** follows: the best batch size of the first round with each attention kernel (also with Unsloth), every setting run **twice** with both speeds shown. Settings closer than about 3 % (or than their own two runs) are marked *within noise*: do not read a ranking into them. The ★ is the fastest of round 2.

## Training in Unsloth Desktop
Unsloth Desktop is a separate app (unsloth.ai) that trains models with its own screens. Lore Forge can hand it your saved data and follow the training, instead of using its own training software. It is never chosen by itself: this computer stays the default.
1. In Unsloth: your picture, **Settings**, **API**, give a name, **Create**. Copy the key (Unsloth shows it only once).
2. In Lore Forge, **Training setup**, card **Unsloth Desktop**: press **Find** (it looks on this computer), paste the key, press **Test**, then **Save**.
3. Step 3, **Where to train**: choose Unsloth Desktop. Press Train. **Keep Unsloth Desktop open** until it finishes.
- The charts, Stop and the finished run look like a normal run. Unsloth chooses the speed settings itself; Lore Forge sends the rank, learning rate, epochs and context length.
- When Unsloth runs on **this computer**, the trained adapter is copied into the run, so Step 4 and Step 5 work as usual. When Unsloth is on **another computer**, the adapter stays in Unsloth's folder (the run says which) and cannot be tested or exported here.
- Not available there: full fine-tuning, two stages (world book first), a model from a folder on this computer, and continuing a run that was stopped (a stop cancels the job in Unsloth; start again).
- If Unsloth is busy with another job, yours waits in its queue and the page says its place. Closing Unsloth forgets a waiting job; a running one goes on, and **Continue** reconnects to it.
- The data contains the GM's secrets. It is sent only to the address you saved. Use an address on this computer or your own network.
