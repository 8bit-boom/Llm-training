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
5. **Stop** saves the current point and ends cleanly after the current step. **Continue the last run** picks up from the saved point. The graphics card is free again as soon as training ends.

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
