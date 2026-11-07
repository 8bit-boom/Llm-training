# Guide: from your world to a model in nd-world

This guide walks through everything once, in order. The pictures show the real windows with a made-up demo world; your numbers will differ.

## Before you start
- **Your computer:** an NVIDIA graphics card with 8 GB of memory or more (16 GB or more is comfortable; your RTX 5090 is ideal), a recent NVIDIA driver (570 or newer for RTX 50 cards), about **15 GB** free disk space for the training software and **20 to 60 GB** more for models, checkpoints and exports, and an internet connection for the first downloads. Windows 11 or Ubuntu 24.04 (or similar).
- **Your world:** a world in nd-world. The program never connects to nd-world; you give it a file.
- **Ollama** (<https://ollama.com>) on this computer, or on the TrueNAS server, so nd-world can use the finished model.

## Install
- **Windows:** run `ND-Lore-Forge-Setup-<version>.exe`. It installs the program, its training worker and the `uv` installer tool. If Windows asks about the *WebView2* runtime, accept: the window needs it.
- **Linux:** download `ND-Lore-Forge-<version>.AppImage`, make it executable (`chmod +x`) and start it. On a newer Ubuntu you may need `sudo apt install libfuse2`.
- Your projects live in your *Documents* (Windows) or home folder (Linux) under *ND Lore Forge*; settings and the training software live in your user settings folder. Uninstalling the program does not delete your projects.

## 1 · Create a project
One project is one world. Press **Create a new project**, type a name.

![The welcome page](guide/welcome.png)

## 2 · Get the world data (Step 1)
In nd-world open **Tools → Export & Backup → Training bundle** and download the `.ndtrain` file. In ND Lore Forge choose it (or drag it onto the page). The program shows what it contains and, for a newer export, what changed. Your GM secrets are included on purpose: this is a model for you, the game master. Do not use it for your players' AI chat.

![Step 1](guide/data.png)

## 3 · Get the computer ready (⚙ Training setup)
Do this once. Every light should be green or yellow. Press **Install the training software** (several GB, 10 to 20 minutes), then **Test the graphics card**. A red light says what to fix.

![Training setup](guide/setup.png)

## 4 · Make the training data (Step 2)
Press **Make the examples**. Look at a few in the review list: **Drop** removes one, **Edit** fixes an answer. Optionally let a **teacher model** (for example Gemma 4 on the TrueNAS) write deeper examples and check them. Then **Save this version**.

![Step 2](guide/dataset.png)

## 5 · Train (Step 3)
Choose a model (the ✓ shows what fits your card), a **Quick test** first (minutes, to see that everything works), then **Normal**. LoRA is the safe default; Full fine-tune is offered for the small models only. Watch the loss go down and read the sample answers. **Stop** saves and ends early; the run can be continued.

![Step 3](guide/train.png)

## 6 · Test (Step 4)
Press **Test this run**: the original and the trained model answer the same held-out questions. Read the verdict and the answers. Ask your own question. (The numbers in this picture are an example.)

![Step 4](guide/test.png)

## 7 · Use it in nd-world (Step 5)
Make the model file (GGUF), install it into Ollama on this computer or the TrueNAS, and ask it one of the training questions as a check. The page then shows how to select the model in nd-world.

![Step 5](guide/use.png)

## When your world changes
On the first page, import the newer export and press **Retrain with the new export**. It makes new data, trains again (optionally continuing from the last model), and can export and install the result.

## Training on the TrueNAS server
Optional: see `docs/TRUENAS_WORKER.md`. In Step 3 choose the server under **Where to train**.

## If something goes wrong
Open **Troubleshooting** (the help panel of this page) or press **Make a problem report** below and send the text.
