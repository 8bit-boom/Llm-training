# Training setup

## What this page does
Gets this computer ready to train: it checks your NVIDIA graphics card and the driver, installs the training software (a Python environment with PyTorch and Unsloth), and tests that the card can really train.

## What you need before you start
- An NVIDIA graphics card (your **RTX 5090** is ideal; 8 GB of memory is the minimum, 16 GB or more is comfortable).
- A recent NVIDIA driver (version **570 or newer** for RTX 50 cards, 560 or newer for older cards).
- About **15 GB** of free disk space for the software (40 GB or more is comfortable; models and checkpoints are big).
- The installer tool **uv**. The installers ship it; if you run from source, install it from <https://docs.astral.sh/uv/>.
- An internet connection for the install.

## What to do
1. Look at the list on the left. Every line should be **green**; yellow is fine, **red** must be fixed first (the line says how).
2. Press **Install the training software**. It downloads several GB, one time (PyTorch, then Unsloth and the training libraries). If Unsloth cannot be installed on your computer, the plain libraries are installed instead; training still works, only slower and with more graphics memory. You can watch the log and **Cancel** any time.
3. Press **Test the graphics card**. The card trains a tiny model for about 30 seconds.
4. When all lines are green, you are ready for Step 3 · Train.

## If something goes wrong
- **"nvidia-smi was not found"**: the NVIDIA driver is not installed. Install it from <https://www.nvidia.com/drivers>, restart, and press **Check again**.
- **"needs NVIDIA driver 570 or newer"**: update the driver; RTX 50 cards cannot run older ones.
- **The install stops with an error**: read the last lines of the log. Most often it is the internet connection or full disk. Fix it and press the button again; it continues where it can.
- **"This PyTorch build does not support the graphics card"**: the driver is too old for the CUDA build that was installed. Update the driver, then install again.

## How long does it take?
The install: 5 to 20 minutes, depending on your internet. The test: about 30 seconds.

## With or without Unsloth
**Unsloth** is an add-on library that makes training faster and use less graphics memory. It is picky: on some computers (Windows especially) it does not install or does not start. Tick **Install without Unsloth** before pressing *Install the training software* and the program installs only the **standard libraries** (Transformers, PEFT and TRL, which every computer with an NVIDIA card can use). Nothing else is needed: the program fetches Python and PyTorch itself, and the model is downloaded from Hugging Face the first time, or taken from a folder you already have (Step 3, *Add a model*). The page tells you which one is installed. Without Unsloth, training is slower and can need more graphics memory than the plan's estimate; a Quick test measures the real figure. Qwen3.5 needs two extra kernel packages to run fast without Unsloth; the setup tries to install them and says so in its log when it cannot (training still works, only much slower). You can also choose per run in Expert mode (*Training engine*: auto, unsloth, standard).

## Unsloth Desktop (optional)
The card at the bottom connects Lore Forge to **Unsloth Desktop** (or Unsloth Studio) so that Step 3 can train there. **Find** looks for it on this computer (the usual ports 8888 and 8000); **Test** says whether it is Unsloth, whether the key works, which graphics card it has and whether it has the training API; **Save** keeps the address (and the key, in the program's secrets file, never in a project or a report). Unsloth Desktop only answers while it is open. To use one on another computer, switch on *Settings, API, Remote & LAN, LAN access* in Unsloth and type that computer's address; the page warns about addresses on the internet that are not encrypted. The key is made in Unsloth (Settings, API) and shown once there.

