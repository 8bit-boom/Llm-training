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
