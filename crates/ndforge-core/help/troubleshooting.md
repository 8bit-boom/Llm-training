# Troubleshooting

Press **Make a problem report** on the guide page to get a text with the facts below, and send it along with your question. It contains no passwords or access keys.

## The program
**The window does not open (Windows).** Install the *Microsoft Edge WebView2 runtime* (<https://developer.microsoft.com/microsoft-edge/webview2/>), then start again.

**The AppImage does not start (Linux).** `chmod +x` the file; on Ubuntu 24.04 install `libfuse2`; or run it with `--appimage-extract-and-run`. A very old system may lack WebKitGTK 4.1.

## Training setup
**"nvidia-smi was not found".** The NVIDIA driver is missing or broken. Install it from <https://www.nvidia.com/drivers> and restart. Test in a terminal: `nvidia-smi`.

**"Driver too old".** RTX 50 cards (like the 5090) need driver **570 or newer**; older cards 560 or newer.

**Installer tool (uv) missing.** Reinstall the program (the installers include it) or install uv from <https://docs.astral.sh/uv/>.

**The install fails halfway.** Press **Install the training software** again: it continues. Typical causes: no internet, a proxy, or too little disk space. The log under the button shows the last lines.

**"PyTorch cannot see an NVIDIA graphics card" (or "the CPU-only build") although the driver is new.** Most often the training software has the CPU-only PyTorch: another package replaced the NVIDIA build while it was being installed. Press **Install the training software** again: the setup now keeps the NVIDIA build, checks it, and puts it back if it was replaced. To look yourself, run this in a terminal, with the path of the environment shown on the setup page (on Windows it is under *AppData\Roaming\NDLoreForge\env*): `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`. A version ending in `+cpu`, or `None` for CUDA, is the CPU-only build. `True` means PyTorch is fine and the graphics card test should pass. If the CUDA version is shown but the card is not available, restart the computer after a driver update and close programs that reserve the card.

**The graphics-card test fails.** Close games and other programs that use the card, update the driver, run the test again.

## Training
**"Out of memory".** Choose a smaller model, **QLoRA**, or the Quick preset; close other programs that use the card (also Ollama: it keeps models in memory, `ollama stop <model>`).

**The model cannot be downloaded.** The first run downloads it from Hugging Face (2 to 60 GB). Check the internet connection and disk space. The six models are not gated; no account is needed.

**Full fine-tune is greyed out.** The reason is written on the card: it needs a bf16 card (RTX 30 series or newer), enough memory (small models only) and disk space. This computer only, not the server.

**Training stopped or the computer restarted.** Open Step 3 and press **Continue** on the run: it resumes from the last checkpoint.

**The loss looks odd.** Gemma 4 E2B and E4B start with a loss of 13 to 15; that is normal. Qwen3.5 starts around 2 to 4. It should fall within the first steps.

## Test and use
**"No questions were kept for testing".** In Step 2 set **Kept for testing** above 0 %, make the examples again, save a version and train again.

**Everything scores low.** The word-overlap score is rough. Read the answers; try the teacher grade; train longer (**Normal** instead of Quick).

**The model invents lore.** Add "not in the lore" examples with the teacher (Step 2) and train again. In nd-world keep retrieval on.

**Ollama cannot be reached.** Is it running (`ollama serve`)? For a server on another computer Ollama must listen on the network (`OLLAMA_HOST=0.0.0.0`). Check the address in Step 5 (default `http://localhost:11434`).

**The installed model answers strangely in nd-world.** Use the check question on Step 5. The template shipped with the model is meant to answer without thinking; if you installed through other tools, install through this page instead.

## Slim or grow a model
**Out of memory while measuring the layers.** Open Expert mode on the page and choose "4-bit" for loading, or measure on fewer samples (64).

**"There is not enough free space".** A made model is a full copy (about 2 GB per billion parameters) and a model that is not on the computer is downloaded first. Free the space or choose a smaller size on the slider.

**"Healing a model of … needs about … GB … this card has …".** The thinned model and the original do not fit this card even when they take turns. Choose a smaller model (a thinner 9 B instead of a 27 B), or heal on a bigger card; thinning and testing still work.

**"The teacher's answers need … GB in … free".** Healing saves the original's answers to disk first (about 4 GB per 20 million tokens). Free the space, choose 10 million tokens, or move the project's folder to a bigger drive.

**"The general text (…) could not be downloaded".** Healing goes on with your world's text only; nothing is lost, but general ability comes back less. Check the internet connection and press *Continue healing* later (the teacher's answers for your text are kept), or untick the general text on the Heal card if you work offline.

**"Healing stopped: the training became unstable".** The error went to infinity. Open Expert mode on the Heal card, lower the learning rate (for example 0.00005) and press *Continue healing*; it goes on from the last checkpoint.

**"The saved teacher answers belong to other data".** The training data or the model changed since the original answered. They are made again (the older answers stay on disk until the model is deleted).

**"The healing result does not fit the slim model".** The saved LoRA does not match the thinner model's layers. Nothing was changed; please make a problem report (Welcome page).

**Ollama or llama.cpp refuses the exported slimmed model.** An unusual number of layers is allowed by the file format, but not every version of Ollama accepts it. Update Ollama, then use *Test it with a question* on Step 5. The text GGUF can also be loaded in llama.cpp directly.

## TrueNAS server
**"Cannot connect".** The address needs the port (`http://truenas:8765`); the container must be running; the token must match. See `docs/TRUENAS_WORKER.md`.

## Teacher
**The teacher keeps failing.** Test the connection first. For Ollama: the model must be downloaded (`ollama pull <name>`), the address like `http://truenas:11434`.

## Still stuck
Send the problem report, what you pressed and what the window said.
