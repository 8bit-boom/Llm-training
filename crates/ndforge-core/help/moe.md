# Cutting a mixture-of-experts model (Inkling-2-Mini)

## What is different
A mixture-of-experts model has many small feed-forward blocks, the **experts**, and a **router** that picks a few of them for every token. Inkling-2-Mini has 128 experts in each of 24 layers and uses 8 per token. Its middle layers (6 to 17) run **three times**. The file is 19.5 GB because the experts are stored in 4-bit and 8-bit packed form; the program cuts them **without decoding** them, byte for byte.

## The cuts
- **Fewer experts** (112, 96, 80 or 64 kept in every layer): a smaller file and less memory, the same speed. It needs the experts to be measured on your text first. Counts are multiples of 16, never below 32, and at most half in one stage.
- **Fewer experts per token**: a little faster. The model's authors did not test it; check the answers.
- **Fewer loop passes** (3 to 2): faster and a smaller memory for the loop, but the model was **never trained** with 2 passes. Auto-cut never does this by itself. The variable-loop research checkpoint (`thinkingmachines/TML-d24-varloop-research`) was trained for it.
- **More passes or more experts per token** are questions for Try first, not improvements.
- **Whole blocks** of 6 layers (expert mode): a quarter of the model per block, very risky.

## Software it needs
The page lists what is installed and what is missing: transformers 5.24 or newer (for measuring, trying, healing and training), llama.cpp b12611 or newer (an older one gives garbage after the first loop pass), vLLM 0.35 or the nightly image (only for a fast teacher; the program falls back to its own slower one). Upgrading transformers only adds the model; Qwen and Gemma keep working. The button **Upgrade transformers** (also **Update** in the card "Python packages" on the Training setup page) does it for you: it shows what it will do, waits for **Confirm**, keeps PyTorch as it is, and puts the old version back by itself if anything fails. **Go back to ...** installs the earlier version again. It is refused while a job, a queue run or an Auto run is active. A button asks Hugging Face whether the model's files changed since the program was tested.

## Order of work on a 32 GB card
1. Decide the goal: smaller file, faster answers, or teaching it your world (then train the uncut model and do not cut).
2. Measure the experts. 3. Try first. 4. Cut 128 to 96. 5. Run the thinking check. 6. Heal only if the check fails. 7. Export with llama.cpp b12611 or newer.

The Recommendations table also shows an NVFP4 size; it is only an estimate (no NVFP4 file is made). Healing uses the packed model in this program as the teacher (vLLM is not used for this model). An importance matrix of the original can be adapted to a cut that removed experts.

Nothing here has been seen working on a real card yet. `docs/MOE_GUIDE.md` has the details and the list of what is not verified.
