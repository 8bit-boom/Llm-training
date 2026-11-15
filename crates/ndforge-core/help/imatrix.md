# Importance matrices

An **importance matrix** is a short measurement. The model reads some typical text, and llama.cpp notes how strongly each part of it is used. When the model is later shrunk to about 4 or 5 bits (**Q4_K_M, Q5_K_M, Q6_K**), the rounding takes more care of the parts that matter, so the small file loses less quality. For **Q8_0 and F16 it does nothing**: those sizes never use it.

This page makes a matrix **on its own, without making any GGUF file**. You keep it in the project, use it in later exports, and can save a copy elsewhere (for example to run `llama-quantize` on another computer, or to share it).

## Your matrices

Every matrix has a label (yours), the model it was made for, its size, the amount of text, whether general text was used, whether the speculative-decoding head (MTP) is in it, its date and who made it (llama.cpp, Unsloth, or *imported*).

- **Save a copy…** copies `imatrix.gguf` (and `imatrix.dat` if there is one) and `imatrix-info.json` (what it was made from) into a new folder named after the matrix, inside the folder you choose. If that name is taken, a number is added: nothing is ever overwritten. The program checks the free space first.
- **Open folder** shows the matrix's folder in your file manager.
- **Make a .dat (legacy)** converts the matrix to the old file format some tools still read (done by `llama-imatrix`; the page says if your build cannot).
- **Delete** removes it from the project (after a question). It is refused while an export or another job uses it. Copies you saved elsewhere stay.
- **Add a matrix file from my computer…** copies a `.gguf` or legacy `.dat` matrix (made on another computer, or received from someone) into the library, with a label. **Add to library** does the same for a matrix that an earlier export saved next to its GGUF files; those are never moved or changed.

## Create a matrix

1. **The model comes from**
   - **A trained run**: its add-on is merged into its base model and converted to a 16-bit GGUF file with llama.cpp, exactly as the export does. If an earlier export of that run left a 16-bit file, it is used instead (no new merging). Needs the training software.
   - **A model**: one of the model list (made, slimmed and healed models are in it), a folder with a full-precision model, or a Hugging Face id (downloaded if needed). It is converted with llama.cpp's converter.
   - **A GGUF file I have**: any GGUF `llama-imatrix` accepts. A 16-bit (F16/BF16) or Q8_0 file is best: **a more precise file gives a better matrix**. To use your saved data you also say which model the file was made from, because that model's chat template writes the practice conversations.
2. **Calibration text**
   - **My saved training data and general text** (the default): your saved data version written through the model's chat template, about 80 %, plus about 20 % general text from the chosen dataset (the same chooser as in Step 5; **Check it** tests a choice). Without general text, only your data is used.
   - **A text file of my own**: plain text, used exactly as it is.
3. **Amount of text**: 64 / **128** / 256 chunks. More is slower and a little better. The page shows a rough time guess; the log shows the real progress.
4. **The speculative-decoding head (MTP)**: measured too when the model has one and your `llama-imatrix` has `--nextn`; otherwise the matrix is made without it and the result says why.
5. **Also make a legacy .dat file** and a **label** are optional.

**Stop** ends the measuring at once and keeps nothing. When the matrix is ready the page shows any notes and, when your llama-imatrix supports it, a few lines of statistics (how many tensors, the largest and the mean value).

## Merge matrices

Tick two or more matrices in the list (and/or add matrix files, and/or use **Unsloth's published matrix** for the base model) and press **Merge**: llama.cpp (`llama-imatrix --in-file a --in-file b`) combines them into a **new** entry; the matrices you merged are not changed. The new entry is used in exports and in **Compare quants** like any other.

- **Merging adds the statistics.** A matrix measured on more text outweighs one measured on less: a large published matrix can drown a small one of your own. The page tells you how much each one weighs. To give yours more say, make it with more chunks.
- **Unsloth's published matrix describes the unmodified base model**, not your fine-tune. It is a reasonable start for a light LoRA, less so for a heavily changed model.
- The published matrix is **downloaded from the Hugging Face Hub** (the repository `unsloth/<model name>-GGUF`, the way Unsloth's own code names it). Press **Look it up** first: the page shows the repository, the file and its **size**, and says what will be downloaded (once, into the project's `imatrix_downloads` folder). Offline, no such repository, a private or gated one, or a model that is a folder on this computer are all explained in plain words.
- The matrices are compared from their headers (architecture, layers, tensors). A mismatch is a **yellow warning**, never a refusal; if llama-imatrix refuses, its own message is shown.
- In **Create a matrix**, the tick *Also merge Unsloth's published matrix for the base model* makes a second, merged entry after yours; yours is kept even if the download fails.

## Use a saved matrix in an export

In Step 5, tick *Better small sizes (importance matrix)*, then choose **Use a saved matrix**. Nothing is measured: the file is copied into the export and given to `llama-quantize` (or Unsloth) as it is. Before the export the page compares the matrix with the model (architecture and number of layers, when they can be read). A **yellow warning** means "this does not look like the same model", or that the file could not be read; it never stops you, `llama-quantize` has the last word. **Add a matrix file from my computer…** is there too.

## When something goes wrong

- *"llama.cpp was not found"*: press **Install llama.cpp now** under the message (or in **⚙ Training setup**): the program downloads the build that fits your computer from GitHub into its own folder, after you have seen the size, and checks that it starts. To use a copy of your own instead, set the environment variable `LLAMA_CPP_DIR` to its folder (<https://github.com/ggml-org/llama.cpp>); that always wins. The program looks in `LLAMA_CPP_DIR`, then in the folder it installed llama.cpp into, then in a `llama.cpp` folder in your home folder, then next to the program; `llama-imatrix` is also looked for on the PATH (the converter script is not). A GGUF file as the source needs only `llama-imatrix`; a model also needs `convert_hf_to_gguf.py`.
- *"could not turn this model into a GGUF file"*: the model is not a full-precision Hugging Face model this llama.cpp knows. Update llama.cpp, or use a GGUF file.
- *"llama-imatrix failed"*: see the log. Often the file is not a model it can read, there is not enough memory, or llama.cpp is old.
- *"not enough free disk space"* or *"The disk is full"*: the 16-bit file of a big model needs as much space as the model itself. Free some space.
- *"The text file … is empty"* / *"was not found"*: choose the file again.

Nothing here has been tried with a real llama.cpp yet; the notes in `docs/IMATRIX.md` ("Not verified yet") list what to check on your computer.
