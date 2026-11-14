# Step 5 · Use in nd-world

## What this step does
Turns a trained run into a file nd-world can use, puts it into **Ollama** (on this computer or on your TrueNAS server), tests it with a question, and shows exactly what to set in nd-world.

## What you need before you start
- A trained run (Step 3). Choose the newest finished one.
- **Ollama** running where you want to use the model: <https://ollama.com> for this computer; on the TrueNAS server nd-world's own Ollama (it must listen on the network, `OLLAMA_HOST=0.0.0.0`).
- For the GGUF file: the training software from **⚙ Training setup** (Unsloth makes the file; without Unsloth, llama.cpp is needed, see below).

## What to do
1. **Make the files.** Choose the run and the **file size**:
   - **Q4_K_M** is the smallest and good for almost everything (recommended).
   - **Q5_K_M**, **Q6_K** and **Q8_0** are larger and a little closer to the original; **F16** is the original precision.
   - You can tick **several sizes** at once (for example Q4, Q5, Q6 and Q8). The model is merged and converted once and then shrunk to each size, so the extra sizes cost little time. Every size becomes its own entry under *Your files* (and can be installed into Ollama under its own name).
   - **Context** is how much text the model reads at once (8192 is a good start; more needs more memory).
   - **Better small sizes (importance matrix)** is an optional tick, off by default, for GGUF files. llama.cpp first reads some of your own training data (as the model sees it, through its chat template) plus about a fifth general text (from the dataset chosen in the *General text* list under the tick: the open default `mlabonne/open-perfectblend`, another preset, or your own; **Check it** tests a choice), and learns which weights matter most; **Q6_K, Q5_K_M and Q4_K_M** are then made with more care and lose less quality. **Q8_0 and F16 stay as they are** (it would change almost nothing there). Choose how much text it reads: **64 / 128 / 256 chunks** (128 is a good start; more is slower and a little better). The page shows a rough time guess; the log shows the real progress. It needs llama.cpp with `llama-imatrix`, which measures the matrix. If the installed Unsloth can apply a matrix, **Unsloth then makes the files with it**; if not, llama.cpp makes them (convert, then `llama-quantize`). The list of files says which one made each GGUF. Without llama.cpp the files are made through Unsloth without a matrix, and the page says so. For a model that has a **speculative-decoding head (MTP)**, such as Qwen3.5, a second tick, **Include the speculative-decoding head (MTP) in the matrix**, is shown (on by default): llama-imatrix then also measures that head (`--nextn`). If your llama-imatrix is too old for it, or refuses, the matrix is made without the head and the page says why. The matrix is saved next to the files as `imatrix.gguf` (with `imatrix.json`), and **Reuse the saved matrix** (on when one exists) skips the measuring next time for the same model, data, settings and general text (another dataset makes a new matrix). If the general text cannot be downloaded the matrix is made from your world's text only; if anything else about it fails (for example an older llama.cpp without `llama-imatrix`) the files are made without it and the page says so. **Cancel** stops the measuring at once.
   Press **Make the files**. This joins your trained add-on into the model and makes a **GGUF** file, a **Modelfile** and a short model card in `exports/`.
   - **Make** chooses what comes out: *GGUF + Modelfile* (for Ollama, the default), *GGUF + the merged model* (also keeps the add-on baked into the model as ordinary safetensors files, 10 to 20 GB more) or *Only the merged model* (no GGUF; it loads in Unsloth, Transformers, vLLM and so on without the add-on, but Ollama cannot use it).
   - **Pictures (vision).** Qwen3.5 and Gemma 4 can read images (the base model's own vision part). Training changes only the language layers, so the *base model's vision file* still fits. For a model that has one, **Make the files** adds it (`mmproj-F16.gguf`, taken from the official GGUF repo of the base model, about 1 GB) next to every GGUF, with a `VISION.md` that says how to use it. **Unsloth Studio** and **llama.cpp** use it for pictures. **Ollama** takes one GGUF file only, so it gets the text model and answers text. A full fine-tune changes the vision part too, so it gets no vision file. A run on a model made on the *Slim or grow a model* page keeps the original's vision part unchanged, so it gets the **original model's** official vision file too. A model with an unusual number of layers is something llama.cpp or Ollama may refuse: use *Test it with a question* before relying on it. If a model has no official GGUF repo, the log says so and the GGUF is text only. The model was trained on text only: it reads pictures as before, but knows nothing new about your pictures.
   - **Save the LoRA add-on…** copies the small add-on of the chosen run (and its tokenizer) to a folder you pick, with a `README.md` that says which base model it needs and how to load it in Unsloth or PEFT or merge it yourself. It needs no training software. A full fine-tune has no separate add-on: its whole model is in the run's folder.
2. **Put it into Ollama.** Pick the server (add the TrueNAS one with its address, for example `http://truenas:11434`), check the model name, then press **Install into Ollama**. Big files take a while: the file is checked, sent and registered.
3. **Test it.** Press **Test it with a question**: one of the questions from the training is asked. You should get an answer that fits your world. If the answer is empty or strange, do not use the model yet.
4. **Use it in nd-world.** Follow the numbered list on the page (it contains the exact setting to copy).

## Important: this model knows your secrets
It was trained on **all** of the world's text, including GM secrets, and it will repeat them. Use it only for yourself. If players can use AI in this world, turn that off or keep them on another model.

## Good to know
- The model was trained **without "thinking"**. The files are prepared so that Ollama and Unsloth Studio answer the same way.
- Keep nd-world's **lore search** on: the model was trained to use the context nd-world gives it.
- Making the files again with another size does not touch the earlier ones.
- **Updating later:** when your world changes, import the new export in Step 1 and press **Retrain with the new export**; it can also make the files and install them for you.

## If something goes wrong
- *"Ollama did not answer"*: start Ollama, or check the address. For another computer, Ollama must listen on the network.
- *"llama.cpp is needed"*: the file could not be made without Unsloth. Install Unsloth in **⚙ Training setup**, or install llama.cpp (<https://github.com/ggml-org/llama.cpp>) and set `LLAMA_CPP_DIR` to its folder. The merged model stays in the export folder.
- *"This llama.cpp has no llama-imatrix"*: the importance matrix was skipped and the files were made normally. Update llama.cpp (llama-imatrix sits next to llama-quantize) if you want it.
- *"The MTP head was left out: …"*: the matrix was made without the speculative-decoding head (an older llama-imatrix without `--nextn`, or it refused). The sizes still got the matrix.
- *"The general text (…) could not be read"*: the note says why: the dataset needs access approval (ask for access and save your Hugging Face token, or choose another), it was not found, it has no usable column, or you are offline. The matrix was made from your world's text only; try again later for a slightly better one.
- *"Ollama refused the request"*: read the reason shown; often the Ollama version is too old for this model family. Update Ollama.
