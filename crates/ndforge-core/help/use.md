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
   Press **Make the files**. This joins your trained add-on into the model and makes a **GGUF** file, a **Modelfile** and a short model card in `exports/`.
   - **Make** chooses what comes out: *GGUF + Modelfile* (for Ollama, the default), *GGUF + the merged model* (also keeps the add-on baked into the model as ordinary safetensors files, 10 to 20 GB more) or *Only the merged model* (no GGUF; it loads in Unsloth, Transformers, vLLM and so on without the add-on, but Ollama cannot use it).
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
- *"Ollama refused the request"*: read the reason shown; often the Ollama version is too old for this model family. Update Ollama.
