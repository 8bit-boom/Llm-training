# Compare quants

A smaller GGUF file (Q4_K_M, Q3, a mixed-precision size) rounds the model's numbers, so it answers a little differently from the 16-bit original. This page **measures how much**, so you can pick a size with facts.

## What is measured

llama.cpp's `llama-perplexity` lets the **16-bit model** read a test text once and saves how likely it finds each next word. Every smaller file then reads the same text and is compared with those saved answers. The main figure is the **KL divergence** (KLD): 0 means identical answers, bigger means further away. The table also shows the median, the 99th percentile and the worst case, how often both files pick the **same top word**, and the perplexity.

> KLD is a **proxy**. Unsloth reports it for its quants too, but it does not replace trying the model on your own questions (Step 4 · Test).

## How to use it

1. **The 16-bit reference**: an F16 file from Step 5 (tick the size "F16" there), a file of your own, or let the program make one with llama.cpp (needs the training software and about two bytes per weight of free disk).
2. **The files to compare**: tick the files of your exports (sizes made with and without an importance matrix, and the mixed-precision sizes, all go in one table), or add a path. After an export, the button **Compare these sizes** on Step 5 brings you here with them ticked.
3. **The test text**: by default the **held-out part of your saved data** (questions the training never saw, and not the text any matrix was measured on). Or a text file of your own.
4. **How much text**: 16 chunks of 512 tokens is the default. The page shows how big the temporary file with the 16-bit model's answers will be **before** anything runs (about two bytes per word of the vocabulary for every compared word: a 248,000-word vocabulary needs about 2 GB for 16 chunks), and refuses when your disk is too small.

Press **Stop** at any time: the files that are measured stay measured, and **Continue** finishes the rest without reading the text again (as long as the saved answers are still there). The big file is deleted when the comparison ends unless you keep it.

## Reading the result

- The **table** is sorted by size. A **●** marks the files on the line of the chart: nothing smaller is closer to the 16-bit model.
- The **chart** shows KLD against size. **Lower left is better.**
- The **words** under it say which size keeps the most, whether a matrix helped (and by how many percent), and how a mixed-precision size compares with the plain one nearest in size.

## When something is wrong

Nothing here stops the program. If `llama-perplexity` is missing, the page has an **Install llama.cpp now** button (the program downloads it into its own folder after you have seen the size); `LLAMA_CPP_DIR` tells the program where a copy of your own is, and always wins. A file that cannot be read, or that is not made from the same model (another vocabulary), fails only its own row.
