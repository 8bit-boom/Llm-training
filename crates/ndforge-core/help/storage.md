# Storage

## What this page does
It shows **where the disk went** and lets you delete what you do not need. Models, exports, the original model's saved answers (the teacher's cache), checkpoints and downloaded models are big: one Auto-cut of a 27 B model can need about 220 GB at its peak.

- The list is made in the background. Big folders take a while, so the rows appear first and the sizes follow ("size not measured yet").
- **Safe to delete** (at the top) lists only the steps *between* your models and files: the 16-bit GGUF when the smaller files exist, the merged model when the GGUF files exist, a cut before it was healed, the teacher's answers and checkpoints of a finished run, leftovers of failed jobs, the 16-bit answers of a quant comparison, old logs. **Nothing is ticked for you.** The button *After a finished Auto-cut keep only the final healed model and the GGUF files* ticks those for you to look at.
- Each row says what it came from and has a chip: **in use** (a job or a run you can continue needs it: it cannot be deleted), **referenced** (a report or an export points to it: it can be deleted, the link then stops working) or **safe**.

## Deleting
1. Tick the rows. The bar shows how much it frees and how the free space on the drive changes.
2. Press *Delete the ticked items…*. The window lists exactly what will go, with its size and place.
3. **Deleting is permanent.** The files are not moved to the Recycle Bin (a bin would not free the space).
4. For a model or more than 5 GB you have to type the word **DELETE**.
5. *Cancel the deletion* stops it between two files. The result lists what was freed and what could not be deleted (on Windows a file that another program has open: close that program and delete again).

## What is never deleted
Your project folder, your data and settings, and any model outside this program's folders (for example the original model on another drive: the program only reads it). Links are never followed: a link is removed itself, never what it points to. Nothing is deleted while a job runs, and nothing that a stopped or failed Auto-cut run still needs to go on.

## Good to know
- The teacher's saved answers: deleting them makes the original answer your texts **again** the next time a model is healed (hours).
- The Hugging Face cache can be shared with other programs. A model this program does not know may be used by another program.
- The vLLM engine volume holds the models the engine downloaded. Its size is not available here; it is refused while an engine container exists.
