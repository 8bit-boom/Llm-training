# Step 4 · Test

## What this step does
Checks whether the trained model really knows your world better than the original. The same questions are put to **both** models and the answers are scored. The questions are the ones set aside in Step 2 (**Kept for testing**): the world book contained their facts, but training never saw them as questions.

## What you need before you start
A finished run from Step 3 and the training software (Training setup). If you trained on the TrueNAS server, the result was already copied back to this computer.

## What to do
1. Choose a run and press **Test this run**. Loading the two models and answering about 60 questions takes a few minutes on the graphics card. **Stop** ends it.
2. Read the verdict at the top, for example *"Better than the original on 86 % of lore questions"*, and the table:
   - **Lore questions** and **Rules questions**: how much of the expected answer the model says. By default this counts matching words, which is a rough guide. Tick **Grade with the teacher** (set up in Step 2) for a fairer grade.
   - **Invents answers about things that do not exist**: made-up names are asked about. A good model says it is not in the lore. **Lower is better.**
   - **General chat sensible**: ordinary game-master questions. If this drops a lot, the model forgot how to talk: train fewer steps or add general chat examples.
3. Open **Read the answers**. The scores are only a guide: your own reading is the real test.
4. **Ask your own question**: both models answer it side by side. **Save this chat as a file** keeps it in the project's *transcripts* folder.
5. **A run on a slimmed or grown model** (made on the *Slim or grow a model* page) has a **Compare with** choice: *the slimmed / grown model it was trained from* (the default) or *the original* that model was made from. The second shows whether training won back what slimming cost. Big models are loaded in 4-bit so that both fit the card; the result says so.
6. **Compare runs** shows every tested run in one table (for example LoRA against full fine-tuning); the best lore score has a star. Slimmed or grown models that were tested on their own page appear there too (*model:…*, type *slim* or *grow*).

## What happens next
Step 5 · Use puts the model you like into Ollama for nd-world.

## Good to know
- A run is tested on a fixed set of questions, so two runs on the same data version can be compared. Runs graded by matching words and runs graded by the teacher cannot.
- If "Kept for testing" was 0 % in Step 2 there are no questions: set it above 0, make the examples again, save a new version and train again.
- The original model answers from the same weights with the trained layers switched off (LoRA) or from a fresh copy (full fine-tune), so the comparison is fair.
