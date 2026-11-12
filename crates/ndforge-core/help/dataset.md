# Step 2 · Make training data

## What this step does
Turns your world into **practice material** for the model: the lore as readable text (the **world book**) and many **questions with answers** made from your entries, rules, sessions, quests and tables. You can look at every example, drop the ones you do not like and fix answers before you save a version. No AI model is needed for the basic examples. An optional **teacher model** can add better ones (see below).

## What you need before you start
Step 1 done: the world data of your project is imported.

## What to do
1. Look at the **recipe**. The defaults are good for a first try. Each line is one kind of example; untick what you do not want:
   - **World book**: the whole lore as text. Keep it on: it is what lets the model *know* your world.
   - **Who or what is it?**, **Other names**, **How things are connected**, **Details**, **Lists**, **Random tables**, **Sessions and quests**, **Rules sections**, **Player characters**: questions with answers built from your data.
   - **GM secrets and hidden entries**: questions such as "What do the players not know about X?". Your secrets are kept, always labelled *(GM secret: ...)*, because this is a model for you, the GM. Turn this off if you do not want the model to repeat secrets.
2. **Different wordings** (1 to 3): how many ways each question is asked. 2 is a good balance.
3. **Kept for testing** (default 10 %): some entries' questions are set aside and never used for training; Step 4 uses them to check what the model learned from the world book.
4. Press **Make the examples**. It takes a few seconds.
5. **Review**: pick a kind, search, and read the examples. **Drop** removes an example from the training data, **Keep** brings it back, **Edit** changes the answer. Your choices are remembered when you make the examples again later.
6. Press **Save this version**. Training always uses a saved version, so you can later see exactly what a model learned from.

## Better examples with a teacher model (optional)
If your world file came from an nd-world that has made **Better training examples** (Export & Backup), they are already in your list: kinds starting with *Teacher:*, with their ✓ / ⚠ / ✗ marks. They stay when you import a newer world file, together with your Drop / Keep / Edit choices. ("Remove all teacher examples" removes them from the list now; they come back the next time you press *Make the examples* while the world file still contains them.)

After you have made the examples, the page shows **Better examples with a teacher model**. A bigger model, for example Gemma 4 running in Ollama on your TrueNAS, reads your world and writes extra practice:
- **Deeper questions**: why and how questions that need several facts.
- **Rules practice**: a concrete situation and the ruling, with the steps.
- **Campaign history**, **NPC role-play** (for entries that have a speaking style; secrets never reach the teacher for this), **Lore search** (the lore is handed over in the question, the way nd-world does it), **Not in the lore** (questions about things that do not exist; the answer says so), **General chat** (about 10 %, so the model keeps its manners) and **House style** (follows your world's standing AI instructions).

How to use it:
1. Enter the teacher's **address** (for Ollama on the TrueNAS: `http://truenas:11434`) and the **model name** (for example `gemma4:e4b`; it must be downloaded there). Press **Test the connection**.
2. Press **Try on 5 pieces first** and read the results in the Review list (kind names start with *t_*), then **Make with the teacher**. It can take a while; press **Stop** at any time. Every answer is stored, so running again only pays for what is new.
3. Every answer is **checked** by a second pass that compares it with the lore. **✗ not supported** answers are dropped for you (bring one back with *Keep* if you disagree), **⚠ partly supported** ones are kept but marked, **not checked** ones (general chat, house style) are yours to judge.

Good to know: the teacher is a language model and can still be wrong. The check removes a lot, not everything, so read a sample. The questions about things that do not exist are made only from invented names that really appear nowhere in your world. If you tick "free its memory when finished", Ollama unloads the teacher so training can use the graphics card.

## What happens next
Step 3 · Train uses the newest saved version. If you import a newer export of your world, the page tells you to make the examples again; your drops and edits stay.

## Good to know
- Your secrets are in the data on purpose. A model trained on it must not be used for your players' AI chat.
- Examples about the same entry are kept together, either all for training or all for testing.
- More examples is not always better: if an entry has a thin description, its questions are thin too. Improving the entry in nd-world and exporting again helps more than any setting here.

## Export for Unsloth Studio
Under **Save**, every saved version has **Export for Unsloth Studio…**. It writes `train.jsonl` (and `eval.jsonl` when the version holds out test questions) in the *chatml* format (`{"messages": [...]}` per line) that the **Train** page of Unsloth Studio / Unsloth Desktop reads, plus a `README.md` with the settings to choose (format chatml, context length, *Train on Completions* on, language layers only). Studio has no plain-text mode, so the world book is turned into answers to "Tell me what the lore says about …" (untick *Include the world book* to leave it out). Several worlds in one project stay together: each example carries its own world's name.

**Train here or in Studio?** Both work on the same data. This program adds what Studio does not: one place for the world data, teacher examples, a test page that scores the model against the original, stop and continue, and a benchmark. Studio is the official tool of the model's makers, so a new model family works there first and its export is the most tested for models with pictures. A good way to decide: train once in each with the same data and compare the two models on the Test page.

