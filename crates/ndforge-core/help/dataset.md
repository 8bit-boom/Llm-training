# Step 2 · Make training data

## What this step does
Turns your world into **practice material** for the model: the lore as readable text (the **world book**) and many **questions with answers** made from your entries, rules, sessions, quests and tables. You can look at every example, drop the ones you do not like and fix answers before you save a version. No AI model is needed for this step.

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

## What happens next
Step 3 · Train uses the newest saved version. If you import a newer export of your world, the page tells you to make the examples again; your drops and edits stay.

## Good to know
- Your secrets are in the data on purpose. A model trained on it must not be used for your players' AI chat.
- Examples about the same entry are kept together, either all for training or all for testing.
- More examples is not always better: if an entry has a thin description, its questions are thin too. Improving the entry in nd-world and exporting again helps more than any setting here.
