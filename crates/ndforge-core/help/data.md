# Step 1 · Get world data

## What this step does
Brings your world into the program as a **training file** (a `.ndtrain` file made by nd-world). Everything the model learns
comes from this file: your entities (characters, places, items ...), the rules, session recaps, facts, quests and random tables.

## What you need before you start
A training file from nd-world. If you do not have one yet:

1. Open nd-world in your browser and log in as the GM.
2. Select the world you want to teach.
3. Open **Tools → 📦 Export & Backup**.
4. Click **Training bundle (for ND Lore Forge)**. A file named `<world>.ndtrain` is downloaded.
5. Copy that file to this computer if nd-world runs on another one (for example your TrueNAS server).

*The file contains all of the world's text, including GM secrets, but no pictures, no player accounts and no passwords. Keep it
as private as you would keep a backup.*

## Several worlds in one project
Choose the training file of **another world** the same way: the program offers to *add* it, and the model is then trained on all the worlds at once. Every example carries the name of its own world, so the model learns which world a question belongs to (and a name that exists in one world only is not mixed into the other). After adding or removing a world, press *Make the examples again* in Step 2; your drops and edits stay. A newer file of a world that is already there updates that world only. The trained model is named after the project when it holds more than one world.

## What to do
1. Click **Choose training file...** or drag the file onto this page.
2. The program checks the file and shows what is inside, for example *312 entities, 27 session recaps, 140 facts*.
3. Check that the numbers look right for your world. Then continue with **Step 2**.

## Updating later
When your world has changed, export a new training file the same way and drop it here. The program shows **what changed**
(new, changed and removed items) and keeps the older file too.

## If something goes wrong
- *"This file is not a training bundle"*: you picked another kind of file. Export it again from the **Training bundle** button.
- *"damaged (checksum does not match)"*: the copy was interrupted. Copy or download the file again.
- *"for the world X, but this project is for Y"*: use **Create a new project** for the other world.
- Your nd-world has no **Training bundle** button: it is too old. Update nd-world. (The older **World JSON** export contains
  fewer things and is not supported yet.)

## How long does it take?
A few seconds, even for a big world.
