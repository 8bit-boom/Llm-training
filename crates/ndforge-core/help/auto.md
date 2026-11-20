# Auto (Auto-cut, one click)

This page does the whole Slim page for you: **measure** the model on reasoning text, **try** a few cuts, **cut** and **heal** in two stages (the original is always the teacher), **check that it still thinks** after every stage, and **export** the GGUF files.

- Move the slider to the size you want (below about 20 B a model that thinks gets clearly weaker; the card says so). The plan under it shows the stages, the healing time, the card and the disk it needs; Start stays off while the plan has a red line.
- **Keep it able to think** (Qwen3.5 / 3.8) uses reasoning examples for measuring and healing, checks the answers, and keeps the thinking template in the GGUF.
- If a healed model fails the thinking check, Auto-cut heals it again (the healed model learns on 80 million tokens in this second heal, 40 million more of the original's answers), then backs the cut off one step and does that stage again; after that it stops and says why. Nothing is deleted from a stage that did not pass.
- **How careful, how fast**: Thorough is the tested recipe. Balanced (chosen for a big model) and Fast give up quality risk for hours: fewer heal tokens, one sequence length in both stages so the original writes its answers once, quick checks, no retry. Each shows its estimated total for your model, each knob shows the hours it saves, and everything below the recipe is listed in the plan and the report. "I have at most N hours" gives up the least harmful settings first, or tells you it cannot be done. A model that thinks very likely loses its reasoning on Fast.
- **Stop** ends the current step safely; **Continue** goes on where it was, also after you closed the program. It takes many hours.
- "Put the GGUF files into Unsloth Studio's share" needs the folder Studio keeps its files in (as this computer sees it) and the same folder inside Studio's container; with an Unsloth API key (Training setup) the container path is found out for you.
- **Show the report** gives the numbers of the whole run. Only the intermediate files that Auto-cut itself made are ever deleted; never your original model.
