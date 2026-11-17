# Auto-cut (one click): from a big model to a smaller one that still thinks

Auto-cut chains the steps of the Slim page that a person would do by hand: look at the model, measure it, try a few cuts, cut, heal, check, export and (optionally) put the GGUF
files into Unsloth Studio's share. It reuses the existing jobs (`reshape_inspect`, `reshape_measure`, `reshape_try`, `reshape`, the heal jobs, `think_check`, export, imatrix adapt); it
adds a planner, a state machine that can be stopped and continued, gates after every stage, and a report. Nothing in it has run on a real model yet ("Not verified yet" at the end).

This file is written in steps: the planner, the orchestrator (state machine, then gates, retry and report), the Studio publish and the page.

## The planner (`crates/ndforge-core/src/reshape/auto_plan.rs`)

Pure functions, no files and no network: `plan(inspect, spec, vram_gb, free_gb) -> AutoPlan`. The exact sizes come from the model's dimensions (`plan::width_params`, the formula the
Slim page uses, tested against the worker's table), not from an estimate. Tests: `crates/ndforge-core/tests/auto_plan.rs`, with the real Qwen3.5 / 3.8 27B config
(`worker/tests/data/qwen3_5_27b_config.json`).

**What is asked** (`AutoSpec`): the model, the size to aim for in billions of parameters, *keep thinking* (default on; refused for a model that is not Qwen3.5 / 3.8), one or two stages (default
two), Quick or Full for Try first, keep the intermediate files (default no), the owner's confirmation of more than four blocks, the export sizes (Q4_K_M is always made) and Studio publish.

**Which cut** (`candidates`, `pick`): width before depth. For every number of whole blocks (0 up to half of them) the MLP width that comes nearest the target is taken, never thinner
than **61.7 % of the original** (a cut of at most 38 %: 17,408 -> 10,752 for the 27B; `WIDTH_FIRST_KEEP`). A cut is *on target* within 4 %. The list is ordered: on target first, then fewer
blocks, then nearer. More than 4 blocks need the owner's confirmation (`needs_confirmation`); without it the planner stops with a plain sentence. The first three on target are what *Try
first* is asked about. A model that cannot be thinned (not Qwen3.5 / 3.8) is cut by blocks only.

For the 27B: 21.1 B is the MLP alone (10,752, exactly 21.13 B); 18 B is blocks plus the thinnest MLP; 14 B needs more than 4 blocks and waits for the confirmation; 3 B is refused.

**Stages** (`split_stages`): two by default. Blocks and a thinner MLP: the blocks in stage A, the width in stage B. Width only: stage A takes the half-way stop (17,408 -> 14,080 ->
10,752), stage B the rest. Blocks only: half of them in each stage. A cut that cannot be divided (one block, one MLP stop) is one stage, with a note. The teacher is always the original.

**Healing per stage**: the thinking recipe's numbers (`runs::heal_choice` with `recipe: thinking`: sequence 8192 or 4096 by the card, batch 1, rank 64) and **40 million tokens** for stage A;
the last stage 40 or **80 million** (`plan::thinking_tokens_m`, 80 when the cut is over 30 %). Without *keep thinking* the plain defaults (20 million from 15 B). The estimate of the card
(`heal_plan_with`) is made per stage on the student's own layer count; "over" blocks the run, "tight" is a warning.

**Disk** (`disk_plan`): a timeline. The original stays; every stage adds its unhealed cut, the teacher's saved answers (196 bytes per token) and its healed model; the export adds a 16-bit GGUF
and the chosen sizes. Without *keep the intermediate files* the cut and the answers of a stage go once the stage passed its checks, the healed model of the stage before once the next one
passed, and the 16-bit GGUF after the export. The plan shows the peak (and at which step), what is left at the end, and refuses to start when the free space is under the peak plus 5 %.
For the 27B to 21.1 B in two stages (stage A 24.46 B at MLP 14,080, stage B 21.14 B at 10,752, 40 million tokens each) the peak is **196.8 GB** with the clean-up (while stage B's healed model is
written) and **308.7 GB** without it (at the export); 110.7 GB are left at the end with the clean-up. They are sums of file sizes, not measured.

**Backing off** (`back_off`): the step Auto-cut takes back after a stage failed twice: one block less while there are blocks, else 1,024 more of the MLP (four stops).

## The orchestrator (`reshape/auto.rs`, effects in `reshape/auto_fx.rs`)

A state machine, `auto::start(project, effects, spec)` / `auto::resume(project, effects, run)`. Its state is **`models/_auto/<run>/auto.json`**, written atomically after every step (the folder
name starts with an underscore like `_scores`, so the model list does not show it). A run is `running | stopped | failed | done | gates_failed`; a run that was `running` when the program
closed is marked `stopped` at the next start (`mark_interrupted`), and **Continue** runs the first step that is not done. Every step looks at the state first (a cut that was made is not made
again; the first answer of Try first is kept between its two requests), so *stopping at any call and continuing gives the same run as one that was never stopped*: the test stops at every one
of the calls of a whole run and compares the finished calls and the saved state.

What the steps do is behind the `Effects` trait (the tests use a stand-in that only records; `RealEffects` calls the jobs of the Slim page, the Heal card, the Thinking check and the export, one at
a time on one card, with the Stop button on the same flag). Worker events are forwarded unchanged with `phase: "auto"` and `auto_step` added; the run's own events are
`{"kind": "auto", "phase": "auto", "auto_step", "status": "running|done|skipped|stopped|failed", "note"}`.

| Step | What it does |
|---|---|
| `plan` | the model's layout (`reshape_inspect`), the training data, the card and the disk; the planner above. A blocker ends the run before any job starts, with the numbers in the message |
| `measure` | `reshape_measure` of the original on the reasoning-aware text (docs/RESHAPE.md, "Measuring on reasoning"): 256 samples, loaded the way the card allows |
| `try` | Try first for the (up to three) candidates: first the widths and the best run of every block count, then the combined cut (those blocks and that width). The cheapest wins; when the two halves of the text name different winners, everything is asked again with 256 samples. The blocks of the winner are cut as Try first found them (explicit layers) |
| `sN.cut` | `reshape` of the stage (the input is the original, or the healed model of the stage before; a width is measured on THAT model, on the reasoning-aware text) |
| `sN.check_cut` | thinking check of the cut (information only). The first one also checks the original, once, from the folder of the first cut (that is where its record of the original is) |
| `sN.heal` | the Heal card's job with the thinking recipe and the stage's tokens; **the teacher is always the original model** (`heal_from_source` with the original as the teacher, also for stage 2) |
| `sN.check_heal` | thinking check of the healed model against the original's numbers (the gates of docs/THINKING_CHECK.md). A failure ends the run with `gates_failed` for now |
| `export` | the GGUF sizes (Q4_K_M and the chosen ones) of the last healed model, its chat template keeping thinking (`thinking_default: "keep"`) |
| `check_gguf` | thinking check of the Q4_K_M GGUF through llama-server (covers the template and the quantisation) |
| `studio_publish` | only when asked (docs/UNSLOTH_API_CONTRACT.md, "Publish") |

Models that do not think (*keep thinking* off) skip the thinking checks and the reasoning text; everything else is the same.

**Clean-up** (`keep_intermediate` off): after the check of stage N passed, its unhealed cut (`model/` and its entry in the model list) and the teacher's saved answers are deleted, and the healed
model of stage N-1; the healed model of the last stage is never deleted. Only what the run made is ever deleted (the run keeps the list, `made`, and the real effects refuse anything else), never the
original model, and nothing from a stage that did not pass. The test records every deletion and compares it with the plan's disk timeline.

**Deviations from the plan** (the code won): the state is in `models/_auto/<run>/`, not `models/<name>/auto.json` (a run makes several models); the check of the original runs after the first cut, not
before the first step (the worker finds the original through the cut's record: `think_check` target `original`); the export measures its importance matrix on the finished model instead of adapting a
matrix of the original (`imatrix_adapt` needs a saved matrix of the original, which a run does not have; docs/IMATRIX.md); stage A heals with 40 million tokens (the plan said 20 in one place and 40 per
stage in another).

## Not verified yet

Nothing here ran on a real model, card or Studio. The planner is arithmetic: the sizes equal the Slim page's, the timeline is a sum, the card estimates are the Heal card's estimates. The state machine
is tested with a stand-in for every job; `RealEffects` (the wiring to the real jobs) has only been compiled and read, not run against a worker.
