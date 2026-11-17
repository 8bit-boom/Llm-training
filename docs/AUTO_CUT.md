# Auto-cut (one click): from a big model to a smaller one that still thinks

Auto-cut chains the steps of the Slim page that a person would do by hand: look at the model, measure it, try a few cuts, cut, heal, check, export and (optionally) put the GGUF
files into Unsloth Studio's share. It reuses the existing jobs (`reshape_inspect`, `reshape_measure`, `reshape_try`, `reshape`, the heal jobs, `think_check`, export, imatrix adapt); it
adds a planner, a state machine that can be stopped and continued, gates after every stage, and a report. Nothing in it has run on a real model yet ("Not verified yet" at the end).

This file covers the planner, the orchestrator (state machine, gates, retry and report), the Studio publish and the page.

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
| `sN.check_heal` | thinking check of the healed model against the original's numbers (the gates of docs/THINKING_CHECK.md). A failure is retried, backed off, or ends the run ("Gates, retry and back-off" below) |
| `export` | the GGUF sizes (Q4_K_M and the chosen ones) of the last healed model, its chat template keeping thinking (`thinking_default: "keep"`) |
| `check_gguf` | thinking check of the Q4_K_M GGUF through llama-server (covers the template and the quantisation) |
| `studio_publish` | only when asked: the GGUF files into Studio's share ("Publishing to Unsloth Studio" below) |

Models that do not think (*keep thinking* off) skip the thinking checks and the reasoning text; everything else is the same.

**The teacher's answers are shared.** Every stage heals with the same teacher, tokenizer, text and settings, so the teacher's answers are kept once in `models/_teacher_cache/<id>/` and each
model's own `distill/cache/` only holds its index (which says where the parts are) and its rows. A stage that needs more rows (a retry with more tokens) adds only the rows that are missing;
a stage that needs fewer reuses the start. In the disk timeline the answers therefore count once (the biggest run), not once per stage. Without *keep the intermediate files* the shared answers go
when the **last stage has passed** its check (and the links of the run's models to them), never earlier, and never while any other model, such as a heal started on the Slim page, still uses them.
A shared folder nobody uses is also removed when a model is deleted. See docs/HEAL_SETTINGS.md ("The shared teacher cache").

**Clean-up** (`keep_intermediate` off): after the check of stage N passed, its unhealed cut (`model/` and its entry in the model list) are deleted (its own cache folder is only a link), and the healed
model of stage N-1; the healed model of the last stage is never deleted. Only what the run made is ever deleted (the run keeps the list, `made`, and the real effects refuse anything else), never the
original model, and nothing from a stage that did not pass. The test records every deletion and compares it with the plan's disk timeline.

**Deviations from the plan** (the code won): the state is in `models/_auto/<run>/`, not `models/<name>/auto.json` (a run makes several models); the check of the original runs after the first cut, not
before the first step (the worker finds the original through the cut's record: `think_check` target `original`); the export measures its importance matrix on the finished model instead of adapting a
matrix of the original (`imatrix_adapt` needs a saved matrix of the original, which a run does not have; docs/IMATRIX.md); the export's importance matrix is switched on (it was off in the state machine's first version) and, with *keep it able to think*, measured with reasoning text in the calibration (docs/IMATRIX.md, "Thinking models"), so "the adapted matrix" of the plan is a matrix measured on the cut model with thinking rows; stage A heals with 40 million tokens (the plan said 20 in one place and 40 per
stage in another).

## Gates, retry and back-off (`reshape/auto.rs`, `gate_failed`)

The healed model of every stage is checked against the original's numbers (the gates of docs/THINKING_CHECK.md). `warn` passes, `fail` (any gate missed) goes through three tries per stage:

1. **Heal again**: the healed model is the student and the original is still the teacher; the stage's tokens plus **40 million** (`RETRY_TOKENS_M`). The new healed model is its own folder (`heal-...`).
2. **Back off one step** (`auto_plan::back_off`): one block less in a stage that cuts blocks, else 1,024 more of the MLP width (never wider than the stage before). The stage is made again from the same input
   (cut, check of the cut, heal, check), the stages after it are planned again from the new stop. When the last stage is backed off the final model is larger than asked; the report and a note say so.
   Backing off is refused when nothing is left to give back or the card cannot heal the larger model.
3. **Give up**: status `gates_failed`, `error` names the stage and the gates, the report is written. A run in that status is not continued (start a new one with another size).

The GGUF check is a gate too: a Q4_K_M file that fails although the healed model passed ends the run with `gates_failed` before anything is published (the files stay).

Every try is a line of `stage.attempts` (`heal`, `heal_again`, `back_off`: model, tokens, verdict, failed gates, numbers). The steps of a retry are set back to pending in the saved state, so a stop at any call and a
continue gives the same run here as well (a test stops at every call of a run with both retries).

**Clean-up and retries**: what a failed try made (its unhealed cut, a healed model that was healed again, the first stage's failed attempt) is deleted only when the stage has passed (never with *keep the
intermediate files*, never for a run that gave up). Only what the run made is removed, never the original, never the healed model that passed. The back-off therefore needs room for the failed try next to the
new one; the plan's peak does not include that.

## The report (`reshape/auto_report.rs`)

When a run ends (`done`, `gates_failed`, `failed`) `models/_auto/<run>/auto_report.json` and `auto_report.md` are written (the state keeps only their names in `report`). A report is a function of the saved state, so it can be
made again at any time, also for a stopped run. It holds: the model, the size asked and reached, blocks removed, MLP width, the time (the steps' own seconds added up), the planned disk (not measured), per stage the size, cut,
tokens, KL at the start and the end of the heal, the loss before and after and the teacher's loss, the tries and their verdicts; the thinking-check table (original, every cut, every healed model, the GGUF) with
the gates that were missed; the exported files; where Studio has them; the notes (retries, back-offs, the halves of Try first that disagreed); every step with its status and seconds.

## Publishing to Unsloth Studio (`unsloth/publish.rs`)

The step `studio_publish` (only when asked) copies the GGUF files into Studio's share and says where they are inside Studio's container. The rules, the calls and what is checked are in
docs/UNSLOTH_API_CONTRACT.md, "Publish"; the settings are **Studio share**, **Path inside the container**, *tell Studio about the exports folder* (default on) and *fine-tune handoff* (default off), in
`Settings.studio_share`. In short: `.partial` folder then rename, a free-space check first, the container path from the settings or from `outputs_dir` of `GET /api/models/checkpoints` (with an API key), the share
confirmed against Studio's run folders, `POST /api/models/scan-folders` only when `GET` says 200. The folder is named `<model> cut to <size>B`. Not set up, the drive missing or the wrong folder is a note on the step, not a failure
of the run; a full disk is a failure that Continue retries. Files are never written outside `<share>/exports`.

Deviations: Studio answers 201 (not 400) for a folder it already has, so there is no "duplicate" case (a 400 containing "already" is still taken as success); under an API key Studio hides the paths of its folder list, so
the registration is not checked against that list.

## The page (`ui/app.js`, "Auto-cut (one click)")

A card at the top of the Slim page. It shows: a size slider (3 B or 30 % of the model up to 95 %, steps of 0.5 B; the default is 76 %), *Keep it able to think* (on for Qwen3.5 / 3.8, off and disabled otherwise), one or two stages,
Full or Quick, *keep the intermediate files*, the confirmation of more than 4 blocks (only when the plan says it is needed), and the Studio publish box with the share fields (Save and Check). Under them the plan: blockers (red, Start
disabled), warnings, the stages (size, cut, healing tokens and hours, card) and the disk. Start / Stop / Continue / Show the report, then a **stepper** of the run's steps (done, running, pending, skipped, failed, with the note of the step), the live text
and bar of the running job, and the report as text.

Tauri commands (`src-tauri/src/lib.rs`): `auto_plan(spec, inspect)` (the planner for the model description the page already has; no job runs), `auto_start(spec, resume)` (a new run, or `resume` = run id; one job at a time, the
same busy flag as the other jobs), `auto_stop` (the Stop flag of the other jobs), `auto_status` (the newest run's state, whether a job runs, the saved Studio share; a run left "running" by a closed program is marked stopped here),
`auto_report(run)` (data and Markdown), `studio_share_set` and `studio_share_check` (the share settings, and the comparison with Studio's outputs folder). Events: `reshape-event` with `phase: "auto"`; the page reloads the state
file at every step event, so the stepper is what the file says. Tests: `ui/smoke/autocut.test.mjs` (the card, the plan request, start / stop / continue, the stepper, the end states, the share), `tests/auto.rs` (`preview`, `report_of`).

## Not verified yet

Nothing here ran on a real model, card or Studio (the publish ran against a stand-in server and temporary folders only). The planner is arithmetic: the sizes equal the Slim page's, the timeline is a sum, the card estimates are the Heal card's estimates. The state machine
is tested with a stand-in for every job; `RealEffects` (the wiring to the real jobs, including `publish`, which reads the global settings) has only been compiled and read, not run against a worker; the page was only run in a stand-in browser, never in the window.

The Auto-cut card says how long the rest takes and when it will be done: "about 3 h 20 min to 20 h left (done around 22:09–tomorrow 14:49)". Durations are rounded to 5 minutes below 10 hours and to 30 minutes from 10 hours up, so they do not claim false precision. An end of the range that falls on another calendar day names it ("tomorrow", or a short weekday name two or more days on), so the high end never reads as earlier than the low end. A finished run shows nothing.
