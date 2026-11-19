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

For the 27B: 21.1 B is the MLP alone (11,008, exactly 21.07 B; a width cut removes the multi-token head, so every width size is about 0.3-0.4 B lower than it was when the head was kept); 18 B is blocks plus the thinnest MLP; 14 B needs more than 4 blocks and waits for the confirmation; 3 B is refused.

**Stages** (`split_stages`): two by default. Blocks and a thinner MLP: the blocks in stage A, the width in stage B. Width only: stage A takes the half-way stop (17,408 -> 14,080 ->
11,008), stage B the rest. Blocks only: half of them in each stage. A cut that cannot be divided (one block, one MLP stop) is one stage, with a note. The teacher is always the original.

**Healing per stage**: the thinking recipe's numbers (`runs::heal_choice` with `recipe: thinking`: sequence 8192 or 4096 by the card, batch 1, rank 64) and **40 million tokens** for stage A;
the last stage 40 or **80 million** (`plan::thinking_tokens_m`, 80 when the cut is over 30 %). Without *keep thinking* the plain defaults (20 million from 15 B). The estimate of the card
(`heal_plan_with`) is made per stage on the student's own layer count; "over" blocks the run, "tight" is a warning.

**Disk** (`disk_plan`): a timeline. The original stays; every stage adds its unhealed cut, the teacher's saved answers (196 bytes per token) and its healed model; the export adds a 16-bit GGUF
and the chosen sizes. Without *keep the intermediate files* the cut and the answers of a stage go once the stage passed its checks, the healed model of the stage before once the next one
passed, and the 16-bit GGUF after the export. The plan shows the peak (and at which step), what is left at the end, and refuses to start when the free space is under the peak plus 5 %.
For the 27B to 21.1 B in two stages (stage A 24.09 B at MLP 14,080, stage B 21.07 B at 11,008, 40 million tokens each) the peak is **195.8 GB** with the clean-up (while stage B's healed model is
written) and **306.7 GB** without it (at the export); 110.5 GB are left at the end with the clean-up. They are sums of file sizes, not measured.

**Backing off** (`back_off`): the step Auto-cut takes back after a stage failed twice: one block less while there are blocks, else 1,024 more of the MLP (four stops).

## The thinking check before healing (`think_gate`)

**Why.** A real run (docs/REAL_RUN_17B.md) cut a 27 B model to about 17 B. The cut with the lowest text-prediction error finished its reasoning in only half of the answers (the original: 97 %),
and the error was even *lower* than the original's. Try first measures how well a cut predicts text, not whether the model can still think, so Auto-cut checks the thinking of its best cuts
before it spends hours healing one.

**What it does** (step `think_gate`, after `try`, only with *keep thinking* and *think the best N cuts before healing* on; N is 2 by default, 1 to 3):

1. The N cuts of Try first with the lowest loss are made one after the other as the FINAL cut (the same job as a stage, `skip_check`: Try first measured them already).
2. The original is checked once (from the folder of the first cut), then each cut. The checked numbers are the same 16 or 32 questions as everywhere (docs/THINKING_CHECK.md).
3. Only the best cut so far stays on the disk. The ranking is `thinkcheck::rank_thinking`: the score is *finishes its reasoning* minus *loops* minus *garbage*; the highest score wins, and scores within
   0.125 of the best are a tie that the lower loss decides (the check asks few questions, so a smaller difference is noise). A cut without a score only counts when no cut has one.
4. When the winner is not the cut Try first chose, the stages are planned again for it, and the notes say so with the numbers: "Try first's lowest loss was 1.267 (3 blocks, MLP 10240), but it kept
   less of its thinking (closed 50 %, loops 50 %); 2 blocks, MLP 12032 was chosen (closed 97 %, loops 0 %)."
5. With **one stage** the winner's cut and its check are the stage's own (`s1.cut` and `s1.check_cut` are skipped: "made and checked in the thinking gate") and its loss is Try first's. With
   **two stages** the checked cuts are deleted (stage 1 makes its own, half-way cut) unless *keep the intermediate files* is on.
6. **It never halts a run.** If even the best cut finished its reasoning in under 0.6 of the original's rate, a warning with the numbers is added ("Even the best cut has lost much of its
   ability to finish its reasoning before healing (closed 50 % against 97 %). Healing may bring some back; the healed model's check decides."). The healed model's own check still gates the run.

Stop and Continue repeat nothing: every checked cut, the cut that was being checked and the original's numbers are in the state (`think_gate`, `baseline`), and a run saved before the gate existed has no
gate step and loads and runs as before.

**Time and disk.** The plan shows `think_gate_minutes`: for each cut writing it (about 500 MB per second) plus the check (15 minutes for Full, 10 for Quick, per 17 B of the model, at least once),
plus one check of the original. It is an estimate. On the disk the gate keeps one final cut at a time (two with one stage: the best so far and the next one); the plan has a line for it and uses it as the
peak when it is the largest. `step_try` also notes when the best cut's loss is below the original's (measure both the same way).

**The report** (`auto_report.json` key `think_gate`, markdown "Before healing") has the table of the checked cuts (loss, finishes reasoning, loops, right answers, chosen) and, when the chosen cut
goes beyond the guardrails (`beyond_guardrails`: the MLP kept below 61.7 % or more than 4 blocks, as with *allow many blocks*), a warning with those numbers and the cut's own thinking numbers.

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
| `think_gate` | the thinking check before healing (the section above): the best cuts of `try` are made and checked, the best thinker is chosen, the stages are planned again when it is not Try first's choice |
| `sN.cut` (skip_check) | when stage 1 makes exactly the cut Try first measured (one stage, or the plan's last cut with its saved blocks), the job is told `skip_check`: the new model is not loaded again to measure its loss (and, for blocks only, nor is the original). The stage keeps Try first's loss as `cut_loss`, and the report says "measured in Try first". A later stage's cut is measured on the healed model as before |
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

## A target size: the measured file (`target_gb`)

Optional (`AutoSpec.target_gb`, the field "Also make a file of about __ GB"). Without it nothing changes. With it the export makes Q4_K_M, the plain sizes just below and just above the target (from the table of bytes per parameter), the measured file `mixed_measured` (docs/MIXED_QUANTS.md, "Fit to a size (measured)", measured on the healed model with the matrix of the export, which was measured on text with thinking when the model keeps thinking) and `f16`, the reference. A new step `compare_quants` (after `check_gguf`, before `studio_publish`) runs Compare quants on them: the test text is the project's saved data when there is any, else general text (seed 1234). The result (`compare` in the state and the report: a table of file, GB, mean KLD with its plus-minus, same top answer and promotion) says whether the measured file is **promoted**: only then is it published to Unsloth Studio; otherwise it stays in the exports and the report says so. The 16-bit file is deleted after the comparison unless `keep_intermediate` (or the person chose `f16`); the plan counts it, 2 GB for the sample file and a guess of the minutes. A comparison that cannot be made does not lose the files: it is noted, the 16-bit file is kept so that Compare quants can be run by hand, and the measured file is not published. Stop and Continue inside the comparison make that call again and nothing else. An `auto.json` from before has none of these fields and loads as before. The caveats of the predicted error apply: the comparison is the judge.

## How long it takes (`reshape/auto_time.rs`, `timing::History`)

Before Start the card shows **How long this will take**: the whole run, step by step, as a range. `AutoPlan.time` (additive, `#[serde(default)]`) is a `TimeEstimate`: `steps` (`StepTime` with `id`, `label`,
`minutes` (low, high), `basis` = `your_history` or `formula`, a note and `parts`), `total_h` (the sum of the steps), `history_steps` and `step_count`, the `basis` sentence, `lever_h`, `notes` and `extra`. The step ids are
those of the run's own steps (`plan`, `measure`, `try`, `think_gate`, `sN.cut`, `sN.check_cut`, `sN.heal`, `sN.check_heal`, `export`, `check_gguf`, `compare_quants`, `studio_publish`), so the same list serves the
window before the run, the "time left" during it and the report after it. A step that is switched off (no thinking, no gate, no target size, no Studio) is not in the list. The report is written in seconds and is not a step.

**Where the numbers come from, in this order**

1. *Your own history.* `timing::History::recent` reads the newest 60 saved job logs (`logs/jobs/*.log`) and keeps, per kind of work, the rate of each job: the teacher's answers (tokens per second x billions of parameters
   of the teacher, from `teacher_pass` and the size of the loaded folder), the training (`train`, with the student's size taken from the merge job that follows it), measuring (`score`) and Try first (`try_pass`) the same
   way, the merge (seconds per GB, `merge_total`), writing a cut (`write`), loading a model (GB per second by load mode, `load_*`) and the thinking check (`think_generate`, seconds per billion parameters, Quick and Full apart).
   The median of the newest five samples of the same size class (0.5 to 2 times the new size; when there is none that close, all of them scaled linearly) is used, with 15 % below and 25 % above as the range.
   A step is tagged `your_history` when the parts that take most of it (the start of the process does not count) came from it.
2. *A formula* built on the reference run below, tagged `formula`. A formula is never shown as a measurement; the basis line says how many steps used your history ("3 of 14 steps use your own earlier runs; the other 11
   are formulas ...", or "None of the 14 steps ..." without any).

Not read from the logs because the worker's records do not carry a size to scale by: the 16-bit conversion and quantising of the export, the thinking check of the GGUF (llama.cpp), the copy to Studio. They are formulas until
the worker records the bytes next to them.

**The reference numbers** (one real run: RTX 5090 with 32 GB, 94 GB of RAM, Windows; a 17.4 B student cut from a 27 B teacher)

| Part | Range | Scales with |
| --- | --- | --- |
| the teacher's answers | 1.0 to 2.5 h per 5 M tokens | tokens, the teacher's size (27 B = 1) |
| the student's training | 1.4 to 2.2 h per 5 M tokens | tokens, the student's size (17.4 B = 1); with the original next to the student ("online") divided by 0.6 |
| the merge | 0.3 to 1.2 min per GB of the healed model | size (the real run: about 35 min for 34 GB; 0.3 is the low end because the arithmetic is faster now) |
| a thinking check | 15 min Full / 10 min Quick per 17 B, at least once (`auto_plan::think_check_only_minutes`, the one formula the plan's own check times use too), x 0.8 to 1.4 | size |
| writing a cut | 0.3 to 0.5 GB per second | size |
| loading a model | 16-bit 0.6 to 1.5 GB/s, 4-bit 0.3 to 1.0, spread over card and memory 0.2 to 0.5 | size |

Guesses (named as such): measuring 0.35 to 1.0 min per billion parameters for 256 texts (x4 when the model is spread over card and memory); a Try-first pass 80,000 to 40,000 tokens x billions per second (the high end
of Try counts one repeat with 256 samples); the 16-bit conversion 0.1 to 0.2 min per GB; each size 0.1 to 0.3 min per GB of the 16-bit file; the measured fit 2 + 0.25 min per billion (0.8 to 1.6 x); the comparison
4 + 3 minutes per file (0.8 to 1.5 x); the copy 0.2 to 1.0 GB/s; 0.5 to 1.0 min for every worker job to start. `think_gate` is the cuts it checks (writing, loading, checking each) plus the original once; without the
check before healing the original is checked once inside `s1.check_cut` instead.

The teacher's answers and the training are most of the run; on the owner's case (27 B to 17.4 B, two stages, 40 M and 80 M tokens) the reference numbers make it a matter of days, more than the heal hours the stage
table shows (`StagePlan.hours` is an older, kinder formula and is unchanged). Once there are earlier runs in the logs the estimate follows them instead.

**Not in the total.** Waiting in the queue; downloading the base model; the computer going to sleep (Windows: switch sleep off for a long run); and a retry. When a thinking check fails the last stage is healed again
(`extra`, id `retry_heal`, shown on its own line "Possible extra, not counted above" and only with Keep thinking, and not at all when the retry is switched off). It happens at most once per stage. **The retry is the
healed model learning on 80 M tokens**, not 40 M more: the student is the healed model and it trains on ALL the tokens of the second heal (the stage's own plus 40 M, rounded up to a budget the heal takes: 40 gives
80, 20 gives 80, 80 stays 80 (it used to fail the heal's own check of the budget)). Only the original's answers are extended, by the difference (a stage on 40 M asks for 40 M more); the estimate counts exactly that.

**The original's answers are written once per sequence length.** The saved answers are kept per teacher, tokenizer, `k`, sequence length (`max_len`) and way of loading the original (`distilljob.shared_store_id`);
a later stage on the same ones finds the earlier stage's rows and writes only the tokens it has more. The estimate knows: when both stages heal on the same sequence length (it is the recipe's own choice per
student size and card: 4096 for both on a 32 GB card for a 19.7 B student after a 26 B one, but 4096 then 8192 on a 36 GB card) the second stage's "the original writes its answers" part is the tokens it has more
(none when both heal on 40 M), else the original writes them again and the step's note says why. The knob "same sequence length in both stages" (`share_teacher_cache`) makes the second case the first by healing
every stage on the shortest sequence length any of them would use and sending it to the heal (`StagePlan.seq_pinned`, `HealSpec.seq`).

**The window.** The block shows the total as hours and minutes (to 5 minutes below 10 h, to 30 minutes from 10 h, like the time left), "if you press Start now it should be done around <clock range, with
"tomorrow" or the weekday>", a collapsible table of every step (range, from your history / formula; the heal shows its teacher, training and merge), the basis line, a one-line caveat and, when the teacher's answers and
the training are more than half of it, the hint of the biggest lever ("fewer tokens or one stage shortens it"). It is asked again whenever the size, the stages, Quick / Full, the thinking check, the target size or any
other choice changes. During a run the "time left" is the sum of the steps not done yet from the same list (a running step has its elapsed time taken off; the training's own estimate replaces the heal's range, plus its
merge). A run saved before the estimate existed keeps the older rule (the stage hours plus fixed allowances).

**Compared with what happened.** The report (`auto_report.json` / `.md`) has, additively, `estimated_minutes`, `actual_minutes` and `estimate_basis` on every step and an `estimate` object (`total_h`, `actual_h`,
`basis`, per step `result`: within / faster / slower than the range) and a section "Estimated and actual time", so that the formulas can be tuned from real runs. Nothing here has been checked against a real run yet.

## Speed settings (`reshape/auto_speed.rs`, `auto_speed.json`)

Healing is most of a run (the teacher's answers and the student's training), so the card has one choice of how careful and how fast, with the estimated total of each for the model and card in front of you.
**Rule: legacy = Thorough.** A spec with every speed field at its default (an older `auto.json`, a spec from an older window) is the recommended recipe and plans and runs exactly as before.

| Preset | What it sets | Risk |
|---|---|---|
| Thorough (recommended recipe) | two stages, the first on 40 M tokens, the last on the recipe's 40 or 80 M, full thinking checks (32 questions), a check of every cut, the best 2 cuts checked before healing, a retry and a back-off | none beyond the cut itself |
| Balanced (chosen for a model of 15 B or more) | the first stage on 20 M tokens, the same sequence length in both stages, quick thinking checks (16 questions), a quick Try first (64 texts) | the first stage learns half as much, the quick checks are noisier; healing below the recipe has failed the thinking checks in a real run, the checks will tell |
| Fast (quick look) | one stage on 20 M tokens, quick checks, the best cut only before healing, no check of the cut, no retry, no back-off, no importance matrix | far below the recipe: a model that thinks will very likely lose its reasoning (a red warning for such a model) |
| Custom | any change in the knobs below | each value below the recipe is listed |

The presets are data (`reshape/auto_speed.json`; `AutoSpec::with_preset`); the window holds the same table (`AC_SPEED_PRESETS`, a smoke test compares them). Any change of a knob turns the choice to Custom; setting
the knobs back to a preset's values makes it that preset again. The values (all `#[serde(default)]` on `AutoSpec`): `preset`, `stage_a_tokens_m` / `last_tokens_m` (None = the recipe; 10 / 20 / 40 / 80),
`think_quick` (None = follows `quick`), `check_cut`, `retry_heal`, `back_off` (default on), `share_teacher_cache` (default off), `imatrix` (default on), `max_hours`.

| Knob | Choices | What it saves | What it gives up |
|---|---|---|---|
| heal tokens of the last stage | 10 / 20 / 40 / 80 M (recipe 40, or 80 above 30 % cut) | linear: every 10 M is about 1.8 to 3.3 h of the student plus the teacher's answers (the window shows each choice's hours) | learns less; below the recipe thinking has failed the checks |
| heal tokens of the first stage | the same (recipe 40) | the same, for the first stage | the second stage starts from a weaker model |
| stages | 2 / 1 | a cut, a check, a merge and a whole heal | everything cut at once; riskier for thinking |
| same sequence length in both stages | off / on | the original writing the same answers twice (hours when the lengths differ; nothing when they do not: the note says so) | the larger student sees its longest reasoning traces cut short |
| thinking checks | Full 32 questions / Quick 16 | about a third of each of 6 or 7 checks | noisier gates |
| Try first | Full 128 texts / Quick 64 | minutes | a close race is less certain |
| check before healing | the best 2 cuts / the best 1 / off | about 20 minutes per cut | a cut with a low error but little thinking may be chosen |
| check of the unhealed cut | on / off | about 20 minutes per stage | none for the result (it only informs); the original is then checked after the first heal |
| retry heal | on / off | nothing unless a check fails: then the 80 M tokens retry | goes straight to cutting one step less |
| back-off | on / off | a whole stage, only if the checks fail twice | the run ends with its report |
| importance matrix | on / off | 10 to 20 minutes | the small sizes are a little less exact |

Not speed knobs and never changed here: the LoRA size, the sequence length (except the shared-length knob), the teacher's top-k, how the original is loaded, gradient checkpointing, the batch, the general-text share, the
kept intermediate files, and everything that is a requested output (the measured file and its comparison, Studio, the sizes).

**Every value below the recipe is said, three times**: a warning in the plan before Start (`auto_speed::speed_warnings`: the heal tokens below the recipe quote that healing below the recipe failed the thinking
checks in a real run), a note of the run, and the report's section "Speed settings" (`auto_speed::deviations`: setting, used, recipe, risk). `plan.speed` (additive, from `auto_plan::with_speed`) carries the total of
every preset (with the retry apart), and every knob with every choice's total and its change in minutes; the window shows them ("−6 h to −11 h").

**"I have at most N hours"** (`max_hours`, 1 to 336). `auto_speed::fit_budget` gives up settings one at a time, the least harmful first, and stops at the first that fits on the HIGH end of the estimate (a retry is
never in it): quick thinking checks, quick Try first, no check of the cut, the same sequence length, one cut checked before healing, no retry, the first stage on 20 M, one stage, the last stage on 20 M (the last
resort: a model that thinks will very likely fail its checks). Only values that are not already lower are touched, and nothing that was asked for (the thinking itself, the sizes, the measured file, Studio) is dropped. The plan
lists what was given up with the hours each saved and its risk; the run does the same when it starts (step `plan`, in the notes and the report: `budget`) and keeps `max_hours`. When even the last rung does not fit the plan is
refused: "Even the fastest settings take about X to Y for this cut, and you have N h. Choose more time or a larger size." A guard watches the clock: before a retry heal or a stage made again, if the estimate of that try plus
the steps still to run would end after the limit (counted from the creation of the run), the run ends `gates_failed` with its report and says "would end after your time limit".

## How far it is (`reshape/auto_progress.rs`, the `auto_progress` event)

The card shows one overall bar, how long the run has taken and how long it will probably take. All of it is an estimate and says so.

- **Percentage.** Every step has a weight: the middle of its time estimate (`plan.time`, at least a minute). Finished steps count at their weight, the running one by how far it is, skipped steps leave the numerator and the denominator. How far the running step is comes from the parts of its estimate (`start`, `load`, `cache`, `train`, `merge`, `check`, `measure` ...): a heal is the teacher's answers (tokens done of tokens in all), then the training (step of total steps), then the merge (GB). The worker's `progress` lines carry an additive `detail` (`{unit, done, total, rate, rate_unit, eta_s}`) for the teacher's answers, the merge, the thinking check, the export and the passes of the search.
- **Time left.** At first the plan's own figure for the rest of the step ("estimating the time left..."); after the step has run for two minutes with three windows of at least a minute (or 5 % done) the live speed: the worker's own figure when it sends one, else the speed of the windows, smoothed (exponential average). The displayed time may not rise by more than 20 % a minute. The steps still to come are counted at their estimates, scaled by how much slower or faster this run has been than the plan for steps of the same kind (between 0.5 and 3, only after one such step has finished). The range is the live part x 0.9 to x 1.25 plus the plan's low and high figures.
- **A retry** reopens the stage's steps: the bar steps back once, and the card says why ("a thinking check failed and the stage is healed again, about N more").
- **Where it shows.** The Auto-cut card (overall bar, "Stage 1 of 2", the line under the bar such as "Teacher cache: 12.4 M of 30.0 M tokens · 3,100 tokens/s · 41 min left", the running step's own bar, a chip and the estimate and time taken of every step); the queue on the Home page; the window title ("Auto-cut 42 % · stage 1 of 2: heal · about 14 h left") and the taskbar progress bar (Windows; refreshed every 5 s, put back when the run ends); a notification when a stage has been healed and checked ("stage 1 of 2 healed, 48 % done, about 14 h left"; the setting `notify_stages`, on by default, needs "Tell me when a job has finished"), and at the end "It took 31 h (estimated 28-40 h)". Webhooks still say only the kind and the result.
- **Saved.** `AutoState.progress` keeps the last snapshot at the steps' boundaries ("stopped at 42 %" after a restart); `StepRec.est` and `est_parts` keep the estimates. A run saved earlier has none of them: its card falls back to the older hours estimate and everything loads and resumes.

Events are forwarded while the job runs (not after it), at most one `auto_progress` a second.

## Find the best cut inside Auto-cut

`search_minutes` (None = the classic Try first on the planner's own few cuts) makes the step `try` the search of the Try first card (docs/TRY_FIRST.md) for that many minutes: the speed presets set it (Thorough 30, Balanced 20, Fast 10), the card has a selector under the speed settings. Its best three cuts become the candidates of the thinking check before healing (`try_cands`). `cut` (a `PinnedCut`: the exact blocks, the width, the loss, the search id and the thinking result when it ran) comes from "Use this cut in Auto-cut" on the Try first card: the plan takes it as the chosen cut (it must make the size asked for and may not cut the first or last block), the steps `try` and `think_gate` are left out, and with one stage those blocks are cut exactly; the card shows it with a Remove button. The time estimate counts the search as its minutes (load included), the later stages are not changed.

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
Full or Quick, **How careful, how fast** (the presets with their totals, the heal tokens and the sequence length of the stages with the hours each choice changes, More speed settings, "I have at most N hours"), *keep the intermediate files*, the confirmation of more than 4 blocks (only when the plan says it is needed), and the Studio publish box with the share fields (Save and Check). Under them the plan: blockers (red, Start
disabled), warnings, the stages (size, cut, healing tokens and hours, card), the disk and **How long this will take** (see "How long it takes"). Start / Stop / Continue / Show the report, then a **stepper** of the run's steps (done, running, pending, skipped, failed, with the note of the step), the live text
and bar of the running job, and the report as text.

Tauri commands (`src-tauri/src/lib.rs`): `auto_plan(spec, inspect)` (the planner for the model description the page already has; no job runs), `auto_start(spec, resume)` (a new run, or `resume` = run id; one job at a time, the
same busy flag as the other jobs), `auto_stop` (the Stop flag of the other jobs), `auto_status` (the newest run's state, whether a job runs, the saved Studio share; a run left "running" by a closed program is marked stopped here),
`auto_report(run)` (data and Markdown), `studio_share_set` and `studio_share_check` (the share settings, and the comparison with Studio's outputs folder). Events: `reshape-event` with `phase: "auto"`; the page reloads the state
file at every step event, so the stepper is what the file says. Tests: `ui/smoke/autocut.test.mjs` (the card, the plan request, start / stop / continue, the stepper, the end states, the share), `tests/auto.rs` (`preview`, `report_of`).

## Not verified yet

Nothing here ran on a real model, card or Studio (the publish ran against a stand-in server and temporary folders only). The planner is arithmetic: the sizes equal the Slim page's, the timeline is a sum, the card estimates are the Heal card's estimates. The state machine
is tested with a stand-in for every job; `RealEffects` (the wiring to the real jobs, including `publish`, which reads the global settings) has only been compiled and read, not run against a worker; the page was only run in a stand-in browser, never in the window.

The speed settings have not run on a real model: whether Balanced or Fast keep thinking is unknown (Fast very likely does not), as is the real saving of each knob and whether the two stages really heal on different sequence lengths on your card (the plan prints both). Events reach the card while a job runs now (they were held back until it ended).

The Auto-cut card says how long the rest takes and when it will be done: "about 3 h 20 min to 20 h left (done around 22:09–tomorrow 14:49)". Durations are rounded to 5 minutes below 10 hours and to 30 minutes from 10 hours up, so they do not claim false precision. An end of the range that falls on another calendar day names it ("tomorrow", or a short weekday name two or more days on), so the high end never reads as earlier than the low end. A finished run shows nothing.

## Where the time went (`timing.rs`)

Every worker job records how long its parts took (`protocol.timing`): loading a model (and in which mode), measuring, one Try pass, writing the model, the thinking check of a cut, the teacher's
answers (tokens per second and the mean row length), training, the merge split into reading / arithmetic / total, the 16-bit conversion, the quantisation, the importance matrix and the thinking
check's generation (tokens per second, the kernels that ran). They are ordinary log lines (`timing: {json}`) in the saved job logs and the `timings` key of a job's result.

`timing.rs` reads those lines back (`parse_log`), adds them up per kind of job and stage with shares (`summarize`) and says what is worth changing (`advice`; each rule answers worth or not worth when its
numbers are there):

| Advice | Worth it when |
| --- | --- |
| a saved 4-bit copy of the original (teacher) | loading it as 4-bit took more than 10 % of the time of these jobs |
| reusing the 16-bit GGUF and the merged folder between exports | the conversion took more than 10 % of the export time and the same model was exported at least twice |
| the merge is bound by the processor or by the disk | always one of the two: arithmetic more than twice the rest = processor, else disk ("put the models on the fastest drive") |
| install the fast kernels | a thinking check or a model load ran on the slow kernels |
| long rows dominate the teacher's answers | the speed in the last fifth is below 0.6 of the median |

It appears in the diagnostics bundle (`timings.json`), in the Auto-cut report (the jobs whose logs started while the run lasted; the section "Where the time went") and on the setup page ("Where the
time went", a button that adds up the latest 30 logs). The numbers decide whether the two things the advice mentions are worth building (a saved 4-bit teacher, reusing the 16-bit GGUF): send the
diagnostics bundle after the next real run.

## Healing in rounds (experimental, `heal_rounds`)

The preset **Stop early when good (periodic checks, experimental)** is Balanced with `heal_rounds {round_m: 5, check: quick}`; the knob "Heal in rounds, stop when good" is in the speed settings (its time
is a range: the first check passes ... every round to the cap). Every stage heals in rounds with its recipe tokens as the cap and the original's numbers passed in. Rounds **replace the fixed-budget retry**:
a stage that ends at its cap goes on for 40 M more tokens on the same run, a plateau goes straight to the back-off, the time limit ends the run. The estimate of a heal in rounds is (best case low, worst case
high) with `StepTime.rounds`; the heal's weight in the progress bar is its worst case, so an early stop makes the bar jump forward. The report has a "Healing in rounds" table per stage. Details: `docs/HEAL_ROUNDS.md`.

## The speed-setting radios follow the click (fix)

The choices of "How careful, how fast" (the levers and "More speed settings") are drawn from what is chosen NOW (`acOptionOn`, the same comparison as the preset check), not from the plan that is still on its way: the plan is asked again after every change (`acRefresh` drops `ac.plan`) and used to redraw the old choice until it arrived, and a click in that window was dropped. The table of the last plan is kept (`ac.speedCache`, for the model it was made for) and a click uses its `set` values, so it is never lost; while the new plan loads the old changes stay, marked "updating…", the preset reads Custom at once, and without any table yet the radios are not offered. `History::recent` keeps its parsed answer in memory while the saved logs are the same ones (name, size, modification time), and `speed_info` makes each distinct spec's plan and time estimate once; the numbers are the same as before (the existing tests). This was tested in the stand-in browser of `ui/smoke/autocut.test.mjs` only: nothing was verified in a real window, and how much faster the plan is has not been measured.


## vLLM (experimental)

`AutoSpec.vllm` (`teacher_cache`, `checks`, `automatic`; absent = the run as it always was, the key is not even written; every preset except Custom switches it off). The card section "vLLM (experimental)" has the two switches and "automatic", each with the change in time from the plan (your measured numbers, scaled by model size, or the report's estimates, said so), the prerequisites (self-test run and its verdict, the model it measured, the long-answer speed, the teacher check per made model) and the plan's reason for every decision.

- **Time estimate** (`plan.time.vllm`): a heal's cache part becomes engine start-up + scoring; every check becomes start-up + generation (the engine starts for each check job: it pays only for long checks; with the 27 B numbers of one PC a Quick check is about as long as with the standard engine). "Automatic" picks vLLM for a step only when its middle estimate is under 90 % of the standard engine's.
- **One engine for all checks of a run**: a check is judged against the original's, so the checks use the engine of the baseline; when a vLLM check falls back, the original is checked again with the standard engine.
- **Order of GPU jobs**: before every step `engine_cleanup` stops leftover `ndforge-vllm-*` containers; no vLLM job overlaps a training (tested with the Mock effects).
- **Heals in rounds** (`early` preset or a custom rounds setting) keep the standard teacher: the engine would start every round.
- **Fallback**: a failing vLLM step goes on with the standard engine, with a note in the run's notes and the report.
- **Report**: "Steps that used vLLM": per step the engine, teacher quantization and identity, measured start-up and speed against the estimate, fallbacks. `AutoState.engines` and `check_engine` are additive; old run files load.
