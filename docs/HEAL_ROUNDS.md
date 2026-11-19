# Healing in rounds (experimental)

The Heal card and Auto-cut can heal in **rounds**: the heal is trained in stretches of a few million tokens (1, 2, 5, 10 or 20; 5 is the default), the model as it is
trained is checked for thinking after a round, and healing **stops at the first round that passes**. The merged model is then checked again, and that check decides.
The aim is speed: a cut that is good after 15 M tokens does not have to learn 40 or 80 M. Everything on this page is labelled **experimental** in the program.

Code: `crates/ndforge-core/src/reshape/heal_rounds.rs` (the rules and the driver), `runs.rs` (`heal` branches to it), `thinkcheck.rs` (target `round`), `auto*.rs`;
worker: `distilljob.py` (`rounds_upto`, `round_lr_factor`, `round_order`, `round_slice`, `round_start`, the `rounds` key of `distill`), `thinkcheck_hf.py` (`adapter`).

## What one round does

1. **The original answers only the next chunk** (`distill_cache` with `upto_general_tokens` and `reuse_rows`). The row list is fixed for the whole run by the cap, so the
   rows of a round are the first rows of a longer one; the answers are the shared teacher cache's usual prefix, so a later normal heal of the same cut reuses them. An early stop
   therefore saves the original's time too, which is most of the time of a heal.
2. **The student trains the round** (`distill` with `rounds`): all project rows (cycled as in a normal heal) and the general rows answered *for this round* (none repeats across
   rounds). The optimizer, the scheduler position and the random state go on from round to round through the HF checkpoint that is forced at the end of every round
   (`ignore_data_skip`; the job slices the round's data itself, so a crash in a round continues exactly where it was). Without a checkpoint it starts from the previous round's
   adapter with a fresh optimizer and says so.
3. **The check** (`think_check` with `adapter`): the student is loaded as it was trained (4-bit NF4 with double quantization for QLoRA) and the round's LoRA is put on top with PEFT,
   **not merged** (a merged 4-bit model is a different model). Its cache key carries a hash of the adapter's content. No check comes before `first_check_m` (default 10 M: the cuts
   of the real runs started far from passing, so a 5 M check is very likely a wasted 15-25 minutes), and none after the last round (the merged model's check is the gate).

## The stop rule, and why

* **Stop at the first passed round check** ("good enough" is the thinking gates against the original's numbers in the same mode, verdict pass or warn). A false stop costs one merge
  and one check (about an hour); asking for a second pass would cost one more round, 3 to 5 hours at 17-20 B, every time.
* **Then merge and check the merged model** (Auto-cut: its own mode, Full in Thorough). This is the real gate. Pass or warn: done. Fail while below the cap: the merged model is
  deleted (with its saved check), `strict` is set, healing goes on, and stopping now needs **two passed round checks in a row** (no merge ping-pong).
* **Plateau (thinking)**: at least half of the cap trained, the last three checked rounds did not raise the best thinking score by 0.125 (the tie width) over the best before them,
  *and* the loss on your text fell by less than 1 % over them. Two signals, because 16 questions alone are too noisy. After a plateau the merged model is checked; if it fails the
  stage ends (more rounds would not help).
* **Plateau (a student that does not think)**: no check at all; half of the cap trained and the loss fell by less than 0.5 % over the last two rounds.
* **Cap**: the latest round's adapter is merged and checked. **Time** (`deadline`, Auto-cut's hour limit): before a round, if the round, the merge and the merged check would end
  after the limit, it stops with the reason "time". **Never** (`stop: "never"`): train to the cap and only record the checks (the A/B mode below).
* Which adapter is merged: always the latest round's; Quick scores are too noisy to pick an older one.

How noisy a Quick check (16 questions) is, for the original's numbers (closed 0.97, loops 0.00): the limits are closed >= 0.873 (14 of 16), accuracy >= 0.776 (13 of 16), loops
<= 0.10 (at most one of 16). The chance that the closed-rate gate alone passes for a model whose true rate is p: p = 0.95 gives 0.96, 0.90 gives 0.79, 0.85 gives 0.56, 0.80 gives 0.35,
0.75 gives 0.20. A model near the line passes by luck about half the time: this is why the merged check exists.

## In Auto-cut

Rounds replace the fixed-budget retry. A stage that ends at its cap without a pass goes on for **one** extension of 40 M tokens on the same run (same student, same answers of the
original, `extend`); a plateau goes straight to the back-off; the time limit ends the run with its report; a second failure is the back-off as before. Worst case for a 40 M stage:
80 M student tokens and 80 M teacher tokens (today: 120 M student, 80 M teacher). The preset **Stop early when good** is Balanced with rounds of 5 M and quick checks (the cap stays the
recipe's tokens); the knob **Heal in rounds** is in the speed settings with the best and the worst case. The original's check is passed in, not repeated. Without `heal_rounds` the
run, the plan and the saved state are what they were.

## Time and overhead

Every round adds the load of the original (3 to 8 minutes), the load of the student for the training and again for the check (2 to 5 minutes each), the check (Quick 10-15,
Full 15-25 minutes; the unmerged LoRA may add 10-25 %), the loss on your text (1-2) and the checkpoint, and three or four process starts: about 20 to 35 minutes at 17-20 B. With
c minutes of work per million tokens (about 36-54: the original 10-15, the student 26-39) the overhead share is O / (O + R c): 27-49 % for 1 M rounds, 16-33 % for 2 M, 7-16 % for
5 M, 4-9 % for 10 M. The round size that costs the least overhead for a heal that passes after N M tokens is sqrt(2 N O / c): about 3.7, 5.2 and 7.3 M for 10, 20 and 40 M. That is why 5 M
is the default. A round with fewer than **8 estimated optimizer steps** (a step reads about half of seq x sequences per update tokens; a 1 M round at 8192 is about 7.6 steps) is refused.
Running every round to the cap costs about 2 to 4.5 hours more than one 40 M heal; stopping at 15 M saves about 13 to 20 hours. All of this is a formula (`auto_time::rounds_estimate`),
not a measurement.

## The learning rate

One cosine planned for the whole budget is wrong when healing stops early: a check at 15 of 40 M would see a model at a high rate. Every round therefore has its own warm-up,
plateau and cool-down, and the optimizer state is kept:

```python
def round_lr_factor(step, s_start, steps, index, warmup, sch):
    t = step - s_start
    peak = max(sch["floor"], sch["gamma"] ** (index - 1))          # gamma 0.85, floor 0.35
    if index == 1:
        w, lo = min(warmup, ceil(0.25 * steps)), 0.0
    else:
        w, lo = min(sch["rewarm_max"], ceil(sch["rewarm_share"] * steps)), sch["min_ratio"] * peak     # 10 % of the round, at most 10 steps
    c = max(1, ceil(sch["cooldown"] * steps))                      # the last 20 %
    if t < w:           return lo + (peak - lo) * (t + 1) / w
    if t >= steps - c:  return peak * (sch["min_ratio"] + (1 - sch["min_ratio"]) * (1 - sqrt(u)))     # min_ratio 0.05
    return peak
```

(The worker clamps `t` to the round and `u` to 1.) A fresh Adam state each round was rejected: it restarts the bias correction. About 20 % of every round is spent cooling down and
re-warming. **Whether repeated short cycles reach the quality of one long cosine run on a 40 M heal is not known.**

## Disk

The original's answers grow about 196 bytes per token (about 1 GB per 5 M), at the cap the same as today. A heal in rounds also keeps two round adapters (rank 64 at about 20 B: 0.3-0.6 GB each)
and two checkpoints (adapter and 8-bit Adam state, 1-2 GB each): about 4 GB more than a normal heal, checked up front. The checkpoints are kept unless the heal passed (a failed heal can go on).

## The A/B the owner can run

Same cut: (a) a normal 40 M heal; (b) `stop: "never"`, 5 M rounds, cap 40. Compare the merged Full check, the loss on your text and the KL at the end. (b) also gives the learning curve
(thinking score against tokens), the real answer to "how many tokens does a cut need". A cheaper first pass: a small student (4-8 B) on 10-20 M.

## Adapter check against merged check

The check during healing looks at NF4(W) + BA (the student as it was trained); the merged model on a 32 GB card is quantized after the merge, NF4(W + BA), when it does not fit in 16 bit. They can
differ on close calls, so the merged check decides, and the report puts the last round check and the merged check side by side (same mode).

## Files and keys

`heal.json` gets a `rounds` block (state, rows, merged checks, baseline, stop, final); `heal_round` events (Rust side only) tell the page and the progress bar where it is; the worker keys are in
`docs/WORKER_CONTRACT.md`. Old `heal.json` and `auto.json` load and resume; a spec without `rounds` / `heal_rounds` makes the same jobs, plan and estimate as before.

## Not verified (needs the owner's GPU)

* the real overhead of a round (the loads of the original and the student, the check with the LoRA on top at 17-20 B);
* whether the check of base + adapter agrees with the merged NF4 check on close calls;
* whether round-wise training with re-warms reaches the quality of one long cosine run at the same tokens;
* how many tokens a real cut needs to pass, and whether early passes happen at all for the owner's cuts (they failed badly unhealed);
* whether the plateau constants stop at the right time;
* streaming re-reads and the order of the general text from the Hub on the owner's network (`reuse_rows` avoids it after round 1; a changed order is detected and said);
* the computer staying awake across a multi-round night on Windows;
* the disk growth of adapters and checkpoints on a real 20 B.

The worker's torch tests of the round schedule, the resume and the adapter check run on tiny random models on a CPU in CI only (torch cannot be installed where this was written).
