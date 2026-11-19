# The thinking check

A cut and healed model can look fine on the usual numbers (text-prediction error, the Test card) and still have lost the habit of **thinking before it answers**. The thinking check asks a
small suite of questions with thinking switched on and says whether the model still writes a reasoning trace, finishes it, reaches the right answer and stays free of loops and garbage,
compared with the original model. It is a smoke test with 32 questions, not a benchmark: a difference of one or two questions is noise.

## Where it fits

Heal with **Keep thinking** (docs/RESHAPE.md), then check: the original, the cut model (before healing: how much thinking the cut alone took away) and the healed model, and a GGUF of the export (does
the file llama.cpp runs still think; the exported template keeps thinking for a model healed with the recipe).

## The suite

`worker/ndforge_worker/suites/think_suite.jsonl` (written for this program; `suite_version` 1; installed with the worker as package data):

| Topic | Questions | Kind of answer |
|---|---|---|
| arith | 8 word problems and sums | a number |
| logic | 8 orderings, small syllogisms, a liar puzzle, a day count | a name, a word, a number or a letter (A-D) |
| code | 8 short Python programs: what do they print? | a number or a short text |
| count | 8 counting and string questions | a number or a word |

The questions are interleaved by topic, so the first 16 (the **Quick** set) hold four of each. Eight more questions are **control questions** asked with thinking OFF (the
prompt ends with the empty reasoning block): they show that the model still answers plainly when it is not asked to think. Every prompt asks for the final answer on the last line
as `Answer: ...`; the scorer reads that line (else the last line, else the last number).

The answers were checked when the file was made (the code answers are computed by running the snippets; `test_think_check.py` runs them again). Changing a question or its answer means
a new `suite_version`, because saved results are keyed by it.

## How a question is asked

Thinking ON (`enable_thinking=True`, the prompt ends with `<think>\n`), sampled with the settings Qwen recommends for thinking: temperature 0.6, top-p 0.95, top-k 20, seed 3407.
At most **3072** new tokens per question (Quick: **2048**; control questions: 512). Because the prompt already opens the reasoning block, the generated text starts inside it;
`thinkcheck.split_output` cuts it at `</think>`.

**The sampling is deliberate.** A greedy check (always the most likely token) did not show the failure of a real run (docs/REAL_RUN_17B.md): the same cuts that looked fine greedy looped or ran
into the limit when sampled, which is how the model is really used. `thinkcheck_hf` therefore always calls `generate` with `do_sample=True` (a test asserts it), and llama-server is asked with the same
settings.

## What is measured

All rates are shares of the questions asked (0 to 1).

| Number | Meaning |
|---|---|
| `think_rate` | a non-empty reasoning trace was written |
| `closed_rate` | `</think>` was written (the trace ended and an answer followed) |
| `accuracy` | the answer after `</think>` is the right one (numbers normalised: `2.50` = `2.5`, `1,250` = `1250`; text compared lower-case without markdown and end punctuation) |
| `loop_rate` | a run of 32 tokens repeated 4 or more times within the last 1024 tokens, or the text compresses to under 15 % of its size (only texts of 2,000 bytes or more) |
| `garbage_rate` | more than 1 % replacement or control characters, or more than 20 % of the letters in another script (the questions are English) |
| `truncated_rate` | the token limit cut the answer off |
| `median_think_tokens`, `p90_think_tokens` | length of the traces, in tokens (about a quarter of the characters when the backend does not count them) |

## The gates

Judged against **the original model's** numbers, from the same suite and the same settings:

| Gate | Pass when |
|---|---|
| `closed_rate` | at least 0.9 x the original's, and at least 0.75 |
| `accuracy` | at least 0.8 x the original's |
| `loop_rate` | at most the original's + 0.10 |
| `garbage_rate` | at most 0.05 |
| `median_think_tokens` | at most 2.5 x the original's (at least 2.5 tokens, for an original that never thinks) |

**Fail**: any gate missed. **Warn**: all met, but one is within 5 % of its limit. **Pass**: otherwise. The cases are in `worker/tests/data/think_gate_cases.json`; the Python
(`thinkcheck.gates`) and, from the Rust step on, the Rust side are both tested against that file.

## The jobs (worker)

* `think_check_plan` `{"backend": "hf" | "llama", "quick": false}`: no model is loaded; answers the question counts, the settings, a rough size (about 50,000 generated tokens for the full suite,
  about 10 to 15 minutes with transformers in 4-bit on a 27B model, about 5 minutes with llama-server, estimates).
* `think_check` `{"model": folder or repo, "backend", "quick", "control": true, "stop_file", ...}`: runs the questions through a backend, one `think_item` event per answered question
  (`id, ok, closed, think_tokens, control`), `progress` events (step `think`), and one result: `{model, backend, suite_version, settings, main: {...}, control: {...}, items: [...], stopped, seconds}`.
  Error code `think_check_failed`.

New job types and events only; nothing existing was renamed.

## Backend 1: transformers (`backend: "hf"`, the default)

`thinkcheck_hf.py` loads the model the way the other jobs do (16-bit, 4-bit NF4 or spread over card and memory, `load: auto` picks by the free graphics memory), writes each question with the
model's own chat template (thinking on: the prompt ends with `<think>\n`; off: with the empty block), and generates in **batches of 8** with left padding and sampling (the seed is
set per batch, so the same questions in the same batches give the same answers). Out of memory halves the batch and goes on; `"batch": 1` asks one at a time. The trace length is read from the position
of `</think>` among the generated token ids. Events: `progress` (step `load`, then step `think`) and one `think_item` per answered question.

**The saved numbers of a model.** With `cache_file` in the job, a finished check is saved (keyed by the model's weights fingerprint, backend, load, suite version and all settings) and the next
identical check is answered from it (`"cached": true`): the original is checked once however often it is compared. A stopped check is never saved. A Hub id is keyed by its name, not its revision.

## Backend 2: llama-server (`backend: "llama"`)

Checks a **GGUF file** the way llama.cpp-based programs (Unsloth Studio, LM Studio) will run it: chat template, quantisation and all. `thinkcheck_llama.py` finds `llama-server` with the
same search as the other llama.cpp tools (`llamacpp.find_tool`; a missing one gives the error code `llama_cpp_missing`, so the page offers the installer), starts it on a free port of
this computer (`-m file.gguf --host 127.0.0.1 --port N --jinja -np 4 -c 16384 -ngl 99`: 4 questions at a time, 4096 tokens each), waits for `/health` (up to 15 minutes), asks the
questions through `/v1/chat/completions` with `chat_template_kwargs: {"enable_thinking": true | false}` and the sampling above, and stops the server in every case (done, Stop, error;
on Linux it also ends if the worker is killed). A server that splits the reasoning into `reasoning_content` and one that leaves `<think>...</think>` in the text are both understood. The Stop
file is looked at twice a second while questions run. The trace length is estimated (a quarter of the characters), the loop test uses words instead of token ids.

## In the app

On the result card of a made model (Slim or grow a model page), under the Heal card: **Thinking check**. Choose Quick (16 questions, shorter limit; the default for a model of 15 B or more) or
Full (32), then *Check the original*, *Check the cut model*, *Check the healed model* (when there is one), or choose / type a GGUF file and *Check this GGUF* (the llama-server way). The table
sets the numbers side by side; the last row is the verdict against the original: **Pass**, **Close to the limit** (all gates met, one within 5 %) or **Fail** (with the gates that were missed,
in words). The original is checked once and kept: a repeated check of the same weights, settings and suite is answered from the saved file. Quick and Full are kept apart; a result of another suite
version is not shown. Stop ends the check between questions and saves nothing.

Where things are: `models/<name>/think_check/<target>.<full|quick>.json` (written by the worker: its key and its result), `STOP` while stopping. Rust: `crates/ndforge-core/src/reshape/thinkcheck.rs`
(the job, the saved results, the gates, tested against `worker/tests/data/think_gate_cases.json`); Tauri commands `think_check_results`, `think_check_plan`, `think_check_start` (events on `reshape-event`
with `phase: "think"`; the Stop button is `reshape_stop`); page `ui/app.js` (`thinkCard`), smoke test `ui/smoke/thinkcheck.test.mjs`.

## In the automatic search of Try first

The search of the Try first card (docs/TRY_FIRST.md) can run this check on the best two cuts it found: each is written, checked (the original once) and deleted, one at a time; the result decides between the two (`rank_thinking`) and shows next to the recommendation. A low text-prediction error does not prove that a model thinks; without this check the search says "Thinking not checked".

## Stopping

Stop asks the worker to end (a STOP file) and, after a grace period, ends the worker **and everything it started**, so `llama-server` does not stay behind holding the graphics card. On Linux and macOS the worker runs in its own process group and the whole group is killed; on Windows the worker is put into a Job Object that is terminated on Stop and also closed when this program ends, however it ends (`crates/ndforge-core/src/envmgr/process.rs`, `ProcessTree`). The tests start a program that starts another one and check that the second is gone after a stop. Not checked: a real `llama-server` with CUDA on Windows.

## Not verified yet

* The card and the Tauri commands were not run in a real window (the commands compile with the program; the page was tested in a stand-in browser and the Rust side with a fake worker).
* Nothing here ran on a real model: the scorer was tested on written outputs and the job on a stand-in backend. Whether the suite is easy or hard for a real Qwen3.8 27B (and for the
  cut), how long its traces are, and whether the gates are in the right place need the owner's runs.
* The gates' numbers come from the plan, not from measurements; expect to adjust them after the first real comparisons.
* The llama backend was tested against a stand-in `llama-server` (a script that speaks the same HTTP), not against a real llama.cpp: the flags (`--jinja`, `-np`, `-c`, `-ngl`), the
  `chat_template_kwargs` field, `top_k` and `seed` in the request, `/health` and the shape of the answer are from llama.cpp's documentation as remembered. On Windows the Job Object (see "Stopping") ends the server together with the worker; the Stop file does too. Whether Unsloth Studio's own chat default matches `enable_thinking` as sent here is not known.
* Batched sampling of a model with linear-attention layers and left padding: the tiny test model (the real architecture, random weights) runs it, but whether the real 27B answers the same batched as
  one at a time is not checked (use `batch: 1` or the llama backend if the numbers look odd).
* The loop test on word lists (the llama backend has no token ids) is a rough stand-in for the token version.

## A round of a heal in rounds (target `round`)

`ThinkSpec.target = "round"` with `round: n` checks the student of the heal (`heal_config.json`'s model) with `distill/run/rounds/r<n>/adapter` put on top and **not merged**: the job carries `adapter` and `double_quant`
(and a `load` that follows how the student was trained). The result is kept in `think_check/round-r<n>.<quick|full>.json`; the cache key has a content hash of the adapter, and only when there is one (the keys of every other check
are unchanged). It looks at the model as it is trained; the merged model can score a little differently (`docs/HEAL_ROUNDS.md`).

## The vLLM engine (experimental, off by default)

> Built from a research report, not yet run on this PC. Run the vLLM self-test first (docs/VLLM.md). Speeds are estimates, not measurements.

Under Training setup, "vLLM engine (experimental)" switches the engine on; then the Thinking check card offers **Engine: Transformers (default) / llama-server (GGUF files) / vLLM (experimental)**.
vLLM asks all questions at once (40 in parallel) and is estimated at about 400 tokens per second (the standard engine: about 55, llama-server: about 165) plus 2-5 minutes of start-up; the plan
says so. Its results are kept in their own files (`<target>.<mode>.vllm.json`) and shown in their own table: a model is judged against the original checked **the same way** (the numbers of two
engines are not identical: same distribution, other random numbers). Before a vLLM check starts, the checklist asks that the engine exists (Docker, Docker inside WSL, WSLc, or vLLM in WSL) and
that nothing else uses more than about 2 GB of the graphics card (vLLM reserves about 88 %).
