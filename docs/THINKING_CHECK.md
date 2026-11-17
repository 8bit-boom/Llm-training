# The thinking check

A cut and healed model can look fine on the usual numbers (text-prediction error, the Test card) and still have lost the habit of **thinking before it answers**. The thinking check asks a
small suite of questions with thinking switched on and says whether the model still writes a reasoning trace, finishes it, reaches the right answer and stays free of loops and garbage,
compared with the original model. It is a smoke test with 32 questions, not a benchmark: a difference of one or two questions is noise.

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

## Not verified yet

* Nothing here ran on a real model: the scorer was tested on written outputs and the job on a stand-in backend. Whether the suite is easy or hard for a real Qwen3.8 27B (and for the
  cut), how long its traces are, and whether the gates are in the right place need the owner's runs.
* The gates' numbers come from the plan, not from measurements; expect to adjust them after the first real comparisons.
* The loop test on word lists (the llama backend has no token ids) is a rough stand-in for the token version.
