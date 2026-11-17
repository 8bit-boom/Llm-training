# A real run: a 27 B model cut to about 17 B

This is what one real run showed, kept as a case study. It explains why the program checks the thinking of a cut before healing, and where several fixes come from.
The numbers are from one run on one computer; the thinking check is noisy (16 questions in Quick), so read them as an illustration, not a benchmark.

## The model

A 27 B model of the Qwen3.5 family: 64 layers, which are 16 blocks of 3 linear-attention layers and 1 full-attention layer, hidden size 5120, MLP width 17,408, with a vision tower and a
multi-token (MTP) head.

## The two cuts

| | Dropped | MLP width | Parameters |
| --- | --- | --- | --- |
| Cut A | layers 36 to 47 (blocks 9, 10, 11) | 10,240 (58.8 % kept) | 17.38 B |
| Cut B | layers 28 to 43 (blocks 7 to 10) | 11,264 | 17.07 B |

## What the measuring said

Text-prediction loss on the calibration text:

| | Loss |
| --- | --- |
| Original | 1.4519 |
| Cut A | 1.2668 (16.9 % lower than the original) |
| Cut B | 1.3989 |

Both cuts had a lower loss than the model they were cut from. That is itself a warning: the original and the cut may have been measured differently (a 4-bit or offloaded load against a
16-bit one), or the calibration text may suit the cut. A lower loss does not mean the model can still think.

## What the thinking check said

The same questions, sampled (not greedy), thinking on:

| | Finishes its reasoning | Right answers | Loops | Cut by the limit | Median length (tokens) |
| --- | --- | --- | --- | --- | --- |
| Original | 0.97 | 0.97 | 0.00 | 0.03 | 121 |
| Cut A | 0.50 | 0.28 | 0.50 | 0.47 | 519 |
| Cut B | 0.44 | 0.25 | 0.59 | 0.56 | 3072 |

With thinking switched off (the control) all three answered 0.88 of the questions right. So the cuts kept their knowledge but lost the ability to finish a line of reasoning: half of
the answers looped or ran into the limit.

Greedy decoding did not show this. A greedy check looked healthy; sampling, which is how the model is really used, exposed the loops. The check samples on purpose.

## Incidents and what fixed them

| | What happened | Fix |
| --- | --- | --- |
| a | Merging or cutting into a folder that did not exist yet failed with "WinError 3": the free space was asked of a path that was not there | Create the missing parents; ask the nearest existing folder for the space |
| b | The computer went to sleep (Modern Standby) in the middle of a long job | Keep-awake in the worker itself, for every long job, and a preflight warning with the exact commands |
| c | Qwen3.5's linear-attention layers ran on the slow PyTorch path: the fast kernels were not installed or not reported | Report which kernels run, and install the fast ones where a ready-made package exists |
| d | A script crashed because a vision-language model hands back a processor, not a tokenizer | One helper, `text_tokenizer`, for everything that tokenizes text; a test that guards it |
| e | The multi-token head was sliced by weight size when the MLP was thinned: never measured, never healed, 0.42 B parameters for nothing | A width cut drops the head (depth-only cuts keep it, with a note) |
| f | A dataset file was rewritten in place while a job might still read it | Never rewrite a file in place while a job may read it: write a temporary file and rename it |

## The rules that came out of it

- Never choose a cut of a reasoning model by loss alone. Check how it thinks, before healing and after.
- A cut that scores a lower loss than the original needs a second look at how both were measured.
- Promote a healed model only if its thinking check passes the gates against the original.
- Never rewrite a file in place while a job may read it.
