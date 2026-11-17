# Documents of ND Lore Forge

Start with the root [README](../README.md). The documents below describe what is built **now** unless a line says "historical": a plan written before its code, kept as it was
(the code and the current documents are right where they differ).

## Using the program
| Document | What it is about |
| --- | --- |
| [INSTALL.md](INSTALL.md) | Installing the program, the training software, llama.cpp |
| [LLAMA_CPP_INSTALL.md](LLAMA_CPP_INSTALL.md) | The llama.cpp installer: where it looks, what it downloads |
| [TRAINING_SPEED.md](TRAINING_SPEED.md) | Training speed: what was looked at and built |
| [TRUENAS_WORKER.md](TRUENAS_WORKER.md) | Training on a server (the worker image) |
| [UNSLOTH_DESKTOP.md](UNSLOTH_DESKTOP.md) | Unsloth Desktop / Studio as a place to train and publish |

## Slimming, healing and exporting models
| Document | What it is about |
| --- | --- |
| [AUTO_CUT.md](AUTO_CUT.md) | Auto-cut: one click from a big model to a smaller one that still thinks |
| [RESHAPE.md](RESHAPE.md) | Slim or grow a model: what was built and decided |
| [TRY_FIRST.md](TRY_FIRST.md) | Trying cuts before making one |
| [HEAL_ANY_MODEL.md](HEAL_ANY_MODEL.md), [HEAL_SETTINGS.md](HEAL_SETTINGS.md) | Healing: any model, the settings, graphics memory, the teacher cache |
| [GENERAL_TEXT.md](GENERAL_TEXT.md) | The general text mixed into healing and the importance matrix |
| [THINKING_CHECK.md](THINKING_CHECK.md) | The thinking check, and how Stop ends everything a job started |
| [IMATRIX.md](IMATRIX.md), [MIXED_QUANTS.md](MIXED_QUANTS.md), [QUANT_COMPARE.md](QUANT_COMPARE.md) | The GGUF export: importance matrix, mixed sizes, comparing sizes |
| [PRUNE_QWEN38_27B.md](PRUNE_QWEN38_27B.md) | A worked example with the 27B model |
| [REAL_RUN_17B.md](REAL_RUN_17B.md) | A real run cutting a 27 B model to about 17 B: the numbers, what went wrong, and the rules that came out of it |

## Contracts with nd-world
| Document | What it is about |
| --- | --- |
| [BUNDLE_FORMAT.md](BUNDLE_FORMAT.md) | The `.ndtrain` training bundle |
| [UNSLOTH_API_CONTRACT.md](UNSLOTH_API_CONTRACT.md) | The Unsloth calls both products share |
| [WORKER_CONTRACT.md](WORKER_CONTRACT.md) | The worker's job types, events and error codes (`contract.json`) |

## Historical plans (not the current state)
| Document | |
| --- | --- |
| [../PLAN.md](../PLAN.md) | The first plan of the whole program |
| [RESHAPE_PLAN.md](RESHAPE_PLAN.md) | The plan of "Slim or grow a model" before it was built |
| [RESHAPE_V2_PLAN.md](RESHAPE_V2_PLAN.md) | The plan of version 2 of it, before its code |
