# Storage: where the disk went, and a guarded way to give it back

Page **🧹 Storage** (left list, after "Compare quants", unlocked with a project). Backend: `crates/ndforge-core/src/storage.rs`; Tauri commands `storage_scan`, `storage_delete`, `storage_cancel`; events on `storage-event`. No worker or contract change.

## What it lists (groups)
| Group | Items | Notes |
|---|---|---|
| Models | `models/<name>` (cut, healed), the cut before healing of a healed model | the model entry goes through the model list and the training runs' check (`reshape::runs::removal_checks`) |
| Exports | `exports/<name>`, its `model-f16.gguf`, its `merged/`, `*-gguf-work`, `*.partial` | an export without its f16 / merged part still lists |
| Heal data | teacher caches (`models/_teacher_cache/<id>`, a model's `distill/cache`), heal checkpoints (`distill/run`), Try-first / search caches, quant-compare logits (`base.kld`), quantfit, downloaded matrices, speed measurements, training checkpoints, the general-text cache | deleting a teacher cache makes the original answer your texts again (hours) |
| Downloaded models | every `models--org--name` of the Hugging Face cache the app uses (marked "in the model list" or "not known to the app: another program may use it"), the vLLM engine volume | the volume is removed with `docker volume rm` / the `wslc` template (`volume_remove`, UNVERIFIED), refused while an engine container exists; its size is "not available" |
| Leftovers | `.partial` files and work folders of failed jobs, job logs older than 14 days | |

Sizes are measured in the background; the rows come first ("size not measured yet"), an earlier measurement is shown while the new one runs (`storage_sizes.json` in the config folder).

## Suggestions
Only intermediates, with a reason, never ticked by themselves: the 16-bit GGUF when quantized files exist, the merged folder when GGUF files exist, a cut before healing when its run is finished, the teacher's answers and heal checkpoints when no heal or Auto-cut run can be continued, leftovers, quant logits, old logs. The preset "After a finished Auto-cut keep only the final healed model and the GGUF files" ticks the intermediates (not logits, not logs).

## The rules of a deletion (hard safety)
* The window sends only opaque ids (`s` + 16 hex digits from the kind, the path, the folder's modification time and the way it is deleted). The program scans again and refuses an id that is not in the fresh scan ("changed since").
* `Roots`: only below `models/`, `exports/`, `quantcheck/`, `quantfit/`, `imatrix_downloads/`, `benchmarks/`, `runs/<name>/checkpoints` of the project, the config folder's `general_cache`, `logs`, `vllm_selftest`, and the Hugging Face cache roots. `..` is refused, a root itself and the project folder are never deleted, a path through a link that leaves the folders is refused. Datasets, settings and any model outside these folders (the owner's original on another drive) are never touched.
* Links (symbolic links, Windows junctions and other reparse points) are never followed: a link is removed itself.
* Nothing is deleted while a job runs (the one-job flag), while an Auto-cut run is running, or when a stopped or failed Auto-cut run that still has steps to do uses the model; the teacher's answers and heal checkpoints are kept while a heal or run can be continued.
* A model, or more than 5 GB, needs the typed word `DELETE`. The model chosen on the Slim or Auto page needs a second confirmation.
* Dry run: `storage_delete {ids, dry_run: true}` returns exactly what a real call removes (items inside another chosen item are not listed twice).
* Permanent: no Recycle Bin (it would not free the space). Windows: `\\?\` for long paths, read-only attributes cleared, a file in use is tried six times, then reported with the program to close. A failure of one item does not stop the others; Cancel stops between two files.

## What the other screens see
The project keeps `storage_deleted.json` (kind, label, date, bytes). An Auto-cut run that used a deleted model gets `made[].removed = true` and the note "Deleted on ... on the Storage page", so its report says it. The lists of models and exports read the folders, so a deleted one is simply not listed; the pages that read a run's own state keep working (tested: Slim list, export list, Compare quants state, Auto-cut report).

## Delete buttons elsewhere
"Delete…" on the Slim page's made models and on the Use page's export list open the Storage page with that item ticked and ask the program what would go; nothing is deleted before the confirmation. The Auto card has "Free the disk space of this run…".

## Not built (follow-up)
Deleting intermediates automatically when an Auto-cut run ends (an option) is not built.

## Tests (cache folders such as quantfit and the general cache are emptied, the folder stays: `Exec::Contents`)
`crates/ndforge-core/tests/storage.rs` (guards, link, id revalidation, dry run = real, failures, cancel, sizes, suggestions, logs, Hugging Face cache, "deleted" screens), `tests/auto.rs` (a stopped run's model is in use, a finished run's is referenced, the report says it was deleted), `ui/smoke/storage.test.mjs`. Not tested on a real Windows disk: junctions, long paths, files in use.
