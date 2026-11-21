# Memory: the panel, "Free GPU", "Free RAM"

Backend `crates/ndforge-core/src/memory.rs`; Tauri commands `memory_status`, `memory_free_plan`, `memory_free` (camelCase arguments: `kind`, `planToken`, `stopRunning`); the panel is in the status footer ("Memory" button and a popover).

## What the panel shows
* **Footer line**: GPU used / free of total, RAM used / free of total, and the app's own share (every few seconds when idle, about every 2 s while a job runs). The numbers come from `nvidia-smi --query-gpu=memory.total,memory.used` and, for RAM, `/proc/meminfo` (Linux) or `Win32_OperatingSystem` (Windows); the app's share is the working sets of this program and its worker processes.
* **Popover**: the same numbers, the engine containers of this program (`ndforge-vllm-*`), the other programs that hold graphics memory (listed read-only with their size when `nvidia-smi --query-compute-apps` says it) and the two buttons.
* **Windows (WDDM) usually cannot give per-process graphics memory** (`[N/A]`): the panel then shows only the totals and says so. The desktop itself, a browser, a game, Ollama, LM Studio or Unsloth Desktop may hold some of it; this program lists what the driver reports and never ends another program ("close it yourself").
* The free RAM number includes Windows' file cache: that is not a leak.

## What "Free GPU" ends (only what THIS program owns)
1. Engine containers named `ndforge-vllm-*` (stopped and removed with the engine's own commands: docker, docker inside WSL, wslc `stop -t 30` / `remove -f`), also leftovers of a crashed run. A RUNNING container that a running job uses is refused.
2. Worker processes (the command line contains `ndforge_worker` AND the process descends from this program or is an orphan whose parent is gone; a worker of another running program is never touched; pids of other users or programs are never touched). A worker of a RUNNING job is refused unless the person ticks "Stop the running job too": the window then uses the normal Stop (state saved, Continue works), waits until the job has ended and only then frees.
3. Releasing the CUDA cache inside idle workers is **not built**: a worker is one process per job and ends with its job, so there is nothing idle to release; a leftover worker is ended instead.

## What "Free RAM" does
Ends the same idle or orphaned workers and, on Windows, trims the working set of this program's own processes (`EmptyWorkingSet`: Windows gives back what is not needed right now; it is read in again when needed; a no-op elsewhere). No purging of the standby list, no ending of system processes, no "RAM cleaner" tricks.
A separate extra with its own warning and confirmation, "Release the memory of the WSL virtual machine", runs `wsl --shutdown`: it ENDS EVERY WSL distribution, and is refused while an engine container exists or a job runs. (WSL keeps memory after vLLM; this gives it back.) The wslc equivalent is not known, so the same command is used.

## The rules (same pattern as Storage)
* `memory_free_plan {kind}` returns the exact list (opaque ids built from the kind, pid and command line / container name) and a **plan token**; the confirmation dialog lists it.
* `memory_free {kind, planToken, stopRunning}`: all arguments are required (no `Option` with a permissive default). It plans again and refuses a token that does not match the fresh plan ("the list has changed"). It does only what the plan lists; an item a running job uses stays undone and is reported.
* The result has the numbers before and after, what was ended, what was skipped and why, what failed and why, and the log lines. Every ended item is logged.
* The pure parsers (`nvidia-smi`, `ps`, PowerShell JSON, `/proc/meminfo`, `docker ps`, `wslc list -a --format json` as the owner's runs showed it) are tested with real sample outputs; the machine sits behind the `Env` trait, the tests use a fake process table and fake containers.

## Automatic use
Before a vLLM engine starts the job log has "GPU free before the engine start: X GB of Y GB; the engine takes Z GB". When a vLLM or GPU job fails for lack of free memory the Heal card shows "Free GPU first" next to the message. Nothing is ended without a click.

## Not verified
`nvidia-smi` per-process output on the owner's Windows PC (WDDM), `EmptyWorkingSet` and the WSL release on a real Windows machine, the real `wslc list` / `remove` on the owner's PC for this feature, the real window.
