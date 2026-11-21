# Memory (the line at the bottom)

## What it shows
The status line at the bottom shows how much **graphics memory** (GPU) and **RAM** are free, and how much this program itself uses. It updates every few seconds, and about every 2 seconds while a job runs. Click it for the details: the engine containers this program started, and the other programs that hold graphics memory (only listed when the driver says so; on Windows it often only gives the totals).

## Free GPU and Free RAM
- **Free GPU** ends only what *this program* started: its vLLM engine containers (also leftovers of a crashed run) and its idle or leftover worker processes. A running job keeps its worker and its engine unless you tick *Stop the running job too* (it is then stopped the normal way: its state is saved and Continue works).
- **Free RAM** ends the same leftover workers and, on Windows, asks Windows to take back the memory this program is not using right now. It never ends other programs and does no "RAM cleaning": Windows manages its file cache itself, and the free number includes that cache.
- **Release WSL memory** (an extra, with its own warning) runs `wsl --shutdown`: WSL keeps memory after vLLM, this gives it back, but it **ends every WSL distribution**. It is refused while an engine exists or a job runs.
- Every button first shows exactly what will be ended, and nothing happens before you confirm. When something was left alone or could not be done, the result says why. Other programs (a browser, a game, Ollama, LM Studio, Unsloth Desktop) are only listed: close them yourself.

## When a job says the card is too full
The message offers *Free GPU first*. It never ends anything by itself; the only thing the program removes on its own is an engine container of its own that an earlier crashed run left behind, at the start of a new vLLM job.
