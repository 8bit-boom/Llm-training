# Training on the TrueNAS server (the worker)

ND Lore Forge can train on the server's graphics card (for example the Tesla T10) instead of your PC. A small program, the **worker**, runs in a container
next to nd-world. The desktop program sends it the training data, shows the same live charts and sample answers as a local run, and copies the trained
model back to your PC when it is done.

> **Status:** the worker program and the protocol are tested (`cargo test -p ndforge-core --test remote`), including a real tiny training run through the real
> server binary. **The container image (`Dockerfile.worker`) has not been built yet** (no Docker daemon was available where it was written): the first build is
> the first test. If it fails, send me the error.

## What you need
- The NVIDIA driver on the server: **560 or newer** for the default image (`cu126`); 550 works if you build with `--build-arg TORCH_CUDA=cu124`. `nvidia-smi` must work on the host.
- A dataset folder for the worker, for example `/mnt/DeadPool/apps/nd-lore-forge-worker`: jobs, checkpoints and **downloaded models** (2 to 20 GB each) live there.
- A long random access token: `openssl rand -hex 24`.

## Install
1. Build the image (or use a published one): `docker build -f Dockerfile.worker -t nd-lore-forge-worker .` (10 to 30 minutes, several GB).
2. Copy `truenas-worker-compose.yml`, change `NDFORGE_TOKEN`, the volume path and the image name, and deploy it as a Custom App / compose project. It already contains the GPU
   block in the same form as nd-world's own compose file. Set `OLLAMA_URL` to nd-world's Ollama (for example `http://ollama:11434` on the same compose network) so the worker can free the card.
3. In ND Lore Forge: **Step 3 · Train → Where to train → Add a server**: name (TrueNAS), address (`http://<truenas-address>:8765`) and the token. Press **Test the connection**: you should see the card (for example
   *Tesla T10 · 15.6 GB · fp16 · free*).

## Training on it
- Choose the server in **Where to train**. The model cards now show what fits **that** card: on a 16 GB T10, Qwen3.5 2B/4B and Gemma 4 E2B fit with 16-bit LoRA; Gemma 4 E4B and Qwen3.5 9B need QLoRA (4-bit), which the plan suggests.
  The T10 has no bf16, so it trains in fp16.
- Press **Start training on TrueNAS**. Before it starts, the worker asks Ollama to unload its models so the card is free. **While training, AI chat, speech-to-text and image generation on that server wait or are
  unavailable** (they share the one card). Do not train on a game night.
- **Stop** saves a checkpoint after the current step. **Continue the last run** reconnects if the training is still running (for example after you closed the program), fetches the result if it finished, or restarts
  from the server's last checkpoint.
- Only **LoRA / QLoRA** run on the server. The finished adapter is copied to your PC (`runs/<name>/adapter`), where Step 5 turns it into a GGUF file as usual. The first run of each model downloads it **on the server**.

## Safety
- The worker is protected by the token only and speaks plain HTTP. Keep it on your home network; do not forward its port to the internet. For access from elsewhere use a VPN (for example Tailscale) or put it behind
  TrueNAS' own HTTPS proxy.
- Every request needs the token; job names and file names are checked so nothing can be written outside the data folder; trained models are unpacked on your PC with the same care.
- The token is stored on your PC in `secrets.json` (readable only by you, **not encrypted**; the operating system's keyring is a planned improvement). Remove a server from the program to delete its token.

## Settings (environment variables of the container)
`NDFORGE_TOKEN` (required, at least 16 characters), `NDFORGE_LISTEN` (default `0.0.0.0:8765`), `NDFORGE_DATA` (`/data`), `OLLAMA_URL` (optional), `NDFORGE_MAX_UPLOAD_MB` (2048), `HF_TOKEN` (only for models that need a login).

## If something goes wrong
- *"The training server did not answer"*: the container is not running, or the address or port is wrong.
- *"refused the access token"*: the token differs from `NDFORGE_TOKEN`.
- *"The server is busy"*: another run is still training there.
- *"no usable graphics card"*: the GPU block is missing from the compose file, or the driver is not installed on the host.
- *The run stops with "out of memory"*: choose QLoRA or a smaller model (the plan does this for you when the card was checked).
