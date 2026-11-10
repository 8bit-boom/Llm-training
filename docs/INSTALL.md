# Installing ND Lore Forge

The in-app **Guide** (📖 in the left column) has the same walkthrough with pictures; this page is for people who build or hand out the installers.

## For users
- **Windows 11 / 10:** run `ND-Lore-Forge-Setup-<version>.exe`. Installs for your user (no administrator needed); the dialog offers "all users". It contains the program, the training worker and `uv`. The window is drawn by Microsoft Edge WebView2 (the installer tells you if it is missing).
- **Linux (Ubuntu 24.04 or similar):** `chmod +x ND-Lore-Forge-<version>.AppImage` and start it (`sudo apt install libfuse2` if it asks for FUSE). A `.deb` is also built.
- First start: **⚙ Training setup → Install the training software** (several GB, one time; needs internet), then **Test the graphics card**.
- Where things are: projects in *Documents/ND Lore Forge* (Windows) or the home folder; settings, access keys (`secrets.json`, readable by you only, not encrypted) and the Python environment in the user's settings folder (`%APPDATA%\nd-lore-forge` or `~/.config/nd-lore-forge`). Uninstalling the program leaves projects and the training software; delete the settings folder to remove the latter.

## Install from a git clone
For people who prefer to build it themselves (or want to follow the latest changes).
- **Linux:** `git clone https://github.com/8bit-boom/Llm-training && cd Llm-training && ./install.sh`, then `./run.sh`. `./install.sh --check` only checks the prerequisites; `--desktop` adds an application-menu entry.
- **Windows:** `git clone https://github.com/8bit-boom/Llm-training`, then in the folder double-click **`install.bat`** (checks git and Rust, fetches `uv`, builds the program; the first time takes several minutes), then **`run.bat`** to start it. **`update.bat`** later: `git pull`, then the same build. The batch files start `install.ps1` for you; from PowerShell the same thing is `powershell -ExecutionPolicy Bypass -File .\install.ps1` (`-Check` only checks; `-Shortcut` adds a desktop shortcut). Note the `.\`: PowerShell does not run a script from the current folder by name alone.
- **You need:** git, [Rust](https://rustup.rs) (on Windows also the Visual Studio C++ Build Tools, which rustup asks for), and on Linux the window libraries: `sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev build-essential curl pkg-config`. The script tells you what is missing.
- **What it does:** downloads `uv` (checksum-verified) into `tools/`, then `cargo build --release -p nd-lore-forge --features custom-protocol` (the feature embeds the window's files in the program). The program finds the worker in `worker/` and `uv` in `tools/` inside the clone, so keep the clone where it is.
- **Update:** `git pull` and run the install script again (Windows: `update.bat`). Then, in the program, **Training setup** installs the Python training software as usual (first start only).
- macOS is not supported (no NVIDIA graphics cards).

## For whoever builds the installers
Both builds are in `.github/workflows/release.yml` (tag `v*` or run by hand) and can be run locally:
- **Linux:** `cargo install tauri-cli --version "^2" --locked`, install the Tauri system libraries (see `ci.yml`), then `packaging/linux/build.sh` -> `dist/ND-Lore-Forge-<version>.AppImage` and `.deb`.
- **Windows:** Rust, `cargo install tauri-cli --version "^2" --locked`, Git for Windows, Inno Setup 6, then `powershell -File packaging\windows\build.ps1` -> `dist\ND-Lore-Forge-Setup-<version>.exe`.
- `packaging/stage.sh <linux|windows>` first copies a clean worker and downloads `uv` (version in `packaging/UV_VERSION`, SHA-256 checked against the release's own checksum file) into `dist/stage/`. The installed program finds them next to itself or in its resource folder (`crates/ndforge-core/src/envmgr/mod.rs`, `resource_dirs_for`; covered by `tests/install.rs`).
- The installers are **not code-signed**: Windows SmartScreen will warn ("More info → Run anyway"). Signing needs a certificate.
- Guide pictures: `scripts/make_guide_screenshots.sh` remakes `ui/guide/*.png` from the real pages with a demo world.

## Without Unsloth, with a model you already have
All you need is this program and a model. **⚙ Training setup → "Install without Unsloth"** installs only the standard libraries (Transformers, PEFT, TRL): the program fetches Python, PyTorch and these itself with `uv`, nothing else has to be installed by hand. A model is either downloaded from Hugging Face the first time (the five built-in ones, or any open model added in Step 3) or taken from a folder on this computer (Step 3, *Add a model from Hugging Face…* → *Choose a folder…*). Without Unsloth training is slower and can use more graphics memory; *Training engine* in Expert mode chooses per run (auto, unsloth, standard).
