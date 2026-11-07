# Installing ND Lore Forge

The in-app **Guide** (📖 in the left column) has the same walkthrough with pictures; this page is for people who build or hand out the installers.

## For users
- **Windows 11 / 10:** run `ND-Lore-Forge-Setup-<version>.exe`. Installs for your user (no administrator needed); the dialog offers "all users". It contains the program, the training worker and `uv`. The window is drawn by Microsoft Edge WebView2 (the installer tells you if it is missing).
- **Linux (Ubuntu 24.04 or similar):** `chmod +x ND-Lore-Forge-<version>.AppImage` and start it (`sudo apt install libfuse2` if it asks for FUSE). A `.deb` is also built.
- First start: **⚙ Training setup → Install the training software** (several GB, one time; needs internet), then **Test the graphics card**.
- Where things are: projects in *Documents/ND Lore Forge* (Windows) or the home folder; settings, access keys (`secrets.json`, readable by you only, not encrypted) and the Python environment in the user's settings folder (`%APPDATA%\nd-lore-forge` or `~/.config/nd-lore-forge`). Uninstalling the program leaves projects and the training software; delete the settings folder to remove the latter.

## For whoever builds the installers
Both builds are in `.github/workflows/release.yml` (tag `v*` or run by hand) and can be run locally:
- **Linux:** `cargo install tauri-cli --version "^2" --locked`, install the Tauri system libraries (see `ci.yml`), then `packaging/linux/build.sh` -> `dist/ND-Lore-Forge-<version>.AppImage` and `.deb`.
- **Windows:** Rust, `cargo install tauri-cli --version "^2" --locked`, Git for Windows, Inno Setup 6, then `powershell -File packaging\windows\build.ps1` -> `dist\ND-Lore-Forge-Setup-<version>.exe`.
- `packaging/stage.sh <linux|windows>` first copies a clean worker and downloads `uv` (version in `packaging/UV_VERSION`, SHA-256 checked against the release's own checksum file) into `dist/stage/`. The installed program finds them next to itself or in its resource folder (`crates/ndforge-core/src/envmgr/mod.rs`, `resource_dirs_for`; covered by `tests/install.rs`).
- The installers are **not code-signed**: Windows SmartScreen will warn ("More info → Run anyway"). Signing needs a certificate.
- Guide pictures: `scripts/make_guide_screenshots.sh` remakes `ui/guide/*.png` from the real pages with a demo world.
