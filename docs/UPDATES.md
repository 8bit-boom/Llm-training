# Updates: how the program updates itself

ND Lore Forge checks its own public GitHub releases (`8bit-boom/Llm-training`, no login, no token), downloads the installer for this system, verifies it and starts it
**only after you click "Download and install"**. Code: `crates/ndforge-core/src/update/`; window: the **Updates** card on the *Training setup* page and the
"Update available" chip in the footer; commands: `update_status`, `update_check`, `update_install`, `update_cancel`.

> **Status.** `release.yml` has never run, and the installer swap has not been tried on a real Windows PC. Nothing here has been exercised against a real release yet (see "What is not verified").

## What happens

1. **Check.** `GET /repos/8bit-boom/Llm-training/releases/latest`. Drafts and prereleases (also a tag like `v1.0.0-rc.1`) are ignored. The release's version is compared with the program's (semantic versions).
   Automatic check at start: only when "Check at start" is on (default), at most once in 24 hours, in the background; the start never waits for it, and a failure at start is silent (the button shows the message). Offline, a GitHub rate limit
   (HTTP 403/429) and a bad answer each give a plain message. The result is kept in `update-state.json` (config folder) so the chip shows an update found earlier even when no check is due.
2. **Asset.** Windows: `ND-Lore-Forge-Setup-<version>.exe`. Linux: `ND-Lore-Forge-<version>.AppImage` (the `.deb` is never picked). Another system: the release is announced but there is nothing to install.
3. **Download.** Into a folder of its own (Windows: `%TEMP%\ndforge-update\<version>`; Linux: `~/Downloads/ND-Lore-Forge-updates/<version>`), with progress and a Cancel button. Only GitHub hosts are contacted; redirects are followed by hand inside that list.
4. **Verify.** The SHA-256 must match the line for the installer in the release's `SHA256SUMS.txt` (a missing file or line stops the update before the installer is downloaded). Then the signature, if a key is built in (below). A file that fails is deleted.
5. **Install, after your click and only while idle.** The update is refused, with the reason, while a job (training, heal, export, quant check ...), an Auto run or a queue run is active, or while the queue has waiting entries that would start by themselves.
   - **Windows:** runs the Inno Setup installer with `/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /NORESTART` and closes the program a moment later; the installer replaces it in place (same `AppId`, so same folder and same per-user / all-users choice) and starts it again. Your projects and settings are not in the install folder.
   - **Linux:** the program does not replace itself. The verified AppImage is made executable and its folder is opened; close the program and start the new file. (Replacing a running AppImage was left out on purpose.)
6. **A failure changes nothing.** A failed check, download, verification or launch leaves the installed version as it is and shows a plain message.

## Signed or checksum only

* **Signed** (a public key is built into the program): an update is installed only if the checksum matches **and** the signature `<installer>.sig` is valid for that key. A missing or bad signature **refuses** the install. A matching checksum never replaces a missing signature.
* **Checksum only** (the default until you make a key): no key is built in, so the checksum is the only proof. It catches a damaged or incomplete download, **not** a tampered release (an attacker who can change the installer on GitHub can change `SHA256SUMS.txt` too). The card says this on screen in this mode; the checksum must still match.
* A key text that is built in but is not a valid key puts the program in a third state where **no** update is installed (fail closed).

### Signature format

`<installer>.sig` is a text file with the **base64** (standard alphabet) of the 64-byte ed25519 signature. The signed message is the ASCII text `ndforge-update-v1:` followed by the 64 lowercase hex characters of the installer's SHA-256.
The public key is 32 raw bytes as base64 (44 characters); the private key file is base64 of the 32-byte seed.

## Set up signing (once, on your own PC)

1. `pip install cryptography`, then `python3 scripts/make_update_key.py` (add `--out PATH` to choose the file; it never overwrites). It prints the **public key** and writes the **private key** to `update-signing-key.private.txt` (readable by you only; never printed).
2. GitHub: *Settings -> Secrets and variables -> Actions -> New repository secret*, name `UPDATE_SIGNING_KEY`, value: the single line in that file. Keep the file somewhere safe and **outside the repository**; if it is lost, make a new pair and ship a release that carries the new public key (users on the old key must install that one by hand).
3. Paste the public key into `UPDATE_PUBLIC_KEY` in `crates/ndforge-core/src/update/key.rs`, commit, and release. Versions built **before** this change have no key and stay in checksum-only mode; they still update, to the first release that has the key. After that, every installer must be signed.
4. Without the secret, `release.yml` skips signing, writes `UNSIGNED-UPDATES.txt` and adds a note to the release text. A program that has a key will then refuse that release.

Test it without CI: `python3 scripts/sign_update.py --key-file update-signing-key.private.txt dist/ND-Lore-Forge-Setup-0.2.0.exe` writes the `.sig` next to it.

## Cut a release

1. Change the version in **one place**: `version` under `[workspace.package]` in `Cargo.toml` (`src-tauri/tauri.conf.json` has no version of its own; Tauri, the Rust crates, the footer, the update check and the Inno Setup build all read it).
2. Commit, then tag exactly `v<version>` (for example `v0.2.0`) and push the tag. `release.yml` starts with a job that **fails the release if the tag differs** from the version (`scripts/check_release_version.py`, also run by the worker tests against the repository).
3. The workflow builds both installers, signs them (when the secret exists), and attaches the installers, the `.sig` files and `SHA256SUMS.txt` to the release with the list of changes as its text. Publish it as a normal release: drafts and prereleases are invisible to the updater.

## How users get updates

Open the program: within a day of a release the footer shows **Update available: x.y.z**; the *Training setup* page has the card with the notes and the button (**Check for updates** looks right now). Switch off "Check at start" there to stop the automatic look.

## Roll back

Installers are never deleted from GitHub. Download the older `ND-Lore-Forge-Setup-<old>.exe` from its release page and run it: Inno Setup replaces the newer version in place (the same `AppId`). On Linux keep the old AppImage and start that one. Projects and settings are separate from the program and stay.
The updater will offer the newer version again at the next check; switch off "Check at start" to stay on the old one.

## What is not verified

* `release.yml` (including the new version job and the signing step) has never run.
* The silent upgrade on a real Windows PC (Inno flags, per-user versus all-users, restart of the program) is built from Inno Setup's documentation and not tried.
* The signing flow with a real key; the tests use a key made inside the test, and the scripts are tested against the program's verifier locally.
* The Linux AppImage path (only the "show the file" step is implemented and tested with a stand-in).
