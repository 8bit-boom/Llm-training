# Installing llama.cpp from the program

Everything that exports GGUF files through llama.cpp, the importance matrix tools, Compare quants (`llama-perplexity`), the mixed-precision sizes and the matrix merge need
llama.cpp. The owner's report was "llama.cpp was not found ... make so program installs it automatically". The program now does: one button, a size shown first,
a check that the programs start, and the worker finds the result without any setting.

Code: `crates/ndforge-core/src/envmgr/llamacpp/` (`choose.rs` the choice of the release file, `net.rs` GitHub, `archive.rs` safe unpacking, `mod.rs` the install, the status,
the finder), the Tauri commands `llamacpp_status / plan / install / cancel / remove / check_update` (`src-tauri/src/lib.rs`), the page (`ui/app.js`, "llama.cpp: the installer"),
the worker's one finder `worker/ndforge_worker/llamacpp.py`. Tests: `crates/ndforge-core/tests/llamacpp_choose.rs`, `llamacpp_install.rs`, `worker/tests/test_llamacpp.py`,
`ui/smoke/llamacpp.test.mjs`.

## What is installed where

`<the program's data folder>/llama.cpp/` (the folder that also holds `env/`: `%APPDATA%\NDLoreForge` on Windows, `~/.config/NDLoreForge` on Linux, `~/Library/Application Support/NDLoreForge`
on macOS; `NDFORGE_CONFIG_DIR` moves it). Exactly the layout the worker's finder accepts:

```
llama.cpp/
  convert_hf_to_gguf.py, convert_hf_to_gguf_update.py     from the source of the SAME release
  conversion/                                              (the current converter imports it)
  gguf-py/gguf/ (+ pyproject.toml, README.md, LICENSE)     the converter inserts gguf-py into sys.path
  models/templates/                                        chat templates the converter reads (conversion/base.py looks in ../models/templates)
  requirements.txt, requirements/                          for reference
  build/bin/llama-quantize(.exe), llama-imatrix, llama-perplexity, llama-cli ... + shared libraries / the CUDA runtime DLLs (Windows CUDA)
  install.json                                             {schema, tag, backend, cuda_version, assets, sha256, source_url, installed_at, os, arch, versions, converter_check, via}
```

The source subset was decided by reading the real `convert_hf_to_gguf.py` (master, 14.8 KB: `import torch`, `sys.path.insert(1, .../gguf-py)`, `from conversion import (...)`) and
`conversion/__init__.py` and `conversion/base.py` (imports only the standard library, numpy, torch, transformers, gguf, and reads `../models/templates/*.jinja`). Not taken: tests, docs,
the C++ sources, the large `models/ggml-vocab-*.gguf` files.

## How the choice is made

A pure function over the release's asset names and a description of the machine (`choose.rs`; tested over old and new naming, extra files and unknown lists).
Names are read by lower-case tokens (os, architecture, backend, the CUDA version in `12.4`, `cu12.2.0` or `cuda12.4` form); `-ci`, `kleidiai`, `hip`, `rocm`, `sycl`, `openvino`,
`opencl`, `kompute` and other architectures are never chosen automatically.

| Machine | Choice (automatic) |
|---|---|
| NVIDIA card, driver readable (Windows x64 and Linux x64) | the highest `...-cuda-<v>-x64` build whose **major** version is not above the major version the driver reports (`CUDA Version: 13.0` in the `nvidia-smi` header, or derived from the driver number): CUDA's minor-version compatibility lets the 13.5 build run on a driver that reports any 13.x, so the minor is informational. Real Windows builds are 12.4 and 13.5, Linux 12.8 and 13.5. An **RTX 50 card** (compute capability 10 or 12) needs a build and a driver of at least 12.8: with a 13.x driver it gets 13.5, with a 12.x driver it gets Vulkan and the plain note "a newer NVIDIA driver is needed for the CUDA build of this card". The runtime package is taken with it (Windows `cudart-llama-bin-win-cuda-<v>-x64.zip`, **no tag in its name**; Linux `cudart-llama-<tag>-bin-ubuntu-cuda-<v>-x64.tar.gz`); on Windows a version without one is skipped. Its DLLs / `.so` files are put next to the programs |
| the above without a usable CUDA file | Vulkan build (works on NVIDIA too), else the CPU build; the reason is shown in the confirmation |
| Windows, no NVIDIA card | Vulkan when an AMD or Intel card is present, else the CPU build (the new `win-cpu-x64`, or the best of the old `avx2` / `avx` / `avx512` / `noavx` / `openblas` ones) |
| Linux x64, no usable CUDA build | the Vulkan build when an AMD or Intel card is present, else the plain `ubuntu-x64` build. With an NVIDIA card the prebuilt `ubuntu-cuda-<v>` tar.gz is chosen by the rule above; **Build with CUDA** from the source stays an Expert-mode extra when `nvcc`, `cmake` and a C++ compiler are on the PATH |
| macOS arm64 / x64 | `macos-arm64` (Metal is part of it) / `macos-x64`; the `kleidiai` variant is not chosen |
| anything else (other architecture, FreeBSD ...) | a plain message with the manual steps |

If nothing matches, the error lists every asset name of the release and points to the manual steps. `.zip` is preferred over `.tar.gz`; both are read.
Expert mode shows a list (automatic, CUDA, Vulkan, processor only, build from the source) and the plan says which of them this release can offer.

## The install, step by step

1. **Release discovery:** `GET https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30` (newest first, prereleases included, drafts skipped; or `/releases/tags/<tag>` alone when `NDFORGE_LLAMA_CPP_TAG` pins one), with a User-Agent and no token. The FIRST release for which the asset choice succeeds is used; a release without a matching file is skipped. A `nightly-tag.txt` in a release that fits nothing is read (at most 512 bytes, allowed hosts, at most 3 of them) as a possible pointer to a tag; if its text looks like a tag the tag is looked up, otherwise it is ignored. If nothing matches, the error says how many releases were checked, the newest tag and its file names, and points to the manual steps. One page of 30 is enough.
2. The plan (tag, backend, reason, files, size) is shown first; **nothing is downloaded until the person presses "Download and install (NNN MB)"** (one click, no typing). The size is the
   files' sizes plus about 30 MB for the source zip (its size is not listed).
3. Downloads (only `github.com`, `codeload.github.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com`; https; redirects followed by hand and only inside this list;
   the proxy named by `HTTPS_PROXY` / `ALL_PROXY` / `HTTP_PROXY` is used, `NO_PROXY` is respected). The size is checked against the release listing, and the **sha256 against the
   `digest` the API lists** (a mismatch deletes the file and refuses it; no digest listed: the download is accepted and its sha256 recorded in `install.json`).
4. Unpacking into a side folder (`llama.cpp.new`): every entry name of an archive is checked before anything is written (`..`, absolute paths, drive letters, backslash tricks refuse the
   whole archive); links are only made when they stay inside the tree; only the executable bit is taken from the archive's permissions; the total size is capped.
5. The source subset (above) from the source zip of the same tag (`https://github.com/ggml-org/llama.cpp/archive/refs/tags/<tag>.zip`).
6. **Self-test** (30 s per call, runner injectable in the tests): `--help` of `llama-quantize`, `llama-imatrix`, `llama-perplexity` (llama-quantize exits with 1 after printing its usage; a start is
   judged by "it printed something and was not killed or stopped by the loader", not by the exit code), `--version` of the last two (recorded in `install.json`), and `convert_hf_to_gguf.py --help`
   with the training environment's Python when it exists (without torch the note says "not checked"; a torch import error is a note, any other failure refuses the install).
   A CUDA build that does not start ("error while loading shared libraries", a missing DLL, a driver that is too old) gives a plain message and **"Install the CPU build instead"** (one click,
   no second question).
7. Only now the new folder is renamed into place. The old install is kept until the new one is there and put back if the rename fails. A failed or cancelled install removes its side folder and
   leaves the old install untouched. Cancel works in every phase (the download loop, the unpacking loop, the self-test's programs, the cmake build).

Progress events go through the same channel as the environment installer (`env-event`, tagged `"task": "llamacpp"`): `step` (Looking up the release / Downloading the program files (NN MB) /
Downloading the CUDA runtime files / Downloading the converter / Unpacking / Checking that it starts / Putting it in place), `progress` (value, bytes, total), `log`.

## How the workers find it

One finder (`worker/ndforge_worker/llamacpp.py`; the Rust `llamacpp::find` mirrors it for the status line). A folder counts when it has `convert_hf_to_gguf.py` and `llama-quantize(.exe)` in
itself, `build/bin` or `build/bin/Release`. The places, in order:

1. `LLAMA_CPP_DIR` (always wins)
2. `NDFORGE_LLAMA_CPP_DIR` (the program sets it for every job when `<data folder>/llama.cpp` exists)
3. `~/llama.cpp`
4. a `llama.cpp` folder next to the program (the repository's parent)

`llama-imatrix` and `llama-perplexity` are looked for next to llama-quantize, in each of those folders and their build folders, then on the PATH. The converter script and llama-quantize are never
taken from the PATH. Export, importance matrices, the matrix merge and Compare quants all call this finder (`exportjob.find_llama_cpp`, `imatrix.find_llama_imatrix`, `quantcheck.find_perplexity`,
`imatrixjob.find_tools`).

A missing llama.cpp is an error with the code **`llama_cpp_missing`** (export, importance matrices, merge, Compare quants). The Rust side ends such an error text with the mark
`[llama_cpp_missing]`; the page strips it and shows **Install llama.cpp now** under the message. The old text stays for logs, with the search order corrected (PATH only for `llama-imatrix` and
`llama-perplexity`) and a sentence that the program can install it.

## Options

| What | How |
|---|---|
| Pin a release | environment variable `NDFORGE_LLAMA_CPP_TAG=b6000` |
| Choose the build | Expert mode on the Training setup page (automatic, CUDA, Vulkan, processor only, build from the source) |
| Build with CUDA (Linux) | needs `nvcc`, `cmake`, a C++ compiler; downloads the source zip, runs `cmake -DGGML_CUDA=ON` and builds `llama-quantize llama-imatrix llama-perplexity` (20 minutes or more; progress from the `[ 45%]` lines; Cancel kills it); git is not needed |
| Update | "Check for a newer release" says when there is one (a hint; nothing is installed by itself); **Update** then installs it like a first install |
| Remove | deletes only `<data folder>/llama.cpp` and only when it has `install.json` (a folder the program did not make is never deleted); a folder named in `LLAMA_CPP_DIR` is never touched |
| Hidden settings for tests / a mirror on this computer | `NDFORGE_LLAMA_CPP_API_URL`, `NDFORGE_LLAMA_CPP_WEB_URL`; only `http://127.0.0.1...` or `localhost` is accepted, anything else is ignored |

## Security rules

No token and no login are needed or sent. Only the hosts above, only https (a local test server excepted). Nothing downloaded is run except the self-test calls (`--help`, `--version`, the converter's
`--help`). Unsigned programs: Windows SmartScreen or an antivirus may warn. Nothing needs administrator rights. Nothing outside `<data folder>/llama.cpp`, its side folders
(`llama.cpp.new`, `llama.cpp.old`) and the failure note `llama.cpp-last-failure.json` is written or deleted.

## Manual alternative

Download a llama.cpp release build and the source (or clone it), make a folder with `convert_hf_to_gguf.py`, `conversion/`, `gguf-py/` and the programs in `build/bin/`, and set the environment
variable `LLAMA_CPP_DIR` to it. It wins over the installed copy. To get rid of the installed copy use **Remove**, or delete `<data folder>/llama.cpp`.

## Deviations

- **Changed after the owner's first run (release discovery, asset names, CUDA rule).** The first version asked `GET /releases/latest`. llama.cpp's publish workflow creates every build release as a *prerelease*, and `/releases/latest` never returns prereleases, so it answered an unrelated release whose only file is `nightly-tag.txt` ("This llama.cpp release has no file for this computer"). Now the list of the newest 30 releases is read and the first one with a fitting file is used (above). The asset names and the CUDA rule were also changed to match the real ones from `release-publish.yml`: Windows CUDA builds are 12.4 and 13.5 with tag-less `cudart-llama-bin-win-cuda-<v>-x64.zip`; Linux has `ubuntu-cuda-12.8` / `13.5` tar.gz with `cudart-llama-<tag>-...` tars; the rule "highest version not above the driver's" became "highest major not above the driver's major" (an RTX 5090 with a 13.x driver now gets the 13.5 build, with a 12.x driver Vulkan plus a note to update the driver). Linux NVIDIA machines get the prebuilt CUDA tar instead of only the source-build option.

- The brief asked for PATH to be used "for imatrix only". The existing code also used the PATH for `llama-perplexity` (Compare quants); that behaviour is kept (the brief said to keep the
  other behaviour unchanged) and the messages now say "llama-imatrix and llama-perplexity are also looked for on the PATH (the converter script is not)".
- `.tar.gz` assets are read as well as `.zip` (the project's newer Linux and macOS releases are, as far as I remember, `.tar.gz`; the zip crate cannot read them, so a small reader on the
  already-present `flate2` was written; `flate2` became a direct dependency, it was already in `Cargo.lock`).
- No `proxy-from-env` feature of `ureq` (it would also have changed the proxy behaviour of the local Ollama / Unsloth connections): the proxy variables are read by hand for the GitHub downloads only.
- The machine-readable code travels as a mark at the end of the error text (`[llama_cpp_missing]`), because the program's error type is a plain string; the worker's own error line has the
  real `code` field.
- The exit code of `llama-quantize --help` is 1; the self-test therefore judges a start by its output and by loader errors, not by the exit code.
- The failure that offers the CPU build is remembered in `llama.cpp-last-failure.json` (the task channel only carries a text); a new install clears it.
- The mixed-size *plan* (`mixed_plan`) reads only a GGUF header and needs no llama.cpp; the notice is shown for the export, where the mixed sizes are made.
- No GPU, GitHub, real binaries or torch were available: see below.

## Not verified yet

Nothing in this list has been run; none of it is claimed to work.

- **The real API answers.** api.github.com is blocked where this was written, so the shape of `GET /releases?per_page=30` (the fields `tag_name`, `draft`, `assets[].name/size/browser_download_url/digest`) is from GitHub's documentation, not seen; the asset names now follow llama.cpp's `release-publish.yml` (read from raw.githubusercontent.com) but no real release list was fetched. Whether `nightly-tag.txt` always holds a bare tag is unknown (it is only used as an optional pointer).
- **The earlier asset-name note.** The names in the older tests are from memory (`llama-<tag>-bin-win-cuda-12.4-x64.zip`, `cudart-llama-bin-win-cuda-12.4-x64.zip`, `...-win-vulkan-x64.zip`, `...-win-cpu-x64.zip`,
  `...-ubuntu-x64`, `...-macos-arm64`, and whether Linux / macOS are `.zip` or `.tar.gz`). GitHub's API was blocked where this was written; the real list of a release is the first thing to check
  (`NDFORGE_LLAMA_CPP_TAG` plus the plan button prints the choice without downloading).
- **CUDA variants for Blackwell.** Whether the `cuda-13.5` build (and the Linux `cuda-12.8` / `13.5` ones) runs on an RTX 5090 with a driver reporting 13.x, and that the minor-version compatibility claim holds for these builds; whether the `cudart` package's DLLs are
  all that the build needs.
- **The driver's CUDA detection** (`CUDA Version:` in the `nvidia-smi` header; the driver-number table is from NVIDIA's release notes) on a real Windows machine.
- **Whether the source subset is enough** for the converter in every release (a script that imports something new would fail the converter check; the fallback would be to unpack the whole source
  tree, which is not implemented). The converter check needs torch and has not been run.
- **Antivirus / SmartScreen** behaviour with the downloaded programs and with files unpacked by this program (no Mark-of-the-Web is written).
- **Linux:** whether the prebuilt Ubuntu build runs on the person's distribution (shared libraries such as `libgomp`, `libvulkan`; the zip's rpath); a Vulkan build needs a Vulkan driver.
- **macOS:** everything (the Metal build, Gatekeeper's quarantine of the downloaded programs).
- **Proxy behaviour** (HTTP proxies with a login are refused with a message; TLS interception with a company certificate is not handled: the downloads use the built-in root certificates).
- **Rate limits** of the unauthenticated API (60 requests per hour per address) are only simulated.
- **The Linux CUDA build from the source:** only the command sequence and the progress parsing are tested (with a stand-in `cmake`); no real build was run.
- **The window** was tested in a stand-in browser, not in the real application (`cargo check -p nd-lore-forge` passes; the program was not started).
