#!/bin/sh
# Remakes ui/guide/*.png: a demo project (made-up world "Ashfall Reach"), a stand-in nvidia-smi that reports an RTX 5090, the real page data, a headless browser.
# Needs: cargo, node with playwright (NODE_PATH), the Rust toolchain. Run from the repository root.
set -e
T=$(mktemp -d)
mkdir -p "$T/bin" "$T/cfg" "$T/projects"
printf '#!/bin/sh\necho "NVIDIA GeForce RTX 5090, 32607, 576.80, 12.0"\n' > "$T/bin/nvidia-smi"; chmod +x "$T/bin/nvidia-smi"
export PATH="$T/bin:$PATH" NDFORGE_CONFIG_DIR="$T/cfg" NDFORGE_PROJECTS_DIR="$T/projects" NDFORGE_ENV_DIR="$T/env" NDFORGE_FREE_DISK_GB=400
P=$(cargo run -q -p ndforge-core --example demo_project -- "$T/projects" dataset trained)
cargo run -q -p ndforge-core --example guide_states -- "$P" > "$T/states.json"
mkdir -p ui/guide
NODE_PATH=${NODE_PATH:-$(npm root -g)} node scripts/guide_shots.js "$T/states.json" ui/guide
rm -rf "$T"
