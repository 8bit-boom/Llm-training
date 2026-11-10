#!/bin/sh
# Install ND Lore Forge from a git clone (Linux):
#   git clone https://github.com/8bit-boom/Llm-training && cd Llm-training && ./install.sh
# Checks what is needed, fetches the `uv` tool into ./tools, builds the program and leaves ./run.sh to start it.
#   ./install.sh --check      only check the prerequisites
#   ./install.sh --desktop    also add an entry to the application menu
# To update: git pull && ./install.sh
set -e
cd "$(dirname "$0")"
ROOT=$PWD
CHECK=0; DESKTOP=0
for a in "$@"; do case "$a" in --check) CHECK=1 ;; --desktop) DESKTOP=1 ;; *) echo "unknown option $a"; exit 2 ;; esac; done

[ "$(uname -s)" = Linux ] || { echo "This script is for Linux. On Windows use install.ps1 (or the installer from the Releases page)."; exit 2; }
[ "$(uname -m)" = x86_64 ] || { echo "Only x86_64 is supported (the training software needs an NVIDIA card)."; exit 2; }

missing=""
need() { command -v "$1" >/dev/null 2>&1 || missing="$missing $1"; }
need git; need curl; need tar; need cc
command -v cargo >/dev/null 2>&1 || { echo "Rust is missing. Install it from https://rustup.rs (one command), open a new terminal, and run this again."; missing="$missing cargo"; }
if command -v pkg-config >/dev/null 2>&1; then
  pkg-config --exists webkit2gtk-4.1 gtk+-3.0 librsvg-2.0 2>/dev/null || {
    echo "The window libraries are missing. On Ubuntu / Debian:"
    echo "  sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev build-essential curl pkg-config"
    missing="$missing window-libraries"; }
else
  missing="$missing pkg-config"
fi
command -v nvidia-smi >/dev/null 2>&1 || echo "Note: nvidia-smi was not found. The program starts without it, but training needs the NVIDIA driver (https://www.nvidia.com/drivers)."
if [ -n "$missing" ]; then echo "Missing:$missing"; exit 1; fi
echo "Prerequisites are fine."
[ "$CHECK" = 1 ] && exit 0

echo "Fetching uv ..."
packaging/fetch_uv.sh linux "$ROOT/tools"
echo "Building the program (the first time takes a few minutes) ..."
cargo build --release -p nd-lore-forge --features custom-protocol
if [ "$DESKTOP" = 1 ]; then
  d="${XDG_DATA_HOME:-$HOME/.local/share}/applications"; mkdir -p "$d"
  printf '[Desktop Entry]\nType=Application\nName=ND Lore Forge\nComment=Teach a small AI model your nd-world world\nExec=%s/run.sh\nIcon=%s/src-tauri/icons/128x128.png\nTerminal=false\nCategories=Utility;\n' "$ROOT" "$ROOT" > "$d/nd-lore-forge.desktop"
  echo "Added to the application menu."
fi
echo
echo "Done. Start it with:  $ROOT/run.sh"
echo "First start: open \"Training setup\" and press \"Install the training software\"."
