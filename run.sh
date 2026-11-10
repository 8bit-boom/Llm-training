#!/bin/sh
# Starts ND Lore Forge from a git clone (after ./install.sh).
cd "$(dirname "$0")"
[ -x target/release/nd-lore-forge ] || { echo "Not built yet: run ./install.sh first."; exit 1; }
exec target/release/nd-lore-forge "$@"
