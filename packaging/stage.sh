#!/bin/sh
# Stages what an installed program needs next to it: a clean copy of the Python worker and the `uv` installer tool (checksum-verified).
# usage: packaging/stage.sh <linux|windows>   -> dist/stage/worker, dist/stage/tools/uv[.exe]
set -e
OS=${1:-linux}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/dist/stage"
rm -rf "$OUT"; mkdir -p "$OUT/worker" "$OUT/tools"
cp -r "$ROOT/worker/ndforge_worker" "$OUT/worker/"
cp "$ROOT/worker/pyproject.toml" "$OUT/worker/"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
"$ROOT/packaging/fetch_uv.sh" "$OS" "$OUT/tools"
echo "staged in $OUT"
