#!/bin/sh
# Stages what an installed program needs next to it: a clean copy of the Python worker and the `uv` installer tool (checksum-verified).
# usage: packaging/stage.sh <linux|windows>   -> dist/stage/worker, dist/stage/tools/uv[.exe]
set -e
OS=${1:-linux}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/dist/stage"
UV=$(cat "$ROOT/packaging/UV_VERSION")
rm -rf "$OUT"; mkdir -p "$OUT/worker" "$OUT/tools"
cp -r "$ROOT/worker/ndforge_worker" "$OUT/worker/"
cp "$ROOT/worker/pyproject.toml" "$OUT/worker/"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
case "$OS" in
  linux) ASSET=uv-x86_64-unknown-linux-gnu.tar.gz ;;
  windows) ASSET=uv-x86_64-pc-windows-msvc.zip ;;
  *) echo "unknown OS $OS"; exit 1 ;;
esac
TMP=$(mktemp -d)
BASE="https://github.com/astral-sh/uv/releases/download/$UV"
curl -fsSL -o "$TMP/$ASSET" "$BASE/$ASSET"
WANT=$(curl -fsSL "$BASE/$ASSET.sha256" | cut -d' ' -f1)
GOT=$(sha256sum "$TMP/$ASSET" | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || { echo "uv checksum mismatch: wanted $WANT, got $GOT"; exit 1; }
if [ "$OS" = linux ]; then
  tar -xzf "$TMP/$ASSET" -C "$TMP"; cp "$TMP"/uv-*/uv "$OUT/tools/uv"; chmod +x "$OUT/tools/uv"
else
  (cd "$TMP" && unzip -q "$ASSET"); cp "$TMP/uv.exe" "$OUT/tools/uv.exe"
fi
rm -rf "$TMP"
echo "staged in $OUT"
