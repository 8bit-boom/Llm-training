#!/bin/sh
# Downloads the `uv` installer tool (the version in packaging/UV_VERSION), checks its SHA-256 against the release's own checksum file and puts it in <dest>.
# usage: packaging/fetch_uv.sh <linux|windows> <dest folder>
set -e
OS=${1:-linux}
DEST=${2:?usage: fetch_uv.sh <linux|windows> <dest folder>}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
UV=$(cat "$ROOT/packaging/UV_VERSION")
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
mkdir -p "$DEST"
if [ "$OS" = linux ]; then
  tar -xzf "$TMP/$ASSET" -C "$TMP"; cp "$TMP"/uv-*/uv "$DEST/uv"; chmod +x "$DEST/uv"
else
  (cd "$TMP" && unzip -q "$ASSET"); cp "$TMP/uv.exe" "$DEST/uv.exe"
fi
rm -rf "$TMP"
echo "uv $UV is in $DEST"
