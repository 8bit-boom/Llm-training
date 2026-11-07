#!/bin/sh
# Builds the Linux AppImage (and a .deb): dist/ND-Lore-Forge-<version>.AppImage. Needs: cargo, `cargo install tauri-cli --version "^2"`, and the Tauri system
# libraries (libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev patchelf), plus curl and tar.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
packaging/stage.sh linux
(cd src-tauri && cargo tauri build --config tauri.release.conf.json --bundles appimage,deb)
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
mkdir -p dist
cp target/release/bundle/appimage/*.AppImage "dist/ND-Lore-Forge-$VERSION.AppImage"
cp target/release/bundle/deb/*.deb "dist/ND-Lore-Forge-$VERSION.deb" 2>/dev/null || true
ls -la dist
