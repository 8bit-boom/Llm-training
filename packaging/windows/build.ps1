# Builds the Windows installer: dist\ND-Lore-Forge-Setup-<version>.exe. Needs: Rust, `cargo install tauri-cli --version "^2"`, Git for Windows (for sh/curl/unzip), Inno Setup 6 (iscc on PATH).
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $root
sh packaging/stage.sh windows
Push-Location src-tauri
cargo tauri build --no-bundle --config tauri.release.conf.json
Pop-Location
$version = (Select-String -Path Cargo.toml -Pattern '^version = "(.*)"' | Select-Object -First 1).Matches.Groups[1].Value
iscc "/DAppVersion=$version" packaging\windows\ndforge.iss
Get-ChildItem dist
