# Install ND Lore Forge from a git clone (Windows, PowerShell):
#   git clone https://github.com/8bit-boom/Llm-training; cd Llm-training; powershell -ExecutionPolicy Bypass -File install.ps1
# Checks what is needed, fetches the `uv` tool into .\tools, builds the program and leaves run.bat to start it.
#   -Check      only check the prerequisites
#   -Shortcut   also put a shortcut on the desktop
# To update: git pull; then run this again.
param([switch]$Check, [switch]$Shortcut)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$root = $PSScriptRoot

$missing = @()
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $missing += "git (https://git-scm.com/download/win)" }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { $missing += "Rust (https://rustup.rs; it also asks for the Visual Studio C++ Build Tools)" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Write-Host "Note: nvidia-smi was not found. The program starts without it, but training needs the NVIDIA driver (https://www.nvidia.com/drivers)." }
if ($missing.Count -gt 0) { Write-Host "Missing:"; $missing | ForEach-Object { Write-Host "  $_" }; exit 1 }
Write-Host "Prerequisites are fine."
if ($Check) { exit 0 }

$uvVersion = (Get-Content packaging\UV_VERSION -Raw).Trim()
$asset = "uv-x86_64-pc-windows-msvc.zip"
$base = "https://github.com/astral-sh/uv/releases/download/$uvVersion"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("ndforge-uv-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
Write-Host "Fetching uv ..."
Invoke-WebRequest "$base/$asset" -OutFile "$tmp\$asset"
$sumText = (Invoke-WebRequest "$base/$asset.sha256").Content
if ($sumText -is [byte[]]) { $sumText = [Text.Encoding]::UTF8.GetString($sumText) }   # served as a binary download: bytes, not text
$want = (($sumText.ToString().Trim()) -split "\s+")[0].ToLower()
$got = (Get-FileHash "$tmp\$asset" -Algorithm SHA256).Hash.ToLower()
if ($want -ne $got) { throw "uv checksum mismatch: wanted $want, got $got" }
Expand-Archive "$tmp\$asset" -DestinationPath "$tmp\x"
New-Item -ItemType Directory -Force -Path "$root\tools" | Out-Null
Copy-Item (Get-ChildItem "$tmp\x" -Recurse -Filter uv.exe | Select-Object -First 1).FullName "$root\tools\uv.exe" -Force
Remove-Item $tmp -Recurse -Force

Write-Host "Building the program (the first time takes several minutes) ..."
cargo build --release -p nd-lore-forge --features custom-protocol
if ($LASTEXITCODE -ne 0) { throw "The build failed." }
if ($Shortcut) {
  $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "ND Lore Forge.lnk"
  $s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
  $s.TargetPath = "$root\target\release\nd-lore-forge.exe"; $s.WorkingDirectory = $root; $s.IconLocation = "$root\src-tauri\icons\icon.ico"; $s.Save()
  Write-Host "Shortcut added to the desktop."
}
Write-Host ""
Write-Host "Done. Start it with run.bat (double-click it)."
Write-Host "First start: open `"Training setup`" and press `"Install the training software`"."
