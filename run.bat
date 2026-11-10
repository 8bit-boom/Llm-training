@echo off
rem Starts ND Lore Forge from this folder (after install.bat).
setlocal
cd /d "%~dp0"
if not exist "%~dp0target\release\nd-lore-forge.exe" (
  echo Not built yet: run install.bat first.
  pause
  exit /b 1
)
start "" "%~dp0target\release\nd-lore-forge.exe"
exit /b 0
