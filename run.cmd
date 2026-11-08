@echo off
rem Starts ND Lore Forge from a git clone (after install.ps1).
if not exist "%~dp0target\release\nd-lore-forge.exe" (
  echo Not built yet: run install.ps1 first.
  pause
  exit /b 1
)
start "" "%~dp0target\release\nd-lore-forge.exe"
