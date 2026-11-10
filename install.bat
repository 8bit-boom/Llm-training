@echo off
rem Installs ND Lore Forge from this folder (a git clone): checks git and Rust, fetches uv, builds the program.
rem Double-click it, or run it in a terminal. Start the program afterwards with run.bat.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo.
  echo The install did not finish. Read the message above, fix what it says, and run install.bat again.
  pause
  exit /b 1
)
echo.
pause
exit /b 0
