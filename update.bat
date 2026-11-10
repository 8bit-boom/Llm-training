@echo off
rem Updates ND Lore Forge: gets the newest version with git, then builds it again (install.bat).
setlocal
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed or not on the PATH: https://git-scm.com/download/win
  pause
  exit /b 1
)
echo Getting the newest version ...
(
  git pull --ff-only
  if errorlevel 1 (
    echo.
    echo The update could not be applied. If you changed files in this folder, put them aside or run "git status" and ask for help.
    pause
    exit /b 1
  )
  echo.
  call "%~dp0install.bat"
  exit /b
)
