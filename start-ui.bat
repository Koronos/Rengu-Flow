@echo off
REM Renga Flow UI — keeps this window open. Close the window to stop the server.
setlocal
cd /d "%~dp0"
title Renga Flow UI

echo.
echo ==========================================
echo   Renga Flow UI
echo   Close this window to stop the server.
echo ==========================================
echo.

where bash >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: bash not found.
  echo Install Git for Windows and use "Git Bash", or run from WSL.
  echo.
  pause
  exit /b 1
)

bash "%~dp0start-ui.sh" %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo.
  echo Finished with error code %EXITCODE%.
)

exit /b %EXITCODE%
