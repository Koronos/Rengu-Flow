@echo off
rem Run rengu via the uv-managed .venv on native Windows (install uv: https://docs.astral.sh/uv/)
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo rengu: uv is required. Install from https://docs.astral.sh/uv/ >&2
  exit /b 1
)

set "RENGU=%~dp0.venv\Scripts\rengu.exe"
if not exist "%RENGU%" (
  echo ==^> Setting up project environment ^(uv sync --inexact^)...
  uv sync --inexact
  if errorlevel 1 exit /b 1
)

"%RENGU%" %*
