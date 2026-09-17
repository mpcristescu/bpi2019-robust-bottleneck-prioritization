@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Phase 1B: temporal and resource audit

.venv\Scripts\python.exe scripts\01_temporal_resource_audit.py
if errorlevel 1 exit /b 1

echo Analysis complete.
exit /b 0
