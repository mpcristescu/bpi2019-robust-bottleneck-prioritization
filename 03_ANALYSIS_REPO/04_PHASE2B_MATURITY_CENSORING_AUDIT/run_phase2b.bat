@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Phase 2B: maturity and censoring audit

.venv\Scripts\python.exe scripts\01_maturity_censoring_audit.py
if errorlevel 1 exit /b 1

echo Analysis complete.
exit /b 0
