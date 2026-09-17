@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
echo 05_PHASE3_DRO_BOTTLENECK_PRIORITIZATION: analysis

.venv\Scripts\python.exe scripts\run_phase3.py
if errorlevel 1 exit /b 1

echo Analysis complete.
exit /b 0
