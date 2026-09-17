@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Phase 2: process structure analysis

.venv\Scripts\python.exe scripts\01_build_primary_cohort.py
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe scripts\02_process_structure_analysis.py
if errorlevel 1 exit /b 1

echo Analysis complete.
exit /b 0
