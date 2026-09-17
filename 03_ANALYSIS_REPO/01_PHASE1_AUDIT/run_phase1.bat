@echo off
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Phase 1: BPI 2019 data audit

.venv\Scripts\python.exe scripts\01_audit_xes_stream.py
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe scripts\02_write_report.py
if errorlevel 1 exit /b 1

echo Analysis complete.
exit /b 0
