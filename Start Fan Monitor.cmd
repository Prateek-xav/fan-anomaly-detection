@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py --open-browser
) else (
  python app.py --open-browser
)
if errorlevel 1 pause
