@echo off
setlocal
set "PROJECT=%~dp0"
if not exist "%PROJECT%.venv\Scripts\pythonw.exe" (
    echo Set up Python 3.12 and install requirements.txt as described in README.md.
    pause
    exit /b 1
)
start "" /D "%PROJECT%" "%PROJECT%.venv\Scripts\pythonw.exe" "%PROJECT%main.py"
