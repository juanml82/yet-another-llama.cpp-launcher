@echo off
setlocal
REM ============================================================
REM  llama-launcher.bat — Windows equivalent of llama-launcher.sh
REM
REM  Starts the Llama.cpp / ik_llama launcher GUI (app.py).
REM  Runs from its own folder and uses the local "venv".
REM ============================================================

cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo [launcher] Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting app.py...
python app.py

endlocal
