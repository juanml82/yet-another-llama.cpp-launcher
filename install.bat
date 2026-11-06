@echo off
setlocal
REM ============================================================
REM  install.bat — Windows equivalent of install.sh
REM
REM  On Windows GTK/PyGObject is not available, so pywebview falls
REM  back to Qt WebEngine (installed via requirements-windows.txt).
REM  No system packages need to be installed.
REM ============================================================

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [install] Python was not found on PATH.
    echo [install] Install Python 3 from https://www.python.org/downloads/
    echo [install] and make sure "Add python.exe to PATH" is checked.
    exit /b 1
)

python -m venv venv
if errorlevel 1 (
    echo [install] Failed to create the virtual environment.
    exit /b 1
)

call venv\Scripts\activate.bat

python -m pip install --upgrade pip
if errorlevel 1 (
    echo [install] Failed to upgrade pip.
    exit /b 1
)

python -m pip install -r requirements-windows.txt
if errorlevel 1 (
    echo [install] Failed to install the Python packages.
    exit /b 1
)

echo.
echo [install] Done. You can now run:  llama-launcher.bat
endlocal
