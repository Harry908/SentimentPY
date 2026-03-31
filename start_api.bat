@echo off
setlocal

REM Ensure commands run from this script's directory
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    echo Create it first: python -m venv .venv
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo Starting API at http://127.0.0.1:8000
uvicorn app.main:app --reload

endlocal
