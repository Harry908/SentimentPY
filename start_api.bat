@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    echo Create it first: python -m venv .venv
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo.
echo API: http://127.0.0.1:8000
echo Docs: http://127.0.0.1:8000/docs
echo.
uvicorn app.main:app --reload

endlocal
