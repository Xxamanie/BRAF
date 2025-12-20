@echo off
REM BRAF Windows Deployment Script

echo Starting BRAF Live Deployment...

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install -r app\requirements-live.txt
python -m playwright install

REM Setup database (SQLite for Windows)
python app\database\setup.py

REM Run migrations
alembic upgrade head

REM Start services
echo BRAF deployment completed successfully!
echo Run: python app\main.py to start the server
pause
