@echo off
echo 🚀 Starting BRAF Continuous Workers
echo ================================
echo.
echo This will run BRAF workers every 15 minutes
echo Press Ctrl+C to stop
echo.
cd /d "%~dp0"
python run_continuous.py --interval 15
pause