@echo off
REM Windows batch script to run the web scraper
REM Usage: run_scraper.bat [options]

cd /d "%~dp0"

echo 🚀 Starting Web Scraper Runner
echo ==============================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs
if not exist "data" mkdir data

REM Check command line arguments
if "%1"=="--help" (
    echo Usage:
    echo   run_scraper.bat           # Run scraping session (sync version)
    echo   run_scraper.bat --async   # Run async version
    echo   run_scraper.bat --status  # Check scraper status
    echo   run_scraper.bat --config  # Create sample config
    echo   run_scraper.bat --help    # Show this help
    goto :end
)

if "%1"=="--status" (
    echo 📊 Checking scraper status...
    python check_scraper_status.py
    goto :end
)

if "%1"=="--config" (
    echo 📝 Creating sample configuration...
    python run_scrape_sync.py --create-config
    goto :end
)

if "%1"=="--async" (
    echo 🔄 Running web scraper (async version)...
    echo Start time: %date% %time%
    echo.
    python run_scrape.py
    set SCRAPER_EXIT_CODE=%errorlevel%
    goto :show_results
)

REM Default: Run the scraper
echo 🔄 Running web scraper (synchronous version)...
echo Start time: %date% %time%
echo.

python run_scrape_sync.py

set SCRAPER_EXIT_CODE=%errorlevel%

:show_results

echo.
echo End time: %date% %time%

if %SCRAPER_EXIT_CODE%==0 (
    echo ✅ Scraper completed successfully
) else (
    echo ❌ Scraper failed with exit code %SCRAPER_EXIT_CODE%
)

echo.
echo 📋 Quick status check:
python check_scraper_status.py --summary

:end
echo.
echo Press any key to exit...
pause >nul