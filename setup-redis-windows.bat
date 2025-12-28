@echo off
echo Setting up Redis for BRAF Worker on Windows...
echo.

echo Checking if Docker Desktop is running...
docker ps >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop is not running.
    echo.
    echo Please choose an option:
    echo 1. Start Docker Desktop manually and run this script again
    echo 2. Install Redis using Chocolatey
    echo 3. Use WSL (Windows Subsystem for Linux)
    echo.
    echo Option 1: Docker Desktop
    echo - Open Docker Desktop from Start Menu
    echo - Wait for it to start (whale icon in system tray)
    echo - Run this script again
    echo.
    echo Option 2: Chocolatey (Recommended)
    echo - Open PowerShell as Administrator
    echo - Run: choco install redis-64
    echo - Run: redis-server
    echo.
    echo Option 3: WSL
    echo - Open WSL terminal
    echo - Run: sudo apt update
    echo - Run: sudo apt install redis-server -y
    echo - Run: sudo service redis-server start
    echo.
    pause
    exit /b 1
)

echo Docker is running! Starting Redis container...
docker run -d -p 6379:6379 --name redis-braf redis:alpine

echo Waiting for Redis to start...
timeout /t 3 /nobreak >nul

echo Testing Redis connection...
docker exec redis-braf redis-cli ping

if %errorlevel% equ 0 (
    echo.
    echo ✅ Redis is running successfully!
    echo You can now run: npm run manager:start
    echo.
) else (
    echo ❌ Redis failed to start properly
    echo Try running: docker logs redis-braf
)

pause