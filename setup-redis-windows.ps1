# Redis Setup for Windows - PowerShell Version
Write-Host "Setting up Redis for BRAF Worker on Windows..." -ForegroundColor Green
Write-Host ""

# Check if Docker is available
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running!" -ForegroundColor Green
    
    # Start Redis container
    Write-Host "Starting Redis container..." -ForegroundColor Yellow
    docker run -d -p 6379:6379 --name redis-braf redis:alpine
    
    # Wait for Redis to start
    Write-Host "Waiting for Redis to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    # Test Redis connection
    Write-Host "Testing Redis connection..." -ForegroundColor Yellow
    $result = docker exec redis-braf redis-cli ping
    
    if ($result -eq "PONG") {
        Write-Host "✅ Redis is running successfully!" -ForegroundColor Green
        Write-Host "You can now run: npm run manager:start" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Redis failed to start properly" -ForegroundColor Red
        Write-Host "Try running: docker logs redis-braf" -ForegroundColor Yellow
    }
    
} catch {
    Write-Host "❌ Docker Desktop is not running." -ForegroundColor Red
    Write-Host ""
    Write-Host "Please choose an option:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Start Docker Desktop" -ForegroundColor Cyan
    Write-Host "- Open Docker Desktop from Start Menu"
    Write-Host "- Wait for it to start (whale icon in system tray)"
    Write-Host "- Run this script again"
    Write-Host ""
    Write-Host "Option 2: Install Redis using Chocolatey (Recommended)" -ForegroundColor Cyan
    Write-Host "- Open PowerShell as Administrator"
    Write-Host "- Run: Set-ExecutionPolicy Bypass -Scope Process -Force"
    Write-Host "- Run: iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
    Write-Host "- Run: choco install redis-64"
    Write-Host "- Run: redis-server"
    Write-Host ""
    Write-Host "Option 3: Use WSL" -ForegroundColor Cyan
    Write-Host "- Open WSL terminal"
    Write-Host "- Run: sudo apt update && sudo apt install redis-server -y"
    Write-Host "- Run: sudo service redis-server start"
    Write-Host "- Run: redis-cli ping"
    Write-Host ""
    Write-Host "Option 4: Continue without Redis (Limited functionality)" -ForegroundColor Cyan
    Write-Host "- Run: npm run simple-worker"
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")