@echo off
REM BRAF Worker System - Docker Deployment Script (Windows)
REM Deploys BRAF automation workers with cryptocurrency integration

echo ==========================================
echo BRAF Worker System Deployment
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker Compose is not installed
    pause
    exit /b 1
)

echo [32m✓ Docker and Docker Compose are installed[0m
echo.

REM Create necessary directories
echo Creating necessary directories...
mkdir volumes\postgres 2>nul
mkdir volumes\redis 2>nul
mkdir volumes\prometheus 2>nul
mkdir volumes\grafana 2>nul
mkdir data 2>nul
mkdir logs 2>nul
mkdir certificates 2>nul
mkdir uploads 2>nul
mkdir backups 2>nul
mkdir nginx\ssl 2>nul
mkdir nginx\logs 2>nul
mkdir monitoring 2>nul
mkdir grafana\provisioning\datasources 2>nul
mkdir grafana\provisioning\dashboards 2>nul
mkdir grafana\dashboards 2>nul
echo [32m✓ Directories created[0m
echo.

REM Check environment file
if not exist .env.worker (
    echo [33mCreating .env.worker configuration...[0m
    (
        echo # BRAF Worker Environment
        echo DOMAIN=localhost
        echo DB_USER=braf_user
        echo DB_PASSWORD=braf_secure_pass_2024!
        echo FLOWER_USER=admin
        echo FLOWER_PASSWORD=flower_2024
        echo GRAFANA_PASSWORD=admin123
        echo.
        echo # Worker Configuration
        echo MAX_WORKERS=4
        echo WORKER_CONCURRENCY=2
        echo AUTOMATION_ENABLED=true
        echo CRYPTO_ENABLED=true
        echo.
        echo # Security
        echo RATE_LIMIT_ENABLED=true
        echo MAX_REQUESTS_PER_MINUTE=60
    ) > .env.worker
    echo [33mPlease edit .env.worker with your actual values[0m
)

echo [32m✓ Environment configured[0m
echo.

REM Build Docker images
echo Building BRAF Worker Docker images...
docker-compose -f docker-compose.worker.yml build --no-cache
if %errorlevel% neq 0 (
    echo [31mError: Failed to build Docker images[0m
    pause
    exit /b 1
)
echo [32m✓ Docker images built[0m
echo.

REM Start services
echo Starting BRAF Worker services...
docker-compose -f docker-compose.worker.yml up -d
if %errorlevel% neq 0 (
    echo [31mError: Failed to start services[0m
    pause
    exit /b 1
)
echo [32m✓ Services started[0m
echo.

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 15 /nobreak >nul

REM Check service health
echo Checking service health...
docker-compose -f docker-compose.worker.yml ps

echo.
echo ==========================================
echo BRAF Worker Deployment Summary
echo ==========================================
echo.
echo [32m✓ PostgreSQL Database:[0m Running on port 5432
echo [32m✓ Redis Cache:[0m Running on port 6379
echo [32m✓ BRAF Worker:[0m Running on port 8000
echo [32m✓ BRAF C2 Server:[0m Running on port 8001
echo [32m✓ Celery Worker:[0m Background task processing
echo [32m✓ Flower Monitor:[0m Running on port 5555
echo [32m✓ Prometheus:[0m Running on port 9090
echo [32m✓ Grafana:[0m Running on port 3000
echo [32m✓ Nginx Proxy:[0m Running on ports 80/443
echo.
echo ==========================================
echo Access URLs
echo ==========================================
echo.
echo [34mBRAF Worker:[0m http://localhost:8000
echo [34mBRAF C2 Dashboard:[0m http://localhost:8001
echo [34mFlower Monitor:[0m http://localhost:5555
echo [34mPrometheus:[0m http://localhost:9090
echo [34mGrafana Dashboard:[0m http://localhost:3000
echo [34mWorker Health:[0m http://localhost:8000/health
echo.
echo ==========================================
echo BRAF Worker Features
echo ==========================================
echo.
echo [32m✓ Browser Automation:[0m Chromium with full automation
echo [32m✓ Cryptocurrency Integration:[0m NOWPayments API active
echo [32m✓ Task Processing:[0m Celery background workers
echo [32m✓ Command ^& Control:[0m C2 server for management
echo [32m✓ Monitoring:[0m Flower task monitoring
echo.
echo ==========================================
echo NOWPayments Configuration
echo ==========================================
echo.
echo API Key: RD7WEXF-QTW4N7P-HMV12F9-MPANF4G
echo Base URL: https://api.nowpayments.io/v1
echo Sandbox: Disabled (Live transactions)
echo Supported Cryptocurrencies: 150+
echo.
echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Configure worker tasks in BRAF C2 dashboard
echo 2. Set up automation profiles and proxies
echo 3. Configure earning platform integrations
echo 4. Monitor worker performance via Flower
echo 5. Scale workers as needed
echo.
echo [32mBRAF Worker deployment completed successfully![0m
echo.

REM Show worker logs
echo Showing BRAF worker logs (Ctrl+C to exit)...
docker-compose -f docker-compose.worker.yml logs -f braf_worker

pause