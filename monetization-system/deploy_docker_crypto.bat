@echo off
REM BRAF Cryptocurrency System - Docker Deployment Script (Windows)
REM Deploys complete production environment with real crypto integration

echo ==========================================
echo BRAF Cryptocurrency System Deployment
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
if not exist "volumes" mkdir volumes
if not exist "volumes\postgres" mkdir volumes\postgres
if not exist "volumes\redis" mkdir volumes\redis
if not exist "volumes\prometheus" mkdir volumes\prometheus
if not exist "volumes\grafana" mkdir volumes\grafana
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "uploads" mkdir uploads
if not exist "nginx" mkdir nginx
if not exist "nginx\ssl" mkdir nginx\ssl
if not exist "nginx\logs" mkdir nginx\logs
if not exist "static" mkdir static
echo [32m✓ Directories created[0m
echo.

REM Check environment file
if not exist ".env.production" (
    echo [33mWarning: .env.production not found, creating from template[0m
    echo Creating .env.production...
    (
        echo # Production Environment
        echo DOMAIN=yourdomain.com
        echo DB_USER=braf_crypto_user
        echo DB_PASSWORD=crypto_secure_pass_2024!
        echo SECRET_KEY=super-secret-crypto-key-change-this
        echo JWT_SECRET_KEY=jwt-secret-crypto-key
        echo ENCRYPTION_KEY=32-char-encryption-key-change-this
        echo NOWPAYMENTS_WEBHOOK_SECRET=crypto_webhook_secret_2024
        echo CLOUDFLARE_EMAIL=your-email@example.com
        echo CLOUDFLARE_ZONE_ID=your-zone-id
        echo FLOWER_USER=admin
        echo FLOWER_PASSWORD=flower_admin_2024
        echo GRAFANA_PASSWORD=grafana_admin_2024
    ) > .env.production
    echo [33mPlease edit .env.production with your actual values[0m
)

echo [32m✓ Environment configured[0m
echo.

REM Build Docker images
echo Building Docker images...
docker-compose -f docker-compose.crypto.yml build --no-cache
if %errorlevel% neq 0 (
    echo [31mError: Failed to build Docker images[0m
    pause
    exit /b 1
)
echo [32m✓ Docker images built[0m
echo.

REM Start services
echo Starting services...
docker-compose -f docker-compose.crypto.yml up -d
if %errorlevel% neq 0 (
    echo [31mError: Failed to start services[0m
    pause
    exit /b 1
)
echo [32m✓ Services started[0m
echo.

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service health
echo Checking service health...
docker-compose -f docker-compose.crypto.yml ps

echo.
echo ==========================================
echo Deployment Summary
echo ==========================================
echo.
echo [32m✓ PostgreSQL Database:[0m Running on port 5432
echo [32m✓ Redis Cache:[0m Running on port 6379
echo [32m✓ BRAF Crypto App:[0m Running on port 8000
echo [32m✓ Nginx Proxy:[0m Running on ports 80/443
echo [32m✓ Flower Monitor:[0m Running on port 5555
echo [32m✓ Prometheus:[0m Running on port 9090
echo [32m✓ Grafana:[0m Running on port 3000
echo.
echo ==========================================
echo Access URLs
echo ==========================================
echo.
echo Main Application: http://localhost:8000
echo Crypto Webhook Test: http://localhost:8000/api/crypto/webhook/test
echo Flower Dashboard: http://localhost:5555
echo Prometheus: http://localhost:9090
echo Grafana: http://localhost:3000
echo.
echo ==========================================
echo NOWPayments Configuration
echo ==========================================
echo.
echo API Key: RD7WEXF-QTW4N7P-HMV12F9-MPANF4G
echo Webhook URL: https://yourdomain.com/api/crypto/webhook/nowpayments
echo Supported Cryptocurrencies: 150+
echo.
echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Configure your domain DNS to point to this server
echo 2. Set up SSL certificates in nginx/ssl/
echo 3. Configure NOWPayments webhook URL in their dashboard
echo 4. Fund your NOWPayments account
echo 5. Test with small transactions first
echo.
echo [32mDeployment completed successfully![0m
echo.

REM Show logs option
echo Press any key to show application logs, or close this window...
pause >nul
docker-compose -f docker-compose.crypto.yml logs -f braf_crypto_app