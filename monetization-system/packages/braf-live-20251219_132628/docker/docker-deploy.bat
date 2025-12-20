@echo off
REM BRAF Docker Deployment Script for Windows
REM Deploys the complete BRAF framework using Docker

echo 🚀 BRAF Docker Deployment System
echo ================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not in PATH
    echo Please install Docker Desktop and try again
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo ✅ Docker is available and running

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not available
    echo Please ensure Docker Compose is installed
    pause
    exit /b 1
)

echo ✅ Docker Compose is available

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "config" mkdir config
if not exist "monitoring\grafana\dashboards" mkdir monitoring\grafana\dashboards
if not exist "monitoring\grafana\datasources" mkdir monitoring\grafana\datasources
if not exist "nginx\ssl" mkdir nginx\ssl

REM Copy environment file if it doesn't exist
if not exist ".env" (
    echo 📋 Creating environment file...
    copy "docker\.env.docker" ".env"
    echo ⚠️  Please update .env file with your configuration values
)

REM Copy nginx configuration
echo 🌐 Setting up Nginx configuration...
copy "docker\nginx.conf" "nginx\nginx.conf"

REM Create Grafana configuration files
echo 📊 Creating Grafana configuration...
echo apiVersion: 1 > monitoring\grafana\datasources\prometheus.yml
echo. >> monitoring\grafana\datasources\prometheus.yml
echo datasources: >> monitoring\grafana\datasources\prometheus.yml
echo   - name: Prometheus >> monitoring\grafana\datasources\prometheus.yml
echo     type: prometheus >> monitoring\grafana\datasources\prometheus.yml
echo     access: proxy >> monitoring\grafana\datasources\prometheus.yml
echo     url: http://braf_prometheus:9090 >> monitoring\grafana\datasources\prometheus.yml
echo     isDefault: true >> monitoring\grafana\datasources\prometheus.yml

echo apiVersion: 1 > monitoring\grafana\dashboards\dashboard.yml
echo. >> monitoring\grafana\dashboards\dashboard.yml
echo providers: >> monitoring\grafana\dashboards\dashboard.yml
echo   - name: 'BRAF Dashboards' >> monitoring\grafana\dashboards\dashboard.yml
echo     orgId: 1 >> monitoring\grafana\dashboards\dashboard.yml
echo     folder: '' >> monitoring\grafana\dashboards\dashboard.yml
echo     type: file >> monitoring\grafana\dashboards\dashboard.yml
echo     disableDeletion: false >> monitoring\grafana\dashboards\dashboard.yml
echo     updateIntervalSeconds: 10 >> monitoring\grafana\dashboards\dashboard.yml
echo     allowUiUpdates: true >> monitoring\grafana\dashboards\dashboard.yml
echo     options: >> monitoring\grafana\dashboards\dashboard.yml
echo       path: /etc/grafana/provisioning/dashboards >> monitoring\grafana\dashboards\dashboard.yml

REM Handle command line arguments
set ACTION=%1
if "%ACTION%"=="" set ACTION=deploy

if "%ACTION%"=="build" goto build
if "%ACTION%"=="deploy" goto deploy
if "%ACTION%"=="stop" goto stop
if "%ACTION%"=="restart" goto restart
if "%ACTION%"=="logs" goto logs
if "%ACTION%"=="status" goto status

echo Usage: %0 [build^|deploy^|stop^|restart^|logs^|status]
echo.
echo Commands:
echo   build   - Build Docker images only
echo   deploy  - Full deployment (default)
echo   stop    - Stop all services
echo   restart - Restart all services
echo   logs    - Show service logs
echo   status  - Show service status
goto end

:build
echo 🏗️ Building BRAF Docker images...
docker-compose -f docker-compose.braf.yml build
if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)
echo ✅ Images built successfully
goto end

:deploy
echo 🚀 Deploying BRAF services...

REM Stop existing services
echo 🛑 Stopping existing services...
docker-compose -f docker-compose.braf.yml down --remove-orphans

REM Build images
echo 🏗️ Building images...
docker-compose -f docker-compose.braf.yml build
if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)

REM Start services
echo 🚀 Starting services...
docker-compose -f docker-compose.braf.yml up -d
if %errorlevel% neq 0 (
    echo ❌ Deployment failed
    pause
    exit /b 1
)

echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo.
echo 🎉 BRAF Framework Deployed Successfully!
echo ========================================
echo.
echo 🌐 Access URLs:
echo • Main Dashboard: http://localhost
echo • Enhanced Dashboard: http://localhost/enhanced-dashboard
echo • API Documentation: http://localhost/docs
echo • Health Check: http://localhost/health
echo.
echo 📊 Monitoring ^& Management:
echo • Grafana: http://localhost:3000 (admin/braf_grafana_2024)
echo • Prometheus: http://localhost:9090
echo • Flower (Celery): http://localhost:5555
echo • RabbitMQ Management: http://localhost:15672 (braf/braf_rabbit_2024)
echo.
echo 🗄️ Database Access:
echo • PostgreSQL: localhost:5432 (braf_user/braf_secure_password_2024)
echo • Redis: localhost:6379
echo.
echo ⚠️  Security Notice:
echo • Change default passwords in .env file for production use
echo • Configure SSL certificates for HTTPS in production
echo • Review and update security settings
goto end

:stop
echo 🛑 Stopping BRAF services...
docker-compose -f docker-compose.braf.yml down
echo ✅ Services stopped
goto end

:restart
echo 🔄 Restarting BRAF services...
docker-compose -f docker-compose.braf.yml restart
echo ✅ Services restarted
goto end

:logs
docker-compose -f docker-compose.braf.yml logs -f
goto end

:status
docker-compose -f docker-compose.braf.yml ps
goto end

:end
pause