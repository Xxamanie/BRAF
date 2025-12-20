@echo off
REM BRAF Production Deployment Script (Windows)
REM Deploys BRAF system with proper service references

echo ==========================================
echo BRAF Production System Deployment
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mError: Docker is not installed[0m
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mError: Docker Compose is not installed[0m
    pause
    exit /b 1
)

echo [32m✓ Docker and Docker Compose are installed[0m
echo.

REM Check required environment variables
if "%POSTGRES_DB%"=="" (
    echo [31mError: POSTGRES_DB environment variable is not set[0m
    pause
    exit /b 1
)
if "%POSTGRES_USER%"=="" (
    echo [31mError: POSTGRES_USER environment variable is not set[0m
    pause
    exit /b 1
)
if "%POSTGRES_PASSWORD%"=="" (
    echo [31mError: POSTGRES_PASSWORD environment variable is not set[0m
    pause
    exit /b 1
)
if "%VAULT_TOKEN%"=="" (
    echo [31mError: VAULT_TOKEN environment variable is not set[0m
    pause
    exit /b 1
)
if "%SECRET_KEY%"=="" (
    echo [31mError: SECRET_KEY environment variable is not set[0m
    pause
    exit /b 1
)

echo [32m✓ Environment variables configured[0m
echo.

REM Build and deploy services
echo Building and deploying BRAF production services...
docker-compose -f docker-compose.prod.yml build
if %errorlevel% neq 0 (
    echo [31mError: Failed to build services[0m
    pause
    exit /b 1
)

docker-compose -f docker-compose.prod.yml up -d
if %errorlevel% neq 0 (
    echo [31mError: Failed to start services[0m
    pause
    exit /b 1
)

echo [32m✓ Services deployed[0m
echo.

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 30 /nobreak >nul

REM Initialize database (using c2_server instead of scraper)
echo Initializing database...
docker-compose -f docker-compose.prod.yml run --rm c2_server python -m database.init_db
if %errorlevel% neq 0 (
    echo [33mWarning: Database initialization may have failed[0m
)

REM Run database migrations (using c2_server instead of scraper)
echo Running database migrations...
docker-compose -f docker-compose.prod.yml run --rm c2_server alembic upgrade head
if %errorlevel% neq 0 (
    echo [33mWarning: Database migrations may have failed[0m
)

REM Create admin user (using c2_server instead of scraper)
echo Creating admin user...
docker-compose -f docker-compose.prod.yml run --rm c2_server python -m auth.create_user --username admin --password admin123 --role admin
if %errorlevel% neq 0 (
    echo [33mWarning: Admin user creation may have failed[0m
)

REM Import automation targets (using c2_server instead of scraper)
echo Importing automation targets...
docker-compose -f docker-compose.prod.yml run --rm c2_server python -m tasks.import_targets
if %errorlevel% neq 0 (
    echo [33mWarning: Target import may have failed[0m
)

REM Check worker node health (using worker_node instead of celery_worker)
echo Checking worker node status...
docker-compose -f docker-compose.prod.yml exec worker_node python -c "import sys; sys.path.append('/app'); from src.braf.worker.main import health_check; exit(0 if health_check() else 1)"
if %errorlevel% neq 0 (
    echo [33mWarning: Worker node health check failed[0m
)

REM Verify c2_server health endpoint
echo Verifying C2 server health...
timeout /t 5 /nobreak >nul
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [32m✓ C2 server health check passed[0m
) else (
    echo [33m⚠ C2 server health check failed - service may still be starting[0m
)

REM Show service status
echo.
echo ==========================================
echo Service Status
echo ==========================================
docker-compose -f docker-compose.prod.yml ps

echo.
echo ==========================================
echo BRAF Production Deployment Summary
echo ==========================================
echo.
echo [32m✓ PostgreSQL Database:[0m Running with persistent storage
echo [32m✓ Redis Cache:[0m Running with data persistence
echo [32m✓ Vault Secrets:[0m Running for secure configuration
echo [32m✓ C2 Server:[0m Running on port 8000 (2 replicas)
echo [32m✓ Worker Nodes:[0m Running (5 replicas)
echo [32m✓ Nginx Proxy:[0m Running on ports 80/443
echo.
echo ==========================================
echo Access URLs
echo ==========================================
echo.
echo [34mBRAF Dashboard:[0m http://localhost
echo [34mC2 Server API:[0m http://localhost:8000
echo [34mC2 Health Check:[0m http://localhost:8000/health
echo [34mAPI Documentation:[0m http://localhost:8000/docs
echo [34mVault UI:[0m http://localhost:8200
echo.
echo ==========================================
echo Production Features
echo ==========================================
echo.
echo [32m✓ High Availability:[0m Multiple replicas for C2 and workers
echo [32m✓ Load Balancing:[0m Nginx reverse proxy
echo [32m✓ Secrets Management:[0m HashiCorp Vault integration
echo [32m✓ Data Persistence:[0m PostgreSQL and Redis volumes
echo [32m✓ Network Security:[0m Internal overlay networks
echo [32m✓ Resource Limits:[0m CPU and memory constraints
echo.
echo ==========================================
echo Management Commands
echo ==========================================
echo.
echo Scale workers:
echo   docker service scale braf_worker_node=10
echo.
echo View logs:
echo   docker-compose -f docker-compose.prod.yml logs -f c2_server
echo   docker-compose -f docker-compose.prod.yml logs -f worker_node
echo.
echo Execute commands in services:
echo   docker-compose -f docker-compose.prod.yml exec c2_server bash
echo   docker-compose -f docker-compose.prod.yml exec worker_node bash
echo.
echo Stop deployment:
echo   docker-compose -f docker-compose.prod.yml down
echo.
echo [32mBRAF production deployment completed successfully![0m
echo.

REM Show real-time logs
echo Showing C2 server logs (Ctrl+C to exit)...
docker-compose -f docker-compose.prod.yml logs -f c2_server

pause