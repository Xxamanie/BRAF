@echo off
REM BRAF Docker Deployment Script with Timeout Configuration (Windows)
REM Enhanced deployment with comprehensive timeout settings

setlocal enabledelayedexpansion

REM Configuration
set COMPOSE_FILE=docker-compose.timeout.yml
set PROJECT_NAME=braf-timeout
set TIMEOUT_DEPLOY=600
set TIMEOUT_HEALTH=300
set TIMEOUT_STOP=120

REM Colors for output (Windows compatible)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Functions
:log_info
echo %BLUE%[INFO]%NC% %~1
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1
goto :eof

:check_dependencies
call :log_info "Checking dependencies..."

docker --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker is not installed"
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker Compose is not installed"
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    call :log_error "Docker daemon is not running"
    exit /b 1
)

call :log_success "All dependencies are available"
goto :eof

:create_directories
call :log_info "Creating required directories..."

if not exist volumes mkdir volumes
if not exist volumes\postgres mkdir volumes\postgres
if not exist volumes\redis mkdir volumes\redis
if not exist volumes\rabbitmq mkdir volumes\rabbitmq
if not exist volumes\prometheus mkdir volumes\prometheus
if not exist volumes\grafana mkdir volumes\grafana
if not exist logs mkdir logs
if not exist data mkdir data
if not exist config mkdir config
if not exist nginx mkdir nginx
if not exist nginx\ssl mkdir nginx\ssl
if not exist monitoring mkdir monitoring
if not exist monitoring\grafana mkdir monitoring\grafana
if not exist monitoring\grafana\dashboards mkdir monitoring\grafana\dashboards
if not exist monitoring\grafana\datasources mkdir monitoring\grafana\datasources

call :log_success "Directories created successfully"
goto :eof

:setup_environment
call :log_info "Setting up environment configuration..."

if not exist .env (
    call :log_warning ".env file not found, creating from template..."
    (
        echo # BRAF Environment Configuration with Timeout Settings
        echo.
        echo # Database Configuration
        echo POSTGRES_PASSWORD=braf_secure_password_%RANDOM%
        echo POSTGRES_DB=braf_db
        echo POSTGRES_USER=braf_user
        echo.
        echo # Redis Configuration
        echo REDIS_PASSWORD=braf_redis_%RANDOM%
        echo.
        echo # RabbitMQ Configuration
        echo RABBITMQ_PASSWORD=braf_rabbit_%RANDOM%
        echo.
        echo # Application Configuration
        echo SECRET_KEY=braf_secret_key_%RANDOM%%RANDOM%
        echo BRAF_ENVIRONMENT=production
        echo DEBUG=false
        echo.
        echo # Monitoring
        echo GRAFANA_PASSWORD=braf_grafana_%RANDOM%
        echo FLOWER_USER=admin
        echo FLOWER_PASSWORD=braf_flower_%RANDOM%
        echo.
        echo # Build Configuration
        echo BUILD_DATE=%DATE%T%TIME%
        echo VERSION=1.0.0
        echo.
        echo # Timeout Configuration
        echo HTTP_TIMEOUT=60
        echo DB_TIMEOUT=30
        echo REDIS_TIMEOUT=5
        echo RABBITMQ_TIMEOUT=30
        echo CELERY_TIMEOUT=3600
        echo BROWSER_TIMEOUT=60
    ) > .env
    call :log_success "Environment file created"
) else (
    call :log_info "Using existing .env file"
)
goto :eof

:build_images
call :log_info "Building Docker images with timeout settings..."

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% build --no-cache --parallel
if errorlevel 1 (
    call :log_error "Failed to build Docker images"
    exit /b 1
)

call :log_success "Docker images built successfully"
goto :eof

:deploy_infrastructure
call :log_info "Deploying infrastructure services..."

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d postgres redis rabbitmq
if errorlevel 1 (
    call :log_error "Failed to deploy infrastructure services"
    exit /b 1
)

call :log_info "Waiting for infrastructure services to be healthy..."
timeout /t 30 /nobreak >nul

call :log_success "Infrastructure services deployed"
goto :eof

:deploy_application
call :log_info "Deploying application services..."

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d braf_app braf_worker_1 braf_worker_2
if errorlevel 1 (
    call :log_error "Failed to deploy application services"
    exit /b 1
)

call :log_info "Waiting for application services to be ready..."
timeout /t 45 /nobreak >nul

call :log_success "Application services deployed"
goto :eof

:deploy_monitoring
call :log_info "Deploying monitoring services..."

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d nginx prometheus grafana flower
if errorlevel 1 (
    call :log_error "Failed to deploy monitoring services"
    exit /b 1
)

call :log_success "Monitoring services deployed"
goto :eof

:show_status
call :log_info "Deployment Status:"
echo.

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% ps
echo.

call :log_info "Access URLs:"
echo   🌐 Main Dashboard:      http://localhost
echo   📊 Enhanced Dashboard:  http://localhost/enhanced-dashboard
echo   📋 API Documentation:   http://localhost/docs
echo   📈 Prometheus:          http://localhost:9090
echo   📊 Grafana:             http://localhost:3000
echo   🌸 Flower:              http://localhost:5555
echo   🐰 RabbitMQ Management: http://localhost:15672
echo.
goto :eof

:stop_services
call :log_info "Stopping BRAF services..."

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down --timeout 30
if errorlevel 1 (
    call :log_warning "Graceful shutdown failed, forcing stop..."
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% kill
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down --volumes
)

call :log_success "Services stopped"
goto :eof

:cleanup
call :log_info "Cleaning up Docker resources..."

docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% down --volumes --remove-orphans
docker system prune -f

call :log_success "Cleanup completed"
goto :eof

:show_logs
if "%~1"=="" (
    call :log_info "Showing logs for all services..."
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% logs -f --tail=50
) else (
    call :log_info "Showing logs for %~1..."
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% logs -f --tail=100 %~1
)
goto :eof

REM Main execution
set ACTION=%~1
if "%ACTION%"=="" set ACTION=deploy

if "%ACTION%"=="deploy" (
    call :log_info "Starting BRAF deployment with timeout configuration..."
    call :check_dependencies
    call :create_directories
    call :setup_environment
    call :build_images
    call :deploy_infrastructure
    call :deploy_application
    call :deploy_monitoring
    call :show_status
    call :log_success "BRAF deployment completed successfully!"
) else if "%ACTION%"=="build" (
    call :log_info "Building BRAF Docker images..."
    call :check_dependencies
    call :setup_environment
    call :build_images
    call :log_success "Build completed!"
) else if "%ACTION%"=="start" (
    call :log_info "Starting BRAF services..."
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d
    call :show_status
) else if "%ACTION%"=="stop" (
    call :stop_services
) else if "%ACTION%"=="restart" (
    call :log_info "Restarting BRAF services..."
    call :stop_services
    timeout /t 5 /nobreak >nul
    docker-compose -f %COMPOSE_FILE% -p %PROJECT_NAME% up -d
    call :show_status
) else if "%ACTION%"=="status" (
    call :show_status
) else if "%ACTION%"=="logs" (
    call :show_logs %~2
) else if "%ACTION%"=="cleanup" (
    call :cleanup
) else if "%ACTION%"=="help" (
    echo BRAF Docker Deployment Script with Timeout Configuration
    echo.
    echo Usage: %~nx0 [COMMAND] [OPTIONS]
    echo.
    echo Commands:
    echo   deploy    Deploy complete BRAF system ^(default^)
    echo   build     Build Docker images only
    echo   start     Start existing services
    echo   stop      Stop all services
    echo   restart   Restart all services
    echo   status    Show service status and URLs
    echo   logs      Show logs ^(optionally for specific service^)
    echo   cleanup   Clean up Docker resources
    echo   help      Show this help message
    echo.
    echo Examples:
    echo   %~nx0 deploy                 # Full deployment
    echo   %~nx0 logs braf_app         # Show app logs
    echo   %~nx0 cleanup               # Clean up resources
) else (
    call :log_error "Unknown command: %ACTION%"
    echo Use '%~nx0 help' for usage information
    exit /b 1
)

endlocal