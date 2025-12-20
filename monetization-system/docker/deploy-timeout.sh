#!/bin/bash

# BRAF Docker Deployment Script with Timeout Configuration
# Enhanced deployment with comprehensive timeout settings

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.timeout.yml"
PROJECT_NAME="braf-timeout"
TIMEOUT_DEPLOY=600  # 10 minutes
TIMEOUT_HEALTH=300  # 5 minutes
TIMEOUT_STOP=120    # 2 minutes

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All dependencies are available"
}

create_directories() {
    log_info "Creating required directories..."
    
    mkdir -p volumes/{postgres,redis,rabbitmq,prometheus,grafana}
    mkdir -p logs data config
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/{dashboards,datasources}
    
    # Set proper permissions
    chmod 755 volumes/*
    chmod 755 logs data config
    
    log_success "Directories created successfully"
}

setup_environment() {
    log_info "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        log_warning ".env file not found, creating from template..."
        cat > .env << EOF
# BRAF Environment Configuration with Timeout Settings

# Database Configuration
POSTGRES_PASSWORD=braf_secure_password_$(date +%s)
POSTGRES_DB=braf_db
POSTGRES_USER=braf_user

# Redis Configuration
REDIS_PASSWORD=braf_redis_$(date +%s)

# RabbitMQ Configuration
RABBITMQ_PASSWORD=braf_rabbit_$(date +%s)

# Application Configuration
SECRET_KEY=braf_secret_key_$(openssl rand -hex 32)
BRAF_ENVIRONMENT=production
DEBUG=false

# Monitoring
GRAFANA_PASSWORD=braf_grafana_$(date +%s)
FLOWER_USER=admin
FLOWER_PASSWORD=braf_flower_$(date +%s)

# Build Configuration
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VERSION=1.0.0

# Timeout Configuration
HTTP_TIMEOUT=60
DB_TIMEOUT=30
REDIS_TIMEOUT=5
RABBITMQ_TIMEOUT=30
CELERY_TIMEOUT=3600
BROWSER_TIMEOUT=60
EOF
        log_success "Environment file created"
    else
        log_info "Using existing .env file"
    fi
}

build_images() {
    log_info "Building Docker images with timeout settings..."
    
    # Build with timeout
    timeout $TIMEOUT_DEPLOY docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build --no-cache --parallel || {
        log_error "Failed to build Docker images within timeout"
        exit 1
    }
    
    log_success "Docker images built successfully"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure services..."
    
    # Deploy database and cache services first
    timeout $TIMEOUT_DEPLOY docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d postgres redis rabbitmq || {
        log_error "Failed to deploy infrastructure services"
        exit 1
    }
    
    # Wait for infrastructure to be healthy
    log_info "Waiting for infrastructure services to be healthy..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps | grep -E "(postgres|redis|rabbitmq)" | grep -q "healthy\|Up"; then
            log_success "Infrastructure services are healthy"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Infrastructure services failed to become healthy"
            exit 1
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for services..."
        sleep 10
        ((attempt++))
    done
}

deploy_application() {
    log_info "Deploying application services..."
    
    # Deploy application services
    timeout $TIMEOUT_DEPLOY docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d braf_app braf_worker_1 braf_worker_2 || {
        log_error "Failed to deploy application services"
        exit 1
    }
    
    # Wait for application to be ready
    log_info "Waiting for application services to be ready..."
    local max_attempts=20
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Application services are ready"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Application services failed to become ready"
            exit 1
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for application..."
        sleep 15
        ((attempt++))
    done
}

deploy_monitoring() {
    log_info "Deploying monitoring services..."
    
    # Deploy monitoring stack
    timeout $TIMEOUT_DEPLOY docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d nginx prometheus grafana flower || {
        log_error "Failed to deploy monitoring services"
        exit 1
    }
    
    log_success "Monitoring services deployed"
}

run_health_checks() {
    log_info "Running comprehensive health checks..."
    
    local services=("postgres" "redis" "rabbitmq" "braf_app" "nginx" "prometheus" "grafana" "flower")
    local failed_services=()
    
    for service in "${services[@]}"; do
        log_info "Checking health of $service..."
        
        if timeout 30 docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T $service echo "Health check" > /dev/null 2>&1; then
            log_success "$service is healthy"
        else
            log_warning "$service health check failed"
            failed_services+=($service)
        fi
    done
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        log_success "All services passed health checks"
    else
        log_warning "Some services failed health checks: ${failed_services[*]}"
    fi
}

show_status() {
    log_info "Deployment Status:"
    echo
    
    # Show service status
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps
    echo
    
    # Show access URLs
    log_info "Access URLs:"
    echo "  🌐 Main Dashboard:      http://localhost"
    echo "  📊 Enhanced Dashboard:  http://localhost/enhanced-dashboard"
    echo "  📋 API Documentation:   http://localhost/docs"
    echo "  📈 Prometheus:          http://localhost:9090"
    echo "  📊 Grafana:             http://localhost:3000"
    echo "  🌸 Flower:              http://localhost:5555"
    echo "  🐰 RabbitMQ Management: http://localhost:15672"
    echo
    
    # Show resource usage
    log_info "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $(docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps -q) 2>/dev/null || true
}

stop_services() {
    log_info "Stopping BRAF services..."
    
    timeout $TIMEOUT_STOP docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --timeout 30 || {
        log_warning "Graceful shutdown timed out, forcing stop..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME kill
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --volumes
    }
    
    log_success "Services stopped"
}

cleanup() {
    log_info "Cleaning up Docker resources..."
    
    # Remove containers
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --volumes --remove-orphans
    
    # Remove images (optional)
    if [ "$1" = "--remove-images" ]; then
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --rmi all
    fi
    
    # Clean up unused resources
    docker system prune -f
    
    log_success "Cleanup completed"
}

show_logs() {
    local service=${1:-}
    
    if [ -n "$service" ]; then
        log_info "Showing logs for $service..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f --tail=100 $service
    else
        log_info "Showing logs for all services..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f --tail=50
    fi
}

backup_data() {
    log_info "Creating backup of BRAF data..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p $backup_dir
    
    # Backup database
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME exec -T postgres pg_dump -U braf_user braf_db > $backup_dir/database.sql
    
    # Backup volumes
    tar -czf $backup_dir/volumes.tar.gz volumes/
    
    # Backup configuration
    cp .env $backup_dir/
    cp $COMPOSE_FILE $backup_dir/
    
    log_success "Backup created in $backup_dir"
}

# Main execution
case "${1:-deploy}" in
    "deploy")
        log_info "Starting BRAF deployment with timeout configuration..."
        check_dependencies
        create_directories
        setup_environment
        build_images
        deploy_infrastructure
        deploy_application
        deploy_monitoring
        run_health_checks
        show_status
        log_success "BRAF deployment completed successfully!"
        ;;
    
    "build")
        log_info "Building BRAF Docker images..."
        check_dependencies
        setup_environment
        build_images
        log_success "Build completed!"
        ;;
    
    "start")
        log_info "Starting BRAF services..."
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
        show_status
        ;;
    
    "stop")
        stop_services
        ;;
    
    "restart")
        log_info "Restarting BRAF services..."
        stop_services
        sleep 5
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
        show_status
        ;;
    
    "status")
        show_status
        ;;
    
    "logs")
        show_logs $2
        ;;
    
    "health")
        run_health_checks
        ;;
    
    "backup")
        backup_data
        ;;
    
    "cleanup")
        cleanup $2
        ;;
    
    "help"|"--help"|"-h")
        echo "BRAF Docker Deployment Script with Timeout Configuration"
        echo
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo
        echo "Commands:"
        echo "  deploy    Deploy complete BRAF system (default)"
        echo "  build     Build Docker images only"
        echo "  start     Start existing services"
        echo "  stop      Stop all services"
        echo "  restart   Restart all services"
        echo "  status    Show service status and URLs"
        echo "  logs      Show logs (optionally for specific service)"
        echo "  health    Run health checks"
        echo "  backup    Create data backup"
        echo "  cleanup   Clean up Docker resources"
        echo "  help      Show this help message"
        echo
        echo "Examples:"
        echo "  $0 deploy                 # Full deployment"
        echo "  $0 logs braf_app         # Show app logs"
        echo "  $0 cleanup --remove-images # Clean up with images"
        ;;
    
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac