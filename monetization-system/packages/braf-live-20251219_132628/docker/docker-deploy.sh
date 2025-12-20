#!/bin/bash

# BRAF Docker Deployment Script
# Deploys the complete BRAF framework using Docker

set -e

echo "🚀 BRAF Docker Deployment System"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Build images
build_images() {
    print_status "Building BRAF Docker images..."
    
    # Make build script executable
    chmod +x docker/docker-build.sh
    
    # Run build script
    ./docker/docker-build.sh
    
    print_success "Images built successfully"
}

# Deploy services
deploy_services() {
    print_status "Deploying BRAF services..."
    
    # Stop existing services
    docker-compose -f docker-compose.braf.yml down --remove-orphans
    
    # Start services
    docker-compose -f docker-compose.braf.yml up -d
    
    print_success "Services deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for PostgreSQL..."
    until docker-compose -f docker-compose.braf.yml exec -T postgres pg_isready -U braf_user -d braf_db; do
        sleep 2
    done
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    until docker-compose -f docker-compose.braf.yml exec -T redis redis-cli ping; do
        sleep 2
    done
    
    # Wait for RabbitMQ
    print_status "Waiting for RabbitMQ..."
    until docker-compose -f docker-compose.braf.yml exec -T rabbitmq rabbitmq-diagnostics ping; do
        sleep 2
    done
    
    # Wait for main application
    print_status "Waiting for BRAF application..."
    until curl -f http://localhost:8000/health > /dev/null 2>&1; do
        sleep 5
    done
    
    print_success "All services are ready"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    docker-compose -f docker-compose.braf.yml exec braf_app alembic upgrade head
    
    print_success "Database migrations completed"
}

# Create initial data
create_initial_data() {
    print_status "Creating initial data..."
    
    docker-compose -f docker-compose.braf.yml exec braf_app python seed_sample_data.py
    
    print_success "Initial data created"
}

# Display deployment information
show_deployment_info() {
    echo ""
    echo "🎉 BRAF Framework Deployed Successfully!"
    echo "========================================"
    echo ""
    echo "🌐 Access URLs:"
    echo "• Main Dashboard: http://localhost"
    echo "• Enhanced Dashboard: http://localhost/enhanced-dashboard"
    echo "• API Documentation: http://localhost/docs"
    echo "• Health Check: http://localhost/health"
    echo ""
    echo "📊 Monitoring & Management:"
    echo "• Grafana: http://localhost:3000 (admin/braf_grafana_2024)"
    echo "• Prometheus: http://localhost:9090"
    echo "• Flower (Celery): http://localhost:5555"
    echo "• RabbitMQ Management: http://localhost:15672 (braf/braf_rabbit_2024)"
    echo ""
    echo "🗄️ Database Access:"
    echo "• PostgreSQL: localhost:5432 (braf_user/braf_secure_password_2024)"
    echo "• Redis: localhost:6379"
    echo ""
    echo "🔧 Management Commands:"
    echo "• View logs: docker-compose -f docker-compose.braf.yml logs -f"
    echo "• Stop services: docker-compose -f docker-compose.braf.yml down"
    echo "• Restart services: docker-compose -f docker-compose.braf.yml restart"
    echo "• Scale workers: docker-compose -f docker-compose.braf.yml up -d --scale braf_worker_1=3"
    echo ""
    echo "⚠️  Security Notice:"
    echo "• Change default passwords in .env file for production use"
    echo "• Configure SSL certificates for HTTPS in production"
    echo "• Review and update security settings"
}

# Main deployment function
main() {
    echo "Starting BRAF Docker deployment..."
    echo ""
    
    check_prerequisites
    build_images
    deploy_services
    wait_for_services
    
    # Optional steps (comment out if not needed)
    # run_migrations
    # create_initial_data
    
    show_deployment_info
}

# Handle script arguments
case "${1:-deploy}" in
    "build")
        check_prerequisites
        build_images
        ;;
    "deploy")
        main
        ;;
    "stop")
        print_status "Stopping BRAF services..."
        docker-compose -f docker-compose.braf.yml down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting BRAF services..."
        docker-compose -f docker-compose.braf.yml restart
        print_success "Services restarted"
        ;;
    "logs")
        docker-compose -f docker-compose.braf.yml logs -f
        ;;
    "status")
        docker-compose -f docker-compose.braf.yml ps
        ;;
    *)
        echo "Usage: $0 {build|deploy|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  build   - Build Docker images only"
        echo "  deploy  - Full deployment (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        exit 1
        ;;
esac