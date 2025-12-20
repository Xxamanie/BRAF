#!/bin/bash

# BRAF Docker Build Script
# Builds all Docker images for the BRAF framework

set -e

echo "🐳 BRAF Docker Build System"
echo "=========================="

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker is running"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs data config monitoring/grafana/{dashboards,datasources} nginx/ssl

# Build base image
print_status "Building BRAF base image..."
docker build -f docker/Dockerfile.base -t braf-base:latest .
if [ $? -eq 0 ]; then
    print_success "Base image built successfully"
else
    print_error "Failed to build base image"
    exit 1
fi

# Build BRAF application image
print_status "Building BRAF application image..."
docker build -f docker/Dockerfile.braf -t braf-app:latest .
if [ $? -eq 0 ]; then
    print_success "BRAF application image built successfully"
else
    print_error "Failed to build BRAF application image"
    exit 1
fi

# Create Grafana datasource configuration
print_status "Creating Grafana configuration..."
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://braf_prometheus:9090
    isDefault: true
EOF

# Create Grafana dashboard configuration
cat > monitoring/grafana/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'BRAF Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Copy nginx configuration
print_status "Setting up Nginx configuration..."
cp docker/nginx.conf nginx/nginx.conf

# Set up environment file
if [ ! -f .env ]; then
    print_status "Creating environment file..."
    cp docker/.env.docker .env
    print_warning "Please update .env file with your actual configuration values"
fi

# Build and start services
print_status "Building and starting BRAF services..."
docker-compose -f docker-compose.braf.yml build

print_success "BRAF Docker images built successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Run: docker-compose -f docker-compose.braf.yml up -d"
echo "3. Access BRAF at: http://localhost"
echo ""
echo "📊 Monitoring URLs:"
echo "• Grafana: http://localhost:3000"
echo "• Prometheus: http://localhost:9090"
echo "• Flower: http://localhost:5555"
echo "• RabbitMQ: http://localhost:15672"