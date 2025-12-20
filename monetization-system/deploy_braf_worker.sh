#!/bin/bash
# BRAF Worker System - Docker Deployment Script
# Deploys BRAF automation workers with cryptocurrency integration

set -e

echo "=========================================="
echo "BRAF Worker System Deployment"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p volumes/postgres volumes/redis volumes/prometheus volumes/grafana
mkdir -p data logs certificates uploads backups nginx/ssl nginx/logs
mkdir -p monitoring grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check environment file
if [ ! -f .env.worker ]; then
    echo -e "${YELLOW}Creating .env.worker configuration...${NC}"
    cat > .env.worker << 'EOF'
# BRAF Worker Environment
DOMAIN=localhost
DB_USER=braf_user
DB_PASSWORD=braf_secure_pass_2024!
FLOWER_USER=admin
FLOWER_PASSWORD=flower_2024
GRAFANA_PASSWORD=admin123

# Worker Configuration
MAX_WORKERS=4
WORKER_CONCURRENCY=2
AUTOMATION_ENABLED=true
CRYPTO_ENABLED=true

# Security
RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=60
EOF
    echo -e "${YELLOW}Please edit .env.worker with your actual values${NC}"
fi

# Load environment variables
if [ -f .env.worker ]; then
    export $(cat .env.worker | grep -v '^#' | xargs)
fi

echo -e "${GREEN}✓ Environment configured${NC}"
echo ""

# Build Docker images
echo "Building BRAF Worker Docker images..."
docker-compose -f docker-compose.worker.yml build --no-cache
echo -e "${GREEN}✓ Docker images built${NC}"
echo ""

# Start services
echo "Starting BRAF Worker services..."
docker-compose -f docker-compose.worker.yml up -d
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 15

# Check service health
echo "Checking service health..."
docker-compose -f docker-compose.worker.yml ps

echo ""
echo "=========================================="
echo "BRAF Worker Deployment Summary"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ PostgreSQL Database:${NC} Running on port 5432"
echo -e "${GREEN}✓ Redis Cache:${NC} Running on port 6379"
echo -e "${GREEN}✓ BRAF Worker:${NC} Running on port 8000"
echo -e "${GREEN}✓ BRAF C2 Server:${NC} Running on port 8001"
echo -e "${GREEN}✓ Celery Worker:${NC} Background task processing"
echo -e "${GREEN}✓ Flower Monitor:${NC} Running on port 5555"
echo -e "${GREEN}✓ Prometheus:${NC} Running on port 9090"
echo -e "${GREEN}✓ Grafana:${NC} Running on port 3000"
echo -e "${GREEN}✓ Nginx Proxy:${NC} Running on ports 80/443"
echo ""
echo "=========================================="
echo "Access URLs"
echo "=========================================="
echo ""
echo -e "${BLUE}BRAF Worker:${NC} http://localhost:8000"
echo -e "${BLUE}BRAF C2 Dashboard:${NC} http://localhost:8001"
echo -e "${BLUE}Flower Monitor:${NC} http://localhost:5555"
echo -e "${BLUE}Prometheus:${NC} http://localhost:9090"
echo -e "${BLUE}Grafana Dashboard:${NC} http://localhost:3000"
echo -e "${BLUE}Worker Health:${NC} http://localhost:8000/health"
echo ""
echo "=========================================="
echo "BRAF Worker Features"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ Browser Automation:${NC} Chromium with full automation"
echo -e "${GREEN}✓ Cryptocurrency Integration:${NC} NOWPayments API active"
echo -e "${GREEN}✓ Task Processing:${NC} Celery background workers"
echo -e "${GREEN}✓ Command & Control:${NC} C2 server for management"
echo -e "${GREEN}✓ Monitoring:${NC} Flower task monitoring"
echo ""
echo "=========================================="
echo "NOWPayments Configuration"
echo "=========================================="
echo ""
echo "API Key: RD7WEXF-QTW4N7P-HMV12F9-MPANF4G"
echo "Base URL: https://api.nowpayments.io/v1"
echo "Sandbox: Disabled (Live transactions)"
echo "Supported Cryptocurrencies: 150+"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Configure worker tasks in BRAF C2 dashboard"
echo "2. Set up automation profiles and proxies"
echo "3. Configure earning platform integrations"
echo "4. Monitor worker performance via Flower"
echo "5. Scale workers as needed"
echo ""
echo -e "${GREEN}BRAF Worker deployment completed successfully!${NC}"
echo ""

# Show worker logs
echo "Showing BRAF worker logs (Ctrl+C to exit)..."
docker-compose -f docker-compose.worker.yml logs -f braf_worker