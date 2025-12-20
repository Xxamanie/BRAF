#!/bin/bash
# BRAF Cryptocurrency System - Docker Deployment Script
# Deploys complete production environment with real crypto integration

set -e

echo "=========================================="
echo "BRAF Cryptocurrency System Deployment"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
mkdir -p data logs uploads nginx/ssl nginx/logs static
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check environment file
if [ ! -f .env.production ]; then
    echo -e "${YELLOW}Warning: .env.production not found, using defaults${NC}"
    echo "Creating .env.production from template..."
    cat > .env.production << 'EOF'
# Production Environment
DOMAIN=yourdomain.com
DB_USER=braf_crypto_user
DB_PASSWORD=crypto_secure_pass_2024!
SECRET_KEY=super-secret-crypto-key-change-this
JWT_SECRET_KEY=jwt-secret-crypto-key
ENCRYPTION_KEY=32-char-encryption-key-change-this
NOWPAYMENTS_WEBHOOK_SECRET=crypto_webhook_secret_2024
CLOUDFLARE_EMAIL=your-email@example.com
CLOUDFLARE_ZONE_ID=your-zone-id
FLOWER_USER=admin
FLOWER_PASSWORD=flower_admin_2024
GRAFANA_PASSWORD=grafana_admin_2024
EOF
    echo -e "${YELLOW}Please edit .env.production with your actual values${NC}"
fi

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

echo -e "${GREEN}✓ Environment configured${NC}"
echo ""

# Build Docker images
echo "Building Docker images..."
docker-compose -f docker-compose.crypto.yml build --no-cache
echo -e "${GREEN}✓ Docker images built${NC}"
echo ""

# Start services
echo "Starting services..."
docker-compose -f docker-compose.crypto.yml up -d
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Check service health
echo "Checking service health..."
docker-compose -f docker-compose.crypto.yml ps

echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ PostgreSQL Database:${NC} Running on port 5432"
echo -e "${GREEN}✓ Redis Cache:${NC} Running on port 6379"
echo -e "${GREEN}✓ BRAF Crypto App:${NC} Running on port 8000"
echo -e "${GREEN}✓ Nginx Proxy:${NC} Running on ports 80/443"
echo -e "${GREEN}✓ Flower Monitor:${NC} Running on port 5555"
echo -e "${GREEN}✓ Prometheus:${NC} Running on port 9090"
echo -e "${GREEN}✓ Grafana:${NC} Running on port 3000"
echo ""
echo "=========================================="
echo "Access URLs"
echo "=========================================="
echo ""
echo "Main Application: http://localhost:8000"
echo "Crypto Webhook Test: http://localhost:8000/api/crypto/webhook/test"
echo "Flower Dashboard: http://localhost:5555"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
echo ""
echo "=========================================="
echo "NOWPayments Configuration"
echo "=========================================="
echo ""
echo "API Key: RD7WEXF-QTW4N7P-HMV12F9-MPANF4G"
echo "Webhook URL: https://${DOMAIN:-yourdomain.com}/api/crypto/webhook/nowpayments"
echo "Supported Cryptocurrencies: 150+"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Configure your domain DNS to point to this server"
echo "2. Set up SSL certificates in nginx/ssl/"
echo "3. Configure NOWPayments webhook URL in their dashboard"
echo "4. Fund your NOWPayments account"
echo "5. Test with small transactions first"
echo ""
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo ""

# Show logs
echo "Showing application logs (Ctrl+C to exit)..."
docker-compose -f docker-compose.crypto.yml logs -f braf_crypto_app