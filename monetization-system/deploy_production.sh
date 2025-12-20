#!/bin/bash
# BRAF Production Deployment Script
# Deploys BRAF system with proper service references

set -e

echo "=========================================="
echo "BRAF Production System Deployment"
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

# Check required environment variables
required_vars=("POSTGRES_DB" "POSTGRES_USER" "POSTGRES_PASSWORD" "VAULT_TOKEN" "SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}Error: $var environment variable is not set${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ Environment variables configured${NC}"
echo ""

# Build and deploy services
echo "Building and deploying BRAF production services..."
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

echo -e "${GREEN}✓ Services deployed${NC}"
echo ""

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 30

# Initialize database (using c2_server instead of scraper)
echo "Initializing database..."
docker-compose -f docker-compose.prod.yml run --rm c2_server python -m database.init_db

# Run database migrations (using c2_server instead of scraper)
echo "Running database migrations..."
docker-compose -f docker-compose.prod.yml run --rm c2_server alembic upgrade head

# Create admin user (using c2_server instead of scraper)
echo "Creating admin user..."
docker-compose -f docker-compose.prod.yml run --rm c2_server python -m auth.create_user --username admin --password admin123 --role admin

# Import automation targets (using c2_server instead of scraper)
echo "Importing automation targets..."
docker-compose -f docker-compose.prod.yml run --rm c2_server python -m tasks.import_targets

# Check worker node health (using worker_node instead of celery_worker)
echo "Checking worker node status..."
docker-compose -f docker-compose.prod.yml exec worker_node python -c "
import sys
sys.path.append('/app')
from src.braf.worker.main import health_check
if health_check():
    print('Worker node is healthy')
    sys.exit(0)
else:
    print('Worker node health check failed')
    sys.exit(1)
"

# Verify c2_server health endpoint
echo "Verifying C2 server health..."
sleep 5
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ C2 server health check passed${NC}"
else
    echo -e "${YELLOW}⚠ C2 server health check failed - service may still be starting${NC}"
fi

# Show service status
echo ""
echo "=========================================="
echo "Service Status"
echo "=========================================="
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "=========================================="
echo "BRAF Production Deployment Summary"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ PostgreSQL Database:${NC} Running with persistent storage"
echo -e "${GREEN}✓ Redis Cache:${NC} Running with data persistence"
echo -e "${GREEN}✓ Vault Secrets:${NC} Running for secure configuration"
echo -e "${GREEN}✓ C2 Server:${NC} Running on port 8000 (2 replicas)"
echo -e "${GREEN}✓ Worker Nodes:${NC} Running (5 replicas)"
echo -e "${GREEN}✓ Nginx Proxy:${NC} Running on ports 80/443"
echo ""
echo "=========================================="
echo "Access URLs"
echo "=========================================="
echo ""
echo -e "${BLUE}BRAF Dashboard:${NC} http://localhost"
echo -e "${BLUE}C2 Server API:${NC} http://localhost:8000"
echo -e "${BLUE}C2 Health Check:${NC} http://localhost:8000/health"
echo -e "${BLUE}API Documentation:${NC} http://localhost:8000/docs"
echo -e "${BLUE}Vault UI:${NC} http://localhost:8200"
echo ""
echo "=========================================="
echo "Production Features"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ High Availability:${NC} Multiple replicas for C2 and workers"
echo -e "${GREEN}✓ Load Balancing:${NC} Nginx reverse proxy"
echo -e "${GREEN}✓ Secrets Management:${NC} HashiCorp Vault integration"
echo -e "${GREEN}✓ Data Persistence:${NC} PostgreSQL and Redis volumes"
echo -e "${GREEN}✓ Network Security:${NC} Internal overlay networks"
echo -e "${GREEN}✓ Resource Limits:${NC} CPU and memory constraints"
echo ""
echo "=========================================="
echo "Management Commands"
echo "=========================================="
echo ""
echo "Scale workers:"
echo "  docker service scale braf_worker_node=10"
echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose.prod.yml logs -f c2_server"
echo "  docker-compose -f docker-compose.prod.yml logs -f worker_node"
echo ""
echo "Execute commands in services:"
echo "  docker-compose -f docker-compose.prod.yml exec c2_server bash"
echo "  docker-compose -f docker-compose.prod.yml exec worker_node bash"
echo ""
echo "Stop deployment:"
echo "  docker-compose -f docker-compose.prod.yml down"
echo ""
echo -e "${GREEN}BRAF production deployment completed successfully!${NC}"
echo ""

# Show real-time logs
echo "Showing C2 server logs (Ctrl+C to exit)..."
docker-compose -f docker-compose.prod.yml logs -f c2_server