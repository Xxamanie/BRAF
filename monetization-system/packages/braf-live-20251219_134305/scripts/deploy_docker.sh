#!/bin/bash
# BRAF Docker Deployment Script

set -e

echo "Starting BRAF Docker Deployment..."

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed"
    exit 1
fi

# Build and deploy
docker-compose -f docker/docker-compose.production.yml build
docker-compose -f docker/docker-compose.production.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 30

# Run migrations
docker-compose -f docker/docker-compose.production.yml exec braf_app alembic upgrade head

# Show status
docker-compose -f docker/docker-compose.production.yml ps

echo "BRAF Docker deployment completed successfully!"
echo "Access your BRAF instance at: http://localhost"
