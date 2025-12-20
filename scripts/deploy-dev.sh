#!/bin/bash
set -e

echo "Starting BRAF development deployment..."

# Build images
docker-compose -f docker-compose.dev.yml build

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for database
echo "Waiting for database..."
sleep 10

# Run migrations
docker-compose -f docker-compose.dev.yml exec c2_server alembic upgrade head

echo "Development deployment complete!"
echo "C2 Dashboard: http://localhost:8000"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"