#!/bin/bash

echo "Checking BRAF system health..."

# Check C2 server
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] C2 Server: Healthy"
else
    echo "[FAIL] C2 Server: Unhealthy"
fi

# Check database
if docker-compose exec postgres pg_isready > /dev/null 2>&1; then
    echo "[OK] Database: Healthy"
else
    echo "[FAIL] Database: Unhealthy"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "[OK] Redis: Healthy"
else
    echo "[FAIL] Redis: Unhealthy"
fi

echo "Health check complete!"