#!/bin/bash
set -e

echo "Starting BRAF Monetization System..."

# Wait for database to be ready
echo "Waiting for database..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "Database is ready!"

# Wait for Redis to be ready
echo "Waiting for Redis..."
while ! nc -z redis 6379; do
  sleep 0.1
done
echo "Redis is ready!"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Initialize default data if needed
echo "Initializing system data..."
python -c "
from config import Config
from security.authentication import SecurityManager
from enterprise.subscription_service import EnterpriseSubscription
import asyncio

async def init_system():
    print('System initialization complete')

asyncio.run(init_system())
"

# Start the application
echo "Starting FastAPI application..."
exec "$@"