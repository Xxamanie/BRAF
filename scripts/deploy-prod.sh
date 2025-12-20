#!/bin/bash
set -e

echo "Starting BRAF production deployment..."

# Check required environment variables
required_vars=("POSTGRES_DB" "POSTGRES_USER" "POSTGRES_PASSWORD" "VAULT_TOKEN" "SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var environment variable is not set"
        exit 1
    fi
done

# Deploy stack
docker stack deploy -c docker-compose.prod.yml braf

echo "Production deployment initiated!"
echo "Check deployment status: docker service ls"