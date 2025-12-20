# BRAF Complete Deployment Commands

## 🚀 Full Command Reference for BRAF Deployment with Timeout Configuration

### Prerequisites Check
```bash
# Check Docker installation
docker --version
docker-compose --version
docker info

# Verify system resources
docker system df
docker system info
```

### 1. Quick Deployment (Recommended)
```bash
# Navigate to monetization-system directory
cd monetization-system

# Windows: Run the deployment script
docker\deploy-timeout.bat deploy

# Linux/Mac: Run the deployment script
chmod +x docker/deploy-timeout.sh
./docker/deploy-timeout.sh deploy
```

### 2. Manual Step-by-Step Deployment

#### Step 1: Environment Setup
```bash
# Create required directories
mkdir -p volumes/{postgres,redis,rabbitmq,prometheus,grafana}
mkdir -p logs data config nginx/ssl
mkdir -p monitoring/grafana/{dashboards,datasources}

# Copy environment template (if not exists)
cp docker/.env.docker .env

# Edit environment variables
nano .env  # or use your preferred editor
```

#### Step 2: Build Docker Images
```bash
# Build all images with timeout configuration
docker-compose -f docker-compose.timeout.yml build --no-cache --parallel

# Or build specific services
docker-compose -f docker-compose.timeout.yml build braf_app
docker-compose -f docker-compose.timeout.yml build braf_worker_1
```

#### Step 3: Deploy Infrastructure Services
```bash
# Start database and cache services first
docker-compose -f docker-compose.timeout.yml up -d postgres redis rabbitmq

# Wait for services to be healthy (check status)
docker-compose -f docker-compose.timeout.yml ps
```

#### Step 4: Deploy Application Services
```bash
# Start main application and workers
docker-compose -f docker-compose.timeout.yml up -d braf_app braf_worker_1 braf_worker_2

# Check application health
curl -f http://localhost:8000/health
```

#### Step 5: Deploy Monitoring Stack
```bash
# Start monitoring and proxy services
docker-compose -f docker-compose.timeout.yml up -d nginx prometheus grafana flower

# Verify all services are running
docker-compose -f docker-compose.timeout.yml ps
```

### 3. Alternative Deployment Methods

#### Using Original Docker Compose
```bash
# Deploy with standard configuration
docker-compose -f docker-compose.braf.yml up -d

# Scale workers
docker-compose -f docker-compose.braf.yml up -d --scale braf_worker_1=3
```

#### Using Docker Swarm (Production)
```bash
# Initialize swarm
docker swarm init

# Deploy as stack
docker stack deploy -c docker-compose.timeout.yml braf-stack

# Check stack status
docker stack services braf-stack
```

### 4. Service Management Commands

#### Start/Stop Services
```bash
# Start all services
docker-compose -f docker-compose.timeout.yml up -d

# Stop all services
docker-compose -f docker-compose.timeout.yml down

# Restart specific service
docker-compose -f docker-compose.timeout.yml restart braf_app

# Stop with timeout
docker-compose -f docker-compose.timeout.yml down --timeout 60
```

#### View Logs
```bash
# All services logs
docker-compose -f docker-compose.timeout.yml logs -f

# Specific service logs
docker-compose -f docker-compose.timeout.yml logs -f braf_app
docker-compose -f docker-compose.timeout.yml logs -f braf_worker_1

# Last 100 lines
docker-compose -f docker-compose.timeout.yml logs --tail=100 braf_app
```

#### Health Checks
```bash
# Check service health
docker-compose -f docker-compose.timeout.yml ps

# Detailed container inspection
docker inspect braf_main_app

# Execute health check manually
docker-compose -f docker-compose.timeout.yml exec braf_app curl -f http://localhost:8000/health
```

### 5. Database Management

#### Database Operations
```bash
# Access PostgreSQL
docker-compose -f docker-compose.timeout.yml exec postgres psql -U braf_user -d braf_db

# Run migrations
docker-compose -f docker-compose.timeout.yml exec braf_app alembic upgrade head

# Create database backup
docker-compose -f docker-compose.timeout.yml exec postgres pg_dump -U braf_user braf_db > backup_$(date +%Y%m%d).sql

# Restore database
docker-compose -f docker-compose.timeout.yml exec -T postgres psql -U braf_user -d braf_db < backup.sql
```

#### Redis Operations
```bash
# Access Redis CLI
docker-compose -f docker-compose.timeout.yml exec redis redis-cli -a your_redis_password

# Clear Redis cache
docker-compose -f docker-compose.timeout.yml exec redis redis-cli -a your_redis_password FLUSHALL

# Monitor Redis
docker-compose -f docker-compose.timeout.yml exec redis redis-cli -a your_redis_password MONITOR
```

### 6. Monitoring and Debugging

#### Access Monitoring Dashboards
```bash
# Open monitoring URLs
# Grafana: http://localhost:3000 (admin/your_grafana_password)
# Prometheus: http://localhost:9090
# Flower: http://localhost:5555
# RabbitMQ: http://localhost:15672 (braf/your_rabbitmq_password)
```

#### Debug Commands
```bash
# Check container resource usage
docker stats

# Inspect container configuration
docker inspect braf_main_app

# Execute shell in container
docker-compose -f docker-compose.timeout.yml exec braf_app /bin/bash

# Check network connectivity
docker-compose -f docker-compose.timeout.yml exec braf_app ping postgres
```

### 7. Scaling and Performance

#### Horizontal Scaling
```bash
# Scale workers
docker-compose -f docker-compose.timeout.yml up -d --scale braf_worker_1=5

# Scale application instances
docker-compose -f docker-compose.timeout.yml up -d --scale braf_app=3
```

#### Performance Monitoring
```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check disk usage
docker system df

# Monitor logs in real-time
docker-compose -f docker-compose.timeout.yml logs -f --tail=0
```

### 8. Backup and Recovery

#### Create Full Backup
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose -f docker-compose.timeout.yml exec postgres pg_dump -U braf_user braf_db > backups/$(date +%Y%m%d_%H%M%S)/database.sql

# Backup volumes
tar -czf backups/$(date +%Y%m%d_%H%M%S)/volumes.tar.gz volumes/

# Backup configuration
cp .env docker-compose.timeout.yml backups/$(date +%Y%m%d_%H%M%S)/
```

#### Restore from Backup
```bash
# Stop services
docker-compose -f docker-compose.timeout.yml down

# Restore volumes
tar -xzf backups/backup_date/volumes.tar.gz

# Start database
docker-compose -f docker-compose.timeout.yml up -d postgres

# Restore database
docker-compose -f docker-compose.timeout.yml exec -T postgres psql -U braf_user -d braf_db < backups/backup_date/database.sql

# Start all services
docker-compose -f docker-compose.timeout.yml up -d
```

### 9. Cleanup Commands

#### Clean Up Resources
```bash
# Stop and remove containers
docker-compose -f docker-compose.timeout.yml down --volumes --remove-orphans

# Remove images
docker-compose -f docker-compose.timeout.yml down --rmi all

# Clean up system
docker system prune -f

# Remove all unused resources
docker system prune -a -f --volumes
```

### 10. Troubleshooting Commands

#### Common Issues
```bash
# Port conflicts
netstat -tulpn | grep :8000
lsof -i :8000

# Container won't start
docker-compose -f docker-compose.timeout.yml logs braf_app

# Network issues
docker network ls
docker network inspect braf_network

# Permission issues (Linux/Mac)
sudo chown -R $USER:$USER volumes/
chmod -R 755 volumes/
```

#### Reset Everything
```bash
# Complete reset (WARNING: This will delete all data)
docker-compose -f docker-compose.timeout.yml down --volumes --remove-orphans
docker system prune -a -f --volumes
rm -rf volumes/ logs/ data/
```

### 11. Production Deployment

#### SSL/HTTPS Setup
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem

# Update nginx configuration for SSL
# Edit docker/nginx-timeout.conf to enable HTTPS
```

#### Security Hardening
```bash
# Update passwords in .env
# Enable firewall
# Configure Docker security
# Set up log rotation
```

### 12. Access URLs After Deployment

Once deployed, access these URLs:

- **Main Dashboard**: http://localhost
- **Enhanced Dashboard**: http://localhost/enhanced-dashboard  
- **API Documentation**: http://localhost/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Flower (Celery)**: http://localhost:5555
- **RabbitMQ Management**: http://localhost:15672

### 13. Environment Variables Reference

Key environment variables in `.env`:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=braf_db
POSTGRES_USER=braf_user

# Redis
REDIS_PASSWORD=your_redis_password

# RabbitMQ
RABBITMQ_PASSWORD=your_rabbitmq_password

# Application
SECRET_KEY=your_secret_key_32_chars_minimum
BRAF_ENVIRONMENT=production
DEBUG=false

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
FLOWER_PASSWORD=your_flower_password

# Timeouts
HTTP_TIMEOUT=60
DB_TIMEOUT=30
REDIS_TIMEOUT=5
CELERY_TIMEOUT=3600
```

### 14. Quick Commands Summary

```bash
# Full deployment
docker\deploy-timeout.bat deploy                    # Windows
./docker/deploy-timeout.sh deploy                   # Linux/Mac

# Manual deployment
docker-compose -f docker-compose.timeout.yml up -d

# Check status
docker-compose -f docker-compose.timeout.yml ps

# View logs
docker-compose -f docker-compose.timeout.yml logs -f

# Stop services
docker-compose -f docker-compose.timeout.yml down

# Clean up
docker-compose -f docker-compose.timeout.yml down --volumes
docker system prune -f
```

This comprehensive guide provides all the commands needed to deploy, manage, and troubleshoot the BRAF system with timeout configuration.