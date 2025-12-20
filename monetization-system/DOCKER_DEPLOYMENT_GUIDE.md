# BRAF Docker Deployment Guide

## 🐳 Complete Docker-Based Deployment System

This guide provides comprehensive instructions for deploying the BRAF (Browser Automation Revenue Framework) using Docker and Docker Compose.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Architecture Overview](#architecture-overview)
4. [Configuration](#configuration)
5. [Deployment](#deployment)
6. [Management](#management)
7. [Monitoring](#monitoring)
8. [Scaling](#scaling)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)

## Prerequisites

### Required Software:
- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **Git**: For cloning the repository
- **Minimum System Requirements**:
  - CPU: 4 cores
  - RAM: 8GB
  - Disk: 50GB free space
  - OS: Linux, macOS, or Windows with WSL2

### Verify Installation:
```bash
docker --version
docker-compose --version
docker info
```

## Quick Start

### 1. Clone Repository:
```bash
git clone https://github.com/your-org/braf-framework.git
cd braf-framework/monetization-system
```

### 2. Configure Environment:
```bash
# Copy environment template
cp docker/.env.docker .env

# Edit configuration (use your preferred editor)
nano .env
```

### 3. Deploy BRAF:
```bash
# Make deployment script executable
chmod +x docker/docker-deploy.sh

# Run deployment
./docker/docker-deploy.sh deploy
```

### 4. Access BRAF:
- **Main Dashboard**: http://localhost
- **Enhanced Dashboard**: http://localhost/enhanced-dashboard
- **API Docs**: http://localhost/docs

## Architecture Overview

### Docker Services:

```
┌─────────────────────────────────────────────────────────┐
│                    Nginx (Reverse Proxy)                │
│                    Port 80, 443                         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌────────▼────────┐
│  BRAF Main App │      │  BRAF Workers   │
│  Ports: 8000-  │      │  (Celery)       │
│  8004          │      │  x2 instances   │
└───────┬────────┘      └────────┬────────┘
        │                        │
        └────────────┬───────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐  ┌────────────▼────────┐
│   PostgreSQL   │  │      Redis          │
│   Port: 5432   │  │      Port: 6379     │
└────────────────┘  └─────────────────────┘
                     │
              ┌──────▼──────┐
              │  RabbitMQ   │
              │  Port: 5672 │
              │  Mgmt: 15672│
              └─────────────┘

┌─────────────────────────────────────────────────────────┐
│              Monitoring Stack                           │
│  Prometheus (9090) | Grafana (3000) | Flower (5555)   │
└─────────────────────────────────────────────────────────┘
```

### Service Descriptions:

1. **braf_app**: Main BRAF application server
   - Handles HTTP requests
   - Serves dashboards and APIs
   - Manages browser automation tasks

2. **braf_worker_1/2**: Celery worker nodes
   - Process background tasks
   - Execute browser automation
   - Handle research operations

3. **postgres**: PostgreSQL database
   - Stores application data
   - User accounts and transactions
   - Research data and logs

4. **redis**: Redis cache
   - Session management
   - Task queue backend
   - Real-time data caching

5. **rabbitmq**: Message queue
   - Task distribution
   - Worker communication
   - Event streaming

6. **nginx**: Reverse proxy
   - Load balancing
   - SSL termination
   - Rate limiting

7. **prometheus**: Metrics collection
   - System monitoring
   - Performance metrics
   - Alert management

8. **grafana**: Visualization
   - Dashboard creation
   - Metric visualization
   - Alert notifications

9. **flower**: Celery monitoring
   - Worker status
   - Task monitoring
   - Queue management

## Configuration

### Environment Variables (.env):

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=braf_db
POSTGRES_USER=braf_user

# Redis Configuration
REDIS_PASSWORD=your_redis_password_here

# RabbitMQ Configuration
RABBITMQ_PASSWORD=your_rabbitmq_password_here

# Application Configuration
SECRET_KEY=your_secret_key_minimum_32_characters
BRAF_ENVIRONMENT=production
DEBUG=false

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here

# API Keys (add your actual keys)
OPENAI_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=...
DISCORD_BOT_TOKEN=...

# Payment Providers
OPAY_API_KEY=...
PALMPAY_API_KEY=...
TON_API_KEY=...

# Cryptocurrency APIs
COINBASE_API_KEY=...
BINANCE_API_KEY=...
```

### Docker Compose Configuration:

The `docker-compose.braf.yml` file defines all services. Key configurations:

- **Resource Limits**: Set in service definitions
- **Network Configuration**: Custom bridge network
- **Volume Mounts**: Persistent data storage
- **Health Checks**: Service availability monitoring

## Deployment

### Full Deployment:

```bash
# Deploy all services
./docker/docker-deploy.sh deploy
```

### Build Only:

```bash
# Build Docker images without deploying
./docker/docker-deploy.sh build
```

### Custom Deployment:

```bash
# Deploy specific services
docker-compose -f docker-compose.braf.yml up -d postgres redis rabbitmq

# Deploy application after infrastructure
docker-compose -f docker-compose.braf.yml up -d braf_app braf_worker_1 braf_worker_2
```

## Management

### Service Control:

```bash
# Stop all services
./docker/docker-deploy.sh stop

# Restart all services
./docker/docker-deploy.sh restart

# View service status
./docker/docker-deploy.sh status

# View logs
./docker/docker-deploy.sh logs

# View specific service logs
docker-compose -f docker-compose.braf.yml logs -f braf_app
```

### Database Management:

```bash
# Access PostgreSQL
docker-compose -f docker-compose.braf.yml exec postgres psql -U braf_user -d braf_db

# Run migrations
docker-compose -f docker-compose.braf.yml exec braf_app alembic upgrade head

# Create database backup
docker-compose -f docker-compose.braf.yml exec postgres pg_dump -U braf_user braf_db > backup.sql

# Restore database
docker-compose -f docker-compose.braf.yml exec -T postgres psql -U braf_user -d braf_db < backup.sql
```

### Redis Management:

```bash
# Access Redis CLI
docker-compose -f docker-compose.braf.yml exec redis redis-cli -a your_redis_password

# Clear cache
docker-compose -f docker-compose.braf.yml exec redis redis-cli -a your_redis_password FLUSHALL
```

## Monitoring

### Access Monitoring Tools:

1. **Grafana**: http://localhost:3000
   - Username: `admin`
   - Password: Set in `.env` (GRAFANA_PASSWORD)
   - Pre-configured dashboards for BRAF metrics

2. **Prometheus**: http://localhost:9090
   - Metrics collection and querying
   - Alert rule configuration

3. **Flower**: http://localhost:5555
   - Celery worker monitoring
   - Task queue visualization

4. **RabbitMQ Management**: http://localhost:15672
   - Username: `braf`
   - Password: Set in `.env` (RABBITMQ_PASSWORD)
   - Queue and exchange management

### Key Metrics to Monitor:

- **Application Performance**:
  - Request rate and latency
  - Error rates
  - Active connections

- **Worker Performance**:
  - Task completion rate
  - Task failure rate
  - Worker utilization

- **Infrastructure**:
  - CPU and memory usage
  - Disk I/O
  - Network throughput

- **Database**:
  - Connection pool usage
  - Query performance
  - Transaction rate

## Scaling

### Horizontal Scaling:

```bash
# Scale workers
docker-compose -f docker-compose.braf.yml up -d --scale braf_worker_1=5

# Scale application instances
docker-compose -f docker-compose.braf.yml up -d --scale braf_app=3
```

### Vertical Scaling:

Edit `docker-compose.braf.yml` to adjust resource limits:

```yaml
services:
  braf_app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Load Balancing:

Nginx automatically load balances across multiple application instances.

## Troubleshooting

### Common Issues:

#### 1. Port Already in Use:
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or change port in docker-compose.braf.yml
```

#### 2. Container Won't Start:
```bash
# Check logs
docker-compose -f docker-compose.braf.yml logs <service_name>

# Check container status
docker ps -a

# Restart specific service
docker-compose -f docker-compose.braf.yml restart <service_name>
```

#### 3. Database Connection Issues:
```bash
# Verify database is running
docker-compose -f docker-compose.braf.yml exec postgres pg_isready

# Check connection string in .env
# Ensure DATABASE_URL is correct
```

#### 4. Worker Not Processing Tasks:
```bash
# Check worker logs
docker-compose -f docker-compose.braf.yml logs braf_worker_1

# Verify RabbitMQ connection
docker-compose -f docker-compose.braf.yml exec rabbitmq rabbitmq-diagnostics ping

# Restart workers
docker-compose -f docker-compose.braf.yml restart braf_worker_1 braf_worker_2
```

### Debug Mode:

```bash
# Enable debug logging
docker-compose -f docker-compose.braf.yml exec braf_app \
  python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Access container shell
docker-compose -f docker-compose.braf.yml exec braf_app /bin/bash
```

## Production Deployment

### Security Hardening:

1. **Change Default Passwords**:
   - Update all passwords in `.env`
   - Use strong, unique passwords (32+ characters)

2. **Enable SSL/TLS**:
   ```bash
   # Generate SSL certificates
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem
   
   # Uncomment SSL configuration in nginx.conf
   ```

3. **Configure Firewall**:
   ```bash
   # Allow only necessary ports
   ufw allow 80/tcp
   ufw allow 443/tcp
   ufw enable
   ```

4. **Enable Docker Security**:
   - Use Docker secrets for sensitive data
   - Run containers as non-root user
   - Enable Docker Content Trust

### Performance Optimization:

1. **Database Tuning**:
   - Adjust PostgreSQL configuration
   - Enable connection pooling
   - Configure appropriate indexes

2. **Redis Optimization**:
   - Set maxmemory policy
   - Enable persistence
   - Configure eviction policies

3. **Application Tuning**:
   - Adjust worker concurrency
   - Configure connection pools
   - Enable caching strategies

### Backup Strategy:

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
docker-compose -f docker-compose.braf.yml exec postgres \
  pg_dump -U braf_user braf_db > backups/db_$DATE.sql

# Backup volumes
docker run --rm -v braf_postgres_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/postgres_data_$DATE.tar.gz /data

# Upload to cloud storage (example: AWS S3)
aws s3 cp backups/db_$DATE.sql s3://your-bucket/backups/
```

### Monitoring & Alerts:

Configure Prometheus alerts in `monitoring/prometheus.yml`:

```yaml
groups:
  - name: braf_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
```

## Conclusion

The BRAF Docker deployment provides a production-ready, scalable infrastructure for browser automation and revenue generation. Follow this guide for successful deployment and operation.

For additional support:
- **Documentation**: See other guides in the repository
- **Issues**: Report on GitHub
- **Community**: Join our Discord/Telegram

**Last Updated**: December 2024
**Version**: 1.0.0