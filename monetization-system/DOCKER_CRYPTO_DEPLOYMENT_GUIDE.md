# BRAF Cryptocurrency System - Docker Deployment Guide

## Overview
Complete Docker deployment guide for the BRAF cryptocurrency system with real NOWPayments integration, supporting 150+ cryptocurrencies with actual blockchain transactions.

## 🚀 **Quick Start**

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 4GB+ RAM
- 20GB+ disk space

### One-Command Deployment

**Linux/macOS:**
```bash
chmod +x deploy_docker_crypto.sh
./deploy_docker_crypto.sh
```

**Windows:**
```cmd
deploy_docker_crypto.bat
```

## 📋 **Detailed Deployment Steps**

### 1. Environment Setup

Create `.env.production` file:
```bash
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
```

### 2. Directory Structure
```
monetization-system/
├── docker-compose.crypto.yml
├── Dockerfile.production
├── .env.production
├── volumes/
│   ├── postgres/
│   ├── redis/
│   ├── prometheus/
│   └── grafana/
├── nginx/
│   ├── ssl/
│   └── logs/
├── data/
├── logs/
└── uploads/
```

### 3. Build and Deploy
```bash
# Build images
docker-compose -f docker-compose.crypto.yml build

# Start services
docker-compose -f docker-compose.crypto.yml up -d

# Check status
docker-compose -f docker-compose.crypto.yml ps
```

## 🏗️ **Architecture Overview**

### Services Deployed

1. **PostgreSQL Database** (Port 5432)
   - Production-optimized configuration
   - Persistent data storage
   - Health checks enabled

2. **Redis Cache** (Port 6379)
   - Session management
   - Caching layer
   - Pub/sub for real-time updates

3. **BRAF Crypto App** (Port 8000)
   - Main application with NOWPayments integration
   - Real cryptocurrency operations
   - Webhook endpoints

4. **Celery Worker**
   - Background crypto processing
   - Transaction monitoring
   - Automated tasks

5. **Nginx Reverse Proxy** (Ports 80/443)
   - SSL termination
   - Load balancing
   - Static file serving

6. **Flower Monitor** (Port 5555)
   - Celery task monitoring
   - Queue management
   - Performance metrics

7. **Prometheus** (Port 9090)
   - Metrics collection
   - System monitoring
   - Alerting

8. **Grafana** (Port 3000)
   - Dashboards
   - Visualization
   - Reporting

## 🔐 **Security Configuration**

### SSL Setup
1. Place SSL certificates in `nginx/ssl/`:
   ```
   nginx/ssl/
   ├── certificate.crt
   └── private.key
   ```

2. Update nginx configuration for HTTPS

### Environment Security
- Change all default passwords
- Use strong encryption keys
- Configure firewall rules
- Enable rate limiting

## 💰 **NOWPayments Integration**

### Configuration
- **API Key**: `RD7WEXF-QTW4N7P-HMV12F9-MPANF4G` (pre-configured)
- **Webhook URL**: `https://yourdomain.com/api/crypto/webhook/nowpayments`
- **Supported Currencies**: 150+ cryptocurrencies
- **Real Blockchain**: Live transactions enabled

### Webhook Setup
1. Login to NOWPayments dashboard
2. Navigate to API settings
3. Set webhook URL: `https://yourdomain.com/api/crypto/webhook/nowpayments`
4. Configure webhook secret in environment

## 🧪 **Testing & Verification**

### Health Checks
```bash
# Test main application
curl http://localhost:8000/api/crypto/webhook/test

# Check NOWPayments integration
curl http://localhost:8000/api/crypto/currencies

# Verify database connection
docker-compose -f docker-compose.crypto.yml exec postgres pg_isready

# Test Redis
docker-compose -f docker-compose.crypto.yml exec redis redis-cli ping
```

### Service Status
```bash
# View all services
docker-compose -f docker-compose.crypto.yml ps

# View logs
docker-compose -f docker-compose.crypto.yml logs -f braf_crypto_app

# Monitor resources
docker stats
```

## 📊 **Monitoring & Maintenance**

### Access URLs
- **Main App**: http://localhost:8000
- **Crypto API**: http://localhost:8000/api/crypto/
- **Flower**: http://localhost:5555
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

### Log Management
```bash
# View application logs
docker-compose -f docker-compose.crypto.yml logs braf_crypto_app

# View all logs
docker-compose -f docker-compose.crypto.yml logs

# Follow logs in real-time
docker-compose -f docker-compose.crypto.yml logs -f
```

### Backup & Recovery
```bash
# Backup database
docker-compose -f docker-compose.crypto.yml exec postgres pg_dump -U braf_crypto_user braf_crypto_prod > backup.sql

# Backup volumes
tar -czf backup-volumes.tar.gz volumes/

# Restore database
docker-compose -f docker-compose.crypto.yml exec -T postgres psql -U braf_crypto_user braf_crypto_prod < backup.sql
```

## 🔧 **Troubleshooting**

### Common Issues

**1. Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Change ports in docker-compose.crypto.yml
```

**2. Memory Issues**
```bash
# Increase Docker memory limit
# Check available memory
free -h

# Adjust service resources in docker-compose.crypto.yml
```

**3. Database Connection**
```bash
# Check database logs
docker-compose -f docker-compose.crypto.yml logs postgres

# Test connection
docker-compose -f docker-compose.crypto.yml exec postgres psql -U braf_crypto_user -d braf_crypto_prod
```

**4. NOWPayments API Issues**
```bash
# Test API connectivity
curl -H "x-api-key: RD7WEXF-QTW4N7P-HMV12F9-MPANF4G" https://api.nowpayments.io/v1/status

# Check application logs for API errors
docker-compose -f docker-compose.crypto.yml logs braf_crypto_app | grep -i nowpayments
```

## 🚀 **Production Deployment**

### Server Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ recommended
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection
- **OS**: Ubuntu 20.04+ or CentOS 8+

### Domain Setup
1. Point domain to server IP
2. Configure DNS records
3. Set up SSL certificates
4. Update DOMAIN in .env.production

### Security Hardening
1. Configure firewall (UFW/iptables)
2. Set up fail2ban
3. Enable automatic security updates
4. Configure log rotation
5. Set up monitoring alerts

## 📈 **Scaling**

### Horizontal Scaling
```yaml
# Add more workers
celery_crypto_worker:
  deploy:
    replicas: 3

# Add more app instances
braf_crypto_app:
  deploy:
    replicas: 2
```

### Performance Optimization
- Increase PostgreSQL shared_buffers
- Add Redis clustering
- Configure CDN for static files
- Implement database read replicas

## 🔄 **Updates & Maintenance**

### Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.crypto.yml build --no-cache
docker-compose -f docker-compose.crypto.yml up -d
```

### Database Migrations
```bash
# Run migrations
docker-compose -f docker-compose.crypto.yml exec braf_crypto_app python -m alembic upgrade head
```

### Cleanup
```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune
```

## 📞 **Support**

### Logs Location
- Application: `logs/`
- Nginx: `nginx/logs/`
- Database: Docker volume
- Redis: Docker volume

### Configuration Files
- Main: `docker-compose.crypto.yml`
- Environment: `.env.production`
- Nginx: `nginx/nginx.conf`
- Monitoring: `monitoring/prometheus.yml`

### Key Endpoints
- Health: `/api/crypto/webhook/test`
- Status: `/api/crypto/balance`
- Currencies: `/api/crypto/currencies`
- Rates: `/api/crypto/rates`

---

## ✅ **Deployment Checklist**

- [ ] Docker and Docker Compose installed
- [ ] Environment variables configured
- [ ] SSL certificates in place
- [ ] Domain DNS configured
- [ ] NOWPayments webhook URL set
- [ ] Firewall rules configured
- [ ] Monitoring alerts set up
- [ ] Backup strategy implemented
- [ ] Testing completed with small amounts

**Status: READY FOR PRODUCTION** 🚀

---

*Docker deployment guide for BRAF Cryptocurrency System*  
*NOWPayments Integration: LIVE*  
*Supported Cryptocurrencies: 150+*  
*Real Blockchain Transactions: ENABLED*