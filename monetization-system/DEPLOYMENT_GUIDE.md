# BRAF Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Browser Automation Revenue Framework (BRAF) in a production environment. The deployment includes the complete monetization system, intelligence layer, research capabilities, and security features.

## Quick Start

### Prerequisites Check
```bash
# Check system requirements
python3 --version  # Should be 3.8+
docker --version   # Should be 20.10+
docker-compose --version  # Should be 2.0+
```

### 1-Minute Docker Deployment
```bash
# Extract deployment package
unzip braf-live-20251219_134305.zip
cd braf-live-20251219_134305

# Deploy with Docker
docker-compose -f docker/docker-compose.production.yml up -d

# Verify deployment
curl http://localhost/health
```

## Detailed Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Step 1: Prepare Environment
```bash
# Create deployment directory
mkdir -p /opt/braf
cd /opt/braf

# Extract package
unzip braf-live-20251219_134305.zip
cd braf-live-20251219_134305
```

#### Step 2: Configure Environment
```bash
# Copy environment template
cp config/.env.example app/.env

# Edit configuration
nano app/.env
```

**Required Environment Variables:**
```bash
# Database Configuration
DATABASE_URL=postgresql://braf_user:secure_password@postgres:5432/braf_db
REDIS_URL=redis://redis:6379/0

# Security Keys (Generate new ones!)
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here

# API Keys (Optional - for full functionality)
STRIPE_SECRET_KEY=sk_live_...
COINBASE_API_KEY=your-coinbase-key
TWITTER_BEARER_TOKEN=your-twitter-token

# Production Settings
BRAF_ENVIRONMENT=production
DEBUG=false
```

#### Step 3: Deploy Services
```bash
# Build and start all services
docker-compose -f docker/docker-compose.production.yml build
docker-compose -f docker/docker-compose.production.yml up -d

# Wait for services to start
sleep 30

# Run database migrations
docker-compose -f docker/docker-compose.production.yml exec braf_app alembic upgrade head

# Create admin user
docker-compose -f docker/docker-compose.production.yml exec braf_app python scripts/create_admin.py
```

#### Step 4: Verify Deployment
```bash
# Check service status
docker-compose -f docker/docker-compose.production.yml ps

# Test application
curl http://localhost/health
curl http://localhost/docs

# Check logs
docker-compose -f docker/docker-compose.production.yml logs -f braf_app
```

### Option 2: Native Linux Deployment

#### Step 1: System Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv postgresql redis-server nginx git curl

# Create BRAF user
sudo useradd -m -s /bin/bash braf
sudo mkdir -p /opt/braf /var/log/braf /var/lib/braf
sudo chown -R braf:braf /opt/braf /var/log/braf /var/lib/braf
```

#### Step 2: Database Setup
```bash
# Configure PostgreSQL
sudo -u postgres createdb braf_db
sudo -u postgres createuser braf_user
sudo -u postgres psql -c "ALTER USER braf_user WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE braf_db TO braf_user;"

# Start and enable services
sudo systemctl start postgresql redis-server
sudo systemctl enable postgresql redis-server
```

#### Step 3: Application Installation
```bash
# Switch to BRAF user
sudo -u braf -i

# Extract application
cd /opt/braf
unzip braf-live-20251219_134305.zip
cd braf-live-20251219_134305

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r app/requirements-live.txt

# Install Playwright browsers
playwright install
```

#### Step 4: Configuration
```bash
# Copy configuration files
cp config/.env.example app/.env

# Edit configuration
nano app/.env

# Set database URL
DATABASE_URL=postgresql://braf_user:secure_password@localhost:5432/braf_db
REDIS_URL=redis://localhost:6379/0
```

#### Step 5: Database Migration
```bash
# Run migrations
cd app
alembic upgrade head

# Create admin user
python scripts/create_admin.py
```

#### Step 6: System Services
```bash
# Copy service files
sudo cp systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start services
sudo systemctl enable braf braf-worker braf-beat
sudo systemctl start braf braf-worker braf-beat

# Check status
sudo systemctl status braf braf-worker braf-beat
```

#### Step 7: Nginx Configuration
```bash
# Copy nginx configuration
sudo cp nginx/braf.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/braf.conf /etc/nginx/sites-enabled/

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default

# Test and restart nginx
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

### Option 3: Windows Deployment

#### Step 1: Prerequisites
```cmd
# Install Python 3.8+ from python.org
# Install Git from git-scm.com
# Install PostgreSQL from postgresql.org (optional)
# Install Redis for Windows (optional)
```

#### Step 2: Application Setup
```cmd
# Create directory
mkdir C:\BRAF
cd C:\BRAF

# Extract package
# Use Windows Explorer or 7-zip to extract braf-live-20251219_134305.zip

cd braf-live-20251219_134305

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r app\requirements-live.txt
python -m playwright install
```

#### Step 3: Configuration
```cmd
# Copy configuration
copy config\.env.example app\.env

# Edit app\.env with your preferred text editor
# Use SQLite for development: DATABASE_URL=sqlite:///./braf.db
```

#### Step 4: Run Application
```cmd
# Run database migrations
cd app
alembic upgrade head

# Start application
python main.py
```

## SSL/HTTPS Configuration

### Using Let's Encrypt (Linux)
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

### Manual SSL Certificate
```bash
# Copy certificate files
sudo cp your-certificate.crt /etc/ssl/certs/braf.crt
sudo cp your-private-key.key /etc/ssl/private/braf.key

# Update nginx configuration
sudo nano /etc/nginx/sites-available/braf.conf

# Add SSL configuration
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/braf.crt;
    ssl_certificate_key /etc/ssl/private/braf.key;
    # ... rest of configuration
}
```

## Monitoring Setup

### Prometheus and Grafana
```bash
# Start monitoring stack
docker run -d --name prometheus -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

docker run -d --name grafana -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana

# Access Grafana at http://localhost:3000 (admin/admin)
```

### Application Monitoring
```bash
# Check application health
curl http://localhost/health

# Monitor logs
tail -f /var/log/braf/application.log

# Check system resources
htop
df -h
free -h
```

## Backup Configuration

### Automated Database Backups
```bash
# Create backup script
sudo tee /opt/braf/backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/braf/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
pg_dump -U braf_user -h localhost braf_db > $BACKUP_DIR/braf_db_$DATE.sql

# Keep only last 7 days
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete

# Compress old backups
find $BACKUP_DIR -name "*.sql" -mtime +1 -exec gzip {} \;
EOF

# Make executable
sudo chmod +x /opt/braf/backup_db.sh

# Schedule daily backups
echo "0 2 * * * /opt/braf/backup_db.sh" | sudo crontab -
```

### Application Data Backup
```bash
# Create application backup script
sudo tee /opt/braf/backup_app.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/braf/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/app_data_$DATE.tar.gz /opt/braf/data /opt/braf/config

# Keep only last 30 days
find $BACKUP_DIR -name "app_data_*.tar.gz" -mtime +30 -delete
EOF

sudo chmod +x /opt/braf/backup_app.sh
```

## Security Hardening

### Firewall Configuration
```bash
# Configure UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Check status
sudo ufw status verbose
```

### Fail2Ban Setup
```bash
# Install fail2ban
sudo apt install fail2ban

# Configure for nginx
sudo tee /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
EOF

# Start and enable
sudo systemctl start fail2ban
sudo systemctl enable fail2ban
```

### Security Headers
```nginx
# Add to nginx configuration
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=63072000" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

## Performance Optimization

### Database Optimization
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
SELECT pg_reload_conf();
```

### Redis Optimization
```bash
# Edit redis configuration
sudo nano /etc/redis/redis.conf

# Add optimizations
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Tuning
```bash
# Increase worker processes
export WORKERS=4  # 2 * CPU cores

# Optimize gunicorn
gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker main:app
```

## Scaling Configuration

### Load Balancer Setup
```nginx
# Nginx load balancer configuration
upstream braf_backend {
    server 10.0.1.10:8000 weight=3;
    server 10.0.1.11:8000 weight=2;
    server 10.0.1.12:8000 weight=1;
}

server {
    listen 80;
    location / {
        proxy_pass http://braf_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Database Replication
```bash
# Setup PostgreSQL streaming replication
# On master server
echo "wal_level = replica" >> /etc/postgresql/13/main/postgresql.conf
echo "max_wal_senders = 3" >> /etc/postgresql/13/main/postgresql.conf
echo "wal_keep_segments = 64" >> /etc/postgresql/13/main/postgresql.conf

# Create replication user
sudo -u postgres psql -c "CREATE USER replicator REPLICATION LOGIN CONNECTION LIMIT 1 ENCRYPTED PASSWORD 'repl_password';"
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
journalctl -u braf -f
journalctl -u braf-worker -f

# Check configuration
python -c "from app.config import settings; print('Config OK')"

# Check dependencies
pip check
```

#### Database Connection Issues
```bash
# Test database connection
psql -h localhost -U braf_user -d braf_db -c "SELECT 1;"

# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection limits
sudo -u postgres psql -c "SHOW max_connections;"
```

#### Performance Issues
```bash
# Monitor resources
htop
iotop
nethogs

# Check database performance
sudo -u postgres psql -d braf_db -c "SELECT * FROM pg_stat_activity;"

# Analyze slow queries
sudo -u postgres psql -d braf_db -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### Log Locations
| Service | Log Location |
|---------|-------------|
| **Application** | `/var/log/braf/application.log` |
| **Nginx** | `/var/log/nginx/access.log`, `/var/log/nginx/error.log` |
| **PostgreSQL** | `/var/log/postgresql/postgresql-13-main.log` |
| **Redis** | `/var/log/redis/redis-server.log` |
| **System** | `journalctl -u braf` |

## Maintenance Procedures

### Regular Maintenance
```bash
# Weekly maintenance script
#!/bin/bash

# Update system packages
sudo apt update && sudo apt upgrade -y

# Restart services
sudo systemctl restart braf braf-worker braf-beat

# Clean old logs
sudo find /var/log -name "*.log" -mtime +30 -delete

# Vacuum database
sudo -u postgres psql -d braf_db -c "VACUUM ANALYZE;"

# Check disk space
df -h

# Check service status
sudo systemctl status braf braf-worker braf-beat nginx postgresql redis
```

### Health Checks
```bash
# Application health check script
#!/bin/bash

# Check HTTP response
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✓ Application is healthy"
else
    echo "✗ Application health check failed"
    exit 1
fi

# Check database
if sudo -u postgres psql -d braf_db -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Database is accessible"
else
    echo "✗ Database check failed"
    exit 1
fi

# Check Redis
if redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is responding"
else
    echo "✗ Redis check failed"
    exit 1
fi
```

## Deployment Verification

### Functional Tests
```bash
# Test user registration
curl -X POST http://localhost/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass123","company_name":"Test Corp"}'

# Test authentication
curl -X POST http://localhost/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test@example.com","password":"testpass123"}'

# Test API endpoints
curl http://localhost/api/v1/system/status
curl http://localhost/docs
```

### Performance Tests
```bash
# Install Apache Bench
sudo apt install apache2-utils

# Test concurrent requests
ab -n 1000 -c 10 http://localhost/

# Test API performance
ab -n 100 -c 5 -H "Authorization: Bearer YOUR_TOKEN" http://localhost/api/v1/dashboard/earnings/1
```

## Post-Deployment Checklist

- [ ] All services running without errors
- [ ] Database migrations completed successfully
- [ ] SSL/HTTPS configured and working
- [ ] Monitoring and alerting configured
- [ ] Backups scheduled and tested
- [ ] Security measures implemented
- [ ] Performance baselines established
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Disaster recovery plan tested

## Support and Maintenance

### Emergency Contacts
- **System Administrator**: [Your contact info]
- **Database Administrator**: [Your contact info]
- **Security Team**: [Your contact info]

### Escalation Procedures
1. **Level 1**: Service restart, basic troubleshooting
2. **Level 2**: Configuration changes, log analysis
3. **Level 3**: Code changes, architecture modifications

### Maintenance Windows
- **Regular Maintenance**: Sundays 2:00-4:00 AM UTC
- **Emergency Maintenance**: As needed with 2-hour notice
- **Major Updates**: Quarterly with 1-week notice

---

**Deployment completed successfully!** 

Your BRAF system is now running in production. Access the application at your configured domain or IP address. Monitor the system regularly and follow the maintenance procedures to ensure optimal performance.

For additional support, refer to the comprehensive documentation included in the deployment package.