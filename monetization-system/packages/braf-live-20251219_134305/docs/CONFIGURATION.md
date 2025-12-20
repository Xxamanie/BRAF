# BRAF Configuration Guide

## Environment Variables

### Database Configuration
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/braf_db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
```

### Redis Configuration
```bash
REDIS_URL=redis://localhost:6379/0
REDIS_SOCKET_TIMEOUT=5
```

### Security Configuration
```bash
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here
```

### API Keys
```bash
# Social Media APIs
TWITTER_BEARER_TOKEN=your-twitter-token
REDDIT_CLIENT_ID=your-reddit-client-id
REDDIT_CLIENT_SECRET=your-reddit-secret

# Payment Providers
STRIPE_SECRET_KEY=your-stripe-key
OPAY_API_KEY=your-opay-key
PALMPAY_API_KEY=your-palmpay-key

# Cryptocurrency APIs
COINBASE_API_KEY=your-coinbase-key
BINANCE_API_KEY=your-binance-key
```

### Monitoring Configuration
```bash
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
SENTRY_DSN=your-sentry-dsn
```

## Service Configuration

### Nginx Configuration
Location: `nginx/braf.conf`

Key settings:
- `client_max_body_size`: Maximum upload size
- `proxy_timeout`: Backend timeout settings
- SSL certificate paths

### Systemd Services
Location: `systemd/`

Services:
- `braf.service`: Main application
- `braf-worker.service`: Background workers
- `braf-beat.service`: Scheduled tasks

### Docker Configuration
Location: `docker/docker-compose.production.yml`

Key settings:
- Resource limits
- Environment variables
- Volume mounts
- Network configuration

## Security Configuration

### SSL/TLS Setup
1. Generate certificates:
```bash
sudo certbot --nginx -d yourdomain.com
```

2. Update nginx configuration with SSL settings

### Firewall Configuration
```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH (if needed)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

### Database Security
1. Change default passwords
2. Enable SSL connections
3. Configure access restrictions
4. Regular backups

## Performance Tuning

### Database Optimization
```sql
-- PostgreSQL settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

### Redis Optimization
```bash
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```

### Application Tuning
- Worker processes: 2 * CPU cores
- Connection pool size: 20-50
- Cache TTL: 300-3600 seconds

## Monitoring Configuration

### Prometheus Metrics
- Application metrics
- System metrics
- Custom business metrics

### Grafana Dashboards
- System overview
- Application performance
- Business metrics
- Alert management

### Log Management
- Structured logging
- Log rotation
- Centralized logging (optional)
