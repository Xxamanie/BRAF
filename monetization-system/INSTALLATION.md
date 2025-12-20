# Installation Guide

## System Requirements

- Python 3.8+
- 2GB RAM minimum
- 10GB disk space
- Ubuntu 20.04+ or similar Linux distribution

## Quick Installation

### 1. Download and Extract
```bash
# Download the package
wget https://github.com/your-repo/braf-monetization/archive/main.zip
unzip main.zip
cd braf-monetization-main
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y nginx postgresql redis-server
```

### 3. Configure Environment
```bash
# Copy environment configuration
cp .env.example .env

# Edit configuration
nano .env
```

### 4. Initialize Database
```bash
# Create database tables
python -c "
from database import engine
from database.models import Base
Base.metadata.create_all(bind=engine)
print('Database initialized successfully')
"
```

### 5. Create First Account
```bash
# Create admin account
python create_account.py
```

### 6. Start Service
```bash
# Development mode
python run_server.py

# Production mode
./deploy.sh
```

## Production Deployment

### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3 python3-pip python3-venv nginx git
```

### 2. Deploy Application
```bash
# Clone repository
git clone https://github.com/your-repo/braf-monetization.git
cd braf-monetization

# Run deployment script
./deploy.sh
```

### 3. Configure SSL (Optional)
```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

### 4. Configure Firewall
```bash
# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

## Verification

### 1. Check Service Status
```bash
sudo systemctl status braf-monetization
```

### 2. Test API
```bash
curl http://localhost:8000/health
```

### 3. Access Web Interface
Open browser to: http://your-server-ip:8000

## Troubleshooting

### Service Won't Start
```bash
# Check logs
sudo journalctl -u braf-monetization -f

# Check configuration
python -c "from config import Config; print(Config.DATABASE_URL)"
```

### Database Issues
```bash
# Reset database
rm braf.db
python -c "
from database import engine
from database.models import Base
Base.metadata.create_all(bind=engine)
"
```

### Permission Issues
```bash
# Fix permissions
sudo chown -R www-data:www-data /opt/braf-monetization
sudo chmod +x /opt/braf-monetization/manage.sh
```

## Maintenance

### Regular Backups
```bash
# Create backup
./manage.sh backup

# Restore backup
cp backups/braf_backup_YYYYMMDD_HHMMSS.db braf_production.db
```

### Updates
```bash
# Update system
./manage.sh update
```

### Monitoring
```bash
# View real-time logs
./manage.sh logs

# Check system status
./manage.sh status
```
