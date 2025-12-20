#!/bin/bash
# BRAF Linux Deployment Script

set -e

echo "Starting BRAF Live Deployment..."

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root"
   exit 1
fi

# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv nginx postgresql redis-server supervisor docker.io docker-compose

# Create BRAF user
sudo useradd -m -s /bin/bash braf || true

# Create directories
sudo mkdir -p /opt/braf /var/log/braf /var/lib/braf
sudo chown -R braf:braf /opt/braf /var/log/braf /var/lib/braf

# Copy application files
sudo cp -r app/* /opt/braf/
sudo chown -R braf:braf /opt/braf

# Install Python dependencies
cd /opt/braf
sudo -u braf python3 -m venv venv
sudo -u braf ./venv/bin/pip install -r requirements-live.txt
sudo -u braf ./venv/bin/playwright install

# Setup database
sudo -u postgres createdb braf_db || true
sudo -u postgres createuser braf_user || true
sudo -u postgres psql -c "ALTER USER braf_user WITH PASSWORD 'braf_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE braf_db TO braf_user;"

# Run migrations
sudo -u braf ./venv/bin/alembic upgrade head

# Install systemd services
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable braf braf-worker braf-beat
sudo systemctl start braf braf-worker braf-beat

# Configure nginx
sudo cp nginx/braf.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/braf.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

echo "BRAF deployment completed successfully!"
echo "Access your BRAF instance at: http://localhost"
