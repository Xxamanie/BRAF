#!/bin/bash
# BRAF Monetization System Deployment Script

set -e

echo "🚀 Deploying BRAF Monetization System..."

# Create application directory
sudo mkdir -p /opt/braf-monetization
sudo chown $USER:$USER /opt/braf-monetization

# Copy application files
cp -r . /opt/braf-monetization/

# Create virtual environment
cd /opt/braf-monetization
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set up database
python -c "
from database import engine
from database.models import Base
Base.metadata.create_all(bind=engine)
print('Database initialized')
"

# Copy systemd service
sudo cp braf-monetization.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable braf-monetization
sudo systemctl start braf-monetization

# Set up nginx (optional)
if command -v nginx &> /dev/null; then
    sudo cp nginx/nginx.conf /etc/nginx/sites-available/braf-monetization
    sudo ln -sf /etc/nginx/sites-available/braf-monetization /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
    echo "✅ Nginx configured"
fi

echo "🎉 Deployment completed!"
echo "📍 Service status: sudo systemctl status braf-monetization"
echo "📍 View logs: sudo journalctl -u braf-monetization -f"
echo "🌐 Access dashboard: http://your-server-ip:8000/dashboard"
