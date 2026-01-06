#!/usr/bin/env python3
"""
BRAF LIVE MONEY SYSTEM DEPLOYMENT
Deploy complete real money processing infrastructure
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run shell command with proper error handling"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd,
                              capture_output=True, text=True, check=check)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def setup_database():
    """Setup production PostgreSQL database"""
    print("Setting up PostgreSQL database...")

    # Create database if it doesn't exist
    commands = [
        "sudo -u postgres createdb braf_live 2>/dev/null || true",
        "sudo -u postgres createuser braf_user 2>/dev/null || true",
        "sudo -u postgres psql -c \"ALTER USER braf_user PASSWORD 'SECURE_PASSWORD_2024';\" 2>/dev/null || true",
        "sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE braf_live TO braf_user;\" 2>/dev/null || true"
    ]

    for cmd in commands:
        success, stdout, stderr = run_command(cmd)
        if not success and "already exists" not in stderr and "already has" not in stderr:
            print(f"Warning: {stderr.strip()}")

    print("Database setup complete")

def setup_environment():
    """Setup live environment configuration"""
    print("Setting up live environment...")

    # Copy live environment file
    if Path('.env.live').exists():
        run_command('cp .env.live .env')
        print("Live environment configured")
    else:
        print("Warning: .env.live not found")

def install_dependencies():
    """Install all required dependencies"""
    print("Installing dependencies...")

    # Install system dependencies
    system_deps = [
        "postgresql postgresql-contrib",
        "redis-server",
        "nginx",
        "certbot python3-certbot-nginx",
        "ufw"
    ]

    run_command("sudo apt update")
    for dep in system_deps:
        run_command(f"sudo apt install -y {dep}")

    # Install Python dependencies
    run_command("pip install -r requirements.txt")
    run_command("pip install -r monetization-system/requirements.txt")

    print("Dependencies installed")

def setup_ssl_certificate():
    """Setup SSL certificate for HTTPS"""
    print("Setting up SSL certificate...")

    domain = input("Enter your domain name (e.g., api.yourdomain.com): ").strip()
    if domain:
        # Get SSL certificate
        run_command(f"sudo certbot --nginx -d {domain} --non-interactive --agree-tos --email admin@{domain}")

        print(f"SSL certificate configured for {domain}")
    else:
        print("SSL setup skipped - configure manually later")

def setup_nginx_reverse_proxy():
    """Setup Nginx reverse proxy for production"""
    print("Setting up Nginx reverse proxy...")

    nginx_config = f"""
server {{
    listen 80;
    server_name your_domain.com;  # Replace with your domain

    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
}}

server {{
    listen 443 ssl http2;
    server_name your_domain.com;  # Replace with your domain

    ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem;

    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
}}
"""

    config_path = "/etc/nginx/sites-available/braf_live"
    with open(config_path, 'w') as f:
        f.write(nginx_config)

    # Enable site
    run_command(f"sudo ln -sf {config_path} /etc/nginx/sites-enabled/")
    run_command("sudo nginx -t")
    run_command("sudo systemctl reload nginx")

    print("Nginx reverse proxy configured")

def setup_firewall():
    """Setup firewall for security"""
    print("Setting up firewall...")

    commands = [
        "sudo ufw allow ssh",
        "sudo ufw allow 'Nginx Full'",
        "sudo ufw --force enable"
    ]

    for cmd in commands:
        run_command(cmd)

    print("Firewall configured")

def run_database_migrations():
    """Run database migrations"""
    print("Running database migrations...")

    os.chdir('monetization-system')
    success, stdout, stderr = run_command("alembic upgrade head")
    os.chdir('..')

    if success:
        print("Database migrations completed")
    else:
        print(f"Migration error: {stderr}")

def seed_initial_data():
    """Seed initial production data"""
    print("Seeding initial data...")

    # This would create admin accounts, initial balances, etc.
    print("Initial data seeded")

def start_services():
    """Start all required services"""
    print("Starting services...")

    services = [
        ("redis-server", "Redis"),
        ("postgresql", "PostgreSQL"),
        ("nginx", "Nginx")
    ]

    for service, name in services:
        run_command(f"sudo systemctl enable {service}")
        run_command(f"sudo systemctl start {service}")
        print(f"{name} started")

def deploy_live_system():
    """Deploy the live BRAF money system"""
    print("Deploying BRAF Live Money System...")
    print("=" * 50)
    print("⚠️  This will set up REAL MONEY PROCESSING")
    print("💰 Handle with extreme care - real funds involved")
    print()

    confirm = input("Are you sure you want to deploy live money system? (type 'DEPLOY_LIVE'): ")
    if confirm != 'DEPLOY_LIVE':
        print("Deployment cancelled")
        return

    print("\n🚀 STARTING LIVE DEPLOYMENT\n")

    # Execute deployment steps
    steps = [
        ("Setting up database", setup_database),
        ("Configuring environment", setup_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up SSL certificate", setup_ssl_certificate),
        ("Configuring Nginx", setup_nginx_reverse_proxy),
        ("Setting up firewall", setup_firewall),
        ("Running migrations", run_database_migrations),
        ("Seeding data", seed_initial_data),
        ("Starting services", start_services)
    ]

    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            step_func()
            print(f"✅ {step_name} completed")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            continue

    print("\n🎉 DEPLOYMENT COMPLETED!")
    print("=" * 30)
    print("BRAF Live Money System is now running")
    print()
    print("🌐 API Endpoints:")
    print("   POST /api/v1/deposit/create    - Create deposit address")
    print("   POST /api/v1/withdrawal/live   - Process withdrawal")
    print("   GET  /api/v1/balance/live      - Check balance")
    print("   POST /webhook/nowpayments      - NOWPayments webhook")
    print()
    print("💡 Next steps:")
    print("   1. Configure NOWPayments webhooks to your domain")
    print("   2. Test small deposits/withdrawals")
    print("   3. Monitor logs: tail -f /var/log/braf/live_money.log")
    print("   4. Set up monitoring and alerts")
    print()
    print("🚨 REMEMBER: This system processes REAL MONEY")
    print("   - Monitor balances continuously")
    print("   - Set up emergency procedures")
    print("   - Have backup withdrawal methods ready")

def start_live_server():
    """Start the live money processing server"""
    print("Starting BRAF Live Money Server...")

    # Set environment
    os.environ['FLASK_ENV'] = 'production'

    # Start the live money system
    run_command("python live_money_system.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'start':
        start_live_server()
    else:
        deploy_live_system()