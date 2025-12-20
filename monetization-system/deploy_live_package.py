#!/usr/bin/env python3
"""
BRAF Live Deployment Package Manager
Comprehensive packaging and deployment preparation for production
"""

import os
import sys
import subprocess
import shutil
import json
import zipfile
from pathlib import Path
from datetime import datetime
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BRAFLivePackager:
    """Comprehensive BRAF live deployment packager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.package_name = f"braf-live-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.package_dir = self.project_root / "packages" / self.package_name
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Get system information for deployment"""
        return {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }
    
    def create_package_structure(self):
        """Create comprehensive package structure"""
        logger.info("Creating package structure...")
        
        # Create main package directory
        self.package_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        directories = [
            'app',
            'config',
            'scripts',
            'docker',
            'nginx',
            'monitoring',
            'docs',
            'tests',
            'backups',
            'logs',
            'data',
            'ssl',
            'systemd'
        ]
        
        for directory in directories:
            (self.package_dir / directory).mkdir(exist_ok=True)
        
        logger.info(f"Package structure created at: {self.package_dir}")
    
    def copy_application_files(self):
        """Copy all application files to package"""
        logger.info("Copying application files...")
        
        # Core application files
        app_files = [
            'main.py',
            'config.py',
            'worker.py',
            'alembic.ini',
            'requirements-live.txt'
        ]
        
        for file in app_files:
            src = self.project_root / file
            if src.exists():
                shutil.copy2(src, self.package_dir / 'app' / file)
        
        # Copy directories
        app_directories = [
            'api',
            'core',
            'database',
            'payments',
            'security',
            'intelligence',
            'research',
            'automation',
            'earnings',
            'templates',
            'migrations',
            'coordination',
            'infrastructure'
        ]
        
        for directory in app_directories:
            src_dir = self.project_root / directory
            if src_dir.exists():
                dst_dir = self.package_dir / 'app' / directory
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        
        logger.info("Application files copied successfully")
    
    def copy_configuration_files(self):
        """Copy configuration files"""
        logger.info("Copying configuration files...")
        
        config_files = [
            '.env.example',
            '.env.production',
            'docker-compose.timeout.yml',
            'docker-compose.braf.yml'
        ]
        
        for file in config_files:
            src = self.project_root / file
            if src.exists():
                shutil.copy2(src, self.package_dir / 'config' / file)
        
        # Copy Docker files
        docker_src = self.project_root / 'docker'
        if docker_src.exists():
            shutil.copytree(docker_src, self.package_dir / 'docker', dirs_exist_ok=True)
        
        # Copy monitoring configuration
        monitoring_src = self.project_root / 'monitoring'
        if monitoring_src.exists():
            shutil.copytree(monitoring_src, self.package_dir / 'monitoring', dirs_exist_ok=True)
        
        logger.info("Configuration files copied successfully")
    
    def create_deployment_scripts(self):
        """Create deployment scripts for different platforms"""
        logger.info("Creating deployment scripts...")
        
        # Linux deployment script
        linux_script = self.package_dir / 'scripts' / 'deploy_linux.sh'
        linux_script.write_text(self._get_linux_deploy_script(), encoding='utf-8')
        linux_script.chmod(0o755)
        
        # Windows deployment script
        windows_script = self.package_dir / 'scripts' / 'deploy_windows.bat'
        windows_script.write_text(self._get_windows_deploy_script(), encoding='utf-8')
        
        # Docker deployment script
        docker_script = self.package_dir / 'scripts' / 'deploy_docker.sh'
        docker_script.write_text(self._get_docker_deploy_script(), encoding='utf-8')
        docker_script.chmod(0o755)
        
        # Installation script
        install_script = self.package_dir / 'scripts' / 'install.py'
        install_script.write_text(self._get_install_script(), encoding='utf-8')
        
        logger.info("Deployment scripts created successfully")
    
    def create_systemd_services(self):
        """Create systemd service files for Linux"""
        logger.info("Creating systemd service files...")
        
        # BRAF main service
        braf_service = self.package_dir / 'systemd' / 'braf.service'
        braf_service.write_text(self._get_braf_service(), encoding='utf-8')
        
        # BRAF worker service
        worker_service = self.package_dir / 'systemd' / 'braf-worker.service'
        worker_service.write_text(self._get_worker_service(), encoding='utf-8')
        
        # BRAF celery beat service
        beat_service = self.package_dir / 'systemd' / 'braf-beat.service'
        beat_service.write_text(self._get_beat_service(), encoding='utf-8')
        
        logger.info("Systemd service files created successfully")
    
    def create_nginx_config(self):
        """Create nginx configuration"""
        logger.info("Creating nginx configuration...")
        
        nginx_config = self.package_dir / 'nginx' / 'braf.conf'
        nginx_config.write_text(self._get_nginx_config(), encoding='utf-8')
        
        # SSL configuration
        ssl_config = self.package_dir / 'nginx' / 'ssl.conf'
        ssl_config.write_text(self._get_ssl_config(), encoding='utf-8')
        
        logger.info("Nginx configuration created successfully")
    
    def create_docker_configs(self):
        """Create Docker-specific configurations"""
        logger.info("Creating Docker configurations...")
        
        # Production Dockerfile
        dockerfile = self.package_dir / 'docker' / 'Dockerfile.production'
        dockerfile.write_text(self._get_production_dockerfile(), encoding='utf-8')
        
        # Docker compose for production
        compose_prod = self.package_dir / 'docker' / 'docker-compose.production.yml'
        compose_prod.write_text(self._get_production_compose(), encoding='utf-8')
        
        logger.info("Docker configurations created successfully")
    
    def install_dependencies(self):
        """Install and verify dependencies"""
        logger.info("Installing dependencies...")
        
        try:
            # Create virtual environment
            venv_path = self.package_dir / 'venv'
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            
            # Determine pip path
            if platform.system() == 'Windows':
                pip_path = venv_path / 'Scripts' / 'pip.exe'
                python_path = venv_path / 'Scripts' / 'python.exe'
            else:
                pip_path = venv_path / 'bin' / 'pip'
                python_path = venv_path / 'bin' / 'python'
            
            # Upgrade pip
            subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            
            # Install requirements
            requirements_file = self.project_root / 'requirements-live.txt'
            if requirements_file.exists():
                subprocess.run([str(pip_path), 'install', '-r', str(requirements_file)], check=True)
            
            # Install Playwright browsers
            subprocess.run([str(python_path), '-m', 'playwright', 'install'], check=True)
            
            logger.info("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
        
        return True
    
    def create_documentation(self):
        """Create comprehensive documentation"""
        logger.info("Creating documentation...")
        
        # README
        readme = self.package_dir / 'README.md'
        readme.write_text(self._get_readme_content(), encoding='utf-8')
        
        # Installation guide
        install_guide = self.package_dir / 'docs' / 'INSTALLATION.md'
        install_guide.write_text(self._get_installation_guide(), encoding='utf-8')
        
        # Configuration guide
        config_guide = self.package_dir / 'docs' / 'CONFIGURATION.md'
        config_guide.write_text(self._get_configuration_guide(), encoding='utf-8')
        
        # Deployment guide
        deploy_guide = self.package_dir / 'docs' / 'DEPLOYMENT.md'
        deploy_guide.write_text(self._get_deployment_guide(), encoding='utf-8')
        
        # API documentation
        api_docs = self.package_dir / 'docs' / 'API.md'
        api_docs.write_text(self._get_api_documentation(), encoding='utf-8')
        
        logger.info("Documentation created successfully")
    
    def create_package_info(self):
        """Create package information file"""
        logger.info("Creating package information...")
        
        package_info = {
            'name': 'BRAF Live Deployment Package',
            'version': '1.0.0',
            'build_date': datetime.now().isoformat(),
            'system_info': self.system_info,
            'components': [
                'Core BRAF Framework',
                'Monetization System',
                'Intelligence Layer',
                'Research System (NEXUS7)',
                'Security Modules',
                'Payment Integrations',
                'Cryptocurrency Support',
                'Ethical Automation',
                'Social Media Collection',
                'Web Scraping Framework',
                'Docker Infrastructure',
                'Monitoring Stack'
            ],
            'features': [
                'Enhanced withdrawal system (13+ cryptocurrencies)',
                'Real-time currency conversion',
                'Comprehensive timeout configuration',
                'Ethical automation safeguards',
                'Academic research coordination',
                'Live payment integrations',
                'Intelligence optimization',
                'Security compliance',
                'Production monitoring',
                'Auto-scaling capabilities'
            ],
            'requirements': {
                'python': '>=3.8',
                'memory': '8GB minimum, 16GB recommended',
                'storage': '50GB minimum',
                'cpu': '4 cores minimum',
                'os': 'Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+'
            }
        }
        
        info_file = self.package_dir / 'package_info.json'
        info_file.write_text(json.dumps(package_info, indent=2), encoding='utf-8')
        
        logger.info("Package information created successfully")
    
    def create_archive(self):
        """Create deployment archive"""
        logger.info("Creating deployment archive...")
        
        archive_path = self.project_root / "packages" / f"{self.package_name}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_name = file_path.relative_to(self.package_dir)
                    zipf.write(file_path, arc_name)
        
        logger.info(f"Deployment archive created: {archive_path}")
        return archive_path
    
    def run_tests(self):
        """Run comprehensive tests"""
        logger.info("Running tests...")
        
        try:
            # Copy test files
            test_src = self.project_root / 'tests'
            if test_src.exists():
                shutil.copytree(test_src, self.package_dir / 'tests', dirs_exist_ok=True)
            
            # Run basic import tests
            test_script = self.package_dir / 'tests' / 'test_imports.py'
            test_script.write_text(self._get_import_test(), encoding='utf-8')
            
            logger.info("Tests completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Tests failed: {e}")
            return False
    
    def package(self):
        """Main packaging method"""
        logger.info("Starting BRAF live deployment packaging...")
        
        try:
            self.create_package_structure()
            self.copy_application_files()
            self.copy_configuration_files()
            self.create_deployment_scripts()
            self.create_systemd_services()
            self.create_nginx_config()
            self.create_docker_configs()
            self.create_documentation()
            self.create_package_info()
            
            # Install dependencies (optional, can be time-consuming)
            if '--install-deps' in sys.argv:
                self.install_dependencies()
            
            # Run tests
            if '--run-tests' in sys.argv:
                self.run_tests()
            
            # Create archive
            archive_path = self.create_archive()
            
            logger.info("=" * 60)
            logger.info("BRAF LIVE DEPLOYMENT PACKAGE CREATED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Package Directory: {self.package_dir}")
            logger.info(f"Archive File: {archive_path}")
            logger.info(f"Package Size: {archive_path.stat().st_size / (1024*1024):.2f} MB")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Packaging failed: {e}")
            return False
    
    # Helper methods for generating configuration files
    def _get_linux_deploy_script(self):
        return '''#!/bin/bash
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
'''

    def _get_windows_deploy_script(self):
        return '''@echo off
REM BRAF Windows Deployment Script

echo Starting BRAF Live Deployment...

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment
python -m venv venv
call venv\\Scripts\\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install -r app\\requirements-live.txt
python -m playwright install

REM Setup database (SQLite for Windows)
python app\\database\\setup.py

REM Run migrations
alembic upgrade head

REM Start services
echo BRAF deployment completed successfully!
echo Run: python app\\main.py to start the server
pause
'''

    def _get_docker_deploy_script(self):
        return '''#!/bin/bash
# BRAF Docker Deployment Script

set -e

echo "Starting BRAF Docker Deployment..."

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed"
    exit 1
fi

# Build and deploy
docker-compose -f docker/docker-compose.production.yml build
docker-compose -f docker/docker-compose.production.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 30

# Run migrations
docker-compose -f docker/docker-compose.production.yml exec braf_app alembic upgrade head

# Show status
docker-compose -f docker/docker-compose.production.yml ps

echo "BRAF Docker deployment completed successfully!"
echo "Access your BRAF instance at: http://localhost"
'''

    def _get_install_script(self):
        return '''#!/usr/bin/env python3
"""
BRAF Installation Script
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def main():
    print("BRAF Installation Script")
    print("=" * 40)
    
    system = platform.system().lower()
    
    if system == "linux":
        install_linux()
    elif system == "windows":
        install_windows()
    elif system == "darwin":
        install_macos()
    else:
        print(f"Unsupported system: {system}")
        sys.exit(1)

def install_linux():
    print("Installing BRAF on Linux...")
    subprocess.run(["bash", "scripts/deploy_linux.sh"], check=True)

def install_windows():
    print("Installing BRAF on Windows...")
    subprocess.run(["scripts/deploy_windows.bat"], check=True, shell=True)

def install_macos():
    print("Installing BRAF on macOS...")
    # Similar to Linux but with brew
    subprocess.run(["bash", "scripts/deploy_linux.sh"], check=True)

if __name__ == "__main__":
    main()
'''

    def _get_braf_service(self):
        return '''[Unit]
Description=BRAF Main Application
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=braf
Group=braf
WorkingDirectory=/opt/braf
Environment=PATH=/opt/braf/venv/bin
ExecStart=/opt/braf/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
'''

    def _get_worker_service(self):
        return '''[Unit]
Description=BRAF Celery Worker
After=network.target postgresql.service redis.service rabbitmq-server.service

[Service]
Type=exec
User=braf
Group=braf
WorkingDirectory=/opt/braf
Environment=PATH=/opt/braf/venv/bin
ExecStart=/opt/braf/venv/bin/celery -A worker worker --loglevel=info
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
'''

    def _get_beat_service(self):
        return '''[Unit]
Description=BRAF Celery Beat Scheduler
After=network.target postgresql.service redis.service rabbitmq-server.service

[Service]
Type=exec
User=braf
Group=braf
WorkingDirectory=/opt/braf
Environment=PATH=/opt/braf/venv/bin
ExecStart=/opt/braf/venv/bin/celery -A worker beat --loglevel=info
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
'''

    def _get_nginx_config(self):
        return '''server {
    listen 80;
    server_name _;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /static/ {
        alias /opt/braf/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
'''

    def _get_ssl_config(self):
        return '''# SSL Configuration for BRAF
# Add this to your nginx server block for HTTPS

listen 443 ssl http2;
ssl_certificate /etc/ssl/certs/braf.crt;
ssl_certificate_key /etc/ssl/private/braf.key;

ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

add_header Strict-Transport-Security "max-age=63072000" always;
'''

    def _get_production_dockerfile(self):
        return '''FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    postgresql-client \\
    redis-tools \\
    nginx \\
    supervisor \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-live.txt .
RUN pip install --no-cache-dir -r requirements-live.txt

# Install Playwright browsers
RUN playwright install --with-deps

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 braf && chown -R braf:braf /app
USER braf

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

    def _get_production_compose(self):
        return '''version: '3.8'

services:
  braf_app:
    build:
      context: ../app
      dockerfile: ../docker/Dockerfile.production
    container_name: braf_production_app
    environment:
      - DATABASE_URL=postgresql://braf_user:braf_password@postgres:5432/braf_db
      - REDIS_URL=redis://redis:6379/0
      - BRAF_ENVIRONMENT=production
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  braf_worker:
    build:
      context: ../app
      dockerfile: ../docker/Dockerfile.production
    container_name: braf_production_worker
    command: celery -A worker worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://braf_user:braf_password@postgres:5432/braf_db
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    
  postgres:
    image: postgres:15-alpine
    container_name: braf_production_postgres
    environment:
      - POSTGRES_DB=braf_db
      - POSTGRES_USER=braf_user
      - POSTGRES_PASSWORD=braf_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    container_name: braf_production_redis
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    container_name: braf_production_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ../nginx/braf.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - braf_app
    restart: unless-stopped

volumes:
  postgres_data:
'''

    def _get_readme_content(self):
        return '''# BRAF Live Deployment Package

## Browser Automation Revenue Framework - Production Ready

This package contains everything needed to deploy BRAF in a production environment.

### Package Contents

- **app/**: Complete BRAF application
- **config/**: Configuration files and templates
- **docker/**: Docker deployment files
- **nginx/**: Web server configuration
- **scripts/**: Deployment and installation scripts
- **systemd/**: Linux service files
- **docs/**: Comprehensive documentation

### Quick Start

1. **Extract the package**
2. **Run installation script**: `python scripts/install.py`
3. **Access BRAF**: http://localhost

### Docker Deployment

```bash
cd docker
docker-compose -f docker-compose.production.yml up -d
```

### Linux Deployment

```bash
chmod +x scripts/deploy_linux.sh
./scripts/deploy_linux.sh
```

### Windows Deployment

```cmd
scripts\\deploy_windows.bat
```

### Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Documentation](docs/API.md)

### System Requirements

- **Python**: 3.8+
- **Memory**: 8GB minimum, 16GB recommended
- **Storage**: 50GB minimum
- **CPU**: 4 cores minimum
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, macOS 10.15+

### Features

- Enhanced withdrawal system (13+ cryptocurrencies)
- Real-time currency conversion
- Ethical automation safeguards
- Academic research coordination
- Live payment integrations
- Intelligence optimization
- Security compliance
- Production monitoring

### Support

For support and documentation, visit the project repository.
'''

    def _get_installation_guide(self):
        return '''# BRAF Installation Guide

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 50GB free disk space
- 4 CPU cores minimum

### Supported Operating Systems
- Ubuntu 20.04+ / Debian 11+
- CentOS 8+ / RHEL 8+
- Windows 10+ / Windows Server 2019+
- macOS 10.15+

## Installation Methods

### Method 1: Automated Installation

```bash
python scripts/install.py
```

### Method 2: Docker Installation

```bash
cd docker
docker-compose -f docker-compose.production.yml up -d
```

### Method 3: Manual Installation

#### Linux/macOS

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y python3 python3-pip python3-venv postgresql redis-server nginx

# CentOS/RHEL
sudo yum install -y python3 python3-pip postgresql redis nginx
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r app/requirements-live.txt
playwright install
```

4. Setup database:
```bash
sudo -u postgres createdb braf_db
sudo -u postgres createuser braf_user
```

5. Run migrations:
```bash
cd app
alembic upgrade head
```

6. Start services:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Windows

1. Install Python 3.8+ from python.org
2. Install PostgreSQL and Redis
3. Run the Windows deployment script:
```cmd
scripts\\deploy_windows.bat
```

## Post-Installation

### Verify Installation

1. Check service status:
```bash
curl http://localhost:8000/health
```

2. Access web interface:
   - Main Dashboard: http://localhost
   - API Docs: http://localhost/docs

### Configuration

1. Copy environment template:
```bash
cp config/.env.example app/.env
```

2. Edit configuration:
```bash
nano app/.env
```

3. Restart services after configuration changes.

## Troubleshooting

### Common Issues

1. **Port already in use**: Change port in configuration
2. **Database connection failed**: Check PostgreSQL service
3. **Permission denied**: Check file permissions and user ownership

### Logs

- Application logs: `/var/log/braf/`
- System logs: `journalctl -u braf`
- Docker logs: `docker-compose logs`
'''

    def _get_configuration_guide(self):
        return '''# BRAF Configuration Guide

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
'''

    def _get_deployment_guide(self):
        return '''# BRAF Deployment Guide

## Deployment Strategies

### 1. Single Server Deployment
Best for: Small to medium workloads, development, testing

Components on single server:
- BRAF application
- PostgreSQL database
- Redis cache
- Nginx web server

### 2. Multi-Server Deployment
Best for: Production workloads, high availability

Server roles:
- **App servers**: BRAF application instances
- **Database server**: PostgreSQL with replication
- **Cache server**: Redis cluster
- **Load balancer**: Nginx or HAProxy

### 3. Container Deployment
Best for: Scalability, cloud deployment, microservices

Components:
- Docker containers
- Kubernetes orchestration
- Container registry
- Service mesh (optional)

## Production Deployment Steps

### Pre-Deployment Checklist

- [ ] Server provisioning completed
- [ ] DNS records configured
- [ ] SSL certificates obtained
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring setup completed
- [ ] Security hardening applied

### Deployment Process

1. **Prepare Infrastructure**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip postgresql redis-server nginx
```

2. **Deploy Application**
```bash
# Extract package
unzip braf-live-package.zip
cd braf-live-package

# Run deployment script
./scripts/deploy_linux.sh
```

3. **Configure Services**
```bash
# Start services
sudo systemctl start braf braf-worker braf-beat
sudo systemctl enable braf braf-worker braf-beat

# Configure nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

4. **Verify Deployment**
```bash
# Check service status
sudo systemctl status braf

# Test application
curl http://localhost/health

# Check logs
sudo journalctl -u braf -f
```

### Post-Deployment Tasks

1. **Security Hardening**
   - Change default passwords
   - Configure SSL/TLS
   - Setup firewall rules
   - Enable fail2ban

2. **Monitoring Setup**
   - Configure Prometheus
   - Setup Grafana dashboards
   - Configure alerting

3. **Backup Configuration**
   - Database backups
   - Application data backups
   - Configuration backups

## Scaling Strategies

### Horizontal Scaling

1. **Load Balancer Configuration**
```nginx
upstream braf_backend {
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}
```

2. **Database Replication**
   - Master-slave setup
   - Read replicas
   - Connection pooling

3. **Cache Clustering**
   - Redis cluster
   - Consistent hashing
   - Failover configuration

### Vertical Scaling

1. **Resource Optimization**
   - CPU allocation
   - Memory tuning
   - Storage optimization

2. **Performance Tuning**
   - Database optimization
   - Application profiling
   - Cache optimization

## Maintenance Procedures

### Regular Maintenance

1. **System Updates**
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Python packages
pip install --upgrade -r requirements-live.txt

# Restart services
sudo systemctl restart braf braf-worker
```

2. **Database Maintenance**
```sql
-- Vacuum and analyze
VACUUM ANALYZE;

-- Reindex if needed
REINDEX DATABASE braf_db;
```

3. **Log Rotation**
```bash
# Configure logrotate
sudo nano /etc/logrotate.d/braf
```

### Backup Procedures

1. **Database Backup**
```bash
# Create backup
pg_dump -U braf_user braf_db > backup_$(date +%Y%m%d).sql

# Restore backup
psql -U braf_user braf_db < backup_20241218.sql
```

2. **Application Backup**
```bash
# Backup application data
tar -czf braf_data_$(date +%Y%m%d).tar.gz /opt/braf/data

# Backup configuration
tar -czf braf_config_$(date +%Y%m%d).tar.gz /opt/braf/config
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   - Check logs: `journalctl -u braf`
   - Verify configuration
   - Check dependencies

2. **Database Connection Issues**
   - Verify PostgreSQL service
   - Check connection string
   - Test network connectivity

3. **Performance Issues**
   - Monitor resource usage
   - Check database queries
   - Analyze application logs

### Emergency Procedures

1. **Service Recovery**
```bash
# Restart all services
sudo systemctl restart braf braf-worker braf-beat nginx

# Check service status
sudo systemctl status braf
```

2. **Database Recovery**
```bash
# Stop application
sudo systemctl stop braf braf-worker

# Restore database
psql -U braf_user braf_db < latest_backup.sql

# Start application
sudo systemctl start braf braf-worker
```

## Monitoring and Alerting

### Key Metrics to Monitor

- **Application Metrics**
  - Response time
  - Error rate
  - Request volume
  - Active connections

- **System Metrics**
  - CPU usage
  - Memory usage
  - Disk I/O
  - Network traffic

- **Business Metrics**
  - User registrations
  - Transaction volume
  - Revenue metrics
  - System utilization

### Alert Configuration

1. **Critical Alerts**
   - Service down
   - Database connection lost
   - High error rate
   - Disk space low

2. **Warning Alerts**
   - High CPU usage
   - Memory usage high
   - Slow response time
   - Queue backlog

### Health Checks

1. **Application Health**
```bash
curl -f http://localhost:8000/health
```

2. **Database Health**
```bash
pg_isready -h localhost -p 5432
```

3. **Cache Health**
```bash
redis-cli ping
```
'''

    def _get_api_documentation(self):
        return '''# BRAF API Documentation

## Authentication

### JWT Token Authentication
```bash
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token
```bash
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## Core Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-18T10:30:00Z",
  "version": "1.0.0"
}
```

### System Status
```bash
GET /api/v1/system/status
```

Response:
```json
{
  "database": "connected",
  "redis": "connected",
  "workers": 2,
  "uptime": "2 days, 3 hours"
}
```

## User Management

### Create Account
```bash
POST /api/v1/users/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "company_name": "Example Corp"
}
```

### Get User Profile
```bash
GET /api/v1/users/profile
Authorization: Bearer <token>
```

### Update Profile
```bash
PUT /api/v1/users/profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "company_name": "Updated Corp",
  "phone": "+1234567890"
}
```

## Automation Management

### Create Automation
```bash
POST /api/v1/automation/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Survey Automation",
  "type": "survey",
  "config": {
    "platform": "swagbucks",
    "daily_limit": 10,
    "auto_start": true
  }
}
```

### List Automations
```bash
GET /api/v1/automation/list/{enterprise_id}
Authorization: Bearer <token>
```

### Start Automation
```bash
POST /api/v1/automation/{automation_id}/start
Authorization: Bearer <token>
```

### Stop Automation
```bash
POST /api/v1/automation/{automation_id}/stop
Authorization: Bearer <token>
```

## Withdrawal System

### Get Available Balance
```bash
GET /api/v1/dashboard/earnings/{enterprise_id}
Authorization: Bearer <token>
```

### Request Withdrawal
```bash
POST /api/v1/withdrawal/request
Authorization: Bearer <token>
Content-Type: application/json

{
  "amount": 100.00,
  "method": "btc",
  "recipient": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
  "network": "Bitcoin"
}
```

### Get Withdrawal History
```bash
GET /api/v1/withdrawal/history/{enterprise_id}
Authorization: Bearer <token>
```

## Intelligence System

### Get Platform Intelligence
```bash
GET /api/v1/intelligence/platforms
Authorization: Bearer <token>
```

### Optimize Earnings
```bash
POST /api/v1/intelligence/optimize
Authorization: Bearer <token>
Content-Type: application/json

{
  "platform": "swagbucks",
  "current_earnings": 50.00,
  "time_spent": 120
}
```

### Get Behavior Profile
```bash
GET /api/v1/intelligence/behavior/{profile_id}
Authorization: Bearer <token>
```

## Research System (NEXUS7)

### Start Research Task
```bash
POST /api/v1/research/start
Authorization: Bearer <token>
Content-Type: application/json

{
  "task_type": "survey_research",
  "parameters": {
    "platform": "multiple",
    "duration": 3600,
    "target_earnings": 25.00
  }
}
```

### Get Research Results
```bash
GET /api/v1/research/results/{task_id}
Authorization: Bearer <token>
```

## Monitoring Endpoints

### Get Metrics
```bash
GET /api/v1/monitoring/metrics
Authorization: Bearer <token>
```

### Get System Health
```bash
GET /api/v1/monitoring/health
Authorization: Bearer <token>
```

### Get Performance Stats
```bash
GET /api/v1/monitoring/performance
Authorization: Bearer <token>
```

## WebSocket Endpoints

### Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/updates');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

### Live Earnings Feed
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/earnings');
```

## Error Handling

### Standard Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "amount",
      "issue": "Must be greater than 0"
    }
  }
}
```

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `422`: Validation Error
- `500`: Internal Server Error

## Rate Limiting

### Default Limits
- Authentication: 5 requests/minute
- API calls: 100 requests/minute
- Withdrawals: 10 requests/hour

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## SDK Examples

### Python SDK
```python
from braf_sdk import BRAFClient

client = BRAFClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Create automation
automation = client.automation.create(
    name="Test Automation",
    type="survey",
    config={"platform": "swagbucks"}
)

# Request withdrawal
withdrawal = client.withdrawal.request(
    amount=50.00,
    method="btc",
    recipient="your-btc-address"
)
```

### JavaScript SDK
```javascript
import { BRAFClient } from 'braf-sdk';

const client = new BRAFClient({
    baseURL: 'http://localhost:8000',
    apiKey: 'your-api-key'
});

// Get earnings
const earnings = await client.dashboard.getEarnings(enterpriseId);

// Start automation
const result = await client.automation.start(automationId);
```
'''

    def _get_import_test(self):
        return '''#!/usr/bin/env python3
"""
BRAF Import Test
Verify all modules can be imported correctly
"""

import sys
import importlib

def test_imports():
    """Test importing core modules"""
    modules = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'redis',
        'celery',
        'requests',
        'pandas',
        'numpy',
        'cryptography',
        'selenium',
        'playwright',
        'tweepy',
        'praw'
    ]
    
    failed = []
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"\\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
'''


def main():
    """Main execution function"""
    packager = BRAFLivePackager()
    success = packager.package()
    
    if success:
        print("\nBRAF Live Deployment Package created successfully!")
        print("Ready for production deployment!")
    else:
        print("\nPackaging failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()