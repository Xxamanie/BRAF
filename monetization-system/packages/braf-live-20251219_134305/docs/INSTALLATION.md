# BRAF Installation Guide

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
scripts\deploy_windows.bat
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
