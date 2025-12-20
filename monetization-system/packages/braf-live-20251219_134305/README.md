# BRAF Live Deployment Package

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
scripts\deploy_windows.bat
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
