"""
Deployment and Configuration Management for BRAF.

This module provides Docker containerization, docker-compose configuration,
environment-specific configuration management, and deployment scripts.
"""

import json
import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Deployment environments."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ServiceType(str, Enum):
    """Types of services in the deployment."""
    
    C2_SERVER = "c2_server"
    WORKER_NODE = "worker_node"
    DATABASE = "database"
    REDIS = "redis"
    MONITORING = "monitoring"
    VAULT = "vault"


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    
    name: str
    service_type: ServiceType
    image: str
    ports: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    health_check: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentConfig:
    """Complete deployment configuration."""
    
    environment: Environment
    services: List[ServiceConfig]
    networks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    volumes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    configs: Dict[str, Any] = field(default_factory=dict)


class DockerfileGenerator:
    """Generates Dockerfiles for BRAF components."""
    
    def __init__(self):
        """Initialize Dockerfile generator."""
        self.base_python_image = "python:3.11-slim"
        self.playwright_image = "mcr.microsoft.com/playwright/python:v1.40.0-jammy"
    
    def generate_c2_dockerfile(self) -> str:
        """Generate Dockerfile for C2 server."""
        return f"""
FROM {self.base_python_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Create non-root user
RUN useradd --create-home --shell /bin/bash braf
RUN chown -R braf:braf /app
USER braf

# Expose ports
EXPOSE 8000 50051 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run C2 server
CMD ["python", "-m", "braf.c2.main"]
"""
    
    def generate_worker_dockerfile(self) -> str:
        """Generate Dockerfile for worker node."""
        return f"""
FROM {self.playwright_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    tesseract-ocr \\
    && rm -rf /var/lib/apt/lists/*

# Install Playwright browsers
RUN playwright install --with-deps chromium firefox webkit

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash braf
RUN chown -R braf:braf /app
USER braf

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import sys; sys.exit(0)"

# Run worker node
CMD ["python", "-m", "braf.worker.main"]
"""
    
    def generate_monitoring_dockerfile(self) -> str:
        """Generate Dockerfile for monitoring stack."""
        return f"""
FROM {self.base_python_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-monitoring.txt .
RUN pip install --no-cache-dir -r requirements-monitoring.txt

# Copy monitoring configuration
COPY monitoring/ ./monitoring/
COPY src/braf/core/monitoring.py ./src/braf/core/

# Create non-root user
RUN useradd --create-home --shell /bin/bash monitoring
RUN chown -R monitoring:monitoring /app
USER monitoring

# Expose Prometheus port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:9090/-/healthy || exit 1

# Run monitoring
CMD ["python", "-m", "monitoring.main"]
"""


class DockerComposeGenerator:
    """Generates docker-compose configurations."""
    
    def __init__(self):
        """Initialize docker-compose generator."""
        pass
    
    def generate_development_compose(self) -> Dict[str, Any]:
        """Generate docker-compose for development environment."""
        return {
            "version": "3.8",
            "services": {
                "postgres": {
                    "image": "postgres:15",
                    "environment": {
                        "POSTGRES_DB": "braf_dev",
                        "POSTGRES_USER": "braf",
                        "POSTGRES_PASSWORD": "braf_dev_password"
                    },
                    "ports": ["5432:5432"],
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data",
                        "./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql"
                    ],
                    "networks": ["braf_network"]
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"],
                    "networks": ["braf_network"]
                },
                "c2_server": {
                    "build": {
                        "context": ".",
                        "dockerfile": "docker/Dockerfile.c2"
                    },
                    "ports": [
                        "8000:8000",  # HTTP API
                        "50051:50051",  # gRPC
                        "8765:8765"   # WebSocket
                    ],
                    "environment": {
                        "BRAF_ENV": "development",
                        "DATABASE_URL": "postgresql://braf:braf_dev_password@postgres:5432/braf_dev",
                        "REDIS_URL": "redis://redis:6379/0",
                        "LOG_LEVEL": "DEBUG"
                    },
                    "depends_on": ["postgres", "redis"],
                    "volumes": [
                        "./config:/app/config",
                        "./logs:/app/logs"
                    ],
                    "networks": ["braf_network"]
                },
                "worker_node": {
                    "build": {
                        "context": ".",
                        "dockerfile": "docker/Dockerfile.worker"
                    },
                    "environment": {
                        "BRAF_ENV": "development",
                        "C2_ENDPOINT": "http://c2_server:8000",
                        "WORKER_ID": "dev_worker_1",
                        "MAX_CONCURRENT_TASKS": "2"
                    },
                    "depends_on": ["c2_server"],
                    "volumes": [
                        "./config:/app/config",
                        "./logs:/app/logs"
                    ],
                    "networks": ["braf_network"],
                    "scale": 2
                },
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "prometheus_data:/prometheus"
                    ],
                    "command": [
                        "--config.file=/etc/prometheus/prometheus.yml",
                        "--storage.tsdb.path=/prometheus",
                        "--web.console.libraries=/etc/prometheus/console_libraries",
                        "--web.console.templates=/etc/prometheus/consoles",
                        "--web.enable-lifecycle"
                    ],
                    "networks": ["braf_network"]
                },
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "ports": ["3000:3000"],
                    "environment": {
                        "GF_SECURITY_ADMIN_PASSWORD": "admin"
                    },
                    "volumes": [
                        "grafana_data:/var/lib/grafana",
                        "./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards",
                        "./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources"
                    ],
                    "networks": ["braf_network"]
                }
            },
            "networks": {
                "braf_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {},
                "prometheus_data": {},
                "grafana_data": {}
            }
        }
    
    def generate_production_compose(self) -> Dict[str, Any]:
        """Generate docker-compose for production environment."""
        return {
            "version": "3.8",
            "services": {
                "postgres": {
                    "image": "postgres:15",
                    "environment": {
                        "POSTGRES_DB": "${POSTGRES_DB}",
                        "POSTGRES_USER": "${POSTGRES_USER}",
                        "POSTGRES_PASSWORD_FILE": "/run/secrets/postgres_password"
                    },
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data"
                    ],
                    "secrets": ["postgres_password"],
                    "networks": ["braf_internal"],
                    "deploy": {
                        "replicas": 1,
                        "resources": {
                            "limits": {
                                "memory": "2G",
                                "cpus": "1.0"
                            }
                        },
                        "restart_policy": {
                            "condition": "on-failure",
                            "delay": "5s",
                            "max_attempts": 3
                        }
                    }
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "volumes": ["redis_data:/data"],
                    "networks": ["braf_internal"],
                    "deploy": {
                        "replicas": 1,
                        "resources": {
                            "limits": {
                                "memory": "512M",
                                "cpus": "0.5"
                            }
                        }
                    }
                },
                "vault": {
                    "image": "vault:latest",
                    "cap_add": ["IPC_LOCK"],
                    "environment": {
                        "VAULT_DEV_ROOT_TOKEN_ID": "${VAULT_ROOT_TOKEN}",
                        "VAULT_DEV_LISTEN_ADDRESS": "0.0.0.0:8200"
                    },
                    "ports": ["8200:8200"],
                    "volumes": [
                        "vault_data:/vault/data",
                        "./vault/config:/vault/config"
                    ],
                    "networks": ["braf_internal"],
                    "deploy": {
                        "replicas": 1,
                        "resources": {
                            "limits": {
                                "memory": "256M",
                                "cpus": "0.25"
                            }
                        }
                    }
                },
                "c2_server": {
                    "image": "${REGISTRY}/braf-c2:${VERSION}",
                    "ports": [
                        "8000:8000",
                        "50051:50051",
                        "8765:8765"
                    ],
                    "environment": {
                        "BRAF_ENV": "production",
                        "DATABASE_URL": "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}",
                        "REDIS_URL": "redis://redis:6379/0",
                        "VAULT_URL": "http://vault:8200",
                        "VAULT_TOKEN": "${VAULT_TOKEN}"
                    },
                    "depends_on": ["postgres", "redis", "vault"],
                    "secrets": ["postgres_password", "vault_token"],
                    "networks": ["braf_internal", "braf_external"],
                    "deploy": {
                        "replicas": 2,
                        "resources": {
                            "limits": {
                                "memory": "1G",
                                "cpus": "1.0"
                            }
                        },
                        "restart_policy": {
                            "condition": "on-failure"
                        },
                        "update_config": {
                            "parallelism": 1,
                            "delay": "10s"
                        }
                    }
                },
                "worker_node": {
                    "image": "${REGISTRY}/braf-worker:${VERSION}",
                    "environment": {
                        "BRAF_ENV": "production",
                        "C2_ENDPOINT": "http://c2_server:8000",
                        "MAX_CONCURRENT_TASKS": "3"
                    },
                    "depends_on": ["c2_server"],
                    "networks": ["braf_internal"],
                    "deploy": {
                        "replicas": 5,
                        "resources": {
                            "limits": {
                                "memory": "2G",
                                "cpus": "2.0"
                            }
                        },
                        "restart_policy": {
                            "condition": "on-failure"
                        }
                    }
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "ports": ["80:80", "443:443"],
                    "volumes": [
                        "./nginx/nginx.conf:/etc/nginx/nginx.conf",
                        "./nginx/ssl:/etc/nginx/ssl"
                    ],
                    "depends_on": ["c2_server"],
                    "networks": ["braf_external", "braf_internal"],
                    "deploy": {
                        "replicas": 1,
                        "resources": {
                            "limits": {
                                "memory": "128M",
                                "cpus": "0.25"
                            }
                        }
                    }
                }
            },
            "networks": {
                "braf_internal": {
                    "driver": "overlay",
                    "internal": True
                },
                "braf_external": {
                    "driver": "overlay"
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {},
                "vault_data": {}
            },
            "secrets": {
                "postgres_password": {
                    "external": True
                },
                "vault_token": {
                    "external": True
                }
            }
        }


class ConfigurationManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, config_dir: Path = Path("config")):
        """Initialize configuration manager."""
        self.config_dir = config_dir
        self.configs: Dict[Environment, Dict[str, Any]] = {}
    
    def load_configurations(self):
        """Load all environment configurations."""
        for env in Environment:
            config_file = self.config_dir / f"{env.value}.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.configs[env] = yaml.safe_load(f)
                logger.info(f"Loaded configuration for {env.value}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
    
    def get_config(self, environment: Environment) -> Dict[str, Any]:
        """Get configuration for specific environment."""
        if environment not in self.configs:
            self.load_configurations()
        
        return self.configs.get(environment, {})
    
    def generate_default_configs(self):
        """Generate default configuration files."""
        configs = {
            Environment.DEVELOPMENT: self._get_development_config(),
            Environment.TESTING: self._get_testing_config(),
            Environment.STAGING: self._get_staging_config(),
            Environment.PRODUCTION: self._get_production_config()
        }
        
        # Create config directory
        self.config_dir.mkdir(exist_ok=True)
        
        # Write configuration files
        for env, config in configs.items():
            config_file = self.config_dir / f"{env.value}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Generated configuration: {config_file}")
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Get development configuration."""
        return {
            "database": {
                "url": "postgresql://braf:braf_dev_password@localhost:5432/braf_dev",
                "pool_size": 5,
                "echo": True
            },
            "redis": {
                "url": "redis://localhost:6379/0"
            },
            "logging": {
                "level": "DEBUG",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "security": {
                "secret_key": "dev_secret_key_change_in_production",
                "use_vault": False
            },
            "monitoring": {
                "prometheus_port": 8000,
                "enable_metrics": True
            },
            "worker": {
                "max_concurrent_tasks": 2,
                "heartbeat_interval": 30,
                "health_check_interval": 60
            },
            "captcha": {
                "test_mode": True,
                "primary_service": "2captcha",
                "fallback_ocr": True
            },
            "browser": {
                "headless": True,
                "max_instances": 5,
                "instance_timeout": 300
            }
        }
    
    def _get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration."""
        config = self._get_development_config()
        config.update({
            "database": {
                "url": "postgresql://braf:braf_test_password@localhost:5432/braf_test",
                "pool_size": 2,
                "echo": False
            },
            "logging": {
                "level": "INFO"
            },
            "captcha": {
                "test_mode": True,
                "primary_service": "test",
                "fallback_ocr": False
            }
        })
        return config
    
    def _get_staging_config(self) -> Dict[str, Any]:
        """Get staging configuration."""
        return {
            "database": {
                "url": "${DATABASE_URL}",
                "pool_size": 10,
                "echo": False
            },
            "redis": {
                "url": "${REDIS_URL}"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "security": {
                "secret_key": "${SECRET_KEY}",
                "use_vault": True,
                "vault_url": "${VAULT_URL}",
                "vault_token": "${VAULT_TOKEN}"
            },
            "monitoring": {
                "prometheus_port": 8000,
                "enable_metrics": True,
                "elasticsearch_host": "${ELASTICSEARCH_HOST}"
            },
            "worker": {
                "max_concurrent_tasks": 3,
                "heartbeat_interval": 30,
                "health_check_interval": 60
            },
            "captcha": {
                "test_mode": False,
                "primary_service": "2captcha",
                "api_key": "${CAPTCHA_API_KEY}",
                "fallback_ocr": True
            },
            "browser": {
                "headless": True,
                "max_instances": 10,
                "instance_timeout": 600
            }
        }
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Get production configuration."""
        config = self._get_staging_config()
        config.update({
            "logging": {
                "level": "WARNING"
            },
            "worker": {
                "max_concurrent_tasks": 5,
                "heartbeat_interval": 15,
                "health_check_interval": 30
            },
            "browser": {
                "max_instances": 20,
                "instance_timeout": 900
            }
        })
        return config


class DeploymentManager:
    """Main deployment manager."""
    
    def __init__(self, project_root: Path = Path(".")):
        """Initialize deployment manager."""
        self.project_root = project_root
        self.dockerfile_generator = DockerfileGenerator()
        self.compose_generator = DockerComposeGenerator()
        self.config_manager = ConfigurationManager(project_root / "config")
    
    def setup_deployment_structure(self):
        """Set up deployment directory structure."""
        directories = [
            "docker",
            "config",
            "scripts",
            "monitoring/prometheus",
            "monitoring/grafana/dashboards",
            "monitoring/grafana/datasources",
            "nginx",
            "vault/config",
            "logs"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Deployment directory structure created")
    
    def generate_dockerfiles(self):
        """Generate all Dockerfiles."""
        docker_dir = self.project_root / "docker"
        
        # Generate Dockerfiles
        dockerfiles = {
            "Dockerfile.c2": self.dockerfile_generator.generate_c2_dockerfile(),
            "Dockerfile.worker": self.dockerfile_generator.generate_worker_dockerfile(),
            "Dockerfile.monitoring": self.dockerfile_generator.generate_monitoring_dockerfile()
        }
        
        for filename, content in dockerfiles.items():
            dockerfile_path = docker_dir / filename
            with open(dockerfile_path, 'w') as f:
                f.write(content.strip())
            logger.info(f"Generated {filename}")
    
    def generate_compose_files(self):
        """Generate docker-compose files."""
        # Development compose
        dev_compose = self.compose_generator.generate_development_compose()
        with open(self.project_root / "docker-compose.dev.yml", 'w') as f:
            yaml.dump(dev_compose, f, default_flow_style=False, indent=2)
        
        # Production compose
        prod_compose = self.compose_generator.generate_production_compose()
        with open(self.project_root / "docker-compose.prod.yml", 'w') as f:
            yaml.dump(prod_compose, f, default_flow_style=False, indent=2)
        
        logger.info("Generated docker-compose files")
    
    def generate_deployment_scripts(self):
        """Generate deployment scripts."""
        scripts_dir = self.project_root / "scripts"
        
        # Development deployment script
        dev_script = """#!/bin/bash
set -e

echo "Starting BRAF development deployment..."

# Build images
docker-compose -f docker-compose.dev.yml build

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Wait for database
echo "Waiting for database..."
sleep 10

# Run migrations
docker-compose -f docker-compose.dev.yml exec c2_server alembic upgrade head

echo "Development deployment complete!"
echo "C2 Dashboard: http://localhost:8000"
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
"""
        
        # Production deployment script
        prod_script = """#!/bin/bash
set -e

echo "Starting BRAF production deployment..."

# Check required environment variables
required_vars=("POSTGRES_DB" "POSTGRES_USER" "POSTGRES_PASSWORD" "VAULT_TOKEN" "SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var environment variable is not set"
        exit 1
    fi
done

# Deploy stack
docker stack deploy -c docker-compose.prod.yml braf

echo "Production deployment initiated!"
echo "Check deployment status: docker service ls"
"""
        
        # Health check script
        health_script = """#!/bin/bash

echo "Checking BRAF system health..."

# Check C2 server
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] C2 Server: Healthy"
else
    echo "[FAIL] C2 Server: Unhealthy"
fi

# Check database
if docker-compose exec postgres pg_isready > /dev/null 2>&1; then
    echo "[OK] Database: Healthy"
else
    echo "[FAIL] Database: Unhealthy"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "[OK] Redis: Healthy"
else
    echo "[FAIL] Redis: Unhealthy"
fi

echo "Health check complete!"
"""
        
        scripts = {
            "deploy-dev.sh": dev_script,
            "deploy-prod.sh": prod_script,
            "health-check.sh": health_script
        }
        
        for filename, content in scripts.items():
            script_path = scripts_dir / filename
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            script_path.chmod(0o755)  # Make executable
            logger.info(f"Generated {filename}")
    
    def generate_monitoring_configs(self):
        """Generate monitoring configuration files."""
        monitoring_dir = self.project_root / "monitoring"
        
        # Prometheus configuration
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "braf-c2",
                    "static_configs": [
                        {"targets": ["c2_server:8000"]}
                    ]
                },
                {
                    "job_name": "braf-workers",
                    "static_configs": [
                        {"targets": ["worker_node:8001"]}
                    ]
                }
            ]
        }
        
        with open(monitoring_dir / "prometheus.yml", 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        # Grafana datasource configuration
        grafana_datasource = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True
                }
            ]
        }
        
        datasources_dir = monitoring_dir / "grafana" / "datasources"
        with open(datasources_dir / "prometheus.yml", 'w') as f:
            yaml.dump(grafana_datasource, f, default_flow_style=False)
        
        logger.info("Generated monitoring configurations")
    
    def full_deployment_setup(self, environment: Environment = Environment.DEVELOPMENT):
        """Perform complete deployment setup."""
        logger.info(f"Setting up deployment for {environment.value} environment")
        
        # Create directory structure
        self.setup_deployment_structure()
        
        # Generate configurations
        self.config_manager.generate_default_configs()
        
        # Generate Docker files
        self.generate_dockerfiles()
        self.generate_compose_files()
        
        # Generate scripts
        self.generate_deployment_scripts()
        
        # Generate monitoring configs
        self.generate_monitoring_configs()
        
        logger.info("Deployment setup complete!")
        
        # Print next steps
        print("\nNext steps:")
        print("1. Review and customize configuration files in ./config/")
        print("2. Set up environment variables for production")
        print("3. Run deployment script:")
        if environment == Environment.DEVELOPMENT:
            print("   ./scripts/deploy-dev.sh")
        else:
            print("   ./scripts/deploy-prod.sh")
        print("4. Check system health: ./scripts/health-check.sh")


def main():
    """Main entry point for deployment management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BRAF Deployment Manager')
    parser.add_argument('--environment', choices=[e.value for e in Environment],
                       default=Environment.DEVELOPMENT.value,
                       help='Target environment')
    parser.add_argument('--setup', action='store_true',
                       help='Perform full deployment setup')
    parser.add_argument('--generate-configs', action='store_true',
                       help='Generate configuration files only')
    parser.add_argument('--generate-docker', action='store_true',
                       help='Generate Docker files only')
    
    args = parser.parse_args()
    
    deployment_manager = DeploymentManager()
    environment = Environment(args.environment)
    
    if args.setup:
        deployment_manager.full_deployment_setup(environment)
    elif args.generate_configs:
        deployment_manager.config_manager.generate_default_configs()
    elif args.generate_docker:
        deployment_manager.generate_dockerfiles()
        deployment_manager.generate_compose_files()
    else:
        print("Use --help for available options")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()