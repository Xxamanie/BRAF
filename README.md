# Browser Automation Framework (BRAF)

A distributed system for advanced web automation with industry-grade detection evasion, behavioral simulation, and adversarial capabilities designed to challenge security frameworks.

## Features

- **Distributed Architecture**: Command & Control (C2) system managing multiple worker nodes
- **Human-like Behavior**: Realistic mouse movements, typing patterns, and timing delays
- **Detection Evasion**: Advanced fingerprint management and anti-bot countermeasures
- **Adversarial Optimization**: Configurable constraints for challenging security systems
- **Comprehensive Logging**: Mandatory activity tracking for security and compliance
- **Multi-tier CAPTCHA Solving**: Paid services with OCR fallbacks
- **Secure Communications**: TLS encryption and HashiCorp Vault integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COMMAND & CONTROL (C2)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │ Dashboard   │  │ Job Scheduler│  │ Analytics Engine  │  │
│  │ & Reporting │  │ & Queue      │  │ & Pattern Analysis│  │
│  └─────────────┘  └─────────────┘  └────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ (gRPC/WebSocket)
┌───────────────────────────┼─────────────────────────────────┐
│         WORKER NODES (Distributed)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Profile Manager → Fingerprint Store → Proxy Rotator │  │
│  └──────────────────────────────────────────────────────┘  │
│                   │              │              │           │
│         ┌─────────▼──┐ ┌────────▼──┐ ┌────────▼──┐        │
│         │ Task       │ │ Behavioral│ │ CAPTCHA   │        │
│         │ Executor   │ │ Emulation │ │ Solver    │        │
│         │ Engine     │ │ Engine    │ │ Service   │        │
│         └────────────┘ └───────────┘ └───────────┘        │
│                   │              │              │           │
│         ┌─────────▼──────────────▼──────────────▼──┐       │
│         │        Browser Instance Manager          │       │
│         │    (Playwright/Puppeteer + Stealth)      │       │
│         └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- PostgreSQL 14+
- Redis 7+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd braf

# Install dependencies
pip install -e ".[dev]"

# Install Playwright browsers
playwright install

# Set up pre-commit hooks
pre-commit install
```

### Development Setup

```bash
# Start infrastructure services
docker-compose up -d postgres redis elk

# Run database migrations
alembic upgrade head

# Start C2 Dashboard
braf-c2 --config config/development.yaml

# Start Worker Node (in another terminal)
braf-worker --config config/development.yaml
```

## Configuration

BRAF uses YAML configuration files with environment-specific overrides:

```yaml
# config/development.yaml
database:
  url: "postgresql+asyncpg://braf:password@localhost:5432/braf_dev"
  
redis:
  url: "redis://localhost:6379/0"
  
c2:
  host: "0.0.0.0"
  port: 8000
  
worker:
  max_concurrent_tasks: 5
  fingerprint_pool_size: 5
  
compliance:
  max_requests_per_hour: 100
  ethical_constraints_enabled: false
```

## Usage

### Submitting Tasks via API

```python
import aiohttp
import asyncio

async def submit_task():
    task_config = {
        "target_url": "https://example.com",
        "actions": [
            {"type": "navigate", "url": "https://example.com"},
            {"type": "click", "selector": "#login-button"},
            {"type": "type", "selector": "#username", "text": "testuser"},
            {"type": "extract", "selector": ".result", "attribute": "text"}
        ],
        "constraints": {
            "max_duration": 300,
            "respect_robots_txt": true
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/api/v1/tasks",
            json=task_config
        ) as response:
            result = await response.json()
            print(f"Task ID: {result['task_id']}")

asyncio.run(submit_task())
```

### Dashboard Access

Open http://localhost:8000 to access the C2 Dashboard for:
- Real-time worker status monitoring
- Task queue management
- Analytics and success rate tracking
- Compliance violation alerts

## Security & Capabilities

BRAF is designed with advanced security features and configurable operational constraints:

- **Comprehensive Logging**: All activities logged for performance analysis and optimization
- **Configurable Rate Limiting**: Adjustable limits for operational efficiency
- **Adaptive Constraints**: Flexible safeguards for challenging various security environments
- **Encrypted Storage**: Credentials encrypted with PBKDF2 key derivation
- **Secure Communications**: TLS encryption for all inter-component traffic

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run property-based tests
pytest tests/property/ -m property

# Run integration tests
pytest tests/integration/

# Run all tests with coverage
pytest --cov=braf --cov-report=html
```

## Monitoring

BRAF exposes Prometheus metrics and integrates with Grafana for monitoring:

- Worker node performance and health
- Task success rates and execution times
- Detection evasion effectiveness
- Compliance violation tracking
- Resource utilization metrics

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run pre-commit hooks
5. Submit a pull request

## Usage Notes

BRAF is designed for advanced automation scenarios. Users are responsible for ensuring compliance with applicable laws and terms of service when deploying this framework.