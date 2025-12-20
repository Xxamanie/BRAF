# 🚀 BRAF Usage Guide - Get Started in 5 Minutes

## 📋 Prerequisites

1. **Python 3.10+** installed
2. **Docker & Docker Compose** installed
3. **Git** for cloning (if needed)

## 🏃‍♂️ Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
# Install Python dependencies
pip install -e .

# Install Playwright browsers
playwright install chromium
```

### Step 2: Set Up Development Environment
```bash
# Generate deployment files
python -m braf.deployment.deployment_manager --setup --environment development

# Start the system
./scripts/deploy-dev.sh
```

### Step 3: Access the System
- **Main Dashboard**: http://localhost:8000
- **Monitoring**: http://localhost:3000 (admin/admin)
- **API Docs**: http://localhost:8000/docs

## 💻 Basic Usage Examples

### Example 1: Simple Web Scraping
```python
import asyncio
from braf.core.models import AutomationTask, AutomationAction, ActionType, TaskPriority
from braf.core.task_executor import init_task_executor

async def scrape_example():
    # Initialize the task executor
    executor = init_task_executor()
    
    # Create a simple scraping task
    task = AutomationTask(
        id="scrape_example_com",
        profile_id="default_profile",
        actions=[
            # Navigate to website
            AutomationAction(
                type=ActionType.NAVIGATE,
                url="https://example.com",
                timeout=30
            ),
            # Wait for page to load
            AutomationAction(
                type=ActionType.WAIT,
                data="3.0",
                timeout=10
            ),
            # Extract the page title
            AutomationAction(
                type=ActionType.EXTRACT,
                selector="h1",
                timeout=10,
                metadata={"attribute": "text"}
            ),
            # Take a screenshot
            AutomationAction(
                type=ActionType.SCREENSHOT,
                data="example_screenshot.png",
                timeout=10
            )
        ],
        priority=TaskPriority.NORMAL,
        timeout=300
    )
    
    # Execute the task
    result = await executor.execute_task(task)
    
    if result.success:
        print(f"✅ Task completed successfully!")
        print(f"⏱️  Execution time: {result.execution_time:.2f} seconds")
        print(f"📊 Actions completed: {result.actions_completed}")
    else:
        print(f"❌ Task failed: {result.error}")

# Run the example
asyncio.run(scrape_example())
```

### Example 2: Form Automation
```python
import asyncio
from braf.core.models import AutomationTask, AutomationAction, ActionType

async def form_automation_example():
    from braf.core.task_executor import init_task_executor
    
    executor = init_task_executor()
    
    task = AutomationTask(
        id="form_fill_example",
        profile_id="form_profile",
        actions=[
            # Navigate to form page
            AutomationAction(
                type=ActionType.NAVIGATE,
                url="https://httpbin.org/forms/post",
                timeout=30
            ),
            # Fill in name field
            AutomationAction(
                type=ActionType.TYPE,
                selector="input[name='custname']",
                data="John Doe",
                timeout=10
            ),
            # Fill in email field
            AutomationAction(
                type=ActionType.TYPE,
                selector="input[name='custemail']",
                data="john@example.com",
                timeout=10
            ),
            # Select delivery option
            AutomationAction(
                type=ActionType.SELECT,
                selector="select[name='delivery']",
                data="fast",
                timeout=10
            ),
            # Submit form
            AutomationAction(
                type=ActionType.CLICK,
                selector="input[type='submit']",
                timeout=10,
                metadata={"submit_form": True}
            )
        ]
    )
    
    result = await executor.execute_task(task)
    print(f"Form automation result: {'Success' if result.success else 'Failed'}")

asyncio.run(form_automation_example())
```

### Example 3: E-commerce Price Monitoring
```python
import asyncio
from braf.core.models import AutomationTask, AutomationAction, ActionType

async def price_monitoring_example():
    from braf.core.task_executor import init_task_executor
    
    executor = init_task_executor()
    
    # Monitor product prices
    products = [
        "https://example-store.com/product1",
        "https://example-store.com/product2"
    ]
    
    for i, product_url in enumerate(products):
        task = AutomationTask(
            id=f"price_monitor_{i}",
            profile_id="price_monitor_profile",
            actions=[
                AutomationAction(
                    type=ActionType.NAVIGATE,
                    url=product_url,
                    timeout=30
                ),
                AutomationAction(
                    type=ActionType.EXTRACT,
                    selector=".price, .cost, [data-price]",
                    timeout=10,
                    metadata={"attribute": "text"}
                ),
                AutomationAction(
                    type=ActionType.EXTRACT,
                    selector="h1, .product-title",
                    timeout=10,
                    metadata={"attribute": "text"}
                )
            ]
        )
        
        result = await executor.execute_task(task)
        if result.success:
            print(f"✅ Monitored product {i+1}")
        else:
            print(f"❌ Failed to monitor product {i+1}: {result.error}")

asyncio.run(price_monitoring_example())
```

## 🔧 Advanced Usage

### Using the REST API
```python
import requests
import json

# API endpoint
api_base = "http://localhost:8000"

# Submit a task via API
task_data = {
    "id": "api_task_001",
    "profile_id": "api_profile",
    "actions": [
        {
            "type": "navigate",
            "url": "https://httpbin.org/html",
            "timeout": 30
        },
        {
            "type": "extract",
            "selector": "h1",
            "timeout": 10,
            "metadata": {"attribute": "text"}
        }
    ],
    "priority": "normal",
    "timeout": 300
}

# Submit task
response = requests.post(f"{api_base}/tasks", json=task_data)
if response.status_code == 200:
    task_id = response.json()["task_id"]
    print(f"✅ Task submitted: {task_id}")
    
    # Check task status
    status_response = requests.get(f"{api_base}/tasks/{task_id}")
    print(f"📊 Task status: {status_response.json()}")
```

### Using Worker Nodes
```bash
# Start a worker node
python -m braf.worker.main --worker-id worker_001 --max-tasks 3

# Start multiple workers
for i in {1..3}; do
    python -m braf.worker.main --worker-id worker_00$i --max-tasks 2 &
done
```

### Custom Profiles and Proxies
```python
from braf.core.profile_manager import init_profile_manager
from braf.core.proxy_rotator import init_proxy_rotator

# Initialize services
profile_manager = init_profile_manager()
proxy_rotator = init_proxy_rotator()

# Create custom profile
profile = await profile_manager.create_profile(
    fingerprint_config={
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "screen_resolution": (1920, 1080),
        "timezone": "America/New_York"
    }
)

# Add proxy configuration
proxy_config = {
    "host": "proxy.example.com",
    "port": 8080,
    "username": "user",
    "password": "pass"
}
await proxy_rotator.add_proxy(proxy_config)
```

## 🎛️ Configuration

### Environment Variables
```bash
# Database
export DATABASE_URL="postgresql://user:pass@localhost:5432/braf"
export REDIS_URL="redis://localhost:6379/0"

# Security
export SECRET_KEY="your-secret-key-here"
export VAULT_URL="http://localhost:8200"

# CAPTCHA Services
export CAPTCHA_API_KEY="your-2captcha-api-key"

# Monitoring
export PROMETHEUS_PORT="8000"
export GRAFANA_PORT="3000"
```

### Configuration Files
Edit `config/development.yaml`:
```yaml
worker:
  max_concurrent_tasks: 3
  heartbeat_interval: 30

browser:
  headless: true
  max_instances: 5

captcha:
  test_mode: false
  primary_service: "2captcha"
  api_key: "${CAPTCHA_API_KEY}"

compliance:
  max_requests_per_hour: 100
  max_form_submissions_per_day: 50
```

## 📊 Monitoring & Debugging

### Check System Health
```bash
# Run health check
./scripts/health-check.sh

# Check logs
docker-compose logs -f c2_server
docker-compose logs -f worker_node
```

### View Metrics
```python
from braf.core.monitoring import get_monitoring_manager

# Get system health
monitoring = get_monitoring_manager()
health = monitoring.get_system_health()
print(f"System status: {health['status']}")
print(f"Active alerts: {health['active_alerts']}")
```

### Compliance Monitoring
```python
from braf.core.compliance_logger import get_compliance_logger

compliance = get_compliance_logger()
metrics = compliance.get_compliance_metrics()
print(f"Total tasks: {metrics.total_tasks}")
print(f"Violations: {metrics.violations}")
print(f"Status: {metrics.status}")
```

## 🚨 Troubleshooting

### Common Issues

**1. Browser Not Starting**
```bash
# Install browsers
playwright install chromium

# Check browser path
playwright install --help
```

**2. Database Connection Issues**
```bash
# Check database status
docker-compose exec postgres pg_isready

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**3. Worker Not Connecting**
```bash
# Check C2 server status
curl http://localhost:8000/health

# Check worker logs
docker-compose logs worker_node
```

**4. High Memory Usage**
```python
# Reduce browser instances
browser_manager = get_browser_instance_manager()
await browser_manager.cleanup_expired_instances()
```

## 🔒 Security Best Practices

### 1. Use Environment Variables
```bash
# Never hardcode secrets
export POSTGRES_PASSWORD="secure-password"
export VAULT_TOKEN="vault-token"
```

### 2. Enable TLS in Production
```yaml
# docker-compose.prod.yml
services:
  nginx:
    volumes:
      - ./ssl:/etc/nginx/ssl
```

### 3. Configure Rate Limits
```python
# Adjust compliance constraints
compliance_logger.constraint_checker.add_constraint(
    EthicalConstraint(
        name="custom_rate_limit",
        max_per_hour=50,  # Reduce for sensitive sites
        severity=SeverityLevel.HIGH
    )
)
```

## 📈 Scaling for Production

### Horizontal Scaling
```bash
# Scale workers
docker-compose up --scale worker_node=5

# Use Docker Swarm
docker stack deploy -c docker-compose.prod.yml braf
```

### Performance Tuning
```yaml
# Optimize for high throughput
worker:
  max_concurrent_tasks: 10
  
browser:
  max_instances: 20
  instance_timeout: 300
```

## 🎯 Use Case Examples

### Quality Assurance Testing
```python
# Test website functionality
qa_task = AutomationTask(
    id="qa_login_test",
    actions=[
        AutomationAction(type=ActionType.NAVIGATE, url="https://app.example.com/login"),
        AutomationAction(type=ActionType.TYPE, selector="#username", data="testuser"),
        AutomationAction(type=ActionType.TYPE, selector="#password", data="testpass"),
        AutomationAction(type=ActionType.CLICK, selector="button[type='submit']"),
        AutomationAction(type=ActionType.EXTRACT, selector=".welcome-message")
    ]
)
```

### Market Research
```python
# Competitor price analysis
research_task = AutomationTask(
    id="competitor_analysis",
    actions=[
        AutomationAction(type=ActionType.NAVIGATE, url="https://competitor.com/products"),
        AutomationAction(type=ActionType.EXTRACT, selector=".product-price", metadata={"extract_all": True}),
        AutomationAction(type=ActionType.EXTRACT, selector=".product-name", metadata={"extract_all": True})
    ]
)
```

---

## 🎉 You're Ready to Go!

**Start with the Quick Start section above, then explore the examples that match your use case.**

Need help? Check the logs, monitoring dashboards, or review the troubleshooting section!