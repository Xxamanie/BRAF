# BRAF API Documentation

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
