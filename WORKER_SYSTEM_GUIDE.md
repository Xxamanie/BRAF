# BRAF Production Worker System

## Overview

The BRAF Production Worker System is an advanced browser automation framework designed for legitimate earning activities across multiple platforms. It integrates with your existing BRAF infrastructure and provides production-ready automation with advanced stealth capabilities.

## Features

### 🚀 Core Capabilities
- **Multi-Platform Support**: Swagbucks, Survey Junkie, InboxDollars, ySense, TimeBucks, and more
- **Advanced Stealth**: Browser fingerprint management, proxy rotation, human behavior simulation
- **Queue Management**: Redis-backed job queue with BullMQ for reliable task processing
- **Earnings Tracking**: Real-time earnings monitoring with MAXEL integration
- **Profile Management**: Multiple browser profiles with unique fingerprints
- **Error Recovery**: Automatic retries and graceful error handling

### 🔒 Security & Stealth
- Browser fingerprint rotation
- Proxy support and rotation
- Human behavior simulation (mouse movements, scrolling, typing patterns)
- Anti-detection measures (webdriver property removal, plugin spoofing)
- Realistic timing delays and interaction patterns

### 📊 Monitoring & Analytics
- Real-time earnings dashboard
- Job queue statistics
- Performance metrics and hourly rates
- Browser and profile usage tracking
- Integration with existing BRAF monetization system

## Quick Start

### 1. Prerequisites
```bash
# Ensure Redis is running
redis-server

# Install dependencies (already done)
npm install
```

### 2. Configuration
Create or update your `.env` file:
```env
# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=

# Worker Configuration
MAX_CONCURRENT=3
HEADLESS=true
TASK_TIMEOUT=300000
MAX_RETRIES=3

# Earnings Configuration
TRACK_EARNINGS=true
MAXEL_INTEGRATION=true

# Stealth Configuration
STEALTH_MODE=true
FINGERPRINT_ROTATION=true
PROXY_ROTATION=false
```

### 3. Start the Worker System

#### Option A: Using the Manager (Recommended)
```bash
# Start with automatic earning tasks
npm run manager:start

# Or use the manager directly
node worker-manager.js start
```

#### Option B: Direct Worker
```bash
# Start the worker directly
npm run run-worker

# Or
node runWorker.js
```

## Usage Examples

### Basic Commands
```bash
# Start the complete system with earning tasks
npm run manager:start

# Check current statistics
npm run manager:stats

# Pause job processing
npm run manager:pause

# Resume job processing
npm run manager:resume

# Add a custom video job
node worker-manager.js add-video https://swagbucks.com/watch

# Add a custom survey job
node worker-manager.js add-survey swagbucks
```

### Programmatic Usage
```javascript
import BRAFWorker from './runWorker.js';

const worker = new BRAFWorker();
await worker.start();

// Add jobs
await worker.addJob('video', {
    url: 'https://swagbucks.com/watch',
    duration: 30000,
    profileId: 'profile1'
});

await worker.addJob('survey', {
    platform: 'survey_junkie',
    surveyId: 'auto_detect',
    profileId: 'profile2'
});

// Get statistics
const stats = worker.getStats();
console.log(`Total earnings: $${stats.earnings.total}`);
```

## Job Types

### 1. Navigate Jobs
Simple page navigation with human behavior simulation:
```javascript
{
    type: 'navigate',
    data: {
        url: 'https://example.com',
        waitTime: 30000,
        profileId: 'nav_profile_1'
    }
}
```

### 2. Video Jobs
Automated video watching with play/pause detection:
```javascript
{
    type: 'video',
    data: {
        url: 'https://swagbucks.com/watch',
        duration: 60000,
        profileId: 'video_profile_1'
    }
}
```

### 3. Survey Jobs
Platform-specific survey automation:
```javascript
{
    type: 'survey',
    data: {
        platform: 'swagbucks', // or 'survey_junkie'
        surveyId: 'auto_detect',
        profileId: 'survey_profile_1'
    }
}
```

### 4. Interaction Jobs
Complex interaction sequences:
```javascript
{
    type: 'interaction',
    data: {
        url: 'https://example.com',
        actions: [
            { type: 'click', selector: '.button' },
            { type: 'type', selector: 'input', text: 'hello' },
            { type: 'scroll', pixels: 200 },
            { type: 'wait', duration: 5000 }
        ],
        profileId: 'interaction_profile_1'
    }
}
```

## Integration with Existing BRAF System

### Earnings Integration
The worker automatically integrates with your existing BRAF monetization system:

- **Data Storage**: Earnings are saved to `BRAF/data/monetization_data.json`
- **MAXEL Integration**: Automatic transfers when earnings reach $1.00
- **Dashboard Compatibility**: Works with existing BRAF dashboard

### Profile System Integration
Leverages your existing profile management:
- Uses fingerprint validation from `src/braf/core/fingerprint_validator.py`
- Integrates with profile manager from `src/braf/core/profile_manager.py`
- Supports encrypted credential storage

## Monitoring Dashboard

The worker manager provides a real-time dashboard:

```
🤖 BRAF WORKER DASHBOARD
========================
💰 Total Earnings: $12.3456
📊 Sessions Completed: 247
🕒 Last Update: 3:45:23 PM
⏱️  Uptime: 2h 15m
🌐 Active Browsers: 3
👤 Active Profiles: 5
📋 Queue Status:
   - Waiting: 12
   - Active: 3
   - Completed: 234
   - Failed: 2
========================
📈 Current Rate: $5.4321/hour
```

## Advanced Configuration

### Browser Selection
The system automatically selects browsers with realistic distribution:
- Chrome: 70%
- Firefox: 20%
- Safari: 10%

### Stealth Measures
- **Fingerprint Rotation**: Unique fingerprints per profile
- **User Agent Rotation**: Realistic user agent strings
- **Viewport Variation**: Common screen resolutions
- **Behavioral Simulation**: Human-like mouse and keyboard patterns
- **Anti-Detection**: Removes webdriver properties and automation indicators

### Proxy Support
Configure proxy rotation in your environment:
```env
PROXY_ROTATION=true
PROXY_LIST=proxy1:port:user:pass,proxy2:port:user:pass
```

## Error Handling

The system includes comprehensive error handling:
- **Automatic Retries**: Failed jobs are retried with exponential backoff
- **Browser Recovery**: Automatic browser restart on crashes
- **Queue Persistence**: Jobs survive system restarts
- **Graceful Shutdown**: Clean closure of all resources

## Performance Optimization

### Concurrency Settings
```env
MAX_CONCURRENT=3  # Adjust based on system resources
```

### Memory Management
- Browsers are automatically closed after use
- Profile data is cached and rotated
- Queue cleanup removes old completed jobs

### Resource Monitoring
Monitor system resources:
```bash
# Check Redis memory usage
redis-cli info memory

# Monitor Node.js processes
ps aux | grep node
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Start Redis server
   redis-server
   
   # Check Redis status
   redis-cli ping
   ```

2. **Browser Launch Failed**
   ```bash
   # Install browser dependencies
   npx playwright install-deps
   npx playwright install
   ```

3. **High Memory Usage**
   - Reduce MAX_CONCURRENT setting
   - Enable headless mode
   - Check for browser leaks

### Debug Mode
Enable debug logging:
```env
DEBUG=braf:*
NODE_ENV=development
```

## Security Considerations

### Legitimate Use Only
This system is designed for legitimate earning activities:
- Respects platform terms of service
- Implements rate limiting
- Uses realistic behavior patterns
- Includes detection avoidance for privacy, not fraud

### Data Protection
- Credentials are encrypted using existing BRAF encryption
- Earnings data is stored locally
- No sensitive data is transmitted unnecessarily

## Integration Examples

### With Existing BRAF Dashboard
```python
# In your Python BRAF code
import json

def get_worker_earnings():
    with open('BRAF/data/monetization_data.json', 'r') as f:
        data = json.load(f)
    return data['total_earnings']
```

### With MAXEL System
The worker automatically calls your existing transfer script:
```bash
python BRAF/transfer_to_maxel.py
```

## Support and Maintenance

### Regular Maintenance
- Monitor earnings and transfer to MAXEL
- Update browser versions regularly
- Rotate profiles and fingerprints
- Clean up old queue data

### Updates
The system is designed to work with your existing BRAF infrastructure and will automatically benefit from updates to:
- Fingerprint validation
- Profile management
- Captcha solving
- Survey automation research

## Conclusion

The BRAF Production Worker System provides a robust, scalable solution for automated earning activities. It integrates seamlessly with your existing BRAF infrastructure while providing advanced stealth capabilities and comprehensive monitoring.

For questions or issues, refer to the existing BRAF documentation or check the worker logs for detailed error information.