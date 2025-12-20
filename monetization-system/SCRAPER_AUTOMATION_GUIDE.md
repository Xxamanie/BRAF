# Web Scraper Automation Guide

## Overview
Complete automation system for running web scraping jobs with robust error handling, logging, SQLite storage, and monitoring capabilities.

## Files Created

### Core Scripts
1. **`run_scrape.py`** - Main scraper runner with comprehensive error handling
2. **`check_scraper_status.py`** - Status monitoring and health checks
3. **`scraper_notifications.py`** - Notification system (email, Slack, webhooks)
4. **`run_scraper.bat`** - Windows batch script for easy execution

### Configuration
5. **`scraper_urls.json`** - URL configuration and scraping settings

## Features

### ✅ Robust Error Handling
- Automatic retries with exponential backoff
- Graceful failure handling
- Detailed error logging with stack traces
- Separate error log file for critical issues

### ✅ Comprehensive Logging
- Dual logging (file + console)
- Timestamped entries
- Log rotation for large files
- Separate error log tracking

### ✅ Idempotent Design
- Safe to run multiple times
- Duplicate detection via data hashing
- Status tracking prevents conflicts
- Database UNIQUE constraints

### ✅ SQLite Integration
- Automatic database initialization
- Efficient data storage
- Full-text search capabilities
- Statistics and analytics

### ✅ Monitoring & Status
- Real-time status tracking
- Health check endpoints
- Database statistics
- Performance metrics

### ✅ Graceful Shutdown
- Signal handling (SIGINT, SIGTERM)
- Resource cleanup
- Status preservation
- Safe interruption

## Usage

### Basic Usage

```bash
# Run scraping session
python run_scrape.py

# Check status
python check_scraper_status.py

# Create sample configuration
python run_scrape.py --create-config

# Get help
python run_scrape.py --help
```

### Windows Batch Script

```cmd
# Run scraper
run_scraper.bat

# Check status
run_scraper.bat --status

# Create config
run_scraper.bat --config

# Help
run_scraper.bat --help
```

### Status Monitoring

```bash
# Detailed status
python check_scraper_status.py

# Brief summary
python check_scraper_status.py --summary

# Health check (exit code 0=healthy, 1=unhealthy)
python check_scraper_status.py --health

# Database statistics
python check_scraper_status.py --database
```

## Configuration

### scraper_urls.json

```json
{
  "urls": [
    "https://example.com/page1",
    "https://example.com/page2"
  ],
  "config": {
    "max_pages_per_run": 10,
    "delay_between_pages": 2,
    "timeout_per_page": 30,
    "max_retries": 3,
    "headless": true
  },
  "schedule": {
    "enabled": true,
    "interval_minutes": 60,
    "max_daily_runs": 24
  },
  "notifications": {
    "email_on_failure": false,
    "webhook_url": null,
    "slack_webhook": null
  }
}
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `max_pages_per_run` | Maximum pages to scrape per session | 10 |
| `delay_between_pages` | Delay between requests (seconds) | 2 |
| `timeout_per_page` | Timeout for each page (seconds) | 30 |
| `max_retries` | Maximum retry attempts | 3 |
| `headless` | Run browser in headless mode | true |

## Logging

### Log Files

- **`logs/scraper.log`** - Main log file with all events
- **`logs/scraper_errors.log`** - Error-only log with stack traces

### Log Format

```
2025-12-20 16:05:44,460 - __main__ - INFO - 🚀 Starting scheduled scrape job
2025-12-20 16:05:44,486 - __main__ - INFO - ✅ SQLite scraper initialized
2025-12-20 16:05:44,488 - __main__ - INFO - 📋 Loaded 5 URLs to scrape
```

### Log Rotation

- Logs automatically archived when exceeding 10MB
- Archived with timestamp: `scraper.log.20251220_160544`
- Old archives can be manually deleted

## Status Tracking

### Status File

Location: `data/scraper_status.json`

```json
{
  "status": "completed",
  "timestamp": "2025-12-20T16:05:52.692287",
  "stats": {
    "pages_scraped": 5,
    "pages_failed": 0,
    "total_content_length": 411,
    "duration_seconds": 8.229135,
    "success_rate": 100.0
  },
  "details": {
    "start_time": "2025-12-20T16:05:44.460038",
    "end_time": "2025-12-20T16:05:52.689173",
    "urls_processed": 5
  }
}
```

### Status Values

- `running` - Scraping in progress
- `completed` - Successfully completed
- `failed` - Failed with errors
- `stopped` - Manually stopped

## Error Handling

### Retry Logic

1. **Exponential Backoff**: 1s, 2s, 4s delays between retries
2. **Max Retries**: Configurable (default: 3)
3. **Timeout Handling**: Separate timeout errors
4. **Error Logging**: All errors logged with context

### Error Types

- **Timeout Errors**: Page load timeout
- **Network Errors**: Connection failures
- **Parse Errors**: Content extraction failures
- **Database Errors**: Storage failures

### Error Recovery

```python
# Automatic retry with backoff
for attempt in range(max_retries):
    try:
        result = await scrape_page(url)
        return result
    except Exception as e:
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)
        else:
            log_error(e)
```

## Notifications

### Email Notifications

Configure in `scraper_urls.json`:

```json
{
  "notifications": {
    "email": {
      "enabled": true,
      "smtp_server": "smtp.gmail.com",
      "smtp_port": 587,
      "sender_email": "your-email@gmail.com",
      "sender_password": "your-app-password",
      "recipients": ["admin@example.com"]
    }
  }
}
```

### Slack Notifications

```json
{
  "notifications": {
    "slack_webhook": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }
}
```

### Webhook Notifications

```json
{
  "notifications": {
    "webhook_url": "https://your-api.com/webhook"
  }
}
```

## Scheduling

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., daily at 2 AM)
4. Action: Start a program
5. Program: `python`
6. Arguments: `C:\path\to\run_scrape.py`
7. Start in: `C:\path\to\monetization-system`

### Linux Cron

```bash
# Edit crontab
crontab -e

# Run every hour
0 * * * * cd /path/to/monetization-system && python run_scrape.py

# Run daily at 2 AM
0 2 * * * cd /path/to/monetization-system && python run_scrape.py

# Run every 6 hours
0 */6 * * * cd /path/to/monetization-system && python run_scrape.py
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

# Run scraper every hour
CMD ["sh", "-c", "while true; do python run_scrape.py; sleep 3600; done"]
```

## Database Integration

### SQLite Storage

- **Location**: `data/scraper.db`
- **Schema**: Optimized for scraping data
- **Indexes**: Fast queries on domain, date, hash
- **Size**: Efficient storage with compression

### Database Operations

```python
from sqlite_scraper import SQLiteWebScraper

# Initialize
scraper = SQLiteWebScraper()

# Get recent data
data = scraper.get_data(limit=10)

# Search content
results = scraper.search_content("keyword")

# Get statistics
stats = scraper.get_stats()
```

## Performance Metrics

### Typical Performance

- **Scraping Speed**: ~1-2 pages/second
- **Database Write**: <10ms per record
- **Memory Usage**: ~50-100MB
- **CPU Usage**: Low (mostly I/O bound)

### Optimization Tips

1. **Increase Concurrency**: Scrape multiple URLs in parallel
2. **Reduce Delays**: Lower `delay_between_pages` for faster sites
3. **Batch Writes**: Group database writes
4. **Cache Results**: Avoid re-scraping recent pages

## Troubleshooting

### Common Issues

#### 1. Scraper Not Running

```bash
# Check status
python check_scraper_status.py --health

# Check logs
tail -f logs/scraper.log
```

#### 2. Database Locked

```bash
# Check for other processes
ps aux | grep run_scrape

# Kill if necessary
kill <PID>
```

#### 3. High Failure Rate

```bash
# Check error log
cat logs/scraper_errors.log

# Increase timeout
# Edit scraper_urls.json: "timeout_per_page": 60
```

#### 4. Memory Issues

```bash
# Reduce pages per run
# Edit scraper_urls.json: "max_pages_per_run": 5

# Enable headless mode
# Edit scraper_urls.json: "headless": true
```

## Best Practices

### 1. Respectful Scraping

- Use appropriate delays between requests
- Respect robots.txt
- Implement rate limiting
- Use proper user agents

### 2. Error Handling

- Always log errors with context
- Implement retry logic
- Handle timeouts gracefully
- Monitor failure rates

### 3. Data Management

- Regular database cleanup
- Archive old logs
- Backup database regularly
- Monitor disk space

### 4. Monitoring

- Check status regularly
- Set up health checks
- Configure notifications
- Review logs periodically

## Security Considerations

### 1. Credentials

- Never commit credentials to git
- Use environment variables
- Encrypt sensitive data
- Rotate passwords regularly

### 2. Database

- Restrict file permissions
- Regular backups
- Encrypt sensitive content
- Sanitize inputs

### 3. Logging

- Don't log sensitive data
- Secure log files
- Regular log rotation
- Monitor access

## Maintenance

### Daily Tasks

- Check scraper status
- Review error logs
- Monitor disk space

### Weekly Tasks

- Review success rates
- Clean old data
- Backup database
- Update URL list

### Monthly Tasks

- Analyze trends
- Optimize configuration
- Update dependencies
- Security audit

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - at least one page scraped |
| 1 | Failure - no pages scraped or error |

## Support

For issues or questions:
1. Check logs: `logs/scraper.log`
2. Check status: `python check_scraper_status.py`
3. Review configuration: `scraper_urls.json`
4. Test manually: `python run_scrape.py`

## Summary

The scraper automation system provides:
- ✅ Robust error handling and retries
- ✅ Comprehensive logging
- ✅ Idempotent operation
- ✅ SQLite storage
- ✅ Status monitoring
- ✅ Easy scheduling
- ✅ Notification support
- ✅ Windows & Linux compatible

Perfect for automated, production-ready web scraping workflows!