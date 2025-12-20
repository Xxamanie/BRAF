# BRAF System Maintenance Scripts - Complete Implementation

## ✅ Updated Maintenance Scripts Created

I've created comprehensive maintenance scripts that use the correct service names (`c2_server`, `worker_node`) instead of the outdated references (`scraper`, `celery_worker`).

## Created Scripts

### 1. Daily Maintenance Script ✅
**Files**: 
- `scripts/daily_maintenance.sh` (Linux/macOS)
- `scripts/daily_maintenance.bat` (Windows)

**Features**:
- ✅ Database backup with compression
- ✅ Old backup cleanup (30 days retention)
- ✅ Database optimization (VACUUM ANALYZE)
- ✅ Selective Redis cache cleanup
- ✅ Log rotation (7 days retention)
- ✅ Disk space monitoring
- ✅ Playwright browser updates
- ✅ Comprehensive health checks for all services
- ✅ System metrics collection
- ✅ Database statistics gathering
- ✅ Temporary file cleanup
- ✅ JSON report generation
- ✅ Email notifications (if configured)

**Service References Updated**:
- ✅ `c2_server` (instead of `scraper`)
- ✅ `worker_node` (instead of `celery_worker`)
- ✅ Correct Docker Compose file (`docker-compose.prod.yml`)

### 2. Weekly Backup Script ✅
**File**: `scripts/weekly_backup.sh`

**Features**:
- ✅ Full database backup with compression
- ✅ Application data backup (data, certificates, uploads)
- ✅ Configuration files backup
- ✅ Docker images export and backup
- ✅ Recent logs backup (30 days)
- ✅ Backup manifest generation
- ✅ Backup integrity verification
- ✅ Old backup cleanup (12 weeks retention)
- ✅ HTML backup report generation
- ✅ Email notifications

### 3. Status Report Generator ✅
**File**: `scripts/send_report.py`

**Features**:
- ✅ System status report generation
- ✅ HTML email formatting
- ✅ SMTP email sending
- ✅ JSON report loading
- ✅ Component health monitoring
- ✅ Performance metrics collection
- ✅ Maintenance status tracking

### 4. Cron Job Setup Script ✅
**File**: `scripts/setup_cron.sh`

**Features**:
- ✅ Automated cron job installation
- ✅ Daily maintenance scheduling (2:00 AM)
- ✅ Weekly backup scheduling (3:00 AM Sundays)
- ✅ Hourly health checks
- ✅ Daily log rotation (1:00 AM)
- ✅ Weekly browser updates (4:00 AM Saturdays)

## Service Name Corrections Applied

### ✅ Before → After
- `scraper` → `c2_server`
- `celery_worker` → `worker_node`
- `docker-compose.prod.yml` (correct file reference)

### ✅ Updated Commands
```bash
# Database operations (using c2_server)
docker-compose -f docker-compose.prod.yml exec -T c2_server python -m database.init_db

# Worker health checks (using worker_node)
docker-compose -f docker-compose.prod.yml exec -T worker_node python -c "from src.braf.worker.main import health_check; exit(0 if health_check() else 1)"

# Playwright updates (using worker_node)
docker-compose -f docker-compose.prod.yml exec -T worker_node playwright install --with-deps chromium
```

## Daily Maintenance Tasks

### 🔄 Automated Daily Tasks (2:00 AM)
1. **Database Backup**: Full PostgreSQL dump with timestamp
2. **Backup Cleanup**: Remove backups older than 30 days
3. **Database Optimization**: VACUUM ANALYZE for performance
4. **Cache Cleanup**: Clean temporary Redis keys
5. **Log Rotation**: Remove logs older than 7 days
6. **Disk Space Check**: Monitor storage usage with alerts
7. **Browser Updates**: Update Playwright Chromium
8. **Health Checks**: Verify all service endpoints
9. **Metrics Collection**: Gather system performance data
10. **Database Stats**: Collect table statistics
11. **Temp Cleanup**: Remove temporary files
12. **Report Generation**: Create JSON status report
13. **Email Notification**: Send status report (if configured)

### 📊 Health Checks Performed
- ✅ C2 Server: `http://localhost:8000/health`
- ✅ Worker Node: Python health check function
- ✅ PostgreSQL: Connection and readiness check
- ✅ Redis: Ping connectivity test
- ✅ Docker Containers: Status verification

## Weekly Backup Tasks

### 🗄️ Comprehensive Weekly Backup (3:00 AM Sundays)
1. **Full Database Backup**: Compressed PostgreSQL dump
2. **Application Data**: Backup data, certificates, uploads
3. **Configuration Files**: Docker Compose, environment, configs
4. **Docker Images**: Export and compress container images
5. **Recent Logs**: Backup last 30 days of logs
6. **Backup Manifest**: JSON metadata file
7. **Integrity Verification**: Test backup file integrity
8. **Cleanup**: Remove backups older than 12 weeks
9. **HTML Report**: Generate detailed backup report
10. **Email Notification**: Send backup status report

## Automated Scheduling

### 📅 Cron Job Schedule
```bash
# Daily maintenance at 2:00 AM
0 2 * * * cd /app && bash scripts/daily_maintenance.sh

# Weekly backup at 3:00 AM on Sundays  
0 3 * * 0 cd /app && bash scripts/weekly_backup.sh

# Hourly health check
0 * * * * curl -f -s http://localhost:8000/health > /dev/null || echo "Health check failed"

# Daily log rotation at 1:00 AM
0 1 * * * find /app/logs -name "*.log" -mtime +7 -delete

# Weekly browser update on Saturdays at 4:00 AM
0 4 * * 6 cd /app && docker-compose -f docker-compose.prod.yml exec -T worker_node playwright install
```

## Installation and Setup

### 1. Linux/macOS Setup
```bash
cd monetization-system

# Make scripts executable
chmod +x scripts/*.sh scripts/*.py

# Install cron jobs
bash scripts/setup_cron.sh

# Test daily maintenance
bash scripts/daily_maintenance.sh

# Test weekly backup
bash scripts/weekly_backup.sh
```

### 2. Windows Setup
```batch
cd monetization-system

REM Run daily maintenance
scripts\daily_maintenance.bat

REM Setup Windows Task Scheduler (manual)
REM Create scheduled tasks for daily and weekly maintenance
```

### 3. Email Configuration
Add to your environment file:
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

## Log Files Generated

### 📁 Daily Logs
- `logs/maintenance_YYYYMMDD.log` - Daily maintenance log
- `logs/cron.log` - Cron job output
- `logs/health_check.log` - Health check failures
- `logs/browser_update.log` - Browser update logs

### 📁 Weekly Logs
- `logs/weekly_backup_YYYYMMDD.log` - Weekly backup log
- `logs/weekly_backup_report_YYYYMMDD.html` - HTML backup report

### 📁 Reports
- `logs/maintenance_report_YYYYMMDD.json` - Daily status report
- `backups/weekly/backup_manifest_YYYYMMDD.json` - Backup metadata

## Monitoring and Alerts

### 🚨 Alert Conditions
- Disk usage > 80% (warning)
- Disk usage > 90% (critical)
- Service health check failures
- Backup integrity failures
- Database connection issues

### 📧 Email Notifications
- Daily maintenance summary
- Weekly backup report
- Critical system alerts
- Health check failures

## File Structure

```
monetization-system/
├── scripts/
│   ├── daily_maintenance.sh      ✅ Linux daily maintenance
│   ├── daily_maintenance.bat     ✅ Windows daily maintenance
│   ├── weekly_backup.sh          ✅ Weekly comprehensive backup
│   ├── send_report.py            ✅ Status report generator
│   └── setup_cron.sh             ✅ Cron job installer
├── logs/                         📁 Log files directory
├── backups/                      📁 Backup files directory
│   ├── daily/                    📁 Daily backups
│   └── weekly/                   📁 Weekly backups
└── volumes/                      📁 Docker volumes
```

## Benefits of Updated Scripts

### ✅ Correct Service Integration
- Uses proper service names (`c2_server`, `worker_node`)
- Compatible with current Docker Compose configuration
- Follows BRAF system architecture

### ✅ Comprehensive Maintenance
- Database optimization and backup
- System health monitoring
- Automated cleanup and rotation
- Performance metrics collection

### ✅ Production Ready
- Error handling and logging
- Email notifications
- Integrity verification
- Automated scheduling

### ✅ Cross-Platform Support
- Linux/macOS shell scripts
- Windows batch files
- Docker container compatibility

## Status: 🎉 COMPLETE

All maintenance scripts have been updated with correct service names and comprehensive functionality:

- ✅ Daily maintenance with 13 automated tasks
- ✅ Weekly backup with integrity verification
- ✅ Status reporting with email notifications
- ✅ Automated cron job scheduling
- ✅ Cross-platform compatibility
- ✅ Production-ready error handling
- ✅ Comprehensive logging and monitoring

The maintenance system is now fully integrated with the BRAF production deployment and ready for automated operation.