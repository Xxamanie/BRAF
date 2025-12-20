#!/bin/bash
# Setup cron jobs for BRAF system maintenance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up BRAF system cron jobs..."
echo "Project directory: $PROJECT_DIR"

# Create cron job entries
CRON_JOBS="
# BRAF System Maintenance Jobs
# Daily maintenance at 2:00 AM
0 2 * * * cd $PROJECT_DIR && bash scripts/daily_maintenance.sh >> logs/cron.log 2>&1

# Weekly full backup at 3:00 AM on Sundays
0 3 * * 0 cd $PROJECT_DIR && bash scripts/weekly_backup.sh >> logs/cron.log 2>&1

# Hourly health check
0 * * * * cd $PROJECT_DIR && curl -f -s http://localhost:8000/health > /dev/null || echo \"[$(date)] Health check failed\" >> logs/health_check.log

# Daily log rotation at 1:00 AM
0 1 * * * cd $PROJECT_DIR && find logs -name \"*.log\" -mtime +7 ! -name \"maintenance_*\" -delete

# Weekly Playwright browser update on Saturdays at 4:00 AM
0 4 * * 6 cd $PROJECT_DIR && docker-compose -f docker-compose.prod.yml exec -T worker_node playwright install --with-deps chromium >> logs/browser_update.log 2>&1
"

# Backup existing crontab
echo "Backing up existing crontab..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || echo "No existing crontab found"

# Add new cron jobs
echo "Adding BRAF maintenance cron jobs..."
(crontab -l 2>/dev/null; echo "$CRON_JOBS") | crontab -

echo "✅ Cron jobs installed successfully!"
echo ""
echo "Installed cron jobs:"
echo "==================="
crontab -l | grep -A 10 "BRAF System Maintenance"

echo ""
echo "📋 Cron job schedule:"
echo "- Daily maintenance: 2:00 AM"
echo "- Weekly backup: 3:00 AM (Sundays)"
echo "- Hourly health check: Every hour"
echo "- Daily log rotation: 1:00 AM"
echo "- Weekly browser update: 4:00 AM (Saturdays)"
echo ""
echo "📁 Log files:"
echo "- Maintenance: logs/maintenance_YYYYMMDD.log"
echo "- Cron output: logs/cron.log"
echo "- Health checks: logs/health_check.log"
echo "- Browser updates: logs/browser_update.log"