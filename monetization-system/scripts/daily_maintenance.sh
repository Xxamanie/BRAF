#!/bin/bash
# Daily maintenance tasks for BRAF system
# Updated to use correct service names (c2_server, worker_node)

LOG_FILE="/app/logs/maintenance_$(date +%Y%m%d).log"
BACKUP_DIR="/app/backups"
COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p /app/logs /app/backups

echo "=========================================" >> $LOG_FILE
echo "Starting BRAF daily maintenance at $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> $LOG_FILE
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" >> $LOG_FILE
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

# 1. Backup database
log_message "Backing up PostgreSQL database..."
if docker-compose -f $COMPOSE_FILE exec -T postgres \
    pg_dump -U ${POSTGRES_USER:-braf_user} ${POSTGRES_DB:-braf_worker} > $BACKUP_DIR/backup_$(date +%Y%m%d_%H%M%S).sql 2>> $LOG_FILE; then
    log_message "Database backup completed successfully"
else
    log_error "Database backup failed"
fi

# 2. Clean old backups (keep 30 days)
log_message "Cleaning old backups (keeping 30 days)..."
DELETED_BACKUPS=$(find $BACKUP_DIR -name "backup_*.sql" -mtime +30 -delete -print 2>> $LOG_FILE | wc -l)
log_message "Deleted $DELETED_BACKUPS old backup files"

# 3. Optimize database
log_message "Optimizing PostgreSQL database..."
if docker-compose -f $COMPOSE_FILE exec -T postgres \
    psql -U ${POSTGRES_USER:-braf_user} -d ${POSTGRES_DB:-braf_worker} -c "VACUUM ANALYZE;" >> $LOG_FILE 2>&1; then
    log_message "Database optimization completed"
else
    log_error "Database optimization failed"
fi

# 4. Clean Redis cache (selective cleanup)
log_message "Cleaning Redis cache..."
if docker-compose -f $COMPOSE_FILE exec -T redis \
    redis-cli --eval - <<< "
    local keys = redis.call('keys', 'temp:*')
    for i=1,#keys do
        redis.call('del', keys[i])
    end
    return #keys
    " >> $LOG_FILE 2>&1; then
    log_message "Redis temporary cache cleaned"
else
    log_warning "Redis cache cleanup had issues"
fi

# 5. Rotate logs (keep 7 days)
log_message "Rotating application logs..."
DELETED_LOGS=$(find /app/logs -name "*.log" -mtime +7 ! -name "maintenance_*" -delete -print 2>> $LOG_FILE | wc -l)
log_message "Deleted $DELETED_LOGS old log files"

# 6. Check disk space
log_message "Checking disk space..."
df -h /app >> $LOG_FILE
DISK_USAGE=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    log_warning "Disk usage is high: ${DISK_USAGE}%"
else
    log_message "Disk usage is acceptable: ${DISK_USAGE}%"
fi

# 7. Update Playwright browsers in worker nodes
log_message "Updating Playwright browsers..."
if docker-compose -f $COMPOSE_FILE exec -T worker_node \
    playwright install --with-deps chromium >> $LOG_FILE 2>&1; then
    log_message "Playwright browsers updated successfully"
else
    log_warning "Playwright browser update had issues"
fi

# 8. Health check all services
log_message "Performing health checks..."

# Check C2 server health
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    log_message "C2 server health check: PASSED"
else
    log_error "C2 server health check: FAILED"
fi

# Check worker node health
if docker-compose -f $COMPOSE_FILE exec -T worker_node \
    python -c "
import sys
sys.path.append('/app')
try:
    from src.braf.worker.main import health_check
    if health_check():
        print('Worker health check: PASSED')
        sys.exit(0)
    else:
        print('Worker health check: FAILED')
        sys.exit(1)
except Exception as e:
    print(f'Worker health check: ERROR - {e}')
    sys.exit(1)
" >> $LOG_FILE 2>&1; then
    log_message "Worker node health check: PASSED"
else
    log_error "Worker node health check: FAILED"
fi

# Check database connectivity
if docker-compose -f $COMPOSE_FILE exec -T postgres \
    pg_isready -U ${POSTGRES_USER:-braf_user} >> $LOG_FILE 2>&1; then
    log_message "Database connectivity check: PASSED"
else
    log_error "Database connectivity check: FAILED"
fi

# Check Redis connectivity
if docker-compose -f $COMPOSE_FILE exec -T redis \
    redis-cli ping >> $LOG_FILE 2>&1; then
    log_message "Redis connectivity check: PASSED"
else
    log_error "Redis connectivity check: FAILED"
fi

# 9. Collect system metrics
log_message "Collecting system metrics..."
{
    echo "=== System Metrics at $(date) ==="
    echo "Docker containers status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo "Memory usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    echo ""
    echo "Disk usage:"
    df -h
    echo ""
} >> $LOG_FILE

# 10. Database statistics
log_message "Collecting database statistics..."
docker-compose -f $COMPOSE_FILE exec -T postgres \
    psql -U ${POSTGRES_USER:-braf_user} -d ${POSTGRES_DB:-braf_worker} -c "
    SELECT 
        schemaname,
        tablename,
        n_tup_ins as inserts,
        n_tup_upd as updates,
        n_tup_del as deletes,
        n_live_tup as live_rows,
        n_dead_tup as dead_rows
    FROM pg_stat_user_tables 
    ORDER BY n_live_tup DESC;
" >> $LOG_FILE 2>&1

# 11. Cleanup temporary files
log_message "Cleaning temporary files..."
TEMP_FILES_DELETED=$(find /tmp -name "playwright-*" -mtime +1 -delete -print 2>> $LOG_FILE | wc -l)
log_message "Deleted $TEMP_FILES_DELETED temporary files"

# 12. Check for security updates (if applicable)
log_message "Checking for security updates..."
if command -v apt-get >/dev/null 2>&1; then
    apt list --upgradable 2>/dev/null | grep -i security >> $LOG_FILE || true
fi

# 13. Generate maintenance report
log_message "Generating maintenance report..."
REPORT_FILE="/app/logs/maintenance_report_$(date +%Y%m%d).json"
cat > $REPORT_FILE << EOF
{
    "maintenance_date": "$(date -Iseconds)",
    "backup_created": true,
    "old_backups_deleted": $DELETED_BACKUPS,
    "old_logs_deleted": $DELETED_LOGS,
    "disk_usage_percent": $DISK_USAGE,
    "temp_files_deleted": $TEMP_FILES_DELETED,
    "services_status": {
        "c2_server": "$(curl -f -s http://localhost:8000/health >/dev/null 2>&1 && echo 'healthy' || echo 'unhealthy')",
        "database": "$(docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U ${POSTGRES_USER:-braf_user} >/dev/null 2>&1 && echo 'healthy' || echo 'unhealthy')",
        "redis": "$(docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping >/dev/null 2>&1 && echo 'healthy' || echo 'unhealthy')"
    },
    "maintenance_duration_seconds": $(($(date +%s) - $(date -d "$(head -2 $LOG_FILE | tail -1 | cut -d']' -f1 | tr -d '[')" +%s)))
}
EOF

# 14. Send status report (if configured)
log_message "Preparing status report..."
if [ -f "/app/scripts/send_report.py" ]; then
    if docker-compose -f $COMPOSE_FILE exec -T c2_server \
        python /app/scripts/send_report.py --report-file $REPORT_FILE >> $LOG_FILE 2>&1; then
        log_message "Status report sent successfully"
    else
        log_warning "Status report sending failed"
    fi
else
    log_message "Status report script not found, skipping email notification"
fi

# 15. Final summary
echo "=========================================" >> $LOG_FILE
echo "BRAF daily maintenance completed at $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

# Display summary
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}BRAF Daily Maintenance Summary${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${BLUE}Completed at:${NC} $(date)"
echo -e "${BLUE}Log file:${NC} $LOG_FILE"
echo -e "${BLUE}Report file:${NC} $REPORT_FILE"
echo -e "${BLUE}Backups deleted:${NC} $DELETED_BACKUPS"
echo -e "${BLUE}Logs deleted:${NC} $DELETED_LOGS"
echo -e "${BLUE}Disk usage:${NC} ${DISK_USAGE}%"
echo -e "${GREEN}=========================================${NC}"

# Exit with appropriate code
if [ $DISK_USAGE -gt 90 ]; then
    log_error "Critical disk usage detected: ${DISK_USAGE}%"
    exit 1
else
    log_message "Daily maintenance completed successfully"
    exit 0
fi