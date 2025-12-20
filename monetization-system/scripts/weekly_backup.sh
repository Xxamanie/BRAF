#!/bin/bash
# Weekly comprehensive backup for BRAF system

LOG_FILE="/app/logs/weekly_backup_$(date +%Y%m%d).log"
BACKUP_DIR="/app/backups/weekly"
COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure directories exist
mkdir -p /app/logs "$BACKUP_DIR"

echo "=========================================" >> $LOG_FILE
echo "Starting BRAF weekly backup at $(date)" >> $LOG_FILE
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

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}BRAF Weekly Comprehensive Backup${NC}"
echo -e "${BLUE}=========================================${NC}"

# 1. Full database backup with compression
log_message "Creating full database backup..."
DB_BACKUP_FILE="$BACKUP_DIR/database_full_$(date +%Y%m%d_%H%M%S).sql.gz"
if docker-compose -f $COMPOSE_FILE exec -T postgres \
    pg_dump -U ${POSTGRES_USER:-braf_user} ${POSTGRES_DB:-braf_worker} --verbose | gzip > "$DB_BACKUP_FILE" 2>> $LOG_FILE; then
    DB_SIZE=$(du -h "$DB_BACKUP_FILE" | cut -f1)
    log_message "Database backup completed: $DB_SIZE"
else
    log_error "Database backup failed"
fi

# 2. Backup application data
log_message "Backing up application data..."
DATA_BACKUP_FILE="$BACKUP_DIR/app_data_$(date +%Y%m%d_%H%M%S).tar.gz"
if tar -czf "$DATA_BACKUP_FILE" -C /app data certificates uploads 2>> $LOG_FILE; then
    DATA_SIZE=$(du -h "$DATA_BACKUP_FILE" | cut -f1)
    log_message "Application data backup completed: $DATA_SIZE"
else
    log_error "Application data backup failed"
fi

# 3. Backup configuration files
log_message "Backing up configuration files..."
CONFIG_BACKUP_FILE="$BACKUP_DIR/config_$(date +%Y%m%d_%H%M%S).tar.gz"
if tar -czf "$CONFIG_BACKUP_FILE" -C /app \
    docker-compose.prod.yml \
    .env.production \
    nginx/ \
    monitoring/ \
    grafana/ \
    scripts/ 2>> $LOG_FILE; then
    CONFIG_SIZE=$(du -h "$CONFIG_BACKUP_FILE" | cut -f1)
    log_message "Configuration backup completed: $CONFIG_SIZE"
else
    log_error "Configuration backup failed"
fi

# 4. Export Docker images
log_message "Exporting Docker images..."
IMAGES_BACKUP_FILE="$BACKUP_DIR/docker_images_$(date +%Y%m%d_%H%M%S).tar"
if docker save $(docker-compose -f $COMPOSE_FILE config --services | xargs -I {} docker-compose -f $COMPOSE_FILE images -q {}) -o "$IMAGES_BACKUP_FILE" 2>> $LOG_FILE; then
    # Compress the images file
    gzip "$IMAGES_BACKUP_FILE"
    IMAGES_SIZE=$(du -h "${IMAGES_BACKUP_FILE}.gz" | cut -f1)
    log_message "Docker images backup completed: $IMAGES_SIZE"
else
    log_error "Docker images backup failed"
fi

# 5. Backup logs (last 30 days)
log_message "Backing up recent logs..."
LOGS_BACKUP_FILE="$BACKUP_DIR/logs_$(date +%Y%m%d_%H%M%S).tar.gz"
if find /app/logs -name "*.log" -mtime -30 -print0 | tar -czf "$LOGS_BACKUP_FILE" --null -T - 2>> $LOG_FILE; then
    LOGS_SIZE=$(du -h "$LOGS_BACKUP_FILE" | cut -f1)
    log_message "Logs backup completed: $LOGS_SIZE"
else
    log_error "Logs backup failed"
fi

# 6. Create backup manifest
log_message "Creating backup manifest..."
MANIFEST_FILE="$BACKUP_DIR/backup_manifest_$(date +%Y%m%d_%H%M%S).json"
cat > "$MANIFEST_FILE" << EOF
{
    "backup_date": "$(date -Iseconds)",
    "backup_type": "weekly_full",
    "system": "BRAF",
    "version": "1.0.0",
    "files": {
        "database": {
            "file": "$(basename "$DB_BACKUP_FILE")",
            "size": "$(du -b "$DB_BACKUP_FILE" 2>/dev/null | cut -f1 || echo 0)",
            "compressed": true
        },
        "app_data": {
            "file": "$(basename "$DATA_BACKUP_FILE")",
            "size": "$(du -b "$DATA_BACKUP_FILE" 2>/dev/null | cut -f1 || echo 0)",
            "compressed": true
        },
        "configuration": {
            "file": "$(basename "$CONFIG_BACKUP_FILE")",
            "size": "$(du -b "$CONFIG_BACKUP_FILE" 2>/dev/null | cut -f1 || echo 0)",
            "compressed": true
        },
        "docker_images": {
            "file": "$(basename "${IMAGES_BACKUP_FILE}.gz")",
            "size": "$(du -b "${IMAGES_BACKUP_FILE}.gz" 2>/dev/null | cut -f1 || echo 0)",
            "compressed": true
        },
        "logs": {
            "file": "$(basename "$LOGS_BACKUP_FILE")",
            "size": "$(du -b "$LOGS_BACKUP_FILE" 2>/dev/null | cut -f1 || echo 0)",
            "compressed": true
        }
    },
    "total_files": 5,
    "backup_location": "$BACKUP_DIR",
    "retention_days": 90
}
EOF

# 7. Calculate total backup size
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log_message "Total backup size: $TOTAL_SIZE"

# 8. Clean old weekly backups (keep 12 weeks = 3 months)
log_message "Cleaning old weekly backups (keeping 12 weeks)..."
OLD_BACKUPS=$(find "$BACKUP_DIR" -name "*.tar.gz" -o -name "*.sql.gz" -mtime +84 | wc -l)
find "$BACKUP_DIR" -name "*.tar.gz" -o -name "*.sql.gz" -mtime +84 -delete 2>> $LOG_FILE
find "$BACKUP_DIR" -name "*.json" -mtime +84 -delete 2>> $LOG_FILE
log_message "Cleaned $OLD_BACKUPS old backup files"

# 9. Verify backup integrity
log_message "Verifying backup integrity..."
INTEGRITY_PASSED=0

# Test database backup
if gunzip -t "$DB_BACKUP_FILE" 2>/dev/null; then
    log_message "✓ Database backup integrity: PASSED"
    ((INTEGRITY_PASSED++))
else
    log_error "✗ Database backup integrity: FAILED"
fi

# Test data backup
if tar -tzf "$DATA_BACKUP_FILE" >/dev/null 2>&1; then
    log_message "✓ Data backup integrity: PASSED"
    ((INTEGRITY_PASSED++))
else
    log_error "✗ Data backup integrity: FAILED"
fi

# Test config backup
if tar -tzf "$CONFIG_BACKUP_FILE" >/dev/null 2>&1; then
    log_message "✓ Config backup integrity: PASSED"
    ((INTEGRITY_PASSED++))
else
    log_error "✗ Config backup integrity: FAILED"
fi

# 10. Generate backup report
log_message "Generating backup report..."
BACKUP_REPORT="/app/logs/weekly_backup_report_$(date +%Y%m%d).html"
cat > "$BACKUP_REPORT" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BRAF Weekly Backup Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
        .success { color: #27ae60; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        .info { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🗄️ BRAF Weekly Backup Report</h1>
        <p>Generated: $(date)</p>
    </div>
    
    <div class="info">
        <strong>Backup Summary:</strong><br>
        Total Size: $TOTAL_SIZE<br>
        Files Created: 5<br>
        Integrity Checks Passed: $INTEGRITY_PASSED/3<br>
        Old Backups Cleaned: $OLD_BACKUPS
    </div>
    
    <h2>📁 Backup Files</h2>
    <table>
        <tr><th>Component</th><th>File</th><th>Size</th><th>Status</th></tr>
        <tr><td>Database</td><td>$(basename "$DB_BACKUP_FILE")</td><td>$DB_SIZE</td><td class="success">✓ Created</td></tr>
        <tr><td>Application Data</td><td>$(basename "$DATA_BACKUP_FILE")</td><td>$DATA_SIZE</td><td class="success">✓ Created</td></tr>
        <tr><td>Configuration</td><td>$(basename "$CONFIG_BACKUP_FILE")</td><td>$CONFIG_SIZE</td><td class="success">✓ Created</td></tr>
        <tr><td>Docker Images</td><td>$(basename "${IMAGES_BACKUP_FILE}.gz")</td><td>$IMAGES_SIZE</td><td class="success">✓ Created</td></tr>
        <tr><td>Logs</td><td>$(basename "$LOGS_BACKUP_FILE")</td><td>$LOGS_SIZE</td><td class="success">✓ Created</td></tr>
    </table>
    
    <div class="info">
        <strong>Next Steps:</strong><br>
        1. Verify backups are accessible<br>
        2. Consider offsite backup storage<br>
        3. Test restore procedures periodically<br>
        4. Monitor backup storage usage
    </div>
</body>
</html>
EOF

# 11. Send backup notification
log_message "Sending backup notification..."
if [ -f "/app/scripts/send_report.py" ]; then
    python /app/scripts/send_report.py --report-file "$MANIFEST_FILE" --email admin@braf.local >> $LOG_FILE 2>&1
    if [ $? -eq 0 ]; then
        log_message "Backup notification sent successfully"
    else
        log_warning "Failed to send backup notification"
    fi
else
    log_message "Backup notification script not found"
fi

# Final summary
echo "=========================================" >> $LOG_FILE
echo "BRAF weekly backup completed at $(date)" >> $LOG_FILE
echo "=========================================" >> $LOG_FILE

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}BRAF Weekly Backup Summary${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${BLUE}Completed at:${NC} $(date)"
echo -e "${BLUE}Total backup size:${NC} $TOTAL_SIZE"
echo -e "${BLUE}Backup location:${NC} $BACKUP_DIR"
echo -e "${BLUE}Integrity checks:${NC} $INTEGRITY_PASSED/3 passed"
echo -e "${BLUE}Old backups cleaned:${NC} $OLD_BACKUPS files"
echo -e "${GREEN}=========================================${NC}"

if [ $INTEGRITY_PASSED -eq 3 ]; then
    log_message "✅ Weekly backup completed successfully"
    exit 0
else
    log_error "⚠️ Weekly backup completed with integrity issues"
    exit 1
fi