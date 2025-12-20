#!/bin/bash
# Complete backup script for BRAF system
# Updated to use correct service names (c2_server, worker_node)

set -e

# Configuration
BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="complete_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"
COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create backup directory
mkdir -p "${BACKUP_PATH}"

# Initialize log file
LOG_FILE="${BACKUP_PATH}/backup.log"
echo "=========================================" > "${LOG_FILE}"
echo "BRAF Complete Backup Started at $(date)" >> "${LOG_FILE}"
echo "=========================================" >> "${LOG_FILE}"

# Function to log with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "${LOG_FILE}"
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "${LOG_FILE}"
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "${LOG_FILE}"
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}BRAF Complete System Backup${NC}"
echo -e "${BLUE}=========================================${NC}"

# 1. Backup PostgreSQL (complete database dump)
log_message "Backing up PostgreSQL database..."
if docker-compose -f $COMPOSE_FILE exec -T postgres \
    pg_dumpall -U ${POSTGRES_USER:-braf_user} > "${BACKUP_PATH}/postgres_complete.sql" 2>> "${LOG_FILE}"; then
    # Compress the SQL dump
    gzip "${BACKUP_PATH}/postgres_complete.sql"
    DB_SIZE=$(du -h "${BACKUP_PATH}/postgres_complete.sql.gz" | cut -f1)
    log_message "PostgreSQL backup completed: $DB_SIZE"
else
    log_error "PostgreSQL backup failed"
fi

# 2. Backup Redis data
log_message "Backing up Redis data..."
if docker-compose -f $COMPOSE_FILE exec -T redis \
    redis-cli --rdb /data/dump.rdb >> "${LOG_FILE}" 2>&1; then
    
    # Copy Redis dump file
    if docker-compose -f $COMPOSE_FILE cp redis:/data/dump.rdb "${BACKUP_PATH}/redis.rdb" 2>> "${LOG_FILE}"; then
        REDIS_SIZE=$(du -h "${BACKUP_PATH}/redis.rdb" | cut -f1)
        log_message "Redis backup completed: $REDIS_SIZE"
    else
        log_error "Failed to copy Redis dump file"
    fi
else
    log_error "Redis backup failed"
fi

# 3. Backup application data
log_message "Backing up application data..."
if [ -d "/app/data" ]; then
    if tar -czf "${BACKUP_PATH}/app_data.tar.gz" -C /app/data . 2>> "${LOG_FILE}"; then
        DATA_SIZE=$(du -h "${BACKUP_PATH}/app_data.tar.gz" | cut -f1)
        log_message "Application data backup completed: $DATA_SIZE"
    else
        log_error "Application data backup failed"
    fi
else
    log_warning "Application data directory not found"
fi

# 4. Backup logs (last 7 days only)
log_message "Backing up recent logs (last 7 days)..."
if find /app/logs -name "*.log" -mtime -7 -print0 | tar -czf "${BACKUP_PATH}/logs_recent.tar.gz" --null -T - 2>> "${LOG_FILE}"; then
    LOGS_SIZE=$(du -h "${BACKUP_PATH}/logs_recent.tar.gz" | cut -f1)
    log_message "Logs backup completed: $LOGS_SIZE"
else
    log_warning "Logs backup had issues (may be empty)"
fi

# 5. Backup configurations
log_message "Backing up system configurations..."
CONFIG_FILES=(
    "/app/$COMPOSE_FILE"
    "/app/.env.production"
    "/app/config"
    "/app/nginx"
    "/app/monitoring"
    "/app/grafana"
    "/app/scripts"
)

for config_item in "${CONFIG_FILES[@]}"; do
    if [ -e "$config_item" ]; then
        cp -r "$config_item" "${BACKUP_PATH}/" 2>> "${LOG_FILE}" || log_warning "Failed to backup $config_item"
    else
        log_warning "Configuration item not found: $config_item"
    fi
done

log_message "Configuration backup completed"

# 6. Backup certificates and uploads
log_message "Backing up certificates and uploads..."
ASSET_DIRS=("/app/certificates" "/app/uploads" "/app/backups")
for asset_dir in "${ASSET_DIRS[@]}"; do
    if [ -d "$asset_dir" ]; then
        dir_name=$(basename "$asset_dir")
        if tar -czf "${BACKUP_PATH}/${dir_name}.tar.gz" -C "$asset_dir" . 2>> "${LOG_FILE}"; then
            ASSET_SIZE=$(du -h "${BACKUP_PATH}/${dir_name}.tar.gz" | cut -f1)
            log_message "$dir_name backup completed: $ASSET_SIZE"
        else
            log_warning "$dir_name backup failed"
        fi
    else
        log_warning "$asset_dir directory not found"
    fi
done

# 7. Export Docker images
log_message "Exporting Docker images..."
IMAGES_FILE="${BACKUP_PATH}/docker_images.tar"
if docker-compose -f $COMPOSE_FILE config --services | xargs -I {} docker-compose -f $COMPOSE_FILE images -q {} | sort -u > /tmp/image_list.txt 2>> "${LOG_FILE}"; then
    if [ -s /tmp/image_list.txt ]; then
        if docker save $(cat /tmp/image_list.txt) -o "$IMAGES_FILE" 2>> "${LOG_FILE}"; then
            # Compress the images file
            gzip "$IMAGES_FILE"
            IMAGES_SIZE=$(du -h "${IMAGES_FILE}.gz" | cut -f1)
            log_message "Docker images backup completed: $IMAGES_SIZE"
        else
            log_error "Docker images export failed"
        fi
    else
        log_warning "No Docker images found to backup"
    fi
    rm -f /tmp/image_list.txt
else
    log_error "Failed to list Docker images"
fi

# 8. Create system information snapshot
log_message "Creating system information snapshot..."
SYSINFO_FILE="${BACKUP_PATH}/system_info.txt"
{
    echo "=== BRAF System Information ==="
    echo "Backup Date: $(date)"
    echo "System: $(uname -a)"
    echo "Docker Version: $(docker --version)"
    echo "Docker Compose Version: $(docker-compose --version)"
    echo ""
    echo "=== Docker Containers ==="
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo "=== Docker Images ==="
    docker-compose -f $COMPOSE_FILE images
    echo ""
    echo "=== System Resources ==="
    df -h
    echo ""
    free -h
    echo ""
    echo "=== Network Configuration ==="
    docker network ls
    echo ""
    echo "=== Environment Variables ==="
    env | grep -E "(POSTGRES|REDIS|BRAF|DATABASE)" | sort
} > "$SYSINFO_FILE" 2>> "${LOG_FILE}"

log_message "System information snapshot created"

# 9. Create checksums for integrity verification
log_message "Creating checksums for integrity verification..."
cd "${BACKUP_PATH}"
if sha256sum * > checksums.sha256 2>> "${LOG_FILE}"; then
    log_message "Checksums created successfully"
else
    log_error "Failed to create checksums"
fi

# 10. Create backup manifest
log_message "Creating backup manifest..."
MANIFEST_FILE="${BACKUP_PATH}/backup_manifest.json"
cat > "$MANIFEST_FILE" << EOF
{
    "backup_info": {
        "timestamp": "$(date -Iseconds)",
        "backup_type": "complete_system",
        "system": "BRAF",
        "version": "1.0.0",
        "backup_name": "$BACKUP_NAME"
    },
    "components": {
        "database": {
            "type": "PostgreSQL",
            "file": "postgres_complete.sql.gz",
            "method": "pg_dumpall",
            "compressed": true
        },
        "cache": {
            "type": "Redis",
            "file": "redis.rdb",
            "method": "redis-cli --rdb",
            "compressed": false
        },
        "application_data": {
            "file": "app_data.tar.gz",
            "compressed": true
        },
        "logs": {
            "file": "logs_recent.tar.gz",
            "retention": "7_days",
            "compressed": true
        },
        "docker_images": {
            "file": "docker_images.tar.gz",
            "compressed": true
        }
    },
    "system_info": {
        "file": "system_info.txt",
        "checksums": "checksums.sha256"
    },
    "backup_location": "$BACKUP_PATH",
    "retention_policy": "30_days"
}
EOF

# 11. Calculate total backup size
TOTAL_SIZE=$(du -sh "${BACKUP_PATH}" | cut -f1)
log_message "Total backup size: $TOTAL_SIZE"

# 12. Compress entire backup
log_message "Compressing complete backup..."
cd "${BACKUP_DIR}"
if tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}" 2>> "${LOG_FILE}"; then
    COMPRESSED_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
    log_message "Backup compression completed: $COMPRESSED_SIZE"
    
    # Remove uncompressed backup directory
    rm -rf "${BACKUP_PATH}"
    
    # Update final log location
    LOG_FILE="${BACKUP_DIR}/${BACKUP_NAME}.log"
    mv "${BACKUP_DIR}/${BACKUP_NAME}/backup.log" "$LOG_FILE" 2>/dev/null || true
else
    log_error "Backup compression failed"
fi

# 13. Upload to cloud storage (if configured)
if [ -f "/app/config/cloud_storage.env" ]; then
    log_message "Uploading to cloud storage..."
    source /app/config/cloud_storage.env
    
    if command -v aws >/dev/null 2>&1; then
        if aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" "s3://${S3_BUCKET}/backups/" >> "$LOG_FILE" 2>&1; then
            log_message "Cloud upload completed successfully"
        else
            log_error "Cloud upload failed"
        fi
    else
        log_warning "AWS CLI not found, skipping cloud upload"
    fi
else
    log_message "Cloud storage not configured, skipping upload"
fi

# 14. Clean old backups (keep 30 days)
log_message "Cleaning old backups (keeping 30 days)..."
OLD_BACKUPS=$(find "${BACKUP_DIR}" -name "complete_backup_*.tar.gz" -mtime +30 | wc -l)
find "${BACKUP_DIR}" -name "complete_backup_*.tar.gz" -mtime +30 -delete 2>> "$LOG_FILE"
find "${BACKUP_DIR}" -name "complete_backup_*.log" -mtime +30 -delete 2>> "$LOG_FILE"
log_message "Cleaned $OLD_BACKUPS old backup files"

# 15. Verify backup integrity
log_message "Verifying backup integrity..."
if [ -f "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ]; then
    if tar -tzf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" >/dev/null 2>&1; then
        log_message "✅ Backup integrity verification: PASSED"
        BACKUP_STATUS="SUCCESS"
    else
        log_error "❌ Backup integrity verification: FAILED"
        BACKUP_STATUS="FAILED"
    fi
else
    log_error "❌ Backup file not found"
    BACKUP_STATUS="FAILED"
fi

# Final summary
echo "=========================================" >> "$LOG_FILE"
echo "BRAF complete backup finished at $(date)" >> "$LOG_FILE"
echo "Status: $BACKUP_STATUS" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}BRAF Complete Backup Summary${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${BLUE}Status:${NC} $BACKUP_STATUS"
echo -e "${BLUE}Completed at:${NC} $(date)"
echo -e "${BLUE}Backup file:${NC} ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo -e "${BLUE}Compressed size:${NC} $COMPRESSED_SIZE"
echo -e "${BLUE}Log file:${NC} $LOG_FILE"
echo -e "${BLUE}Old backups cleaned:${NC} $OLD_BACKUPS"
echo -e "${GREEN}=========================================${NC}"

# Send notification (if configured)
if [ -f "/app/scripts/send_report.py" ]; then
    log_message "Sending backup notification..."
    python /app/scripts/send_report.py --report-file "$MANIFEST_FILE" >> "$LOG_FILE" 2>&1 || log_warning "Failed to send notification"
fi

if [ "$BACKUP_STATUS" = "SUCCESS" ]; then
    echo -e "${GREEN}✅ Complete backup completed successfully!${NC}"
    echo "Backup saved to: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
    exit 0
else
    echo -e "${RED}❌ Backup completed with errors${NC}"
    exit 1
fi