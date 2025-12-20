#!/bin/bash
# Restore script for BRAF system backups
# Restores from complete backup archives

set -e

# Configuration
BACKUP_DIR="/app/backups"
RESTORE_DIR="/app/restore_temp"
COMPOSE_FILE="docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log_message() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] BACKUP_FILE"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -f, --force             Force restore without confirmation"
    echo "  --database-only         Restore database only"
    echo "  --config-only           Restore configuration only"
    echo "  --data-only             Restore application data only"
    echo "  --verify-only           Verify backup integrity only"
    echo ""
    echo "Examples:"
    echo "  $0 /app/backups/complete_backup_20241220_140530.tar.gz"
    echo "  $0 --database-only backup_file.tar.gz"
    echo "  $0 --verify-only backup_file.tar.gz"
}

# Parse command line arguments
FORCE_RESTORE=false
DATABASE_ONLY=false
CONFIG_ONLY=false
DATA_ONLY=false
VERIFY_ONLY=false
BACKUP_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -f|--force)
            FORCE_RESTORE=true
            shift
            ;;
        --database-only)
            DATABASE_ONLY=true
            shift
            ;;
        --config-only)
            CONFIG_ONLY=true
            shift
            ;;
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        -*)
            log_error "Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            BACKUP_FILE="$1"
            shift
            ;;
    esac
done

# Check if backup file is provided
if [ -z "$BACKUP_FILE" ]; then
    log_error "Backup file not specified"
    show_usage
    exit 1
fi

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}BRAF System Restore${NC}"
echo -e "${BLUE}=========================================${NC}"

log_message "Backup file: $BACKUP_FILE"

# Verify backup integrity
log_message "Verifying backup integrity..."
if tar -tzf "$BACKUP_FILE" >/dev/null 2>&1; then
    log_message "✅ Backup integrity verification: PASSED"
else
    log_error "❌ Backup integrity verification: FAILED"
    exit 1
fi

if [ "$VERIFY_ONLY" = true ]; then
    log_message "✅ Backup verification completed successfully"
    exit 0
fi

# Create restore directory
log_message "Creating restore directory..."
rm -rf "$RESTORE_DIR"
mkdir -p "$RESTORE_DIR"

# Extract backup
log_message "Extracting backup archive..."
if tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR" 2>/dev/null; then
    log_message "✅ Backup extracted successfully"
else
    log_error "❌ Failed to extract backup"
    exit 1
fi

# Find the backup directory inside the extracted archive
BACKUP_CONTENT_DIR=$(find "$RESTORE_DIR" -maxdepth 1 -type d -name "complete_backup_*" | head -1)
if [ -z "$BACKUP_CONTENT_DIR" ]; then
    log_error "Backup content directory not found"
    exit 1
fi

log_message "Backup content directory: $BACKUP_CONTENT_DIR"

# Load backup manifest if available
MANIFEST_FILE="$BACKUP_CONTENT_DIR/backup_manifest.json"
if [ -f "$MANIFEST_FILE" ]; then
    log_message "Loading backup manifest..."
    BACKUP_TIMESTAMP=$(grep -o '"timestamp": "[^"]*"' "$MANIFEST_FILE" | cut -d'"' -f4)
    BACKUP_TYPE=$(grep -o '"backup_type": "[^"]*"' "$MANIFEST_FILE" | cut -d'"' -f4)
    log_message "Backup timestamp: $BACKUP_TIMESTAMP"
    log_message "Backup type: $BACKUP_TYPE"
else
    log_warning "Backup manifest not found"
fi

# Confirmation prompt (unless forced)
if [ "$FORCE_RESTORE" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️  WARNING: This will restore the BRAF system from backup${NC}"
    echo -e "${YELLOW}   Current data may be overwritten!${NC}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_message "Restore cancelled by user"
        rm -rf "$RESTORE_DIR"
        exit 0
    fi
fi

# Stop services before restore
log_message "Stopping BRAF services..."
if docker-compose -f $COMPOSE_FILE down 2>/dev/null; then
    log_message "✅ Services stopped successfully"
else
    log_warning "Failed to stop some services"
fi

# Restore database
if [ "$DATABASE_ONLY" = true ] || [ "$CONFIG_ONLY" = false ] && [ "$DATA_ONLY" = false ]; then
    log_message "Restoring PostgreSQL database..."
    
    # Start only database service
    docker-compose -f $COMPOSE_FILE up -d postgres
    sleep 10
    
    # Find database backup file
    DB_BACKUP_FILE=""
    if [ -f "$BACKUP_CONTENT_DIR/postgres_complete.sql.gz" ]; then
        DB_BACKUP_FILE="$BACKUP_CONTENT_DIR/postgres_complete.sql.gz"
        gunzip -c "$DB_BACKUP_FILE" | docker-compose -f $COMPOSE_FILE exec -T postgres \
            psql -U ${POSTGRES_USER:-braf_user} -d postgres 2>/dev/null
    elif [ -f "$BACKUP_CONTENT_DIR/postgres_complete.sql" ]; then
        DB_BACKUP_FILE="$BACKUP_CONTENT_DIR/postgres_complete.sql"
        docker-compose -f $COMPOSE_FILE exec -T postgres \
            psql -U ${POSTGRES_USER:-braf_user} -d postgres < "$DB_BACKUP_FILE" 2>/dev/null
    fi
    
    if [ -n "$DB_BACKUP_FILE" ]; then
        log_message "✅ Database restore completed"
    else
        log_error "❌ Database backup file not found"
    fi
fi

# Restore Redis data
if [ "$DATABASE_ONLY" = true ] || [ "$CONFIG_ONLY" = false ] && [ "$DATA_ONLY" = false ]; then
    if [ -f "$BACKUP_CONTENT_DIR/redis.rdb" ]; then
        log_message "Restoring Redis data..."
        
        # Start Redis service
        docker-compose -f $COMPOSE_FILE up -d redis
        sleep 5
        
        # Stop Redis, copy dump file, and restart
        docker-compose -f $COMPOSE_FILE stop redis
        docker-compose -f $COMPOSE_FILE cp "$BACKUP_CONTENT_DIR/redis.rdb" redis:/data/dump.rdb
        docker-compose -f $COMPOSE_FILE start redis
        
        log_message "✅ Redis restore completed"
    else
        log_warning "Redis backup file not found"
    fi
fi

# Restore application data
if [ "$DATA_ONLY" = true ] || [ "$CONFIG_ONLY" = false ] && [ "$DATABASE_ONLY" = false ]; then
    if [ -f "$BACKUP_CONTENT_DIR/app_data.tar.gz" ]; then
        log_message "Restoring application data..."
        
        # Backup existing data
        if [ -d "/app/data" ]; then
            mv /app/data "/app/data.backup.$(date +%Y%m%d_%H%M%S)"
        fi
        
        # Create data directory and restore
        mkdir -p /app/data
        tar -xzf "$BACKUP_CONTENT_DIR/app_data.tar.gz" -C /app/data
        
        log_message "✅ Application data restore completed"
    else
        log_warning "Application data backup not found"
    fi
fi

# Restore configurations
if [ "$CONFIG_ONLY" = true ] || [ "$DATABASE_ONLY" = false ] && [ "$DATA_ONLY" = false ]; then
    log_message "Restoring configurations..."
    
    # Restore configuration files
    CONFIG_FILES=(
        "$COMPOSE_FILE"
        ".env.production"
    )
    
    for config_file in "${CONFIG_FILES[@]}"; do
        if [ -f "$BACKUP_CONTENT_DIR/$config_file" ]; then
            cp "$BACKUP_CONTENT_DIR/$config_file" "/app/"
            log_message "✅ Restored $config_file"
        fi
    done
    
    # Restore configuration directories
    CONFIG_DIRS=("config" "nginx" "monitoring" "grafana" "scripts")
    
    for config_dir in "${CONFIG_DIRS[@]}"; do
        if [ -d "$BACKUP_CONTENT_DIR/$config_dir" ]; then
            # Backup existing directory
            if [ -d "/app/$config_dir" ]; then
                mv "/app/$config_dir" "/app/${config_dir}.backup.$(date +%Y%m%d_%H%M%S)"
            fi
            
            # Restore directory
            cp -r "$BACKUP_CONTENT_DIR/$config_dir" "/app/"
            log_message "✅ Restored $config_dir directory"
        fi
    done
fi

# Restore certificates and uploads
if [ "$DATA_ONLY" = true ] || [ "$CONFIG_ONLY" = false ] && [ "$DATABASE_ONLY" = false ]; then
    ASSET_ARCHIVES=("certificates.tar.gz" "uploads.tar.gz")
    
    for archive in "${ASSET_ARCHIVES[@]}"; do
        if [ -f "$BACKUP_CONTENT_DIR/$archive" ]; then
            asset_dir=$(basename "$archive" .tar.gz)
            log_message "Restoring $asset_dir..."
            
            # Backup existing directory
            if [ -d "/app/$asset_dir" ]; then
                mv "/app/$asset_dir" "/app/${asset_dir}.backup.$(date +%Y%m%d_%H%M%S)"
            fi
            
            # Create directory and restore
            mkdir -p "/app/$asset_dir"
            tar -xzf "$BACKUP_CONTENT_DIR/$archive" -C "/app/$asset_dir"
            
            log_message "✅ $asset_dir restore completed"
        fi
    done
fi

# Restore Docker images (if requested)
if [ -f "$BACKUP_CONTENT_DIR/docker_images.tar.gz" ]; then
    echo ""
    read -p "Do you want to restore Docker images? (y/n): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_message "Restoring Docker images..."
        
        gunzip -c "$BACKUP_CONTENT_DIR/docker_images.tar.gz" | docker load
        
        log_message "✅ Docker images restore completed"
    fi
fi

# Verify checksums
if [ -f "$BACKUP_CONTENT_DIR/checksums.sha256" ]; then
    log_message "Verifying restored files..."
    
    cd "$BACKUP_CONTENT_DIR"
    if sha256sum -c checksums.sha256 >/dev/null 2>&1; then
        log_message "✅ File integrity verification: PASSED"
    else
        log_warning "⚠️ Some files failed integrity check"
    fi
    cd - >/dev/null
fi

# Start services
log_message "Starting BRAF services..."
if docker-compose -f $COMPOSE_FILE up -d 2>/dev/null; then
    log_message "✅ Services started successfully"
else
    log_error "❌ Failed to start some services"
fi

# Wait for services to be ready
log_message "Waiting for services to be ready..."
sleep 30

# Health check
log_message "Performing health checks..."
HEALTH_PASSED=0

# Check C2 server
if curl -f -s http://localhost:8000/health >/dev/null 2>&1; then
    log_message "✅ C2 server health check: PASSED"
    ((HEALTH_PASSED++))
else
    log_error "❌ C2 server health check: FAILED"
fi

# Check database
if docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U ${POSTGRES_USER:-braf_user} >/dev/null 2>&1; then
    log_message "✅ Database health check: PASSED"
    ((HEALTH_PASSED++))
else
    log_error "❌ Database health check: FAILED"
fi

# Check Redis
if docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping >/dev/null 2>&1; then
    log_message "✅ Redis health check: PASSED"
    ((HEALTH_PASSED++))
else
    log_error "❌ Redis health check: FAILED"
fi

# Cleanup
log_message "Cleaning up temporary files..."
rm -rf "$RESTORE_DIR"

# Final summary
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}BRAF System Restore Summary${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${BLUE}Restore completed at:${NC} $(date)"
echo -e "${BLUE}Backup file:${NC} $BACKUP_FILE"
echo -e "${BLUE}Health checks passed:${NC} $HEALTH_PASSED/3"

if [ $HEALTH_PASSED -eq 3 ]; then
    echo -e "${GREEN}✅ System restore completed successfully!${NC}"
    echo -e "${GREEN}All services are healthy and operational${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ System restore completed with warnings${NC}"
    echo -e "${YELLOW}Some services may need manual intervention${NC}"
    exit 1
fi