#!/bin/bash
# BRAF Monetization System Management Script

case "$1" in
    start)
        sudo systemctl start braf-monetization
        echo "✅ Service started"
        ;;
    stop)
        sudo systemctl stop braf-monetization
        echo "✅ Service stopped"
        ;;
    restart)
        sudo systemctl restart braf-monetization
        echo "✅ Service restarted"
        ;;
    status)
        sudo systemctl status braf-monetization
        ;;
    logs)
        sudo journalctl -u braf-monetization -f
        ;;
    update)
        echo "🔄 Updating system..."
        git pull
        sudo systemctl restart braf-monetization
        echo "✅ System updated"
        ;;
    backup)
        echo "💾 Creating backup..."
        timestamp=$(date +%Y%m%d_%H%M%S)
        cp braf_production.db "backups/braf_backup_$timestamp.db"
        echo "✅ Backup created: backups/braf_backup_$timestamp.db"
        ;;
    seed-data)
        echo "🌱 Seeding sample data..."
        python seed_sample_data.py
        echo "✅ Sample data seeded"
        ;;
    create-account)
        echo "👤 Creating new account..."
        python create_account.py
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|update|backup|seed-data|create-account}"
        exit 1
        ;;
esac
