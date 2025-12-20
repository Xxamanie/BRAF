#!/usr/bin/env python3
"""
BRAF Monetization System Startup Script
Initializes the complete system and starts all services
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from database.service import DatabaseService
from config import Config

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'sqlalchemy', 'alembic', 
        'redis', 'celery', 'stripe', 'web3', 'selenium',
        'cryptography', 'pyotp', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed")
    return True

def setup_database():
    """Initialize database and create tables"""
    print("\n🗄️ Setting up database...")
    
    try:
        # Import models to ensure they're registered
        from database.models import Base
        from database import engine
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
        
        # Test database connection
        with DatabaseService() as db:
            # Create a test enterprise if none exists
            existing = db.get_enterprise_by_email("admin@braf.com")
            if not existing:
                admin_enterprise = db.create_enterprise(
                    name="BRAF Admin",
                    email="admin@braf.com",
                    subscription_tier="enterprise"
                )
                
                # Create admin subscription
                db.create_subscription(admin_enterprise.id, {
                    "subscription_id": "sub_admin_001",
                    "tier": "enterprise",
                    "amount": 999.00
                })
                
                print(f"✅ Created admin enterprise: {admin_enterprise.id}")
            else:
                print(f"✅ Admin enterprise exists: {existing.id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def check_configuration():
    """Check system configuration"""
    print("\n⚙️ Checking configuration...")
    
    # Check environment file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found, using defaults")
    else:
        print("✅ .env file found")
    
    # Validate critical config
    missing_config = Config.validate_config()
    if missing_config:
        print(f"⚠️ Missing configuration: {', '.join(missing_config)}")
        print("Some features may not work without proper configuration")
    else:
        print("✅ Configuration validated")
    
    # Check database URL
    if Config.DATABASE_URL.startswith('sqlite'):
        print("✅ Using SQLite database (development)")
    else:
        print("✅ Using PostgreSQL database (production)")
    
    return True

def start_background_services():
    """Start background services (Redis, Celery)"""
    print("\n🔄 Starting background services...")
    
    # Check if Redis is available
    try:
        import redis
        r = redis.Redis.from_url(Config.REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"⚠️ Redis not available: {e}")
        print("Background tasks will not work without Redis")
    
    # Note: In production, Celery workers should be started separately
    print("ℹ️ Start Celery workers manually: celery -A worker.celery worker --loglevel=info")
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("\n📊 Creating sample data...")
    
    try:
        with DatabaseService() as db:
            # Get admin enterprise
            admin = db.get_enterprise_by_email("admin@braf.com")
            if not admin:
                print("⚠️ Admin enterprise not found")
                return False
            
            # Create sample automations
            existing_automations = db.get_automations(admin.id)
            if not existing_automations:
                # Survey automation
                survey_automation = db.create_automation({
                    "enterprise_id": admin.id,
                    "template_type": "survey",
                    "platform": "swagbucks",
                    "config": {"max_surveys": 5, "daily_limit": 10.0}
                })
                
                # Video automation
                video_automation = db.create_automation({
                    "enterprise_id": admin.id,
                    "template_type": "video",
                    "platform": "youtube",
                    "config": {"video_count": 20, "device_type": "desktop"}
                })
                
                print(f"✅ Created sample automations")
                
                # Create sample earnings
                db.record_earning({
                    "automation_id": survey_automation.id,
                    "amount": 5.50,
                    "platform": "swagbucks",
                    "task_type": "survey_completion"
                })
                
                db.record_earning({
                    "automation_id": video_automation.id,
                    "amount": 3.25,
                    "platform": "youtube",
                    "task_type": "video_viewing"
                })
                
                print(f"✅ Created sample earnings")
            else:
                print(f"✅ Sample data already exists ({len(existing_automations)} automations)")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting API server...")
    
    try:
        import uvicorn
        from main import app
        
        print("✅ FastAPI application loaded")
        print(f"🌐 Starting server on http://127.0.0.1:8001")
        print("📚 API Documentation: http://127.0.0.1:8001/docs")
        print("🏥 Health Check: http://127.0.0.1:8001/health")
        print("\n🎯 Ready to accept requests!")
        print("Press Ctrl+C to stop the server")
        
        # Start the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            reload=Config.is_development(),
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 BRAF Monetization System Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check configuration
    if not check_configuration():
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        sys.exit(1)
    
    # Start background services
    if not start_background_services():
        print("⚠️ Background services not fully available")
    
    # Create sample data
    if not create_sample_data():
        print("⚠️ Sample data creation failed")
    
    print("\n" + "=" * 50)
    print("✅ System initialization complete!")
    print("🎉 BRAF Monetization System is ready!")
    
    # Start API server (this will block)
    start_api_server()

if __name__ == "__main__":
    main()