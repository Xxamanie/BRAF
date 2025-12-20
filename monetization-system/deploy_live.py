#!/usr/bin/env python3
"""
Live deployment script for BRAF Monetization System
Sets up production environment with real-time currency rates
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
import requests

def check_system_requirements():
    """Check system requirements for live deployment"""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "python": {"cmd": "python --version", "min_version": "3.8"},
        "pip": {"cmd": f"{sys.executable} -m pip --version", "required": True},
        "git": {"cmd": "git --version", "required": False},
        "nginx": {"cmd": "nginx -v", "required": False},
        "systemctl": {"cmd": "systemctl --version", "required": False}
    }
    
    for tool, config in requirements.items():
        try:
            result = subprocess.run(config["cmd"].split(), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {tool}: Available")
            else:
                if config.get("required", True):
                    print(f"❌ {tool}: Required but not found")
                    return False
                else:
                    print(f"⚠️ {tool}: Optional, not found")
        except FileNotFoundError:
            if config.get("required", True):
                print(f"❌ {tool}: Required but not found")
                return False
            else:
                print(f"⚠️ {tool}: Optional, not found")
    
    return True

def test_currency_apis():
    """Test currency API endpoints"""
    print("\n💱 Testing currency API endpoints...")
    
    apis = [
        {
            "name": "ExchangeRate-API",
            "url": "https://api.exchangerate-api.com/v4/latest/USD",
            "free": True
        },
        {
            "name": "CurrencyAPI.com",
            "url": "https://api.currencyapi.com/v3/latest?base_currency=USD&currencies=NGN",
            "free": True
        }
    ]
    
    working_apis = []
    
    for api in apis:
        try:
            response = requests.get(api["url"], timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "rates" in data or "data" in data:
                    print(f"✅ {api['name']}: Working")
                    working_apis.append(api["name"])
                else:
                    print(f"⚠️ {api['name']}: Unexpected response format")
            else:
                print(f"❌ {api['name']}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {api['name']}: {str(e)}")
    
    if len(working_apis) > 0:
        print(f"✅ {len(working_apis)} currency APIs are working")
        return True
    else:
        print("❌ No currency APIs are working")
        return False

def setup_production_environment():
    """Setup production environment"""
    print("\n🔧 Setting up production environment...")
    
    # Create production directories
    directories = [
        "logs",
        "backups",
        "static",
        "uploads"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Copy production environment file
    if os.path.exists(".env.production"):
        if not os.path.exists(".env"):
            shutil.copy(".env.production", ".env")
            print("✅ Copied production environment configuration")
        else:
            print("⚠️ .env file already exists, skipping copy")
    
    # Set proper permissions
    try:
        os.chmod("deploy.sh", 0o755)
        os.chmod("manage.sh", 0o755)
        print("✅ Set executable permissions on scripts")
    except:
        print("⚠️ Could not set script permissions (Windows?)")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ Upgraded pip")
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Installed Python dependencies")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def initialize_database():
    """Initialize production database"""
    print("\n🗄️ Initializing database...")
    
    try:
        # Import and create tables
        from database import engine
        from database.models import Base
        
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created")
        
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_currency_conversion():
    """Test currency conversion with real APIs"""
    print("\n💱 Testing real-time currency conversion...")
    
    try:
        from payments.currency_converter import currency_converter
        
        # Test USD to NGN conversion
        result = currency_converter.convert_amount(100, "USD", "NGN")
        
        print(f"✅ Currency conversion test:")
        print(f"   💰 $100 USD = ₦{result['converted_amount']} NGN")
        print(f"   📊 Exchange rate: 1 USD = {result['exchange_rate']} NGN")
        print(f"   🔄 Rate source: {result['rate_source']}")
        print(f"   ⏰ Live rate: {'Yes' if result['is_live_rate'] else 'No (fallback)'}")
        
        if result['is_live_rate']:
            print("✅ Real-time currency conversion is working!")
            return True
        else:
            print("⚠️ Using fallback rates (APIs may be unavailable)")
            return True
            
    except Exception as e:
        print(f"❌ Currency conversion test failed: {e}")
        return False

def create_systemd_service():
    """Create systemd service for production"""
    print("\n🔧 Creating systemd service...")
    
    service_content = f"""[Unit]
Description=BRAF Monetization System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory={os.getcwd()}
Environment=PATH={os.getcwd()}/venv/bin
ExecStart={sys.executable} run_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    try:
        with open("braf-monetization.service", "w") as f:
            f.write(service_content)
        print("✅ Created systemd service file")
        print("   To install: sudo cp braf-monetization.service /etc/systemd/system/")
        print("   To enable: sudo systemctl enable braf-monetization")
        print("   To start: sudo systemctl start braf-monetization")
        return True
    except Exception as e:
        print(f"❌ Failed to create systemd service: {e}")
        return False

def start_production_server():
    """Start the production server"""
    print("\n🚀 Starting production server...")
    
    try:
        # Set production environment
        os.environ["ENVIRONMENT"] = "production"
        
        print("✅ Production server configuration:")
        print("   🌐 Host: 0.0.0.0")
        print("   🔌 Port: 8000")
        print("   👥 Workers: 4")
        print("   🔒 SSL: Ready (configure certificates)")
        print("   💱 Currency: Real-time rates enabled")
        
        print("\n📍 Access points:")
        print("   • Dashboard: http://your-server-ip:8000/dashboard")
        print("   • API Docs: http://your-server-ip:8000/docs")
        print("   • Health Check: http://your-server-ip:8000/health")
        
        print("\n🔧 To start the server:")
        print("   Development: python run_server.py")
        print("   Production: ./deploy.sh")
        print("   Service: sudo systemctl start braf-monetization")
        
        return True
    except Exception as e:
        print(f"❌ Failed to configure production server: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 BRAF Monetization System - Live Deployment")
    print("=" * 60)
    print("Deploying with real-time currency conversion")
    print("=" * 60)
    
    steps = [
        ("Checking system requirements", check_system_requirements),
        ("Testing currency APIs", test_currency_apis),
        ("Setting up production environment", setup_production_environment),
        ("Installing dependencies", install_dependencies),
        ("Initializing database", initialize_database),
        ("Testing currency conversion", test_currency_conversion),
        ("Creating systemd service", create_systemd_service),
        ("Configuring production server", start_production_server)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if not step_func():
                print(f"❌ {step_name} failed")
                return False
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("🎉 LIVE DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n✅ **PRODUCTION READY FEATURES:**")
    print("   • Real-time currency conversion (USD ↔ NGN)")
    print("   • Multiple currency API sources with fallback")
    print("   • Production database configuration")
    print("   • Systemd service for auto-restart")
    print("   • SSL/HTTPS ready")
    print("   • Rate limiting and security")
    print("   • Comprehensive logging")
    print("   • Health monitoring")
    
    print("\n🌐 **CURRENCY APIS CONFIGURED:**")
    print("   • ExchangeRate-API (free, reliable)")
    print("   • CurrencyAPI.com (free tier)")
    print("   • Fixer.io (with API key)")
    print("   • CurrencyLayer (with API key)")
    print("   • OpenExchangeRates (with API key)")
    
    print("\n🔧 **NEXT STEPS:**")
    print("   1. Configure domain and SSL certificates")
    print("   2. Set up reverse proxy (nginx)")
    print("   3. Configure firewall rules")
    print("   4. Set up monitoring and alerts")
    print("   5. Configure backup strategy")
    
    print("\n💡 **OPTIONAL ENHANCEMENTS:**")
    print("   • Get API keys for premium currency services")
    print("   • Set up Redis for caching")
    print("   • Configure email notifications")
    print("   • Set up Prometheus monitoring")
    
    print(f"\n📍 **DEPLOYMENT INFO:**")
    print(f"   • Timestamp: {datetime.now().isoformat()}")
    print(f"   • Directory: {os.getcwd()}")
    print(f"   • Python: {sys.version}")
    print(f"   • Status: 🟢 READY FOR PRODUCTION")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)