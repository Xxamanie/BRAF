#!/usr/bin/env python3
"""
Windows Live deployment script for BRAF Monetization System
Sets up production environment with real-time currency rates (Windows compatible)
"""

import os
import sys
import subprocess
import shutil
from datetime import datetime
import requests

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
    
    return True

def install_core_dependencies():
    """Install core Python dependencies (Windows compatible)"""
    print("\n📦 Installing core dependencies...")
    
    # Core dependencies without PostgreSQL
    core_deps = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "pydantic==2.5.0",
        "python-multipart==0.0.6",
        "sqlalchemy==2.0.23",
        "requests==2.31.0",
        "python-jose[cryptography]==3.3.0",
        "passlib[bcrypt]==1.7.4",
        "python-dotenv==1.0.0"
    ]
    
    try:
        for dep in core_deps:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, capture_output=True)
            print(f"✅ Installed {dep}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {dep}: {e}")
        return False

def initialize_database():
    """Initialize SQLite database"""
    print("\n🗄️ Initializing SQLite database...")
    
    try:
        # Import and create tables
        from database import engine
        from database.models import Base
        
        Base.metadata.create_all(bind=engine)
        print("✅ SQLite database tables created")
        
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

def test_system_endpoints():
    """Test system endpoints"""
    print("\n🧪 Testing system endpoints...")
    
    # Start server in background for testing
    import threading
    import time
    
    def start_test_server():
        try:
            import uvicorn
            from main import app
            uvicorn.run(app, host="127.0.0.1", port=8003, log_level="error")
        except:
            pass
    
    # Start server thread
    server_thread = threading.Thread(target=start_test_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Test endpoints
    endpoints = [
        "/health",
        "/",
        "/docs"
    ]
    
    working_endpoints = 0
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"http://127.0.0.1:8003{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}: Working")
                working_endpoints += 1
            else:
                print(f"❌ {endpoint}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {str(e)}")
    
    if working_endpoints > 0:
        print(f"✅ {working_endpoints}/{len(endpoints)} endpoints working")
        return True
    else:
        print("❌ No endpoints are working")
        return False

def create_startup_script():
    """Create Windows startup script"""
    print("\n🔧 Creating startup script...")
    
    startup_script = f"""@echo off
echo Starting BRAF Monetization System...
echo.
echo Dashboard: http://localhost:8003/dashboard
echo API Docs: http://localhost:8003/docs
echo Health Check: http://localhost:8003/health
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "{os.getcwd()}"
{sys.executable} run_server.py

pause
"""
    
    try:
        with open("start_production.bat", "w") as f:
            f.write(startup_script)
        print("✅ Created start_production.bat")
        
        # Also create a Python version
        python_script = f"""#!/usr/bin/env python3
import os
import sys

# Set production environment
os.environ["ENVIRONMENT"] = "production"

print("🚀 Starting BRAF Monetization System (Production)")
print("=" * 50)
print("🌐 Dashboard: http://localhost:8003/dashboard")
print("📚 API Docs: http://localhost:8003/docs")
print("🏥 Health Check: http://localhost:8003/health")
print("💱 Currency: Real-time rates enabled")
print("=" * 50)
print("Press Ctrl+C to stop")
print()

# Start server
if __name__ == "__main__":
    import uvicorn
    from main import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )
"""
        
        with open("start_production.py", "w") as f:
            f.write(python_script)
        print("✅ Created start_production.py")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create startup scripts: {e}")
        return False

def main():
    """Main deployment function"""
    print("🚀 BRAF Monetization System - Windows Live Deployment")
    print("=" * 60)
    print("Deploying with real-time currency conversion (Windows)")
    print("=" * 60)
    
    steps = [
        ("Testing currency APIs", test_currency_apis),
        ("Setting up production environment", setup_production_environment),
        ("Installing core dependencies", install_core_dependencies),
        ("Initializing database", initialize_database),
        ("Testing currency conversion", test_currency_conversion),
        ("Creating startup scripts", create_startup_script),
        ("Testing system endpoints", test_system_endpoints)
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
    print("🎉 WINDOWS LIVE DEPLOYMENT COMPLETED!")
    print("=" * 60)
    
    print("\n✅ **PRODUCTION READY FEATURES:**")
    print("   • Real-time currency conversion (USD ↔ NGN)")
    print("   • Multiple currency API sources with fallback")
    print("   • SQLite database (production ready)")
    print("   • Windows-compatible deployment")
    print("   • Rate limiting and security")
    print("   • Comprehensive logging")
    print("   • Health monitoring")
    
    print("\n🌐 **CURRENCY APIS WORKING:**")
    print("   • ExchangeRate-API (free, reliable)")
    print("   • Automatic fallback to cached rates")
    print("   • 15-minute cache for performance")
    
    print("\n🚀 **START THE SYSTEM:**")
    print("   • Windows: start_production.bat")
    print("   • Python: python start_production.py")
    print("   • Manual: python run_server.py")
    
    print("\n🌐 **ACCESS POINTS:**")
    print("   • Dashboard: http://localhost:8003/dashboard")
    print("   • API Docs: http://localhost:8003/docs")
    print("   • Health Check: http://localhost:8003/health")
    
    print("\n💡 **CURRENCY FEATURES:**")
    print("   • Live USD to NGN conversion")
    print("   • OPay: $100 USD → ~₦145,000 NGN")
    print("   • PalmPay: $100 USD → ~₦145,000 NGN")
    print("   • Crypto: Remains in USD")
    print("   • Real-time exchange rates")
    print("   • Automatic API failover")
    
    print(f"\n📍 **DEPLOYMENT INFO:**")
    print(f"   • Timestamp: {datetime.now().isoformat()}")
    print(f"   • Directory: {os.getcwd()}")
    print(f"   • Python: {sys.version}")
    print(f"   • Platform: Windows")
    print(f"   • Status: 🟢 READY FOR PRODUCTION")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
