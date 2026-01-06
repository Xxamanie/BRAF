#!/usr/bin/env python3
"""
Simple System Startup
Starts the BRAF monetization system without the complex live orchestrator
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_system():
    """Start the BRAF system in simple mode"""
    
    print("🚀 Starting BRAF Monetization System")
    print("=" * 50)
    
    # Load environment
    env_file = Path('.env.production')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Production environment loaded")
    
    # Set default values
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8003))
    
    print(f"🌐 Starting server on {host}:{port}")
    print(f"📊 Dashboard: http://localhost:{port}/dashboard")
    print(f"🔧 API Docs: http://localhost:{port}/docs")
    print(f"🧠 Intelligence: http://localhost:{port}/api/v1/intelligence/status")
    
    print("\n💡 System Features:")
    print("   ✅ Enterprise account management")
    print("   ✅ Browser automation framework")
    print("   ✅ Intelligence system with ML optimization")
    print("   ✅ Real-time currency conversion")
    print("   ✅ OPay/PalmPay integration (demo mode)")
    print("   ✅ Comprehensive API endpoints")
    
    print("\n⚠️  Note: Running in demo mode - no real money transactions")
    print("   Configure live API credentials in .env.production for real operations")
    
    print("\n" + "=" * 50)
    
    try:
        # Import and start the FastAPI app
        from main import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 System shutdown requested")
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        print(f"\n❌ System startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_system()
