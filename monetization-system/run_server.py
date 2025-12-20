#!/usr/bin/env python3
"""
Simple server runner for BRAF Monetization System
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())
    
    print("🚀 Starting BRAF Monetization Server...")
    print("📍 Server will be available at: http://127.0.0.1:8003")
    print("📚 API Documentation: http://127.0.0.1:8003/docs")
    print("🏠 Dashboard: http://127.0.0.1:8003/dashboard")
    print("🔐 Login: http://127.0.0.1:8003/login")
    print("📝 Register: http://127.0.0.1:8003/register")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8003,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")