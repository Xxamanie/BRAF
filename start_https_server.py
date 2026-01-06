#!/usr/bin/env python3
"""
HTTPS Server for BRAF - Windows SSL
"""
import ssl
import uvicorn
from pathlib import Path

def start_https_server():
    """Start HTTPS server with SSL certificates"""
    
    # Check for SSL files
    cert_file = Path("ssl_cert.pem")
    key_file = Path("ssl_key.pem")
    
    if not cert_file.exists() or not key_file.exists():
        print("SSL certificates not found!")
        print("Run: python setup_windows_ssl.py")
        return
    
    print("Starting HTTPS server...")
    print("HTTPS URL: https://127.0.0.1:8443")
    print("Dashboard: https://127.0.0.1:8443/dashboard")
    print("Press Ctrl+C to stop")
    
    # Start HTTPS server
    uvicorn.run(
        "monetization-system.main:app",
        host="127.0.0.1",
        port=8443,
        ssl_keyfile="ssl_key.pem",
        ssl_certfile="ssl_cert.pem",
        reload=False
    )

if __name__ == "__main__":
    start_https_server()
