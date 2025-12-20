#!/usr/bin/env python3
"""
Startup script for Maxel webhook server with configuration
"""

import os
import sys
from maxel_webhook_server import app, logger

def setup_environment():
    """Setup environment variables and configuration"""
    
    # Default configuration
    config = {
        "HOST": "0.0.0.0",
        "PORT": "8080",
        "DEBUG": "false",
        "MAXEL_SECRET": "your_secure_maxel_secret_here"
    }
    
    # Load from environment or use defaults
    for key, default_value in config.items():
        if key not in os.environ:
            os.environ[key] = default_value
            logger.info(f"Using default {key}: {default_value}")
        else:
            logger.info(f"Using environment {key}: {os.environ[key]}")
    
    # Validate critical settings
    if os.environ["MAXEL_SECRET"] == "your_secure_maxel_secret_here":
        logger.warning("⚠️  Using default MAXEL_SECRET! Set MAXEL_SECRET environment variable for production!")
    
    return config

def print_startup_info():
    """Print startup information"""
    print("=" * 60)
    print("🚀 MAXEL WEBHOOK SERVER")
    print("=" * 60)
    print(f"Host: {os.environ.get('HOST', '0.0.0.0')}")
    print(f"Port: {os.environ.get('PORT', '8080')}")
    print(f"Debug: {os.environ.get('DEBUG', 'false')}")
    print(f"Secret configured: {'✅' if os.environ.get('MAXEL_SECRET') else '❌'}")
    print("=" * 60)
    print("Endpoints:")
    print("  GET  /         - Server info")
    print("  GET  /health   - Health check")
    print("  POST /webhook  - Maxel webhook handler")
    print("=" * 60)
    print("Environment Variables:")
    print("  MAXEL_SECRET - Authentication secret (required)")
    print("  HOST         - Server host (default: 0.0.0.0)")
    print("  PORT         - Server port (default: 8080)")
    print("  DEBUG        - Debug mode (default: false)")
    print("=" * 60)

def main():
    """Main startup function"""
    try:
        # Setup environment
        setup_environment()
        
        # Print startup info
        print_startup_info()
        
        # Get configuration
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", 8080))
        debug = os.environ.get("DEBUG", "false").lower() == "true"
        
        # Start server
        logger.info("Starting Maxel webhook server...")
        app.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()