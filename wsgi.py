#!/usr/bin/env python3
"""
WSGI entry point for BRAF application
"""

import os
import sys

# Add the monetization-system directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'monetization-system'))

try:
    # Try to import from monetization-system
    from main import app
except ImportError:
    try:
        # Fallback to maxelpay webhook server
        from MAXELPAY_webhook_server import app
    except ImportError:
        # Last resort: create a simple Flask app
        from flask import Flask, jsonify
        
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            return jsonify({"status": "healthy", "service": "BRAF"})
        
        @app.route('/')
        def root():
            return jsonify({
                "service": "BRAF - Browser Automation Framework",
                "status": "running",
                "endpoints": {
                    "health": "/health"
                }
            })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
