#!/usr/bin/env python3
"""
WSGI Entry Point for Production Deployment
Gunicorn WSGI server entry point for the BRAF monetization system
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the Flask application
try:
    # Try importing from main.py first
    from main import app
except ImportError:
    try:
        # Try importing from start_system.py
        from start_system import app
    except ImportError:
        try:
            # Try importing from run_server.py
            from run_server import app
        except ImportError:
            # Create a minimal Flask app as fallback
            from flask import Flask, jsonify
            
            app = Flask(__name__)
            
            @app.route('/')
            def index():
                return jsonify({
                    'status': 'running',
                    'message': 'BRAF Monetization System',
                    'version': '1.0.0'
                })
            
            @app.route('/health')
            def health():
                return jsonify({
                    'status': 'healthy',
                    'timestamp': os.environ.get('TIMESTAMP', 'unknown')
                })

# Configure for production
if __name__ != '__main__':
    # Production configuration
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False
    app.config['TESTING'] = False

if __name__ == '__main__':
    # Development server (not used in production)
    app.run(host='0.0.0.0', port=8080, debug=False)
