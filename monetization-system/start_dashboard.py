#!/usr/bin/env python3
"""
BRAF Dashboard Server
Simple HTTP server to serve the BRAF dashboard
"""
import os
import sys
import http.server
import socketserver
import webbrowser
from pathlib import Path

def start_dashboard(port=8080, open_browser=True):
    """Start the BRAF dashboard server"""
    
    # Change to the monetization-system directory
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    print(f"🚀 Starting BRAF Dashboard Server")
    print(f"📁 Serving from: {dashboard_dir}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}/dashboard/")
    
    # Create a custom handler that serves files with proper MIME types
    class BRAFHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers for local development
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def log_message(self, format, *args):
            # Custom logging format
            print(f"📊 {self.address_string()} - {format % args}")
    
    try:
        with socketserver.TCPServer(("", port), BRAFHandler) as httpd:
            print(f"✅ Server started successfully!")
            print(f"📋 Dashboard URL: http://localhost:{port}/dashboard/")
            print(f"📄 Results API: http://localhost:{port}/data/results.json")
            print(f"🔧 Enhanced API: http://localhost:{port}/data/enhanced_results.json")
            print(f"\n💡 Press Ctrl+C to stop the server")
            
            # Open browser automatically
            if open_browser:
                dashboard_url = f"http://localhost:{port}/dashboard/"
                print(f"🌐 Opening dashboard in browser...")
                webbrowser.open(dashboard_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use")
            print(f"💡 Try a different port: python start_dashboard.py --port 8081")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function with command line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start BRAF Dashboard Server')
    parser.add_argument('--port', '-p', type=int, default=8080, 
                       help='Port to serve on (default: 8080)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    start_dashboard(port=args.port, open_browser=not args.no_browser)

if __name__ == "__main__":
    main()
