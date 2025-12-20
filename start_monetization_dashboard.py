#!/usr/bin/env python3
"""
BRAF Monetization Dashboard Server
HTTP server for the monetization dashboard with earnings data
"""
import os
import sys
import json
import http.server
import socketserver
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3

def generate_monetization_data():
    """Generate monetization data from earnings database and automation results"""
    dashboard_dir = Path(__file__).parent
    
    # Initialize monetization data structure
    monetization_data = {
        'total_earnings': 0.0,
        'pending_earnings': 0.0,
        'withdrawn_amount': 0.0,
        'platforms': [],
        'recent_activity': [],
        'performance': {
            'success_rate': 0.0,
            'total_tasks': 0,
            'avg_execution_time': 0.0
        }
    }
    
    # Try to load from earnings database
    earnings_db = dashboard_dir / 'earnings.db'
    if earnings_db.exists():
        try:
            conn = sqlite3.connect(str(earnings_db))
            cursor = conn.cursor()
            
            # Get total earnings
            cursor.execute('SELECT SUM(amount) FROM earnings WHERE status = "completed"')
            result = cursor.fetchone()
            monetization_data['total_earnings'] = result[0] if result[0] else 0.0
            
            # Get pending earnings
            cursor.execute('SELECT SUM(amount) FROM earnings WHERE status = "pending"')
            result = cursor.fetchone()
            monetization_data['pending_earnings'] = result[0] if result[0] else 0.0
            
            # Get withdrawn amount
            cursor.execute('SELECT SUM(amount) FROM withdrawals WHERE status = "completed"')
            result = cursor.fetchone()
            monetization_data['withdrawn_amount'] = result[0] if result[0] else 0.0
            
            # Get platform statistics
            cursor.execute('''
                SELECT name, total_earned, status, last_updated 
                FROM platforms 
                ORDER BY total_earned DESC
            ''')
            
            for row in cursor.fetchall():
                monetization_data['platforms'].append({
                    'name': row[0],
                    'total_earned': row[1],
                    'status': row[2],
                    'last_updated': row[3],
                    'tasks_completed': 0  # Will be updated below
                })
            
            # Get recent activity
            cursor.execute('''
                SELECT platform, task_type, amount, timestamp, status 
                FROM earnings 
                ORDER BY timestamp DESC 
                LIMIT 20
            ''')
            
            for row in cursor.fetchall():
                activity_type = 'earning' if row[4] == 'completed' else 'pending'
                monetization_data['recent_activity'].append({
                    'type': activity_type,
                    'title': f'{row[1].replace("_", " ").title()} Completed',
                    'details': f'{row[0]} platform',
                    'amount': row[2] if row[4] == 'completed' else 0,
                    'timestamp': row[3]
                })
            
            # Get performance stats
            cursor.execute('SELECT COUNT(*) FROM earnings')
            total_tasks = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT COUNT(*) FROM earnings WHERE status = "completed"')
            completed_tasks = cursor.fetchone()[0] or 0
            
            monetization_data['performance'] = {
                'success_rate': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                'total_tasks': total_tasks,
                'avg_execution_time': 2.5  # Simulated average
            }
            
            conn.close()
            print(f"✅ Loaded earnings data from database")
            
        except Exception as e:
            print(f"⚠️  Could not load from earnings database: {e}")
    
    # Try to load from automation results
    automation_results = dashboard_dir / 'data' / 'automation_results.json'
    if automation_results.exists():
        try:
            with open(automation_results, 'r') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            if results:
                # Add simulated earnings from automation results
                successful_tasks = [r for r in results if r.get('result', {}).get('success', False)]
                automation_earnings = len(successful_tasks) * 0.15  # $0.15 per successful automation
                
                monetization_data['total_earnings'] += automation_earnings
                
                # Add automation platform
                monetization_data['platforms'].append({
                    'name': 'Automation Workers',
                    'total_earned': automation_earnings,
                    'status': 'active',
                    'last_updated': datetime.now().isoformat(),
                    'tasks_completed': len(successful_tasks)
                })
                
                # Add automation activity
                for result in results[-5:]:  # Last 5 results
                    success = result.get('result', {}).get('success', False)
                    monetization_data['recent_activity'].append({
                        'type': 'earning' if success else 'error',
                        'title': 'Automation Task',
                        'details': f"Task ID: {result.get('task_id', 'unknown')}",
                        'amount': 0.15 if success else 0,
                        'timestamp': result.get('timestamp', datetime.now().isoformat())
                    })
                
                print(f"✅ Added automation earnings: ${automation_earnings:.2f}")
                
        except Exception as e:
            print(f"⚠️  Could not load automation results: {e}")
    
    # Try to load from basic scraping results
    scraping_results = dashboard_dir / 'data' / 'results.json'
    if scraping_results.exists():
        try:
            with open(scraping_results, 'r') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            if results:
                # Add simulated earnings from scraping results
                successful_tasks = [r for r in results if r.get('success', False)]
                scraping_earnings = len(successful_tasks) * 0.10  # $0.10 per successful scrape
                
                monetization_data['total_earnings'] += scraping_earnings
                
                # Add scraping platform
                monetization_data['platforms'].append({
                    'name': 'Scraping Workers',
                    'total_earned': scraping_earnings,
                    'status': 'active',
                    'last_updated': datetime.now().isoformat(),
                    'tasks_completed': len(successful_tasks)
                })
                
                # Add scraping activity
                for result in results[-5:]:  # Last 5 results
                    success = result.get('success', False)
                    monetization_data['recent_activity'].append({
                        'type': 'earning' if success else 'error',
                        'title': 'Scraping Task',
                        'details': f"URL: {result.get('url', 'unknown')[:50]}...",
                        'amount': 0.10 if success else 0,
                        'timestamp': datetime.now().isoformat()
                    })
                
                print(f"✅ Added scraping earnings: ${scraping_earnings:.2f}")
                
        except Exception as e:
            print(f"⚠️  Could not load scraping results: {e}")
    
    # If no data found, create sample data
    if monetization_data['total_earnings'] == 0:
        print("📊 Generating sample monetization data...")
        
        # Sample earnings data
        monetization_data['total_earnings'] = 47.85
        monetization_data['pending_earnings'] = 12.30
        monetization_data['withdrawn_amount'] = 25.00
        
        # Sample platforms
        monetization_data['platforms'] = [
            {
                'name': 'Web Scraping Workers',
                'total_earned': 28.50,
                'status': 'active',
                'last_updated': datetime.now().isoformat(),
                'tasks_completed': 285
            },
            {
                'name': 'Browser Automation',
                'total_earned': 19.35,
                'status': 'active',
                'last_updated': (datetime.now() - timedelta(hours=2)).isoformat(),
                'tasks_completed': 129
            }
        ]
        
        # Sample activity
        monetization_data['recent_activity'] = [
            {
                'type': 'earning',
                'title': 'Survey Completion',
                'details': 'Automated survey on platform XYZ',
                'amount': 2.50,
                'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()
            },
            {
                'type': 'earning',
                'title': 'Data Collection',
                'details': 'Scraped product information',
                'amount': 1.75,
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat()
            },
            {
                'type': 'withdrawal',
                'title': 'PayPal Withdrawal',
                'details': 'Withdrawal to user@example.com',
                'amount': 0,
                'timestamp': (datetime.now() - timedelta(hours=3)).isoformat()
            }
        ]
        
        # Sample performance
        monetization_data['performance'] = {
            'success_rate': 94.2,
            'total_tasks': 414,
            'avg_execution_time': 3.7
        }
    
    return monetization_data

def start_dashboard(port=8082, open_browser=True):
    """Start the monetization dashboard server"""
    
    # Change to the BRAF directory
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    print(f"💰 Starting BRAF Monetization Dashboard Server")
    print(f"📁 Serving from: {dashboard_dir}")
    print(f"🌐 Port: {port}")
    print(f"🔗 URL: http://localhost:{port}/dashboard/")
    
    # Generate monetization data
    print(f"📊 Generating monetization data...")
    monetization_data = generate_monetization_data()
    
    # Save monetization data
    data_dir = dashboard_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    full_data = {
        'timestamp': datetime.now().isoformat(),
        'monetization_data': monetization_data
    }
    
    with open(data_dir / 'monetization_data.json', 'w') as f:
        json.dump(full_data, f, indent=2)
    
    print(f"💾 Monetization data saved")
    print(f"💰 Total earnings: ${monetization_data['total_earnings']:.2f}")
    print(f"⏳ Pending: ${monetization_data['pending_earnings']:.2f}")
    print(f"💸 Withdrawn: ${monetization_data['withdrawn_amount']:.2f}")
    print(f"🏢 Active platforms: {len(monetization_data['platforms'])}")
    
    # Create a custom handler
    class MonetizationHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # Add CORS headers
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def log_message(self, format, *args):
            # Custom logging format
            print(f"💰 {self.address_string()} - {format % args}")
    
    try:
        with socketserver.TCPServer(("", port), MonetizationHandler) as httpd:
            print(f"✅ Monetization dashboard started successfully!")
            print(f"💰 Dashboard URL: http://localhost:{port}/dashboard/")
            print(f"📊 Data API: http://localhost:{port}/data/monetization_data.json")
            print(f"\n💡 Press Ctrl+C to stop the server")
            
            # Open browser automatically
            if open_browser:
                dashboard_url = f"http://localhost:{port}/dashboard/"
                print(f"🌐 Opening monetization dashboard in browser...")
                webbrowser.open(dashboard_url)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monetization dashboard stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use")
            print(f"💡 Try a different port: python start_monetization_dashboard.py --port 8083")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function with command line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start BRAF Monetization Dashboard Server')
    parser.add_argument('--port', '-p', type=int, default=8082, 
                       help='Port to serve on (default: 8082)')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    start_dashboard(port=args.port, open_browser=not args.no_browser)

if __name__ == "__main__":
    main()