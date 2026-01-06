#!/usr/bin/env python3
"""
Scraper Status Monitor
Check the status of the last scraping run and provide detailed information.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

STATUS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'scraper_status.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'scraper.log')

class ScraperStatusMonitor:
    """Monitor and report scraper status"""
    
    def __init__(self):
        self.status_file = STATUS_FILE
        self.log_file = LOG_FILE
    
    def load_status(self) -> Optional[Dict]:
        """Load the latest status from file"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"❌ Error loading status: {e}")
        return None
    
    def get_log_tail(self, lines: int = 20) -> list:
        """Get the last N lines from the log file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return f.readlines()[-lines:]
        except Exception as e:
            print(f"❌ Error reading log: {e}")
        return []
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"
    
    def get_status_emoji(self, status: str) -> str:
        """Get emoji for status"""
        status_emojis = {
            'running': '🔄',
            'completed': '✅',
            'failed': '❌',
            'stopped': '🛑',
            'unknown': '❓'
        }
        return status_emojis.get(status, '❓')
    
    def print_detailed_status(self):
        """Print detailed status information"""
        print("🔍 Scraper Status Monitor")
        print("=" * 40)
        
        status_data = self.load_status()
        
        if not status_data:
            print("❌ No status file found. Scraper may not have run yet.")
            print(f"Expected status file: {self.status_file}")
            return
        
        # Basic status info
        status = status_data.get('status', 'unknown')
        timestamp = status_data.get('timestamp', 'unknown')
        
        print(f"Status: {self.get_status_emoji(status)} {status.upper()}")
        print(f"Last Update: {timestamp}")
        
        # Parse timestamp for age calculation
        try:
            last_update = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = datetime.now() - last_update.replace(tzinfo=None)
            print(f"Age: {self.format_duration(age.total_seconds())}")
        except:
            print("Age: Unknown")
        
        # Statistics
        stats = status_data.get('stats', {})
        if stats:
            print(f"\n📊 Statistics:")
            print(f"   Pages Scraped: {stats.get('pages_scraped', 0)}")
            print(f"   Pages Failed: {stats.get('pages_failed', 0)}")
            print(f"   Content Length: {stats.get('total_content_length', 0):,} chars")
            
            if 'duration_seconds' in stats:
                print(f"   Duration: {self.format_duration(stats['duration_seconds'])}")
            
            if 'success_rate' in stats:
                print(f"   Success Rate: {stats['success_rate']:.1f}%")
        
        # Details
        details = status_data.get('details', {})
        if details:
            print(f"\n📋 Details:")
            for key, value in details.items():
                if key == 'traceback':
                    continue  # Skip traceback in summary
                print(f"   {key}: {value}")
        
        # Errors
        errors = stats.get('errors', [])
        if errors:
            print(f"\n⚠️  Errors ({len(errors)}):")
            for i, error in enumerate(errors[-5:], 1):  # Show last 5 errors
                print(f"   {i}. {error}")
        
        # Recent log entries
        print(f"\n📝 Recent Log Entries:")
        log_lines = self.get_log_tail(10)
        if log_lines:
            for line in log_lines:
                print(f"   {line.strip()}")
        else:
            print("   No log entries found")
    
    def print_summary(self):
        """Print brief status summary"""
        status_data = self.load_status()
        
        if not status_data:
            print("❌ No status available")
            return
        
        status = status_data.get('status', 'unknown')
        timestamp = status_data.get('timestamp', 'unknown')
        stats = status_data.get('stats', {})
        
        print(f"{self.get_status_emoji(status)} {status.upper()} | "
              f"Scraped: {stats.get('pages_scraped', 0)} | "
              f"Failed: {stats.get('pages_failed', 0)} | "
              f"Last: {timestamp}")
    
    def check_health(self) -> bool:
        """Check if scraper is healthy"""
        status_data = self.load_status()
        
        if not status_data:
            return False
        
        status = status_data.get('status')
        timestamp = status_data.get('timestamp')
        
        # Check if status is good
        if status not in ['completed', 'running']:
            return False
        
        # Check if last run was recent (within 2 hours)
        try:
            last_update = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = datetime.now() - last_update.replace(tzinfo=None)
            if age > timedelta(hours=2):
                return False
        except:
            return False
        
        return True
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            from database_manager import ScraperDatabaseManager
            manager = ScraperDatabaseManager()
            stats = manager.get_stats()
            
            print(f"\n🗄️  Database Statistics:")
            print(f"   Total Records: {stats.get('total_records', 0)}")
            print(f"   Unique Domains: {stats.get('unique_domains', 0)}")
            print(f"   Database Size: {stats.get('database_size_mb', 0)} MB")
            print(f"   Recent Records: {stats.get('recent_records', 0)}")
            
        except ImportError:
            print("\n🗄️  Database statistics not available")
        except Exception as e:
            print(f"\n❌ Error getting database stats: {e}")

def main():
    """Main entry point"""
    import sys
    
    monitor = ScraperStatusMonitor()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--summary":
            monitor.print_summary()
        elif command == "--health":
            healthy = monitor.check_health()
            print("✅ Healthy" if healthy else "❌ Unhealthy")
            sys.exit(0 if healthy else 1)
        elif command == "--database":
            monitor.get_database_stats()
        elif command == "--help":
            print("Usage:")
            print("  python check_scraper_status.py           # Detailed status")
            print("  python check_scraper_status.py --summary # Brief summary")
            print("  python check_scraper_status.py --health  # Health check")
            print("  python check_scraper_status.py --database # Database stats")
            print("  python check_scraper_status.py --help    # This help")
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        # Default: detailed status
        monitor.print_detailed_status()
        monitor.get_database_stats()

if __name__ == "__main__":
    main()
