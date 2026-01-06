#!/usr/bin/env python3
"""
SQLite Database Manager for Web Scraper
"""
import sqlite3
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'scraper.db')

class ScraperDatabaseManager:
    """Manage the SQLite database for scraped data"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure database and directory exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        if not os.path.exists(self.db_path):
            from automation.ethical_web_scraper import init_database
            init_database()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total records
        cursor.execute('SELECT COUNT(*) FROM scraped_data')
        total_records = cursor.fetchone()[0]
        
        # Unique domains
        cursor.execute('SELECT COUNT(DISTINCT domain) FROM scraped_data')
        unique_domains = cursor.fetchone()[0]
        
        # Records by domain
        cursor.execute('''
            SELECT domain, COUNT(*) as count 
            FROM scraped_data 
            GROUP BY domain 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        top_domains = cursor.fetchall()
        
        # Recent activity (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        cursor.execute('''
            SELECT COUNT(*) FROM scraped_data 
            WHERE scraped_at > ?
        ''', (week_ago,))
        recent_records = cursor.fetchone()[0]
        
        # Database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        conn.close()
        
        return {
            'total_records': total_records,
            'unique_domains': unique_domains,
            'top_domains': top_domains,
            'recent_records': recent_records,
            'database_size_mb': round(db_size / (1024 * 1024), 2),
            'database_path': self.db_path
        }
    
    def search_content(self, query: str, limit: int = 20) -> List[Dict]:
        """Search scraped content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT url, domain, title, content, word_count, scraped_at
            FROM scraped_data 
            WHERE content LIKE ? OR title LIKE ?
            ORDER BY scraped_at DESC 
            LIMIT ?
        ''', (f'%{query}%', f'%{query}%', limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'url': row[0],
                'domain': row[1],
                'title': row[2],
                'content_preview': row[3][:200] + '...' if len(row[3]) > 200 else row[3],
                'word_count': row[4],
                'scraped_at': row[5]
            }
            for row in results
        ]
    
    def get_domain_data(self, domain: str, limit: int = 50) -> List[Dict]:
        """Get all data for a specific domain"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT url, title, content, word_count, scraped_at
            FROM scraped_data 
            WHERE domain = ?
            ORDER BY scraped_at DESC 
            LIMIT ?
        ''', (domain, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'url': row[0],
                'title': row[1],
                'content': row[2],
                'word_count': row[3],
                'scraped_at': row[4]
            }
            for row in results
        ]
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Remove data older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM scraped_data WHERE scraped_at < ?', (cutoff_date,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def export_to_json(self, output_file: str, domain: str = None):
        """Export data to JSON file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if domain:
            cursor.execute('''
                SELECT url, domain, title, content, word_count, scraped_at
                FROM scraped_data 
                WHERE domain = ?
                ORDER BY scraped_at DESC
            ''', (domain,))
        else:
            cursor.execute('''
                SELECT url, domain, title, content, word_count, scraped_at
                FROM scraped_data 
                ORDER BY scraped_at DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        data = [
            {
                'url': row[0],
                'domain': row[1],
                'title': row[2],
                'content': row[3],
                'word_count': row[4],
                'scraped_at': row[5]
            }
            for row in results
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return len(data)

def main():
    """Command line interface"""
    import sys
    
    manager = ScraperDatabaseManager()
    
    if len(sys.argv) < 2:
        print("📊 Scraper Database Manager")
        print("=" * 30)
        
        stats = manager.get_stats()
        print(f"Database: {stats['database_path']}")
        print(f"Size: {stats['database_size_mb']} MB")
        print(f"Total records: {stats['total_records']}")
        print(f"Unique domains: {stats['unique_domains']}")
        print(f"Recent records (7 days): {stats['recent_records']}")
        
        print("\nTop domains:")
        for domain, count in stats['top_domains']:
            print(f"  {domain}: {count} pages")
        
        print("\nUsage:")
        print("  python database_manager.py stats")
        print("  python database_manager.py search 'keyword'")
        print("  python database_manager.py domain example.com")
        print("  python database_manager.py cleanup 30")
        print("  python database_manager.py export output.json")
        return
    
    command = sys.argv[1]
    
    if command == "stats":
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2))
    
    elif command == "search" and len(sys.argv) > 2:
        query = sys.argv[2]
        results = manager.search_content(query)
        print(f"Found {len(results)} results for '{query}':")
        for result in results:
            print(f"  {result['domain']}: {result['title']}")
    
    elif command == "domain" and len(sys.argv) > 2:
        domain = sys.argv[2]
        results = manager.get_domain_data(domain)
        print(f"Found {len(results)} pages for {domain}:")
        for result in results:
            print(f"  {result['scraped_at']}: {result['title']}")
    
    elif command == "cleanup" and len(sys.argv) > 2:
        days = int(sys.argv[2])
        deleted = manager.cleanup_old_data(days)
        print(f"Deleted {deleted} records older than {days} days")
    
    elif command == "export" and len(sys.argv) > 2:
        output_file = sys.argv[2]
        domain = sys.argv[3] if len(sys.argv) > 3 else None
        count = manager.export_to_json(output_file, domain)
        print(f"Exported {count} records to {output_file}")
    
    else:
        print("Invalid command. Use: stats, search, domain, cleanup, or export")

if __name__ == "__main__":
    main()
