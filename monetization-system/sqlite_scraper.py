#!/usr/bin/env python3
"""
SQLite-based Web Scraper - Clean implementation
"""
import sqlite3
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

# Database configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'scraper.db')

@dataclass
class ScrapedData:
    """Structure for scraped data"""
    url: str
    domain: str
    title: str
    content: str
    word_count: int
    scraped_at: datetime
    data_hash: str

class SQLiteWebScraper:
    """Web scraper with SQLite storage"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraped_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                domain TEXT,
                title TEXT,
                content TEXT,
                word_count INTEGER,
                data_hash TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, data_hash)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON scraped_data(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scraped_at ON scraped_data(scraped_at)')
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at: {self.db_path}")
    
    def save_data(self, scraped_data: ScrapedData) -> bool:
        """Save scraped data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO scraped_data 
                (url, domain, title, content, word_count, data_hash, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                scraped_data.url,
                scraped_data.domain,
                scraped_data.title,
                scraped_data.content,
                scraped_data.word_count,
                scraped_data.data_hash,
                scraped_data.scraped_at
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Failed to save data: {e}")
            return False
    
    def get_data(self, domain: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if domain:
                cursor.execute('''
                    SELECT url, domain, title, content, word_count, scraped_at 
                    FROM scraped_data 
                    WHERE domain = ? 
                    ORDER BY scraped_at DESC 
                    LIMIT ?
                ''', (domain, limit))
            else:
                cursor.execute('''
                    SELECT url, domain, title, content, word_count, scraped_at 
                    FROM scraped_data 
                    ORDER BY scraped_at DESC 
                    LIMIT ?
                ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'url': row[0],
                    'domain': row[1],
                    'title': row[2],
                    'content': row[3][:200] + '...' if len(row[3]) > 200 else row[3],
                    'word_count': row[4],
                    'scraped_at': row[5]
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"❌ Failed to retrieve data: {e}")
            return []
    
    def search_content(self, query: str, limit: int = 20) -> List[Dict]:
        """Search content in database"""
        try:
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
                    'content': row[3][:200] + '...' if len(row[3]) > 200 else row[3],
                    'word_count': row[4],
                    'scraped_at': row[5]
                }
                for row in results
            ]
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM scraped_data')
            total_records = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT domain) FROM scraped_data')
            unique_domains = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(word_count) FROM scraped_data')
            total_words = cursor.fetchone()[0] or 0
            
            conn.close()
            
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_records': total_records,
                'unique_domains': unique_domains,
                'total_words': total_words,
                'database_size_kb': round(db_size / 1024, 2),
                'database_path': self.db_path
            }
            
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {}
    
    def add_sample_data(self):
        """Add sample data for testing"""
        sample_pages = [
            {
                'url': 'https://example.com/article1',
                'domain': 'example.com',
                'title': 'Introduction to SQLite Database',
                'content': 'SQLite is a lightweight, serverless database engine that is perfect for small to medium applications. It stores data in a single file and requires no setup or administration. This makes it ideal for web scraping projects where you need to store collected data efficiently.'
            },
            {
                'url': 'https://news.example.com/tech-news',
                'domain': 'news.example.com',
                'title': 'Latest Technology News',
                'content': 'The technology industry continues to evolve rapidly with new innovations in artificial intelligence, machine learning, and web development. Companies are investing heavily in automation and data collection technologies.'
            },
            {
                'url': 'https://blog.example.com/web-scraping',
                'domain': 'blog.example.com',
                'title': 'Web Scraping Best Practices',
                'content': 'When building web scrapers, it is important to follow ethical guidelines and respect website terms of service. Always implement rate limiting, handle errors gracefully, and store data efficiently using databases like SQLite.'
            }
        ]
        
        for page in sample_pages:
            content = page['content']
            scraped_data = ScrapedData(
                url=page['url'],
                domain=page['domain'],
                title=page['title'],
                content=content,
                word_count=len(content.split()),
                scraped_at=datetime.now(),
                data_hash=hashlib.sha256(content.encode()).hexdigest()[:16]
            )
            
            if self.save_data(scraped_data):
                print(f"✅ Added: {page['title']}")
            else:
                print(f"❌ Failed to add: {page['title']}")

def demo():
    """Demonstration of SQLite scraper functionality"""
    print("🚀 SQLite Web Scraper Demo")
    print("=" * 30)
    
    # Initialize scraper
    scraper = SQLiteWebScraper()
    
    # Add sample data
    print("\n1. Adding sample data...")
    scraper.add_sample_data()
    
    # Show statistics
    print("\n2. Database statistics:")
    stats = scraper.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Show all data
    print("\n3. All scraped data:")
    data = scraper.get_data(limit=10)
    for item in data:
        print(f"   🌐 {item['domain']}")
        print(f"      Title: {item['title']}")
        print(f"      Words: {item['word_count']}")
        print(f"      Date: {item['scraped_at']}")
        print()
    
    # Search functionality
    print("4. Search results for 'SQLite':")
    results = scraper.search_content('SQLite')
    for result in results:
        print(f"   📄 {result['title']}")
        print(f"      {result['content']}")
        print()
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    demo()