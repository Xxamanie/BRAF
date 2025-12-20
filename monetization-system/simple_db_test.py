#!/usr/bin/env python3
"""
Simple SQLite database test for scraper
"""
import sqlite3
import os
from datetime import datetime
import json

# Database configuration
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'scraper.db')

def init_database():
    """Initialize SQLite database for scraped data"""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraped_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            domain TEXT,
            title TEXT,
            content TEXT,
            html TEXT,
            screenshots TEXT,  -- JSON string of base64 screenshots
            links TEXT,        -- JSON string of links array
            metadata TEXT,     -- JSON string of metadata
            word_count INTEGER,
            data_hash TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(url, data_hash)
        )
    ''')
    
    # Create indexes for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON scraped_data(domain)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scraped_at ON scraped_data(scraped_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_hash ON scraped_data(data_hash)')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {DB_PATH}")

def test_insert_data():
    """Test inserting sample data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sample data
    sample_data = {
        'url': 'https://example.com/test-page',
        'domain': 'example.com',
        'title': 'Test Page for SQLite Integration',
        'content': 'This is sample content to test the SQLite database integration with our web scraper. ' * 5,
        'html': '<html><head><title>Test</title></head><body><p>Sample HTML content</p></body></html>',
        'screenshots': json.dumps({'full_page': 'base64_screenshot_data_here'}),
        'links': json.dumps(['https://example.com/link1', 'https://example.com/link2']),
        'metadata': json.dumps({'description': 'Test page', 'keywords': 'test,sqlite'}),
        'word_count': len('This is sample content to test the SQLite database integration with our web scraper. '.split()) * 5,
        'data_hash': 'test_hash_12345',
        'scraped_at': datetime.now()
    }
    
    cursor.execute('''
        INSERT OR REPLACE INTO scraped_data 
        (url, domain, title, content, html, screenshots, links, metadata, 
         word_count, data_hash, scraped_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        sample_data['url'],
        sample_data['domain'],
        sample_data['title'],
        sample_data['content'],
        sample_data['html'],
        sample_data['screenshots'],
        sample_data['links'],
        sample_data['metadata'],
        sample_data['word_count'],
        sample_data['data_hash'],
        sample_data['scraped_at']
    ))
    
    conn.commit()
    conn.close()
    print("✅ Sample data inserted successfully")

def test_query_data():
    """Test querying data from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT url, domain, title, word_count, scraped_at 
        FROM scraped_data 
        ORDER BY scraped_at DESC 
        LIMIT 5
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    print(f"\n📊 Found {len(results)} records:")
    print("-" * 60)
    for row in results:
        print(f"Domain: {row[1]}")
        print(f"Title: {row[2]}")
        print(f"Words: {row[3]}")
        print(f"Date: {row[4]}")
        print(f"URL: {row[0]}")
        print("-" * 60)

def get_database_stats():
    """Get basic database statistics"""
    if not os.path.exists(DB_PATH):
        print("❌ Database doesn't exist yet")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total records
    cursor.execute('SELECT COUNT(*) FROM scraped_data')
    total = cursor.fetchone()[0]
    
    # Unique domains
    cursor.execute('SELECT COUNT(DISTINCT domain) FROM scraped_data')
    domains = cursor.fetchone()[0]
    
    # Database size
    db_size = os.path.getsize(DB_PATH)
    
    conn.close()
    
    print(f"\n📈 Database Statistics:")
    print(f"   Total records: {total}")
    print(f"   Unique domains: {domains}")
    print(f"   Database size: {db_size / 1024:.1f} KB")
    print(f"   Database path: {DB_PATH}")

if __name__ == "__main__":
    print("🧪 Testing SQLite Database Integration")
    print("=" * 40)
    
    # Test 1: Initialize database
    print("1. Initializing database...")
    init_database()
    
    # Test 2: Insert sample data
    print("\n2. Inserting sample data...")
    test_insert_data()
    
    # Test 3: Query data
    print("\n3. Querying data...")
    test_query_data()
    
    # Test 4: Get stats
    print("\n4. Database statistics...")
    get_database_stats()
    
    print("\n✅ All tests completed successfully!")
    print(f"💡 Database created at: {DB_PATH}")