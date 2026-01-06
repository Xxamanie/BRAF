#!/usr/bin/env python3
"""
Synchronous Playwright Web Scraper
Uses sync_playwright for simpler, more reliable scraping
"""
import os
import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import logging

# Try to import Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not available. Install with: pip install playwright")

logger = logging.getLogger(__name__)

@dataclass
class ScrapedResult:
    """Structure for scraped data"""
    url: str
    domain: str
    title: str
    content: str
    word_count: int
    scraped_at: datetime
    data_hash: str
    success: bool
    error: Optional[str] = None

class SyncPlaywrightScraper:
    """Synchronous Playwright-based web scraper with SQLite storage"""
    
    def __init__(self, db_path: str = None, headless: bool = True):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), 'data', 'scraper.db')
        self.headless = headless
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists and get its schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scraped_data'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check if success column exists
            cursor.execute("PRAGMA table_info(scraped_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'success' not in columns:
                # Add missing columns
                cursor.execute('ALTER TABLE scraped_data ADD COLUMN success BOOLEAN DEFAULT 1')
                cursor.execute('ALTER TABLE scraped_data ADD COLUMN error TEXT')
                logger.info("✅ Added success and error columns to existing table")
        else:
            # Create new table with all columns
            cursor.execute('''
                CREATE TABLE scraped_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    domain TEXT,
                    title TEXT,
                    content TEXT,
                    word_count INTEGER,
                    data_hash TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 1,
                    error TEXT,
                    UNIQUE(url, data_hash)
                )
            ''')
            logger.info("✅ Created new scraped_data table")
        
        # Create indexes for performance (with IF NOT EXISTS to avoid errors)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON scraped_data(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scraped_at ON scraped_data(scraped_at)')
        
        # Only create success index if column exists
        cursor.execute("PRAGMA table_info(scraped_data)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'success' in columns:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_success ON scraped_data(success)')
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized at: {self.db_path}")
    
    def run_single_scrape(self, target: Dict) -> ScrapedResult:
        """
        Scrape a single URL using synchronous Playwright
        
        Args:
            target: Dictionary with 'url' key
            
        Returns:
            ScrapedResult with scraped data or error info
        """
        if not PLAYWRIGHT_AVAILABLE:
            return ScrapedResult(
                url=target["url"],
                domain=urlparse(target["url"]).netloc,
                title="",
                content="",
                word_count=0,
                scraped_at=datetime.now(),
                data_hash="",
                success=False,
                error="Playwright not available"
            )
        
        try:
            with sync_playwright() as p:
                # Launch browser with optimized settings
                browser = p.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-web-security",
                        "--disable-features=VizDisplayCompositor"
                    ]
                )
                
                # Create new page with realistic settings
                page = browser.new_page()
                page.set_viewport_size({"width": 1920, "height": 1080})
                page.set_extra_http_headers({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                })
                
                # Navigate to target URL
                logger.info(f"📥 Navigating to: {target['url']}")
                page.goto(target["url"], timeout=60000, wait_until="domcontentloaded")
                
                # Wait for page to stabilize
                page.wait_for_timeout(3000)
                
                # Extract data
                title = page.title()
                content = page.inner_text("body")[:4000]  # Limit content to 4000 chars
                
                # Clean up content
                content = content.strip()
                word_count = len(content.split()) if content else 0
                
                # Generate hash for duplicate detection
                data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                
                # Close browser
                browser.close()
                
                result = ScrapedResult(
                    url=target["url"],
                    domain=urlparse(target["url"]).netloc,
                    title=title,
                    content=content,
                    word_count=word_count,
                    scraped_at=datetime.now(),
                    data_hash=data_hash,
                    success=True
                )
                
                logger.info(f"✅ Successfully scraped: {title[:50]}...")
                return result
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Failed to scrape {target['url']}: {error_msg}")
            
            return ScrapedResult(
                url=target["url"],
                domain=urlparse(target["url"]).netloc,
                title="",
                content="",
                word_count=0,
                scraped_at=datetime.now(),
                data_hash="",
                success=False,
                error=error_msg
            )
    
    def save_result(self, result: ScrapedResult) -> bool:
        """Save scraped result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if success and error columns exist
            cursor.execute("PRAGMA table_info(scraped_data)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'success' in columns and 'error' in columns:
                # Use full schema with success/error columns
                cursor.execute('''
                    INSERT OR REPLACE INTO scraped_data 
                    (url, domain, title, content, word_count, data_hash, scraped_at, success, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.url,
                    result.domain,
                    result.title,
                    result.content,
                    result.word_count,
                    result.data_hash,
                    result.scraped_at,
                    result.success,
                    result.error
                ))
            else:
                # Use legacy schema without success/error columns
                cursor.execute('''
                    INSERT OR REPLACE INTO scraped_data 
                    (url, domain, title, content, word_count, data_hash, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.url,
                    result.domain,
                    result.title,
                    result.content,
                    result.word_count,
                    result.data_hash,
                    result.scraped_at
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save result to database: {e}")
            return False
    
    def scrape_urls(self, urls: List[str], delay_seconds: int = 2) -> Dict:
        """
        Scrape multiple URLs with delay between requests
        
        Args:
            urls: List of URLs to scrape
            delay_seconds: Delay between requests
            
        Returns:
            Dictionary with statistics and results
        """
        results = []
        stats = {
            'total_urls': len(urls),
            'successful': 0,
            'failed': 0,
            'total_content_length': 0,
            'total_words': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        for i, url in enumerate(urls):
            # Create target dictionary
            target = {"url": url}
            
            # Scrape the URL
            result = self.run_single_scrape(target)
            results.append(result)
            
            # Save to database
            if self.save_result(result):
                logger.info(f"💾 Saved result for {url}")
            else:
                logger.warning(f"⚠️  Failed to save result for {url}")
            
            # Update statistics
            if result.success:
                stats['successful'] += 1
                stats['total_content_length'] += len(result.content)
                stats['total_words'] += result.word_count
            else:
                stats['failed'] += 1
                stats['errors'].append(f"{url}: {result.error}")
            
            # Delay between requests (except for last URL)
            if i < len(urls) - 1 and delay_seconds > 0:
                import time
                logger.info(f"⏱️  Waiting {delay_seconds} seconds...")
                time.sleep(delay_seconds)
        
        # Finalize statistics
        stats['end_time'] = datetime.now()
        stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()
        stats['success_rate'] = (stats['successful'] / stats['total_urls']) * 100 if stats['total_urls'] > 0 else 0
        
        return {
            'stats': stats,
            'results': results
        }
    
    def get_recent_data(self, limit: int = 10) -> List[Dict]:
        """Get recent scraped data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT url, domain, title, content, word_count, scraped_at, success, error
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
                    'scraped_at': row[5],
                    'success': bool(row[6]),
                    'error': row[7]
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve data: {e}")
            return []

def test_scraper():
    """Test the synchronous scraper"""
    print("🧪 Testing Synchronous Playwright Scraper")
    print("=" * 40)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright not available for testing")
        return
    
    # Initialize scraper
    scraper = SyncPlaywrightScraper(headless=True)
    
    # Test URLs
    test_urls = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://jsonplaceholder.typicode.com/posts/1"
    ]
    
    # Scrape URLs
    print(f"📋 Scraping {len(test_urls)} URLs...")
    results = scraper.scrape_urls(test_urls, delay_seconds=1)
    
    # Display results
    stats = results['stats']
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {stats['successful']}")
    print(f"   ❌ Failed: {stats['failed']}")
    print(f"   📝 Total words: {stats['total_words']}")
    print(f"   ⏱️  Duration: {stats['duration_seconds']:.1f} seconds")
    print(f"   📈 Success rate: {stats['success_rate']:.1f}%")
    
    if stats['errors']:
        print(f"\n⚠️  Errors:")
        for error in stats['errors']:
            print(f"   - {error}")
    
    # Show recent data
    print(f"\n📋 Recent scraped data:")
    recent_data = scraper.get_recent_data(5)
    for item in recent_data:
        status = "✅" if item['success'] else "❌"
        print(f"   {status} {item['domain']}: {item['title'][:40]}...")

if __name__ == "__main__":
    test_scraper()
