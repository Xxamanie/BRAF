"""
ACTUAL WEB SCRAPER - Collects real data from websites
"""
import asyncio
from playwright.async_api import async_playwright
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
import hashlib
import sqlite3
import os

@dataclass
class ScrapedPage:
    """Real scraped data structure"""
    url: str
    domain: str
    title: str
    content: str
    html: str
    screenshots: Dict[str, str]  # Base64 encoded screenshots
    links: List[str]
    metadata: Dict
    collected_at: datetime
    data_hash: str

# SQLite Database Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'scraper.db')

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
    
    # Create index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON scraped_data(domain)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scraped_at ON scraped_data(scraped_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_hash ON scraped_data(data_hash)')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at: {DB_PATH}")

def save_scraped_data(page_data: ScrapedPage) -> bool:
    """Save scraped page data to SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert complex data to JSON strings
        screenshots_json = json.dumps({k: v.hex() if isinstance(v, bytes) else v 
                                     for k, v in page_data.screenshots.items()})
        links_json = json.dumps(page_data.links)
        metadata_json = json.dumps(page_data.metadata)
        
        # Calculate word count
        word_count = len(page_data.content.split())
        
        cursor.execute('''
            INSERT OR REPLACE INTO scraped_data 
            (url, domain, title, content, html, screenshots, links, metadata, 
             word_count, data_hash, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            page_data.url,
            page_data.domain,
            page_data.title,
            page_data.content,
            page_data.html,
            screenshots_json,
            links_json,
            metadata_json,
            word_count,
            page_data.data_hash,
            page_data.collected_at
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Failed to save data to database: {e}")
        return False

def get_scraped_data(domain: str = None, limit: int = 100) -> List[Dict]:
    """Retrieve scraped data from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
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
                'content': row[3][:500] + '...' if len(row[3]) > 500 else row[3],
                'word_count': row[4],
                'scraped_at': row[5]
            }
            for row in results
        ]
        
    except Exception as e:
        print(f"❌ Failed to retrieve data: {e}")
        return []

class RealWebScraper:
    """ACTUAL web scraper using Playwright (real browser automation)"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.browser = None
        self.context = None
        # Initialize database on first use
        init_database()
        
    async def __aenter__(self):
        """Initialize browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def scrape_page(self, url: str, scroll: bool = False) -> ScrapedPage:
        """ACTUALLY scrape a webpage"""
        page = await self.context.new_page()
        
        try:
            # Navigate to URL
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Take screenshot
            screenshot = await page.screenshot(full_page=True, type='jpeg', quality=80)
            
            # Scroll if needed (for infinite scroll pages)
            if scroll:
                await self._auto_scroll(page)
            
            # Extract data
            title = await page.title()
            html = await page.content()
            
            # Extract content - multiple strategies
            content = await self._extract_content(page)
            
            # Extract links
            links = await self._extract_links(page, url)
            
            # Extract metadata
            metadata = await self._extract_metadata(page)
            
            # Take additional screenshots of specific elements
            element_screenshots = await self._capture_elements(page)
            
            # Generate data hash
            data_hash = hashlib.sha256(content.encode()).hexdigest()
            
            return ScrapedPage(
                url=url,
                domain=urlparse(url).netloc,
                title=title,
                content=content,
                html=html[:1000000],  # Limit HTML storage
                screenshots={
                    'full_page': screenshot,
                    **element_screenshots
                },
                links=links,
                metadata=metadata,
                collected_at=datetime.now(),
                data_hash=data_hash
            )
            
        finally:
            await page.close()
    
    async def _auto_scroll(self, page):
        """Auto-scroll for infinite scroll pages"""
        last_height = await page.evaluate('document.body.scrollHeight')
        
        for _ in range(10):  # Max 10 scrolls
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await asyncio.sleep(2)  # Wait for content to load
            
            new_height = await page.evaluate('document.body.scrollHeight')
            if new_height == last_height:
                break
            last_height = new_height
    
    async def _extract_content(self, page) -> str:
        """Extract main content from page"""
        # Try multiple content extraction strategies
        
        # Strategy 1: Article tag (most reliable for news)
        article_content = await page.evaluate('''
            () => {
                const article = document.querySelector('article');
                if (article) return article.innerText;
                
                // Strategy 2: Common content classes
                const selectors = [
                    '.post-content', '.entry-content', '.article-body',
                    '.story-content', '.content', '.main-content'
                ];
                
                for (const selector of selectors) {
                    const element = document.querySelector(selector);
                    if (element) return element.innerText;
                }
                
                // Strategy 3: Largest text block
                const paragraphs = Array.from(document.querySelectorAll('p'));
                let maxLength = 0;
                let mainContent = '';
                
                for (const p of paragraphs) {
                    const text = p.innerText.trim();
                    if (text.length > maxLength) {
                        maxLength = text.length;
                        mainContent = text;
                    }
                }
                
                return mainContent || document.body.innerText;
            }
        ''')
        
        # Clean the content
        cleaned = re.sub(r'\s+', ' ', article_content).strip()
        return cleaned[:50000]  # Limit content length
    
    async def _extract_links(self, page, base_url: str) -> List[str]:
        """Extract and normalize links"""
        links = await page.evaluate('''
            () => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => a.href)
                    .filter(href => 
                        href && 
                        !href.startsWith('#') && 
                        !href.startsWith('javascript:') &&
                        !href.startsWith('mailto:') &&
                        !href.startsWith('tel:')
                    );
            }
        ''')
        
        # Normalize and deduplicate
        normalized_links = []
        seen = set()
        
        for link in links:
            try:
                # Resolve relative URLs
                absolute_url = urljoin(base_url, link)
                parsed = urlparse(absolute_url)
                
                if parsed.scheme in ('http', 'https'):
                    # Remove fragments and query params for deduplication
                    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                    
                    if clean_url not in seen:
                        seen.add(clean_url)
                        normalized_links.append(absolute_url)
            except:
                continue
        
        return normalized_links[:100]  # Limit to 100 links
    
    async def _extract_metadata(self, page) -> Dict:
        """Extract page metadata"""
        metadata = await page.evaluate('''
            () => {
                const metaTags = Array.from(document.querySelectorAll('meta'));
                const metaData = {};
                
                metaTags.forEach(tag => {
                    const name = tag.getAttribute('name') || 
                                tag.getAttribute('property') || 
                                tag.getAttribute('itemprop');
                    const content = tag.getAttribute('content');
                    
                    if (name && content) {
                        metaData[name] = content;
                    }
                });
                
                // Get Open Graph data
                const ogData = {};
                metaTags.forEach(tag => {
                    const property = tag.getAttribute('property');
                    if (property && property.startsWith('og:')) {
                        ogData[property] = tag.getAttribute('content');
                    }
                });
                
                // Get JSON-LD structured data
                const jsonLdScripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
                const jsonLd = jsonLdScripts.map(script => {
                    try {
                        return JSON.parse(script.textContent);
                    } catch (e) {
                        return null;
                    }
                }).filter(data => data !== null);
                
                return {
                    meta_tags: metaData,
                    open_graph: ogData,
                    json_ld: jsonLd,
                    images: Array.from(document.querySelectorAll('img')).map(img => img.src).slice(0, 20),
                    scripts_count: document.querySelectorAll('script').length,
                    stylesheets_count: document.querySelectorAll('link[rel="stylesheet"]').length
                };
            }
        ''')
        
        return metadata
    
    async def _capture_elements(self, page) -> Dict[str, str]:
        """Capture screenshots of specific elements"""
        screenshots = {}
        
        # Capture common elements
        selectors = {
            'header': 'header, .header, #header',
            'main_article': 'article, .article, main',
            'hero_image': '.hero-image, .featured-image, .main-image'
        }
        
        for name, selector in selectors.items():
            try:
                element = await page.wait_for_selector(selector, timeout=2000)
                if element:
                    screenshot = await element.screenshot(type='jpeg', quality=70)
                    screenshots[name] = screenshot
            except:
                continue
        
        return screenshots

# REAL USAGE EXAMPLE
async def collect_real_data():
    """ACTUAL data collection in action"""
    urls_to_scrape = [
        'https://news.ycombinator.com',
        'https://www.reddit.com/r/programming',
        'https://techcrunch.com',
        'https://www.theverge.com/tech'
    ]
    
    async with RealWebScraper(headless=True) as scraper:
        scraped_data = []
        
        for url in urls_to_scrape:
            print(f"Scraping: {url}")
            try:
                page_data = await scraper.scrape_page(url, scroll=True)
                scraped_data.append(page_data)
                
                print(f"✓ Collected: {page_data.title}")
                print(f"  Content length: {len(page_data.content)} chars")
                print(f"  Links found: {len(page_data.links)}")
                
                # Save to SQLite database instead of JSON file
                if save_scraped_data(page_data):
                    print(f"  ✅ Saved to database")
                else:
                    print(f"  ❌ Failed to save to database")
                
            except Exception as e:
                print(f"✗ Failed to scrape {url}: {e}")
        
        return scraped_data

def view_scraped_data():
    """View recently scraped data from database"""
    print("\n📊 Recently Scraped Data:")
    print("=" * 50)
    
    data = get_scraped_data(limit=10)
    for item in data:
        print(f"🌐 {item['domain']}")
        print(f"   Title: {item['title'][:60]}...")
        print(f"   Words: {item['word_count']}")
        print(f"   Date: {item['scraped_at']}")
        print(f"   URL: {item['url']}")
        print("-" * 40)

# Run it
# asyncio.run(collect_real_data())
# view_scraped_data()