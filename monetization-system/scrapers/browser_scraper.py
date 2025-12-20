#!/usr/bin/env python3
"""
Browser Scraper
Full browser automation using Playwright
Best for: JavaScript-heavy sites, dynamic content, complex interactions
"""
from typing import Dict
from urllib.parse import urlparse
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import Playwright
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available. Install with: pip install playwright")

def run(target: Dict) -> Dict:
    """
    Browser-based scraper using Playwright
    
    Args:
        target: Dictionary with 'url' key and optional 'headless', 'timeout'
        
    Returns:
        Dictionary with scraped data or error info
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {
            "url": target["url"],
            "domain": urlparse(target["url"]).netloc,
            "title": "",
            "content": "",
            "word_count": 0,
            "scraped_at": datetime.now(),
            "data_hash": "",
            "success": False,
            "error": "Playwright not available",
            "scraper_type": "browser"
        }
    
    url = target.get("url")
    headless = target.get("headless", True)
    timeout = target.get("timeout", 60000)
    
    try:
        logger.info(f"📥 Browser scraping: {url}")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox"
                ]
            )
            
            # Create page
            page = browser.new_page()
            page.set_viewport_size({"width": 1920, "height": 1080})
            
            # Navigate to URL
            page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            
            # Wait for page to stabilize
            page.wait_for_timeout(3000)
            
            # Extract data
            title = page.title()
            content = page.inner_text("body")[:4000]
            
            # Clean content
            content = content.strip()
            word_count = len(content.split()) if content else 0
            
            # Generate hash
            data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Close browser
            browser.close()
            
            result = {
                "url": url,
                "domain": urlparse(url).netloc,
                "title": title,
                "content": content,
                "word_count": word_count,
                "scraped_at": datetime.now(),
                "data_hash": data_hash,
                "success": True,
                "scraper_type": "browser"
            }
            
            logger.info(f"✅ Browser scraped: {title[:50]}...")
            return result
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Browser scraping failed: {error_msg}")
        return {
            "url": url,
            "domain": urlparse(url).netloc,
            "title": "",
            "content": "",
            "word_count": 0,
            "scraped_at": datetime.now(),
            "data_hash": "",
            "success": False,
            "error": error_msg,
            "scraper_type": "browser"
        }

def test():
    """Test browser scraper"""
    print("🧪 Testing Browser Scraper")
    print("=" * 30)
    
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright not available")
        return
    
    test_urls = [
        {"url": "https://httpbin.org/html"},
        {"url": "https://example.com"}
    ]
    
    for target in test_urls:
        result = run(target)
        status = "✅" if result["success"] else "❌"
        print(f"{status} {target['url']}")
        if result["success"]:
            print(f"   Title: {result['title'][:50]}...")
            print(f"   Words: {result['word_count']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    test()