#!/usr/bin/env python3
"""
BRAF Browser Scraper
Playwright-based scraper for JavaScript-rendered content
"""
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(target):
    """
    Run browser scraper on target
    
    Args:
        target: Dictionary with 'url' and optional metadata
        
    Returns:
        Dictionary with scraping results
    """
    url = target.get('url', '')
    start_time = time.time()
    
    logger.info(f"📥 Browser scraping: {url}")
    
    try:
        # Try to import playwright
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            # Fallback to requests if playwright not available
            logger.warning("Playwright not available, falling back to HTTP scraper")
            from . import http_scraper
            return http_scraper.run(target)
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Set user agent
            page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Navigate to page
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for content to load
            page.wait_for_timeout(2000)
            
            # Extract content
            title = page.title()
            text_content = page.evaluate('document.body.innerText')
            
            # Count words
            word_count = len(text_content.split()) if text_content else 0
            
            # Clean title
            title = title.strip()[:100] + '...' if len(title) > 100 else title
            
            browser.close()
            
            execution_time = time.time() - start_time
            
            result = {
                'url': url,
                'success': True,
                'scraper_used': 'browser',
                'execution_time': execution_time,
                'word_count': word_count,
                'title': title,
                'content_length': len(text_content) if text_content else 0
            }
            
            logger.info(f"✅ Browser scraped: {title}")
            
            return result
            
    except Exception as e:
        execution_time = time.time() - start_time
        
        result = {
            'url': url,
            'success': False,
            'error': str(e),
            'scraper_used': 'browser',
            'execution_time': execution_time
        }
        
        logger.error(f"❌ Browser scraping failed: {url} - {e}")
        
        return result

def main():
    """Test the browser scraper"""
    test_targets = [
        {"url": "https://quotes.toscrape.com/js/"},
        {"url": "https://example.com"}
    ]
    
    print("🧪 Testing Browser Scraper")
    print("=" * 30)
    
    for target in test_targets:
        result = run(target)
        url = result['url']
        success = result['success']
        time_taken = result.get('execution_time', 0)
        
        print(f"{'✅' if success else '❌'} {url} - {time_taken:.2f}s")
        
        if success:
            print(f"   Words: {result.get('word_count', 0)}")
            print(f"   Title: {result.get('title', 'N/A')}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")

if __name__ == "__main__":
    main()