#!/usr/bin/env python3
"""
BRAF HTTP Scraper
Fast HTTP-based scraper for static content
"""
import requests
from bs4 import BeautifulSoup
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run(target):
    """
    Run HTTP scraper on target
    
    Args:
        target: Dictionary with 'url' and optional metadata
        
    Returns:
        Dictionary with scraping results
    """
    url = target.get('url', '')
    start_time = time.time()
    
    logger.info(f"📥 HTTP scraping: {url}")
    
    try:
        # Make HTTP request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text content
        text = soup.get_text()
        word_count = len(text.split())
        
        # Get title
        title = soup.title.string if soup.title else 'No title'
        title = title.strip()[:100] + '...' if len(title) > 100 else title
        
        execution_time = time.time() - start_time
        
        result = {
            'url': url,
            'success': True,
            'scraper_used': 'http',
            'execution_time': execution_time,
            'word_count': word_count,
            'title': title,
            'status_code': response.status_code,
            'content_length': len(response.text)
        }
        
        logger.info(f"✅ HTTP scraped: {title} ({response.status_code})")
        
        return result
        
    except requests.exceptions.RequestException as e:
        execution_time = time.time() - start_time
        
        result = {
            'url': url,
            'success': False,
            'error': str(e),
            'scraper_used': 'http',
            'execution_time': execution_time
        }
        
        logger.error(f"❌ HTTP scraping failed: {url} - {e}")
        
        return result
    
    except Exception as e:
        execution_time = time.time() - start_time
        
        result = {
            'url': url,
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'scraper_used': 'http',
            'execution_time': execution_time
        }
        
        logger.error(f"❌ Unexpected error: {url} - {e}")
        
        return result

def main():
    """Test the HTTP scraper"""
    test_targets = [
        {"url": "https://example.com"},
        {"url": "https://httpbin.org/html"},
        {"url": "https://jsonplaceholder.typicode.com/posts/1"}
    ]
    
    print("🧪 Testing HTTP Scraper")
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
