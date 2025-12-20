#!/usr/bin/env python3
"""
HTTP Scraper
Fast, lightweight scraping using requests library
Best for: Static pages, APIs, simple content
"""
import requests
from typing import Dict
from urllib.parse import urlparse
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def run(target: Dict) -> Dict:
    """
    HTTP-based scraper using requests library
    
    Args:
        target: Dictionary with 'url' key and optional 'headers', 'timeout'
        
    Returns:
        Dictionary with scraped data or error info
    """
    url = target.get("url")
    headers = target.get("headers", {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    timeout = target.get("timeout", 30)
    
    try:
        logger.info(f"📥 HTTP scraping: {url}")
        
        # Make HTTP request
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Extract content
        content = response.text
        
        # Try to extract title from HTML
        title = ""
        if "<title>" in content.lower():
            start = content.lower().find("<title>") + 7
            end = content.lower().find("</title>", start)
            if end > start:
                title = content[start:end].strip()
        
        # Limit content size
        content = content[:4000]
        
        # Calculate word count
        word_count = len(content.split())
        
        # Generate hash
        data_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        result = {
            "url": url,
            "domain": urlparse(url).netloc,
            "title": title or f"Page from {urlparse(url).netloc}",
            "content": content,
            "word_count": word_count,
            "scraped_at": datetime.now(),
            "data_hash": data_hash,
            "success": True,
            "scraper_type": "http",
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", "unknown")
        }
        
        logger.info(f"✅ HTTP scraped: {title[:50]}... ({response.status_code})")
        return result
        
    except requests.Timeout:
        error_msg = f"Timeout after {timeout}s"
        logger.error(f"❌ {error_msg}: {url}")
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
            "scraper_type": "http"
        }
        
    except requests.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        logger.error(f"❌ {error_msg}: {url}")
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
            "scraper_type": "http"
        }
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"❌ {error_msg}: {url}")
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
            "scraper_type": "http"
        }

def test():
    """Test HTTP scraper"""
    print("🧪 Testing HTTP Scraper")
    print("=" * 30)
    
    test_urls = [
        {"url": "https://httpbin.org/html"},
        {"url": "https://example.com"},
        {"url": "https://jsonplaceholder.typicode.com/posts/1"}
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