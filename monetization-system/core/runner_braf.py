#!/usr/bin/env python3
"""
BRAF Simple Runner Interface
Simple interface for running BRAF with targets
"""
import sys
import os
from typing import List, Dict

def run_targets(targets: List[Dict]) -> List[Dict]:
    """
    Simple interface to run BRAF with targets
    
    Args:
        targets: List of target dictionaries with 'url' and optional metadata
        
    Returns:
        List of scraping results
    """
    print(f"🚀 BRAF Runner - Processing {len(targets)} targets")
    
    # Import scrapers
    try:
        from scrapers.http_scraper import run as http_run
        from scrapers.browser_scraper import run as browser_run
    except ImportError:
        print("❌ Scrapers not found. Using basic implementation.")
        return basic_scraper(targets)
    
    results = []
    
    for target in targets:
        url = target.get('url', '')
        requires_js = target.get('requires_js', False)
        
        print(f"📥 Processing: {url}")
        
        try:
            if requires_js:
                print(f"🖥️  Using browser scraper for: {url}")
                result = browser_run(target)
            else:
                print(f"🌐 Using HTTP scraper for: {url}")
                result = http_run(target)
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing {url}: {e}")
            results.append({
                'url': url,
                'success': False,
                'error': str(e),
                'scraper_used': 'error'
            })
    
    successful = sum(1 for r in results if r.get('success', False))
    print(f"📊 Results: {successful}/{len(results)} successful")
    
    return results

def basic_scraper(targets: List[Dict]) -> List[Dict]:
    """Basic fallback scraper using requests"""
    import requests
    from bs4 import BeautifulSoup
    import time
    
    results = []
    
    for target in targets:
        url = target.get('url', '')
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            word_count = len(text.split())
            
            execution_time = time.time() - start_time
            
            result = {
                'url': url,
                'success': True,
                'scraper_used': 'basic_http',
                'execution_time': execution_time,
                'word_count': word_count,
                'title': soup.title.string if soup.title else 'No title'
            }
            
            print(f"✅ {url} - {word_count} words in {execution_time:.2f}s")
            
        except Exception as e:
            result = {
                'url': url,
                'success': False,
                'error': str(e),
                'scraper_used': 'basic_http'
            }
            print(f"❌ {url} - Error: {e}")
        
        results.append(result)
    
    return results

def main():
    """Demo usage of the runner"""
    targets = [
        {"url": "https://example.com", "requires_js": False},
        {"url": "https://httpbin.org/html", "requires_js": False},
        {"url": "https://jsonplaceholder.typicode.com/posts/1", "requires_js": False},
    ]
    
    results = run_targets(targets)
    
    print(f"\n🎯 Demo completed with {len(results)} results")
    for result in results:
        url = result.get('url', 'unknown')
        success = result.get('success', False)
        scraper = result.get('scraper_used', 'unknown')
        print(f"   {'✅' if success else '❌'} {url} ({scraper})")

if __name__ == "__main__":
    main()
