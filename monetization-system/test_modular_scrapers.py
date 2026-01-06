#!/usr/bin/env python3
"""
Test Modular Scrapers
Demonstrates HTTP and Browser scraping with fallback functionality
"""
import sys
import os
from datetime import datetime

# Add scrapers to path
sys.path.append(os.path.dirname(__file__))

def test_individual_scrapers():
    """Test each scraper individually"""
    print("🧪 Testing Individual Scrapers")
    print("=" * 40)
    
    # Test HTTP scraper
    print("\n📡 HTTP Scraper Test:")
    try:
        from scrapers.http_scraper import run as http_run
        
        target = {"url": "https://httpbin.org/html"}
        result = http_run(target)
        
        if result["success"]:
            print(f"✅ HTTP Success: {result['title'][:50]}...")
            print(f"   Words: {result['word_count']}")
            print(f"   Status: {result.get('status_code', 'N/A')}")
        else:
            print(f"❌ HTTP Failed: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ HTTP Scraper Error: {e}")
    
    # Test Browser scraper
    print("\n🖥️  Browser Scraper Test:")
    try:
        from scrapers.browser_scraper import run as browser_run
        
        target = {"url": "https://example.com", "headless": True}
        result = browser_run(target)
        
        if result["success"]:
            print(f"✅ Browser Success: {result['title'][:50]}...")
            print(f"   Words: {result['word_count']}")
        else:
            print(f"❌ Browser Failed: {result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Browser Scraper Error: {e}")

def test_scraper_selection():
    """Test scraper selection and fallback"""
    print("\n🔄 Testing Scraper Selection & Fallback")
    print("=" * 40)
    
    try:
        from scrapers import SCRAPERS
        
        print(f"📋 Available scrapers: {list(SCRAPERS.keys())}")
        
        # Test with different URLs
        test_cases = [
            {
                "url": "https://httpbin.org/html",
                "description": "Simple HTML page (good for HTTP)"
            },
            {
                "url": "https://example.com",
                "description": "Basic static page"
            }
        ]
        
        for case in test_cases:
            print(f"\n🌐 Testing: {case['description']}")
            print(f"   URL: {case['url']}")
            
            # Test with HTTP scraper
            target = {"url": case["url"]}
            http_result = SCRAPERS["http"](target)
            http_status = "✅" if http_result["success"] else "❌"
            print(f"   HTTP: {http_status} ({http_result.get('word_count', 0)} words)")
            
            # Test with Browser scraper
            target["headless"] = True
            browser_result = SCRAPERS["browser"](target)
            browser_status = "✅" if browser_result["success"] else "❌"
            print(f"   Browser: {browser_status} ({browser_result.get('word_count', 0)} words)")
            
    except Exception as e:
        print(f"❌ Scraper selection test failed: {e}")

def test_performance_comparison():
    """Compare performance between scrapers"""
    print("\n⚡ Performance Comparison")
    print("=" * 40)
    
    test_url = "https://httpbin.org/html"
    
    try:
        from scrapers import SCRAPERS
        import time
        
        # Test HTTP scraper performance
        start_time = time.time()
        http_result = SCRAPERS["http"]({"url": test_url})
        http_duration = time.time() - start_time
        
        # Test Browser scraper performance
        start_time = time.time()
        browser_result = SCRAPERS["browser"]({"url": test_url, "headless": True})
        browser_duration = time.time() - start_time
        
        print(f"📊 Performance Results for {test_url}:")
        print(f"   HTTP Scraper:")
        print(f"     Duration: {http_duration:.2f}s")
        print(f"     Success: {'✅' if http_result['success'] else '❌'}")
        print(f"     Words: {http_result.get('word_count', 0)}")
        
        print(f"   Browser Scraper:")
        print(f"     Duration: {browser_duration:.2f}s")
        print(f"     Success: {'✅' if browser_result['success'] else '❌'}")
        print(f"     Words: {browser_result.get('word_count', 0)}")
        
        if http_result['success'] and browser_result['success']:
            speed_ratio = browser_duration / http_duration
            print(f"   📈 Browser is {speed_ratio:.1f}x slower than HTTP")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

def test_configuration_loading():
    """Test configuration loading"""
    print("\n⚙️  Configuration Test")
    print("=" * 40)
    
    try:
        import json
        
        config_file = os.path.join(os.path.dirname(__file__), 'scraper_config.json')
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"✅ Configuration loaded:")
            print(f"   URLs: {len(config.get('urls', []))}")
            print(f"   Preferred scraper: {config.get('scraper_config', {}).get('preferred_scraper', 'default')}")
            print(f"   Fallback enabled: {config.get('scraper_config', {}).get('fallback_enabled', 'default')}")
            print(f"   Max pages: {config.get('scraper_config', {}).get('max_pages_per_run', 'default')}")
            
            # Show URL-specific configs
            url_configs = config.get('url_specific_config', {})
            if url_configs:
                print(f"   URL-specific configs: {len(url_configs)}")
                for url, settings in url_configs.items():
                    print(f"     {url}: {settings.get('preferred_scraper', 'default')}")
        else:
            print("❌ Configuration file not found")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def run_all_tests():
    """Run all tests"""
    print("🚀 Modular Scraper Test Suite")
    print("=" * 50)
    print(f"Test Time: {datetime.now()}")
    
    tests = [
        ("Individual Scrapers", test_individual_scrapers),
        ("Scraper Selection", test_scraper_selection),
        ("Performance Comparison", test_performance_comparison),
        ("Configuration Loading", test_configuration_loading)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print(f"\n{'='*50}")
    print("🎉 Test suite completed!")

if __name__ == "__main__":
    run_all_tests()
