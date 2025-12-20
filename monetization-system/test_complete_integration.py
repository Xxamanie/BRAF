#!/usr/bin/env python3
"""
Complete Integration Test
Tests the full Playwright + SQLite + Automation pipeline
"""
import os
import sys
import json
from datetime import datetime

def test_sync_scraper():
    """Test the synchronous Playwright scraper"""
    print("🧪 Testing Synchronous Playwright Scraper")
    print("=" * 40)
    
    try:
        from sync_playwright_scraper import SyncPlaywrightScraper
        
        # Initialize scraper
        scraper = SyncPlaywrightScraper(headless=True)
        
        # Test single URL
        test_url = "https://httpbin.org/html"
        target = {"url": test_url}
        
        print(f"📥 Testing single URL: {test_url}")
        result = scraper.run_single_scrape(target)
        
        if result.success:
            print(f"✅ Success: {result.title[:50]}...")
            print(f"   Content: {len(result.content)} chars")
            print(f"   Words: {result.word_count}")
        else:
            print(f"❌ Failed: {result.error}")
        
        return result.success
        
    except ImportError:
        print("❌ Playwright scraper not available")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_database_integration():
    """Test database operations"""
    print("\n🗄️  Testing Database Integration")
    print("=" * 40)
    
    try:
        from database_manager import ScraperDatabaseManager
        
        manager = ScraperDatabaseManager()
        stats = manager.get_stats()
        
        print(f"📊 Database Statistics:")
        print(f"   Total Records: {stats.get('total_records', 0)}")
        print(f"   Unique Domains: {stats.get('unique_domains', 0)}")
        print(f"   Database Size: {stats.get('database_size_mb', 0)} MB")
        
        # Test search
        results = manager.search_content("example")
        print(f"   Search Results: {len(results)} for 'example'")
        
        return stats.get('total_records', 0) > 0
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_status_monitoring():
    """Test status monitoring system"""
    print("\n📊 Testing Status Monitoring")
    print("=" * 40)
    
    try:
        from check_scraper_status import ScraperStatusMonitor
        
        monitor = ScraperStatusMonitor()
        status_data = monitor.load_status()
        
        if status_data:
            status = status_data.get('status', 'unknown')
            timestamp = status_data.get('timestamp', 'unknown')
            stats = status_data.get('stats', {})
            
            print(f"📈 Last Status: {status.upper()}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Pages Scraped: {stats.get('pages_scraped', 0)}")
            print(f"   Success Rate: {stats.get('success_rate', 0):.1f}%")
            
            return status in ['completed', 'running']
        else:
            print("❌ No status data found")
            return False
            
    except Exception as e:
        print(f"❌ Status monitoring error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️  Testing Configuration")
    print("=" * 40)
    
    config_file = os.path.join(os.path.dirname(__file__), 'scraper_urls.json')
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            urls = config.get('urls', [])
            settings = config.get('config', {})
            
            print(f"📋 Configuration Loaded:")
            print(f"   URLs: {len(urls)}")
            print(f"   Max Pages: {settings.get('max_pages_per_run', 'default')}")
            print(f"   Delay: {settings.get('delay_between_pages', 'default')}s")
            print(f"   Retries: {settings.get('max_retries', 'default')}")
            
            return len(urls) > 0
        else:
            print("❌ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def run_integration_test():
    """Run complete integration test"""
    print("🚀 Complete Integration Test")
    print("=" * 50)
    print(f"Test Time: {datetime.now()}")
    print()
    
    tests = [
        ("Configuration", test_configuration),
        ("Database Integration", test_database_integration),
        ("Status Monitoring", test_status_monitoring),
        ("Sync Scraper", test_sync_scraper),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)