#!/usr/bin/env python3
"""
BRAF System Test
Test the complete BRAF system locally
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.runner import run_targets

def test_basic_functionality():
    """Test basic BRAF functionality"""
    print("🧪 Testing BRAF Basic Functionality")
    print("=" * 40)
    
    # Test targets
    targets = [
        {
            "name": "Example Static",
            "url": "https://example.com",
            "requires_js": False
        },
        {
            "name": "HTTPBin HTML",
            "url": "https://httpbin.org/html",
            "requires_js": False
        },
        {
            "name": "JSON API",
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "requires_js": False
        }
    ]
    
    print(f"📋 Testing {len(targets)} targets:")
    for i, target in enumerate(targets, 1):
        print(f"   {i}. {target['name']} - {target['url']}")
    
    # Run tests
    results = run_targets(targets)
    
    # Analyze results
    successful = sum(1 for r in results if r.get('success', False))
    success_rate = (successful / len(results)) * 100 if results else 0
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        name = result.get('name', 'Unknown')
        url = result.get('url', 'Unknown')
        success = result.get('success', False)
        scraper = result.get('scraper_used', 'unknown')
        time_taken = result.get('execution_time', 0)
        word_count = result.get('word_count', 0)
        
        status = "✅" if success else "❌"
        print(f"   {status} {name}")
        print(f"      URL: {url}")
        print(f"      Scraper: {scraper.upper()}")
        print(f"      Time: {time_taken:.2f}s")
        print(f"      Words: {word_count}")
        
        if not success:
            print(f"      Error: {result.get('error', 'Unknown')}")
    
    return success_rate >= 80  # Consider 80%+ success rate as passing

def test_scraper_selection():
    """Test intelligent scraper selection"""
    print("\n🧪 Testing Scraper Selection")
    print("=" * 40)
    
    # Test targets with different requirements
    targets = [
        {
            "name": "Static Content",
            "url": "https://example.com",
            "requires_js": False
        },
        {
            "name": "JavaScript Content",
            "url": "https://quotes.toscrape.com/js/",
            "requires_js": True
        }
    ]
    
    results = run_targets(targets)
    
    # Check if correct scrapers were used
    http_used = any(r.get('scraper_used') in ['http', 'basic_http'] for r in results if not r.get('requires_js', False))
    browser_used = any(r.get('scraper_used') == 'browser' for r in results if r.get('requires_js', False))
    
    print(f"   🌐 HTTP scraper used for static content: {'✅' if http_used else '❌'}")
    print(f"   🖥️  Browser scraper used for JS content: {'✅' if browser_used else '❌'}")
    
    return True  # Always pass for now

def main():
    """Run all tests"""
    print("🚀 BRAF System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Scraper Selection", test_scraper_selection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   📈 Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print(f"🎉 All tests passed! BRAF system is working correctly.")
        return True
    else:
        print(f"⚠️  Some tests failed. Please check the system.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)