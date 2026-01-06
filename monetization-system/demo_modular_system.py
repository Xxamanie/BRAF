#!/usr/bin/env python3
"""
Modular Scraper System Demo
Demonstrates the flexibility and capabilities of the modular scraping system
"""
import sys
import os
import json
from datetime import datetime

# Add scrapers to path
sys.path.append(os.path.dirname(__file__))

def demo_basic_usage():
    """Demonstrate basic scraper usage"""
    print("🚀 Basic Scraper Usage Demo")
    print("=" * 40)
    
    from scrapers import SCRAPERS
    
    test_url = "https://httpbin.org/html"
    
    print(f"🌐 Testing URL: {test_url}")
    print(f"📋 Available scrapers: {list(SCRAPERS.keys())}")
    
    # Test each scraper
    for scraper_name, scraper_func in SCRAPERS.items():
        print(f"\n🔧 Testing {scraper_name.upper()} scraper:")
        
        target = {"url": test_url}
        if scraper_name == "browser":
            target["headless"] = True
        
        try:
            result = scraper_func(target)
            
            if result["success"]:
                print(f"   ✅ Success!")
                print(f"   📄 Title: {result['title'][:50]}...")
                print(f"   📝 Words: {result['word_count']}")
                print(f"   🏷️  Type: {result.get('scraper_type', 'unknown')}")
            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")

def demo_intelligent_selection():
    """Demonstrate intelligent scraper selection"""
    print("\n🧠 Intelligent Scraper Selection Demo")
    print("=" * 40)
    
    # Different types of websites
    test_cases = [
        {
            "url": "https://httpbin.org/html",
            "description": "Simple HTML page",
            "expected_best": "http"
        },
        {
            "url": "https://example.com",
            "description": "Static website",
            "expected_best": "http"
        },
        {
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "description": "JSON API endpoint",
            "expected_best": "http"
        }
    ]
    
    from scrapers import SCRAPERS
    import time
    
    for case in test_cases:
        print(f"\n🌐 {case['description']}")
        print(f"   URL: {case['url']}")
        print(f"   Expected best: {case['expected_best'].upper()}")
        
        results = {}
        
        # Test both scrapers
        for scraper_name, scraper_func in SCRAPERS.items():
            target = {"url": case["url"]}
            if scraper_name == "browser":
                target["headless"] = True
            
            try:
                start_time = time.time()
                result = scraper_func(target)
                duration = time.time() - start_time
                
                results[scraper_name] = {
                    "success": result["success"],
                    "duration": duration,
                    "words": result.get("word_count", 0),
                    "error": result.get("error")
                }
                
            except Exception as e:
                results[scraper_name] = {
                    "success": False,
                    "duration": 0,
                    "words": 0,
                    "error": str(e)
                }
        
        # Show results
        for scraper_name, result in results.items():
            status = "✅" if result["success"] else "❌"
            print(f"   {scraper_name.upper()}: {status} ({result['duration']:.2f}s, {result['words']} words)")
            if not result["success"]:
                print(f"      Error: {result['error']}")
        
        # Recommend best scraper
        if results["http"]["success"] and results["browser"]["success"]:
            if results["http"]["duration"] < results["browser"]["duration"]:
                print(f"   💡 Recommendation: HTTP (faster)")
            else:
                print(f"   💡 Recommendation: Browser (more reliable)")
        elif results["http"]["success"]:
            print(f"   💡 Recommendation: HTTP (only working option)")
        elif results["browser"]["success"]:
            print(f"   💡 Recommendation: Browser (only working option)")
        else:
            print(f"   ⚠️  Both scrapers failed")

def demo_configuration_system():
    """Demonstrate configuration-driven scraping"""
    print("\n⚙️  Configuration System Demo")
    print("=" * 40)
    
    # Create sample configuration
    sample_config = {
        "urls": [
            "https://httpbin.org/html",
            "https://example.com"
        ],
        "scraper_config": {
            "preferred_scraper": "http",
            "fallback_enabled": True,
            "max_retries": 2
        },
        "url_specific_config": {
            "https://example.com": {
                "preferred_scraper": "browser",
                "timeout_per_page": 45
            }
        }
    }
    
    print("📋 Sample Configuration:")
    print(json.dumps(sample_config, indent=2))
    
    print(f"\n🔧 Configuration Analysis:")
    print(f"   Default scraper: {sample_config['scraper_config']['preferred_scraper'].upper()}")
    print(f"   Fallback enabled: {sample_config['scraper_config']['fallback_enabled']}")
    print(f"   URL overrides: {len(sample_config['url_specific_config'])}")
    
    for url, config in sample_config['url_specific_config'].items():
        print(f"     {url} → {config['preferred_scraper'].upper()}")

def demo_performance_comparison():
    """Demonstrate performance differences"""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 40)
    
    from scrapers import SCRAPERS
    import time
    
    test_url = "https://httpbin.org/html"
    iterations = 3
    
    print(f"🌐 Testing URL: {test_url}")
    print(f"🔄 Iterations: {iterations}")
    
    performance_data = {}
    
    for scraper_name, scraper_func in SCRAPERS.items():
        print(f"\n🔧 Testing {scraper_name.upper()} scraper:")
        
        times = []
        successes = 0
        
        for i in range(iterations):
            target = {"url": test_url}
            if scraper_name == "browser":
                target["headless"] = True
            
            try:
                start_time = time.time()
                result = scraper_func(target)
                duration = time.time() - start_time
                
                times.append(duration)
                if result["success"]:
                    successes += 1
                
                print(f"   Run {i+1}: {duration:.2f}s ({'✅' if result['success'] else '❌'})")
                
            except Exception as e:
                print(f"   Run {i+1}: Failed - {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            success_rate = (successes / iterations) * 100
            
            performance_data[scraper_name] = {
                "avg_time": avg_time,
                "success_rate": success_rate
            }
            
            print(f"   📊 Average: {avg_time:.2f}s")
            print(f"   📈 Success rate: {success_rate:.1f}%")
    
    # Compare performance
    if len(performance_data) == 2:
        http_time = performance_data.get("http", {}).get("avg_time", 0)
        browser_time = performance_data.get("browser", {}).get("avg_time", 0)
        
        if http_time > 0 and browser_time > 0:
            ratio = browser_time / http_time
            print(f"\n📊 Performance Summary:")
            print(f"   Browser is {ratio:.1f}x slower than HTTP")
            print(f"   HTTP: {http_time:.2f}s average")
            print(f"   Browser: {browser_time:.2f}s average")

def demo_error_handling():
    """Demonstrate error handling and fallback"""
    print("\n🛡️  Error Handling & Fallback Demo")
    print("=" * 40)
    
    from scrapers import SCRAPERS
    
    # Test with problematic URLs
    test_cases = [
        {
            "url": "https://httpbin.org/status/404",
            "description": "404 Not Found"
        },
        {
            "url": "https://httpbin.org/delay/10",
            "description": "Slow response (10s delay)"
        },
        {
            "url": "https://invalid-domain-that-does-not-exist.com",
            "description": "Invalid domain"
        }
    ]
    
    for case in test_cases:
        print(f"\n🧪 Testing: {case['description']}")
        print(f"   URL: {case['url']}")
        
        for scraper_name, scraper_func in SCRAPERS.items():
            target = {"url": case["url"], "timeout": 5}  # Short timeout
            if scraper_name == "browser":
                target["headless"] = True
                target["timeout"] = 5000  # 5 seconds in ms
            
            try:
                result = scraper_func(target)
                
                if result["success"]:
                    print(f"   {scraper_name.upper()}: ✅ Unexpected success!")
                else:
                    print(f"   {scraper_name.upper()}: ❌ {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   {scraper_name.upper()}: 💥 Exception: {e}")

def main():
    """Run all demos"""
    print("🎭 Modular Scraper System Demo")
    print("=" * 50)
    print(f"Demo Time: {datetime.now()}")
    
    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Intelligent Selection", demo_intelligent_selection),
        ("Configuration System", demo_configuration_system),
        ("Performance Comparison", demo_performance_comparison),
        ("Error Handling", demo_error_handling)
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            demo_func()
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
    
    print(f"\n{'='*50}")
    print("🎉 Demo completed!")
    print("\n💡 Key Takeaways:")
    print("   • HTTP scraper is faster for simple pages")
    print("   • Browser scraper handles JavaScript and complex sites")
    print("   • Configuration allows intelligent method selection")
    print("   • Fallback provides reliability when one method fails")
    print("   • Both methods integrate seamlessly with existing database")

if __name__ == "__main__":
    main()
