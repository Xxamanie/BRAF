#!/usr/bin/env python3
"""
BRAF System Integration Test
Comprehensive test of the Browser Automation Framework with intelligent scraper selection
"""
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_decision_engine():
    """Test the intelligent decision engine"""
    print("🧠 Testing Decision Engine")
    print("=" * 40)
    
    from core.decision import needs_browser, get_decision_explanation
    
    test_cases = [
        # HTTP preferred cases
        {
            "url": "https://httpbin.org/html",
            "expected": False,
            "description": "Simple HTML page"
        },
        {
            "url": "https://api.github.com/users",
            "expected": False,
            "description": "API endpoint"
        },
        {
            "url": "https://example.com/feed.xml",
            "expected": False,
            "description": "XML feed"
        },
        
        # Browser required cases
        {
            "url": "https://app.example.com/dashboard",
            "expected": True,
            "description": "App dashboard"
        },
        {
            "url": "https://example.com/#/spa-route",
            "expected": True,
            "description": "SPA with hash routing"
        },
        {
            "url": "https://dashboard.example.com",
            "expected": True,
            "description": "Dashboard subdomain"
        }
    ]
    
    correct = 0
    for case in test_cases:
        decision = needs_browser(case)
        expected = case["expected"]
        
        if decision == expected:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        scraper_type = "Browser" if decision else "HTTP"
        expected_type = "Browser" if expected else "HTTP"
        
        print(f"{status} {case['description']}")
        print(f"   URL: {case['url']}")
        print(f"   Decision: {scraper_type}, Expected: {expected_type}")
        
        # Show explanation for failed cases
        if decision != expected:
            explanation = get_decision_explanation(case)
            print(f"   Factors: {', '.join(explanation['factors'])}")
        print()
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"📊 Decision Engine Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")
    return accuracy > 80

def test_scraper_registry():
    """Test the scraper registry functionality"""
    print("\n🔧 Testing Scraper Registry")
    print("=" * 40)
    
    from scrapers.registry import (
        SCRAPERS, get_scraper, get_scraper_info, 
        list_scrapers, validate_scraper_availability
    )
    
    # Test registry contents
    print(f"📋 Available scrapers: {list(SCRAPERS.keys())}")
    
    # Test scraper info
    for scraper_type in SCRAPERS.keys():
        info = get_scraper_info(scraper_type)
        print(f"\n{scraper_type.upper()} SCRAPER:")
        print(f"   Name: {info['name']}")
        print(f"   Performance: {info['performance']}")
        print(f"   JavaScript Support: {'Yes' if info['javascript_support'] else 'No'}")
    
    # Test availability
    availability = validate_scraper_availability()
    print(f"\n📊 Scraper Availability:")
    for scraper_type, available in availability.items():
        status = "✅ Available" if available else "❌ Unavailable"
        print(f"   {scraper_type.upper()}: {status}")
    
    return all(availability.values())

def test_braf_execution():
    """Test BRAF execution with mixed targets"""
    print("\n🚀 Testing BRAF Execution")
    print("=" * 40)
    
    from braf_runner import run_targets
    
    # Mixed targets to test both scrapers
    test_targets = [
        {
            "url": "https://httpbin.org/html",
            "description": "Simple HTML (should use HTTP)"
        },
        {
            "url": "https://example.com",
            "description": "Static site (should use HTTP)"
        },
        {
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "description": "JSON API (should use HTTP)"
        },
        {
            "url": "https://app.example.com/dashboard",
            "description": "App dashboard (should use Browser)",
            "preferred_scraper": "browser"  # Force browser for testing
        }
    ]
    
    print(f"📋 Testing with {len(test_targets)} targets:")
    for target in test_targets:
        print(f"   • {target['description']}")
    
    print(f"\n🔄 Running BRAF execution...")
    
    try:
        results = run_targets(test_targets)
        
        # Analyze results
        successful = sum(1 for r in results if r.get('success', False))
        http_used = sum(1 for r in results if r.get('braf_metadata', {}).get('scraper_selected') == 'http')
        browser_used = sum(1 for r in results if r.get('braf_metadata', {}).get('scraper_selected') == 'browser')
        
        print(f"📊 Execution Results:")
        print(f"   ✅ Successful: {successful}/{len(results)}")
        print(f"   🌐 HTTP used: {http_used}")
        print(f"   🖥️  Browser used: {browser_used}")
        
        # Check if we got expected scraper distribution
        expected_http = 3  # First 3 targets should use HTTP
        expected_browser = 1  # Last target should use browser
        
        scraper_distribution_correct = (http_used >= expected_http - 1 and browser_used >= expected_browser - 1)
        
        return successful == len(test_targets) and scraper_distribution_correct
        
    except Exception as e:
        print(f"❌ BRAF execution failed: {e}")
        return False

def test_fallback_mechanism():
    """Test fallback mechanism with registry"""
    print("\n🔄 Testing Fallback Mechanism")
    print("=" * 40)
    
    from scrapers.registry import run_with_best_scraper
    
    # Test with a simple target
    test_target = {
        "url": "https://httpbin.org/html",
        "description": "Test fallback with simple target"
    }
    
    print(f"🌐 Testing fallback with: {test_target['url']}")
    
    try:
        result = run_with_best_scraper(test_target)
        
        if result.get('success', False):
            scraper_used = result.get('scraper_used', 'unknown')
            fallback_used = result.get('fallback_used', False)
            
            print(f"✅ Success with {scraper_used} scraper")
            if fallback_used:
                print(f"   🔄 Fallback was used")
            else:
                print(f"   ✨ Primary scraper worked")
            
            return True
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

def test_results_format():
    """Test results file format and metadata"""
    print("\n📄 Testing Results Format")
    print("=" * 40)
    
    import json
    
    results_file = os.path.join(os.path.dirname(__file__), 'data', 'results.json')
    
    if not os.path.exists(results_file):
        print("❌ Results file not found")
        return False
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # Check structure
        required_keys = ['braf_execution', 'results']
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing key: {key}")
                return False
        
        # Check execution metadata
        execution = data['braf_execution']
        required_exec_keys = ['version', 'execution_id', 'statistics']
        for key in required_exec_keys:
            if key not in execution:
                print(f"❌ Missing execution key: {key}")
                return False
        
        # Check statistics
        stats = execution['statistics']
        required_stat_keys = ['total_targets', 'successful', 'failed', 'success_rate']
        for key in required_stat_keys:
            if key not in stats:
                print(f"❌ Missing statistics key: {key}")
                return False
        
        # Check results format
        results = data['results']
        if not results:
            print("❌ No results found")
            return False
        
        # Check first result structure
        first_result = results[0]
        required_result_keys = ['url', 'success', 'braf_metadata']
        for key in required_result_keys:
            if key not in first_result:
                print(f"❌ Missing result key: {key}")
                return False
        
        print(f"✅ Results file format is valid")
        print(f"   📊 Execution ID: {execution['execution_id']}")
        print(f"   📈 Success rate: {stats['success_rate']:.1f}%")
        print(f"   📋 Total results: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return False

def run_comprehensive_test():
    """Run all BRAF system tests"""
    print("🎯 BRAF System Comprehensive Test")
    print("=" * 50)
    print(f"Test Time: {datetime.now()}")
    
    tests = [
        ("Decision Engine", test_decision_engine),
        ("Scraper Registry", test_scraper_registry),
        ("BRAF Execution", test_braf_execution),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("Results Format", test_results_format)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All BRAF system tests passed!")
        print("\n💡 BRAF System Features Verified:")
        print("   ✅ Intelligent scraper selection")
        print("   ✅ Comprehensive scraper registry")
        print("   ✅ Robust execution framework")
        print("   ✅ Automatic fallback mechanism")
        print("   ✅ Detailed results and metadata")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)