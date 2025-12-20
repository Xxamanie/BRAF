#!/usr/bin/env python3
"""
BRAF Local Runner
Local execution runner for BRAF system with enhanced capabilities
"""
import sys
import os
from pathlib import Path

# Add monetization-system to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'monetization-system'))

try:
    from core.runner import run_targets
    from enhanced_braf_runner_fixed import EnhancedBRAFRunner
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    """Main local runner execution"""
    print("🚀 BRAF Local Runner")
    print("=" * 50)
    
    # Default targets for local testing
    default_targets = [
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
        },
        {
            "name": "Quotes SPA",
            "url": "https://quotes.toscrape.com/js/",
            "requires_js": True
        },
        {
            "name": "Hacker News",
            "url": "https://news.ycombinator.com",
            "requires_js": False
        }
    ]
    
    print("📋 Default Targets:")
    for i, target in enumerate(default_targets, 1):
        js_indicator = "🖥️ JS" if target.get('requires_js') else "🌐 HTTP"
        print(f"   {i}. {target['name']} ({js_indicator})")
        print(f"      {target['url']}")
    
    print(f"\n🎯 Processing {len(default_targets)} targets...")
    
    # Run the targets
    results = run_targets(default_targets)
    
    # Display detailed results
    print("\n📊 Execution Results:")
    print("=" * 50)
    
    successful = 0
    total_time = 0
    http_count = 0
    browser_count = 0
    
    for result in results:
        name = result.get('name', 'Unknown')
        url = result.get('url', 'Unknown')
        success = result.get('success', False)
        scraper = result.get('scraper_used', 'unknown')
        exec_time = result.get('execution_time', 0)
        word_count = result.get('word_count', 0)
        
        if success:
            successful += 1
        
        total_time += exec_time
        
        if scraper == 'http':
            http_count += 1
        elif scraper == 'browser':
            browser_count += 1
        
        status = "✅" if success else "❌"
        
        print(f"\n{status} {name}")
        print(f"   URL: {url}")
        print(f"   Scraper: {scraper.upper()}")
        print(f"   Time: {exec_time:.2f}s")
        print(f"   Words: {word_count}")
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
        
        # Show ML decision info if available
        if 'decision_explanation' in result:
            decision = result['decision_explanation']
            confidence = decision.get('confidence', 0) * 100
            print(f"   🧠 ML Confidence: {confidence:.1f}%")
    
    # Summary statistics
    success_rate = (successful / len(results)) * 100 if results else 0
    
    print(f"\n📈 Summary Statistics:")
    print("=" * 50)
    print(f"✅ Success Rate: {success_rate:.1f}% ({successful}/{len(results)})")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"🌐 HTTP Scraper: {http_count} targets")
    print(f"🖥️  Browser Scraper: {browser_count} targets")
    
    if total_time > 0:
        avg_time = total_time / len(results)
        print(f"📊 Average Time: {avg_time:.2f}s per target")
    
    # Check for enhanced results file
    results_file = project_root / 'monetization-system' / 'data' / 'enhanced_results.json'
    if results_file.exists():
        print(f"\n💾 Enhanced results saved to: {results_file}")
        print("🌐 View dashboard: http://localhost:8081/dashboard/")
    
    print(f"\n🎉 Local runner execution completed!")
    return results

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n🛑 Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)