#!/usr/bin/env python3
"""
Test BRAF with custom targets including JavaScript-rendered content
"""
import sys
import os

# Add monetization-system to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'monetization-system'))

from core.runner import run_targets

# Your custom targets
TARGETS = [
    {
        "name": "Example Static",
        "url": "https://example.com",
        "requires_js": False
    },
    {
        "name": "Quotes SPA",
        "url": "https://quotes.toscrape.com/js/",
        "requires_js": True
    }
]

def main():
    print("🎯 Testing BRAF with Custom Targets")
    print("=" * 50)
    
    for i, target in enumerate(TARGETS, 1):
        print(f"{i}. {target['name']}")
        print(f"   URL: {target['url']}")
        print(f"   Requires JS: {target['requires_js']}")
    
    print("\n🚀 Starting BRAF execution...")
    
    # Run the targets
    results = run_targets(TARGETS)
    
    print("\n📊 Detailed Results:")
    print("=" * 50)
    
    for result in results:
        name = result.get('name', 'Unknown')
        url = result.get('url', 'Unknown')
        success = result.get('success', False)
        scraper = result.get('scraper_used', 'unknown')
        exec_time = result.get('execution_time', 0)
        word_count = result.get('word_count', 0)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        
        print(f"\n🎯 {name}")
        print(f"   Status: {status}")
        print(f"   URL: {url}")
        print(f"   Scraper: {scraper.upper()}")
        print(f"   Time: {exec_time:.2f}s")
        print(f"   Words: {word_count}")
        
        if result.get('error'):
            print(f"   Error: {result['error']}")
        
        # Show decision explanation if available
        if 'decision_explanation' in result:
            decision = result['decision_explanation']
            confidence = decision.get('confidence', 0) * 100
            print(f"   ML Confidence: {confidence:.1f}%")
            
            if 'features' in decision:
                features = decision['features']
                print(f"   Features: JS={features.get('has_js_indicators', False)}, "
                      f"SPA={features.get('is_spa', False)}, "
                      f"Static={features.get('is_static_content', False)}")

if __name__ == "__main__":
    main()
