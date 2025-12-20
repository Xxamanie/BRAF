#!/usr/bin/env python3
"""
Test the simple runner interface with your exact code
"""
from core.runner import run_targets

TARGETS = [
    {"url": "https://example.com", "requires_js": False}
]

if __name__ == "__main__":
    results = run_targets(TARGETS)
    print(f"\n🎯 Final Results:")
    for result in results:
        print(f"   URL: {result.get('url')}")
        print(f"   Success: {result.get('success')}")
        print(f"   Scraper: {result.get('scraper_used')}")
        if result.get('decision_explanation'):
            explanation = result['decision_explanation']
            print(f"   Decision: {explanation.get('decision')} (confidence: {explanation.get('confidence', 0):.3f})")