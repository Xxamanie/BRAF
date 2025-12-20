#!/usr/bin/env python3
"""
BRAF GitHub Actions Scraper
Main scraper script for GitHub Actions automation
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.runner import run_targets

def load_targets():
    """Load targets from configuration file"""
    targets_file = Path(__file__).parent / 'scraper_targets.json'
    
    if targets_file.exists():
        with open(targets_file, 'r') as f:
            data = json.load(f)
            return data.get('targets', [])
    
    # Default targets if file doesn't exist
    return [
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

def save_results(results, targets_processed):
    """Save results to JSON file"""
    # Create data directory
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Prepare results data
    results_data = {
        "github_actions_execution": {
            "timestamp": datetime.now().isoformat(),
            "targets_processed": targets_processed,
            "workflow_run": os.environ.get('GITHUB_RUN_ID', 'local')
        },
        "results": results
    }
    
    # Save to file
    results_file = data_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    return results_file

def main():
    """Main execution function"""
    print("🚀 BRAF GitHub Actions Scraper")
    print("=" * 50)
    print(f"⏰ Execution time: {datetime.now().isoformat()}")
    
    # Load targets
    targets = load_targets()
    print(f"📋 Loaded {len(targets)} targets")
    
    # Display targets
    for i, target in enumerate(targets, 1):
        js_indicator = "🖥️ JS" if target.get('requires_js') else "🌐 HTTP"
        print(f"   {i}. {target.get('name', 'Unknown')} ({js_indicator})")
    
    # Run scraping
    print(f"\n🎯 Starting scraping process...")
    results = run_targets(targets)
    
    # Calculate statistics
    successful = sum(1 for r in results if r.get('success', False))
    success_rate = (successful / len(results)) * 100 if results else 0
    
    # Save results
    save_results(results, len(targets))
    
    # Display summary
    print(f"\n📊 Execution Summary:")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    print(f"   💾 Results saved to: data/results.json")
    
    # Display individual results
    print(f"\n📋 Individual Results:")
    for result in results:
        url = result.get('url', 'unknown')
        success = result.get('success', False)
        scraper = result.get('scraper_used', 'unknown')
        print(f"   {'✅' if success else '❌'} {url} ({scraper})")
    
    print(f"\n🎉 BRAF scraping completed successfully!")
    
    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)