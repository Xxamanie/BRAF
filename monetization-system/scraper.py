#!/usr/bin/env python3
"""
BRAF GitHub Actions Scraper
Main scraper script for automated GitHub Actions execution
"""
import os
import sys
import json
from datetime import datetime
from typing import List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def load_targets() -> List[Dict]:
    """Load scraping targets from configuration"""
    
    # Default targets if no config file exists
    default_targets = [
        {"url": "https://httpbin.org/html", "description": "Test HTML page"},
        {"url": "https://example.com", "description": "Example domain"},
        {"url": "https://jsonplaceholder.typicode.com/posts/1", "description": "JSON API test"},
        {"url": "https://news.ycombinator.com", "description": "Hacker News"},
        {"url": "https://httpbin.org/json", "description": "JSON endpoint"},
    ]
    
    # Try to load from config file
    config_file = "scraper_targets.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('targets', default_targets)
        except Exception as e:
            print(f"⚠️  Error loading config file: {e}")
            print("📋 Using default targets")
    
    return default_targets

def main():
    """Main scraper execution"""
    print("🚀 BRAF GitHub Actions Scraper")
    print("=" * 50)
    print(f"⏰ Execution time: {datetime.now().isoformat()}")
    
    # Load targets
    targets = load_targets()
    print(f"📋 Loaded {len(targets)} targets")
    
    # Import and run BRAF
    try:
        from core.runner import run_targets
        print("✨ Using Enhanced BRAF Runner")
        
        # Execute scraping
        results = run_targets(targets)
        
        # Save results with timestamp
        results_with_metadata = {
            "github_actions_execution": {
                "timestamp": datetime.now().isoformat(),
                "targets_processed": len(targets),
                "results_count": len(results),
                "workflow_run": os.environ.get('GITHUB_RUN_NUMBER', 'local'),
                "repository": os.environ.get('GITHUB_REPOSITORY', 'local'),
                "ref": os.environ.get('GITHUB_REF', 'local')
            },
            "targets": targets,
            "results": results
        }
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Save to results.json for GitHub Actions
        with open("data/results.json", "w") as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        # Show summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📊 Execution Summary:")
        print(f"   ✅ Successful: {successful}/{len(results)}")
        print(f"   📈 Success rate: {(successful/len(results)*100):.1f}%")
        print(f"   💾 Results saved to: data/results.json")
        
        # Show individual results
        print(f"\n📋 Individual Results:")
        for result in results:
            url = result.get('url', 'unknown')
            success = result.get('success', False)
            scraper = result.get('scraper_used', 'unknown')
            status = "✅" if success else "❌"
            print(f"   {status} {url} ({scraper})")
        
        print(f"\n🎉 BRAF scraping completed successfully!")
        return 0
        
    except ImportError as e:
        print(f"❌ Error importing BRAF: {e}")
        print("💡 Make sure all dependencies are installed")
        return 1
        
    except Exception as e:
        print(f"❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)