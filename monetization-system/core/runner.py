#!/usr/bin/env python3
"""
BRAF Simple Runner Interface
Simple interface for running BRAF with targets
"""
import sys
import os
from typing import List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from enhanced_braf_runner_fixed import EnhancedBRAFRunner
    ENHANCED_AVAILABLE = True
except ImportError:
    try:
        from braf_runner import run_targets as basic_run_targets
        ENHANCED_AVAILABLE = False
    except ImportError:
        print("❌ No BRAF runner available. Please ensure the system is properly set up.")
        sys.exit(1)

def run_targets(targets: List[Dict]) -> List[Dict]:
    """
    Simple interface to run BRAF with targets
    
    Args:
        targets: List of target dictionaries with 'url' and optional metadata
        
    Returns:
        List of scraping results
    """
    print(f"🚀 BRAF Runner - Processing {len(targets)} targets")
    
    # Convert legacy format if needed
    processed_targets = []
    for target in targets:
        if isinstance(target, dict):
            processed_target = target.copy()
            
            # Handle legacy 'requires_js' field
            if 'requires_js' in processed_target:
                requires_js = processed_target.pop('requires_js')
                if requires_js:
                    processed_target['preferred_scraper'] = 'browser'
                else:
                    processed_target['preferred_scraper'] = 'http'
            
            processed_targets.append(processed_target)
        else:
            # Handle string URLs
            processed_targets.append({"url": str(target)})
    
    # Use enhanced runner if available
    if ENHANCED_AVAILABLE:
        print("✨ Using Enhanced BRAF Runner with machine learning")
        
        runner = EnhancedBRAFRunner(
            max_workers=4,
            max_browser_workers=2,
            enable_analytics=True
        )
        
        def progress_callback(progress: float, result: Dict):
            scraper = result.get('scraper_used', 'unknown')
            success = "✅" if result.get('success', False) else "❌"
            url = result.get('url', 'unknown')
            print(f"   📈 {progress:.1f}% - {success} {scraper.upper()} - {url}")
        
        results = runner.run_enhanced(
            processed_targets,
            parallel=len(processed_targets) > 1,
            progress_callback=progress_callback,
            save_results=True
        )
        
        # Show summary
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📊 Results: {successful}/{len(results)} successful")
        
        if hasattr(runner, 'execution_stats'):
            stats = runner.execution_stats
            if stats.get('parallel_speedup', 0) > 1:
                print(f"⚡ Parallel speedup: {stats['parallel_speedup']:.2f}x")
        
        return results
    
    else:
        print("📝 Using Basic BRAF Runner")
        return basic_run_targets(processed_targets)

def main():
    """Demo usage of the runner"""
    # Example targets
    targets = [
        {"url": "https://example.com", "requires_js": False},
        {"url": "https://httpbin.org/html"},
        {"url": "https://jsonplaceholder.typicode.com/posts/1"},
    ]
    
    results = run_targets(targets)
    
    print(f"\n🎯 Demo completed with {len(results)} results")
    for result in results:
        url = result.get('url', 'unknown')
        success = result.get('success', False)
        scraper = result.get('scraper_used', 'unknown')
        print(f"   {'✅' if success else '❌'} {url} ({scraper})")

if __name__ == "__main__":
    main()