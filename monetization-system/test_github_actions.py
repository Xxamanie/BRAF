#!/usr/bin/env python3
"""
Test GitHub Actions Scraper Locally
Simulates the GitHub Actions environment for testing
"""
import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def simulate_github_env():
    """Simulate GitHub Actions environment variables"""
    os.environ['GITHUB_RUN_NUMBER'] = '123'
    os.environ['GITHUB_REPOSITORY'] = 'test/braf-scraper'
    os.environ['GITHUB_REF'] = 'refs/heads/main'

def test_scraper():
    """Test the scraper script locally"""
    print("🧪 Testing GitHub Actions Scraper Locally")
    print("=" * 50)
    
    # Simulate GitHub environment
    simulate_github_env()
    
    # Import and run the scraper
    try:
        import scraper
        exit_code = scraper.main()
        
        if exit_code == 0:
            print("\n✅ Scraper test completed successfully!")
            
            # Check if results file was created
            if os.path.exists("data/results.json"):
                with open("data/results.json", 'r') as f:
                    results = json.load(f)
                
                print(f"📊 Results file created with {len(results.get('results', []))} results")
                
                # Show GitHub Actions metadata
                gh_metadata = results.get('github_actions_execution', {})
                print(f"🤖 GitHub Actions metadata:")
                print(f"   Workflow run: {gh_metadata.get('workflow_run')}")
                print(f"   Repository: {gh_metadata.get('repository')}")
                print(f"   Timestamp: {gh_metadata.get('timestamp')}")
                
            else:
                print("⚠️  No results file found")
                
        else:
            print(f"❌ Scraper test failed with exit code: {exit_code}")
            
        return exit_code
        
    except Exception as e:
        print(f"❌ Error testing scraper: {e}")
        import traceback
        traceback.print_exc()
        return 1

def test_targets_config():
    """Test the targets configuration"""
    print("\n🎯 Testing Targets Configuration")
    print("-" * 30)
    
    if os.path.exists("scraper_targets.json"):
        with open("scraper_targets.json", 'r') as f:
            config = json.load(f)
        
        targets = config.get('targets', [])
        settings = config.get('settings', {})
        
        print(f"📋 Found {len(targets)} targets:")
        for i, target in enumerate(targets, 1):
            url = target.get('url', 'unknown')
            desc = target.get('description', 'No description')
            scraper = target.get('preferred_scraper', 'auto')
            print(f"   {i}. {url}")
            print(f"      Description: {desc}")
            print(f"      Scraper: {scraper}")
        
        print(f"\n⚙️  Settings:")
        for key, value in settings.items():
            print(f"   {key}: {value}")
            
    else:
        print("❌ No scraper_targets.json found")

def main():
    """Main test function"""
    print("🚀 BRAF GitHub Actions Test Suite")
    print("=" * 60)
    
    # Test configuration
    test_targets_config()
    
    # Test scraper
    exit_code = test_scraper()
    
    print(f"\n🏁 Test completed with exit code: {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
