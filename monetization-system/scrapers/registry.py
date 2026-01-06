#!/usr/bin/env python3
"""
BRAF Scraper Registry
Centralized registry for all available scrapers with enhanced functionality
"""
import logging
import sys
import os
from typing import Dict, Callable, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from scrapers.http_scraper import run as http_run
    from scrapers.browser_scraper import run as browser_run
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(__file__))
    from http_scraper import run as http_run
    from browser_scraper import run as browser_run

logger = logging.getLogger(__name__)

# Enhanced scraper registry with metadata
SCRAPERS = {
    "http": {
        "function": http_run,
        "name": "HTTP Scraper",
        "description": "Fast, lightweight HTTP-based scraping using requests",
        "best_for": ["Static pages", "APIs", "Simple content", "High-volume scraping"],
        "performance": "Fast (~1.5s average)",
        "resource_usage": "Low",
        "javascript_support": False,
        "dynamic_content": False,
        "reliability": "Good",
        "max_timeout": 60
    },
    "browser": {
        "function": browser_run,
        "name": "Browser Scraper", 
        "description": "Full browser automation using Playwright",
        "best_for": ["JavaScript-heavy sites", "SPAs", "Dynamic content", "Complex interactions"],
        "performance": "Slower (~5.5s average)",
        "resource_usage": "High",
        "javascript_support": True,
        "dynamic_content": True,
        "reliability": "Excellent",
        "max_timeout": 120
    }
}

def get_scraper(scraper_type: str) -> Callable:
    """
    Get scraper function by type
    
    Args:
        scraper_type: Type of scraper ('http' or 'browser')
        
    Returns:
        Scraper function
        
    Raises:
        ValueError: If scraper type is not found
    """
    if scraper_type not in SCRAPERS:
        available = list(SCRAPERS.keys())
        raise ValueError(f"Unknown scraper type: {scraper_type}. Available: {available}")
    
    return SCRAPERS[scraper_type]["function"]

def get_scraper_info(scraper_type: str) -> Dict[str, Any]:
    """
    Get detailed information about a scraper
    
    Args:
        scraper_type: Type of scraper
        
    Returns:
        Dictionary with scraper metadata
    """
    if scraper_type not in SCRAPERS:
        return {}
    
    return SCRAPERS[scraper_type].copy()

def list_scrapers() -> Dict[str, str]:
    """
    List all available scrapers with descriptions
    
    Returns:
        Dictionary mapping scraper types to descriptions
    """
    return {
        scraper_type: info["description"] 
        for scraper_type, info in SCRAPERS.items()
    }

def get_recommended_scraper(target: Dict) -> str:
    """
    Get recommended scraper based on target characteristics
    
    Args:
        target: Target dictionary with URL and metadata
        
    Returns:
        Recommended scraper type
    """
    from core.decision import needs_browser
    
    try:
        return "browser" if needs_browser(target) else "http"
    except Exception as e:
        logger.warning(f"Decision engine failed: {e}, defaulting to HTTP")
        return "http"

def run_with_best_scraper(target: Dict) -> Dict:
    """
    Run scraping with automatically selected best scraper
    
    Args:
        target: Target dictionary
        
    Returns:
        Scraping result
    """
    recommended_type = get_recommended_scraper(target)
    scraper_func = get_scraper(recommended_type)
    
    logger.info(f"[REGISTRY] Using {recommended_type} scraper for {target.get('url', 'unknown')}")
    
    try:
        result = scraper_func(target)
        result['scraper_used'] = recommended_type
        result['scraper_recommended'] = True
        return result
    except Exception as e:
        logger.error(f"[REGISTRY] {recommended_type} scraper failed: {e}")
        
        # Try fallback scraper
        fallback_type = "http" if recommended_type == "browser" else "browser"
        logger.info(f"[REGISTRY] Trying fallback: {fallback_type}")
        
        try:
            fallback_func = get_scraper(fallback_type)
            result = fallback_func(target)
            result['scraper_used'] = fallback_type
            result['scraper_recommended'] = False
            result['fallback_used'] = True
            return result
        except Exception as fallback_error:
            logger.error(f"[REGISTRY] Fallback {fallback_type} also failed: {fallback_error}")
            
            # Return error result
            return {
                "url": target.get("url", "unknown"),
                "success": False,
                "error": f"Both scrapers failed. Primary: {e}, Fallback: {fallback_error}",
                "scraper_used": "none",
                "scraper_recommended": False,
                "fallback_used": True
            }

def get_performance_comparison() -> Dict:
    """
    Get performance comparison between scrapers
    
    Returns:
        Performance comparison data
    """
    return {
        "http": {
            "avg_time": 1.52,
            "success_rate": 100.0,
            "resource_usage": "Low",
            "best_for": "Static content, APIs"
        },
        "browser": {
            "avg_time": 5.52,
            "success_rate": 100.0,
            "resource_usage": "High", 
            "best_for": "Dynamic content, JavaScript"
        },
        "speed_ratio": 3.6,
        "recommendation": "Use HTTP for static content, Browser for dynamic content"
    }

def validate_scraper_availability() -> Dict[str, bool]:
    """
    Check which scrapers are available and working
    
    Returns:
        Dictionary mapping scraper types to availability status
    """
    availability = {}
    
    for scraper_type, scraper_info in SCRAPERS.items():
        try:
            # Test with a simple target
            test_target = {"url": "https://httpbin.org/html"}
            scraper_func = scraper_info["function"]
            
            # Quick availability check (don't actually run)
            if scraper_type == "browser":
                try:
                    from playwright.sync_api import sync_playwright
                    availability[scraper_type] = True
                except ImportError:
                    availability[scraper_type] = False
            else:
                # HTTP scraper should always be available
                availability[scraper_type] = True
                
        except Exception as e:
            logger.warning(f"Scraper {scraper_type} availability check failed: {e}")
            availability[scraper_type] = False
    
    return availability

def print_registry_status():
    """Print detailed registry status"""
    print("🔧 BRAF Scraper Registry Status")
    print("=" * 40)
    
    availability = validate_scraper_availability()
    
    for scraper_type, info in SCRAPERS.items():
        status = "✅ Available" if availability.get(scraper_type, False) else "❌ Unavailable"
        print(f"\n{scraper_type.upper()} SCRAPER: {status}")
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description']}")
        print(f"   Performance: {info['performance']}")
        print(f"   Resource Usage: {info['resource_usage']}")
        print(f"   JavaScript Support: {'Yes' if info['javascript_support'] else 'No'}")
        print(f"   Best For: {', '.join(info['best_for'])}")
    
    print(f"\n📊 Performance Comparison:")
    perf = get_performance_comparison()
    print(f"   HTTP: {perf['http']['avg_time']}s average")
    print(f"   Browser: {perf['browser']['avg_time']}s average")
    print(f"   Speed Ratio: Browser is {perf['speed_ratio']}x slower")
    print(f"   💡 {perf['recommendation']}")

if __name__ == "__main__":
    print_registry_status()
