#!/usr/bin/env python3
"""
BRAF Runner - Browser Automation Framework
Intelligent scraping with automatic method selection and comprehensive logging
"""
import json
import os
import logging
from datetime import datetime
from typing import List, Dict
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from core.decision import needs_browser, get_decision_explanation
from scrapers.registry import SCRAPERS, get_scraper, run_with_best_scraper

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_FILE = os.path.join(DATA_DIR, 'results.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'braf.log')

# Setup logging
def setup_logging():
    """Setup BRAF logging"""
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_targets(targets: List[Dict]) -> List[Dict]:
    """
    BRAF main execution function
    Processes targets with intelligent scraper selection
    
    Args:
        targets: List of target dictionaries with URLs and metadata
        
    Returns:
        List of scraping results
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"[BRAF] Starting execution with {len(targets)} targets")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    results = []
    stats = {
        'total_targets': len(targets),
        'successful': 0,
        'failed': 0,
        'http_used': 0,
        'browser_used': 0,
        'fallback_used': 0,
        'start_time': datetime.now().isoformat()
    }
    
    for i, target in enumerate(targets, 1):
        url = target.get('url', 'unknown')
        logger.info(f"[BRAF] Processing target {i}/{len(targets)}: {url}")
        
        try:
            # Get decision explanation for logging
            explanation = get_decision_explanation(target)
            logger.info(f"[BRAF] Decision analysis: {explanation['decision']} "
                       f"(confidence: {explanation['confidence']:.2f})")
            
            # Determine scraper type
            scraper_type = "browser" if needs_browser(target) else "http"
            
            # Get and run scraper
            scraper = SCRAPERS[scraper_type]["function"]
            
            logger.info(f"[BRAF] Running {scraper_type} scraper → {url}")
            
            # Execute scraping
            result = scraper(target)
            
            # Add BRAF metadata
            result['braf_metadata'] = {
                'scraper_selected': scraper_type,
                'decision_factors': explanation['factors'],
                'complexity_score': explanation['complexity_score'],
                'processed_at': datetime.now().isoformat(),
                'target_index': i
            }
            
            # Update statistics
            if result.get('success', False):
                stats['successful'] += 1
                logger.info(f"[BRAF] ✅ Success: {result.get('title', 'No title')[:50]}...")
            else:
                stats['failed'] += 1
                logger.warning(f"[BRAF] ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Track scraper usage
            if scraper_type == 'http':
                stats['http_used'] += 1
            else:
                stats['browser_used'] += 1
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"[BRAF] 💥 Critical error processing {url}: {e}", exc_info=True)
            
            # Create error result
            error_result = {
                'url': url,
                'success': False,
                'error': f"BRAF execution error: {str(e)}",
                'braf_metadata': {
                    'scraper_selected': 'none',
                    'error_type': 'execution_error',
                    'processed_at': datetime.now().isoformat(),
                    'target_index': i
                }
            }
            
            results.append(error_result)
            stats['failed'] += 1
    
    # Finalize statistics
    stats['end_time'] = datetime.now().isoformat()
    stats['success_rate'] = (stats['successful'] / stats['total_targets']) * 100 if stats['total_targets'] > 0 else 0
    
    # Log final statistics
    logger.info("[BRAF] Execution completed")
    logger.info(f"[BRAF] 📊 Statistics:")
    logger.info(f"[BRAF]    ✅ Successful: {stats['successful']}")
    logger.info(f"[BRAF]    ❌ Failed: {stats['failed']}")
    logger.info(f"[BRAF]    🌐 HTTP used: {stats['http_used']}")
    logger.info(f"[BRAF]    🖥️  Browser used: {stats['browser_used']}")
    logger.info(f"[BRAF]    📈 Success rate: {stats['success_rate']:.1f}%")
    
    # Save results with metadata
    results_with_metadata = {
        'braf_execution': {
            'version': '1.0',
            'execution_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'statistics': stats
        },
        'results': results
    }
    
    # Write results to file
    try:
        with open(RESULTS_FILE, "w", encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        logger.info(f"[BRAF] 💾 Results saved to: {RESULTS_FILE}")
        
    except Exception as e:
        logger.error(f"[BRAF] Failed to save results: {e}")
    
    return results

def run_targets_with_fallback(targets: List[Dict]) -> List[Dict]:
    """
    Enhanced BRAF runner with automatic fallback
    
    Args:
        targets: List of target dictionaries
        
    Returns:
        List of scraping results with fallback handling
    """
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"[BRAF] Starting enhanced execution with fallback for {len(targets)} targets")
    
    results = []
    
    for i, target in enumerate(targets, 1):
        url = target.get('url', 'unknown')
        logger.info(f"[BRAF] Processing target {i}/{len(targets)}: {url}")
        
        # Use the registry's intelligent runner with fallback
        result = run_with_best_scraper(target)
        
        # Add BRAF metadata
        result['braf_metadata'] = {
            'execution_mode': 'enhanced_with_fallback',
            'processed_at': datetime.now().isoformat(),
            'target_index': i
        }
        
        # Log result
        if result.get('success', False):
            scraper_used = result.get('scraper_used', 'unknown')
            fallback_used = result.get('fallback_used', False)
            status_msg = f"✅ Success with {scraper_used}"
            if fallback_used:
                status_msg += " (fallback)"
            logger.info(f"[BRAF] {status_msg}")
        else:
            logger.warning(f"[BRAF] ❌ Failed: {result.get('error', 'Unknown error')}")
        
        results.append(result)
    
    # Save results
    try:
        with open(RESULTS_FILE, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"[BRAF] 💾 Results saved to: {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"[BRAF] Failed to save results: {e}")
    
    return results

def demo_braf_execution():
    """Demonstrate BRAF execution with sample targets"""
    print("🚀 BRAF Execution Demo")
    print("=" * 30)
    
    # Sample targets with different characteristics
    sample_targets = [
        {
            "url": "https://httpbin.org/html",
            "description": "Simple HTML page"
        },
        {
            "url": "https://example.com",
            "description": "Static website"
        },
        {
            "url": "https://jsonplaceholder.typicode.com/posts/1",
            "description": "JSON API endpoint"
        },
        {
            "url": "https://news.ycombinator.com",
            "description": "News aggregator"
        }
    ]
    
    print(f"📋 Processing {len(sample_targets)} sample targets:")
    for target in sample_targets:
        print(f"   • {target['url']} - {target['description']}")
    
    print("\n🔄 Running BRAF execution...")
    
    # Run with standard BRAF
    results = run_targets(sample_targets)
    
    print(f"\n📊 Execution Results:")
    successful = sum(1 for r in results if r.get('success', False))
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   📁 Results saved to: {RESULTS_FILE}")
    
    # Show decision breakdown
    http_count = sum(1 for r in results if r.get('braf_metadata', {}).get('scraper_selected') == 'http')
    browser_count = sum(1 for r in results if r.get('braf_metadata', {}).get('scraper_selected') == 'browser')
    
    print(f"   🌐 HTTP scraper used: {http_count}")
    print(f"   🖥️  Browser scraper used: {browser_count}")

def test_decision_accuracy():
    """Test decision engine accuracy"""
    print("\n🧪 Testing Decision Engine Accuracy")
    print("=" * 40)
    
    test_cases = [
        {"url": "https://httpbin.org/html", "expected_scraper": "http"},
        {"url": "https://example.com", "expected_scraper": "http"},
        {"url": "https://api.github.com/users", "expected_scraper": "http"},
        {"url": "https://app.example.com/dashboard", "expected_scraper": "browser"},
        {"url": "https://example.com/#/app", "expected_scraper": "browser"},
    ]
    
    correct = 0
    for case in test_cases:
        decision = "browser" if needs_browser(case) else "http"
        expected = case["expected_scraper"]
        
        if decision == expected:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} {case['url']}")
        print(f"   Decision: {decision}, Expected: {expected}")
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"\n📈 Decision Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    # Run demo
    demo_braf_execution()
    
    # Test decision accuracy
    test_decision_accuracy()
    
    print(f"\n💡 BRAF Features:")
    print(f"   • Intelligent scraper selection based on URL analysis")
    print(f"   • Comprehensive logging and error handling")
    print(f"   • Automatic fallback between scraping methods")
    print(f"   • Detailed execution metadata and statistics")
    print(f"   • JSON results output with full traceability")
