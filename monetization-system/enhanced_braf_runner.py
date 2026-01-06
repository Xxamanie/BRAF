#!/usr/bin/env python3
"""
Enhanced BRAF Runner - Next Generation Browser Automation Framework
Combines intelligent decision-making, parallel processing, and advanced analytics
"""
import json
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from core.enhanced_decision import enhanced_engine
from core.parallel_executor import ParallelExecutor
from core.analytics_engine import BRAFAnalytics
from scrapers.registry import SCRAPERS

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_FILE = os.path.join(DATA_DIR, 'enhanced_results.json')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'enhanced_braf.log')

class EnhancedBRAFRunner:
    """Next-generation BRAF runner with advanced capabilities"""
    
    def __init__(self, max_workers: int = 6, max_browser_workers: int = 2, 
                 enable_analytics: bool = True):
        """
        Initialize Enhanced BRAF Runner
        
        Args:
            max_workers: Maximum parallel workers
            max_browser_workers: Maximum browser workers (resource intensive)
            enable_analytics: Enable analytics and learning
        """
        self.max_workers = max_workers
        self.max_browser_workers = max_browser_workers
        self.enable_analytics = enable_analytics
        
        # Initialize components
        self.parallel_executor = ParallelExecutor(max_workers, max_browser_workers)
        self.analytics = BRAFAnalytics() if enable_analytics else None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Execution statistics
        self.execution_stats = {
            'total_targets': 0,
            'successful': 0,
            'failed': 0,
            'http_used': 0,
            'browser_used': 0,
            'fallback_used': 0,
            'decision_overrides': 0,
            'start_time': None,
            'end_time': None,
            'parallel_speedup': 0
        }
    
    def _setup_logging(self):
        """Setup enhanced logging"""
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
    
    def run_enhanced(self, targets: List[Dict], 
                    parallel: bool = True,
                    progress_callback: Optional[Callable] = None,
                    save_results: bool = True) -> List[Dict]:
        """
        Enhanced BRAF execution with all advanced features
        
        Args:
            targets: List of target dictionaries
            parallel: Enable parallel processing
            progress_callback: Optional progress callback
            save_results: Save results to file
            
        Returns:
            List of enhanced results
        """
        self.logger.info(f"[ENHANCED-BRAF] Starting enhanced execution of {len(targets)} targets")
        self.logger.info(f"[ENHANCED-BRAF] Configuration: parallel={parallel}, analytics={self.enable_analytics}")
        
        # Initialize statistics
        self.execution_stats['total_targets'] = len(targets)
        self.execution_stats['start_time'] = datetime.now()
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Enhanced progress callback
        def enhanced_progress_callback(progress: float, result: Dict):
            # Update statistics
            if result.get('success', False):
                self.execution_stats['successful'] += 1
            else:
                self.execution_stats['failed'] += 1
            
            scraper_used = result.get('scraper_used', 'unknown')
            if scraper_used == 'http':
                self.execution_stats['http_used'] += 1
            elif scraper_used == 'browser':
                self.execution_stats['browser_used'] += 1
            
            if result.get('fallback_used', False):
                self.execution_stats['fallback_used'] += 1
            
            # Record analytics
            if self.analytics:
                recommended_scraper = result.get('recommended_scraper', 'http')
                decision_score = result.get('decision_score', 0.5)
                complexity_score = result.get('complexity_score', 0.0)
                
                self.analytics.record_execution(
                    result, recommended_scraper, decision_score, complexity_score
                )
            
            # Call user callback
            if progress_callback:
                progress_callback(progress, result)
            
            self.logger.info(f"[ENHANCED-BRAF] Progress: {progress:.1f}% - "
                           f"{scraper_used} scraper - {result.get('url', 'unknown')}")
        
        # Execute with appropriate method
        if parallel and len(targets) > 1:
            self.logger.info("[ENHANCED-BRAF] Using parallel execution")
            results = self._run_parallel_enhanced(targets, enhanced_progress_callback)
        else:
            self.logger.info("[ENHANCED-BRAF] Using sequential execution")
            results = self._run_sequential_enhanced(targets, enhanced_progress_callback)
        
        # Finalize statistics
        self.execution_stats['end_time'] = datetime.now()
        execution_time = (self.execution_stats['end_time'] - self.execution_stats['start_time']).total_seconds()
        
        # Calculate parallel speedup estimate
        if parallel:
            estimated_sequential = self._estimate_sequential_time(results)
            self.execution_stats['parallel_speedup'] = estimated_sequential / execution_time if execution_time > 0 else 1
        
        # Log final statistics
        self._log_final_statistics()
        
        # Save results if requested
        if save_results:
            self._save_enhanced_results(results)
        
        return results
    
    def _run_parallel_enhanced(self, targets: List[Dict], progress_callback: Callable) -> List[Dict]:
        """Run targets in parallel with enhanced features"""
        # Pre-process targets with enhanced decision engine
        enhanced_targets = []
        for target in targets:
            enhanced_target = target.copy()
            
            # Get decision explanation
            explanation = enhanced_engine.get_decision_explanation(target)
            enhanced_target['_decision_explanation'] = explanation
            enhanced_target['recommended_scraper'] = explanation['decision']
            enhanced_target['decision_score'] = explanation['total_score']
            enhanced_target['complexity_score'] = explanation['complexity_score']
            
            enhanced_targets.append(enhanced_target)
        
        # Execute in parallel
        results = self.parallel_executor.execute_parallel(enhanced_targets, progress_callback)
        
        # Post-process results
        for i, result in enumerate(results):
            if i < len(enhanced_targets):
                target = enhanced_targets[i]
                result['recommended_scraper'] = target.get('recommended_scraper', 'http')
                result['decision_score'] = target.get('decision_score', 0.5)
                result['complexity_score'] = target.get('complexity_score', 0.0)
                result['decision_explanation'] = target.get('_decision_explanation', {})
        
        return results
    
    def _run_sequential_enhanced(self, targets: List[Dict], progress_callback: Callable) -> List[Dict]:
        """Run targets sequentially with enhanced features"""
        results = []
        
        for i, target in enumerate(targets):
            self.logger.info(f"[ENHANCED-BRAF] Processing target {i+1}/{len(targets)}: {target.get('url', 'unknown')}")
            
            # Get enhanced decision
            explanation = enhanced_engine.get_decision_explanation(target)
            scraper_type = explanation['decision']
            
            # Execute with selected scraper
            try:
                scraper_func = SCRAPERS[scraper_type]["function"]
                
                start_time = time.time()
                result = scraper_func(target)
                execution_time = time.time() - start_time
                
                # Add enhanced metadata
                result['scraper_used'] = scraper_type
                result['execution_time'] = execution_time
                result['recommended_scraper'] = scraper_type
                result['decision_score'] = explanation['total_score']
                result['complexity_score'] = explanation['complexity_score']
                result['decision_explanation'] = explanation
                result['enhanced_execution'] = True
                result['processed_at'] = datetime.now().isoformat()
                
            except Exception as e:
                self.logger.error(f"[ENHANCED-BRAF] Error executing {scraper_type} scraper: {e}")
                
                result = {
                    'url': target.get('url', 'unknown'),
                    'success': False,
                    'error': f"Enhanced execution error: {str(e)}",
                    'scraper_used': scraper_type,
                    'execution_time': time.time() - start_time if 'start_time' in locals() else 0,
                    'enhanced_execution': True,
                    'processed_at': datetime.now().isoformat()
                }
            
            results.append(result)
            
            # Progress callback
            progress = ((i + 1) / len(targets)) * 100
            progress_callback(progress, result)
        
        return results
    
    def _estimate_sequential_time(self, results: List[Dict]) -> float:
        """Estimate sequential execution time for speedup calculation"""
        total_time = 0
        avg_http_time = 1.5  # seconds
        avg_browser_time = 5.5  # seconds
        
        for result in results:
            scraper_used = result.get('scraper_used', 'http')
            if scraper_used == 'browser':
                total_time += avg_browser_time
            else:
                total_time += avg_http_time
        
        return total_time
    
    def _log_final_statistics(self):
        """Log comprehensive final statistics"""
        stats = self.execution_stats
        execution_time = (stats['end_time'] - stats['start_time']).total_seconds()
        success_rate = (stats['successful'] / stats['total_targets']) * 100 if stats['total_targets'] > 0 else 0
        
        self.logger.info("[ENHANCED-BRAF] Execution completed")
        self.logger.info(f"[ENHANCED-BRAF] 📊 Enhanced Statistics:")
        self.logger.info(f"[ENHANCED-BRAF]    ⏱️  Total time: {execution_time:.2f}s")
        self.logger.info(f"[ENHANCED-BRAF]    ✅ Successful: {stats['successful']}")
        self.logger.info(f"[ENHANCED-BRAF]    ❌ Failed: {stats['failed']}")
        self.logger.info(f"[ENHANCED-BRAF]    📈 Success rate: {success_rate:.1f}%")
        self.logger.info(f"[ENHANCED-BRAF]    🌐 HTTP used: {stats['http_used']}")
        self.logger.info(f"[ENHANCED-BRAF]    🖥️  Browser used: {stats['browser_used']}")
        self.logger.info(f"[ENHANCED-BRAF]    🔄 Fallback used: {stats['fallback_used']}")
        
        if stats['parallel_speedup'] > 1:
            self.logger.info(f"[ENHANCED-BRAF]    ⚡ Parallel speedup: {stats['parallel_speedup']:.2f}x")
        
        # Analytics summary
        if self.analytics:
            self.logger.info("[ENHANCED-BRAF]    📊 Analytics enabled - data recorded for learning")
    
    def _save_enhanced_results(self, results: List[Dict]):
        """Save results with enhanced metadata"""
        enhanced_results = {
            'enhanced_braf_execution': {
                'version': '2.0',
                'execution_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'features': {
                    'enhanced_decision_engine': True,
                    'parallel_processing': self.max_workers > 1,
                    'analytics_enabled': self.enable_analytics,
                    'machine_learning': True
                },
                'configuration': {
                    'max_workers': self.max_workers,
                    'max_browser_workers': self.max_browser_workers
                },
                'statistics': self.execution_stats
            },
            'results': results
        }
        
        try:
            with open(RESULTS_FILE, "w", encoding='utf-8') as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            
            self.logger.info(f"[ENHANCED-BRAF] 💾 Enhanced results saved to: {RESULTS_FILE}")
            
        except Exception as e:
            self.logger.error(f"[ENHANCED-BRAF] Failed to save results: {e}")
    
    def get_performance_report(self, days: int = 30) -> Dict:
        """Get comprehensive performance report"""
        if not self.analytics:
            return {'error': 'Analytics not enabled'}
        
        return self.analytics.get_performance_report(days)
    
    def get_optimization_suggestions(self) -> List[Dict]:
        """Get AI-powered optimization suggestions"""
        if not self.analytics:
            return []
        
        return self.analytics.get_optimization_suggestions()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old analytics data"""
        if self.analytics:
            self.analytics.cleanup_old_data(days_to_keep)

def run_enhanced_demo():
    """Demonstrate enhanced BRAF capabilities"""
    print("🚀 Enhanced BRAF Runner Demo")
    print("=" * 50)
    
    # Sample targets with diverse characteristics
    targets = [
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
        },
        {
            "url": "https://httpbin.org/json",
            "description": "JSON endpoint"
        },
        {
            "url": "https://app.example.com/dashboard",
            "description": "App dashboard (forced browser)",
            "preferred_scraper": "browser"
        }
    ]
    
    print(f"📋 Processing {len(targets)} targets with enhanced BRAF:")
    for i, target in enumerate(targets, 1):
        print(f"   {i}. {target['url']} - {target['description']}")
    
    # Progress callback
    def progress_callback(progress: float, result: Dict):
        scraper = result.get('scraper_used', 'unknown')
        success = "✅" if result.get('success', False) else "❌"
        print(f"   📈 {progress:.1f}% - {success} {scraper.upper()} - {result.get('url', 'unknown')}")
    
    # Initialize enhanced runner
    runner = EnhancedBRAFRunner(
        max_workers=4,
        max_browser_workers=2,
        enable_analytics=True
    )
    
    print(f"\n🔄 Running enhanced BRAF execution...")
    start_time = time.time()
    
    # Execute with enhanced features
    results = runner.run_enhanced(
        targets,
        parallel=True,
        progress_callback=progress_callback,
        save_results=True
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Show enhanced results
    print(f"\n📊 Enhanced Execution Results:")
    print(f"   ⏱️  Total time: {execution_time:.2f}s")
    print(f"   ✅ Successful: {runner.execution_stats['successful']}/{runner.execution_stats['total_targets']}")
    print(f"   📈 Success rate: {(runner.execution_stats['successful']/runner.execution_stats['total_targets']*100):.1f}%")
    print(f"   🌐 HTTP used: {runner.execution_stats['http_used']}")
    print(f"   🖥️  Browser used: {runner.execution_stats['browser_used']}")
    
    if runner.execution_stats['parallel_speedup'] > 1:
        print(f"   ⚡ Parallel speedup: {runner.execution_stats['parallel_speedup']:.2f}x")
    
    # Show decision insights
    print(f"\n🧠 Decision Engine Insights:")
    for result in results[:3]:  # Show first 3
        explanation = result.get('decision_explanation', {})
        if explanation:
            print(f"   • {result.get('url', 'unknown')}")
            print(f"     Decision: {explanation.get('decision', 'unknown')} "
                  f"(confidence: {explanation.get('confidence', 0):.2f})")
            print(f"     Score: {explanation.get('total_score', 0):.3f}")
    
    # Performance report
    print(f"\n📈 Performance Report:")
    report = runner.get_performance_report(days=1)
    if 'error' not in report:
        overall = report.get('overall_statistics', {})
        print(f"   Total executions: {overall.get('total_executions', 0)}")
        print(f"   Average HTTP time: {overall.get('avg_http_time', 0):.2f}s")
        print(f"   Average Browser time: {overall.get('avg_browser_time', 0):.2f}s")
    
    # Optimization suggestions
    print(f"\n💡 AI Optimization Suggestions:")
    suggestions = runner.get_optimization_suggestions()
    if suggestions:
        for suggestion in suggestions[:3]:
            print(f"   • {suggestion.get('suggestion', 'No suggestion')}")
            print(f"     Reason: {suggestion.get('reason', 'No reason')}")
    else:
        print("   No specific suggestions at this time")
    
    print(f"\n🎉 Enhanced BRAF Features Demonstrated:")
    print(f"   ✅ Intelligent decision engine with machine learning")
    print(f"   ✅ Parallel processing with load balancing")
    print(f"   ✅ Advanced analytics and performance tracking")
    print(f"   ✅ AI-powered optimization suggestions")
    print(f"   ✅ Comprehensive logging and metadata")

if __name__ == "__main__":
    run_enhanced_demo()
