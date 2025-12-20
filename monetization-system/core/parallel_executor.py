#!/usr/bin/env python3
"""
BRAF Parallel Executor
High-performance parallel processing for multiple targets with intelligent load balancing
"""
import asyncio
import concurrent.futures
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
import threading
from queue import Queue
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.enhanced_decision import enhanced_engine
from scrapers.registry import SCRAPERS

logger = logging.getLogger(__name__)

class ParallelExecutor:
    """High-performance parallel executor for BRAF operations"""
    
    def __init__(self, max_workers: int = 4, max_browser_workers: int = 2):
        """
        Initialize parallel executor
        
        Args:
            max_workers: Maximum total worker threads
            max_browser_workers: Maximum browser workers (resource intensive)
        """
        self.max_workers = max_workers
        self.max_browser_workers = max_browser_workers
        self.results_queue = Queue()
        self.stats = {
            'total_targets': 0,
            'completed': 0,
            'successful': 0,
            'failed': 0,
            'http_used': 0,
            'browser_used': 0,
            'start_time': None,
            'end_time': None
        }
        
    def execute_parallel(self, targets: List[Dict], progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Execute targets in parallel with intelligent load balancing
        
        Args:
            targets: List of target dictionaries
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of results
        """
        logger.info(f"[PARALLEL] Starting parallel execution of {len(targets)} targets")
        
        self.stats['total_targets'] = len(targets)
        self.stats['start_time'] = datetime.now()
        
        # Separate targets by scraper type for load balancing
        http_targets = []
        browser_targets = []
        
        for i, target in enumerate(targets):
            target['_index'] = i  # Track original order
            
            if enhanced_engine.needs_browser(target):
                browser_targets.append(target)
            else:
                http_targets.append(target)
        
        logger.info(f"[PARALLEL] Load distribution: {len(http_targets)} HTTP, {len(browser_targets)} Browser")
        
        results = [None] * len(targets)  # Pre-allocate results array
        
        # Use ThreadPoolExecutor for better resource management
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit HTTP tasks (can run more concurrently)
            http_futures = []
            for target in http_targets:
                future = executor.submit(self._execute_single, target, 'http')
                http_futures.append((future, target['_index']))
            
            # Submit browser tasks with limited concurrency
            browser_futures = []
            browser_semaphore = threading.Semaphore(self.max_browser_workers)
            
            for target in browser_targets:
                future = executor.submit(self._execute_single_with_semaphore, 
                                       target, 'browser', browser_semaphore)
                browser_futures.append((future, target['_index']))
            
            # Collect results as they complete
            all_futures = http_futures + browser_futures
            
            for future in concurrent.futures.as_completed([f[0] for f in all_futures]):
                # Find the original index for this future
                original_index = 0
                for f, idx in all_futures:
                    if f == future:
                        original_index = idx
                        break
                
                try:
                    result = future.result()
                    results[original_index] = result
                    
                    self.stats['completed'] += 1
                    
                    if result.get('success', False):
                        self.stats['successful'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                    # Track scraper usage
                    scraper_used = result.get('scraper_used', 'unknown')
                    if scraper_used == 'http':
                        self.stats['http_used'] += 1
                    elif scraper_used == 'browser':
                        self.stats['browser_used'] += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress = (self.stats['completed'] / self.stats['total_targets']) * 100
                        progress_callback(progress, result)
                    
                    logger.info(f"[PARALLEL] Completed {self.stats['completed']}/{self.stats['total_targets']} "
                               f"({scraper_used}): {result.get('url', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"[PARALLEL] Task failed: {e}")
                    self.stats['failed'] += 1
                    
                    # Create error result
                    error_result = {
                        'url': 'unknown',
                        'success': False,
                        'error': f"Parallel execution error: {str(e)}",
                        'scraper_used': 'none'
                    }
                    results[original_index] = error_result
        
        self.stats['end_time'] = datetime.now()
        
        # Remove None results (shouldn't happen, but safety check)
        results = [r for r in results if r is not None]
        
        # Log final statistics
        execution_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        success_rate = (self.stats['successful'] / self.stats['total_targets']) * 100
        
        logger.info(f"[PARALLEL] Execution completed in {execution_time:.2f}s")
        logger.info(f"[PARALLEL] Success rate: {success_rate:.1f}% ({self.stats['successful']}/{self.stats['total_targets']})")
        logger.info(f"[PARALLEL] Scraper usage: HTTP={self.stats['http_used']}, Browser={self.stats['browser_used']}")
        
        return results
    
    def _execute_single_with_semaphore(self, target: Dict, scraper_type: str, semaphore: threading.Semaphore) -> Dict:
        """Execute single target with semaphore for resource limiting"""
        with semaphore:
            return self._execute_single(target, scraper_type)
    
    def _execute_single(self, target: Dict, scraper_type: str) -> Dict:
        """Execute single target with specified scraper"""
        start_time = time.time()
        url = target.get('url', 'unknown')
        
        try:
            # Get scraper function
            scraper_func = SCRAPERS[scraper_type]["function"]
            
            # Execute scraping
            result = scraper_func(target)
            
            execution_time = time.time() - start_time
            
            # Add metadata
            result['scraper_used'] = scraper_type
            result['execution_time'] = execution_time
            result['parallel_execution'] = True
            result['processed_at'] = datetime.now().isoformat()
            
            # Update performance data for learning
            enhanced_engine.update_performance(
                url, scraper_type, result.get('success', False), execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[PARALLEL] Error executing {scraper_type} scraper for {url}: {e}")
            
            # Update performance data for failed execution
            enhanced_engine.update_performance(url, scraper_type, False, execution_time)
            
            return {
                'url': url,
                'success': False,
                'error': f"{scraper_type} scraper error: {str(e)}",
                'scraper_used': scraper_type,
                'execution_time': execution_time,
                'parallel_execution': True,
                'processed_at': datetime.now().isoformat()
            }
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            stats['total_execution_time'] = (stats['end_time'] - stats['start_time']).total_seconds()
            stats['success_rate'] = (stats['successful'] / stats['total_targets']) * 100 if stats['total_targets'] > 0 else 0
            stats['targets_per_second'] = stats['total_targets'] / stats['total_execution_time'] if stats['total_execution_time'] > 0 else 0
        
        return stats

class AsyncParallelExecutor:
    """Async version of parallel executor for even better performance"""
    
    def __init__(self, max_concurrent: int = 10, max_browser_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.max_browser_concurrent = max_browser_concurrent
        self.browser_semaphore = None
        
    async def execute_async(self, targets: List[Dict]) -> List[Dict]:
        """Execute targets asynchronously"""
        logger.info(f"[ASYNC] Starting async execution of {len(targets)} targets")
        
        # Create semaphore for browser tasks
        self.browser_semaphore = asyncio.Semaphore(self.max_browser_concurrent)
        
        # Create tasks
        tasks = []
        for i, target in enumerate(targets):
            target['_index'] = i
            task = asyncio.create_task(self._execute_single_async(target))
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[ASYNC] Task {i} failed: {result}")
                processed_results.append({
                    'url': targets[i].get('url', 'unknown'),
                    'success': False,
                    'error': f"Async execution error: {str(result)}",
                    'scraper_used': 'none'
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_async(self, target: Dict) -> Dict:
        """Execute single target asynchronously"""
        scraper_type = 'browser' if enhanced_engine.needs_browser(target) else 'http'
        
        if scraper_type == 'browser':
            async with self.browser_semaphore:
                return await self._run_in_executor(target, scraper_type)
        else:
            return await self._run_in_executor(target, scraper_type)
    
    async def _run_in_executor(self, target: Dict, scraper_type: str) -> Dict:
        """Run scraper in thread executor"""
        loop = asyncio.get_event_loop()
        
        def sync_execute():
            executor = ParallelExecutor()
            return executor._execute_single(target, scraper_type)
        
        return await loop.run_in_executor(None, sync_execute)

def run_parallel_demo():
    """Demonstrate parallel execution capabilities"""
    print("🚀 BRAF Parallel Execution Demo")
    print("=" * 40)
    
    # Sample targets for parallel processing
    targets = [
        {"url": "https://httpbin.org/html", "description": "Simple HTML"},
        {"url": "https://example.com", "description": "Static site"},
        {"url": "https://jsonplaceholder.typicode.com/posts/1", "description": "JSON API"},
        {"url": "https://news.ycombinator.com", "description": "News site"},
        {"url": "https://httpbin.org/json", "description": "JSON endpoint"},
        {"url": "https://httpbin.org/xml", "description": "XML endpoint"},
        # Add some browser targets
        {"url": "https://app.example.com/dashboard", "preferred_scraper": "browser"},
        {"url": "https://dashboard.example.com", "preferred_scraper": "browser"},
    ]
    
    print(f"📋 Processing {len(targets)} targets in parallel:")
    for target in targets:
        print(f"   • {target['url']} - {target.get('description', 'No description')}")
    
    # Progress callback
    def progress_callback(progress: float, result: Dict):
        print(f"   📈 Progress: {progress:.1f}% - {result.get('url', 'unknown')}")
    
    # Execute in parallel
    executor = ParallelExecutor(max_workers=6, max_browser_workers=2)
    
    print(f"\n🔄 Running parallel execution...")
    start_time = time.time()
    
    results = executor.execute_parallel(targets, progress_callback)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Show results
    stats = executor.get_statistics()
    
    print(f"\n📊 Parallel Execution Results:")
    print(f"   ⏱️  Total time: {execution_time:.2f}s")
    print(f"   ✅ Successful: {stats['successful']}/{stats['total_targets']}")
    print(f"   📈 Success rate: {stats.get('success_rate', 0):.1f}%")
    print(f"   🌐 HTTP used: {stats['http_used']}")
    print(f"   🖥️  Browser used: {stats['browser_used']}")
    print(f"   🚀 Targets/second: {stats.get('targets_per_second', 0):.2f}")
    
    # Compare with sequential execution estimate
    avg_http_time = 1.5  # seconds
    avg_browser_time = 5.5  # seconds
    
    estimated_sequential = (stats['http_used'] * avg_http_time + 
                           stats['browser_used'] * avg_browser_time)
    
    speedup = estimated_sequential / execution_time if execution_time > 0 else 1
    
    print(f"\n💡 Performance Improvement:")
    print(f"   📏 Estimated sequential time: {estimated_sequential:.2f}s")
    print(f"   ⚡ Parallel speedup: {speedup:.2f}x faster")

async def run_async_demo():
    """Demonstrate async parallel execution"""
    print("\n🌟 BRAF Async Parallel Execution Demo")
    print("=" * 40)
    
    targets = [
        {"url": "https://httpbin.org/html"},
        {"url": "https://example.com"},
        {"url": "https://jsonplaceholder.typicode.com/posts/1"},
        {"url": "https://httpbin.org/json"},
    ]
    
    executor = AsyncParallelExecutor(max_concurrent=8, max_browser_concurrent=2)
    
    print(f"🔄 Running async execution of {len(targets)} targets...")
    start_time = time.time()
    
    results = await executor.execute_async(targets)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    successful = sum(1 for r in results if r.get('success', False))
    
    print(f"📊 Async Execution Results:")
    print(f"   ⏱️  Total time: {execution_time:.2f}s")
    print(f"   ✅ Successful: {successful}/{len(targets)}")
    print(f"   🚀 Targets/second: {len(targets)/execution_time:.2f}")

if __name__ == "__main__":
    # Run parallel demo
    run_parallel_demo()
    
    # Run async demo
    print("\n" + "="*50)
    asyncio.run(run_async_demo())