#!/usr/bin/env python3
"""
Modular Scraper Runner
Supports multiple scraping methods with automatic fallback
"""
import logging
import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import signal
import time

# Add scrapers to path
sys.path.append(os.path.dirname(__file__))

from scrapers import SCRAPERS

# Configuration
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'modular_scraper.log')
ERROR_LOG_FILE = os.path.join(LOG_DIR, 'modular_scraper_errors.log')
STATUS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'modular_scraper_status.json')

# Default URLs to scrape
DEFAULT_URLS = [
    'https://httpbin.org/html',
    'https://example.com',
    'https://jsonplaceholder.typicode.com/posts/1',
]

# Scraping configuration
SCRAPE_CONFIG = {
    'max_pages_per_run': 10,
    'delay_between_pages': 2,
    'timeout_per_page': 30,
    'max_retries': 3,
    'preferred_scraper': 'http',  # 'http' or 'browser'
    'fallback_enabled': True,     # Try other scraper if preferred fails
    'headless': True
}

class ModularScraperRunner:
    """Modular scraper runner supporting multiple scraping methods"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.stats = {
            'pages_scraped': 0,
            'pages_failed': 0,
            'http_used': 0,
            'browser_used': 0,
            'fallback_used': 0,
            'total_content_length': 0,
            'errors': []
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.shutdown_requested = False
        
        # Initialize database
        self.init_database()
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(LOG_DIR, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Error logger
        error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        logging.getLogger().addHandler(error_handler)
    
    def setup_directories(self):
        """Ensure required directories exist"""
        directories = [
            LOG_DIR,
            os.path.join(os.path.dirname(__file__), 'data'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.warning(f"Received signal {signum}, shutting down...")
        self.shutdown_requested = True
    
    def init_database(self):
        """Initialize database for storing results"""
        try:
            from sync_playwright_scraper import SyncPlaywrightScraper
            # Use existing database infrastructure
            self.db_scraper = SyncPlaywrightScraper()
            self.logger.info("✅ Database initialized")
        except ImportError:
            self.logger.warning("⚠️  Database scraper not available")
            self.db_scraper = None
    
    def save_result(self, result: Dict) -> bool:
        """Save scraping result to database"""
        if not self.db_scraper:
            return False
        
        try:
            # Convert to ScrapedResult format
            from sync_playwright_scraper import ScrapedResult
            
            scraped_result = ScrapedResult(
                url=result['url'],
                domain=result['domain'],
                title=result['title'],
                content=result['content'],
                word_count=result['word_count'],
                scraped_at=result['scraped_at'],
                data_hash=result['data_hash'],
                success=result['success'],
                error=result.get('error')
            )
            
            return self.db_scraper.save_result(scraped_result)
            
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
            return False
    
    def scrape_with_method(self, target: Dict, method: str) -> Dict:
        """Scrape using specific method"""
        if method not in SCRAPERS:
            return {
                "success": False,
                "error": f"Unknown scraper method: {method}"
            }
        
        try:
            scraper_func = SCRAPERS[method]
            result = scraper_func(target)
            
            # Track which scraper was used
            if method == 'http':
                self.stats['http_used'] += 1
            elif method == 'browser':
                self.stats['browser_used'] += 1
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"{method} scraper failed: {str(e)}"
            }
    
    def scrape_single_url(self, url: str) -> tuple[bool, Dict]:
        """Scrape single URL with fallback support"""
        target = {
            "url": url,
            "headless": SCRAPE_CONFIG['headless'],
            "timeout": SCRAPE_CONFIG['timeout_per_page'] * 1000  # Convert to ms for browser
        }
        
        preferred_method = SCRAPE_CONFIG['preferred_scraper']
        fallback_enabled = SCRAPE_CONFIG['fallback_enabled']
        
        for attempt in range(SCRAPE_CONFIG['max_retries']):
            if self.shutdown_requested:
                return False, {'error': 'Shutdown requested'}
            
            try:
                self.logger.info(f"📥 Scraping {url} with {preferred_method} (attempt {attempt + 1})")
                
                # Try preferred method
                result = self.scrape_with_method(target, preferred_method)
                
                if result.get('success'):
                    # Success with preferred method
                    self.stats['pages_scraped'] += 1
                    self.stats['total_content_length'] += len(result.get('content', ''))
                    
                    # Save to database
                    self.save_result(result)
                    
                    return True, result
                
                # Preferred method failed, try fallback if enabled
                if fallback_enabled:
                    fallback_method = 'browser' if preferred_method == 'http' else 'http'
                    
                    self.logger.info(f"🔄 Trying fallback method: {fallback_method}")
                    fallback_result = self.scrape_with_method(target, fallback_method)
                    
                    if fallback_result.get('success'):
                        # Success with fallback
                        self.stats['pages_scraped'] += 1
                        self.stats['fallback_used'] += 1
                        self.stats['total_content_length'] += len(fallback_result.get('content', ''))
                        
                        # Save to database
                        self.save_result(fallback_result)
                        
                        return True, fallback_result
                
                # Both methods failed, will retry
                error_msg = result.get('error', 'Unknown error')
                if attempt == SCRAPE_CONFIG['max_retries'] - 1:
                    self.stats['errors'].append(f"{url}: {error_msg}")
                    return False, {'error': error_msg}
                
                # Wait before retry
                time.sleep(2 ** attempt)
                
            except Exception as e:
                error_msg = f"Error scraping {url}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                if attempt == SCRAPE_CONFIG['max_retries'] - 1:
                    self.stats['errors'].append(error_msg)
                    return False, {'error': str(e)}
                
                time.sleep(2 ** attempt)
        
        return False, {'error': 'Max retries exceeded'}
    
    def load_urls_config(self) -> List[str]:
        """Load URLs from configuration"""
        config_file = os.path.join(os.path.dirname(__file__), 'scraper_urls.json')
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    urls = config.get('urls', DEFAULT_URLS)
                    
                    # Update scraper config if present
                    scraper_config = config.get('scraper_config', {})
                    SCRAPE_CONFIG.update(scraper_config)
                    
                    self.logger.info(f"Loaded {len(urls)} URLs from config")
                    return urls
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
        
        return DEFAULT_URLS
    
    def save_status(self, status: str, details: Dict = None):
        """Save current status"""
        try:
            status_data = {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'details': details or {}
            }
            
            with open(STATUS_FILE, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save status: {e}")
    
    def run_scraping_session(self) -> bool:
        """Run complete scraping session"""
        self.logger.info("🚀 Starting modular scraping session")
        self.save_status('running')
        
        try:
            # Load URLs
            urls = self.load_urls_config()
            urls = urls[:SCRAPE_CONFIG['max_pages_per_run']]
            
            self.logger.info(f"📋 Scraping {len(urls)} URLs")
            self.logger.info(f"⚙️  Preferred method: {SCRAPE_CONFIG['preferred_scraper']}")
            self.logger.info(f"🔄 Fallback enabled: {SCRAPE_CONFIG['fallback_enabled']}")
            
            # Scrape each URL
            for i, url in enumerate(urls):
                if self.shutdown_requested:
                    break
                
                success, result = self.scrape_single_url(url)
                
                if success:
                    scraper_type = result.get('scraper_type', 'unknown')
                    self.logger.info(f"✅ Success ({scraper_type}): {url}")
                else:
                    self.logger.warning(f"❌ Failed: {url} - {result.get('error', 'Unknown')}")
                    self.stats['pages_failed'] += 1
                
                # Delay between requests
                if i < len(urls) - 1 and not self.shutdown_requested:
                    time.sleep(SCRAPE_CONFIG['delay_between_pages'])
            
            # Calculate final stats
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            final_stats = {
                **self.stats,
                'duration_seconds': duration,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'urls_processed': len(urls),
                'success_rate': (self.stats['pages_scraped'] / len(urls)) * 100 if urls else 0
            }
            
            # Log results
            self.logger.info("📊 Scraping session completed:")
            self.logger.info(f"   ✅ Pages scraped: {self.stats['pages_scraped']}")
            self.logger.info(f"   ❌ Pages failed: {self.stats['pages_failed']}")
            self.logger.info(f"   🌐 HTTP used: {self.stats['http_used']}")
            self.logger.info(f"   🖥️  Browser used: {self.stats['browser_used']}")
            self.logger.info(f"   🔄 Fallback used: {self.stats['fallback_used']}")
            self.logger.info(f"   📝 Total content: {self.stats['total_content_length']} chars")
            self.logger.info(f"   ⏱️  Duration: {duration:.1f} seconds")
            self.logger.info(f"   📈 Success rate: {final_stats['success_rate']:.1f}%")
            
            self.save_status('completed', final_stats)
            return self.stats['pages_scraped'] > 0
            
        except Exception as e:
            error_msg = f"Critical error: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.save_status('failed', {'error': error_msg})
            return False

def main():
    """Main entry point"""
    runner = ModularScraperRunner()
    
    try:
        success = runner.run_scraping_session()
        return success
    except KeyboardInterrupt:
        runner.logger.info("🛑 Interrupted by user")
        return False
    except Exception as e:
        runner.logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_modular_scraper.py           # Run scraping session")
            print("  python run_modular_scraper.py --help    # Show this help")
            sys.exit(0)
    
    # Run the scraper
    try:
        success = main()
        exit_code = 0 if success else 1
        
        print(f"\n{'✅' if success else '❌'} Modular scraping {'completed' if success else 'failed'}")
        print(f"📋 Check logs at: {LOG_FILE}")
        print(f"📊 Status file: {STATUS_FILE}")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
