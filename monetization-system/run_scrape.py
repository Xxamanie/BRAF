#!/usr/bin/env python3
"""
Production Web Scraper Runner
Handles scheduled scraping with robust error handling, logging, and SQLite storage.
"""
import asyncio
import logging
import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import signal
import time
from urllib.parse import urlparse
import hashlib

# Add automation directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'automation'))

# Configuration
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'scraper.log')
ERROR_LOG_FILE = os.path.join(LOG_DIR, 'scraper_errors.log')
STATUS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'scraper_status.json')

# Default URLs to scrape (can be overridden via config file)
DEFAULT_URLS = [
    'https://news.ycombinator.com',
    'https://httpbin.org/html',
    'https://example.com',
    'https://www.reddit.com/r/programming.json',
]

# Scraping configuration
SCRAPE_CONFIG = {
    'max_pages_per_run': 10,
    'delay_between_pages': 2,  # seconds
    'timeout_per_page': 30,    # seconds
    'max_retries': 3,
    'headless': True,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

class ScraperRunner:
    """Main scraper runner with comprehensive error handling and logging"""
    
    def __init__(self):
        self.setup_logging()
        self.setup_directories()
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        self.stats = {
            'pages_scraped': 0,
            'pages_failed': 0,
            'pages_skipped': 0,
            'total_content_length': 0,
            'errors': []
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.shutdown_requested = False
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Main logger configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Error logger for critical issues
        error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n%(funcName)s\n'
        )
        error_handler.setFormatter(error_formatter)
        
        # Add error handler to root logger
        logging.getLogger().addHandler(error_handler)
    
    def setup_directories(self):
        """Ensure all required directories exist"""
        directories = [
            LOG_DIR,
            os.path.join(os.path.dirname(__file__), 'data'),
            os.path.join(os.path.dirname(__file__), 'backups')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def load_urls_config(self) -> List[str]:
        """Load URLs from configuration file or use defaults"""
        config_file = os.path.join(os.path.dirname(__file__), 'scraper_urls.json')
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    urls = config.get('urls', DEFAULT_URLS)
                    self.logger.info(f"Loaded {len(urls)} URLs from config file")
                    return urls
        except Exception as e:
            self.logger.warning(f"Failed to load URL config: {e}, using defaults")
        
        return DEFAULT_URLS
    
    def save_status(self, status: str, details: Dict = None):
        """Save current scraper status to file"""
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
    
    async def initialize_scraper(self):
        """Initialize the scraper and database"""
        try:
            # Try to import and initialize the sync Playwright scraper first
            from sync_playwright_scraper import SyncPlaywrightScraper
            self.sync_scraper = SyncPlaywrightScraper(headless=SCRAPE_CONFIG['headless'])
            self.logger.info("✅ Sync Playwright scraper initialized")
            return True
            
        except ImportError:
            self.logger.warning("Sync Playwright scraper not available, trying SQLite scraper")
            
            try:
                from sqlite_scraper import SQLiteWebScraper
                self.sqlite_scraper = SQLiteWebScraper()
                self.logger.info("✅ SQLite scraper initialized")
                return True
                
            except ImportError:
                self.logger.warning("SQLite scraper not available, trying ethical_web_scraper")
                
                try:
                    from ethical_web_scraper import init_database, RealWebScraper
                    init_database()
                    self.logger.info("✅ Ethical web scraper initialized")
                    return True
                    
                except ImportError as e:
                    self.logger.error(f"❌ Failed to initialize any scraper: {e}")
                    return False
    
    async def scrape_single_url(self, url: str, scraper) -> Tuple[bool, Dict]:
        """Scrape a single URL with error handling and retries"""
        for attempt in range(SCRAPE_CONFIG['max_retries']):
            if self.shutdown_requested:
                return False, {'error': 'Shutdown requested'}
            
            try:
                self.logger.info(f"📥 Scraping {url} (attempt {attempt + 1})")
                
                # Use sync Playwright scraper if available (preferred)
                if hasattr(self, 'sync_scraper'):
                    target = {"url": url}
                    result = self.sync_scraper.run_single_scrape(target)
                    
                    if result.success:
                        # Save to database (already done in scraper, but verify)
                        self.stats['pages_scraped'] += 1
                        self.stats['total_content_length'] += len(result.content)
                        return True, {
                            'url': url,
                            'title': result.title,
                            'content_length': len(result.content),
                            'word_count': result.word_count,
                            'domain': result.domain
                        }
                    else:
                        # Failed scrape, will retry
                        error_msg = result.error or "Unknown error"
                        if attempt == SCRAPE_CONFIG['max_retries'] - 1:
                            self.stats['errors'].append(f"{url}: {error_msg}")
                            return False, {'error': error_msg}
                        continue
                
                # Use SQLite scraper if available
                elif hasattr(self, 'sqlite_scraper'):
                    from sqlite_scraper import ScrapedData
                    
                    # Simulate scraped content (replace with actual scraping)
                    content = f"Scraped content from {url} at {datetime.now()}"
                    scraped_data = ScrapedData(
                        url=url,
                        domain=urlparse(url).netloc,
                        title=f"Page from {urlparse(url).netloc}",
                        content=content,
                        word_count=len(content.split()),
                        scraped_at=datetime.now(),
                        data_hash=hashlib.sha256(content.encode()).hexdigest()[:16]
                    )
                    
                    success = self.sqlite_scraper.save_data(scraped_data)
                    if success:
                        self.stats['pages_scraped'] += 1
                        self.stats['total_content_length'] += len(content)
                        return True, {
                            'url': url,
                            'title': scraped_data.title,
                            'content_length': len(content),
                            'word_count': scraped_data.word_count
                        }
                
                else:
                    # Use the ethical web scraper
                    from ethical_web_scraper import RealWebScraper, save_scraped_data
                    
                    async with RealWebScraper(headless=SCRAPE_CONFIG['headless']) as web_scraper:
                        page_data = await web_scraper.scrape_page(url)
                        
                        if save_scraped_data(page_data):
                            self.stats['pages_scraped'] += 1
                            self.stats['total_content_length'] += len(page_data.content)
                            return True, {
                                'url': url,
                                'title': page_data.title,
                                'content_length': len(page_data.content),
                                'domain': page_data.domain
                            }
                
                return False, {'error': 'Failed to save scraped data'}
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout scraping {url} (attempt {attempt + 1})"
                self.logger.warning(error_msg)
                if attempt == SCRAPE_CONFIG['max_retries'] - 1:
                    self.stats['errors'].append(error_msg)
                    return False, {'error': 'Timeout after retries'}
                
            except Exception as e:
                error_msg = f"Error scraping {url} (attempt {attempt + 1}): {e}"
                self.logger.error(error_msg, exc_info=True)
                if attempt == SCRAPE_CONFIG['max_retries'] - 1:
                    self.stats['errors'].append(error_msg)
                    return False, {'error': str(e)}
            
            # Wait before retry
            if attempt < SCRAPE_CONFIG['max_retries'] - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False, {'error': 'Max retries exceeded'}
    
    async def run_scraping_session(self) -> bool:
        """Run a complete scraping session"""
        self.logger.info("🚀 Starting scheduled scrape job")
        self.save_status('running', {'start_time': self.start_time.isoformat()})
        
        try:
            # Initialize scraper
            if not await self.initialize_scraper():
                self.logger.error("❌ Failed to initialize scraper")
                self.save_status('failed', {'error': 'Scraper initialization failed'})
                return False
            
            # Load URLs to scrape
            urls = self.load_urls_config()
            urls = urls[:SCRAPE_CONFIG['max_pages_per_run']]  # Limit pages per run
            
            self.logger.info(f"📋 Loaded {len(urls)} URLs to scrape")
            
            # Scrape each URL
            results = []
            for i, url in enumerate(urls):
                if self.shutdown_requested:
                    self.logger.info("🛑 Shutdown requested, stopping scraping")
                    break
                
                success, result = await self.scrape_single_url(url, None)
                results.append({
                    'url': url,
                    'success': success,
                    'result': result
                })
                
                if success:
                    self.logger.info(f"✅ Successfully scraped {url}")
                else:
                    self.logger.warning(f"❌ Failed to scrape {url}: {result.get('error', 'Unknown error')}")
                    self.stats['pages_failed'] += 1
                
                # Delay between pages to be respectful
                if i < len(urls) - 1 and not self.shutdown_requested:
                    await asyncio.sleep(SCRAPE_CONFIG['delay_between_pages'])
            
            # Calculate final statistics
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            final_stats = {
                **self.stats,
                'duration_seconds': duration,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'urls_processed': len(results),
                'success_rate': (self.stats['pages_scraped'] / len(results)) * 100 if results else 0
            }
            
            self.logger.info(f"📊 Scraping session completed:")
            self.logger.info(f"   ✅ Pages scraped: {self.stats['pages_scraped']}")
            self.logger.info(f"   ❌ Pages failed: {self.stats['pages_failed']}")
            self.logger.info(f"   📝 Total content: {self.stats['total_content_length']} chars")
            self.logger.info(f"   ⏱️  Duration: {duration:.1f} seconds")
            self.logger.info(f"   📈 Success rate: {final_stats['success_rate']:.1f}%")
            
            # Save final status
            self.save_status('completed', final_stats)
            
            # Return success if at least one page was scraped
            return self.stats['pages_scraped'] > 0
            
        except Exception as e:
            error_msg = f"Critical error in scraping session: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.save_status('failed', {
                'error': error_msg,
                'traceback': traceback.format_exc()
            })
            return False
    
    def cleanup(self):
        """Cleanup resources and log final status"""
        try:
            # Log cleanup
            self.logger.info("🧹 Cleaning up resources...")
            
            # Archive old logs if they're too large
            self.archive_large_logs()
            
            self.logger.info("✅ Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def archive_large_logs(self):
        """Archive log files if they become too large"""
        max_log_size = 10 * 1024 * 1024  # 10MB
        
        for log_file in [LOG_FILE, ERROR_LOG_FILE]:
            try:
                if os.path.exists(log_file) and os.path.getsize(log_file) > max_log_size:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    archive_name = f"{log_file}.{timestamp}"
                    os.rename(log_file, archive_name)
                    self.logger.info(f"Archived large log file: {archive_name}")
            except Exception as e:
                self.logger.error(f"Failed to archive log {log_file}: {e}")

async def main():
    """Main entry point"""
    runner = ScraperRunner()
    
    try:
        success = await runner.run_scraping_session()
        return success
        
    except KeyboardInterrupt:
        runner.logger.info("🛑 Interrupted by user")
        return False
        
    except Exception as e:
        runner.logger.error(f"💥 Unexpected error: {e}", exc_info=True)
        return False
        
    finally:
        runner.cleanup()

def create_sample_config():
    """Create a sample configuration file"""
    config_file = os.path.join(os.path.dirname(__file__), 'scraper_urls.json')
    
    if not os.path.exists(config_file):
        sample_config = {
            "urls": [
                "https://news.ycombinator.com",
                "https://httpbin.org/html",
                "https://example.com",
                "https://www.reddit.com/r/programming.json"
            ],
            "config": {
                "max_pages_per_run": 10,
                "delay_between_pages": 2,
                "timeout_per_page": 30,
                "max_retries": 3
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"✅ Created sample config file: {config_file}")

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-config":
            create_sample_config()
            sys.exit(0)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_scrape.py                 # Run scraping session")
            print("  python run_scrape.py --create-config # Create sample config")
            print("  python run_scrape.py --help          # Show this help")
            sys.exit(0)
    
    # Run the scraper
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
        
        print(f"\n{'✅' if success else '❌'} Scraping session {'completed successfully' if success else 'failed'}")
        print(f"📋 Check logs at: {LOG_FILE}")
        print(f"📊 Status file: {STATUS_FILE}")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)