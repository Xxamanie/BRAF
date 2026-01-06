"""
Modular Scraper System
Supports multiple scraping methods: HTTP and Browser-based
"""

from .registry import SCRAPERS, get_scraper, run_with_best_scraper
from .http_scraper import run as http_run
from .browser_scraper import run as browser_run

__all__ = ['SCRAPERS', 'get_scraper', 'run_with_best_scraper', 'http_run', 'browser_run']
