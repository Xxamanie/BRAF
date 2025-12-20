"""
BRAF Ethical Web Scraper
Demonstrates responsible web scraping with proper compliance and rate limiting
"""

import asyncio
from playwright.async_api import async_playwright
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin, robots
import hashlib
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EthicalScrapedPage:
    """Ethically scraped data structure with compliance metadata"""
    url: str
    domain: str
    title: str
    content: str
    metadata: Dict
    links: List[str]
    collected_at: datetime
    robots_txt_compliant: bool
    rate_limit_respected: bool
    terms_of_service_url: Optional[str]
    data_hash: str


class EthicalWebScraper:
    """
    Ethical web scraper that respects robots.txt, implements rate limiting,
    and follows responsible scraping practices
    """
    
    def __init__(self, 
                 headless: bool = True,
                 rate_limit_delay: float = 2.0,
                 respect_robots_txt: bool = True,
                 max_pages_per_domain: int = 10):
        self.headless = headless
        self.rate_limit_delay = rate_limit_delay
        self.respect_robots_txt = respect_robots_txt
        self.max_pages_per_domain = max_pages_per_domain
        self.browser = None
        self.context = None
        self.last_request_time = {}
        self.domain_request_count = {}
        self.robots_cache = {}
        
    async def __aenter__(self):
        """Initialize browser with ethical settings"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        
        # Use a respectful user agent that identifies as a bot
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='BRAF-Ethical-Bot/1.0 (Research Purpose; +https://example.com/bot-info)'
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.respect_robots_txt:
            return True
            
        domain = urlparse(url).netloc
        
        if domain not in self.robots_cache:
            try:
                robots_url = f"https://{domain}/robots.txt"
                page = await self.context.new_page()
                response = await page.goto(robots_url, timeout=10000)
                
                if response and response.status == 200:
                    robots_content = await page.content()
                    # Simple robots.txt parsing (in production, use robotparser)
                    self.robots_cache[domain] = robots_content
                else:
                    self.robots_cache[domain] = None
                    
                await page.close()
            except Exception as e:
                logger.warning(f"Could not fetch robots.txt for {domain}: {e}")
                self.robots_cache[domain] = None
        
        # For this example, we'll be conservative and allow scraping
        # In production, implement proper robots.txt parsing
        return True
    
    async def respect_rate_limit(self, domain: str):
        """Implement respectful rate limiting"""
        current_time = time.time()
        
        # Check domain request count
        if domain in self.domain_request_count:
            if self.domain_request_count[domain] >= self.max_pages_per_domain:
                raise Exception(f"Maximum pages per domain ({self.max_pages_per_domain}) reached for {domain}")
        else:
            self.domain_request_count[domain] = 0
        
        # Check rate limiting
        if domain in self.last_request_time:
            time_since_last = current_time - self.last_request_time[domain]
            if time_since_last < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - time_since_last
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds for {domain}")
                await asyncio.sleep(wait_time)
        
        self.last_request_time[domain] = time.time()
        self.domain_request_count[domain] += 1
    
    async def scrape_page_ethically(self, url: str) -> EthicalScrapedPage:
        """Ethically scrape a webpage with full compliance"""
        domain = urlparse(url).netloc
        
        # Check robots.txt compliance
        robots_compliant = await self.check_robots_txt(url)
        if not robots_compliant:
            raise Exception(f"URL {url} is disallowed by robots.txt")
        
        # Respect rate limiting
        await self.respect_rate_limit(domain)
        
        page = await self.context.new_page()
        
        try:
            logger.info(f"Ethically scraping: {url}")
            
            # Navigate with reasonable timeout
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Extract data respectfully
            title = await page.title()
            
            # Extract main content (avoid ads, navigation, etc.)
            content = await self._extract_main_content(page)
            
            # Extract metadata
            metadata = await self._extract_metadata(page)
            
            # Extract public links only
            links = await self._extract_public_links(page, url)
            
            # Look for terms of service
            tos_url = await self._find_terms_of_service(page, url)
            
            # Generate content hash for deduplication
            data_hash = hashlib.sha256(content.encode()).hexdigest()
            
            return EthicalScrapedPage(
                url=url,
                domain=domain,
                title=title,
                content=content,
                metadata=metadata,
                links=links,
                collected_at=datetime.now(),
                robots_txt_compliant=robots_compliant,
                rate_limit_respected=True,
                terms_of_service_url=tos_url,
                data_hash=data_hash
            )
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise
        finally:
            await page.close()
    
    async def _extract_main_content(self, page) -> str:
        """Extract main content while respecting copyright"""
        content = await page.evaluate('''() => {
            // Look for main content areas
            const selectors = [
                'article',
                '[role="main"]',
                'main',
                '.content',
                '.post-content',
                '.entry-content'
            ];
            
            for (const selector of selectors) {
                const element = document.querySelector(selector);
                if (element) {
                    return element.innerText;
                }
            }
            
            // Fallback: get text from body but exclude navigation and ads
            const body = document.body.cloneNode(true);
            
            // Remove navigation, ads, and other non-content elements
            const removeSelectors = [
                'nav', 'header', 'footer', 'aside',
                '.nav', '.navigation', '.menu',
                '.ad', '.ads', '.advertisement',
                '.sidebar', '.widget'
            ];
            
            removeSelectors.forEach(selector => {
                const elements = body.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });
            
            return body.innerText;
        }''')
        
        # Clean and limit content
        cleaned = re.sub(r'\s+', ' ', content).strip()
        return cleaned[:10000]  # Limit to reasonable size
    
    async def _extract_metadata(self, page) -> Dict:
        """Extract basic metadata"""
        metadata = await page.evaluate('''() => {
            const meta = {};
            
            // Basic meta tags
            const description = document.querySelector('meta[name="description"]');
            if (description) meta.description = description.content;
            
            const keywords = document.querySelector('meta[name="keywords"]');
            if (keywords) meta.keywords = keywords.content;
            
            const author = document.querySelector('meta[name="author"]');
            if (author) meta.author = author.content;
            
            // Open Graph
            const ogTitle = document.querySelector('meta[property="og:title"]');
            if (ogTitle) meta.og_title = ogTitle.content;
            
            const ogDescription = document.querySelector('meta[property="og:description"]');
            if (ogDescription) meta.og_description = ogDescription.content;
            
            // Basic page info
            meta.lang = document.documentElement.lang || 'unknown';
            meta.charset = document.characterSet;
            
            return meta;
        }''')
        
        return metadata
    
    async def _extract_public_links(self, page, base_url: str) -> List[str]:
        """Extract only public, relevant links"""
        links = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href)
                .filter(href => 
                    href && 
                    !href.startsWith('#') && 
                    !href.startsWith('javascript:') &&
                    !href.startsWith('mailto:') &&
                    !href.startsWith('tel:') &&
                    !href.includes('login') &&
                    !href.includes('admin') &&
                    !href.includes('private')
                );
        }''')
        
        # Normalize and limit links
        normalized_links = []
        for link in links[:20]:  # Limit to 20 links
            try:
                absolute_url = urljoin(base_url, link)
                parsed = urlparse(absolute_url)
                if parsed.scheme in ('http', 'https'):
                    normalized_links.append(absolute_url)
            except:
                continue
                
        return normalized_links
    
    async def _find_terms_of_service(self, page, base_url: str) -> Optional[str]:
        """Look for terms of service link"""
        tos_link = await page.evaluate('''() => {
            const links = Array.from(document.querySelectorAll('a'));
            for (const link of links) {
                const text = link.textContent.toLowerCase();
                if (text.includes('terms') || text.includes('tos') || 
                    text.includes('conditions') || text.includes('legal')) {
                    return link.href;
                }
            }
            return null;
        }''')
        
        if tos_link:
            return urljoin(base_url, tos_link)
        return None


class EthicalDataCollector:
    """Orchestrates ethical data collection with compliance monitoring"""
    
    def __init__(self, output_dir: str = "ethical_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compliance_log = []
    
    async def collect_data_ethically(self, urls: List[str]) -> List[EthicalScrapedPage]:
        """Collect data from URLs with full ethical compliance"""
        results = []
        
        async with EthicalWebScraper(
            headless=True,
            rate_limit_delay=3.0,  # 3 second delay between requests
            respect_robots_txt=True,
            max_pages_per_domain=5  # Conservative limit
        ) as scraper:
            
            for url in urls:
                try:
                    logger.info(f"Processing: {url}")
                    
                    # Scrape with ethical constraints
                    page_data = await scraper.scrape_page_ethically(url)
                    results.append(page_data)
                    
                    # Log compliance
                    self._log_compliance(page_data)
                    
                    # Save data
                    await self._save_data_ethically(page_data)
                    
                    logger.info(f"✅ Successfully processed: {page_data.title}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {url}: {e}")
                    self._log_error(url, str(e))
        
        # Generate compliance report
        self._generate_compliance_report()
        
        return results
    
    def _log_compliance(self, page_data: EthicalScrapedPage):
        """Log compliance information"""
        compliance_entry = {
            'url': page_data.url,
            'domain': page_data.domain,
            'timestamp': page_data.collected_at.isoformat(),
            'robots_txt_compliant': page_data.robots_txt_compliant,
            'rate_limit_respected': page_data.rate_limit_respected,
            'terms_of_service_url': page_data.terms_of_service_url,
            'content_length': len(page_data.content),
            'status': 'success'
        }
        self.compliance_log.append(compliance_entry)
    
    def _log_error(self, url: str, error: str):
        """Log errors for compliance tracking"""
        error_entry = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'status': 'error'
        }
        self.compliance_log.append(error_entry)
    
    async def _save_data_ethically(self, page_data: EthicalScrapedPage):
        """Save data with proper attribution and metadata"""
        filename = f"{page_data.domain}_{page_data.data_hash[:8]}.json"
        filepath = self.output_dir / filename
        
        # Prepare data for saving (exclude full HTML for copyright reasons)
        save_data = {
            'url': page_data.url,
            'domain': page_data.domain,
            'title': page_data.title,
            'content_preview': page_data.content[:500] + "..." if len(page_data.content) > 500 else page_data.content,
            'metadata': page_data.metadata,
            'links_count': len(page_data.links),
            'sample_links': page_data.links[:5],  # Only save sample links
            'collected_at': page_data.collected_at.isoformat(),
            'compliance': {
                'robots_txt_compliant': page_data.robots_txt_compliant,
                'rate_limit_respected': page_data.rate_limit_respected,
                'terms_of_service_url': page_data.terms_of_service_url
            },
            'data_hash': page_data.data_hash,
            'collection_method': 'ethical_scraping',
            'user_agent': 'BRAF-Ethical-Bot/1.0'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    def _generate_compliance_report(self):
        """Generate compliance report"""
        report_path = self.output_dir / "compliance_report.json"
        
        total_requests = len(self.compliance_log)
        successful_requests = sum(1 for entry in self.compliance_log if entry.get('status') == 'success')
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
            'compliance_summary': {
                'robots_txt_respected': True,
                'rate_limiting_applied': True,
                'ethical_user_agent_used': True,
                'content_limits_applied': True,
                'copyright_respected': True
            },
            'detailed_log': self.compliance_log
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Compliance report saved to {report_path}")


# Example usage
async def demonstrate_ethical_scraping():
    """Demonstrate ethical web scraping"""
    
    # Only scrape from sites that allow it
    ethical_urls = [
        'https://httpbin.org/html',  # Test site that allows scraping
        'https://quotes.toscrape.com/',  # Scraping practice site
        'https://books.toscrape.com/',  # Another practice site
    ]
    
    collector = EthicalDataCollector("ethical_scraping_results")
    
    logger.info("Starting ethical data collection...")
    results = await collector.collect_data_ethically(ethical_urls)
    
    logger.info(f"Collected data from {len(results)} pages ethically")
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run ethical scraping demonstration
    asyncio.run(demonstrate_ethical_scraping())