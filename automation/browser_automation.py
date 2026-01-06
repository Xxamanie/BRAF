#!/usr/bin/env python3
"""
BRAF Browser Automation Module
Advanced browser automation with behavioral patterns
"""
import time
import random
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BrowserAutomation:
    """Advanced browser automation with human-like behavior"""
    
    def __init__(self, headless=True, slow_mo=None):
        self.headless = headless
        self.slow_mo = slow_mo or random.randint(50, 150)
        self.page = None
        self.browser = None
        
    def launch_browser(self):
        """Launch browser with stealth settings"""
        try:
            from playwright.sync_api import sync_playwright
            
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=[
                    '--no-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            # Create context with realistic settings
            context = self.browser.new_context(
                viewport={'width': 1366, 'height': 768},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            self.page = context.new_page()
            
            # Add stealth scripts
            self.page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            logger.info("Browser launched successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            return False
    
    def human_type(self, selector: str, text: str, delay_range=(50, 150)):
        """Type text with human-like delays"""
        if not self.page:
            return False
            
        try:
            element = self.page.wait_for_selector(selector, timeout=10000)
            element.click()
            
            # Clear existing text
            element.fill('')
            
            # Type with random delays
            for char in text:
                element.type(char)
                delay = random.randint(*delay_range)
                time.sleep(delay / 1000)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to type in {selector}: {e}")
            return False
    
    def human_click(self, selector: str, delay_range=(100, 300)):
        """Click with human-like behavior"""
        if not self.page:
            return False
            
        try:
            # Random delay before click
            delay = random.randint(*delay_range)
            time.sleep(delay / 1000)
            
            element = self.page.wait_for_selector(selector, timeout=10000)
            
            # Get element bounds for realistic click position
            box = element.bounding_box()
            if box:
                # Click at random position within element
                x = box['x'] + random.randint(5, int(box['width'] - 5))
                y = box['y'] + random.randint(5, int(box['height'] - 5))
                self.page.mouse.click(x, y)
            else:
                element.click()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to click {selector}: {e}")
            return False
    
    def scroll_page(self, direction='down', amount=None):
        """Scroll page with human-like behavior"""
        if not self.page:
            return False
            
        try:
            if amount is None:
                amount = random.randint(200, 600)
            
            if direction == 'down':
                self.page.mouse.wheel(0, amount)
            elif direction == 'up':
                self.page.mouse.wheel(0, -amount)
            
            # Random pause after scroll
            time.sleep(random.uniform(0.5, 1.5))
            return True
            
        except Exception as e:
            logger.error(f"Failed to scroll: {e}")
            return False
    
    def wait_for_element(self, selector: str, timeout=10000):
        """Wait for element with timeout"""
        if not self.page:
            return None
            
        try:
            return self.page.wait_for_selector(selector, timeout=timeout)
        except Exception as e:
            logger.error(f"Element {selector} not found: {e}")
            return None
    
    def extract_data(self, selectors: Dict[str, str]) -> Dict[str, str]:
        """Extract data using CSS selectors"""
        if not self.page:
            return {}
            
        data = {}
        
        for key, selector in selectors.items():
            try:
                element = self.page.query_selector(selector)
                if element:
                    data[key] = element.inner_text().strip()
                else:
                    data[key] = None
            except Exception as e:
                logger.error(f"Failed to extract {key}: {e}")
                data[key] = None
        
        return data
    
    def take_screenshot(self, path: str = None) -> str:
        """Take screenshot for debugging"""
        if not self.page:
            return None
            
        try:
            if path is None:
                path = f"screenshot_{int(time.time())}.png"
            
            self.page.screenshot(path=path)
            logger.info(f"Screenshot saved: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    def close(self):
        """Close browser and cleanup"""
        try:
            if self.browser:
                self.browser.close()
            if hasattr(self, 'playwright'):
                self.playwright.stop()
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")

def run_automation_task(task_config: Dict) -> Dict:
    """Run a browser automation task"""
    automation = BrowserAutomation(
        headless=task_config.get('headless', True),
        slow_mo=task_config.get('slow_mo', 100)
    )
    
    try:
        # Launch browser
        if not automation.launch_browser():
            return {'success': False, 'error': 'Failed to launch browser'}
        
        # Navigate to URL
        url = task_config.get('url')
        if not url:
            return {'success': False, 'error': 'No URL provided'}
        
        automation.page.goto(url, wait_until='networkidle')
        
        # Execute actions
        actions = task_config.get('actions', [])
        results = {}
        
        for action in actions:
            action_type = action.get('type')
            
            if action_type == 'click':
                success = automation.human_click(action.get('selector'))
                results[f"click_{action.get('name', 'unknown')}"] = success
                
            elif action_type == 'type':
                success = automation.human_type(
                    action.get('selector'), 
                    action.get('text', '')
                )
                results[f"type_{action.get('name', 'unknown')}"] = success
                
            elif action_type == 'extract':
                data = automation.extract_data(action.get('selectors', {}))
                results[f"extract_{action.get('name', 'unknown')}"] = data
                
            elif action_type == 'wait':
                time.sleep(action.get('duration', 1))
                results[f"wait_{action.get('name', 'unknown')}"] = True
                
            elif action_type == 'scroll':
                success = automation.scroll_page(
                    action.get('direction', 'down'),
                    action.get('amount')
                )
                results[f"scroll_{action.get('name', 'unknown')}"] = success
        
        # Take screenshot if requested
        if task_config.get('screenshot'):
            screenshot_path = automation.take_screenshot()
            results['screenshot'] = screenshot_path
        
        return {
            'success': True,
            'url': url,
            'results': results,
            'execution_time': time.time()
        }
        
    except Exception as e:
        logger.error(f"Automation task failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'url': task_config.get('url', 'unknown')
        }
        
    finally:
        automation.close()

def main():
    """Test browser automation"""
    task_config = {
        'url': 'https://quotes.toscrape.com/js/',
        'headless': False,
        'screenshot': True,
        'actions': [
            {
                'type': 'wait',
                'name': 'page_load',
                'duration': 2
            },
            {
                'type': 'extract',
                'name': 'quotes',
                'selectors': {
                    'title': 'title',
                    'quote_count': '.quote'
                }
            },
            {
                'type': 'scroll',
                'name': 'scroll_down',
                'direction': 'down',
                'amount': 300
            }
        ]
    }
    
    print("🤖 Testing Browser Automation")
    result = run_automation_task(task_config)
    
    if result['success']:
        print("✅ Automation completed successfully")
        print(f"📊 Results: {result['results']}")
    else:
        print(f"❌ Automation failed: {result.get('error')}")

if __name__ == "__main__":
    main()
