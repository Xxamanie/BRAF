"""
Browser Instance Manager with stealth capabilities for BRAF.

This module manages Playwright browser instances with advanced stealth features,
fingerprint application, and bot detection monitoring.
"""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from braf.core.models import BrowserFingerprint, ProxyConfig, DetectionScore

logger = logging.getLogger(__name__)


class StealthConfig:
    """Configuration for stealth features and bot detection evasion."""
    
    def __init__(self):
        """Initialize stealth configuration with default settings."""
        # User agent overrides
        self.override_user_agent = True
        
        # WebGL fingerprint overrides
        self.override_webgl = True
        
        # Canvas fingerprint overrides
        self.override_canvas = True
        
        # Audio context overrides
        self.override_audio_context = True
        
        # Navigator properties
        self.override_navigator = True
        
        # Screen properties
        self.override_screen = True
        
        # Timezone override
        self.override_timezone = True
        
        # Language overrides
        self.override_languages = True
        
        # Plugin overrides
        self.override_plugins = True
        
        # Permission overrides
        self.override_permissions = True
        
        # Chrome runtime detection
        self.hide_chrome_runtime = True
        
        # Automation detection
        self.hide_automation_flags = True


class BrowserInstance:
    """Managed browser instance with stealth capabilities."""
    
    def __init__(
        self, 
        browser: Browser, 
        context: BrowserContext, 
        fingerprint: BrowserFingerprint,
        proxy_config: Optional[ProxyConfig] = None
    ):
        """
        Initialize browser instance.
        
        Args:
            browser: Playwright browser instance
            context: Browser context
            fingerprint: Applied browser fingerprint
            proxy_config: Optional proxy configuration
        """
        self.browser = browser
        self.context = context
        self.fingerprint = fingerprint
        self.proxy_config = proxy_config
        self.pages: List[Page] = []
        self.detection_score = 0.0
        self.created_at = asyncio.get_event_loop().time()
        self.last_activity = self.created_at
    
    async def new_page(self) -> Page:
        """
        Create a new page with stealth configuration.
        
        Returns:
            Configured page instance
        """
        page = await self.context.new_page()
        
        # Apply stealth scripts
        await self._apply_stealth_scripts(page)
        
        # Track page
        self.pages.append(page)
        self.last_activity = asyncio.get_event_loop().time()
        
        logger.debug(f"Created new page, total pages: {len(self.pages)}")
        return page
    
    async def close_page(self, page: Page) -> None:
        """
        Close a page and remove from tracking.
        
        Args:
            page: Page to close
        """
        if page in self.pages:
            self.pages.remove(page)
        
        await page.close()
        self.last_activity = asyncio.get_event_loop().time()
        
        logger.debug(f"Closed page, remaining pages: {len(self.pages)}")
    
    async def close(self) -> None:
        """Close browser instance and cleanup resources."""
        # Close all pages
        for page in self.pages[:]:  # Copy list to avoid modification during iteration
            await self.close_page(page)
        
        # Close context and browser
        await self.context.close()
        await self.browser.close()
        
        logger.info("Browser instance closed")
    
    async def _apply_stealth_scripts(self, page: Page) -> None:
        """
        Apply stealth scripts to page to avoid detection.
        
        Args:
            page: Page to configure
        """
        # Override navigator.webdriver
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        # Override chrome runtime
        await page.add_init_script("""
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
        """)
        
        # Override permissions
        await page.add_init_script("""
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
        
        # Override plugins
        await page.add_init_script(f"""
            Object.defineProperty(navigator, 'plugins', {{
                get: () => {json.dumps(self.fingerprint.plugins)},
            }});
        """)
        
        # Override languages
        await page.add_init_script(f"""
            Object.defineProperty(navigator, 'languages', {{
                get: () => {json.dumps(self.fingerprint.languages)},
            }});
        """)
        
        # Override WebGL
        await page.add_init_script(f"""
            const getParameter = WebGLRenderingContext.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === 37445) {{
                    return '{self.fingerprint.webgl_vendor}';
                }}
                if (parameter === 37446) {{
                    return '{self.fingerprint.webgl_renderer}';
                }}
                return getParameter(parameter);
            }};
        """)
        
        # Override canvas fingerprinting
        await page.add_init_script(f"""
            const toDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function() {{
                const context = this.getContext('2d');
                if (context) {{
                    context.fillText('{self.fingerprint.canvas_hash}', 0, 0);
                }}
                return toDataURL.apply(this, arguments);
            }};
        """)
        
        # Override audio context
        await page.add_init_script(f"""
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) {{
                const createAnalyser = AudioContext.prototype.createAnalyser;
                AudioContext.prototype.createAnalyser = function() {{
                    const analyser = createAnalyser.apply(this, arguments);
                    const getFloatFrequencyData = analyser.getFloatFrequencyData;
                    analyser.getFloatFrequencyData = function(array) {{
                        getFloatFrequencyData.apply(this, arguments);
                        // Add fingerprint-specific noise
                        for (let i = 0; i < array.length; i++) {{
                            array[i] += Math.sin(i * 0.1 + {hash(self.fingerprint.audio_context_hash) % 1000}) * 0.001;
                        }}
                    }};
                    return analyser;
                }};
            }}
        """)


class BrowserInstanceManager:
    """Manager for browser instances with stealth capabilities and fingerprint application."""
    
    def __init__(self, stealth_config: Optional[StealthConfig] = None):
        """
        Initialize browser instance manager.
        
        Args:
            stealth_config: Optional stealth configuration
        """
        self.stealth_config = stealth_config or StealthConfig()
        self.playwright: Optional[Playwright] = None
        self.active_instances: Dict[str, BrowserInstance] = {}
        self.temp_dirs: List[str] = []
    
    async def initialize(self) -> None:
        """Initialize Playwright and browser management."""
        if self.playwright is None:
            self.playwright = await async_playwright().start()
            logger.info("Browser instance manager initialized")
    
    async def create_instance(
        self,
        fingerprint: BrowserFingerprint,
        proxy_config: Optional[ProxyConfig] = None,
        headless: bool = True,
        user_data_dir: Optional[str] = None
    ) -> BrowserInstance:
        """
        Create a new browser instance with fingerprint and stealth configuration.
        
        Args:
            fingerprint: Browser fingerprint to apply
            proxy_config: Optional proxy configuration
            headless: Whether to run in headless mode
            user_data_dir: Optional user data directory
            
        Returns:
            Configured browser instance
        """
        if not self.playwright:
            await self.initialize()
        
        # Create temporary user data directory if not provided
        if not user_data_dir:
            temp_dir = tempfile.mkdtemp(prefix="braf_browser_")
            self.temp_dirs.append(temp_dir)
            user_data_dir = temp_dir
        
        # Configure launch options
        launch_options = {
            "headless": headless,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=VizDisplayCompositor",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",  # Faster loading
                "--disable-javascript-harmony-shipping",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-field-trial-config",
                "--disable-ipc-flooding-protection",
                "--no-first-run",
                "--no-default-browser-check",
                "--no-pings",
                "--password-store=basic",
                "--use-mock-keychain",
                f"--user-agent={fingerprint.user_agent}"
            ]
        }
        
        # Add proxy configuration if provided
        if proxy_config:
            proxy_dict = {
                "server": f"{proxy_config.proxy_type}://{proxy_config.host}:{proxy_config.port}"
            }
            if proxy_config.username and proxy_config.password:
                proxy_dict["username"] = proxy_config.username
                proxy_dict["password"] = proxy_config.password
            
            launch_options["proxy"] = proxy_dict
        
        # Launch browser
        browser = await self.playwright.chromium.launch(**launch_options)
        
        # Create context with fingerprint
        context_options = {
            "viewport": {
                "width": fingerprint.screen_resolution[0],
                "height": fingerprint.screen_resolution[1]
            },
            "user_agent": fingerprint.user_agent,
            "locale": fingerprint.languages[0] if fingerprint.languages else "en-US",
            "timezone_id": fingerprint.timezone,
            "user_data_dir": user_data_dir,
            "java_script_enabled": True,
            "accept_downloads": False,
            "ignore_https_errors": True
        }
        
        context = await browser.new_context(**context_options)
        
        # Create browser instance
        instance = BrowserInstance(browser, context, fingerprint, proxy_config)
        
        # Generate instance ID and track
        instance_id = f"instance_{len(self.active_instances)}"
        self.active_instances[instance_id] = instance
        
        logger.info(f"Created browser instance {instance_id} with fingerprint")
        return instance
    
    async def close_instance(self, instance: BrowserInstance) -> None:
        """
        Close browser instance and cleanup.
        
        Args:
            instance: Browser instance to close
        """
        # Find and remove from active instances
        instance_id = None
        for iid, inst in self.active_instances.items():
            if inst == instance:
                instance_id = iid
                break
        
        if instance_id:
            del self.active_instances[instance_id]
        
        # Close the instance
        await instance.close()
        
        logger.info(f"Closed browser instance {instance_id}")
    
    async def check_detection_signals(self, page: Page) -> DetectionScore:
        """
        Check page for bot detection signals.
        
        Args:
            page: Page to analyze
            
        Returns:
            Detection score with risk assessment
        """
        detection_factors = {}
        
        try:
            # Check for common bot detection scripts
            content = await page.content()
            
            # Known bot detection services
            detection_services = [
                "distil", "perimeterx", "datadome", "cloudflare", 
                "recaptcha", "hcaptcha", "funcaptcha", "arkose"
            ]
            
            for service in detection_services:
                if service.lower() in content.lower():
                    detection_factors[f"{service}_detected"] = 0.3
            
            # Check for automation detection
            webdriver_detected = await page.evaluate("""
                () => {
                    return window.navigator.webdriver === true;
                }
            """)
            
            if webdriver_detected:
                detection_factors["webdriver_exposed"] = 0.5
            
            # Check for missing chrome object
            chrome_missing = await page.evaluate("""
                () => {
                    return typeof window.chrome === 'undefined';
                }
            """)
            
            if chrome_missing:
                detection_factors["chrome_missing"] = 0.2
            
            # Check for suspicious navigator properties
            navigator_check = await page.evaluate("""
                () => {
                    const suspicious = [];
                    if (navigator.plugins.length === 0) suspicious.push('no_plugins');
                    if (navigator.languages.length === 0) suspicious.push('no_languages');
                    if (!navigator.permissions) suspicious.push('no_permissions');
                    return suspicious;
                }
            """)
            
            for issue in navigator_check:
                detection_factors[f"navigator_{issue}"] = 0.1
            
            # Check for headless detection
            headless_signals = await page.evaluate("""
                () => {
                    const signals = [];
                    if (window.outerHeight === 0) signals.push('zero_outer_height');
                    if (window.outerWidth === 0) signals.push('zero_outer_width');
                    if (!window.chrome || !window.chrome.runtime) signals.push('no_chrome_runtime');
                    return signals;
                }
            """)
            
            for signal in headless_signals:
                detection_factors[f"headless_{signal}"] = 0.2
            
        except Exception as e:
            logger.warning(f"Error checking detection signals: {e}")
            detection_factors["check_error"] = 0.1
        
        # Calculate overall score
        total_score = min(1.0, sum(detection_factors.values()))
        
        # Generate recommendations
        recommendations = []
        if total_score > 0.7:
            recommendations.append("High detection risk - consider rotating fingerprint")
        if "webdriver_exposed" in detection_factors:
            recommendations.append("WebDriver detection - check stealth configuration")
        if any("headless" in key for key in detection_factors):
            recommendations.append("Headless detection - consider running with GUI")
        
        return DetectionScore(
            score=total_score,
            factors=detection_factors,
            recommendations=recommendations
        )
    
    async def activate_cooldown(self, instance: BrowserInstance, duration: float = 300) -> None:
        """
        Activate cooldown procedure for detected instance.
        
        Args:
            instance: Browser instance to cool down
            duration: Cooldown duration in seconds
        """
        logger.warning(f"Activating cooldown for browser instance (duration: {duration}s)")
        
        # Close all pages
        for page in instance.pages[:]:
            await instance.close_page(page)
        
        # Wait for cooldown period
        await asyncio.sleep(duration)
        
        logger.info("Cooldown period completed")
    
    async def get_instance_stats(self) -> Dict:
        """
        Get statistics for all active browser instances.
        
        Returns:
            Statistics dictionary
        """
        current_time = asyncio.get_event_loop().time()
        
        stats = {
            "total_instances": len(self.active_instances),
            "total_pages": sum(len(inst.pages) for inst in self.active_instances.values()),
            "average_detection_score": 0.0,
            "instances_by_age": {"new": 0, "medium": 0, "old": 0},
            "temp_directories": len(self.temp_dirs)
        }
        
        if self.active_instances:
            # Calculate average detection score
            total_detection = sum(inst.detection_score for inst in self.active_instances.values())
            stats["average_detection_score"] = total_detection / len(self.active_instances)
            
            # Categorize by age
            for instance in self.active_instances.values():
                age = current_time - instance.created_at
                if age < 300:  # 5 minutes
                    stats["instances_by_age"]["new"] += 1
                elif age < 1800:  # 30 minutes
                    stats["instances_by_age"]["medium"] += 1
                else:
                    stats["instances_by_age"]["old"] += 1
        
        return stats
    
    async def cleanup_inactive_instances(self, max_idle_time: float = 1800) -> int:
        """
        Clean up instances that have been inactive for too long.
        
        Args:
            max_idle_time: Maximum idle time in seconds
            
        Returns:
            Number of instances cleaned up
        """
        current_time = asyncio.get_event_loop().time()
        cleanup_count = 0
        
        instances_to_cleanup = []
        for instance_id, instance in self.active_instances.items():
            if current_time - instance.last_activity > max_idle_time:
                instances_to_cleanup.append((instance_id, instance))
        
        for instance_id, instance in instances_to_cleanup:
            await self.close_instance(instance)
            cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} inactive browser instances")
        return cleanup_count
    
    async def shutdown(self) -> None:
        """Shutdown browser manager and cleanup all resources."""
        # Close all active instances
        for instance in list(self.active_instances.values()):
            await self.close_instance(instance)
        
        # Close Playwright
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        
        # Cleanup temporary directories
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
        
        self.temp_dirs.clear()
        
        logger.info("Browser instance manager shutdown complete")


# Global browser manager instance
_browser_manager: Optional[BrowserInstanceManager] = None


def get_browser_manager() -> BrowserInstanceManager:
    """
    Get global browser instance manager.
    
    Returns:
        Browser instance manager
    """
    global _browser_manager
    
    if _browser_manager is None:
        _browser_manager = BrowserInstanceManager()
    
    return _browser_manager


async def init_browser_manager(stealth_config: Optional[StealthConfig] = None) -> BrowserInstanceManager:
    """
    Initialize global browser instance manager.
    
    Args:
        stealth_config: Optional stealth configuration
        
    Returns:
        Initialized browser manager
    """
    global _browser_manager
    
    _browser_manager = BrowserInstanceManager(stealth_config)
    await _browser_manager.initialize()
    
    return _browser_manager