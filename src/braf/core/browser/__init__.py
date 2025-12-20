"""
Browser management module for BRAF.

This module provides browser instance management with stealth capabilities,
fingerprint application, and behavioral automation utilities.
"""

from braf.core.browser.browser_instance import (
    BrowserInstance,
    BrowserInstanceManager,
    StealthConfig,
    get_browser_manager,
    init_browser_manager
)

# Alias for compatibility
get_browser_instance_manager = get_browser_manager
from braf.core.browser.automation_utils import (
    AutomationExecutor,
    PageAnalyzer,
    execute_automation_sequence
)

__all__ = [
    # Browser instance management
    "BrowserInstance",
    "BrowserInstanceManager", 
    "StealthConfig",
    "get_browser_manager",
    "get_browser_instance_manager",
    "init_browser_manager",
    
    # Automation utilities
    "AutomationExecutor",
    "PageAnalyzer",
    "execute_automation_sequence"
]