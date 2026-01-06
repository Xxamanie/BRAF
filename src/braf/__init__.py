"""
Browser Automation Framework (BRAF)

A distributed system for ethical web automation with advanced detection evasion
and behavioral emulation capabilities.
"""

__version__ = "0.1.0"
__author__ = "BRAF Team"
__email__ = "team@braf.dev"

from braf.core.models import (
    AutomationAction,
    AutomationTask,
    BrowserFingerprint,
    ComplianceLog,
    Profile,
)

__all__ = [
    "AutomationAction",
    "AutomationTask", 
    "BrowserFingerprint",
    "ComplianceLog",
    "Profile",
]
