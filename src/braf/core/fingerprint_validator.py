"""
Fingerprint validation utilities for BRAF.

This module provides validation functions to ensure fingerprint consistency
and detect potential issues that could lead to detection.
"""

import logging
from typing import Dict, List, Optional, Tuple

from braf.core.models import BrowserFingerprint

logger = logging.getLogger(__name__)


class FingerprintValidator:
    """Validator for browser fingerprint consistency and realism."""
    
    # Known inconsistent combinations that could trigger detection
    INCONSISTENT_COMBINATIONS = [
        # Windows user agent with Mac platform
        ("Windows", "MacIntel"),
        ("Macintosh", "Win32"),
        ("X11", "Win32"),
        ("X11", "MacIntel"),
    ]
    
    # Minimum expected fonts for different platforms
    MIN_FONTS_BY_PLATFORM = {
        "Win32": 10,
        "MacIntel": 8,
        "Linux x86_64": 6,
    }
    
    # Expected WebGL vendors by platform
    WEBGL_VENDORS_BY_PLATFORM = {
        "Win32": ["Google Inc.", "NVIDIA Corporation", "AMD", "Intel Inc."],
        "MacIntel": ["Intel Inc.", "AMD", "Apple Inc."],
        "Linux x86_64": ["Mesa", "NVIDIA Corporation", "AMD", "Intel Inc."],
    }
    
    def validate_fingerprint(self, fingerprint: BrowserFingerprint) -> Tuple[bool, List[str]]:
        """
        Validate fingerprint for consistency and realism.
        
        Args:
            fingerprint: Browser fingerprint to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check user agent and platform consistency
        ua_platform_issues = self._validate_user_agent_platform(fingerprint)
        issues.extend(ua_platform_issues)
        
        # Check screen resolution realism
        screen_issues = self._validate_screen_resolution(fingerprint)
        issues.extend(screen_issues)
        
        # Check WebGL consistency
        webgl_issues = self._validate_webgl_consistency(fingerprint)
        issues.extend(webgl_issues)
        
        # Check font list realism
        font_issues = self._validate_font_list(fingerprint)
        issues.extend(font_issues)
        
        # Check hardware consistency
        hardware_issues = self._validate_hardware_consistency(fingerprint)
        issues.extend(hardware_issues)
        
        # Check language consistency
        language_issues = self._validate_language_consistency(fingerprint)
        issues.extend(language_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_user_agent_platform(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Validate user agent and platform consistency."""
        issues = []
        
        user_agent = fingerprint.user_agent
        platform = fingerprint.platform
        
        # Check for known inconsistent combinations
        for ua_part, platform_part in self.INCONSISTENT_COMBINATIONS:
            if ua_part in user_agent and platform == platform_part:
                issues.append(f"Inconsistent user agent '{ua_part}' with platform '{platform_part}'")
        
        # Check timezone consistency with user agent
        if "Windows" in user_agent and fingerprint.timezone.startswith("America/"):
            # This is fine - Windows in Americas
            pass
        elif "Macintosh" in user_agent and not fingerprint.timezone.startswith(("America/", "Europe/")):
            # Mac users are typically in Americas or Europe
            issues.append(f"Unusual timezone '{fingerprint.timezone}' for Mac user agent")
        
        return issues
    
    def _validate_screen_resolution(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Validate screen resolution realism."""
        issues = []
        
        width, height = fingerprint.screen_resolution
        
        # Check for common resolutions
        common_resolutions = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
            (2560, 1440), (1280, 720), (1600, 900), (1024, 768)
        ]
        
        if (width, height) not in common_resolutions:
            # Check if it's at least a reasonable aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 1.2 or aspect_ratio > 2.5:
                issues.append(f"Unusual screen resolution {width}x{height} with aspect ratio {aspect_ratio:.2f}")
        
        # Check for unrealistic resolutions
        if width < 800 or height < 600:
            issues.append(f"Screen resolution {width}x{height} is too small for modern browsers")
        
        if width > 7680 or height > 4320:  # 8K resolution
            issues.append(f"Screen resolution {width}x{height} is unusually high")
        
        return issues
    
    def _validate_webgl_consistency(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Validate WebGL vendor and renderer consistency."""
        issues = []
        
        platform = fingerprint.platform
        webgl_vendor = fingerprint.webgl_vendor
        webgl_renderer = fingerprint.webgl_renderer
        
        # Check if WebGL vendor is appropriate for platform
        expected_vendors = self.WEBGL_VENDORS_BY_PLATFORM.get(platform, [])
        
        if expected_vendors:
            vendor_match = any(vendor in webgl_vendor for vendor in expected_vendors)
            if not vendor_match:
                issues.append(f"WebGL vendor '{webgl_vendor}' unusual for platform '{platform}'")
        
        # Check for ANGLE consistency (Windows-specific)
        if platform == "Win32" and "ANGLE" not in webgl_renderer and "Google Inc." in webgl_vendor:
            issues.append("Google WebGL vendor on Windows should typically use ANGLE renderer")
        
        # Check for Mac-specific patterns
        if platform == "MacIntel" and "Intel Iris" in webgl_renderer and "Intel Inc." not in webgl_vendor:
            issues.append("Intel Iris renderer should have Intel Inc. as vendor")
        
        return issues
    
    def _validate_font_list(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Validate font list realism."""
        issues = []
        
        fonts = fingerprint.fonts
        platform = fingerprint.platform
        
        # Check minimum font count
        min_fonts = self.MIN_FONTS_BY_PLATFORM.get(platform, 5)
        if len(fonts) < min_fonts:
            issues.append(f"Too few fonts ({len(fonts)}) for platform '{platform}', expected at least {min_fonts}")
        
        # Check for platform-specific fonts
        if platform == "Win32":
            windows_fonts = ["Arial", "Times New Roman", "Calibri", "Segoe UI"]
            if not any(font in fonts for font in windows_fonts):
                issues.append("Missing common Windows fonts")
        
        elif platform == "MacIntel":
            mac_fonts = ["Helvetica", "Times", "Courier", "Arial"]
            if not any(font in fonts for font in mac_fonts):
                issues.append("Missing common Mac fonts")
        
        # Check for duplicate fonts
        if len(fonts) != len(set(fonts)):
            issues.append("Duplicate fonts in font list")
        
        return issues
    
    def _validate_hardware_consistency(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Validate hardware specification consistency."""
        issues = []
        
        concurrency = fingerprint.hardware_concurrency
        memory = fingerprint.device_memory
        
        # Check reasonable hardware concurrency
        if concurrency < 1 or concurrency > 32:
            issues.append(f"Unrealistic hardware concurrency: {concurrency}")
        
        # Check reasonable device memory
        if memory < 1 or memory > 128:
            issues.append(f"Unrealistic device memory: {memory}GB")
        
        # Check consistency between concurrency and memory
        if concurrency >= 16 and memory < 8:
            issues.append(f"High CPU cores ({concurrency}) with low memory ({memory}GB) is unusual")
        
        if concurrency <= 2 and memory > 32:
            issues.append(f"Low CPU cores ({concurrency}) with high memory ({memory}GB) is unusual")
        
        return issues
    
    def _validate_language_consistency(self, fingerprint: BrowserFingerprint) -> List[str]:
        """Validate language and locale consistency."""
        issues = []
        
        languages = fingerprint.languages
        timezone = fingerprint.timezone
        
        # Check language format
        for lang in languages:
            if not (len(lang) == 2 or (len(lang) == 5 and lang[2] == '-')):
                issues.append(f"Invalid language format: '{lang}'")
        
        # Check timezone and language consistency
        if timezone.startswith("America/") and not any("en" in lang for lang in languages):
            # Americas without English is unusual but not impossible
            pass
        
        if timezone.startswith("Europe/") and "en-US" in languages and "en-GB" not in languages:
            # European timezone with US English but no British English
            issues.append("European timezone with US English locale may be suspicious")
        
        return issues
    
    def calculate_realism_score(self, fingerprint: BrowserFingerprint) -> float:
        """
        Calculate realism score for fingerprint.
        
        Args:
            fingerprint: Browser fingerprint to score
            
        Returns:
            Realism score from 0.0 (unrealistic) to 1.0 (highly realistic)
        """
        is_valid, issues = self.validate_fingerprint(fingerprint)
        
        if is_valid:
            return 1.0
        
        # Deduct points for each issue
        score = 1.0 - (len(issues) * 0.1)
        return max(0.0, score)
    
    def suggest_improvements(self, fingerprint: BrowserFingerprint) -> List[str]:
        """
        Suggest improvements for fingerprint realism.
        
        Args:
            fingerprint: Browser fingerprint to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        is_valid, issues = self.validate_fingerprint(fingerprint)
        
        if is_valid:
            return ["Fingerprint appears realistic"]
        
        # Provide specific suggestions based on issues
        for issue in issues:
            if "Inconsistent user agent" in issue:
                suggestions.append("Update user agent to match platform")
            elif "Unusual screen resolution" in issue:
                suggestions.append("Use a more common screen resolution")
            elif "WebGL vendor" in issue:
                suggestions.append("Adjust WebGL vendor to match platform")
            elif "Too few fonts" in issue:
                suggestions.append("Add more platform-appropriate fonts")
            elif "Missing common" in issue and "fonts" in issue:
                suggestions.append("Include standard system fonts for the platform")
            elif "hardware concurrency" in issue:
                suggestions.append("Use realistic CPU core count")
            elif "device memory" in issue:
                suggestions.append("Use realistic memory amount")
            elif "language format" in issue:
                suggestions.append("Fix language code format (e.g., 'en-US')")
        
        return suggestions


# Global validator instance
_validator = FingerprintValidator()


def validate_fingerprint(fingerprint: BrowserFingerprint) -> Tuple[bool, List[str]]:
    """
    Validate fingerprint using global validator.
    
    Args:
        fingerprint: Browser fingerprint to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    return _validator.validate_fingerprint(fingerprint)


def calculate_fingerprint_realism_score(fingerprint: BrowserFingerprint) -> float:
    """
    Calculate realism score for fingerprint.
    
    Args:
        fingerprint: Browser fingerprint to score
        
    Returns:
        Realism score from 0.0 to 1.0
    """
    return _validator.calculate_realism_score(fingerprint)