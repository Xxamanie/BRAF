"""
CAPTCHA solving module for BRAF.

This module provides CAPTCHA solving capabilities using multiple services
with OCR fallbacks and test mode support.
"""

from braf.core.captcha.captcha_solver import (
    CaptchaSolver,
    TwoCaptchaService,
    AntiCaptchaService,
    OCRFallback,
    TestModeCaptchaSolver,
    get_captcha_solver,
    init_captcha_solver
)

__all__ = [
    "CaptchaSolver",
    "TwoCaptchaService", 
    "AntiCaptchaService",
    "OCRFallback",
    "TestModeCaptchaSolver",
    "get_captcha_solver",
    "init_captcha_solver"
]
