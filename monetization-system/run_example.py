#!/usr/bin/env python3
"""
Direct execution example matching your exact code
"""
from core.runner import run_targets

TARGETS = [
    {"url": "https://example.com", "requires_js": False}
]

run_targets(TARGETS)