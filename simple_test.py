#!/usr/bin/env python3
"""
Simple test to verify BRAF dependencies and basic imports work
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import asyncio
        import fastapi
        import uvicorn
        import playwright
        import redis
        import celery
        print("SUCCESS: Basic dependencies imported")
        return True
    except ImportError as e:
        print(f"FAILED: Basic dependencies import failed: {e}")
        return False

def test_braf_imports():
    """Test BRAF-specific imports"""
    try:
        from src.braf.core.task_executor import TaskExecutor
        print("SUCCESS: BRAF core imports working")
        return True
    except ImportError as e:
        print(f"FAILED: BRAF core imports failed: {e}")
        return False

def test_monetization_imports():
    """Test monetization system imports"""
    try:
        from monetization_system.crypto.real_crypto_infrastructure import RealCryptoInfrastructure
        from monetization_system.external_bypass_engine import ExternalBypassEngine
        from monetization_system.value_source_engine import ValueSourceEngine
        from monetization_system.settlement_monitor import SettlementMonitor
        from monetization_system.true_stealth_engine import TrueStealthEngine
        print("SUCCESS: Enhanced monetization system imports working")
        return True
    except ImportError as e:
        print(f"FAILED: Enhanced monetization system imports failed: {e}")
        return False

def test_supporting_modules():
    """Test supporting module imports"""
    try:
        from monetization_system.external_bypass.kyc_evasion import SyntheticIdentityGenerator
        from monetization_system.external_bypass.aml_bypass import TransactionStructuringEngine
        from monetization_system.value_sources.survey_hijacking import SurveyAutomationEngine
        print("SUCCESS: Supporting module imports working")
        return True
    except ImportError as e:
        print(f"FAILED: Supporting module imports failed: {e}")
        return False

def main():
    print("Testing BRAF Dependencies and Imports")
    print("=" * 40)

    tests = [
        test_basic_imports,
        test_braf_imports,
        test_monetization_imports,
        test_supporting_modules
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Results: {passed}/{len(tests)} test groups passed")

    if passed == len(tests):
        print("SUCCESS: All BRAF dependencies and imports working!")
        print("Enhanced BRAF system is ready for operation.")
        return 0
    else:
        print("FAILED: Some dependencies or imports are missing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())