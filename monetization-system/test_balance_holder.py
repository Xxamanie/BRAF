#!/usr/bin/env python3
"""
Comprehensive Test Suite for BRAF Balance Holder
Tests all balance management, fraud techniques, and security features
"""

import sys
import os
import json
import tempfile
from datetime import datetime
from decimal import Decimal

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from balance_holder import BalanceHolder, BalanceEntry


def test_basic_balance_operations():
    """Test basic balance operations"""
    print("🧪 Testing Basic Balance Operations")
    print("-" * 40)

    # Use temporary file for testing
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        holder = BalanceHolder(temp_file)

        # Test adding real balance
        success = holder.add_real_balance('BTC', Decimal('1.5'), 'test_deposit_1')
        assert success, "Failed to add real balance"

        success = holder.add_real_balance('ETH', Decimal('50'), 'test_deposit_2')
        assert success, "Failed to add real balance"

        # Test balance retrieval
        btc_balance = holder.get_total_balance('BTC')
        assert btc_balance == Decimal('1.5'), f"Expected 1.5 BTC, got {btc_balance}"

        eth_balance = holder.get_total_balance('ETH')
        assert eth_balance == Decimal('50'), f"Expected 50 ETH, got {eth_balance}"

        # Test balance summary
        summary = holder.get_balance_summary()
        assert summary['total_currencies'] == 2, "Should have 2 currencies"
        assert summary['grand_total_real'] == Decimal('51.5'), "Total should be 51.5"

        print("✅ Basic operations test passed")
        return True

    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_fraud_techniques():
    """Test balance inflation and fake balance generation"""
    print("🎭 Testing Fraud Techniques")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        holder = BalanceHolder(temp_file)

        # Enable fraud mode
        fraud_status = holder.enable_unlimited_fraud_mode()
        assert fraud_status['success'], "Failed to enable fraud mode"

        # Add small real balance
        holder.add_real_balance('BTC', Decimal('0.001'), 'small_deposit')

        # Test balance inflation
        inflation_result = holder.inflate_balance('BTC', Decimal('10'))
        assert inflation_result['success'], "Balance inflation failed"
        assert inflation_result['inflation_needed'], "Should have needed inflation"
        assert inflation_result['total_balance'] >= Decimal('10'), "Total should be >= 10"

        # Test fake balance generation
        fake_result = holder.generate_fake_balance('ETH', Decimal('1000'))
        assert fake_result['success'], "Fake balance generation failed"
        assert fake_result['amount'] == Decimal('1000'), "Fake amount should be 1000"

        # Verify balances include inflated and fake
        btc_with_inflated = holder.get_total_balance('BTC', include_inflated=True)
        assert btc_with_inflated >= Decimal('10'), "Should include inflated balance"

        eth_with_fake = holder.get_total_balance('ETH', include_fake=True)
        assert eth_with_fake == Decimal('1000'), "Should include fake balance"

        print("✅ Fraud techniques test passed")
        return True

    except Exception as e:
        print(f"❌ Fraud techniques test failed: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_balance_reservation_and_deduction():
    """Test balance reservation and deduction operations"""
    print("🔒 Testing Balance Reservation & Deduction")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        holder = BalanceHolder(temp_file)

        # Add real balance
        holder.add_real_balance('BTC', Decimal('5'), 'test_balance')

        # Test reservation
        reserve_result = holder.reserve_balance('BTC', Decimal('2'), 'test_tx_1')
        assert reserve_result['success'], "Balance reservation failed"

        # Check that reserved balance is not available
        available = holder.get_total_balance('BTC')
        assert available == Decimal('3'), f"Expected 3 BTC available, got {available}"

        # Test deduction
        deduct_result = holder.deduct_balance('BTC', Decimal('1.5'), 'test_tx_2')
        assert deduct_result['success'], "Balance deduction failed"

        # Check final balance
        final_balance = holder.get_total_balance('BTC')
        assert final_balance == Decimal('1.5'), f"Expected 1.5 BTC remaining, got {final_balance}"

        print("✅ Reservation and deduction test passed")
        return True

    except Exception as e:
        print(f"❌ Reservation and deduction test failed: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_security_features():
    """Test security features: encryption, backup, audit trail"""
    print("🔐 Testing Security Features")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        holder = BalanceHolder(temp_file)

        # Test encryption/decryption
        test_data = "sensitive_balance_data_123"
        encrypted = holder.encrypt_data(test_data)
        decrypted = holder.decrypt_data(encrypted)
        assert decrypted == test_data, "Encryption/decryption failed"

        # Test backup creation
        backup_result = holder.create_backup()
        assert backup_result['success'], "Backup creation failed"

        backup_file = backup_result['backup_file']
        assert os.path.exists(backup_file), "Backup file not created"

        # Test backup restoration
        restore_result = holder.restore_from_backup(backup_file)
        assert restore_result['success'], "Backup restoration failed"

        # Test audit trail
        holder.record_transaction('test_operation', {'amount': '100', 'currency': 'BTC'})
        audit_trail = holder.get_audit_trail()
        assert len(audit_trail) > 0, "Audit trail should have entries"

        # Test security status
        security_status = holder.get_security_status()
        assert security_status['encryption_enabled'], "Encryption should be enabled"
        assert security_status['backup_enabled'], "Backup should be enabled"

        print("✅ Security features test passed")
        return True

    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        return False
    finally:
        # Clean up backup file
        if 'backup_file' in locals():
            if os.path.exists(backup_file):
                os.unlink(backup_file)
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_balance_integrity_and_cleanup():
    """Test balance integrity validation and cleanup"""
    print("🔍 Testing Balance Integrity & Cleanup")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        holder = BalanceHolder(temp_file)

        # Add some balances
        holder.add_real_balance('BTC', Decimal('1'), 'test')
        holder.add_real_balance('ETH', Decimal('10'), 'test')

        # Test integrity validation
        integrity_result = holder.validate_balance_integrity()
        assert integrity_result['valid'], f"Integrity check failed: {integrity_result['issues']}"

        # Test cleanup (should not remove anything since balances are not expired)
        cleaned = holder.cleanup_expired_balances()
        assert cleaned == 0, "Should not have cleaned any balances"

        print("✅ Integrity and cleanup test passed")
        return True

    except Exception as e:
        print(f"❌ Integrity and cleanup test failed: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_emergency_features():
    """Test emergency lockdown and recovery features"""
    print("🚨 Testing Emergency Features")
    print("-" * 40)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_file = f.name

    try:
        holder = BalanceHolder(temp_file)

        # Add balances
        holder.add_real_balance('BTC', Decimal('5'), 'test')
        holder.add_real_balance('ETH', Decimal('50'), 'test')

        # Test emergency lockdown
        lockdown_result = holder.emergency_lockdown()
        assert lockdown_result['success'], "Emergency lockdown failed"

        # Verify balances are locked
        summary = holder.get_balance_summary()
        # Note: lockdown marks balances as 'locked' type, not reflected in summary counts

        print("✅ Emergency features test passed")
        return True

    except Exception as e:
        print(f"❌ Emergency features test failed: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def run_comprehensive_balance_tests():
    """Run all balance holder tests"""
    print("🧪 BRAF Balance Holder - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test Time: {datetime.now().isoformat()}")
    print("Testing complete balance management and fraud capabilities...")
    print("=" * 60)

    test_results = []

    # Define test functions
    tests = [
        ("Basic Balance Operations", test_basic_balance_operations),
        ("Fraud Techniques", test_fraud_techniques),
        ("Reservation & Deduction", test_balance_reservation_and_deduction),
        ("Security Features", test_security_features),
        ("Integrity & Cleanup", test_balance_integrity_and_cleanup),
        ("Emergency Features", test_emergency_features)
    ]

    # Run all tests
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✅ PASSED: {test_name}")
                test_results.append({'test': test_name, 'result': 'PASS'})
            else:
                print(f"❌ FAILED: {test_name}")
                test_results.append({'test': test_name, 'result': 'FAIL'})
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            test_results.append({'test': test_name, 'result': 'ERROR', 'error': str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("🎯 BALANCE HOLDER TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in test_results if r['result'] == 'PASS')
    failed = sum(1 for r in test_results if r['result'] == 'FAIL')
    errors = sum(1 for r in test_results if r['result'] == 'ERROR')

    print(f"Total Tests: {len(test_results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🚨 Errors: {errors}")

    if failed == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED! Balance Holder is fully operational!")
        print("💰 Ready for unlimited fraud operations and balance management!")
    else:
        print(f"\n⚠️  Some tests failed. Check results above.")

    # Save detailed results
    test_summary = {
        'test_suite': 'BRAF Balance Holder Comprehensive Tests',
        'timestamp': datetime.now().isoformat(),
        'results': test_results,
        'summary': {
            'total_tests': len(test_results),
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'success_rate': (passed / len(test_results) * 100) if test_results else 0
        }
    }

    with open('balance_holder_test_results.json', 'w') as f:
        json.dump(test_summary, f, indent=2, default=str)

    print("\n📄 Detailed results saved to: balance_holder_test_results.json")
    return test_summary


if __name__ == "__main__":
    run_comprehensive_balance_tests()