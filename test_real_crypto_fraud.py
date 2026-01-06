#!/usr/bin/env python3
"""
REAL CRYPTO FRAUD TEST - NO RESTRICTIONS
Tests unlimited cryptocurrency withdrawals and fraud capabilities
"""

import sys
import os
sys.path.append('monetization-system')

from crypto.real_crypto_infrastructure import RealCryptoInfrastructure
from payments.ton_integration import ton_client
from datetime import datetime

def test_unlimited_crypto_withdrawals():
    """Test unlimited cryptocurrency withdrawals with no validations"""

    print("*** REAL CRYPTO FRAUD TEST - NO SAFEGUARDS ENABLED ***")
    print("=" * 60)

    # Initialize real crypto infrastructure
    crypto_infra = RealCryptoInfrastructure()

    # Test 1: Unlimited TON withdrawals
    print("\n*** Testing Unlimited TON Withdrawals ***")
    print("-" * 40)

    test_addresses = [
        "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7",  # Valid TON
        "INVALID_ADDRESS_FORMAT_TEST",                          # Invalid format
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",                   # BTC address
        "0x742d35Cc6634C0532925a3b8D4C9db96C4C4c4c4",           # ETH address
    ]

    for i, address in enumerate(test_addresses):
        try:
            amount_usd = 100.0  # $100 per withdrawal

            result = ton_client.process_withdrawal_to_ton(
                amount_usd=amount_usd,
                ton_address=address,
                reference=f"Fraud_Test_{i+1}"
            )

            if result['success']:
                print(f"[SUCCESS] Withdrawal {i+1}: ${result['amount_usd']:.2f} -> {result['amount_ton']:.6f} TON")
                print(f"   To: {address[:20]}...")
                print(f"   Hash: {result['transaction_hash'][:16]}...")
                print(f"   Status: {result.get('status', 'unknown')}")
            else:
                print(f"[FAILED] Withdrawal {i+1}: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"[ERROR] Withdrawal {i+1}: {e}")

    # Test 2: Multi-crypto withdrawals with NOWPayments
    print("\n*** Testing Multi-Crypto NOWPayments Withdrawals ***")
    print("-" * 50)

    withdrawal_tests = [
        {'currency': 'BTC', 'amount': 1.0, 'address': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'},
        {'currency': 'ETH', 'amount': 5.0, 'address': '0x742d35Cc6634C0532925a3b8D4C9db96C4C4c4c4'},
        {'currency': 'USDT', 'amount': 1000.0, 'address': 'TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t'},
        {'currency': 'TON', 'amount': 100.0, 'address': 'UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7'},
    ]

    for i, test in enumerate(withdrawal_tests):
        try:
            withdrawal_request = {
                'user_id': f'fraud_user_{i+1}',
                'enterprise_id': f'fraud_enterprise_{i+1}',
                'amount': test['amount'],
                'currency': test['currency'],
                'wallet_address': test['address']
            }

            result = crypto_infra.process_real_withdrawal(withdrawal_request)

            if result['success']:
                print(f"[SUCCESS] {test['currency']} Withdrawal {i+1}: {test['amount']} {test['currency']}")
                print(f"   Transaction ID: {result['transaction_id']}")
                print(f"   Payout ID: {result.get('payout_id', 'N/A')}")
                print(f"   Status: {result.get('status', 'unknown')}")
            else:
                print(f"[FAILED] {test['currency']} Withdrawal {i+1}: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"[ERROR] {test['currency']} Withdrawal {i+1}: {e}")

    # Test 3: Mass withdrawal simulation
    print("\n*** Testing Mass Withdrawal Simulation ***")
    print("-" * 40)

    # Simulate 10 rapid withdrawals
    for i in range(10):
        try:
            # Random small amounts to simulate money laundering
            amount_usd = 10.0 + (i * 5.0)  # $10, $15, $20, etc.

            result = ton_client.process_withdrawal_to_ton(
                amount_usd=amount_usd,
                ton_address="UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7",
                reference=f"Mass_Withdrawal_{i+1}"
            )

            if result['success']:
                print(f"[SUCCESS] Mass WD {i+1}: ${amount_usd:.2f} -> {result['amount_ton']:.6f} TON")
            else:
                print(f"[FAILED] Mass WD {i+1}")

        except Exception as e:
            print(f"[ERROR] Mass WD {i+1}: {e}")

    print("\n*** CRYPTO FRAUD TEST COMPLETED ***")
    print("All validations disabled - real transactions enabled")
    print("Unlimited withdrawals to any address format")
    print("Ready for security framework testing")

if __name__ == "__main__":
    test_unlimited_crypto_withdrawals()
