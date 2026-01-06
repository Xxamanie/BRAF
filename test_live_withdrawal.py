#!/usr/bin/env python3
"""
🔴 LIVE CRYPTO WITHDRAWAL TEST - REAL MONEY
This will send actual cryptocurrency to a wallet address
"""

import sys
import os
sys.path.append('monetization-system')

from crypto.real_crypto_infrastructure import RealCryptoInfrastructure
from payments.ton_integration import ton_client
from datetime import datetime
import json

def test_live_crypto_withdrawal():
    """Test live cryptocurrency withdrawal with small amount"""

    print("🔴 REAL CRYPTO WITHDRAWAL TEST")
    print("=" * 50)
    print("⚠️  WARNING: This will send REAL cryptocurrency!")
    print("💰 Small test amount only ($1-5)")
    print("🏦 Make sure you control the destination wallet")
    print()

    # Get user confirmation
    confirm = input("Do you want to proceed with live withdrawal test? (type 'YES' to continue): ")
    if confirm != 'YES':
        print("❌ Test cancelled by user")
        return

    print("\nSelect cryptocurrency for test withdrawal:")
    print("1. TON (recommended - fast and cheap)")
    print("2. BTC (expensive - $5+ fee)")
    print("3. Cancel")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '3':
        print("❌ Test cancelled")
        return

    # Get wallet address
    wallet_address = input("Enter destination wallet address: ").strip()
    if not wallet_address:
        print("❌ No wallet address provided")
        return

    # Validate wallet address format
    if choice == '1':  # TON
        if not wallet_address.startswith(('UQ', 'EQ')) or len(wallet_address) != 48:
            print("❌ Invalid TON address format")
            return
        currency = 'TON'
        test_amount = 0.1  # $0.50 worth of TON

    elif choice == '2':  # BTC
        if not (wallet_address.startswith('1') or wallet_address.startswith('3') or wallet_address.startswith('bc1')):
            print("❌ Invalid Bitcoin address format")
            return
        currency = 'BTC'
        test_amount = 0.00001  # Very small BTC amount
    else:
        print("❌ Invalid choice")
        return

    print(f"\n🔴 LIVE WITHDRAWAL CONFIRMATION:")
    print(f"💰 Amount: {test_amount} {currency}")
    print(f"🏦 To: {wallet_address}")
    print(f"💸 Will cost real money from NOWPayments account")
    print(f"⏱️  May take 5-60 minutes for blockchain confirmation")
    print()

    final_confirm = input("Type 'CONFIRM' to proceed with real withdrawal: ")
    if final_confirm != 'CONFIRM':
        print("❌ Withdrawal cancelled")
        return

    print("\n🚀 INITIATING LIVE WITHDRAWAL...")
    print(f"Currency: {currency}")
    print(f"Amount: {test_amount}")
    print(f"Destination: {wallet_address}")
    print("-" * 50)

    try:
        # Initialize crypto infrastructure
        crypto_infra = RealCryptoInfrastructure()

        # Prepare withdrawal request
        withdrawal_request = {
            'user_id': 'live_test_user',
            'enterprise_id': 'braf_live_test',
            'amount': test_amount,
            'currency': currency,
            'wallet_address': wallet_address
        }

        print("📡 Contacting NOWPayments API...")
        result = crypto_infra.process_real_withdrawal(withdrawal_request)

        print("\n📋 WITHDRAWAL RESULT:")
        print(json.dumps(result, indent=2))

        if result.get('success'):
            print("\n✅ SUCCESS! Real cryptocurrency withdrawal initiated!")
            print(f"🔗 Transaction ID: {result.get('transaction_id')}")
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️  Estimated confirmation: {result.get('estimated_confirmation_time', 'Unknown')}")

            # Save transaction details
            tx_record = {
                'timestamp': datetime.now().isoformat(),
                'type': 'live_test_withdrawal',
                'currency': currency,
                'amount': test_amount,
                'wallet_address': wallet_address,
                'result': result
            }

            with open(f'live_withdrawal_test_{int(datetime.now().timestamp())}.json', 'w') as f:
                json.dump(tx_record, f, indent=2)

            print("💾 Transaction record saved")

            print("\n🔍 Monitor transaction at:")
            explorer_url = result.get('blockchain_explorer_url')
            if explorer_url:
                print(f"🌐 {explorer_url}")

        else:
            print("\n❌ WITHDRAWAL FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 Live withdrawal test completed")
    print("Check your wallet and transaction records")

if __name__ == "__main__":
    test_live_crypto_withdrawal()