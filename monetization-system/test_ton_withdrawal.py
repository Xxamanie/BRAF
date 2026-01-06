#!/usr/bin/env python3
"""
Test TON Withdrawal System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from payments.ton_integration import ton_client

def test_ton_system():
    """Test TON integration system"""
    
    print("💎 Testing TON Integration System")
    print("=" * 50)
    
    # Test TON address validation
    ton_address = "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7"
    print(f"Testing address: {ton_address}")
    
    is_valid = ton_client.validate_ton_address(ton_address)
    print(f"Address validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    if not is_valid:
        print("Cannot proceed with invalid address")
        return
    
    # Test TON price fetching
    print("\n💰 Getting TON price...")
    price_result = ton_client.get_current_ton_price()
    if price_result['success']:
        print(f"✅ TON Price: ${price_result['price_usd']:.2f} USD")
    else:
        print(f"⚠️  Price fetch failed: {price_result.get('error')}")
    
    # Test USD to TON conversion
    print("\n🔄 Testing USD to TON conversion...")
    amount_usd = 100.0
    conversion_result = ton_client.convert_usd_to_ton(amount_usd)
    if conversion_result['success']:
        print(f"✅ ${amount_usd:.2f} USD = {conversion_result['amount_ton']:.6f} TON")
        print(f"   Rate: 1 TON = ${conversion_result['conversion_rate']:.2f} USD")
    else:
        print(f"❌ Conversion failed: {conversion_result.get('error')}")
        return
    
    # Test withdrawal processing
    print("\n🏦 Testing withdrawal processing...")
    withdrawal_result = ton_client.process_withdrawal_to_ton(
        amount_usd=amount_usd,
        ton_address=ton_address,
        reference="Test withdrawal from BRAF system"
    )
    
    if withdrawal_result['success']:
        print("✅ Withdrawal processed successfully!")
        print(f"   Withdrawal ID: {withdrawal_result['withdrawal_id']}")
        print(f"   Transaction Hash: {withdrawal_result['transaction_hash']}")
        print(f"   Amount: ${withdrawal_result['amount_usd']:.2f} USD → {withdrawal_result['amount_ton']:.6f} TON")
        print(f"   Status: {withdrawal_result.get('status', 'pending')}")
        
        if withdrawal_result.get('demo_mode'):
            print("   ⚠️  Demo Mode: No real funds transferred")
    else:
        print(f"❌ Withdrawal failed: {withdrawal_result.get('error')}")
    
    print("\n🎯 TON Integration Test Complete!")

if __name__ == "__main__":
    try:
        test_ton_system()
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
