#!/usr/bin/env python3
"""
Test currency conversion for OPay/PalmPay withdrawals
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8003"
ENTERPRISE_ID = "e9e9d28b-62d1-4452-b0df-e1f1cf6e4721"  # Test account

def test_currency_conversion():
    """Test currency conversion functionality"""
    print("🧪 Testing Currency Conversion for OPay/PalmPay")
    print("=" * 50)
    
    # Test OPay withdrawal with currency conversion
    print("\n📱 Testing OPay Withdrawal (USD to NGN conversion):")
    
    withdrawal_data = {
        "amount": 100.0,  # $100 USD
        "provider": "opay",
        "recipient": "+234XXXXXXXXXX"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/withdrawal/create/{ENTERPRISE_ID}",
            json=withdrawal_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OPay withdrawal request successful!")
            print(f"   💰 Original USD Amount: ${result['original_amount_usd']}")
            print(f"   🔄 Converted Amount: {result['converted_amount']} {result['currency']}")
            print(f"   📊 Exchange Rate: 1 USD = {result['exchange_rate']} {result['currency']}")
            print(f"   💸 Fee: {result['fee']} {result['currency']}")
            print(f"   💵 Net Amount: {result['net_amount']} {result['currency']}")
            print(f"   🆔 Transaction ID: {result['transaction_id']}")
        else:
            print(f"❌ OPay withdrawal failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error testing OPay withdrawal: {e}")
    
    # Test PalmPay withdrawal
    print("\n💳 Testing PalmPay Withdrawal (USD to NGN conversion):")
    
    withdrawal_data = {
        "amount": 50.0,  # $50 USD
        "provider": "palmpay",
        "recipient": "+234XXXXXXXXXX"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/withdrawal/create/{ENTERPRISE_ID}",
            json=withdrawal_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ PalmPay withdrawal request successful!")
            print(f"   💰 Original USD Amount: ${result['original_amount_usd']}")
            print(f"   🔄 Converted Amount: {result['converted_amount']} {result['currency']}")
            print(f"   📊 Exchange Rate: 1 USD = {result['exchange_rate']} {result['currency']}")
            print(f"   💸 Fee: {result['fee']} {result['currency']}")
            print(f"   💵 Net Amount: {result['net_amount']} {result['currency']}")
            print(f"   🆔 Transaction ID: {result['transaction_id']}")
        else:
            print(f"❌ PalmPay withdrawal failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error testing PalmPay withdrawal: {e}")
    
    # Test Crypto withdrawal (should remain in USD)
    print("\n₿ Testing Crypto Withdrawal (USD - no conversion):")
    
    withdrawal_data = {
        "amount": 200.0,  # $200 USD
        "provider": "crypto",
        "recipient": "TXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx"  # Sample USDT address
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/withdrawal/create/{ENTERPRISE_ID}",
            json=withdrawal_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Crypto withdrawal request successful!")
            print(f"   💰 Original USD Amount: ${result['original_amount_usd']}")
            print(f"   💵 Amount: {result['converted_amount']} {result['currency']}")
            print(f"   💸 Fee: {result['fee']} {result['currency']}")
            print(f"   💵 Net Amount: {result['net_amount']} {result['currency']}")
            print(f"   🆔 Transaction ID: {result['transaction_id']}")
        else:
            print(f"❌ Crypto withdrawal failed: {response.status_code}")
            print(f"   Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error testing crypto withdrawal: {e}")
    
    # Test withdrawal history to see currency display
    print("\n📊 Testing Withdrawal History:")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/dashboard/withdrawals/{ENTERPRISE_ID}")
        
        if response.status_code == 200:
            result = response.json()
            withdrawals = result.get("recent_withdrawals", [])
            print(f"✅ Found {len(withdrawals)} recent withdrawals:")
            
            for i, withdrawal in enumerate(withdrawals[:3], 1):
                print(f"   {i}. {withdrawal['provider'].upper()}: {withdrawal['net_amount']} {withdrawal.get('currency', 'USD')} - {withdrawal['status']}")
        else:
            print(f"❌ Failed to get withdrawal history: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Error getting withdrawal history: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Currency conversion testing completed!")
    print("📍 Key Points:")
    print("   • OPay/PalmPay: USD earnings converted to NGN for withdrawal")
    print("   • Crypto: Remains in USD (no conversion needed)")
    print("   • Exchange rates: Fetched from live APIs with fallback rates")
    print("   • Fees: Calculated in the withdrawal currency")

if __name__ == "__main__":
    test_currency_conversion()