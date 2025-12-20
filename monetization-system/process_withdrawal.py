#!/usr/bin/env python3
"""
Process withdrawal request: $5,000 USD to OPay account 8161129466
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://127.0.0.1:8003"
ENTERPRISE_ID = "e9e9d28b-62d1-4452-b0df-e1f1cf6e4721"  # Test account

def process_withdrawal():
    """Process the $5,000 USD withdrawal to OPay"""
    
    print("💰 PROCESSING WITHDRAWAL REQUEST")
    print("=" * 50)
    print(f"🕐 Request Time: {datetime.now().isoformat()}")
    print(f"💵 Amount: $5,000 USD")
    print(f"📱 Provider: OPay")
    print(f"🔢 Account: 8161129466")
    print("=" * 50)
    
    # First, check available balance
    print("\n💳 Checking available balance...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/dashboard/overview/{ENTERPRISE_ID}")
        if response.status_code == 200:
            data = response.json()
            balance_info = data.get('data', {})
            available_balance = balance_info.get('available_balance', 0)
            
            print(f"✅ Available Balance: ${available_balance:.2f} USD")
            
            if available_balance < 5000:
                print(f"❌ Insufficient balance! Need $5,000 but only have ${available_balance:.2f}")
                return False
            else:
                print(f"✅ Sufficient balance for withdrawal")
        else:
            print(f"❌ Could not check balance: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Balance check error: {e}")
        return False
    
    # Get current exchange rate
    print("\n💱 Getting current USD to NGN exchange rate...")
    try:
        from payments.currency_converter import currency_converter
        
        rate_info = currency_converter.get_rate_info("USD", "NGN")
        current_rate = rate_info["rate"]
        is_live = not rate_info["is_fallback"]
        
        print(f"📊 Current Rate: 1 USD = {current_rate} NGN")
        print(f"🔄 Rate Source: {'Live API' if is_live else 'Fallback'}")
        
        # Calculate conversion
        calc = currency_converter.calculate_withdrawal_amounts(5000, "opay")
        
        print(f"\n💰 Withdrawal Calculation:")
        print(f"   💵 USD Amount: ${calc['original_usd_amount']}")
        print(f"   🔄 NGN Amount: ₦{calc['converted_amount']:,.0f}")
        print(f"   💸 Fee (1.5%): ₦{calc['fee_amount']:,.0f}")
        print(f"   💰 Net Amount: ₦{calc['net_amount']:,.0f}")
        print(f"   ✅ Valid: {'Yes' if calc['is_valid'] else 'No'}")
        
        if not calc['is_valid']:
            print(f"❌ Withdrawal amount is below minimum threshold")
            return False
            
    except Exception as e:
        print(f"❌ Currency conversion error: {e}")
        return False
    
    # Process the withdrawal
    print(f"\n🚀 Processing withdrawal to OPay account 8161129466...")
    
    withdrawal_data = {
        "amount": 5000.0,
        "provider": "opay",
        "recipient": "8161129466"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/withdrawal/create/{ENTERPRISE_ID}",
            json=withdrawal_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ WITHDRAWAL REQUEST SUCCESSFUL!")
            print("=" * 50)
            print(f"🆔 Transaction ID: {result['transaction_id']}")
            print(f"📊 Status: {result['status']}")
            print(f"💵 USD Amount: ${result['original_amount_usd']}")
            print(f"💰 NGN Amount: ₦{result['converted_amount']:,.0f}")
            print(f"📊 Exchange Rate: 1 USD = {result['exchange_rate']} NGN")
            print(f"💸 Fee: ₦{result['fee']:,.0f} NGN")
            print(f"💰 Net Amount: ₦{result['net_amount']:,.0f} NGN")
            print(f"📱 OPay Account: 8161129466")
            print(f"⏰ Estimated Completion: {result.get('estimated_completion', 'N/A')}")
            print("=" * 50)
            
            print(f"\n📋 WITHDRAWAL SUMMARY:")
            print(f"   • You requested: $5,000 USD")
            print(f"   • You will receive: ₦{result['net_amount']:,.0f} NGN")
            print(f"   • To OPay account: 8161129466")
            print(f"   • Processing time: 1-3 hours")
            print(f"   • Transaction ID: {result['transaction_id']}")
            
            return True
            
        else:
            error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
            error_message = error_data.get('detail', f'HTTP {response.status_code}')
            
            print("❌ WITHDRAWAL REQUEST FAILED!")
            print("=" * 50)
            print(f"Error: {error_message}")
            print(f"Status Code: {response.status_code}")
            
            return False
            
    except Exception as e:
        print("❌ WITHDRAWAL REQUEST ERROR!")
        print("=" * 50)
        print(f"Error: {str(e)}")
        return False

def main():
    """Main function"""
    success = process_withdrawal()
    
    if success:
        print("\n🎉 Withdrawal request completed successfully!")
        print("📱 Check your OPay account 8161129466 in 1-3 hours")
        print("🌐 View status at: http://127.0.0.1:8003/dashboard")
    else:
        print("\n❌ Withdrawal request failed!")
        print("🔧 Please check the error details above")
        print("🌐 Try again at: http://127.0.0.1:8003/request-withdrawal")

if __name__ == "__main__":
    main()