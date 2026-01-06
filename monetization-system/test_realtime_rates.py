#!/usr/bin/env python3
"""
Test real-time currency conversion with multiple API sources
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from payments.currency_converter import currency_converter
import time

def test_realtime_conversion():
    """Test real-time currency conversion"""
    print("🧪 Testing Real-time Currency Conversion")
    print("=" * 50)
    
    # Test different currency pairs
    test_pairs = [
        ("USD", "NGN", 100),
        ("USD", "NGN", 50),
        ("USD", "NGN", 25),
    ]
    
    for from_curr, to_curr, amount in test_pairs:
        print(f"\n💱 Converting {amount} {from_curr} to {to_curr}:")
        
        try:
            # Get detailed rate info
            rate_info = currency_converter.get_rate_info(from_curr, to_curr)
            print(f"   📊 Rate: 1 {from_curr} = {rate_info['rate']} {to_curr}")
            print(f"   🔄 Source: {'Live API' if not rate_info['is_fallback'] else 'Fallback'}")
            print(f"   ⏰ Cached: {'Yes' if rate_info['is_cached'] else 'No'}")
            if rate_info['cache_age_minutes']:
                print(f"   🕐 Cache age: {rate_info['cache_age_minutes']:.1f} minutes")
            
            # Convert amount
            result = currency_converter.convert_amount(amount, from_curr, to_curr)
            print(f"   💰 Result: {amount} {from_curr} = {result['converted_amount']} {to_curr}")
            print(f"   ✅ Live rate: {'Yes' if result['is_live_rate'] else 'No'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test withdrawal calculation
    print(f"\n🏦 Testing Withdrawal Calculations:")
    
    providers = ["opay", "palmpay", "crypto"]
    test_amount = 100  # $100 USD
    
    for provider in providers:
        print(f"\n   📱 {provider.upper()} Withdrawal:")
        try:
            calc = currency_converter.calculate_withdrawal_amounts(test_amount, provider)
            
            print(f"      💵 USD Amount: ${calc['original_usd_amount']}")
            print(f"      🔄 Converted: {calc['converted_amount']} {calc['provider_currency']}")
            print(f"      📊 Rate: 1 USD = {calc['exchange_rate']} {calc['provider_currency']}")
            print(f"      💸 Fee: {calc['fee_amount']} {calc['provider_currency']}")
            print(f"      💰 Net Amount: {calc['net_amount']} {calc['provider_currency']}")
            print(f"      ✅ Valid: {'Yes' if calc['is_valid'] else 'No'}")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test API performance
    print(f"\n⚡ Testing API Performance:")
    
    start_time = time.time()
    for i in range(5):
        try:
            result = currency_converter.get_exchange_rate("USD", "NGN")
            print(f"   Request {i+1}: {result} NGN/USD")
        except Exception as e:
            print(f"   Request {i+1}: Error - {e}")
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 5
    print(f"   📊 Average response time: {avg_time:.2f} seconds")
    
    print("\n" + "=" * 50)
    print("🎉 Real-time Currency Testing Completed!")

if __name__ == "__main__":
    test_realtime_conversion()
