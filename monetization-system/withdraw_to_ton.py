#!/usr/bin/env python3
"""
Withdraw to TON Address
Process withdrawal to the user's TON wallet
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from payments.ton_integration import ton_client
from datetime import datetime
import json

def main():
    """Process withdrawal to TON address"""
    
    # User's TON wallet address
    ton_address = "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7"
    
    print("💎 BRAF TON Withdrawal System")
    print("=" * 50)
    print(f"Destination: {ton_address}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get amount from command line or user input
    if len(sys.argv) > 1:
        try:
            amount_usd = float(sys.argv[1])
        except ValueError:
            print("❌ Invalid amount format")
            return
    else:
        try:
            amount_input = input("Enter withdrawal amount in USD: $")
            amount_usd = float(amount_input)
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Withdrawal cancelled")
            return
    
    if amount_usd < 10:
        print("❌ Minimum withdrawal amount is $10 USD")
        return
    
    print(f"💰 Processing withdrawal of ${amount_usd:.2f} USD...")
    print()
    
    # Process the withdrawal
    result = ton_client.process_withdrawal_to_ton(
        amount_usd=amount_usd,
        ton_address=ton_address,
        reference=f"BRAF_Withdrawal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    if result['success']:
        print("🎉 Withdrawal Successful!")
        print("=" * 30)
        print(f"💵 USD Amount: ${result['amount_usd']:.2f}")
        print(f"💎 TON Amount: {result['amount_ton']:.6f} TON")
        print(f"📈 TON Price: ${result['ton_price_usd']:.2f} USD")
        print(f"🏦 To Address: {result['to_address']}")
        print(f"🔗 Transaction Hash: {result['transaction_hash']}")
        print(f"⚡ Network Fee: {result.get('network_fee_ton', 0.01):.6f} TON")
        print(f"📊 Status: {result.get('status', 'pending')}")
        print(f"🕒 Timestamp: {result['timestamp']}")
        
        if result.get('demo_mode'):
            print()
            print("⚠️  DEMO MODE ACTIVE")
            print("   This is a simulated transaction for testing")
            print("   No real TON cryptocurrency was transferred")
            print("   To enable real transfers, configure TON API credentials")
        
        # Save transaction record
        record_file = f"ton_withdrawal_{result['withdrawal_id']}.json"
        with open(record_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Transaction record saved: {record_file}")
        
    else:
        print("❌ Withdrawal Failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
