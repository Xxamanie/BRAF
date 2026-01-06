#!/usr/bin/env python3
"""
Process TON Withdrawal
Process withdrawal to TON wallet address UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from payments.ton_integration import ton_client
from datetime import datetime
import json

def process_ton_withdrawal(amount_usd: float, ton_address: str, memo: str = ""):
    """Process TON withdrawal"""
    
    print(f"🚀 Processing TON Withdrawal")
    print(f"=" * 50)
    print(f"Amount: ${amount_usd:.2f} USD")
    print(f"TON Address: {ton_address}")
    print(f"Memo: {memo}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Validate TON address
    print("📋 Validating TON address...")
    if not ton_client.validate_ton_address(ton_address):
        print("❌ Invalid TON wallet address format")
        return False
    print("✅ TON address is valid")
    print()
    
    # Get current TON price
    print("💰 Getting current TON price...")
    price_result = ton_client.get_current_ton_price()
    if price_result['success']:
        ton_price = price_result['price_usd']
        print(f"✅ Current TON price: ${ton_price:.2f} USD")
    else:
        print("⚠️  Using fallback price")
        ton_price = 2.45
    print()
    
    # Convert USD to TON
    print("🔄 Converting USD to TON...")
    conversion_result = ton_client.convert_usd_to_ton(amount_usd)
    if conversion_result['success']:
        ton_amount = conversion_result['amount_ton']
        print(f"✅ Conversion: ${amount_usd:.2f} USD = {ton_amount:.6f} TON")
        print(f"   Exchange rate: 1 TON = ${conversion_result['conversion_rate']:.2f} USD")
    else:
        print(f"❌ Currency conversion failed: {conversion_result.get('error')}")
        return False
    print()
    
    # Process withdrawal
    print("🏦 Processing TON withdrawal...")
    withdrawal_result = ton_client.process_withdrawal_to_ton(
        amount_usd=amount_usd,
        ton_address=ton_address,
        reference=memo
    )
    
    if withdrawal_result['success']:
        print("✅ TON withdrawal processed successfully!")
        print()
        print("📄 Transaction Details:")
        print(f"   Withdrawal ID: {withdrawal_result['withdrawal_id']}")
        print(f"   Transaction Hash: {withdrawal_result['transaction_hash']}")
        print(f"   USD Amount: ${withdrawal_result['amount_usd']:.2f}")
        print(f"   TON Amount: {withdrawal_result['amount_ton']:.6f} TON")
        print(f"   TON Price: ${withdrawal_result['ton_price_usd']:.2f} USD")
        print(f"   To Address: {withdrawal_result['to_address']}")
        print(f"   Network Fee: {withdrawal_result.get('network_fee_ton', 0.01):.6f} TON")
        print(f"   Status: {withdrawal_result.get('status', 'pending')}")
        print(f"   Timestamp: {withdrawal_result['timestamp']}")
        
        if withdrawal_result.get('demo_mode'):
            print()
            print("⚠️  DEMO MODE: This is a simulated transaction")
            print("   No real TON was transferred")
            print("   To process real transactions, configure TON API credentials")
        
        # Save transaction record
        transaction_record = {
            'type': 'ton_withdrawal',
            'withdrawal_id': withdrawal_result['withdrawal_id'],
            'transaction_hash': withdrawal_result['transaction_hash'],
            'amount_usd': withdrawal_result['amount_usd'],
            'amount_ton': withdrawal_result['amount_ton'],
            'ton_price_usd': withdrawal_result['ton_price_usd'],
            'to_address': withdrawal_result['to_address'],
            'network_fee_ton': withdrawal_result.get('network_fee_ton', 0.01),
            'status': withdrawal_result.get('status', 'pending'),
            'timestamp': withdrawal_result['timestamp'],
            'demo_mode': withdrawal_result.get('demo_mode', False)
        }
        
        # Save to file
        filename = f"ton_withdrawal_{withdrawal_result['withdrawal_id']}.json"
        with open(filename, 'w') as f:
            json.dump(transaction_record, f, indent=2)
        
        print(f"💾 Transaction record saved to: {filename}")
        return True
        
    else:
        print(f"❌ TON withdrawal failed: {withdrawal_result.get('error')}")
        return False

def main():
    """Main function"""
    
    # User's TON wallet address
    ton_address = "UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7"
    
    print("💎 TON Withdrawal Processor")
    print("=" * 50)
    print()
    
    # Get withdrawal amount from user
    try:
        if len(sys.argv) > 1:
            amount_usd = float(sys.argv[1])
        else:
            amount_input = input("Enter withdrawal amount in USD (minimum $10): $")
            amount_usd = float(amount_input)
        
        if amount_usd < 10:
            print("❌ Minimum withdrawal amount is $10 USD")
            return
        
        # Get optional memo
        if len(sys.argv) > 2:
            memo = sys.argv[2]
        else:
            memo = input("Enter memo (optional): ").strip()
        
        if not memo:
            memo = f"BRAF Withdrawal - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Process withdrawal
        success = process_ton_withdrawal(amount_usd, ton_address, memo)
        
        if success:
            print()
            print("🎉 Withdrawal completed successfully!")
            print("   Check your TON wallet for the transaction")
        else:
            print()
            print("💥 Withdrawal failed!")
            
    except ValueError:
        print("❌ Invalid amount. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\n👋 Withdrawal cancelled by user")
    except Exception as e:
        print(f"💥 Error: {e}")

if __name__ == "__main__":
    main()
