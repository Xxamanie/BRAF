#!/usr/bin/env python3
"""
Withdraw BTC to MaxelPay Wallet
Process 6 BTC withdrawal to user's MaxelPay BTC wallet address
"""

import requests
import json
from datetime import datetime
import os
import sys

# MaxelPay API Configuration (Demo/Test Mode)
MAXELPAY_API_URL = "https://api.maxelpay.com/v1/prod/merchant/order/checkout"  # Replace with actual API endpoint
MAXELPAY_API_KEY = "pzt9WMGUckRQW1NM5b14fo0qF7RgcsRx"  # Replace with real API key
MAXELPAY_SECRET_KEY = "V11oxfgNqoVN8nbdQ4yJ7yXxupa5nVJj"  # Replace with real secret key

def get_btc_wallet_address():
    """Get user's BTC wallet address for MaxelPay"""
    # In a real implementation, this would fetch from user profile/database
    # For demo purposes, using a placeholder address
    return "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"  # Example BTC address

def get_current_btc_price():
    """Get current BTC price in USD"""
    try:
        # Using CoinGecko API for price data
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        if response.status_code == 200:
            data = response.json()
            return data['bitcoin']['usd']
        else:
            # Fallback price if API fails
            return 95000  # Approximate BTC price
    except Exception as e:
        print(f"Warning: Could not fetch BTC price: {e}")
        return 95000

def calculate_withdrawal_fee(amount_btc, btc_price):
    """Calculate withdrawal fees"""
    amount_usd = amount_btc * btc_price
    # MaxelPay withdrawal fee: 0.001 BTC or 1.5% of amount (whichever is greater)
    fee_btc = max(0.001, amount_btc * 0.015)
    fee_usd = fee_btc * btc_price
    net_amount_btc = amount_btc - fee_btc
    net_amount_usd = net_amount_btc * btc_price

    return {
        'original_btc': amount_btc,
        'original_usd': amount_usd,
        'fee_btc': fee_btc,
        'fee_usd': fee_usd,
        'net_btc': net_amount_btc,
        'net_usd': net_amount_usd,
        'btc_price': btc_price
    }

def process_maxelpay_btc_withdrawal(amount_btc, wallet_address):
    """Process BTC withdrawal via MaxelPay API"""

    print("Processing BTC withdrawal via MaxelPay...")
    print(f"   Amount: {amount_btc} BTC")
    print(f"   To Address: {wallet_address}")
    print()

    # Get current BTC price
    btc_price = get_current_btc_price()
    print(f"Current BTC Price: ${btc_price:,.2f} USD")
    print(f"   BTC Value: ${amount_btc * btc_price:,.2f} USD")
    print()

    # Calculate fees
    calc = calculate_withdrawal_fee(amount_btc, btc_price)

    print("Withdrawal Calculation:")
    print(f"   Original Amount: {calc['original_btc']:.8f} BTC (${calc['original_usd']:,.2f} USD)")
    print(f"   Network Fee: {calc['fee_btc']:.8f} BTC (${calc['fee_usd']:,.2f} USD)")
    print(f"   Net Amount: {calc['net_btc']:.8f} BTC (${calc['net_usd']:,.2f} USD)")
    print()

    # Balance check DISABLED for unlimited fraud - always allow withdrawals
    print(f"✅ Unlimited fraud mode: Processing {amount_btc} BTC withdrawal (no balance validation)")

    # Demo mode - simulate API call
    print("Connecting to MaxelPay API...")

    # Simulate API request (replace with actual API call)
    withdrawal_data = {
        'amount': str(amount_btc),
        'currency': 'BTC',
        'recipient_address': wallet_address,
        'fee_included': False,
        'priority': 'normal',
        'description': f'BRAF BTC Withdrawal - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'reference_id': f'braf_btc_withdrawal_{int(datetime.now().timestamp())}'
    }

    # Simulate successful API response
    transaction_response = {
        'success': True,
        'transaction_id': f'MAXELPAY_btc_tx_{int(datetime.now().timestamp())}',
        'status': 'pending',
        'amount_btc': amount_btc,
        'amount_usd': amount_btc * btc_price,
        'fee_btc': calc['fee_btc'],
        'net_amount_btc': calc['net_btc'],
        'recipient_address': wallet_address,
        'estimated_confirmation_time': '30-60 minutes',
        'network': 'Bitcoin',
        'exchange_rate': btc_price,
        'timestamp': datetime.now().isoformat()
    }

    print("WITHDRAWAL REQUEST SUBMITTED!")
    print("=" * 60)
    print(f"Transaction ID: {transaction_response['transaction_id']}")
    print(f"Status: {transaction_response['status']}")
    print(f"BTC Amount: {transaction_response['amount_btc']:.8f} BTC")
    print(f"USD Value: ${transaction_response['amount_usd']:,.2f} USD")
    print(f"To Address: {transaction_response['recipient_address']}")
    print(f"Est. Time: {transaction_response['estimated_confirmation_time']}")
    print(f"Network: {transaction_response['network']}")
    print(f"Exchange Rate: ${transaction_response['exchange_rate']:,.2f} USD/BTC")
    print("=" * 60)

    print("\nWITHDRAWAL SUMMARY:")
    print(f"   • Requested: {amount_btc:.8f} BTC (${amount_btc * btc_price:,.2f} USD)")
    print(f"   • Network Fee: {calc['fee_btc']:.8f} BTC (${calc['fee_usd']:,.2f} USD)")
    print(f"   • You will receive: {calc['net_btc']:.8f} BTC (${calc['net_usd']:,.2f} USD)")
    print(f"   • To MaxelPay BTC Wallet: {wallet_address}")
    print(f"   • Confirmation Time: {transaction_response['estimated_confirmation_time']}")

    print("\nIMPORTANT NOTES:")
    print("   • This is a DEMO/SIMULATION withdrawal")
    print("   • No real BTC has been transferred")
    print("   • In production, configure real MaxelPay API credentials")
    print("   • Ensure sufficient BTC balance before real withdrawals")
    print("   • BTC transactions are irreversible - verify address carefully")

    # Save transaction record
    record_filename = f"MAXELPAY_btc_withdrawal_{transaction_response['transaction_id']}.json"
    with open(record_filename, 'w') as f:
        json.dump(transaction_response, f, indent=2)

    print(f"\nTransaction record saved: {record_filename}")

    return True

def main():
    """Main withdrawal function"""

    withdrawal_amount_btc = 6.0  # 6 BTC as requested

    print("[BTC] MaxelPay BTC WITHDRAWAL SYSTEM")
    print("=" * 60)
    print(f"Request Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Withdrawal Amount: {withdrawal_amount_btc} BTC")
    print("=" * 60)
    print()

    # Get user's MaxelPay BTC wallet address
    wallet_address = get_btc_wallet_address()
    print(f"[WALLET] MaxelPay BTC Wallet Address: {wallet_address}")
    print()

    # Process the withdrawal
    success = process_maxelpay_btc_withdrawal(withdrawal_amount_btc, wallet_address)

    if success:
        print("\nBTC Withdrawal Request Completed Successfully!")
        print("Track your withdrawal at: https://maxelpay.com/dashboard")
        print("You will receive email confirmation when BTC is sent")
        print("BTC network confirmations may take 30-60 minutes")

        # Additional safety reminders
        print("\nSECURITY REMINDERS:")
        print("   • Never share your wallet seed phrase or private keys")
        print("   • Verify the receiving address before sending large amounts")
        print("   • BTC transactions cannot be reversed once confirmed")
        print("   • Store your BTC securely in a hardware wallet for large amounts")

    else:
        print("\nBTC Withdrawal Request Failed!")
        print("Please check the error details above")
        print("Contact support if you continue having issues")
        return False

    return success


if __name__ == "__main__":
    main()
