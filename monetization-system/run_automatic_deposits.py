#!/usr/bin/env python3
"""
Run Automatic Deposit Sender
Monitors BRAF earnings and automatically sends live cryptocurrency deposits
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from automatic_deposit_sender import AutomaticDepositSender


def main():
    """Main function to run automatic deposit sender"""
    print("🚀 BRAF Automatic Deposit Sender")
    print("=" * 50)
    print("This will monitor earnings and automatically send live crypto deposits")
    print("when payout thresholds are reached.")
    print()
    print("Supported currencies and thresholds:")
    print("  • TON: $1.00 USD")
    print("  • USDT: $5.00 USD")
    print("  • BTC: $10.00 USD")
    print()
    print("⚠️  WARNING: This sends REAL cryptocurrency!")
    print("   Ensure you have sufficient balances in your payment accounts.")
    print("=" * 50)

    # Confirm with user
    try:
        confirm = input("Start automatic deposit monitoring? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("❌ Automatic deposits cancelled by user")
            return
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        return

    print("\n✅ Starting automatic deposit monitoring...")

    # Initialize and run
    sender = AutomaticDepositSender()
    sender.run_continuous_monitoring()


if __name__ == "__main__":
    main()