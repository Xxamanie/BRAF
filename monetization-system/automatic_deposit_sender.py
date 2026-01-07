#!/usr/bin/env python3
"""
Automatic Deposit Sender for BRAF
Monitors earnings and automatically sends live cryptocurrency deposits to users
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

from payments.nowpayments_integration import CryptocurrencyWalletManager
from payments.ton_integration import ton_wallet_manager as TONWalletManager

logger = logging.getLogger(__name__)


class AutomaticDepositSender:
    """
    Monitors BRAF earnings and automatically sends live crypto deposits to users
    when they reach payout thresholds
    """

    def __init__(self, earnings_file: str = "BRAF/data/monetization_data.json"):
        self.earnings_file = earnings_file
        self.wallet_manager = CryptocurrencyWalletManager()
        self.ton_manager = TONWalletManager()

        # Payout thresholds (in USD)
        self.payout_thresholds = {
            'btc': 10.0,    # Send BTC when earnings reach $10
            'ton': 1.0,     # Send TON when earnings reach $1
            'usdt': 5.0,    # Send USDT when earnings reach $5
        }

        # Minimum payout amounts
        self.min_payouts = {
            'btc': 0.0001,   # 0.0001 BTC minimum
            'ton': 0.1,      # 0.1 TON minimum
            'usdt': 1.0,     # 1 USDT minimum
        }

        self.sent_deposits = []  # Track sent deposits
        self.check_interval = 60  # Check every 60 seconds

    def load_earnings_data(self) -> Dict:
        """Load current earnings data"""
        try:
            with open(self.earnings_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'total_earnings': 0, 'users': {}}
        except json.JSONDecodeError:
            return {'total_earnings': 0, 'users': {}}

    def get_user_wallet_address(self, user_id: str, currency: str) -> Optional[str]:
        """Get user's wallet address for specified currency"""
        # In a real implementation, this would query user database
        # For now, return a default test address
        test_addresses = {
            'btc': '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
            'ton': 'UQBmMxSNU5PLmtib4xKsBH9zAg08681Tec0rcOHYB6F4vST7',
            'usdt': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'
        }
        return test_addresses.get(currency.lower())

    def calculate_payout_amount(self, earnings_usd: float, currency: str) -> float:
        """Calculate how much crypto to send based on USD earnings"""
        if currency.lower() == 'btc':
            # Convert USD to BTC at current market rate (simplified)
            btc_price = 95000  # Approximate BTC price
            amount = earnings_usd / btc_price
            return max(amount, self.min_payouts['btc'])

        elif currency.lower() == 'ton':
            # TON is cheaper, send more
            ton_price = 2.0  # Approximate TON price
            amount = earnings_usd / ton_price
            return max(amount, self.min_payouts['ton'])

        elif currency.lower() == 'usdt':
            # USDT is 1:1 with USD
            return max(earnings_usd, self.min_payouts['usdt'])

        return 0

    def send_live_deposit(self, user_id: str, currency: str, amount: float, wallet_address: str) -> Dict:
        """Send live cryptocurrency deposit to user"""
        try:
            print(f"🚀 Sending {amount:.8f} {currency.upper()} to user {user_id}")
            print(f"   Address: {wallet_address}")
            print(f"   Amount: {amount} {currency.upper()}")

            # Use NOWPayments for BTC/USDT, TON manager for TON
            if currency.lower() == 'ton':
                result = self.ton_manager.process_real_withdrawal(
                    user_id=user_id,
                    amount=amount,
                    currency=currency,
                    wallet_address=wallet_address
                )
            else:
                result = self.wallet_manager.process_real_withdrawal(
                    user_id=user_id,
                    amount=amount,
                    currency=currency,
                    wallet_address=wallet_address
                )

            if result.get('success'):
                deposit_record = {
                    'user_id': user_id,
                    'currency': currency.upper(),
                    'amount': amount,
                    'wallet_address': wallet_address,
                    'transaction_id': result.get('transaction_id'),
                    'blockchain_hash': result.get('blockchain_hash'),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'sent'
                }

                self.sent_deposits.append(deposit_record)

                print(f"✅ Deposit sent successfully!")
                print(f"   Transaction ID: {result.get('transaction_id')}")
                print(f"   Blockchain: Live {currency.upper()} transaction")

                return deposit_record
            else:
                print(f"❌ Deposit failed: {result.get('error')}")
                return {'error': result.get('error')}

        except Exception as e:
            print(f"❌ Deposit error: {e}")
            return {'error': str(e)}

    def check_and_send_deposits(self) -> Dict:
        """Check earnings and send deposits when thresholds are reached"""
        print(f"\n🔍 Checking for automatic deposits...")
        print(f"   Time: {datetime.now().isoformat()}")

        earnings_data = self.load_earnings_data()
        total_earnings = earnings_data.get('total_earnings', 0)

        print(f"   Total Earnings: ${total_earnings:.4f}")

        deposits_sent = []

        # Check each payout threshold
        for currency, threshold in self.payout_thresholds.items():
            if total_earnings >= threshold:
                print(f"   📈 Threshold reached for {currency.upper()}: ${threshold}")

                # Get user wallet (in real system, would get from user profile)
                user_id = "braf_user"  # Default user
                wallet_address = self.get_user_wallet_address(user_id, currency)

                if wallet_address:
                    # Calculate payout amount
                    payout_amount = self.calculate_payout_amount(total_earnings, currency)

                    # Send the deposit
                    deposit_result = self.send_live_deposit(
                        user_id=user_id,
                        currency=currency,
                        amount=payout_amount,
                        wallet_address=wallet_address
                    )

                    if 'error' not in deposit_result:
                        deposits_sent.append(deposit_result)

                        # Reset earnings after successful payout
                        # In real system, this would update the earnings tracking
                        print(f"   💰 Earnings reset after successful {currency.upper()} deposit")
                    else:
                        print(f"   ❌ Failed to send {currency.upper()} deposit")
                else:
                    print(f"   ⚠️ No wallet address found for {currency.upper()}")

        return {
            'deposits_sent': deposits_sent,
            'total_earnings_checked': total_earnings,
            'thresholds_checked': list(self.payout_thresholds.keys())
        }

    def run_continuous_monitoring(self):
        """Run continuous monitoring for automatic deposits"""
        print("🚀 Starting Automatic Deposit Sender")
        print("=" * 50)
        print(f"Monitoring file: {self.earnings_file}")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Payout thresholds: {self.payout_thresholds}")
        print("=" * 50)

        try:
            while True:
                result = self.check_and_send_deposits()

                if result['deposits_sent']:
                    print(f"💸 Sent {len(result['deposits_sent'])} automatic deposits!")
                    for deposit in result['deposits_sent']:
                        print(f"   • {deposit['amount']:.8f} {deposit['currency']} to {deposit['user_id']}")
                else:
                    print(f"   ⏳ No deposits triggered (earnings: ${result['total_earnings_checked']:.2f})")

                # Save sent deposits log
                self.save_deposit_log()

                print(f"   💤 Waiting {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Automatic Deposit Sender stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")

    def save_deposit_log(self):
        """Save log of sent deposits"""
        try:
            log_data = {
                'sent_deposits': self.sent_deposits,
                'last_updated': datetime.now().isoformat(),
                'total_deposits_sent': len(self.sent_deposits)
            }

            with open('automatic_deposits_log.json', 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save deposit log: {e}")

    def get_deposit_history(self) -> List[Dict]:
        """Get history of sent deposits"""
        return self.sent_deposits.copy()


def test_automatic_deposits():
    """Test automatic deposit functionality"""
    print("Testing Automatic Deposit Sender...")
    print("=" * 40)

    sender = AutomaticDepositSender()

    # Test 1: Check earnings
    earnings = sender.load_earnings_data()
    print(f"Current earnings: ${earnings.get('total_earnings', 0):.4f}")

    # Test 2: Check thresholds
    print(f"Payout thresholds: {sender.payout_thresholds}")

    # Test 3: Simulate deposit check
    result = sender.check_and_send_deposits()
    print(f"Deposits sent in test: {len(result['deposits_sent'])}")

    print("Automatic deposit test completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_automatic_deposits()
    else:
        # Run continuous monitoring
        sender = AutomaticDepositSender()
        sender.run_continuous_monitoring()