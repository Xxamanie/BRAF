#!/usr/bin/env python3
"""
NOWPayments Integration - Real Cryptocurrency Payment Processing
Handles actual blockchain transactions for BTC, ETH, XMR, and 150+ cryptocurrencies
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


class NOWPaymentsIntegration:
    """
    Real cryptocurrency payment processing via NOWPayments API
    Supports 150+ cryptocurrencies with actual blockchain transactions
    """
    
    def __init__(self):
        self.api_key = os.getenv('NOWPAYMENTS_API_KEY', 'RD7WEXF-QTW4N7P-HMV12F9-MPANF4G')
        self.base_url = os.getenv('NOWPAYMENTS_BASE_URL', 'https://api.nowpayments.io/v1')
        self.sandbox = os.getenv('NOWPAYMENTS_SANDBOX', 'false').lower() == 'true'
        
        if self.sandbox:
            self.base_url = 'https://api-sandbox.nowpayments.io/v1'
        
        self.headers = {
            'x-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        logger.info(f"NOWPayments initialized - Mode: {'SANDBOX' if self.sandbox else 'LIVE'}")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to NOWPayments API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=data)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NOWPayments API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            return {'success': False, 'error': str(e)}
    
    def get_api_status(self) -> Dict:
        """Check API status and connectivity"""
        response = self._make_request('GET', '/status')
        return response
    
    def get_available_currencies(self) -> List[str]:
        """Get list of all available cryptocurrencies"""
        response = self._make_request('GET', '/currencies')
        
        if 'currencies' in response:
            return response['currencies']
        return []
    
    def get_available_full_currencies(self) -> List[Dict]:
        """Get detailed information about available currencies"""
        response = self._make_request('GET', '/full-currencies')
        
        if 'currencies' in response:
            return response['currencies']
        return []
    
    def get_minimum_payment_amount(self, currency_from: str, currency_to: str) -> Dict:
        """Get minimum payment amount for currency pair"""
        response = self._make_request('GET', f'/min-amount', {
            'currency_from': currency_from,
            'currency_to': currency_to
        })
        return response
    
    def get_estimated_price(self, amount: float, currency_from: str, currency_to: str) -> Dict:
        """Get estimated exchange price"""
        response = self._make_request('GET', '/estimate', {
            'amount': amount,
            'currency_from': currency_from,
            'currency_to': currency_to
        })
        return response
    
    def create_payment(self, price_amount: float, price_currency: str, 
                      pay_currency: str, order_id: str = None,
                      order_description: str = None, ipn_callback_url: str = None,
                      success_url: str = None, cancel_url: str = None) -> Dict:
        """
        Create a new payment
        This generates a payment address for the user to send crypto to
        """
        data = {
            'price_amount': price_amount,
            'price_currency': price_currency,
            'pay_currency': pay_currency
        }
        
        if order_id:
            data['order_id'] = order_id
        if order_description:
            data['order_description'] = order_description
        if ipn_callback_url:
            data['ipn_callback_url'] = ipn_callback_url
        if success_url:
            data['success_url'] = success_url
        if cancel_url:
            data['cancel_url'] = cancel_url
        
        response = self._make_request('POST', '/payment', data)
        
        if 'payment_id' in response:
    
            logger.info(f"Payment created: {response['payment_id']}")
        
        return response
    
    def get_payment_status(self, payment_id: str) -> Dict:
        """Get payment status by payment ID"""
        response = self._make_request('GET', f'/payment/{payment_id}')
        return response
    
    def get_payment_list(self, limit: int = 10, page: int = 0, 
                        sortBy: str = 'created_at', orderBy: str = 'desc',
                        dateFrom: str = None, dateTo: str = None) -> Dict:
        """Get list of payments"""
        params = {
            'limit': limit,
            'page': page,
            'sortBy': sortBy,
            'orderBy': orderBy
        }
        
        if dateFrom:
            params['dateFrom'] = dateFrom
        if dateTo:
            params['dateTo'] = dateTo
        
        response = self._make_request('GET', '/payment', params)
        return response
    
    def create_payout(self, withdrawals: List[Dict]) -> Dict:
        """
        Create payout (mass withdrawal)
        withdrawals: List of {'address': 'wallet_address', 'currency': 'btc', 'amount': 0.001}
        """
        data = {'withdrawals': withdrawals}
        response = self._make_request('POST', '/payout', data)
        return response
    
    def get_payout_status(self, payout_id: str) -> Dict:
        """Get payout status"""
        response = self._make_request('GET', f'/payout/{payout_id}')
        return response
    
    def get_balance(self) -> Dict:
        """Get account balance for all currencies"""
        response = self._make_request('GET', '/balance')
        return response
    
    def process_withdrawal(self, user_id: str, amount: float, currency: str, 
                         wallet_address: str, memo: str = None) -> Dict:
        """
        Process real cryptocurrency withdrawal
        This sends actual crypto to the user's wallet
        """
        try:
            # Validate currency
            available_currencies = self.get_available_currencies()
            if currency.lower() not in [c.lower() for c in available_currencies]:
                return {
                    'success': False,
                    'error': f'Currency {currency} not supported'
                }
            
            # Check minimum amount
            min_amount_response = self.get_minimum_payment_amount('usd', currency.lower())
            if 'min_amount' in min_amount_response:
                min_amount = float(min_amount_response['min_amount'])
                if amount < min_amount:
                    return {
                        'success': False,
                        'error': f'Amount below minimum: {min_amount} {currency}'
                    }
            
            # Create withdrawal
            withdrawal_data = {
                'address': wallet_address,
                'currency': currency.lower(),
                'amount': amount
            }
            
            if memo:
                withdrawal_data['extra_id'] = memo
            
            payout_response = self.create_payout([withdrawal_data])
            
            if 'id' in payout_response:
                logger.info(f"Real withdrawal created: {payout_response['id']} for user {user_id}")
                return {
                    'success': True,
                    'payout_id': payout_response['id'],
                    'status': payout_response.get('status', 'pending'),
                    'amount': amount,
                    'currency': currency,
                    'address': wallet_address,
                    'transaction_id': payout_response.get('id'),
                    'blockchain_hash': None,  # Will be available later
                    'created_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': payout_response.get('message', 'Withdrawal failed')
                }
                
        except Exception as e:
            logger.error(f"Withdrawal processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_supported_currencies_with_details(self) -> Dict[str, Dict]:
        """Get detailed information about supported currencies"""
        currencies = self.get_available_full_currencies()
        
        currency_map = {}
        for currency in currencies:
            currency_map[currency['code']] = {
                'name': currency['name'],
                'network': currency.get('network', 'mainnet'),
                'is_popular': currency.get('is_popular', False),
                'has_extra_id': currency.get('has_extra_id', False),
                'extra_id_name': currency.get('extra_id_name', ''),
                'min_amount': currency.get('min_amount', 0),
                'max_amount': currency.get('max_amount', 0)
            }
        
        return currency_map
    
    def validate_wallet_address(self, currency: str, address: str) -> Dict:
        """Validate cryptocurrency wallet address"""
        # Basic validation - in production, use more sophisticated validation
        if not address or len(address) < 10:
            return {'valid': False, 'error': 'Invalid address format'}
        
        # Currency-specific validation
        if currency.lower() == 'btc':
            if not (address.startswith('1') or address.startswith('3') or address.startswith('bc1')):
                return {'valid': False, 'error': 'Invalid Bitcoin address format'}
        elif currency.lower() == 'eth':
            if not (address.startswith('0x') and len(address) == 42):
                return {'valid': False, 'error': 'Invalid Ethereum address format'}
        elif currency.lower() == 'xmr':
            if len(address) != 95:
                return {'valid': False, 'error': 'Invalid Monero address format'}
        
        return {'valid': True}
    
    def get_real_time_rates(self, base_currency: str = 'usd') -> Dict[str, float]:
        """Get real-time cryptocurrency exchange rates"""
        currencies = self.get_available_currencies()
        rates = {}
        
        for currency in currencies[:20]:  # Limit to prevent API overload
            try:
                estimate = self.get_estimated_price(1, base_currency, currency)
                if 'estimated_amount' in estimate:
                    rates[currency] = float(estimate['estimated_amount'])
            except:
                continue
        
        return rates
    
    def create_invoice(self, price_amount: float, price_currency: str,
                      order_id: str, order_description: str = None) -> Dict:
        """Create payment invoice for receiving crypto payments"""
        data = {
            'price_amount': price_amount,
            'price_currency': price_currency,
            'order_id': order_id,
            'order_description': order_description or f'BRAF Payment {order_id}'
        }
        
        response = self._make_request('POST', '/invoice', data)
        return response


class CryptocurrencyWalletManager:
    """
    Manages cryptocurrency wallets and addresses for the BRAF system
    Integrates with NOWPayments for actual blockchain operations
    """
    
    def __init__(self):
        self.nowpayments = NOWPaymentsIntegration()
        self.supported_currencies = [
            'btc', 'eth', 'ltc', 'bch', 'xmr', 'dash', 'zec', 'doge',
            'usdt', 'usdc', 'dai', 'busd', 'bnb', 'ada', 'dot', 'link',
            'uni', 'aave', 'comp', 'mkr', 'snx', 'yfi', 'sushi', 'crv',
            'ton', 'trx', 'sol', 'avax', 'matic', 'ftm', 'atom', 'luna'
        ]
    
    def get_deposit_address(self, user_id: str, currency: str) -> Dict:
        """
        Generate deposit address for user to receive cryptocurrency
        This creates a real blockchain address
        """
        try:
            # Create payment to get deposit address
            payment = self.nowpayments.create_payment(
                price_amount=1,  # Minimum amount
                price_currency='usd',
                pay_currency=currency.lower(),
                order_id=f'deposit_{user_id}_{currency}_{int(datetime.now().timestamp())}'
            )
            
            if 'pay_address' in payment:
                return {
                    'success': True,
                    'address': payment['pay_address'],
                    'currency': currency.upper(),
                    'network': payment.get('network', 'mainnet'),
                    'memo': payment.get('payin_extra_id'),
                    'payment_id': payment['payment_id'],
                    'expires_at': payment.get('created_at')  # Add expiration logic
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate deposit address'
                }
                
        except Exception as e:
            logger.error(f"Failed to generate deposit address: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_real_withdrawal(self, user_id: str, amount: float, 
                              currency: str, wallet_address: str, memo: str = None) -> Dict:
        """Process real cryptocurrency withdrawal to user's wallet"""
        
        # Validate address
        validation = self.nowpayments.validate_wallet_address(currency, wallet_address)
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error']
            }
        
        # Process withdrawal through NOWPayments
        result = self.nowpayments.process_withdrawal(
            user_id=user_id,
            amount=amount,
            currency=currency,
            wallet_address=wallet_address,
            memo=memo
        )
        
        return result
    
    def get_transaction_status(self, transaction_id: str) -> Dict:
        """Get real transaction status from blockchain"""
        try:
            # Check if it's a payment or payout
            payment_status = self.nowpayments.get_payment_status(transaction_id)
            if 'payment_status' in payment_status:
                return {
                    'transaction_id': transaction_id,
                    'status': payment_status['payment_status'],
                    'blockchain_hash': payment_status.get('outcome_hash'),
                    'confirmations': payment_status.get('outcome_confirmations', 0),
                    'amount': payment_status.get('pay_amount'),
                    'currency': payment_status.get('pay_currency'),
                    'created_at': payment_status.get('created_at'),
                    'updated_at': payment_status.get('updated_at')
                }
            
            # Try as payout
            payout_status = self.nowpayments.get_payout_status(transaction_id)
            if 'status' in payout_status:
                return {
                    'transaction_id': transaction_id,
                    'status': payout_status['status'],
                    'blockchain_hash': payout_status.get('hash'),
                    'amount': payout_status.get('amount'),
                    'currency': payout_status.get('currency'),
                    'created_at': payout_status.get('created_at')
                }
            
            return {'error': 'Transaction not found'}
            
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {'error': str(e)}
    
    def get_wallet_balance(self) -> Dict:
        """Get real wallet balances from NOWPayments"""
        try:
            balance_response = self.nowpayments.get_balance()
            
            if 'balances' in balance_response:
                balances = {}
                for currency, amount in balance_response['balances'].items():
                    if float(amount) > 0:
                        balances[currency.upper()] = float(amount)
                
                return {
                    'success': True,
                    'balances': balances,
                    'total_currencies': len(balances),
                    'updated_at': datetime.now().isoformat()
                }
            
            return {'success': False, 'error': 'No balance data available'}
            
        except Exception as e:
            logger.error(f"Failed to get wallet balance: {e}")
            return {'success': False, 'error': str(e)}


def test_nowpayments_integration():
    """Test NOWPayments integration with real API"""
    print("Testing NOWPayments Integration...")
    
    # Initialize
    nowpayments = NOWPaymentsIntegration()
    
    # Test 1: API Status
    print("\n1. Testing API Status...")
    status = nowpayments.get_api_status()
    print(f"API Status: {status}")
    
    # Test 2: Available Currencies
    print("\n2. Getting Available Currencies...")
    currencies = nowpayments.get_available_currencies()
    print(f"Available Currencies: {len(currencies)} total")
    print(f"Sample: {currencies[:10]}")
    
    # Test 3: Currency Details
    print("\n3. Getting Currency Details...")
    currency_details = nowpayments.get_supported_currencies_with_details()
    print(f"Detailed Currencies: {len(currency_details)}")
    
    # Test 4: Minimum Amounts
    print("\n4. Testing Minimum Amounts...")
    min_btc = nowpayments.get_minimum_payment_amount('usd', 'btc')
    print(f"Minimum BTC: {min_btc}")
    
    # Test 5: Price Estimation
    print("\n5. Testing Price Estimation...")
    estimate = nowpayments.get_estimated_price(100, 'usd', 'btc')
    print(f"100 USD = {estimate.get('estimated_amount', 'N/A')} BTC")
    
    # Test 6: Wallet Manager
    print("\n6. Testing Wallet Manager...")
    wallet_manager = CryptocurrencyWalletManager()
    
    # Get balance
    balance = wallet_manager.get_wallet_balance()
    print(f"Wallet Balance: {balance}")
    
    print("\nNOWPayments integration test completed!")


if __name__ == "__main__":
    test_nowpayments_integration()