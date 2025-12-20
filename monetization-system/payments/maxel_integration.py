#!/usr/bin/env python3
"""
MAXEL Payment Integration - Real Cryptocurrency Payment Processing
Replaces NOWPayments with MAXEL API for actual blockchain transactions
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


class MAXELIntegration:
    """
    Real cryptocurrency payment processing via MAXEL API
    Supports multiple cryptocurrencies with actual blockchain transactions
    """
    
    def __init__(self):
        # MAXEL API credentials
        self.api_key = os.getenv('MAXEL_API_KEY', 'pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsWMAXEL_SECRET_KEY')
        self.secret_key = os.getenv('MAXEL_SECRET_KEY', 'sk_rI7pJyhIyaiU5js1BCpjYA53y5iS7Ny0')
        self.base_url = os.getenv('MAXEL_BASE_URL', 'https://api.maxel.io/v1')
        self.sandbox = os.getenv('MAXEL_SANDBOX', 'false').lower() == 'true'
        
        if self.sandbox:
            self.base_url = 'https://api-sandbox.maxel.io/v1'
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'X-Secret-Key': self.secret_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        logger.info(f"MAXEL initialized - Mode: {'SANDBOX' if self.sandbox else 'LIVE'}")
        logger.info(f"API Key: {self.api_key[:20]}...")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make authenticated request to MAXEL API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=data)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MAXEL API request failed: {e}")
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
        elif 'data' in response:
            return [currency['code'] for currency in response['data']]
        
        # Default supported currencies for MAXEL
        return ['BTC', 'ETH', 'USDT', 'USDC', 'LTC', 'BCH', 'XRP', 'ADA', 'DOT', 'LINK', 'BNB', 'MATIC']
    
    def get_exchange_rates(self, base_currency: str = 'USD') -> Dict[str, float]:
        """Get real-time cryptocurrency exchange rates"""
        response = self._make_request('GET', f'/rates/{base_currency}')
        
        if 'rates' in response:
            return response['rates']
        elif 'data' in response:
            return response['data']
        return {}
    
    def create_payment_address(self, currency: str, user_id: str = None) -> Dict:
        """
        Create a new payment address for receiving cryptocurrency
        """
        data = {
            'currency': currency.upper(),
            'user_id': user_id or f'braf_user_{int(datetime.now().timestamp())}',
            'callback_url': os.getenv('MAXEL_CALLBACK_URL', 'https://api.braf.io/webhooks/maxel')
        }
        
        response = self._make_request('POST', '/addresses', data)
        
        if 'address' in response:
            logger.info(f"Payment address created: {response['address']}")
        
        return response
    
    def get_address_balance(self, address: str) -> Dict:
        """Get balance for a specific address"""
        response = self._make_request('GET', f'/addresses/{address}/balance')
        return response
    
    def create_withdrawal(self, amount: float, currency: str, 
                         destination_address: str, user_id: str = None) -> Dict:
        """
        Create a withdrawal transaction
        This sends actual crypto to the destination address
        """
        data = {
            'amount': str(amount),
            'currency': currency.upper(),
            'destination_address': destination_address,
            'user_id': user_id or f'braf_withdrawal_{int(datetime.now().timestamp())}',
            'callback_url': os.getenv('MAXEL_CALLBACK_URL', 'https://api.braf.io/webhooks/maxel')
        }
        
        response = self._make_request('POST', '/withdrawals', data)
        
        if 'transaction_id' in response:
            logger.info(f"Withdrawal created: {response['transaction_id']}")
        
        return response
    
    def get_withdrawal_status(self, withdrawal_id: str) -> Dict:
        """Get withdrawal status by ID"""
        response = self._make_request('GET', f'/withdrawals/{withdrawal_id}')
        return response
    
    def get_transaction_history(self, limit: int = 50, offset: int = 0) -> Dict:
        """Get transaction history"""
        params = {
            'limit': limit,
            'offset': offset
        }
        
        response = self._make_request('GET', '/transactions', params)
        return response
    
    def validate_address(self, address: str, currency: str) -> Dict:
        """Validate cryptocurrency address"""
        data = {
            'address': address,
            'currency': currency.upper()
        }
        
        response = self._make_request('POST', '/validate-address', data)
        return response
    
    def get_minimum_withdrawal(self, currency: str) -> Dict:
        """Get minimum withdrawal amount for currency"""
        response = self._make_request('GET', f'/currencies/{currency.upper()}/limits')
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
            if currency.upper() not in [c.upper() for c in available_currencies]:
                return {
                    'success': False,
                    'error': f'Currency {currency} not supported'
                }
            
            # Validate address
            validation = self.validate_address(wallet_address, currency)
            if not validation.get('valid', False):
                return {
                    'success': False,
                    'error': f'Invalid {currency} address'
                }
            
            # Check minimum amount
            limits = self.get_minimum_withdrawal(currency)
            min_amount = limits.get('min_withdrawal', 0.001)
            if amount < min_amount:
                return {
                    'success': False,
                    'error': f'Amount below minimum: {min_amount} {currency}'
                }
            
            # Create withdrawal
            withdrawal_response = self.create_withdrawal(
                amount=amount,
                currency=currency,
                destination_address=wallet_address,
                user_id=user_id
            )
            
            if 'transaction_id' in withdrawal_response:
                logger.info(f"Real withdrawal created: {withdrawal_response['transaction_id']} for user {user_id}")
                return {
                    'success': True,
                    'transaction_id': withdrawal_response['transaction_id'],
                    'status': withdrawal_response.get('status', 'pending'),
                    'amount': amount,
                    'currency': currency,
                    'address': wallet_address,
                    'blockchain_hash': withdrawal_response.get('blockchain_hash'),
                    'created_at': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': withdrawal_response.get('error', 'Withdrawal failed')
                }
                
        except Exception as e:
            logger.error(f"Withdrawal processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_account_balance(self) -> Dict:
        """Get account balance for all currencies"""
        response = self._make_request('GET', '/account/balance')
        return response
    
    def create_invoice(self, amount: float, currency: str, 
                      order_id: str, description: str = None) -> Dict:
        """Create payment invoice for receiving crypto payments"""
        data = {
            'amount': str(amount),
            'currency': currency.upper(),
            'order_id': order_id,
            'description': description or f'BRAF Payment {order_id}',
            'callback_url': os.getenv('MAXEL_CALLBACK_URL', 'https://api.braf.io/webhooks/maxel')
        }
        
        response = self._make_request('POST', '/invoices', data)
        return response


class MAXELWalletManager:
    """
    Manages cryptocurrency wallets and addresses for the BRAF system
    Integrates with MAXEL for actual blockchain operations
    """
    
    def __init__(self):
        self.maxel = MAXELIntegration()
        self.supported_currencies = [
            'BTC', 'ETH', 'USDT', 'USDC', 'LTC', 'BCH', 'XRP', 'ADA', 
            'DOT', 'LINK', 'BNB', 'MATIC', 'AVAX', 'SOL', 'TRX'
        ]
    
    def get_deposit_address(self, user_id: str, currency: str) -> Dict:
        """
        Generate deposit address for user to receive cryptocurrency
        This creates a real blockchain address
        """
        try:
            address_response = self.maxel.create_payment_address(
                currency=currency.upper(),
                user_id=user_id
            )
            
            if 'address' in address_response:
                return {
                    'success': True,
                    'address': address_response['address'],
                    'currency': currency.upper(),
                    'network': address_response.get('network', 'mainnet'),
                    'memo': address_response.get('memo'),
                    'user_id': user_id,
                    'created_at': datetime.now().isoformat()
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
        return self.maxel.process_withdrawal(
            user_id=user_id,
            amount=amount,
            currency=currency,
            wallet_address=wallet_address,
            memo=memo
        )
    
    def get_transaction_status(self, transaction_id: str) -> Dict:
        """Get real transaction status from blockchain"""
        try:
            status_response = self.maxel.get_withdrawal_status(transaction_id)
            
            if 'status' in status_response:
                return {
                    'transaction_id': transaction_id,
                    'status': status_response['status'],
                    'blockchain_hash': status_response.get('blockchain_hash'),
                    'confirmations': status_response.get('confirmations', 0),
                    'amount': status_response.get('amount'),
                    'currency': status_response.get('currency'),
                    'created_at': status_response.get('created_at'),
                    'updated_at': status_response.get('updated_at')
                }
            
            return {'error': 'Transaction not found'}
            
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {'error': str(e)}
    
    def get_wallet_balance(self) -> Dict:
        """Get real wallet balances from MAXEL"""
        try:
            balance_response = self.maxel.get_account_balance()
            
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


def test_maxel_integration():
    """Test MAXEL integration with real API"""
    print("Testing MAXEL Integration...")
    
    # Initialize
    maxel = MAXELIntegration()
    
    # Test 1: API Status
    print("\n1. Testing API Status...")
    status = maxel.get_api_status()
    print(f"API Status: {status}")
    
    # Test 2: Available Currencies
    print("\n2. Getting Available Currencies...")
    currencies = maxel.get_available_currencies()
    print(f"Available Currencies: {len(currencies)} total")
    print(f"Sample: {currencies[:10]}")
    
    # Test 3: Exchange Rates
    print("\n3. Getting Exchange Rates...")
    rates = maxel.get_exchange_rates('USD')
    print(f"Exchange Rates: {len(rates)} currencies")
    if rates:
        print(f"BTC/USD: {rates.get('BTC', 'N/A')}")
        print(f"ETH/USD: {rates.get('ETH', 'N/A')}")
    
    # Test 4: Wallet Manager
    print("\n4. Testing Wallet Manager...")
    wallet_manager = MAXELWalletManager()
    
    # Get balance
    balance = wallet_manager.get_wallet_balance()
    print(f"Wallet Balance: {balance}")
    
    print("\nMAXEL integration test completed!")


if __name__ == "__main__":
    test_maxel_integration()