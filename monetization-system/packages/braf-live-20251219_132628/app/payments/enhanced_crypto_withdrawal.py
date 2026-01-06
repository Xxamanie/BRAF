"""
Enhanced Cryptocurrency Withdrawal System
Supports multiple cryptocurrencies with different networks
"""

import hashlib
import hmac
import time
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

class EnhancedCryptoWithdrawal:
    """Enhanced cryptocurrency withdrawal processor"""
    
    def __init__(self):
        self.supported_cryptos = {
            'btc': {
                'name': 'Bitcoin',
                'symbol': 'BTC',
                'networks': ['Bitcoin'],
                'min_amount': 0.001,
                'max_amount': 10.0,
                'fee': 0.0005,
                'confirmations': 3
            },
            'eth': {
                'name': 'Ethereum',
                'symbol': 'ETH',
                'networks': ['Ethereum'],
                'min_amount': 0.01,
                'max_amount': 100.0,
                'fee': 0.005,
                'confirmations': 12
            },
            'usdt': {
                'name': 'Tether',
                'symbol': 'USDT',
                'networks': ['ERC20', 'TRC20', 'BEP20'],
                'min_amount': 10.0,
                'max_amount': 50000.0,
                'fee': {'ERC20': 5.0, 'TRC20': 1.0, 'BEP20': 1.0},
                'confirmations': {'ERC20': 12, 'TRC20': 20, 'BEP20': 15}
            },
            'usdc': {
                'name': 'USD Coin',
                'symbol': 'USDC',
                'networks': ['ERC20', 'BEP20', 'Polygon'],
                'min_amount': 10.0,
                'max_amount': 50000.0,
                'fee': {'ERC20': 5.0, 'BEP20': 1.0, 'Polygon': 0.1},
                'confirmations': {'ERC20': 12, 'BEP20': 15, 'Polygon': 30}
            },
            'bnb': {
                'name': 'Binance Coin',
                'symbol': 'BNB',
                'networks': ['BEP20'],
                'min_amount': 0.1,
                'max_amount': 1000.0,
                'fee': 0.005,
                'confirmations': 15
            },
            'ada': {
                'name': 'Cardano',
                'symbol': 'ADA',
                'networks': ['Cardano'],
                'min_amount': 10.0,
                'max_amount': 10000.0,
                'fee': 1.0,
                'confirmations': 20
            },
            'xmr': {
                'name': 'Monero',
                'symbol': 'XMR',
                'networks': ['Monero'],
                'min_amount': 0.1,
                'max_amount': 1000.0,
                'fee': 0.01,
                'confirmations': 10
            },
            'zcash': {
                'name': 'Zcash',
                'symbol': 'ZEC',
                'networks': ['Zcash'],
                'min_amount': 0.1,
                'max_amount': 1000.0,
                'fee': 0.001,
                'confirmations': 6
            },
            'dash': {
                'name': 'Dash',
                'symbol': 'DASH',
                'networks': ['Dash'],
                'min_amount': 0.1,
                'max_amount': 1000.0,
                'fee': 0.001,
                'confirmations': 6
            },
            'ton': {
                'name': 'TON Coin',
                'symbol': 'TON',
                'networks': ['TON'],
                'min_amount': 1.0,
                'max_amount': 10000.0,
                'fee': 0.01,
                'confirmations': 1
            },
            'trx': {
                'name': 'Tron',
                'symbol': 'TRX',
                'networks': ['Tron'],
                'min_amount': 100.0,
                'max_amount': 100000.0,
                'fee': 1.0,
                'confirmations': 20
            },
            'ltc': {
                'name': 'Litecoin',
                'symbol': 'LTC',
                'networks': ['Litecoin'],
                'min_amount': 0.1,
                'max_amount': 1000.0,
                'fee': 0.001,
                'confirmations': 6
            },
            'sol': {
                'name': 'Solana',
                'symbol': 'SOL',
                'networks': ['Solana'],
                'min_amount': 0.1,
                'max_amount': 1000.0,
                'fee': 0.00025,
                'confirmations': 32
            }
        }
        
        # Simulated exchange rates (in production, fetch from API)
        self.exchange_rates = {
            'BTC': 42000.0,
            'ETH': 2500.0,
            'USDT': 1.0,
            'USDC': 1.0,
            'BNB': 300.0,
            'ADA': 0.45,
            'XMR': 150.0,
            'ZEC': 35.0,
            'DASH': 45.0,
            'TON': 2.45,
            'TRX': 0.08,
            'LTC': 75.0,
            'SOL': 65.0
        }
    
    def validate_address(self, crypto: str, address: str, network: str = None) -> Dict[str, Any]:
        """Validate cryptocurrency address format"""
        try:
            if crypto not in self.supported_cryptos:
                return {'valid': False, 'error': 'Unsupported cryptocurrency'}
            
            # Address validation patterns
            patterns = {
                'btc': r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$',
                'eth': r'^0x[a-fA-F0-9]{40}$',
                'usdt': {
                    'ERC20': r'^0x[a-fA-F0-9]{40}$',
                    'TRC20': r'^T[A-Za-z1-9]{33}$',
                    'BEP20': r'^0x[a-fA-F0-9]{40}$'
                },
                'usdc': {
                    'ERC20': r'^0x[a-fA-F0-9]{40}$',
                    'BEP20': r'^0x[a-fA-F0-9]{40}$',
                    'Polygon': r'^0x[a-fA-F0-9]{40}$'
                },
                'bnb': r'^0x[a-fA-F0-9]{40}$',
                'ada': r'^addr1[a-z0-9]{98}$',
                'xmr': r'^4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}$',
                'zcash': r'^t1[a-zA-Z0-9]{33}$|^zs1[a-z0-9]{75}$',
                'dash': r'^X[1-9A-HJ-NP-Za-km-z]{33}$',
                'ton': r'^[UE]Q[A-Za-z0-9_-]{46}$',
                'trx': r'^T[A-Za-z1-9]{33}$',
                'ltc': r'^[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}$',
                'sol': r'^[1-9A-HJ-NP-Za-km-z]{32,44}$'
            }
            
            import re
            pattern = patterns.get(crypto)
            
            if isinstance(pattern, dict) and network:
                pattern = pattern.get(network)
            
            if not pattern:
                return {'valid': False, 'error': 'No validation pattern available'}
            
            is_valid = bool(re.match(pattern, address))
            
            return {
                'valid': is_valid,
                'crypto': crypto.upper(),
                'network': network,
                'address_type': self._get_address_type(crypto, address, network)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _get_address_type(self, crypto: str, address: str, network: str = None) -> str:
        """Determine address type for better user feedback"""
        if crypto == 'btc':
            if address.startswith('1'):
                return 'Legacy (P2PKH)'
            elif address.startswith('3'):
                return 'Script (P2SH)'
            elif address.startswith('bc1'):
                return 'Bech32 (SegWit)'
        elif crypto in ['eth', 'usdt', 'usdc', 'bnb'] and network in ['ERC20', 'BEP20']:
            return f'{network} Address'
        elif crypto == 'ton':
            if address.startswith('UQ'):
                return 'User Wallet'
            elif address.startswith('EQ'):
                return 'Smart Contract'
        
        return 'Standard Address'
    
    def calculate_withdrawal_fee(self, crypto: str, network: str = None, amount_usd: float = 0) -> Dict[str, Any]:
        """Calculate withdrawal fees for different cryptocurrencies"""
        try:
            if crypto not in self.supported_cryptos:
                return {'error': 'Unsupported cryptocurrency'}
            
            config = self.supported_cryptos[crypto]
            fee_config = config['fee']
            
            # Get network-specific fee if applicable
            if isinstance(fee_config, dict) and network:
                fee = fee_config.get(network, 0)
            else:
                fee = fee_config
            
            # Convert USD amount to crypto amount
            rate = self.exchange_rates.get(crypto.upper(), 1.0)
            crypto_amount = amount_usd / rate if rate > 0 else 0
            
            # Calculate fee in crypto and USD
            if crypto.upper() in ['USDT', 'USDC']:
                # Stablecoins - fee is in USD
                fee_usd = fee
                fee_crypto = fee
            else:
                # Other cryptos - fee is in native currency
                fee_crypto = fee
                fee_usd = fee * rate
            
            net_crypto_amount = crypto_amount - fee_crypto
            net_usd_amount = amount_usd - fee_usd
            
            return {
                'crypto': crypto.upper(),
                'network': network,
                'amount_usd': amount_usd,
                'amount_crypto': crypto_amount,
                'fee_crypto': fee_crypto,
                'fee_usd': fee_usd,
                'net_crypto': net_crypto_amount,
                'net_usd': net_usd_amount,
                'exchange_rate': rate,
                'min_amount': config['min_amount'],
                'max_amount': config['max_amount']
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def process_withdrawal(self, withdrawal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process cryptocurrency withdrawal"""
        try:
            crypto = withdrawal_data.get('method', '').lower()
            network = withdrawal_data.get('network')
            address = withdrawal_data.get('recipient')
            amount_usd = float(withdrawal_data.get('amount', 0))
            memo = withdrawal_data.get('memo', '')
            
            # Validate cryptocurrency
            if crypto not in self.supported_cryptos:
                return {'success': False, 'error': 'Unsupported cryptocurrency'}
            
            # Validate address
            address_validation = self.validate_address(crypto, address, network)
            if not address_validation['valid']:
                return {'success': False, 'error': f"Invalid address: {address_validation.get('error', 'Unknown error')}"}
            
            # Calculate fees
            fee_calculation = self.calculate_withdrawal_fee(crypto, network, amount_usd)
            if 'error' in fee_calculation:
                return {'success': False, 'error': fee_calculation['error']}
            
            # Check minimum/maximum amounts
            config = self.supported_cryptos[crypto]
            crypto_amount = fee_calculation['amount_crypto']
            
            if crypto_amount < config['min_amount']:
                return {'success': False, 'error': f"Minimum withdrawal: {config['min_amount']} {crypto.upper()}"}
            
            if crypto_amount > config['max_amount']:
                return {'success': False, 'error': f"Maximum withdrawal: {config['max_amount']} {crypto.upper()}"}
            
            # Generate transaction ID
            transaction_id = self._generate_transaction_id(crypto, network)
            
            # In production, this would integrate with actual blockchain APIs
            # For demo, we simulate the withdrawal process
            withdrawal_result = self._simulate_blockchain_withdrawal(
                crypto, network, address, fee_calculation['net_crypto'], memo, transaction_id
            )
            
            if withdrawal_result['success']:
                return {
                    'success': True,
                    'transaction_id': transaction_id,
                    'crypto': crypto.upper(),
                    'network': network,
                    'address': address,
                    'amount_usd': amount_usd,
                    'amount_crypto': fee_calculation['net_crypto'],
                    'fee_usd': fee_calculation['fee_usd'],
                    'fee_crypto': fee_calculation['fee_crypto'],
                    'exchange_rate': fee_calculation['exchange_rate'],
                    'status': 'processing',
                    'estimated_completion': (datetime.utcnow() + timedelta(hours=2)).isoformat(),
                    'confirmations_required': config.get('confirmations', 6),
                    'blockchain_tx': withdrawal_result.get('blockchain_tx'),
                    'memo': memo
                }
            else:
                return {'success': False, 'error': withdrawal_result.get('error', 'Withdrawal failed')}
                
        except Exception as e:
            return {'success': False, 'error': f"Processing error: {str(e)}"}
    
    def _generate_transaction_id(self, crypto: str, network: str = None) -> str:
        """Generate unique transaction ID"""
        timestamp = str(int(time.time()))
        crypto_part = crypto.upper()
        network_part = f"_{network}" if network else ""
        random_part = hashlib.md5(f"{timestamp}{crypto}{network}".encode()).hexdigest()[:8]
        
        return f"WD_{crypto_part}{network_part}_{timestamp}_{random_part}"
    
    def _simulate_blockchain_withdrawal(self, crypto: str, network: str, address: str, 
                                      amount: float, memo: str, transaction_id: str) -> Dict[str, Any]:
        """Simulate blockchain withdrawal (replace with actual API calls in production)"""
        try:
            # Simulate different success rates for different cryptos
            success_rates = {
                'btc': 0.98,
                'eth': 0.95,
                'usdt': 0.99,
                'usdc': 0.99,
                'bnb': 0.97,
                'ada': 0.96,
                'xmr': 0.94,  # Privacy coins might have lower success rates
                'zcash': 0.93,
                'dash': 0.95,
                'ton': 0.99,  # Fast networks
                'trx': 0.98,
                'ltc': 0.97,
                'sol': 0.96
            }
            
            import random
            success_rate = success_rates.get(crypto, 0.95)
            
            if random.random() < success_rate:
                # Simulate successful withdrawal
                blockchain_tx = self._generate_blockchain_tx(crypto, network)
                
                return {
                    'success': True,
                    'blockchain_tx': blockchain_tx,
                    'status': 'broadcasted',
                    'message': f'Transaction broadcasted to {crypto.upper()} network'
                }
            else:
                # Simulate failure
                error_messages = [
                    'Insufficient network fees',
                    'Network congestion',
                    'Temporary service unavailable',
                    'Address validation failed on blockchain'
                ]
                
                return {
                    'success': False,
                    'error': random.choice(error_messages)
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _generate_blockchain_tx(self, crypto: str, network: str = None) -> str:
        """Generate simulated blockchain transaction hash"""
        import secrets
        
        # Different hash formats for different blockchains
        if crypto in ['btc', 'ltc', 'dash']:
            # Bitcoin-style transaction hash
            return secrets.token_hex(32)
        elif crypto in ['eth', 'usdt', 'usdc', 'bnb'] and network in ['ERC20', 'BEP20']:
            # Ethereum-style transaction hash
            return '0x' + secrets.token_hex(32)
        elif crypto == 'xmr':
            # Monero transaction hash (longer)
            return secrets.token_hex(32)
        elif crypto == 'ton':
            # TON transaction hash format
            return secrets.token_hex(32)
        else:
            # Default format
            return secrets.token_hex(32)
    
    def get_withdrawal_status(self, transaction_id: str) -> Dict[str, Any]:
        """Get withdrawal status (simulate blockchain confirmation)"""
        try:
            # Extract crypto from transaction ID
            crypto_part = transaction_id.split('_')[1]
            crypto = crypto_part.lower().replace('erc20', 'usdt').replace('trc20', 'usdt').replace('bep20', 'usdt')
            
            if crypto not in self.supported_cryptos:
                return {'error': 'Invalid transaction ID'}
            
            config = self.supported_cryptos[crypto]
            required_confirmations = config.get('confirmations', 6)
            
            # Simulate confirmation progress
            import random
            current_confirmations = random.randint(0, required_confirmations + 2)
            
            if current_confirmations >= required_confirmations:
                status = 'completed'
                progress = 100
            else:
                status = 'confirming'
                progress = int((current_confirmations / required_confirmations) * 100)
            
            return {
                'transaction_id': transaction_id,
                'status': status,
                'confirmations': current_confirmations,
                'required_confirmations': required_confirmations,
                'progress': progress,
                'estimated_completion': 'Within 1 hour' if status == 'confirming' else 'Completed'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_supported_cryptocurrencies(self) -> Dict[str, Any]:
        """Get list of supported cryptocurrencies with details"""
        result = {}
        
        for crypto, config in self.supported_cryptos.items():
            result[crypto] = {
                'name': config['name'],
                'symbol': config['symbol'],
                'networks': config['networks'],
                'min_amount': config['min_amount'],
                'max_amount': config['max_amount'],
                'current_rate': self.exchange_rates.get(config['symbol'], 1.0),
                'estimated_fee_usd': self._estimate_fee_usd(crypto, config['fee'])
            }
        
        return result
    
    def _estimate_fee_usd(self, crypto: str, fee_config) -> float:
        """Estimate fee in USD for display purposes"""
        if isinstance(fee_config, dict):
            # Return average fee for multi-network cryptos
            fees = list(fee_config.values())
            avg_fee = sum(fees) / len(fees)
            return avg_fee
        else:
            # Single network crypto
            rate = self.exchange_rates.get(crypto.upper(), 1.0)
            if crypto.upper() in ['USDT', 'USDC']:
                return fee_config  # Already in USD
            else:
                return fee_config * rate
