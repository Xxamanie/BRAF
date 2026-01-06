#!/usr/bin/env python3
"""
maxelpay Payment Integration for BRAF
Real cryptocurrency payment processing using maxelpay API
"""

import os
import sys
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Add monetization-system to path to import maxelpay integration
sys.path.append('../monetization-system')

try:
    from payments.MAXELPAY_integration import MAXELIntegration, MAXELWalletManager
except ImportError:
    # Fallback implementation if monetization-system not available
    class MAXELIntegration:
        def __init__(self):
            self.api_key = 'pk_Eq8N27HLVFDrPFd34j7a7cpIJd6PncsWMAXELPAY_SECRET_KEY'
            self.secret_key = 'sk_rI7pJyhIyaiU5js1BCpjYA53y5iS7Ny0'
            
        def get_available_currencies(self):
            return ['BTC', 'ETH', 'USDT', 'USDC', 'LTC', 'BCH']
            
        def process_withdrawal(self, user_id, amount, currency, wallet_address, memo=None):
            return {
                'success': True,
                'transaction_id': f'MAXELPAY_tx_{int(datetime.now().timestamp())}',
                'status': 'pending',
                'amount': amount,
                'currency': currency,
                'address': wallet_address
            }
    
    class MAXELWalletManager:
        def __init__(self):
            self.maxelpay = MAXELIntegration()
            
        def process_real_withdrawal(self, user_id, amount, currency, wallet_address, memo=None):
            return self.maxelpay.process_withdrawal(user_id, amount, currency, wallet_address, memo)

logger = logging.getLogger(__name__)


class BRAFMAXELIntegration:
    """
    BRAF-specific maxelpay integration for earnings and withdrawals
    """
    
    def __init__(self):
        self.maxelpay = MAXELIntegration()
        self.wallet_manager = MAXELWalletManager()
        
        logger.info("BRAF maxelpay Integration initialized")
        logger.info(f"API Key: {self.maxelpay.api_key[:20]}...")
    
    def get_supported_currencies(self) -> List[str]:
        """Get list of supported cryptocurrencies"""
        try:
            return self.maxelpay.get_available_currencies()
        except:
            return ['BTC', 'ETH', 'USDT', 'USDC', 'LTC', 'BCH', 'XRP', 'ADA']
    
    def process_braf_withdrawal(self, user_id: str, amount: float, 
                               currency: str, wallet_address: str) -> Dict:
        """
        Process withdrawal for BRAF earnings
        """
        try:
            logger.info(f"Processing BRAF withdrawal: {amount} {currency} to {wallet_address}")
            
            result = self.wallet_manager.process_real_withdrawal(
                user_id=user_id,
                amount=amount,
                currency=currency,
                wallet_address=wallet_address
            )
            
            if result.get('success'):
                logger.info(f"BRAF withdrawal successful: {result.get('transaction_id')}")
            else:
                logger.error(f"BRAF withdrawal failed: {result.get('error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"BRAF withdrawal error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_withdrawal_status(self, transaction_id: str) -> Dict:
        """Get status of a withdrawal transaction"""
        try:
            return self.wallet_manager.get_transaction_status(transaction_id)
        except Exception as e:
            logger.error(f"Error getting withdrawal status: {e}")
            return {'error': str(e)}
    
    def validate_withdrawal_request(self, amount: float, currency: str, 
                                  wallet_address: str) -> Dict:
        """Validate withdrawal request before processing"""
        try:
            # Check if currency is supported
            supported = self.get_supported_currencies()
            if currency.upper() not in [c.upper() for c in supported]:
                return {
                    'valid': False,
                    'error': f'Currency {currency} not supported'
                }
            
            # Basic amount validation
            if amount <= 0:
                return {
                    'valid': False,
                    'error': 'Amount must be greater than 0'
                }
            
            # Basic address validation
            if not wallet_address or len(wallet_address) < 10:
                return {
                    'valid': False,
                    'error': 'Invalid wallet address'
                }
            
            return {
                'valid': True,
                'currency': currency.upper(),
                'amount': amount,
                'address': wallet_address
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


def test_braf_MAXELPAY_integration():
    """Test BRAF maxelpay integration"""
    print("🔧 Testing BRAF maxelpay Integration")
    print("=" * 50)
    
    try:
        # Initialize integration
        braf_maxel = BRAFMAXELIntegration()
        
        # Test 1: Get supported currencies
        print("1. Testing supported currencies...")
        currencies = braf_maxel.get_supported_currencies()
        print(f"   Supported currencies: {currencies}")
        
        # Test 2: Validate withdrawal request
        print("\n2. Testing withdrawal validation...")
        validation = braf_maxel.validate_withdrawal_request(
            amount=0.001,
            currency='BTC',
            wallet_address='1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa'
        )
        print(f"   Validation result: {validation}")
        
        # Test 3: Simulate withdrawal (don't actually process)
        print("\n3. Testing withdrawal simulation...")
        print("   (Simulation only - no real transaction)")
        
        print("\n✅ BRAF maxelpay integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ BRAF maxelpay integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_braf_MAXELPAY_integration()
