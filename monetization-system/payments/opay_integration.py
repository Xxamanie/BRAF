"""
OPay Live Integration for Real Money Transfers
Handles actual OPay API calls for withdrawals in Nigerian Naira
"""

import os
import json
import hmac
import hashlib
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OPayIntegration:
    """Real OPay API integration for live money transfers"""
    
    def __init__(self):
        self.merchant_id = os.getenv('OPAY_MERCHANT_ID')
        self.api_key = os.getenv('OPAY_API_KEY')
        self.secret_key = os.getenv('OPAY_SECRET_KEY')
        self.base_url = os.getenv('OPAY_BASE_URL', 'https://api.opayweb.com/v3')
        self.webhook_secret = os.getenv('OPAY_WEBHOOK_SECRET')
        
        # Exit simulation mode - force real mode
        self.demo_mode = False
        logger.info("OPay integration: Real mode enabled - simulation mode exited")
    
    def _generate_signature(self, payload: str, timestamp: str) -> str:
        """Generate HMAC signature for OPay API"""
        message = f"{timestamp}{payload}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to OPay API"""
        if self.demo_mode:
            return self._demo_response(endpoint, data)
        
        url = f"{self.base_url}/{endpoint}"
        timestamp = str(int(datetime.now().timestamp()))
        payload = json.dumps(data, separators=(',', ':'))
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'MerchantId': self.merchant_id,
            'Timestamp': timestamp,
            'Signature': self._generate_signature(payload, timestamp)
        }
        
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OPay API request failed: {e}")
            return {
                'code': '02000',
                'message': f'API request failed: {str(e)}',
                'data': None
            }
    
    def _demo_response(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo response for testing"""
        if endpoint == 'transfer':
            return {
                'code': '00000',
                'message': 'SUCCESS',
                'data': {
                    'reference': f"OPAY_DEMO_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'status': 'SUCCESS',
                    'amount': data.get('amount'),
                    'fee': str(int(float(data.get('amount', '0')) * 0.015)),  # 1.5% fee
                    'recipient': data.get('recipient', {}).get('phoneNumber'),
                    'transactionId': f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                }
            }
        elif endpoint == 'balance':
            return {
                'code': '00000',
                'message': 'SUCCESS',
                'data': {
                    'availableBalance': '1000000.00',  # 1M NGN demo balance
                    'currency': 'NGN'
                }
            }
        else:
            return {
                'code': '00000',
                'message': 'SUCCESS',
                'data': {}
            }
    
    def transfer_money(self, phone_number: str, amount_ngn: float, reference: str) -> Dict[str, Any]:
        """
        Transfer money to OPay account
        
        Args:
            phone_number: Recipient's phone number
            amount_ngn: Amount in Nigerian Naira
            reference: Unique transaction reference
            
        Returns:
            Dict containing transaction result
        """
        data = {
            'reference': reference,
            'amount': str(int(amount_ngn * 100)),  # Convert to kobo
            'currency': 'NGN',
            'recipient': {
                'phoneNumber': phone_number,
                'name': 'BRAF User'  # In production, get from KYC data
            },
            'reason': 'BRAF Automation Earnings Withdrawal',
            'callbackUrl': f"{os.getenv('BASE_URL', 'http://localhost:8003')}/api/v1/webhooks/opay"
        }
        
        logger.info(f"Initiating OPay transfer: {amount_ngn} NGN to {phone_number}")
        result = self._make_request('transfer', data)
        
        # Log transaction
        if result.get('code') == '00000':
            logger.info(f"OPay transfer successful: {result.get('data', {}).get('reference')}")
        else:
            logger.error(f"OPay transfer failed: {result.get('message')}")
        
        return result
    
    def check_balance(self) -> Dict[str, Any]:
        """Check OPay merchant account balance"""
        return self._make_request('balance', {})
    
    def verify_transaction(self, reference: str) -> Dict[str, Any]:
        """Verify transaction status"""
        data = {'reference': reference}
        return self._make_request('transaction/status', data)
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate Nigerian phone number format"""
        # Remove any non-digit characters
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Check if it's a valid Nigerian number
        if len(clean_number) == 11 and clean_number.startswith(('070', '080', '081', '090', '091')):
            return True
        elif len(clean_number) == 13 and clean_number.startswith('234'):
            return True
        
        return False
    
    def format_phone_number(self, phone_number: str) -> str:
        """Format phone number for OPay API"""
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        if len(clean_number) == 11:
            return clean_number
        elif len(clean_number) == 13 and clean_number.startswith('234'):
            return clean_number[3:]  # Remove country code
        
        return clean_number

# Global instance
opay_client = OPayIntegration()
