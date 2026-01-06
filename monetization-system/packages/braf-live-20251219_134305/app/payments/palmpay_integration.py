"""
PalmPay Live Integration for Real Money Transfers
Handles actual PalmPay API calls for withdrawals in Nigerian Naira
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

class PalmPayIntegration:
    """Real PalmPay API integration for live money transfers"""
    
    def __init__(self):
        self.merchant_id = os.getenv('PALMPAY_MERCHANT_ID')
        self.api_key = os.getenv('PALMPAY_API_KEY')
        self.secret_key = os.getenv('PALMPAY_SECRET_KEY')
        self.base_url = os.getenv('PALMPAY_BASE_URL', 'https://api.palmpay.com/v1')
        self.webhook_secret = os.getenv('PALMPAY_WEBHOOK_SECRET')
        
        # Validate credentials
        if not all([self.merchant_id, self.api_key, self.secret_key]):
            logger.warning("PalmPay credentials not configured - running in demo mode")
            self.demo_mode = True
        else:
            self.demo_mode = False
    
    def _generate_signature(self, payload: str, timestamp: str) -> str:
        """Generate HMAC signature for PalmPay API"""
        message = f"{self.merchant_id}{timestamp}{payload}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest().upper()
        return signature
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to PalmPay API"""
        if self.demo_mode:
            return self._demo_response(endpoint, data)
        
        url = f"{self.base_url}/{endpoint}"
        timestamp = str(int(datetime.now().timestamp() * 1000))  # PalmPay uses milliseconds
        payload = json.dumps(data, separators=(',', ':'))
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'X-Merchant-Id': self.merchant_id,
            'X-Timestamp': timestamp,
            'X-Signature': self._generate_signature(payload, timestamp),
            'X-Request-Id': f"BRAF_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        }
        
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"PalmPay API request failed: {e}")
            return {
                'responseCode': '01',
                'responseMessage': f'API request failed: {str(e)}',
                'data': None
            }
    
    def _demo_response(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo response for testing"""
        if endpoint == 'transfer/wallet':
            return {
                'responseCode': '00',
                'responseMessage': 'SUCCESS',
                'data': {
                    'transactionReference': f"PALMPAY_DEMO_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'status': 'SUCCESS',
                    'amount': data.get('amount'),
                    'fee': str(int(float(data.get('amount', '0')) * 0.02)),  # 2% fee
                    'recipient': data.get('beneficiary', {}).get('phoneNumber'),
                    'transactionId': f"PP_TXN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                }
            }
        elif endpoint == 'account/balance':
            return {
                'responseCode': '00',
                'responseMessage': 'SUCCESS',
                'data': {
                    'availableBalance': '2000000.00',  # 2M NGN demo balance
                    'currency': 'NGN',
                    'accountNumber': self.merchant_id
                }
            }
        else:
            return {
                'responseCode': '00',
                'responseMessage': 'SUCCESS',
                'data': {}
            }
    
    def transfer_money(self, phone_number: str, amount_ngn: float, reference: str) -> Dict[str, Any]:
        """
        Transfer money to PalmPay wallet
        
        Args:
            phone_number: Recipient's phone number
            amount_ngn: Amount in Nigerian Naira
            reference: Unique transaction reference
            
        Returns:
            Dict containing transaction result
        """
        data = {
            'transactionReference': reference,
            'amount': str(amount_ngn),
            'currency': 'NGN',
            'beneficiary': {
                'phoneNumber': self.format_phone_number(phone_number),
                'accountName': 'BRAF User',  # In production, get from KYC data
                'bankCode': 'PALMPAY'
            },
            'narration': 'BRAF Automation Earnings Withdrawal',
            'callbackUrl': f"{os.getenv('BASE_URL', 'http://localhost:8003')}/api/v1/webhooks/palmpay"
        }
        
        logger.info(f"Initiating PalmPay transfer: {amount_ngn} NGN to {phone_number}")
        result = self._make_request('transfer/wallet', data)
        
        # Log transaction
        if result.get('responseCode') == '00':
            logger.info(f"PalmPay transfer successful: {result.get('data', {}).get('transactionReference')}")
        else:
            logger.error(f"PalmPay transfer failed: {result.get('responseMessage')}")
        
        return result
    
    def check_balance(self) -> Dict[str, Any]:
        """Check PalmPay merchant account balance"""
        return self._make_request('account/balance', {})
    
    def verify_transaction(self, reference: str) -> Dict[str, Any]:
        """Verify transaction status"""
        data = {'transactionReference': reference}
        return self._make_request('transaction/status', data)
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate Nigerian phone number format for PalmPay"""
        # Remove any non-digit characters
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # PalmPay accepts these Nigerian network prefixes
        valid_prefixes = [
            '070', '080', '081', '090', '091',  # MTN
            '081', '080', '070',  # Glo
            '070', '080', '081', '090',  # Airtel
            '081', '080'  # 9mobile
        ]
        
        if len(clean_number) == 11:
            return any(clean_number.startswith(prefix) for prefix in valid_prefixes)
        elif len(clean_number) == 13 and clean_number.startswith('234'):
            return any(clean_number[3:].startswith(prefix) for prefix in valid_prefixes)
        
        return False
    
    def format_phone_number(self, phone_number: str) -> str:
        """Format phone number for PalmPay API (requires +234 format)"""
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        if len(clean_number) == 11:
            return f"+234{clean_number[1:]}"  # Remove first 0, add +234
        elif len(clean_number) == 13 and clean_number.startswith('234'):
            return f"+{clean_number}"  # Add + prefix
        elif len(clean_number) == 10:
            return f"+234{clean_number}"  # Add +234 prefix
        
        return f"+234{clean_number}"  # Default format
    
    def get_transaction_fee(self, amount_ngn: float) -> float:
        """Calculate PalmPay transaction fee"""
        # PalmPay fee structure (example - check actual rates)
        if amount_ngn <= 5000:
            return 50.0  # Flat fee for small amounts
        elif amount_ngn <= 50000:
            return amount_ngn * 0.015  # 1.5% for medium amounts
        else:
            return min(amount_ngn * 0.02, 2000.0)  # 2% capped at 2000 NGN

# Global instance
palmpay_client = PalmPayIntegration()
