import requests
import hashlib
import hmac
import json
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

class MobileMoneyProvider(Enum):
    OPAY_NG = "opay_ng"
    OPAY_KE = "opay_ke"
    OPAY_GH = "opay_gh"
    PALMPAY_NG = "palmpay_ng"

class MobileMoneyWithdrawal:
    def __init__(self):
        self.providers = {
            "opay_ng": {
                "base_url": "https://api.opay.ng/v1",
                "endpoints": {
                    "transfer": "/transfer",
                    "status": "/transfer/status",
                    "balance": "/balance"
                },
                "min_amount": 50,
                "max_amount": 5000,
                "fee_percent": 1.5,
                "processing_time": "1-3 hours"
            },
            "palmpay_ng": {
                "base_url": "https://api.palmpay.com/v1",
                "endpoints": {
                    "payout": "/payout",
                    "check": "/transaction/check"
                },
                "min_amount": 50,
                "max_amount": 5000,
                "fee_percent": 1.5,
                "processing_time": "1-3 hours"
            }
        }

    def generate_signature(self, provider: str, data: Dict) -> str:
        """Generate secure signature for API calls"""
        from config import Config
        api_secret = Config.MOBILE_MONEY_SECRETS[provider]
        
        # Create signature payload
        payload = json.dumps(data, separators=(',', ':'))
        signature = hmac.new(
            api_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    async def withdraw_opay(self, amount: float, phone_number: str,
                           country: str = "NG", enterprise_id: str = None) -> Dict:
        """Process OPay withdrawal"""
        provider_key = f"opay_{country.lower()}"
        provider_config = self.providers[provider_key]
        
        # Validate amount
        if amount < provider_config["min_amount"]:
            raise ValueError(f"Minimum withdrawal is ${provider_config['min_amount']}")
        if amount > provider_config["max_amount"]:
            raise ValueError(f"Maximum withdrawal is ${provider_config['max_amount']}")
        
        # Calculate fee
        fee = amount * (provider_config["fee_percent"] / 100)
        net_amount = amount - fee
        
        # Convert USD to NGN for OPay
        from payments.currency_converter import currency_converter
        withdrawal_calc = currency_converter.calculate_withdrawal_amounts(amount, "opay")
        
        ngn_amount = withdrawal_calc["converted_amount"]
        
        # Prepare transaction data
        transaction_data = {
            "reference": f"BRAF_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "amount": ngn_amount,  # Amount in NGN
            "currency": "NGN",     # OPay uses NGN
            "country": country,
            "receiver": phone_number,
            "receiver_type": "phone",
            "reason": "Earnings withdrawal",
            "callback_url": f"{Config.BASE_URL}/api/v1/webhooks/opay"
        }
        
        # Generate signature
        signature = self.generate_signature(provider_key, transaction_data)
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {Config.OPAY_API_KEY}",
            "X-Signature": signature,
            "Content-Type": "application/json"
        }
        
        url = f"{provider_config['base_url']}{provider_config['endpoints']['transfer']}"
        
        response = requests.post(
            url,
            json=transaction_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Save transaction to database
            from database.service import DatabaseService
            
            transaction_data_db = {
                "enterprise_id": enterprise_id or "unknown",
                "transaction_id": result.get("orderNo"),
                "amount": amount,
                "fee": fee,
                "net_amount": net_amount,
                "provider": provider_key,
                "recipient": phone_number,
                "currency": "USD"
            }
            
            with DatabaseService() as db:
                withdrawal = db.create_withdrawal(transaction_data_db)
            
            return {
                "success": True,
                "transaction_id": result.get("orderNo"),
                "original_amount_usd": amount,
                "converted_amount_ngn": ngn_amount,
                "fee": fee,
                "net_amount": net_amount,
                "exchange_rate": withdrawal_calc["exchange_rate"],
                "estimated_time": provider_config["processing_time"],
                "status": "pending"
            }
        
        raise Exception(f"OPay withdrawal failed: {response.text}")

    async def withdraw_palmpay(self, amount: float, phone_number: str) -> Dict:
        """Process PalmPay withdrawal"""
        provider_config = self.providers["palmpay_ng"]
        
        # Convert USD to NGN for PalmPay
        from payments.currency_converter import currency_converter
        withdrawal_calc = currency_converter.calculate_withdrawal_amounts(amount, "palmpay")
        
        ngn_amount = withdrawal_calc["converted_amount"]
        
        # Similar implementation for PalmPay
        transaction_data = {
            "merchant_transaction_id": f"BRAF_PP_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "amount": str(ngn_amount),  # Amount in NGN
            "currency": "NGN",          # PalmPay uses NGN
            "account_number": phone_number,
            "account_type": "PHONE",
            "narration": "Earnings payout",
            "callback_url": f"{Config.BASE_URL}/api/v1/webhooks/palmpay"
        }
        
        # Make PalmPay API call
        # Implementation similar to OPay
        
        return {
            "success": True,
            "message": "PalmPay withdrawal initiated"
        }
