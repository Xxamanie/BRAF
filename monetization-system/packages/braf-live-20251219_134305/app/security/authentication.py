import pyotp
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
from database.service import DatabaseService

class SecurityManager:
    def __init__(self):
        self.failed_attempts = {}
        self.whitelist_cache = {}

    def setup_2fa(self, enterprise_id: str) -> Dict:
        """Setup two-factor authentication"""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        # Generate QR code for authenticator app
        provisioning_uri = totp.provisioning_uri(
            name=f"enterprise-{enterprise_id}@braf.com",
            issuer_name="BRAF Monetization"
        )
        
        # Store secret securely
        with DatabaseService() as db:
            db.save_2fa_secret(enterprise_id, secret)
        
        return {
            "secret": secret,  # For backup purposes only
            "provisioning_uri": provisioning_uri,
            "qr_code_url": f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={provisioning_uri}"
        }

    def verify_2fa(self, enterprise_id: str, token: str) -> bool:
        """Verify 2FA token"""
        # Check rate limiting
        if self.is_rate_limited(enterprise_id):
            return False
        
        # Get secret from database
        with DatabaseService() as db:
            secret = db.get_2fa_secret(enterprise_id)
            if not secret:
                return False
        
        totp = pyotp.TOTP(secret)
        is_valid = totp.verify(token, valid_window=1)
        
        if not is_valid:
            self.record_failed_attempt(enterprise_id)
        
        return is_valid

    def is_whitelisted(self, enterprise_id: str, address: str) -> bool:
        """Check if address is in whitelist"""
        whitelist = self.get_whitelist(enterprise_id)
        
        # Check exact match
        if address in whitelist:
            return True
        
        # For crypto addresses, check case-insensitive
        if address.lower() in [a.lower() for a in whitelist]:
            return True
        
        return False

    def get_whitelist(self, enterprise_id: str) -> List[str]:
        """Get withdrawal whitelist for enterprise"""
        if enterprise_id in self.whitelist_cache:
            return self.whitelist_cache[enterprise_id]
        
        whitelist = db.get_withdrawal_whitelist(enterprise_id)
        self.whitelist_cache[enterprise_id] = whitelist
        
        return whitelist

    def add_to_whitelist(self, enterprise_id: str, address: str,
                        address_type: str = "crypto") -> bool:
        """Add address to withdrawal whitelist"""
        # Validate address based on type
        if address_type == "crypto":
            if not self.validate_crypto_address(address):
                return False
        elif address_type == "mobile":
            if not self.validate_phone_number(address):
                return False
        
        # Add to database
        db.add_to_whitelist(enterprise_id, address, address_type)
        
        # Clear cache
        if enterprise_id in self.whitelist_cache:
            del self.whitelist_cache[enterprise_id]
        
        return True

    def detect_suspicious_activity(self, enterprise_id: str,
                                  activity: Dict) -> Optional[Dict]:
        """Detect suspicious activity patterns"""
        alerts = []
        
        # Check for abnormal withdrawal patterns
        if activity.get("type") == "withdrawal":
            if self.is_abnormal_withdrawal(enterprise_id, activity):
                alerts.append({
                    "type": "abnormal_withdrawal",
                    "severity": "high",
                    "activity": activity
                })
        
        # Check for multiple failed attempts
        if activity.get("type") == "authentication":
            if self.is_brute_force_attempt(enterprise_id, activity):
                alerts.append({
                    "type": "brute_force_attempt",
                    "severity": "critical",
                    "activity": activity
                })
        
        # Check for geographic anomalies
        if self.is_geographic_anomaly(enterprise_id, activity):
            alerts.append({
                "type": "geographic_anomaly",
                "severity": "medium",
                "activity": activity
            })
        
        if alerts:
            return {
                "enterprise_id": enterprise_id,
                "alerts": alerts,
                "timestamp": datetime.utcnow().isoformat(),
                "action_required": any(a["severity"] in ["high", "critical"] for a in alerts)
            }
        
        return None

    def get_transaction_limits(self, enterprise_id: str, kyc_level: int) -> Dict:
        """Get transaction limits based on KYC level"""
        limits = {
            0: {  # No KYC
                "daily_withdrawal": 500,
                "monthly_withdrawal": 5000,
                "max_per_transaction": 500,
                "currency_restrictions": ["USDT", "OPay", "PalmPay"]
            },
            1: {  # Basic KYC
                "daily_withdrawal": 2000,
                "monthly_withdrawal": 20000,
                "max_per_transaction": 2000,
                "currency_restrictions": []
            },
            2: {  # Full KYC
                "daily_withdrawal": 10000,
                "monthly_withdrawal": 100000,
                "max_per_transaction": 5000,
                "currency_restrictions": []
            }
        }
        
        return limits.get(kyc_level, limits[0])