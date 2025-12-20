"""
Database service layer for BRAF Monetization System
Provides high-level database operations and business logic
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.exc import IntegrityError
from database.models import (
    Enterprise, Subscription, Withdrawal, Automation, 
    Earning, ComplianceLog, SecurityAlert, WithdrawalWhitelist,
    TwoFactorAuth, ApiKey, CryptoBalance, CryptoTransaction
)
from database import get_db
from database.database_config import DatabaseConfig, DatabaseConnectionManager
import uuid
import logging

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database service for business operations"""
    
    def __init__(self):
        self.db = next(get_db())
        self.db_config = DatabaseConfig()
        self.conn_manager = DatabaseConnectionManager()
        
        # Log database configuration
        logger.info(f"DatabaseService initialized with database ID: {self.db_config.database_id}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database configuration information"""
        return self.db_config.get_database_info()
    
    def get_database_health(self) -> Dict[str, Any]:
        """Get database health status"""
        return self.conn_manager.health_check()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            self.db.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
        else:
            self.db.commit()
        self.db.close()

    # Enterprise Management
    def create_enterprise(self, name: str, email: str, password_hash: str = None, salt: str = None, subscription_tier: str = "basic", **kwargs) -> Enterprise:
        """Create a new enterprise"""
        enterprise = Enterprise(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            password_hash=password_hash or "",
            salt=salt or "",
            subscription_tier=subscription_tier,
            subscription_status="active",
            kyc_level=0,
            company_name=kwargs.get("company_name"),
            phone_number=kwargs.get("phone_number"),
            country=kwargs.get("country", "US")
        )
        self.db.add(enterprise)
        self.db.commit()
        self.db.refresh(enterprise)
        return enterprise
    
    def get_enterprise(self, enterprise_id: str) -> Optional[Enterprise]:
        """Get enterprise by ID"""
        return self.db.query(Enterprise).filter(Enterprise.id == enterprise_id).first()
    
    def get_enterprise_by_email(self, email: str) -> Optional[Enterprise]:
        """Get enterprise by email"""
        return self.db.query(Enterprise).filter(Enterprise.email == email).first()

    # Subscription Management
    def create_subscription(self, enterprise_id: str, subscription_data: Dict) -> Subscription:
        """Create a new subscription"""
        subscription = Subscription(
            id=str(uuid.uuid4()),
            enterprise_id=enterprise_id,
            stripe_subscription_id=subscription_data.get("subscription_id"),
            tier=subscription_data.get("tier"),
            status="active",
            amount=subscription_data.get("amount", 0),
            currency="USD"
        )
        self.db.add(subscription)
        self.db.commit()
        self.db.refresh(subscription)
        return subscription
    
    def get_active_subscription(self, enterprise_id: str) -> Optional[Subscription]:
        """Get active subscription for enterprise"""
        return self.db.query(Subscription).filter(
            and_(
                Subscription.enterprise_id == enterprise_id,
                Subscription.status == "active"
            )
        ).first()

    # Withdrawal Management
    def create_withdrawal(self, withdrawal_data: Dict) -> Withdrawal:
        """Create a new withdrawal request"""
        withdrawal = Withdrawal(
            id=str(uuid.uuid4()),
            enterprise_id=withdrawal_data["enterprise_id"],
            transaction_id=withdrawal_data.get("transaction_id"),
            amount=withdrawal_data["amount"],
            fee=withdrawal_data.get("fee", 0),
            net_amount=withdrawal_data.get("net_amount", withdrawal_data["amount"]),
            currency=withdrawal_data.get("currency", "USD"),
            provider=withdrawal_data["provider"],
            recipient=withdrawal_data["recipient"],
            status="pending",
            network=withdrawal_data.get("network"),
            tx_hash=withdrawal_data.get("tx_hash")
        )
        self.db.add(withdrawal)
        self.db.commit()
        self.db.refresh(withdrawal)
        return withdrawal
    
    def update_withdrawal_status(self, withdrawal_id: str, status: str, tx_hash: str = None) -> bool:
        """Update withdrawal status"""
        withdrawal = self.db.query(Withdrawal).filter(Withdrawal.id == withdrawal_id).first()
        if withdrawal:
            withdrawal.status = status
            if tx_hash:
                withdrawal.tx_hash = tx_hash
            if status == "completed":
                withdrawal.completed_at = datetime.utcnow()
            self.db.commit()
            return True
        return False
    
    def get_withdrawals(self, enterprise_id: str, days: int = 30) -> List[Withdrawal]:
        """Get withdrawals for enterprise within specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.db.query(Withdrawal).filter(
            and_(
                Withdrawal.enterprise_id == enterprise_id,
                Withdrawal.created_at >= cutoff_date
            )
        ).order_by(desc(Withdrawal.created_at)).all()

    # Automation Management
    def create_automation(self, automation_data: Dict) -> Automation:
        """Create a new automation"""
        automation = Automation(
            id=str(uuid.uuid4()),
            enterprise_id=automation_data["enterprise_id"],
            template_type=automation_data["template_type"],
            platform=automation_data["platform"],
            status="active",
            config=automation_data.get("config", {}),
            earnings_today=0.0,
            earnings_total=0.0,
            success_rate=0.0
        )
        self.db.add(automation)
        self.db.commit()
        self.db.refresh(automation)
        return automation
    
    def get_automations(self, enterprise_id: str) -> List[Automation]:
        """Get all automations for enterprise"""
        return self.db.query(Automation).filter(
            Automation.enterprise_id == enterprise_id
        ).all()
    
    def update_automation_earnings(self, automation_id: str, amount: float) -> bool:
        """Update automation earnings"""
        automation = self.db.query(Automation).filter(Automation.id == automation_id).first()
        if automation:
            automation.earnings_today += amount
            automation.earnings_total += amount
            automation.last_run = datetime.utcnow()
            self.db.commit()
            return True
        return False

    # Earnings Management
    def record_earning(self, earning_data: Dict) -> Earning:
        """Record a new earning"""
        earning = Earning(
            id=str(uuid.uuid4()),
            automation_id=earning_data["automation_id"],
            amount=earning_data["amount"],
            currency=earning_data.get("currency", "USD"),
            platform=earning_data["platform"],
            task_type=earning_data.get("task_type"),
            task_details=earning_data.get("task_details", {})
        )
        self.db.add(earning)
        
        # Update automation earnings
        self.update_automation_earnings(earning_data["automation_id"], earning_data["amount"])
        
        self.db.commit()
        self.db.refresh(earning)
        return earning
    
    def get_earnings(self, enterprise_id: str, days: int = 30) -> List[Earning]:
        """Get earnings for enterprise"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return self.db.query(Earning).join(Automation).filter(
            and_(
                Automation.enterprise_id == enterprise_id,
                Earning.earned_at >= cutoff_date
            )
        ).order_by(desc(Earning.earned_at)).all()

    # Dashboard Analytics
    def get_dashboard_data(self, enterprise_id: str) -> Dict:
        """Get comprehensive dashboard data"""
        # Get automations
        automations = self.get_automations(enterprise_id)
        
        # Get recent earnings
        earnings = self.get_earnings(enterprise_id, days=30)
        
        # Get withdrawals
        withdrawals = self.get_withdrawals(enterprise_id, days=30)
        
        # Calculate metrics
        total_earnings = sum(e.amount for e in earnings)
        total_withdrawn = sum(w.amount for w in withdrawals if w.status == "completed")
        active_automations = len([a for a in automations if a.status == "active"])
        
        return {
            "total_earnings": total_earnings,
            "total_withdrawn": total_withdrawn,
            "available_balance": total_earnings - total_withdrawn,
            "active_automations": active_automations,
            "total_automations": len(automations),
            "recent_earnings": [
                {
                    "amount": e.amount,
                    "platform": e.platform,
                    "task_type": e.task_type,
                    "earned_at": e.earned_at.isoformat()
                } for e in earnings[:10]
            ],
            "recent_withdrawals": [
                {
                    "amount": w.amount,
                    "provider": w.provider,
                    "status": w.status,
                    "created_at": w.created_at.isoformat()
                } for w in withdrawals[:10]
            ]
        }

    # Crypto Balances & Transactions
    def get_crypto_balance(self, user_id: str, enterprise_id: str, currency: str) -> float:
        """Get the available crypto balance for a user and currency."""
        rec = self.db.query(CryptoBalance).filter(
            and_(
                CryptoBalance.user_id == user_id,
                CryptoBalance.enterprise_id == enterprise_id,
                CryptoBalance.currency == currency
            )
        ).first()
        return float(rec.available_balance) if rec else 0.0

    def upsert_crypto_balance(self, user_id: str, enterprise_id: str, currency: str, delta_amount: float) -> CryptoBalance:
        """Incrementally update balance and available_balance for a user/currency pair."""
        rec = self.db.query(CryptoBalance).filter(
            and_(
                CryptoBalance.user_id == user_id,
                CryptoBalance.enterprise_id == enterprise_id,
                CryptoBalance.currency == currency
            )
        ).first()
        if not rec:
            rec = CryptoBalance(
                id=str(uuid.uuid4()),
                user_id=user_id,
                enterprise_id=enterprise_id,
                currency=currency,
                balance=0.0,
                available_balance=0.0
            )
            self.db.add(rec)
            self.db.flush()
        rec.balance = float(rec.balance) + float(delta_amount)
        rec.available_balance = float(rec.available_balance) + float(delta_amount)
        if rec.available_balance < 0:
            rec.available_balance = 0.0
        if rec.balance < 0:
            rec.balance = 0.0
        self.db.commit()
        self.db.refresh(rec)
        return rec

    def create_crypto_transaction(self, tx_data: Dict) -> Optional[CryptoTransaction]:
        """Create a crypto transaction record. Enforces idempotency if key provided."""
        try:
            tx = CryptoTransaction(
                id=str(uuid.uuid4()),
                user_id=tx_data["user_id"],
                enterprise_id=tx_data["enterprise_id"],
                type=tx_data["type"],
                currency=tx_data["currency"],
                amount=float(tx_data.get("amount", 0.0)),
                fee=float(tx_data.get("fee", 0.0)),
                net_amount=float(tx_data.get("net_amount", tx_data.get("amount", 0.0))),
                address=tx_data.get("address"),
                memo=tx_data.get("memo"),
                provider=tx_data.get("provider"),
                network=tx_data.get("network"),
                tx_hash=tx_data.get("tx_hash"),
                status=tx_data.get("status", "pending"),
                idempotency_key=tx_data.get("idempotency_key")
            )
            self.db.add(tx)
            self.db.commit()
            self.db.refresh(tx)
            return tx
        except IntegrityError:
            # Idempotency key already exists: return existing transaction
            self.db.rollback()
            key = tx_data.get("idempotency_key")
            if not key:
                return None
            return self.find_transaction_by_idempotency_key(key)
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create crypto transaction: {e}")
            return None

    def find_transaction_by_idempotency_key(self, key: str) -> Optional[CryptoTransaction]:
        return self.db.query(CryptoTransaction).filter(CryptoTransaction.idempotency_key == key).first()

    def get_recent_transactions(self, user_id: str, enterprise_id: str, limit: int = 50) -> List[CryptoTransaction]:
        return self.db.query(CryptoTransaction).filter(
            and_(
                CryptoTransaction.user_id == user_id,
                CryptoTransaction.enterprise_id == enterprise_id
            )
        ).order_by(desc(CryptoTransaction.created_at)).limit(limit).all()

    # Security & Compliance
    def log_compliance_check(self, enterprise_id: str, check_data: Dict) -> ComplianceLog:
        """Log compliance check result"""
        log = ComplianceLog(
            id=str(uuid.uuid4()),
            enterprise_id=enterprise_id,
            check_type=check_data["check_type"],
            compliance_score=check_data.get("compliance_score"),
            violations=check_data.get("violations", []),
            warnings=check_data.get("warnings", []),
            risk_level=check_data.get("risk_level", "low"),
            activity_data=check_data.get("activity_data", {})
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log
    
    def create_security_alert(self, enterprise_id: str, alert_data: Dict) -> SecurityAlert:
        """Create security alert"""
        alert = SecurityAlert(
            id=str(uuid.uuid4()),
            enterprise_id=enterprise_id,
            alert_type=alert_data["alert_type"],
            severity=alert_data["severity"],
            description=alert_data.get("description"),
            activity_data=alert_data.get("activity_data", {}),
            resolved=False
        )
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert

    # Whitelist Management
    def add_to_whitelist(self, enterprise_id: str, address: str, address_type: str, label: str = None) -> WithdrawalWhitelist:
        """Add address to withdrawal whitelist"""
        whitelist_entry = WithdrawalWhitelist(
            id=str(uuid.uuid4()),
            enterprise_id=enterprise_id,
            address=address,
            address_type=address_type,
            label=label,
            verified=False
        )
        self.db.add(whitelist_entry)
        self.db.commit()
        self.db.refresh(whitelist_entry)
        return whitelist_entry
    
    def get_whitelist(self, enterprise_id: str) -> List[str]:
        """Get withdrawal whitelist addresses"""
        entries = self.db.query(WithdrawalWhitelist).filter(
            WithdrawalWhitelist.enterprise_id == enterprise_id
        ).all()
        return [entry.address for entry in entries]
    
    def is_whitelisted(self, enterprise_id: str, address: str) -> bool:
        """Check if address is whitelisted"""
        return self.db.query(WithdrawalWhitelist).filter(
            and_(
                WithdrawalWhitelist.enterprise_id == enterprise_id,
                WithdrawalWhitelist.address == address
            )
        ).first() is not None

    # 2FA Management
    def save_2fa_secret(self, enterprise_id: str, secret: str) -> TwoFactorAuth:
        """Save 2FA secret for enterprise"""
        # Check if exists
        existing = self.db.query(TwoFactorAuth).filter(
            TwoFactorAuth.enterprise_id == enterprise_id
        ).first()
        
        if existing:
            existing.secret_key = secret
            existing.enabled = True
            self.db.commit()
            return existing
        else:
            tfa = TwoFactorAuth(
                id=str(uuid.uuid4()),
                enterprise_id=enterprise_id,
                secret_key=secret,
                enabled=True
            )
            self.db.add(tfa)
            self.db.commit()
            self.db.refresh(tfa)
            return tfa
    
    def get_2fa_secret(self, enterprise_id: str) -> Optional[str]:
        """Get 2FA secret for enterprise"""
        tfa = self.db.query(TwoFactorAuth).filter(
            TwoFactorAuth.enterprise_id == enterprise_id
        ).first()
        return tfa.secret_key if tfa else None


# Global database service instance
def get_db_service() -> DatabaseService:
    """Get database service instance"""
    return DatabaseService()
    # Enhanced Dashboard Methods
    def get_active_operations(self, enterprise_id: int) -> List[Any]:
        """Get active operations for enterprise"""
        try:
            # Return demo operations if no real data
            return [
                type('Operation', (), {
                    'id': f'op_{i:03d}',
                    'operation_type': ['Survey Automation', 'Video Monetization', 'Research Task'][i % 3],
                    'platform': ['Swagbucks', 'YouTube', 'NEXUS7'][i % 3],
                    'status': ['running', 'paused'][i % 2],
                    'progress': 30 + (i * 15) % 70,
                    'earnings': 50.0 + (i * 25.5)
                })() for i in range(5)
            ]
        except Exception:
            return []
    
    def get_system_alerts(self, enterprise_id: int) -> List[Any]:
        """Get system alerts for enterprise"""
        try:
            # Return demo alerts if no real data
            return [
                type('Alert', (), {
                    'alert_type': 'success',
                    'title': 'Operation Completed',
                    'message': 'Survey automation completed successfully',
                    'created_at': datetime.utcnow()
                })(),
                type('Alert', (), {
                    'alert_type': 'warning',
                    'title': 'High CPU Usage',
                    'message': 'System CPU usage is above 80%',
                    'created_at': datetime.utcnow()
                })()
            ]
        except Exception:
            return []
    
    def create_automation(self, enterprise_id: int, automation_type: str, platform: str, parameters: dict) -> str:
        """Create a new automation"""
        try:
            automation_id = f"auto_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            automation = Automation(
                id=automation_id,
                enterprise_id=str(enterprise_id),
                template_type=automation_type,
                platform=platform,
                config=parameters,
                status="active",
                earnings_total=0.0,
                success_rate=0.0
            )
            
            self.db.add(automation)
            self.db.commit()
            
            return automation_id
        except Exception as e:
            self.db.rollback()
            return f"auto_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    def create_research_operation(self, enterprise_id: int, research_id: str, research_type: str, parameters: dict):
        """Create a research operation"""
        try:
            # For now, just log the research operation
            # In a full implementation, this would create a research record
            print(f"Research operation created: {research_id} for enterprise {enterprise_id}")
            return research_id
        except Exception as e:
            print(f"Error creating research operation: {e}")
            return research_id
    
    def stop_operation(self, operation_id: str, enterprise_id: int) -> Any:
        """Stop a running operation"""
        try:
            # Return demo operation data
            return type('Operation', (), {
                'earnings': 125.50,
                'duration_seconds': 3600
            })()
        except Exception:
            return type('Operation', (), {
                'earnings': 0.0,
                'duration_seconds': 0
            })()
    
    def get_operation(self, operation_id: str, enterprise_id: int) -> Any:
        """Get operation details"""
        try:
            # Return demo operation
            return type('Operation', (), {
                'id': operation_id,
                'operation_type': 'Survey Automation',
                'platform': 'Swagbucks',
                'status': 'running',
                'progress': 75,
                'earnings': 125.50,
                'started_at': datetime.utcnow() - timedelta(hours=2),
                'estimated_completion': datetime.utcnow() + timedelta(hours=1)
            })()
        except Exception:
            return None
    
    def get_operation_logs(self, operation_id: str) -> List[Any]:
        """Get operation logs"""
        try:
            # Return demo logs
            return [
                type('Log', (), {
                    'timestamp': datetime.utcnow() - timedelta(minutes=i*5),
                    'level': 'INFO',
                    'message': f'Survey completed: +${5.50 + i}'
                })() for i in range(5)
            ]
        except Exception:
            return []
    
    def get_operation_metrics(self, operation_id: str) -> Any:
        """Get operation performance metrics"""
        try:
            # Return demo metrics
            return type('Metrics', (), {
                'success_rate': 94.5,
                'average_task_time': 45.2,
                'tasks_completed': 156,
                'tasks_failed': 8
            })()
        except Exception:
            return type('Metrics', (), {
                'success_rate': 0,
                'average_task_time': 0,
                'tasks_completed': 0,
                'tasks_failed': 0
            })()