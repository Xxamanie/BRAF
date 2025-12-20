from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Enterprise(Base):
    __tablename__ = "enterprises"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(255), nullable=False)
    subscription_tier = Column(String(50), default="basic")
    subscription_status = Column(String(50), default="active")
    kyc_level = Column(Integer, default=0)
    company_name = Column(String(255))
    phone_number = Column(String(50))
    country = Column(String(10), default="US")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="enterprise")
    withdrawals = relationship("Withdrawal", back_populates="enterprise")
    automations = relationship("Automation", back_populates="enterprise")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    stripe_subscription_id = Column(String(255), unique=True)
    tier = Column(String(50), nullable=False)
    status = Column(String(50), default="active")
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    enterprise = relationship("Enterprise", back_populates="subscriptions")

class Withdrawal(Base):
    __tablename__ = "withdrawals"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    transaction_id = Column(String(255), unique=True)
    amount = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    net_amount = Column(Float, nullable=False)
    currency = Column(String(10), nullable=False)
    provider = Column(String(50), nullable=False)  # opay, palmpay, crypto
    recipient = Column(String(255), nullable=False)  # phone or wallet address
    status = Column(String(50), default="pending")
    network = Column(String(50))  # For crypto: TRC20, ERC20, etc.
    tx_hash = Column(String(255))  # Blockchain transaction hash
    estimated_completion = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    enterprise = relationship("Enterprise", back_populates="withdrawals")

class Automation(Base):
    __tablename__ = "automations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    template_type = Column(String(50), nullable=False)  # survey, video, content
    platform = Column(String(50), nullable=False)  # swagbucks, youtube, etc.
    status = Column(String(50), default="active")
    config = Column(JSON)  # Automation configuration
    earnings_today = Column(Float, default=0.0)
    earnings_total = Column(Float, default=0.0)
    success_rate = Column(Float, default=0.0)
    last_run = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    enterprise = relationship("Enterprise", back_populates="automations")
    earnings = relationship("Earning", back_populates="automation")

class Earning(Base):
    __tablename__ = "earnings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    automation_id = Column(String, ForeignKey("automations.id"), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    platform = Column(String(50), nullable=False)
    task_type = Column(String(50))  # survey, video_view, etc.
    task_details = Column(JSON)
    earned_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    automation = relationship("Automation", back_populates="earnings")

class ComplianceLog(Base):
    __tablename__ = "compliance_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    check_type = Column(String(50), nullable=False)
    compliance_score = Column(Float)
    violations = Column(JSON)
    warnings = Column(JSON)
    risk_level = Column(String(20))
    activity_data = Column(JSON)
    checked_at = Column(DateTime, default=datetime.utcnow)

class SecurityAlert(Base):
    __tablename__ = "security_alerts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    description = Column(Text)
    activity_data = Column(JSON)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class WithdrawalWhitelist(Base):
    __tablename__ = "withdrawal_whitelist"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    address = Column(String(255), nullable=False)
    address_type = Column(String(20), nullable=False)  # crypto, mobile
    label = Column(String(100))
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class TwoFactorAuth(Base):
    __tablename__ = "two_factor_auth"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False, unique=True)
    secret_key = Column(String(255), nullable=False)  # Encrypted TOTP secret
    backup_codes = Column(JSON)  # Encrypted backup codes
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)

class ApiKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    permissions = Column(JSON)  # List of allowed operations
    last_used = Column(DateTime)
    expires_at = Column(DateTime)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class CryptoBalance(Base):
    __tablename__ = "crypto_balances"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    currency = Column(String(10), nullable=False)
    balance = Column(Float, nullable=False, default=0.0)
    available_balance = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    enterprise = relationship("Enterprise")

    __table_args__ = (
        UniqueConstraint('user_id', 'enterprise_id', 'currency', name='uq_crypto_balance_user_ent_curr'),
    )

class CryptoTransaction(Base):
    __tablename__ = "crypto_transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    enterprise_id = Column(String, ForeignKey("enterprises.id"), nullable=False)
    type = Column(String(20), nullable=False)  # deposit or withdrawal
    currency = Column(String(10), nullable=False)
    amount = Column(Float, nullable=False)
    fee = Column(Float, default=0.0)
    net_amount = Column(Float, nullable=False)
    address = Column(String(255))
    memo = Column(String(255))
    provider = Column(String(50))
    network = Column(String(50))
    tx_hash = Column(String(255))
    status = Column(String(50), default="pending")
    idempotency_key = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    enterprise = relationship("Enterprise")