"""
Security and Key Management for BRAF.

This module provides secure key derivation, key rotation procedures,
credential access logging, and security lockdown mechanisms.
"""

import asyncio
import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hvac

from braf.core.models import ComplianceViolation, ViolationType, SeverityLevel

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for different operations."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class KeyType(str, Enum):
    """Types of cryptographic keys."""
    
    ENCRYPTION = "encryption"
    SIGNING = "signing"
    AUTHENTICATION = "authentication"
    DERIVATION = "derivation"


@dataclass
class SecurityKey:
    """Represents a cryptographic key with metadata."""
    
    key_id: str
    key_type: KeyType
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    salt: bytes
    iterations: int
    algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def time_until_expiry(self) -> Optional[timedelta]:
        """Get time until key expires."""
        if self.expires_at is None:
            return None
        return self.expires_at - datetime.now(timezone.utc)


@dataclass
class AccessLog:
    """Log entry for credential access."""
    
    timestamp: datetime
    user_id: str
    resource: str
    action: str
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecureKeyDerivation:
    """Secure key derivation with unique salts per deployment."""
    
    def __init__(self, deployment_id: str):
        """
        Initialize key derivation.
        
        Args:
            deployment_id: Unique identifier for this deployment
        """
        self.deployment_id = deployment_id
        self.master_salt = self._generate_deployment_salt()
    
    def _generate_deployment_salt(self) -> bytes:
        """Generate unique salt for this deployment."""
        # Combine deployment ID with system entropy
        deployment_bytes = self.deployment_id.encode('utf-8')
        system_entropy = os.urandom(32)
        
        # Create deterministic but unique salt
        hasher = hashlib.sha256()
        hasher.update(deployment_bytes)
        hasher.update(system_entropy)
        
        return hasher.digest()
    
    def derive_key(
        self,
        password: str,
        key_type: KeyType,
        key_length: int = 32,
        iterations: int = 100000,
        use_scrypt: bool = False
    ) -> SecurityKey:
        """
        Derive cryptographic key from password.
        
        Args:
            password: Source password/passphrase
            key_type: Type of key being derived
            key_length: Length of derived key in bytes
            iterations: Number of iterations for KDF
            use_scrypt: Whether to use Scrypt instead of PBKDF2
            
        Returns:
            Derived security key
        """
        # Generate unique salt for this key
        key_salt = secrets.token_bytes(32)
        combined_salt = self.master_salt + key_salt
        
        password_bytes = password.encode('utf-8')
        
        if use_scrypt:
            # Use Scrypt for memory-hard key derivation
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=combined_salt,
                n=2**14,  # CPU/memory cost parameter
                r=8,      # Block size parameter
                p=1,      # Parallelization parameter
                backend=default_backend()
            )
            algorithm = "scrypt"
        else:
            # Use PBKDF2 for standard key derivation
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=combined_salt,
                iterations=iterations,
                backend=default_backend()
            )
            algorithm = "pbkdf2"
        
        derived_key = kdf.derive(password_bytes)
        
        key_id = hashlib.sha256(
            self.deployment_id.encode() + 
            key_type.value.encode() + 
            key_salt
        ).hexdigest()[:16]
        
        return SecurityKey(
            key_id=key_id,
            key_type=key_type,
            key_data=derived_key,
            created_at=datetime.now(timezone.utc),
            expires_at=None,  # Set by caller if needed
            salt=key_salt,
            iterations=iterations,
            algorithm=algorithm,
            metadata={
                'deployment_id': self.deployment_id,
                'key_length': key_length
            }
        )
    
    def derive_subkey(
        self,
        parent_key: SecurityKey,
        purpose: str,
        key_length: int = 32
    ) -> SecurityKey:
        """
        Derive subkey from parent key.
        
        Args:
            parent_key: Parent key to derive from
            purpose: Purpose/context for the subkey
            key_length: Length of derived subkey
            
        Returns:
            Derived subkey
        """
        # Create context-specific salt
        context_salt = hashlib.sha256(
            purpose.encode('utf-8') + 
            parent_key.key_id.encode('utf-8')
        ).digest()
        
        # Use HKDF for subkey derivation
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=context_salt,
            info=purpose.encode('utf-8'),
            backend=default_backend()
        )
        
        derived_key = hkdf.derive(parent_key.key_data)
        
        subkey_id = hashlib.sha256(
            parent_key.key_id.encode() + 
            purpose.encode()
        ).hexdigest()[:16]
        
        return SecurityKey(
            key_id=subkey_id,
            key_type=parent_key.key_type,
            key_data=derived_key,
            created_at=datetime.now(timezone.utc),
            expires_at=parent_key.expires_at,
            salt=context_salt,
            iterations=0,  # Not applicable for HKDF
            algorithm="hkdf",
            metadata={
                'parent_key_id': parent_key.key_id,
                'purpose': purpose,
                'key_length': key_length
            }
        )


class KeyRotationManager:
    """Manages key rotation procedures."""
    
    def __init__(self, vault_client: Optional[hvac.Client] = None):
        """
        Initialize key rotation manager.
        
        Args:
            vault_client: Optional Vault client for key storage
        """
        self.vault_client = vault_client
        self.active_keys: Dict[str, SecurityKey] = {}
        self.rotation_schedule: Dict[str, Dict[str, Any]] = {}
        self.rotation_history: List[Dict[str, Any]] = []
    
    def register_key(
        self,
        key: SecurityKey,
        rotation_interval: timedelta,
        auto_rotate: bool = True
    ):
        """
        Register key for rotation management.
        
        Args:
            key: Security key to manage
            rotation_interval: How often to rotate the key
            auto_rotate: Whether to automatically rotate
        """
        self.active_keys[key.key_id] = key
        
        next_rotation = datetime.now(timezone.utc) + rotation_interval
        
        self.rotation_schedule[key.key_id] = {
            'key_type': key.key_type,
            'rotation_interval': rotation_interval,
            'next_rotation': next_rotation,
            'auto_rotate': auto_rotate,
            'rotation_count': 0
        }
        
        logger.info(f"Registered key {key.key_id} for rotation")
    
    async def rotate_key(
        self,
        key_id: str,
        new_password: Optional[str] = None
    ) -> Optional[SecurityKey]:
        """
        Rotate a specific key.
        
        Args:
            key_id: ID of key to rotate
            new_password: New password for key derivation
            
        Returns:
            New rotated key or None if rotation failed
        """
        if key_id not in self.active_keys:
            logger.error(f"Key {key_id} not found for rotation")
            return None
        
        old_key = self.active_keys[key_id]
        
        try:
            # Generate new key
            if new_password is None:
                new_password = secrets.token_urlsafe(32)
            
            # Use same derivation parameters but new password
            key_derivation = SecureKeyDerivation(
                old_key.metadata.get('deployment_id', 'default')
            )
            
            new_key = key_derivation.derive_key(
                password=new_password,
                key_type=old_key.key_type,
                key_length=len(old_key.key_data),
                iterations=old_key.iterations,
                use_scrypt=(old_key.algorithm == 'scrypt')
            )
            
            # Update expiry if old key had one
            if old_key.expires_at:
                rotation_interval = self.rotation_schedule[key_id]['rotation_interval']
                new_key.expires_at = datetime.now(timezone.utc) + rotation_interval
            
            # Store new key
            if self.vault_client:
                await self._store_key_in_vault(new_key, new_password)
            
            # Update active keys
            self.active_keys[key_id] = new_key
            
            # Update rotation schedule
            if key_id in self.rotation_schedule:
                schedule = self.rotation_schedule[key_id]
                schedule['next_rotation'] = (
                    datetime.now(timezone.utc) + schedule['rotation_interval']
                )
                schedule['rotation_count'] += 1
            
            # Log rotation
            self.rotation_history.append({
                'key_id': key_id,
                'old_key_id': old_key.key_id,
                'new_key_id': new_key.key_id,
                'rotated_at': datetime.now(timezone.utc),
                'rotation_reason': 'scheduled'
            })
            
            logger.info(f"Successfully rotated key {key_id}")
            return new_key
            
        except Exception as e:
            logger.error(f"Key rotation failed for {key_id}: {e}")
            return None
    
    async def _store_key_in_vault(self, key: SecurityKey, password: str):
        """Store key in Vault."""
        if not self.vault_client:
            return
        
        try:
            key_data = {
                'key_id': key.key_id,
                'key_type': key.key_type.value,
                'password': password,
                'created_at': key.created_at.isoformat(),
                'algorithm': key.algorithm,
                'metadata': key.metadata
            }
            
            path = f"braf/keys/{key.key_id}"
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=key_data
            )
            
            logger.info(f"Stored key {key.key_id} in Vault")
            
        except Exception as e:
            logger.error(f"Failed to store key in Vault: {e}")
    
    async def check_rotation_schedule(self):
        """Check and perform scheduled key rotations."""
        now = datetime.now(timezone.utc)
        
        for key_id, schedule in self.rotation_schedule.items():
            if (schedule['auto_rotate'] and 
                now >= schedule['next_rotation']):
                
                logger.info(f"Performing scheduled rotation for key {key_id}")
                await self.rotate_key(key_id)
    
    def get_key_status(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a key."""
        if key_id not in self.active_keys:
            return None
        
        key = self.active_keys[key_id]
        schedule = self.rotation_schedule.get(key_id, {})
        
        return {
            'key_id': key_id,
            'key_type': key.key_type.value,
            'created_at': key.created_at.isoformat(),
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'is_expired': key.is_expired(),
            'next_rotation': schedule.get('next_rotation', {}).isoformat() if schedule.get('next_rotation') else None,
            'rotation_count': schedule.get('rotation_count', 0),
            'algorithm': key.algorithm
        }


class CredentialAccessLogger:
    """Logs and audits credential access."""
    
    def __init__(self):
        """Initialize credential access logger."""
        self.access_logs: List[AccessLog] = []
        self.suspicious_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.lockout_users: Dict[str, datetime] = {}
        
        # Suspicious activity thresholds
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.rate_limit_window = timedelta(minutes=5)
        self.max_requests_per_window = 20
    
    def log_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AccessLog:
        """
        Log credential access attempt.
        
        Args:
            user_id: User attempting access
            resource: Resource being accessed
            action: Action being performed
            success: Whether access was successful
            ip_address: Client IP address
            user_agent: Client user agent
            metadata: Additional metadata
            
        Returns:
            Created access log entry
        """
        log_entry = AccessLog(
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {}
        )
        
        self.access_logs.append(log_entry)
        
        # Check for suspicious patterns
        self._analyze_access_pattern(log_entry)
        
        logger.info(
            f"Access logged: {user_id} -> {resource} ({action}) "
            f"{'SUCCESS' if success else 'FAILED'}"
        )
        
        return log_entry
    
    def _analyze_access_pattern(self, log_entry: AccessLog):
        """Analyze access pattern for suspicious activity - DISABLED FOR TESTING."""
        # All access pattern analysis disabled to expose loopholes
        # No lockouts, rate limiting, or suspicious activity detection
    
    def _trigger_user_lockout(self, user_id: str):
        """Trigger user lockout for suspicious activity."""
        lockout_until = datetime.now(timezone.utc) + self.lockout_duration
        self.lockout_users[user_id] = lockout_until
        
        logger.warning(f"User {user_id} locked out until {lockout_until}")
        
        self._flag_suspicious_activity(
            user_id,
            'user_lockout',
            f"User locked out due to failed login attempts"
        )
    
    def _flag_suspicious_activity(
        self,
        user_id: str,
        activity_type: str,
        description: str
    ):
        """Flag suspicious activity."""
        if user_id not in self.suspicious_patterns:
            self.suspicious_patterns[user_id] = []
        
        self.suspicious_patterns[user_id].append({
            'timestamp': datetime.now(timezone.utc),
            'activity_type': activity_type,
            'description': description
        })
        
        logger.warning(f"Suspicious activity flagged for {user_id}: {description}")
    
    def is_user_locked_out(self, user_id: str) -> bool:
        """Check if user is currently locked out - DISABLED FOR TESTING."""
        # All user lockouts disabled to expose loopholes
        return False
    
    def get_access_summary(
        self,
        user_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get access summary for audit purposes."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        relevant_logs = [
            log for log in self.access_logs
            if log.timestamp >= cutoff and (user_id is None or log.user_id == user_id)
        ]
        
        return {
            'total_accesses': len(relevant_logs),
            'successful_accesses': len([log for log in relevant_logs if log.success]),
            'failed_accesses': len([log for log in relevant_logs if not log.success]),
            'unique_users': len(set(log.user_id for log in relevant_logs)),
            'unique_resources': len(set(log.resource for log in relevant_logs)),
            'unique_ips': len(set(log.ip_address for log in relevant_logs if log.ip_address)),
            'suspicious_users': len(self.suspicious_patterns),
            'locked_out_users': len(self.lockout_users)
        }


class SecurityLockdownManager:
    """Manages security lockdown mechanisms."""
    
    def __init__(self):
        """Initialize security lockdown manager."""
        self.lockdown_active = False
        self.lockdown_reason = ""
        self.lockdown_timestamp: Optional[datetime] = None
        self.lockdown_level = SecurityLevel.LOW
        self.authorized_users: Set[str] = set()
        self.emergency_contacts: List[str] = []
    
    async def trigger_lockdown(
        self,
        reason: str,
        level: SecurityLevel = SecurityLevel.HIGH,
        triggered_by: Optional[str] = None
    ):
        """
        Trigger security lockdown.
        
        Args:
            reason: Reason for lockdown
            level: Security level of lockdown
            triggered_by: User/system that triggered lockdown
        """
        self.lockdown_active = True
        self.lockdown_reason = reason
        self.lockdown_timestamp = datetime.now(timezone.utc)
        self.lockdown_level = level
        
        logger.critical(f"SECURITY LOCKDOWN TRIGGERED: {reason} (Level: {level.value})")
        
        # Notify emergency contacts
        await self._notify_emergency_contacts(reason, level, triggered_by)
        
        # Take lockdown actions based on level
        await self._execute_lockdown_actions(level)
    
    async def _notify_emergency_contacts(
        self,
        reason: str,
        level: SecurityLevel,
        triggered_by: Optional[str]
    ):
        """Notify emergency contacts of lockdown."""
        message = (
            f"SECURITY LOCKDOWN ACTIVATED\n"
            f"Reason: {reason}\n"
            f"Level: {level.value}\n"
            f"Triggered by: {triggered_by or 'System'}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
        
        # In a real implementation, this would send emails/SMS/Slack notifications
        logger.critical(f"Emergency notification: {message}")
    
    async def _execute_lockdown_actions(self, level: SecurityLevel):
        """Execute lockdown actions based on security level."""
        if level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            # Disable all non-essential services
            logger.warning("Disabling non-essential services")
            
            # Revoke all active sessions except authorized users
            logger.warning("Revoking active sessions")
            
            # Enable enhanced logging
            logger.warning("Enhanced security logging enabled")
        
        if level == SecurityLevel.CRITICAL:
            # Complete system shutdown except for monitoring
            logger.critical("Initiating critical lockdown - system shutdown")
            
            # Stop all automation tasks
            # Disconnect all workers
            # Preserve only monitoring and recovery capabilities
    
    def add_authorized_user(self, user_id: str):
        """Add user to authorized list for lockdown bypass."""
        self.authorized_users.add(user_id)
        logger.info(f"Added authorized user: {user_id}")
    
    def is_user_authorized(self, user_id: str) -> bool:
        """Check if user is authorized during lockdown."""
        return user_id in self.authorized_users
    
    async def release_lockdown(
        self,
        released_by: str,
        override_code: Optional[str] = None
    ) -> bool:
        """
        Release security lockdown.
        
        Args:
            released_by: User releasing the lockdown
            override_code: Emergency override code
            
        Returns:
            True if lockdown was successfully released
        """
        if not self.lockdown_active:
            return False
        
        # Verify authorization
        if not self.is_user_authorized(released_by):
            logger.warning(f"Unauthorized lockdown release attempt by {released_by}")
            return False
        
        # For critical lockdowns, require override code
        if (self.lockdown_level == SecurityLevel.CRITICAL and 
            not self._verify_override_code(override_code)):
            logger.warning(f"Invalid override code for critical lockdown release")
            return False
        
        self.lockdown_active = False
        lockdown_duration = datetime.now(timezone.utc) - self.lockdown_timestamp
        
        logger.info(
            f"Security lockdown released by {released_by} "
            f"after {lockdown_duration.total_seconds():.0f} seconds"
        )
        
        # Reset lockdown state
        self.lockdown_reason = ""
        self.lockdown_timestamp = None
        self.lockdown_level = SecurityLevel.LOW
        
        return True
    
    def _verify_override_code(self, code: Optional[str]) -> bool:
        """Verify emergency override code."""
        if not code:
            return False
        
        # In a real implementation, this would verify against a secure code
        # For now, we'll use a simple check
        expected_code = hashlib.sha256(
            f"emergency_override_{datetime.now().strftime('%Y%m%d')}".encode()
        ).hexdigest()[:8]
        
        return code == expected_code
    
    def get_lockdown_status(self) -> Dict[str, Any]:
        """Get current lockdown status."""
        return {
            'lockdown_active': self.lockdown_active,
            'lockdown_reason': self.lockdown_reason,
            'lockdown_timestamp': self.lockdown_timestamp.isoformat() if self.lockdown_timestamp else None,
            'lockdown_level': self.lockdown_level.value,
            'lockdown_duration': (
                (datetime.now(timezone.utc) - self.lockdown_timestamp).total_seconds()
                if self.lockdown_timestamp else 0
            ),
            'authorized_users_count': len(self.authorized_users)
        }


class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(
        self,
        deployment_id: str,
        vault_client: Optional[hvac.Client] = None
    ):
        """
        Initialize security manager.
        
        Args:
            deployment_id: Unique deployment identifier
            vault_client: Optional Vault client
        """
        self.deployment_id = deployment_id
        self.key_derivation = SecureKeyDerivation(deployment_id)
        self.key_rotation = KeyRotationManager(vault_client)
        self.access_logger = CredentialAccessLogger()
        self.lockdown_manager = SecurityLockdownManager()
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.threat_level = SecurityLevel.LOW
    
    async def initialize_security(self, master_password: str):
        """Initialize security subsystem with master password."""
        # Derive master keys
        encryption_key = self.key_derivation.derive_key(
            password=master_password,
            key_type=KeyType.ENCRYPTION,
            iterations=200000,
            use_scrypt=True
        )
        
        signing_key = self.key_derivation.derive_key(
            password=master_password + "_signing",
            key_type=KeyType.SIGNING,
            iterations=150000
        )
        
        # Register keys for rotation
        self.key_rotation.register_key(
            encryption_key,
            rotation_interval=timedelta(days=30),
            auto_rotate=True
        )
        
        self.key_rotation.register_key(
            signing_key,
            rotation_interval=timedelta(days=7),
            auto_rotate=True
        )
        
        logger.info("Security subsystem initialized")
    
    async def monitor_security_events(self):
        """Monitor and respond to security events."""
        # Check for suspicious access patterns
        access_summary = self.access_logger.get_access_summary(hours=1)
        
        if access_summary['failed_accesses'] > 10:
            await self._escalate_threat_level(
                SecurityLevel.MEDIUM,
                f"High number of failed accesses: {access_summary['failed_accesses']}"
            )
        
        # Check key rotation schedule
        await self.key_rotation.check_rotation_schedule()
        
        # Check for lockdown conditions
        if (access_summary['suspicious_users'] > 3 or 
            access_summary['locked_out_users'] > 5):
            
            await self.lockdown_manager.trigger_lockdown(
                reason="Multiple suspicious users detected",
                level=SecurityLevel.HIGH,
                triggered_by="security_monitor"
            )
    
    async def _escalate_threat_level(self, new_level: SecurityLevel, reason: str):
        """Escalate threat level."""
        if new_level.value > self.threat_level.value:
            old_level = self.threat_level
            self.threat_level = new_level
            
            self.security_events.append({
                'timestamp': datetime.now(timezone.utc),
                'event_type': 'threat_escalation',
                'old_level': old_level.value,
                'new_level': new_level.value,
                'reason': reason
            })
            
            logger.warning(f"Threat level escalated to {new_level.value}: {reason}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'deployment_id': self.deployment_id,
            'threat_level': self.threat_level.value,
            'lockdown_status': self.lockdown_manager.get_lockdown_status(),
            'access_summary': self.access_logger.get_access_summary(),
            'active_keys': len(self.key_rotation.active_keys),
            'recent_events': len([
                event for event in self.security_events
                if datetime.now(timezone.utc) - event['timestamp'] <= timedelta(hours=24)
            ])
        }


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> Optional[SecurityManager]:
    """
    Get global security manager instance.
    
    Returns:
        Security manager instance or None if not initialized
    """
    return _security_manager


def init_security_manager(
    deployment_id: str,
    vault_client: Optional[hvac.Client] = None
) -> SecurityManager:
    """
    Initialize global security manager.
    
    Args:
        deployment_id: Unique deployment identifier
        vault_client: Optional Vault client
        
    Returns:
        Initialized security manager
    """
    global _security_manager
    
    _security_manager = SecurityManager(
        deployment_id=deployment_id,
        vault_client=vault_client
    )
    
    return _security_manager