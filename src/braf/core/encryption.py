"""
Encryption utilities for secure credential storage in BRAF.

This module provides PBKDF2-based encryption for storing sensitive data
with configurable iterations and secure key derivation.
"""

import base64
import hashlib
import json
import os
import secrets
from typing import Any, Dict, Tuple

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet


class CredentialVault:
    """Secure credential storage using PBKDF2 key derivation and Fernet encryption."""
    
    def __init__(self, master_key: str, iterations: int = 100000):
        """
        Initialize credential vault with master key.
        
        Args:
            master_key: Master password for key derivation
            iterations: PBKDF2 iterations (default: 100,000)
        """
        self.master_key = master_key.encode('utf-8')
        self.iterations = iterations
    
    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive encryption key using PBKDF2.
        
        Args:
            salt: Random salt for key derivation
            
        Returns:
            Derived encryption key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.iterations,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.master_key))
    
    def encrypt_credentials(self, credentials: Dict[str, Any]) -> Tuple[str, str]:
        """
        Encrypt credentials dictionary.
        
        Args:
            credentials: Dictionary of credential data to encrypt
            
        Returns:
            Tuple of (encrypted_data, salt) as base64 strings
        """
        # Generate random salt
        salt = secrets.token_bytes(32)
        
        # Derive encryption key
        key = self._derive_key(salt)
        
        # Create Fernet cipher
        cipher = Fernet(key)
        
        # Serialize and encrypt credentials
        credentials_json = json.dumps(credentials, sort_keys=True)
        encrypted_data = cipher.encrypt(credentials_json.encode('utf-8'))
        
        # Return base64 encoded data and salt
        return (
            base64.b64encode(encrypted_data).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8')
        )
    
    def decrypt_credentials(self, encrypted_data: str, salt: str) -> Dict[str, Any]:
        """
        Decrypt credentials from encrypted data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            salt: Base64 encoded salt used for encryption
            
        Returns:
            Decrypted credentials dictionary
            
        Raises:
            ValueError: If decryption fails or data is invalid
        """
        try:
            # Decode base64 data
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            salt_bytes = base64.b64decode(salt.encode('utf-8'))
            
            # Derive encryption key
            key = self._derive_key(salt_bytes)
            
            # Create Fernet cipher
            cipher = Fernet(key)
            
            # Decrypt and deserialize credentials
            decrypted_data = cipher.decrypt(encrypted_bytes)
            credentials_json = decrypted_data.decode('utf-8')
            
            return json.loads(credentials_json)
            
        except Exception as e:
            raise ValueError(f"Failed to decrypt credentials: {str(e)}")
    
    def verify_master_key(self, encrypted_data: str, salt: str) -> bool:
        """
        Verify that the master key can decrypt the given data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            salt: Base64 encoded salt
            
        Returns:
            True if master key is correct, False otherwise
        """
        try:
            self.decrypt_credentials(encrypted_data, salt)
            return True
        except ValueError:
            return False


class SecureKeyManager:
    """Manager for secure key generation and rotation."""
    
    @staticmethod
    def generate_master_key(length: int = 32) -> str:
        """
        Generate a cryptographically secure master key.
        
        Args:
            length: Key length in bytes
            
        Returns:
            Base64 encoded master key
        """
        key_bytes = secrets.token_bytes(length)
        return base64.b64encode(key_bytes).decode('utf-8')
    
    @staticmethod
    def generate_salt(length: int = 32) -> str:
        """
        Generate a cryptographically secure salt.
        
        Args:
            length: Salt length in bytes
            
        Returns:
            Base64 encoded salt
        """
        salt_bytes = secrets.token_bytes(length)
        return base64.b64encode(salt_bytes).decode('utf-8')
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash password using PBKDF2.
        
        Args:
            password: Password to hash
            salt: Optional salt (generates new if not provided)
            
        Returns:
            Tuple of (hashed_password, salt) as base64 strings
        """
        if salt is None:
            salt_bytes = secrets.token_bytes(32)
        else:
            salt_bytes = base64.b64decode(salt.encode('utf-8'))
        
        # Hash password using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            iterations=100000,
        )
        
        hashed = kdf.derive(password.encode('utf-8'))
        
        return (
            base64.b64encode(hashed).decode('utf-8'),
            base64.b64encode(salt_bytes).decode('utf-8')
        )
    
    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Password to verify
            hashed_password: Base64 encoded hashed password
            salt: Base64 encoded salt
            
        Returns:
            True if password is correct, False otherwise
        """
        try:
            computed_hash, _ = SecureKeyManager.hash_password(password, salt)
            return secrets.compare_digest(computed_hash, hashed_password)
        except Exception:
            return False


class AuthorizationTokenManager:
    """Manager for authorization tokens used in compliance logging."""
    
    def __init__(self, secret_key: str):
        """
        Initialize token manager with secret key.
        
        Args:
            secret_key: Secret key for token generation
        """
        self.secret_key = secret_key.encode('utf-8')
    
    def generate_token(self, profile_id: str, worker_id: str, timestamp: str) -> str:
        """
        Generate authorization token for compliance logging.
        
        Args:
            profile_id: Profile identifier
            worker_id: Worker node identifier
            timestamp: ISO timestamp string
            
        Returns:
            Authorization token
        """
        # Create token payload
        payload = f"{profile_id}:{worker_id}:{timestamp}"
        
        # Generate HMAC signature
        signature = hashlib.hmac.new(
            self.secret_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Combine payload and signature
        token_data = f"{payload}:{signature}"
        
        # Return base64 encoded token
        return base64.b64encode(token_data.encode('utf-8')).decode('utf-8')
    
    def verify_token(self, token: str, profile_id: str, worker_id: str, timestamp: str) -> bool:
        """
        Verify authorization token.
        
        Args:
            token: Authorization token to verify
            profile_id: Expected profile identifier
            worker_id: Expected worker node identifier
            timestamp: Expected ISO timestamp string
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Decode token
            token_data = base64.b64decode(token.encode('utf-8')).decode('utf-8')
            
            # Split payload and signature
            parts = token_data.split(':')
            if len(parts) != 4:
                return False
            
            token_profile, token_worker, token_timestamp, signature = parts
            
            # Verify payload matches expected values
            if (token_profile != profile_id or 
                token_worker != worker_id or 
                token_timestamp != timestamp):
                return False
            
            # Verify signature
            expected_payload = f"{profile_id}:{worker_id}:{timestamp}"
            expected_signature = hashlib.hmac.new(
                self.secret_key,
                expected_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return secrets.compare_digest(signature, expected_signature)
            
        except Exception:
            return False


# Global instances
_credential_vault: Optional[CredentialVault] = None
_token_manager: Optional[AuthorizationTokenManager] = None


def init_encryption(master_key: str, secret_key: str) -> Tuple[CredentialVault, AuthorizationTokenManager]:
    """
    Initialize global encryption components.
    
    Args:
        master_key: Master key for credential encryption
        secret_key: Secret key for authorization tokens
        
    Returns:
        Tuple of (credential_vault, token_manager)
    """
    global _credential_vault, _token_manager
    
    _credential_vault = CredentialVault(master_key)
    _token_manager = AuthorizationTokenManager(secret_key)
    
    return _credential_vault, _token_manager


def get_credential_vault() -> CredentialVault:
    """Get global credential vault instance."""
    if _credential_vault is None:
        raise RuntimeError("Encryption not initialized. Call init_encryption() first.")
    return _credential_vault


def get_token_manager() -> AuthorizationTokenManager:
    """Get global authorization token manager instance."""
    if _token_manager is None:
        raise RuntimeError("Encryption not initialized. Call init_encryption() first.")
    return _token_manager
