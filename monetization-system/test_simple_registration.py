#!/usr/bin/env python3
"""
Simple test for registration without complex dependencies
"""

from database.service import DatabaseService
from database.models import Enterprise
import uuid
import hashlib
import secrets

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Hash password with salt"""
    if not salt:
        salt = secrets.token_hex(32)
    
    # Create password hash
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    )
    
    return password_hash.hex(), salt

def test_direct_registration():
    """Test registration directly with database"""
    try:
        print("🧪 Testing direct database registration...")
        
        # Create database service
        db = DatabaseService()
        
        # Test data
        email = f"direct{int(__import__('time').time())}@example.com"
        password = "testpassword123"
        
        # Check if email exists
        existing = db.get_enterprise_by_email(email)
        if existing:
            print(f"❌ Email {email} already exists")
            return False
        
        # Hash password
        password_hash, salt = hash_password(password)
        
        # Create enterprise
        enterprise = db.create_enterprise(
            name="Direct Test User",
            email=email,
            password_hash=password_hash,
            salt=salt,
            subscription_tier="basic",
            company_name="Test Company",
            phone_number="+1234567890",
            country="US"
        )
        
        print(f"✅ Enterprise created: {enterprise.id}")
        print(f"📧 Email: {enterprise.email}")
        print(f"🔐 Password hash: {enterprise.password_hash[:20]}...")
        
        # Test login verification
        from api.routes.enterprise import verify_password
        if verify_password(password, enterprise.password_hash, enterprise.salt):
            print("✅ Password verification works")
        else:
            print("❌ Password verification failed")
        
        db.db.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct registration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_direct_registration()