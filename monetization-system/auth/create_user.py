#!/usr/bin/env python3
"""
User Creation Module for BRAF System
Creates admin and regular users with proper authentication
"""

import sys
import argparse
import logging
import hashlib
import secrets
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.models import Enterprise
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def hash_password(password: str, salt: str = None) -> tuple:
    """Hash password with salt"""
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Create password hash using PBKDF2
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # 100k iterations
    ).hex()
    
    return password_hash, salt

def create_user(username: str, password: str, role: str = "user", email: str = None):
    """Create a new user in the system"""
    try:
        logger.info(f"🔐 Creating user: {username} with role: {role}")
        
        # Get database URL from config
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        session = SessionLocal()
        
        try:
            # Use email if provided, otherwise use username@braf.local
            user_email = email or f"{username}@braf.local"
            
            # Check if user already exists
            existing_user = session.query(Enterprise).filter_by(email=user_email).first()
            if existing_user:
                logger.warning(f"⚠️ User with email {user_email} already exists")
                return False
            
            # Hash password
            password_hash, salt = hash_password(password)
            
            # Determine subscription tier based on role
            subscription_tier = "enterprise" if role == "admin" else "free"
            
            # Create new user
            new_user = Enterprise(
                name=username.title(),
                email=user_email,
                password_hash=password_hash,
                salt=salt,
                subscription_tier=subscription_tier,
                subscription_status="active",
                kyc_level=3 if role == "admin" else 1,
                company_name=f"{username.title()} Company",
                phone_number="+1234567890",
                country="US"
            )
            
            session.add(new_user)
            session.commit()
            
            logger.info(f"✅ User created successfully:")
            logger.info(f"   📧 Email: {user_email}")
            logger.info(f"   👤 Name: {username.title()}")
            logger.info(f"   🎭 Role: {role}")
            logger.info(f"   📊 Tier: {subscription_tier}")
            logger.info(f"   🆔 ID: {new_user.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ User creation failed: {e}")
            session.rollback()
            return False
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def create_admin_user(username: str = "admin", password: str = "admin123"):
    """Create default admin user"""
    return create_user(
        username=username,
        password=password,
        role="admin",
        email=f"{username}@braf.local"
    )

def list_users():
    """List all users in the system"""
    try:
        logger.info("📋 Listing all users...")
        
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        session = SessionLocal()
        
        try:
            users = session.query(Enterprise).all()
            
            if not users:
                logger.info("📭 No users found in the system")
                return
            
            logger.info(f"👥 Found {len(users)} users:")
            for user in users:
                logger.info(f"   🆔 {user.id}")
                logger.info(f"   📧 {user.email}")
                logger.info(f"   👤 {user.name}")
                logger.info(f"   📊 {user.subscription_tier}")
                logger.info(f"   📅 Created: {user.created_at}")
                logger.info("   " + "-" * 40)
                
        except Exception as e:
            logger.error(f"❌ Failed to list users: {e}")
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")

def verify_user(email: str, password: str) -> bool:
    """Verify user credentials"""
    try:
        database_url = Config.get_database_url()
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        
        session = SessionLocal()
        
        try:
            user = session.query(Enterprise).filter_by(email=email).first()
            if not user:
                return False
            
            # Hash provided password with stored salt
            password_hash, _ = hash_password(password, user.salt)
            
            return password_hash == user.password_hash
            
        except Exception as e:
            logger.error(f"❌ User verification failed: {e}")
            return False
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def main():
    """Main entry point for user creation"""
    parser = argparse.ArgumentParser(description="Create users for BRAF system")
    parser.add_argument("--username", required=True, help="Username for the new user")
    parser.add_argument("--password", required=True, help="Password for the new user")
    parser.add_argument("--role", default="user", choices=["user", "admin"], help="User role")
    parser.add_argument("--email", help="Email address (optional)")
    parser.add_argument("--list", action="store_true", help="List all users")
    parser.add_argument("--create-admin", action="store_true", help="Create default admin user")
    
    args = parser.parse_args()
    
    if args.list:
        list_users()
        return
    
    if args.create_admin:
        if create_admin_user():
            logger.info("✅ Default admin user created successfully")
        else:
            logger.error("❌ Failed to create default admin user")
            sys.exit(1)
        return
    
    # Create user with provided arguments
    if create_user(args.username, args.password, args.role, args.email):
        logger.info("✅ User created successfully")
    else:
        logger.error("❌ Failed to create user")
        sys.exit(1)

if __name__ == "__main__":
    main()