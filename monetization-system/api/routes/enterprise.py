"""
Enterprise Management API endpoints
Handles enterprise registration, profile management, and account operations
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import hashlib
import secrets

from database.service import DatabaseService
from security.authentication import SecurityManager
from enterprise.subscription_service import EnterpriseSubscription
from config import Config

router = APIRouter(prefix="/api/v1/enterprise", tags=["Enterprise"])


class EnterpriseRegistrationRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    subscription_tier: str = "enterprise"
    company_name: Optional[str] = None
    phone_number: Optional[str] = None
    country: str = "US"


class EnterpriseLoginRequest(BaseModel):
    email: EmailStr
    password: str
    totp_code: Optional[str] = None


class EnterpriseProfileUpdate(BaseModel):
    name: Optional[str] = None
    company_name: Optional[str] = None
    phone_number: Optional[str] = None
    country: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    totp_code: str


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


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verify password against hash"""
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == password_hash


@router.post("/register")
async def register_enterprise(request: EnterpriseRegistrationRequest):
    """Register a new enterprise account"""
    try:
        with DatabaseService() as db:
            # Check if email already exists
            existing = db.get_enterprise_by_email(request.email)
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Force free tier for all accounts during beta period
            request.subscription_tier = "free"
            
            # Hash password
            password_hash, salt = hash_password(request.password)
            
            # Create enterprise with password
            enterprise = db.create_enterprise(
                name=request.name,
                email=request.email,
                password_hash=password_hash,
                salt=salt,
                subscription_tier=request.subscription_tier,
                company_name=request.company_name,
                phone_number=request.phone_number,
                country=request.country
            )
            
            # Create initial subscription
            subscription_data = {
                "subscription_id": f"sub_{enterprise.id}_{datetime.utcnow().timestamp()}",
                "tier": request.subscription_tier,
                "amount": Config.SUBSCRIPTION_TIERS[request.subscription_tier]["price"] / 100,  # Convert from cents
                "status": "trial"  # Start with trial
            }
            
            subscription = db.create_subscription(enterprise.id, subscription_data)
            
            # Extract data while still in session
            enterprise_id = enterprise.id
            subscription_tier = subscription.tier
            subscription_status = subscription.status
        
        # Set up 2FA (simplified for now)
        tfa_setup = {
            "qr_code_url": "https://example.com/qr",
            "secret": "EXAMPLE2FA",
            "setup_required": True
        }
        
        return {
            "success": True,
            "enterprise_id": enterprise_id,
            "message": "Enterprise account created successfully",
            "subscription": {
                "tier": subscription_tier,
                "status": subscription_status,
                "features": Config.SUBSCRIPTION_TIERS[request.subscription_tier]["features"]
            },
            "two_factor_auth": {
                "qr_code_url": tfa_setup["qr_code_url"],
                "secret": tfa_setup["secret"],  # Show once for backup
                "setup_required": True
            },
            "next_steps": [
                "Set up two-factor authentication using the QR code",
                "Complete your profile information",
                "Add payment method to activate subscription",
                "Start creating automations"
            ]
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login")
async def login_enterprise(request: EnterpriseLoginRequest):
    """Login to enterprise account"""
    try:
        with DatabaseService() as db:
            # Get enterprise by email
            enterprise = db.get_enterprise_by_email(request.email)
            if not enterprise:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Verify password
            if not verify_password(request.password, enterprise.password_hash, enterprise.salt):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # Check 2FA if provided
            if request.totp_code:
                security_manager = SecurityManager()
                if not security_manager.verify_2fa(enterprise.id, request.totp_code):
                    raise HTTPException(status_code=401, detail="Invalid 2FA code")
            
            # Get subscription info
            subscription = db.get_active_subscription(enterprise.id)
            
            # Get dashboard data
            dashboard_data = db.get_dashboard_data(enterprise.id)
            
            return {
                "success": True,
                "enterprise": {
                    "id": enterprise.id,
                    "name": enterprise.name,
                    "email": enterprise.email,
                    "subscription_tier": enterprise.subscription_tier,
                    "kyc_level": enterprise.kyc_level,
                    "created_at": enterprise.created_at.isoformat()
                },
                "subscription": {
                    "tier": subscription.tier if subscription else "none",
                    "status": subscription.status if subscription else "inactive",
                    "features": Config.SUBSCRIPTION_TIERS.get(enterprise.subscription_tier, {}).get("features", [])
                },
                "dashboard": dashboard_data,
                "session_token": f"session_{enterprise.id}_{datetime.utcnow().timestamp()}",  # Simple token for demo
                "message": "Login successful"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@router.get("/profile/{enterprise_id}")
async def get_enterprise_profile(enterprise_id: str):
    """Get enterprise profile information"""
    try:
        with DatabaseService() as db:
            enterprise = db.get_enterprise(enterprise_id)
            if not enterprise:
                raise HTTPException(status_code=404, detail="Enterprise not found")
            
            subscription = db.get_active_subscription(enterprise_id)
            automations = db.get_automations(enterprise_id)
            dashboard_data = db.get_dashboard_data(enterprise_id)
            
            return {
                "enterprise": {
                    "id": enterprise.id,
                    "name": enterprise.name,
                    "email": enterprise.email,
                    "subscription_tier": enterprise.subscription_tier,
                    "subscription_status": enterprise.subscription_status,
                    "kyc_level": enterprise.kyc_level,
                    "created_at": enterprise.created_at.isoformat(),
                    "updated_at": enterprise.updated_at.isoformat()
                },
                "subscription": {
                    "tier": subscription.tier if subscription else "none",
                    "status": subscription.status if subscription else "inactive",
                    "amount": subscription.amount if subscription else 0,
                    "features": Config.SUBSCRIPTION_TIERS.get(enterprise.subscription_tier, {}).get("features", [])
                },
                "statistics": {
                    "total_automations": len(automations),
                    "active_automations": len([a for a in automations if a.status == "active"]),
                    "total_earnings": dashboard_data["total_earnings"],
                    "available_balance": dashboard_data["available_balance"],
                    "total_withdrawn": dashboard_data["total_withdrawn"]
                },
                "limits": {
                    "max_automations": Config.SUBSCRIPTION_TIERS.get(enterprise.subscription_tier, {}).get("max_automations", 1),
                    "daily_earnings_limit": Config.SUBSCRIPTION_TIERS.get(enterprise.subscription_tier, {}).get("daily_earnings_limit", 10)
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")


@router.put("/profile/{enterprise_id}")
async def update_enterprise_profile(enterprise_id: str, request: EnterpriseProfileUpdate):
    """Update enterprise profile"""
    try:
        with DatabaseService() as db:
            enterprise = db.get_enterprise(enterprise_id)
            if not enterprise:
                raise HTTPException(status_code=404, detail="Enterprise not found")
            
            # Update fields
            if request.name:
                enterprise.name = request.name
            
            # TODO: Add company_name, phone_number, country fields to Enterprise model
            
            enterprise.updated_at = datetime.utcnow()
            db.db.commit()
            
            return {
                "success": True,
                "message": "Profile updated successfully",
                "enterprise": {
                    "id": enterprise.id,
                    "name": enterprise.name,
                    "email": enterprise.email,
                    "updated_at": enterprise.updated_at.isoformat()
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")


@router.post("/upgrade-subscription")
async def upgrade_subscription(enterprise_id: str, new_tier: str, payment_method_id: str):
    """Upgrade enterprise subscription"""
    try:
        # Validate tier
        valid_tiers = ["basic", "pro", "enterprise"]
        if new_tier not in valid_tiers:
            raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {valid_tiers}")
        
        with DatabaseService() as db:
            enterprise = db.get_enterprise(enterprise_id)
            if not enterprise:
                raise HTTPException(status_code=404, detail="Enterprise not found")
            
            # Check if it's actually an upgrade
            tier_hierarchy = {"basic": 1, "pro": 2, "enterprise": 3}
            current_level = tier_hierarchy.get(enterprise.subscription_tier, 0)
            new_level = tier_hierarchy.get(new_tier, 0)
            
            if new_level <= current_level:
                raise HTTPException(status_code=400, detail="Can only upgrade to higher tier")
            
            # Process payment with Stripe
            subscription_service = EnterpriseSubscription(Config.STRIPE_SECRET_KEY)
            
            try:
                result = subscription_service.create_subscription(
                    enterprise_id=enterprise_id,
                    tier=new_tier,
                    payment_method_id=payment_method_id
                )
                
                # Update enterprise tier
                enterprise.subscription_tier = new_tier
                enterprise.subscription_status = "active"
                enterprise.updated_at = datetime.utcnow()
                
                # Create new subscription record
                subscription = db.create_subscription(enterprise_id, {
                    "subscription_id": result["subscription_id"],
                    "tier": new_tier,
                    "amount": Config.SUBSCRIPTION_TIERS[new_tier]["price"] / 100,
                    "status": "active"
                })
                
                db.db.commit()
                
                return {
                    "success": True,
                    "message": f"Successfully upgraded to {new_tier} tier",
                    "subscription": {
                        "tier": new_tier,
                        "status": "active",
                        "features": Config.SUBSCRIPTION_TIERS[new_tier]["features"],
                        "limits": {
                            "max_automations": Config.SUBSCRIPTION_TIERS[new_tier]["max_automations"],
                            "daily_earnings_limit": Config.SUBSCRIPTION_TIERS[new_tier]["daily_earnings_limit"]
                        }
                    },
                    "stripe_data": result
                }
                
            except Exception as stripe_error:
                raise HTTPException(status_code=400, detail=f"Payment failed: {str(stripe_error)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upgrade failed: {str(e)}")


@router.get("/list")
async def list_enterprises(limit: int = 50, offset: int = 0):
    """List all enterprises (admin endpoint)"""
    try:
        with DatabaseService() as db:
            # Get enterprises with pagination
            from database.models import Enterprise
            enterprises = db.db.query(Enterprise).offset(offset).limit(limit).all()
            
            return {
                "enterprises": [
                    {
                        "id": ent.id,
                        "name": ent.name,
                        "email": ent.email,
                        "subscription_tier": ent.subscription_tier,
                        "subscription_status": ent.subscription_status,
                        "kyc_level": ent.kyc_level,
                        "created_at": ent.created_at.isoformat()
                    } for ent in enterprises
                ],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": len(enterprises)
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list enterprises: {str(e)}")


@router.delete("/{enterprise_id}")
async def delete_enterprise(enterprise_id: str, confirmation_code: str):
    """Delete enterprise account (requires confirmation)"""
    try:
        # Simple confirmation - in production, use proper verification
        expected_code = f"DELETE_{enterprise_id[-8:]}"
        if confirmation_code != expected_code:
            raise HTTPException(status_code=400, detail=f"Invalid confirmation code. Expected: {expected_code}")
        
        with DatabaseService() as db:
            enterprise = db.get_enterprise(enterprise_id)
            if not enterprise:
                raise HTTPException(status_code=404, detail="Enterprise not found")
            
            # TODO: Implement cascade deletion of related records
            # For now, just mark as deleted
            enterprise.subscription_status = "deleted"
            enterprise.updated_at = datetime.utcnow()
            db.db.commit()
            
            return {
                "success": True,
                "message": "Enterprise account deleted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")